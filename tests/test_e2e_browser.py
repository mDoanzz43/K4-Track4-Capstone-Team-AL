"""Browser end-to-end test (skipped when Playwright/Chromium is unavailable).

Starts the real server, then in headless Chromium:
  * draws on the canvas pad -> extracts -> lets the robot draw;
  * feeds synthetic hand keypoints (pinch = pen down) into the air-drawing
    controller, checks the gesture logic, extracts the result;
  * loads the vendored MediaPipe model + WASM and runs it once on a blank frame.
"""
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api")
ROOT = Path(__file__).resolve().parent.parent

SYNTH = """
() => {
  const air = window.__air;
  air.pen.clear();
  const hand = (cx, cy, gap) => {
    const lm = Array.from({length: 21}, () => ({x: cx, y: cy + 0.15}));
    lm[0] = {x: cx, y: cy + 0.30}; lm[9] = {x: cx, y: cy + 0.05};
    lm[4] = {x: cx - gap / 2, y: cy}; lm[8] = {x: cx + gap / 2, y: cy};
    return lm;
  };
  let t = 0;
  const line = (a, b, n) => Array.from({length: n}, (_, i) => [a[0] + (b[0]-a[0]) * i / (n-1), a[1] + (b[1]-a[1]) * i / (n-1)]);
  const go = (pts, gap) => { for (const [x, y] of pts) { air.feed(hand(x, y, gap), t); t += 33; } };
  go(line([0.6, 0.2], [0.6, 0.2], 5), 0.20);    // hover, open hand
  go(line([0.6, 0.2], [0.6, 0.7], 40), 0.02);   // pinch -> stroke 1
  go(line([0.6, 0.7], [0.4, 0.7], 10), 0.20);   // open hand -> pen up while moving
  go(line([0.62, 0.45], [0.40, 0.45], 30), 0.03); // pinch -> stroke 2
  air.feed(null, t);                             // hand leaves the frame
  const s = air.pen.strokes;
  return {lens: s.map(x => x.length), first: s[0][0], last: s[0][s[0].length - 1], down: air.pen.down};
}
"""


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app:app", "--port", str(port)],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.1)
    yield url
    proc.terminate()
    proc.wait(10)


@pytest.fixture(scope="module")
def page(server):
    with sync_api.sync_playwright() as pw:
        try:
            browser = pw.chromium.launch()
        except Exception as e:  # browsers not installed
            pytest.skip(f"Chromium unavailable: {e}")
        pg = browser.new_page(viewport={"width": 1300, "height": 1100})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(server)
        pg.wait_for_selector("#conn.ok", timeout=30000)
        pg.errors = errors
        yield pg
        browser.close()


def _wait_extract(pg):
    pg.wait_for_function("document.querySelector('#extract-info').textContent.match(/điểm|Lỗi/)", timeout=15000)
    return pg.inner_text("#extract-info")


def test_canvas_to_robot(page):
    box = page.locator("#pad").bounding_box()
    P = lambda u, v: (box["x"] + u * box["width"], box["y"] + v * box["height"])
    for stroke in ([(0.2, 0.85), (0.5, 0.15), (0.8, 0.85)], [(0.33, 0.55), (0.67, 0.55)]):
        page.mouse.move(*P(*stroke[0]))
        page.mouse.down()
        for q in stroke[1:]:
            page.mouse.move(*P(*q), steps=8)
        page.mouse.up()
    page.click("#extract")
    assert "nét" in _wait_extract(page)
    page.fill("#speed", "6")
    page.dispatch_event("#speed", "input")
    page.click("#draw")
    page.wait_for_function("document.querySelector('#robot-info').textContent.includes('Hoàn thành')", timeout=90000)
    assert "RMS" in page.inner_text("#robot-info")
    assert not page.errors


def test_hand_pinch_gestures(page):
    page.click("[data-tab=cam]")
    r = page.evaluate(SYNTH)
    assert len(r["lens"]) == 2 and min(r["lens"]) > 10, r
    assert not r["down"]
    # x is mirrored (selfie view): un-mirrored 0.6 -> 0.4
    assert abs(r["first"][0] - 0.4) < 0.02 and r["last"][1] > r["first"][1] + 0.4
    page.click("#extract")
    assert "2 nét" in _wait_extract(page)
    assert not page.errors


def test_mediapipe_model_loads_and_runs(page):
    ok = page.evaluate(
        """async () => {
            await window.__air.loadModel();
            const c = document.createElement('canvas'); c.width = 320; c.height = 240;
            const g = c.getContext('2d'); g.fillStyle = '#888'; g.fillRect(0, 0, 320, 240);
            await window.__air.landmarker.setOptions({runningMode: 'IMAGE'});
            const res = window.__air.landmarker.detect(c);
            await window.__air.landmarker.setOptions({runningMode: 'VIDEO'});
            return Array.isArray(res.landmarks) && res.landmarks.length === 0;
        }"""
    )
    assert ok
    assert not page.errors
