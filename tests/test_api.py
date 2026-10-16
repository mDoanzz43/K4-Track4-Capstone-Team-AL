import base64

import cv2
from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


def _data_url(img):
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def test_index_and_health():
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/api/health").json()["ok"]


def test_extract_endpoint(make_text, make_photo):
    r = client.post("/api/extract", json={"image": _data_url(make_text("A")), "mode": "canvas"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["stats"]["num_strokes"] >= 1
    assert data["images"]["skeleton"].startswith("data:image/png;base64,")
    r = client.post("/api/extract", json={"image": _data_url(make_photo()), "mode": "photo"})
    assert r.status_code == 200, r.text


def test_extract_errors(make_text):
    assert client.post("/api/extract", json={"image": "bm90IGFuIGltYWdl", "mode": "canvas"}).status_code == 400
    blank = make_text(" ")
    assert client.post("/api/extract", json={"image": _data_url(blank), "mode": "canvas"}).status_code == 422
    assert client.post("/api/extract", json={"image": _data_url(blank), "mode": "bogus"}).status_code == 422


def test_websocket_draw_session():
    with client.websocket_connect("/ws/draw") as ws:
        assert ws.receive_json()["type"] == "ready"
        assert ws.receive_bytes()[:2] == b"\xff\xd8"  # idle JPEG frame
        ws.send_json({"type": "start", "strokes": [[[0.3, 0.3], [0.7, 0.3]]], "speed": 8, "camera": "top"})
        started = ws.receive_json()
        assert started["type"] == "started"
        frames, inked, done = 0, 0, None
        while done is None:
            try:
                msg = ws.receive_json()
            except Exception:
                frames += 1  # a binary frame
                continue
            if msg["type"] == "progress":
                inked += sum(len(pl) for pl in msg["ink"])
            elif msg["type"] == "done":
                done = msg
        assert frames > 3 and inked > 10
        assert done["stats"]["tracking_rms_mm"] < 1.5

        ws.send_json({"type": "start", "strokes": "nope"})
        assert ws.receive_json()["type"] == "error"


def test_extract_ink_and_sketch_modes(make_text, make_portrait):
    r = client.post("/api/extract", json={"image": _data_url(make_text("B")), "mode": "ink"})
    assert r.status_code == 200, r.text
    assert r.json()["stats"]["est_seconds"] > 1
    r = client.post(
        "/api/extract",
        json={"image": _data_url(make_portrait()), "mode": "sketch", "detail": 0.4, "hatching": True},
    )
    assert r.status_code == 200, r.text
    st = r.json()["stats"]
    assert st["method"] == "sketch" and st["hatch_strokes"] > 0
    bad = client.post("/api/extract", json={"image": _data_url(make_portrait()), "mode": "sketch", "detail": 3})
    assert bad.status_code == 422


def test_websocket_accepts_large_sketch(make_portrait):
    strokes = client.post("/api/extract", json={"image": _data_url(make_portrait()), "mode": "sketch"}).json()["strokes"]
    assert len(strokes) > 60  # above the old limit
    with client.websocket_connect("/ws/draw") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.receive_bytes()
        ws.send_json({"type": "reset", "strokes": strokes})
        assert ws.receive_bytes()[:2] == b"\xff\xd8"  # preview frame with the plan, no error
