"""FastAPI server: OpenCV stroke extraction + streaming PyBullet robot drawing.

HTTP
  GET  /                 single-page web UI
  GET  /api/health       liveness probe
  POST /api/extract      image (data URL) -> strokes + debug images

WebSocket /ws/draw
  client -> server (JSON):
    {"type": "start", "strokes": [[[x, y], ...], ...], "speed": 1.0,
     "camera": "perspective", "shadows": false, "show_plan": true}
    {"type": "stop"}
    {"type": "camera", "name": "top"}       (works while idle or drawing)
    {"type": "reset", "strokes": [...]}     (idle only: clear ink, preview plan)
  server -> client:
    binary messages: JPEG frames of the simulation
    JSON: {"type": "ready"} | {"type": "progress", ...} | {"type": "done", "stats": {...}}
          | {"type": "error", "message": "..."}
"""
from __future__ import annotations

import asyncio
import logging
import mimetypes
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .robot_sim import CAMERAS, DrawingSim
from .trajectory import TrajectoryConfig, build_trajectory, world_to_unit
from .vision import NoCharacterFound, decode_image, encode_png_b64, extract_strokes, render_strokes

log = logging.getLogger("robot_draw")
# MediaPipe (vendored for the hand-keypoint webcam mode) needs correct types.
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/octet-stream", ".task")
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
MAX_POINTS = 4000
MAX_STROKES = 60

app = FastAPI(title="Robot Character Drawing")
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


class ExtractRequest(BaseModel):
    image: str = Field(..., description="data URL or base64 encoded PNG/JPEG")
    mode: str = Field("canvas", pattern="^(canvas|photo)$")
    method: str = Field("skeleton", pattern="^(skeleton|outline)$")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "cameras": list(CAMERAS)}


@app.post("/api/extract")
def extract(req: ExtractRequest):
    try:
        img = decode_image(req.image)
    except Exception:
        raise HTTPException(400, "Không đọc được ảnh (invalid image)")
    t0 = time.perf_counter()
    try:
        res = extract_strokes(img, req.mode, req.method)
    except NoCharacterFound as e:
        raise HTTPException(422, f"Không tìm thấy ký tự: {e}")
    stats = dict(res.stats, ms=round((time.perf_counter() - t0) * 1000, 1))
    return {
        "strokes": [[[round(x, 4), round(y, 4)] for x, y in s] for s in res.strokes],
        "stats": stats,
        "images": {
            "binary": encode_png_b64(res.binary),
            "skeleton": encode_png_b64(res.skeleton),
            "strokes": encode_png_b64(render_strokes(res.strokes, 256)),
        },
    }


def _validate_strokes(raw) -> list[list[tuple[float, float]]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("strokes must be a non-empty list")
    if len(raw) > MAX_STROKES:
        raise ValueError(f"too many strokes (max {MAX_STROKES})")
    out, total = [], 0
    for s in raw:
        pts = np.asarray(s, float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 1 or not np.isfinite(pts).all():
            raise ValueError("each stroke must be a list of [x, y] pairs")
        total += len(pts)
        out.append([(float(x), float(y)) for x, y in np.clip(pts, 0.0, 1.0)])
    if total > MAX_POINTS:
        raise ValueError(f"too many points (max {MAX_POINTS})")
    return out


def _jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return buf.tobytes()


@app.websocket("/ws/draw")
async def ws_draw(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()
    cfg = TrajectoryConfig()
    sim: DrawingSim | None = None
    out_q: asyncio.Queue = asyncio.Queue(maxsize=3)
    stop_event = threading.Event()
    job: asyncio.Future | None = None
    sim_lock = threading.Lock()  # one trajectory at a time per connection

    async def sender():
        while True:
            item = await out_q.get()
            if isinstance(item, bytes):
                await ws.send_bytes(item)
            else:
                await ws.send_json(item)

    def put(item):
        """Thread-safe, back-pressured enqueue from the simulation thread."""
        fut = asyncio.run_coroutine_threadsafe(out_q.put(item), loop)
        fut.result()

    def run_job(strokes, speed, show_plan):
        with sim_lock:
            traj = build_trajectory(strokes, cfg)
            put({"type": "started", "duration": round(traj.duration, 2), "steps": len(traj)})

            def on_frame(frame, info):
                ink = [world_to_unit(np.asarray(pl), cfg).round(4).tolist() for pl in info.new_ink if pl]
                put(
                    {
                        "type": "progress",
                        "step": info.step,
                        "total": info.total,
                        "t": round(info.t, 2),
                        "pen_down": info.pen_down,
                        "stroke": info.stroke_id,
                        "tip": [round(v, 4) for v in info.tip],
                        "ink": ink,
                    }
                )
                put(_jpeg(frame))

            stats = sim.run(traj, on_frame, speed=speed, stop_event=stop_event, show_plan=show_plan)
            put({"type": "done", "stats": stats})

    def idle_frame():
        with sim._lock:
            return _jpeg(sim.render())

    send_task = asyncio.create_task(sender())
    try:
        sim = await loop.run_in_executor(None, lambda: DrawingSim(cfg))
        await out_q.put({"type": "ready", "cameras": list(CAMERAS)})
        await out_q.put(await loop.run_in_executor(None, idle_frame))
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")
            if kind == "start":
                if job is not None and not job.done():
                    await out_q.put({"type": "error", "message": "Robot đang vẽ, hãy dừng trước."})
                    continue
                try:
                    strokes = _validate_strokes(msg.get("strokes"))
                except ValueError as e:
                    await out_q.put({"type": "error", "message": str(e)})
                    continue
                speed = float(np.clip(float(msg.get("speed", 1.0)), 0.25, 8.0))
                with sim._lock:
                    sim.shadows = bool(msg.get("shadows", False))
                    sim.set_camera(str(msg.get("camera", sim.camera)))
                stop_event.clear()
                job = loop.run_in_executor(None, run_job, strokes, speed, bool(msg.get("show_plan", True)))
                job.add_done_callback(lambda f: f.exception() and log.error("job failed: %r", f.exception()))
            elif kind == "stop":
                stop_event.set()
            elif kind == "camera":
                with sim._lock:
                    sim.set_camera(str(msg.get("name", "perspective")))
                    if "shadows" in msg:
                        sim.shadows = bool(msg["shadows"])
                if job is None or job.done():
                    await out_q.put(await loop.run_in_executor(None, idle_frame))
            elif kind == "reset":
                # Clear the paper; optionally preview the planned strokes on it.
                if job is None or job.done():
                    plan = []
                    if msg.get("strokes"):
                        try:
                            traj = build_trajectory(_validate_strokes(msg["strokes"]), cfg)
                            plan = sim.plan_from(traj)
                        except ValueError as e:
                            await out_q.put({"type": "error", "message": str(e)})
                    with sim._lock:
                        sim.ink, sim.plan = [], plan
                    await out_q.put(await loop.run_in_executor(None, idle_frame))
    except WebSocketDisconnect:
        pass
    except Exception as e:  # pragma: no cover - logged for diagnosis
        log.exception("websocket error")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        stop_event.set()
        if job is not None:
            try:
                # unblock a producer waiting on a full queue, then let it finish
                while not job.done():
                    while not out_q.empty():
                        out_q.get_nowait()
                    await asyncio.sleep(0.02)
            except Exception:
                pass
        send_task.cancel()
        if sim is not None:
            sim.close()
