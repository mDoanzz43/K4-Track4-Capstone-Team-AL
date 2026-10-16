"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  tab: "draw",
  strokes: null,        // extracted strokes (unit box)
  ws: null,
  running: false,
  frameUrl: null,
  inkPolys: [],         // polylines received while drawing
};

/* ------------------------------------------------------------------ tabs */
document.querySelectorAll(".tab").forEach((btn) =>
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
    state.tab = btn.dataset.tab;
    $("tab-draw").classList.toggle("hidden", state.tab !== "draw");
    $("tab-cam").classList.toggle("hidden", state.tab !== "cam");
  })
);

/* ------------------------------------------------------------ draw pad */
const pad = $("pad");
const pctx = pad.getContext("2d");
let padStrokes = [];
let current = null;

function redrawPad() {
  pctx.fillStyle = "#fff";
  pctx.fillRect(0, 0, pad.width, pad.height);
  pctx.strokeStyle = "#000";
  pctx.lineCap = "round";
  pctx.lineJoin = "round";
  for (const s of padStrokes) {
    pctx.lineWidth = s.w;
    pctx.beginPath();
    s.pts.forEach(([x, y], i) => (i ? pctx.lineTo(x, y) : pctx.moveTo(x, y)));
    if (s.pts.length === 1) pctx.lineTo(s.pts[0][0] + 0.1, s.pts[0][1]);
    pctx.stroke();
  }
}
function padPos(e) {
  const r = pad.getBoundingClientRect();
  return [((e.clientX - r.left) * pad.width) / r.width, ((e.clientY - r.top) * pad.height) / r.height];
}
pad.addEventListener("pointerdown", (e) => {
  pad.setPointerCapture(e.pointerId);
  current = { w: +$("brush").value, pts: [padPos(e)] };
  padStrokes.push(current);
  redrawPad();
});
pad.addEventListener("pointermove", (e) => {
  if (!current) return;
  current.pts.push(padPos(e));
  redrawPad();
});
["pointerup", "pointercancel", "pointerleave"].forEach((ev) => pad.addEventListener(ev, () => (current = null)));
$("undo").onclick = () => { padStrokes.pop(); redrawPad(); };
$("clear").onclick = () => { padStrokes = []; redrawPad(); };
redrawPad();

/* ------------------------------------------------- webcam: hand keypoints */
const air = new AirDraw.AirDrawer({
  video: $("video"),
  overlay: $("air-overlay"),
  wrap: $("air-wrap"),
  status: $("air-status"),
  penState: $("pen-state"),
});
window.__air = air; // handle for automated tests

$("cam-start").onclick = async () => {
  if (air.running) {
    air.stop();
    $("cam-start").textContent = "Bật webcam";
    return;
  }
  $("cam-start").disabled = true;
  try {
    await air.start();
    $("cam-start").textContent = "Tắt webcam";
  } catch (err) {
    air.stop();
    setInfo("extract-info", "Không mở được webcam / mô hình tay: " + err.message, "err");
  } finally {
    $("cam-start").disabled = false;
  }
};
$("air-undo").onclick = () => { air.pen.undo(); air.draw(); };
$("air-clear").onclick = () => { air.pen.clear(); air.draw(); };

/* ----------------------------------------------------------- extraction */
function setInfo(id, text, cls = "") {
  const el = $(id);
  el.textContent = text;
  el.className = "info " + cls;
}

$("extract").onclick = async () => {
  let image, mode;
  if (state.tab === "draw") {
    if (!padStrokes.length) return setInfo("extract-info", "Hãy vẽ một ký tự trước.", "err");
    image = pad.toDataURL("image/png");
    mode = "canvas";
  } else {
    if (!air.pen.hasInk()) return setInfo("extract-info", "Hãy chụm ngón cái + ngón trỏ để viết một ký tự trước.", "err");
    image = air.pen.toImage();
    mode = "canvas"; // the air-drawn strokes are rasterised, then go through the same OpenCV pipeline
  }
  $("extract").disabled = true;
  setInfo("extract-info", "Đang xử lý…");
  try {
    const res = await fetch("/api/extract", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image, mode, method: $("method").value }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    state.strokes = data.strokes;
    $("img-binary").src = data.images.binary;
    $("img-skeleton").src = data.images.skeleton;
    $("img-strokes").src = data.images.strokes;
    const s = data.stats;
    setInfo("extract-info", `${s.num_strokes} nét, ${s.num_points} điểm điều khiển · ${s.ms} ms`, "ok");
    state.inkPolys = [];
    drawTrace();
    send({ type: "reset", strokes: $("plan").checked ? state.strokes : null });
    updateButtons();
  } catch (err) {
    state.strokes = null;
    setInfo("extract-info", "Lỗi: " + err.message, "err");
    updateButtons();
  } finally {
    $("extract").disabled = false;
  }
};

/* --------------------------------------------------------- trace panel */
const trace = $("trace");
const tctx = trace.getContext("2d");
function toTrace([u, v]) {
  const m = 20, s = trace.width - 2 * m;
  return [m + u * s, m + v * s];
}
function polyline(ctx, pts, color, width) {
  if (!pts.length) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = ctx.lineJoin = "round";
  ctx.beginPath();
  pts.forEach((p, i) => { const [x, y] = toTrace(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  if (pts.length === 1) { const [x, y] = toTrace(pts[0]); ctx.lineTo(x + 0.1, y); }
  ctx.stroke();
}
function drawTrace(tip) {
  tctx.fillStyle = "#fff";
  tctx.fillRect(0, 0, trace.width, trace.height);
  if (state.strokes && $("plan").checked) state.strokes.forEach((s) => polyline(tctx, s, "#b9cdf5", 2));
  state.inkPolys.forEach((s) => polyline(tctx, s, "#1c2a7a", 3));
  if (tip) {
    const [x, y] = toTrace(tip.uv);
    tctx.fillStyle = tip.down ? "#d92d20" : "#98a2b3";
    tctx.beginPath();
    tctx.arc(x, y, 5, 0, Math.PI * 2);
    tctx.fill();
  }
}
drawTrace();
$("plan").onchange = () => drawTrace();

/* ------------------------------------------------------------ websocket */
const CHAR_SIZE = 0.2, PAPER = [0.62, 0.0]; // must match backend TrajectoryConfig
function tipToUnit([x, y]) {
  return [-(y - PAPER[1]) / CHAR_SIZE + 0.5, -(x - PAPER[0]) / CHAR_SIZE + 0.5];
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/draw`);
  ws.binaryType = "blob";
  state.ws = ws;
  setConn("Đang khởi tạo mô phỏng…", "");
  ws.onmessage = (ev) => {
    if (typeof ev.data !== "string") {
      const url = URL.createObjectURL(ev.data);
      $("sim").src = url;
      if (state.frameUrl) URL.revokeObjectURL(state.frameUrl);
      state.frameUrl = url;
      return;
    }
    const msg = JSON.parse(ev.data);
    switch (msg.type) {
      case "ready":
        setConn("PyBullet sẵn sàng", "ok");
        $("sim-overlay").classList.add("hidden");
        updateButtons();
        break;
      case "started":
        setInfo("robot-info", `Bắt đầu vẽ · thời lượng mô phỏng ${msg.duration}s`);
        break;
      case "progress": {
        msg.ink.forEach((pl, i) => {
          // Only the first polyline of a batch can continue the previous one
          // (the pen stayed down across the frame boundary).
          const last = state.inkPolys[state.inkPolys.length - 1];
          if (i === 0 && last && last.open) last.push(...pl.slice(1));
          else state.inkPolys.push(Object.assign(pl, { open: true }));
        });
        if (!msg.pen_down) state.inkPolys.forEach((p) => (p.open = false));
        drawTrace({ uv: tipToUnit(msg.tip), down: msg.pen_down });
        $("bar").style.width = `${(100 * msg.step) / msg.total}%`;
        setInfo("robot-info", `t = ${msg.t.toFixed(2)} s · ${msg.pen_down ? "đang vẽ" : "nhấc bút"}` + (msg.stroke >= 0 ? ` · nét ${msg.stroke + 1}` : ""));
        break;
      }
      case "done": {
        state.running = false;
        const s = msg.stats;
        setInfo(
          "robot-info",
          (s.stopped ? "Đã dừng." : "Hoàn thành!") +
            `\nMô phỏng ${s.sim_time}s (thực ${s.wall_time}s)` +
            `\nSai số bám quỹ đạo: RMS ${s.tracking_rms_mm} mm, max ${s.tracking_max_mm} mm`,
          s.stopped ? "" : "ok"
        );
        drawTrace();
        updateButtons();
        break;
      }
      case "error":
        setInfo("robot-info", "Lỗi: " + msg.message, "err");
        state.running = false;
        updateButtons();
        break;
    }
  };
  ws.onclose = () => {
    state.running = false;
    setConn("Mất kết nối – đang thử lại…", "err");
    $("sim-overlay").textContent = "Mất kết nối tới server";
    $("sim-overlay").classList.remove("hidden");
    updateButtons();
    setTimeout(connect, 2000);
  };
}
function setConn(text, cls) {
  const el = $("conn");
  el.textContent = text;
  el.className = "pill " + cls;
}
function wsReady() { return state.ws && state.ws.readyState === WebSocket.OPEN; }
function send(obj) { if (wsReady()) state.ws.send(JSON.stringify(obj)); }

function updateButtons() {
  const ready = wsReady() && $("sim-overlay").classList.contains("hidden");
  $("draw").disabled = !ready || !state.strokes || state.running;
  $("stop").disabled = !state.running;
}

$("speed").oninput = () => ($("speed-val").textContent = `${$("speed").value}×`);
$("speed").oninput();
$("camera").onchange = () => send({ type: "camera", name: $("camera").value, shadows: $("shadows").checked });
$("shadows").onchange = $("camera").onchange;
$("draw").onclick = () => {
  if (!state.strokes) return;
  state.inkPolys = [];
  state.running = true;
  drawTrace();
  updateButtons();
  send({
    type: "start",
    strokes: state.strokes,
    speed: +$("speed").value,
    camera: $("camera").value,
    shadows: $("shadows").checked,
    show_plan: $("plan").checked,
  });
};
$("stop").onclick = () => send({ type: "stop" });

connect();
