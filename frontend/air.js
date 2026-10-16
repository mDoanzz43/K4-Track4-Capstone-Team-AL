"use strict";
/*
 * Air drawing with hand keypoints.
 *
 * MediaPipe Hand Landmarker (WASM, CPU delegate, vendored under
 * /static/vendor/mediapipe) returns 21 keypoints per frame. The pen position
 * is the midpoint of the thumb tip (4) and index tip (8); the pen is DOWN while
 * those two are pinched together. The pinch distance is normalised by the hand
 * size (wrist 0 -> middle-finger MCP 9) so it works at any distance from the
 * camera, and uses hysteresis + a short debounce so the pen does not flicker.
 * Positions are smoothed with a One-Euro filter (low jitter when still, low lag
 * when moving fast).
 *
 * The resulting strokes are rasterised onto a white square canvas and go
 * through exactly the same OpenCV pipeline as the drawing pad.
 */
(function () {
  const MP_BASE = "/static/vendor/mediapipe";
  const PINCH_DOWN = 0.32; // pinch / hand-size below which the pen goes down
  const PINCH_UP = 0.48; // ... and above which it lifts again (hysteresis)
  const DEBOUNCE = 2; // consecutive frames needed to change pen state
  const MIN_STEP = 0.003; // min movement (normalised) to record a new point

  // Keypoint pairs to draw the hand skeleton.
  const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
  ];

  class LowPass {
    constructor() { this.y = null; }
    filter(x, a) { this.y = this.y === null ? x : a * x + (1 - a) * this.y; return this.y; }
  }
  /** One-Euro filter (Casiez et al. 2012) for one scalar signal. */
  class OneEuro {
    constructor(minCutoff = 1.5, beta = 8.0, dCutoff = 1.0) {
      Object.assign(this, { minCutoff, beta, dCutoff });
      this.reset();
    }
    reset() { this.x = new LowPass(); this.dx = new LowPass(); this.tPrev = null; }
    static alpha(cutoff, dt) { const tau = 1 / (2 * Math.PI * cutoff); return 1 / (1 + tau / dt); }
    filter(x, tSec) {
      if (this.tPrev === null) { this.tPrev = tSec; this.dx.filter(0, 1); return this.x.filter(x, 1); }
      const dt = Math.max(1e-3, tSec - this.tPrev);
      this.tPrev = tSec;
      const prev = this.x.y;
      const dx = this.dx.filter((x - prev) / dt, OneEuro.alpha(this.dCutoff, dt));
      const cutoff = this.minCutoff + this.beta * Math.abs(dx);
      return this.x.filter(x, OneEuro.alpha(cutoff, dt));
    }
  }

  /** Pure gesture -> strokes logic (no DOM, testable with synthetic landmarks). */
  class AirPen {
    constructor(aspect = 4 / 3) {
      this.aspect = aspect; // width / height of the video, to measure distances isotropically
      this.strokes = [];
      this.fx = new OneEuro();
      this.fy = new OneEuro();
      this.reset();
    }
    reset() {
      this.down = false;
      this.counter = 0;
      this.current = null;
      this.cursor = null;
      this.pinch = null;
      this.fx.reset();
      this.fy.reset();
    }
    clear() { this.strokes = []; this.reset(); }
    undo() { this.strokes.pop(); this.current = null; }
    _dist(a, b) { return Math.hypot((a.x - b.x) * this.aspect, a.y - b.y); }

    /**
     * @param landmarks array of 21 {x, y} in [0, 1] image coords (un-mirrored), or null
     * @param tMs timestamp in ms
     * @param forceDown pen forced down (keyboard)
     */
    update(landmarks, tMs, forceDown = false) {
      if (!landmarks) {
        // Hand lost: finish the stroke, forget the filter state.
        this._penUp();
        this.reset();
        return this.state();
      }
      const scale = Math.max(1e-3, this._dist(landmarks[0], landmarks[9]));
      this.pinch = this._dist(landmarks[4], landmarks[8]) / scale;
      const t = tMs / 1000;
      // Mirror x so the drawing matches the selfie-style preview.
      const mx = 1 - (landmarks[4].x + landmarks[8].x) / 2;
      const my = (landmarks[4].y + landmarks[8].y) / 2;
      this.cursor = { x: this.fx.filter(mx, t), y: this.fy.filter(my, t) };

      const wantDown = forceDown || (this.down ? this.pinch < PINCH_UP : this.pinch < PINCH_DOWN);
      if (wantDown !== this.down) {
        this.counter += 1;
        if (this.counter >= DEBOUNCE || forceDown) {
          this.counter = 0;
          if (wantDown) this._penDown(); else this._penUp();
        }
      } else {
        this.counter = 0;
      }
      if (this.down) {
        const last = this.current[this.current.length - 1];
        if (!last || Math.hypot((this.cursor.x - last[0]) * this.aspect, this.cursor.y - last[1]) > MIN_STEP) {
          this.current.push([this.cursor.x, this.cursor.y]);
        }
      }
      return this.state();
    }
    _penDown() {
      this.down = true;
      this.current = [];
      this.strokes.push(this.current);
    }
    _penUp() {
      if (this.down && this.current && this.current.length === 0) this.strokes.pop();
      this.down = false;
      this.current = null;
    }
    state() { return { down: this.down, cursor: this.cursor, pinch: this.pinch, strokes: this.strokes.length }; }
    hasInk() { return this.strokes.some((s) => s.length > 0); }

    /** Rasterise the strokes, fitted and centred, onto a white square canvas (PNG data URL). */
    toImage(size = 400, thickness = 14) {
      const pts = this.strokes.flat().map(([x, y]) => [x * this.aspect, y]);
      const c = document.createElement("canvas");
      c.width = c.height = size;
      const g = c.getContext("2d");
      g.fillStyle = "#fff";
      g.fillRect(0, 0, size, size);
      if (!pts.length) return c.toDataURL("image/png");
      const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
      const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
      const span = Math.max(x1 - x0, y1 - y0, 1e-3);
      const inner = size - 2 * (thickness + 20);
      const s = inner / span;
      const ox = (size - (x1 - x0) * s) / 2, oy = (size - (y1 - y0) * s) / 2;
      g.strokeStyle = "#000";
      g.lineWidth = thickness;
      g.lineCap = g.lineJoin = "round";
      for (const st of this.strokes) {
        if (!st.length) continue;
        g.beginPath();
        st.forEach(([x, y], i) => {
          const px = ox + (x * this.aspect - x0) * s, py = oy + (y - y0) * s;
          i ? g.lineTo(px, py) : g.moveTo(px, py);
        });
        if (st.length === 1) { const [x, y] = st[0]; g.lineTo(ox + (x * this.aspect - x0) * s + 0.1, oy + (y - y0) * s); }
        g.stroke();
      }
      return c.toDataURL("image/png");
    }
  }

  /** Webcam + MediaPipe + overlay controller. */
  class AirDrawer {
    constructor({ video, overlay, wrap, status, penState }) {
      Object.assign(this, { video, overlay, wrap, status, penState });
      this.pen = new AirPen();
      this.stream = null;
      this.landmarker = null;
      this.lastVideoTime = -1;
      this.keyDown = false;
      this.fps = 0;
      this.raf = null;
      this.lastLandmarks = null;
      // Space = pen down while the webcam runs. preventDefault on both events so a
      // focused button (e.g. "Tắt webcam" right after clicking it) is not activated.
      const spaceKey = (e) => e.code === "Space" && this.stream && !/INPUT|SELECT|TEXTAREA/.test(e.target.tagName);
      window.addEventListener("keydown", (e) => {
        if (!spaceKey(e)) return;
        e.preventDefault();
        this.keyDown = true;
      });
      window.addEventListener("keyup", (e) => {
        if (e.code === "Space") this.keyDown = false;
        if (spaceKey(e)) e.preventDefault();
      });
      this.draw();
    }
    get running() { return !!this.stream; }

    async loadModel() {
      if (this.landmarker) return;
      this.setStatus("Đang tải mô hình nhận diện tay…");
      const { FilesetResolver, HandLandmarker } = await import(`${MP_BASE}/vision_bundle.mjs`);
      const fileset = await FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
      this.landmarker = await HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: `${MP_BASE}/hand_landmarker.task`, delegate: "CPU" },
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.6,
        minHandPresenceConfidence: 0.6,
        minTrackingConfidence: 0.5,
      });
    }

    async start() {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("Trình duyệt không hỗ trợ webcam (cần https hoặc localhost)");
      const modelReady = this.loadModel();
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });
      this.video.srcObject = this.stream;
      await new Promise((r) => (this.video.readyState >= 1 ? r() : (this.video.onloadedmetadata = r)));
      const vw = this.video.videoWidth || 640, vh = this.video.videoHeight || 480;
      this.wrap.style.aspectRatio = `${vw} / ${vh}`;
      this.overlay.width = vw;
      this.overlay.height = vh;
      this.pen.aspect = vw / vh;
      await modelReady;
      this.setStatus("Sẵn sàng – giơ tay lên");
      this.loop();
    }

    stop() {
      this.stream?.getTracks().forEach((t) => t.stop());
      this.stream = null;
      cancelAnimationFrame(this.raf);
      this.pen.update(null, performance.now());
      this.lastLandmarks = null;
      this.setStatus("Webcam đang tắt");
      this.draw();
      this.showPen();
    }

    loop() {
      if (!this.stream) return;
      const now = performance.now();
      if (this.video.readyState >= 2 && this.video.currentTime !== this.lastVideoTime) {
        this.lastVideoTime = this.video.currentTime;
        const t0 = performance.now();
        const res = this.landmarker.detectForVideo(this.video, now);
        const ms = performance.now() - t0;
        this.fps = 0.9 * this.fps + 0.1 * (1000 / Math.max(16, ms));
        this.feed(res.landmarks && res.landmarks.length ? res.landmarks[0] : null, now);
        this.setStatus(
          (this.lastLandmarks ? "Đã thấy tay" : "Không thấy tay") +
          ` · ${ms.toFixed(0)} ms/khung` +
          (this.pen.pinch !== null && this.lastLandmarks ? ` · chụm ${this.pen.pinch.toFixed(2)}` : "")
        );
      }
      this.raf = requestAnimationFrame(() => this.loop());
    }

    /** Feed one frame of landmarks (also used by automated tests). */
    feed(landmarks, tMs) {
      this.lastLandmarks = landmarks;
      this.pen.update(landmarks, tMs, this.keyDown);
      this.draw();
      this.showPen();
    }

    showPen() {
      const d = this.pen.down;
      this.penState.textContent = d ? "✍️ Đang vẽ" : "✋ Nhấc bút";
      this.penState.classList.toggle("down", d);
    }
    setStatus(t) { this.status.textContent = t; }

    draw() {
      const g = this.overlay.getContext("2d");
      const W = this.overlay.width, H = this.overlay.height;
      g.clearRect(0, 0, W, H);
      if (!this.stream) {
        g.fillStyle = "#111";
        g.fillRect(0, 0, W, H);
      }
      // strokes drawn so far
      g.lineCap = g.lineJoin = "round";
      g.lineWidth = Math.max(4, W / 90);
      g.strokeStyle = "#ffd400";
      g.shadowColor = "rgba(0,0,0,.6)";
      g.shadowBlur = 4;
      for (const st of this.pen.strokes) {
        if (!st.length) continue;
        g.beginPath();
        st.forEach(([x, y], i) => (i ? g.lineTo(x * W, y * H) : g.moveTo(x * W, y * H)));
        if (st.length === 1) g.lineTo(st[0][0] * W + 0.1, st[0][1] * H);
        g.stroke();
      }
      g.shadowBlur = 0;
      // hand skeleton (mirrored to match the preview)
      const lm = this.lastLandmarks;
      if (lm) {
        g.strokeStyle = "rgba(0, 220, 255, .9)";
        g.lineWidth = 2;
        for (const [a, b] of HAND_CONNECTIONS) {
          g.beginPath();
          g.moveTo((1 - lm[a].x) * W, lm[a].y * H);
          g.lineTo((1 - lm[b].x) * W, lm[b].y * H);
          g.stroke();
        }
        g.fillStyle = "#fff";
        for (const p of lm) {
          g.beginPath();
          g.arc((1 - p.x) * W, p.y * H, 3, 0, 2 * Math.PI);
          g.fill();
        }
      }
      // pen cursor
      const c = this.pen.cursor;
      if (lm && c) {
        g.beginPath();
        g.arc(c.x * W, c.y * H, this.pen.down ? 9 : 12, 0, 2 * Math.PI);
        g.lineWidth = 3;
        g.strokeStyle = this.pen.down ? "#ff3b30" : "#ffffff";
        g.fillStyle = this.pen.down ? "rgba(255,59,48,.6)" : "rgba(255,255,255,.15)";
        g.fill();
        g.stroke();
      }
    }
  }

  window.AirDraw = { AirPen, AirDrawer, OneEuro, HAND_CONNECTIONS, PINCH_DOWN, PINCH_UP };
})();
