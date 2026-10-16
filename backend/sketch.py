"""Turn a photo (portrait, object, scene) into a pen sketch the robot can draw.

Pipeline (all OpenCV, CPU):
1. Optional face-centred crop with the YuNet face detector (cv2.FaceDetectorYN,
   a 230 kB ONNX model shipped in backend/models) so the robot draws the person
   instead of the cluttered background.
2. Contrast normalisation (CLAHE) + edge-preserving smoothing (bilateral).
3. Line art: adaptive (local mean) threshold -> thin dark lines around facial
   features, hair, clothes; tiny specks removed. The line mask then goes through
   the same skeleton -> graph -> stroke pipeline as characters.
4. Optional tone hatching: dark / very dark regions (relative to the image's own
   brightness distribution) are filled with parallel strokes at 45 deg and
   cross-hatched at -45 deg, like a pencil drawing.

`detail` in [0, 1] trades drawing time for fidelity (more lines, denser hatching).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .vision import NoCharacterFound, _resize_max, finalize_strokes, strokes_from_mask

MODEL = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
_detector = None


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * float(np.clip(t, 0.0, 1.0))


def detect_face(img: np.ndarray):
    """Largest face as (x, y, w, h) in `img` pixels, or None."""
    global _detector
    if not MODEL.exists():
        return None
    small = _resize_max(img, 640)
    s = img.shape[1] / small.shape[1]
    h, w = small.shape[:2]
    try:
        if _detector is None:
            _detector = cv2.FaceDetectorYN.create(str(MODEL), "", (w, h), 0.7, 0.3, 50)
        _detector.setInputSize((w, h))
        _, faces = _detector.detect(small)
    except cv2.error:
        return None
    if faces is None or len(faces) == 0:
        return None
    x, y, fw, fh = max(faces[:, :4].tolist(), key=lambda f: f[2] * f[3])
    return tuple(int(round(v * s)) for v in (x, y, fw, fh))


def face_crop_box(shape, face):
    """Head-and-shoulders box around a face, clipped to the image."""
    H, W = shape[:2]
    x, y, fw, fh = face
    cx, cy = x + fw / 2, y + fh / 2 + 0.25 * fh
    bw, bh = 2.3 * fw, 2.9 * fh
    x0, y0 = int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2))
    x1, y1 = int(min(W, cx + bw / 2)), int(min(H, cy + bh / 2))
    return x0, y0, x1 - x0, y1 - y0


def person_mask(img: np.ndarray, face) -> np.ndarray:
    """Foreground (head + shoulders) mask via GrabCut seeded from the face box.

    Seeds: a definite-foreground ellipse inside the face, a probable-foreground
    head ellipse and a widening torso trapezoid below it, and definite
    background along the left/right/top borders away from the head.
    """
    h, w = img.shape[:2]
    fx, fy, fw, fh = face
    cx = fx + fw / 2
    m = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    head_c = (int(cx), int(fy + 0.45 * fh))
    cv2.ellipse(m, head_c, (int(0.8 * fw), int(0.85 * fh)), 0, 0, 360, cv2.GC_PR_FGD, -1)
    torso = np.array(
        [[cx - 0.55 * fw, fy + 1.05 * fh], [cx + 0.55 * fw, fy + 1.05 * fh], [cx + 1.6 * fw, h], [cx - 1.6 * fw, h]], np.int32
    )
    cv2.fillPoly(m, [torso], cv2.GC_PR_FGD)
    b = max(3, w // 60)
    border = np.zeros((h, w), bool)
    border[:, :b] = border[:, -b:] = True
    border[:b, :] = True
    m[border & (m == cv2.GC_PR_BGD)] = cv2.GC_BGD
    cv2.ellipse(m, (int(cx), int(fy + 0.5 * fh)), (int(0.3 * fw), int(0.35 * fh)), 0, 0, 360, cv2.GC_FGD, -1)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(img, m, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    fg = np.isin(m, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return np.full((h, w), 255, np.uint8)
    # Keep every sizeable part (head, torso, arms may be separated by GrabCut).
    areas = stats[:, cv2.CC_STAT_AREA].copy()
    areas[0] = 0
    keep = areas >= 0.1 * areas.max()
    fg = (keep[labels] * 255).astype(np.uint8)
    # fill holes
    inv = cv2.bitwise_not(fg)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    for i in range(1, n):
        x, y, cw, ch, _ = stats[i]
        if x > 0 and y > 0 and x + cw < w and y + ch < h:
            fg[labels == i] = 255
    return fg


def silhouette_strokes(fg: np.ndarray, min_len: float):
    """Outer contour of the foreground as (y, x) strokes, split where it runs
    along the image border (that is the frame, not the person)."""
    h, w = fg.shape
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in contours:
        pts = c.reshape(-1, 2)
        on_border = (pts[:, 0] <= 1) | (pts[:, 1] <= 1) | (pts[:, 0] >= w - 2) | (pts[:, 1] >= h - 2)
        run = []
        for p, ob in zip(pts, on_border):
            if ob:
                if len(run) > 1:
                    out.append(np.array(run, float))
                run = []
            else:
                run.append((p[1], p[0]))
        if len(run) > 1:
            out.append(np.array(run, float))
    return [r for r in out if np.linalg.norm(np.diff(r, axis=0), axis=1).sum() >= min_len]


def _remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return (keep[labels] * 255).astype(np.uint8)


def hatch(mask: np.ndarray, angle: float, spacing: int, min_len: int, phase: int = 0):
    """Parallel hatch segments filling `mask`, as (y, x) 2-point strokes.

    The mask is rotated so the hatch direction becomes horizontal, rows are
    scanned every `spacing` px, each run of mask pixels becomes a segment, and
    its end points are rotated back. Alternate rows are reversed so the
    nearest-neighbour ordering naturally produces a zig-zag (short pen-up hops).
    """
    h, w = mask.shape
    c = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(c, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    M[0, 2] += nw / 2 - c[0]
    M[1, 2] += nh / 2 - c[1]
    rot = cv2.warpAffine(mask, M, (nw, nh), flags=cv2.INTER_NEAREST)
    Minv = cv2.invertAffineTransform(M)
    segs = []
    for k, y in enumerate(range(spacing // 2 + phase, nh, spacing)):
        row = np.concatenate([[0], (rot[y] > 0).astype(np.int8), [0]])
        d = np.diff(row)
        starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1
        for x0, x1 in zip(starts, ends):
            if x1 - x0 < min_len:
                continue
            pts = np.array([[x0 + 1, y, 1.0], [x1 - 1, y, 1.0]]) @ Minv.T  # (x, y) in the original image
            if k % 2:
                pts = pts[::-1]
            segs.append(pts[:, ::-1])  # -> (y, x)
    return segs


def sketch_strokes(
    img: np.ndarray, detail: float = 0.5, hatching: bool = True, face_crop: bool = True, remove_bg: bool = True
):
    """BGR photo -> ExtractionResult with line-art (+ hatching) pen strokes.

    face_crop: crop head-and-shoulders around the largest detected face.
    remove_bg: when a face is found, segment the person (GrabCut) and only
               draw inside it, plus its silhouette.
    """
    img = _resize_max(img, 1400)
    face = detect_face(img) if (face_crop or remove_bg) else None
    if face is not None and face_crop:
        x, y, w, h = face_crop_box(img.shape, face)
        img = img[y : y + h, x : x + w]
        bbox = (x, y, w, h)
        face = (face[0] - x, face[1] - y, face[2], face[3])
    else:
        bbox = (0, 0, img.shape[1], img.shape[0])

    work = int(_lerp(320, 480, detail))
    s = work / max(img.shape[:2])
    img = cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    if face is not None:
        face = tuple(v * s for v in face)
    fg = person_mask(img, face) if (remove_bg and face is not None) else None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.std() < 4:
        raise NoCharacterFound("Ảnh gần như đồng màu, không có gì để vẽ")
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    smooth = cv2.bilateralFilter(gray, 9, 40, 7)

    # ---- line art
    C = _lerp(11, 5, detail)  # lower C -> more (fainter) lines
    lines = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, C)
    lines = cv2.morphologyEx(lines, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if fg is not None:
        lines = cv2.bitwise_and(lines, cv2.erode(fg, np.ones((3, 3), np.uint8)))
    lines = _remove_small(lines, int(_lerp(60, 15, detail)))
    lines = cv2.copyMakeBorder(lines, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    raw, sk = strokes_from_mask(lines, "skeleton", spur_len=_lerp(10, 5, detail))
    raw = [np.asarray(r, float) - 2 for r in raw]  # undo the border
    sk = sk[2:-2, 2:-2]
    min_line = _lerp(22, 8, detail)
    raw = [r for r in raw if len(r) > 1 and np.linalg.norm(np.diff(r, axis=0), axis=1).sum() >= min_line]
    if fg is not None:
        raw = silhouette_strokes(fg, min_line) + raw
    n_lines = len(raw)

    # ---- tone hatching
    hatch_segs = []
    tone_vis = np.zeros_like(gray)
    if hatching:
        tone = cv2.GaussianBlur(gray, (0, 0), 3)
        spacing = int(round(_lerp(10, 5, detail)))
        min_len = max(4, spacing)
        for q, angle, phase in ((0.30, 45, 0), (0.12, -45, spacing // 2)):
            t = np.quantile(tone, q)
            m = ((tone < t) * 255).astype(np.uint8)
            if fg is not None:
                m = cv2.bitwise_and(m, cv2.erode(fg, np.ones((7, 7), np.uint8)))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            m = _remove_small(m, 150)
            tone_vis = np.maximum(tone_vis, (m > 0).astype(np.uint8) * (90 if q > 0.2 else 160))
            hatch_segs += hatch(m, angle, spacing, min_len, phase)
    raw = raw + hatch_segs
    if not raw:
        raise NoCharacterFound("Không trích được nét nào từ ảnh (thử tăng mức chi tiết)")

    # Debug views: line mask + tone regions, and the skeleton + hatch lines.
    binary = np.maximum(lines[2:-2, 2:-2], tone_vis)
    skel = sk.copy().astype(np.uint8)
    for seg in hatch_segs:
        p = np.round(seg[:, ::-1]).astype(np.int32)
        cv2.line(skel, tuple(p[0]), tuple(p[1]), 1, 1)
    return finalize_strokes(
        raw,
        binary=binary,
        skeleton=skel.astype(bool),
        bbox=bbox,
        method="sketch",
        min_len=0,
        eps=1.0,
        extra_stats={
            "face_found": face is not None,
            "background_removed": fg is not None,
            "line_strokes": n_lines,
            "hatch_strokes": len(hatch_segs),
        },
    )
