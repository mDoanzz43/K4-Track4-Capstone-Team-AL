"""Character detection and stroke extraction with OpenCV.

Pipeline
--------
1. Binarise: canvas images use a fixed threshold; photos use illumination
   normalisation (divide by a heavily blurred background) + Otsu.
2. Clean-up: morphology + connected-component filtering that keeps the main
   character and its nearby parts (dots, diacritics) and drops paper edges / noise.
3. Crop & rescale the character to a fixed working size.
4. Skeletonise (Zhang-Suen thinning + a topology-preserving clean-up pass so the
   skeleton is strictly 1 pixel wide).
5. Convert the skeleton into a graph (endpoints / junction clusters / edges),
   prune short spurs, then chain edges into pen strokes, preferring the
   straightest continuation at junctions.
6. Smooth + simplify (Douglas-Peucker) the strokes and order them greedily to
   minimise pen-up travel.

All output coordinates are normalised to the unit box: x to the right,
y downward, the character centred in [0, 1] x [0, 1] with its aspect ratio kept.
"""
from __future__ import annotations

import base64
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

WORK_SIZE = 200  # longest side of the cropped character, in pixels

# 8-neighbourhood offsets in clockwise order starting at north: (dy, dx)
_N8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


class NoCharacterFound(ValueError):
    pass


@dataclass
class ExtractionResult:
    strokes: list[list[tuple[float, float]]]  # normalised coordinates
    strokes_px: list[np.ndarray]  # same strokes in pixel coordinates of `binary`
    binary: np.ndarray  # cropped binary character (uint8 0/255)
    skeleton: np.ndarray  # cropped skeleton (uint8 0/255)
    bbox: tuple[int, int, int, int]  # x, y, w, h in the (resized) input image
    stats: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Image decoding helpers
# --------------------------------------------------------------------------- #
def decode_image(data: bytes | str) -> np.ndarray:
    """Decode raw bytes or a (data-URL) base64 string into a BGR image.

    Transparent pixels (e.g. from an HTML canvas) are composited onto white.
    """
    if isinstance(data, str):
        if data.startswith("data:"):
            data = data.split(",", 1)[1]
        data = base64.b64decode(data)
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Cannot decode image")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        img = (rgb * alpha + 255.0 * (1 - alpha)).astype(np.uint8)
    return img


def encode_png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


# --------------------------------------------------------------------------- #
# Binarisation and character isolation
# --------------------------------------------------------------------------- #
def _resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def binarize(img: np.ndarray, mode: str) -> np.ndarray:
    """Return a uint8 mask (255 = ink) for dark ink on a light background.

    In "ink" mode (uploaded images of text / line art) the polarity is detected
    automatically, so light-on-dark images (chalk, white calligraphy on black)
    work too.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == "canvas":
        _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        return bw
    if mode == "ink":
        _, probe = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if probe.mean() > 127:  # "ink" covers most of the image -> it is the background
            gray = 255 - gray

    # Photo: flatten uneven lighting by dividing by an estimate of the paper.
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    k = max(15, (max(gray.shape) // 8) | 1)
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    norm = cv2.divide(gray, background, scale=255)
    otsu_t, bw = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Otsu on an almost blank page picks a threshold inside paper noise; require
    # real contrast (ink must be at least ~12% darker than the local paper).
    if otsu_t > 225:
        _, bw = cv2.threshold(norm, 225, 255, cv2.THRESH_BINARY_INV)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return bw


def isolate_character(bw: np.ndarray, mode: str) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Select the ink to draw.

    Canvas: everything the user drew is intentional, so every component except
    single-pixel specks is kept (multiple letters, diacritics, dots...).
    Photo: start from the main component and grow a cluster with every
    reasonably sized component lying close to it (other letters of a word,
    diacritics), so isolated dirt / paper edges far away are ignored.
    """
    h, w = bw.shape
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n <= 1:
        raise NoCharacterFound("No ink found in the image")

    min_area = max(4, int(0.00015 * h * w)) if mode != "canvas" else 4
    cand = []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < min_area:
            continue
        touches = x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1
        # In photos, huge blobs touching the border are paper edges / shadows.
        if mode != "canvas" and touches and (cw > 0.6 * w or ch > 0.6 * h):
            continue
        cand.append(i)
    if not cand:
        raise NoCharacterFound("No character-like region found")

    if mode in ("canvas", "ink"):
        keep = cand
    else:
        # Main component: largest area, with a mild preference for the image centre.
        def score(i):
            cx, cy = centroids[i]
            d = math.hypot(cx / w - 0.5, cy / h - 0.5)
            return stats[i][4] * (1.0 - 0.8 * d)

        main = max(cand, key=score)
        marea = stats[main][4]
        size = max(stats[main][2], stats[main][3])  # typical glyph size
        x0, y0 = stats[main][0], stats[main][1]
        x1, y1 = x0 + stats[main][2], y0 + stats[main][3]
        keep = [main]
        rest = [i for i in cand if i != main and stats[i][4] >= 0.02 * marea]
        gap = 0.5 * size  # letters in a word / diacritics sit closer than this
        grown = True
        while grown:  # grow the cluster until no nearby component remains
            grown = False
            for i in list(rest):
                x, y, cw, ch, _ = stats[i]
                dx = max(x0 - (x + cw), x - x1, 0)
                dy = max(y0 - (y + ch), y - y1, 0)
                if max(dx, dy) <= gap:
                    keep.append(i)
                    rest.remove(i)
                    x0, y0, x1, y1 = min(x0, x), min(y0, y), max(x1, x + cw), max(y1, y + ch)
                    grown = True

    mask = np.isin(labels, keep).astype(np.uint8) * 255
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    return mask, (int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1))


def crop_and_scale(mask: np.ndarray, bbox, work_size: int = WORK_SIZE) -> tuple[np.ndarray, float]:
    """Crop to bbox and rescale so the longest side equals `work_size` (+margin)."""
    x, y, w, h = bbox
    crop = mask[y : y + h, x : x + w]
    s = work_size / max(w, h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA)
    _, crop = cv2.threshold(crop, 127, 255, cv2.THRESH_BINARY)
    m = 6
    crop = cv2.copyMakeBorder(crop, m, m, m, m, cv2.BORDER_CONSTANT, value=0)
    # Fill tiny holes / gaps caused by scanning artefacts.
    crop = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return crop, s


# --------------------------------------------------------------------------- #
# Skeletonisation
# --------------------------------------------------------------------------- #
def zhang_suen_thinning(mask: np.ndarray) -> np.ndarray:
    """Vectorised Zhang-Suen thinning. Input/Output: bool arrays."""
    img = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            P2 = img[:-2, 1:-1]; P3 = img[:-2, 2:]; P4 = img[1:-1, 2:]; P5 = img[2:, 2:]
            P6 = img[2:, 1:-1]; P7 = img[2:, :-2]; P8 = img[1:-1, :-2]; P9 = img[:-2, :-2]
            seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            c = img[1:-1, 1:-1] == 1
            cond = c & (B >= 2) & (B <= 6) & (A == 1)
            if step == 0:
                cond &= (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
            else:
                cond &= (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
            if cond.any():
                img[1:-1, 1:-1][cond] = 0
                changed = True
        if not changed:
            break
    return img[1:-1, 1:-1].astype(bool)


def _build_simple_point_lut() -> np.ndarray:
    """LUT over 8-neighbour configurations: True if the centre pixel can be
    removed without changing topology (8-connected fg / 4-connected bg) and it
    is not an endpoint."""
    lut = np.zeros(256, bool)
    for code in range(256):
        nb = [(code >> i) & 1 for i in range(8)]  # order of _N8
        if sum(nb) < 2:
            continue
        fg = [i for i in range(8) if nb[i]]
        # foreground components among neighbours (8-adjacency)
        seen, comps = set(), 0
        for s in fg:
            if s in seen:
                continue
            comps += 1
            stack = [s]
            seen.add(s)
            while stack:
                a = stack.pop()
                for b in fg:
                    if b not in seen:
                        dy = _N8[a][0] - _N8[b][0]
                        dx = _N8[a][1] - _N8[b][1]
                        if max(abs(dy), abs(dx)) == 1:
                            seen.add(b)
                            stack.append(b)
        if comps != 1:
            continue
        # background components (4-adjacency) that touch a 4-neighbour
        bg = [i for i in range(8) if not nb[i]]
        seen, bcomps = set(), 0
        for s in bg:
            if s in seen or s % 2 == 1:  # start only from 4-neighbours (even idx)
                continue
            bcomps += 1
            stack = [s]
            seen.add(s)
            while stack:
                a = stack.pop()
                for b in bg:
                    if b not in seen:
                        dy = _N8[a][0] - _N8[b][0]
                        dx = _N8[a][1] - _N8[b][1]
                        if abs(dy) + abs(dx) == 1:
                            seen.add(b)
                            stack.append(b)
        if bcomps == 1:
            lut[code] = True
    return lut


_SIMPLE_LUT = _build_simple_point_lut()


def _neighbour_code(sk: np.ndarray, y: int, x: int) -> int:
    code = 0
    for i, (dy, dx) in enumerate(_N8):
        if sk[y + dy, x + dx]:
            code |= 1 << i
    return code


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Return a strictly 1-pixel-wide boolean skeleton (with a 1px zero border)."""
    sk = zhang_suen_thinning(mask > 0)
    sk[0, :] = sk[-1, :] = sk[:, 0] = sk[:, -1] = False
    # Remove remaining staircase pixels sequentially (topology preserving).
    changed = True
    while changed:
        changed = False
        ys, xs = np.nonzero(sk)
        for y, x in zip(ys, xs):
            code = _neighbour_code(sk, y, x)
            # only thin "corner" pixels: those having two 4-neighbours that are
            # themselves 8-adjacent (N&E, E&S, S&W, W&N)
            n4 = [(code >> i) & 1 for i in (0, 2, 4, 6)]
            corner = any(n4[i] and n4[(i + 1) % 4] for i in range(4))
            if corner and _SIMPLE_LUT[code]:
                sk[y, x] = False
                changed = True
    return sk


# --------------------------------------------------------------------------- #
# Skeleton -> graph -> strokes
# --------------------------------------------------------------------------- #
def _neighbours(sk: np.ndarray, p):
    y, x = p
    for dy, dx in _N8:
        if sk[y + dy, x + dx]:
            yield (y + dy, x + dx)


def skeleton_graph(sk: np.ndarray):
    """Decompose a skeleton into nodes and edges.

    Returns (node_pos, edges) where node_pos maps node id -> (y, x) float and
    edges is a list of (n_a, n_b, [ (y, x), ... ]) polylines including both
    node positions. Closed loops without nodes get a synthetic node.
    """
    kernel = np.ones((3, 3), np.float32)
    kernel[1, 1] = 0
    deg = cv2.filter2D(sk.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT).astype(int)
    deg[~sk] = 0

    junction = sk & (deg >= 3)
    endpoint = sk & (deg <= 1)
    n_j, jlab = cv2.connectedComponents(junction.astype(np.uint8), connectivity=8)

    node_of: dict = {}
    node_pos: dict = {}
    nid = 0
    for j in range(1, n_j):
        ys, xs = np.nonzero(jlab == j)
        for y, x in zip(ys, xs):
            node_of[(y, x)] = nid
        node_pos[nid] = (float(ys.mean()), float(xs.mean()))
        nid += 1
    for y, x in zip(*np.nonzero(endpoint)):
        node_of[(y, x)] = nid
        node_pos[nid] = (float(y), float(x))
        nid += 1

    edges = []
    visited = set()  # visited non-node pixels
    used_links = set()  # direct node-pixel -> node-pixel links

    for start_px, start_node in list(node_of.items()):
        for nb in _neighbours(sk, start_px):
            if nb in node_of:
                if node_of[nb] == start_node:
                    continue
                key = frozenset((start_px, nb))
                if key in used_links:
                    continue
                used_links.add(key)
                edges.append((start_node, node_of[nb], [node_pos[start_node], node_pos[node_of[nb]]]))
                continue
            if nb in visited:
                continue
            path = [start_px, nb]
            visited.add(nb)
            prev, cur = start_px, nb
            end_node = None
            while True:
                nxt = None
                for q in _neighbours(sk, cur):
                    if q == prev or q in visited and q not in node_of:
                        continue
                    if q in node_of:
                        if node_of[q] == start_node and len(path) <= 3:
                            continue
                        nxt = q
                        break
                    nxt = q
                if nxt is None:
                    break
                path.append(nxt)
                if nxt in node_of:
                    end_node = node_of[nxt]
                    break
                visited.add(nxt)
                prev, cur = cur, nxt
            if end_node is None:
                # dead end (should not happen for clean skeletons) -> make a node
                end_node = nid
                node_pos[nid] = (float(cur[0]), float(cur[1]))
                nid += 1
            pts = [node_pos[start_node]] + [(float(y), float(x)) for y, x in path[1:-1]] + [node_pos[end_node]]
            edges.append((start_node, end_node, pts))

    # Closed loops with no nodes (e.g. "O", "0").
    for y, x in zip(*np.nonzero(sk)):
        p = (y, x)
        if p in visited or p in node_of:
            continue
        loop = [p]
        visited.add(p)
        prev, cur = None, p
        while True:
            nxt = None
            for q in _neighbours(sk, cur):
                if q != prev and q not in visited:
                    nxt = q
                    break
            if nxt is None:
                break
            loop.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt
        node_pos[nid] = (float(y), float(x))
        pts = [(float(a), float(b)) for a, b in loop] + [(float(y), float(x))]
        edges.append((nid, nid, pts))
        nid += 1

    return node_pos, edges


def _polyline_length(pts) -> float:
    a = np.asarray(pts, float)
    return float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum()) if len(a) > 1 else 0.0


def prune_spurs(sk: np.ndarray, min_len: float, iterations: int = 3) -> np.ndarray:
    """Remove short branches that end in an endpoint and attach to a junction."""
    sk = sk.copy()
    for _ in range(iterations):
        node_pos, edges = skeleton_graph(sk)
        degree: dict = {}
        for a, b, _ in edges:
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
        removed = False
        for a, b, pts in edges:
            if a == b:
                continue
            da, db = degree[a], degree[b]
            if (da == 1) != (db == 1) and _polyline_length(pts) < min_len:
                tip = pts[0] if da == 1 else pts[-1]
                # erase pixels of the spur except the part inside the junction
                for y, x in (pts[:-1] if da == 1 else pts[1:]):
                    sk[int(round(y)), int(round(x))] = False
                sk[int(round(tip[0])), int(round(tip[1]))] = False
                removed = True
        if not removed:
            break
        sk = skeletonize(sk.astype(np.uint8) * 255) if removed else sk
    return sk


def _direction(pts, at_start: bool, look: int = 6):
    a = np.asarray(pts, float)
    if len(a) < 2:
        return np.zeros(2)
    k = min(look, len(a) - 1)
    v = a[k] - a[0] if at_start else a[-1] - a[-1 - k]
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def chain_strokes(node_pos, edges) -> list[list[tuple[float, float]]]:
    """Join graph edges into as few pen strokes as practical."""
    adj: dict = {}
    for i, (a, b, _) in enumerate(edges):
        adj.setdefault(a, []).append(i)
        adj.setdefault(b, []).append(i)
    used = [False] * len(edges)

    def remaining_degree(n):
        return sum(1 for e in adj.get(n, []) if not used[e])

    strokes = []
    while not all(used):
        # Prefer starting at a free stroke tip, then any odd-degree node
        # (Euler-path heuristic), top-left first like a human writer.
        cand = [n for n in adj if remaining_degree(n) > 0]
        tips = [n for n in cand if len(adj[n]) == 1]
        odd = [n for n in cand if remaining_degree(n) % 2 == 1]
        pool = tips or odd or cand
        start = min(pool, key=lambda n: (node_pos[n][0] + node_pos[n][1]))
        stroke: list = []
        cur = start
        heading = None
        while True:
            options = [e for e in adj[cur] if not used[e]]
            if not options:
                break

            def oriented(e):
                a, b, pts = edges[e]
                return pts if a == cur else pts[::-1]

            if heading is None:
                e = options[0] if len(options) == 1 else min(options, key=lambda e: -_polyline_length(edges[e][2]))
            else:
                e = max(options, key=lambda e: float(np.dot(heading, _direction(oriented(e), True))))
            pts = oriented(e)
            used[e] = True
            stroke.extend(pts if not stroke else pts[1:])
            heading = _direction(pts, False)
            a, b, _ = edges[e]
            cur = b if a == cur else a
        if len(stroke) >= 2:
            strokes.append(stroke)
    return strokes


def extend_endpoints(strokes, mask: np.ndarray, free_ends):
    """Thinning retracts open stroke ends by up to the stroke half-width (most
    visible on blunt/square ends). Push each free end along its tangent while
    still well inside the ink, so the robot's thin pen reaches the true ends."""
    dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    h, w = dt.shape
    free = np.asarray(list(free_ends), float).reshape(-1, 2)

    def is_free(pt):  # only real stroke tips, never ends sitting on a junction
        return len(free) > 0 and np.min(np.abs(free - pt).sum(1)) < 0.5

    out = []
    for s in strokes:
        a = np.asarray(s, float)
        if len(a) < 3 or np.linalg.norm(a[0] - a[-1]) < 1.5:
            out.append(a)
            continue
        ends = []
        for at_start in (True, False):
            p0 = a[0] if at_start else a[-1]
            if not is_free(p0):
                ends.append(p0)
                continue
            d = -_direction(a, at_start, look=8) if at_start else _direction(a, False, look=8)
            r = dt[int(round(p0[0])), int(round(p0[1]))]
            pos = p0.copy()
            for _ in range(int(2 * r) + 1):
                nxt = pos + d
                y, x = int(round(nxt[0])), int(round(nxt[1]))
                if not (0 <= y < h and 0 <= x < w) or dt[y, x] <= max(1.0, 0.5 * r):
                    break
                pos = nxt
            ends.append(pos)
        a = np.vstack([ends[0][None], a, ends[1][None]])
        out.append(a)
    return out


def smooth_and_simplify(stroke, eps: float = 0.8, window: int = 5):
    a = np.asarray(stroke, np.float32)
    closed = len(a) > 3 and np.linalg.norm(a[0] - a[-1]) < 1.5
    if len(a) >= window:
        k = np.ones(window, np.float32) / window
        pad = window // 2
        if closed:
            ext = np.concatenate([a[-pad - 1 : -1], a, a[1 : pad + 1]])
            sm = np.stack([np.convolve(ext[:, i], k, mode="valid") for i in range(2)], 1)
            sm[-1] = sm[0]
        else:
            ext = np.concatenate([np.repeat(a[:1], pad, 0), a, np.repeat(a[-1:], pad, 0)])
            sm = np.stack([np.convolve(ext[:, i], k, mode="valid") for i in range(2)], 1)
            sm[0], sm[-1] = a[0], a[-1]
        a = sm
    if closed:
        # approxPolyDP degenerates when first == last; simplify as a closed ring.
        ring = cv2.approxPolyDP(a[:-1].reshape(-1, 1, 2), eps, True).reshape(-1, 2)
        return np.vstack([ring, ring[:1]])
    return cv2.approxPolyDP(a.reshape(-1, 1, 2), eps, False).reshape(-1, 2)


def order_strokes(strokes):
    """Greedy nearest-neighbour ordering (allowing reversal) from the top-left.

    Vectorised over the remaining strokes, so it stays fast for the hundreds
    of strokes produced by sketches / hatching.
    """
    remaining = [np.asarray(s, float) for s in strokes]
    if not remaining:
        return []
    starts = np.array([s[0] for s in remaining])
    ends = np.array([s[-1] for s in remaining])
    closed = np.linalg.norm(starts - ends, axis=1) < 1e-6
    alive = np.ones(len(remaining), bool)
    ordered = []
    pos = np.array([0.0, 0.0])
    for _ in range(len(remaining)):
        d0 = np.linalg.norm(starts - pos, axis=1)
        d1 = np.where(closed, np.inf, np.linalg.norm(ends - pos, axis=1))
        d0[~alive] = np.inf
        d1[~alive] = np.inf
        i0, i1 = int(np.argmin(d0)), int(np.argmin(d1))
        if d1[i1] < d0[i0]:
            i, s = i1, remaining[i1][::-1]
        else:
            i, s = i0, remaining[i0]
        alive[i] = False
        ordered.append(s)
        pos = s[-1]
    return ordered


def strokes_from_mask(mask: np.ndarray, method: str = "skeleton", spur_len: float = 16.0):
    """Binary ink mask (uint8, 255 = ink) -> raw strokes in (y, x) pixel coords
    plus the skeleton / contour image used to obtain them."""
    if method == "outline":
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        raw = []
        for c in contours:
            if cv2.arcLength(c, True) < 15:
                continue
            pts = c.reshape(-1, 2)[:, ::-1].astype(float)  # (y, x)
            raw.append(np.vstack([pts, pts[:1]]))
        sk = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(sk, contours, -1, 1, 1)
        return raw, sk.astype(bool)
    sk = skeletonize(mask)
    sk = prune_spurs(sk, min_len=spur_len)
    node_pos, edges = skeleton_graph(sk)
    degree: dict = {}
    for a, b, _ in edges:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    free_ends = [node_pos[n] for n, d in degree.items() if d == 1]
    return extend_endpoints(chain_strokes(node_pos, edges), mask, free_ends), sk


def finalize_strokes(raw, *, binary, skeleton, bbox, method, min_len: float = 3.0, eps: float = 0.8, extra_stats=None):
    """Raw (y, x) pixel strokes -> smoothed, simplified, ordered, normalised result."""
    strokes_px = []
    for s in raw:
        xy = np.asarray(s, float)[:, ::-1]
        if _polyline_length(xy) < min_len and len(raw) > 1:
            continue
        strokes_px.append(smooth_and_simplify(xy, eps=eps))
    if not strokes_px:
        raise NoCharacterFound("Too little ink to extract strokes")
    strokes_px = order_strokes(strokes_px)

    # Normalise to the unit box, centred, aspect ratio kept.
    allp = np.vstack(strokes_px)
    mn, mx = allp.min(0), allp.max(0)
    span = max(float((mx - mn).max()), 1.0)
    off = (np.array([1.0, 1.0]) - (mx - mn) / span) / 2
    strokes = [[(float(x), float(y)) for x, y in ((s - mn) / span + off)] for s in strokes_px]

    total_len = sum(_polyline_length(s) for s in strokes)
    stats = {
        "num_strokes": len(strokes),
        "num_points": int(sum(len(s) for s in strokes)),
        "ink_length": round(total_len, 3),
        "method": method,
    }
    stats.update(extra_stats or {})
    return ExtractionResult(
        strokes=strokes,
        strokes_px=strokes_px,
        binary=binary,
        skeleton=(skeleton.astype(np.uint8) * 255),
        bbox=bbox,
        stats=stats,
    )


MODES = ("canvas", "photo", "ink")


def extract_strokes(img: np.ndarray, mode: str = "canvas", method: str = "skeleton") -> ExtractionResult:
    """Full pipeline: BGR image -> ordered, normalised pen strokes.

    mode: "canvas" (drawing pad), "photo" (one handwritten character in a
    photo) or "ink" (uploaded image of text / calligraphy / line art: every
    ink component is kept and a finer working resolution is used).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    img = _resize_max(img, 1000 if mode == "ink" else 800)
    bw = binarize(img, mode)
    mask, bbox = isolate_character(bw, mode)
    work = 320 if mode == "ink" else WORK_SIZE  # complex glyphs (Hán tự) need more pixels
    crop, _ = crop_and_scale(mask, bbox, work)
    raw, sk = strokes_from_mask(crop, method, spur_len=0.08 * WORK_SIZE)
    return finalize_strokes(raw, binary=crop, skeleton=sk, bbox=bbox, method=method)


def render_strokes(strokes, size: int = 256, margin: int = 16) -> np.ndarray:
    """Debug rendering of normalised strokes (colour = order; start dots when few)."""
    img = np.full((size, size, 3), 255, np.uint8)
    sc = size - 2 * margin
    many = len(strokes) > 40
    for i, s in enumerate(strokes):
        hue = int(150 * i / max(1, len(strokes)))
        col = cv2.cvtColor(np.uint8([[[hue, 220, 200]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        pts = (np.asarray(s) * sc + margin).astype(np.int32)
        cv2.polylines(img, [pts], False, col, 1 if many else 2, cv2.LINE_AA)
        if not many:
            cv2.circle(img, tuple(int(v) for v in pts[0]), 4, col, -1, cv2.LINE_AA)
    return img
