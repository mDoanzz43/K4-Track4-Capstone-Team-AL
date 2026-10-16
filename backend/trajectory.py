"""Turn normalised 2-D strokes into a timed 3-D pen-tip trajectory.

Coordinates: strokes come in the unit box (x right, y down). They are mapped
onto the paper lying on the table in front of the robot so that the character
reads upright for someone standing behind the robot (image "up" = +X world,
image "right" = -Y world).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrajectoryConfig:
    paper_center: tuple[float, float] = (0.62, 0.0)  # world XY of the paper centre (m)
    paper_z: float = 0.62  # height of the paper / table top (m); the robot base sits here too
    char_size: float = 0.20  # side of the square the character is fitted into (m)
    lift: float = 0.04  # pen-up clearance above the paper (m)
    press: float = 0.0015  # how far below the surface the pen is commanded when down (m)
    draw_speed: float = 0.10  # m/s while drawing
    travel_speed: float = 0.25  # m/s while the pen is up
    z_speed: float = 0.08  # m/s for pen up/down moves
    dt: float = 1.0 / 240.0  # control period (matches the PyBullet step)


@dataclass
class Trajectory:
    points: np.ndarray  # (N, 3) target pen-tip positions, one per control step
    pen_down: np.ndarray  # (N,) bool, True while the segment is a drawing segment
    stroke_id: np.ndarray  # (N,) int, -1 while travelling
    duration: float

    def __len__(self) -> int:
        return len(self.points)


def to_world(strokes, cfg: TrajectoryConfig) -> list[np.ndarray]:
    cx, cy = cfg.paper_center
    out = []
    for s in strokes:
        a = np.asarray(s, float)
        u, v = a[:, 0] - 0.5, a[:, 1] - 0.5
        xw = cx - v * cfg.char_size
        yw = cy - u * cfg.char_size
        out.append(np.stack([xw, yw], 1))
    return out


def world_to_unit(xy: np.ndarray, cfg: TrajectoryConfig) -> np.ndarray:
    """Inverse of :func:`to_world` (used to report the inked path to the UI)."""
    cx, cy = cfg.paper_center
    xy = np.atleast_2d(xy)
    u = -(xy[:, 1] - cy) / cfg.char_size + 0.5
    v = -(xy[:, 0] - cx) / cfg.char_size + 0.5
    return np.stack([u, v], 1)


def _ease(p0: np.ndarray, p1: np.ndarray, speed: float, dt: float) -> np.ndarray:
    """Samples from p0 (exclusive) to p1 (inclusive) with a smooth cosine
    velocity profile so the arm does not jerk at the ends of pen-up moves."""
    d = float(np.linalg.norm(p1 - p0))
    if d < 1e-9:
        return np.empty((0, 3))
    # Cosine (minimum-jerk-like) profile with peak speed `speed`: T = pi*d / (2v)
    n = max(2, int(np.ceil(np.pi * d / (2 * speed) / dt)))
    t = (1 - np.cos(np.pi * np.arange(1, n + 1) / n)) / 2
    return p0 + np.outer(t, p1 - p0)


def resample_polyline(xy: np.ndarray, step: float) -> np.ndarray:
    """Uniform arc-length resampling (keeps start and end)."""
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    L = np.concatenate([[0], np.cumsum(seg)])
    if L[-1] < 1e-9:
        return xy[:1]
    n = max(2, int(np.ceil(L[-1] / step)) + 1)
    s = np.linspace(0, L[-1], n)
    return np.stack([np.interp(s, L, xy[:, 0]), np.interp(s, L, xy[:, 1])], 1)


def build_trajectory(strokes, cfg: TrajectoryConfig | None = None, home=None) -> Trajectory:
    cfg = cfg or TrajectoryConfig()
    world = to_world(strokes, cfg)
    z_up = cfg.paper_z + cfg.lift
    z_dn = cfg.paper_z - cfg.press
    pts = []
    down = []
    sid = []

    cur = np.array(home if home is not None else [cfg.paper_center[0] - 0.1, cfg.paper_center[1], z_up + 0.06], float)
    pts.append(cur[None]); down.append([False]); sid.append([-1])

    def push(arr, is_down, k):
        nonlocal cur
        if len(arr):
            pts.append(arr)
            down.append(np.full(len(arr), is_down))
            sid.append(np.full(len(arr), k))
            cur = arr[-1]

    for k, xy in enumerate(world):
        start = np.array([xy[0, 0], xy[0, 1], z_up])
        push(_ease(cur, start, cfg.travel_speed, cfg.dt), False, -1)
        push(_ease(cur, np.array([xy[0, 0], xy[0, 1], z_dn]), cfg.z_speed, cfg.dt), False, -1)
        # Constant-speed drawing along the arc-length-resampled stroke.
        dense = resample_polyline(xy, cfg.draw_speed * cfg.dt)
        draw = np.column_stack([dense[1:], np.full(len(dense) - 1, z_dn)])
        push(draw, True, k)
        push(_ease(cur, np.array([cur[0], cur[1], z_up]), cfg.z_speed, cfg.dt), False, -1)

    end = np.array([cfg.paper_center[0] - 0.1, cfg.paper_center[1], z_up + 0.06])
    push(_ease(cur, end, cfg.travel_speed, cfg.dt), False, -1)

    P = np.vstack(pts)
    return Trajectory(
        points=P,
        pen_down=np.concatenate(down).astype(bool),
        stroke_id=np.concatenate(sid).astype(int),
        duration=len(P) * cfg.dt,
    )
