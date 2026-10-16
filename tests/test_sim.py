import numpy as np
import pytest

from backend.robot_sim import CAMERAS, DrawingSim
from backend.trajectory import TrajectoryConfig, build_trajectory, world_to_unit


@pytest.fixture(scope="module")
def sim():
    s = DrawingSim(TrajectoryConfig())
    yield s
    s.close()


def test_ik_accuracy(sim):
    cfg = sim.cfg
    for dx, dy in [(-0.1, -0.1), (0.1, 0.1), (0.1, -0.1), (-0.1, 0.1), (0, 0)]:
        target = [cfg.paper_center[0] + dx, cfg.paper_center[1] + dy, cfg.paper_z]
        q = sim.ik(target)
        sim._set_ik_joints(q)
        import pybullet as p

        ls = p.getLinkState(sim.ik_robot, 6, computeForwardKinematics=True, physicsClientId=sim.ik_cid)
        rot = np.array(p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)
        tip = np.array(ls[4]) + rot @ np.array([0, 0, 0.15])
        assert np.linalg.norm(tip - target) < 1e-4
        # pen points straight down
        assert rot[:, 2] @ np.array([0, 0, -1]) > 0.9999


def test_robot_draws_what_was_planned(sim):
    strokes = [[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8)], [(0.2, 0.5), (0.5, 0.5)]]
    tr = build_trajectory(strokes, sim.cfg)
    frames = []
    stats = sim.run(tr, lambda f, i: frames.append((f, i)), speed=8, realtime=False)
    assert not stats["stopped"]
    assert stats["tracking_rms_mm"] < 1.5 and stats["tracking_max_mm"] < 3.0
    assert stats["ink_polylines"] == 2
    # inked points (back in unit coords) lie on the planned strokes
    ink = np.vstack([world_to_unit(np.asarray(pl)[:, :2], sim.cfg) for pl in sim.ink])
    dense = np.vstack([np.linspace(a, b, 200) for s in strokes for a, b in zip(s[:-1], s[1:])])
    d = np.min(np.linalg.norm(ink[:, None] - dense[None], axis=2), axis=1)
    assert d.max() * sim.cfg.char_size < 0.004  # < 4 mm anywhere
    # and the planned strokes are fully covered by ink
    d2 = np.min(np.linalg.norm(dense[:, None] - ink[None], axis=2), axis=1)
    assert d2.max() * sim.cfg.char_size < 0.004
    f, info = frames[-1]
    assert f.shape == (sim.height, sim.width, 3) and info.step == info.total


def test_ink_visible_in_render_and_cameras(sim):
    sim.set_camera("top")
    img = sim.render().astype(int)
    # dark navy ink pixels exist on the paper after the previous drawing
    navy = (img[:, :, 0] > 90) & (img[:, :, 2] < 60) & (img[:, :, 1] < 70)
    assert navy.sum() > 100
    for name in CAMERAS:
        sim.set_camera(name)
        assert sim.render().mean() > 20


def test_stop_event(sim):
    import threading

    ev = threading.Event()
    tr = build_trajectory([[(0.1, 0.1), (0.9, 0.9)]], sim.cfg)
    count = []

    def on_frame(f, i):
        count.append(1)
        if len(count) == 2:
            ev.set()

    stats = sim.run(tr, on_frame, speed=4, realtime=False, stop_event=ev)
    assert stats["stopped"]


def test_incremental_ink_cache_matches_full_redraw(sim):
    sim.set_camera("top")
    tr = build_trajectory([[(0.1, 0.5), (0.9, 0.5)], [(0.5, 0.1), (0.5, 0.9)]], sim.cfg)
    sim.run(tr, lambda f, i: None, speed=8, realtime=False)  # renders incrementally along the way
    cached = sim.render()
    sim._ink_cache = None  # force a full redraw from scratch
    assert np.abs(sim.render().astype(int) - cached.astype(int)).max() <= 1
    sim.clear_paper()
    assert not sim.ink and sim.render().shape == cached.shape
