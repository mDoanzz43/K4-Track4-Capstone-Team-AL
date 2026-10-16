import numpy as np

from backend.trajectory import TrajectoryConfig, build_trajectory, to_world, world_to_unit


def test_world_mapping_roundtrip():
    cfg = TrajectoryConfig()
    s = [[(0.1, 0.2), (0.9, 0.7)]]
    w = to_world(s, cfg)[0]
    assert np.allclose(world_to_unit(w, cfg), np.asarray(s[0]))
    # image "up" (smaller v) is further from the robot (+X)
    assert w[0, 0] > w[1, 0]


def test_trajectory_speeds_and_pen_states():
    cfg = TrajectoryConfig()
    strokes = [[(0.2, 0.2), (0.8, 0.2)], [(0.5, 0.3), (0.5, 0.9)]]
    tr = build_trajectory(strokes, cfg)
    step = np.linalg.norm(np.diff(tr.points, axis=0), axis=1)
    # never faster than the fastest configured speed
    assert step.max() <= cfg.travel_speed * cfg.dt * 1.01
    down = tr.points[tr.pen_down]
    assert np.allclose(down[:, 2], cfg.paper_z - cfg.press)
    # drawing happens at the configured speed
    dsteps = step[tr.pen_down[1:] & tr.pen_down[:-1]]
    assert np.allclose(np.median(dsteps), cfg.draw_speed * cfg.dt, rtol=0.05)
    assert set(np.unique(tr.stroke_id)) == {-1, 0, 1}
    # travel between strokes happens with the pen lifted
    travel = tr.points[~tr.pen_down]
    assert travel[:, 2].max() >= cfg.paper_z + cfg.lift - 1e-9
    # inked length matches the stroke lengths
    ink_len = step[tr.pen_down[1:] & tr.pen_down[:-1]].sum()
    assert abs(ink_len - (0.6 + 0.6) * cfg.char_size) < 0.01
