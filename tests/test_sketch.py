import cv2
import numpy as np
import pytest

from backend.sketch import MODEL, detect_face, hatch, person_mask, sketch_strokes
from backend.vision import NoCharacterFound


def test_hatch_fills_mask_with_parallel_segments():
    mask = np.zeros((200, 200), np.uint8)
    cv2.rectangle(mask, (50, 50), (149, 149), 255, -1)
    segs = hatch(mask, 0, spacing=10, min_len=5)
    assert len(segs) == 10  # one segment per 10 px row inside the square
    for s in segs:
        (y0, x0), (y1, x1) = s
        assert abs(y0 - y1) < 1.5 and 45 <= min(x0, x1) <= 56 and 143 <= max(x0, x1) <= 152
    # alternate rows reversed (zig-zag)
    assert (segs[0][1][1] - segs[0][0][1]) * (segs[1][1][1] - segs[1][0][1]) < 0
    diag = hatch(mask, 45, spacing=8, min_len=4)
    for s in diag:
        d = s[1] - s[0]
        assert abs(abs(d[0]) - abs(d[1])) < 2.5  # 45 degrees
        for y, x in s:
            assert 47 <= x <= 152 and 47 <= y <= 152


def test_sketch_without_face(make_portrait):
    img = make_portrait()
    res = sketch_strokes(img, detail=0.5, hatching=True)
    st = res.stats
    assert st["method"] == "sketch" and not st["face_found"]  # cartoon face: detector finds nothing
    assert st["line_strokes"] > 5 and st["hatch_strokes"] > 10
    allp = np.vstack(res.strokes)
    assert allp.min() >= -1e-6 and allp.max() <= 1 + 1e-6
    # hatching lies in the dark jacket / hair regions: check in pixel space
    h = res.binary.shape[0]
    hatch_pts = np.vstack(res.strokes_px[-1:])  # any stroke works for the bounds check
    assert hatch_pts[:, 1].max() <= h + 1


def test_detail_controls_amount_of_strokes(make_portrait):
    img = make_portrait()
    lo = sketch_strokes(img, detail=0.0, hatching=True)
    hi = sketch_strokes(img, detail=1.0, hatching=True)
    assert hi.stats["num_strokes"] > lo.stats["num_strokes"]
    no_hatch = sketch_strokes(img, detail=0.5, hatching=False)
    assert no_hatch.stats["hatch_strokes"] == 0


def test_person_mask_keeps_the_person(make_portrait):
    img = make_portrait()
    fg = person_mask(img, (120, 70, 160, 200))  # face box around the synthetic head
    assert fg[170, 200] == 255  # face centre
    assert fg[420, 200] == 255  # jacket
    assert fg[20, 10] == 0 and fg[20, 390] == 0  # background corners
    with_bg = sketch_strokes(img, 0.5)
    assert with_bg.stats["background_removed"] is False  # no face detected -> no segmentation


def test_face_detector_model_present_and_no_false_positive():
    assert MODEL.exists() and MODEL.stat().st_size > 100_000
    assert detect_face(np.full((300, 300, 3), 200, np.uint8)) is None


def test_blank_image_raises():
    with pytest.raises(NoCharacterFound):
        sketch_strokes(np.full((300, 300, 3), 128, np.uint8))
