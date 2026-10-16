import cv2
import numpy as np
import pytest

from backend.vision import (
    NoCharacterFound,
    decode_image,
    extract_strokes,
    skeletonize,
    zhang_suen_thinning,
)


def _coverage(res) -> float:
    """Fraction of ink pixels within reach (half stroke width + 3 px) of the
    extracted strokes, measured in the pixel frame of the binary crop."""
    binary = res.binary
    half_width = cv2.distanceTransform(binary, cv2.DIST_L2, 5).max()
    canvas = np.full(binary.shape, 255, np.uint8)
    for s in res.strokes_px:
        cv2.polylines(canvas, [np.round(s).astype(np.int32)], False, 0, 1)
    dist = cv2.distanceTransform(canvas, cv2.DIST_L2, 5)
    return float((dist[binary > 0] <= half_width + 3).mean())


def test_thinning_is_one_pixel_wide():
    img = np.zeros((60, 60), np.uint8)
    cv2.rectangle(img, (10, 25), (50, 35), 255, -1)
    sk = skeletonize(img)
    assert sk.sum() >= 25
    # no 2x2 fully-set block remains
    blocks = sk[:-1, :-1] & sk[1:, :-1] & sk[:-1, 1:] & sk[1:, 1:]
    assert not blocks.any()
    assert zhang_suen_thinning(img > 0).sum() >= sk.sum()


@pytest.mark.parametrize(
    "ch,max_strokes",
    [("I", 1), ("L", 1), ("O", 1), ("T", 2), ("A", 3), ("X", 3), ("8", 2), ("B", 3), ("k", 3)],
)
def test_canvas_characters(make_text, ch, max_strokes):
    res = extract_strokes(make_text(ch), "canvas")
    assert 1 <= len(res.strokes) <= max_strokes, res.stats
    allp = np.vstack(res.strokes)
    assert allp.min() >= -1e-6 and allp.max() <= 1 + 1e-6
    assert _coverage(res) > 0.98


def test_closed_loop_is_closed(make_text):
    res = extract_strokes(make_text("O"), "canvas")
    s = np.asarray(res.strokes[0])
    assert np.linalg.norm(s[0] - s[-1]) < 1e-6
    # the loop must go all the way round (no gap): consecutive points close together
    assert np.linalg.norm(np.diff(s, axis=0), axis=1).max() < 0.25


def test_outline_method(make_text):
    res = extract_strokes(make_text("O"), "canvas", method="outline")
    assert len(res.strokes) == 2  # outer and inner contour


def test_photo_mode_isolates_character(make_photo):
    res = extract_strokes(make_photo(), "photo")
    # The desk edge and dirt specks must be ignored: a 'T' is 1-2 strokes.
    assert 1 <= len(res.strokes) <= 2, res.stats
    x, y, w, h = res.bbox
    assert 200 < x < 260 and 120 < y < 170 and 150 < w < 230 and 180 < h < 250


def test_blank_inputs_raise():
    with pytest.raises(NoCharacterFound):
        extract_strokes(np.full((300, 300, 3), 255, np.uint8), "canvas")
    rng = np.random.default_rng(1)
    paper = np.clip(200 + rng.normal(0, 3, (300, 300, 3)), 0, 255).astype(np.uint8)
    with pytest.raises(NoCharacterFound):
        extract_strokes(paper, "photo")


def test_decode_transparent_png():
    rgba = np.zeros((50, 50, 4), np.uint8)  # fully transparent -> white
    rgba[20:30, 20:30] = [0, 0, 0, 255]
    ok, buf = cv2.imencode(".png", rgba)
    img = decode_image(buf.tobytes())
    assert img[0, 0].tolist() == [255, 255, 255]
    assert img[25, 25].tolist() == [0, 0, 0]


def _poly(img, pts, t):
    cv2.polylines(img, [np.array(pts, np.int32)], False, (0, 0, 0), t, cv2.LINE_AA)


def test_canvas_keeps_every_drawn_part():
    """Regression: drawing 'ÂY' used to keep only the 'Y' (largest component)."""
    img = np.full((400, 400, 3), 255, np.uint8)
    _poly(img, [(35, 120), (130, 95), (175, 122)], 18)  # circumflex
    _poly(img, [(38, 265), (90, 155), (160, 270)], 18)  # A legs
    _poly(img, [(55, 215), (145, 215)], 18)  # A bar
    _poly(img, [(230, 150), (320, 205)], 18)  # Y left arm
    _poly(img, [(370, 148), (255, 285)], 18)  # Y right arm
    res = extract_strokes(img, "canvas")
    x, y, w, h = res.bbox
    assert x < 40 and y < 100 and x + w > 360 and y + h > 270, res.bbox
    assert len(res.strokes) >= 4
    for method in ("skeleton", "outline"):
        assert extract_strokes(img, "canvas", method).bbox == res.bbox


def test_photo_keeps_nearby_letters_drops_far_blobs(make_photo):
    img = make_photo()
    cv2.line(img, (450, 150), (520, 360), (60, 55, 50), 10, cv2.LINE_AA)  # second letter nearby
    cv2.circle(img, (40, 455), 9, (60, 55, 50), -1)  # far-away blot in a corner
    res = extract_strokes(img, "photo")
    x, y, w, h = res.bbox
    assert x + w > 510  # the nearby letter is included
    assert x > 200 and y + h < 400  # the blot is not


def test_ink_mode_keeps_all_components_and_detects_polarity():
    """Uploaded text: a 'Hán-like' glyph made of separate parts must keep every
    part, and white-on-black input must give the same strokes as black-on-white."""
    img = np.full((400, 400, 3), 230, np.uint8)
    _poly(img, [(60, 60), (340, 60)], 16)  # 一
    _poly(img, [(110, 130), (290, 130)], 16)  # second bar
    cv2.rectangle(img, (120, 200), (280, 340), (20, 20, 20), 16)  # 口
    _poly(img, [(40, 380), (60, 360)], 10)  # small dot-like stroke far away
    a = extract_strokes(img, "ink")
    x, y, w, h = a.bbox
    assert x < 45 and y < 60 and x + w > 330 and y + h > 375
    assert len(a.strokes) >= 4
    b = extract_strokes(255 - img, "ink")
    assert b.bbox == a.bbox and len(b.strokes) == len(a.strokes)


def test_cjk_character_if_font_available():
    from pathlib import Path

    font_path = Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
    PIL = pytest.importorskip("PIL")
    if not font_path.exists():
        pytest.skip("Noto CJK font not installed")
    from PIL import Image, ImageDraw, ImageFont

    im = Image.new("RGB", (360, 400), (240, 236, 228))
    ImageDraw.Draw(im).text((20, 20), "龍", font=ImageFont.truetype(str(font_path), 300), fill=(20, 20, 20))
    res = extract_strokes(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR), "ink")
    assert 8 <= len(res.strokes) <= 30
    assert _coverage(res) > 0.95
