import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def text_image(ch: str, thickness: int = 26, size: int = 400) -> np.ndarray:
    img = np.full((size, size, 3), 255, np.uint8)
    cv2.putText(img, ch, (int(size * 0.22), int(size * 0.8)), cv2.FONT_HERSHEY_SIMPLEX, size / 44, (0, 0, 0), thickness, cv2.LINE_AA)
    return img


def photo_image(seed: int = 0) -> np.ndarray:
    """Synthetic webcam photo: uneven lighting, noise, a dark desk edge and a
    handwritten-looking 'T' drawn with grey-ish ink."""
    rng = np.random.default_rng(seed)
    H, W = 480, 640
    yy, xx = np.mgrid[0:H, 0:W]
    light = 210 - 0.15 * xx + 0.06 * yy
    img = np.clip(light + rng.normal(0, 5, (H, W)), 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img, (0, 0), (W, 35), (50, 50, 60), -1)
    cv2.line(img, (230, 150), (420, 145), (60, 55, 50), 10, cv2.LINE_AA)
    cv2.line(img, (325, 148), (330, 360), (60, 55, 50), 10, cv2.LINE_AA)
    for _ in range(40):  # specks of dirt
        cv2.circle(img, (int(rng.integers(0, W)), int(rng.integers(40, H))), 1, (120, 120, 120), -1)
    return img


@pytest.fixture
def make_text():
    return text_image


@pytest.fixture
def make_photo():
    return photo_image


def portrait_like(seed: int = 0) -> np.ndarray:
    """Synthetic 'portrait': light background, skin-toned head with eyes,
    mouth and hair, and a dark jacket (gives line art + dark tone regions)."""
    rng = np.random.default_rng(seed)
    img = np.full((480, 400, 3), (205, 215, 220), np.uint8)
    cv2.ellipse(img, (200, 170), (85, 110), 0, 0, 360, (150, 180, 215), -1)  # face
    cv2.ellipse(img, (200, 95), (95, 55), 0, 180, 360, (30, 30, 40), -1)  # hair
    for x in (165, 235):
        cv2.ellipse(img, (x, 160), (16, 8), 0, 0, 360, (250, 250, 250), -1)
        cv2.circle(img, (x, 160), 6, (40, 30, 20), -1)
        cv2.line(img, (x - 20, 138), (x + 20, 136), (35, 35, 35), 5)  # eyebrows
    cv2.line(img, (200, 170), (192, 215), (110, 130, 170), 3)  # nose
    cv2.ellipse(img, (200, 240), (32, 12), 0, 0, 180, (60, 60, 140), 4)  # mouth
    pts = np.array([[60, 480], [110, 330], [290, 330], [340, 480]], np.int32)
    cv2.fillPoly(img, [pts], (45, 40, 45))  # jacket
    img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    return img


@pytest.fixture
def make_portrait():
    return portrait_like
