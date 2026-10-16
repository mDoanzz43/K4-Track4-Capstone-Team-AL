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
