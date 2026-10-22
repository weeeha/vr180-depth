import cv2
import numpy as np
import pytest

from vr180_depth.align import estimate_vertical_offset, shift_vertical
from vr180_depth.export import depth_to_matte16, depth_to_mm16
from vr180_depth.fuse import fit_scale_offset


def _textured(h=360, w=640, seed=0):
    img = (np.random.default_rng(seed).random((h, w)) * 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (0, 0), 1.5)


def test_estimates_vertical_offset():
    left = _textured()
    m = np.float32([[1, 0, -12], [0, 1, 3]])  # right eye: 12 px disparity, 3 px lower
    right = cv2.warpAffine(left, m, (640, 360), borderMode=cv2.BORDER_REFLECT)
    dy = estimate_vertical_offset(left, right, max_disparity=64)
    assert dy == pytest.approx(3.0, abs=0.5)
    fixed = shift_vertical(right, dy)
    assert estimate_vertical_offset(left, fixed, max_disparity=64) == pytest.approx(0.0, abs=0.5)


def test_fit_scale_offset_ignores_outliers():
    rng = np.random.default_rng(1)
    rel = rng.random((200, 200))
    disp = 2.5 * rel + 3.0
    disp[rng.random(disp.shape) < 0.1] = 80.0  # bad matches
    disp[:10] = np.nan
    scale, offset = fit_scale_offset(rel, disp)
    assert scale == pytest.approx(2.5, rel=0.01)
    assert offset == pytest.approx(3.0, rel=0.01)


def test_matte_endpoints():
    depth = np.array([[0.5, 10.0, 50.0, np.nan]])
    matte = depth_to_matte16(depth, near_m=0.5, far_m=10.0)
    assert matte.tolist() == [[65535, 0, 0, 0]]


def test_mm_encoding():
    depth = np.array([[1.234, np.nan, 100.0]])
    assert depth_to_mm16(depth).tolist() == [[1234, 0, 65535]]
