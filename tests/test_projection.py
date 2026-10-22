import numpy as np
import pytest

from vr180_depth.projection import PinholeView, equirect_to_pinhole_maps, split_stereo


def test_focal_from_fov():
    assert PinholeView(1280, 720, 90).focal_px == pytest.approx(640)


def test_view_center_samples_eye_center():
    view = PinholeView(641, 361, 90)
    mx, my = equirect_to_pinhole_maps(2000, 2000, view)
    assert mx[180, 320] == pytest.approx(999.5, abs=1e-3)
    assert my[180, 320] == pytest.approx(999.5, abs=1e-3)


def test_view_edge_is_45_degrees():
    view = PinholeView(641, 361, 90)
    mx, _ = equirect_to_pinhole_maps(2000, 2000, view)
    # Right edge of a 90 degree view looks 45 degrees right: 3/4 across a 180 degree eye.
    assert mx[180, 640] == pytest.approx(0.75 * 2000 - 0.5, abs=0.5)


def test_pitch_up_samples_upper_half():
    view = PinholeView(641, 361, 90)
    _, my = equirect_to_pinhole_maps(2000, 2000, view, pitch_deg=30)
    assert my[180, 320] == pytest.approx((0.5 - 30 / 180) * 2000 - 0.5, abs=0.5)


def test_split_layouts():
    frame = np.zeros((4, 8), np.uint8)
    frame[:, 4:] = 1
    left, right = split_stereo(frame, "sbs")
    assert left.max() == 0 and right.min() == 1
    left, right = split_stereo(frame, "sbs", swap=True)
    assert left.min() == 1
    top, bottom = split_stereo(frame.T, "tb")
    assert top.max() == 0 and bottom.min() == 1
