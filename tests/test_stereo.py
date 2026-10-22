import numpy as np
import pytest

from vr180_depth.pipeline import FrameProcessor, Settings
from vr180_depth.stereo import disparity_to_depth, fill_holes, max_disparity_for
from vr180_depth.synthetic import render_vr180_sbs


def test_disparity_to_depth():
    depth = disparity_to_depth(np.array([[10.0, np.nan, 0.0]]), focal_px=500, baseline_m=0.06)
    assert depth[0, 0] == pytest.approx(3.0)
    assert np.isnan(depth[0, 1]) and np.isnan(depth[0, 2])


def test_max_disparity_is_multiple_of_16():
    d = max_disparity_for(640, 0.065, 0.4)
    assert d % 16 == 0 and d >= 640 * 0.065 / 0.4


def test_fill_holes_prefers_background():
    disp = np.array([[20.0, np.nan, np.nan, 5.0]])
    assert fill_holes(disp).tolist() == [[20.0, 5.0, 5.0, 5.0]]


@pytest.fixture(scope="module")
def synthetic_result():
    frame = render_vr180_sbs(eye_size=1024, baseline_m=0.065, card_z=1.5, wall_z=4.0)
    s = Settings(width=640, height=360, fov_deg=90, baseline_m=0.065)
    return FrameProcessor(s).process(frame)


def test_recovers_card_depth(synthetic_result):
    depth = synthetic_result.depth_m
    assert np.nanmedian(depth[160:200, 300:340]) == pytest.approx(1.5, rel=0.08)


def test_recovers_wall_depth(synthetic_result):
    depth = synthetic_result.depth_m
    assert np.nanmedian(depth[160:200, 540:600]) == pytest.approx(4.0, rel=0.15)


def test_synthetic_eyes_need_no_vertical_fix(synthetic_result):
    assert abs(synthetic_result.vertical_offset_px) < 0.5
