"""Classical stereo matching (OpenCV SGBM) and disparity -> depth."""

from __future__ import annotations

import cv2
import numpy as np


def max_disparity_for(focal_px: float, baseline_m: float, min_depth_m: float) -> int:
    """Search range needed to see objects as close as ``min_depth_m``.

    SGBM wants a multiple of 16.
    """
    d = focal_px * baseline_m / min_depth_m
    return max(16, int(np.ceil(d / 16.0)) * 16)


def compute_disparity(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    num_disparities: int,
    block_size: int = 5,
) -> np.ndarray:
    """Left-referenced disparity in pixels; NaN where matching failed."""
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size**2,
        P2=32 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan
    return disp


def fill_holes(disp: np.ndarray) -> np.ndarray:
    """Fill invalid pixels row by row with the farther of the nearest valid
    neighbours. Stereo holes are mostly occlusions next to a foreground edge,
    and the hidden surface belongs to the background, so the smaller disparity
    is the better guess.
    """
    valid = np.isfinite(disp)
    if valid.all() or not valid.any():
        return disp
    h, w = disp.shape
    cols = np.arange(w)[None, :]
    rows = np.arange(h)[:, None]

    left_idx = np.maximum.accumulate(np.where(valid, cols, 0), axis=1)
    right_idx = np.minimum.accumulate(np.where(valid, cols, w - 1)[:, ::-1], axis=1)[:, ::-1]
    from_left = np.where(valid[rows, left_idx], disp[rows, left_idx], np.nan)
    from_right = np.where(valid[rows, right_idx], disp[rows, right_idx], np.nan)
    return np.where(valid, disp, np.fmin(from_left, from_right))


def disparity_to_depth(disp: np.ndarray, focal_px: float, baseline_m: float) -> np.ndarray:
    """Depth in metres. NaN stays NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = (focal_px * baseline_m) / disp
    depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
    return depth
