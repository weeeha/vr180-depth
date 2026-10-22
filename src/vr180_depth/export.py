"""Encode depth for files that compositing tools can read."""

from __future__ import annotations

import cv2
import numpy as np


def depth_to_mm16(depth_m: np.ndarray) -> np.ndarray:
    """uint16 millimetres (max 65.5 m). 0 means unknown."""
    mm = np.nan_to_num(depth_m * 1000.0, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(np.round(mm), 0, 65535).astype(np.uint16)


def depth_to_matte16(depth_m: np.ndarray, near_m: float, far_m: float) -> np.ndarray:
    """uint16 matte: white at ``near_m``, black at ``far_m`` and beyond.

    The ramp is linear in inverse depth (1/z), like disparity, so most of the
    grey levels go to the near range, where stereo is accurate and where
    focus and fog effects need them.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / depth_m
    lo, hi = 1.0 / far_m, 1.0 / near_m
    v = np.nan_to_num((inv - lo) / (hi - lo), nan=0.0, posinf=1.0, neginf=0.0)
    return np.round(np.clip(v, 0.0, 1.0) * 65535).astype(np.uint16)


def preview_frame(view_bgr: np.ndarray, matte16: np.ndarray) -> np.ndarray:
    """Flat view and colour-mapped depth side by side."""
    colored = cv2.applyColorMap((matte16 >> 8).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[matte16 == 0] = 0
    return cv2.hconcat([view_bgr, colored])
