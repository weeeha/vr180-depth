"""Split stereo frames and reproject VR180 eyes into flat (pinhole) views.

Coordinate convention: x right, y down, z forward. A VR180 eye is an
equirectangular image covering 180 x 180 degrees around the forward axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PinholeView:
    """A flat output view. ``fov_deg`` is the horizontal field of view."""

    width: int
    height: int
    fov_deg: float

    @property
    def focal_px(self) -> float:
        return (self.width / 2) / np.tan(np.radians(self.fov_deg) / 2)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.width - 1) / 2, (self.height - 1) / 2)


def split_stereo(frame: np.ndarray, layout: str = "sbs", swap: bool = False):
    """Return (left, right) from a side-by-side ("sbs") or top-bottom ("tb") frame."""
    h, w = frame.shape[:2]
    if layout == "sbs":
        a, b = frame[:, : w // 2], frame[:, w // 2 : (w // 2) * 2]
    elif layout == "tb":
        a, b = frame[: h // 2], frame[h // 2 : (h // 2) * 2]
    else:
        raise ValueError(f"unknown layout {layout!r}, expected 'sbs' or 'tb'")
    return (b, a) if swap else (a, b)


def _pitch_matrix(pitch_deg: float) -> np.ndarray:
    # Positive pitch looks up (toward -y). Pitch rotates about the baseline
    # axis, so a pitched pair stays rectified. Yaw would not, so there is none.
    p = np.radians(pitch_deg)
    c, s = np.cos(p), np.sin(p)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def equirect_to_pinhole_maps(
    eye_w: int,
    eye_h: int,
    view: PinholeView,
    pitch_deg: float = 0.0,
    eye_h_fov_deg: float = 180.0,
    eye_v_fov_deg: float = 180.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``cv2.remap`` maps that sample an equirect eye into ``view``.

    Use the same maps for both eyes: two parallel cameras seen through the
    same pinhole projection form a rectified pair, so matching points share a
    row and disparity = focal_px * baseline / depth.
    """
    f = view.focal_px
    cx, cy = view.center
    xs = (np.arange(view.width, dtype=np.float64) - cx) / f
    ys = (np.arange(view.height, dtype=np.float64) - cy) / f
    x, y = np.meshgrid(xs, ys)
    rays = np.stack([x, y, np.ones_like(x)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    rays = rays @ _pitch_matrix(pitch_deg).T

    lon = np.arctan2(rays[..., 0], rays[..., 2])
    lat = np.arcsin(np.clip(rays[..., 1], -1.0, 1.0))
    map_x = (lon / np.radians(eye_h_fov_deg) + 0.5) * eye_w - 0.5
    map_y = (lat / np.radians(eye_v_fov_deg) + 0.5) * eye_h - 0.5
    return map_x.astype(np.float32), map_y.astype(np.float32)


def apply_maps(img: np.ndarray, maps: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return cv2.remap(img, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
