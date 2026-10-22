"""Fix small vertical misalignment between the eyes.

Stereo matching only searches along rows, so even a 2 px vertical offset
between lenses wrecks the result. We measure it from feature matches and
shift the right eye to cancel it.
"""

from __future__ import annotations

import cv2
import numpy as np


def estimate_vertical_offset(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    max_disparity: float,
    min_matches: int = 30,
) -> float | None:
    """Median (right_y - left_y) over plausible matches, or None if too few."""
    orb = cv2.ORB_create(nfeatures=4000)
    kl, dl = orb.detectAndCompute(left_gray, None)
    kr, dr = orb.detectAndCompute(right_gray, None)
    if dl is None or dr is None:
        return None

    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(dl, dr)
    if len(matches) < min_matches:
        return None
    pl = np.float32([kl[m.queryIdx].pt for m in matches])
    pr = np.float32([kr[m.trainIdx].pt for m in matches])
    dx = pl[:, 0] - pr[:, 0]
    dy = pr[:, 1] - pl[:, 1]
    plausible = (dx >= -2) & (dx <= max_disparity) & (np.abs(dy) < 20)
    if plausible.sum() < min_matches:
        return None
    return float(np.median(dy[plausible]))


def shift_vertical(img: np.ndarray, dy: float) -> np.ndarray:
    """Move image content up by ``dy`` pixels (down if negative)."""
    h, w = img.shape[:2]
    m = np.float32([[1, 0, 0], [0, 1, -dy]])
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
