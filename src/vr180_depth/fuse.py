"""Optional AI refinement: AI detail, stereo scale.

A monocular model (Depth Anything V2) gives sharp, dense depth but only up
to an unknown scale and offset. Stereo gives real scale but has holes and
soft edges. We fit the AI output to the stereo disparity with one scale and
one offset per frame, and keep the AI's detail with the stereo's metres.
"""

from __future__ import annotations

import cv2
import numpy as np

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def fit_scale_offset(
    relative: np.ndarray,
    disparity: np.ndarray,
    iterations: int = 3,
    keep: float = 0.8,
    max_samples: int = 200_000,
) -> tuple[float, float] | None:
    """Least-squares ``disparity ~ scale * relative + offset``, refit after
    dropping the worst (1 - keep) residuals each round. None if too few
    valid pixels.
    """
    mask = np.isfinite(disparity) & np.isfinite(relative)
    x = relative[mask].astype(np.float64)
    y = disparity[mask].astype(np.float64)
    if x.size < 100:
        return None
    if x.size > max_samples:
        pick = np.random.default_rng(0).choice(x.size, max_samples, replace=False)
        x, y = x[pick], y[pick]

    inliers = np.ones(x.size, dtype=bool)
    scale = offset = 0.0
    for _ in range(iterations):
        a = np.stack([x[inliers], np.ones(inliers.sum())], axis=1)
        (scale, offset), *_ = np.linalg.lstsq(a, y[inliers], rcond=None)
        residual = np.abs(scale * x + offset - y)
        inliers = residual <= np.quantile(residual, keep)
    return float(scale), float(offset)


class MonoDepth:
    """Depth Anything V2 through Hugging Face transformers."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str | None = None):
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise SystemExit(
                "AI refinement needs the extras: pip install 'vr180-depth[ai]'"
            ) from exc
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.model = model
        self._pipe = pipeline("depth-estimation", model=model, device=device)

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        """Relative inverse depth (bigger = closer), same size as ``bgr``."""
        from PIL import Image

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pred = self._pipe(Image.fromarray(rgb))["predicted_depth"]
        rel = pred.squeeze().float().cpu().numpy()
        h, w = bgr.shape[:2]
        return cv2.resize(rel, (w, h), interpolation=cv2.INTER_CUBIC)
