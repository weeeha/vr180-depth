"""A synthetic VR180 scene with known depth, for tests and the demo.

Two parallel eyes ``baseline_m`` apart look at a textured card in front of
a textured wall. Every pixel is ray traced into an equirect eye, so the
parallax is exact and the true depth of each surface is known.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _texture(seed: int, size: int = 512, sigma: float = 1.2) -> np.ndarray:
    t = np.random.default_rng(seed).random((size, size)).astype(np.float32)
    # Blur a wrapped copy so the texture tiles without seams.
    pad = int(4 * sigma) + 1
    t = cv2.GaussianBlur(np.pad(t, pad, mode="wrap"), (0, 0), sigma)[pad:-pad, pad:-pad]
    t -= t.min()
    return t / t.max()


def _sample(tex: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    n = tex.shape[0]
    mx = np.mod(u, n).astype(np.float32)
    my = np.mod(v, n).astype(np.float32)
    return cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def render_vr180_sbs(
    eye_size: int = 1024,
    baseline_m: float = 0.065,
    card_z: float = 1.5,
    wall_z: float = 4.0,
    card_half: tuple[float, float] = (0.5, 0.35),
    card_x: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Side-by-side VR180 frame (BGR uint8), left eye on the left."""
    wall_tex, card_tex = _texture(seed), _texture(seed + 1)
    angles = ((np.arange(eye_size) + 0.5) / eye_size - 0.5) * np.pi
    lon, lat = np.meshgrid(angles, angles)
    dx, dy, dz = np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)
    forward = dz > 1e-3
    dz = np.where(forward, dz, 1.0)

    eyes = []
    for ox in (-baseline_m / 2, baseline_m / 2):
        t_card = card_z / dz
        cx, cy = ox + t_card * dx, t_card * dy
        hit = forward & (np.abs(cx - card_x) <= card_half[0]) & (np.abs(cy) <= card_half[1])
        t_wall = wall_z / dz
        wall = _sample(wall_tex, (ox + t_wall * dx) * 20, t_wall * dy * 20)
        card = _sample(card_tex, cx * 40, cy * 40)

        img = np.zeros((eye_size, eye_size, 3), np.float32)
        img[:] = (wall * 0.85)[..., None] * np.float32([1.0, 0.8, 0.6])
        img[hit] = (0.3 + 0.7 * card[hit])[:, None] * np.float32([0.4, 0.7, 1.0])
        img[~forward] = 0
        eyes.append(img)

    return (np.clip(np.hstack(eyes), 0, 1) * 255).astype(np.uint8)


def render_demo_video(path: str | Path, frames: int = 48, eye_size: int = 1024, fps: float = 24.0) -> Path:
    """Short clip: the card drifts toward and away from the camera."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (eye_size * 2, eye_size))
    for i in range(frames):
        phase = 2 * np.pi * i / frames
        frame = render_vr180_sbs(
            eye_size=eye_size,
            card_z=1.75 + 0.75 * np.sin(phase),
            card_x=0.3 * np.cos(phase),
        )
        writer.write(frame)
    writer.release()
    return path
