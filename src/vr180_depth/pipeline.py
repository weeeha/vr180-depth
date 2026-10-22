"""Frame loop: split eyes -> flatten -> align -> match -> depth -> files."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from . import __version__
from .align import estimate_vertical_offset, shift_vertical
from .export import depth_to_matte16, depth_to_mm16, preview_frame
from .fuse import DEFAULT_MODEL, fit_scale_offset, fuse
from .projection import PinholeView, apply_maps, equirect_to_pinhole_maps, split_stereo
from .stereo import compute_disparity, disparity_to_depth, fill_holes, max_disparity_for

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


@dataclass
class Settings:
    layout: str = "sbs"  # sbs | tb
    projection: str = "equirect"  # equirect (VR180) | flat (already pinhole)
    swap_eyes: bool = False
    width: int = 1280
    height: int = 720
    fov_deg: float = 90.0
    pitch_deg: float = 0.0
    baseline_m: float = 0.065  # unverified for the Calf; see README
    min_depth_m: float = 0.4
    near_m: float = 0.4
    far_m: float = 20.0
    block_size: int = 5
    align: str = "once"  # once | every | off
    fill: bool = True
    ai: bool = False
    ai_model: str = DEFAULT_MODEL
    ai_tolerance: float = 0.2  # stereo overrides AI past this relative gap
    every: int = 1
    start: int = 0
    max_frames: int | None = None
    preview: bool = True


@dataclass
class FrameResult:
    view: np.ndarray
    depth_m: np.ndarray
    vertical_offset_px: float
    fit: tuple[float, float] | None
    stereo_share: float | None = None  # with --ai: pixels where stereo overrode AI


class FrameProcessor:
    """Stateful per-clip processor: caches remap tables, the eye offset and
    a smoothed AI fit so they stay steady from frame to frame.
    """

    FIT_SMOOTHING = 0.3  # weight of each new frame's scale/offset

    def __init__(self, settings: Settings):
        self.s = settings
        self.view: PinholeView | None = None
        self._maps = None
        self._dy: float | None = None
        self._fit: tuple[float, float] | None = None
        self._mono = None
        if settings.ai:
            from .fuse import MonoDepth

            self._mono = MonoDepth(settings.ai_model)

    def _flatten(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left, right = split_stereo(frame, self.s.layout, self.s.swap_eyes)
        if self.s.projection == "flat":
            if self.view is None:
                self.view = PinholeView(left.shape[1], left.shape[0], self.s.fov_deg)
            return left, right
        if self._maps is None:
            self.view = PinholeView(self.s.width, self.s.height, self.s.fov_deg)
            self._maps = equirect_to_pinhole_maps(left.shape[1], left.shape[0], self.view, self.s.pitch_deg)
        return apply_maps(left, self._maps), apply_maps(right, self._maps)

    def process(self, frame: np.ndarray) -> FrameResult:
        left, right = self._flatten(frame)
        f = self.view.focal_px
        num_disp = max_disparity_for(f, self.s.baseline_m, self.s.min_depth_m)
        lg = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        rg = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        if self.s.align != "off" and (self._dy is None or self.s.align == "every"):
            self._dy = estimate_vertical_offset(lg, rg, num_disp) or 0.0
        dy = self._dy or 0.0
        if abs(dy) > 0.25:
            rg = shift_vertical(rg, dy)

        disp = compute_disparity(lg, rg, num_disp, self.s.block_size)

        fit = stereo_share = None
        if self._mono is not None:
            rel = self._mono(left)
            new = fit_scale_offset(rel, disp)
            if new is not None:
                a = self.FIT_SMOOTHING
                self._fit = new if self._fit is None else (
                    a * new[0] + (1 - a) * self._fit[0],
                    a * new[1] + (1 - a) * self._fit[1],
                )
            if self._fit is not None:
                fit = self._fit
                disp, stereo_share = fuse(rel, disp, fit[0], fit[1], self.s.ai_tolerance)

        if fit is None and self.s.fill:
            disp = fill_holes(disp)

        depth = disparity_to_depth(disp, f, self.s.baseline_m)
        return FrameResult(left, depth, dy, fit, stereo_share)


def iter_frames(path: Path, start: int = 0, every: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    if path.suffix.lower() in IMAGE_SUFFIXES:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"could not read image {path}")
        yield 0, img
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"could not open video {path}")
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx = start
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - start) % every == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()


def video_fps(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
    cap.release()
    return fps if fps and fps > 0 else 24.0


def run(input_path: str | Path, out_dir: str | Path, s: Settings, log: Callable[[str], None] = print) -> dict:
    """Process a clip or photo. Writes view/, depth_mm/, matte/, preview.mp4
    and meta.json into ``out_dir``. Returns the metadata.
    """
    src, out = Path(input_path), Path(out_dir)
    dirs = {name: out / name for name in ("view", "depth_mm", "matte")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    proc = FrameProcessor(s)
    is_image = src.suffix.lower() in IMAGE_SUFFIXES
    writer = None
    written, offsets = 0, []
    for idx, frame in iter_frames(src, s.start, max(1, s.every)):
        if s.max_frames is not None and written >= s.max_frames:
            break
        r = proc.process(frame)
        matte = depth_to_matte16(r.depth_m, s.near_m, s.far_m)
        name = f"{idx:06d}.png"
        cv2.imwrite(str(dirs["view"] / name), r.view)
        cv2.imwrite(str(dirs["depth_mm"] / name), depth_to_mm16(r.depth_m))
        cv2.imwrite(str(dirs["matte"] / name), matte)

        if s.preview:
            pf = preview_frame(r.view, matte)
            if is_image:
                cv2.imwrite(str(out / "preview.png"), pf)
            else:
                if writer is None:
                    fps = video_fps(src) / max(1, s.every)
                    h, w = pf.shape[:2]
                    writer = cv2.VideoWriter(str(out / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(pf)

        offsets.append(r.vertical_offset_px)
        written += 1
        if written == 1 or written % 10 == 0:
            valid = np.isfinite(r.depth_m)
            med = float(np.nanmedian(r.depth_m)) if valid.any() else float("nan")
            msg = f"frame {idx}: {valid.mean():.0%} valid, median depth {med:.2f} m"
            if r.stereo_share is not None:
                msg += f", stereo overrode AI on {r.stereo_share:.0%}"
            log(msg)
    if writer is not None:
        writer.release()
    if written == 0:
        raise SystemExit("no frames processed; check --start / --every / --max-frames")

    view = proc.view
    meta = {
        "tool": f"vr180-depth {__version__}",
        "source": str(src),
        "frames_written": written,
        "view": {"width": view.width, "height": view.height, "fov_deg": view.fov_deg, "focal_px": view.focal_px},
        "vertical_offset_px": offsets[0] if offsets else 0.0,
        "encoding": {
            "view": "8-bit BGR flat view of the left eye; depth maps line up with this image",
            "depth_mm": "uint16 depth in millimetres; 0 = unknown",
            "matte": f"uint16, linear in 1/depth: 65535 at {s.near_m} m, 0 at {s.far_m} m and beyond",
        },
        "settings": asdict(s),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta
