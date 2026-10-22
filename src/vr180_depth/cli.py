"""Command line: ``vr180-depth run`` and ``vr180-depth demo``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .fuse import DEFAULT_MODEL
from .pipeline import Settings, run


def _add_settings(p: argparse.ArgumentParser) -> None:
    d = Settings()
    g = p.add_argument_group("input")
    g.add_argument("--layout", choices=["sbs", "tb"], default=d.layout, help="side-by-side or top-bottom eyes")
    g.add_argument("--projection", choices=["equirect", "flat"], default=d.projection,
                   help="equirect = VR180 eyes; flat = eyes already rectified pinhole images")
    g.add_argument("--swap-eyes", action="store_true", help="right eye is stored first")
    g.add_argument("--start", type=int, default=d.start, help="first frame index")
    g.add_argument("--every", type=int, default=d.every, help="process every Nth frame")
    g.add_argument("--max-frames", type=int, default=d.max_frames, help="stop after N frames")

    g = p.add_argument_group("output view")
    g.add_argument("--width", type=int, default=d.width)
    g.add_argument("--height", type=int, default=d.height)
    g.add_argument("--fov", type=float, default=d.fov_deg, help="horizontal field of view in degrees")
    g.add_argument("--pitch", type=float, default=d.pitch_deg, help="tilt the view up (+) or down (-), degrees")

    g = p.add_argument_group("depth")
    g.add_argument("--baseline", type=float, default=d.baseline_m,
                   help="distance between the lenses in metres (default is a guess; calibrate it)")
    g.add_argument("--min-depth", type=float, default=d.min_depth_m, help="closest distance to search, metres")
    g.add_argument("--near", type=float, default=d.near_m, help="matte is white at this distance")
    g.add_argument("--far", type=float, default=d.far_m, help="matte is black at this distance")
    g.add_argument("--block-size", type=int, default=d.block_size, help="SGBM block size (odd)")
    g.add_argument("--align", choices=["once", "every", "off"], default=d.align,
                   help="fix vertical eye offset: measure once, every frame, or never")
    g.add_argument("--no-fill", action="store_true", help="leave stereo holes empty")
    g.add_argument("--ai", action="store_true", help="refine with Depth Anything V2 (needs the [ai] extras)")
    g.add_argument("--ai-model", default=DEFAULT_MODEL)
    g.add_argument("--no-preview", action="store_true", help="skip preview.mp4 / preview.png")


def _settings(a: argparse.Namespace) -> Settings:
    return Settings(
        layout=a.layout, projection=a.projection, swap_eyes=a.swap_eyes,
        width=a.width, height=a.height, fov_deg=a.fov, pitch_deg=a.pitch,
        baseline_m=a.baseline, min_depth_m=a.min_depth, near_m=a.near, far_m=a.far,
        block_size=a.block_size, align=a.align, fill=not a.no_fill,
        ai=a.ai, ai_model=a.ai_model, every=a.every, start=a.start,
        max_frames=a.max_frames, preview=not a.no_preview,
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="vr180-depth", description="VR180 / stereo video to per-frame depth maps.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="make depth maps from a stereo video or photo")
    r.add_argument("input", type=Path)
    r.add_argument("-o", "--out", type=Path, required=True)
    _add_settings(r)

    d = sub.add_parser("demo", help="render a synthetic VR180 clip with known depth, then run it")
    d.add_argument("-o", "--out", type=Path, default=Path("demo-out"))
    d.add_argument("--frames", type=int, default=48)
    _add_settings(d)

    a = p.parse_args(argv)
    s = _settings(a)
    if a.cmd == "demo":
        from .synthetic import render_demo_video

        clip = render_demo_video(a.out / "demo-input.mp4", frames=a.frames)
        print(f"rendered {clip} (card 1.0 to 2.5 m, wall 4.0 m)")
        meta = run(clip, a.out, s)
    else:
        meta = run(a.input, a.out, s)
    print(f"wrote {meta['frames_written']} frames to {a.out}")


if __name__ == "__main__":
    main()
