import json

import cv2
import numpy as np

from vr180_depth.pipeline import Settings, run
from vr180_depth.synthetic import render_vr180_sbs


def test_run_on_photo_writes_outputs(tmp_path):
    photo = tmp_path / "sbs.png"
    cv2.imwrite(str(photo), render_vr180_sbs(eye_size=768))
    meta = run(photo, tmp_path / "out", Settings(width=480, height=270), log=lambda _: None)

    out = tmp_path / "out"
    assert meta["frames_written"] == 1
    assert json.loads((out / "meta.json").read_text())["view"]["width"] == 480
    depth = cv2.imread(str(out / "depth_mm" / "000000.png"), cv2.IMREAD_UNCHANGED)
    assert depth.dtype == np.uint16 and depth.shape == (270, 480)
    assert (out / "matte" / "000000.png").exists()
    assert (out / "view" / "000000.png").exists()
    assert (out / "preview.png").exists()
