"""Camera-to-arm transform from assets/calibration.yaml (scripts/calibrate_camera.py)."""

from pathlib import Path

import numpy as np
import yaml

CALIB = Path(__file__).resolve().parents[1] / "assets" / "calibration.yaml"


class CameraToBase:
    def __init__(self, path=CALIB):
        c = yaml.safe_load(Path(path).read_text())
        self.T = np.array(c["T_base_cam"], dtype=float)
        self.k = float(c.get("depth_scale", 1.0))
        self.rms_mm = c.get("rms_mm")

    def point(self, p_cam):
        """Camera-frame point (m) -> arm base frame (m)."""
        return self.T[:3, :3] @ (self.k * np.asarray(p_cam, dtype=float)) + self.T[:3, 3]

    def rotation(self, R_cam):
        return self.T[:3, :3] @ np.asarray(R_cam, dtype=float)
