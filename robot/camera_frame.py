"""Camera-to-arm transform from assets/calibration.yaml (scripts/calibrate_camera.py).

A calibration belongs to one camera: check_serial() refuses a live camera whose serial differs from the one the
file was fitted with, so another camera can never silently use it (ignore_serial=True overrides, deliberately).
"""

from pathlib import Path

import numpy as np
import yaml

CALIB = Path(__file__).resolve().parents[1] / "assets" / "calibration.yaml"


class CameraMismatch(RuntimeError):
    """The live camera is not the one assets/calibration.yaml was fitted with."""


def mismatch_message(live, calibrated):
    return (f"camera {live} != calibration {calibrated}: recalibrate (scripts.calibrate_camera) or connect the "
            f"calibrated camera")


class CameraToBase:
    def __init__(self, path=CALIB, ignore_serial=False):
        c = yaml.safe_load(Path(path).read_text())
        self.T = np.array(c["T_base_cam"], dtype=float)
        self.k = float(c.get("depth_scale", 1.0))
        self.rms_mm = c.get("rms_mm")
        self.serial = str(c["camera_serial"]) if c.get("camera_serial") else None
        self.ignore_serial = ignore_serial

    def check_serial(self, live):
        """Raise CameraMismatch unless `live` (the live camera's serial; None if the source does not report one) is
        the camera this calibration was fitted with."""
        if self.ignore_serial:
            return
        if live is None or self.serial is None or str(live) != self.serial:
            raise CameraMismatch(mismatch_message(live or "unknown", self.serial or "unknown"))

    def point(self, p_cam):
        """Camera-frame point (m) -> arm base frame (m)."""
        return self.T[:3, :3] @ (self.k * np.asarray(p_cam, dtype=float)) + self.T[:3, 3]

    def rotation(self, R_cam):
        return self.T[:3, :3] @ np.asarray(R_cam, dtype=float)
