"""Pinhole intrinsics shared by the detector, tracker, simulator and RealSense source."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_fov(cls, width: int, height: int, fovy_deg: float) -> "Intrinsics":
        f = (height / 2) / np.tan(np.radians(fovy_deg) / 2)
        return cls(f, f, (width - 1) / 2, (height - 1) / 2, width, height)

    @classmethod
    def from_realsense(cls, rs_intr) -> "Intrinsics":
        return cls(rs_intr.fx, rs_intr.fy, rs_intr.ppx, rs_intr.ppy, rs_intr.width, rs_intr.height)

    def as_dict(self) -> dict:
        return asdict(self)

    def project(self, P: np.ndarray) -> np.ndarray:
        """Nx3 camera-frame points (OpenCV convention: x right, y down, z forward) -> Nx2 pixels."""
        z = P[:, 2]
        return np.stack([self.fx * P[:, 0] / z + self.cx, self.fy * P[:, 1] / z + self.cy], axis=1)

    def backproject(self, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Pixel coordinates and depth (m) -> Nx3 camera-frame points."""
        return np.stack([(u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z], axis=1)

    def rays(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Pixel coordinates -> Nx3 ray directions with z = 1."""
        return np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u, dtype=float)], axis=1)
