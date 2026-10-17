"""Does a cube pose agree with the current image? This is the tracker's "am I lost?" signal.

The cube at pose T is ray-cast into the image. On its pixels that are not hidden by something in
front of it (measured depth well in front of the model), we check the color mask and the depth. Cube-
colored pixels just outside the silhouette mean the pose is lagging a moving cube.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import Intrinsics
from .geometry import HALF, cube_corners


@dataclass(frozen=True)
class Check:
    visible: float  # fraction of the model silhouette not hidden by an occluder
    color: float  # fraction of unhidden silhouette pixels with the cube's color
    depth: float  # fraction of unhidden silhouette pixels within depth tolerance
    leak: float  # cube-colored pixels just outside the silhouette / inside it
    n_pixels: int

    # Thresholds from the sim runs: correct poses have depth agreement >= 0.95 in 99.96% of frames,
    # wrong ones mostly 0.70-0.93. Re-check the distribution on the real camera (run_live --log).
    def ok(self, min_visible=0.3, min_color=0.8, min_depth=0.95, max_leak=0.2) -> bool:
        return (self.visible >= min_visible and self.color >= min_color and self.depth >= min_depth
                and self.leak <= max_leak)

    def score(self) -> float:
        return self.color + self.depth - self.leak


EMPTY = Check(0.0, 0.0, 0.0, 1.0, 0)


class ConsistencyChecker:
    def __init__(self, K: Intrinsics, depth_scale=0.001, occlusion_margin=0.012, depth_tol=0.008, ring_px=4,
                 occluder_band_px=4, smear_tol=0.015, min_incidence_cos=0.25):
        self.K = K
        self.depth_scale = depth_scale
        self.occlusion_margin = occlusion_margin
        self.depth_tol = depth_tol
        self.ring_px = ring_px
        self.occluder_band_px = occluder_band_px  # D455 at 848x480 smears depth edges by about 2-4 px
        self.smear_tol = smear_tol  # a smeared pixel's depth is within this of the occluder next to it
        self.min_incidence_cos = min_incidence_cos  # faces seen beyond about 75 deg from face-on are not depth-scored

    @staticmethod
    def _raycast(T: np.ndarray, d: np.ndarray, with_incidence=False):
        """z-depth where rays d (Nx3, z = 1) hit the cube at pose T, and a hit mask (slab test).
        With with_incidence, also |cos| of the angle between each ray and the face it hits."""
        R, t = T[:3, :3], T[:3, 3]
        o_c = -R.T @ t
        d_c = d @ R  # = (R^T d)^T
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-HALF - o_c) / d_c
            t2 = (HALF - o_c) / d_c
        near = np.nan_to_num(np.minimum(t1, t2), nan=-np.inf)
        tmin = np.max(near, axis=1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=1)
        hit = (tmin <= tmax) & (tmax > 0)
        if not with_incidence:
            return tmin, hit
        axis = np.argmax(near, axis=1)  # the slab entered last is the face hit
        cos = np.abs(d_c[np.arange(len(d)), axis]) / np.linalg.norm(d, axis=1)
        return tmin, hit, cos

    def check(self, T: np.ndarray, mask: np.ndarray, depth: np.ndarray, others=()) -> Check:
        """others: poses of the other cubes; where one of them is in front, the pixel is occluded."""
        K, H, W = self.K, self.K.height, self.K.width
        corners = cube_corners(T)
        if (corners[:, 2] < 0.1).any():
            return EMPTY
        uv = K.project(corners)
        pad = self.ring_px + 2
        x0, y0 = np.floor(uv.min(axis=0)).astype(int) - pad
        x1, y1 = np.ceil(uv.max(axis=0)).astype(int) + pad
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1 - x0 < 3 or y1 - y0 < 3:
            return EMPTY

        # ray-cast the cube (slab test in the cube frame); with z = 1 rays, t is the z-depth
        vv, uu = np.mgrid[y0:y1, x0:x1]
        d = K.rays(uu.ravel().astype(float), vv.ravel().astype(float))
        tmin, hit, cos_inc = self._raycast(T, d, with_incidence=True)
        sil = hit.reshape(vv.shape)
        cos_inc = cos_inc.reshape(vv.shape)
        if sil.sum() < 10:
            return EMPTY
        z_model = tmin.reshape(vv.shape)
        known = np.zeros_like(sil)
        for To in others:
            if np.linalg.norm(To[:3, 3] - T[:3, 3]) > 0.12:
                continue
            zo, ho = self._raycast(To, d)
            known |= (ho & (zo < tmin - 0.002)).reshape(vv.shape)

        kernel = np.ones((3, 3), np.uint8)
        sil8 = sil.astype(np.uint8)
        inner = cv2.erode(sil8, kernel, iterations=1).astype(bool)  # ignore the blurry boundary
        if inner.sum() < 10:
            inner = sil
        outer = cv2.dilate(sil8, kernel, iterations=self.ring_px).astype(bool)
        ring = outer & ~cv2.dilate(sil8, kernel, iterations=2).astype(bool)

        z_meas = depth[y0:y1, x0:x1].astype(np.float64) * self.depth_scale
        m = mask[y0:y1, x0:x1] > 0
        valid = z_meas > 0
        # something in front of the model hides it, unless that something has this cube's own color:
        # a cube cannot hide behind itself, so that means the pose is behind the real cube
        occluded = (valid & (z_meas < z_model - self.occlusion_margin) & ~m) | (known & sil)
        unocc = inner & ~occluded
        n_inner, n_unocc = int(inner.sum()), int(unocc.sum())
        if n_unocc == 0:
            return Check(0.0, 0.0, 0.0, 0.0, n_inner)
        color = float((unocc & m).sum() / n_unocc)
        # real stereo depth smears 2-3 px across a cube's outline, so depth is scored further inside
        deep = cv2.erode(sil8, kernel, iterations=3).astype(bool)
        if deep.sum() < 10:
            deep = inner
        # stereo "edge fattening": an occluder's depth smears a few px onto the cube behind it. Those
        # pixels are cube-colored but read too close, so they are skipped for depth (not for color).
        # Only excuse what smearing can explain: a pixel next to an occluder that reads too close *and*
        # at the occluder's own depth. A wrong pose's mismatches show some other surface and still count.
        occ_px = occluded & valid
        inv = np.where(occ_px, 1.0 / np.maximum(z_meas, 1e-3), 0.0).astype(np.float32)
        inv_near = cv2.dilate(inv, kernel, iterations=self.occluder_band_px)  # closest occluder nearby
        z_occ = np.where(inv_near > 0, 1.0 / np.maximum(inv_near, 1e-6), np.inf)
        smeared = (valid & ~occluded & (z_meas < z_model - self.depth_tol)
                   & (np.abs(z_meas - z_occ) < self.smear_tol))
        # Faces seen nearly edge-on have noisy stereo depth, and a tiny pose error moves their depth a lot:
        # skip them, and widen the tolerance with the viewing angle elsewhere (8 mm when seen face-on).
        uv_valid = deep & ~occluded & ~smeared & valid & (cos_inc > self.min_incidence_cos)
        tol = self.depth_tol / np.maximum(cos_inc, 0.5)
        depth_ok = float((uv_valid & (np.abs(z_meas - z_model) < tol)).sum() / max(1, uv_valid.sum()))
        # colored pixels outside that are not explained by an occluder covering the cube
        leak = float((ring & m).sum() / max(1, (sil & m).sum()))
        return Check(n_unocc / n_inner, color, depth_ok, leak, n_inner)
