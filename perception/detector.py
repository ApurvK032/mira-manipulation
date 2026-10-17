"""Global per-frame cube detector: color mask + depth -> 6D cube pose (up to cube symmetry).

It runs on every frame with no memory, so a cube that moved anywhere in view is found again in the
next frame. Its pose is accurate to a few mm; M3T refines it.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from .camera import Intrinsics
from .geometry import CUBE_SIZE, HALF, make_T


@dataclass
class ColorSpec:
    name: str
    # list of (lo, hi) OpenCV HSV bounds, H in [0, 179]
    hsv_ranges: list
    bgr: tuple = (255, 255, 255)  # drawing color


DEFAULT_COLORS = {
    # low brightness floors so shaded faces and dim light still count; saturation separates the cubes
    # from wood and skin. Tune for real lighting with scripts/tune_colors.py.
    "red": ColorSpec("red", [((0, 140, 30), (7, 255, 255)), ((172, 140, 30), (179, 255, 255))], (40, 40, 230)),
    "blue": ColorSpec("blue", [((100, 120, 25), (128, 255, 255))], (230, 90, 30)),
    "green": ColorSpec("green", [((48, 100, 25), (85, 255, 255))], (60, 200, 60)),
    "yellow": ColorSpec("yellow", [((20, 150, 30), (35, 255, 255))], (40, 220, 240)),
}


def load_colors(path=None) -> dict:
    """Color specs from a YAML file written by scripts/tune_colors.py, else the defaults."""
    from pathlib import Path

    import yaml

    path = Path(path) if path else Path(__file__).resolve().parents[1] / "assets" / "colors.yaml"
    if not path.exists():
        return dict(DEFAULT_COLORS)
    data = yaml.safe_load(path.read_text())
    return {n: ColorSpec(n, [tuple(map(tuple, r)) for r in d["hsv_ranges"]], tuple(d.get("bgr", (255, 255, 255))))
            for n, d in data.items()}


@dataclass
class Detection:
    color: str
    T: np.ndarray  # 4x4 cube pose in the camera frame
    n_pixels: int
    bbox: tuple  # x, y, w, h
    extent: tuple  # visible face extent (m) along the two in-face axes
    n_faces: int
    quality: float  # 0..1, how complete the visible faces are


@dataclass
class DetectorOutput:
    detections: dict = field(default_factory=dict)  # color -> best Detection
    masks: dict = field(default_factory=dict)  # color -> uint8 mask (full frame)
    rejected: dict = field(default_factory=dict)  # color -> list of rejection reasons


def _plane_ransac(P: np.ndarray, rng: np.random.Generator, thresh: float, iters: int = 48):
    n = len(P)
    if n < 12:
        return None
    idx = rng.integers(0, n, size=(iters, 3))
    a, b, c = P[idx[:, 0]], P[idx[:, 1]], P[idx[:, 2]]
    normals = np.cross(b - a, c - a)
    norms = np.linalg.norm(normals, axis=1)
    ok = norms > 1e-9
    if not ok.any():
        return None
    normals = normals[ok] / norms[ok, None]
    offsets = np.einsum("kj,kj->k", normals, a[ok])
    dist = np.abs(normals @ P.T - offsets[:, None])
    counts = (dist < thresh).sum(axis=1)
    best = int(np.argmax(counts))
    inliers = dist[best] < thresh
    # least-squares refit on the inliers
    Q = P[inliers]
    centroid = Q.mean(axis=0)
    _, _, vt = np.linalg.svd(Q - centroid, full_matrices=False)
    normal = vt[2]
    if normal @ centroid > 0:  # face the camera (camera at the origin)
        normal = -normal
    inliers = np.abs((P - centroid) @ normal) < thresh
    return normal, float(normal @ centroid), inliers


def _robust_span(x: np.ndarray, lo: float = 2, hi: float = 98):
    a, b = np.percentile(x, [lo, hi])
    return float(a), float(b)


class CubeDetector:
    def __init__(
        self,
        intrinsics: Intrinsics,
        colors: dict | None = None,
        depth_scale: float = 0.001,
        min_pixels: int = 300,  # a cube at 0.5-0.8 m covers about 1,800-2,900 px; partly hidden ones still > 300
        max_pixels: int = 12000,  # larger than a cube can appear at >= 0.35 m (wall or furniture patches)
        max_blobs: int = 3,  # per color, largest plausible first: background specks are skipped cheaply
        max_points: int = 700,
        plane_thresh: float = 0.003,
        max_range: float = 2.5,
        seed: int = 0,
    ):
        self.K = intrinsics
        self.colors = colors or DEFAULT_COLORS
        self.depth_scale = depth_scale
        self.min_pixels = min_pixels
        self.max_blobs = max_blobs
        self.max_pixels = max_pixels
        self.max_points = max_points
        self.plane_thresh = plane_thresh
        self.max_range = max_range  # m; cubes on the bench are well inside this
        self.rng = np.random.default_rng(seed)
        self.kernel = np.ones((3, 3), np.uint8)

    def color_masks(self, color_bgr: np.ndarray) -> dict:
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        masks = {}
        for name, spec in self.colors.items():
            m = None
            for lo, hi in spec.hsv_ranges:
                r = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
                m = r if m is None else cv2.bitwise_or(m, r)
            masks[name] = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.kernel)
        return masks

    def detect(self, color_bgr: np.ndarray, depth: np.ndarray, colors=None) -> DetectorOutput:
        out = DetectorOutput()
        out.masks = self.color_masks(color_bgr)
        for name in colors or self.colors:
            mask = out.masks[name]
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            best, reasons = None, []
            order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
            areas = stats[order, cv2.CC_STAT_AREA]
            too_big = int((areas > self.max_pixels).sum())
            if too_big:
                reasons.extend(["too many pixels for a cube"] * too_big)
            tried = 0
            for i in order[(areas <= self.max_pixels) & (areas >= self.min_pixels)]:
                if tried >= self.max_blobs:
                    break
                area = int(stats[i, cv2.CC_STAT_AREA])
                x, y, w, h = (int(v) for v in stats[i, :4])
                # cheap size test before the 3D fit: at its depth, a 5 cm cube covers at most about
                # 2.5 face areas (three faces seen at an angle); walls and screens of a cube color are larger
                vs, us = np.nonzero(labels[y : y + h, x : x + w] == i)
                zs = depth[vs[::7] + y, us[::7] + x]
                zs = zs[zs > 0]
                if len(zs) >= 10:
                    face_px = (self.K.fx * CUBE_SIZE / (np.median(zs) * self.depth_scale)) ** 2
                    if area > 2.5 * face_px * 1.5:
                        reasons.append("too many pixels for a cube at its depth")
                        continue
                tried += 1
                pad = 10  # room for the ring of surrounding pixels used by the flatness test
                xa, ya = max(0, x - pad), max(0, y - pad)
                xb, yb = min(mask.shape[1], x + w + pad), min(mask.shape[0], y + h + pad)
                det, why = self._blob_pose(name, labels[ya:yb, xa:xb] == i, depth[ya:yb, xa:xb], xa, ya)
                if det is None:
                    reasons.append(why)
                    continue
                det.n_pixels, det.bbox = area, (x, y, w, h)
                if best is None or det.quality * det.n_pixels > best.quality * best.n_pixels:
                    best = det
            if best is not None:
                out.detections[name] = best
            if reasons:
                out.rejected[name] = reasons
        return out

    def _blob_pose(self, name, blob, depth_roi, x0, y0):
        # sample depth away from the blob border, where stereo depth smears across edges
        inner = cv2.erode(blob.astype(np.uint8), self.kernel, iterations=2).astype(bool)
        if inner.sum() < 20:
            inner = blob
        vs, us = np.nonzero(inner)
        z = depth_roi[vs, us].astype(np.float64) * self.depth_scale
        valid = z > 0.05
        if valid.sum() < 20:
            return None, "no depth"
        us, vs, z = us[valid] + x0, vs[valid] + y0, z[valid]
        zm = np.median(z)
        if zm > self.max_range:
            return None, "beyond working range"
        keep = np.abs(z - zm) < 0.06
        if keep.sum() < 20:
            return None, "depth too scattered"
        us, vs, z = us[keep], vs[keep], z[keep]
        if len(z) > self.max_points:
            sel = self.rng.choice(len(z), self.max_points, replace=False)
            us, vs, z = us[sel], vs[sel], z[sel]
        P = self.K.backproject(us.astype(float), vs.astype(float), z)

        # every visible cube point lies within one cube diagonal of the visible points' centroid
        if np.percentile(np.linalg.norm(P - P.mean(axis=0), axis=1), 98) > CUBE_SIZE * np.sqrt(3):
            return None, "too large for a cube"

        fit1 = _plane_ransac(P, self.rng, self.plane_thresh)
        if fit1 is None:
            return None, "no plane"
        n1, d1, in1 = fit1
        if in1.sum() < 15:
            return None, "no plane"

        # a cube stands out from what surrounds it; a colored patch flush with the table (tape, paper,
        # a blurred ghost) has its surroundings on the same plane
        b8 = blob.astype(np.uint8)
        ring = cv2.dilate(b8, self.kernel, iterations=7).astype(bool) & ~cv2.dilate(b8, self.kernel, iterations=3).astype(bool)
        rv, ru = np.nonzero(ring)
        rz = depth_roi[rv, ru].astype(np.float64) * self.depth_scale
        rok = rz > 0.05
        if rok.sum() >= 30:
            R3 = self.K.backproject((ru[rok] + x0).astype(float), (rv[rok] + y0).astype(float), rz[rok])
            if np.mean(np.abs(R3 @ n1 - d1) < 0.004) > 0.5:
                return None, "flat patch, not a cube"

        # optional second face
        n2 = d2 = None
        rest = P[~in1]
        if len(rest) >= max(20, 0.12 * len(P)):
            fit2 = _plane_ransac(rest, self.rng, self.plane_thresh)
            if fit2 is not None and fit2[2].sum() >= 15 and abs(fit2[0] @ n1) < 0.26:
                n2 = fit2[0] - (fit2[0] @ n1) * n1
                n2 /= np.linalg.norm(n2)
                d2 = float(n2 @ rest[fit2[2]].mean(axis=0))

        F1 = P[in1]
        if n2 is not None:
            a2 = n2
        else:
            # in-face orientation from the minimum-area rectangle of the face points
            u_ax = np.cross(n1, [0.0, 0.0, 1.0] if abs(n1[2]) < 0.9 else [1.0, 0.0, 0.0])
            u_ax /= np.linalg.norm(u_ax)
            v_ax = np.cross(n1, u_ax)
            uv = np.stack([F1 @ u_ax, F1 @ v_ax], axis=1).astype(np.float32)
            (_, _), (_, _), ang = cv2.minAreaRect(uv * 1000.0)
            th = np.radians(ang)
            a2 = np.cos(th) * u_ax + np.sin(th) * v_ax
        a3 = np.cross(n1, a2)

        s2_lo, s2_hi = _robust_span(F1 @ a2)
        s3_lo, s3_hi = _robust_span(P @ a3)
        ext2, ext3 = s2_hi - s2_lo, s3_hi - s3_lo
        if max(ext2, ext3) > CUBE_SIZE + 0.012:
            return None, "face too large"
        if max(ext2, ext3) < 0.018:
            return None, "face too small"

        c1 = d1 - HALF  # along n1: the face is half an edge in front of the center
        c2 = d2 - HALF if n2 is not None else 0.5 * (s2_lo + s2_hi)
        c3 = 0.5 * (s3_lo + s3_hi)
        center = c1 * n1 + c2 * a2 + c3 * a3
        R = np.stack([a2, a3, n1], axis=1)
        quality = float(min(1.0, ext2 / CUBE_SIZE) * min(1.0, ext3 / CUBE_SIZE))
        return Detection(name, make_T(R, center), 0, (0, 0, 0, 0), (ext2, ext3), 2 if n2 is not None else 1, quality), ""
