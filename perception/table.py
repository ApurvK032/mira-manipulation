"""Support-plane prior: a cube resting on the table (or on another cube) has one face up.

The table plane is fitted from depth (dominant plane), so this needs no camera-to-table calibration.
"""

import numpy as np

from .camera import Intrinsics
from .geometry import HALF, make_T


def fit_table_plane(depth: np.ndarray, K: Intrinsics, depth_scale=0.001, stride=8, iters=80, thresh=0.005, seed=0):
    """Dominant plane n.x = d in the camera frame, n pointing toward the camera. None if not found."""
    rng = np.random.default_rng(seed)
    z = depth[::stride, ::stride].astype(np.float64) * depth_scale
    vv, uu = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    ok = (z > 0.1) & (z < 3.0)
    if ok.sum() < 200:
        return None
    P = K.backproject(uu[ok].astype(float), vv[ok].astype(float), z[ok])
    idx = rng.integers(0, len(P), size=(iters, 3))
    a, b, c = P[idx[:, 0]], P[idx[:, 1]], P[idx[:, 2]]
    nrm = np.cross(b - a, c - a)
    ln = np.linalg.norm(nrm, axis=1)
    good = ln > 1e-9
    nrm, a = nrm[good] / ln[good, None], a[good]
    off = np.einsum("kj,kj->k", nrm, a)
    counts = (np.abs(nrm @ P.T - off[:, None]) < thresh).sum(axis=1)
    k = int(np.argmax(counts))
    inl = np.abs(P @ nrm[k] - off[k]) < thresh
    if inl.mean() < 0.25:
        return None
    Q = P[inl]
    cen = Q.mean(axis=0)
    n = np.linalg.svd(Q - cen, full_matrices=False)[2][2]
    if n @ cen > 0:
        n = -n
    return n, float(n @ cen)


def upright(T: np.ndarray, plane, tol=0.006):
    """If the cube rests on the plane or on one cube, return (tilt_deg, T snapped face-up); else None."""
    if plane is None:
        return None
    n, d = plane
    h = float(n @ T[:3, 3] - d)
    if not (abs(h - HALF) < tol or abs(h - 3 * HALF) < tol):
        return None
    R = T[:3, :3]
    dots = R.T @ n
    k = int(np.argmax(np.abs(dots)))
    axis = R[:, k] * np.sign(dots[k])
    tilt = float(np.degrees(np.arccos(np.clip(abs(dots[k]), -1, 1))))
    v = np.cross(axis, n)
    s, c = np.linalg.norm(v), float(axis @ n)
    if s < 1e-9:
        return tilt, T
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R_align = np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)
    # keep the center on the support: height snapped to exactly half or one and a half edges
    h_snap = HALF if abs(h - HALF) < tol else 3 * HALF
    center = T[:3, 3] + (h_snap - h) * n
    return tilt, make_T(R_align @ R, center)
