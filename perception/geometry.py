"""Cube constants, rigid transforms and cube-symmetry-aware pose errors."""

import itertools
from pathlib import Path

import numpy as np

CUBE_SIZE = 0.05  # m, edge length
HALF = CUBE_SIZE / 2


def make_T(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _cube_symmetries() -> np.ndarray:
    """The 24 proper rotations that map a cube onto itself (signed permutation matrices)."""
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for row, (col, s) in enumerate(zip(perm, signs)):
                m[row, col] = s
            if np.linalg.det(m) > 0:
                mats.append(m)
    return np.array(mats)


CUBE_SYMMETRIES = _cube_symmetries()


def _sym_traces(R_a: np.ndarray, R_b: np.ndarray) -> np.ndarray:
    # trace(R_a^T R_b S_k) for every symmetry S_k
    M = R_a.T @ R_b
    return np.einsum("ij,kji->k", M, CUBE_SYMMETRIES)


def rot_error_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Smallest rotation angle between two cube orientations, modulo the cube's symmetry."""
    c = np.clip((_sym_traces(R_a, R_b).max() - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def closest_symmetric(R_ref: np.ndarray, R: np.ndarray) -> np.ndarray:
    """The symmetric equivalent of R that is closest to R_ref."""
    return R @ CUBE_SYMMETRIES[np.argmax(_sym_traces(R_ref, R))]


def cube_corners(T: np.ndarray, half: float = HALF) -> np.ndarray:
    """8x3 corner positions of a cube with pose T."""
    signs = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)
    return (T[:3, :3] @ (signs * half).T).T + T[:3, 3]


def write_cube_obj(path: Path, size: float = CUBE_SIZE) -> Path:
    """Triangulated cube centered at the origin, counter-clockwise faces, units in meters."""
    h = size / 2
    # vertex index = (x>0) + 2*(y>0) + 4*(z>0)
    verts = [((i & 1) * 2 - 1, ((i >> 1) & 1) * 2 - 1, ((i >> 2) & 1) * 2 - 1) for i in range(8)]
    faces = [
        (1, 3, 7), (1, 7, 5),  # +x
        (0, 4, 6), (0, 6, 2),  # -x
        (2, 6, 7), (2, 7, 3),  # +y
        (0, 1, 5), (0, 5, 4),  # -y
        (4, 5, 7), (4, 7, 6),  # +z
        (0, 2, 3), (0, 3, 1),  # -z
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"v {x * h:.6f} {y * h:.6f} {z * h:.6f}" for x, y, z in verts]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n")
    return path
