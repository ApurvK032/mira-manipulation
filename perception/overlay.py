"""Draw cube estimates on the color image."""

import cv2
import numpy as np

from .camera import Intrinsics
from .geometry import cube_corners

EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
STATE_BGR = {"TRACKING": (60, 220, 60), "OCCLUDED": (0, 200, 255), "LOST": (60, 60, 230)}


def draw_cube(img, K: Intrinsics, T, bgr, thickness=2):
    c = cube_corners(T)
    if (c[:, 2] < 0.05).any():
        return
    uv = np.round(K.project(c)).astype(int)
    for a, b in EDGES:
        cv2.line(img, tuple(uv[a]), tuple(uv[b]), bgr, thickness, cv2.LINE_AA)


def draw_result(img, K: Intrinsics, result, colors, gt=None, fps_text=""):
    out = img.copy()
    if gt:
        for T in gt.values():
            if T is not None:
                draw_cube(out, K, T, (255, 255, 255), 1)
    y = 18
    for n, est in result.cubes.items():
        if est.T is not None:
            draw_cube(out, K, est.T, STATE_BGR[est.state] if est.state != "TRACKING" else colors[n].bgr, 2)
        txt = f"{n:6s} {est.state:8s}"
        if est.T is not None:
            p = est.T[:3, 3] * 1000
            txt += f" ({p[0]:6.1f},{p[1]:6.1f},{p[2]:6.1f}) mm"
        cv2.putText(out, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, STATE_BGR[est.state], 1, cv2.LINE_AA)
        y += 18
    t = result.timing_ms
    txt = f"detect {t['detect']:.1f}  track {t['track']:.1f}  check {t['check']:.1f}  total {t['total']:.1f} ms {fps_text}"
    cv2.putText(out, txt, (8, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, txt, (8, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out
