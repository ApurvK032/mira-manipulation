"""Per-frame cube perception: global detector + M3T tracking + consistency check.

States per cube:
  TRACKING  pose verified against this frame's color and depth (safe to act on)
  OCCLUDED  too little of the cube is visible to verify; pose is the tracker's best guess
  LOST      no pose; the detector will pick it up as soon as the cube is visible again
"""

import time
from dataclasses import dataclass, field

import numpy as np

from .camera import Intrinsics
from .consistency import Check, ConsistencyChecker, EMPTY
from .detector import CubeDetector
from .geometry import closest_symmetric, make_T
from .table import fit_table_plane, upright

TRACKING, OCCLUDED, LOST = "TRACKING", "OCCLUDED", "LOST"
# depth agreement is the reliable signal (see consistency.py); color can be lower under hand shadows
ACQUIRE = dict(min_visible=0.5, min_color=0.75, min_depth=0.95, max_leak=0.15)
FULL_VIEW = 0.85  # a check this unhidden fully constrains the cube's rotation
FAST_STEP = 0.008  # m per frame (about 0.5 m/s at 60 fps)


@dataclass
class CubeEstimate:
    name: str
    state: str
    T: np.ndarray | None  # 4x4 pose in the camera frame
    source: str = ""  # track | reinit | detector
    check: Check = EMPTY


@dataclass
class FrameResult:
    cubes: dict = field(default_factory=dict)
    timing_ms: dict = field(default_factory=dict)


class CubePerception:
    def __init__(self, K: Intrinsics, names, colors=None, use_m3t=True, depth_scale=0.001,
                 max_occluded_frames=60, model_dir=None):
        self.K = K
        self.names = list(names)
        self.detector = CubeDetector(K, colors=colors, depth_scale=depth_scale)
        self.checker = ConsistencyChecker(K, depth_scale=depth_scale)
        self.m3t = None
        if use_m3t:
            from .tracker import M3TCubeTrackers, MODEL_DIR

            self.m3t = M3TCubeTrackers(K, self.names, model_dir or MODEL_DIR, depth_scale=depth_scale)
        self.state = {n: LOST for n in self.names}
        self.T = {n: None for n in self.names}
        self.occluded_frames = {n: 0 for n in self.names}
        self.last_full = {n: None for n in self.names}  # pose at the last mostly-unhidden verified frame
        self.max_occluded_frames = max_occluded_frames
        self.iteration = 0
        self.depth_scale = depth_scale
        self.plane = None  # table plane (n, d) in the camera frame, refreshed from depth

    def process(self, color_bgr: np.ndarray, depth_u16: np.ndarray) -> FrameResult:
        it = self.iteration
        self.iteration += 1
        res = FrameResult()
        t0 = time.perf_counter()
        if self.plane is None or it % 30 == 0:
            self.plane = fit_table_plane(depth_u16, self.K, self.depth_scale) or self.plane
        dout = self.detector.detect(color_bgr, depth_u16, colors=self.names)
        t1 = time.perf_counter()
        if self.m3t is not None:
            self.m3t.set_images(color_bgr, depth_u16)
        t_track = t_check = 0.0
        for n in self.names:
            est, dt_tr, dt_ch = self._update(n, it, dout.detections.get(n), dout.masks[n], depth_u16)
            t_track += dt_tr
            t_check += dt_ch
            res.cubes[n] = est
        t2 = time.perf_counter()
        res.timing_ms = {"detect": 1e3 * (t1 - t0), "track": 1e3 * t_track, "check": 1e3 * t_check,
                         "total": 1e3 * (t2 - t0)}
        return res

    def _update(self, n, it, det, mask, depth):
        dt_track = dt_check = 0.0
        cands = []  # (source, T, check)
        prev = self.T[n]
        # other cubes known to be here (e.g. the cube stacked on this one) explain the pixels they cover
        others = [self.T[m] for m in self.names if m != n and self.state[m] != LOST and self.T[m] is not None]

        if self.m3t is not None and self.m3t.started[n] and self.state[n] != LOST:
            a = time.perf_counter()
            T_trk = self.m3t.track(n, it)
            b = time.perf_counter()
            cands.append(("track", T_trk, self.checker.check(T_trk, mask, depth, others)))
            dt_track += b - a
            dt_check += time.perf_counter() - b
        tracked_ok = bool(cands) and cands[0][2].ok()

        # temporal prior: a partly hidden cube that is not moving keeps the rotation it had when it
        # was last fully visible (M3T's rotation can drift when something passes in front of the cube)
        full = self.last_full[n]
        if (cands and full is not None and cands[0][2].visible < FULL_VIEW
                and np.linalg.norm(cands[0][1][:3, 3] - full[:3, 3]) < 0.003):
            T_hold = make_T(closest_symmetric(cands[0][1][:3, :3], full[:3, :3]), cands[0][1][:3, 3])
            a = time.perf_counter()
            cands.append(("hold", T_hold, self.checker.check(T_hold, mask, depth, others)))
            dt_check += time.perf_counter() - a

        # re-initialize from the global detector when tracking failed, or when the detector sees the
        # cube somewhere else than the tracker (then both are verified and the better one wins)
        need_detector = det is not None and (
            not tracked_ok or np.linalg.norm(det.T[:3, 3] - cands[0][1][:3, 3]) > 0.03)
        if need_detector:
            T_det = det.T
            if prev is not None:
                T_det = make_T(closest_symmetric(prev[:3, :3], det.T[:3, :3]), det.T[:3, 3])
            a = time.perf_counter()
            cands.append(("detector", T_det, self.checker.check(T_det, mask, depth, others)))
            dt_check += time.perf_counter() - a
            if self.m3t is not None:
                a = time.perf_counter()
                T_ref = self.m3t.reset(n, T_det, it)
                b = time.perf_counter()
                cands.append(("reinit", T_ref, self.checker.check(T_ref, mask, depth, others)))
                dt_track += b - a
                dt_check += time.perf_counter() - b

        # hysteresis: a new lock (from the detector) needs stronger evidence than keeping one
        good = [c for c in cands if (c[2].ok() if c[0] in ("track", "hold") else c[2].ok(**ACQUIRE))]
        # support prior: a cube resting on the table or on a cube is face-up; fix small tilts, and do not
        # trust a resting pose that is clearly tilted unless the face-up version also checks out
        prior = []
        for src, T, chk in good:
            up = upright(T, self.plane)
            if up is None or up[0] <= 3.0:
                prior.append((src, T, chk))
                continue
            a = time.perf_counter()
            chk_up = self.checker.check(up[1], mask, depth, others)
            dt_check += time.perf_counter() - a
            if chk_up.ok():
                prior.append((src, up[1], chk_up))
            elif up[0] <= 8.0:
                prior.append((src, T, chk))
        good = prior
        if good:
            held = [c for c in good if c[0] == "hold"]
            src, T, chk = held[0] if held else max(good, key=lambda c: c[2].score())
            if (prev is not None and chk.visible < 0.6 and self.state[n] != LOST
                    and np.linalg.norm(T[:3, 3] - prev[:3, 3]) > FAST_STEP):
                # moving fast and mostly hidden: being carried by a hand. Keep tracking, do not vouch for it.
                if self.m3t is not None:
                    self.m3t.set_pose(n, T)
                self.state[n], self.T[n] = OCCLUDED, T
                return CubeEstimate(n, OCCLUDED, T, src, chk), dt_track, dt_check
            if chk.visible >= FULL_VIEW:
                self.last_full[n] = T
            if self.m3t is not None and src == "detector":
                self.m3t.reset(n, T, it, refine_steps=0)  # continue tracking from the verified pose
            elif self.m3t is not None:
                self.m3t.set_pose(n, T)  # continue from the chosen (possibly face-up corrected) pose
            self.state[n], self.T[n] = TRACKING, T
            self.occluded_frames[n] = 0
            return CubeEstimate(n, TRACKING, T, src, chk), dt_track, dt_check

        trk = cands[0] if cands and cands[0][0] == "track" else None
        if trk is not None and trk[2].visible < 0.3 and self.occluded_frames[n] < self.max_occluded_frames:
            if self.m3t is not None and need_detector:
                self.m3t.set_pose(n, trk[1])
            self.state[n], self.T[n] = OCCLUDED, trk[1]
            self.occluded_frames[n] += 1
            return CubeEstimate(n, OCCLUDED, trk[1], "track", trk[2]), dt_track, dt_check

        self.state[n] = LOST
        self.occluded_frames[n] = 0
        # last_full is kept: if the cube is found again at the same spot (e.g. the bottom cube of a new
        # stack, never fully visible again), its rotation from before is still the best one
        best = max(cands, key=lambda c: c[2].score())[2] if cands else EMPTY
        return CubeEstimate(n, LOST, None, "", best), dt_track, dt_check
