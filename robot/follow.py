"""Hover over a cube and follow it as it moves (shared by scripts.follow_cube and robot.controller).

Each new perception frame gives a hover target: straight above the cube, fingertips at HOVER (8 cm above a
cube's top), fingers along the cube's faces (the alignment nearest the current wrist angle, so the wrist
never flips 90 deg), solved with IK warm-started from the last target (0.2 ms). The arm is re-aimed with
one smooth driver-interpolated move (quintic, >= 0.3 s, path checked) when the target has moved, at most
every REAIM s; SafeArm.follow_check guards tracking and unexpected effort in between. (Driver 1.8.x handles
fast streamed position commands badly: see robot/arm.py.)
Followed only while the cube is at table height: a cube lifted in a hand is not chased upward (the gripper
would come down onto it); the arm waits where it is.
"""

import json
import time

import numpy as np

from .grasp import face_yaw
from .ik import IK, gripper_down_rotation

HOVER = 0.14  # fingertip height (arm frame, m): 8 cm above a cube's top; straight-down reach is 20-34 cm here
RADIUS = (0.20, 0.34)
FLOOR = 0.11  # fingertips never below this while following
LIFTED = 0.06  # cube centre above this (m, arm frame): lifted off the table, do not follow
REAIM = 0.15  # s between re-aims at most


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class HoverFollower:
    """height: fingertip height of the hover (arm frame, m); floor: the lowest the fingertips may go on a
    re-aim. Defaults: HOVER / FLOOR (empty gripper). The controller also uses one at carry height while
    holding the top cube, to follow the bottom cube."""

    def __init__(self, arm, ik: IK | None = None, vmax=1.0, log_path=None, height=HOVER, floor=FLOOR):
        self.arm, self.ik = arm, ik or IK()
        self.height, self.floor = height, floor
        self.vmax = min(vmax, arm.limits.max_joint_speed)
        self.q_tgt = self.q_goal = None
        self.t_aim, self.yaw, self.last_search = 0.0, None, 0.0
        self.active = False
        self.log_path, self.last_log = log_path, 0.0

    def hover_point(self, p):
        """Hover point over p, moved into the reachable ring if needed; and whether it was moved."""
        x, y = p[0], p[1]
        r0 = np.hypot(x, y)
        r = np.clip(r0, *RADIUS)
        a = np.arctan2(y, x)
        a = np.clip(a, -np.arccos(0.1 / r), np.arccos(0.1 / r))  # keep x >= 0.1 (in front of the base)
        return np.array([r * np.cos(a), r * np.sin(a), self.height]), bool(r0 < RADIUS[0] or r0 > RADIUS[1])

    def pick_yaw(self, R):
        """Finger direction along the cube's faces: the alignment nearest the last one (or, at first, the
        gripper's current finger direction). Always aligned, so any wrist turn happens while hovering and the
        descent to grasp needs none; since the last choice was aligned, the nearest is within a few degrees
        of it while the cube turns, so it never flips 90 deg. Changes under 3 deg are ignored."""
        y0 = face_yaw(R)
        cands = [y0 + k * np.pi / 2 for k in range(4)]
        ref = self.yaw
        if ref is None:
            G = self.ik.fk(self.arm.q())[1]
            ref = np.arctan2(G[1, 1], G[0, 1])
        best = min(cands, key=lambda c: abs(wrap(c - ref)))
        d = wrap(best - ref)
        return ref if self.yaw is not None and abs(d) < np.radians(3) else ref + d

    def aim(self, p, R):
        """New hover target from the cube's pose (arm frame). Returns 'lifted', 'edge' (outside the ring:
        aiming at the nearest reachable point), 'ok', or 'no ik' (target kept)."""
        if p[2] > LIFTED:
            return "lifted"
        pt, clamped = self.hover_point(p)
        yaw = self.pick_yaw(R)
        seed = self.q_tgt if self.q_tgt is not None else self.arm.q()
        q, ep, er = self.ik.solve(pt, gripper_down_rotation(yaw), seed)
        if (ep >= 2e-3 or er >= 2.0) and time.time() - self.last_search > 1.0:
            # far from the last solution (e.g. the first target, from the watch pose): multi-seed search,
            # at most once a second so the loop keeps its rate
            self.last_search = time.time()
            az = np.arctan2(pt[1], pt[0])
            seeds = [np.array([az, a, b, w, 0, 0]) for a in (0.4, 0.9, 1.4) for b in (0.4, 1.0, 1.6) for w in (0.5, 1.2)]
            q, ep, er = min((self.ik.solve(pt, gripper_down_rotation(yaw), sd) for sd in seeds),
                            key=lambda r: r[1] + r[2] / 100)
        if ep >= 2e-3 or er >= 2.0:
            return "no ik"
        self.q_tgt, self.yaw = q, yaw
        return "edge" if clamped else "ok"

    def start(self):
        self.arm.follow_begin()
        self.q_goal = self.arm.q()
        if self.q_tgt is None:
            self.q_tgt = self.q_goal.copy()
        self.t_aim, self.active = 0.0, True

    def stop(self):
        """Stop following (the arm finishes its current smooth move and holds there)."""
        self.active = False

    def step(self):
        """Guards every tick; re-aim when the target moved (about 2 mm at the gripper) and REAIM has passed."""
        self.arm.follow_check()
        if time.time() - self.t_aim >= REAIM and np.abs(self.q_tgt - self.q_goal).max() > 0.008:
            self.arm.follow_goal(self.q_tgt, self.vmax, self.floor)
            self.q_goal, self.t_aim = self.q_tgt.copy(), time.time()

    def settled(self):
        """True once the arm has reached its last re-aim goal (nearly still at the hover point)."""
        return self.q_goal is not None and np.abs(self.arm.q() - self.q_goal).max() < 0.01

    def reset(self):
        """Forget the last target (e.g. a new cube, or a new height)."""
        self.q_tgt = self.q_goal = None
        self.yaw, self.active = None, False

    def log(self, state, p_cube):
        if self.log_path is None or time.time() - self.last_log < 0.05:
            return
        self.last_log = time.time()
        tip = self.arm.model.fk(self.arm.q())[0]
        tgt = self.arm.model.fk(self.q_tgt)[0] if self.q_tgt is not None else None
        rec = {"t": round(time.time(), 3), "state": state,
               "cube": None if p_cube is None else np.round(p_cube * 1000, 1).tolist(),
               "target": None if tgt is None else np.round(tgt * 1000, 1).tolist(),
               "tip": np.round(tip * 1000, 1).tolist(),
               "effort_peak": np.round(getattr(self.arm, "follow_peak", np.zeros(6)), 2).tolist()}
        self.arm.follow_peak = np.zeros(6)  # peak |unexpected effort| per joint since the last record
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
