"""Always-on stacking controller: wake up, follow the top cube while it moves, stack it on the bottom one
once both rest, and keep watching. The arm goes back to rest only when the operator asks for it.

Main loop (about 20 Hz; 100 Hz while following):
  top already on bottom (verified)                    -> wait at the WATCH pose (cubes in full view)
  both resting on the table, reachable, fingers clear -> one pick-and-place attempt, straight down from the
                                                         hover point when already over the cube
  otherwise, top cube in view on the table            -> hover over it and follow it (robot.follow), so the
                                                         pick starts right above it the moment it rests
  otherwise                                           -> wait at WATCH; the banner says what it waits for
A stacked cube ends at WATCH. A failed attempt (missed grasp, drop, bad placement) rises and goes straight
after the cube again; a cube moved before the grasp is just followed again (not counted as a failure).

During the pick and place the cubes are re-read about every 50 ms:
  carrying: the bottom cube moved -> stop, hold the cube until the bottom one rests, re-plan, go on
  lowering onto it: it moved -> stop, rise, re-plan
  holding the cube with no way to place it (bottom cube gone or unreachable for 30 s) -> put the cube
    back where it was picked, if that spot is clear
Operator commands (buttons on the live page): pause (stop and hold), resume, rest (put down anything held,
go to rest, brake), start (wake up from rest). A dry run (--dry) takes commands from runs/dry_command.json.
"""

import json
import signal
import threading
import time
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path

import numpy as np

from .arm import REST, WATCH, ArmModel, Limits, MoveLog, SafeArm, SafetyStop, minjerk
from .camera_frame import CameraMismatch, CameraToBase
from .follow import HoverFollower
from .grasp import (HALF, REACH, TABLE_Z, GripperShape, RestTracker, check_live_camera, cubes_in_base, plan_top_down,
                    read_stats, top_face_tilt_deg)
from .ik import IK
from .reaim import plan_ff, quintic
from .stream import Streamer

ROOT = Path(__file__).resolve().parents[1]
GRASP = 0.010  # fingertips 10 mm above the cube centre: hold its top 15 mm
HANG = GRASP + HALF  # fingertips to the bottom face of the held cube
PLACE = 2 * HALF + 0.002 + GRASP  # fingertips over the bottom cube's centre: held cube 2 mm above contact
HELD = 0.010  # finger position (m) above which the closed gripper is holding the cube
OPEN = 0.031  # finger position (m) above which the gripper is open (fully open is 0.035)


def is_holding(g):
    """Closed on a cube: the fingers stopped between nearly shut and open."""
    return HELD <= g <= OPEN
COMMANDS = ("pause", "rest")
BOTTOM_STILL = 0.5  # s the bottom cube must rest before the held cube is lowered onto it
DRY_COMMANDS = ROOT / "runs" / "dry_command.json"


def read_command(stats: dict, dry: bool) -> dict:
    """Latest operator command {seq, c}. The real arm takes only the live page's buttons; a dry run (no
    arm) takes only runs/dry_command.json, so a test can never command the real arm, and the page's
    buttons never drive a test."""
    if dry:
        try:
            return json.loads(DRY_COMMANDS.read_text())
        except (OSError, ValueError):
            return {}
    return stats.get("command") or {}


class Stop(Exception):
    """An attempt ended early for a reason that is not a failure of the arm (a command, a moved cube)."""


class Retarget(Exception):
    """The top cube moved before the grasp: go back to following it (not a failed attempt)."""


class Poller(threading.Thread):
    """Reads the live viewer's stats in the background at about 50 Hz; .latest / .t are the newest."""

    def __init__(self):
        super().__init__(daemon=True)
        self.latest, self.t = None, 0.0

    def run(self):
        while True:
            try:
                self.latest, self.t = read_stats(timeout=0.5), time.time()
            except OSError:
                pass
            time.sleep(0.02)


class DryArm:
    """Stands in for SafeArm with no hardware: same checks and timing, motion only simulated."""

    def __init__(self):
        self.limits = Limits()
        self.model = ArmModel()
        m = self.model.m
        rng = np.array([m.joint(f"joint_{i}").range for i in range(6)])
        self.q_min_raw, self.q_max_raw = rng[:, 0], rng[:, 1]
        self.q_min = self.q_min_raw + self.limits.joint_margin
        self.q_max = self.q_max_raw - self.limits.joint_margin
        self.v_max = np.full(6, 3.0)
        self._q, self._g = REST.copy(), 0.0
        self.position_mode = False

    check_target, _check_path, _check_ff = SafeArm.check_target, SafeArm._check_path, SafeArm._check_ff

    def q(self):
        return self._q.copy()

    def move_joints(self, q_goal, speed=None, label="move", gripper_floor=None, watch=None, watch_every=0.05):
        q_goal, q0 = np.asarray(q_goal, dtype=float), self._q.copy()
        self.check_target(q_goal, q0, gripper_floor=gripper_floor)
        speed = min(speed or self.limits.max_joint_speed, self.limits.max_joint_speed)
        dq = np.abs(q_goal - q0)
        T = max(self.limits.min_move_time, 1.875 * dq.max() / speed)
        log, t0 = MoveLog(), time.time()
        self._traj = None
        while (t := time.time() - t0) < T:
            self._q = q0 + (q_goal - q0) * minjerk(t / T)
            if watch is not None and (why := watch(t)):
                log.stopped = why
                return log
            time.sleep(watch_every)
        self._q = q_goal.copy()
        self.position_mode = True
        return log

    def hold(self):
        if getattr(self, "_traj", None) is not None:
            self._q, _ = self._state()
            self._traj = None

    def follow_begin(self):
        self.position_mode = True
        self._traj = None  # quintic from the state at the last re-aim (models the driver blending)
        self._v = np.zeros(6)

    def _state(self):
        if self._traj is None:
            return self._q.copy(), np.zeros(6)
        t0, T, c = self._traj
        s = min(time.time() - t0, T)
        q = sum(c[k] * s**k for k in range(6))
        v = sum(k * c[k] * s**(k - 1) for k in range(1, 6))
        return q, (v if s < T else np.zeros(6))

    def follow_goal(self, q_goal, speed, gripper_floor, min_time=0.3, accel=3.0, goal_velocity=None):
        q_goal = np.asarray(q_goal, dtype=float)
        q0, v0 = self._state()
        self._q = q0
        if goal_velocity is not None:  # feedforward re-aim (robot.reaim.plan_ff), as SafeArm.follow_goal
            g, v_end, T = plan_ff(q0, v0, q_goal, np.asarray(goal_velocity, dtype=float),
                                  min(speed, self.limits.max_joint_speed), accel, min_time, self.v_max, self.q_min, self.q_max)
            if np.any(v_end):
                try:
                    self._check_ff(q0, v0, g, v_end, T, gripper_floor)
                    self._traj = (time.time(), T, quintic(q0, v0, g, v_end, T))
                    return T
                except SafetyStop:
                    pass  # near a limit or the floor: a plain re-aim instead (below)
        self.check_target(q_goal, q0, n=21, gripper_floor=gripper_floor)
        dq = np.abs(q_goal - q0)
        T = max(min_time, 1.875 * dq.max() / min(speed, self.limits.max_joint_speed), np.sqrt(5.77 * dq.max() / accel))
        # quintic: start at (q0, v0, 0), end at (q_goal, 0, 0)
        d = q_goal - q0
        c = [q0, v0, np.zeros(6), (10 * d - 6 * v0 * T) / T**3, (-15 * d + 8 * v0 * T) / T**4, (6 * d - 3 * v0 * T) / T**5]
        self._traj = (time.time(), T, c)
        return T

    def follow_check(self):
        self._q, self._v = self._state()

    def stream_begin(self, rate=100.0, speed=None, accel=4.0, tau=0.06, window=0.3):
        q0, _ = self._state()
        self._q, self._traj, self.position_mode = q0, None, True
        self._st = Streamer(rate, min(speed or self.limits.max_joint_speed, self.limits.max_joint_speed), accel, tau)
        self._st.reset(q0)
        self._st_goal = q0.copy()
        self.stream_rejects = 0

    def stream_tick(self, q_goal, gripper_floor):
        """As SafeArm.stream_tick; the driver's straight-line move to each point over one tick is modelled as a
        first-degree trajectory from the current commanded state."""
        # clip to this arm's own joint limits (the IK clips to the model's, which can differ by a mrad): a policy
        # pushing a joint to its limit is followed to the limit instead of stopping the run
        q_goal = np.clip(np.asarray(q_goal, dtype=float), self.q_min, self.q_max)
        last = self._st.q.copy()
        if np.abs(q_goal - self._st_goal).max() > 1e-6:
            try:
                self.check_target(q_goal, last, n=11, gripper_floor=gripper_floor)
                self._st_goal, self.stream_rejects = q_goal.copy(), 0
            except SafetyStop:
                # a goal that fails the checks is not followed: keep heading to the last accepted one, and stop only
                # if goals keep failing for 0.5 s (a target grazing the floor for a tick should not end the attempt)
                self.stream_rejects += 1
                if self.stream_rejects * self._st.dt >= 0.5:
                    raise
        q_next, _ = self._st.step(self._st_goal)
        # the generator can overshoot its goal by a few mrad: keep each point inside the joint limits, stopping there
        if np.any(q_next < self.q_min) or np.any(q_next > self.q_max):
            at = (q_next < self.q_min) | (q_next > self.q_max)
            q_next = np.clip(q_next, self.q_min, self.q_max)
            self._st.q, self._st.v = q_next.copy(), np.where(at, 0.0, self._st.v)
        self.check_target(q_next, last, n=2, gripper_floor=gripper_floor)
        q0, _ = self._state()
        dt = self._st.dt
        self._traj = (time.time(), dt, [q0, (q_next - q0) / dt, np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)])
        return q_next

    def velocities(self):
        return self._state()[1] if getattr(self, "_traj", None) is not None else np.zeros(6)

    def gripper_position(self):
        return self._g

    def open_gripper(self, position=0.035, duration=0.8, blocking=True):
        self._g = position
        return position

    def close_gripper(self, force=None, settle=0.8, mode="external_effort"):
        self._g = 0.027  # pretend it closed on the cube
        return self._g

    def squeeze_gripper(self, force=20.0, start=None, timeout=6.0):
        self._g = 0.027
        return self._g

    def gripper_idle(self):
        pass

    def rest_and_idle(self, speed=None):
        self.move_joints(REST, speed=speed, label="to rest")
        self.position_mode = False

    def close(self):
        pass


class StackController:
    def __init__(self, arm, cal: CameraToBase, top="red", bottom="green", still_s=1.0, log_path=None,
                 speeds=None):
        self.arm, self.cal, self.top, self.bottom = arm, cal, top, bottom
        self.dry = isinstance(arm, DryArm)  # a dry run neither takes the page's buttons nor writes its banner
        self.ik, self.shape = IK(), GripperShape()
        self.rest_tracker = RestTracker(still_s=still_s)
        self.recent = {}  # name -> deque of (t, center) over the last 0.3 s, for jitter-free checks
        self.cubes, self.stale, self.t_stats = {}, True, 0.0
        self.cmd_seq, self.pending = None, None
        self.mode = "rest"  # rest | awake | paused
        self.holding, self.pick_plan = False, None
        self.last_status, self.blocked = None, None
        self.fails, self.max_fails = 0, 4  # pause after this many failed attempts in a row
        self.log_path = log_path
        self.v = dict(approach=1.0, descend=0.45, lift=0.5, carry=0.8, lower=0.3, rise=0.6, watch=1.0)
        self.v.update(speeds or {})
        self.follow = HoverFollower(arm, ik=self.ik)  # hover over the top cube while it moves
        self.carry_follow = HoverFollower(arm, ik=self.ik)  # holding it: follow the bottom cube at carry height
        self.lost_since = None
        self.poller = Poller()
        self.poller.start()

    # ----- perception and commands -----
    def refresh(self):
        """Take in the newest perception snapshot (from the background poller), if there is a new one."""
        s, t = self.poller.latest, self.poller.t
        if s is None or time.time() - t > 0.5:
            self.stale = True
            return
        self.stale = False
        if t == self.t_stats:
            return
        now = self.t_stats = t
        self.cubes = cubes_in_base(s, self.cal)
        self.rest_tracker.update(now, self.cubes)
        for name, (state, p, R, _) in self.cubes.items():
            d = self.recent.setdefault(name, deque())
            if state == "TRACKING" and p is not None:
                d.append((now, p, R))
            while d and now - d[0][0] > 0.3:
                d.popleft()
        cmd = read_command(s, self.dry)
        if self.cmd_seq is None:
            self.cmd_seq = cmd.get("seq", 0)  # ignore anything pressed before this run started
        elif cmd.get("seq", 0) != self.cmd_seq:
            self.cmd_seq = cmd.get("seq", 0)
            self.pending = cmd.get("c")
            self.event("command", command=self.pending)

    def where(self, name, window=0.3):
        """Jitter-free current centre: median of the last `window` s of TRACKING samples (None if none)."""
        d = self.recent.get(name)
        if not d or time.time() - d[-1][0] > 0.3:
            return None
        now = d[-1][0]
        return np.median(np.array([x[1] for x in d if now - x[0] <= window]), axis=0)

    def rotation(self, name):
        """Latest TRACKING rotation (None if none in the last 0.3 s)."""
        d = self.recent.get(name)
        return d[-1][2] if d and time.time() - d[-1][0] <= 0.3 else None

    def status(self, msg):
        if msg == self.last_status:
            return
        self.last_status = msg
        print(time.strftime("%H:%M:%S"), msg, flush=True)
        if self.dry:
            return
        try:
            urllib.request.urlopen("http://127.0.0.1:8080/status?msg=" + urllib.parse.quote(msg[:200]), timeout=0.5)
        except OSError:
            pass

    def event(self, kind, **kw):
        if self.log_path is None:
            return
        rec = {"t": round(time.time(), 2), "event": kind}
        rec.update({k: (np.round(v * 1000, 1).tolist() if isinstance(v, np.ndarray) else v) for k, v in kw.items()})
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    # ----- planning -----
    def make_plan(self, ct, Rt, cb, Rb, start):
        table_z = min(ct[2], cb[2]) - HALF
        carry_z = max(cb[2] + HALF + HANG + 0.03, ct[2] + 0.085)
        qt, yaw_t = plan_top_down(ct, Rt, {"above": HALF + 0.06, "grasp": GRASP, "carry": carry_z - ct[2]},
                                  ik=self.ik, obstacles=[(self.bottom, cb, Rb)], start=start, shape=self.shape,
                                  seed=start)
        qb, yaw_b = plan_top_down(cb, Rb, {"carry": carry_z - cb[2], "place": PLACE}, ik=self.ik, seed=qt["carry"])
        P = dict(ct=ct, Rt=Rt, cb=cb, Rb=Rb, table_z=table_z, carry_z=carry_z, qt=qt, qb=qb, yaw_t=yaw_t)
        g, c = table_z + 0.012, carry_z - 0.02
        for a, b, f in [(start, qt["above"], None), (qt["above"], qt["grasp"], g), (qt["grasp"], qt["carry"], g),
                        (qt["carry"], qb["carry"], c), (qb["carry"], qb["place"], None), (qb["place"], qb["carry"], None),
                        (qb["carry"], WATCH, None)]:
            self.arm.check_target(b, a, gripper_floor=f)
        return P

    def replan_place(self, P, cb, Rb, start):
        qb, _ = plan_top_down(cb, Rb, {"carry": P["carry_z"] - cb[2], "place": PLACE}, ik=self.ik, seed=start)
        self.arm.check_target(qb["carry"], start, gripper_floor=P["carry_z"] - 0.02)
        self.arm.check_target(qb["place"], qb["carry"])
        P.update(cb=cb, Rb=Rb, qb=qb)

    # ----- watch callbacks (run during moves) -----
    def watch_top(self, ct):
        unseen = [None]

        def f(t):
            self.refresh()
            if self.pending in COMMANDS:
                return self.pending
            if self.stale:
                return "camera feed stalled"
            p = self.where(self.top)
            if p is None:
                unseen[0] = unseen[0] or time.time()
                return f"{self.top} cube out of sight" if time.time() - unseen[0] > 0.3 else None
            unseen[0] = None
            # horizontal only: the gripper hovering over the cube pulls its estimated height down about 3 mm
            # (it hides part of the top face); its x, y stay within about 1 mm
            if np.linalg.norm(p[:2] - ct[:2]) > 0.006:
                return f"{self.top} cube moved"
            return None
        return f

    def watch_descent(self, ct):
        """Coming down onto the top cube: stop if it is moved sideways. The fingers hide it more and more
        on the way down, so moments out of sight are not a reason to stop (unlike watch_top)."""
        def f(t):
            self.refresh()
            if self.pending in COMMANDS:
                return self.pending
            p = self.where(self.top)
            if p is not None and np.linalg.norm(p[:2] - ct[:2]) > 0.006:
                return f"{self.top} cube moved"
            return None
        return f

    def watch_bottom(self, cb, tol):
        def f(t):
            self.refresh()
            if self.pending in COMMANDS:
                return self.pending
            if self.arm.gripper_position() < HELD:
                return "dropped"
            p = self.where(self.bottom)  # the held cube and the gripper can hide it: only act on a clear move
            if p is not None and np.linalg.norm(p[:2] - cb[:2]) > tol:
                return f"{self.bottom} cube moved"
            return None
        return f

    def watch_commands(self, t):
        self.refresh()
        return self.pending if self.pending in COMMANDS else None

    # ----- one attempt -----
    def attempt(self, ct, Rt, cb, Rb):
        """Pick the top cube and place it on the bottom one. Returns None on a verified stack, else why not.
        Raises Stop for operator commands; SafetyStop from the arm passes through."""
        arm, v = self.arm, self.v
        t0 = time.time()
        P = self.make_plan(ct, Rt, cb, Rb, start=arm.q())
        self.event("plan", ms=round((time.time() - t0) * 1000), top=ct, bottom=cb, yaw=round(float(np.degrees(P["yaw_t"]))))
        if arm.gripper_position() < 0.03:
            arm.open_gripper(0.035)
        self.refresh()  # planning took a few hundred ms: take in the newest view before checking the cube
        self.status(f"Picking up the {self.top} cube")
        tip = arm.model.fk(arm.q())[0]
        if not (np.linalg.norm(tip[:2] - P["ct"][:2]) < 0.015 and tip[2] < 0.16):  # not already hovering over it
            log = arm.move_joints(P["qt"]["above"], speed=v["approach"], label=f"above {self.top}",
                                  watch=self.watch_top(P["ct"]))
            if log.stopped in COMMANDS:
                raise Stop(log.stopped)
            if log.stopped:
                raise Retarget(log.stopped)
        p = self.where(self.top)
        if p is None or np.linalg.norm(p[:2] - P["ct"][:2]) > 0.005:
            raise Retarget(f"{self.top} cube moved")
        floor = P["table_z"] + 0.012
        log = arm.move_joints(P["qt"]["grasp"], speed=v["descend"], label="down to grasp", gripper_floor=floor,
                              watch=self.watch_descent(P["ct"]))
        if log.stopped in COMMANDS:
            raise Stop(log.stopped)
        p = self.where(self.top)
        if log.stopped or (p is not None and np.linalg.norm(p[:2] - P["ct"][:2]) > 0.006):
            raise Retarget(log.stopped or f"{self.top} cube moved")  # it was moved while we came down
        # (the lift is not interruptible: stopped half way, the held cube would be below the carry height)
        g = arm.close_gripper(15.0, settle=0.6)
        if g < HELD:
            arm.open_gripper(0.035)
            arm.move_joints(P["qt"]["above"], speed=v["rise"], label="back up", gripper_floor=floor)
            return f"missed the grasp (fingers at {g * 1000:.1f} mm)"
        self.holding, self.pick_plan = True, P
        arm.move_joints(P["qt"]["carry"], speed=v["lift"], label="lift", gripper_floor=floor)
        if arm.gripper_position() < HELD:
            self.holding = False
            return "dropped the cube while lifting"
        return self.place(P)

    def place(self, P):
        """Carry the held cube over the bottom one and set it down. If the bottom cube moves, follow it at
        carry height (like the top cube before the pick) and lower once it has rested BOTTOM_STILL s with
        the arm settled over it."""
        arm, v = self.arm, self.v
        carry_floor = P["carry_z"] - 0.02
        self.status(f"Carrying the {self.top} cube to the {self.bottom} cube")
        # the carry is itself 'follow the bottom cube at carry height' (the same follower as for the top cube),
        # so a moving bottom cube is tracked from the first moment, not after a stop
        cf = self.carry_follow
        cf.reset()
        cf.height, cf.floor = P["carry_z"], carry_floor
        last_t = None
        t_wait = time.time()
        while True:
            # follow the bottom cube until it rests, with the arm settled over it
            self.refresh()
            if self.pending in COMMANDS:
                cf.stop()
                raise Stop(self.pending)
            if arm.gripper_position() < HELD:
                if cf.active:
                    self.end_carry_follow()
                self.holding = False
                return "dropped the cube while carrying"
            now = time.time()
            r = self.rest_tracker.resting(self.bottom, now, still_s=BOTTOM_STILL)
            tip = arm.model.fk(arm.q())[0]
            if (r is not None and np.linalg.norm(tip[:2] - r[0][:2]) < 0.006
                    and (not cf.active or cf.settled())):
                if cf.active:
                    self.end_carry_follow()
                try:
                    self.replan_place(P, r[0], r[1], arm.q())
                except (ValueError, SafetyStop) as e:
                    return self.put_back(f"cannot reach the {self.bottom} cube ({e})")
                if np.abs(arm.q() - P["qb"]["carry"]).max() > 0.01:
                    arm.move_joints(P["qb"]["carry"], speed=v["carry"], label=f"over {self.bottom}",
                                    gripper_floor=carry_floor)
                log = arm.move_joints(P["qb"]["place"], speed=v["lower"], label=f"onto {self.bottom}",
                                      watch=self.watch_bottom(P["cb"], 0.004))
                if not log.stopped:
                    break
                if log.stopped == "dropped":
                    self.holding = False
                    return "dropped the cube while lowering"
                arm.move_joints(P["qb"]["carry"], speed=v["rise"], label="rise", gripper_floor=carry_floor)
                if log.stopped in COMMANDS:
                    raise Stop(log.stopped)
                self.event("follow_bottom", why=log.stopped)
                t_wait = time.time()
                continue
            state, pb, Rb, _ = self.cubes.get(self.bottom, ("not seen", None, None, 0.0))
            if state in ("TRACKING", "OCCLUDED") and pb is not None:
                # the held cube hides part of it, so its pose jitters more: aim at a 0.15 s median, once per frame
                pm = self.where(self.bottom, window=0.15)
                res = cf.aim(pm if pm is not None else pb, Rb) if self.t_stats != last_t else "ok"
                last_t = self.t_stats
            else:
                res = "unseen"
            if res in ("ok", "edge") and cf.q_tgt is not None:
                t_wait = now
                if not cf.active:
                    cf.start()
                cf.step()
                self.status(f"Carrying the {self.top} cube to the {self.bottom} cube, following it")
            else:
                if cf.active:
                    cf.step()  # guards only: no new target
                self.status(f"Holding the {self.top} cube: waiting for the {self.bottom} cube ({res})")
                if now - t_wait > 30.0:
                    if cf.active:
                        self.end_carry_follow()
                    return self.put_back(f"the {self.bottom} cube did not come back within reach")
            time.sleep(0.01)
        arm.open_gripper(0.035)
        self.holding = False
        arm.move_joints(P["qb"]["carry"], speed=v["rise"], label="rise")
        msg = "not checked"
        for _ in range(8):  # the stack can take a moment to settle in the tracker
            time.sleep(0.2)
            self.refresh()
            ok, msg, m = self.stack_check()
            if ok:
                self.event("placed", **m)
                return None
        return f"placement check failed: {msg}"

    def end_carry_follow(self):
        """Stop following the bottom cube and let the last smooth move finish."""
        self.carry_follow.stop()
        t0 = time.time()
        while time.time() - t0 < 1.5 and not self.carry_follow.settled():
            self.arm.follow_check()
            time.sleep(0.01)

    def put_back(self, why):
        """Set the held cube down where it was picked (if nothing is there now), then report why."""
        P, arm = self.pick_plan, self.arm
        p = self.where(self.bottom)
        if P is None or (p is not None and np.linalg.norm(p[:2] - P["ct"][:2]) < 0.07):
            raise Stop(f"holding the {self.top} cube, {why}; its pick-up spot is taken")
        self.status(f"{why}: putting the {self.top} cube back down")
        floor = P["table_z"] + 0.012
        arm.move_joints(P["qt"]["carry"], speed=self.v["carry"], label="back over the pick-up spot",
                        gripper_floor=P["carry_z"] - 0.02)
        arm.move_joints(P["qt"]["grasp"], speed=self.v["lower"], label="put it down", gripper_floor=floor)
        arm.open_gripper(0.035)
        self.holding = False
        arm.move_joints(P["qt"]["above"], speed=self.v["rise"], label="back up", gripper_floor=floor)
        return f"{why}; put the {self.top} cube back"

    # ----- decisions -----
    def stack_check(self, strict=True):
        pt, pb, Rt = self.where(self.top), self.where(self.bottom), self.rotation(self.top)
        if pt is None or pb is None or Rt is None:
            return False, f"cannot see both cubes ({self.cubes.get(self.top, ['?'])[0]}, {self.cubes.get(self.bottom, ['?'])[0]})", {}
        d = (pt - pb) * 1000
        tilt = top_face_tilt_deg(Rt)
        m = {"dz_mm": round(float(d[2]), 1), "dxy_mm": round(float(np.hypot(d[0], d[1])), 1),
             "dx_mm": round(float(d[0]), 1), "dy_mm": round(float(d[1]), 1), "tilt_deg": round(tilt, 1)}
        # strict: the check right after placing. Loose: 'already stacked' while watching, since stacked
        # cubes currently read 4-5 mm short (docs/reliability/run_20261016_174531.md)
        lo, hi, dxy, tl = (42, 58, 12, 6) if strict else (38, 62, 15, 8)
        ok = lo < m["dz_mm"] < hi and m["dxy_mm"] < dxy and tilt < tl
        return ok, f"{self.top} is {m['dz_mm']:.0f} mm above {self.bottom}, {m['dxy_mm']:.1f} mm off centre", m

    def ready(self):
        """(ct, Rt, cb, Rb) when both cubes rest on the table within reach, else (None, why)."""
        now = time.time()
        out = []
        for name in (self.top, self.bottom):
            state, p, _, vis = self.cubes.get(name, ("not seen", None, None, 0.0))
            r = self.rest_tracker.resting(name, now)
            if r is None:
                return None, (f"waiting for the {name} cube ({state})" if state != "TRACKING" else
                              f"waiting for the {name} cube to rest")
            c, R, _ = r
            other = self.where(self.bottom if name == self.top else self.top)
            if other is not None and np.linalg.norm(other[:2] - c[:2]) < 0.035 and c[2] > other[2] + 0.03:
                return None, f"the {name} cube is on top of the other one: unstack them"
            if not TABLE_Z[0] < c[2] < TABLE_Z[1]:
                return None, f"the {name} cube is not on the table (z {c[2] * 1000:.0f} mm)"
            rad = np.hypot(c[0], c[1])
            if not (REACH[0] < rad < REACH[1] and c[0] > 0.1):
                return None, (f"the {name} cube is out of reach ({rad * 100:.0f} cm from the arm base; "
                              f"{REACH[0] * 100:.0f}-{REACH[1] * 100:.0f} cm works)")
            out += [c, R]
        return tuple(out), None

    # ----- arm modes -----
    def wake(self):
        if np.abs(self.arm.q() - REST).max() > 0.15 and np.abs(self.arm.q() - WATCH).max() > 0.15:
            raise RuntimeError("arm is neither at rest nor at the watch pose; not starting")
        self.status("Waking up: moving to the watch pose")
        self.arm.open_gripper(0.035)
        self.arm.move_joints(WATCH, speed=self.v["watch"], label="to watch pose")
        self.mode = "awake"
        self.event("awake")

    def go_rest(self):
        self.end_follow()
        if self.holding:
            self.put_back("going to rest")
        self.status("Going to rest")
        self.go_watch_safe(watch=False)
        self.arm.rest_and_idle(speed=0.6)
        self.arm.gripper_idle()
        self.mode = "rest"
        self.event("rest")
        self.status("At rest (brakes on). Press Start to wake the arm.")

    def go_watch_safe(self, watch=True, to_watch=True):
        """To WATCH from wherever the arm stopped: straight up first if the fingertips are low (near the
        cubes); with to_watch=False, only that. Stops early for an operator command (handled by the main
        loop) unless watch=False."""
        w = self.watch_commands if watch else None
        q = self.arm.q()
        tip = self.arm.model.fk(q)[0]
        if tip[2] < 0.10:
            try:
                qs, _ = plan_top_down(np.array([tip[0], tip[1], 0.0]), np.eye(3), {"up": 0.13}, ik=self.ik, seed=q)
                if self.arm.move_joints(qs["up"], speed=self.v["rise"], label="rise", watch=w).stopped:
                    return
            except ValueError:
                pass
        if to_watch and np.abs(self.arm.q() - WATCH).max() > 0.02:
            self.arm.move_joints(WATCH, speed=self.v["watch"], label="to watch pose", watch=w)

    def end_follow(self):
        """Stop following and let the last smooth move finish (so the next move starts from rest)."""
        if not self.follow.active:
            return
        self.follow.stop()
        t0 = time.time()
        while time.time() - t0 < 1.5 and not self.follow.settled():
            self.arm.follow_check()
            time.sleep(0.01)

    def idle(self, msg):
        """Wait at the watch pose (cubes in full view)."""
        self.end_follow()
        self.lost_since = None
        if np.abs(self.arm.q() - WATCH).max() > 0.02:
            self.go_watch_safe()
        self.status(msg)

    def hover_or_watch(self, why):
        """Not ready to pick: hover over the top cube and follow it while it is on the table and in view;
        otherwise wait at the watch pose."""
        state, p, R, _ = self.cubes.get(self.top, ("not seen", None, None, 0.0))
        if state in ("TRACKING", "OCCLUDED") and p is not None:
            self.lost_since = None
            res = self.follow.aim(p, R)
            if res in ("ok", "edge"):
                if not self.follow.active:
                    self.follow.start()
                self.follow.step()
                self.status(f"Following the {self.top} cube ({why})" if res == "ok" else
                            f"Following: the {self.top} cube is outside the reachable ring, waiting at the edge ({why})")
                return
            if res == "lifted" and self.follow.active:
                self.follow.step()  # guards only: no new target
                self.status(f"The {self.top} cube is lifted off the table: waiting where I am")
                return
        if self.follow.active:  # out of sight (a hand over it?): wait where we are for a moment
            self.lost_since = self.lost_since or time.time()
            if time.time() - self.lost_since < 2.0:
                self.follow.step()
                self.status(f"Waiting: the {self.top} cube is {state}")
                return
        self.idle(f"Watching: {why}")

    # ----- main loop -----
    def run(self):
        t0 = time.time()
        while self.poller.latest is None and time.time() - t0 < 3:
            time.sleep(0.05)
        self.refresh()
        if self.stale:
            raise RuntimeError("live perception (scripts.web_view) is not answering on port 8080")
        if np.abs(self.arm.q() - REST).max() > 0.15:
            raise RuntimeError("arm is not at the rest pose; not starting")
        self.status("Controller ready. Press Start: the arm follows the red cube and stacks it once both cubes rest.")
        while True:
            try:
                self.tick()
            except Stop as e:
                why = str(e)
                self.follow.stop()
                self.carry_follow.stop()
                if why in COMMANDS:
                    self.pending = why  # handled at the top of the loop
                else:
                    self.arm.hold()
                    self.mode = "paused"
                    self.status(f"Paused: {why}. Press Resume to continue.")
                    self.event("paused", why=why)
            except SafetyStop as e:
                self.follow.stop()
                self.carry_follow.stop()
                self.mode = "paused"
                self.holding = is_holding(self.arm.gripper_position())
                self.event("safety_stop", why=str(e))
                self.status(f"SAFETY STOP: {e}. Holding position. Press Resume to continue or Rest to finish.")
            except Exception as e:  # a bug must not leave the arm unattended: hold, pause, report
                import traceback

                traceback.print_exc()
                try:
                    if self.arm.position_mode:
                        self.arm.hold()
                    self.holding = is_holding(self.arm.gripper_position())
                except Exception:
                    traceback.print_exc()
                self.follow.stop()
                self.carry_follow.stop()
                self.mode = "paused" if self.mode != "rest" else "rest"
                self.event("error", why=f"{type(e).__name__}: {e}")
                self.status(f"Controller error ({type(e).__name__}: {e}). Holding position; press Resume or Rest.")
            time.sleep(0.01 if self.follow.active else 0.05)  # guards at 100 Hz while following

    def tick(self):
        self.refresh()
        cmd, self.pending = self.pending, None
        if cmd == "rest" and self.mode != "rest":
            self.go_rest()
            return
        if self.mode == "rest":
            if cmd == "start":
                self.wake()
            return
        if cmd == "pause":
            self.follow.stop()
            self.arm.hold()
            self.mode = "paused"
            self.event("paused")
        if self.mode == "paused":
            if cmd not in ("resume", "start"):
                self.status("Paused (holding position). Press Resume to continue or Rest to finish.")
                return
            self.mode = "awake"
            self.event("resumed")
        if self.stale:
            self.status("Waiting for the live camera feed")
            return
        self.step()

    def step(self):
        """One decision from the current view."""
        if self.holding:  # e.g. resumed after a pause while carrying
            self.end_follow()
            r = self.rest_tracker.resting(self.bottom, time.time())
            if r is None:
                self.status(f"Holding the {self.top} cube: waiting for the {self.bottom} cube to rest")
                return
            P = self.pick_plan
            try:
                self.replan_place(P, r[0], r[1], self.arm.q())
            except (ValueError, SafetyStop) as e:
                self.status(f"Holding the {self.top} cube: cannot reach the {self.bottom} cube ({e})")
                return
            self.finish(self.place(P), t0=time.time())
            return
        ok, msg, m = self.stack_check(strict=False)
        if ok:
            self.idle(f"Stacked ({msg}). Watching.")
            return
        cubes, why = self.ready()
        if cubes is not None:
            ct, Rt, cb, Rb = cubes
            key = np.concatenate([ct, cb])
            if self.blocked is not None and np.abs(self.blocked[0] - key).max() < 0.005:
                why = f"cannot do this placement: {self.blocked[1]}. Move a cube"
            elif not self.follow.active or self.follow.settled():
                self.end_follow()
                t0 = time.time()
                self.event("attempt", top=ct, bottom=cb)
                try:
                    why = self.attempt(ct, Rt, cb, Rb)
                except ValueError as e:  # planning: unreachable, or the fingers would not clear the other cube
                    self.blocked = (key, str(e))
                    self.event("cannot_plan", why=str(e))
                    return
                except Retarget as e:  # it moved before the grasp: follow it again (not a failure)
                    self.event("retarget", why=str(e))
                    self.go_watch_safe(to_watch=False)
                    return
                self.finish(why, t0)
                return
            else:
                why = "settling over it"
        self.hover_or_watch(why)

    def finish(self, why, t0):
        self.blocked = None
        self.fails = 0 if why is None else self.fails + 1
        self.event("attempt_end", ok=why is None, why=why, seconds=round(time.time() - t0, 1))
        if not self.holding:  # stacked: watch from the watch pose; failed: rise and go after the cube again
            self.go_watch_safe(to_watch=why is None)
        if why is None:
            self.status(f"Stacked in {time.time() - t0:.0f} s. Watching.")
        elif self.fails >= self.max_fails:
            self.fails = 0
            raise Stop(f"{self.max_fails} attempts in a row failed (last: {why})")
        else:
            self.status(f"Attempt failed: {why}. Trying again when the cubes rest.")


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ip", default="192.168.1.3")
    ap.add_argument("--top", default="red")
    ap.add_argument("--bottom", default="green")
    ap.add_argument("--still", type=float, default=1.0, help="s a cube must rest before the arm goes for it")
    ap.add_argument("--dry", action="store_true", help="no arm: plan and simulate the motion only")
    ap.add_argument("--ignore-serial", action="store_true",
                    help="use assets/calibration.yaml although the live camera is not the one it was fitted with")
    args = ap.parse_args()

    def on_term(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)

    log_dir = ROOT / "runs" / "controller"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}{'_dry' if args.dry else ''}.jsonl"
    cal = CameraToBase(ignore_serial=args.ignore_serial)
    try:
        check_live_camera(cal)  # before connecting to the arm; every snapshot is checked again (cubes_in_base)
    except CameraMismatch as e:
        raise SystemExit(str(e))
    arm = DryArm() if args.dry else SafeArm(args.ip)
    ctl = StackController(arm, cal, args.top, args.bottom, still_s=args.still, log_path=log_path)
    print(f"log: {log_path}", flush=True)
    try:
        ctl.run()
    except KeyboardInterrupt:
        pass
    finally:
        shutdown(ctl)


def shutdown(ctl):
    """Leave the arm safe when the program ends for any reason: back to rest with the brakes on (putting
    down a held cube first); if that is not possible, hold position and say so. Never leave it awake and
    unattended."""
    arm = ctl.arm
    try:
        if arm.position_mode:
            ctl.status("Controller stopping: taking the arm to rest")
            ctl.pending = None
            ctl.go_rest()
        ctl.status("Controller stopped (arm at rest, brakes on)")
    except Exception as e:  # last resort: keep holding where it is
        try:
            arm.hold()
        except Exception:
            pass
        ctl.status(f"Controller stopped while the arm is awake ({type(e).__name__}: {e}): it is holding position. "
                   f"Power it off with the switch once safe.")
        return
    arm.gripper_idle()
    arm.close()


if __name__ == "__main__":
    main()
