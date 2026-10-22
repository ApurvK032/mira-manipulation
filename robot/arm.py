"""Safety wrapper around the Trossen WidowX AI driver. Every arm command goes through here.

- Joint targets must be inside the driver's joint limits minus a margin.
- The whole planned path is checked with Trossen's MuJoCo model: the gripper and wrist stay inside a
  workspace box above the table. (The model matches the real arm's gripper position to 0.4 mm.)
- Move time comes from a maximum joint speed, so moves are never faster than allowed.
- Moves run non-blocking and are watched at about 100 Hz. If the arm falls behind the planned
  trajectory (blocked) or feels unexpected external effort (collision), it holds where it is.
- hold() keeps the motors in position mode. The arm is only set to idle (braked) at the rest pose.
  The power switch is the hard stop.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import trossen_arm as ta

from .reaim import evaluate, plan_ff, quintic
from .stream import Streamer

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "assets" / "trossen_arm_mujoco" / "wxai" / "wxai_follower.xml"

REST = np.zeros(6)  # folded on its own supports: where the arm sits when idle
HOME = np.array([0.0, np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0])  # Trossen's demo home pose
# Awake pose between tasks: gripper 25 cm out, 25 cm up, tilted 60 deg down. Hides none of the cube
# workspace (18-36 cm from the base, up to a stack) from the D455, and reaches above any cube in about 2 s.
WATCH = np.array([0.0, 1.12, 1.41, -1.34, 0.0, 0.0])


class SafetyStop(RuntimeError):
    pass


@dataclass
class Limits:
    max_joint_speed: float = 1.0  # rad/s peak along a move (driver allows 6.3; home takes about 3 s)
    min_move_time: float = 1.5  # s
    joint_margin: float = 0.05  # rad kept inside the driver's joint limits
    # workspace box in the arm base frame (m): x forward, y left, z up from the mounting surface
    box_min: tuple = (-0.05, -0.45, 0.04)
    box_max: tuple = (0.62, 0.45, 0.55)
    # rad between planned and measured joint angles before stopping: 0.08 at rest speed, growing with
    # speed because faster moves lag a little more (1.2 deg observed at 0.3 rad/s)
    tracking_tol: float = 0.08
    tracking_tol_per_speed: float = 0.06  # extra rad per rad/s of peak speed
    gripper_force: float = 15.0  # N used to close on an object (driver max 100 N)
    gripper_open: float = 0.035  # m per finger (gripper range 0-0.040 m)
    # Nm of unexpected external effort per joint; None = record only. The driver's external effort
    # includes unmodeled gravity (about 6 Nm on the elbow when lifting off the rest pose), so limits
    # are set from logged moves rather than guessed.
    # From the first rest <-> home moves on 2026-10-16 (1.5 x max observed + 2 Nm). This only covers that
    # path; other poses may need a gravity-residual model before the limits can be tightened.
    effort_limit: tuple | None = (2.3, 6.3, 16.6, 6.3, 2.4, 2.2)


def minjerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class ArmModel:
    """Forward kinematics from Trossen's MuJoCo model."""

    def __init__(self, path=MODEL):
        self.m = mujoco.MjModel.from_xml_path(str(path))
        self.d = mujoco.MjData(self.m)
        self.qadr = [self.m.joint(f"joint_{i}").qposadr[0] for i in range(6)]
        self.points = ["link_4", "link_5", "link_6"]  # wrist bodies, checked together with the gripper tip

    def fk(self, q):
        self.d.qpos[:] = 0
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        ee = self.d.site("ee_site").xpos.copy()
        pts = [self.d.body(b).xpos.copy() for b in self.points]
        return ee, np.array([ee, *pts])


@dataclass
class MoveLog:
    t: list = field(default_factory=list)
    q: list = field(default_factory=list)
    q_plan: list = field(default_factory=list)
    effort: list = field(default_factory=list)
    stopped: str | None = None  # why the move was stopped early by its watch callback, if it was


class SafeArm:
    def __init__(self, ip="192.168.1.3", limits: Limits | None = None, clear_error=True):
        self.limits = limits or Limits()
        self.model = ArmModel()
        self.drv = ta.TrossenArmDriver()
        self.drv.configure(ta.Model.wxai_v0, ta.StandardEndEffector.wxai_v0_follower, ip, clear_error)
        jl = self.drv.get_joint_limits()
        self.q_min_raw = np.array([j.position_min for j in jl[:6]])
        self.q_max_raw = np.array([j.position_max for j in jl[:6]])
        self.q_min = self.q_min_raw + self.limits.joint_margin
        self.q_max = self.q_max_raw - self.limits.joint_margin
        self.v_max = np.array([j.velocity_max for j in jl[:6]])
        self.position_mode = False
        self.logs: list[MoveLog] = []

    # ----- state -----
    def q(self) -> np.ndarray:
        return np.array(self.drv.get_arm_positions(), dtype=float)

    def ee(self) -> np.ndarray:
        return np.array(list(self.drv.get_cartesian_positions()), dtype=float)

    def efforts(self) -> np.ndarray:
        return np.array(self.drv.get_arm_external_efforts(), dtype=float)

    def velocities(self) -> np.ndarray:
        return np.array(self.drv.get_arm_velocities(), dtype=float)

    def modes(self):
        return [str(m).split(".")[-1] for m in self.drv.get_modes()]

    # ----- checks -----
    def check_target(self, q_goal: np.ndarray, q_start: np.ndarray, n=41, gripper_floor=None):
        """Raise SafetyStop unless the joint path from q_start to q_goal stays inside every limit.

        gripper_floor (m): the lowest the fingertips may go on this move, replacing the box floor for
        them only. Lower for grasp moves near the table (measured table height plus a clearance, never
        below the table); higher while carrying a cube, which hangs below the fingertips.
        """
        lim = self.limits
        lo = np.minimum(self.q_min, q_start - 1e-3)  # the start pose itself may sit on a limit (rest pose)
        hi = np.maximum(self.q_max, q_start + 1e-3)
        if np.allclose(q_goal, REST, atol=1e-6):
            # the rest pose sits on the joint limits by design: check it against the driver's own limits
            lo, hi = np.minimum(lo, self.q_min_raw - 1e-3), np.maximum(hi, self.q_max_raw + 1e-3)
        if np.any(q_goal < lo) or np.any(q_goal > hi):
            raise SafetyStop(f"joint target outside limits: {np.round(q_goal, 3)} not in "
                             f"[{np.round(self.q_min, 2)}, {np.round(self.q_max, 2)}]")
        self._check_path([q_start + (q_goal - q_start) * minjerk(s) for s in np.linspace(0, 1, n)], q_start,
                         gripper_floor)

    def _check_path(self, qs, q_start, gripper_floor=None):
        """Raise SafetyStop unless every joint configuration in qs (a path from q_start) keeps the gripper and the
        links inside the workspace box (gripper_floor as in check_target)."""
        lim = self.limits
        bmin, bmax = np.array(lim.box_min), np.array(lim.box_max)
        # the start pose may sit below the box floor (the rest pose is low); the path may not go lower
        floor = min(bmin[2], self.model.fk(q_start)[1][:, 2].min() - 1e-3)
        for i, q in enumerate(qs):
            _, pts = self.model.fk(q)
            for p, name in zip(pts, ["gripper", *self.model.points]):
                f = gripper_floor if (name == "gripper" and gripper_floor is not None) else floor
                if p[0] < bmin[0] or p[1] < bmin[1] or p[2] < f or np.any(p > bmax):
                    raise SafetyStop(f"{name} would leave the workspace at {np.round(p, 3)} "
                                     f"({100 * i / max(len(qs) - 1, 1):.0f}% along the move)")

    def _check_ff(self, q0, v0, q_goal, v_end, T, gripper_floor):
        """Checks for a feedforward re-aim: the goal and the straight path (check_target), the planned quintic
        itself (it may bulge past the straight path), and where the arm would coast to if the next re-aim came
        0.15 s late. Returns the joint-space bounds of all that, for follow_check."""
        self.check_target(q_goal, q0, n=21, gripper_floor=gripper_floor)
        qs = list(evaluate(quintic(q0, v0, q_goal, v_end, T), np.linspace(0, T, 21))[0])
        coast = q_goal + v_end * 0.15
        lo, hi = np.minimum(self.q_min, q0 - 1e-3), np.maximum(self.q_max, q0 + 1e-3)
        for q in qs + [coast]:
            if np.any(q < lo) or np.any(q > hi):
                raise SafetyStop(f"feedforward re-aim would pass a joint limit: {np.round(q, 3)}")
        self._check_path(qs + [coast], q0, gripper_floor)
        pts = np.array(qs + [coast, q0, q_goal])
        return pts.min(axis=0), pts.max(axis=0)

    # ----- motion -----
    def _ensure_position_mode(self):
        if not self.position_mode:
            self.drv.set_arm_modes(ta.Mode.position)  # targets start at the current position
            self.position_mode = True
            time.sleep(0.5)  # let the motors take the load before effort readings are used as a baseline

    def hold(self):
        """Stop where the arm is and keep holding (motors stay on)."""
        self.drv.set_arm_positions(self.q(), 0.1, False)

    def move_joints(self, q_goal, speed=None, label="move", gripper_floor=None, watch=None,
                    watch_every=0.05) -> MoveLog:
        """Move along a checked minimum-jerk joint path. watch(t) is called about every watch_every s;
        if it returns a reason, the arm slows to a stop on the planned path (about 0.4 s) and holds there,
        and the returned log has .stopped set to that reason."""
        q_goal = np.asarray(q_goal, dtype=float)
        q0 = self.q()
        self.check_target(q_goal, q0, gripper_floor=gripper_floor)
        speed = min(speed or self.limits.max_joint_speed, self.limits.max_joint_speed)
        dq = np.abs(q_goal - q0)
        # minimum jerk peaks at 1.875x the average speed: size the time so the peak stays under `speed`
        T = max(self.limits.min_move_time, 1.875 * dq.max() / speed, *(1.875 * dq / (0.5 * self.v_max)))
        self._ensure_position_mode()
        base_effort = self.efforts()
        log = MoveLog()
        t0 = time.time()
        self.drv.set_arm_positions(q_goal, T, False)
        next_watch = watch_every
        try:
            while True:
                t = time.time() - t0
                if watch is not None and t >= next_watch and t < T:
                    next_watch = t + watch_every
                    why = watch(t)
                    if why:
                        self._stop_on_path(q0, q_goal, T, t)
                        log.stopped = why
                        break
                q = self.q()
                q_plan = q0 + (q_goal - q0) * minjerk(t / T)
                e = self.efforts() - base_effort
                log.t.append(t), log.q.append(q), log.q_plan.append(q_plan), log.effort.append(e)
                tol = self.limits.tracking_tol + self.limits.tracking_tol_per_speed * speed
                if t > 0.3 and np.max(np.abs(q - q_plan)) > tol:
                    raise SafetyStop(f"{label}: arm is not following the plan (off by "
                                     f"{np.degrees(np.max(np.abs(q - q_plan))):.0f} deg): blocked?")
                if self.limits.effort_limit is not None and np.any(np.abs(e) > np.array(self.limits.effort_limit)):
                    j = int(np.argmax(np.abs(e) / np.array(self.limits.effort_limit)))
                    raise SafetyStop(f"{label}: unexpected effort {e[j]:.1f} Nm on joint {j}: collision?")
                if t > T + 0.5:
                    break
                time.sleep(0.01)
        except BaseException:
            self.hold()
            raise
        finally:
            self.logs.append(log)
        return log

    # ----- following a moving target -----
    # Driver 1.8.x interpolates commands shorter than 0.2 s linearly, and fast re-commanding is unstable
    # before driver 1.9.3, so a moving target is followed with the same smooth (quintic, >= 0.3 s) moves
    # as move_joints, re-aimed a few times a second; the guards run between re-aims (follow_check).
    def follow_begin(self):
        self._ensure_position_mode()
        q = self.q()
        self._fb = dict(start=q, goal=q, effort=self.efforts(), speed=self.limits.max_joint_speed)

    def follow_goal(self, q_goal, speed, gripper_floor, min_time=0.3, accel=3.0, goal_velocity=None) -> float:
        """Re-aim: one smooth move from where the arm is to q_goal (path checked like move_joints).
        Returns its duration. Call follow_check() often in between. The duration also keeps the peak joint
        acceleration under `accel` rad/s^2 (a quintic peaks at 5.77 dq / T^2): short, sharp re-aims swing
        the whole arm on the base joint and read as unexpected effort there.

        goal_velocity (opt-in, rad/s per joint, from robot.reaim.GoalVelocity): a feedforward re-aim. The move aims
        at where the moving goal will be when it ends and arrives with the goal's velocity (robot.reaim.plan_ff,
        through the driver's feedforward velocities), so the arm does not brake at every re-aim. Without it,
        behaviour is exactly as before. Near a joint limit or the workspace edge (where the feedforward move or
        its coast would leave them), it falls back to a plain re-aim."""
        q_goal = np.asarray(q_goal, dtype=float)
        q0 = self.q()
        speed = min(speed, self.limits.max_joint_speed)
        if goal_velocity is not None:
            v0 = self.velocities()
            g, v_end, T = plan_ff(q0, v0, q_goal, np.asarray(goal_velocity, dtype=float), speed, accel, min_time,
                                  self.v_max, self.q_min, self.q_max)
            if np.any(v_end):
                try:
                    lo, hi = self._check_ff(q0, v0, g, v_end, T, gripper_floor)
                    self.drv.set_arm_positions(g, T, False, v_end, np.zeros(6))
                    self._fb.update(start=q0, goal=g, speed=speed, lo=lo, hi=hi)
                    return T
                except SafetyStop:
                    pass  # near a limit or the floor: a plain re-aim instead, with the usual checks (below)
        self._fb.pop("lo", None)
        self._fb.pop("hi", None)
        self.check_target(q_goal, q0, n=21, gripper_floor=gripper_floor)
        dq = np.abs(q_goal - q0)
        T = max(min_time, 1.875 * dq.max() / speed, np.sqrt(5.77 * dq.max() / accel),
                *(1.875 * dq / (0.5 * self.v_max)))
        self.drv.set_arm_positions(q_goal, T, False)
        self._fb.update(start=q0, goal=q_goal, speed=speed)
        return T

    def follow_check(self):
        """Guards while following (call at about 100 Hz). The arm must stay between the last re-aim's
        start and goal (plus the move tolerance), and must not feel unexpected effort compared with a
        reading taken while it was still. Raises SafetyStop and holds."""
        lim, fb = self.limits, self._fb
        try:
            q = self.q()
            tol = lim.tracking_tol + lim.tracking_tol_per_speed * fb["speed"]
            lo = fb["lo"] if "lo" in fb else np.minimum(fb["start"], fb["goal"])  # lo/hi: a feedforward re-aim
            hi = fb["hi"] if "hi" in fb else np.maximum(fb["start"], fb["goal"])
            lo, hi = lo - tol, hi + tol
            if np.any(q < lo) or np.any(q > hi):
                raise SafetyStop("following: arm is off its path: blocked?")
            if lim.effort_limit is not None:
                e = self.efforts()
                if np.abs(np.array(self.drv.get_arm_velocities())).max() < 0.03:
                    fb["effort"] = fb["effort"] + 0.1 * (e - fb["effort"])  # still: refresh the baseline
                d = e - fb["effort"]
                self.follow_peak = np.maximum(getattr(self, "follow_peak", np.zeros(6)), np.abs(d))
                if np.any(np.abs(d) > np.array(lim.effort_limit)):
                    j = int(np.argmax(np.abs(d) / np.array(lim.effort_limit)))
                    raise SafetyStop(f"following: unexpected effort {d[j]:.1f} Nm on joint {j}: collision?")
        except BaseException:
            self.hold()
            raise

    # ----- streaming targets (driver 1.9.3+) -----
    # Driver 1.9.3 joins goals re-sent every few milliseconds into continuous motion, so a moving target can be
    # streamed instead of re-aimed: robot.stream.Streamer turns each goal into smooth, speed- and acceleration-limited
    # points, one per tick, each sent with its velocity. Needs driver and firmware 1.9.x; on 1.8.3 use follow_goal.
    def stream_begin(self, rate=100.0, speed=None, accel=4.0, tau=0.06, window=0.3):
        """Start streaming from where the arm is. Call stream_tick every 1/rate s and follow_check in between
        (its tracking window covers the points commanded in the last `window` s)."""
        self._ensure_position_mode()
        q = self.q()
        self._st = Streamer(rate, min(speed or self.limits.max_joint_speed, self.limits.max_joint_speed), accel, tau)
        self._st.reset(q)
        self._st_goal, self._st_recent = q.copy(), [q.copy()] * max(2, int(round(window * rate)))
        self.stream_rejects = 0
        self._fb = dict(start=q, goal=q, effort=self.efforts(), speed=self._st.speed, lo=q.copy(), hi=q.copy())

    def stream_tick(self, q_goal, gripper_floor):
        """One tick toward q_goal. A new goal is checked first like any move (joint limits, workspace box, fingertip
        floor, on the path from the last commanded point), and so is every step; a goal that fails is not followed
        (the last accepted goal is kept) until goals have failed for 0.5 s, which raises SafetyStop. Returns the
        commanded point."""
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
        q_next, v_next = self._st.step(self._st_goal)
        # the generator can overshoot its goal by a few mrad: keep each point inside the joint limits, stopping there
        if np.any(q_next < self.q_min) or np.any(q_next > self.q_max):
            at = (q_next < self.q_min) | (q_next > self.q_max)
            q_next = np.clip(q_next, self.q_min, self.q_max)
            self._st.q, self._st.v = q_next.copy(), np.where(at, 0.0, self._st.v)
            v_next = self._st.v.copy()
        self.check_target(q_next, last, n=2, gripper_floor=gripper_floor)
        self.drv.set_arm_positions(q_next, self._st.dt, False, v_next, np.zeros(6))
        self._st_recent = self._st_recent[1:] + [q_next]
        rec = np.array(self._st_recent)
        self._fb.update(start=last, goal=q_next, lo=rec.min(axis=0), hi=rec.max(axis=0))
        return q_next

    def _stop_on_path(self, q0, q_goal, T, t, lead=0.15, duration=0.4):
        """Slow down and stop at the planned position `lead` s ahead (still on the checked path)."""
        q_stop = q0 + (q_goal - q0) * minjerk((t + lead) / T)
        self.drv.set_arm_positions(q_stop, duration, False)
        time.sleep(duration + 0.1)

    # ----- gripper -----
    def gripper_position(self) -> float:
        return float(self.drv.get_gripper_position())

    def open_gripper(self, position=None, duration=0.8, blocking=True) -> float:
        """Open to a finger position (m) under position control; returns the reached position (with
        blocking=False, returns at once with the position the fingers are at now)."""
        position = self.limits.gripper_open if position is None else position
        position = float(np.clip(position, 0.0, 0.040))
        self.drv.set_gripper_mode(ta.Mode.position)
        self.drv.set_gripper_position(position, duration, blocking)
        return self.gripper_position()

    def close_gripper(self, force=None, settle=0.8, mode="external_effort") -> float:
        """Close with a limited force (N) so it stops on whatever is between the fingers; returns the
        finger position: about half the object's width, or near 0 if it closed on nothing.
        mode: "external_effort" (default, as always) or "effort" (total effort: the steadiest hold of a cube picked
        off the table on firmware 1.9.4, 18 N with no force spike; scripts/grip_lift_test.py, 2026-10-20)."""
        force = min(self.limits.gripper_force if force is None else force, 40.0)
        if mode == "effort":
            self.drv.set_gripper_mode(ta.Mode.effort)
            self.drv.set_gripper_effort(-force, 0.0, False)
        else:
            self.drv.set_gripper_mode(ta.Mode.external_effort)
            self.drv.set_gripper_external_effort(-force, 0.0, False)
        time.sleep(settle)
        return self.gripper_position()

    def squeeze_gripper(self, force=20.0, start=None, timeout=6.0) -> float:
        """Close on an object in position mode and squeeze it with about `force` N, then hold that position target.
        start (m per finger): close slowly to this target first (just inside the object's width, e.g. 25.5 mm for a
        5 cm cube, which reads about 27.5 mm when held). Then, only while the fingers are stopped against the object,
        step the target in until the gripper's effort reaches `force`. Returns the finger position (m); near 0 if it
        closed on nothing. On firmware 1.9.4 the external-effort close (close_gripper) was seen to chatter against a
        cube and let it slip; position mode held it steadily (scripts/gripper_hold_test.py)."""
        force = min(force, 40.0)
        self.drv.set_gripper_mode(ta.Mode.position)
        goal = self.gripper_position() if start is None else float(start)
        self.drv.set_gripper_position(goal, 0.6, False)
        time.sleep(0.8)
        t0 = time.time()
        while time.time() - t0 < timeout and goal > 0.0:
            pos, still = self.gripper_position(), abs(self.drv.get_gripper_velocity()) < 0.003
            if not still:
                time.sleep(0.05)
                continue
            blocked = pos - goal > 0.0005  # stopped short of the target: the object holds the fingers apart
            if blocked and abs(self.drv.get_gripper_effort()) >= force:
                break
            goal = max(0.0, goal - (0.0003 if blocked else 0.002))
            self.drv.set_gripper_position(goal, 0.1, False)
            time.sleep(0.15)
        time.sleep(0.2)
        return self.gripper_position()

    def gripper_idle(self):
        self.drv.set_gripper_mode(ta.Mode.idle)

    def rest_and_idle(self, speed=None):
        """Go back to the rest pose slowly, then brake the joints (the arm sits on its own supports)."""
        self.move_joints(REST, speed=speed, label="to rest")
        self.drv.set_arm_modes(ta.Mode.idle)
        self.position_mode = False

    def close(self):
        self.drv.cleanup()
