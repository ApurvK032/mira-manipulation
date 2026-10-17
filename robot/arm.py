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
        bmin, bmax = np.array(lim.box_min), np.array(lim.box_max)
        # the start pose may sit below the box floor (the rest pose is low); the path may not go lower
        floor = min(bmin[2], self.model.fk(q_start)[1][:, 2].min() - 1e-3)
        for s in np.linspace(0, 1, n):
            q = q_start + (q_goal - q_start) * minjerk(s)
            _, pts = self.model.fk(q)
            for p, name in zip(pts, ["gripper", *self.model.points]):
                f = gripper_floor if (name == "gripper" and gripper_floor is not None) else floor
                if p[0] < bmin[0] or p[1] < bmin[1] or p[2] < f or np.any(p > bmax):
                    raise SafetyStop(f"{name} would leave the workspace at {np.round(p, 3)} "
                                     f"({100 * s:.0f}% along the move)")

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

    def follow_goal(self, q_goal, speed, gripper_floor, min_time=0.3, accel=3.0) -> float:
        """Re-aim: one smooth move from where the arm is to q_goal (path checked like move_joints).
        Returns its duration. Call follow_check() often in between. The duration also keeps the peak joint
        acceleration under `accel` rad/s^2 (a quintic peaks at 5.77 dq / T^2): short, sharp re-aims swing
        the whole arm on the base joint and read as unexpected effort there."""
        q_goal = np.asarray(q_goal, dtype=float)
        q0 = self.q()
        speed = min(speed, self.limits.max_joint_speed)
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
            lo, hi = np.minimum(fb["start"], fb["goal"]) - tol, np.maximum(fb["start"], fb["goal"]) + tol
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

    def _stop_on_path(self, q0, q_goal, T, t, lead=0.15, duration=0.4):
        """Slow down and stop at the planned position `lead` s ahead (still on the checked path)."""
        q_stop = q0 + (q_goal - q0) * minjerk((t + lead) / T)
        self.drv.set_arm_positions(q_stop, duration, False)
        time.sleep(duration + 0.1)

    # ----- gripper -----
    def gripper_position(self) -> float:
        return float(self.drv.get_gripper_position())

    def open_gripper(self, position=None, duration=0.8) -> float:
        """Open to a finger position (m) under position control; returns the reached position."""
        position = self.limits.gripper_open if position is None else position
        position = float(np.clip(position, 0.0, 0.040))
        self.drv.set_gripper_mode(ta.Mode.position)
        self.drv.set_gripper_position(position, duration, True)
        return self.gripper_position()

    def close_gripper(self, force=None, settle=0.8) -> float:
        """Close with a limited force (N) so it stops on whatever is between the fingers; returns the
        finger position: about half the object's width, or near 0 if it closed on nothing."""
        force = min(self.limits.gripper_force if force is None else force, 40.0)
        self.drv.set_gripper_mode(ta.Mode.external_effort)
        self.drv.set_gripper_external_effort(-force, 0.0, False)
        time.sleep(settle)
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
