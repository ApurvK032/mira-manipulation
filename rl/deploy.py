"""Run the RL chain on the WidowX AI: the grab policy until red is picked up, a scripted lift to carry height, then
the placement policy until red stands on green with the gripper open (as rl/eval_chained.py does in simulation).

The arm is driven exactly the way system ID measured it (scripts/sysid_record.py): 20 times a second the policy's
action moves a fingertip target (lag_mode="target": at most 3 cm / 15 deg ahead of the fingertips), IK turns it into
joint angles, and SafeArm re-aims a smooth move at it at most every REAIM s (driver 1.8.3); follow_check() runs every
10 ms in between, so joint limits, the workspace box, the fingertip floor and the effort guards all still apply.
Observations are rebuilt from the real joints, the fingertip from FK, the live tracker's cube poses (arm frame) and the
previous action, and normalized with the checkpoint's own VecNormalize statistics (rl/check_deploy_obs.py checks them
against the training environment's).

Main loop (always on, like robot.controller): at rest until Start; then at the WATCH pose; when red is on the table
within reach, one attempt: to a hover pose 13 cm up, the chain, then back to WATCH. The attempt ends with a stack by
StackCubes' own success rule (on the tracker's readings, held 0.5 s): the arm stops, waits for the fingers to finish
opening, rises straight up 4 cm, and only then goes to WATCH (moving away earlier drags the released cube). A timeout
(15 s), a cube lost for too long or a SafetyStop also ends it; anything held is carried back to the table first (never
a scripted stack), and the next attempt starts once red is back on the table.
  - Any SafetyStop: the arm holds where it is and the runner pauses (Resume or Go to rest on the live page).
  - A cube the current phase needs not seen for over 0.5 s: hold; resume when it is seen again, give up after 3 s.
    (While red is held its position comes from the fingertips: the fingers hide it.)
Operator commands: start / pause / resume / rest from the live page (/command); a dry run reads only
runs/dry_command.json. On exit the arm goes back to rest.

Usage:  python -m rl.deploy <grab_model> <place_model> --dry [--episodes 10] [--autostart]   # no arm: rl/dry_world.py
        python -m rl.deploy <grab_model> <place_model>                                         # real arm, at the switch
Logs every policy step to runs/deploy/run_<time>[_dry].jsonl.
"""

import argparse
import json
import signal
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rl.envs.moving_cube import CTRL_DT, GRASP_Z, HALF, OPEN, TIP_BOX, down_rotation, wrap
from rl.envs.stack_cubes import CARRY_Z, STACK_STEPS
from robot.arm import REST, WATCH, SafeArm, SafetyStop
from robot.camera_frame import CameraMismatch, CameraToBase
from robot.controller import Poller, read_command
from robot.grasp import REACH, check_live_camera, cubes_in_base
from robot.ik import IK
from robot.reaim import GoalVelocity

ROOT = Path(__file__).resolve().parents[1]
TICK, REAIM, SPEED = 0.01, 0.15, 1.0  # as in scripts/sysid_record.py, which measured this path
DEADBAND = 0.008  # rad: re-aim only when the joint target moved more than this (as sysid_record)
# Heights are kept relative to the table, measured from red's resting height at the start of each attempt (on the
# real bench the table is 1-7 mm above the arm's base plane depending on where; in simulation it is at 0):
TARGET_MIN = TIP_BOX["z"][0]  # m above the table: the lowest fingertip target, as in training (1.5 cm)
FLOOR = 0.012  # m above the table: SafeArm's hard floor for the fingertip site (fingers end 4.5 mm below it)
TABLE_Z = (-0.005, 0.015)  # plausible table heights in the arm frame; outside this, use the nearest
START_TIP, START_YAW = np.array([0.27, 0.0, 0.13]), np.pi / 2  # the system-ID hover pose; table starts hover 8-14 cm up
MAX_STEPS = 300  # policy steps per attempt (15 s), as the training episodes
LOST_S, GIVE_UP_S = 0.5, 3.0
HOLDING = (0.015, 0.031)  # finger position (m) when closed on a cube, as rl.envs.stack_cubes counts it
RELEASED = 0.030  # finger position (m) the fingers must reach before the arm moves away from a released cube
WIDE, WIDE_OK = 0.040, 0.036  # after a stack: open fully (driver max), and lift only past a 45-deg cube's corners (35.4)
# Stacked, by StackCubes' success rule: red centred on green within 10 mm, the height difference in this band, both
# upright (tilt < 8 deg), gripper open, for STACK_STEPS steps in a row (0.5 s). On the real arm the band is wider
# because the tracker reads stacked cubes 4-5 mm short (REPORT.md, reliability run).
STACK_DZ = {"dry": (0.046, 0.056), "real": (0.040, 0.058)}
PUT_BACK_Z = 0.14  # m above the table: fingertip height carrying red back (its bottom 10.5 cm up, 5.5 cm over a cube)
HYSTERESIS = 0.3  # binary gripper as trained: switching needs |action| above this
GRIP_FORCE = 15.0  # N (robot.arm.Limits.gripper_force)
CHAIN_KWARGS = dict(binary_gripper=True, release_needs_rise=False, place_bonus=0.3, stack_reward=2.5, knock_penalty=6.0,
                    premature_release_penalty=1.0, gripper_hysteresis=HYSTERESIS, grasp_bonus=2.0, lag_mode="target",
                    green_shove_penalty=0.3, family_probs=[1, 0, 0, 0, 0])


# ----- observations, exactly as the training environments build them -----
@dataclass
class Reading:
    p: np.ndarray  # cube centre, arm frame (m)
    yaw: float  # angle of a horizontal face axis (rad; the cube is 90-deg symmetric)
    tilt: float = 0.0  # deg from upright


@dataclass
class Meas:
    tip: np.ndarray  # fingertip site from FK of q
    R: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    finger: float  # gripper finger position (m)


def grip_yaw(R):
    return float(np.arctan2(R[1, 1], R[0, 1]))  # finger opening direction


def cube_yaw(R):
    a = min((R[:, i] for i in range(3)), key=lambda v: abs(v[2]))
    return float(np.arctan2(a[1], a[0]))


def cube_tilt(R):
    return float(np.degrees(np.arccos(np.clip(max(abs(R[2, i]) for i in range(3)), 0, 1))))


def _cube_part(m: Meas, gy, cur: Reading, prev: Reading):
    dyaw = wrap(cur.yaw - gy)
    return [cur.p - m.tip, (cur.p[:2] - prev.p[:2]) / CTRL_DT, [np.sin(4 * dyaw), np.cos(4 * dyaw)]]


def grab_obs(m: Meas, cur: dict, prev: dict, prev_action):
    """MovingCubeGrasp(obstacle=True) observation (34), as rl.eval_chained.grab_obs builds it."""
    gy = grip_yaw(m.R)
    return np.concatenate([m.tip, [np.sin(2 * gy), np.cos(2 * gy)], *_cube_part(m, gy, cur["red"], prev["red"]),
                           [cur["red"].p[2] - HALF], [m.finger / OPEN], m.q, 0.1 * m.qd, prev_action,
                           cur["green"].p - m.tip]).astype(np.float32)


def stack_obs(m: Meas, cur: dict, prev: dict, prev_action):
    """StackCubes observation (41), as StackCubes._obs builds it."""
    gy = grip_yaw(m.R)
    parts = [m.tip, [np.sin(2 * gy), np.cos(2 * gy)]]
    for n in ("red", "green"):
        parts += _cube_part(m, gy, cur[n], prev[n])
    parts += [[cur["red"].p[2] - HALF], cur["red"].p - (cur["green"].p + [0, 0, 2 * HALF]), [m.finger / OPEN], m.q,
              0.1 * m.qd, prev_action]
    return np.concatenate(parts).astype(np.float32)


def clip_tip(p, z_min=TARGET_MIN):
    """The training environments' fingertip target limits (StackCubes._clip_tip); z_min moves the floor with the table."""
    r = np.clip(np.hypot(p[0], p[1]), *TIP_BOX["r"])
    a = np.clip(np.arctan2(p[1], p[0]), -np.arccos(0.1 / r), np.arccos(0.1 / r))
    return np.array([r * np.cos(a), r * np.sin(a), np.clip(p[2], z_min, TIP_BOX["z"][1])])


class Commander:
    """Policy action -> fingertip target, gripper angle and binary gripper command, as lag_mode="target" does it."""

    def __init__(self, ik: IK, q, closed=False, z_min=TARGET_MIN):
        self.ik, self.z_min = ik, z_min
        # joint limits 0.05 rad inside the model's, where the training environments' IK clips its solution
        rng = np.array([ik.m.joint(f"joint_{i}").range for i in range(6)])
        self.q_lo, self.q_hi = rng[:, 0] + 0.05, rng[:, 1] - 0.05
        self.reset(q, closed)

    def reset(self, q, closed=None):
        tip, R = self.ik.fk(q)
        self.tip_target, self.yaw_target, self.q_tgt = tip, grip_yaw(R), np.array(q, dtype=float)
        if closed is not None:
            self.closed = closed

    def target(self, a, tip, R):
        """Move the target by the action (in place); returns (tip target, gripper angle)."""
        t = clip_tip(self.tip_target + 0.01 * a[:3], self.z_min)
        lead = t - tip
        if np.linalg.norm(lead) > 0.03:
            t = tip + lead * 0.03 / np.linalg.norm(lead)
        gy = grip_yaw(R)
        yaw = gy + np.clip(wrap(self.yaw_target + np.radians(5) * a[3] - gy), -np.radians(15), np.radians(15))
        self.tip_target, self.yaw_target = t, yaw
        return t, yaw

    def step(self, a, tip, R):
        """Returns (joint target, gripper closed?)."""
        a = np.clip(np.asarray(a, dtype=float), -1, 1)
        t, yaw = self.target(a, tip, R)
        self.q_tgt = np.clip(self.ik.solve(t, down_rotation(yaw), self.q_tgt, iters=100)[0], self.q_lo, self.q_hi)
        # near the table the IK can leave the fingertips a few mm below the target (it keeps them pointing straight
        # down); raise such a goal to just above SafeArm's floor so it is followed instead of rejected
        floor = self.z_min - TARGET_MIN + FLOOR + 0.001
        for _ in range(3):  # a few tries: at the edge of reach one raise may not be enough
            z = self.ik.fk(self.q_tgt)[0][2]
            if z >= floor:
                break
            t = np.array([t[0], t[1], t[2] + floor - z + 0.001])
            self.q_tgt = np.clip(self.ik.solve(t, down_rotation(yaw), self.q_tgt, iters=100)[0], self.q_lo, self.q_hi)
        self.closed = a[4] > HYSTERESIS if not self.closed else a[4] > -HYSTERESIS
        return self.q_tgt, self.closed


def load_policy(model, make_env):
    """PPO policy plus its observation normalization; accepts model_<steps> and the final model(.zip)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    p = Path(model)
    stem = p.name.removesuffix(".zip")
    steps = stem.rsplit("_", 1)[-1]
    vn = p.parent / (f"vecnormalize_{steps}.pkl" if steps.isdigit() else "vecnormalize.pkl")
    pol = PPO.load(str(p.parent / stem), device="cpu")
    norm = VecNormalize.load(str(vn), DummyVecEnv([make_env]))
    norm.training, norm.norm_reward = False, False

    def act(obs):
        return pol.predict(norm.normalize_obs(obs[None]), deterministic=True)[0][0]
    act.n_obs = pol.observation_space.shape[0]
    return act


def load_chain(grab_model, place_model):
    from rl.envs.moving_cube import MovingCubeGrasp
    from rl.envs.stack_cubes import StackCubes

    grab = load_policy(grab_model, lambda: MovingCubeGrasp(obstacle=True))
    place = load_policy(place_model, lambda: StackCubes(**CHAIN_KWARGS))
    if grab.n_obs != 34 or place.n_obs != 41:
        raise SystemExit(f"expected a 34-number grab policy (green as obstacle) and a 41-number placement policy, "
                         f"got {grab.n_obs} and {place.n_obs}")
    return grab, place


# ----- perception -----
class Cubes:
    """Newest TRACKING reading of each cube (arm frame) from the tracker's snapshots, and how old it is."""

    def __init__(self, src, cal: CameraToBase):
        self.src, self.cal = src, cal
        self.t_stats, self.stats = 0.0, None
        self.seen = {}  # name -> (time, Reading)
        self.recent = {}  # name -> [(time, position)] over the last 0.5 s of TRACKING readings

    def update(self):
        s, t = self.src.latest, self.src.t
        if s is None or t == self.t_stats:
            return
        self.t_stats, self.stats = t, s
        for name, (state, p, R, _) in cubes_in_base(s, self.cal).items():
            if state == "TRACKING" and p is not None:
                self.seen[name] = (t, Reading(p, cube_yaw(R), cube_tilt(R)))
                self.recent[name] = [x for x in self.recent.get(name, []) if t - x[0] <= 0.5] + [(t, p)]

    def stale(self):
        return self.stats is None or time.time() - self.t_stats > LOST_S

    def median(self, name, window=0.3):
        """Median position over the last `window` s of TRACKING readings within 1 cm of the newest one (the cube
        resting where it is now; None if no readings)."""
        h = self.recent.get(name, [])
        if not h:
            return None
        last = h[-1][1]
        near = [p for t, p in h if time.time() - t <= window and np.linalg.norm(p - last) < 0.01]
        return np.median(near, axis=0) if near else last

    def get(self, name):
        """(Reading or None, age in s)."""
        if name not in self.seen:
            return None, np.inf
        t, r = self.seen[name]
        return r, time.time() - t


class Stop(Exception):
    """An operator command ended the current motion."""


# ----- the runner -----
class Runner:
    def __init__(self, arm, src, cal, grab, place, dry, log_path, world=None, ff=False, stream=0.0):
        self.arm, self.cubes, self.grab, self.place = arm, Cubes(src, cal), grab, place
        self.dry, self.world, self.log_path = dry, world, log_path
        self.ik = IK()
        self.cmd = Commander(self.ik, arm.q())
        self.mode, self.pending, self.cmd_seq = "rest", None, None
        self.holding = False  # closed on red (see policy_step)
        self._last_finger = 1.0
        self.table_z = 0.0  # table height in the arm frame (m), measured at each attempt
        self.pick_spot = START_TIP[:2].copy()  # where red was when the attempt started (for putting it back)
        self.last_status = None
        self.deadband = DEADBAND
        self.grip_mode, self.grip_force = "external_effort", GRIP_FORCE  # --grip-mode / --grip-force
        self.ff, self.gv = ff, GoalVelocity()  # ff: feedforward re-aims (robot.reaim), opt-in with --ff
        self.stream = stream  # Hz; > 0: stream targets (driver 1.9.3+, SafeArm.stream_tick) instead of re-aiming
        seeds = [np.array([0.0, a, b, w, 0, 0]) for a in (0.4, 0.9, 1.4) for b in (0.4, 1.0, 1.6) for w in (0.5, 1.2)]
        self.q_start = min((self.ik.solve(START_TIP, down_rotation(START_YAW), s) for s in seeds),
                           key=lambda r: r[1] + r[2] / 100)[0]

    # ----- status, log, commands -----
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

    def log(self, **rec):
        rec["t"] = round(time.time(), 3)
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v)
                                 for k, v in rec.items()}) + "\n")

    def poll(self):
        """Take in the newest snapshot and any new operator command (dry: only runs/dry_command.json)."""
        self.cubes.update()
        cmd = read_command(self.cubes.stats or {}, self.dry)
        if self.cmd_seq is None:
            self.cmd_seq = cmd.get("seq", 0)  # ignore anything pressed before this run started
        elif cmd.get("seq", 0) != self.cmd_seq:
            self.cmd_seq = cmd.get("seq", 0)
            self.pending = cmd.get("c")
            self.log(event="command", command=self.pending)

    def interrupt(self, t=None):
        """watch callback for move_joints: a pause or rest command stops the move."""
        self.poll()
        return self.pending if self.pending in ("pause", "rest") else None

    def move(self, q, speed, label, stoppable=True):
        log = self.arm.move_joints(q, speed=speed, label=label, gripper_floor=self.table_z + FLOOR,
                                   watch=self.interrupt if stoppable else None)
        if log.stopped:
            raise Stop(log.stopped)

    def tip(self):
        return self.ik.fk(self.arm.q())

    def meas(self):
        q = self.arm.q()
        tip, R = self.ik.fk(q)
        return Meas(tip, R, q, self.arm.velocities(), self.arm.gripper_position())

    # ----- scripted moves -----
    def straight_to(self, z, speed, label, stoppable=True):
        """Fingertips straight up or down to height z, keeping x, y and the gripper angle."""
        q = self.arm.q()
        tip, R = self.ik.fk(q)
        q_z, err, _ = self.ik.solve(np.array([tip[0], tip[1], z]), down_rotation(grip_yaw(R)), q)
        if err > 0.005:
            raise SafetyStop(f"{label}: no IK solution at {np.round(tip[:2], 3)}, z {z:.3f}")
        self.move(q_z, speed, label, stoppable=stoppable)

    def to_watch(self, stoppable=True):
        """Up to 13 cm if the fingertips are low, then to WATCH (cubes in full view)."""
        tip, _ = self.tip()
        if tip[2] < 0.11 and np.abs(self.arm.q() - WATCH).max() > 0.02:
            self.straight_to(0.13, 0.5, "rise", stoppable)
        if np.abs(self.arm.q() - WATCH).max() > 0.02:
            self.move(WATCH, 0.8, "to watch pose", stoppable=stoppable)

    def put_back(self, stoppable=True):
        """Holding red with no plan for it: carry it at carry height to where it was picked (or, if green is there
        now, a clear spot on the table), set it down and open. Never a scripted stack: a stack is the policies' own."""
        tip, R = self.tip()
        green, _ = self.cubes.get("green")
        spot = self.pick_spot
        if green is not None and np.linalg.norm(spot - green.p[:2]) < 0.08:
            cands = [np.array([r * np.cos(a), r * np.sin(a)]) for r in (0.24, 0.28, 0.32) for a in np.linspace(-0.7, 0.7, 15)]
            cands = [c for c in cands if np.linalg.norm(c - green.p[:2]) >= 0.08]
            spot = min(cands, key=lambda c: np.linalg.norm(c - tip[:2]))
        self.status("Putting the red cube back on the table")
        z_carry = max(tip[2], self.table_z + PUT_BACK_Z)
        if tip[2] < z_carry - 0.005:
            self.straight_to(z_carry, 0.4, "lift to carry height", stoppable)
        q = self.arm.q()
        q_over, err, _ = self.ik.solve(np.array([spot[0], spot[1], z_carry]), down_rotation(grip_yaw(R)), q)
        if err > 0.005:
            raise SafetyStop(f"put back: no IK solution over {np.round(spot, 3)}")
        self.move(q_over, 0.6, "carry to the put-back spot", stoppable)
        self.straight_to(self.table_z + HALF + GRASP_Z + 0.008, 0.3, "put down", stoppable)
        self.arm.open_gripper(OPEN)
        self.holding = False
        self.straight_to(0.13, 0.5, "rise", stoppable)

    def wake(self):
        if np.abs(self.arm.q() - REST).max() > 0.15 and np.abs(self.arm.q() - WATCH).max() > 0.15:
            raise RuntimeError("arm is neither at rest nor at the watch pose; not starting")
        self.status("Waking up: moving to the watch pose")
        self.arm.open_gripper(OPEN)
        self.move(WATCH, 0.8, "to watch pose")
        self.mode = "awake"
        self.log(event="awake")

    def go_rest(self):
        self.pending = None
        self.arm.hold()
        if self.holding:
            self.put_back(stoppable=False)
        self.status("Going to rest")
        self.to_watch(stoppable=False)
        self.arm.rest_and_idle(speed=0.6)
        self.arm.gripper_idle()
        self.mode = "rest"
        self.log(event="rest")
        self.status("At rest (brakes on). Press Start to wake the arm.")

    # ----- one attempt: grab -> lift -> place -----
    def attempt(self):
        """Returns (outcome, seconds). Outcomes: stacked, timeout, lost, dropped. Raises Stop / SafetyStop."""
        red = self.cubes.median("red")
        if red is None:
            red = self.cubes.get("red")[0].p
        self.table_z = float(np.clip(red[2] - HALF, *TABLE_Z))
        self.pick_spot = red[:2].copy()
        self.cmd.z_min = self.table_z + TARGET_MIN
        self.log(event="attempt", table_z=round(self.table_z, 4))
        self.status("Attempt: to the start pose")
        self.move(self.q_start, 0.8, "to start pose")
        self.arm.open_gripper(OPEN)
        self.begin_motion()
        self.cmd.reset(self.arm.q(), closed=False)
        self.gv.reset()
        self.holding, self._last_finger = False, 1.0
        st = dict(prev_action=np.zeros(5), hist={}, phase="grab", steps=0, stacked=0, z_close=None, hold_since=None,
                  q_goal=self.arm.q(), t_aim=0.0, v_end=np.zeros(6))
        t0 = next_step = next_tick = time.time()
        tick = 1.0 / self.stream if self.stream else TICK
        self.status("Attempt: grab")
        while True:
            self.arm.follow_check()
            self.poll()
            if self.pending in ("pause", "rest"):
                self.arm.hold()
                raise Stop(self.pending)
            now = time.time()
            if now >= next_step:
                next_step = next_step + CTRL_DT if now - next_step < CTRL_DT else now + CTRL_DT  # no bursts after a slow step
                outcome = self.policy_step(st)
                if outcome:
                    return outcome, now - t0
            if st["hold_since"] is not None:
                if now - st["hold_since"] > GIVE_UP_S:
                    return "lost", now - t0
            elif self.stream:  # one smooth point per tick toward the current target
                self.arm.stream_tick(self.cmd.q_tgt, self.table_z + FLOOR)
            elif now - st["t_aim"] >= REAIM and (np.abs(self.cmd.q_tgt - st["q_goal"]).max() > self.deadband
                                                 or (self.ff and np.any(st["v_end"]))):  # ff: once more after it stops
                v_end = self.gv.get() if self.ff else None
                self.arm.follow_goal(self.cmd.q_tgt, SPEED, self.table_z + FLOOR, goal_velocity=v_end)
                st["q_goal"], st["t_aim"] = self.cmd.q_tgt.copy(), time.time()
                st["v_end"] = v_end if self.ff else np.zeros(6)
            next_tick = next_tick + tick if time.time() - next_tick < tick else time.time() + tick  # paced, no bursts
            time.sleep(max(0.0, next_tick - time.time()))

    def begin_motion(self):
        """Start following the policy's targets: streaming (driver 1.9.3+) or re-aiming (1.8.3)."""
        if self.stream:
            self.arm.stream_begin(rate=self.stream)
        else:
            self.arm.follow_begin()

    def policy_step(self, st):
        """One 20 Hz step: observe, act, move the target. Returns "stacked" / "timeout" when the attempt is over."""
        now, m = time.time(), self.meas()
        # Holding red, as StackCubes._red_grasped counts it: closed, fingers at a cube's width, and the fingertips at the
        # cube. The fingers must have stopped there (a gripper closing on air passes through the same range; on the real
        # arm, 2026-10-20, that read as a grasp while the arm happened to rise), and red's last fresh reading must be at
        # the fingertips when the grasp is first counted; then it stays counted while the fingers stay in range.
        in_range = self.cmd.closed and HOLDING[0] < m.finger < HOLDING[1]
        settled = in_range and abs(m.finger - self._last_finger) < 0.0003
        self._last_finger = m.finger
        if not in_range:
            self.holding = False
        elif not self.holding and settled:
            red, age = self.cubes.get("red")
            self.holding = bool(red is not None and age <= LOST_S
                                and np.linalg.norm(m.tip[:2] - red.p[:2]) < 0.02
                                and abs(m.tip[2] - (red.p[2] + GRASP_Z)) < 0.02)
        cur, ages = {}, {}
        for n in ("red", "green"):
            cur[n], ages[n] = self.cubes.get(n)
        if self.holding and ages["red"] > 0.1:  # the fingers hide it: it hangs from the fingertips
            cur["red"], ages["red"] = Reading(m.tip - [0, 0, GRASP_Z], grip_yaw(m.R)), 0.0
        need = ("green",) if st["phase"] == "lift" else ("red", "green")
        lost = [n for n in need if ages[n] > LOST_S] + (["tracker"] if self.cubes.stale() else [])
        if lost:
            if st["hold_since"] is None:
                st["hold_since"] = now
                self.arm.hold()
                self.status(f"Holding: {', '.join(lost)} not seen")
                self.log(event="hold", lost=lost)
            return None
        if st["hold_since"] is not None:  # seen again: carry on from where the arm stopped
            st["hold_since"] = None
            self.begin_motion()
            self.cmd.reset(m.q)
            self.gv.reset()
            st["q_goal"], st["v_end"] = m.q, np.zeros(6)
            self.status(f"Attempt: {st['phase']}")
            self.log(event="resume")
        hist = st["hist"]
        for n in ("red", "green"):
            hist[n] = (hist.get(n, []) + [cur[n]])[-2:]
        prev = {n: hist[n][0] for n in hist}

        # phase changes, as rl.eval_chained: grab until red is held and 1 cm up, lift to carry height, then place
        if st["phase"] == "grab":
            st["z_close"] = (m.tip[2] if st["z_close"] is None else st["z_close"]) if self.holding else None
            if self.holding and m.tip[2] > st["z_close"] + 0.01:
                st["phase"] = "lift"
                self.status("Attempt: lift")
        if st["phase"] == "lift" and (m.tip[2] > CARRY_Z - 0.005 or not self.holding):
            st["phase"] = "place"
            self.status("Attempt: place")
        if st["phase"] == "grab":
            a = self.grab(grab_obs(m, cur, prev, st["prev_action"]))
        elif st["phase"] == "lift":
            a = np.array([0.0, 0.0, 1.0, 0.0, 1.0])
        else:
            a = self.place(stack_obs(m, cur, prev, st["prev_action"]))
        a = np.clip(np.asarray(a, dtype=float), -1, 1)
        was_closed = self.cmd.closed
        q_tgt, closed = self.cmd.step(a, m.tip, m.R)
        self.gv.update(time.time(), q_tgt)
        if closed and not was_closed:
            self.arm.close_gripper(self.grip_force, settle=0.0, mode=self.grip_mode)
        elif was_closed and not closed:
            self.arm.open_gripper(OPEN, duration=0.2, blocking=False)
        st["prev_action"] = a
        st["steps"] += 1

        # stacked: StackCubes' success rule (see STACK_DZ), on the tracker's readings
        dz_lo, dz_hi = STACK_DZ["dry" if self.dry else "real"]
        red, green = cur["red"], cur["green"]
        on = (st["phase"] == "place" and not self.holding and m.finger > HOLDING[1]
              and np.linalg.norm(red.p[:2] - green.p[:2]) < 0.010 and dz_lo < red.p[2] - green.p[2] < dz_hi
              and red.tilt < 8 and green.tilt < 8)
        st["stacked"] = st["stacked"] + 1 if on else 0
        self.log(phase=st["phase"], a=a, tip=m.tip, target=self.cmd.tip_target, q=m.q, q_tgt=q_tgt, finger=m.finger,
                 red=cur["red"].p, green=cur["green"].p, age=[round(min(ages["red"], 99), 3), round(min(ages["green"], 99), 3)])
        if st["stacked"] >= STACK_STEPS:
            return "stacked"
        if st["phase"] in ("lift", "place") and self.cmd.closed and m.finger < HOLDING[0]:
            # closed on nothing while it should be carrying red: never let the placement policy move without it
            # (2026-10-20: after a false grasp it flew to green and came down on it with an empty gripper)
            return "dropped"
        if st["steps"] >= MAX_STEPS:
            return "timeout"
        return None

    def end_attempt(self):
        """Stop where the arm is (the last re-aim is not finished), put back anything held; otherwise let the fingers
        finish opening, rise straight up 4 cm (an open gripper moving sideways or rising early can drag a released
        cube and tip the stack), and only then go to WATCH."""
        self.arm.hold()
        if self.holding:
            self.put_back()  # ends 13 cm up with the gripper open
        else:
            if not self.cmd.closed:
                # open all the way first: a cube turned about 45 deg to the fingers (70.7 mm across its corners) can
                # stay wedged in the policies' 35 mm-per-finger opening and be carried up (seen in a dry run)
                self.arm.open_gripper(WIDE, duration=0.3, blocking=False)
                t0 = time.time()
                while self.arm.gripper_position() < WIDE_OK and time.time() - t0 < 1.0:
                    time.sleep(TICK)
                g = self.arm.gripper_position()
                if g < RELEASED:
                    raise SafetyStop(f"gripper did not open (reads {1000 * g:.0f} mm)")
                if g < WIDE_OK:
                    raise SafetyStop(f"gripper stopped at {1000 * g:.1f} mm opening wide: the cube may be wedged "
                                     f"between the fingers; not lifting")
            # 4 cm if the wrist allows it; near a joint limit (seen in a dry run: joint 3) a shorter rise still clears
            # a released cube, and with none possible the path-checked move to WATCH is the way out
            tip, R = self.tip()
            q = self.arm.q()
            for dz in (0.04, 0.03, 0.02):
                q_up, err, _ = self.ik.solve(np.array([tip[0], tip[1], tip[2] + dz]), down_rotation(grip_yaw(R)), q)
                if err <= 0.005:
                    self.move(q_up, 0.3, "rise clear")
                    break
            else:
                self.log(event="no_rise", tip=tip)
        self.to_watch()

    def red_on_table(self):
        """Red is seen resting on the table within reach (not on green): time for an attempt."""
        red, age = self.cubes.get("red")
        green, gage = self.cubes.get("green")
        if red is None or age > LOST_S or green is None or gage > LOST_S:
            return False, "waiting to see both cubes"
        if np.linalg.norm(red.p[:2] - green.p[:2]) < 0.03 and red.p[2] > green.p[2] + 0.03:
            return False, "stacked. Watching."
        r = np.hypot(red.p[0], red.p[1])
        if not (REACH[0] < r < REACH[1] and red.p[0] > 0.1 and red.p[2] < 0.06):
            return False, f"waiting for the red cube on the table within reach ({100 * r:.0f} cm from the base)"
        return True, ""

    # ----- main loop -----
    def run(self, episodes=None, autostart=False, max_attempts=3):
        """Always on. With episodes (dry runs only), the dry world is reset after each episode and the run ends after
        that many; returns the per-episode results. An episode ends with a stack checked in place, after max_attempts
        attempts, on a SafetyStop, or when red is not back on the table 3 s after an attempt (it was knocked out of
        reach, or the policies stacked it and only the in-place check missed it: the simulator's verdict tells)."""
        dry_eps = episodes is not None and self.world is not None
        results, ep, attempts, last, t_ep, idle_since = [], 0, 0, None, time.time(), None

        def end_episode(outcome, in_place=None):
            """Record the episode; returns True when the run is over (else the world has been reset). in_place: the
            simulator's verdict when the runner's own check passed (before the arm rose and went to WATCH)."""
            nonlocal ep, attempts, t_ep, idle_since
            truth = self.world.truth()
            results.append(dict(episode=ep, outcome=outcome, attempts=attempts, stacked_true=truth["stacked"],
                                stacked_in_place=None if in_place is None else in_place["stacked"],
                                seconds=round(time.time() - t_ep, 1), truth=truth, truth_in_place=in_place))
            print(f"episode {ep}: {outcome} ({attempts} attempt(s)); simulator says stacked={truth['stacked']}"
                  + ("" if in_place is None else f" (in place: {in_place['stacked']})"), flush=True)
            ep, attempts, idle_since = ep + 1, 0, None
            if ep >= episodes or outcome == "safety":
                return True
            self.world.reset(seed=self.world.seed0 + ep, q_arm=self.arm.q())
            t_ep = time.time()
            return False

        if autostart:
            self.pending = "start"
        while True:
            self.poll()
            c, self.pending = self.pending, None
            if c == "rest" and self.mode != "rest":
                self.go_rest()
            elif c == "pause" and self.mode == "awake":
                self.arm.hold()
                self.mode = "paused"
                self.status("Paused (holding). Resume or Go to rest.")
            elif c in ("start", "resume") and self.mode != "awake":
                try:
                    if self.mode == "rest":
                        self.wake()
                    else:
                        self.mode = "awake"
                        if self.holding:
                            self.put_back()
                        self.to_watch()
                except Stop:
                    continue
                except SafetyStop as e:
                    self.arm.hold()
                    self.mode = "paused"
                    self.status(f"SAFETY STOP: {e}. Holding. Resume or Go to rest.")
                    continue
            if self.mode != "awake":
                time.sleep(0.05)
                continue
            go, why = self.red_on_table()
            if not go:
                self.status(why[0].upper() + why[1:])
                if dry_eps and attempts:
                    idle_since = idle_since or time.time()
                    if time.time() - idle_since > 3.0 and end_episode(f"{last}, then: {why}"):
                        return results
                time.sleep(0.05)
                continue
            idle_since = None
            outcome, secs, in_place = None, 0.0, None
            try:
                outcome, secs = self.attempt()
                if dry_eps and outcome == "stacked":
                    in_place = self.world.truth()
                self.end_attempt()
            except Stop:
                continue
            except SafetyStop as e:
                self.arm.hold()
                self.mode, outcome = "paused", "safety"
                self.status(f"SAFETY STOP: {e}. Holding. Resume or Go to rest.")
            attempts, last = attempts + 1, outcome
            self.log(event="attempt_end", outcome=outcome, seconds=round(secs, 1))
            self.status(f"Attempt {attempts}: {outcome} after {secs:.1f} s")
            if dry_eps and (outcome in ("stacked", "safety") or attempts >= max_attempts) and end_episode(outcome, in_place):
                return results


def shutdown(run: Runner):
    """Leave the arm safe whatever happened: put down anything held, back to rest, brakes on; if that fails, hold."""
    arm = run.arm
    try:
        if arm.position_mode:
            run.status("Stopping: taking the arm to rest")
            run.go_rest()
    except Exception as e:
        try:
            arm.hold()
        except Exception:
            pass
        run.status(f"Stopped while the arm is awake ({type(e).__name__}: {e}): it is holding position. "
                   f"Power it off with the switch once safe.")
        return
    arm.gripper_idle()
    arm.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("grab_model")
    ap.add_argument("place_model")
    ap.add_argument("--ip", default="192.168.1.3")
    ap.add_argument("--dry", action="store_true", help="no arm: DryArm and a simulated scene (rl/dry_world.py)")
    ap.add_argument("--ignore-serial", action="store_true",
                    help="use assets/calibration.yaml although the live camera is not the one it was fitted with")
    ap.add_argument("--episodes", type=int, default=10, help="dry only: episodes before stopping")
    ap.add_argument("--autostart", action="store_true", help="dry only: start without a command")
    ap.add_argument("--seed", type=int, default=4242, help="dry only: scene seed of the first episode")
    ap.add_argument("--latency", type=float, default=0.033, help="dry only: simulated tracker latency (s)")
    ap.add_argument("--dropout", type=float, default=0.0, help="dry only: chance per frame a cube goes unseen")
    ap.add_argument("--no-hands", action="store_true", help="dry only: nobody slides the cubes")
    ap.add_argument("--dry-instant-gripper", action="store_true",
                    help="dry only: the simulated gripper moves at once (the training envs' default) instead of at the "
                         "real gripper's speed")
    ap.add_argument("--deadband", type=float, default=DEADBAND, help="rad: smallest joint-target change that re-aims")
    ap.add_argument("--grip-mode", choices=["external_effort", "effort"], default="external_effort",
                    help="gripper close: external_effort (default, as before) or effort (steadiest hold of a picked "
                         "cube on firmware 1.9.4, scripts/grip_lift_test.py)")
    ap.add_argument("--grip-force", type=float, default=GRIP_FORCE, help="N for the gripper close (default 15)")
    ap.add_argument("--stream", type=float, default=0.0, help="Hz (e.g. 50 or 100): stream targets with SafeArm."
                    "stream_tick instead of re-aiming; needs driver and firmware 1.9.x. Train/evaluate with lag_mode=stream")
    ap.add_argument("--ff", action="store_true", help="feedforward re-aims (robot.reaim): each ends moving at the "
                    "target's velocity instead of at rest; train/evaluate with lag_mode=reaim_ff to match")
    args = ap.parse_args()
    if args.autostart and not args.dry:
        raise SystemExit("--autostart is for dry runs only: the real arm starts from the live page's Start button")

    import torch
    torch.set_num_threads(1)

    def on_term(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)

    grab, place = load_chain(args.grab_model, args.place_model)
    log_dir = ROOT / "runs" / "deploy"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}{'_dry' if args.dry else ''}.jsonl"
    cal = CameraToBase(ignore_serial=args.ignore_serial)
    world = None
    if not args.dry:
        try:
            check_live_camera(cal)  # before connecting to the arm; every snapshot is checked again (cubes_in_base)
        except CameraMismatch as e:
            raise SystemExit(str(e))
    if args.dry:
        from rl.dry_world import REAL_GRIPPER_SPEED, SimDryArm, SimWorld

        world = SimWorld(cal, CHAIN_KWARGS, seed=args.seed, latency=args.latency, dropout=args.dropout,
                         hands=not args.no_hands,
                         gripper_speed=None if args.dry_instant_gripper else REAL_GRIPPER_SPEED)
        arm = SimDryArm(world)
        world.reset(seed=args.seed, q_arm=arm.q_cmd())
        world.start()
        src = world
    else:
        arm = SafeArm(args.ip)
        src = Poller()
        src.start()
    run = Runner(arm, src, cal, grab, place, args.dry, log_path, world, ff=args.ff, stream=args.stream)
    run.grip_mode, run.grip_force = args.grip_mode, args.grip_force
    run.deadband = args.deadband
    print(f"log: {log_path}", flush=True)
    results = None
    try:
        results = run.run(episodes=args.episodes if args.dry else None, autostart=args.autostart)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown(run)
        if world is not None:
            world.stop()
    if results:
        n = len(results)
        ok = sum(r["stacked_true"] for r in results)
        first = sum(r["stacked_true"] and r["attempts"] == 1 for r in results)
        said = [r for r in results if r["outcome"] == "stacked"]
        print(f"dry run: {ok}/{n} stacked at the end (simulator's check), {first}/{n} on the first attempt; "
              f"runner said stacked {len(said)}/{n}, simulator agreed in place for "
              f"{sum(bool(r['stacked_in_place']) for r in said)} of those; "
              f"median {np.median([r['seconds'] for r in results]):.1f} s per episode", flush=True)
        (log_path.with_suffix(".summary.json")).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
