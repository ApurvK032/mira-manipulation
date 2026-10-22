"""The world for a dry run of rl/deploy.py: camera, cubes, hands and the arm's motors, simulated in real time.

A MuJoCo StackCubes scene (rl.envs.stack_cubes, table starts) is stepped against the wall clock in a background
thread:
  - the arm's joints follow the trajectory the DryArm commands (robot.controller.DryArm models the driver's smooth
    re-aims; the scene's position servos stand in for the motors), and the gripper follows open / close;
  - hands slide the cubes as in training (StackCubes' own hand model), starting `hand_delay` s into each episode;
  - a fake tracker publishes both cubes at 60 Hz in the live viewer's stats.json format (camera frame, through the
    inverse of assets/calibration.yaml), `latency` s late, with the training noise (1.5 mm, 2 deg), and optionally
    unseen ("LOST") for 0.05-0.3 s at a time.
SimDryArm is the DryArm with its readings (joints, speeds, gripper) taken from the scene, as SafeArm reads the real
arm's. No hardware is touched: nothing here imports the arm driver's connection.
"""

import threading
import time

import mujoco
import numpy as np

from rl.envs.moving_cube import CLOSED, CTRL_DT, OPEN
from rl.envs.stack_cubes import StackCubes
from robot.controller import DryArm


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


REAL_GRIPPER_SPEED = (0.048, 0.125)  # m/s per finger closing / opening: the real gripper, firmware 1.9.4 (2026-10-20)


class SimWorld(threading.Thread):
    def __init__(self, cal, env_kwargs, seed=0, latency=0.033, fps=60.0, noise=0.0015, yaw_noise_deg=2.0,
                 dropout=0.0, hands=True, hand_delay=2.0, gripper_speed=REAL_GRIPPER_SPEED):
        super().__init__(daemon=True)
        kw = dict(env_kwargs)
        if not hands:
            kw.update(push_red_prob=0.0, push_green_prob=0.0)
        self.env = StackCubes(**kw)
        self.cal, self.seed0 = cal, seed
        self.latency, self.frame_dt, self.noise, self.yaw_noise = latency, 1.0 / fps, noise, np.radians(yaw_noise_deg)
        self.dropout, self.hand_delay = dropout, hand_delay
        # the gripper moves at the real gripper's speed (None: at once, as the training envs by default); with the
        # instant gripper, a gripper closing on air never looked like a grasp in dry runs (it did on the real arm)
        self.gripper_speed = gripper_speed
        self.rng = np.random.default_rng(seed)
        self.lock = threading.Lock()
        self.arm = None  # the SimDryArm whose commands drive the scene
        self.latest, self.t = None, 0.0  # newest stats snapshot and when it was published (wall clock), as Poller
        self.running = True
        self.max_behind = 0.0  # s the simulation fell behind the wall clock at most

    # ----- episodes -----
    def reset(self, seed, q_arm):
        """New table start (StackCubes' own sampling); the arm wherever it is now, gripper as commanded."""
        e = self.env
        with self.lock:
            e.reset(seed=seed)
            d = e.d
            d.qpos[e.qadr], d.qvel[e.dadr], d.ctrl[:6] = q_arm, 0.0, q_arm
            g = OPEN if self.arm is None else self.arm.grip_cmd
            d.qpos[e.finger] = d.qpos[e.finger_r] = g
            d.ctrl[6] = g
            mujoco.mj_forward(e.m, d)
            for h in e.hands.values():  # hands start once the arm is on its way
                h["start"] += self.hand_delay
                h["end"] += self.hand_delay
            e.steps, self.sim_t, self.t0 = 0, 0.0, time.time()
            self.pushes, self.next_ctrl, self.next_frame = {}, 0.0, 0.0
            self.frames, self.unseen = [], {"red": 0.0, "green": 0.0}
            self.was_grasped = False

    def stop(self):
        self.running = False

    # ----- simulation -----
    def run(self):
        while self.running:
            with self.lock:
                behind = (time.time() - self.t0) - self.sim_t
                self.max_behind = max(self.max_behind, behind)
                n = int(min(behind, 0.1) / self.env.m.opt.timestep)  # catch up, at most 0.1 s per pass
                for _ in range(n):
                    self._substep()
            time.sleep(0.001)

    def _substep(self):
        e, d = self.env, self.env.d
        if self.sim_t >= self.next_ctrl:  # hands, as StackCubes.step does once per control step
            self.next_ctrl += CTRL_DT
            e.steps = int(round(self.sim_t / CTRL_DT))
            self.pushes = {n: e._hand_velocity(n, h) for n, h in e.hands.items()}
            grasped = e._red_grasped()
            rh = e.hands["red"]
            if rh["active"]:
                rp, _, _ = e._cube("red")
                tip, _ = e._tip()
                if np.linalg.norm(tip[:2] - rp[:2]) < 0.02 and tip[2] < rp[2] + 0.045:
                    rh["active"] = False
            if grasped and not self.was_grasped and e.green_hand_armed:
                e.hands["green"] = e._new_hand(True, e.steps * CTRL_DT + e.np_random.uniform(0.0, 1.5),
                                               e.np_random.uniform(0.5, 2.5))
                e.green_hand_armed = False
            self.was_grasped = grasped
        if self.arm is not None:
            d.ctrl[:6] = self.arm.q_cmd()
            goal = self.arm.grip_cmd
            if self.gripper_speed is None:
                d.ctrl[6] = goal
            else:
                v = self.gripper_speed[0] if goal < d.ctrl[6] else self.gripper_speed[1]
                step = v * e.m.opt.timestep
                d.ctrl[6] += min(max(goal - d.ctrl[6], -step), step)
        for n, pv in self.pushes.items():
            if pv is not None:
                _, v = e.adr[n]
                d.qvel[v:v + 2] = pv[0]
                d.qvel[v + 5] = pv[1]
        mujoco.mj_step(e.m, d)
        self.sim_t += e.m.opt.timestep
        if self.sim_t >= self.next_frame:
            self.next_frame += self.frame_dt
            self._frame()

    def _frame(self):
        """One tracker frame: noisy poses now, published `latency` s later."""
        e = self.env
        cubes = {}
        for n in ("red", "green"):
            if self.sim_t < self.unseen[n] or (self.dropout and self.rng.random() < self.dropout):
                if self.sim_t >= self.unseen[n]:
                    self.unseen[n] = self.sim_t + self.rng.uniform(0.05, 0.3)
                cubes[n] = {"state": "LOST", "pos": None, "rot": None, "check": [0.0, 0.0, 0.0, 0.0], "source": ""}
                continue
            q, _ = e.adr[n]
            R = np.empty(9)
            mujoco.mju_quat2Mat(R, e.d.qpos[q + 3:q + 7])
            R = rot_z(self.rng.normal(0, self.yaw_noise)) @ R.reshape(3, 3)
            p = e.d.qpos[q:q + 3] + self.rng.normal(0, self.noise, 3)
            Rbc, tbc = self.cal.T[:3, :3], self.cal.T[:3, 3]
            p_cam = Rbc.T @ (p - tbc) / self.cal.k
            cubes[n] = {"state": "TRACKING", "pos": (1000 * p_cam).tolist(), "rot": (Rbc.T @ R).tolist(),
                        "check": [1.0, 1.0, 1.0, 0.0], "source": "sim"}
        self.frames.append((self.sim_t, cubes))
        while self.frames and self.frames[0][0] <= self.sim_t - self.latency:
            _, c = self.frames.pop(0)
            self.latest, self.t = {"t": time.time(), "cubes": c, "command": {},
                                   "camera_serial": self.cal.serial}, time.time()

    # ----- readings for the arm, and the simulator's own verdict -----
    def read(self):
        e = self.env
        with self.lock:
            return e.d.qpos[e.qadr].copy(), e.d.qvel[e.dadr].copy(), float(e.d.qpos[e.finger])

    def truth(self):
        """The simulator's check (StackCubes' success test, without the 0.5 s): red centred on green within
        10 mm, both upright, gripper open."""
        e = self.env
        with self.lock:
            rp, _, rt = e._cube("red")
            gp, _, gt = e._cube("green")
            opening = float(e.d.qpos[e.finger])
        on = bool(np.linalg.norm(rp[:2] - gp[:2]) < 0.010 and 0.046 < rp[2] - gp[2] < 0.056 and rt < 8 and gt < 8)
        return dict(stacked=on and opening > 0.031, red=np.round(rp * 1000, 1).tolist(),
                    green=np.round(gp * 1000, 1).tolist(), tilt=[round(rt, 1), round(gt, 1)], knocked=bool(rt > 30 or gt > 30))


class SimDryArm(DryArm):
    """DryArm (same checks and driver timing) whose joint, speed and gripper readings come from the SimWorld."""

    def __init__(self, world: SimWorld):
        super().__init__()
        self.world, self.grip_cmd = world, OPEN
        world.arm = self

    def q_cmd(self):
        """Where the driver is commanding the joints right now."""
        if getattr(self, "_traj", None) is not None:
            return self._state()[0]
        return self._q.copy()

    def q(self):
        return self.world.read()[0]

    def velocities(self):
        return self.world.read()[1]

    def gripper_position(self):
        return self.world.read()[2]

    def open_gripper(self, position=OPEN, duration=0.8, blocking=True):
        self.grip_cmd = position
        if blocking:
            time.sleep(duration)
        return self.gripper_position()

    def close_gripper(self, force=None, settle=0.8, mode="external_effort"):
        self.grip_cmd = CLOSED
        time.sleep(settle)
        return self.gripper_position()

