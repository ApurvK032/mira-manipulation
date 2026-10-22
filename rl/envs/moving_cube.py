"""Grab a moving cube: a Gymnasium environment for the WidowX AI in MuJoCo.

A simulated hand slides a 5 cm cube around the table at random speeds and turns. The hand lets go only
once the gripper has caught up with the cube (fingertips over it and low), so waiting for it to stop is
not an option: the policy has to intercept a moving cube, grasp it and lift it. Success: the cube held
4 cm above the table for 0.5 s.

Control at 20 Hz. Action (5, in [-1, 1]): fingertip move dx, dy, dz (up to 1 cm per step), wrist turn
(up to 5 deg per step), gripper (-1 open .. +1 closed). Moves go through a few damped-least-squares IK
iterations to joint targets for the arm's position actuators, with the fingertip target clipped to a safe
workspace (the reachable ring 18-36 cm from the base, never lower than 1.5 cm above the table).

Observation (31): fingertip position and gripper angle (as sin/cos of 2x, the gripper is 180-deg
symmetric), cube position relative to the fingertips, cube velocity, cube angle relative to the gripper
(sin/cos of 4x, the cube is 90-deg symmetric), cube height above its resting height, gripper opening,
joint angles and speeds, previous action. Cube readings are delayed by one or two control steps and noisy,
like the real perception (robot.controller reads ~60 fps poses with ~30 ms latency).
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from rl.envs.variations import Variations
from robot.reaim import Reaim  # the deployment path's re-aims (lag_mode "reaim" / "reaim_ff")
from robot.stream import Streamer  # the streaming path of driver 1.9.3+ (lag_mode "stream")

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "assets" / "trossen_arm_mujoco" / "wxai" / "wxai_follower.xml"

HALF = 0.025  # cube half edge (m)
CTRL_DT = 0.05  # 20 Hz
GRASP_Z = 0.010  # fingertips this far above the cube centre hold its top 15 mm (as on the real arm)
RING = (0.20, 0.34)  # where cubes start and move (m from the base)
TIP_BOX = dict(r=(0.16, 0.38), z=(0.015, 0.20))  # fingertip target limits
OPEN, CLOSED = 0.035, 0.0  # gripper carriage positions (m)
SUCCESS_LIFT, SUCCESS_STEPS = 0.04, 10


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def down_rotation(yaw):
    """Gripper pointing straight down, fingers opening along the horizontal direction yaw."""
    x = np.array([0.0, 0.0, -1.0])
    y = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    return np.stack([x, y, np.cross(x, y)], axis=1)


class MovingCubeGrasp(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps=160, static_prob=0.15, max_speed=0.12, obs_noise=0.0015, yaw_noise_deg=2.0,
                 delay_steps=(1, 2), obstacle=False, lag_tau=(0.0, 0.0), lag_mode="measured", obs_dropout=0.0, leash=0.03, knock_penalty=5.0, bump_penalty=0.0,
                 calib_error=(0.0, 0.0), obs_bias=0.0, friction_scale=None, mass_scale=None,
                 stream_accel=(3.0, 6.0), stream_tau=(0.04, 0.1), gripper_speed=None, gripper_speed_scale=None, seed=None):
        """obstacle: a second (green) cube rests on the table 8+ cm away. The policy also sees its position
        (observation 34), the hand never slides red into it, and knocking it over or shoving it 2 cm ends the
        episode (as when grabbing red next to green before stacking). lag_tau: the arm's joint targets are
        low-pass filtered with a random time constant in this range (s), as in rl.envs.stack_cubes.
        lag_mode: "measured" (each step's move is from where the fingertips are; with lag that makes the arm
        slow, not just late: the first grab2 run never caught a cube) or "target" (the policy moves a commanded
        target, kept at most 3 cm / 15 deg ahead of the fingertips, which the lagging arm follows: late but at
        full speed, like the real arm following SafeArm targets). obs_dropout: per step, the chance that the cube
        readings freeze for 1-6 steps (hidden from the camera). lag_mode="reaim": "target", but the arm follows the
        deployment path's smooth re-aims (robot.reaim.Reaim; "reaim_ff": with feedforward end velocities)
        instead of a first-order lag; lag_tau is unused, and delay_steps
        should cover perception only. leash: how far (m) the target may run ahead of the fingertips.
        knock_penalty: charged when red is knocked over (ends the episode). bump_penalty: charged each step red moves
        more than 3 mm or tilts more than 5 deg while neither held nor pushed by the hand (the fingers bumped it),
        to teach a precise grasp. calib_error, obs_bias, friction_scale, mass_scale: per-episode sim-to-real
        variations (rl.envs.variations)."""
        # gripper_speed: (closing, opening) m/s per finger, opt-in; None sets the gripper target at once, as before.
        # Real WidowX AI gripper, firmware 1.9.4: closes at about 0.048 m/s, opens at about 0.125 m/s (real-arm log,
        # 2026-10-20); the simulated one otherwise shuts in about 0.09 s.
        # gripper_speed_scale=(lo, hi): per episode, both speeds times a factor drawn from this range (training margin).
        self.gripper_speed = tuple(gripper_speed) if gripper_speed is not None else None
        self.gripper_speed_base, self.gripper_speed_scale = self.gripper_speed, gripper_speed_scale
        super().__init__()
        self.obstacle, self.lag_tau, self.target_mode = obstacle, lag_tau, lag_mode in ("target", "reaim", "reaim_ff", "stream")
        self.reaim = Reaim(ff=lag_mode == "reaim_ff") if lag_mode in ("reaim", "reaim_ff") else None
        # "stream": the driver 1.9.3 streaming path (robot.stream.Streamer at 100 Hz), with per-episode limits
        self.stream = Streamer() if lag_mode == "stream" else None
        self.stream_accel, self.stream_tau = stream_accel, stream_tau
        self.obs_dropout, self.leash = obs_dropout, leash
        self.knock_penalty, self.bump_penalty = knock_penalty, bump_penalty
        self.max_steps, self.static_prob, self.max_speed = max_steps, static_prob, max_speed
        self.obs_noise, self.yaw_noise = obs_noise, np.radians(yaw_noise_deg)
        self.delay_steps = delay_steps
        spec = mujoco.MjSpec.from_file(str(MODEL))
        w = spec.worldbody
        w.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.1], friction=[0.8, 0.005, 0.0001])
        cube = w.add_body(name="cube", pos=[0.28, 0.0, HALF])
        cube.add_freejoint(name="cube_free")
        cube.add_geom(name="cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=[HALF] * 3, mass=0.05,
                      friction=[1.0, 0.005, 0.0001], rgba=[1, 0.1, 0.1, 1])
        if obstacle:
            g = w.add_body(name="green", pos=[0.26, 0.10, HALF])
            g.add_freejoint(name="green_free")
            g.add_geom(name="green", type=mujoco.mjtGeom.mjGEOM_BOX, size=[HALF] * 3, mass=0.05,
                       friction=[1.0, 0.005, 0.0001], rgba=[0.1, 0.8, 0.1, 1])
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.var = Variations(self.m, "table", {"red": "cube", **({"green": "green"} if obstacle else {})}, calib_error,
                              obs_bias, friction_scale, mass_scale)
        self.d_ik = mujoco.MjData(self.m)
        self.n_sub = int(round(CTRL_DT / self.m.opt.timestep))
        self.site = self.m.site("ee_site").id
        joints = [self.m.joint(f"joint_{i}") for i in range(6)]
        self.qadr = np.array([j.qposadr[0] for j in joints])
        self.dadr = np.array([j.dofadr[0] for j in joints])
        rng = np.array([j.range for j in joints])
        self.q_lo, self.q_hi = rng[:, 0] + 0.05, rng[:, 1] - 0.05
        if self.reaim:
            self.reaim.lo, self.reaim.hi = self.q_lo, self.q_hi
        self.finger = self.m.joint("left_carriage_joint").qposadr[0]
        fj = self.m.joint("cube_free")
        self.cq, self.cv = fj.qposadr[0], fj.dofadr[0]
        self.action_space = gym.spaces.Box(-1.0, 1.0, (5,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (34 if obstacle else 31,), np.float32)
        if obstacle:
            self.gq = self.m.joint("green_free").qposadr[0]
        self._seed = seed

    # ----- helpers -----
    def _tip(self, d=None):
        d = d or self.d
        return d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()

    def _grip_yaw(self, R):
        return float(np.arctan2(R[1, 1], R[0, 1]))  # finger opening direction

    def _cube(self):
        p = self.d.qpos[self.cq:self.cq + 3].copy()
        qw, qx, qy, qz = self.d.qpos[self.cq + 3:self.cq + 7]
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, np.array([qw, qx, qy, qz]))
        R = R.reshape(3, 3)
        a = min((R[:, i] for i in range(3)), key=lambda v: abs(v[2]))  # a horizontal face axis
        tilt = np.degrees(np.arccos(np.clip(max(abs(R[2, i]) for i in range(3)), 0, 1)))
        return p, float(np.arctan2(a[1], a[0])), float(tilt)

    def _ik(self, q, p_goal, R_goal, iters=2, damping=0.05, w_rot=0.3):
        d = self.d_ik
        d.qpos[:] = self.d.qpos
        d.qpos[self.qadr] = q
        jacp, jacr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_comPos(self.m, d)
            p, R = self._tip(d)
            e = np.concatenate([p_goal - p, w_rot * 0.5 * sum(np.cross(R[:, i], R_goal[:, i]) for i in range(3))])
            mujoco.mj_jacSite(self.m, d, jacp, jacr, self.site)
            J = np.vstack([jacp[:, self.dadr], w_rot * jacr[:, self.dadr]])
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), e)
            d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + np.clip(dq, -0.15, 0.15), self.q_lo, self.q_hi)
        return d.qpos[self.qadr].copy()

    def _move_target(self, tip, R, a):
        """Commanded fingertip target and gripper angle, moved by the action and kept on a leash."""
        t = self._clip_tip(self.tip_target + 0.01 * a[:3])
        lead = t - tip
        if np.linalg.norm(lead) > self.leash:
            t = tip + lead * self.leash / np.linalg.norm(lead)
        self.tip_target = t
        gy = self._grip_yaw(R)
        dy = wrap(self.yaw_target + np.radians(5) * a[3] - gy)
        self.yaw_target = gy + np.clip(dy, -np.radians(15), np.radians(15))
        return self.tip_target, self.yaw_target

    def _clip_tip(self, p):
        r = np.clip(np.hypot(p[0], p[1]), *TIP_BOX["r"])
        a = np.arctan2(p[1], p[0])
        a = np.clip(a, -np.arccos(0.1 / r), np.arccos(0.1 / r))  # in front of the base
        return np.array([r * np.cos(a), r * np.sin(a), np.clip(p[2], *TIP_BOX["z"])])

    def _random_ring_point(self):
        r = self.np_random.uniform(*RING)
        a = self.np_random.uniform(-1.0, 1.0)
        return np.array([r * np.cos(a), r * np.sin(a)])

    # ----- the hand that slides the cube -----
    def _push_velocity(self):
        if not self.pushing:
            return None
        h = self.hand
        h["t"] += CTRL_DT
        if h["t"] > h["seg_end"]:  # new segment: speed and heading change (sometimes a pause)
            h["seg_end"] = h["t"] + self.np_random.uniform(0.4, 1.5)
            h["speed"] = 0.0 if self.np_random.random() < 0.15 else self.np_random.uniform(0.02, self.max_speed)
            h["heading"] += self.np_random.normal(0, 1.2)
            h["yaw_rate"] = self.np_random.normal(0, 0.6)
        p, _, _ = self._cube()
        r, a = np.hypot(p[0], p[1]), np.arctan2(p[1], p[0])
        v = h["speed"] * np.array([np.cos(h["heading"]), np.sin(h["heading"])])
        # keep it in the ring: steer back toward the middle when near an edge
        if r < RING[0] + 0.01 or r > RING[1] - 0.01 or abs(a) > 0.95:
            mid = 0.27 * np.array([np.cos(0.0), np.sin(0.0)]) - p[:2]
            h["heading"] = float(np.arctan2(mid[1], mid[0]))
            v = h["speed"] * np.array([np.cos(h["heading"]), np.sin(h["heading"])])
        if self.obstacle:  # never slide red into green
            gp = self.d.qpos[self.gq:self.gq + 2]
            if np.linalg.norm(gp - p[:2]) < 0.08:
                away = p[:2] - gp
                h["heading"] = float(np.arctan2(away[1], away[0]))
                v = h["speed"] * np.array([np.cos(h["heading"]), np.sin(h["heading"])])
        return v, h["yaw_rate"]

    def _green_disturbed(self):
        q = self.d.qpos[self.gq + 3:self.gq + 7]
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, q)
        R = R.reshape(3, 3)
        tilt = np.degrees(np.arccos(np.clip(max(abs(R[2, i]) for i in range(3)), 0, 1)))
        return tilt > 20 or np.linalg.norm(self.d.qpos[self.gq:self.gq + 2] - self.green_start) > 0.02

    def _hand_lets_go(self):
        tip, _ = self._tip()
        p, _, _ = self._cube()
        caught = np.linalg.norm(tip[:2] - p[:2]) < 0.02 and tip[2] < p[2] + 0.045
        return caught or self.hand["t"] > self.hand["release_at"]

    # ----- observation -----
    def _obs(self):
        tip, R = self._tip()
        gy = self._grip_yaw(R)
        k = self.delay
        cp, cyaw = self.cube_hist[-1 - k]
        cp_prev, _ = self.cube_hist[-2 - k]
        vel = (cp[:2] - cp_prev[:2]) / CTRL_DT
        rel = cp - tip
        dyaw = wrap(cyaw - gy)
        q = self.d.qpos[self.qadr]
        qd = self.d.qvel[self.dadr]
        return np.concatenate([
            tip, [np.sin(2 * gy), np.cos(2 * gy)], rel, vel, [np.sin(4 * dyaw), np.cos(4 * dyaw)],
            [cp[2] - self.rest_z], [self.d.qpos[self.finger] / OPEN], q, 0.1 * qd, self.prev_action,
        ] + ([self.green_hist[-1 - k] - tip] if self.obstacle else [])).astype(np.float32)

    def _record_cube(self):
        if self.cube_hist and (self.stale > 0 or self.np_random.random() < self.obs_dropout):
            # readings frozen (hidden from the camera)
            self.stale = self.stale - 1 if self.stale > 0 else int(self.np_random.integers(0, 6))
            self.cube_hist.append(self.cube_hist[-1])
            self.cube_hist = self.cube_hist[-max(6, self.delay + 2):]
            if self.obstacle:
                self.green_hist.append(self.green_hist[-1])
                self.green_hist = self.green_hist[-max(6, self.delay + 2):]
            return
        p, yaw = self.var.reading("red", *self._cube()[:2])
        noisy = p + self.np_random.normal(0, self.obs_noise, 3)
        self.cube_hist.append((noisy, yaw + self.np_random.normal(0, self.yaw_noise)))
        self.cube_hist = self.cube_hist[-max(6, self.delay + 2):]
        if self.obstacle:
            self.green_hist.append(self.var.reading("green", self.d.qpos[self.gq:self.gq + 3])[0]
                                   + self.np_random.normal(0, self.obs_noise, 3))
            self.green_hist = self.green_hist[-max(6, self.delay + 2):]

    # ----- gym API -----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed if seed is not None else self._seed)
        self._seed = None
        self.var.sample(self.np_random)
        mujoco.mj_resetData(self.m, self.d)
        # arm: fingertips 8-14 cm up over a random spot, pointing down (as when following or watching)
        start = np.append(self._random_ring_point(), self.np_random.uniform(0.08, 0.14))
        yaw0 = self.np_random.uniform(-np.pi / 2, np.pi / 2)
        az = np.arctan2(start[1], start[0])
        q = np.array([az, 1.0, 1.0, 1.0, 0.0, 0.0])
        for _ in range(12):
            q = self._ik(q, start, down_rotation(yaw0), iters=10)
        self.d.qpos[self.qadr] = q
        self.d.ctrl[:6] = q
        self.q_cmd = q.copy()
        self.tau = self.np_random.uniform(*self.lag_tau)
        if self.reaim:
            self.reaim.reset(q)
        if self.stream:
            self.stream.accel = self.np_random.uniform(*self.stream_accel)
            self.stream.tau = self.np_random.uniform(*self.stream_tau)
            self.stream.reset(q)
            self.s_prev, self.s_next, self.stream_goal = (np.array(q, dtype=float) for _ in range(3))
        self.d.qpos[self.finger] = OPEN
        self.d.ctrl[6] = OPEN
        self.grip_goal = OPEN  # the commanded gripper target (ctrl[6] lags it with gripper_speed)
        if self.gripper_speed_base is not None and self.gripper_speed_scale is not None:  # drawn only when set
            k = self.np_random.uniform(*self.gripper_speed_scale)
            self.gripper_speed = (self.gripper_speed_base[0] * k, self.gripper_speed_base[1] * k)
        # cube: random spot and angle in the ring
        cp = self._random_ring_point()
        cyaw = self.np_random.uniform(-np.pi, np.pi)
        self.d.qpos[self.cq:self.cq + 3] = [cp[0], cp[1], HALF]
        self.d.qpos[self.cq + 3:self.cq + 7] = [np.cos(cyaw / 2), 0, 0, np.sin(cyaw / 2)]
        if self.obstacle:
            while True:
                gp = self._random_ring_point()
                if np.linalg.norm(gp - cp) >= 0.08:
                    break
            gyaw = self.np_random.uniform(-np.pi, np.pi)
            self.d.qpos[self.gq:self.gq + 3] = [gp[0], gp[1], HALF]
            self.d.qpos[self.gq + 3:self.gq + 7] = [np.cos(gyaw / 2), 0, 0, np.sin(gyaw / 2)]
            self.green_start = np.array([gp[0], gp[1]])
        mujoco.mj_forward(self.m, self.d)
        self.rest_z = HALF
        self.pushing = self.np_random.random() > self.static_prob
        self.hand = dict(t=0.0, seg_end=0.0, speed=0.0, heading=self.np_random.uniform(-np.pi, np.pi), yaw_rate=0.0,
                         release_at=self.np_random.uniform(2.0, 7.0))
        self.delay = int(self.np_random.integers(self.delay_steps[0], self.delay_steps[1] + 1))
        self.prev_action = np.zeros(5)
        self.steps, self.held_steps, self.grasped_before = 0, 0, False
        tip, R = self._tip()
        self.tip_target, self.yaw_target = tip.copy(), self._grip_yaw(R)
        self.cube_hist, self.green_hist, self.stale = [], [], 0
        for _ in range(max(4, self.delay + 2)):
            self._record_cube()
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        tip, R = self._tip()
        if self.target_mode:
            target, yaw = self._move_target(tip, R, a)
            seed = self.reaim.goal if self.reaim else (self.stream_goal if self.stream else self.q_cmd)  # last goal, as
            # rl/deploy.py does
            q_goal = self._ik(seed, target, down_rotation(yaw), iters=3)
        else:
            target = self._clip_tip(tip + 0.01 * a[:3])
            yaw = self._grip_yaw(R) + np.radians(5) * a[3]
            q_goal = self._ik(self.d.qpos[self.qadr], target, down_rotation(yaw))
        self.grip_goal = OPEN * (1 - (a[4] + 1) / 2)
        if self.gripper_speed is None:
            self.d.ctrl[6] = self.grip_goal
        cp0 = self.d.qpos[self.cq:self.cq + 3].copy()  # red before this step (for bump_penalty)
        push = self._push_velocity()
        alpha = 1.0 if self.tau <= 1e-3 else min(1.0, self.m.opt.timestep / self.tau)
        if self.reaim:
            self.reaim.aim(q_goal)
        if self.stream:
            self.stream_goal = q_goal
            k = max(1, int(round(self.stream.dt / self.m.opt.timestep)))  # physics steps per stream tick
        for i in range(self.n_sub):
            if self.reaim:
                self.q_cmd = self.reaim.advance(self.m.opt.timestep)
            elif self.stream:  # a new point each tick; the driver interpolates linearly to it over the tick
                if i % k == 0:
                    self.s_prev, self.s_next = self.s_next, self.stream.step(q_goal)[0]
                self.q_cmd = self.s_prev + (self.s_next - self.s_prev) * ((i % k + 1) / k)
            else:
                self.q_cmd += alpha * (q_goal - self.q_cmd)  # the arm's slower response (none by default)
            self.d.ctrl[:6] = self.q_cmd
            if self.gripper_speed is not None:  # the real gripper's speed (opt-in): the actuator target follows
                g = self.d.ctrl[6]                # the command at most this fast
                v = self.gripper_speed[0] if self.grip_goal < g else self.gripper_speed[1]
                step = v * self.m.opt.timestep
                self.d.ctrl[6] = g + min(max(self.grip_goal - g, -step), step)
            if push is not None:
                v, yr = push
                self.d.qvel[self.cv:self.cv + 2] = v
                self.d.qvel[self.cv + 5] = yr
            mujoco.mj_step(self.m, self.d)
        if self.pushing and self._hand_lets_go():
            self.pushing = False
        self._record_cube()
        self.steps += 1

        # ----- reward -----
        tip, R = self._tip()
        cp, cyaw, tilt = self._cube()
        grasp_pt = cp + [0, 0, GRASP_Z]
        dist = np.linalg.norm(tip - grasp_pt)
        dyaw = wrap(cyaw - self._grip_yaw(R))
        opening = self.d.qpos[self.finger]
        between = np.linalg.norm(tip[:2] - cp[:2]) < 0.015 and abs(tip[2] - grasp_pt[2]) < 0.015
        grasped = between and 0.015 < opening < 0.031
        lift = cp[2] - self.rest_z
        # Per-step shaping is capped at about 2.1, and the success bonus is worth more than every remaining
        # step at that cap, so finishing always pays more than hovering just short of success (the first
        # version paid ~6 per step for an almost-lifted cube and only +20 for success, and learned not to
        # finish), and finishing sooner pays more.
        r = 0.5 * (1.0 - np.tanh(8.0 * dist))                      # get to the grasp point
        r += 0.1 * np.cos(4 * dyaw) * (dist < 0.05)                # fingers along the faces when close
        r += 0.5 * grasped                                          # holding it
        r += 1.0 * grasped * np.clip(lift / SUCCESS_LIFT, 0, 1)     # lifting it
        r -= 0.01 + 0.01 * float(np.sum(a[:4] ** 2)) + 0.02 * float(np.sum((a - self.prev_action) ** 2))
        if self.bump_penalty and push is None and not grasped and (np.linalg.norm(cp[:2] - cp0[:2]) > 0.003 or tilt > 5):
            r -= self.bump_penalty  # the fingers bumped red
            info_bump = True
        else:
            info_bump = False
        self.held_steps = self.held_steps + 1 if (grasped and lift > SUCCESS_LIFT) else 0
        terminated, info = False, {"success": False}
        if self.held_steps >= SUCCESS_STEPS:
            r += 10.0 + 2.2 * (self.max_steps - self.steps)
            terminated, info["success"] = True, True
        elif tilt > 30 and not grasped:  # knocked over
            r -= self.knock_penalty
            terminated, info["knocked"] = True, True
        elif self.obstacle and self._green_disturbed():  # knocked over or shoved the other cube
            r -= 3.0
            terminated, info["knocked_green"] = True, True
        elif not (RING[0] - 0.04 < np.hypot(cp[0], cp[1]) < RING[1] + 0.04):  # pushed out of reach
            r -= 2.0
            terminated, info["out"] = True, True
        truncated = self.steps >= self.max_steps
        self.prev_action = a
        info.update(dist=float(dist), grasped=bool(grasped), pushing=bool(self.pushing), bump=info_bump)
        return self._obs(), float(r), terminated, truncated, info
