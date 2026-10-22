"""Stack red on green while a person interferes: a Gymnasium environment for the WidowX AI in MuJoCo.

The full task the scripted controller (robot.controller) does today, as an RL problem: pick the red cube
(even while a hand slides it around), carry it, follow the green cube if it is moved while red is held, set
red down on it, release, and leave a standing stack. Success: red resting on green (centred within 10 mm,
both upright), the gripper open and clear of it, for 0.5 s.

Episodes start from four families (as in OmniReset: practise the end of the task as well as the start):
  0 "table"   red and green on the table, the arm hovering somewhere; red may be slid by a hand
  1 "carry"   red already held at carry height somewhere; green on the table
  2 "near"    red held a few cm above green, off-centre
  3 "place"   red held just above green, nearly aligned
  4 "grasp"   the gripper open just above red on the table, roughly aligned (practises the grasp itself)
A hand may slide green while red is held (and lets go after a while). The arm's joint targets are low-pass
filtered with a random time constant (0-0.15 s), standing in for the real arm's slower response.

Control (20 Hz) and action space as in rl.envs.moving_cube. Observation (41): fingertip position, gripper
angle (sin/cos 2x); for red and for green: position relative to the fingertips, velocity, angle relative
to the gripper (sin/cos 4x); red's height above the table; red relative to its place on green; gripper
opening; joint angles and speeds; previous action. Cube readings are delayed one or two steps and noisy.
"""

import gymnasium as gym
import mujoco
import numpy as np

from .variations import Variations
from robot.stream import Streamer
from .moving_cube import CLOSED, CTRL_DT, GRASP_Z, HALF, MODEL, OPEN, RING, TIP_BOX, Reaim, down_rotation, wrap

CARRY_Z = 0.12  # fingertip height while carrying (held cube bottom ~8.5 cm up)
STACK_STEPS = 10
FAMILIES = ("table", "carry", "near", "place", "grasp", "handover")  # handover: states recorded from the chain


class StackCubes(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps=300, family_probs=(0.4, 0.25, 0.2, 0.15), push_red_prob=0.6,
                 push_green_prob=0.5, max_speed=0.10, lag_tau=(0.0, 0.15), obs_noise=0.0015, yaw_noise_deg=2.0,
                 delay_steps=(1, 2), binary_gripper=False, release_needs_rise=True, place_bonus=0.0,
                 stack_reward=2.0, knock_penalty=5.0, premature_release_penalty=0.0, gripper_hysteresis=0.0,
                 grasp_bonus=0.0, lag_mode="measured", green_shove_penalty=0.0, obs_dropout=0.0, leash=0.03, handover_file=None, calib_error=(0.0, 0.0),
                 obs_bias=0.0, friction_scale=None, mass_scale=None, stream_accel=(3.0, 6.0),
                 stream_tau=(0.04, 0.1), gripper_speed=None, gripper_speed_scale=None, seed=None):
        """The defaults are the first version (run stack1). Later runs pass: binary_gripper (action > 0 closes,
        otherwise open), release_needs_rise=False (open is enough to count as released), place_bonus (per
        step while holding red within 8 mm of its place on green), a larger stack_reward and a smaller
        knock_penalty, and family_probs weighted toward the end of the task. Also premature_release_penalty
        (charged when the gripper opens on red away from its place on green) and gripper_hysteresis (with the
        binary gripper, switching needs |action| above it, so exploration noise near 0 cannot drop the cube), and
        grasp_bonus (once per episode, the first time red is picked up), and lag_mode="target" (see
        rl.envs.moving_cube: with "measured", the default and the stack1-4 runs, the lag makes the arm slow,
        not just late). green_shove_penalty: per step while green moves more than 3 mm without a hand pushing it
        (teaches setting red down gently). obs_dropout: per step and cube, the chance that its reading freezes for
        1-6 steps, as when a hand or the arm hides it from the real camera. lag_mode="reaim": the deployment path's
        re-aims instead of a first-order lag (robot.reaim.Reaim, "reaim_ff" with feedforward; lag_tau unused, delay_steps = perception). leash: how far (m) the target may run ahead of the fingertips. handover_file: npz(s) from
        rl/record_handover.py (comma-separated) for the "handover" family: the full simulator state at the moment the
        chain (grab policy, scripted lift) hands red to the placement policy, with the steps already used.
        calib_error, obs_bias, friction_scale, mass_scale: per-episode sim-to-real variations (rl.envs.variations)."""
        # gripper_speed: (closing, opening) m/s per finger, opt-in; None sets the gripper target at once, as before.
        # Real WidowX AI gripper, firmware 1.9.4: closes at about 0.048 m/s, opens at about 0.125 m/s (real-arm log,
        # 2026-10-20); the simulated one otherwise shuts in about 0.09 s.
        # gripper_speed_scale=(lo, hi): per episode, both speeds times a factor drawn from this range (training margin).
        self.gripper_speed = tuple(gripper_speed) if gripper_speed is not None else None
        self.gripper_speed_base, self.gripper_speed_scale = self.gripper_speed, gripper_speed_scale
        super().__init__()
        self.binary_gripper, self.release_needs_rise = binary_gripper, release_needs_rise
        self.place_bonus, self.stack_reward, self.knock_penalty = place_bonus, stack_reward, knock_penalty
        self.premature_release_penalty, self.gripper_hysteresis = premature_release_penalty, gripper_hysteresis
        probs = list(family_probs) + [0.0] * (len(FAMILIES) - len(family_probs))  # older runs gave 4 families
        self.max_steps, self.family_probs = max_steps, np.array(probs) / np.sum(probs)
        self.grasp_bonus, self.target_mode = grasp_bonus, lag_mode in ("target", "reaim", "reaim_ff", "stream")
        self.reaim = Reaim(ff=lag_mode == "reaim_ff") if lag_mode in ("reaim", "reaim_ff") else None  # robot.reaim
        # "stream": the driver 1.9.3 streaming path (robot.stream.Streamer at 100 Hz), with per-episode limits
        self.stream = Streamer() if lag_mode == "stream" else None
        self.stream_accel, self.stream_tau = stream_accel, stream_tau
        self.green_shove_penalty, self.obs_dropout, self.leash = green_shove_penalty, obs_dropout, leash
        self.handovers = None
        if handover_file:
            parts = [np.load(f) for f in str(handover_file).split(",")]
            self.handovers = {k: np.concatenate([d[k] for d in parts]) for k in ("qpos", "qvel", "ctrl", "steps")}
        self.push_red_prob, self.push_green_prob, self.max_speed = push_red_prob, push_green_prob, max_speed
        self.lag_tau, self.obs_noise, self.yaw_noise = lag_tau, obs_noise, np.radians(yaw_noise_deg)
        self.delay_steps = delay_steps
        spec = mujoco.MjSpec.from_file(str(MODEL))
        w = spec.worldbody
        w.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.1], friction=[0.8, 0.005, 0.0001])
        for name, pos, rgba in (("red", [0.30, -0.08, HALF], [1, 0.1, 0.1, 1]), ("green", [0.26, 0.08, HALF], [0.1, 0.8, 0.1, 1])):
            b = w.add_body(name=name, pos=pos)
            b.add_freejoint(name=name + "_free")
            b.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[HALF] * 3, mass=0.05,
                       friction=[1.0, 0.005, 0.0001], rgba=rgba)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
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
        self.finger_r = self.m.joint("right_carriage_joint").qposadr[0]
        self.adr = {n: (self.m.joint(n + "_free").qposadr[0], self.m.joint(n + "_free").dofadr[0]) for n in ("red", "green")}
        self.var = Variations(self.m, "table", {"red": "red", "green": "green"}, calib_error, obs_bias, friction_scale,
                              mass_scale)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (5,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (41,), np.float32)
        self._seed = seed

    # ----- state helpers -----
    def _tip(self, d=None):
        d = d or self.d
        return d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()

    @staticmethod
    def _grip_yaw(R):
        return float(np.arctan2(R[1, 1], R[0, 1]))

    def _cube(self, n):
        q, _ = self.adr[n]
        p = self.d.qpos[q:q + 3].copy()
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, self.d.qpos[q + 3:q + 7])
        R = R.reshape(3, 3)
        a = min((R[:, i] for i in range(3)), key=lambda v: abs(v[2]))
        tilt = np.degrees(np.arccos(np.clip(max(abs(R[2, i]) for i in range(3)), 0, 1)))
        return p, float(np.arctan2(a[1], a[0])), float(tilt)

    def _set_cube(self, n, p, yaw):
        q, v = self.adr[n]
        self.d.qpos[q:q + 3] = p
        self.d.qpos[q + 3:q + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.d.qvel[v:v + 6] = 0

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

    def _place_arm(self, tip_goal, yaw):
        az = np.arctan2(tip_goal[1], tip_goal[0])
        q = np.array([az, 1.0, 1.0, 1.0, 0.0, 0.0])
        for _ in range(15):
            q = self._ik(q, tip_goal, down_rotation(yaw), iters=10)
        self.d.qpos[self.qadr] = q
        self.d.ctrl[:6] = q
        self.q_cmd = q.copy()

    def _clip_tip(self, p):
        r = np.clip(np.hypot(p[0], p[1]), *TIP_BOX["r"])
        a = np.clip(np.arctan2(p[1], p[0]), -np.arccos(0.1 / r), np.arccos(0.1 / r))
        return np.array([r * np.cos(a), r * np.sin(a), np.clip(p[2], *TIP_BOX["z"])])

    def _ring_point(self):
        r, a = self.np_random.uniform(*RING), self.np_random.uniform(-1.0, 1.0)
        return np.array([r * np.cos(a), r * np.sin(a)])

    def _apart_points(self, min_sep=0.08):
        while True:
            a, b = self._ring_point(), self._ring_point()
            if np.linalg.norm(a - b) >= min_sep:
                return a, b

    def _red_grasped(self):
        tip, _ = self._tip()
        p, _, _ = self._cube("red")
        between = np.linalg.norm(tip[:2] - p[:2]) < 0.015 and abs(tip[2] - (p[2] + GRASP_Z)) < 0.015
        return between and 0.015 < self.d.qpos[self.finger] < 0.031

    # ----- hands sliding cubes -----
    def _new_hand(self, active, start, duration):
        return dict(active=active, start=start, end=start + duration, seg_end=0.0, speed=0.0,
                    heading=self.np_random.uniform(-np.pi, np.pi), yaw_rate=0.0)

    def _hand_velocity(self, n, h):
        t = self.steps * CTRL_DT
        if not h["active"] or t < h["start"]:
            return None
        if t > h["end"]:
            h["active"] = False
            return None
        if t > h["seg_end"]:
            h["seg_end"] = t + self.np_random.uniform(0.4, 1.5)
            h["speed"] = 0.0 if self.np_random.random() < 0.15 else self.np_random.uniform(0.02, self.max_speed)
            h["heading"] += self.np_random.normal(0, 1.2)
            h["yaw_rate"] = self.np_random.normal(0, 0.6)
        p, _, _ = self._cube(n)
        other, _, _ = self._cube("green" if n == "red" else "red")
        r, a = np.hypot(p[0], p[1]), np.arctan2(p[1], p[0])
        heading = h["heading"]
        if r < RING[0] + 0.01 or r > RING[1] - 0.01 or abs(a) > 0.95:  # back toward the middle of the ring
            mid = np.array([0.27, 0.0]) - p[:2]
            heading = h["heading"] = float(np.arctan2(mid[1], mid[0]))
        if other[2] < 2 * HALF and np.linalg.norm(other[:2] - p[:2]) < 0.08:  # do not shove it into the other cube
            away = p[:2] - other[:2]
            heading = h["heading"] = float(np.arctan2(away[1], away[0]))
        return h["speed"] * np.array([np.cos(heading), np.sin(heading)]), h["yaw_rate"]

    # ----- observation -----
    def _record(self):
        for n in ("red", "green"):
            if self.hist[n] and self.stale[n] > 0:  # reading frozen (hidden from the camera)
                self.stale[n] -= 1
                self.hist[n].append(self.hist[n][-1])
            else:
                if self.hist[n] and self.np_random.random() < self.obs_dropout:
                    self.stale[n] = int(self.np_random.integers(0, 6))
                    self.hist[n].append(self.hist[n][-1])
                else:
                    p, yaw = self.var.reading(n, *self._cube(n)[:2])
                    self.hist[n].append((p + self.np_random.normal(0, self.obs_noise, 3), yaw + self.np_random.normal(0, self.yaw_noise)))
            self.hist[n] = self.hist[n][-max(6, self.delay + 2):]

    def _obs(self):
        tip, R = self._tip()
        gy = self._grip_yaw(R)
        k = self.delay
        parts = [tip, [np.sin(2 * gy), np.cos(2 * gy)]]
        seen = {}
        for n in ("red", "green"):
            p, yaw = self.hist[n][-1 - k]
            p_prev, _ = self.hist[n][-2 - k]
            dyaw = wrap(yaw - gy)
            seen[n] = p
            parts += [p - tip, (p[:2] - p_prev[:2]) / CTRL_DT, [np.sin(4 * dyaw), np.cos(4 * dyaw)]]
        parts += [[seen["red"][2] - HALF], seen["red"] - (seen["green"] + [0, 0, 2 * HALF]),
                  [self.d.qpos[self.finger] / OPEN], self.d.qpos[self.qadr], 0.1 * self.d.qvel[self.dadr], self.prev_action]
        return np.concatenate(parts).astype(np.float32)

    # ----- gym API -----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed if seed is not None else self._seed)
        self._seed = None
        self.var.sample(self.np_random)
        for _ in range(10):
            fam = int(self.np_random.choice(len(FAMILIES), p=self.family_probs)) if not options else options.get("family", 0)
            if self._reset_family(fam):
                break
        else:
            self._reset_family(0)
            fam = 0
        self.family = FAMILIES[fam]
        self.grip_goal = float(self.d.ctrl[6])  # the commanded gripper target (ctrl[6] lags it with gripper_speed)
        if self.gripper_speed_base is not None and self.gripper_speed_scale is not None:  # drawn only when set
            k = self.np_random.uniform(*self.gripper_speed_scale)
            self.gripper_speed = (self.gripper_speed_base[0] * k, self.gripper_speed_base[1] * k)
        self.tau = self.np_random.uniform(*self.lag_tau)
        if self.reaim:
            self.reaim.reset(self.q_cmd)
        if self.stream:
            self.stream.accel = self.np_random.uniform(*self.stream_accel)
            self.stream.tau = self.np_random.uniform(*self.stream_tau)
            self.stream.reset(self.q_cmd)
            self.s_prev, self.s_next, self.stream_goal = (np.array(self.q_cmd, dtype=float) for _ in range(3))
        self.delay = int(self.np_random.integers(self.delay_steps[0], self.delay_steps[1] + 1))
        self.prev_action = np.zeros(5)
        self.steps, self.stack_steps = 0, 0
        if self.family == "handover":
            self.steps = self.handover_steps  # the time the chain had already used
        self.was_grasped = self.family not in ("table", "grasp")
        self.picked_up = self.was_grasped
        tip, R = self._tip()
        self.tip_target, self.yaw_target = tip.copy(), self._grip_yaw(R)
        self.hist, self.stale = {"red": [], "green": []}, {"red": 0, "green": 0}
        for _ in range(max(4, self.delay + 2)):
            self._record()
        self.green_last = self._cube("green")[0]
        return self._obs(), {"family": self.family}

    def _reset_family(self, fam):
        mujoco.mj_resetData(self.m, self.d)
        yaw = self.np_random.uniform(-np.pi / 2, np.pi / 2)
        red_hand = self._new_hand(False, 0.0, 0.0)
        green_hand = self._new_hand(False, 0.0, 0.0)
        if fam == 5:
            if self.handovers is None:
                raise ValueError('family "handover" needs handover_file')
            i = int(self.np_random.integers(len(self.handovers["steps"])))
            self.d.qpos[:], self.d.qvel[:] = self.handovers["qpos"][i], self.handovers["qvel"][i]
            self.d.ctrl[:] = self.handovers["ctrl"][i]
            self.q_cmd = self.d.ctrl[:6].copy()
            self.handover_steps = int(self.handovers["steps"][i])
            mujoco.mj_forward(self.m, self.d)
            if not self._red_grasped():
                return False
            # as in the carry family, timed from the hand-over (the episode clock starts at handover_steps)
            green_hand = self._new_hand(self.np_random.random() < self.push_green_prob,
                                        self.handover_steps * CTRL_DT + self.np_random.uniform(0.0, 2.0),
                                        self.np_random.uniform(0.5, 2.5))
        elif fam == 4:
            rp, gp = self._apart_points()
            ryaw = self.np_random.uniform(-np.pi, np.pi)
            self._set_cube("red", [rp[0], rp[1], HALF], ryaw)
            self._set_cube("green", [gp[0], gp[1], HALF], self.np_random.uniform(-np.pi, np.pi))
            off = self.np_random.uniform(-0.008, 0.008, 2)
            yaw = ryaw + self.np_random.normal(0, np.radians(8))
            self._place_arm(np.array([rp[0] + off[0], rp[1] + off[1], HALF + GRASP_Z + self.np_random.uniform(0.01, 0.03)]), yaw)
            self.d.qpos[self.finger] = self.d.qpos[self.finger_r] = OPEN
            self.d.ctrl[6] = OPEN
        elif fam == 0:
            rp, gp = self._apart_points()
            self._set_cube("red", [rp[0], rp[1], HALF], self.np_random.uniform(-np.pi, np.pi))
            self._set_cube("green", [gp[0], gp[1], HALF], self.np_random.uniform(-np.pi, np.pi))
            self._place_arm(np.append(self._ring_point(), self.np_random.uniform(0.08, 0.14)), yaw)
            self.d.qpos[self.finger] = self.d.qpos[self.finger_r] = OPEN
            self.d.ctrl[6] = OPEN
            red_hand = self._new_hand(self.np_random.random() < self.push_red_prob, 0.0, self.np_random.uniform(2.0, 7.0))
        else:
            gp = self._ring_point()
            gyaw = self.np_random.uniform(-np.pi, np.pi)
            self._set_cube("green", [gp[0], gp[1], HALF], gyaw)
            if fam == 1:
                while True:
                    tp = self._ring_point()
                    if np.linalg.norm(tp - gp) >= 0.08:
                        break
                tip = np.array([tp[0], tp[1], CARRY_Z])
            elif fam == 2:
                off = self.np_random.uniform(-0.04, 0.04, 2)
                tip = np.array([gp[0] + off[0], gp[1] + off[1], 3 * HALF + self.np_random.uniform(0.01, 0.04) + GRASP_Z])
            else:
                off = self.np_random.uniform(-0.005, 0.005, 2)
                yaw = gyaw + self.np_random.normal(0, np.radians(4))
                tip = np.array([gp[0] + off[0], gp[1] + off[1], 3 * HALF + self.np_random.uniform(0.002, 0.01) + GRASP_Z])
            self._place_arm(tip, yaw)
            real_tip, R = self._tip_after_kinematics()
            self._set_cube("red", real_tip - [0, 0, GRASP_Z], self._grip_yaw(R))
            self.d.qpos[self.finger] = self.d.qpos[self.finger_r] = HALF
            self.d.ctrl[6] = CLOSED
            # settle the grasp for 0.2 s with the arm held; reject the start state if the cube slipped out
            for _ in range(int(0.2 / self.m.opt.timestep)):
                mujoco.mj_step(self.m, self.d)
            if not self._red_grasped():
                return False
            green_hand = self._new_hand(self.np_random.random() < self.push_green_prob,
                                        self.np_random.uniform(0.0, 2.0), self.np_random.uniform(0.5, 2.5))
        mujoco.mj_forward(self.m, self.d)
        self.hands = {"red": red_hand, "green": green_hand}
        self.green_hand_armed = fam in (0, 4) and self.np_random.random() < self.push_green_prob
        return True

    def _tip_after_kinematics(self):
        mujoco.mj_kinematics(self.m, self.d)
        return self._tip()

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        tip, R = self._tip()
        if self.target_mode:
            target = self._clip_tip(self.tip_target + 0.01 * a[:3])
            lead = target - tip
            if np.linalg.norm(lead) > self.leash:
                target = tip + lead * self.leash / np.linalg.norm(lead)
            gy = self._grip_yaw(R)
            yaw = gy + np.clip(wrap(self.yaw_target + np.radians(5) * a[3] - gy), -np.radians(15), np.radians(15))
            self.tip_target, self.yaw_target = target, yaw
            seed = self.reaim.goal if self.reaim else (self.stream_goal if self.stream else self.q_cmd)  # last goal, as
            # rl/deploy.py does
            q_goal = self._ik(seed, target, down_rotation(yaw), iters=3)
        else:
            target = self._clip_tip(tip + 0.01 * a[:3])
            yaw = self._grip_yaw(R) + np.radians(5) * a[3]
            q_goal = self._ik(self.d.qpos[self.qadr], target, down_rotation(yaw))
        was_closed = self.grip_goal < OPEN / 2
        if self.binary_gripper:
            h = self.gripper_hysteresis
            closed = a[4] > h if not was_closed else a[4] > -h
            self.grip_goal = CLOSED if closed else OPEN
        else:
            self.grip_goal = OPEN * (1 - (a[4] + 1) / 2)
        if self.gripper_speed is None:
            self.d.ctrl[6] = self.grip_goal
        held_before = self._red_grasped()
        opening_now = was_closed and self.grip_goal >= OPEN / 2
        rp0, gp0 = self._cube("red")[0], self._cube("green")[0]  # where red was when the gripper opened
        pushes = {n: self._hand_velocity(n, h) for n, h in self.hands.items()}
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
                self.q_cmd += alpha * (q_goal - self.q_cmd)  # the arm's slower response
            self.d.ctrl[:6] = self.q_cmd
            if self.gripper_speed is not None:  # the real gripper's speed (opt-in): the actuator target follows
                g = self.d.ctrl[6]                # the command at most this fast
                v = self.gripper_speed[0] if self.grip_goal < g else self.gripper_speed[1]
                step = v * self.m.opt.timestep
                self.d.ctrl[6] = g + min(max(self.grip_goal - g, -step), step)
            for n, pv in pushes.items():
                if pv is not None:
                    _, v = self.adr[n]
                    self.d.qvel[v:v + 2] = pv[0]
                    self.d.qvel[v + 5] = pv[1]
            mujoco.mj_step(self.m, self.d)
        self.steps += 1
        grasped = self._red_grasped()
        # the hand on red lets go once the gripper catches up with it
        rh = self.hands["red"]
        if rh["active"]:
            rp, _, _ = self._cube("red")
            tip, _ = self._tip()
            if np.linalg.norm(tip[:2] - rp[:2]) < 0.02 and tip[2] < rp[2] + 0.045:
                rh["active"] = False
        # in table starts, a hand may slide green some time after red is picked up
        if grasped and not self.was_grasped and self.green_hand_armed:
            self.hands["green"] = self._new_hand(True, self.steps * CTRL_DT + self.np_random.uniform(0.0, 1.5),
                                                 self.np_random.uniform(0.5, 2.5))
            self.green_hand_armed = False
        self._record()

        # ----- reward -----
        tip, R = self._tip()
        gy = self._grip_yaw(R)
        rp, ryaw, rtilt = self._cube("red")
        gp, gyaw, gtilt = self._cube("green")
        opening = self.d.qpos[self.finger]
        on_green = (np.linalg.norm(rp[:2] - gp[:2]) < 0.010 and 0.046 < rp[2] - gp[2] < 0.056
                    and rtilt < 8 and gtilt < 8)
        released = opening > 0.031 and (tip[2] > rp[2] + GRASP_Z + 0.008 or not self.release_needs_rise)
        info = {"success": False, "family": self.family, "grasped": bool(grasped)}
        if on_green and released:
            r = self.stack_reward
            self.stack_steps += 1
        else:
            self.stack_steps = 0
            if not grasped:
                d = np.linalg.norm(tip - (rp + [0, 0, GRASP_Z]))
                r = 0.5 * (1 - np.tanh(8 * d)) + 0.1 * np.cos(4 * wrap(ryaw - gy)) * (d < 0.05)
            else:
                d_place = np.linalg.norm(rp - (gp + [0, 0, 2 * HALF]))
                r = 1.0 + 0.8 * (1 - np.tanh(8 * d_place)) + 0.1 * np.cos(4 * wrap(ryaw - gyaw)) * (d_place < 0.05)
                r += self.place_bonus * (np.linalg.norm(rp[:2] - gp[:2]) < 0.008 and rp[2] - gp[2] < 0.06)
        r -= 0.01 + 0.01 * float(np.sum(a[:4] ** 2)) + 0.02 * float(np.sum((a - self.prev_action) ** 2))
        if self.green_shove_penalty and not self.hands["green"]["active"] and np.linalg.norm(gp[:2] - self.green_last[:2]) > 0.003:
            r -= self.green_shove_penalty  # shoved green (e.g. coming down on its edge)
            info["shoved_green"] = True
        self.green_last = gp
        if held_before and opening_now:  # letting go: only right above its place on green
            off = np.linalg.norm(rp0[:2] - gp0[:2]) > 0.008 or rp0[2] - gp0[2] > 2 * HALF + 0.015
            if off:
                r -= self.premature_release_penalty
                info["premature_release"] = True
        if self.was_grasped and not grasped and not on_green and rp[2] < HALF + 0.01:
            r -= 1.0  # dropped it on the table (it can still be picked up again)
            info["dropped"] = True
        if grasped and not self.picked_up:
            r += self.grasp_bonus
            self.picked_up = True
        self.was_grasped = grasped or (self.was_grasped and on_green)
        terminated = False
        if self.stack_steps >= STACK_STEPS:
            r += 10.0 + 2.4 * (self.max_steps - self.steps)
            terminated, info["success"] = True, True
        elif (rtilt > 30 and not grasped) or gtilt > 30:
            r -= self.knock_penalty
            terminated, info["knocked"] = True, True
        elif not (RING[0] - 0.04 < np.hypot(gp[0], gp[1]) < RING[1] + 0.04) or \
                (not grasped and not (RING[0] - 0.04 < np.hypot(rp[0], rp[1]) < RING[1] + 0.04)):
            r -= 2.0
            terminated, info["out"] = True, True
        truncated = self.steps >= self.max_steps
        self.prev_action = a
        return self._obs(), float(r), terminated, truncated, info
