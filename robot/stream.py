"""Streaming joint targets to the arm at a high rate (driver 1.9.3+), shared by SafeArm, DryArm and the simulator
(rl.envs lag_mode="stream").

Driver 1.9.3 fixed the unstable motion of goals re-sent faster than the controller's trajectory rate, so a goal can
be sent every few milliseconds and consecutive trajectories join up. Sending a policy's raw targets that way would
still jerk the arm: each command shorter than 0.2 s is a straight-line segment, and a target that jumps asks for an
instant change of speed. So the computer runs a small trajectory generator at the stream rate. Each tick the
commanded joints move toward the goal at a limited speed and acceleration, all joints in step (the whole velocity
vector is scaled, so the direction of motion is kept), braking in time to stop at the goal. The driver receives the
next point with its velocity as feed-forward.
"""

import numpy as np

RATE = 100.0  # Hz
SPEED = 1.0  # rad/s, per joint (SafeArm's max_joint_speed)
ACCEL = 4.0  # rad/s^2, per joint
TAU = 0.06  # s: close to the goal the approach is first order with this time constant


class Streamer:
    def __init__(self, rate=RATE, speed=SPEED, accel=ACCEL, tau=TAU):
        self.dt, self.speed, self.accel, self.tau = 1.0 / rate, speed, accel, tau
        self.q = self.v = None

    def reset(self, q, v=None):
        self.q = np.array(q, dtype=float)
        self.v = np.zeros_like(self.q) if v is None else np.array(v, dtype=float)

    def step(self, goal):
        """One tick toward goal: returns the joints and joint velocities to command."""
        err = np.asarray(goal, dtype=float) - self.q
        v_des = err / self.tau
        a = np.abs(err)
        # scale the whole vector: under the speed cap, and slow enough for every joint to brake to a stop at the goal
        brake = np.sqrt(2 * self.accel * a)
        ratio = max(1.0, np.abs(v_des).max() / self.speed,
                    *(np.abs(v_des)[a > 1e-9] / brake[a > 1e-9]) if np.any(a > 1e-9) else [1.0])
        v_des = v_des / ratio
        dv = v_des - self.v
        dv = dv / max(1.0, np.abs(dv).max() / (self.accel * self.dt))
        self.v = self.v + dv
        self.q = self.q + self.v * self.dt
        return self.q.copy(), self.v.copy()
