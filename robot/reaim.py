"""Smooth re-aims for following a moving joint goal on driver 1.8.3, shared by SafeArm.follow_goal, DryArm and the
simulator (rl.envs lag_mode="reaim").

The driver interpolates a command longer than 0.2 s with a quintic from where the arm is to the goal, and re-commanding
fast is unstable before driver 1.9.3, so a moving goal is followed by re-aiming a few times a second. A plain re-aim
ends at rest at its goal, so the arm brakes every time even while the goal keeps moving. A feedforward re-aim
(set_arm_positions' goal_feedforward_velocities) aims at where the goal will be when the move ends and arrives moving
with the goal's own velocity instead (plan_ff).
"""

import numpy as np

EVERY, DEADBAND = 3, 0.008  # re-aim at most every 3rd 20 Hz step (0.15 s), only when the goal moved this much (rad)
MIN_T, SPEED, ACCEL, V_MAX = 0.3, 1.0, 3.0, 3.0  # as SafeArm.follow_goal's defaults and DryArm's joint speed limit
V_FF = 0.6  # rad/s: cap on a feedforward end velocity (per joint)
FF_GAIN = 0.5  # share of the goal's velocity fed forward: full feedforward overshoots when the goal slows or turns
# (dry tests: 0.5 cut the tracking error on 10 cm/s ramps by about 20%; every gain hurt on a 0.2-1.2 Hz sweep)


def quintic(q0, v0, q1, v1, T):
    """Coefficients c[0..5] of q(s) = sum c[k] s^k from (q0, v0, no acceleration) to (q1, v1, no acceleration)."""
    d = q1 - q0
    return [q0, v0, np.zeros_like(q0), (10 * d - (6 * v0 + 4 * v1) * T) / T**3,
            (-15 * d + (8 * v0 + 7 * v1) * T) / T**4, (6 * d - 3 * (v0 + v1) * T) / T**5]


def evaluate(c, s):
    """Position, velocity and acceleration of the quintic at time s (scalar or array, then one row per time)."""
    s = np.asarray(s, dtype=float)[..., None] if np.ndim(s) else float(s)
    q = sum(c[k] * s**k for k in range(6))
    v = sum(k * c[k] * s**(k - 1) for k in range(1, 6))
    a = sum(k * (k - 1) * c[k] * s**(k - 2) for k in range(2, 6))
    return q, v, a


def reaim_time(q0, v0, q1, v1=None, speed=SPEED, accel=ACCEL, min_t=MIN_T, v_max=V_MAX):
    """Duration of one re-aim. Without an end velocity it is SafeArm's formula (a quintic from rest to rest peaks at
    1.875 dq / T in speed and 5.77 dq / T^2 in acceleration). With one, the time grows from min_t until the planned
    quintic keeps every joint under the speed cap (or the starting speed, if higher; at most half the joint's
    maximum) and the acceleration cap; None if no time up to twice the rest-to-rest one does."""
    dq = np.abs(q1 - q0)
    rest = max(min_t, 1.875 * dq.max() / speed, np.sqrt(5.77 * dq.max() / accel), *(1.875 * dq / (0.5 * v_max)))
    if v1 is None:
        return rest
    # moving ends: the rest-to-rest formula would count the cruise at the goal's speed as extra distance to cover
    # (and, with plan_ff's intercept, feed on itself), so start from the minimum and check the planned quintic
    v_lim = min(max(speed, np.abs(v0).max()), 0.5 * np.min(v_max))
    T = min_t
    while T <= 2 * rest:
        _, v, a = evaluate(quintic(q0, v0, q1, v1, T), np.linspace(0, T, 41))
        if np.abs(v).max() <= v_lim and np.abs(a).max() <= accel:
            return T
        T *= 1.1
    return None


def plan_ff(q0, v0, q_goal, v_goal, speed=SPEED, accel=ACCEL, min_t=MIN_T, v_max=V_MAX, lo=None, hi=None):
    """A feedforward re-aim at a goal moving with v_goal: aim at where the goal will be when the move ends,
    arriving with the goal's velocity (intercept; aiming at where the goal is now, with that end velocity, makes a
    stretched move dawdle). Returns (goal to command, end velocity, duration). Falls back to a plain re-aim (at
    q_goal, ending at rest) when no feedforward move fits the caps, or when the intercept point or where the arm
    would coast to in the next 0.15 s is outside the joint limits lo/hi (if given)."""
    caps, best, T = (speed, accel, min_t, v_max), None, min_t
    if np.any(v_goal):
        for _ in range(5):  # fixed point: the duration sets the intercept point, which sets the duration
            g = q_goal + v_goal * T
            T_g = reaim_time(q0, v0, g, v_goal, *caps)
            if T_g is None:
                break
            best = (g, T_g)  # T_g is checked for this g
            if abs(T_g - T) < 1e-3:
                break
            T = T_g
    if best is not None and lo is not None:
        coast = best[0] + v_goal * 0.15
        if np.any(np.minimum(best[0], coast) < lo) or np.any(np.maximum(best[0], coast) > hi):
            best = None
    if best is not None:
        return best[0], np.asarray(v_goal, dtype=float), best[1]
    return np.asarray(q_goal, dtype=float), np.zeros(6), reaim_time(q0, v0, q_goal, None, *caps)


class GoalVelocity:
    """The joint goal's velocity over the last `window` s (for a feedforward re-aim's end velocity): each joint
    capped at V_FF (scaling all joints together), and zero below `floor` rad/s so a still goal ends at rest. A jump
    (any joint moving more than `jump` rad between two updates, which smooth targets never do: a policy moves its
    target at most 1 cm per step) clears the history, so the next re-aim is a plain one instead of extrapolating
    the jump."""

    def __init__(self, window=0.15, cap=V_FF, floor=0.02, jump=0.06, gain=FF_GAIN):
        self.window, self.cap, self.floor, self.jump, self.gain = window, cap, floor, jump, gain
        self.reset()

    def reset(self):
        self.hist = []

    def update(self, t, q_goal):
        q_goal = np.array(q_goal, dtype=float)
        if self.hist and np.abs(q_goal - self.hist[-1][1]).max() > self.jump:
            self.hist = []
        self.hist.append((t, q_goal))
        while len(self.hist) > 2 and self.hist[-1][0] - self.hist[1][0] >= self.window - 1e-9:
            self.hist.pop(0)

    def get(self):
        if len(self.hist) < 2 or self.hist[-1][0] - self.hist[0][0] <= 0:
            return np.zeros(6)
        (t0, q0), (t1, q1) = self.hist[0], self.hist[-1]
        v = self.gain * (q1 - q0) / (t1 - t0)
        v[np.abs(v) < self.floor] = 0.0
        m = np.abs(v).max()
        return v * (self.cap / m) if m > self.cap else v


class Reaim:
    """The deployment path's arm motion for the simulator: the joint goal is re-aimed at most every EVERY control
    steps and only when it moved more than DEADBAND rad (with feedforward, also once more after it stops, so the arm
    ends at rest). Each re-aim is a quintic from the commanded position and velocity, as robot.controller.DryArm
    models the driver. ff=False is the plain re-aim that system ID measured on the real arm."""

    def __init__(self, ff=False, ctrl_dt=0.05, lo=None, hi=None, ff_gain=FF_GAIN):
        self.ff, self.ctrl_dt, self.lo, self.hi = ff, ctrl_dt, lo, hi  # lo/hi: joint limits, for plan_ff
        self.gv = GoalVelocity(gain=ff_gain)

    def reset(self, q):
        self.c, self.T, self.s = [np.array(q, dtype=float)] + [np.zeros(6)] * 5, 0.0, 0.0
        self.goal, self.since, self.k, self.v_end = np.array(q, dtype=float), EVERY, 0, np.zeros(6)
        self.gv.reset()

    def state(self):
        s = min(self.s, self.T)
        q = sum(self.c[k] * s**k for k in range(6))
        v = sum(k * self.c[k] * s**(k - 1) for k in range(1, 6)) if s < self.T else np.zeros(6)
        return q, v

    def aim(self, q_goal):
        """Once per control step, before its physics steps."""
        self.since += 1
        self.k += 1
        if self.ff:
            self.gv.update(self.k * self.ctrl_dt, q_goal)
        moved = np.abs(q_goal - self.goal).max() > DEADBAND
        if self.since < EVERY or not (moved or (self.ff and np.any(self.v_end))):
            return
        q0, v0 = self.state()
        if self.ff:
            g, v1, T = plan_ff(q0, v0, q_goal, self.gv.get(), lo=self.lo, hi=self.hi)
            self.c, self.v_end = quintic(q0, v0, g, v1, T), v1
        else:
            T = reaim_time(q0, v0, q_goal)
            self.c = quintic(q0, v0, q_goal, np.zeros(6), T)
        self.T, self.s, self.goal, self.since = T, 0.0, np.array(q_goal, dtype=float), 0

    def advance(self, dt):
        """One physics step: the position command."""
        self.s += dt
        return self.state()[0]
