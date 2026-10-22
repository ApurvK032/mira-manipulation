"""System identification, part 2: fit the simulator's arm lag to a recording from scripts/sysid_record.py.

The RL environments (lag_mode="target") model the arm as following its commanded joint targets through a
first-order lag with time constant tau. The real pipeline (policy targets at 20 Hz -> SafeArm follow_goal
re-aims -> the arm) may also add a pure delay. This replays the recorded targets through
  dq/dt = (q_target(t - delay) - q) / tau
for a grid of (tau, delay) and finds the pair that best matches the measured joints (RMSE over the moving
joints), per test phase and overall. Use the result to set lag_tau in training (e.g. a range around tau).

It also replays the targets through the simulator's re-aim model (robot.reaim.Reaim, lag_mode="reaim", or
"reaim_ff" for a recording made with --ff), which has no fitted parameters, and reports how closely that model
matches the arm (plus the best extra first-order motor lag on top of it).

Usage:  python -m rl.sysid_fit runs/sysid/sysid_<time>.npz
"""

import argparse

import numpy as np

from robot.reaim import Reaim
from robot.stream import Streamer


def simulate(t, u, tau, delay, q0):
    """First-order lag with a pure delay, on the recorded time grid."""
    q = np.empty_like(u)
    q[0] = q0
    td = t - delay
    idx = np.clip(np.searchsorted(t, td, side="right") - 1, 0, len(t) - 1)
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        a = 1.0 if tau <= 1e-4 else min(1.0, dt / tau)
        q[i] = q[i - 1] + a * (u[idx[i]] - q[i - 1])
    return q


def fit(t, u, y, joints):
    best = None
    for tau in np.arange(0.0, 0.41, 0.01):
        for delay in np.arange(0.0, 0.21, 0.01):
            q = simulate(t, u, tau, delay, y[0])
            rmse = float(np.sqrt(np.mean((q[:, joints] - y[:, joints]) ** 2)))
            if best is None or rmse < best[0]:
                best = (rmse, tau, delay)
    return best


def replay_reaim(t, u, q0, ff, dt=0.002, ctrl_dt=0.05, ff_gain=1.0):
    """The joints the re-aim model commands for the logged targets (policy steps every ctrl_dt from t[0], each
    taking the newest logged target), sampled at the log's times."""
    r = Reaim(ff=ff, ctrl_dt=ctrl_dt, ff_gain=ff_gain)
    r.reset(q0)
    tt = np.arange(t[0], t[-1], dt)
    q, next_step = np.empty((len(tt), u.shape[1])), t[0]
    for i, ti in enumerate(tt):
        if ti >= next_step:
            r.aim(u[np.searchsorted(t, ti, side="right") - 1])
            next_step += ctrl_dt
        q[i] = r.advance(dt)
    return np.array([np.interp(t, tt, q[:, j]) for j in range(u.shape[1])]).T


def replay_stream(t, u, q0, rate, dt=0.002, ctrl_dt=0.05):
    """The joints SafeArm.stream_tick commands for the logged targets (robot.stream.Streamer at `rate`, default
    limits), with the driver's straight line between points, sampled at the log's times."""
    st = Streamer(rate)
    st.reset(q0)
    tt = np.arange(t[0], t[-1], dt)
    q, nxt, prev, cur, t_tick = np.empty((len(tt), u.shape[1])), t[0], q0.copy(), q0.copy(), t[0]
    for i, ti in enumerate(tt):
        if ti >= nxt:
            prev, cur, t_tick = cur, st.step(u[np.searchsorted(t, ti, side="right") - 1])[0], ti
            nxt += st.dt
        q[i] = prev + (cur - prev) * min(1.0, (ti - t_tick) / st.dt)
    return np.array([np.interp(t, tt, q[:, j]) for j in range(u.shape[1])]).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    args = ap.parse_args()
    d = np.load(args.log)
    t, u, y, phase = d["t"], d["q_target"], d["q_meas"], d["phase"]
    moving = np.where(np.ptp(u, axis=0) > 0.01)[0]  # joints the tests actually moved
    print(f"{len(t)} samples over {t[-1] - t[0]:.0f} s at {len(t) / (t[-1] - t[0]):.0f} Hz; fitting joints {moving.tolist()}")
    zero_lag = float(np.sqrt(np.mean((u[:, moving] - y[:, moving]) ** 2)))
    print(f"if the arm followed instantly: RMSE {np.degrees(zero_lag):.2f} deg")
    for name, sel in (("steps", phase == 1), ("sweep", phase == 2), ("all", np.ones_like(phase, bool))):
        if sel.sum() < 50:
            continue
        rmse, tau, delay = fit(t[sel], u[sel], y[sel], moving)
        print(f"{name:5s}: tau {tau:.2f} s, delay {delay:.2f} s -> RMSE {np.degrees(rmse):.2f} deg")
    ff = bool(d["ff"]) if "ff" in d.files else False
    gain = float(d["ff_gain"]) if "ff_gain" in d.files else 1.0  # recordings before the gain existed used 1.0
    rate = float(d["stream"]) if "stream" in d.files else 0.0
    if rate:
        qr = replay_stream(t, u, y[0], rate)
        print(f"streaming model ({rate:g} Hz, robot.stream defaults, no fitted parameters):")
    else:
        qr = replay_reaim(t, u, y[0], ff, ff_gain=gain)
        print(f"re-aim model ({f'reaim_ff, gain {gain:g}' if ff else 'reaim'}, no fitted parameters):")
    for name, sel in (("steps", phase == 1), ("sweep", phase == 2), ("all", np.ones_like(phase, bool))):
        if sel.sum() < 50:
            continue
        rm = float(np.sqrt(np.mean((qr[sel][:, moving] - y[sel][:, moving]) ** 2)))
        extra = min((float(np.sqrt(np.mean((simulate(t[sel], qr[sel], tau, 0.0, y[sel][0])[:, moving]
                                             - y[sel][:, moving]) ** 2))), tau) for tau in np.arange(0, 0.11, 0.01))
        print(f"{name:5s}: RMSE {np.degrees(rm):.2f} deg; with extra motor lag {extra[1]:.2f} s: {np.degrees(extra[0]):.2f} deg")
    if len(d["events"]):
        for kind, (t_done, t_cmd, g) in zip(d["event_kinds"], d["events"]):
            print(f"gripper {kind}: {t_done - t_cmd:.2f} s, fingers at {g * 1000:.1f} mm")
    print("set lag_tau in training to a range around the fitted tau (and add the delay as observation/action delay)")


if __name__ == "__main__":
    main()
