"""Sanity check for rl.envs.MovingCubeGrasp: a hand-written policy that uses the true cube state.

If this cannot grasp and lift a moving cube, the simulation (contacts, gripper, IK, workspace) is broken and
RL would only learn the bug. Also reports simulation speed.

Usage:  python -m rl.scripted_check [--episodes 50]
"""

import argparse
import time

import numpy as np

from rl.envs.moving_cube import GRASP_Z, HALF, MovingCubeGrasp, wrap


def scripted_action(env, noisy=False):
    tip, R = env._tip()
    cp, cyaw, _ = env._cube()
    v = env.d.qvel[env.cv:env.cv + 2]
    if noisy:  # the same delayed, noisy cube readings the RL policy gets
        cp, cyaw = env.cube_hist[-1 - env.delay]
        prev, _ = env.cube_hist[-2 - env.delay]
        v = (cp[:2] - prev[:2]) / 0.05
    lead = cp[:2] + 0.15 * v  # aim a little ahead of a moving cube
    dyaw = wrap(cyaw - env._grip_yaw(R))
    dyaw = (dyaw + np.pi / 4) % (np.pi / 2) - np.pi / 4  # nearest face alignment
    xy_err = lead - tip[:2]
    close = np.linalg.norm(cp[:2] - tip[:2]) < 0.008 and tip[2] < cp[2] + GRASP_Z + 0.006
    st = env.script_state
    if st == "approach":
        z_goal = cp[2] + 0.06
        if np.linalg.norm(xy_err) < 0.01 and abs(dyaw) < np.radians(8):
            st = "descend"
    if st == "descend":
        z_goal = cp[2] + GRASP_Z
        if close:
            st, env.close_t = "grasp", 0
    if st == "grasp":
        z_goal = cp[2] + GRASP_Z
        env.close_t += 1
        if env.close_t > 6:
            st = "lift"
    if st == "lift":
        z_goal = HALF + GRASP_Z + 0.07
    env.script_state = st
    dz = z_goal - tip[2] if st != "lift" else 0.07
    a = np.zeros(5)
    a[:2] = np.clip(xy_err / 0.01, -1, 1) if st in ("approach", "descend") else 0.0
    a[2] = np.clip(dz / 0.01, -1, 1)
    a[3] = np.clip(dyaw / np.radians(5), -1, 1) if st in ("approach", "descend") else 0.0
    a[4] = 1.0 if st in ("grasp", "lift") else -1.0
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--noisy", action="store_true", help="use the delayed, noisy cube readings (as the RL policy sees)")
    args = ap.parse_args()
    env = MovingCubeGrasp(seed=0)
    stats = {"success": 0, "knocked": 0, "out": 0, "timeout": 0}
    steps_to_success, t0, total_steps = [], time.time(), 0
    for ep in range(args.episodes):
        env.reset(seed=ep)
        env.script_state, env.close_t = "approach", 0
        for t in range(env.max_steps):
            _, r, term, trunc, info = env.step(scripted_action(env, args.noisy))
            total_steps += 1
            if term or trunc:
                if info.get("success"):
                    stats["success"] += 1
                    steps_to_success.append(t + 1)
                elif info.get("knocked"):
                    stats["knocked"] += 1
                elif info.get("out"):
                    stats["out"] += 1
                else:
                    stats["timeout"] += 1
                break
    dt = time.time() - t0
    print(f"scripted policy ({'noisy, delayed readings' if args.noisy else 'true cube state'}) over {args.episodes} episodes: {stats}")
    if steps_to_success:
        print(f"time to success: median {np.median(steps_to_success) * 0.05:.1f} s")
    print(f"speed: {total_steps / dt:.0f} control steps/s on one core ({total_steps / dt * env.n_sub:.0f} physics steps/s)")


if __name__ == "__main__":
    main()
