"""Sanity check for rl.envs.StackCubes: a hand-written stacking policy that uses the true cube states.

It picks red (following it if it is slid), carries it at carry height over green (following green if it
is moved), lowers, releases and backs off. If this cannot stack in the simulation, RL would only learn a
bug. Per start family success rates, plus speed.

Usage:  python -m rl.scripted_stack [--episodes 80]
"""

import argparse
import collections
import json
import time

import numpy as np

from rl.envs.moving_cube import GRASP_Z, HALF, wrap
from rl.envs.stack_cubes import CARRY_Z, StackCubes


def face(dyaw):
    return (dyaw + np.pi / 4) % (np.pi / 2) - np.pi / 4  # nearest 90-deg alignment


def act(env):
    tip, R = env._tip()
    gy = env._grip_yaw(R)
    rp, ryaw, _ = env._cube("red")
    gp, gyaw, _ = env._cube("green")
    st = env.script
    xy_goal, z_goal, yaw_err, grip = tip[:2], tip[2], 0.0, -1.0
    if st == "approach":
        xy_goal, z_goal, yaw_err = rp[:2], rp[2] + 0.06, face(wrap(ryaw - gy))
        if np.linalg.norm(rp[:2] - tip[:2]) < 0.01 and abs(yaw_err) < np.radians(8):
            st = "descend"
    if st == "descend":
        xy_goal, z_goal, yaw_err = rp[:2], rp[2] + GRASP_Z, face(wrap(ryaw - gy))
        if np.linalg.norm(rp[:2] - tip[:2]) < 0.008 and tip[2] < rp[2] + GRASP_Z + 0.006:
            st, env.t_state = "grasp", 0
    if st == "grasp":
        grip, env.t_state = 1.0, env.t_state + 1
        if env.t_state > 6:
            st = "lift" if env._red_grasped() else "approach"
    if st == "lift":
        grip, z_goal = 1.0, CARRY_Z
        if tip[2] > CARRY_Z - 0.01:
            st = "carry"
    if st == "carry":
        grip, xy_goal, z_goal = 1.0, gp[:2], CARRY_Z
        yaw_err = face(wrap(gyaw - ryaw))
        if np.linalg.norm(rp[:2] - gp[:2]) < 0.006 and abs(yaw_err) < np.radians(6):
            st = "lower"
    if st == "lower":
        grip, xy_goal = 1.0, gp[:2]
        yaw_err = face(wrap(gyaw - ryaw))
        z_goal = gp[2] + 2 * HALF + GRASP_Z + 0.003
        if np.linalg.norm(rp[:2] - gp[:2]) > 0.012:
            st = "carry"
        elif tip[2] < z_goal + 0.003:
            st, env.t_state = "release", 0
    if st == "release":
        grip, env.t_state = -1.0, env.t_state + 1
        if env.t_state > 6:
            st = "rise"
    if st == "rise":
        z_goal = rp[2] + GRASP_Z + 0.04
    if st in ("approach", "descend") and env._red_grasped():
        st = "lift"
    if st in ("lift", "carry", "lower") and not env._red_grasped():
        st = "approach"
    env.script = st
    a = np.zeros(5)
    a[:2] = np.clip((xy_goal - tip[:2]) / 0.01, -1, 1)
    a[2] = np.clip((z_goal - tip[2]) / 0.01, -1, 1)
    a[3] = np.clip(yaw_err / np.radians(5), -1, 1)
    a[4] = grip
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=80)
    ap.add_argument("--env-kwargs", default="{}", help="JSON dict for StackCubes (to match a training setup)")
    args = ap.parse_args()
    env = StackCubes(seed=0, **json.loads(args.env_kwargs))
    res = collections.defaultdict(collections.Counter)
    t0, n = time.time(), 0
    for ep in range(args.episodes):
        _, info = env.reset(seed=ep)
        env.script, env.t_state = ("approach" if info["family"] == "table" else "lift"), 0
        for t in range(env.max_steps):
            _, r, term, trunc, inf = env.step(act(env))
            n += 1
            if term or trunc:
                out = ("success" if inf.get("success") else "knocked" if inf.get("knocked")
                       else "out" if inf.get("out") else "timeout")
                res[info["family"]][out] += 1
                break
    for fam, c in res.items():
        tot = sum(c.values())
        print(f"{fam:6s}: success {c['success']}/{tot}  {dict(c)}")
    allc = sum((c for c in res.values()), collections.Counter())
    print(f"overall success {allc['success'] / sum(allc.values()):.0%}; {n / (time.time() - t0):.0f} steps/s")


if __name__ == "__main__":
    main()
