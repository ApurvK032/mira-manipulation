"""Evaluate skill chaining on rl.envs.StackCubes: the grab-a-moving-cube policy (trained on MovingCubeGrasp) until
red is picked up, a straight lift to carry height (where the stacking policy's "carry" starts are), then the
stacking policy for carry, place and release.

The grab policy sees red through the same 31-number observation it was trained on, rebuilt from the stacking
environment's state (same delayed, noisy readings). Its gripper command goes through the stacking
environment's gripper mapping.

Usage:  python -m rl.eval_chained <grab_model> <stack_model> [--episodes 100] [--env-kwargs '{...}']
"""

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from rl.envs.moving_cube import CTRL_DT, HALF, OPEN, MovingCubeGrasp, wrap
from rl.envs.stack_cubes import CARRY_Z, StackCubes


def vecnorm_path(model):
    p = Path(model)
    if p.stem == "model":  # a run's final save
        return str(p.parent / "vecnormalize.pkl")
    return str(p.parent / f"vecnormalize_{p.stem.split('_')[-1]}.pkl")


def grab_obs(env: StackCubes, with_green=False):
    """The MovingCubeGrasp observation, for red (plus green's position for the obstacle version), from the
    stacking environment's state."""
    tip, R = env._tip()
    gy = env._grip_yaw(R)
    k = env.delay
    cp, cyaw = env.hist["red"][-1 - k]
    prev, _ = env.hist["red"][-2 - k]
    dyaw = wrap(cyaw - gy)
    return np.concatenate([
        tip, [np.sin(2 * gy), np.cos(2 * gy)], cp - tip, (cp[:2] - prev[:2]) / CTRL_DT, [np.sin(4 * dyaw), np.cos(4 * dyaw)],
        [cp[2] - HALF], [env.d.qpos[env.finger] / OPEN], env.d.qpos[env.qadr], 0.1 * env.d.qvel[env.dadr], env.prev_action,
    ] + ([env.hist["green"][-1 - k][0] - tip] if with_green else [])).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grab_model")
    ap.add_argument("stack_model")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--env-kwargs", default="{}")
    args = ap.parse_args()
    kwargs = json.loads(args.env_kwargs)
    grab = PPO.load(args.grab_model, device="cpu")
    with_green = grab.observation_space.shape[0] == 34
    grab_norm = VecNormalize.load(vecnorm_path(args.grab_model), DummyVecEnv([lambda: MovingCubeGrasp(obstacle=with_green)]))
    grab_norm.training = False
    stack = PPO.load(args.stack_model, device="cpu")
    venv = VecNormalize.load(vecnorm_path(args.stack_model), DummyVecEnv([lambda: StackCubes(**kwargs)]))
    venv.training, venv.norm_reward = False, False
    res, handover, fail_phase = {}, [], {}
    for ep in range(args.episodes):
        venv.seed(args.seed + ep)
        obs = venv.reset()
        env = venv.venv.envs[0]
        fam = env.family
        phase = "grab" if fam in ("table", "grasp") else "stack"
        for t in range(env.max_steps):
            if phase == "grab":
                rp, _, _ = env._cube("red")
                if env._red_grasped() and rp[2] > HALF + 0.01:
                    phase = "lift"
            if phase == "lift" and (env._tip()[0][2] > CARRY_Z - 0.005 or not env._red_grasped()):
                phase = "stack"
                handover.append(t * CTRL_DT)
            if phase == "grab":
                act, _ = grab.predict(grab_norm.normalize_obs(grab_obs(env, with_green)[None]), deterministic=True)
            elif phase == "lift":
                act = np.array([[0.0, 0.0, 1.0, 0.0, 1.0]], dtype=np.float32)
            else:
                act, _ = stack.predict(obs, deterministic=True)
            obs, _, done, infos = venv.step(act)
            if done[0]:
                info = infos[0]
                r = ("success" if info.get("success") else "knocked" if info.get("knocked")
                     else "out" if info.get("out") else "timeout")
                res.setdefault(fam, []).append(r)
                if r != "success":
                    fail_phase.setdefault(f"{r} in {phase}", 0)
                    fail_phase[f"{r} in {phase}"] += 1
                break
    for fam, rs in res.items():
        c = {k: rs.count(k) for k in ("success", "knocked", "out", "timeout")}
        print(f"{fam:6s} ({len(rs):3d}): success {c['success'] / len(rs):5.1%}  knocked {c['knocked']}  out {c['out']}  timeout {c['timeout']}")
    print("failures by phase:", fail_phase)
    allr = [x for rs in res.values() for x in rs]
    print(f"overall {allr.count('success') / len(allr):.1%}; grab handed over after median "
          f"{np.median(handover) if handover else float('nan'):.1f} s")


if __name__ == "__main__":
    main()
