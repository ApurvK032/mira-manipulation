"""Record the states the chain hands over in (grab policy -> scripted lift -> placement policy), so the placement
policy can train from what the chain actually gives it: StackCubes(family "handover", handover_file=...).

The "carry" start family holds red in a perfect grasp with the arm at rest; a real hand-over has red where the grab
caught it (off-centre, turned), the arm still rising, and green wherever it was pushed. This runs the grab policy and
the lift exactly as rl/eval_chained.py does, from full-task ("table") starts, and saves the full simulator state at
each hand-over with red held: qpos, qvel, ctrl, and the step it happened at.

Usage:  python -m rl.record_handover <grab_model> --episodes 500 --seed 0 --env-kwargs '{...}' --out runs/rl/handover/x.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.envs.moving_cube import HALF, MovingCubeGrasp
from rl.envs.stack_cubes import CARRY_Z, StackCubes
from rl.eval_chained import grab_obs, vecnorm_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grab_model")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env-kwargs", default="{}", help="StackCubes kwargs (arm model, delays...); starts are forced to table")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    kwargs = dict(json.loads(args.env_kwargs), family_probs=[1, 0, 0, 0, 0])
    grab = PPO.load(args.grab_model, device="cpu")
    with_green = grab.observation_space.shape[0] == 34
    grab_norm = VecNormalize.load(vecnorm_path(args.grab_model), DummyVecEnv([lambda: MovingCubeGrasp(obstacle=with_green)]))
    grab_norm.training = False
    env = StackCubes(**kwargs)
    rec = {k: [] for k in ("qpos", "qvel", "ctrl", "steps")}
    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        phase = "grab"
        for t in range(env.max_steps):
            if phase == "grab":
                rp, _, _ = env._cube("red")
                if env._red_grasped() and rp[2] > HALF + 0.01:
                    phase = "lift"
            if phase == "lift" and (env._tip()[0][2] > CARRY_Z - 0.005 or not env._red_grasped()):
                if env._red_grasped():
                    rec["qpos"].append(env.d.qpos.copy())
                    rec["qvel"].append(env.d.qvel.copy())
                    rec["ctrl"].append(env.d.ctrl.copy())
                    rec["steps"].append(t)
                break
            if phase == "grab":
                act, _ = grab.predict(grab_norm.normalize_obs(grab_obs(env, with_green)[None]), deterministic=True)
                act = act[0]
            else:
                act = np.array([0.0, 0.0, 1.0, 0.0, 1.0])
            _, _, term, trunc, _ = env.step(act)
            if term or trunc:
                break
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **{k: np.array(v) for k, v in rec.items()})
    print(f"{len(rec['steps'])} hand-overs from {args.episodes} episodes -> {args.out}")


if __name__ == "__main__":
    main()
