"""Evaluate a trained policy on rl.envs.MovingCubeGrasp: success rate, failure modes, time to grasp.

Usage:  python -m rl.eval_policy runs/rl/<run>/model [--episodes 200] [--vecnorm runs/rl/<run>/vecnormalize.pkl]
Episodes are split by whether the cube was being slid when the run started (moving) or not (static).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.envs.moving_cube import MovingCubeGrasp
from rl.envs.stack_cubes import StackCubes

ENVS = {"moving_cube": MovingCubeGrasp, "stack": StackCubes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--env", choices=sorted(ENVS), default="moving_cube")
    ap.add_argument("--vecnorm")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--env-kwargs", default="{}")
    args = ap.parse_args()
    model_path = Path(args.model)
    vn = args.vecnorm or (str(model_path.parent / "vecnormalize.pkl") if model_path.name in ("model", "model.zip")
                          else str(model_path.parent / f"vecnormalize_{model_path.stem.split('_')[-1]}.pkl"))
    kwargs = json.loads(args.env_kwargs)
    venv = DummyVecEnv([lambda: ENVS[args.env](**kwargs)])
    venv = VecNormalize.load(vn, venv)
    venv.training, venv.norm_reward = False, False
    model = PPO.load(str(model_path), device="cpu")
    out = {}
    times = []
    for ep in range(args.episodes):
        venv.seed(args.seed + ep)
        obs = venv.reset()
        env = venv.venv.envs[0]
        kind = env.family if args.env == "stack" else ("moving" if env.pushing else "static")
        for t in range(env.max_steps):
            act, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = venv.step(act)
            if done[0]:
                info = infos[0]
                res = ("success" if info.get("success") else "knocked" if info.get("knocked")
                       else "out" if info.get("out") else "timeout")
                out.setdefault(kind, []).append(res)
                if res == "success":
                    times.append((t + 1) * 0.05)
                break
    for kind, rs in out.items():
        if rs:
            c = {k: rs.count(k) for k in ("success", "knocked", "out", "timeout")}
            print(f"{kind:6s} ({len(rs):3d} episodes): success {c['success'] / len(rs):5.1%}  "
                  f"knocked {c['knocked']}  out {c['out']}  timeout {c['timeout']}")
    allr = [x for rs in out.values() for x in rs]
    print(f"overall success {allr.count('success') / len(allr):.1%}; time to success median "
          f"{np.median(times) if times else float('nan'):.1f} s")


if __name__ == "__main__":
    main()
