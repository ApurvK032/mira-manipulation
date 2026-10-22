"""Train a PPO policy for rl.envs.MovingCubeGrasp (simulation on the CPU cores, network on the GPU).

Usage:  python -m rl.train_ppo [--env moving_cube|stack] --steps 20000000 [--envs 20] [--name run1]
                              [--resume runs/rl/<run>/model_XXXX]
Writes runs/rl/<name>/: progress.log (one line per ~200k steps: success rate, reward, speed), TensorBoard
logs, checkpoints (model_<steps>.zip + vecnormalize_<steps>.pkl) and final model.zip / vecnormalize.pkl.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from rl.envs.moving_cube import MovingCubeGrasp
from rl.envs.stack_cubes import StackCubes

ENVS = {"moving_cube": MovingCubeGrasp, "stack": StackCubes}

ROOT = Path(__file__).resolve().parents[1]


def make_env(env_name, rank, seed, kwargs):
    def f():
        return ENVS[env_name](seed=seed + rank, **kwargs)
    return f


class Progress(BaseCallback):
    """Compact progress lines, success-rate logging, and checkpoints."""

    def __init__(self, out: Path, every=200_000, ckpt_every=2_000_000):
        super().__init__()
        self.out, self.every, self.ckpt_every = out, every, ckpt_every
        self.next_log, self.next_ckpt, self.t0 = every, ckpt_every, time.time()
        self.outcomes = []  # 'success' / 'knocked' / 'out' / 'timeout' for recent episodes

    def _on_training_start(self):
        # a resumed run starts at its checkpoint's step count: log and save relative to that
        self.start = self.num_timesteps
        self.next_log, self.next_ckpt, self.t0 = self.start + self.every, self.start + self.ckpt_every, time.time()

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.outcomes.append("success" if info.get("success") else "knocked" if info.get("knocked")
                                     else "out" if info.get("out") else "timeout")
        self.outcomes = self.outcomes[-2000:]
        n = self.num_timesteps
        if n >= self.next_log:
            self.next_log += self.every
            rec = self.outcomes[-500:]
            sr = np.mean([o == "success" for o in rec]) if rec else 0.0
            kn = np.mean([o == "knocked" for o in rec]) if rec else 0.0
            ep = [e["r"] for e in self.model.ep_info_buffer]
            el = [e["l"] for e in self.model.ep_info_buffer]
            fps = (n - self.start) / (time.time() - self.t0)
            self.logger.record("rollout/success_rate", sr)
            self.logger.record("rollout/knocked_rate", kn)
            line = (f"{time.strftime('%H:%M:%S')}  steps {n / 1e6:6.2f}M  success {sr:5.1%}  knocked {kn:5.1%}  "
                    f"ep_reward {np.mean(ep) if ep else 0:7.1f}  ep_len {np.mean(el) if el else 0:5.0f}  {fps:5.0f} steps/s")
            print(line, flush=True)
            with open(self.out / "progress.log", "a") as fh:
                fh.write(line + "\n")
        if n >= self.next_ckpt:
            self.next_ckpt += self.ckpt_every
            self.model.save(self.out / f"model_{n}")
            self.model.get_vec_normalize_env().save(str(self.out / f"vecnormalize_{n}.pkl"))
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=sorted(ENVS), default="moving_cube")
    ap.add_argument("--steps", type=float, default=20e6)
    ap.add_argument("--envs", type=int, default=20)
    ap.add_argument("--name", default=time.strftime("ppo_%Y%m%d_%H%M%S"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--resume", help="checkpoint path without .zip (its vecnormalize_<steps>.pkl is loaded too)")
    ap.add_argument("--env-kwargs", default="{}", help="JSON dict passed to MovingCubeGrasp")
    args = ap.parse_args()

    out = ROOT / "runs" / "rl" / args.name
    out.mkdir(parents=True, exist_ok=True)
    kwargs = json.loads(args.env_kwargs)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    venv = VecMonitor(SubprocVecEnv([make_env(args.env, i, args.seed * 1000, kwargs) for i in range(args.envs)]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.resume:
        stem = Path(args.resume).stem
        vn = "vecnormalize.pkl" if stem == "model" else f"vecnormalize_{stem.rsplit('_', 1)[-1]}.pkl"  # final save or checkpoint
        venv = VecNormalize.load(str(Path(args.resume).parent / vn), venv)
        # --lr applies on resume too (the saved schedule would restart at its full rate); the default equals the
        # rate every earlier run used
        lr = args.lr
        sched = lambda p: lr * max(p, 0.1)
        model = PPO.load(args.resume, env=venv, device=device, tensorboard_log=str(out / "tb"),
                         custom_objects={"learning_rate": sched, "lr_schedule": sched})
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        lr = args.lr
        model = PPO(
            "MlpPolicy", venv, n_steps=256, batch_size=args.envs * 256 // 2, n_epochs=5, gamma=0.99, gae_lambda=0.95,
            learning_rate=lambda p: lr * max(p, 0.1), clip_range=0.2, ent_coef=0.0, vf_coef=0.5,
            max_grad_norm=0.5, target_kl=0.03,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-0.5,
                               activation_fn=torch.nn.ELU),
            device=device, seed=args.seed, verbose=0, tensorboard_log=str(out / "tb"))
    print(f"training {args.name} on {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}), "
          f"{args.envs} envs, {args.steps / 1e6:.0f}M steps -> {out}", flush=True)
    model.learn(total_timesteps=int(args.steps), callback=Progress(out), reset_num_timesteps=not args.resume)
    model.save(out / "model")
    venv.save(str(out / "vecnormalize.pkl"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
