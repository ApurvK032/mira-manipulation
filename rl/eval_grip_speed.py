"""Chain pairs on the standard 150 full-task starts (seeds 4242, 5242, 6242 x 50), streaming arm model, with the
simulated gripper as trained (instant) and at the real gripper's speed, each clean and with all sim-to-real
variations. Generalises runs/rl/gripspeed/eval_grip_speed.py (the 5090's, for grab12 + place11) to any pairs.

Usage:  python -m rl.eval_grip_speed grab13:place12 grab13:place11 grab12:place12 [--out runs/rl/gripspeed2] [--procs 12]
A pair "grabN:placeM" means runs/rl/grabN_ppo/model + runs/rl/placeM_ppo/model. Logs: <out>/<pair>_<cond>_<seed>.log;
prints a summary table.
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = dict(binary_gripper=True, release_needs_rise=False, place_bonus=0.3, stack_reward=2.5, knock_penalty=6.0,
            premature_release_penalty=1.0, gripper_hysteresis=0.3, grasp_bonus=2.0, green_shove_penalty=0.3,
            family_probs=[1, 0, 0, 0, 0], lag_mode="stream")
VAR = dict(calib_error=[0.003, 1.0], obs_bias=0.003, friction_scale=[0.6, 1.4], mass_scale=[0.6, 1.6])
REAL = dict(gripper_speed=[0.048, 0.125])  # the real gripper, measured on the arm (m/s per finger, closing / opening)
CONDS = {"base_off": BASE, "base_real": dict(BASE, **REAL), "all_off": dict(BASE, **VAR), "all_real": dict(BASE, **VAR, **REAL)}
SEEDS = (4242, 5242, 6242)


def model(name):
    return f"runs/rl/{name}_ppo/model"


def run(out, pair, cond, seed):
    grab, place = pair.split(":")
    log = out / f"{grab}_{place}_{cond}_{seed}.log"
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with open(log, "w") as fh:
        subprocess.run([sys.executable, "-m", "rl.eval_chained", model(grab), model(place), "--episodes", "50", "--seed",
                        str(seed), "--env-kwargs", json.dumps(CONDS[cond])], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                       env=env)


def summary(out, pair, cond):
    grab, place = pair.split(":")
    ok = n = 0
    per, ho, fails = [], [], {}
    for s in SEEDS:
        txt = (out / f"{grab}_{place}_{cond}_{s}.log").read_text()
        m = re.search(r"^table\s+\(\s*(\d+)\): success\s+([\d.]+)%", txt, re.M)
        k, p = int(m.group(1)), float(m.group(2))
        n, ok = n + k, ok + round(k * p / 100)
        per.append(f"{p:.0f}")
        h = re.search(r"handed over after median ([\d.]+)", txt)
        ho.append(h.group(1) if h else "-")
        f = re.search(r"failures by phase: (\{.*\})", txt)
        for key, v in (ast.literal_eval(f.group(1)) if f else {}).items():
            fails[key] = fails.get(key, 0) + v
    return f"{pair:16s} {cond:10s} {ok:3d}/{n} = {100 * ok / n:5.1f}%  {'/'.join(per):12s}  {', '.join(ho):15s}  {fails}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+")
    ap.add_argument("--out", default="runs/rl/gripspeed2")
    ap.add_argument("--procs", type=int, default=12)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(p, c, s) for p in args.pairs for c in CONDS for s in SEEDS]
    with ThreadPoolExecutor(args.procs) as ex:
        list(ex.map(lambda j: run(out, *j), jobs))
    print(f"{'pair':16s} {'condition':10s} {'success':>15s}  per seed       grab hands over   failures by phase")
    for p in args.pairs:
        for c in CONDS:
            print(summary(out, p, c))


if __name__ == "__main__":
    main()
