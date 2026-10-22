# Handoff: continue MIRA's RL work on the RTX 5090 PC

Written 2026-10-17 at about 16:40, from the Claude Code session on the lab PC (WSL2, RTX 4070). Read this first, then
`docs/rl/REPORT.md` for the full history, results and lessons.

## The project in one paragraph

**MIRA (Markerless Interactive Reactive Arm)**, https://github.com/ApurvK032/mira-manipulation (public, MIT).
- **Hardware:** a Trossen WidowX AI arm (driver and firmware 1.8.3) and a fixed RealSense D455.
- **Task:** stack a red 5 cm cube on a green one. Perception is markerless (M3T edge/region tracking), not
  AprilTags.
- **Behavior:** it reacts while people move the cubes. It follows red, picks it once it rests, follows green while
  carrying, places, and re-picks after any failure.
- **On the real arm today** (committed and working): a scripted, always-on controller, `python -m robot.controller`.
- **The RL work** (MuJoCo + PPO, everything under `rl/`, **not committed**) aims to replace the scripted pick and
  place with learned policies that react faster and better under interference.

## Where things stand

- **Policies (simulation):**
  - grab a moving cube next to green, with arm lag and dropouts: `runs/rl/grab4_ppo/model_14000016`, **94.7%**;
  - place a held cube on a moving green: `runs/rl/place2_ppo/model_42000048`, 78.7% on random held starts.
- **Chain** (grab4 → scripted lift → place2) on 150 full-task starts: **94.0%** at the trained lag. The scripted
  stacker gets 57% in the same environment.
- **System ID on the real arm** (16:28, `runs/sysid/sysid_20261017_162847.npz`, fitted by `rl/sysid_fit.py`):
  - the arm follows targets sent through `SafeArm.follow_goal` with **lag tau 0.16 s and delay 0.16 s**, RMSE 0.33°;
  - the simulated deployment path gives the same numbers, so the lag comes from the re-aim scheme (re-aims of at
    least 0.3 s, at most every 0.15 s, needed by driver 1.8.3), not the motors.
- **Chain at the real arm's values** (tau 0.16 s, total delay 4 or 5 steps of 0.05 s, which includes 1–2 steps of
  perception delay): grab4 + place2 92.0% (138/150) both times.
  - It degrades with a slower arm: tau 0.25 s → 84.7%; tau 0.35 s → 58%.
- **Training when the PC was handed over** (both resumed, lag randomized over tau 0–0.35 s and delay 1–4 steps):
  - `grab5_ppo`, from grab4;
  - `place4_ppo`, from `place3_ppo/model_44000048`, a placement run with no lag change: 81% → 87% training success
    on placement starts.
  - Both ran to the end on the lab PC (grab5 to 20M steps, place4 to 52M). Their final saves are
    `runs/rl/<run>/model.zip` plus `vecnormalize.pkl`, and `train_ppo --resume`, `eval_policy` and `eval_chained` all
    accept them.
  - Their evaluations at the real arm's lag are in `runs/rl/lageval/` (`single_*`, `final_*`) and in REPORT.md.
- **Final lag-robust results** (details in REPORT.md, "Lag-robust runs"), at tau 0.16 s and delay 5:
  - grab5 87.3% alone, place4 86.0% alone (were 85.3% and 82.7%);
  - chain **grab5 + place2: 94.0% (141/150). This is the baseline to beat and the current deploy candidate.**
  - chain grab5 + place4: 91.3% (137/150), and 93.3% at delay 4. Within noise, but not better: the scripted-lift
    handover states differ from place4's training starts.

## Rules (from the user; keep them)

1. **No `git commit` or `git push` until the user says "commit" or "push".** When they do:
   - author "Apurv K" with their GitHub noreply email (copy the lab repo's local `git config user.email`, or ask);
   - several commits grouped by area;
   - each message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
2. **This PC is shared.** Check `nvidia-smi` and `htop` before starting anything big. Ask the user if someone else
   is using the GPU, and leave CPU cores for others unless the user says the machine is free.
3. **No real-arm motion from here** unless the user has moved the arm and camera to this PC and is at the power
   switch. Dry runs must never drive the real arm: dry mode uses `robot.controller.DryArm` and
   `runs/dry_command.json`.
4. **Never call `hardware_reset` on the D455.** Keep any web service on localhost or the tailnet.
5. **Report honestly.** Simulation success is not real success; say what was and wasn't verified.

## Setup on this PC (no sudo needed)

```bash
git clone https://github.com/ApurvK032/mira-manipulation.git ~/mira && cd ~/mira
tar -xzf /path/to/mira_rl_handoff_*.tar.gz          # adds rl/, docs/rl/, scripts/sysid_record.py, runs/rl, runs/sysid
# training env (micromamba is root-free; any conda or venv with Python 3.12 works too)
mkdir -p ~/.local/bin && curl -fsSL -o ~/.local/bin/micromamba https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 && chmod +x ~/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=~/micromamba
~/.local/bin/micromamba create -y -n rlwx-train -c conda-forge python=3.12 pip
PY=~/micromamba/envs/rlwx-train/bin/python
$PY -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # cu128 covers the 5090 (sm_120)
$PY -m pip install -r rl/requirements-train.txt
$PY -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"   # expect sm_120 in the list
```

**Smoke test** (each takes about a minute):

```bash
$PY -m rl.scripted_stack --episodes 40                       # expect about 80% overall (all start families)
$PY -m rl.eval_chained runs/rl/grab4_ppo/model_14000016 runs/rl/place2_ppo/model_42000048 --episodes 50 --seed 4242 \
  --env-kwargs '{"binary_gripper": true, "release_needs_rise": false, "place_bonus": 0.3, "stack_reward": 2.5, "knock_penalty": 6.0, "premature_release_penalty": 1.0, "gripper_hysteresis": 0.3, "grasp_bonus": 2.0, "lag_mode": "target", "green_shove_penalty": 0.3, "family_probs": [1, 0, 0, 0, 0], "lag_tau": [0.16, 0.16], "delay_steps": [5, 5]}'
# expect about 92% (46/50 on this seed)
```

## Updating from a newer bundle

Don't extract a newer bundle directly over a repo you have edited: it would overwrite your changes to files it
contains. Instead:
- extract it into a temp folder;
- copy `runs/` over;
- diff and merge the code and doc files.

## Tasks, in order

### 1. Fine-tune both policies around the real arm's values

Narrow the lag range to the measured values with margin: **tau 0.10–0.25 s, delay 3–6 steps** (0.15–0.30 s). Set
`--envs` to about half the free cores per run. Training runs on the CPU; the GPU only runs the network.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
G=runs/rl/grab5_ppo/model    # final save of the finished run (with vecnormalize.pkl)
P=runs/rl/place4_ppo/model
nohup $PY -u -m rl.train_ppo --env moving_cube --steps 8e6 --envs 12 --name grab6_ppo --resume $G \
  --env-kwargs '{"obstacle": true, "lag_tau": [0.10, 0.25], "delay_steps": [3, 6], "lag_mode": "target", "obs_dropout": 0.03}' > runs/rl_grab6.log 2>&1 &
nohup $PY -u -m rl.train_ppo --env stack --steps 10e6 --envs 12 --name place5_ppo --resume $P \
  --env-kwargs '{"binary_gripper": true, "release_needs_rise": false, "place_bonus": 0.3, "stack_reward": 2.5, "knock_penalty": 6.0, "premature_release_penalty": 1.0, "gripper_hysteresis": 0.3, "grasp_bonus": 2.0, "lag_mode": "target", "green_shove_penalty": 0.3, "obs_dropout": 0.03, "lag_tau": [0.10, 0.25], "delay_steps": [3, 6], "family_probs": [0, 0.4, 0.3, 0.3, 0]}' > runs/rl_place5.log 2>&1 &
```

- **Progress:** `runs/rl/<run>/progress.log` gets a line every 200k steps (success, knock-over rate, steps/s).
  Checkpoints are saved every 2M steps as `model_<steps>.zip` plus `vecnormalize_<steps>.pkl`. The step counts are
  not round, so find them with `ls -t`.
- **If training success stalls or drops for 3M+ steps,** stop and rethink rather than running longer. See the
  lessons below.

### 2. Evaluate, and compare against the 94% baseline

```bash
# single skills (150 episodes); use the same env-kwargs as training but with fixed lag
$PY -m rl.eval_policy runs/rl/grab6_ppo/model_<N> --env moving_cube --episodes 150 \
  --env-kwargs '{"obstacle": true, "lag_tau": [0.16, 0.16], "delay_steps": [5, 5], "lag_mode": "target"}'
# chain on the standard 150 full-task starts: seeds 4242, 5242, 6242 with 50 episodes each (run all three in parallel)
for s in 4242 5242 6242; do $PY -m rl.eval_chained runs/rl/grab6_ppo/model_<N> runs/rl/place5_ppo/model_<M> \
  --episodes 50 --seed $s --env-kwargs '<the chain kwargs from the smoke test>' > runs/rl/lageval/new_$s.log & done
```

- Report the chain at tau 0.16 with delay 4 and with delay 5, and also at the old trained setting (default
  `lag_tau`/`delay_steps`) to check nothing regressed.
- A new pair becomes the deploy candidate only if it beats grab5 + place2 (94.0% at tau 0.16 s, delay 5).
  Also try the new grab policy with place2, and the new placement policy with grab5.
- **Idea for the placement run:** add starts taken from the scripted-lift handover (red just lifted to
  `CARRY_Z` by the chain) to `StackCubes`' start families, so placement trains on what the chain actually
  hands it.
- Add a dated section to `docs/rl/REPORT.md` in the same style: a results table, failures by phase, and the
  checkpoint paths to use.

### 3. Write the deployment runner, dry-tested only

The deployment runner is `rl/deploy.py`, which runs the chain on the real arm.
- **Interface:** policy actions → fingertip and wrist targets → IK (`robot.ik.IK`) → `SafeArm.follow_goal`. This is
  exactly what system ID measured, and it already enforces joint limits, the workspace box and effort guards.
- **Observations:** build the exact observations the policies trained on: 34 numbers for the grab policy (see
  `rl/eval_chained.grab_obs`) and 41 for placement (`StackCubes._obs`). Use real joints, the fingertip from FK, the
  previous action, and the same normalization (the checkpoint's `vecnormalize_<steps>.pkl`, with `training=False`).
- **Perception:** the live tracker serves `http://127.0.0.1:8080/stats.json`. Convert with
  `robot.grasp.read_stats` + `cubes_in_base`, and the calibration as in `robot/controller.py`.
- **Phases:** grab → scripted lift to `CARRY_Z` → place → release → watch pose, as in `rl/eval_chained.py`.
  - Re-aim at most every 0.15 s (`REAIM`), and call `follow_check()` every tick.
  - Map the gripper command with the same hysteresis as training.
  - Stop and hold on any SafetyStop, or if tracking is lost for over 0.5 s.
- **Commands:** start/pause/rest through the live page's `/command`, like `robot/controller.py`. Dry mode must read
  `runs/dry_command.json`, never the live command.
- **Dry test:** `--dry` uses `DryArm` plus a fake perception source: a MuJoCo `StackCubes` instance stepped
  alongside, publishing a stats dict with the same fields. Verify that it stacks in the dry loop and that the
  observations match the env's own `_obs()` to within float error.
- **Real-arm tests** happen later, on the lab PC with the user at the power switch. Hand over a checklist and the
  exact command.

### 4. If there is time: fewer knock-overs

Knocking cubes over is the main failure in every chain evaluation. Ideas, in REPORT.md's "Next steps": contact-force
penalties, or train grab and place together as one hierarchical policy.

## Lessons and gotchas (learned the hard way)

- **Success must out-pay hovering.** The success bonus is `10 + 2.x * steps_left`. Almost finishing once paid better
  than finishing.
- **Model lag as a target the arm follows** (`lag_mode="target"`), not as a slower arm.
- **End-to-end stacking never learned the grasp.** Skill chaining did (grab → scripted lift → place).
- **Release was never discovered with a continuous gripper.** Binary gripper plus hysteresis, plus a
  premature-release penalty.
- **Shell gotchas:**
  - `VAR=... && cmd &` runs the whole chain in a background subshell, so `VAR` is not set afterwards.
  - `pgrep -f pattern` matches your own shell's command line, so wait loops can spin forever.
  - `pkill` once killed its own shell. Stop runs by explicit PID.
- **More than 2 steps of observation delay used to crash at reset.** Fixed in `rl/envs` (the reading history grows
  with the delay). Old settings behave the same.
- **Speeds:** about 2.0–2.5k steps/s per run with 12 envs on the 7900X when two run at once. 10M steps took about
  45 minutes alone.

## Files

- `rl/envs/moving_cube.py` (`MovingCubeGrasp`), `rl/envs/stack_cubes.py` (`StackCubes`): the environments.
- `rl/train_ppo.py`: `--env moving_cube|stack`, `--resume`, `--env-kwargs`.
- `rl/eval_policy.py`, `rl/eval_chained.py`: evaluation.
- `rl/scripted_check.py`, `rl/scripted_stack.py`: scripted baselines.
- `scripts/sysid_record.py`, `rl/sysid_fit.py`: system ID.
- `rl/make_handoff.sh`: packs all of this again.
- `robot/arm.py`: `SafeArm` (`follow_begin`, `follow_goal`, `follow_check`, `move_joints`); `WATCH` and `REST` poses.
- `robot/controller.py`: the scripted always-on controller, `DryArm`, and `read_command`.
- `robot/follow.py`, `robot/grasp.py`, `robot/ik.py`.
- `docs/rl/REPORT.md`: the full log of results.
