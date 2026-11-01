# Reinforcement learning report

This report covers MIRA's reinforcement learning work, from the first simulated environments (2026-10-17) to the
first runs on the real arm (2026-10-31).
- The [summary](#summary-as-of-2026-10-31) is current.
- The sections after it are a dated log, oldest first. Each section's numbers are as of its date; later sections
  supersede them.
- How to run the policies on the arm: [DEPLOY_CHECKLIST.md](DEPLOY_CHECKLIST.md).

## Summary (as of 2026-10-31)

- **Approach:**
  - **The chain:** a learned grab policy, a short scripted lift, then a learned placement policy.
  - **Training:** PPO, in MuJoCo, against a system-identified model of the WidowX AI while a simulated hand moves the
    cubes.
  - **Input:** cube poses from the markerless tracker, not images. The policies use the same perception as the
    scripted controller.
- **Simulation (grab13 + place12):**
  - **72%** stacked over 150 full-task trials at the real gripper speed (108/150);
  - **66%** with calibration, tracker, friction and mass errors added.
  - Earlier policies reached 94% with an idealised instant gripper. Modelling the real gripper's speed dropped them
    to 12%, and fine-tuning for it produced grab13 + place12.
- **On the real arm (2026-10-31, grab13 + place12):**
  - The grab policy picked up red in all 14 attempts after the first fixes.
  - 8 of the 12 attempts that weren't stress tests ended with red stacked on green, and 4 met the runner's full
    stacking rule (in 4.7–7.3 s).
  - In 6 of those 12, a person moved the cubes mid-task. They ended stacked as often as the undisturbed ones (4 of 6
    each).
  - The main remaining failure is the place policy grabbing red again after letting go. See
    [First real-arm runs of the RL chain](#first-real-arm-runs-of-the-rl-chain-rtx-5090-pc-2026-10-31).
- **What didn't work, and why:**
  1. A single end-to-end stacking policy learned to carry, place and release (up to 91% on near-finished starts),
     but never to grasp: with red under its fingers, it acts as if already holding it.
  2. The first arm-lag model made the arm slow instead of late. It was replaced (`lag_mode="target"`).
  3. Three reward-design bugs, each found and fixed (details below).
- **Next:**
  1. Stop the placement policy from re-grasping after the release.
  2. Log joint efforts at every step.
  3. Train with the real gripper's closing profile, and penalise fast dives near the table.
  4. Benchmark the RL chain against the scripted controller, with a person interfering.

## Contents

- **Simulation (2026-10-17 to 2026-10-18)**
  - [Setup](#setup)
  - [Skill B: grab a moving cube (rl/envs/moving_cube.py)](#skill-b-grab-a-moving-cube-rlenvsmoving_cubepy)
  - [Skill D: full stacking under interference (rl/envs/stack_cubes.py)](#skill-d-full-stacking-under-interference-rlenvsstack_cubespy)
  - [Final numbers (09:40)](#final-numbers-0940)
  - [Afternoon runs (14:10–15:35): fewer knock-overs, robust to tracking dropouts](#afternoon-runs-14101535-fewer-knock-overs-robust-to-tracking-dropouts)
  - [System ID tools and a lag stress test (16:00)](#system-id-tools-and-a-lag-stress-test-1600)
  - [Lag-robust runs (16:12–17:43): grab5 and place4](#lag-robust-runs-16121743-grab5-and-place4)
  - [Deployment runner, dry only (RTX 5090 PC, 17:00–18:00)](#deployment-runner-dry-only-rtx-5090-pc-17001800)
  - [Fine-tuning at the real arm's lag (RTX 5090 PC, 17:51–19:28): grab6 and place5, no improvement](#fine-tuning-at-the-real-arms-lag-rtx-5090-pc-17511928-grab6-and-place5-no-improvement)
  - [The re-aim arm model (lab PC, 19:30–20:35): the deployment path is slower than the fitted lag](#the-re-aim-arm-model-lab-pc-19302035-the-deployment-path-is-slower-than-the-fitted-lag)
  - [Feedforward re-aims and the speed ceiling of driver 1.8.3 (lab PC, 20:40–22:15)](#feedforward-re-aims-and-the-speed-ceiling-of-driver-183-lab-pc-20402215)
  - [Fine-tuning for the real arm's motion: place6, hand-over starts (22:15–22:30)](#fine-tuning-for-the-real-arms-motion-place6-hand-over-starts-22152230)
  - [place7 and grab8 results, and where the plateau is (00:05)](#place7-and-grab8-results-and-where-the-plateau-is-0005)
  - [Grasp-precision fine-tunes: grab9 and grab10 (00:15–02:00)](#grasp-precision-fine-tunes-grab9-and-grab10-00150200)
  - [Sim-to-real variations: sensitivity and a robustness fine-tune (02:30–)](#sim-to-real-variations-sensitivity-and-a-robustness-fine-tune-0230)
- **Driver upgrade and streaming (2026-10-18 to 2026-10-20)**
  - [What a driver upgrade would buy, in simulation (morning, 2026-10-18)](#what-a-driver-upgrade-would-buy-in-simulation-morning-2026-10-18)
  - [Streaming for driver 1.9.3 (evening, 2026-10-18; the upgrade is planned for 2026-10-19)](#streaming-for-driver-193-evening-2026-10-18-the-upgrade-is-planned-for-2026-10-19)
  - [Upgrade day (2026-10-20): firmware 1.9.4, driver 1.9.3, and streaming on the real arm](#upgrade-day-2026-10-20-firmware-194-driver-193-and-streaming-on-the-real-arm)
- **The real arm, gripper and camera (2026-10-20 to 2026-10-21)**
  - [The arm and camera on the RTX 5090 PC (evening, 2026-10-20): calibration, grip, first RL attempt](#the-arm-and-camera-on-the-rtx-5090-pc-evening-2026-10-20-calibration-grip-first-rl-attempt)
  - [The real gripper's speed breaks grab12 + place11 (2026-10-21, simulation only)](#the-real-grippers-speed-breaks-grab12--place11-2026-10-21-simulation-only)
  - [Fine-tuning for the real gripper speed: grab13 and place12 (lab PC, 2026-10-21, simulation only)](#fine-tuning-for-the-real-gripper-speed-grab13-and-place12-lab-pc-2026-10-21-simulation-only)
- **First real-arm runs (2026-10-31)**
  - [First real-arm runs of the RL chain (RTX 5090 PC, 2026-10-31)](#first-real-arm-runs-of-the-rl-chain-rtx-5090-pc-2026-10-31)
- [Code map](#code-map)

## Setup

- **Hardware:**
  - training on the lab PC: an RTX 4070 for the networks, and a Ryzen 9 7900X for the simulation (20–24 parallel
    copies);
  - some runs, and all real-arm sessions, on a shared RTX 5090 PC.
- **Training environment:** a separate micromamba env `rlwx-train` (PyTorch 2.11 + CUDA 12.8, Stable-Baselines3 2.9,
  Gymnasium 1.4, MuJoCo 3.16), so the perception env `rlwx` is untouched.
- **Algorithm:** PPO, MLP 256×256 (ELU), observation and reward normalization, 20 Hz control.
- **Action:** fingertip move (±1 cm/step) + wrist turn (±5°/step) + gripper, through two damped-least-squares IK
  iterations to the arm's joint position actuators, clipped to a safe workspace.
- **Observation:** cube readings delayed one or two steps and noisy (1.5 mm, 2°), like the real perception.
- **Speed:** about 3.4–3.9k steps/s while training. The raw simulation reaches about 9.6k steps/s; the rest is PPO overhead.

## Skill B: grab a moving cube (`rl/envs/moving_cube.py`)

A hand slides the cube at up to 12 cm/s with turns and pauses, and lets go only once the gripper catches up.
Success means the cube is lifted 4 cm and held for 0.5 s.

| Policy | Success | Moving / still | Median time |
|---|---|---|---|
| Scripted, true cube state | 95% (38/40) | — | 2.8 s |
| **Scripted, same delayed and noisy readings** | **97% (97/100)** | — | 2.9 s |
| RL run 1 (reward bug, stopped at 3M) | ~4–9% | — | — |
| **RL run 2, 10M steps (200 eval episodes)** | **93%** | 92% / 100% | **2.6 s** |

- **Run 1 found a reward bug:** holding an almost-lifted cube paid about 6 per step, while success paid only +20
  and ended the episode. The policy learned not to finish. In the fix, per-step shaping is capped at about 2.1 and
  the success bonus is worth more than all the remaining steps.
- **Run 2:** 93% success, slightly below the scripted baseline on identical inputs (97%) but faster (2.6 s vs 2.9 s),
  and still improving when stopped (94.8% training success at 10M). Checkpoint: `runs/rl/run2_ppo/model_10000000`.
- **Takeaway:** for this narrow skill, a good hand-written policy is about as good as RL. RL's value should show in
  harder, longer tasks, which is why the GPU time moved to stacking.

## Skill D: full stacking under interference (`rl/envs/stack_cubes.py`)

- **The task:** pick red (possibly while it is slid), carry it, follow green if green is moved while red is held,
  place red, release, and leave a standing stack (red centered within 10 mm, both upright, gripper open) for 0.5 s.
- **OmniReset-style start families:**
  - table: full task;
  - carry: red held elsewhere;
  - near: red held a few cm above and off green;
  - place: red held just above green.
- **Arm lag:** the simulated arm's joint targets are low-pass filtered with a random 0–0.15 s time constant, standing
  in for the real arm's slower response.
- **Held-cube start states:** all 134 sampled kept the cube in the gripper (a problem other projects have had to fix).

**Scripted stacker** (true state, with lag): **79% overall** (table 23/29, carry 15/19, near 13/19, place 12/13).

| Run | What changed | Result |
|---|---|---|
| stack1 (4M steps) | First version | 0% success. Releasing needed a precise continuous gripper command held for 10 steps; never discovered |
| stack2 (4M) | Binary gripper; open counts as released; place bonus; bigger stack reward; 30/20/20/30 starts | place 87.5% but near/carry/table 0%: it learned to **let go immediately** as a reflex, dropping red everywhere |
| **stack3** (resumed from stack2 at 4M) | Penalty for opening away from the place; gripper hysteresis (±0.3) | Learning backward like OmniReset. At 8M: place 87.5%, near 28.6%, carry 22.2%, table 0% |

| stack3 at 14M | (continued) | place 91%, near 53%, carry 54%, **table 0%** (31/35 timeouts) |
| stack4 (resumed from stack3 at 14M) | New start family "grasp" (gripper open just above red); one-time +2 pick-up bonus; starts 25/15/15/25/20 | At 18M: place 87%, near 50%, carry 50%, **grasp 0%, table 0%** |

**Why the full task from the table fails:** the end-to-end policy learned the end of the task (carry, place,
release) but not the grasp. From a grasp start, with red right under the open fingers, it *lifts and closes at
once*, as if it were already holding red. The situation looks like its many practiced "carrying red" states. It
may untangle with more training; the grasp family was only 4M steps old at the time.

**Skill chaining** (`rl/eval_chained.py`): skill B's grab policy until red is picked up, a straight lift to carry
height, then the stacking policy. On full-task starts (table), 80 episodes: **35% success** (end-to-end: 0%).
Failures by phase (60 episodes): knocked over during the grab 16 (the grab policy never saw green: it pushes red
into it or hits it), knocked over during carry and place 13, timeouts 9, out of reach 4. Next: retrain the grab
with green on the table as an obstacle, and keep training placement.

### Fixing the arm-lag model, then a better chain

**A flaw in the first lag model:** in the first lag model, each step's 1 cm move was measured from where the lagging
fingertips were. So lag made the arm *slow*, not just *late*: at the strongest lag it moved at about 6 cm/s instead
of 20. A grab policy trained with it never caught a cube in 4M steps (`grab2_ppo`).
- **The fix, `lag_mode="target"`:** the policy moves a commanded target (at most 3 cm and 15° ahead of the
  fingertips), and the lagging arm follows it. The arm is late but at full speed, like the real arm following
  SafeArm targets.
- **Scripted grab with green present and lag:** 32/40 with the old model, 37/40 with the new one.

**New grab policy** (`grab3_ppo`, 8M steps): trained with green on the table as an obstacle (it sees green; knocking
or shoving green ends the episode) and the corrected lag. Evaluation, 150 episodes: **92.7% success** (moving
93.8%, still 85.7%), median **2.4 s**. The scripted grab in the same setting manages about 92.5%. RL matches it and
is faster.

**Placement specialist** (`place1_ppo`): the stacking policy resumed from stack4 at 22M, trained only on held starts
(carry 40%, near 30%, place 30%) with the corrected lag. After 2M more steps: **76% training success** on those starts.

**Full chain on full-task (table) starts**, 100 episodes, same environment, lag included:

| System | Inputs | Success |
|---|---|---|
| End-to-end stacking policy (stack4, 22M) | delayed, noisy | 0% |
| First chain (grab trained without green or lag) | delayed, noisy | 35% |
| **Chain: grab3 + lift to carry height + place1 (24M)** | **delayed, noisy** | **78%** |
| Scripted stacker (`rl/scripted_stack.py`) | **true state** | 57% (35 timeouts: its fixed thresholds struggle with lag) |

- **Failures of the chain:** knocked over during the grab 8, during placement 4, timeouts 10.
- **Caveats:**
  - the scripted stacker is a quick reference script, not the tuned real-arm controller (`robot/controller.py`);
  - the lift between the two policies is scripted;
  - all of this is in simulation.

## Final numbers (09:40)

**Placement specialist** (`place1_ppo`, final, 37M steps). Evaluation on held starts, 150 episodes: **80.0% success**
(carry 78.6%, near 78.9%, place 83.8%), median 1.3 s. Main failure: knocked over (23 of 30).

**Full chain on full-task (table) starts**, final policies (`grab3_ppo/model_8000016` + scripted lift +
`place1_ppo/model_36000048`), 150 episodes:

| System | Inputs | Success on full-task starts |
|---|---|---|
| End-to-end stacking policy (stack4) | delayed, noisy | 0% |
| First chain (grab without green or lag) | delayed, noisy | 35% |
| Chain with place1 at 24M | delayed, noisy | 78% |
| **Final chain (place1 at 36M)** | **delayed, noisy** | **85.3% (128/150)** |
| Scripted stacker, same environment | true state | 57% |

- **The 22 failures:** knocked over during the grab 7, during placement 7, grab timeouts 6, placement timeouts 2.
- **Median time:** the grab hands over after about 2.0–2.4 s; a full stack takes roughly 4–5 s.

## Afternoon runs (14:10–15:35): fewer knock-overs, robust to tracking dropouts

Two runs side by side, 12 parallel simulations each, both resumed from the best checkpoints:
- **grab4** (from grab3 at 8M, +7M steps): with **simulated perception dropouts**: per step, a 3% chance the cube
  readings freeze for 1–6 steps (about 12% of steps frozen), as when a hand or the arm hides a cube.
- **place2** (from place1 at 36M, +7M steps): knock-over penalty 3 → 6, plus a new **green-shove penalty** (0.3 per
  step while green moves more than 3 mm without a hand pushing it), to teach setting red down gently. Also trained
  with dropouts.

| Evaluation | Result |
|---|---|
| grab4, no dropouts (150 episodes) | **94.7%**, median 2.4 s (grab3: 92.7%) |
| grab4 with dropouts | 92.0% |
| grab3 with dropouts | 92.7%: it was already robust to dropouts at this rate. The dropout training did not change it measurably (difference within ±4%) |
| place2 on random held starts (150) | 78.7% (place1: 80.0%): unchanged on these harder, random held positions |
| **Chain grab4 + place2, same 150 full-task starts as before** | **94.0% (141/150)**, was 85.3% |
| Chain with dropouts (100 starts) | **93% (93/100)** |

- **Chain failures** (9 of 150): knocked over during placement 4, during the grab 1, placement timeouts 3, grab timeout 1.
  Knock-overs fell from 14 to 5 out of 150.
- **Policies to use:**
  - `runs/rl/grab4_ppo/model_14000016`;
  - `runs/rl/place2_ppo/model_42000048`;
  - with `rl/eval_chained.py` and the scripted lift between them.

## System ID tools and a lag stress test (16:00)

**Tools, both tested in dry mode** (no arm motion):
- `scripts/sysid_record.py` drives the arm the way a policy would: fingertip targets at 20 Hz through SafeArm's smooth
  re-aims (`follow_goal`). It logs commanded vs measured joints at about 100 Hz to `runs/sysid/*.npz`.
  - Sequence: rest → watch → hover at (27, 0, 13) cm → ±3 cm steps in x, y and z → a 15 s y sweep at 0.2–1.2 Hz →
    gripper close/open timing → watch → rest.
  - The fingertips stay between 11.5 and 15 cm above the table, and every re-aim is path-checked.
  - `--dry` uses the simulated arm; without it, it needs someone at the power switch.
- `rl/sysid_fit.py` replays the logged targets through a first-order lag plus a pure delay, and grid-searches
  (tau, delay) for the lowest joint RMSE, per test and overall. Runs in about 10 s.

**The deployment path alone already adds lag.** On the dry arm (which models the driver's quintic re-aims), the fit
gives **tau 0.16 s, delay 0.15 s**. That comes from the re-aim scheme itself: each re-aim takes at least 0.3 s, and
re-aims happen at most every 0.15 s, which driver 1.8.3 needs. The real arm can only add to this.

**The chain at larger lags** (grab4 + place2, the same 150 full-task starts, lag fixed per run):

| Arm lag tau / delay | Success | Grab hands over after |
|---|---|---|
| trained range (0–0.15 s / 0.05–0.10 s) | 94.0% (141/150) | about 2.2 s |
| 0.16 s / 0.15 s (dry deployment path) | 92.0% (138/150) | 3.1 s |
| 0.25 s / 0.20 s | 84.7% (127/150) | 4.6 s |
| 0.35 s / 0.20 s | 58.0% (87/150) | 5.4 s |

So at the deployment path's own lag the policies still work, but past it they degrade fast, mostly by knocking
cubes over. Next: train both policies with a wider lag range (tau 0–0.35 s, delay 1–4 steps), then narrow it to the
real arm's fitted values once system ID has run.

**Real arm, 16:28** (`runs/sysid/sysid_20261017_162847.npz`; 35 s, loop at 66 Hz, no safety stops; fingertips
never below 114.9 mm):

| Test | tau | delay | RMSE with the fit (instant-arm RMSE 2.22°) |
|---|---|---|---|
| steps | 0.16 s | 0.11 s | 0.27° |
| sweep | 0.17 s | 0.16 s | 0.32° |
| all | **0.16 s** | **0.16 s** | 0.33° |

- **The real arm matches the dry model** (0.16 s / 0.15 s). The motors track the driver's trajectories closely. Almost
  all the lag comes from the re-aim scheme (re-aims of at least 0.3 s, at most every 0.15 s), not from the arm.
- At these values the current chain scores **92% in simulation** (row 2 of the table above), so arm lag should not
  block deployment.
- **With perception delay on top** (the camera and tracker add 1–2 steps, so the total delay is 4–5 steps,
  0.20–0.25 s), the chain at tau 0.16 s still scores **92.0% (138/150) at 4 steps and 92.0% (138/150) at 5 steps**.
  grab5 after 2M lag-robust steps (16M) with place2 scores 90.0% (135/150) at 5 steps, within noise. Delay barely
  matters; the slow-arm tau was what hurt in the stress test.
- **Faster reactions need faster re-aims.** That needs driver/firmware 1.9.3+, which handles fast re-commanding.
- The gripper timings in the log (0.81 s) only measure the script's own waits (0.8 s), not the gripper.

(`rl/envs`: the camera-reading history now grows with the delay. Before, delays above 2 steps crashed at reset.
Settings that trained with 1–2 steps behave the same as before.)

## Lag-robust runs (16:12–17:43): grab5 and place4

- **Both resumed:** grab5 from grab4 (+6M steps, to 20M) and place4 from place3 at 44M (+8M, to 52M).
- **Arm lag randomized** over tau 0–0.35 s and delay 1–4 steps.
- **Evaluated at the real arm's values** (tau 0.16 s; 5 steps of total delay unless noted), on the same starts as
  before.

| Evaluation | Before | Lag-robust |
|---|---|---|
| Grab alone, 150 episodes | grab4 85.3% | **grab5 87.3%** (knock-overs 5 → 3) |
| Placement alone, 150 held starts | place2 82.7%, median 2.2 s | **place4 86.0%**, median 1.9 s |
| Chain with place2, 150 full-task starts | grab4: 92.0% (138/150) | **grab5: 94.0% (141/150)** |
| Chain grab5 + place4, delay 5 / delay 4 | | 91.3% (137/150) / 93.3% (140/150) |
| Chain grab5 + place4 at the old trained lag | grab4 + place2: 94.0% | 88.7% (133/150) |

- **Each skill alone got better, but the chain did not improve with place4.** grab5 + place4 is within noise of
  grab5 + place2 at the real values (about ±2.5% standard error per 150 episodes). It is a little worse at the old,
  lower lag, mostly from knock-overs during placement.
  - A likely reason: the chain hands over from the scripted lift. Those start states differ from place4's training
    families (carry, near, place).
- **Deploy candidate:** `runs/rl/grab5_ppo/model` + `runs/rl/place2_ppo/model_42000048`, 94.0% at the real arm's
  values. Keep grab5 + place4 as the alternative.
- **Next (on the 5090):**
  - fine-tune around the real values (tau 0.10–0.25 s, delay 3–6 steps);
  - add the scripted-lift handover states to the placement training starts, so the placement policy trains on what
    the chain actually gives it.

## Deployment runner, dry only (RTX 5090 PC, 17:00–18:00)

`rl/deploy.py` runs the chain on the arm, and nothing else is needed from the lab PC beyond the live tracker. **It
has not driven the real arm**: everything below is in simulation, with `robot.controller.DryArm`.
`docs/rl/DEPLOY_CHECKLIST.md` has the exact commands and what to watch for on the first real run.

- **Control path:** the one system ID measured (`scripts/sysid_record.py`).
  - The policy runs at 20 Hz and moves a fingertip target.
  - IK turns the target into joints, and `SafeArm.follow_goal` re-aims at most every 0.15 s.
  - `follow_check()` runs every 10 ms, so joint limits, the workspace box, the fingertip floor and the effort guards
    all still apply.
- **Observations:** rebuilt from the measured joints, FK, the live tracker's poses (`stats.json` → arm frame) and the
  previous action, and normalized with each checkpoint's VecNormalize file. While red is held, its position comes
  from the fingertips (the fingers hide it).
- **Attempt:** WATCH → hover (27, 0, 13) cm → grab → scripted lift → place. It ends when StackCubes' success rule
  holds on the tracker's readings for 0.5 s.
- **After release:** stop and open the gripper fully (40 mm per finger). Rise straight up 4 cm only once the fingers
  read ≥ 36 mm, then go to WATCH. If the fingers stall below that, hold and pause: something is wedged.
- **Failed attempt:** a timeout (15 s), a cube unseen for 3 s, or a SafetyStop. Anything held is carried back to
  the table, never stacked by script.
- **Operator commands:** start/pause/resume/rest from the live page; a dry run reads only `runs/dry_command.json`.
- **New files:**
  - `rl/dry_world.py`: the dry world (camera, cubes, hands, motors in real time);
  - `rl/check_deploy_obs.py`: the observation check;
  - `docs/rl/DEPLOY_CHECKLIST.md`: the real-arm checklist.
- **Changes in `robot/`:** additions only; existing callers behave exactly as before.
  - `SafeArm.velocities()` and `DryArm.velocities()`;
  - `open_gripper(..., blocking=True)`: non-blocking is opt-in.

**Observations match training** (`rl/check_deploy_obs.py`): the chain runs inside StackCubes, but every observation
is rebuilt through the deployment path (FK, cube poses through the camera frame and back, the deploy builders).

| Check, 50 full-task starts from seed 4242, tau 0.16 s, delay 5 | Result |
|---|---|
| Grab and placement observations vs the environment's, same fingertip input | 6e-8 (float32 rounding) |
| Fingertip target and gripper angle updates | identical |
| Chain grab4 + place2 / grab5 + place2 acting on the deploy observations | 46/50 / 47/50, the same as `eval_chained` |

- **One known difference, left as is.** StackCubes reads the fingertip from MuJoCo's `site_xpos`, which `mj_step`
  computes before integrating. So in training the fingertip is one physics step (2 ms) older than the joint angles.
  - The runner uses FK of the current joints.
  - The gap is a median of 0.2 mm (99th percentile 1.1 mm), and it changes actions by a median of 0.004.
  - This is part of what the policies trained on, so the environment is not changed.

**Closed loop in the dry world** (DryArm's driver timing, MuJoCo cubes and motors, a fake tracker at 60 Hz with
33 ms latency and the training noise, and hands sliding the cubes as in training). An episode allows up to 3
attempts; "stacked" is StackCubes' own test on the true state.

| Runner version | Policies | Starts | Stacked (simulator) | First attempt | Notes |
|---|---|---|---|---|---|
| First version: moved away while the fingers opened | grab4 + place2 | 10, seed 4242 | 4/10 | 4/10 | 3 of the 7 stacks it declared were dragged off-centre afterwards |
| Same, no hands | grab4 + place2 | 10, seed 4242 | 4/10 | 3/10 | 7/7 declared stacks were right in place; 3 spoiled after |
| **Final release sequence** | grab4 + place2 | 10, seed 4242 | **8/10** | 6/10 | every declared stack confirmed in place and at WATCH |
| Final, re-aim deadband 0.002 rad (default 0.008) | grab4 + place2 | 10, seed 4242 | 8/10 | 4/10 | no gain: keep 0.008, as system ID measured |
| **Final** | grab4 + place2 | 10, seed 5242 | **9/10** | 5/10 | |
| Final, tracker dropouts (3% per frame, 0.05–0.3 s) | grab5 + place2 | 10, seed 4242 | 9/10 | 6/10 | 10/10 declared stacks right in place; 7 holds for unseen cubes, all resumed; one stack carried up wedged in the open gripper |
| Same, gripper opened fully before rising | grab5 + place2 | 10, seed 4242 | 7/10 | 6/10 | 6/6 declared stacks intact at WATCH; 3 attempts ran out of time with red left on green (1 centred) |

- **Commands, dry:** start → pause mid-grab → resume → rest, and SIGTERM mid-lift (red carried back, arm to rest),
  all as intended.
- **Four bugs found and fixed in the dry world:**
  1. The safety floor sat above the policies' own lowest target, so a normal grasp move was refused. Both are now
     relative to the table.
  2. The fallback could put red down on green, which is a scripted stack. It now carries red back to the table.
  3. The arm moved away while the fingers were still opening, dragging the released cube.
  4. The table height came from one noisy reading, and then from a median that mixed in readings of red elsewhere.
     It is now the median over the cube resting where it is.
  5. A cube turned about 45° to the fingers measures 70.7 mm across its corners. It stayed wedged in the policies'
     70 mm opening and was carried up to WATCH. The runner now opens fully (80 mm) before rising, and stops if the
     fingers stall.
- **What it means:** with the final runner, grab4 + place2 stacked 17/20 in the dry world, and grab5 + place2 with
  tracker dropouts stacked 16/20. The same pairs get 92% and 94% in StackCubes at the fitted lag.
  - Every stack the runner declared was right by the simulator's test in place.
  - With the final release sequence, none was disturbed on the way to WATCH.
  - Repeats of the same seed differ, because the dry world runs in real time.
  - 20 episodes is too few to tell these apart.
  - The dry world differs from training in ways that matter: the arm follows quintic re-aims capped at 3 rad/s²
    (what the real driver does), not a first-order lag; the cubes start from WATCH-pose episodes; and failed attempts
    are retried.
  - The grab took a median of about 6 s in the first runs, against 3.1 s in StackCubes at the same fitted lag. For
    the policy's larger moves, the re-aim scheme is probably slower than the tau 0.16 s it fits on system ID's
    ±3 cm steps.
- **Not verified, needs the real arm:**
  - the gripper's real timing (the simulated gripper closes in 0.09 s);
  - the base-joint effort guard during the policies' re-aims (known to trip on driver 1.8.3 in fast following);
  - the tracker with the arm and gripper in view;
  - the real table and calibration;
  - whether the real gripper reaches 36 mm per finger reliably (its range is 0–40 mm).
  - DryArm checks joint limits, the workspace box and the floor, but not tracking error or effort.

## Fine-tuning at the real arm's lag (RTX 5090 PC, 17:51–19:28): grab6 and place5, no improvement

- **Runs (Task 1):** both resumed from the final saves, with the lag narrowed to the real values plus margin (tau
  0.10–0.25 s, delay 3–6 steps), 3% observation dropouts, and 8 parallel copies each.
  - grab6: grab5 + 8M steps, to 28M.
  - place5: place4 + 10M steps, to 62M. Same starts as place4 (carry 40%, near 30%, place 30%).
  - Speed on the 9950X: 3.4k and 2.3k steps/s side by side (lab PC: 2.1k and 1.4k with 12 copies each). place5
    slowed to about 1.7k once another user's jobs started.
- **Evaluation (Task 2):** the standard 150 full-task starts (seeds 4242, 5242, 6242 × 50). The evaluation here
  reproduces the lab PC's numbers exactly (grab5 + place2 141/150; place4 alone 86.0%).

| Chain, 150 full-task starts | tau 0.16 s, delay 5 | delay 4 | old trained lag |
|---|---|---|---|
| **grab5 + place2 (deploy candidate)** | **94.0% (141/150)** | 92.0% (138/150) | 93.3% (140/150) |
| grab5 + place5 | 93.3% (140/150) | | |
| grab5 + place5 at 54M (2M into the run) | 94.7% (142/150) | | |
| grab6 + place2 | 89.3% (134/150) | | |
| grab6 + place5 | 89.3% (134/150) | 86.7% (130/150) | |

| Single skill, 150 episodes, tau 0.16 s, delay 5 | Before | After |
|---|---|---|
| Grab | grab5 87.3% (knocked 3) | grab6 81.3% (knocked 8); 80.7% at 22M |
| Placement, held starts | place4 86.0% | place5 84.7% (86.7% at 54M) |

**Failures by phase**, chain at tau 0.16 s, delay 5, out of 150:

| Pair | Knocked in grab | Grab timeout or out | Knocked in placement | Placement timeout |
|---|---|---|---|---|
| grab5 + place2 | 2 | 1 | 3 | 3 |
| grab5 + place5 | 2 | 1 | 6 | 1 |
| grab6 + place2 | 8 | 1 | 3 | 4 |
| grab6 + place5 | 8 | 1 | 5 | 2 |

- **Neither run beats the baseline.**
  - grab6 got worse. The narrower, longer-delay lag range made the grab knock cubes over about four times as often,
    and success never recovered during the run (training success hovered around 80–87%).
  - place5 is the same as place4 within noise (±2.5% per 150 episodes).
- **Deploy candidate unchanged:** `runs/rl/grab5_ppo/model` + `runs/rl/place2_ppo/model_42000048`.
  - grab6 and place5 (`model.zip` + `vecnormalize.pkl`) are kept on the 5090 PC only and are not needed.
- **Logs:** `runs/rl/lageval/t1_*` and the runs' `progress.log` / `config.json`.
- **What to try instead:** model the arm's real re-aim scheme in the simulator before training on it again (below).

**The arm's real re-aim scheme, for the simulator.** System ID showed the real arm follows SafeArm's re-aims as
DryArm models them, so this is what an opt-in `lag_mode` should reproduce:
- **When:** re-aim at most every 0.15 s (every 3rd 20 Hz step), and only if the joint target moved more than
  0.008 rad (0.002 was tried in the dry world: no gain).
- **How:** each re-aim is a quintic from the current commanded (q, v) to (q_goal, 0, 0), so velocity stays
  continuous. Its duration is T = max(0.3, 1.875·max|dq|/1.0, √(5.77·max|dq|/3.0), 1.875·|dq|/(0.5·v_max)), the
  formula `SafeArm.follow_goal` uses (DryArm takes v_max = 3.0).
- **Physics:** each substep sets `ctrl` to the quintic's position. `delay_steps` then covers perception only
  (1–2 steps), because the arm's own delay is modelled directly.
- **Compatibility:** existing modes must stay bit-identical (check: grab4 + place2 at seed 4242, tau 0.16 s,
  delay 5 gives 46/50; grab5 + place2 gives 47/50).
- **Leave as is:** the environment's fingertip reading (`site_xpos`, one 2 ms substep old); the policies trained on
  it.
- **Why:** in the real-time dry world, the chain stacked 16–17 of 20, and grabs took about 6 s against 3.1 s in
  StackCubes at the fitted lag. The re-aim scheme looks slower for the policies' moves than the tau 0.16 s it fits
  on system ID's small steps.

## The re-aim arm model (lab PC, 19:30–20:35): the deployment path is slower than the fitted lag

**What it is.** `lag_mode="reaim"` in both envs (`rl.envs.moving_cube.Reaim`) models what the real arm does when a
policy drives it through `SafeArm.follow_goal`, the deployment path:
- re-aim at most every 3rd control step (0.15 s), and only when the joint goal moved more than 0.008 rad;
- each re-aim is a quintic from the commanded position and velocity to the goal at rest, with
  T = max(0.3 s, 1.875 |dq| / 1 rad/s, sqrt(5.77 |dq| / 3 rad/s²), 1.875 |dq| / (0.5 v_max)), as in
  `follow_goal` and `DryArm`;
- the position actuators follow the quintic every physics step;
- `delay_steps` then covers perception only (1–2 steps).

The 5090 session specified it from `DryArm` and its dry-world runs.

**Checked against the real arm.** Replaying the system-ID recording through `Reaim`, with no fitted parameters and
ideal motor tracking, gives a joint RMSE of **0.29°** overall: 0.13° on the steps and 0.40° on the sweep. The fitted
first-order lag gives 0.33° (0.27° / 0.32°). Adding motor lag on top doesn't improve the fit, so the motors follow
their commands closely, and the re-aim model is the more faithful one.

**Older modes are unchanged.** The 4242-seed chain checks still give exactly 46/50 (grab4 + place2) and 47/50
(grab5 + place2) at tau 0.16 s and delay 5.

**The deploy candidate under the re-aim model:** grab5 + place2 scores **52.0% (78/150)**, against 94.0% under the
fitted lag.
- The grab hands over after a median **6.2 s** instead of 3.2 s. That matches the about 6 s the 5090 saw in its
  real-time dry world.
- Failures are mostly placement timeouts within the 15 s episodes, plus knock-overs.

**Why it's slower: the re-aims, not the arm.** Pushing the target at full action for 1 s (open loop, same starts):

| Arm model | moved after 0.5 s | after 1 s | peak fingertip speed |
|---|---|---|---|
| fitted lag (tau 0.16 s) | 13.0 cm | 22.5 cm | 0.35 m/s |
| re-aim, 3 cm leash | 3.2 cm | 8.6 cm | 0.13 m/s |
| re-aim, 6 cm / 10 cm leash | 3.5 / 3.5 cm | 10.5 / 11.2 cm | 0.16 / 0.18 m/s |

- **Why:** every re-aim plans a move that ends at rest. The 3 rad/s² acceleration cap stretches even a 0.1 rad
  re-aim to about 0.44 s, and the next re-aim comes 0.15 s later. So the arm keeps braking.
- **The first-order lag fitted on system ID's small steps** has no speed or acceleration limit, which is why it
  overstated the speed for larger moves.
- **A longer leash doesn't fix it,** and the current policies collapse with it: **3% at 6 cm, 0% at 10 cm**, almost
  all knock-overs. The policy doesn't observe its target, so a far-running target makes it overshoot.

**Fixes:**
1. **Fine-tune both policies under the re-aim model** (running): grab7 from grab5 (8M steps) and place6 from place2
   (10M steps). Both use 12 envs, delay 1–2 steps, dropouts, the 3 cm leash and **lr 1e-4**.
   - `train_ppo --resume` used to ignore `--lr` and restart the saved schedule at full rate (3e-4). It now applies
     `--lr`, and the default equals the rate every earlier run used.
   - That restart may be why the 5090's grab6 lost skill.
2. **A faster deployment path** (needs the real arm). Driver 1.8.3's `set_arm_positions` accepts
   `goal_feedforward_velocities`. A re-aim could then end moving at the target's velocity instead of at rest, which
   removes the stop-and-go. It would need a matching sim model and a system-ID run on the arm.
3. **Driver and firmware 1.9.3+**, which handle fast re-commanding.

## Feedforward re-aims and the speed ceiling of driver 1.8.3 (lab PC, 20:40–22:15)

**What was built (opt-in; plain re-aims stay the default everywhere):**
- `robot/reaim.py` is now the one re-aim implementation, shared by `SafeArm.follow_goal`, `DryArm` and the
  simulator. The plain path is unchanged: the 4242-seed checks gave exactly 40.0% (reaim) and 94.0% (target) with
  the same failures, and the real-arm replay gave 0.29°.
- **Feedforward re-aims** (`follow_goal(..., goal_velocity=)`; `--ff` in `scripts/sysid_record.py` and
  `rl/deploy.py`; `lag_mode="reaim_ff"` in the envs). Each re-aim aims at where the goal will be when the move ends
  and arrives with a share of the goal's velocity (gain 0.5), through the driver's `goal_feedforward_velocities`.
  1.8.3 accepts that argument.
- **Safety:**
  - the planned quintic and a 0.15 s coast past its end are path- and limit-checked;
  - near a joint limit or the floor, it falls back to a plain re-aim instead of stopping;
  - a goal jump (more than 0.06 rad in one step) clears the velocity estimate, so jumps are never extrapolated;
  - once the goal stops, one more re-aim ends at rest.

**Results:**

| Test | plain re-aims | feedforward |
|---|---|---|
| Goal moving steadily at 0.2 / 0.4 / 0.6 rad/s: lag | 0.16 / 0.28 / 0.42 s | about 0 (gain 1) |
| Env push, 1 s at full action: distance, peak speed | 8.6 cm, 0.13 m/s | 12.7 cm, 0.22 m/s (gain 1) |
| Dry system-ID test, ramps: tracking error | 1.12° RMSE | 0.90° (gain 0.5), 1.37° (gain 1) |
| Dry system-ID test, 0.2–1.2 Hz sweep: worst error | 9.9° | 15° (gain 0.5), 25° (gain 1) |
| Chain grab5 + place2, 150 starts | 52.0% | 28.0% (gain 0.5), 2.0% (gain 1) |

- **Feedforward helps on steady motion but overshoots** when the goal slows or reverses. With re-aims only every
  0.15 s, the goal-velocity estimate lags, and extrapolating it costs precision exactly where grasping needs it.
- **In the real-time dry world** (`rl/deploy.py --dry`, which retries with no time limit), grab5 + place2 with plain
  re-aims stacked **9/10** (5/10 on the first attempt), median 20.8 s per episode, scenes from seed 7000.
  - With `--ff`, the same scenes stopped after 2 episodes (1 stacked). The runner's scripted rise found no IK
    solution near the inner edge of reach (15 cm from the base) after a failed placement. The runner held and went
    to rest, as it should.
  - So with plain re-aims the deploy candidate is slow but viable. The 52% above counts single 15 s episodes with no
    retries.
- **The current policies knock cubes over under feedforward.** It would need policies trained with `reaim_ff`, and
  a real-arm check that the driver follows feedforward velocities as `DryArm` models it.
  - That system-ID run is prepared: `docs/rl/DEPLOY_CHECKLIST.md`, "Optional: system ID with feedforward
    re-aims".
- **Raising the acceleration cap** (plain re-aims) helps a little: 0.13 → 0.16 / 0.17 m/s peak at 5 / 8 rad/s².
  It was 3 because sharper re-aims tripped the base joint's effort guard.

**Fine-tuning under the plain re-aim model (grab7, from grab5, 8M steps, lr 1e-4):** no gain.
- grab7 alone: 66.7% (grab5: 66.0%).
- Chain grab7 + place2: 55.3% (83/150; grab5 + place2: 52.0%).
- Training success stayed at about 68% throughout. Most failures are timeouts: the arm can't catch a cube moving at
  up to 12 cm/s within the 8 s episode.

**Conclusion.** Through driver 1.8.3's re-aims, every knob (feedforward, leash, acceleration cap) tops out at about
0.13–0.22 m/s of fingertip speed. That is about as fast as the simulated hand moves the cubes. The ceiling is
structural: re-aims at most every 0.15 s, each at least 0.3 s long, because re-commanding faster is unstable before
driver 1.9.3. **Upgrading the driver and firmware to 1.9.3+** is the change that would let the policies run at the
speed they were trained for: stream targets at 50–100 Hz, with no re-aims. After the upgrade, re-run system ID.

## Fine-tuning for the real arm's motion: place6, hand-over starts (22:15–22:30)

**place6** (from place2, 10M steps under the plain re-aim model, lr 1e-4). All results are under the re-aim model,
150 starts each:

| Evaluation | Before | After |
|---|---|---|
| Placement alone, held starts | place2 66.0% | **place6 72.7%** |
| Placement from recorded chain hand-overs | place2 61.3% | **place6 87.3%** |
| Chain grab5 + place2 → grab5 + place6 | 52.0% | **74.7%** (112/150) |
| Chain grab7 + place6 | | 72.7% (109/150); 77.3% with 22.5 s episodes |
| Chain grab5 + place2 with 22.5 s episodes | 69.3% | |
| Chain grab7 + place6 under the old fitted lag | (grab5 + place2: 94.0%) | 84.7% |

- **Placement was the weak half under the real arm's motion,** and fine-tuning it there gained about 20 points.
- **The cost:** some performance under the old fitted-lag model, which the real arm doesn't follow.
- **Much of the remaining gap is time.** The episodes are 15 s; the deploy runner retries with no time limit, and
  in the dry world it already stacked 9/10.
- **Remaining failures** (grab7 + place6): grab phase 24 (14 knock-overs, 9 timeouts), placement 17 (9 timeouts,
  8 knock-overs).

**Hand-over starts.** `rl/record_handover.py` runs the chain (grab policy, scripted lift) exactly as
`rl/eval_chained.py` does and saves the full simulator state at each hand-over with red held. StackCubes' new
family "handover" (`handover_file=`) restores one, with the episode clock at the step the chain had reached; green's
push is timed from the hand-over.
- 2,405 hand-overs were recorded from 3,000 runs (grab5 and grab7, plain re-aims; median at step 118 of 300).
- Older settings are unchanged: the 4242-seed checks still give 94.0% and 40.0% with the same failures.

**Running now** (12 envs each, plain re-aims, lr 1e-4):
- **place7**, from place6: 50% of its starts are hand-overs (those after step 240 are left out), 50% carry, near
  and place.
- **grab8**, from grab7: 12 s episodes instead of 8, with the hand at most 10 cm/s as in the chain's starts. Its
  training success started at 80%, against about 68% with 8 s episodes, so many of the grab's failures were the
  time limit.

## place7 and grab8 results, and where the plateau is (00:05)

All results are under the plain re-aim model (the real arm on driver 1.8.3), 150 full-task starts:

| Pair | Chain | Grab-phase knock-overs | Grab hands over after |
|---|---|---|---|
| **grab5 + place6** | **74.7%** (112/150) | 5 | 6.1 s |
| grab7 + place7 | 74.0% (111/150) | 14 | 5.6 s |
| grab7 + place6 | 72.7% (109/150) | 14 | 5.6 s |
| grab8 + place7, 22.5 s episodes | 70.0% (105/150) | 25 | 5.1 s |
| grab8 + place6 | 66.0% (99/150) | 24 | 5.0 s |
| grab8 + place7 | 64.0% (96/150) | 24 | 5.0 s |
| grab8 + place7 under the old fitted lag | 82.0% (123/150) | 12 | 3.2 s |

- **grab8** (from grab7, 12 s episodes) is better alone but worse in the chain.
  - Alone: 72.0% at 8 s, 77.3% at 12 s, median 5.0 s (grab7: 66.7% / 75.3%, 5.9 s).
  - In the chain it is faster but tips red far more often.
- **place7** (half its starts from recorded hand-overs) did not improve on place6.
  - From hand-overs: 82.7% (place6: 87.3%). Held starts: 74.7% (72.7%).
  - Training success stayed at about 80%.
- **What the grab knock-overs are** (grab7, 150 starts, 14 knock-overs, instrumented):
  - 13 were red tipped by the fingers during the final descent and close; 1 was green;
  - 11 of the 14 happened with the hand already idle.
  - So the cause is grasp precision with the re-aiming arm (fingers arriving late or misaligned), not the hand.
  - grab5, trained with the old lag model, tips red much less in the chain (5 of 150).

**Plateau.** Under the deployment path of driver 1.8.3, every fine-tune lands at about 72–75% on 15 s chain
episodes. In the real-time dry world, which retries with no time limit, the chain stacked 9/10.

**Recommended pairs:**
- Real arm on driver 1.8.3 (plain re-aims): **grab5 + place6** (`runs/rl/grab5_ppo/model`,
  `runs/rl/place6_ppo/model`).
- After a driver/firmware 1.9.3+ upgrade (fast streaming, close to the fitted-lag model): **grab5 + place2** (94% in
  simulation).

## Grasp-precision fine-tunes: grab9 and grab10 (00:15–02:00)

**Setup.** Both resumed from grab5 (the most careful grabber), under the plain re-aim model, at lr 5e-5 to stay close
to it, 8M steps, 12 envs.
- **Two new opt-in penalties in `MovingCubeGrasp`.** Defaults are unchanged: grab5 alone still gives exactly
  66.0% with the same counts.
  - `bump_penalty`: 0.5 per step while red moves more than 3 mm or tilts more than 5° and is neither held nor
    pushed by the hand, i.e. the fingers bumped it.
  - `knock_penalty`: 10 for knocking red over (was a fixed 5).
- **grab9** keeps the 8 s episodes. **grab10** uses 12 s episodes and a hand of at most 10 cm/s, like the chain's
  starts.

| Plain re-aim model, 150 starts | grab5 (before) | grab9 | grab10 |
|---|---|---|---|
| Grab alone, 8 s episodes | 66.0%, 0 knock-overs | 62.0%, 5 | 66.7%, 6 |
| Grab alone, 12 s episodes | **80.0%, 0 knock-overs** | 74.0%, 8 | 76.0%, 6 |
| Chain with place6, 15 s episodes | **74.7%**, 5 knocked in grab | 72.0%, 10 | 72.7%, 13 |
| Chain with place6, 22.5 s episodes | **82.7%** (124/150), 6 | 76.7%, 10 | 82.0%, 13 |
| Chain with place6, old fitted-lag model | | 72.0% | 85.3% |

- **Neither beat grab5.** Every grab fine-tune under the re-aim model so far (grab7, grab8, grab9, grab10) has tipped
  red more often than grab5, even with explicit bump and knock penalties and a halved learning rate.
- **A plausible reason:** under the slow, re-aiming arm, the policy learns to push its target further ahead to make
  up for the lag, and that overshoots at the grasp. The penalties did not outweigh it. Not investigated further.
- **The time limit is most of the gap.** grab5 alone goes from 66% at 8 s to 80% at 12 s. Its chain with place6 goes
  from 74.7% at 15 s to 82.7% at 22.5 s. The deploy runner has no time limit.

**The deploy candidate for driver 1.8.3 stays grab5 + place6** (82.7% with 22.5 s episodes; 9/10 in the dry world,
with grab5 + place2). Further grab fine-tuning under these dynamics looks like diminishing returns. The faster-arm
path (driver 1.9.3+) is the larger lever.

## Sim-to-real variations: sensitivity and a robustness fine-tune (02:30–)

**New, opt-in** (`rl/envs/variations.py`; no random draws when off; the 4242-seed checks and grab5 alone reproduce
exactly). Each episode draws:
- `calib_error=(m, deg)`: one camera-to-base calibration error, a turn about the workspace centre plus a shift, applied
  to every cube reading;
- `obs_bias=m`: a constant tracker offset per cube;
- `friction_scale=(lo, hi)`: friction of the cubes and the table together (MuJoCo uses the larger of the two
  surfaces' friction);
- `mass_scale=(lo, hi)`: each cube's mass and inertia.

**Sensitivity of grab5 + place6** (plain re-aims, 15 s, 150 starts each):

| Variation | Success | vs 74.7% |
|---|---|---|
| Calibration error 3 mm / 1° | 70.0% | −4.7 |
| Calibration error 5 mm / 2° | 63.3% | −11.4 |
| Tracker bias 4 mm per cube | 66.7% | −8.0 |
| Friction ×0.5–1.5 | 63.3% | −11.4 |
| Mass ×0.5–2.0 | 66.7% | −8.0 |
| All combined (3 mm / 1°, 3 mm bias, friction ×0.6–1.4, mass ×0.6–1.6) | **54.7%** | **−20.0** |

- Every variation costs something, and together they cost 20 points.
- Most added failures are grab timeouts and placement knock-overs: misread positions mean misaligned fingers.

**Fine-tunes running,** trained with calibration error 4 mm / 1.5°, 4 mm bias, friction ×0.5–1.5 and mass ×0.5–2.0,
under plain re-aims:
- **place9**, from place6, lr 1e-4, 10M steps;
- **grab11**, from grab5, lr 5e-5, with the bump and knock penalties, 8M steps.

**Results** (plain re-aims, 150 starts):

| Pair | All variations | No variations | Stronger calibration error (4 mm / 1.5°) |
|---|---|---|---|
| grab5 + place6 (before) | 54.7% | 74.7% | 54.0% |
| grab5 + place9 | 52.7% | 72.0% | 56.0% |
| grab11 + place9 | 57.3% | 68.7% | |
| grab11 + place6 | **59.3%** | | |

- **Robustness training barely helped:** at best +4.6 points with the variations on (about the noise), and a few
  points lost without them. Training success hardly moved (place9 71 → 76%, grab11 57 → 61%).
- **Why, most likely:** the policies see each cube relative to the fingertip, the cube from the camera and the
  fingertip from the joint angles. A calibration error or tracker bias moves the cube relative to the real fingers,
  and nothing the policy observes reveals it, so training cannot correct it.
  - The fix is on the perception side: keep calibration tight (3 mm already costs about 5 points), or close the
    loop visually by tracking the gripper with the camera, so the cube is measured relative to the fingers it
    actually sees.
- **Friction and weight** are physical, so a policy can learn to be gentler. Friction was the larger single cost
  (−11 points, mostly placement knock-overs).
  - **place10** (from place6, friction ×0.5–1.5 and weight ×0.5–2 only, 15M steps, 20 envs) tests this.

**place10 results** (06:25; plain re-aims, 150 starts):

| Pair | Friction and weight varied | No variations |
|---|---|---|
| grab5 + place6 | 59.3% (22 knocked in placement) | 74.7% |
| **grab5 + place10** | **61.3%** (18 knocked in placement) | **75.3%** |
| grab11 + place10 | 59.3% | |
| grab11 + place6 | | 69.3% |

- place10 is at least as good as place6 with no variations, and slightly better with them (+2 points, 4 fewer
  placement knock-overs). Within noise, but at no cost.
- grab11 is worse than grab5 without variations (69.3% vs 74.7%). The grab stays grab5.

**Pairs to use (simulation; nothing RL has run on the real arm yet):**
- Driver 1.8.3 (plain re-aims): **grab5 + place10** (`runs/rl/grab5_ppo/model`, `runs/rl/place10_ppo/model`); place6 is
  equivalent.
- After a driver 1.9.3+ upgrade (fast streaming): **grab5 + place2**, then re-check under the new arm behaviour.

## What a driver upgrade would buy, in simulation (morning, 2026-10-18)

An upgraded arm that follows streamed targets closely is modelled as `lag_mode="target"` with a small lag (perception
delay 1–2 steps). That is what driver 1.9.3 should allow, if it streams smoothly. Chains on the same 150 starts; "all
variations" = calibration error 3 mm / 1°, tracking bias 3 mm, friction ×0.6–1.4, mass ×0.6–1.6:

| Arm model | Pair | No variations | All variations |
|---|---|---|---|
| Driver 1.8.3, plain re-aims | grab5 + place10 | 75.3% | 55–61% |
| Streaming, 0.05 s lag | grab5 + place2 | **90.7%** | **86.0%** |
| Streaming, 0.1 s lag | grab5 + place2 | **92.0%** | **90.7%** |
| Streaming, 0.05 s lag | grab5 + place10 (tuned for the slow arm) | 42.7% | 37.3% |

- **With a fast arm, the real-world variations cost only 1–5 points** (20 on the slow arm): the arm corrects small
  errors mid-motion.
- **Policies are specific to the arm model.** place10 times out on a fast arm (53 placement timeouts), so after an
  upgrade use place2.
- **This assumes the upgraded driver streams smoothly,** which only a real-arm test can confirm. Trossen does not give
  the rate its 1.9.3 fix covers.

## Streaming for driver 1.9.3 (evening, 2026-10-18; the upgrade is planned for 2026-10-19)

**`robot/stream.py`:** a trajectory generator on the computer at the stream rate (100 Hz by default).
- Each tick, the commanded joints move toward the goal with a speed cap (1 rad/s) and an acceleration cap
  (4 rad/s²), all joints in step, braking in time to stop at the goal.
- The driver gets each point with its velocity as feed-forward.
- Unit check: a 0.5 rad step settles in 0.67 s with 0.3° overshoot, and a goal moving at 0.4 rad/s is tracked with
  0.05 s lag (the 1.8.3 re-aims: 0.28 s).

**Wired in, opt-in, and dry-tested:**
- `SafeArm.stream_begin` / `stream_tick`:
  - each new goal is checked like a move, and so is every step;
  - a goal that fails is not followed (the last good one is kept), and goals failing for 0.5 s raise SafetyStop;
  - `follow_check` guards as before, its tracking window covering the last 0.3 s of points.
- `DryArm`: the same, with the driver's straight line between points modelled.
- `scripts/sysid_record.py --stream HZ`, and `rl/sysid_fit.py`, which replays a streaming recording through the
  generator. Dry at 50 and 100 Hz: the model replays the dry arm to 0.03–0.18°; the sweep tracks a median 5 mm off
  target (real arm on 1.8.3: 16 mm); equivalent to a 0.07 s lag plus a 0.05 s delay.
- `rl/deploy.py --stream HZ`: one point per tick on a paced clock. Two deploy fixes found in dry streaming runs
  apply to both modes:
  - IK goals are clipped to the joint limits, as the training env's IK does;
  - a goal whose fingertip IK lands under SafeArm's floor is raised just above it. The floor is 3 mm under the
    target floor, and the IK can miss the target by that much near the table.
- `lag_mode="stream"` in both envs: the generator at 100 Hz inside each 20 Hz step, with the acceleration cap
  (3–6 rad/s²) and approach time constant (0.04–0.1 s) randomized per episode. Old modes reproduce exactly.

**Driver 1.9.3 compatibility (no arm):** every driver call the code makes exists in 1.9.3 with the same arguments, in a
throwaway env. Details and the upgrade steps are in [FIRMWARE_UPGRADE.md](FIRMWARE_UPGRADE.md). New: `scripts/backup_arm_config.py`
saves the controller's configuration with the current driver, before flashing.

**Correction to the earlier upgrade estimate.** The 86–91% for an upgraded arm came from the first-order "target"
model, which has no speed or acceleration limit. Under the realistic streaming model, grab5 + place2 scores 58.0%
(87/150), and 54.7% with the real-world variations. Grabs take about 4 s, and most failures are placement knock-overs
(21) and timeouts (21). The policies were never trained for an arm with these limits, so:
- **grab12** (from grab5) and **place11** (from place2) are fine-tuning under `lag_mode="stream"` with the real-world
  variations (calibration error 4 mm / 1.5°, 4 mm bias, friction ×0.5–1.5, mass ×0.5–2). They use 12 envs,
  lr 1e-4, the bump and knock penalties for the grab, and 8M / 10M steps.

**Results** (21:10; streaming model, 150 starts; variations as in the robustness tests):

| Pair | Streaming (acceleration 3–6 rad/s²) | With all variations |
|---|---|---|
| grab5 + place2 (before) | 58.0% | 54.7% |
| grab5 + place11 | 76.7% | 66.0% |
| grab12 + place2 | 81.3% | |
| **grab12 + place11** | **89.3%** (134/150), grab 3.2 s | **76.7%** (115/150) |
| grab12 + place11, acceleration 3 rad/s², tau 0.08 s | 86.7%, grab 3.7 s | |
| grab12 + place11, acceleration 6 rad/s², tau 0.05 s | **93.3%** (140/150), grab 2.7 s | |

- **Fine-tuning for the streaming arm gained 31 points,** and both halves contributed.
- **A higher acceleration limit helps:** 87% at 3 rad/s², 93% at 6. Raise `robot/stream.py`'s ACCEL from 4 if the
  force guard stays quiet on the real arm.
- **Against driver 1.8.3** (best: grab5 + place10, 75% clean, 55–61% with variations), streaming with these
  policies is +14 points clean and about +17 with the variations, with grabs about twice as fast.
- **Pairs to use:**
  - streaming (driver 1.9.x): **grab12 + place11** (`runs/rl/grab12_ppo/model`, `runs/rl/place11_ppo/model`);
  - re-aims (1.8.3, or if streaming is not smooth): grab5 + place10.

## Upgrade day (2026-10-20): firmware 1.9.4, driver 1.9.3, and streaming on the real arm

**Upgrade.** Both lab arms were flashed to firmware 1.9.4 from a Linux PC over USB: the second arm (192.168.1.2)
and the project arm (192.168.1.3).
- `trossen-arm==1.9.3` is now in rlwx and rlwx-train. `trossen183` is a separate env with 1.8.3, for any 1.8.3 arm.
- Each arm's configuration was saved with the old driver before flashing and with the new one after. The
  backups are in `runs/arm_backup/`, with copies on the Windows Desktop.
- `scripts/compare_arm_config.py`: **factory calibration unchanged on both arms.** 0 fields changed, 0 removed, and
  7 new `position_offset` fields, all 0. No homing needed.
- **Lesson:** after flashing, each controller answered on the network only after a power-cycle with Ethernet
  plugged in.

**System ID on the project arm** (same 35 s test each time; fingertips never below 112 mm; no guard trips):

| Mode | Lag fit (tau + delay) | Model match, no fitted parameters | Sweep: off target, median / 95% | Peak base-joint effort |
|---|---|---|---|---|
| Re-aims, 1.8.3 (2026-10-17) | 0.16 + 0.16 s | re-aim model 0.29° | 16 / 39 mm | 1.08 Nm |
| Re-aims, 1.9.4 | 0.15 + 0.16 s | re-aim model 0.30° | 16 / 39 mm | 1.10 Nm |
| Streaming 20 Hz | 0.09 + 0.06 s | streaming model 0.16° | 5 / 35 mm | 1.77 Nm |
| Streaming 50 Hz | 0.08 + 0.06 s | streaming model 0.13° | 5 / 28 mm | 1.97 Nm |
| **Streaming 100 Hz** | **0.08 + 0.05 s** | **streaming model 0.08°** | **5 / 27 mm** | 1.97 Nm |

- **Re-aims behave exactly as on 1.8.3.** The upgrade did not change the arm.
- **Streaming works at every rate.** The simulator's streaming model (`robot/stream.py`, `lag_mode="stream"`, used to
  train grab12 + place11) matches the real arm to within 0.08–0.16°. So the simulated 89% (77% with variations)
  rests on a model the real arm follows.
- **Smoothness.** On a common 20 Hz grid, all streaming rates peak at about 4.1 rad/s² (the generator's limit) and
  about 55 rad/s³ jerk. Re-aims: 2.8 rad/s² and 29 rad/s³. Streaming is brisker by design, not rougher at higher
  rates.
- **Limit to watch:** the base joint's peak effort, 1.97 Nm, against its 2.3 Nm guard. Keep the acceleration at
  4 rad/s². Raising it to 6 (93% in simulation) would probably trip the guard unless the base-joint limit is
  re-examined.
- **Next:** the first RL run with streaming, `rl/deploy.py` with grab12 + place11 at `--stream 100`.

## The arm and camera on the RTX 5090 PC (evening, 2026-10-20): calibration, grip, first RL attempt

The project arm (192.168.1.3, firmware 1.9.4, driver 1.9.3) and the D455 are now connected to the RTX 5090 PC
(native Ubuntu, shared, headless). Everything below was done with the operator at the power switch.

**Setup.**
- `rlwx` built with `scripts/setup_env.sh`. One fix: its env check also matched `rlwx-train`.
- `pyrealsense2==2.58.3.10794` added from PyPI, matching conda-forge's librealsense 2.58.3. On the lab PC the camera
  ran through the Windows streamer, so `rlwx` never had it.
- Perception runs on a virtual display (`~/.local/bin/xvfb-run`): about 60 fps, 7 ms per frame, 16–19 ms from
  capture to pose.
- The live page is reached through an SSH tunnel (`ssh -N -L 8080:127.0.0.1:8080 <user>@<host>`). The browser
  stream is 10 fps at JPEG quality 60, because 30 fps lagged several seconds through the tunnel.

**Camera serial.** `assets/calibration.yaml` said serial `338122302706`, but the camera in front of the arm (its
sticker confirms it) is `151422252817`.
- The cause: `calibrate_camera.py` wrote a hard-coded serial.
- It now records the live serial.
- A **serial check** refuses a camera that isn't the calibrated one:
  - `scripts.web_view` checks at start-up, before opening its port;
  - `rl.deploy`, `robot.controller` and `scripts.follow_cube` check before connecting to the arm, and every
    snapshot after (`robot.grasp.cubes_in_base`);
  - message: "camera X != calibration Y: recalibrate (scripts.calibrate_camera) or connect the calibrated camera";
  - `--ignore-serial` overrides it deliberately (for example, `web_view` while calibrating a new camera);
  - calibration runs are never blocked.

**Recalibration.**
- **Hover check:** with the cube centred by eye under the fingers, the old calibration was off by **10.7 mm** in
  xy, and read cube heights about 1 cm low.
- **Two attempts with the cube placed in the gripper by hand both slipped.** Closing at the rest pose in
  external-effort mode chattered against the cube.
- **New `calibrate_camera pick` phase:**
  1. hover over the cube on the table;
  2. the operator centres it, then presses Resume on the page;
  3. straight down to 36 mm above the table (the table, not the camera's height, sets the grasp);
  4. squeeze, check the grip, a 1 cm slip check, then the 18 poses;
  5. put the cube back, rest, fit.
  Optional flags: `--prior-current` starts from the current calibration, `--grip-z` sets the grasp height.
- **Result:** **18/18 poses, 2.08 mm RMS (max 3.3 mm), rigid fit (depth scale 1.000)**. The camera pose moved
  13.2 mm / 0.46°. Cubes on the table read 28 mm high again (old file: 15–17 mm). Re-checked by eye: centred.
- **Backups:** `assets/calibration_before_*.yaml`.

**The gripper on firmware 1.9.4** (`scripts/grip_lift_test.py`: the arm picks the cube off the table per variant,
lifts it 6 cm, wiggles ±2 cm and holds):

| Grip | Held | Finger | Effort median / peak | Reversals | Slip |
|---|---|---|---|---|---|
| external effort 15 N (`close_gripper` default) | yes | 27.3 mm | 18 / 30 N | 0 | 0.05 mm |
| external effort 25 N | yes | 27.0 mm | 28 / 37 N | 0 | none |
| **effort 18 N** | **yes** | 27.3 mm | **18 / 18 N** | 0 | none |
| position, 2.0 / 1.0 / 0.5 mm inside contact | yes | 26.5 mm | 84 / 84 N | 0 | ≤ 0.12 mm |

- Picked off the table, every mode held. The chatter seen earlier only happened when closing at the rest pose on a
  cube held by hand.
- **Effort 18 N is the gentlest steady grip**, with no closing spike. It is opt-in in the runner:
  `rl.deploy --grip-mode effort --grip-force 18`, and `close_gripper(mode="effort")`. The default is unchanged.
- Position mode squeezes at about 84 N at any offset: the controller saturates.
- The real gripper is **slow**: it closes at about 45 mm/s (0.75 s open to shut) and opens in about 0.2 s. The
  simulated one closes in 0.09 s.

**First real RL attempt** (grab12 + place11, `--stream 100`, effort 18 N): **a deployment-runner bug, not a policy
failure.**
1. The grab policy closes the gripper early, 10 cm above the cube. It does the same in simulation (4 of 4 dry
   attempts), where it is harmless because it reopens before grasping.
2. The slow real gripper passed through the 15–31 mm "holding" range while the arm happened to rise.
3. The runner, which checked only the finger width, counted that as a grasp and lift, and switched to placement
   0.4 s in.
4. The placement policy carried the "held" cube to green and lowered it, so the empty gripper came down on green.
5. Joint 1 felt 6.6 Nm (guard 6.3), and the SafetyStop held the arm. The operator sent it to rest.

Fixed, dry only so far (3/3 stacked):
- a grasp counts only when the fingers have **stopped** at a cube's width with the fingertips **at red**, as
  `StackCubes._red_grasped`;
- a new outcome, **dropped**: the attempt ends if the gripper closes on nothing during lift or place.

**Other fixes from the day.**
- `web_view` now **exits if the camera loop dies**. When the D455 dropped off USB, it had kept serving the last
  cube poses as if live, and consumers timestamp by fetch time.
- Calibration runs reject stale stats and samples, and return to rest on any error.
- The deploy runner's post-release rise tries 4, 3 then 2 cm (a 4 cm rise hit a joint limit at one pose).

**Hardware notes.**
- The D455 must be on a USB 3 port; on USB 2 it cannot stream 848×480 at 60 fps. Twice it re-enumerated after a
  bump and needed a replug.
- The arm's USB Ethernet adapter (on a USB hub) disconnected once (19:14) and needs replugging before the next run.

**Not verified yet:**
- the RL chain on the real arm after the fixes;
- grasp timing with the real gripper speed;
- the joint-1 effort guard under the policies' streamed motions.

**Next (no lab needed):**
- give the simulated gripper the real speed (opt-in), in the training envs and the dry world;
- measure grab12 + place11 with it;
- fine-tune only if it drops.

## The real gripper's speed breaks grab12 + place11 (2026-10-21, simulation only)

**New, opt-in:** `gripper_speed=(closing, opening)` m/s per finger, in `StackCubes` and `MovingCubeGrasp`.
- The actuator target follows the commanded one at most that fast. The commanded state (`grip_goal`) drives the
  binary gripper's logic.
- Off by default, and bit-identical then: the streaming baseline reproduces per seed and per failure count.
- `gripper_speed_scale=(lo, hi)`, also opt-in: per episode, both speeds times a factor drawn from this range (a
  training margin). Nothing is drawn unless it is set.
- Real values from the 2026-10-20 arm log: **closing 0.048 m/s, opening 0.125 m/s** (0.70 s and 0.20 s full
  travel). The simulated gripper otherwise shuts in about 0.09 s.
- The dry world (`rl/dry_world.py`) now uses the real speed by default; `rl.deploy --dry-instant-gripper` gives the
  old behaviour.

**grab12 + place11, standard 150 full-task starts, streaming arm** (`rl.eval_chained`, env kwargs as the 21:10
results plus `gripper_speed`):

| Gripper | Clean | All variations (3 mm / 1°, 3 mm bias, friction ×0.6–1.4, mass ×0.6–1.6) |
|---|---|---|
| As trained | 89.3% (134/150) | 76.7% (115/150) |
| **Real speed** | **12.0% (18/150)** | **13.3% (20/150)** |
| Real close, instant open | 13.3% | |
| Instant close, real open | 83.3% | |
| 2× the real speed | 74.7% | |
| 4× the real speed | 83.3% | |
| Real speed × 0.7–1.3, per episode (`gripper_speed_scale`) | 20.0% (30/150) | |

- **The slow close is the cause.** The grab policy expects the fingers shut within about 0.1 s; the real ones need
  about 0.2 s just to reach the cube.
  - It closes 5.7 times per episode instead of 3.2, and only half its attempts reach the hand-over.
  - Its grasps are weak, so cubes tilt while lifting and placing.
  - Making the hand-over wait for the fingers to stop does not help (4/30 against 5/30). It is the policies, not
    the code around them.
- **Dry world** (deploy runner, `--stream 100`, effort 18 N):
  - instant gripper: 5/5 stacked;
  - real speed: 0/2 (three timeouts, then a SafetyStop: joint 3 targeted at −1.521 rad against the −1.520 limit as
    the policy drove the wrist to its limit).
  - The streaming path should clip goals to SafeArm's margin, not just the IK's.
- **Logs:** `runs/rl/gripspeed/`.

**Next: fine-tune with the real gripper (not started).**
1. grab12, then place11, under the same setup (streaming model, real-world variations), adding
   `"gripper_speed": [0.048, 0.125], "gripper_speed_scale": [0.7, 1.3]`.
2. Judge on the 150 starts at the real speed, against 12% now; and at the fast gripper, against 89%.
3. Then the dry world, then the real arm.

## Fine-tuning for the real gripper speed: grab13 and place12 (lab PC, 2026-10-21, simulation only)

**Merged first:** the RTX 5090 PC's real-arm session (2026-10-20) and its opt-in gripper speed (2026-10-21).
- Nothing on the lab PC had changed since the last sync, so both applied cleanly in order.
- Checks: the gripper-speed option is off by default, and grab12 + place11 on seed 4242 gives exactly 88.0% with the
  same failures as before the merge and on the 5090. `--lr` is honoured on resume (the schedule starts at 1e-4).

**Training:** grab13 from grab12 (8M steps) and place12 from place11 (10M steps), side by side, 12 envs each, about
1,900 and 1,350 steps/s.
- Settings: each run's own settings plus `"gripper_speed": [0.048, 0.125], "gripper_speed_scale": [0.7, 1.3]`, at
  lr 1e-4.
- Training success: grab13 63% → about 73–76% (knock-overs 18% → 11%); place12 71% → about 75–78%.
- Neither stalled by the 3M-step rule. place12 hovered around 75–77% for its last few million steps.

**Chain on the standard 150 full-task starts** (`rl/eval_grip_speed.py`, streaming model; "all" = calibration
3 mm / 1°, 3 mm bias, friction ×0.6–1.4, mass ×0.6–1.6; logs in `runs/rl/gripspeed2/`):

| Pair | Real gripper, clean | Real gripper, all variations | As trained, clean | As trained, all variations |
|---|---|---|---|---|
| grab12 + place11 (before) | 12.0% | 13.3% | 89.3% | 76.7% |
| **grab13 + place12** | **72.0%** (108/150) | **66.0%** (99/150) | 84.0% | 74.0% |
| grab13 + place11 | 70.0% | 60.0% | 88.0% | 82.0% |
| grab12 + place12 | 12.7% | 14.0% | 85.3% | 66.7% |

- **The grab was the bottleneck.** grab12 stays at about 13% whichever placement follows. grab13 lifts the chain to
  70–72% at the real gripper speed.
- **place12 adds a little at the real speed:** 72.0 vs 70.0% clean, 66.0 vs 60.0% with variations, mostly fewer
  placement knock-overs (26 vs 37 with variations). With the as-trained fast gripper, place11 is slightly better.
- **Remaining failures at the real speed** (grab13 + place12, clean): knocked over in placement 19, in the grab 13,
  in the lift 3, placement timeouts 7. The grab hands over after a median 3.2–3.9 s.

**Streaming path fix** (`robot/arm.py`, `robot/controller.py`; found by the 5090 in a dry run, where joint 3 targeted
-1.521 rad against SafeArm's -1.520):
- `stream_tick` clips each goal to the arm's own joint limits (the IK clips to the model's, which differ by about a
  mrad).
- It also clips each generated point, stopping the generator there (it overshoots its goal by a few mrad).
- A goal 1 mrad past joint 3's limit is now followed to the limit with no SafetyStop. Normal goals are unchanged. The
  generator itself (`robot/stream.py`, the simulator's model) is untouched.

**Dry run, grab13 + place12** (deploy runner, real gripper speed, `--stream 100 --grip-mode effort --grip-force 18`):
- **4/5 stacked** by the simulator's check, 4/5 on the first attempt, median 13.6 s per episode, no SafetyStops.
- One "dropped" first attempt was caught by the new guard, and the retry stacked (the simulator later found that stack
  shifted).
- The 5090's dry run of grab12 + place11 at the real speed stacked 0/2.

**Next on the real arm:** `rl.deploy runs/rl/grab13_ppo/model runs/rl/place12_ppo/model --stream 100 --grip-mode effort
--grip-force 18 --ip 192.168.1.3`. Placement knock-overs are the main remaining failure in simulation.

## First real-arm runs of the RL chain (RTX 5090 PC, 2026-10-31)

**Setup:**
- grab13 + place12, with `--stream 100 --grip-mode effort --grip-force 18`.
- The bench and camera were calibrated on 2026-10-20 (2.08 mm RMS). A pick-measured xy correction was added (below).
- **The policies are unchanged.** Every fix below is in the runner or the arm wrapper.
- The operator stood at the power switch throughout and filmed the attempts.

**Result:** 17 attempts in five runner sessions.
- **The first 3 attempts found hardware problems** (below). Each was fixed before the next session.
- **The 14 attempts after that (sessions 4 and 5):**
  - **the grab policy picked up red every time**, in about 2 s (1.4–3.3 s);
  - **it set red down on green and let go in 11 of the 14**;
  - **4 stacks met the runner's full rule**, in **4.7, 5.7, 5.5 and 7.3 s** from the start of the grab. The rule: red within 10 mm of green's centre, at the right height, upright, and released for 0.5 s;
  - **4 more ended with red stacked on green but failed the rule.** The place policy grabbed red again after letting go (see "Main remaining failure");
  - 2 were stress tests by the operator: green taken away just after the place, and green slid about 10 cm during the carry. Both ended in an effort-guard safety stop, as designed.
- **In 6 of the other 12 attempts, a person moved the cubes mid-task.** Usually that meant pushing green during the carry; once it meant moving red during the reach.
  - The run notes didn't record these. They were found by lining the phone videos up with the logs.
- **Of those 12 attempts, 8 ended with red stacked on green:**
  - **undisturbed:** 4 of 6 ended stacked, 3 meeting the full rule;
  - **with a person moving the cubes:** 4 of 6 ended stacked, 1 meeting the full rule. In that one, green was pushed around during the carry; the place policy followed it and stacked in 7.3 s.

| Session | # | Runner outcome | s | Grab s | Re-grasps after release | Ends stacked | Person moving cubes | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | controller stop | about 3 | | | | | Missed grasp. Then a fast reopen ran past the gripper's 44 mm limit and the controller stopped (power cycle). |
| 2 | 1 | safety | 0.9 | | | | | Dive overshoot: the next streamed point fell 1 mm below the floor. |
| 3 | 1 | safety | 0.9 | | | | | The same, with the target floor 6 mm higher. |
| 4 | 1 | safety (stress test) | 15 | 3.0 | 4 | | removed green | Red placed on green. Then the operator removed green, and the base guard tripped (2.3 Nm) during the rise. |
| 4 | 2 | **stacked** | 4.7 | 2.1 | 0 | yes | | Released 1 mm from centre. |
| 4 | 3 | dropped | 7.7 | 2.1 | 3 | yes | pushed green during the carry | Placed on green where it was pushed, then grabbed red again and closed on air. |
| 4 | 4 | dropped | 6.2 | 1.8 | 2 | | | Set down 12 mm off-centre (cube off-centre in the grip). Red ended on the table just behind green. |
| 4 | 5 | dropped | 10.9 | 2.1 | 4 | yes | pushed green during re-grasping | |
| 4 | 6 | dropped | 3.5 | 1.4 | 1 | | moved green during the lift | On green, then grabbed again and closed on air. |
| 4 | 7 | **stacked** | 5.7 | 2.3 | 1 | yes | | Released 1 mm from centre. |
| 4 | 8 | dropped | 6.2 | 2.2 | 1 | | moved red during the reach, then green during the carry | Red ended on the table. |
| 4 | 9 | safety (stress test) | about 2 | 1.9 | 1 | | slid green about 10 cm during the carry | The policy followed and placed; then green moved again. Joint 1 guard (−6.5 Nm). |
| 5 | 1 | timeout | 15 | 2.2 | 9 | yes | | On green at 4.5 s, then re-grasping until the timeout. |
| 5 | 2 | lost | 8.7 | 2.0 | 0 | yes | pushed green during the carry | Followed green and stacked on it. Tracking lost red in the grip, so the runner logged "lost". |
| 5 | 3 | dropped | 9.6 | 3.3 | 3 | | | Cube 14 mm off-centre in the grip. Red ended on the table just behind green. |
| 5 | 4 | **stacked** | 5.5 | 2.5 | 0 | yes | | |
| 5 | 5 | **stacked** | 7.3 | 2.2 | 0 | yes | pushed green several times during the carry | Followed green and stacked on it. |

- Session 5 ran with the in-hand offset fix (below).
- "Ends stacked" is the runner's own check right after the attempt ("Stacked. Watching.").
- Grab times and re-grasp counts come from the per-step logs. The "Person" column comes from the phone videos.
- **Videos:** [session 4](../media/rl_session4.mp4) (attempts 2–9) and [session 5](../media/rl_session5.mp4) (all of it), both at 2×. They are lined up with the logs to within about half a second, using the grasp moments.

**What the real arm showed:**
1. **The real gripper is slow to start closing.** In effort mode at 18 N, 35 → 25 mm took 0.35 s; training assumed
   0.21 s.
   - The grab policy closes and rises at the same time.
   - In session 1 the fingertips were above the cube before the fingers reached it.
2. **A fast position-mode open overshoots.** "Open to 35 mm in 0.2 s" from shut ran past the controller's limit
   (40 mm plus 4 mm tolerance), and the controller stopped.
   - It left the fingers at 47 mm, and every gripper command then tripped the limit again.
   - A power cycle cleared it.
   - Opens at about 0.044 m/s stop on target.
3. **The grab policy dives at about 0.2 m/s.** The streaming generator brakes at a limited rate, so near the table the
   real fingertips overshoot the target by 10–15 mm. Twice the next streamed point fell 1 mm below the hard floor. The
   dry world doesn't show this: its arm stops on target.
4. **Main remaining failure: grabbing red again after the release.** The place policy sets red down well, but then
   often closes on it again and re-releases instead of letting go and holding still. The 0.5 s released check then
   fails, and the runner reports "dropped" (closed on air) or "timeout", although red is usually left on green.
   - Counted stacks had 0–1 re-grasps.
   - The other placements had 1–9.
5. **Off-centre grasps.** The grasp centres the cube between the fingers, but it can sit 12–16 mm off along the finger
   pads. In training the cube is always centred, so the place policy lines up the fingertips, not the cube, with green.
6. **The effort guards tripped only in the stress tests.** Per-step joint efforts aren't logged yet, so peak efforts in
   normal attempts are unknown.

**Changes (the policies are unchanged):**
- **`robot/arm.py`:**
  - `open_gripper` never opens faster than 0.08 m/s per finger.
  - **Floor brake in `stream_tick`:** a streamed point below the floor or outside the workspace box stops the stream
    at the last checked point and carries on. It raises a SafetyStop only if this lasts 0.5 s. The floor is still
    never commanded.
- **`rl/deploy.py`:**
  - The policy opens the gripper at up to 0.07 m/s, and the wide open at up to 0.05 m/s.
  - **Grip hold:** when the grab policy closes with the fingertips at red, the arm waits until the fingers stop, for
    at most 0.8 s. `--no-grip-hold` turns this off.
  - On the real arm the lowest fingertip target is 6 mm higher than in training (`TARGET_MARGIN`), so the dive
    overshoot ends near the trained height.
  - **In-hand offset fix:** red's offset from the fingertips is measured during the scripted lift. While red is carried,
    both cubes are shifted by that offset in what the place policy sees. `--no-inhand-fix` turns this off.
  - A stack counts only with the gripper commanded open.
  - The end of an attempt always opens wide before rising.
  - Wake refuses to start if the gripper reads past 43.5 mm.
- **`rl/dry_world.py`:**
  - The dry world uses the real close profile.
  - Opens take the commanded time.
  - With these, the dry run reproduced a bug: the policy closing again after the release carried red off the stack.
    The end-of-attempt open fixes it.
- **Calibration:**
  - `scripts/pick_bias.py` measures where the gripper centres on the cube against where the camera puts it, over
    4 picks.
  - It found an xy correction of **(−5.9, +4.2) mm**, with residuals of 0.3 mm or less.
  - The correction is saved as `xy_correction_m` in `assets/calibration.yaml`, and `robot/camera_frame.py` applies it.
  - `scripts/pick_check.py` is a slow single-pick check.

**The dry world now shows the main failure too.** With the real close profile, dry runs of the final code ended
stacked in 2/4, 2/3 and 4/4 attempts on the 5090, and 2/5 on the lab PC (runner-counted 1/5, median 25.6 s). The
misses look like the real ones: red set on green, then grabbed again until the timeout. Before these changes the dry
world, with its faster gripper, stacked 4/5. A fix for the re-grasp can therefore be checked dry before the arm.

**Next:**
1. **Stop re-grasping after the release.** Freeze the arm with the gripper open once red sits on green, while the
   0.5 s check runs. A first version exists but hasn't been run, dry or real; check it in the dry world first.
2. **Log joint efforts** per step.
3. **Train with the real close profile** and a penalty for diving fast near the table. The floor brake only works
   around the dive.
4. **Run the scripted controller and the RL chain side by side** with a person moving the cubes.

## Code map

| Path | What |
|---|---|
| `rl/envs/moving_cube.py`, `rl/envs/stack_cubes.py` | The grab and stacking environments |
| `rl/envs/variations.py` | Per-episode sim-to-real variations (calibration error, tracker bias, friction, mass), off by default |
| `rl/train_ppo.py`, `rl/eval_policy.py` | Training (`--env moving_cube\|stack`, `--resume`) and single-policy evaluation |
| `rl/eval_chained.py`, `rl/eval_grip_speed.py` | Evaluation of the grab → lift → place chain, including at the real gripper speed |
| `rl/record_handover.py` | Records the chain's real hand-over states, so the placement policy can train from them |
| `rl/scripted_check.py`, `rl/scripted_stack.py` | Scripted baselines and sanity checks |
| `scripts/sysid_record.py`, `rl/sysid_fit.py` | System identification: record on the arm (or `--dry`), then fit the motion model |
| `rl/deploy.py`, `rl/dry_world.py`, `rl/check_deploy_obs.py` | The real-arm runner, its simulated stand-in, and the observation check |
| `scripts/pick_bias.py`, `scripts/pick_check.py` | Measure and check the camera-to-arm xy correction by picking the cube |
| `rl/make_sync_bundle.sh` | Packs uncommitted work, policies and logs to continue on another machine |
| `runs/` (not in the repository) | Checkpoints, training logs, evaluation and deployment logs |
