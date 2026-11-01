# Running the RL policies on the real arm

This page explains how to run the learned chain on the WidowX AI with `rl/deploy.py`: grab policy → scripted lift →
placement policy. It is the procedure used for the first real-arm runs on 2026-10-31. Their results, and the reason
behind each safeguard below, are in
[REPORT.md](REPORT.md#first-real-arm-runs-of-the-rl-chain-rtx-5090-pc-2026-10-31).

**Someone must be at the arm's power switch whenever it can move.**

`py` and `pyt` below are the Pythons of the `rlwx` and `rlwx-train` environments (see the README's
[Setup](../../README.md#setup)). Run every command from the repository root.

## What runs

```
D455 ─► scripts.web_view (env rlwx: perception, 127.0.0.1:8080) ─stats.json─► rl.deploy (env rlwx-train) ─► SafeArm ─► arm
```

- **Policies:** grab13 + place12, each a `runs/rl/<run>/model.zip` with its `vecnormalize.pkl` and `config.json`.
  - They aren't in the repository; [REPORT.md](REPORT.md) lists the settings each was trained with.
  - They decide at 20 Hz from the cube poses relative to the fingertips. IK turns their fingertip target into joint
    targets.
- **Motion:**
  - With `--stream 100`, SafeArm streams joint targets to the arm at 100 Hz through a trajectory generator (1 rad/s,
    4 rad/s²). This needs firmware 1.9.x.
  - Without `--stream`, the arm re-aims at most every 0.15 s instead.
- **Every SafeArm guard applies:** joint limits, the workspace box along each path, a fingertip floor 1.2 cm above
  the table, tracking error, and the effort guards.
- **Grip:** `--grip-mode effort --grip-force 18` is the steadiest hold on firmware 1.9.4 (`scripts/grip_lift_test.py`).

**One attempt:**
1. **Start pose.** From the watch pose, the fingertips go to (27, 0, 13) cm.
2. **Grab policy,** until red is held 1 cm up.
   - A grasp counts only when the fingers have stopped at a cube's width with the fingertips at red.
   - When the policy closes at red, the arm holds still until the fingers stop (at most 0.8 s), because the real
     gripper starts closing slowly.
3. **Scripted lift** to 12 cm. Red's offset in the grip is measured here.
4. **Placement policy,** until the stack check passes for 0.5 s:
   - red centred on green within 10 mm, with 40–58 mm between the cube centres (stacked cubes read 4–5 mm short);
   - both cubes tilted less than 8°, and the gripper commanded open.
   While red is carried, both cubes are shifted by red's in-hand offset in what the policy sees.
5. **Release.** The gripper opens fully (40 mm per finger). The arm rises only once the fingers read at least
   36 mm, because a cube turned 45° to the fingers can wedge. Then back to the watch pose.

**Outcomes:**

| Outcome | Meaning |
|---|---|
| `stacked` | The stack check passed. |
| `dropped` | The gripper closed on nothing during the lift or the placement. |
| `lost` | A cube was unseen for 3 s. The arm holds after 0.5 s and resumes if the cube reappears. |
| `timeout` | 15 s passed. If red is still held, it is carried back at 14 cm and set down; there is no scripted stacking. |
| `safety` | A SafetyStop. The arm holds and the runner pauses: press Resume or Go to rest on the live page. |

- **After an attempt:** if red sits on green, the page shows "Stacked. Watching." and the runner waits. Otherwise the
  next attempt starts.
- **On exit** (Ctrl-C, SIGTERM or an error): the runner puts down anything held and returns the arm to rest with the
  brakes on.

## Before a session

1. **Environments:**
   - `rlwx` (perception and arm scripts) and `rlwx-train` (the runner);
   - both with `trossen-arm==1.9.3`, matching firmware 1.9.x.
2. **Observation check** (no arm, about 10 s). Both "same fingertip" rows must be about 1e-7:
   ```bash
   pyt -m rl.check_deploy_obs runs/rl/grab13_ppo/model runs/rl/place12_ppo/model --episodes 50 --seed 4242
   ```
3. **Dry run** (no arm, about 2 min). The dry world uses the real gripper's timing, so expect some re-grasp
   timeouts, as on the real arm:
   ```bash
   pyt -m rl.deploy runs/rl/grab13_ppo/model runs/rl/place12_ppo/model --stream 100 --grip-mode effort --grip-force 18 --dry --autostart --episodes 3
   ```
4. **Camera and calibration:**
   - Use the calibrated D455, on USB 3. Any other camera is refused by its serial number.
   - Hover check: `py -m scripts.follow_cube --cube red`. The open fingers should straddle red evenly, within about
     3 mm by eye.
   - **A small xy offset:**
     - measure a correction with `py -m scripts.pick_bias --save` (4 picks; it backs up `assets/calibration.yaml`);
     - confirm it with a second `scripts.pick_bias` run (about 0 mm) or `scripts.pick_check`.
   - **A moved camera:** recalibrate with `py -m scripts.calibrate_camera pick --prior-current` (about 10 min; back up
     `assets/calibration.yaml` first).
5. **Gripper:**
   - Opened to 40 mm, it must read at least 36 mm per finger; otherwise every stack ends in a pause.
   - If it reads past 43.5 mm (beyond the controller's 44 mm limit), push the fingers in by hand first. The runner
     refuses to wake the arm until you do.
6. **Workspace:**
   - cubes 20–34 cm in front of the arm base, at least 10 cm apart;
   - nothing else within reach;
   - the area under the start pose clear.

## Run

1. **Perception:** `py -m scripts.web_view --source realsense --cubes red,green`.
   - On a headless machine, run it under `xvfb-run -a`.
   - Both cubes should show `TRACKING` at http://127.0.0.1:8080.
2. **Arm:** resting on its supports, powered, with a hand on the switch.
3. **Runner:**
   ```bash
   pyt -m rl.deploy runs/rl/grab13_ppo/model runs/rl/place12_ppo/model --stream 100 --grip-mode effort --grip-force 18
   ```
4. **Start:** press Start on the live page. Pause holds the arm, Resume continues, and Go to rest puts down anything
   held and parks the arm.
5. **Order:**
   1. one watched attempt with still cubes;
   2. a series with still cubes;
   3. a series with a person sliding red or pushing green mid-task.
6. **Logs:**
   - `runs/deploy/run_<time>.jsonl` has every policy step: action, fingertip, target, joints, gripper, cube readings
     and their ages;
   - it also has the events: attempt, `grip_hold`, `inhand_offset`, hold, resume, command, `attempt_end`;
   - the console log is next to it.

| Option | Effect |
|---|---|
| `--dry` | A simulated arm and scene in real time. It never reads the page's buttons, so it can't drive the real arm. |
| `--no-grip-hold` | Let the grab policy move while the fingers are still closing. |
| `--no-inhand-fix` | Give the placement policy the cubes as seen, without correcting for red's offset in the grip. |
| `--ip` | The arm's address (default 192.168.1.3). |
| `--ignore-serial` | Accept a camera other than the calibrated one. Only use this while calibrating a new camera. |

## What to watch

- **Re-grasping after the release** (the main failure on 2026-10-31):
  - The placement policy sets red on green, then closes on it again instead of holding still.
  - The runner then reports `dropped` or `timeout`, though red is often left on green.
  - A fix is being prepared: freeze the arm with the gripper open once red rests on green.
- **Effort guards** (base 2.3 Nm, joint 1 6.3 Nm):
  - So far they tripped only when a person removed green, or slid it, during the carry.
  - A trip holds the arm. If trips become frequent, slow the motion rather than loosening a guard.
- **Fast dives near the table:**
  - The grab policy dives at about 0.2 m/s, and the stream can overshoot by 10–15 mm.
  - A streamed point below the floor stops the stream at the last checked point. A SafetyStop follows only if this
    lasts 0.5 s.
  - On the real arm the lowest target sits 6 mm higher than in training.
- **Gripper opens:**
  - A fast position-mode open overshoots past the 44 mm limit and stops the controller, which then needs a power
    cycle.
  - `open_gripper` caps the speed at 0.08 m/s per finger. The runner opens at up to 0.07 m/s, or 0.05 m/s for the
    full open.
- **Off-centre grasps:**
  - The cube can sit 12–16 mm off along the finger pads.
  - The in-hand fix corrects for this during the placement.
- **Off-centre stacks:** if red is left on green but off-centre, the runner waits ("Stacked. Watching."). It never
  unstacks, so lift red off by hand to continue.
- **Joint efforts aren't logged per step yet.** Only guard trips record an effort.

## Hardware notes

- **Camera:** keep the D455 on a USB 3 port. Never call `hardware_reset()` on it; unplug and replug instead.
- **Arm network:** the computer needs a static address on 192.168.1.0/24.
- **Driver and firmware:** the driver's minor version must match the firmware's. The upgrade procedure is in
  [FIRMWARE_UPGRADE.md](FIRMWARE_UPGRADE.md).
- **Motion model:** to re-measure it, run `scripts.sysid_record` (the arm makes small moves around the start pose for
  about 2 minutes), then `rl.sysid_fit`. See REPORT.md, "System ID tools and a lag stress test".
