# Running the RL chain on the real arm (rl/deploy.py): checklist

For the lab PC, with the arm and the D455 attached and **someone at the power switch the whole time**. Nothing in this
file has been run on the real arm yet. What has been checked, in dry runs only, is listed at the end.

## What runs

```
D455 -> scripts.web_view (env rlwx: perception, 127.0.0.1:8080)  --stats.json-->  rl.deploy (env rlwx-train)  -->  SafeArm
```

- **`rl/deploy.py` drives the arm through the path system ID measured** (`scripts/sysid_record.py`):
  - the policy runs at 20 Hz and moves a fingertip target (at most 3 cm and 15° ahead of the fingertips);
  - IK turns the target into joints, and `SafeArm.follow_goal` re-aims at most every 0.15 s;
  - `follow_check()` runs every 10 ms.
- **Every SafeArm guard still applies:** joint limits, the workspace box, the fingertip floor, tracking error and the
  effort guards.
- **One attempt:**
  1. WATCH, then the system-ID hover pose (27, 0, 13) cm;
  2. grab policy, until red is held and 1 cm up;
  3. scripted lift to 12 cm;
  4. placement policy, until the stack passes StackCubes' own success rule for 0.5 s (by the live tracker; see the
     stack-check risk below);
  5. hold, open the gripper fully (40 mm per finger), and rise straight up 4 cm slowly only once the fingers read
     at least 36 mm, then WATCH.
     - Moving away while the fingers were still opening knocked 3 of 7 dry stacks off-centre before this was added.
     - A cube turned about 45° to the fingers (70.7 mm across its corners) can wedge in the policies' 70 mm opening;
       one was carried up in a dry run.
     - If the fingers stall below 36 mm, the runner holds and pauses instead of lifting.
- **Ending an attempt early:**
  - **Timeout (15 s):** if red is held, it is carried back to where it was picked (or to a clear spot 8+ cm from
    green) and set down. It is carried with the fingertips 14 cm above the table, so red clears a cube on the table
    by 5.5 cm. There is no scripted stacking: every stack is the policies' own.
  - **Cube not seen for 0.5 s:** hold, and resume when it is seen again. After 3 s, end the attempt the same way.
- **SafetyStop:** hold and pause. Resume or Go to rest on the live page.
- **Program exit (Ctrl-C, SIGTERM, error):** put down anything held, go to WATCH, then rest with the brakes on.

## Before the first run

1. **Code on the lab PC:**
   - `rl/` (with `deploy.py`, `dry_world.py`, `check_deploy_obs.py`);
   - `robot/arm.py` and `robot/controller.py` (small additions: `velocities()`, non-blocking `open_gripper`);
   - `rl/requirements-train.txt` (adds `pyyaml`);
   - the two checkpoints, each `model*.zip` with its matching `vecnormalize*.pkl`.
2. **Training env:** `~/micromamba/envs/rlwx-train/bin/python -m pip install pyyaml==6.0.3` (the deploy runner reads
   `assets/calibration.yaml`).
   - `robot/` changes keep every existing call the same: `open_gripper` still blocks unless `blocking=False` is passed,
     and `velocities()` is new.
3. **Observation check on the lab PC** (no arm, about 10 s). Both "same fingertip" rows must be about 1e-7:
   ```bash
   ~/micromamba/envs/rlwx-train/bin/python -m rl.check_deploy_obs <grab_model> <place_model> --episodes 50 --seed 4242
   ```
4. **Dry run on the lab PC** (no arm; simulated scene, about 30 s per episode):
   ```bash
   ~/micromamba/envs/rlwx-train/bin/python -m rl.deploy <grab_model> <place_model> --dry --autostart --episodes 3
   ```
5. **Calibration:**
   - `assets/calibration.yaml` must be current (2.1 mm RMS, 2026-10-16).
   - If the camera has moved, recalibrate first (`py -m scripts.calibrate_camera`).
6. **Workspace:**
   - cubes 20–34 cm from the arm base, in front of it (x > 10 cm);
   - nothing else within reach;
   - the area under the hover pose (27, 0) cm clear.

## Run

1. **Perception** (env `rlwx`): `py -m scripts.web_view --cubes red,green`. Check that both cubes show `TRACKING` at
   http://127.0.0.1:8080.
2. **Arm at rest** (folded on its supports), powered, hand on the switch.
3. **Runner** (env `rlwx-train`, real arm). The final saves are `runs/rl/<run>/model`:
   ```bash
   ~/micromamba/envs/rlwx-train/bin/python -m rl.deploy <grab_model> <place_model>
   ```
4. **Press Start** on the live page. Pause holds the arm, Resume continues, and Go to rest puts down anything held
   and parks the arm.
5. **First session plan:**
   1. Cubes still, 10+ cm apart. Watch one attempt end to end, then Go to rest.
   2. Cubes still, five attempts.
   3. Then slide red slowly while the arm grabs.
6. **Logs:** `runs/deploy/run_<time>.jsonl` has every policy step (action, fingertip, target, joints, gripper, cube
   readings and their ages) plus events (attempt, hold, resume, command, attempt_end).

## Watch for (known risks, not tested on hardware)

- **The effort guard and fast following.** The base-joint effort guard tripped during fast following on driver 1.8.3
  (README, known issues), and the policies re-aim more often and more sharply than the scripted follower. A trip
  pauses the runner safely. If it trips often, slow the re-aims (`SPEED`, or `accel` in `follow_goal`) rather than
  loosening the guard.
- **Wide opening.** Check once, before the first run, that the real gripper reads at least 36 mm per finger when
  opened to 40 mm (`open_gripper(0.040)`, then `gripper_position()`). If not, every stack ends in a pause.
- **Gripper timing has never been measured.**
  - The simulated gripper closes and opens in about 0.09 s; the real one is slower.
  - The runner closes with 15 N (force mode, non-blocking) and opens over 0.2 s.
  - A slow close shows as the grab policy lifting before the fingers hold, and so a missed grasp, which the next
    attempt handles.
- **Low fingertips.**
  - The fingertip target may go down to 1.5 cm above the table, as in training.
  - SafeArm's hard floor is 1.2 cm above the table, with the table height taken from red's resting height at the
    start of each attempt.
  - The table height is the median of red's readings over 0.3 s, clipped to -5..+15 mm.
  - The fingers end 4.5 mm below the fingertip site.
  - A SafetyStop "gripper would leave the workspace" near the table means a re-aim's path dipped below the floor.
- **Red hidden while held.** While the gripper holds red, red's position is taken from the fingertips (red hangs
  10 mm below them, as in simulation), not from the tracker.
- **Stack check.**
  - It uses StackCubes' rule: red centred on green within 10 mm, both cubes tilted under 8°, gripper open, held for
    10 policy steps (0.5 s).
  - On the real arm only, it accepts 40–58 mm between the cube centres instead of 46–56, because stacked cubes read
    4–5 mm short.
  - If the check never passes, the attempt times out, so look at `red`/`green` in the log.
- **Off-centre stacks.** A placement released just before the timeout can leave red on green, off-centre. The runner
  then reports "Stacked. Watching." and waits: it does not pick red off green, because the grab policy only learned
  cubes on the table. Unstack by hand to continue.
- **Put-back.** It carries red at 14 cm (held cube bottom about 10.5 cm up) with a joint-space move. The path check
  covers the gripper and wrist, not the held cube. Watch the first one.

## Verified so far (dry, on the RTX 5090 PC, 2026-10-17)

See `docs/rl/REPORT.md`, section "Deployment runner (dry only)".

## Optional: system ID with feedforward re-aims (lab PC, about 2 minutes, no cubes needed)

Feedforward re-aims (`--ff`; `robot/reaim.py`, opt-in, off by default) end each re-aim moving at a share of the
target's velocity (gain 0.5) instead of stopping. They aim at where the target will be when the move ends. The
point of this run: find out whether the real driver (1.8.3) follows `goal_feedforward_velocities` the way `DryArm`
models it. It is not needed for the plain-re-aim deployment above.

**What the arm does** (same as the first system-ID run, 2026-10-17 16:28):
- rest → WATCH → hover at (27, 0, 13) cm;
- ±3 cm moves in x, y and z, as 0.3 s ramps, holding 1.5 s each;
- a 15 s side-to-side sweep (±3 cm, 0.2–1.2 Hz);
- the gripper closes and opens twice;
- WATCH → rest (brakes on).

**In the dry run:**
- the fingertips stayed at or above 115 mm;
- they moved at most 3.9 cm from hover in the ramps and 4.7 cm in the sweep (targets ±3 cm), so up to about 2 cm
  of overshoot;
- every re-aim is path-checked;
- near a joint limit or the 11 cm floor, the run falls back to a plain re-aim.

**Before:**
- the arm rests on its supports;
- your hand is on the power switch;
- nothing within about 10 cm of the hover point.

```bash
~/micromamba/envs/rlwx/bin/python -m scripts.sysid_record --ff           # real arm; writes runs/sysid/sysid_<time>_ff.npz
~/micromamba/envs/rlwx/bin/python -m rl.sysid_fit runs/sysid/sysid_<time>_ff.npz
```

**Stop it** (power switch) if the arm swings more than about 5 cm from the hover point, or the motion looks jerky or
oscillating.

**Reading the result:** `rl.sysid_fit` replays the targets through the simulator's `reaim_ff` model.
- An RMSE close to the plain run's 0.29° means the driver matches the model.
- A much larger RMSE means it handles feedforward differently, and `reaim_ff` training would need a new model first.

## Updates from the first real-arm session on the RTX 5090 PC (2026-10-20)

- **Camera serial check.** `web_view`, `rl.deploy`, `robot.controller` and `follow_cube` refuse a camera that isn't
  the one `assets/calibration.yaml` was fitted with (`--ignore-serial` overrides). To calibrate a new camera, start
  `web_view --ignore-serial`; the saved calibration records the live serial.
- **Calibrate by picking.** `scripts.calibrate_camera pick --ip 192.168.1.3 --prior-current`:
  1. the arm hovers over red on the table;
  2. you centre it, then press Resume;
  3. it picks it 36 mm above the table, visits 18 poses, puts it back and fits.
  Go to rest stops it at any time. It replaces placing the cube in the gripper by hand at the rest pose, which slipped.
- **Grip.** Use `--grip-mode effort --grip-force 18` with `rl.deploy`: the steadiest hold on firmware 1.9.4
  (`scripts/grip_lift_test.py`). The default (external effort, 15 N) is unchanged.
- **Grasp detection** now needs the fingers stopped at a cube's width with the fingertips at red. Closing on air no
  longer counts as a grasp (it did on the first real attempt, and the placement policy then came down on green).
- **New outcome, "dropped":** the gripper closed on nothing during lift or place, so the attempt ends.
- **Perception.** `web_view` exits if the camera loop dies, so a dropped camera can't serve frozen poses. Keep the
  D455 on USB 3.
- **Real gripper speed:** about 0.75 s to close, about 0.2 s to open (simulation: 0.09 s). Expect slower grabs than
  in simulation until the policies are trained with it.
