# MIRA: project report

MIRA was developed in Professor Zhutian Chen's lab, on the lab's arms, cameras and computers.

**Status (2026-11-01):** the arm and the camera have been moved to other projects. This page records where MIRA
stands, its limitations, and the plan for when the hardware is available again.

**Details elsewhere:**
- [README](../README.md): an overview, demos and how to run everything.
- [RL report](rl/REPORT.md): the learning work, and every real-arm attempt.
- [Original plan](PLAN.md): how each phase turned out.

## Timeline

| Date | Milestone |
|---|---|
| 2026-10-16 | Scripted controller: 10/10 stacks in a reliability run ([log](reliability/run_20261016_174531.md)) |
| 2026-10-17 to 10-21 | RL in simulation: grab and placement policies, system ID of the arm, and fine-tuning for the real gripper's speed |
| 2026-10-20 | Firmware 1.9.4 and 100 Hz streaming on the arm; camera-to-arm calibration at 2.08 mm RMS |
| 2026-10-31 | First real-arm runs of the learned policies: 17 attempts, 4 counted stacks |
| 2026-11-01 | Second real-arm day: re-grasp fix, retry from here, recalibration; 60 attempts, 19 counted stacks |

## What the project shows

- **Markerless 6D tracking of two cubes from their CAD models, at camera rate (about 60 fps), on a real D455.**
  - Each cube has an explicit state: `TRACKING`, `OCCLUDED` or `LOST`.
  - Every pose is checked against the image before it is used.
  - In simulation, there were 0 wrong poses in 16,274 frames across 7 adversarial scenarios.
- **A camera-to-arm calibration in which the arm holds the cube up to the camera.**
  - It uses 18 poses, and reaches 2.1–2.3 mm RMS.
  - A pick-based check measures the remaining error where it matters, at the grasp. It caught a 4.8 mm shift after
    the arm was remounted, and the correction brought it to 0.6 mm.
- **A safety wrapper around the arm.** It checks every move before it runs, and stops on tracking error or unexpected
  joint effort.
  - In 77 real-arm attempts of the learned policies, it stopped the arm 24 times, with no damage reported.
- **A reactive scripted controller.** It follows a cube as it is moved, retries after a failure, and stacked 10/10 in
  its reliability run.
- **Sim-to-real reinforcement learning:** PPO in MuJoCo, against a system-identified model of the arm and the real
  gripper's speed. The policies run unchanged on the real arm. They:
  - stack red on green: on 2026-11-01, undisturbed stacks took 3.5–5.5 s from the start of the grab;
  - follow a cube slid by hand (stacked in 12.2 s after red was moved up to 19 cm);
  - follow a moved target cube;
  - with the always-on runner, rebuild a stack after it is knocked over.
- **Sim-to-real problems found on the hardware and fixed:**
  - On 2026-10-31: the gripper's slow start when closing, an overshoot when it opens fast, overshoot of fast dives
    near the table, and cubes held off-centre in the grip.
  - On 2026-11-01: re-grasping after the release, the release height, a calibration shift after a remount, and guard
    trips from abrupt stops.

## Results

| Area | Result |
|---|---|
| Perception | ~60 fps, ~8 ms of compute per frame; 0 wrong poses in 16,274 simulated frames |
| Calibration | 2.08 mm RMS (2026-10-20); a pick-measured residual of 0.6 mm after the 2026-11-01 correction |
| Scripted controller | 10/10 stacked, 9 on the first attempt (2026-10-16) |
| RL, simulation | 72% stacked over 150 trials at the real gripper speed; 66% with calibration, tracker, friction and mass errors |
| RL, real arm, 2026-10-31 | 17 attempts: the grab picked up red in 14 of 14 after the first 3 setup attempts; 8 of 12 normal attempts ended stacked, 4 counted |
| RL, real arm, 2026-11-01 | 60 attempts, 19 counted stacks (median 4.6 s). With the final code and no deliberate disturbance: 6 of 15; with both cubes in the trained range at the start, 5 of 8. 20 safety stops, 11 with a cube beyond the trained range |

A counted stack meets the runner's full rule: red within 10 mm of green's centre, at stack height, upright, released and
still for 0.5 s. The real-arm numbers come from sessions in which the runner was being fixed. They are not a benchmark.
Every attempt is listed in the [RL report](rl/REPORT.md#second-real-arm-day-release-retry-and-recalibration-rtx-5090-pc-2026-11-01).

## Limitations

- **The re-grasp fix is not learned.** It, and "release ends placing", are runner rules.
- **The RL success rates come from small samples,** taken while the runner changed. There is no clean benchmark yet.
- **The runner doesn't yet refuse starts outside the trained range.** Cubes placed beyond it (20–34 cm from the base)
  caused most of the safety stops on 2026-11-01.
- **The base-joint effort guard (2.3 Nm) stops fast carries** when the target cube is moved far.
- **Red lands off centre.** In most placements on 2026-11-01 it landed about 9 mm toward the arm; the cause isn't
  measured yet.
- **The calibration is out of date.** The arm and the camera have been moved since, so the next session starts with a
  recalibration.

## Future plans (when the arm is ready again)

### 0. Setup and recalibration (first session, about 30 minutes)
1. Arm network and camera:
   - confirm it is the project arm (by its address and MAC);
   - the D455 on USB 3.
2. Calibrate, then check:
   - back up `assets/calibration.yaml`;
   - `scripts.calibrate_camera pick --prior-current`;
   - `scripts.pick_bias --picks 6 --save`;
   - a 2-pick check, which should read under 1 mm.
3. A dry run, then one watched RL attempt.

### 1. Runner robustness (can be built and dry-tested before the arm is back)
1. **Start only in the trained range:** both cubes 20–34 cm from the base, as the scripted follow mode already does.
   This would have prevented 11 of the 20 stops on 2026-11-01.
2. **Put-back and rest after a stop:** if the straight lift has no reachable pose (far out), fall back to a lower lift
   or a joint-space move, so a stop never ends with the runner exiting.
3. **Lifting off a cube that a finger rests on:** open the gripper and lift slowly first, so the guard doesn't read the
   unloading as a collision. Keep the guard limits as they are.
4. **Smoother carries:** lower the streaming acceleration in the place phase, staying within the range the policy was
   trained with, so fast follows don't trip the base guard. Changing the 2.3 Nm limit itself would be a separate,
   deliberate decision.

### 2. Placement accuracy
1. **Measure the bias:** does red really overhang, or does the camera misread stacked cubes? Stacked cubes already
   read 4–5 mm low.
2. **Then correct it:** an aim correction, or a stack prior in perception.

### 3. Retrain the placement policy (lab PC)
1. **Episodes that continue after the stack,** rewarding letting go at stack height and backing off. This removes the
   need for the runner's freeze.
2. **Training conditions:**
   - the real gripper's closing profile;
   - a penalty on fast dives near the table;
   - a penalty on base acceleration while carrying.
3. **Checks:** in simulation, then dry, then on the arm.

### 4. A clean benchmark (one session, about an hour)
1. **Fixed cube spots** (tape marks), both 25–32 cm out.
2. **Both controllers, 20+ trials each,** the scripted controller and the RL chain, on the same day with the same
   code.
   - Record success, time to stack, and guard stops.
   - Report the results with confidence intervals.
3. **Disturbances, 10 trials each:**
   - red slid 5–10 cm during the grab;
   - green pushed 3–5 cm during the carry;
   - the stack knocked over.
