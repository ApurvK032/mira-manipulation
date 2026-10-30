# Upgrade day: driver 1.9.3 + firmware 1.9.x, then streaming

**Before starting:**
- someone at the power switch for every step that moves the arm;
- the arm resting on its supports;
- the arm upgraded answers at **192.168.1.3** (the PC needs a static address on 192.168.1.0/24); every
  command below passes `--ip 192.168.1.3` (the scripts default to .3);
- only the arm at .3 is touched. (The lab's second arm was also flashed to 1.9.4 on 2026-10-20, so it
  needs driver 1.9.3 too.)

**Environments.** `rlwx` runs perception and the arm scripts; `rlwx-train` runs the RL runner. Both have
`trossen-arm==1.8.3` until step 4.

## What was checked the evening before (no arm)

- **Every driver call our code makes exists in 1.9.3 with the same arguments.** Checked in a throwaway env
  `trossen193`.
  - New in 1.9.3: `discover`, `MODEL_NAME`, `ERROR_INFORMATION`, used only by `scripts/check_arm.py`.
  - Numpy arrays are accepted for targets and feed-forward velocities, and the joint-limit fields are unchanged.
  - 1.9.3 prints a notice on connect: default motor gains change in 1.10. It doesn't apply to 1.9.x.
- **Streaming is implemented and dry-tested** (needs 1.9.x on the arm; on 1.8.3 keep the re-aims):
  `SafeArm.stream_begin` / `stream_tick`, `scripts/sysid_record.py --stream HZ`, `rl/deploy.py --stream HZ`, and
  `lag_mode="stream"` in the simulator.

## Steps

### 1. Back up the configuration, with the current 1.8.3 driver (no motion)

```bash
~/micromamba/envs/rlwx/bin/python -m scripts.check_arm --ip 192.168.1.3            # versions, joints, errors (read only)
~/micromamba/envs/rlwx/bin/python -m scripts.backup_arm_config --ip 192.168.1.3    # -> runs/arm_backup/arm_config_<time>_controller-1.8.3.yaml
```

Copy the YAML off this PC (for example, attach it to a release in the private repo).

### 2. Flash firmware 1.9.x (newest on the downloads page; 1.9.4 per the changelog)

This needs a **Linux PC with admin rights**; this WSL box has no sudo. Follow
[Trossen's firmware upgrade guide](https://docs.trossenrobotics.com/trossen_arm/main/getting_started/firmware_upgrade.html):
install `teensy_loader_cli` and PJRC's udev rule, connect the controller by USB to micro-USB, then:

```bash
unzip firmware-wxai_v0.zip
teensy_loader_cli --mcu=TEENSY41 -s firmware-wxai_v0.hex
```

### 3. Power-cycle the controller; the arm stays at rest

Then back up again with the new driver (after step 4) and compare: the factory calibration must be unchanged.

```bash
~/micromamba/envs/rlwx/bin/python -m scripts.backup_arm_config --ip 192.168.1.3
~/micromamba/envs/rlwx/bin/python -m scripts.compare_arm_config runs/arm_backup/arm_config_<before>_controller-1.8.3.yaml runs/arm_backup/arm_config_<after>_controller-1.9.4.yaml
```

Recovery at any point: `~/micromamba/envs/rlwx/bin/python -m scripts.to_rest --ip 192.168.1.3`.

### 4. Install driver 1.9.3 in both environments

```bash
~/micromamba/envs/rlwx/bin/python -m pip install trossen-arm==1.9.3
~/micromamba/envs/rlwx-train/bin/python -m pip install trossen-arm==1.9.3
~/micromamba/envs/rlwx/bin/python -m scripts.check_arm --ip 192.168.1.3            # expect driver 1.9.3, controller 1.9.x, no error
```

### 5. Smoke test at low speed (arm moves)

```bash
~/micromamba/envs/rlwx/bin/python -m scripts.sysid_record --ip 192.168.1.3         # the same test as 2026-10-17 (re-aims), about 45 s
~/micromamba/envs/rlwx/bin/python -m rl.sysid_fit runs/sysid/sysid_<time>.npz
```

Expect the same fit as on 1.8.3 (re-aim model about 0.3° RMSE). If not, stop: the arm behaves differently on
1.9.x.

### 6. Streaming ladder (arm moves; stop at the first jerk, force-guard trip or odd noise)

```bash
~/micromamba/envs/rlwx/bin/python -m scripts.sysid_record --ip 192.168.1.3 --stream 20
~/micromamba/envs/rlwx/bin/python -m scripts.sysid_record --ip 192.168.1.3 --stream 50
~/micromamba/envs/rlwx/bin/python -m scripts.sysid_record --ip 192.168.1.3 --stream 100
~/micromamba/envs/rlwx/bin/python -m rl.sysid_fit runs/sysid/sysid_<time>_stream<HZ>.npz
```

- **The fitter replays each recording through the streaming model.** Read it like this:
  - a small "streaming model" RMSE (dry runs: 0.03–0.2°) means the driver follows streamed points as modelled;
  - a fitted lag well under 0.16 s + 0.16 s means the arm got faster.
- **Use the highest rate that ran clean** for the next step.

### 7. First RL run with streaming (perception on, cubes still, then slid slowly)

```bash
~/micromamba/envs/rlwx/bin/python -m scripts.web_view --cubes red,green          # perception (camera streamer running on Windows)
~/micromamba/envs/rlwx-train/bin/python -m rl.deploy <grab> <place> --stream <HZ> --dry --autostart --episodes 3   # dry first
~/micromamba/envs/rlwx-train/bin/python -m rl.deploy <grab> <place> --stream <HZ> --ip 192.168.1.3                  # real; press Start
```

**Policies for streaming:** `runs/rl/grab12_ppo/model` + `runs/rl/place11_ppo/model`.
- In simulation: 89% at the default limits, 93% with acceleration 6 rad/s², and 77% with realistic calibration,
  tracking, friction and weight errors.
- Start at the default acceleration (4 rad/s² in `robot/stream.py`). Raise it only if the step-6 recordings show no
  force-guard trips.

### If streaming is not smooth

Re-aims still work on 1.9.x: run `rl.deploy` without `--stream`, with **grab5 + place10** (the 1.8.3 pair).

## Notes

- **Streaming safety:**
  - every new target is checked like any move (joint limits, workspace box, fingertip floor), and so is every
    10 ms step;
  - a target that fails isn't followed (the arm keeps to the last good one), and the run stops if targets keep
    failing for 0.5 s;
  - the tracking and force guards (`follow_check`) run every tick, as before.
- **Limits used by streaming:** joint speed 1 rad/s and joint acceleration 4 rad/s². Lower them in
  `robot/stream.py` if the force guard trips.
- **Going back** to 1.8.x firmware is "strongly not recommended" by Trossen: it resets the factory calibration.
