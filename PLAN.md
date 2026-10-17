# RL-widowxai: reactive cube stacking on a WidowX AI

## Goal

One Trossen WidowX AI arm and one fixed RGB-D camera overlooking the workbench. Pick a 5 cm cube of
one color (e.g. blue) and stack it on another (e.g. red). The system must react within about 0.1 s:

- the cube is moved by hand during the approach → the arm follows it;
- the grasp misses or the cube slips → the system notices and grasps again;
- the place fails or the stack gets knocked over → the system notices and tries again.

Plan: build the perception and control loop with a scripted controller first. Then train an RL policy
in MuJoCo and drop it into the same loop.

## Where the speed comes from

Every part runs in one fixed-rate closed loop. Nothing is planned once and then run open loop.

```
RealSense 60 Hz ──► color detector ──► M3T tracker ──► world state ──► controller ──► trossen_arm
 (color + depth)    (HSV + depth,     (pym3t, region   (cube poses in   (state machine,  set_cartesian_positions
                    init / re-init)   + depth, 6D)     base frame,      later RL policy)  goal_time ≈ 1 tick,
                                                       vel, flags)      30 Hz             blocking=False
```

- **The controller reads the latest cube pose on every tick.** A moved cube is just a new target on
  the next tick, so no replanning step is needed.
- **Detecting failure needs no vision.** The gripper opening after a close tells us right away
  whether we grasped the cube: about 5 cm wide means the cube is in the gripper, fully closed means
  we missed. After release, check that the top cube sits about 5 cm above the bottom cube, within
  about 1 cm in xy, and stays there for about 0.3 s.
- **Estimated latency** (to be measured in Phase 1): about 17–33 ms camera, 5–10 ms
  detection + tracking, about 33 ms command interpolation. That puts the arm's reaction at about
  60–80 ms.

## What the existing pieces give us (checked in their source)

**DLR-RM/3DObjectTracking (use the `M3T` folder, the latest version)**
- 6D tracking from a mesh, fusing region (color contour) and depth. It is very fast on CPU and
  handles occlusions it doesn't know about by using depth, e.g. a hand or the gripper.
- It **only tracks**. Its only detectors are `StaticDetector` (a fixed pose) and `ManualDetector`
  (you click 4 points). **We write our own**: an HSV color mask plus depth gives the cube's 3D
  centroid, and `minAreaRect` gives its yaw.
- It **has no lost-track signal. We add one**: project the tracked cube into the image and compare
  it with the color mask (IoU) and with depth. If they disagree for N frames, re-initialize from the
  color detector.
- The default RealSense stream is 960x540 color and 848x480 depth at 60 Hz.

**agimus-project/pym3t**: Python bindings for M3T. They expose `DummyColorCamera` and
`DummyDepthCamera`, so we push numpy frames in ourselves. We can also set `body.body2world_pose` from
outside, which lets our detector start or reset tracking. We own the stream with `pyrealsense2` and
feed the same frame to both the detector and the tracker.

**TrossenRobotics/trossen_arm** (driver, updated 2026-10-15)
- `set_cartesian_positions(pose6, InterpolationSpace.cartesian, goal_time, blocking=False)`. With
  `goal_time` ≤ 0.2 s it interpolates linearly, which suits streaming a target every tick.
- `get_gripper_position()` and `get_robot_output().cartesian` give gripper and end-effector state.
- External efforts are available to detect contact and collisions.

**trossen_arm_mujoco**: the WXAI MJCF model is already in `~/claude-code/hri/dex/third_party/`.

## Phases

### 0. Environment (two machines)

| Machine | Use |
|---|---|
| Lab PC: WSL2, RTX 4070, 24 cores, 30 GB | Day-to-day development, simulation, short RL runs, first hardware tests |
| Shared PC: native Ubuntu, RTX 5090 (when free) | Long RL training; real-time hardware runs if WSL adds latency |

- Code must run on both machines: a `uv` venv with Python 3.12 and a lockfile, nothing specific to
  WSL in the code, and machine settings in a config file (camera serial, arm IP, render backend).
- Build pym3t (M3T needs Eigen, GLEW, GLFW and OpenCV). On the shared PC we may not have `sudo`; if
  so, use pym3t's conda `environment.yaml`. conda-forge provides these libraries and librealsense
  without root.
- **Rendering:** on WSL, GLFW goes through WSLg (`DISPLAY=:0`); per the dex README, EGL and OSMesa
  don't work there. On native Ubuntu, either works.
- **Camera on WSL:** the RealSense goes through `usbipd-win`. Measure the frame rate and the delay
  from capture to arrival there once, and compare with native Ubuntu. Use whichever machine meets
  the latency target for real-time runs.
- **RTX 5090 (Blackwell):** needs CUDA ≥ 12.8 builds of PyTorch, JAX or MJX. Pin these versions per
  machine.
- **Shared PC:** other users' CPU load can add jitter. Pin the perception and control loop to its own
  cores (`taskset`).
- **Arm over Ethernet** (default 192.168.1.2): on WSL, mirrored networking lets WSL reach the arm's
  subnet directly. On Ubuntu, a static IP on the same subnet.

### Hardware choice

| Device | Role | Why |
|---|---|---|
| **D455**, fixed, about 0.6–0.8 m from the cubes, looking down at 45–60° | Scene camera for Phase 1 onward | Global-shutter color (no rolling-shutter skew when a cube moves fast), 95 mm stereo baseline for good depth at this range |
| D405 | Later: wrist camera for fine alignment before the grasp or place | Short-range depth (about 7–50 cm), very accurate close up |
| ZED 2 | Not used | Its depth needs the GPU SDK, it's weaker under 1 m, and it adds latency |
| WidowX AI #1 | Does the task | |
| WidowX AI #2 | Spare; later a leader arm for demos, or a second worker | |

### 1. Perception proof of concept (no arm)

**Status (2026-10-16): done in simulation.** 0 wrong `TRACKING` frames out of 16,274; moved cubes are
found again in 0–1 frames; compute 6.3 ms for 2 cubes. All pass bars met except stack height (47.5 vs
50 mm). Details: [docs/poc/README.md](docs/poc/README.md). Next: the same tests on the real D455.

Fixed D455, two or more matte cubes, arm parked. Exposure and white balance locked.

**Main rule: the pipeline is never confident and wrong.** When it isn't sure, it reports `LOST`.
A pose marked `TRACKING` must be correct, within the stated accuracy.

Components, built in this order:
1. **Global detector, run on every frame.** A per-color HSV mask, then connected blobs, then a depth
   check: the blob's size matches 5 cm at its measured depth, and the top face is flat. Output is a
   3D centroid and a yaw. Because it runs on every frame with no memory, a cube moved anywhere is
   found again in the next frame. This is the "AprilTag" property.
2. **M3T tracking per cube** (pym3t, region + depth modalities). It starts from the detector's pose
   and gives the precise 6D pose.
3. **Agreement check per frame:** compare the tracker's projected silhouette with the color mask
   (IoU), and its centroid with the detector's. On disagreement, re-initialize from the detector.
   Each cube gets a state: `TRACKING`, `REACQUIRING` or `LOST`.
4. **Frame sources:** the same pipeline runs on (a) synthetic MuJoCo sequences with ground truth,
   (b) recorded RealSense sequences, and (c) a live RealSense.
5. **Overlay and logs:** draw cube outlines, poses, states and the time each stage takes, and log
   every frame to a file for offline analysis.

Poses are in the camera frame for this phase. Phase 2 calibrates them to the arm base, using the arm
itself as ground truth.

**Tests and pass bars** (run first on sim, then on the real camera):

| Test | How | Pass |
|---|---|---|
| Static jitter | Cubes still for 60 s | Position σ < 1 mm, yaw σ < 0.5° |
| Small motion | Push a cube 10 mm along a ruler | Measured 10 ± 1 mm, visible in the very next frame |
| Fast move | Grab a cube and put it somewhere else, fast | `TRACKING` again within 3 frames (50 ms) after the hand leaves; no wrong `TRACKING` frames |
| Hand clutter | Hands moving around and touching the cubes for 5 min | 0 frames that are `TRACKING` with the wrong cube or > 1 cm error |
| Partial occlusion | A finger covering about half of a cube | Correct pose or `LOST`, never wrong |
| Stack | Blue cube on the red cube | Both tracked, height difference 50 ± 2 mm |
| Lighting | Room lights off and on, shadows | Hand-clutter bar still met |
| Absolute accuracy | Cubes on marked spots of a ChArUco board | Position error < 3 mm |
| Latency | Compute time from frame in to pose out | p95 < 10 ms for 2 cubes (camera transfer measured separately) |

### 2. Arm interface and scripted baseline
- Hand-eye calibration with the arm as ground truth: grasp a cube, move it to about 30 known poses,
  and fit the camera-to-base transform. That also gives a real accuracy number for the tracker.
- A cube's yaw is symmetric every 90°. Pick the grasp yaw closest to the current gripper yaw.
- Wrap `trossen_arm` with: connect and home, stream end-effector targets, gripper control, a
  workspace box, speed and acceleration limits, a stop key, and an effort-based collision stop.
- State machine checked on every tick: `APPROACH → DESCEND → GRASP → VERIFY_GRASP → LIFT → CARRY →
  LOWER → RELEASE → VERIFY_PLACE → RETREAT`. Any failed check jumps back to `APPROACH` using the
  cube's *current* pose: the cube moved, the grasp missed, the cube dropped, or the place failed.
- **Done when:** the arm stacks the cubes 10 times out of 10 from random positions, and recovers
  when you move the cube during the approach, pull it out of the gripper, or knock the stack over.

### 3. RL in simulation
- MuJoCo environment: the WXAI model plus two cubes, with **the same observation and action interface
  as the real loop**.
  - Observation: end-effector pose, gripper opening, and each cube's pose and velocity relative to
    the end effector.
  - Action: change in end-effector xyz and yaw, plus a gripper command, at 30 Hz.
- Domain randomization:
  - Noise on observed cube poses, 30–100 ms of latency, and dropped detections (occlusion).
  - Friction and mass.
  - **Disturbances:** move the cube to a new spot mid-episode, push the top cube off after a place.
    These are what teach the policy to recover.
- PPO with a staged dense reward (reach, grasp, lift, align, place, release) plus a success bonus.
  Train on the RTX 4070. MuJoCo MJX or Warp can run thousands of environments in parallel.
- Options if plain PPO is slow to learn: a behavior-cloning warm start from the scripted controller,
  or residual RL on top of it.

### 4. Sim-to-real
- Swap the policy into the controller slot. Keep the safety layer from Phase 2 and the
  `VERIFY_*` checks as watchdogs.
- Compare against the baseline on success rate, time to stack, and recovery time after a disturbance.

## Proposed layout

```
perception/   camera.py  color_detector.py  tracker.py  calibration.py
robot/        arm.py  (trossen_arm wrapper + safety limits)
control/      loop.py  state_machine.py  policy.py
sim/          env.py  train.py
assets/       cube_5cm.obj  m3t/*.yaml  camera.yaml
scripts/      check_camera.py  check_arm.py  calibrate.py  run.py
```

## Open questions
- Exact camera model? D435 or D415. A D435 has a minimum depth of about 0.28 m, which affects where
  it can be mounted.
- Is every cube a unique solid color? If two cubes share a color, the detector needs to tell them
  apart by tracking continuity.
- Gripper: the stock WidowX AI parallel gripper? It needs to open past 5 cm with some margin.
