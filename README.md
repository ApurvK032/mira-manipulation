# MIRA: Markerless Interactive Reactive Arm

*Markerless 6D tracking from CAD models, driving a reactive arm. First task: stacking cubes while a person interferes.*

A Trossen WidowX AI arm and one fixed Intel RealSense D455 stack a red 5 cm cube on a green one, while a
person keeps interfering. The arm hovers over the red cube and follows it as it is moved. It picks
red the moment it comes to rest. While carrying, it follows green if green is moved. It stacks, then
stays awake: if the stack is knocked over, a grasp misses or red is pulled away mid-grasp, it simply
goes again.

The perception is tagless: no AprilTags, no markers. It uses color and depth with DLR-RM's M3T tracker,
and checks every pose against the image before anything acts on it. A scripted controller drives the
arm today. It is the baseline and the infrastructure for an RL policy trained in MuJoCo (see
[Roadmap](#roadmap) and [PLAN.md](PLAN.md)).

| | |
|---|---|
| Perception | 6D poses of both cubes at **~60 fps**, ~8 ms compute per frame; a moved cube is re-found in 0–1 frames |
| Simulation check | **0 wrong poses in 16,274 frames** across 7 adversarial scenarios ([docs/poc](docs/poc/README.md)) |
| Camera-to-arm calibration | **2.1 mm RMS** (cube held in the gripper, 18 poses) |
| Scripted stacking | **10/10 stacked**, 9 on the first attempt ([docs/reliability](docs/reliability/run_20261016_174531.md)) |
| Following a moving cube | **~0.25 s** lag; within 0.7 mm over a still cube |

## How it works

### Perception (`perception/`)

Every camera frame goes through three steps:

1. **Global detector** (`detector.py`). A per-color HSV mask feeds a depth-based box fit that gives a
   6D cube pose from scratch, with no memory between frames, so a cube moved anywhere in view is found
   again in the next frame. Blobs that are not cube-shaped (a hand, a red stick, flat red tape) are
   rejected.
2. **M3T tracking** (`tracker.py`). DLR-RM's 3DObjectTracking (M3T, through pym3t) refines and tracks
   each cube from color contour and depth.
3. **Consistency check** (`consistency.py`, `table.py`, `pipeline.py`). The tracked cube is ray-cast
   into the image and compared with the color mask and the depth. The other cube counts as a known
   occluder. Each cube gets a state:
   - `TRACKING`: verified, safe to act on.
   - `OCCLUDED`: mostly hidden; the pose is a best guess.
   - `LOST`: no pose.
   The rule is "never confidently wrong": a pose that fails the check is never reported as `TRACKING`.

### Arm and control (`robot/`)

- **`arm.py`: safety wrapper.** Every command goes through it.
  - Joint limits with a margin.
  - The whole joint path is checked against a workspace box, using forward kinematics from Trossen's
    MuJoCo model.
  - Speed limits.
  - Tracking-error and unexpected-effort guards.
  - The arm is only braked (idle) at its rest pose.
- **`controller.py`: the always-on controller.**
  - *Watch:* the arm waits at a pose that hides none of the workspace from the camera.
  - *Follow:* it hovers over the red cube and follows it (`follow.py`).
  - *Pick:* once both cubes have rested about 1 s, it plans a top-down grasp. The finger direction is
    chosen so the open gripper keeps 8 mm from the other cube, checked against the real finger meshes.
    It descends straight from the hover point, and stops and re-follows if red is moved on the way down.
  - *Carry:* it follows the green cube at carry height, and lowers once green has rested 0.5 s.
  - *Verify and watch again.*
  - Any failure (missed grasp, drop, bad placement) leads straight to another attempt.
  - Buttons on the live page: Start, Pause, Resume, Go to rest.
- **`follow.py`: hover-and-follow.** Smooth re-aims a few times a second, with the fingers kept along
  the cube's faces and acceleration capped.
- **`grasp.py`, `ik.py`, `camera_frame.py`:** grasp planning, IK on the MuJoCo model, and the
  camera-to-base transform.

## Hardware

- **Arm:** Trossen WidowX AI (follower), on Ethernet, driver `trossen-arm` matched to the firmware (1.8.3 here).
- **Camera:** Intel RealSense D455 on a fixed mount, 0.6–0.8 m from the cubes, looking down at
  45–60°; 848×480 at 60 fps, color and depth, on **USB 3**.
- **Cubes:** 5 cm, matte, red and green (other colors are in `assets/colors.yaml`).
- **Computer:** developed on Windows 11 + WSL2 (Ubuntu) with an RTX 4070. Runs natively on Ubuntu too.

## Setup (no sudo needed)

```bash
scripts/setup_env.sh        # micromamba env "rlwx" (conda-forge) + builds pym3t with RealSense support
alias py=~/micromamba/envs/rlwx/bin/python
```

Run everything from the repo root as `py -m <module>`. M3T builds its cube models into
`assets/m3t_models/` on first use.

**Camera on a Windows host, pipeline in WSL** (no usbipd needed). Run the streamer with a
Windows Python that has `pyrealsense2`:

```powershell
python scripts\win_camera_stream.py --host 127.0.0.1 --wb 3000
```

With WSL **mirrored networking**, WSL sees it at 127.0.0.1:5600. Mirrored networking is also required
for the arm, whose UDP stream doesn't reach WSL behind NAT: set `networkingMode=mirrored` under
`[wsl2]` in `%UserProfile%\.wslconfig`. On native Ubuntu, use `--source realsense` instead.

**Arm network:** give the computer a static IP on the arm's subnet (the default arm IP is 192.168.1.2;
pass `--ip` if yours differs).

## Running

Start the live perception first; everything else reads it.

| What | Command |
|---|---|
| Live perception + browser view (127.0.0.1:8080) | `py -m scripts.web_view --cubes red,green` |
| **Always-on follow-and-stack controller** (press Start on the page) | `py -m robot.controller` |
| Follow mode only (hover over red, no grasping) | `py -m scripts.follow_cube` |
| Dry run of either, with no arm (simulated motion, commands from `runs/dry_command.json`) | add `--dry` |
| Camera-to-arm calibration (cube held in the gripper) | `py -m scripts.calibrate_camera plan` … `fit` |
| One scripted stack / a reliability run with retries and a log | `py -m scripts.stack_test` / `py -m scripts.reliability_run --trials 10` |
| Bring-up steps: read the arm, home, gripper, hover, pick | `scripts.check_arm`, `arm_home`, `gripper_test`, `hover_test`, `pick_test` |
| Take the arm to rest from wherever it was left | `py -m scripts.to_rest` |
| Simulation proof of concept (all scenarios, GIFs, plots) | `py -m scripts.eval_sim --gif` |
| Camera check, recording, replay, color tuning | `scripts.check_camera`, `record`, `run_live --source <dir>`, `tune_colors` |

The live page shows the overlay (switchable to depth or color masks), each cube's state and position,
fps, compute time and capture-to-pose latency, the controller's status, and the arm buttons. It
listens on localhost only; forward the port (for example over SSH) to view it from another machine.

## Safety

- Keep a hand on the arm's power switch whenever it can move. The controller wakes the arm only when
  you press **Start**.
- Every move is checked before it runs: joints, the workspace box along the whole path, and a floor
  for the fingertips. It is watched while it runs (tracking error, unexpected effort). Any error pauses
  and holds; when the program exits, the arm goes back to rest.
- Slide or hold cubes low by their sides while the arm follows them: it hovers right above the cube.
- Dry runs (`--dry`) never read the page's buttons, so a test can never drive the real arm.
- D455: never call `hardware_reset()`. In testing it left the camera unresponsive. Unplug and replug instead.

## Layout

```
perception/   detector, M3T tracker wrapper, consistency check, table prior, camera sources, overlay
robot/        safety wrapper, IK, grasp planning, follower, always-on controller
sim/          MuJoCo bench scene (D455-like noise, hand, distractors), scenarios, render worker
scripts/      live view, calibration, bring-up and stacking tests, follow mode, sim evaluation, camera tools
assets/       cube colors, calibration, Trossen's MuJoCo arm model (BSD-3), table texture
docs/         proof of concept (sim), calibration checks, reliability runs
PLAN.md       project plan and phase status
```

## Roadmap

- **RL policy in MuJoCo.**
  - *Observation:* cube poses and velocities, arm and gripper state.
  - *Action:* small fingertip moves plus the gripper command, through the same safety wrapper.
  - *Reward:* a verified stack, fast and smooth, without drops or collisions.
  - *Training:* with disturbances (cubes moved, knocked over, pulled away mid-grasp) and the camera's
    noise and delay. Start as a residual policy on top of the scripted controller.
- **Disturbance benchmark.** Success rate, time to stack, reaction time and smoothness while a person
  interferes, comparing the scripted controller with RL.
- **Known issues:**
  - The base-joint effort guard trips during fast following on driver 1.8.3. Upgrading to driver and
    firmware 1.9.3 or newer fixes the underlying streaming behavior.
  - Placement lands about 4 mm short in x, which points to a calibration bias.
  - Stacked cubes read 4–5 mm short; the fix is a stack prior in perception.

## WSL graphics note

M3T's GLEW is built for GLX. GLFW 3.5 picks Wayland when WSLg's Wayland socket exists, and then GLEW
can't initialize, so `perception/__init__.py` keeps GLFW on X11. MuJoCo rendering runs in a separate
process (`sim/render_worker.py`), because M3T calls `glfwTerminate` after it builds its models.

## Acknowledgments

- [DLR-RM/3DObjectTracking](https://github.com/DLR-RM/3DObjectTracking) (M3T) and its Python bindings
  [pym3t](https://github.com/agimus-project/pym3t), cloned and built by `scripts/setup_env.sh`.
- [Trossen Robotics](https://github.com/TrossenRobotics) for the WidowX AI driver and the MuJoCo model
  in `assets/trossen_arm_mujoco` (BSD 3-Clause, see `assets/LICENSE.trossen_arm_mujoco`).

## License

MIT (see [LICENSE](LICENSE)). The Trossen MuJoCo model in `assets/trossen_arm_mujoco` keeps its own BSD 3-Clause license.
