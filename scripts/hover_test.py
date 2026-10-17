"""End-to-end check of camera + calibration + arm: hover the open gripper just above a cube on the table.

Waits until the cube rests still and fully visible (no hand near it), converts its pose to the arm frame
with assets/calibration.yaml, then: rest -> 8 cm above the cube's top -> 3 cm above it (hold) -> back.  (8 cm is about the highest a straight-down
gripper reaches here before the wrist joint limit.)
The gripper points straight down with its fingers lined up with the cube's faces.

Usage:  python -m scripts.hover_test --plan     # wait for the cube, plan and check; no motion
        python -m scripts.hover_test            # plan, then move (someone at the power switch)
Needs scripts.web_view running.
"""

import argparse
import time

import numpy as np

from robot.arm import REST, SafeArm, SafetyStop
from robot.camera_frame import CameraToBase
from robot.grasp import HALF, plan_top_down, top_face_tilt_deg, wait_for_cube


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.1.3")
    ap.add_argument("--cube", default="red")
    ap.add_argument("--gap", type=float, default=0.03, help="fingertip height above the cube's top, m")
    ap.add_argument("--hold", type=float, default=6.0)
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()

    cal = CameraToBase()
    print(f"calibration rms {cal.rms_mm} mm; waiting for a resting {args.cube} cube ...", flush=True)
    center, R = wait_for_cube(cal, args.cube)
    targets = {"above": HALF + 0.08, "hover": HALF + args.gap}
    try:
        qs, yaw = plan_top_down(center, R, targets)
    except ValueError as e:
        raise SystemExit(str(e))
    print(f"cube centre in the arm frame: {np.round(center * 1000, 1).tolist()} mm (top face tilt "
          f"{top_face_tilt_deg(R):.1f} deg); fingers along {np.degrees(yaw):.0f} deg")
    for name, h in targets.items():
        print(f"  {name:5s}: fingertip at z {(center[2] + h) * 1000:.1f} mm, joints {np.round(qs[name], 3).tolist()}")

    arm = SafeArm(args.ip)
    try:
        q0 = arm.q()
        if np.abs(q0 - REST).max() > 0.15:
            print("arm is not at the rest pose; not starting")
            return
        for a, b in ((q0, qs["above"]), (qs["above"], qs["hover"]), (qs["hover"], qs["above"]), (qs["above"], REST)):
            arm.check_target(b, a)
        print("plan: all moves pass the workspace and joint checks")
        if args.plan:
            return
        print(f"gripper open at {arm.open_gripper(0.035) * 1000:.1f} mm", flush=True)
        try:
            arm.move_joints(qs["above"], speed=0.8, label="above the cube")
            arm.move_joints(qs["hover"], speed=0.3, label="down to hover")
            p_tip = arm.model.fk(arm.q())[0]
            print(f"HOVERING: fingertips at {np.round(p_tip * 1000, 1).tolist()} mm, cube top centre at "
                  f"{np.round((center + [0, 0, HALF]) * 1000, 1).tolist()} mm "
                  f"(xy offset {np.linalg.norm(p_tip[:2] - center[:2]) * 1000:.1f} mm by the model). "
                  f"Look: is the gripper centred over the cube?", flush=True)
            time.sleep(args.hold)
            arm.move_joints(qs["above"], speed=0.5, label="back up")
            arm.rest_and_idle(speed=0.8)
            print("back at rest")
        except SafetyStop as e:
            print(f"SAFETY STOP: {e}; returning to rest slowly", flush=True)
            time.sleep(1.0)
            arm.rest_and_idle(speed=0.4)
        finally:
            arm.gripper_idle()
    finally:
        if not arm.position_mode:
            arm.close()


if __name__ == "__main__":
    main()
