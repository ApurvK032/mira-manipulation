"""Pick a cube off the table, lift it, check it came along, put it back down, and return to rest.

Top-down grasp: the gripper points straight down with its fingers along two opposite cube faces. The
fingertips descend to 10 mm above the cube's centre (holding its top 15 mm), close with a
limited force, and the finger reading confirms the grasp (about half the cube width; near 0 = missed).

Usage:  python -m scripts.pick_test --plan     # wait for the cube, plan and check; no motion
        python -m scripts.pick_test            # do it (someone at the power switch)
Needs scripts.web_view running and assets/calibration.yaml.
"""

import argparse
import time

import numpy as np

from robot.arm import REST, SafeArm, SafetyStop
from robot.camera_frame import CameraToBase
from robot.grasp import HALF, plan_top_down, read_cube, top_face_tilt_deg, wait_for_cube


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.1.3")
    ap.add_argument("--cube", default="red")
    ap.add_argument("--depth", type=float, default=0.010, help="fingertip height relative to the cube centre, m (0.010: hold the top 15 mm)")
    ap.add_argument("--lift", type=float, default=0.08, help="lift height, m")
    ap.add_argument("--force", type=float, default=15.0)
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()

    cal = CameraToBase()
    print(f"calibration rms {cal.rms_mm} mm; waiting for a resting {args.cube} cube ...", flush=True)
    center, R = wait_for_cube(cal, args.cube)
    table_z = center[2] - HALF
    floor = table_z + 0.012  # fingertips never closer than 12 mm to the table
    heights = {"above": HALF + 0.08, "grasp": args.depth, "lifted": args.depth + args.lift}
    qs, yaw = plan_top_down(center, R, heights)
    print(f"cube centre {np.round(center * 1000, 1).tolist()} mm, top tilt {top_face_tilt_deg(R):.1f} deg, "
          f"table at z {table_z * 1000:.1f} mm; fingers along {np.degrees(yaw):.0f} deg")
    for k, h in heights.items():
        print(f"  {k:6s}: fingertips at z {(center[2] + h) * 1000:.1f} mm, joints {np.round(qs[k], 3).tolist()}")

    arm = SafeArm(args.ip)
    try:
        q0 = arm.q()
        if np.abs(q0 - REST).max() > 0.15:
            print("arm is not at the rest pose; not starting")
            return
        moves = [(q0, qs["above"]), (qs["above"], qs["grasp"]), (qs["grasp"], qs["lifted"]),
                 (qs["lifted"], qs["grasp"]), (qs["grasp"], qs["above"]), (qs["above"], REST)]
        for a, b in moves:
            arm.check_target(b, a, gripper_floor=floor)
        print("plan: all moves pass the workspace and joint checks")
        if args.plan:
            return
        try:
            print(f"gripper open at {arm.open_gripper(0.035) * 1000:.1f} mm", flush=True)
            arm.move_joints(qs["above"], speed=0.8, label="above the cube")
            arm.move_joints(qs["grasp"], speed=0.25, label="down to grasp", gripper_floor=floor)
            g = arm.close_gripper(args.force, settle=0.8)
            print(f"closed: fingers at {g * 1000:.1f} mm", flush=True)
            if g < 0.010:
                print("MISSED: nothing between the fingers; opening and backing off")
                arm.open_gripper(0.035)
                arm.move_joints(qs["above"], speed=0.4, label="back up", gripper_floor=floor)
                arm.rest_and_idle(speed=0.8)
                return
            arm.move_joints(qs["lifted"], speed=0.3, label="lift", gripper_floor=floor)
            time.sleep(0.8)
            g2 = arm.gripper_position()
            state, p_cam, _, vis = read_cube(cal, args.cube)
            tip = arm.model.fk(arm.q())[0]
            seen = (f"camera sees it at z {p_cam[2] * 1000:.0f} mm ({state}, {vis:.0%} unhidden), "
                    f"{(p_cam[2] - center[2]) * 1000:.0f} mm above where it was") if p_cam is not None else f"camera: {state}"
            print(f"LIFTED: fingers at {g2 * 1000:.1f} mm, fingertips at z {tip[2] * 1000:.0f} mm; {seen}", flush=True)
            if g2 < 0.010:
                print("the cube slipped out while lifting")
            time.sleep(1.5)
            arm.move_joints(qs["grasp"], speed=0.25, label="put it back", gripper_floor=floor)
            print(f"released: fingers at {arm.open_gripper(0.035) * 1000:.1f} mm", flush=True)
            arm.move_joints(qs["above"], speed=0.4, label="back up", gripper_floor=floor)
            arm.rest_and_idle(speed=0.8)
            time.sleep(1.0)
            state, p_end, _, _ = read_cube(cal, args.cube)
            if p_end is not None:
                print(f"back at rest; cube now at {np.round(p_end * 1000, 1).tolist()} mm "
                      f"({np.linalg.norm(p_end[:2] - center[:2]) * 1000:.1f} mm from where it started)")
            else:
                print(f"back at rest; cube {state}")
        except SafetyStop as e:
            print(f"SAFETY STOP: {e}; opening and returning to rest slowly", flush=True)
            time.sleep(1.0)
            arm.open_gripper(0.035)
            arm.move_joints(qs["above"], speed=0.3, label="retreat", gripper_floor=floor)
            arm.rest_and_idle(speed=0.4)
        finally:
            arm.gripper_idle()
    finally:
        if not arm.position_mode:
            arm.close()


if __name__ == "__main__":
    main()
