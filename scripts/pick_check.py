"""Slow pick check: is the gripper centred on the red cube when it comes down to pick it?

The arm hovers over red (where the calibrated camera sees it), comes down slowly to 1 cm above the cube's top and waits:
the operator looks from above and from the side whether the open fingers straddle the cube evenly, then presses
Resume on the live page. It then goes down to the grasp height, closes gently (effort mode), lifts 2 cm, holds, puts
the cube back, opens, rises and goes to rest. Go to rest / Pause on the page (or Ctrl-C) stops at any point: hold, rise
(setting a held cube down first), rest.

Printed at the pause: where the camera sees the cube relative to the fingertips (x, y in the arm frame). If the
operator sees the fingers off-centre, that offset and the operator's estimate together say how far the calibration is
off.

Usage:  python -m scripts.pick_check --ip 192.168.1.3 [--grip-z 36] [--force 18] [--dry]
"""

import argparse
import json
import time
import urllib.request

import numpy as np

from scripts.calibrate_camera import GRIP_Z, HOVER_Z, STATS, Operator, read_cube_now

ABOVE = 0.060  # m above the table: fingertips 1 cm above the 50 mm cube top
LIFT = 0.020  # m lift with the cube


class Stop(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.1.3")
    ap.add_argument("--grip-z", type=float, default=GRIP_Z * 1000, help="mm above the table to grasp at (50 = top)")
    ap.add_argument("--force", type=float, default=18.0, help="N, effort-mode close")
    ap.add_argument("--speed", type=float, default=0.5)
    ap.add_argument("--dry", action="store_true", help="DryArm; commands from runs/dry_command.json")
    args = ap.parse_args()

    from robot.arm import REST, WATCH, SafeArm, SafetyStop
    from robot.camera_frame import CameraToBase
    from robot.controller import DryArm
    from robot.grasp import REACH, plan_top_down
    from robot.ik import IK, gripper_down_rotation

    grip_z, floor = args.grip_z / 1000.0, 0.012
    s = json.load(urllib.request.urlopen(STATS, timeout=2))
    if not s.get("fps") or s["fps"] < 20 or time.time() - s.get("t", 0) > 0.5:
        raise SystemExit("live perception (scripts.web_view) is not running at camera rate; not starting")
    cal, ik = CameraToBase(), IK()
    c, R = read_cube_now(cal)
    if c is None or not (REACH[0] < np.hypot(c[0], c[1]) < REACH[1] and c[0] > 0.1):
        raise SystemExit("red must be flat on the table, 18-36 cm in front of the arm, and seen steadily")
    qp, yaw = plan_top_down(np.array([c[0], c[1], 0.0]), R, {"hover": HOVER_Z, "above": ABOVE, "grip": grip_z}, ik=ik)
    arm = DryArm() if args.dry else SafeArm(args.ip)
    op = Operator(args.dry)
    holding = False

    def move(q, label, spd, fl=floor):
        if arm.move_joints(q, speed=spd, label=label, gripper_floor=fl, watch=op.watch).stopped:
            raise Stop()

    def straight(z, label, spd=0.2):
        tip, _ = ik.fk(arm.q())
        q, err, _ = ik.solve(np.array([tip[0], tip[1], z]), gripper_down_rotation(yaw), arm.q())
        if err > 0.002:
            raise SafetyStop(f"{label}: no IK solution at z {z * 1000:.0f} mm")
        move(q, label, spd)

    try:
        if np.abs(arm.q() - REST).max() > 0.15:
            raise SystemExit("arm is not at the rest pose; not starting")
        print(f"red seen at {np.round(c * 1000).tolist()} mm, {100 * np.hypot(c[0], c[1]):.1f} cm from the base", flush=True)
        op.status("Pick check: waking, then over the red cube")
        arm.open_gripper(0.035)
        move(WATCH, "to watch pose", args.speed, fl=None)
        move(qp["hover"], "over the red cube", args.speed)
        move(qp["above"], "slowly down to 1 cm above the cube", 0.2)
        tip, _ = ik.fk(arm.q())
        cn, _ = read_cube_now(cal)
        if cn is not None:
            d = (cn[:2] - tip[:2]) * 1000
            print(f"at the pause the camera sees the cube {d[0]:+.1f} mm in x (forward), {d[1]:+.1f} mm in y (left) of the "
                  f"fingertips ({np.hypot(*d):.1f} mm)", flush=True)
        op.status("Pick check: are the fingers centred around red? Resume = pick it, Go to rest = stop")
        while True:
            cmd = op.poll()
            if cmd in ("resume", "start"):
                break
            if cmd in ("rest", "pause"):
                raise Stop()
            time.sleep(0.1)
        op.status("Pick check: picking slowly")
        straight(grip_z, f"down to {grip_z * 1000:.0f} mm", spd=0.15)
        g = arm.close_gripper(args.force, settle=1.2, mode="effort")
        print(f"closed: fingers at {g * 1000:.1f} mm per finger (a 5 cm cube held flat reads about 27.3)", flush=True)
        if g < 0.010:
            arm.open_gripper(0.035)
            straight(HOVER_Z, "rise")
            raise SafetyStop("the gripper closed on nothing")
        holding = True
        straight(grip_z + LIFT, "lift 2 cm", spd=0.15)
        time.sleep(2.0)
        g2 = arm.gripper_position()
        print(f"held 2 s at 2 cm: fingers {g2 * 1000:.1f} mm ({'steady' if abs(g2 - g) < 0.0008 else 'moved: slipping?'})",
              flush=True)
        op.status(f"Pick check: held at {g2 * 1000:.1f} mm; putting it back")
        straight(grip_z + 0.003, "put down", spd=0.15)
        arm.open_gripper(0.035)
        holding = False
        straight(HOVER_Z, "rise", spd=0.3)
        move(WATCH, "to watch pose", args.speed, fl=None)
        arm.rest_and_idle(speed=args.speed)
        op.status("Pick check done: arm at rest")
    except (Stop, SafetyStop, KeyboardInterrupt, Exception) as e:
        if isinstance(e, SystemExit):
            raise
        why = "operator stop" if isinstance(e, (Stop, KeyboardInterrupt)) else f"{type(e).__name__}: {e}"
        op.status(f"Pick check stopped ({why}): back to rest")
        try:
            arm.hold()
            tip, _ = ik.fk(arm.q())
            if holding and tip[2] < grip_z + LIFT + 0.01:
                straight(grip_z + 0.003, "put down", spd=0.15)
                arm.open_gripper(0.035)
            if tip[2] < 0.10:
                arm.open_gripper(0.035)
                straight(HOVER_Z, "rise", spd=0.3)
        except Exception as e2:
            print(f"(recovery move refused: {e2})", flush=True)
        time.sleep(0.5)
        arm.rest_and_idle(speed=args.speed / 2)
    finally:
        try:
            arm.gripper_idle()
        except Exception:
            pass
        if not args.dry:
            arm.close()


if __name__ == "__main__":
    main()
