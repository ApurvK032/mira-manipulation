"""Stack one cube on another with checks and retries (first version of the scripted controller).

Each attempt: wait for both cubes to rest -> pick the top cube (top-down, by its top 15 mm, fingers on the pair of faces
whose approach keeps the open gripper 8 mm clear of the other cube) -> check the grip -> lift to a
carry height that clears the other cube -> carry over it -> re-read the bottom cube (re-plan if it
moved) -> lower until 2 mm above contact, faces aligned -> open -> rise -> verify with the camera
(top cube about 50 mm above the bottom one, centred within 12 mm). Any failure (missed grasp, drop,
bad placement) returns to rest and retries with fresh detections, up to --attempts times.

Usage:  python -m scripts.stack_test --plan      # wait for the cubes, plan and check; no motion
        python -m scripts.stack_test             # do it (someone at the power switch)
        python -m scripts.stack_test --top green --bottom red
"""

import argparse
import time

import numpy as np

from robot.arm import REST, SafeArm, SafetyStop
from robot.camera_frame import CameraToBase
from robot.grasp import HALF, plan_top_down, read_cube, top_face_tilt_deg, wait_for_cubes

# Fingertips 10 mm above the top cube's centre: the fingers hold its top 15 mm, so the gripper stays
# well clear of the table and of the bottom cube when placing.
GRASP = 0.010
HANG = GRASP + HALF  # fingertips to the bottom face of the held cube
# Fingertip height over the bottom cube's centre when placing: the held cube's centre sits at
# fingertips - GRASP, which should be 2 * HALF + 2 mm above the bottom cube's centre (2 mm above contact).
PLACE = 2 * HALF + 0.002 + GRASP


def measure(cal, top, bottom):
    """Where the top cube sits on the bottom one: dict of mm/deg, or None if either cube is not posed."""
    _, pt, Rt, _ = read_cube(cal, top)
    _, pb, _, _ = read_cube(cal, bottom)
    if pt is None or pb is None:
        return None
    d = (pt - pb) * 1000
    return {"dz_mm": round(float(d[2]), 1), "dxy_mm": round(float(np.hypot(d[0], d[1])), 1),
            "dx_mm": round(float(d[0]), 1), "dy_mm": round(float(d[1]), 1), "tilt_deg": round(top_face_tilt_deg(Rt), 1)}


def stacked(cal, top, bottom):
    m = measure(cal, top, bottom)
    if m is None:
        st, sb = read_cube(cal, top)[0], read_cube(cal, bottom)[0]
        return False, f"cannot see both cubes ({top} {st}, {bottom} {sb})"
    ok = 42 < m["dz_mm"] < 58 and m["dxy_mm"] < 12 and m["tilt_deg"] < 6
    return ok, (f"{top} is {m['dz_mm']:.0f} mm above {bottom}, {m['dxy_mm']:.1f} mm off centre "
                f"(x {m['dx_mm']:+.1f}, y {m['dy_mm']:+.1f}), tilted {m['tilt_deg']:.1f} deg")


def plan(cal, top, bottom, still_s=1.8, timeout=180.0, notify=None):
    cubes = wait_for_cubes(cal, [top, bottom], still_s=still_s, timeout=timeout, notify=notify)
    (ct, Rt), (cb, Rb) = cubes[top], cubes[bottom]
    table_z = min(ct[2], cb[2]) - HALF
    # fingertip height while carrying: the held cube (its bottom HANG below the fingertips) clears the
    # bottom cube by 3 cm; higher would shrink the straight-down reach far from the base
    carry_z = max(cb[2] + HALF + HANG + 0.03, ct[2] + 0.085)
    # the fingers close on the pair of faces that keeps the open gripper clear of the bottom cube, all
    # the way from rest down to the grasp and back up
    qt, yaw_t = plan_top_down(ct, Rt, {"above": HALF + 0.06, "grasp": GRASP, "carry": carry_z - ct[2]},
                              obstacles=[(bottom, cb, Rb)], start=REST)
    qb, yaw_b = plan_top_down(cb, Rb, {"carry": carry_z - cb[2], "place": PLACE})
    return dict(ct=ct, cb=cb, Rb=Rb, table_z=table_z, carry_z=carry_z, qt=qt, qb=qb, yaw_t=yaw_t, yaw_b=yaw_b)


def check_plan(arm, P):
    grasp_floor = P["table_z"] + 0.012
    carry_floor = P["carry_z"] - 0.02
    moves = [(arm.q(), P["qt"]["above"], None), (P["qt"]["above"], P["qt"]["grasp"], grasp_floor),
             (P["qt"]["grasp"], P["qt"]["carry"], grasp_floor), (P["qt"]["carry"], P["qb"]["carry"], carry_floor),
             (P["qb"]["carry"], P["qb"]["place"], None), (P["qb"]["place"], P["qb"]["carry"], None),
             (P["qb"]["carry"], REST, None)]
    for a, b, f in moves:
        arm.check_target(b, a, gripper_floor=f)


def attempt(arm, cal, P, top, bottom, force):
    grasp_floor, carry_floor = P["table_z"] + 0.012, P["carry_z"] - 0.02
    arm.open_gripper(0.035)
    arm.move_joints(P["qt"]["above"], speed=0.8, label=f"above {top}")
    _, p_now, _, _ = read_cube(cal, top)
    if p_now is None or np.linalg.norm(p_now - P["ct"]) > 0.01:
        return f"{top} cube moved before the grasp"
    arm.move_joints(P["qt"]["grasp"], speed=0.25, label="down to grasp", gripper_floor=grasp_floor)
    g = arm.close_gripper(force, settle=0.8)
    if g < 0.010:
        arm.open_gripper(0.035)
        arm.move_joints(P["qt"]["above"], speed=0.4, label="back up", gripper_floor=grasp_floor)
        return f"missed the grasp (fingers at {g * 1000:.1f} mm)"
    print(f"  grasped: fingers at {g * 1000:.1f} mm", flush=True)
    arm.move_joints(P["qt"]["carry"], speed=0.3, label="lift", gripper_floor=grasp_floor)
    if arm.gripper_position() < 0.010:
        return "dropped the cube while lifting"
    arm.move_joints(P["qb"]["carry"], speed=0.5, label=f"carry over {bottom}", gripper_floor=carry_floor)
    if arm.gripper_position() < 0.010:
        return "dropped the cube while carrying"
    _, pb_now, Rb_now, _ = read_cube(cal, bottom)
    if pb_now is None:
        return f"lost sight of the {bottom} cube"
    if np.linalg.norm(pb_now[:2] - P["cb"][:2]) > 0.01:
        print(f"  {bottom} moved {np.linalg.norm(pb_now[:2] - P['cb'][:2]) * 1000:.0f} mm: re-planning the place", flush=True)
        qb, _ = plan_top_down(pb_now, Rb_now, {"carry": P["carry_z"] - pb_now[2], "place": PLACE})
        arm.move_joints(qb["carry"], speed=0.4, label=f"over the moved {bottom}", gripper_floor=carry_floor)
        P["qb"], P["cb"] = qb, pb_now
    arm.move_joints(P["qb"]["place"], speed=0.2, label="lower onto the cube")
    arm.open_gripper(0.035)
    arm.move_joints(P["qb"]["carry"], speed=0.4, label="rise")
    time.sleep(1.0)
    P["placed"] = measure(cal, top, bottom)  # in place, arm still above the stack
    ok, msg = stacked(cal, top, bottom)
    return None if ok else f"placement check failed: {msg}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.1.3")
    ap.add_argument("--top", default="red")
    ap.add_argument("--bottom", default="green")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--force", type=float, default=15.0)
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()

    cal = CameraToBase()
    ok, msg = stacked(cal, args.top, args.bottom)
    if ok:
        print(f"already stacked: {msg}")
        return
    arm = SafeArm(args.ip)
    try:
        if np.abs(arm.q() - REST).max() > 0.15:
            print("arm is not at the rest pose; not starting")
            return
        for k in range(1, args.attempts + 1):
            print(f"attempt {k}: waiting for both cubes to rest ...", flush=True)
            P = plan(cal, args.top, args.bottom)
            print(f"  {args.top} at {np.round(P['ct'] * 1000, 1).tolist()} mm, {args.bottom} at "
                  f"{np.round(P['cb'] * 1000, 1).tolist()} mm, table z {P['table_z'] * 1000:.1f} mm, "
                  f"carry with fingertips at z {P['carry_z'] * 1000:.0f} mm", flush=True)
            check_plan(arm, P)
            print("  plan: all moves pass the workspace and joint checks", flush=True)
            if args.plan:
                return
            try:
                why = attempt(arm, cal, P, args.top, args.bottom, args.force)
            except SafetyStop as e:
                why = f"safety stop: {e}"
                print(f"  SAFETY STOP: {e}; holding, then returning to rest slowly", flush=True)
                time.sleep(1.0)
            arm.rest_and_idle(speed=0.6)
            if why is None:
                time.sleep(1.0)
                ok, msg = stacked(cal, args.top, args.bottom)
                print(f"STACKED on attempt {k}: {msg}")
                return
            print(f"  attempt {k} failed: {why}", flush=True)
        print(f"gave up after {args.attempts} attempts")
    finally:
        arm.gripper_idle()
        if not arm.position_mode:
            arm.close()


if __name__ == "__main__":
    main()
