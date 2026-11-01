"""Measure where the gripper really centres on a cube, relative to where the camera sees it, and (with --save) store the
difference as the calibration's xy correction (assets/calibration.yaml: xy_correction_m; robot.camera_frame applies it).

Why it works: the gripper's two fingers close symmetrically, so a cube picked off-centre is pushed until it sits midway
between them. The arm picks the red cube where the camera sees it, lifts it 1 cm, puts it back on the same spot and
opens: the cube now sits where the gripper's centre was. The camera's reading moves by exactly the calibration error
along the fingers' closing direction (across it, the pads don't push the cube). Picks alternate the fingers between the
cube's two face directions, so both horizontal components are measured; the arm goes to the watch pose between picks
so the camera sees the cube unhidden.

Each pick: read the cube (1 s median, at the watch pose) -> over it -> slowly down to the grasp height -> close
(effort mode) -> lift 1 cm -> put back -> open -> rise -> watch pose -> read again.
Go to rest / Pause on the live page (or Ctrl-C) stops: hold, set a held cube down, rise, rest.

Usage:  python -m scripts.pick_bias --ip 192.168.1.3 [--picks 4] [--save] [--dry]
Without --save it only reports. With --save it backs up assets/calibration.yaml and adds the new error to any correction
already in the file (so a verification run after saving should measure about 0).
"""

import argparse
import json
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np
import yaml

from scripts.calibrate_camera import GRIP_Z, HOVER_Z, OUT, STATS, Operator

ROOT = Path(__file__).resolve().parents[1]


class Stop(Exception):
    pass


def read_cube(cal, seconds=1.0, min_unhidden=0.9):
    """Median arm-frame position and the last rotation of the red cube over `seconds`, from fresh, mostly unhidden
    TRACKING readings: (p, R, n) or (None, None, n)."""
    from robot.grasp import cubes_in_base

    pts, R, t0 = [], None, time.time()
    while time.time() - t0 < seconds:
        s = json.load(urllib.request.urlopen(STATS, timeout=1))
        if time.time() - s.get("t", 0) < 0.2:
            state, p, Rm, vis = cubes_in_base(s, cal).get("red", ("", None, None, 0.0))
            if state == "TRACKING" and p is not None and vis >= min_unhidden:
                pts.append(p)
                R = Rm
        time.sleep(0.03)
    return (np.median(pts, axis=0), R, len(pts)) if len(pts) >= 10 else (None, None, len(pts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.1.3")
    ap.add_argument("--picks", type=int, default=4, help="picks, alternating the finger direction (even number)")
    ap.add_argument("--grip-z", type=float, default=GRIP_Z * 1000, help="mm above the table to grasp at")
    ap.add_argument("--force", type=float, default=18.0, help="N, effort-mode close")
    ap.add_argument("--speed", type=float, default=0.5)
    ap.add_argument("--save", action="store_true", help="store the measured error in assets/calibration.yaml (backup first)")
    ap.add_argument("--dry", action="store_true", help="DryArm; commands from runs/dry_command.json; never saves")
    args = ap.parse_args()

    from robot.arm import REST, WATCH, SafeArm, SafetyStop
    from robot.camera_frame import CameraToBase
    from robot.controller import DryArm
    from robot.grasp import REACH, face_yaw
    from robot.ik import IK, gripper_down_rotation

    grip_z, floor = args.grip_z / 1000.0, 0.012
    s = json.load(urllib.request.urlopen(STATS, timeout=2))
    if not s.get("fps") or s["fps"] < 20 or time.time() - s.get("t", 0) > 0.5:
        raise SystemExit("live perception (scripts.web_view) is not running at camera rate; not starting")
    cal, ik = CameraToBase(), IK()
    c, R, n = read_cube(cal)
    if c is None or not (REACH[0] < np.hypot(c[0], c[1]) < REACH[1] - 0.02 and c[0] > 0.1):
        raise SystemExit("red must be flat on the table, about 20-33 cm in front of the arm, and seen steadily")
    arm = DryArm() if args.dry else SafeArm(args.ip)
    op = Operator(args.dry)
    holding, results = False, []
    yaw = None

    def move(q, label, spd, fl=floor):
        if arm.move_joints(q, speed=spd, label=label, gripper_floor=fl, watch=op.watch).stopped:
            raise Stop()

    def to_xyz(p, label, spd=0.2):
        q, err, _ = ik.solve(np.asarray(p, dtype=float), gripper_down_rotation(yaw), arm.q())
        if err > 0.002:
            raise SafetyStop(f"{label}: no IK solution at {np.round(np.asarray(p) * 1000).tolist()} mm")
        move(q, label, spd)

    def over(p, label):
        """From anywhere above the table to straight over p at hover height, gripper turned to `yaw`."""
        az = np.arctan2(p[1], p[0])
        seeds = [arm.q()] + [np.array([az, a, b, w, 0, 0]) for a in (0.4, 0.9, 1.4) for b in (0.4, 1.0, 1.6) for w in (0.5, 1.2)]
        q, err, _ = min((ik.solve(np.array([p[0], p[1], HOVER_Z]), gripper_down_rotation(yaw), sd) for sd in seeds),
                        key=lambda r: r[1] + r[2] / 100)
        if err > 0.002:
            raise SafetyStop(f"{label}: no IK solution over {np.round(p[:2] * 1000).tolist()} mm")
        move(q, label, args.speed)

    try:
        if np.abs(arm.q() - REST).max() > 0.15:
            raise SystemExit("arm is not at the rest pose; not starting")
        op.status("Pick bias: waking")
        arm.open_gripper(0.035)
        move(WATCH, "to watch pose", args.speed, fl=None)
        for i in range(args.picks):
            c, R, n = read_cube(cal)
            if c is None:
                raise SafetyStop(f"the red cube is not seen steadily and unhidden ({n} samples)")
            base = face_yaw(R)
            # fingers along one face direction, then the other; of each pair of opposite yaws, the one nearer the wrist's
            # zero (less wrist roll)
            yaw = base + (np.pi / 2 if i % 2 else 0.0)
            yaw = min((yaw + k * np.pi for k in (-2, -1, 0, 1, 2)), key=abs)
            axis = np.array([np.cos(yaw), np.sin(yaw)])
            op.status(f"Pick bias {i + 1}/{args.picks}: fingers closing along {np.degrees(yaw):.0f} deg (Go to rest stops)")
            over(c, f"pick {i + 1}: over the cube")
            to_xyz([c[0], c[1], grip_z], f"pick {i + 1}: down", spd=0.15)
            g = arm.close_gripper(args.force, settle=1.2, mode="effort")
            if g < 0.010:
                arm.open_gripper(0.035)
                to_xyz([c[0], c[1], HOVER_Z], "rise", spd=0.3)
                raise SafetyStop("the gripper closed on nothing")
            holding = True
            to_xyz([c[0], c[1], grip_z + 0.010], f"pick {i + 1}: lift 1 cm", spd=0.15)
            time.sleep(0.3)
            to_xyz([c[0], c[1], grip_z + 0.002], f"pick {i + 1}: put back", spd=0.15)
            arm.open_gripper(0.035)
            holding = False
            to_xyz([c[0], c[1], HOVER_Z], f"pick {i + 1}: rise", spd=0.3)
            move(WATCH, "to watch pose", args.speed, fl=None)
            time.sleep(0.5)
            c2, _, n2 = read_cube(cal)
            if c2 is None:
                raise SafetyStop(f"the red cube is not seen steadily after pick {i + 1} ({n2} samples)")
            d = c2[:2] - c[:2]
            along = float(d @ axis)
            results.append(dict(pick=i + 1, yaw_deg=round(float(np.degrees(yaw)), 1), axis=np.round(axis, 4).tolist(),
                                before_mm=np.round(c * 1000, 1).tolist(), after_mm=np.round(c2 * 1000, 1).tolist(),
                                along_mm=round(1000 * along, 2), across_mm=round(1000 * float(d @ [-axis[1], axis[0]]), 2),
                                grip_mm=round(g * 1000, 2)))
            print(f"pick {i + 1}: fingers along {np.degrees(yaw):5.0f} deg, grip {g * 1000:.1f} mm; cube moved "
                  f"{1000 * along:+.1f} mm along the fingers, {results[-1]['across_mm']:+.1f} mm across", flush=True)
        op.status("Pick bias: done, back to rest")
        arm.rest_and_idle(speed=args.speed)
    except (Stop, SafetyStop, KeyboardInterrupt, Exception) as e:
        if isinstance(e, SystemExit):
            raise
        why = "operator stop" if isinstance(e, (Stop, KeyboardInterrupt)) else f"{type(e).__name__}: {e}"
        op.status(f"Pick bias stopped ({why}): back to rest")
        try:
            arm.hold()
            tip, _ = ik.fk(arm.q())
            if holding and tip[2] < grip_z + 0.03:
                to_xyz([tip[0], tip[1], grip_z + 0.002], "put down", spd=0.15)
            arm.open_gripper(0.035)
            if tip[2] < 0.10:
                to_xyz([tip[0], tip[1], HOVER_Z], "rise", spd=0.3)
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

    out = ROOT / "runs" / "calib"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"pick_bias_{time.strftime('%Y%m%d_%H%M%S')}{'_dry' if args.dry else ''}.json"
    if len(results) < 2:
        path.write_text(json.dumps(dict(results=results), indent=1))
        print(f"too few picks to estimate both directions; saved {path}")
        return
    # least squares: each pick measures e . axis_i
    A = np.array([r["axis"] for r in results])
    b = np.array([r["along_mm"] for r in results]) / 1000.0
    e, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = 1000 * (A @ e - b)
    print(f"\ncamera error at the grasp (arm frame): x {1000 * e[0]:+.1f} mm, y {1000 * e[1]:+.1f} mm "
          f"({1000 * np.hypot(*e):.1f} mm); residuals {np.round(resid, 1).tolist()} mm", flush=True)
    path.write_text(json.dumps(dict(error_m=np.round(e, 5).tolist(), results=results), indent=1))
    print(f"saved {path}")
    if args.save and not args.dry:
        c_file = yaml.safe_load(OUT.read_text())
        old = np.array(c_file.get("xy_correction_m") or [0.0, 0.0], dtype=float)
        new = old + e
        backup = OUT.with_name(f"calibration_before_pickbias_{time.strftime('%Y%m%d_%H%M')}.yaml")
        shutil.copy(OUT, backup)
        c_file["xy_correction_m"] = np.round(new, 5).tolist()
        c_file["xy_correction_note"] = (f"scripts/pick_bias.py {time.strftime('%Y-%m-%d %H:%M')}: {len(results)} picks, "
                                        f"{path.relative_to(ROOT)}; previous {np.round(old * 1000, 1).tolist()} mm")
        OUT.write_text(yaml.safe_dump(c_file, sort_keys=False))
        print(f"saved xy_correction_m {np.round(new * 1000, 1).tolist()} mm to {OUT.relative_to(ROOT)} (backup "
              f"{backup.name}); restart scripts.web_view is not needed, but restart any arm script so it reloads it")
    elif args.save:
        print("dry run: nothing saved")


if __name__ == "__main__":
    main()
