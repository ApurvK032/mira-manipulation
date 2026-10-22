"""Which gripper mode holds a cube picked off the table? The arm picks the red cube once per variant, as the RL runner
would grab it: from straight above, fingers along its faces.

For each variant: hover over the cube (where the calibrated camera sees it), straight down to GRIP_Z above the table,
close with the variant, lift 6 cm, hold, wiggle +-2 cm sideways, hold again, put it back down, open, rise. The finger
readings are logged at about 50 Hz throughout: chatter shows as open/close reversals, a slip as the fingers closing
further while held (readings in the two holds, and before the lift vs after the wiggle). Variants as in
scripts/gripper_hold_test.py (extN, effN, posD).

Go to rest / Pause on the live page (or Ctrl-C) stops: hold, put down anything held when low enough, back to rest at
half speed, gripper idle.

Usage:  python -m scripts.grip_lift_test --ip 192.168.1.3 [--variants ext15 ...] [--dry]
Writes runs/gripper/lift_<time>.json.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from scripts.calibrate_camera import GRIP_Z, HOVER_Z, STATS, Operator, read_cube_now
from scripts.gripper_hold_test import OPEN, find_contact, record

ROOT = Path(__file__).resolve().parents[1]
LIFT = 0.06  # m above the grip height
WIGGLE = 0.02  # m sideways, each way


def assess(before, held, after):
    """before / after: finger position (m) just before the lift and after the wiggle; held: record() rows from the
    two holds at the top. Chatter shows as open/close reversals, a slip as the fingers closing further."""
    pos = held[:, 1] * 1000
    v = held[:, 2]
    moving = v[np.abs(v) > 0.005]
    return dict(pos_mm=round(float(np.median(pos)), 2), spread_mm=round(float(np.ptp(pos)), 2),
                closed_further_mm=round(1000 * float(before - after), 2),
                reversals=int(np.sum(np.diff(np.sign(moving)) != 0)) if len(moving) > 1 else 0,
                effort=round(float(np.median(np.abs(held[:, 3]))), 1), effort_peak=round(float(np.abs(held[:, 3]).max()), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.1.3")
    ap.add_argument("--variants", nargs="+", default=["ext15", "ext25", "eff18", "pos2.0", "pos1.0", "pos0.5"])
    ap.add_argument("--grip-z", type=float, default=GRIP_Z * 1000, help="mm above the table (50 is the cube's top)")
    ap.add_argument("--speed", type=float, default=0.6)
    ap.add_argument("--dry", action="store_true", help="DryArm, commands from runs/dry_command.json")
    args = ap.parse_args()

    import trossen_arm as ta
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
        raise SystemExit("the red cube must be flat on the table, 18-36 cm in front of the arm, and seen steadily")
    arm = DryArm() if args.dry else SafeArm(args.ip)
    drv = getattr(arm, "drv", None)
    op = Operator(args.dry)
    results, holding, yaw, logs = {}, False, None, {}

    def readings(seconds):
        if drv is None:  # dry: a steady hold
            n = int(seconds / 0.02)
            return np.c_[np.arange(n) * 0.02, np.full(n, arm.gripper_position()), np.zeros(n), np.full(n, 15.0),
                         np.full(n, 15.0)]
        return record(drv, seconds)
    out = ROOT / "runs" / "gripper"
    out.mkdir(parents=True, exist_ok=True)

    class Stop(Exception):
        pass

    def move(q, label, spd=args.speed, fl=floor):
        if arm.move_joints(q, speed=spd, label=label, gripper_floor=fl, watch=op.watch).stopped:
            raise Stop()

    def to_xyz(p, label, spd=0.3):
        q, err, _ = ik.solve(np.asarray(p, dtype=float), gripper_down_rotation(yaw), arm.q())
        if err > 0.002:
            raise SafetyStop(f"{label}: no IK solution at {np.round(np.asarray(p) * 1000).tolist()} mm")
        move(q, label, spd)

    def close(v):
        if drv is None:
            return arm.close_gripper()
        if v.startswith("ext"):
            drv.set_gripper_mode(ta.Mode.external_effort)
            drv.set_gripper_external_effort(-float(v[3:]), 0.0, False)
        elif v.startswith("eff"):
            drv.set_gripper_mode(ta.Mode.effort)
            drv.set_gripper_effort(-float(v[3:]), 0.0, False)
        else:
            contact = find_contact(drv)
            if contact is None:
                return None
            drv.set_gripper_position(contact - float(v[3:]) / 1000.0, 0.3, False)
        time.sleep(0.8)
        return arm.gripper_position()

    def open_(blocking=True):
        arm.open_gripper(OPEN) if blocking else arm.open_gripper(OPEN, duration=0.5, blocking=False)

    try:
        if np.abs(arm.q() - REST).max() > 0.15:
            raise SystemExit("arm is not at the rest pose; not starting")
        op.status("Grip lift test: waking")
        open_()
        move(WATCH, "to watch pose", fl=None)
        for v in args.variants:
            c, R = read_cube_now(cal)
            if c is None:
                raise SafetyStop("the red cube is not seen steadily on the table")
            qp, yaw = plan_top_down(np.array([c[0], c[1], 0.0]), R, {"hover": HOVER_Z, "grip": grip_z}, ik=ik, seed=arm.q())
            op.status(f"Grip lift test {v}: picking the red cube (Go to rest stops)")
            move(qp["hover"], f"{v}: over the cube")
            move(qp["grip"], f"{v}: down", spd=0.3)
            tip, _ = ik.fk(arm.q())
            g = close(v)
            if g is None or g < 0.010:
                print(f"{v}: closed on nothing ({g})", flush=True)
                results[v] = dict(held=False, note="closed on nothing")
                open_()
                to_xyz([tip[0], tip[1], HOVER_Z], f"{v}: rise")
                continue
            holding = True
            before = arm.gripper_position()
            to_xyz([tip[0], tip[1], grip_z + LIFT], f"{v}: lift")
            h1 = readings(1.5)
            for dy in (WIGGLE, -WIGGLE, 0.0):
                to_xyz([tip[0], tip[1] + dy, grip_z + LIFT], f"{v}: wiggle", spd=0.5)
            h2 = readings(1.0)
            after = arm.gripper_position()
            a = assess(before, np.vstack([h1, h2]), after)
            logs[v] = np.round(np.vstack([h1, h2]), 5).tolist()
            still = after > 0.010
            results[v] = dict(held=bool(still and a.get("closed_further_mm", 99) < 1.0), grip_mm=round(g * 1000, 2), **a)
            print(f"{v}: grip {g * 1000:.1f} mm -> {'HELD' if results[v]['held'] else 'SLIPPED'}; {a}", flush=True)
            op.status(f"Grip lift test {v}: {'held' if results[v]['held'] else 'slipped'} "
                      f"(effort {a.get('effort', float('nan')):.0f} N, reversals {a.get('reversals')})")
            to_xyz([tip[0], tip[1], grip_z + 0.003], f"{v}: put down")
            open_()
            holding = False
            to_xyz([tip[0], tip[1], HOVER_Z], f"{v}: rise")
        op.status("Grip lift test: done, back to rest")
        move(WATCH, "to watch pose", fl=None)
        arm.rest_and_idle(speed=args.speed)
    except (Stop, SafetyStop, KeyboardInterrupt, Exception) as e:
        why = "operator stop" if isinstance(e, (Stop, KeyboardInterrupt)) else f"{type(e).__name__}: {e}"
        op.status(f"Grip lift test stopped ({why}): back to rest at half speed")
        try:
            arm.hold()
            if holding:
                tip, _ = ik.fk(arm.q())
                if tip[2] < grip_z + LIFT + 0.02:  # low and over the table: set the cube down before leaving
                    q, err, _ = ik.solve(np.array([tip[0], tip[1], grip_z + 0.003]), gripper_down_rotation(yaw), arm.q())
                    if err <= 0.002:
                        arm.move_joints(q, speed=0.2, label="put down", gripper_floor=floor)
                        open_()
        except Exception as e2:
            print(f"(could not set the cube down: {e2})", flush=True)
        time.sleep(0.5)
        arm.rest_and_idle(speed=args.speed / 2)
    finally:
        try:
            arm.gripper_idle()
        except Exception:
            pass
        path = out / f"lift_{time.strftime('%Y%m%d_%H%M%S')}{'_dry' if args.dry else ''}.json"
        path.write_text(json.dumps(dict(args=vars(args), results=results, logs=logs)))
        print(f"\nsaved {path}", flush=True)
        if not args.dry:
            arm.close()
    print("\nsummary:")
    for v, r in results.items():
        print(f"  {v:7s} {'HELD   ' if r.get('held') else 'SLIPPED'} grip {r.get('grip_mm')} mm, effort {r.get('effort')} N "
              f"(peak {r.get('effort_peak')}), reversals {r.get('reversals')}, closed further {r.get('closed_further_mm')} mm")


if __name__ == "__main__":
    main()
