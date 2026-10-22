"""How steadily does the gripper hold a cube, per control mode? The arm joints are never commanded (arm stays at rest).

Firmware 1.9.4 + driver 1.9.3: closing on a cube in external-effort mode (SafeArm.close_gripper) was seen to chatter,
opening and closing again and again at the cube's width, and the cube slipped out on the first arm move; position mode
held it, but squeezed at about 83 N. Each variant here: the live page's banner names it and asks the operator to hold
the cube square between the open fingers and press Resume (Go to rest ends the test); then close, hold HOLD s while
the operator tugs and twists the cube, open. (--countdown S instead: a timed countdown, printed only on the console.) Logged at about 50 Hz: finger position (chatter
shows as reversals; a slip as the fingers closing further during the hold), velocity, effort.

Variants (default: ext15 ext25 eff18 pos2.0 pos1.0 pos0.5):
  extN    external-effort mode, N newtons (SafeArm.close_gripper's mode; it uses 15)
  effN    effort mode, N newtons of total effort
  posD    position mode: close slowly until the cube stops the fingers (contact), then target D mm inside that

Usage:  python -m scripts.gripper_hold_test --ip 192.168.1.3 [--variants ext15 pos1.0 ...]
Writes runs/gripper/hold_<time>.json.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trossen_arm as ta

ROOT = Path(__file__).resolve().parents[1]
OPEN = 0.035


def wait_for_resume(op, msg):
    """Show msg on the live page and wait for Resume (True) or Go to rest / Pause (False)."""
    op.status(msg)
    while True:
        c = op.poll()
        if c in ("resume", "start"):
            return True
        if c in ("rest", "pause"):
            return False
        time.sleep(0.1)


def countdown(s, what):
    for k in range(int(round(s)), 0, -1):
        print(f"  {what} in {k} s: hold the red cube square between the fingers", flush=True)
        time.sleep(1.0)


def record(drv, seconds, dt=0.02):
    rows, t0 = [], time.time()
    while (t := time.time() - t0) < seconds:
        rows.append((t, drv.get_gripper_position(), drv.get_gripper_velocity(), drv.get_gripper_effort(),
                     drv.get_gripper_external_effort()))
        time.sleep(dt)
    return np.array(rows)


def summary(r, settle=0.5):
    late = r[r[:, 0] > settle]
    v = late[:, 2]
    moving = v[np.abs(v) > 0.005]  # m/s: ignore sensor noise when still
    reversals = int(np.sum(np.diff(np.sign(moving)) != 0)) if len(moving) > 1 else 0
    first, last = np.median(late[:10, 1]), np.median(late[-10:, 1])
    return dict(pos_mm=round(1000 * float(np.median(late[:, 1])), 2),
                pos_spread_mm=round(1000 * float(np.ptp(late[:, 1])), 2),
                closed_further_mm=round(1000 * float(first - last), 2),  # > 1 mm: the cube slipped (fingers followed)
                reversals=reversals, effort=round(float(np.median(late[:, 3])), 2),
                effort_peak=round(float(np.abs(late[:, 3]).max()), 2),
                external_effort=round(float(np.median(late[:, 4])), 2))


def find_contact(drv, step=0.0003, dt=0.05, timeout=8.0):
    """Close slowly in position mode until something stops the fingers; returns that finger position (m) or None."""
    drv.set_gripper_mode(ta.Mode.position)
    goal, t0 = drv.get_gripper_position(), time.time()
    while time.time() - t0 < timeout and goal > 0.003:
        goal -= step
        drv.set_gripper_position(goal, dt, False)
        time.sleep(dt)
        pos = drv.get_gripper_position()
        if pos - goal > 0.001 and abs(drv.get_gripper_velocity()) < 0.003:
            drv.set_gripper_position(pos, 0.1, False)  # stop pushing: hold where the cube stopped them
            time.sleep(0.2)
            return drv.get_gripper_position()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.1.3")
    ap.add_argument("--variants", nargs="+", default=["ext15", "ext25", "eff18", "pos2.0", "pos1.0", "pos0.5"])
    ap.add_argument("--hold", type=float, default=5.0, help="s to hold (tug and twist the cube meanwhile)")
    ap.add_argument("--countdown", type=float, default=0.0,
                    help="s of console countdown before each close instead of waiting for Resume on the live page")
    args = ap.parse_args()

    op = None
    if not args.countdown:
        from scripts.calibrate_camera import Operator

        op = Operator(dry=False)
    drv = ta.TrossenArmDriver()
    drv.configure(ta.Model.wxai_v0, ta.StandardEndEffector.wxai_v0_follower, args.ip, False)
    out, results = ROOT / "runs" / "gripper", {}
    out.mkdir(parents=True, exist_ok=True)
    try:
        print(f"arm joint modes (untouched): {[str(m).split('.')[-1] for m in drv.get_modes()]}", flush=True)
        drv.set_gripper_mode(ta.Mode.position)
        drv.set_gripper_position(OPEN, 0.8, True)
        for v in args.variants:
            print(f"\n{v}:", flush=True)
            if op is None:
                countdown(args.countdown, "closing")
            elif not wait_for_resume(op, f"Grip test {v}: hold the red cube square between the open fingers, "
                                         f"then press Resume (Go to rest ends the test)"):
                print("  ended from the page", flush=True)
                break
            extra = {}
            if v.startswith("ext"):
                drv.set_gripper_mode(ta.Mode.external_effort)
                drv.set_gripper_external_effort(-float(v[3:]), 0.0, False)
            elif v.startswith("eff"):
                drv.set_gripper_mode(ta.Mode.effort)
                drv.set_gripper_effort(-float(v[3:]), 0.0, False)
            else:
                contact = find_contact(drv)
                if contact is None:
                    print("  no contact found (nothing between the fingers?)", flush=True)
                    drv.set_gripper_position(OPEN, 0.8, True)
                    continue
                target = contact - float(v[3:]) / 1000.0
                drv.set_gripper_position(target, 0.3, False)
                extra = dict(contact_mm=round(1000 * contact, 2), target_mm=round(1000 * target, 2))
            print("  closed: tug and twist the cube now", flush=True)
            if op is not None:
                op.status(f"Grip test {v}: closed. Tug and twist the cube for 5 s")
            r = record(drv, args.hold + 0.5)
            results[v] = dict(summary(r), **extra, log=np.round(r, 5).tolist())
            s = results[v]
            print(f"  {v}: finger {s['pos_mm']} mm (spread {s['pos_spread_mm']}, closed further "
                  f"{s['closed_further_mm']} mm), reversals {s['reversals']}, effort {s['effort']} "
                  f"(peak {s['effort_peak']}), external {s['external_effort']}"
                  + (f", contact {extra['contact_mm']} -> target {extra['target_mm']} mm" if extra else ""), flush=True)
            drv.set_gripper_mode(ta.Mode.position)
            drv.set_gripper_position(OPEN, 0.8, True)
            print("  opened", flush=True)
            if op is not None:
                op.status(f"Grip test {v}: done ({s['pos_mm']} mm, effort {abs(s['effort']):.0f} N, "
                          f"closed further {s['closed_further_mm']} mm). Opened.")
    finally:
        try:
            drv.set_gripper_mode(ta.Mode.idle)
        finally:
            path = out / f"hold_{time.strftime('%Y%m%d_%H%M%S')}.json"
            path.write_text(json.dumps(dict(args=vars(args), results=results)))
            print(f"\nsaved {path}; gripper idle", flush=True)
            if op is not None:
                op.status("Grip test finished: gripper idle")
            drv.cleanup()


if __name__ == "__main__":
    main()
