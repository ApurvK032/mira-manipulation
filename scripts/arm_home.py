"""First supervised motion: rest -> home -> rest, slowly, with every safety check on.

Usage:  python -m scripts.arm_home --dry-run      # connect, read, check the planned path; no motion
        python -m scripts.arm_home                # move (someone at the arm with a hand on the power switch)
Logs joint angles, the planned trajectory and external efforts to runs/arm_logs/ for tuning the limits.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import trossen_arm as ta

from robot.arm import HOME, REST, SafeArm, SafetyStop

LOGS = Path(__file__).resolve().parents[1] / "runs" / "arm_logs"


def summarize(arm: SafeArm, label: str, log):
    q, qp, e = np.array(log.q), np.array(log.q_plan), np.array(log.effort)
    print(f"  {label}: {log.t[-1]:.1f} s, max tracking error {np.degrees(np.abs(q - qp).max()):.1f} deg, "
          f"max |external effort| per joint (Nm) {np.round(np.abs(e).max(axis=0), 2).tolist()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.1.3")
    ap.add_argument("--speed", type=float, default=0.3, help="peak joint speed, rad/s (capped by the limits)")
    ap.add_argument("--hold", type=float, default=2.0, help="seconds to hold at home")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    arm = SafeArm(args.ip)
    q0 = arm.q()
    print(f"modes {arm.modes()}")
    print(f"joints now (rad) {np.round(q0, 3).tolist()}  gripper at {np.round(arm.ee()[:3], 3).tolist()} m")
    if np.abs(q0 - REST).max() > 0.15:
        print("arm is not at the rest pose; refusing to start from here")
        arm.close()
        return
    arm.check_target(HOME, q0)
    ee_home, _ = arm.model.fk(HOME)
    T = max(arm.limits.min_move_time, 1.875 * np.abs(HOME - q0).max() / min(args.speed, arm.limits.max_joint_speed))
    print(f"plan: rest -> home in about {T:.1f} s, gripper ends at {np.round(ee_home, 3).tolist()} m "
          f"(forward, left, up from the base); path checked against the workspace box")
    if args.dry_run:
        print("dry run: no motion")
        arm.close()
        return

    try:
        for k in (3, 2, 1):
            print(f"moving in {k} ...", flush=True)
            time.sleep(1.0)
        log = arm.move_joints(HOME, speed=args.speed, label="to home")
        summarize(arm, "to home", log)
        print(f"  at home: gripper at {np.round(arm.ee()[:3], 3).tolist()} m (model said "
              f"{np.round(ee_home, 3).tolist()})", flush=True)
        time.sleep(args.hold)
        arm.rest_and_idle(speed=args.speed)
        summarize(arm, "to rest", arm.logs[-1])
        print(f"back at rest, modes {arm.modes()}")
    except SafetyStop as e:
        if np.abs(arm.q() - REST).max() < 0.08:
            # stopped while settling onto the rest supports: it is at rest, so just brake it there
            print(f"SAFETY STOP near the rest pose ({e}); braking there")
            arm.drv.set_arm_modes(ta.Mode.idle)
            arm.position_mode = False
        else:
            print(f"SAFETY STOP: {e}\n  holding position for 3 s, then returning to rest at half speed", flush=True)
            time.sleep(3.0)
            try:
                arm.rest_and_idle(speed=args.speed / 2)
                print("returned to rest")
            except SafetyStop as e2:
                print(f"SAFETY STOP again: {e2}\n  holding for 120 s: support the arm and switch power off.")
                time.sleep(120)
    finally:
        LOGS.mkdir(parents=True, exist_ok=True)
        path = LOGS / f"home_{time.strftime('%Y%m%d_%H%M%S')}.npz"
        np.savez(path, **{f"{i}_{k}": np.array(getattr(l, k)) for i, l in enumerate(arm.logs)
                          for k in ("t", "q", "q_plan", "effort")})
        print(f"log: {path}")
        if arm.position_mode is False:
            arm.close()


if __name__ == "__main__":
    main()
