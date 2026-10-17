"""Gripper check at the rest pose (the arm itself does not move): open, then close with limited force.

Usage:  python -m scripts.gripper_test [--open 0.035] [--force 15]
"""

import argparse
import time

from robot.arm import REST, SafeArm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.1.3")
    ap.add_argument("--open", type=float, default=0.035, help="finger position to open to, m")
    ap.add_argument("--force", type=float, default=15.0, help="closing force, N")
    args = ap.parse_args()
    arm = SafeArm(args.ip)
    try:
        print(f"arm joints {[round(x, 3) for x in arm.q()]} (at rest), gripper at {arm.gripper_position() * 1000:.1f} mm")
        print("opening ...", flush=True)
        p = arm.open_gripper(args.open, duration=0.8)
        print(f"  open: finger position {p * 1000:.1f} mm (commanded {args.open * 1000:.1f} mm)", flush=True)
        time.sleep(1.5)
        print(f"closing with {args.force:.0f} N ...", flush=True)
        p = arm.close_gripper(args.force, settle=1.0)
        print(f"  closed: finger position {p * 1000:.1f} mm (near 0 = nothing between the fingers)")
    finally:
        arm.gripper_idle()
        arm.close()


if __name__ == "__main__":
    main()
