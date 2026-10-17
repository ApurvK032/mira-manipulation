"""Find WidowX AI arms on the network and read their state. Does not move the arm.

Usage:  python -m scripts.check_arm                    # the lab arm (192.168.1.3), or scan if the driver can
        python -m scripts.check_arm --ip 192.168.1.2
Before connecting, make sure the arm is resting (folded in its sleep pose or supported): whether the
driver changes the motor mode on connect / cleanup is not documented, and a held arm would drop.
The driver version must match the arm's firmware (lab arm: 1.8.3, see environment.yml).
"""

import argparse

import numpy as np
import trossen_arm as ta


def read_arm(ip: str, follower: bool = True, clear_error: bool = False):
    ee = ta.StandardEndEffector.wxai_v0_follower if follower else ta.StandardEndEffector.wxai_v0_leader
    drv = ta.TrossenArmDriver()
    drv.configure(ta.Model.wxai_v0, ee, ip, clear_error)
    try:
        print(f"  driver {drv.get_driver_version()}  controller {drv.get_controller_version()}")
        print(f"  error: {drv.get_error_information() or 'none'}")
        print(f"  joint modes: {[str(m).split('.')[-1] for m in drv.get_modes()]}")
        print(f"  arm joints (rad): {np.round(drv.get_arm_positions(), 3).tolist()}")
        print(f"  gripper (m):      {drv.get_gripper_position():.4f}")
        print(f"  end effector (m, axis-angle): {np.round(list(drv.get_cartesian_positions()), 4).tolist()}")
        print(f"  joint temperatures (C): {np.round(drv.get_all_rotor_temperatures(), 1).tolist()}")
    finally:
        drv.cleanup()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", action="append", help="arm IP (repeatable); default: scan the subnet")
    ap.add_argument("--subnet", default="192.168.1")
    ap.add_argument("--leader", action="store_true", help="arm has the leader (teleop) end effector")
    ap.add_argument("--clear-error", action="store_true", help="clear the controller's error state when connecting")
    args = ap.parse_args()

    ips = args.ip
    if not ips and not hasattr(ta.TrossenArmDriver, "discover"):
        ips = ["192.168.1.3"]  # driver 1.8.x has no discovery; the lab arm's address
    if not ips:
        print(f"Scanning {args.subnet}.1-254 ...")
        found = ta.TrossenArmDriver.discover(subnet=args.subnet, ip_start=1, ip_end=254, timeout=0.01)
        for r in found:
            print(f"  found {r.ip}  model={ta.MODEL_NAME[r.model]}  firmware={r.firmware_version}  "
                  f"error={ta.ERROR_INFORMATION[r.error_state]}")
        ips = [r.ip for r in found]
        if not ips:
            print("No arm found. Check the Ethernet cable and that this machine has an IP on the arm subnet "
                  f"({args.subnet}.x). On WSL, use mirrored networking (see README).")
            return
    for ip in ips:
        print(f"\nArm {ip}")
        read_arm(ip, follower=not args.leader, clear_error=args.clear_error)


if __name__ == "__main__":
    main()
