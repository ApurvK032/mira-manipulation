"""Back up the arm controller's persistent configuration to a YAML file, before a firmware upgrade.

Trossen's firmware upgrade guide: use the driver that matches the CURRENT firmware (here 1.8.3) to save every
configuration before flashing; an upgrade keeps compatible fields, and a downgrade resets everything, including the
factory calibration (joint characteristics). Connects, reads the versions, saves, disconnects. Does not enable or move
the motors. As with scripts.check_arm: make sure the arm is resting (folded on its supports) before connecting.

Usage:  python -m scripts.backup_arm_config [--ip 192.168.1.3]
Writes  runs/arm_backup/arm_config_<time>_controller-<version>.yaml
"""

import argparse
import time
from pathlib import Path

import numpy as np
import trossen_arm as ta

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.1.3")
    args = ap.parse_args()
    drv = ta.TrossenArmDriver()
    drv.configure(ta.Model.wxai_v0, ta.StandardEndEffector.wxai_v0_follower, args.ip, False)
    try:
        ctrl, driver = drv.get_controller_version(), drv.get_driver_version()
        print(f"driver {driver}, controller {ctrl}; arm joints (rad) {np.round(drv.get_arm_positions(), 3).tolist()}")
        out = ROOT / "runs" / "arm_backup"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"arm_config_{time.strftime('%Y%m%d_%H%M%S')}_controller-{ctrl}.yaml"
        drv.save_configs_to_file(str(path))
        print(f"saved {path} ({path.stat().st_size} bytes). Keep a copy off this PC too.")
    finally:
        drv.cleanup()


if __name__ == "__main__":
    main()
