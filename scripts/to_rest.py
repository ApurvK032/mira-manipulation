"""Recovery: take the arm to rest from wherever it was left (e.g. a controller that stopped while awake).

Refuses to move if the gripper looks closed on something. Path: straight up if the fingertips are low,
then the watch pose, then rest (brakes on); every move checked by SafeArm as usual.
Usage:  python -m scripts.to_rest
"""

import numpy as np

from robot.arm import REST, WATCH, SafeArm
from robot.grasp import plan_top_down
from robot.ik import IK


def main():
    arm = SafeArm()
    try:
        q, g = arm.q(), arm.gripper_position()
        tip = arm.model.fk(q)[0]
        print(f"joints {np.round(q, 2).tolist()}, fingertips at {np.round(tip * 1000).tolist()} mm, gripper {g * 1000:.1f} mm")
        if np.abs(q - REST).max() < 0.05:
            print("already at rest")
            return
        if 0.010 <= g <= 0.031:
            print("the gripper may be holding something: not moving. Take the cube out, then run again.")
            return
        if tip[2] < 0.10:
            qs, _ = plan_top_down(np.array([tip[0], tip[1], 0.0]), np.eye(3), {"up": 0.13}, ik=IK(), seed=q)
            arm.move_joints(qs["up"], speed=0.4, label="straight up")
        arm.move_joints(WATCH, speed=0.6, label="to watch pose")
        arm.rest_and_idle(speed=0.6)
        print("at rest (brakes on)")
    finally:
        arm.gripper_idle()
        if not arm.position_mode:
            arm.close()


if __name__ == "__main__":
    main()
