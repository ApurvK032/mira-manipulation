"""Inverse kinematics on Trossen's MuJoCo model of the WidowX AI (damped least squares).

The target is the fingertip site (`ee_site`, the point the driver's cartesian pose reports) with an
orientation. Joint limits are respected. The result is joint angles, so motion still goes through
SafeArm.move_joints and its path checks.
"""

import mujoco
import numpy as np

from .arm import MODEL


def gripper_rotation(pitch_down=0.0, yaw=0.0) -> np.ndarray:
    """Gripper pointing forward (+x of the base), tilted down by pitch and turned left by yaw (rad),
    with the fingers opening horizontally (gripper z axis kept as close to world up as possible)."""
    d = np.array([np.cos(pitch_down) * np.cos(yaw), np.cos(pitch_down) * np.sin(yaw), -np.sin(pitch_down)])
    z = np.array([0.0, 0.0, 1.0]) - d * d[2]
    z /= np.linalg.norm(z)
    y = np.cross(z, d)
    return np.stack([d, y, z], axis=1)


def gripper_down_rotation(yaw=0.0) -> np.ndarray:
    """Gripper pointing straight down, fingers opening along the horizontal direction `yaw` (rad)."""
    x = np.array([0.0, 0.0, -1.0])
    y = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    return np.stack([x, y, np.cross(x, y)], axis=1)


class IK:
    def __init__(self, path=MODEL):
        self.m = mujoco.MjModel.from_xml_path(str(path))
        self.d = mujoco.MjData(self.m)
        joints = [self.m.joint(f"joint_{i}") for i in range(6)]
        self.qadr = [j.qposadr[0] for j in joints]
        self.dadr = [j.dofadr[0] for j in joints]
        # keep solutions inside the same margin SafeArm checks (robot.arm.Limits.joint_margin)
        from .arm import Limits

        margin = Limits().joint_margin + 0.005
        self.lo = np.array([j.range[0] for j in joints]) + margin
        self.hi = np.array([j.range[1] for j in joints]) - margin
        self.site = self.m.site("ee_site").id

    def fk(self, q):
        self.d.qpos[:] = 0
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def solve(self, p_target, R_target, q_seed, iters=300, damping=0.02, w_rot=0.3):
        q = np.array(q_seed, dtype=float)
        jacp, jacr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for _ in range(iters):
            p, R = self.fk(q)
            e_p = p_target - p
            e_r = 0.5 * sum(np.cross(R[:, i], R_target[:, i]) for i in range(3))
            if np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(e_r) < 2e-3:
                break
            mujoco.mj_forward(self.m, self.d)
            mujoco.mj_jacSite(self.m, self.d, jacp, jacr, self.site)
            J = np.vstack([jacp[:, self.dadr], w_rot * jacr[:, self.dadr]])
            e = np.concatenate([e_p, w_rot * e_r])
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        p, R = self.fk(q)
        rot_err = np.degrees(np.arccos(np.clip((np.trace(R.T @ R_target) - 1) / 2, -1, 1)))
        return q, float(np.linalg.norm(p_target - p)), float(rot_err)
