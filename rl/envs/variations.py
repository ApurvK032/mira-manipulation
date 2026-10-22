"""Per-episode sim-to-real variations for the environments, all off by default (no random draws when off, so older
settings reproduce exactly).

- calib_error=(pos_m, rot_deg): a camera-to-base calibration error. One rigid transform per episode (a turn about the
  vertical through the workspace centre and a shift) applied to every cube reading. The lab calibration is 2.1 mm
  RMS.
- obs_bias=m: a constant offset per cube reading for the episode (tracker bias; real placements landed about 4 mm
  off in x).
- friction_scale=(lo, hi): scales the sliding friction of the cubes and the table together (MuJoCo takes the larger
  of two surfaces' friction, so scaling only the cube would barely change cube-on-table contact).
- mass_scale=(lo, hi): scales each cube's mass and inertia.
"""

import numpy as np

CENTRE = np.array([0.27, 0.0, 0.0])  # middle of the cube workspace, the pivot of a calibration turn


class Variations:
    def __init__(self, m, table, cubes, calib_error=(0.0, 0.0), obs_bias=0.0, friction_scale=None, mass_scale=None):
        """cubes: {reading name: model body/geom name}."""
        self.m, self.calib_error, self.obs_bias = m, tuple(calib_error), obs_bias
        self.friction_scale, self.mass_scale = friction_scale, mass_scale
        self.active = bool(self.calib_error[0] or self.calib_error[1] or obs_bias or friction_scale or mass_scale)
        self.table = m.geom(table).id
        self.cubes = {n: (m.geom(g).id, m.body(g).id) for n, g in cubes.items()}
        self.base_table = m.geom_friction[self.table].copy()
        self.base = {n: (m.geom_friction[g].copy(), float(m.body_mass[b]), m.body_inertia[b].copy())
                     for n, (g, b) in self.cubes.items()}
        self.shift, self.turn, self.bias = np.zeros(3), 0.0, {n: np.zeros(3) for n in cubes}

    def sample(self, rng):
        """New variations for an episode (call at reset, before the scene settles)."""
        if not self.active:
            return
        if self.friction_scale:
            s = rng.uniform(*self.friction_scale)
            self.m.geom_friction[self.table, 0] = self.base_table[0] * s
            for n, (g, _) in self.cubes.items():
                self.m.geom_friction[g, 0] = self.base[n][0][0] * s
        if self.mass_scale:
            for n, (_, b) in self.cubes.items():
                s = rng.uniform(*self.mass_scale)
                self.m.body_mass[b], self.m.body_inertia[b] = self.base[n][1] * s, self.base[n][2] * s
        pos, rot = self.calib_error
        self.shift = rng.uniform(-pos, pos, 3) if pos else np.zeros(3)
        self.turn = float(np.radians(rng.uniform(-rot, rot))) if rot else 0.0
        self.bias = {n: (rng.uniform(-self.obs_bias, self.obs_bias, 3) if self.obs_bias else np.zeros(3))
                     for n in self.cubes}

    def reading(self, n, p, yaw=0.0):
        """What the camera would report for cube n at true position p and yaw."""
        if not self.active:
            return p, yaw
        d = p - CENTRE
        c, s = np.cos(self.turn), np.sin(self.turn)
        q = CENTRE + np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]]) + self.shift + self.bias[n]
        return q, yaw + self.turn
