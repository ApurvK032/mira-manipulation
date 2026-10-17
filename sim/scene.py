"""MuJoCo workbench seen by a fixed D455-like camera.

Renders color (with motion blur), aligned depth (with stereo-like noise, edge dropouts and holes),
and per-frame ground truth: cube poses in the camera frame and how much of each cube the hand hides.
All moving things are mocap bodies driven by a scenario, so there is no physics to tune.
"""

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("DISPLAY", ":0")

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from perception.camera import Intrinsics
from perception.geometry import HALF, inv_T, make_T

ASSETS = Path(__file__).resolve().parents[1] / "assets"

CUBE_RGB = {
    "red": (0.72, 0.05, 0.05),
    "blue": (0.06, 0.20, 0.72),
    "green": (0.08, 0.50, 0.15),
    "yellow": (0.85, 0.68, 0.04),
}
SKIN_RGB = (0.86, 0.63, 0.50)
HAND_GROUP = 2  # visible by default; hidden in the visibility pass
HIDDEN = make_T(np.eye(3), (0.0, 0.0, -2.0))

# D455 color at 848x480: about 90 x 59 degrees field of view
WIDTH, HEIGHT, FOVY = 848, 480, 59.0
DEPTH_FX, BASELINE, SUBPIXEL = 447.0, 0.095, 0.08  # stereo noise model: sigma_z = subpixel * z^2 / (f * B)


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """World pose of an OpenCV camera (x right, y down, z forward) at eye looking at target."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = target - eye
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(f, x)
    return make_T(np.stack([x, y, f], axis=1), eye)


def _fmt(v) -> str:
    return " ".join(f"{x:.6g}" for x in v)


def build_xml(T_world_cam: np.ndarray, cubes) -> str:
    x_mj, y_mj = T_world_cam[:3, 0], -T_world_cam[:3, 1]
    cube_bodies = "\n".join(
        f'<body name="cube_{n}" mocap="true" pos="0 0 -2"><geom name="cube_{n}" type="box" size="{HALF} {HALF} {HALF}" '
        f'rgba="{_fmt(CUBE_RGB[n])} 1" material="matte"/></body>'
        for n in cubes
    )
    fingers = "\n".join(
        f'<geom type="capsule" fromto="0.036 {y} 0 0.036 {y} -0.045" size="0.0085" material="skin" group="{HAND_GROUP}"/>'
        for y in (-0.027, -0.009, 0.009, 0.027)
    )
    return f"""
<mujoco model="cube_bench">
  <visual>
    <global offwidth="{WIDTH}" offheight="{HEIGHT}"/>
    <quality shadowsize="4096" offsamples="0"/>  <!-- no MSAA: segmentation IDs must stay exact -->
    <headlight ambient="0.22 0.22 0.22" diffuse="0.25 0.25 0.25" specular="0 0 0"/>
    <map znear="0.05" zfar="20"/>
  </visual>
  <asset>
    <texture name="wood" type="2d" file="{ASSETS / 'maple_tabletop.png'}"/>
    <material name="table" texture="wood" texrepeat="3 3" specular="0.08" shininess="0.15"/>
    <material name="matte" specular="0.12" shininess="0.1"/>
    <material name="skin" rgba="{_fmt(SKIN_RGB)} 1" specular="0.15" shininess="0.2"/>
    <material name="floor" rgba="0.35 0.35 0.38 1"/>
  </asset>
  <worldbody>
    <light name="key" pos="0.3 -0.3 1.3" dir="-0.25 0.2 -1" diffuse="0.65 0.65 0.62" castshadow="true"/>
    <light name="fill" pos="-0.6 0.6 1.1" dir="0.45 -0.45 -1" diffuse="0.3 0.3 0.32" castshadow="false"/>
    <geom name="floor" type="plane" size="4 4 0.1" pos="0 0 -0.75" material="floor"/>
    <geom name="table" type="box" size="0.6 0.45 0.02" pos="0 0 -0.02" material="table"/>
    <camera name="d455" pos="{_fmt(T_world_cam[:3, 3])}" xyaxes="{_fmt(x_mj)} {_fmt(y_mj)}" fovy="{FOVY}"/>
    {cube_bodies}
    <body name="distractor" mocap="true" pos="0 0 -2">
      <geom name="distractor" type="box" size="0.06 0.012 0.012" rgba="0.75 0.07 0.06 1" material="matte"/>
    </body>
    <body name="tape" mocap="true" pos="0 0 -2">
      <geom name="tape" type="box" size="0.022 0.022 0.0008" rgba="0.74 0.06 0.05 1" material="matte"/>
    </body>
    <body name="hand" mocap="true" pos="0 0 -2">
      <geom type="box" size="0.045 0.042 0.012" material="skin" group="{HAND_GROUP}"/>
      {fingers}
      <geom type="capsule" fromto="-0.036 0 0 -0.036 0 -0.04" size="0.010" material="skin" group="{HAND_GROUP}"/>
      <geom type="capsule" fromto="-0.04 0 0.02 -0.30 0 0.18" size="0.034" material="skin" group="{HAND_GROUP}"/>
    </body>
  </worldbody>
</mujoco>"""


@dataclass
class Frame:
    t: float
    color: np.ndarray  # HxWx3 uint8 BGR
    depth: np.ndarray  # HxW uint16, millimeters, 0 = invalid
    gt: dict  # cube name -> 4x4 pose in camera frame (None if hidden)
    visibility: dict  # cube name -> fraction of the cube's pixels not hidden by the hand
    pixels: dict  # cube name -> visible pixel count


class BenchScene:
    def __init__(self, cubes=("red", "blue", "green", "yellow"), cam_eye=(0.0, 0.52, 0.50),
                 cam_target=(0.0, 0.0, 0.02), exposure_s=0.008, seed=0):
        self.cubes = list(cubes)
        self.T_world_cam = look_at(cam_eye, cam_target)
        self.T_cam_world = inv_T(self.T_world_cam)
        self.model = mujoco.MjModel.from_xml_string(build_xml(self.T_world_cam, self.cubes))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, HEIGHT, WIDTH)
        self.intrinsics = Intrinsics.from_fov(WIDTH, HEIGHT, FOVY)
        self.exposure = exposure_s
        self.rng = np.random.default_rng(seed)
        m = self.model
        self.mocap = {n: m.body_mocapid[m.body(f"cube_{n}").id] for n in self.cubes}
        self.mocap["hand"] = m.body_mocapid[m.body("hand").id]
        self.mocap["distractor"] = m.body_mocapid[m.body("distractor").id]
        self.mocap["tape"] = m.body_mocapid[m.body("tape").id]
        self.cube_geom = {n: m.geom(f"cube_{n}").id for n in self.cubes}
        self.base_light = m.light_diffuse.copy()
        self.base_head = (m.vis.headlight.diffuse.copy(), m.vis.headlight.ambient.copy())
        self.opt_nohand = mujoco.MjvOption()
        self.opt_nohand.geomgroup[HAND_GROUP] = 0

    def _set(self, name, T):
        T = HIDDEN if T is None else T
        i = self.mocap[name]
        self.data.mocap_pos[i] = T[:3, 3]
        q = Rotation.from_matrix(T[:3, :3]).as_quat()  # xyzw
        self.data.mocap_quat[i] = (q[3], q[0], q[1], q[2])

    def apply(self, state):
        for n in self.cubes:
            self._set(n, state.cubes.get(n))
        self._set("hand", state.hand)
        self._set("distractor", state.distractor)
        self._set("tape", state.tape)
        s = state.light
        self.model.light_diffuse[:] = self.base_light * s
        self.model.vis.headlight.diffuse[:] = self.base_head[0] * s
        self.model.vis.headlight.ambient[:] = self.base_head[1] * (0.4 + 0.6 * s)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def _render_rgb(self):
        self.renderer.update_scene(self.data, "d455")
        return self.renderer.render().astype(np.float32)

    def render(self, scenario, t: float) -> Frame:
        # motion blur: average sub-frames over the exposure when something moves fast
        speed = scenario.max_speed(t)
        k = int(np.clip(np.ceil(speed * self.exposure / 0.002), 1, 6))
        offsets = np.linspace(-self.exposure / 2, self.exposure / 2, k) if k > 1 else [0.0]
        acc = np.zeros((HEIGHT, WIDTH, 3), np.float32)
        for dt in offsets:
            self.apply(scenario.state(t + dt))
            acc += self._render_rgb()
        rgb = acc / len(offsets)
        self.apply(scenario.state(t))

        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, "d455")
        z = self.renderer.render().astype(np.float64)
        self.renderer.disable_depth_rendering()

        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, "d455")
        seg = self.renderer.render()[..., 0]
        self.renderer.update_scene(self.data, "d455", scene_option=self.opt_nohand)
        seg_nohand = self.renderer.render()[..., 0]
        self.renderer.disable_segmentation_rendering()

        state = scenario.state(t)
        gt, vis, px = {}, {}, {}
        for n in self.cubes:
            Tw = state.cubes.get(n)
            gid = self.cube_geom[n]
            seen, full = int((seg == gid).sum()), int((seg_nohand == gid).sum())
            px[n] = seen
            vis[n] = seen / full if full > 0 else 0.0
            gt[n] = None if Tw is None or full == 0 else self.T_cam_world @ Tw
        return Frame(t, self._color_noise(rgb), self._depth_noise(z), gt, vis, px)

    def _color_noise(self, rgb):
        img = cv2.GaussianBlur(rgb, (0, 0), 0.7)
        img += self.rng.normal(0, 2.0, img.shape).astype(np.float32)
        return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    def _depth_noise(self, z):
        h, w = z.shape
        sigma = SUBPIXEL * z**2 / (DEPTH_FX * BASELINE)
        corr = cv2.GaussianBlur(self.rng.standard_normal((h, w)).astype(np.float32), (0, 0), 1.5)
        corr /= corr.std() + 1e-9
        zn = z + sigma * (0.85 * corr + 0.5 * self.rng.standard_normal((h, w)))
        # stereo drops and smears depth at depth discontinuities
        zf = z.astype(np.float32)
        grad = cv2.dilate(zf, np.ones((3, 3))) - cv2.erode(zf, np.ones((3, 3)))
        edge = cv2.dilate((grad > 0.01).astype(np.uint8), np.ones((3, 3))).astype(bool)
        drop = edge & (self.rng.random((h, w)) < 0.6)
        drop |= self.rng.random((h, w)) < 0.005
        zn[drop] = 0.0
        zn[(z > 6.0) | (z < 0.1)] = 0.0
        return np.clip(np.round(zn * 1000.0), 0, 65535).astype(np.uint16)
