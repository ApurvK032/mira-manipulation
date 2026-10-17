"""M3T (DLR-RM 3DObjectTracking, via pym3t) cube trackers.

One M3T Tracker per cube, all sharing two dummy cameras that we fill with numpy frames. Each cube can
then be started, reset and stepped independently. The world frame is the color camera frame.
"""

from pathlib import Path

import numpy as np
import pym3t

from .camera import Intrinsics
from .geometry import write_cube_obj

MODEL_DIR = Path(__file__).resolve().parents[1] / "assets" / "m3t_models"


class M3TCubeTrackers:
    def __init__(self, intrinsics: Intrinsics, names, model_dir: Path = MODEL_DIR, depth_scale: float = 0.001,
                 n_corr_iterations: int = 5, n_update_iterations: int = 2):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        obj = write_cube_obj(model_dir / "cube_5cm.obj")
        intr = pym3t.Intrinsics(fu=intrinsics.fx, fv=intrinsics.fy, ppu=intrinsics.cx, ppv=intrinsics.cy,
                                width=intrinsics.width, height=intrinsics.height)
        self.color_camera = pym3t.DummyColorCamera("color")
        self.color_camera.camera2world_pose = np.eye(4)
        self.color_camera.intrinsics = intr
        self.depth_camera = pym3t.DummyDepthCamera("depth", depth_scale=depth_scale)
        self.depth_camera.camera2world_pose = np.eye(4)  # depth is aligned to color
        self.depth_camera.intrinsics = intr

        self.cubes = {}
        for n in names:
            body = pym3t.Body(name=n, geometry_path=obj.as_posix(), geometry_unit_in_meter=1.0,
                              geometry_counterclockwise=True, geometry_enable_culling=True,
                              geometry2body_pose=np.eye(4))
            link = pym3t.Link(f"{n}_link", body)
            region_model = pym3t.RegionModel(f"{n}_region_model", body, (model_dir / f"{n}_region_model.bin").as_posix())
            depth_model = pym3t.DepthModel(f"{n}_depth_model", body, (model_dir / f"{n}_depth_model.bin").as_posix())
            region = pym3t.RegionModality(f"{n}_region", body, self.color_camera, region_model)
            region.MeasureOcclusions(self.depth_camera)
            depth = pym3t.DepthModality(f"{n}_depth", body, self.depth_camera, depth_model)
            depth.MeasureOcclusions()
            # M3T's defaults (2-5 cm) suit noisy sensors and large objects. For a 5 cm cube and a D455's
            # ~1 mm noise at 0.7 m they leave depth too weak against the contour term, and the tracker
            # can settle on a rotated cube with the same outline. Weight depth to the cube's scale.
            depth.considered_distances = [0.04, 0.015, 0.008]
            depth.standard_deviations = [0.02, 0.006, 0.003]
            link.AddModality(region)
            link.AddModality(depth)
            optimizer = pym3t.Optimizer(f"{n}_optimizer", link)
            tracker = pym3t.Tracker(f"{n}_tracker", n_corr_iterations=n_corr_iterations,
                                    n_update_iterations=n_update_iterations, synchronize_cameras=False)
            tracker.AddOptimizer(optimizer)
            if not tracker.SetUp():
                raise RuntimeError(f"M3T set up failed for {n}")
            self.cubes[n] = (tracker, body, link)
        self.started = {n: False for n in names}

    def set_images(self, color_bgr: np.ndarray, depth_u16: np.ndarray):
        self.color_camera.image = color_bgr
        self.depth_camera.image = depth_u16

    def pose(self, name) -> np.ndarray:
        return np.array(self.cubes[name][1].body2world_pose, dtype=float)

    def set_pose(self, name, T: np.ndarray):
        """Move the cube without re-learning its color histograms."""
        _, body, link = self.cubes[name]
        body.body2world_pose = T
        link.link2world_pose = T

    def reset(self, name, T: np.ndarray, iteration: int, refine_steps: int = 2) -> np.ndarray:
        """Put the cube at T, re-learn its color histograms there, and refine."""
        tracker, body, link = self.cubes[name]
        body.body2world_pose = T
        link.link2world_pose = T
        tracker.StartModalities(iteration)
        for _ in range(refine_steps):
            tracker.ExecuteTrackingStep(iteration)
        self.started[name] = True
        return self.pose(name)

    def track(self, name, iteration: int) -> np.ndarray:
        self.cubes[name][0].ExecuteTrackingStep(iteration)
        return self.pose(name)
