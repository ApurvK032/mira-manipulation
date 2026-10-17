"""Camera-to-arm calibration with a cube held in the gripper.

The arm moves the red cube through poses in front of it. At each pose the arm gives the fingertip
pose (forward kinematics from measured joint angles) and the live perception gives the cube's
position in the camera frame. A least-squares fit finds the camera pose in the arm base frame, the
cube's offset in the gripper, and a depth scale correction for this camera.

Phases (run one at a time):
  python -m scripts.calibrate_camera plan       # IK + path checks + predicted image positions; no motion
  python -m scripts.calibrate_camera open       # open the gripper at the rest pose (then place the cube)
  python -m scripts.calibrate_camera run        # close on the cube, visit the poses, fit, save
  python -m scripts.calibrate_camera fit [file] # refit saved measurements
  python -m scripts.calibrate_camera release    # open the gripper to take the cube out
Needs scripts.web_view running (it supplies the cube positions). The prior is ft_server's primary
camera calibration, used only to choose poses that stay in view and to start the fit.
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "calib"
OUT = ROOT / "assets" / "calibration.yaml"
STATS = "http://127.0.0.1:8080/stats.json"
CUBE = "red"
# camera pose of ft_server's primary in its rig frame (2026-08-29), copied so the plan works without the repo
PRIOR_T_RIG_CAM = np.array([
    [-0.9978835448, 0.0014922827, 0.0650092617, 0.0227958363],
    [-0.0608336886, 0.3317486612, -0.9414043170, 0.8321111099],
    [-0.0229715769, -0.9433666301, -0.3309557490, 0.4306599156],
    [0.0, 0.0, 0.0, 1.0]])
T_RIG_BASE = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])  # yaw +pi/2 at the rig origin
K_CAM = np.array([[428.175, 0, 424.930], [0, 427.588, 243.320], [0, 0, 1]])
W, H = 848, 480


def prior_T_cam_base():
    return np.linalg.inv(PRIOR_T_RIG_CAM) @ T_RIG_BASE


def poses():
    """Fingertip targets in the base frame: a 3 x 3 x 2 grid with varied gripper tilt and yaw."""
    orient = [(0, 0), (25, 15), (15, -15)]
    out, k = [], 0
    for xi, x in enumerate((0.25, 0.30, 0.35)):
        ys = (-0.14, 0.0, 0.14) if xi % 2 == 0 else (0.14, 0.0, -0.14)  # snake order: short moves
        for y in ys:
            for z in ((0.15, 0.28) if k % 2 == 0 else (0.28, 0.15)):
                pitch, yaw = orient[len(out) % 3]
                out.append((x, y, z, pitch, yaw))
            k += 1
    return out


def plan(verbose=True):
    from robot.arm import REST, ArmModel, Limits
    from robot.ik import IK, gripper_rotation

    ik, model, lim = IK(), ArmModel(), Limits()
    T = prior_T_cam_base()
    qs, prev, rows = [], REST, []
    for x, y, z, pitch, yaw in poses():
        seed = np.array([np.arctan2(y, x), 0.6, 0.6, 0.0, 0.0, 0.0])
        for extra in (0, 20, 35):  # unreachable as asked (e.g. horizontal right over the base): tilt further down
            q, ep, er = ik.solve(np.array([x, y, z]), gripper_rotation(np.radians(pitch + extra), np.radians(yaw)), seed)
            if ep < 0.001 and er < 1.0:
                pitch += extra
                break
        p_ee, R_ee = ik.fk(q)
        cube = p_ee + R_ee @ np.array([-0.02, 0, 0])  # rough cube centre, 2 cm inside the fingertips
        pc = T[:3, :3] @ cube + T[:3, 3]
        u, v = K_CAM[0, 0] * pc[0] / pc[2] + K_CAM[0, 2], K_CAM[1, 1] * pc[1] / pc[2] + K_CAM[1, 2]
        rows.append(dict(target=[x, y, z], pitch=pitch, yaw=yaw, q=q.tolist(), ik_err_mm=ep * 1000, ik_err_deg=er,
                         range_m=float(np.linalg.norm(pc)), uv=[float(u), float(v)]))
        qs.append(q)
    ok = True
    for i, r in enumerate(rows):
        bad = []
        if r["ik_err_mm"] > 1 or r["ik_err_deg"] > 1:
            bad.append("IK")
        if not (40 < r["uv"][0] < W - 40 and 40 < r["uv"][1] < H - 40):
            bad.append("out of view")
        if r["range_m"] < 0.45:
            bad.append("too close for depth")
        r["problems"] = bad
        ok &= not bad
        if verbose:
            print(f"  pose {i:2d} target {r['target']} pitch {r['pitch']:2d} yaw {r['yaw']:3d}: IK {r['ik_err_mm']:.2f} mm, "
                  f"predicted pixel ({r['uv'][0]:4.0f},{r['uv'][1]:4.0f}) at {r['range_m']:.2f} m {'  <- ' + ', '.join(bad) if bad else ''}")
    return rows, qs, ok


def check_paths(arm, qs):
    from robot.arm import REST

    seq = [arm.q(), *qs, REST]
    for a, b in zip(seq[:-1], seq[1:]):
        arm.check_target(np.asarray(b), np.asarray(a))


def camera_sample(n=15, dt=0.033):
    pts = []
    for _ in range(n):
        s = json.load(urllib.request.urlopen(STATS, timeout=1))
        c = s["cubes"].get(CUBE, {})
        if c.get("state") == "TRACKING" and c.get("pos"):
            pts.append(c["pos"])
        time.sleep(dt)
    if len(pts) < 0.8 * n:
        return None, len(pts)
    P = np.array(pts) / 1000.0
    if np.linalg.norm(P.std(axis=0)) > 0.0015:
        return None, len(pts)
    return np.median(P, axis=0), len(pts)


def fit(data, use_scale=True):
    """Solve camera pose, cube offset in the fingertip frame, and a depth scale factor."""
    Ree = np.array([d["R_ee"] for d in data]); tee = np.array([d["p_ee"] for d in data])
    meas = np.array([d["p_cam"] for d in data])
    T0 = prior_T_cam_base()
    x0 = np.r_[Rotation.from_matrix(T0[:3, :3]).as_rotvec(), T0[:3, 3], -0.02, 0, 0, 0.0]

    def resid(x):
        R = Rotation.from_rotvec(x[:3]).as_matrix()
        k = np.exp(x[9]) if use_scale else 1.0
        cube = np.einsum("nij,j->ni", Ree, x[6:9]) + tee
        return ((cube @ R.T + x[3:6]) - k * meas).ravel()

    keep = np.ones(len(data), bool)
    for _ in range(3):
        idx = np.repeat(keep, 3)
        r = least_squares(lambda x: resid(x)[idx], x0, loss="soft_l1", f_scale=0.003)
        err = np.linalg.norm(resid(r.x).reshape(-1, 3), axis=1)
        new = err < max(3 * np.median(err[keep]), 0.006)
        if (new == keep).all():
            break
        keep = new
    R = Rotation.from_rotvec(r.x[:3]).as_matrix()
    T_cam_base = np.eye(4); T_cam_base[:3, :3] = R; T_cam_base[:3, 3] = r.x[3:6]
    return dict(T_cam_base=T_cam_base, cube_offset=r.x[6:9], k=float(np.exp(r.x[9])) if use_scale else 1.0,
                err=err, keep=keep)


def report_and_save(data, source):
    res = {s: fit(data, use_scale=s) for s in (False, True)}
    for s, f in res.items():
        e = f["err"][f["keep"]] * 1000
        print(f"  fit {'with depth scale' if s else 'rigid          '}: {f['keep'].sum()}/{len(data)} poses, "
              f"rms {np.sqrt(np.mean(e**2)):.2f} mm, max {e.max():.2f} mm, depth scale {f['k']:.4f}")
    best = res[True] if (np.sqrt(np.mean((res[True]["err"][res[True]["keep"]]) ** 2))
                         < 0.85 * np.sqrt(np.mean((res[False]["err"][res[False]["keep"]]) ** 2))) else res[False]
    T_base_cam = np.linalg.inv(best["T_cam_base"])
    T0 = np.linalg.inv(prior_T_cam_base())
    dR = Rotation.from_matrix(T0[:3, :3].T @ T_base_cam[:3, :3]).magnitude()
    print(f"  camera in the arm frame: {np.round(T_base_cam[:3, 3] * 1000, 1).tolist()} mm; "
          f"vs ft_server prior: {np.linalg.norm(T_base_cam[:3, 3] - T0[:3, 3]) * 1000:.1f} mm, {np.degrees(dR):.2f} deg")
    print(f"  cube centre in the fingertip frame: {np.round(best['cube_offset'] * 1000, 1).tolist()} mm")
    e = best["err"][best["keep"]] * 1000
    out = {
        "camera_serial": "338122302706",
        "solved_on": time.strftime("%Y-%m-%d %H:%M"),
        "method": "cube in gripper, least squares (camera pose, cube offset, depth scale)",
        "convention": "T_base_cam maps camera-frame points (OpenCV) into the arm base frame; "
                      "apply depth_scale to camera-frame positions first: p_base = T_base_cam @ (depth_scale * p_cam)",
        "T_base_cam": np.round(T_base_cam, 8).tolist(),
        "depth_scale": round(best["k"], 6),
        "cube_offset_in_fingertip_frame_m": np.round(best["cube_offset"], 5).tolist(),
        "poses_used": int(best["keep"].sum()), "poses_total": len(data),
        "rms_mm": round(float(np.sqrt(np.mean(e**2))), 3), "max_mm": round(float(e.max()), 3),
        "data": str(Path(source).resolve().relative_to(ROOT)) if Path(source).resolve().is_relative_to(ROOT) else str(source),
        "stream": "848x480 color, depth aligned to color",
    }
    OUT.write_text(yaml.safe_dump(out, sort_keys=False))
    print(f"  saved {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["plan", "open", "run", "fit", "release"])
    ap.add_argument("file", nargs="?")
    ap.add_argument("--ip", default="192.168.1.3")
    ap.add_argument("--speed", type=float, default=0.8)
    args = ap.parse_args()

    if args.phase == "fit":
        path = Path(args.file) if args.file else sorted(RUNS.glob("*.json"))[-1]
        report_and_save(json.loads(path.read_text()), path)
        return
    from robot.arm import SafeArm

    rows, qs, ok = plan(verbose=args.phase == "plan")
    arm = SafeArm(args.ip)
    try:
        if args.phase == "plan":
            if not ok:
                print("plan has problems (see above); paths not checked")
            else:
                check_paths(arm, qs)
                print(f"paths: all {len(qs) + 1} moves pass the workspace and joint checks\nplan OK")
        elif args.phase == "open":
            print(f"gripper open at {arm.open_gripper(0.035) * 1000:.1f} mm: place the {CUBE} cube between the fingers")
            arm.gripper_idle()
        elif args.phase == "release":
            print(f"gripper open at {arm.open_gripper(0.035) * 1000:.1f} mm: take the cube out")
            arm.gripper_idle()
        elif args.phase == "run":
            run(arm, qs, ok, args.speed)
    finally:
        if not arm.position_mode:
            arm.close()


def run(arm, qs, ok, speed):
    from robot.arm import REST, SafetyStop
    from robot.ik import IK

    if not ok:
        print("plan has problems; not moving")
        return
    if np.abs(arm.q() - REST).max() > 0.15:
        print("arm is not at the rest pose; not starting")
        return
    check_paths(arm, qs)
    s = json.load(urllib.request.urlopen(STATS, timeout=2))
    if not s.get("fps") or s["fps"] < 20:
        print("live perception (scripts.web_view) is not running at camera rate; not starting")
        return
    grip = arm.close_gripper(15.0, settle=1.0)
    print(f"gripper closed at {grip * 1000:.1f} mm")
    if grip < 0.010:
        print("the gripper closed on nothing: place the cube between the fingers and run again")
        arm.open_gripper(0.035)
        arm.gripper_idle()
        return
    ik = IK()
    data = []
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"calib_{time.strftime('%Y%m%d_%H%M%S')}.json"
    try:
        for i, q in enumerate(qs):
            arm.move_joints(q, speed=speed, label=f"pose {i}")
            time.sleep(0.6)
            qm = arm.q()
            p_ee, R_ee = ik.fk(qm)
            pc, n = camera_sample()
            g = arm.gripper_position()
            if g < 0.010:
                raise SafetyStop("the cube slipped out of the gripper")
            if pc is None:
                print(f"  pose {i:2d}: cube not tracked steadily ({n}/15 frames), skipped", flush=True)
                continue
            data.append(dict(i=i, q=qm.tolist(), p_ee=p_ee.tolist(), R_ee=R_ee.tolist(), p_cam=pc.tolist(),
                             samples=n, gripper=g))
            print(f"  pose {i:2d}: fingertip {np.round(p_ee * 1000, 1).tolist()} mm (arm), cube "
                  f"{np.round(pc * 1000, 1).tolist()} mm (camera)", flush=True)
        arm.rest_and_idle(speed=speed)
    except SafetyStop as e:
        print(f"SAFETY STOP: {e}; returning to rest at half speed", flush=True)
        time.sleep(1.0)
        arm.rest_and_idle(speed=speed / 2)
    finally:
        path.write_text(json.dumps(data))
        print(f"measurements: {path} ({len(data)} poses)")
    if len(data) >= 8:
        report_and_save(data, path)
    else:
        print("too few poses for a fit")


if __name__ == "__main__":
    main()
