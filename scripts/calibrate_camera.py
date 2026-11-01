"""Camera-to-arm calibration with a cube held in the gripper.

The arm moves the red cube through poses in front of it. At each pose the arm gives the fingertip
pose (forward kinematics from measured joint angles) and the live perception gives the cube's
position in the camera frame. A least-squares fit finds the camera pose in the arm base frame, the
cube's offset in the gripper, and a depth scale correction for this camera.

Phases (run one at a time):
  python -m scripts.calibrate_camera plan       # IK + path checks + predicted image positions; no motion
  python -m scripts.calibrate_camera open       # open the gripper at the rest pose (then place the cube)
  python -m scripts.calibrate_camera run        # close on the cube, visit the poses, fit, save
  python -m scripts.calibrate_camera pick       # or: wake, hover over the cube on the table, wait for Resume while the
                                                # operator centres it, pick it, visit the poses, put it back, fit, save
  python -m scripts.calibrate_camera fit [file] # refit saved measurements
  python -m scripts.calibrate_camera release    # open the gripper to take the cube out
Needs scripts.web_view running (it supplies the cube positions). The prior is ft_server's primary
camera calibration, used only to choose poses that stay in view and to start the fit.
"""

import argparse
import json
import sys
import time
import urllib.parse
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


PRIOR_CURRENT = False  # --prior-current: start from this camera's assets/calibration.yaml instead of the old rig's pose


def prior_T_cam_base():
    if PRIOR_CURRENT:
        return np.linalg.inv(np.array(yaml.safe_load(OUT.read_text())["T_base_cam"], dtype=float))
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
        fresh = time.time() - s.get("t", 0) < 0.2  # a frozen snapshot would look like a perfectly steady cube
        if fresh and c.get("state") == "TRACKING" and c.get("pos"):
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
          f"vs {'the previous calibration' if PRIOR_CURRENT else 'ft_server prior'}: "
          f"{np.linalg.norm(T_base_cam[:3, 3] - T0[:3, 3]) * 1000:.1f} mm, {np.degrees(dR):.2f} deg")
    print(f"  cube centre in the fingertip frame: {np.round(best['cube_offset'] * 1000, 1).tolist()} mm")
    e = best["err"][best["keep"]] * 1000
    serial = data[0].get("camera_serial")
    if not serial:
        print("  WARNING: these measurements do not record the camera serial; the saved calibration will be refused "
              "by every live script until a camera with serial 'unknown' is used (recalibrate)")
    out = {
        "camera_serial": str(serial or "unknown"),
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
    ap.add_argument("phase", choices=["plan", "open", "run", "pick", "fit", "release"])
    ap.add_argument("file", nargs="?")
    ap.add_argument("--ip", default="192.168.1.3")
    ap.add_argument("--speed", type=float, default=0.8)
    ap.add_argument("--close-delay", type=float, default=0.0,
                    help="run: seconds of countdown before the gripper closes on the cube (time to hold it in place)")
    ap.add_argument("--grip-mm", type=float, nargs=2, metavar=("LO", "HI"),
                    help="run: accepted gripper reading (mm per finger) after the close; outside it the gripper opens "
                         "and the run stops at rest before any pose (a cube held across an edge or tilted reads wide)")
    ap.add_argument("--grip-z", type=float, default=GRIP_Z * 1000,
                    help="pick: fingertip height (mm above the table) to grasp the cube at; 50 is its top")
    ap.add_argument("--dry", action="store_true", help="pick: DryArm instead of the arm (commands from "
                    "runs/dry_command.json); nothing is saved")
    ap.add_argument("--prior-current", action="store_true",
                    help="plan and fit start from assets/calibration.yaml (this camera) instead of the old rig's prior")
    args = ap.parse_args()
    global PRIOR_CURRENT
    PRIOR_CURRENT = args.prior_current and OUT.exists()

    if args.phase == "fit":
        path = Path(args.file) if args.file else sorted(RUNS.glob("*.json"))[-1]
        report_and_save(json.loads(path.read_text()), path)
        return
    from robot.arm import SafeArm

    rows, qs, ok = plan(verbose=args.phase == "plan")
    if args.dry and args.phase != "pick":
        raise SystemExit("--dry is for the pick phase only")
    if args.dry:
        from robot.controller import DryArm

        arm = DryArm()
    else:
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
            run(arm, qs, ok, args.speed, close_delay=args.close_delay, grip_mm=args.grip_mm)
        elif args.phase == "pick":
            pick_and_run(arm, qs, ok, args.speed, dry=args.dry, grip_z=args.grip_z / 1000.0)
    finally:
        if not arm.position_mode:
            arm.close()


def run(arm, qs, ok, speed, close_delay=0.0, grip_mm=None):
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
    if not s.get("fps") or s["fps"] < 20 or time.time() - s.get("t", 0) > 0.5:
        print("live perception (scripts.web_view) is not running at camera rate (or its stats are stale); not starting")
        return
    serial = s.get("camera_serial")  # recorded in the new calibration; a calibration run never checks it
    if not serial:
        print("live perception does not report the camera serial (update scripts.web_view); not starting")
        return
    print(f"calibrating camera {serial}")
    for k in range(int(round(close_delay)), 0, -1):
        print(f"closing the gripper in {k} s: hold the {CUBE} cube between the fingers", flush=True)
        time.sleep(1.0)
    grip = arm.close_gripper(15.0, settle=1.0)
    print(f"gripper closed at {grip * 1000:.1f} mm")
    if grip < 0.010:
        print("the gripper closed on nothing: place the cube between the fingers and run again")
        arm.open_gripper(0.035)
        arm.gripper_idle()
        return
    if grip_mm is not None and not grip_mm[0] <= grip * 1000 <= grip_mm[1]:
        print(f"grip {grip * 1000:.1f} mm is outside {grip_mm[0]:.1f}-{grip_mm[1]:.1f} mm (cube not held flat?): "
              f"opening the gripper; the arm stays at rest, no poses visited")
        arm.open_gripper(0.035)
        arm.gripper_idle()
        return
    print(f"grip accepted; visiting {len(qs)} poses", flush=True)
    ik = IK()
    data, stopped = [], False
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
                             samples=n, gripper=g, camera_serial=serial))
            print(f"  pose {i:2d}: fingertip {np.round(p_ee * 1000, 1).tolist()} mm (arm), cube "
                  f"{np.round(pc * 1000, 1).tolist()} mm (camera)", flush=True)
        arm.rest_and_idle(speed=speed)
    except SafetyStop as e:
        print(f"SAFETY STOP: {e}; returning to rest at half speed", flush=True)
        time.sleep(1.0)
        arm.rest_and_idle(speed=speed / 2)
    except KeyboardInterrupt:  # operator stop (Ctrl-C / SIGINT): move_joints has already held the arm where it was
        stopped = True
        print("stopped by the operator: holding, then returning to rest at half speed; no fit", flush=True)
        time.sleep(1.0)
        arm.rest_and_idle(speed=speed / 2)
    except Exception as e:  # e.g. perception gone mid-run: never leave the arm out at a pose
        stopped = True
        print(f"error mid-run ({type(e).__name__}: {e}): holding, then returning to rest at half speed; no fit",
              flush=True)
        arm.hold()
        time.sleep(1.0)
        arm.rest_and_idle(speed=speed / 2)
    finally:
        path.write_text(json.dumps(data))
        print(f"measurements: {path} ({len(data)} poses)")
        if stopped or not data or arm.gripper_position() < 0.010:
            arm.gripper_idle()  # do not leave the gripper squeezing (on nothing) after a stop
    if stopped:
        print("operator stop: measurements kept; refit later with `fit <file>` if they are good")
    elif len(data) >= 8:
        report_and_save(data, path)
    else:
        print("too few poses for a fit")


# ----- pick phase: the arm picks the cube off the table itself (square, repeatable grip), then visits the poses -----
# Heights of the fingertip site above the table, which is the arm's base plane (z = 0: the arm stands on it). They do
# not come from the camera: before calibrating, its heights can be off by a centimetre (2026-10-20: cubes read 17 mm
# instead of 25), and grasping at a seen height took the fingers down to the cube's bottom edges.
HOVER_Z, LIFT_Z = 0.12, 0.13  # straight-down reachable over the whole 20-36 cm ring (0.17 is not, at 32 cm)
GRIP_Z = 0.036  # default fingertip height for the pick: 14 mm below the 50 mm top; the finger ends (4.5 mm lower) stay
#                 above the cube's centre, so the pads hold its upper part
SQUEEZE_N = 20.0  # N: grip force, position mode (SafeArm.squeeze_gripper)
SQUEEZE_START = 0.0255  # m per finger: first close target, just inside a 5 cm cube (held flat it reads about 27.5 mm)
HELD_MM = (22.0, 28.5)  # finger reading (mm) that means squeezing the cube; wider is the cube's edge or nothing


class Stop(Exception):
    """The operator aborted from the live page."""


class Operator:
    """Commands from the live page (scripts.web_view /command); a dry run reads only runs/dry_command.json."""

    def __init__(self, dry):
        from robot.controller import read_command

        self.read, self.dry = read_command, dry
        self.seq = self._cmd().get("seq", 0)  # ignore anything pressed before this run

    def _cmd(self):
        try:
            s = json.load(urllib.request.urlopen(STATS, timeout=1))
        except OSError:
            s = {}
        return self.read(s, self.dry)

    def poll(self):
        c = self._cmd()
        if c.get("seq", 0) != self.seq:
            self.seq = c.get("seq", 0)
            return c.get("c")
        return None

    def watch(self, t=None):
        """move_joints watch callback: Pause or Go to rest stops the move."""
        c = self.poll()
        return f"operator: {c}" if c in ("pause", "rest") else None

    def status(self, msg):
        print(msg, flush=True)
        if not self.dry:
            try:
                urllib.request.urlopen(f"{STATS.rsplit('/', 1)[0]}/status?msg=" + urllib.parse.quote(msg[:200]),
                                       timeout=0.5)
            except OSError:
                pass


def read_cube_now(cal, window=0.4):
    """Median of fresh TRACKING readings of the cube over `window` s, in the arm frame: (centre, R) or (None, None)."""
    from robot.grasp import cubes_in_base

    pts, R, t0 = [], None, time.time()
    while time.time() - t0 < window:
        s = json.load(urllib.request.urlopen(STATS, timeout=1))
        if time.time() - s.get("t", 0) < 0.2:
            state, p, Rm, _ = cubes_in_base(s, cal).get(CUBE, ("", None, None, 0.0))
            if state == "TRACKING" and p is not None:
                pts.append(p)
                R = Rm
        time.sleep(0.02)
    return (np.median(pts, axis=0), R) if len(pts) >= 5 else (None, None)


def pick_and_run(arm, qs, ok, speed, dry=False, grip_z=GRIP_Z):
    """Wake, hover over the cube, wait while the operator centres it under the fingers (Resume on the live page),
    pick it straight down with a position-mode squeeze, visit the poses, put it back where it was, rest; then fit.
    Go to rest / Pause on the page (or Ctrl-C) stops at any time: hold, then back to rest at half speed, no fit."""
    from robot.arm import REST, WATCH, SafetyStop
    from robot.camera_frame import CameraToBase
    from robot.grasp import REACH, plan_top_down
    from robot.ik import IK, gripper_down_rotation

    if not ok:
        print("plan has problems; not moving")
        return
    if np.abs(arm.q() - REST).max() > 0.15:
        print("arm is not at the rest pose; not starting")
        return
    s = json.load(urllib.request.urlopen(STATS, timeout=2))
    if not s.get("fps") or s["fps"] < 20 or time.time() - s.get("t", 0) > 0.5:
        print("live perception (scripts.web_view) is not running at camera rate (or its stats are stale); not starting")
        return
    serial = s.get("camera_serial")
    if not serial:
        print("live perception does not report the camera serial (update scripts.web_view); not starting")
        return
    cal, ik = CameraToBase(ignore_serial=True), IK()  # only the starting guess; a calibration run is for a new or moved camera
    c, R = read_cube_now(cal)
    if c is None:
        print(f"the {CUBE} cube is not seen steadily: put it flat on the table in front of the arm")
        return
    r = float(np.hypot(c[0], c[1]))
    if not (REACH[0] < r < REACH[1] and c[0] > 0.1):
        print(f"the {CUBE} cube is {100 * r:.0f} cm from the base: put it {100 * REACH[0]:.0f}-{100 * REACH[1]:.0f} cm "
              f"in front of the arm")
        return
    floor = 0.012  # fingertips never below 12 mm over the table
    on_table = np.array([c[0], c[1], 0.0])  # heights below are from the table, not the camera's (uncalibrated) height
    qp, yaw = plan_top_down(on_table, R, {"hover": HOVER_Z, "grip": grip_z, "lift": LIFT_Z}, ik=ik)
    op = Operator(dry)
    print(f"calibrating camera {serial}{' (DRY: no arm, nothing saved)' if dry else ''}; {CUBE} cube seen at "
          f"{np.round(c * 1000).tolist()} mm, {100 * r:.0f} cm from the base", flush=True)

    def straight(z, label, spd=0.3):
        q = arm.q()
        tip, _ = ik.fk(q)
        q_z, err, _ = ik.solve(np.array([tip[0], tip[1], z]), gripper_down_rotation(yaw), q)
        if err > 0.002:
            raise SafetyStop(f"{label}: no IK solution at z {z * 1000:.0f} mm")
        if arm.move_joints(q_z, speed=spd, label=label, gripper_floor=floor, watch=op.watch).stopped:
            raise Stop()

    def move(q, label, spd=speed, fl=floor):
        if arm.move_joints(q, speed=spd, label=label, gripper_floor=fl, watch=op.watch).stopped:
            raise Stop()

    data, stopped, holding, spot = [], False, False, None
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"calib_{time.strftime('%Y%m%d_%H%M%S')}{'_dry' if dry else ''}.json"
    try:
        op.status(f"Calibration: waking, then hovering over the {CUBE} cube")
        arm.open_gripper(0.035)
        move(WATCH, "to watch pose", fl=None)  # from rest: the workspace box's own floor rules
        move(qp["hover"], f"over the {CUBE} cube")
        op.status(f"Slide the {CUBE} cube under the fingers, square to them, then press Resume (Go to rest aborts)")
        t0, last = time.time(), 0.0
        while True:
            c_ = op.poll()
            if c_ in ("resume", "start"):
                break
            if c_ in ("rest", "pause"):
                raise Stop()
            if time.time() - t0 > 300:
                raise Stop()
            if time.time() - last > 1.0:
                last = time.time()
                cn, _ = read_cube_now(cal, window=0.1)
                tip, _ = ik.fk(arm.q())
                if cn is not None:
                    d = (cn[:2] - tip[:2]) * 1000
                    print(f"  camera sees the cube {d[0]:+.0f} mm in x, {d[1]:+.0f} mm in y from the fingertips", flush=True)
            time.sleep(0.1)
        tip, _ = ik.fk(arm.q())
        cn, _ = read_cube_now(cal)
        if cn is not None:
            d = (cn[:2] - tip[:2]) * 1000
            print(f"centred by eye: the current calibration puts the cube {np.hypot(*d):.1f} mm from the fingertips "
                  f"({d[0]:+.1f} x, {d[1]:+.1f} y)", flush=True)
        spot = tip[:2].copy()
        op.status("Picking the cube")
        straight(grip_z, f"down to {grip_z * 1000:.0f} mm above the table")
        grip = arm.squeeze_gripper(SQUEEZE_N, start=SQUEEZE_START)
        eff = abs(arm.drv.get_gripper_effort()) if hasattr(arm, "drv") else float("nan")
        print(f"gripper closed at {grip * 1000:.1f} mm, effort {eff:.1f} N (position mode)", flush=True)
        if not HELD_MM[0] <= grip * 1000 <= HELD_MM[1]:
            arm.open_gripper(0.035)
            straight(HOVER_Z, "rise")
            raise SafetyStop(f"the gripper stopped at {grip * 1000:.1f} mm: not squeezing the cube "
                             f"({HELD_MM[0]:.0f}-{HELD_MM[1]:.1f} mm expected)")
        holding = True
        straight(grip_z + 0.010, "lift 1 cm (slip check)", spd=0.15)
        time.sleep(0.5)
        if abs(arm.gripper_position() - grip) > 0.0008:
            arm.open_gripper(0.035)
            holding = False
            straight(HOVER_Z, "rise")
            raise SafetyStop(f"the cube slipped on the first centimetre (fingers {grip * 1000:.1f} -> "
                             f"{arm.gripper_position() * 1000:.1f} mm)")
        straight(LIFT_Z, "lift")
        time.sleep(0.5)
        if arm.gripper_position() < 0.010:
            raise SafetyStop("the cube dropped while lifting")
        check_paths(arm, qs)
        op.status(f"Calibration: visiting {len(qs)} poses (Go to rest stops)")
        for i, q in enumerate(qs):
            move(q, f"pose {i}")
            time.sleep(0.6)
            qm = arm.q()
            p_ee, R_ee = ik.fk(qm)
            pc, n = camera_sample()
            g = arm.gripper_position()
            if g < 0.010:
                holding = False
                raise SafetyStop("the cube slipped out of the gripper")
            if pc is None:
                print(f"  pose {i:2d}: cube not tracked steadily ({n}/15 frames), skipped", flush=True)
                continue
            data.append(dict(i=i, q=qm.tolist(), p_ee=p_ee.tolist(), R_ee=R_ee.tolist(), p_cam=pc.tolist(),
                             samples=n, gripper=g, camera_serial=serial))
            print(f"  pose {i:2d}: fingertip {np.round(p_ee * 1000, 1).tolist()} mm (arm), cube "
                  f"{np.round(pc * 1000, 1).tolist()} mm (camera)", flush=True)
        op.status(f"Calibration: putting the {CUBE} cube back")
        q_over, err, _ = ik.solve(np.array([spot[0], spot[1], LIFT_Z]), gripper_down_rotation(yaw), arm.q())
        if err > 0.002:
            raise SafetyStop("no IK solution back over the pick spot")
        move(q_over, "back over the pick spot")
        straight(grip_z + 0.003, "put down")
        arm.open_gripper(0.035)
        holding = False
        straight(HOVER_Z, "rise")
        arm.rest_and_idle(speed=speed)
        arm.gripper_idle()
    except (Stop, SafetyStop, KeyboardInterrupt, Exception) as e:
        stopped = True
        why = "operator stop" if isinstance(e, (Stop, KeyboardInterrupt)) else f"{type(e).__name__}: {e}"
        op.status(f"Calibration stopped ({why}): holding, then back to rest at half speed; no fit")
        try:
            arm.hold()
        except Exception:
            pass
        time.sleep(1.0)
        arm.rest_and_idle(speed=speed / 2)
        arm.gripper_idle()  # never leave the gripper squeezing; a held cube drops near the rest pose
        if holding:
            print(f"(the {CUBE} cube was released at the rest pose)", flush=True)
    finally:
        path.write_text(json.dumps(data))
        print(f"measurements: {path} ({len(data)} poses)")
    if stopped:
        print("stopped: measurements kept; refit later with `fit <file>` if they are good")
    elif dry:
        print(f"dry run: {len(data)} poses sampled; not fitted, assets/calibration.yaml untouched")
    elif len(data) >= 8:
        report_and_save(data, path)
        op.status("Calibration saved; restart scripts.web_view and do the hover check")
    else:
        print("too few poses for a fit")


if __name__ == "__main__":
    main()
