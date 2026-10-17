"""Follow a cube with the gripper: hover straight above it and track it as it moves (no grasping).

Buttons on the live page (scripts.web_view): Start (rest -> watch pose -> over the cube, then follow),
Pause (hold where it is), Resume, Go to rest (back to rest, brakes on).

Each new perception frame: latest cube pose (TRACKING or OCCLUDED) -> hover point 14 cm up (8 cm above the
cube's top), fingers aligned with the cube's faces (the alignment nearest the current wrist angle, so the
wrist never flips 90 deg) -> IK warm-started from the last target (0.2 ms). When that target has moved, at
most every 0.15 s, the arm is re-aimed with one smooth driver-interpolated move (quintic, >= 0.3 s; path
checked like any move, fingertips >= 11 cm); between re-aims SafeArm.follow_check guards tracking and
unexpected effort at 100 Hz. (Driver 1.8.x handles fast streamed commands badly, see robot/arm.py.)
If the cube leaves the reachable ring (20-34 cm from the base), the gripper waits at the nearest point it can
reach; if the cube is out of sight, it waits where it is.

Usage:  python -m scripts.follow_cube [--cube red] [--dry]
Logs to runs/follow/run_<time>.jsonl (cube, target and fingertip positions at 20 Hz) for lag analysis.
"""

import argparse
import json
import signal
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from robot.arm import REST, WATCH, SafeArm, SafetyStop
from robot.camera_frame import CameraToBase
from robot.controller import DryArm, read_command
from robot.follow import FLOOR, RADIUS, HoverFollower
from robot.grasp import cubes_in_base, read_stats

ROOT = Path(__file__).resolve().parents[1]
DT = 0.01
DRY = False  # set in main: a dry run does not write the live page's banner


def status(msg, last=[None]):
    if msg == last[0]:
        return
    last[0] = msg
    print(time.strftime("%H:%M:%S"), msg, flush=True)
    if DRY:
        return
    try:
        urllib.request.urlopen("http://127.0.0.1:8080/status?msg=" + urllib.parse.quote(msg[:200]), timeout=0.5)
    except OSError:
        pass


class Stats(threading.Thread):
    """Polls the live viewer at about 50 Hz in the background; .latest is the newest snapshot."""

    def __init__(self):
        super().__init__(daemon=True)
        self.latest, self.t = None, 0.0

    def run(self):
        while True:
            try:
                self.latest, self.t = read_stats(timeout=0.5), time.time()
            except OSError:
                pass
            time.sleep(0.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.1.3")
    ap.add_argument("--cube", default="red")
    ap.add_argument("--speed", type=float, default=1.0, help="peak joint speed, rad/s (capped by robot.arm.Limits)")
    ap.add_argument("--dry", action="store_true", help="no arm: simulate the motion only")
    args = ap.parse_args()

    def on_term(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)
    global DRY
    DRY = args.dry

    log_dir = ROOT / "runs" / "follow"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}{'_dry' if args.dry else ''}.jsonl"
    arm = DryArm() if args.dry else SafeArm(args.ip)
    cal = CameraToBase()
    f = HoverFollower(arm, vmax=args.speed, log_path=log_path)

    def aim(snap):
        state, p, R, _ = cubes_in_base(snap, cal).get(args.cube, ("not seen", None, None, 0.0))
        if state not in ("TRACKING", "OCCLUDED") or p is None:
            return state, None, "unseen"
        return state, p, f.aim(p, R)
    stats = Stats()
    stats.start()
    t0 = time.time()
    while stats.latest is None and time.time() - t0 < 3:
        time.sleep(0.05)
    if stats.latest is None:
        raise SystemExit("live perception (scripts.web_view) is not answering on port 8080")
    if np.abs(arm.q() - REST).max() > 0.15:
        raise SystemExit("arm is not at the rest pose; not starting")
    cmd_seq = read_command(stats.latest, args.dry).get("seq", 0)
    mode, last_snap_t = "rest", 0.0
    status(f"Follow mode ready. Press Start: the gripper will hover over the {args.cube} cube and follow it.")
    print(f"log: {log_path}", flush=True)
    try:
        while True:
            now = time.time()
            snap = stats.latest
            cmd = None
            c = read_command(snap, args.dry)
            if c.get("seq", 0) != cmd_seq:
                cmd_seq, cmd = c.get("seq", 0), c.get("c")
            try:
                if cmd == "rest" and mode != "rest":
                    status("Going to rest")
                    arm.hold()
                    arm.move_joints(WATCH, speed=0.8, label="to watch pose")
                    arm.rest_and_idle(speed=0.6)
                    arm.gripper_idle()
                    mode = "rest"
                    status("At rest (brakes on). Press Start to follow again.")
                elif cmd == "start" and mode == "rest":
                    status("Waking up")
                    arm.move_joints(WATCH, speed=0.8, label="to watch pose")
                    state, p, res = aim(snap)
                    if res in ("ok", "edge"):
                        arm.move_joints(f.q_tgt, speed=0.8, label="over the cube", gripper_floor=FLOOR)
                    f.start()
                    mode = "follow"
                elif cmd == "pause" and mode == "follow":
                    arm.hold()
                    mode = "paused"
                    status("Paused (holding position). Press Resume to follow again or Go to rest.")
                elif cmd in ("resume", "start") and mode == "paused":
                    f.start()
                    mode = "follow"
                if mode == "follow":
                    if stats.t > last_snap_t:  # a new perception snapshot: new target
                        last_snap_t = stats.t
                        state, p, res = aim(snap)
                        f.log(state, p)
                        if p is None:
                            status(f"Following: {args.cube} cube {state}, waiting where I am")
                        elif res == "lifted":
                            status(f"Following: {args.cube} cube is lifted off the table, waiting where I am")
                        elif res == "edge":
                            status(f"Following: {args.cube} cube is outside the reachable ring "
                                   f"({RADIUS[0] * 100:.0f}-{RADIUS[1] * 100:.0f} cm from the base): waiting at the edge")
                        else:
                            status(f"Following the {args.cube} cube")
                    if time.time() - stats.t > 0.5:
                        raise SafetyStop("camera feed stalled")
                    f.step()
            except SafetyStop as e:
                mode = "paused"
                status(f"SAFETY STOP: {e}. Holding position. Press Resume to follow again or Go to rest.")
            except Exception as e:  # a bug must not leave the arm unattended: hold and pause
                import traceback

                traceback.print_exc()
                if arm.position_mode:
                    arm.hold()
                mode = "paused" if mode != "rest" else "rest"
                status(f"Follow error ({type(e).__name__}: {e}). Holding position; press Resume or Go to rest.")
            time.sleep(max(0.0, DT - (time.time() - now)))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if arm.position_mode:
                status("Follow mode stopping: taking the arm to rest")
                arm.hold()
                arm.move_joints(WATCH, speed=0.6, label="to watch pose")
                arm.rest_and_idle(speed=0.6)
            status("Follow mode stopped (arm at rest, brakes on)")
            arm.gripper_idle()
            arm.close()
        except Exception as e:
            arm.hold()
            status(f"Follow mode stopped while the arm is awake ({e}): it is holding position. "
                   f"Power it off with the switch once safe.")


if __name__ == "__main__":
    main()
