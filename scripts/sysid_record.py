"""System identification, part 1: record how the real arm follows targets sent the way an RL policy sends them.

A policy outputs a fingertip target 20 times a second; on the real arm those targets go through SafeArm's
smooth re-aims (follow_goal). This script does exactly that with known target patterns and logs commanded vs
measured joints, so rl/sysid_fit.py can fit the simulator's arm lag (the lag_tau of rl.envs with
lag_mode="target") to the real WidowX AI.

Motions (small, straight down over the table, fingertips never below 11.5 cm, every re-aim path-checked):
  rest -> watch pose -> hover at (27, 0, 13) cm
  1. steps: fingertip target jumps +-3 cm in x, +-3 cm in y, +2/-1.5 cm in z, holding 1.5 s each
     (with --ff: 0.3 s ramps instead of jumps)
  2. sweep: target swings +-3 cm in y with the frequency rising from 0.2 to 1.2 Hz over 15 s
  3. gripper: close and open twice (empty), timing the fingers
  -> watch pose -> rest (brakes on)

Usage:  python -m scripts.sysid_record --dry     # simulated arm, no motion (checks the script)
        python -m scripts.sysid_record           # real arm, someone at the power switch
        add --ff for feedforward re-aims (robot.reaim: each re-aim ends moving at the target's velocity)
        add --stream HZ (e.g. 20, 50, 100) to stream targets instead (SafeArm.stream_tick; driver/firmware 1.9.x)
Writes runs/sysid/sysid_<time>[_ff|_stream<HZ>][_dry].npz
"""

import argparse
import time
from pathlib import Path

import numpy as np

from robot.arm import REST, WATCH, SafeArm, SafetyStop
from robot.controller import DryArm
from robot.ik import IK, gripper_down_rotation
from robot.reaim import FF_GAIN, GoalVelocity

ROOT = Path(__file__).resolve().parents[1]
HOVER = np.array([0.27, 0.0, 0.13])
FLOOR = 0.11
POLICY_DT, TICK, REAIM = 0.05, 0.01, 0.15


def step_targets():
    """(duration, fingertip offset) segments for the step test."""
    segs = []
    for axis, (up, down) in enumerate(((0.03, -0.03), (0.03, -0.03), (0.02, -0.015))):
        for off in (up, 0.0, down, 0.0):
            d = np.zeros(3)
            d[axis] = off
            segs.append((1.5, d))
    return segs


def sweep_offset(t, T=15.0, f0=0.2, f1=1.2, amp=0.03):
    """Linear chirp in y: phase = 2 pi (f0 t + (f1 - f0) t^2 / (2 T))."""
    return np.array([0.0, amp * np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * T))), 0.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.1.3")
    ap.add_argument("--dry", action="store_true", help="simulated arm, no motion")
    ap.add_argument("--ff", action="store_true", help="feedforward re-aims (robot.reaim)")
    ap.add_argument("--ff-gain", type=float, default=FF_GAIN, help="share of the goal's velocity fed forward")
    ap.add_argument("--stream", type=float, default=0.0, help="Hz: stream targets (driver 1.9.x) instead of re-aiming")
    args = ap.parse_args()

    arm = DryArm() if args.dry else SafeArm(args.ip)
    ik = IK()
    log = {k: [] for k in ("t", "q_meas", "q_target", "tip_target", "phase", "effort", "gripper", "v_end")}
    gv, reaims = GoalVelocity(gain=args.ff_gain), []  # reaims: (time, T) of every re-aim
    events = []
    try:
        if np.abs(arm.q() - REST).max() > 0.15:
            raise SystemExit("arm is not at the rest pose; not starting")
        print("to watch pose", flush=True)
        arm.move_joints(WATCH, speed=0.8, label="to watch pose")
        seeds = [np.array([0.0, a, b, w, 0, 0]) for a in (0.4, 0.9, 1.4) for b in (0.4, 1.0, 1.6) for w in (0.5, 1.2)]
        q_hover = min((ik.solve(HOVER, gripper_down_rotation(np.pi / 2), s) for s in seeds), key=lambda r: r[1] + r[2] / 100)[0]
        print("to hover pose", flush=True)
        arm.move_joints(q_hover, speed=0.6, label="to hover", gripper_floor=FLOOR)
        arm.open_gripper(0.035)
        if args.stream:
            if args.ff:
                raise SystemExit("--stream and --ff are different modes; pick one")
            arm.stream_begin(rate=args.stream)
        else:
            arm.follow_begin()
        q_tgt, q_goal, t_aim, v_last = q_hover.copy(), q_hover.copy(), 0.0, np.zeros(6)
        t0 = time.time()

        def run(phase, duration, offset_fn):
            nonlocal q_tgt, q_goal, t_aim, v_last
            start, next_policy = time.time(), 0.0
            tick = 1.0 / args.stream if args.stream else TICK
            next_tick = start
            while (el := time.time() - start) < duration:
                if el >= next_policy:  # a policy step: new fingertip target
                    next_policy += POLICY_DT
                    tip = HOVER + offset_fn(el)
                    q_tgt = ik.solve(tip, gripper_down_rotation(np.pi / 2), q_tgt)[0]
                    log_tip = tip
                    gv.update(time.time(), q_tgt)
                arm.follow_check()
                moved = np.abs(q_tgt - q_goal).max() > 0.008
                if args.stream:
                    arm.stream_tick(q_tgt, FLOOR)
                elif time.time() - t_aim >= REAIM and (moved or (args.ff and np.any(v_last))):
                    v_last = gv.get() if args.ff else np.zeros(6)  # feedforward: one more re-aim once it stops
                    T = arm.follow_goal(q_tgt, 1.0, FLOOR, goal_velocity=v_last if args.ff else None)
                    q_goal, t_aim = q_tgt.copy(), time.time()
                    reaims.append((t_aim - t0, T))
                log["t"].append(time.time() - t0)
                log["q_meas"].append(arm.q())
                log["q_target"].append(q_tgt.copy())
                log["tip_target"].append(log_tip)
                log["phase"].append(phase)
                log["effort"].append(arm.efforts() if hasattr(arm, "efforts") else np.zeros(6))
                log["gripper"].append(arm.gripper_position())
                log["v_end"].append(v_last.copy())
                next_tick = next_tick + tick if time.time() - next_tick < tick else time.time() + tick
                time.sleep(max(0.0, next_tick - time.time()))

        print("test 1: steps", flush=True)
        prev = np.zeros(3)
        for dur, off in step_targets():
            # feedforward: 0.3 s ramps (10 cm/s) from the previous offset, as policy targets move (at most 1 cm per
            # step); an instant jump would read as a very fast goal and make the feedforward re-aims overshoot
            if args.ff:
                run(1, dur, lambda el, a=prev, b=off: a + (b - a) * min(1.0, el / 0.3))
            else:
                run(1, dur, lambda el, off=off: off)
            prev = off
        print("test 2: sweep", flush=True)
        run(2, 15.0, sweep_offset)
        run(2, 1.5, lambda el: np.zeros(3))
        print("test 3: gripper timing", flush=True)
        arm.hold()
        for k in range(2):
            t_cmd = time.time()
            g = arm.close_gripper(10.0, settle=0.8)
            events.append(("close", time.time() - t0, t_cmd - t0, g))
            t_cmd = time.time()
            g = arm.open_gripper(0.035)
            events.append(("open", time.time() - t0, t_cmd - t0, g))
        print("back to watch pose and rest", flush=True)
        arm.move_joints(WATCH, speed=0.6, label="to watch pose")
        arm.rest_and_idle(speed=0.6)
    except SafetyStop as e:
        print(f"SAFETY STOP: {e}", flush=True)
    except KeyboardInterrupt:
        print("stopped by Ctrl-C", flush=True)
    finally:
        out = ROOT / "runs" / "sysid"
        out.mkdir(parents=True, exist_ok=True)
        mode = f"_stream{args.stream:g}" if args.stream else ("_ff" if args.ff else "")
        path = out / f"sysid_{time.strftime('%Y%m%d_%H%M%S')}{mode}{'_dry' if args.dry else ''}.npz"
        np.savez(path, **{k: np.array(v) for k, v in log.items()},
                 events=np.array([(e[1], e[2], e[3]) for e in events]) if events else np.zeros((0, 3)),
                 event_kinds=np.array([e[0] for e in events]), ff=args.ff, ff_gain=args.ff_gain, stream=args.stream,
                 reaims=np.array(reaims).reshape(-1, 2))
        print(f"saved {path} ({len(log['t'])} samples)", flush=True)
        try:
            if arm.position_mode:  # stopped early: same exit as follow mode (watch pose, then rest)
                print("taking the arm to rest", flush=True)
                arm.hold()
                arm.move_joints(WATCH, speed=0.6, label="to watch pose")
                arm.rest_and_idle(speed=0.6)
        except SafetyStop as e:
            print(f"could not reach rest ({e}): arm holding; use python -m scripts.to_rest", flush=True)
        arm.gripper_idle()
        arm.close()

if __name__ == "__main__":
    main()
