"""Reliability run: stack the cubes N times while a person rearranges them between trials.

Each trial waits for both cubes to rest on the table (within reach, tracked, still for 3 s; one may partly
hide the other, and the grasp planner checks the open fingers clear the other cube),
then runs the stack routine (scripts.stack_test) with up to --attempts tries. After each trial the arm
returns to rest and the next trial waits for the cubes to be unstacked and placed somewhere new.
The live page (scripts.web_view) shows a status banner so the person can see when the arm will move.

Usage:  python -m scripts.reliability_run [--trials 10] [--top red --bottom green]
Logs every trial to runs/reliability/run_<time>.jsonl and writes a summary to docs/reliability/.
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from robot.arm import REST, SafeArm, SafetyStop
from robot.camera_frame import CameraToBase
from robot.grasp import read_cube
from scripts.stack_test import attempt, check_plan, measure, plan, stacked

ROOT = Path(__file__).resolve().parents[1]


def status(msg):
    print(msg, flush=True)
    try:
        urllib.request.urlopen("http://127.0.0.1:8080/status?msg=" + urllib.parse.quote(msg), timeout=1)
    except OSError:
        pass


def plan_or_wait(arm, cal, top, bottom, t):
    """Plan once the cubes are still. If a placement cannot be planned (out of the straight-down reach,
    or a pose the safety checks refuse), say so and wait for the cubes to be moved. Not an attempt."""
    while True:
        try:
            P = plan(cal, top, bottom, still_s=3.0, timeout=900.0, notify=lambda m: status(f"Trial {t}: waiting: {m}"))
            check_plan(arm, P)
            return P
        except (ValueError, SafetyStop) as e:
            status(f"Trial {t}: cannot plan this placement: {e}. Move the cubes, then step back.")
            before = {n: read_cube(cal, n)[1] for n in (top, bottom)}
            t0 = time.time()
            while time.time() - t0 < 900:
                time.sleep(0.5)
                now = {n: read_cube(cal, n)[1] for n in (top, bottom)}
                if any(before[n] is None or now[n] is None or np.linalg.norm(now[n] - before[n]) > 0.02 for n in now):
                    break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.1.3")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--top", default="red")
    ap.add_argument("--bottom", default="green")
    args = ap.parse_args()

    cal = CameraToBase()
    log_dir = ROOT / "runs" / "reliability"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{stamp}.jsonl"
    trials, prev = [], None
    arm = SafeArm(args.ip)
    try:
        if np.abs(arm.q() - REST).max() > 0.15:
            print("arm is not at the rest pose; not starting")
            return
        for t in range(1, args.trials + 1):
            status(f"Trial {t}/{args.trials}: unstack the cubes, place them in new spots, then step back. "
                   f"The arm moves 3 s after both are still.")
            rec = {"trial": t, "attempts": []}
            t_start = None
            for k in range(1, args.attempts + 1):
                P = plan_or_wait(arm, cal, args.top, args.bottom, t)
                if k == 1:
                    rec["start"] = {args.top: np.round(P["ct"] * 1000, 1).tolist(), args.bottom: np.round(P["cb"] * 1000, 1).tolist()}
                    if prev is not None:
                        rec["moved_from_previous_mm"] = round(float(min(np.linalg.norm(P["ct"][:2] - prev[0][:2]),
                                                                        np.linalg.norm(P["cb"][:2] - prev[1][:2]))) * 1000, 0)
                    prev = (P["ct"], P["cb"])
                    t_start = time.time()
                status(f"Trial {t}/{args.trials}, attempt {k}: arm moving (stacking {args.top} on {args.bottom})")
                t0 = time.time()
                try:
                    why = attempt(arm, cal, P, args.top, args.bottom, 15.0)
                except SafetyStop as e:
                    why = f"safety stop: {e}"
                    time.sleep(1.0)
                arm.rest_and_idle(speed=0.6)
                rec["attempts"].append({"attempt": k, "ok": why is None, "reason": why, "seconds": round(time.time() - t0, 1),
                                        "placed": P.pop("placed", None), "fingers_yaw_deg": round(float(np.degrees(P["yaw_t"])), 0)})
                if why is None:
                    break
                status(f"Trial {t}, attempt {k} failed: {why}. Retrying once the cubes are still ...")
            # success = an attempt passed its own check right after releasing (arm still above the stack).
            # The re-check from the rest pose is recorded too, retried for 3 s because the bottom cube of a
            # stack can drop out of tracking for a moment.
            ok = any(a["ok"] for a in rec["attempts"])
            fin, msg = None, "not checked"
            for _ in range(15):
                time.sleep(0.2)
                fin, (good, msg) = measure(cal, args.top, args.bottom), stacked(cal, args.top, args.bottom)
                if fin is not None:
                    break
            rec.update(success=ok, final=fin, check=msg, seconds_to_stack=round(time.time() - t_start, 1))
            trials.append(rec)
            with log_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            status(f"Trial {t}/{args.trials}: {'STACKED' if ok else 'FAILED'} after {len(rec['attempts'])} attempt(s): {msg}")
            time.sleep(2.0)
        status("Reliability run finished")
    except (KeyboardInterrupt, TimeoutError) as e:
        status(f"Reliability run stopped: {e or 'interrupted'}")
    finally:
        arm.gripper_idle()
        if not arm.position_mode:
            arm.close()
        summarize(trials, log_path, stamp)


def placed(r):
    """The successful attempt's in-place measurement (arm above the stack); older logs only have the check
    from the rest pose, where the bottom cube can be out of sight."""
    for a in r["attempts"]:
        if a["ok"] and a.get("placed"):
            return a["placed"]
    return r.get("final")


def summarize(trials, log_path, stamp):
    if not trials:
        return
    n = len(trials)
    ok = [r for r in trials if any(a["ok"] for a in r["attempts"])]  # older records: success was the rest re-check
    first = [r for r in ok if len(r["attempts"]) == 1]
    reasons = [a["reason"] for r in trials for a in r["attempts"] if not a["ok"]]
    fin = [placed(r) for r in ok if placed(r)]
    lines = [f"# Reliability run {stamp}", "",
             f"- Trials: {n}; stacked: **{len(ok)}/{n}**; on the first attempt: {len(first)}/{n}",
             f"- Attempts in total: {sum(len(r['attempts']) for r in trials)}; failed attempts: {len(reasons)}"]
    if fin:
        dxy = np.array([f["dxy_mm"] for f in fin]); dz = np.array([f["dz_mm"] for f in fin])
        lines.append(f"- Placement: off centre median {np.median(dxy):.1f} mm (max {dxy.max():.1f}); "
                     f"height median {np.median(dz):.1f} mm (true 50)")
        d = np.array([(f["dx_mm"], f["dy_mm"]) for f in fin if "dx_mm" in f])
        if len(d) >= 3:  # a consistent offset direction points at a calibration bias
            lines.append(f"- Offset of top from bottom (arm frame): mean x {d[:, 0].mean():+.1f} mm, y {d[:, 1].mean():+.1f} mm "
                         f"(spread {d[:, 0].std():.1f}, {d[:, 1].std():.1f})")
    secs = [r["seconds_to_stack"] for r in ok]
    if secs:
        lines.append(f"- Time from cubes still to stacked: median {np.median(secs):.0f} s")
    if reasons:
        lines += ["", "## Failed attempts"] + [f"- {x}" for x in reasons]
    lines += ["", "| Trial | Result | Attempts | Off centre (mm) | Height (mm) | Tilt (deg) | Start positions (mm) |",
              "|---|---|---|---|---|---|---|"]
    for r in trials:
        f = placed(r) or {}
        res = "stacked" if any(a["ok"] for a in r["attempts"]) else "failed"
        lines.append(f"| {r['trial']} | {res} | {len(r['attempts'])} | {f.get('dxy_mm', '-')} | "
                     f"{f.get('dz_mm', '-')} | {f.get('tilt_deg', '-')} | {r.get('start')} |")
    lines += ["", f"Raw log: `{log_path.relative_to(ROOT)}`"]
    out = ROOT / "docs" / "reliability" / f"run_{stamp}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"summary: {out}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == "--summarize":  # re-summarize a saved log
        path = Path(sys.argv[2]).resolve()
        summarize([json.loads(l) for l in path.read_text().splitlines() if l.strip()], path,
                  path.stem.replace("run_", ""))
    else:
        main()
