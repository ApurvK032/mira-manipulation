"""Summarize docs/poc/results_m3t.json as markdown tables (printed; pasted into docs/poc/README.md).

Usage:  python -m scripts.poc_report [results_m3t.json]
"""

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "poc"


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "results_m3t.json"
    R = json.loads(path.read_text())
    scen = [k for k in R if k != "config"]

    print("| Scenario | Frames | Wrong TRACKING frames | Position error median / p95 / max (mm) | "
          "Rotation error median / p95 / max (deg) | Compute p50 / p95 (ms) |")
    print("|---|---|---|---|---|---|")
    all_pe, all_wrong, all_frames = [], 0, 0
    for s in scen:
        r = R[s]
        cubes = [d for d in r["cubes"].values() if "pos_med" in d]
        pm = max(d["pos_med"] for d in cubes)
        pp = max(d["pos_p95"] for d in cubes)
        px = max(d["pos_max"] for d in cubes)
        rm = max(d["rot_med"] for d in cubes)
        rp = max(d["rot_p95"] for d in cubes)
        rx = max(d["rot_max"] for d in cubes)
        tm = r["timing_ms"]["total"]
        cube_frames = sum(d["frames"] for d in r["cubes"].values())
        all_wrong += len(r["wrong_tracking_frames"])
        all_frames += cube_frames
        print(f"| {s} | {r['frames']} | {len(r['wrong_tracking_frames'])} of {cube_frames} | {pm:.2f} / {pp:.2f} / {px:.2f} | "
              f"{rm:.2f} / {rp:.2f} / {rx:.2f} | {tm['p50']:.1f} / {tm['p95']:.1f} |")
    print(f"\nWrong TRACKING cube-frames in total: {all_wrong} of {all_frames}\n")

    re = [(s, e) for s in scen for e in R[s]["events"] if "reacquire_frames" in e]
    print("| Event | Count | Found again (frames): median / max | Not found again |")
    print("|---|---|---|---|")
    for kind in ("teleport", "release", "knock"):
        v = [e["reacquire_frames"] for _, e in re if e["kind"] == kind and e["reacquire_frames"] is not None]
        miss = sum(1 for _, e in re if e["kind"] == kind and e["reacquire_frames"] is None)
        if v or miss:
            print(f"| {kind} | {len(v) + miss} | {np.median(v):.0f} / {max(v)} ({1000 * np.median(v) / 60:.0f} / "
                  f"{1000 * max(v) / 60:.0f} ms) | {miss} |")

    print("\n| Static / slide check | Result |")
    print("|---|---|")
    for s in scen:
        for e in R[s]["events"]:
            if e["kind"] == "static" and "jitter_mm" in e:
                print(f"| {s} {e['cube']} jitter | {e['jitter_mm']:.2f} mm, TRACKING {100 * e['tracking_fraction']:.0f}% of frames |")
            if e["kind"] == "slide" and "measured_mm" in e:
                lag = e.get("motion_lag_frames")
                print(f"| {s} {e['cube']} pushed {e['commanded_mm']:.1f} mm, rotated {e['dyaw_deg']:.0f} deg | measured "
                      f"{e['measured_mm']:.2f} mm (error {e['displacement_error_mm']:.2f} mm), lag {lag} frames |")


if __name__ == "__main__":
    main()
