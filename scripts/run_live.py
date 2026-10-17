"""Run cube perception live on a RealSense (or on a recording) with an overlay window.

Usage:  python -m scripts.run_live                         # first RealSense, all four colors
        python -m scripts.run_live --cubes blue,red
        python -m scripts.run_live --source recordings/hand_moves
        python -m scripts.run_live --no-m3t                # detector only, for comparison
Keys: q quit, space pause, s save snapshot.
Shows per-cube state and pose, compute time, and capture-to-pose latency.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")  # M3T's OpenMP pool, read when pym3t loads
# small matrices only: one BLAS thread, and OpenMP threads that sleep instead of spinning, so the two
# pools do not fight over cores (that showed up as 50 ms stalls every time the table plane was refit)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from perception.detector import load_colors
from perception.overlay import draw_result
from perception.pipeline import CubePerception
from perception.sources import open_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="realsense")
    ap.add_argument("--cubes", default="red,blue,green,yellow")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--no-m3t", action="store_true")
    ap.add_argument("--log", help="CSV file for per-frame poses and states")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    kw = {"fps": args.fps} if args.source.startswith("realsense") else {}
    src = open_source(args.source, **kw)
    colors = load_colors()
    names = [n for n in args.cubes.split(",") if n in colors]
    per = CubePerception(src.intrinsics, names, colors={n: colors[n] for n in names},
                         use_m3t=not args.no_m3t, depth_scale=src.depth_scale)
    log = None
    if args.log:
        log = csv.writer(open(args.log, "w", newline=""))
        log.writerow(["frame", "t_capture_ms", "latency_ms", "cube", "state", "x", "y", "z", "compute_ms"])

    lat_hist, comp_hist = [], []
    t_last, fps_txt = time.time(), ""
    n = 0
    paused = False
    snap_dir = Path("docs/snapshots")
    while True:
        if not paused:
            f = src.read()
            if f is None:
                break
            r = per.process(f.color, f.depth)
            now = time.time() * 1000.0
            latency = now - f.t_capture_ms if args.source.startswith("realsense") else float("nan")
            lat_hist.append(latency)
            comp_hist.append(r.timing_ms["total"])
            n += 1
            if time.time() - t_last > 1.0:
                fps = n / (time.time() - t_last)
                lat = np.array(lat_hist[-120:])
                lat_txt = f"capture->pose p50 {np.median(lat):.1f} ms" if np.isfinite(lat).all() else "replay"
                fps_txt = f"| {fps:.0f} fps | {lat_txt}"
                print(f"{fps:5.1f} fps  compute p50 {np.median(comp_hist[-120:]):.1f} ms  {lat_txt}  "
                      + "  ".join(f"{k}:{v.state[0]}" for k, v in r.cubes.items()), flush=True)
                n, t_last = 0, time.time()
            if log:
                for k, e in r.cubes.items():
                    p = e.T[:3, 3] if e.T is not None else (np.nan, np.nan, np.nan)
                    log.writerow([f.index, f.t_capture_ms, latency, k, e.state, *p, r.timing_ms["total"]])
            if not args.headless:
                img = draw_result(f.color, src.intrinsics, r, colors, fps_text=fps_txt)
                cv2.imshow("cubes", img)
        if not args.headless:
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" "):
                paused = not paused
            if k == ord("s"):
                snap_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(snap_dir / f"live_{time.strftime('%H%M%S')}.png"), img)
    src.close()


if __name__ == "__main__":
    main()
