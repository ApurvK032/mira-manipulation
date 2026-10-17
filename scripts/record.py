"""Record aligned color + depth from a RealSense, to replay through the pipeline at your desk.

Usage:  python -m scripts.record --seconds 20 --out recordings/hand_moves
Replay: python -m scripts.run_live --source recordings/hand_moves
"""

import argparse
import json
import queue
import threading
import time
from pathlib import Path

import cv2

from perception.sources import RealSenseSource


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--out", default=f"recordings/{time.strftime('%Y%m%d_%H%M%S')}")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = RealSenseSource(serial=args.serial, fps=args.fps)
    q = queue.Queue(maxsize=600)

    def writer():
        while (item := q.get()) is not None:
            i, f = item
            cv2.imwrite(str(out / f"color_{i:06d}.png"), f.color, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            cv2.imwrite(str(out / f"depth_{i:06d}.png"), f.depth, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    threads = [threading.Thread(target=writer, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()
    stamps = []
    print(f"Recording {args.seconds:.0f} s from {src.name} {src.serial} ... move the cubes around")
    t_end = time.time() + args.seconds
    i = 0
    while time.time() < t_end:
        f = src.read()
        q.put((i, f))
        stamps.append([f.index, f.t_capture_ms, f.t_arrival_ms])
        i += 1
    src.close()
    for _ in threads:
        q.put(None)
    for t in threads:
        t.join()
    meta = {"intrinsics": src.intrinsics.as_dict(), "depth_scale": src.depth_scale, "camera": src.name,
            "serial": src.serial, "locked": src.locked, "frames": stamps}
    (out / "meta.json").write_text(json.dumps(meta))
    print(f"Saved {i} frames to {out}")


if __name__ == "__main__":
    main()
