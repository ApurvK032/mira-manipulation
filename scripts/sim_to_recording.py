"""Write a simulated scenario in the scripts/record.py format (to test replay and run_live offline).

Usage:  python -m scripts.sim_to_recording teleport recordings/sim_teleport
"""

import json
import sys
from pathlib import Path

import cv2

from perception.camera import Intrinsics
from sim.render_worker import render_frames
from sim.scene import FOVY, HEIGHT, WIDTH


def main():
    name, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    stamps = []
    for i, f in enumerate(render_frames(name)):
        cv2.imwrite(str(out / f"color_{i:06d}.png"), f.color, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        cv2.imwrite(str(out / f"depth_{i:06d}.png"), f.depth, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        stamps.append([i, 1000.0 * f.t, 1000.0 * f.t])
    meta = {"intrinsics": Intrinsics.from_fov(WIDTH, HEIGHT, FOVY).as_dict(), "depth_scale": 0.001,
            "camera": "sim D455", "serial": "sim", "locked": {}, "frames": stamps}
    (out / "meta.json").write_text(json.dumps(meta))
    print(f"wrote {len(stamps)} frames to {out}")


if __name__ == "__main__":
    main()
