"""List RealSense cameras and measure what one actually delivers: frame rate, drops, latency, depth.

Usage:  python -m scripts.check_camera                # list devices, test the first one for 10 s
        python -m scripts.check_camera --serial 123 --fps 90 --seconds 20
Saves a color + depth snapshot to docs/camera_check/.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "camera_check"


def list_devices():
    import pyrealsense2 as rs

    ctx = rs.context()
    devs = []
    for d in ctx.query_devices():
        info = {k: d.get_info(getattr(rs.camera_info, k)) for k in ("name", "serial_number", "firmware_version",
                                                                      "usb_type_descriptor", "product_line")
                if d.supports(getattr(rs.camera_info, k))}
        devs.append(info)
        print("  ", info)
    return devs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial")
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()

    print("RealSense devices:")
    devs = list_devices()
    if not devs:
        print("No RealSense found. On WSL: attach it with usbipd (see README), then rerun.")
        return
    from perception.sources import RealSenseSource

    src = RealSenseSource(serial=args.serial, width=args.width, height=args.height, fps=args.fps)
    print(f"\nTesting {src.name} {src.serial} on USB {src.usb} at {args.width}x{args.height}@{args.fps}")
    if not str(src.usb).startswith("3"):
        print("  WARNING: not on USB 3; color + depth at this rate needs USB 3")
    print("  locked:", src.locked, " depth scale:", src.depth_scale)
    print("  intrinsics:", src.intrinsics)

    numbers, lat, gaps, t_arr = [], [], [], []
    t_end = time.time() + args.seconds
    last = None
    while time.time() < t_end:
        f = src.read()
        now = time.time() * 1000.0
        numbers.append(f.index)
        lat.append(now - f.t_capture_ms)
        t_arr.append(f.t_arrival_ms)
        last = f
    src.close()

    numbers, lat, t_arr = np.array(numbers), np.array(lat), np.array(t_arr)
    dn = np.diff(numbers)
    dropped = int((dn - 1).clip(min=0).sum())
    dt = np.diff(t_arr)
    print(f"\n  frames received {len(numbers)} in {args.seconds:.0f} s = {len(numbers) / args.seconds:.1f} fps "
          f"(camera frames skipped: {dropped}, includes ones we dropped to stay on the newest frame)")
    print(f"  inter-arrival ms  p50 {np.median(dt):.1f}  p95 {np.percentile(dt, 95):.1f}  max {dt.max():.1f}")
    print(f"  capture -> aligned frame in Python ms  p50 {np.median(lat):.1f}  p95 {np.percentile(lat, 95):.1f}  "
          f"max {lat.max():.1f}")

    d = last.depth.astype(np.float64) * src.depth_scale
    h, w = d.shape
    center = d[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
    print(f"  depth valid {100 * (d > 0).mean():.1f}% (center {100 * (center > 0).mean():.1f}%), "
          f"center median {np.median(center[center > 0]):.3f} m")
    OUT.mkdir(parents=True, exist_ok=True)
    dm = cv2.applyColorMap(cv2.convertScaleAbs(last.depth, alpha=255.0 / (1.5 / src.depth_scale)), cv2.COLORMAP_TURBO)
    path = OUT / f"{src.serial}.png"
    cv2.imwrite(str(path), np.hstack([last.color, dm]))
    print(f"  snapshot: {path}")


if __name__ == "__main__":
    main()
