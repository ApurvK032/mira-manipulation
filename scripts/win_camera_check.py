"""Camera check and short recording that runs on Windows Python (only pyrealsense2, numpy, opencv).

Used when the RealSense is plugged into the Windows host of a WSL machine: the native Windows driver
reads the camera, and the recording (scripts/record.py format) is replayed in WSL with run_live.

Usage (Windows):  python win_camera_check.py --out C:\\path\\rec --seconds 5
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

MODES = [(848, 480, 60), (848, 480, 30), (640, 480, 30), (640, 480, 15), (424, 240, 30), (424, 240, 15)]


def start(serial, modes):
    for w, h, fps in modes:
        pipe, cfg = rs.pipeline(), rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        try:
            return pipe, pipe.start(cfg), (w, h, fps)
        except RuntimeError as e:
            print(f"  mode {w}x{h}@{fps} not available: {e}")
    raise SystemExit("no color+depth mode worked")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--exposure", type=float, help="manual RGB exposure, units of 100 us (e.g. 80 = 8 ms)")
    ap.add_argument("--gain", type=float, default=32)
    ap.add_argument("--wb", type=float, help="manual white balance in K (e.g. 4200)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ctx = rs.context()
    for d in ctx.query_devices():
        print("device:", {k: d.get_info(getattr(rs.camera_info, k)) for k in
                          ("name", "serial_number", "firmware_version", "usb_type_descriptor")})
    pipe, profile, mode = start(args.serial, MODES)
    dev = profile.get_device()
    usb = dev.get_info(rs.camera_info.usb_type_descriptor)
    print(f"streaming color+depth {mode[0]}x{mode[1]}@{mode[2]} on USB {usb}")
    depth_sensor = dev.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    for s in dev.query_sensors():
        if s.supports(rs.option.global_time_enabled):
            s.set_option(rs.option.global_time_enabled, 1)
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1)
    color_sensor = next(s for s in dev.query_sensors() if "RGB" in s.get_info(rs.camera_info.name))
    # Never "lock" by reading the exposure option: in auto mode it keeps its last manual value, not
    # what auto exposure is using. Use explicit values (--exposure/--gain/--wb) or stay in auto.
    locked = {}
    if args.exposure:
        color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        color_sensor.set_option(rs.option.exposure, args.exposure)
        color_sensor.set_option(rs.option.gain, args.gain)
        locked.update(exposure=args.exposure, gain=args.gain)
    else:
        color_sensor.set_option(rs.option.enable_auto_exposure, 1)
    if args.wb:
        color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
        color_sensor.set_option(rs.option.white_balance, args.wb)
        locked["white_balance"] = args.wb
    else:
        color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
    print("color settings:", locked or "auto exposure + auto white balance")
    t_end = time.time() + args.settle
    while time.time() < t_end:
        pipe.wait_for_frames()
    if depth_sensor.supports(rs.option.laser_power):
        print("laser power:", depth_sensor.get_option(rs.option.laser_power))

    align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print(f"color intrinsics: fx {intr.fx:.1f} fy {intr.fy:.1f} cx {intr.ppx:.1f} cy {intr.ppy:.1f} "
          f"model {intr.model} coeffs {np.round(intr.coeffs, 4).tolist()}")

    stamps, lat, nums = [], [], []
    i = 0
    t_end = time.time() + args.seconds
    while time.time() < t_end:
        fs = pipe.wait_for_frames()
        t_arr = time.time() * 1000.0
        fs = align.process(fs)
        c, d = fs.get_color_frame(), fs.get_depth_frame()
        lat.append(t_arr - c.get_timestamp())
        nums.append(c.get_frame_number())
        cv2.imwrite(str(out / f"color_{i:06d}.png"), np.asanyarray(c.get_data()), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        cv2.imwrite(str(out / f"depth_{i:06d}.png"), np.asanyarray(d.get_data()), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        stamps.append([c.get_frame_number(), c.get_timestamp(), t_arr])
        i += 1
    pipe.stop()
    nums, lat = np.array(nums), np.array(lat)
    gaps = int((np.diff(nums) - 1).clip(min=0).sum())
    print(f"recorded {i} frames in {args.seconds:.0f} s = {i / args.seconds:.1f} fps (camera frames skipped: {gaps})")
    print(f"capture -> frame in Python ms: p50 {np.median(lat):.1f}  p95 {np.percentile(lat, 95):.1f}")
    meta = {"intrinsics": {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
                           "width": intr.width, "height": intr.height},
            "depth_scale": depth_scale, "camera": "D455", "serial": dev.get_info(rs.camera_info.serial_number),
            "usb": usb, "mode": mode, "locked": locked, "frames": stamps}
    (out / "meta.json").write_text(json.dumps(meta))
    print("saved to", out)


if __name__ == "__main__":
    main()
