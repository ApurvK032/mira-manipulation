"""Stream a RealSense from Windows into WSL over TCP (runs on Windows Python: pyrealsense2 + numpy).

The camera is read natively on Windows. Each frame (BGR color + depth aligned to color) goes to the
WSL receiver (perception.sources.NetworkSource, e.g. scripts/web_view.py --source tcp:...). Only the
newest frame is sent: if the link is busy, older frames are dropped instead of queuing up latency.

Usage (Windows):  python win_camera_stream.py --host <WSL eth0 IP> --port 5600 --wb 3000
Color: auto exposure stays on (manual RGB exposure gives black frames through the Windows driver).
"""

import argparse
import json
import socket
import struct
import threading
import time

import numpy as np
import pyrealsense2 as rs

MAGIC = b"RSF1"
HEADER = struct.Struct("<4sQddII")  # magic, frame number, t_capture_ms, t_send_ms, color bytes, depth bytes


def connect(host, port):
    while True:
        try:
            s = socket.create_connection((host, port), timeout=3)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(None)
            return s
        except OSError:
            print(f"waiting for the WSL receiver at {host}:{port} ...", flush=True)
            time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--wb", type=float, default=3000, help="fixed white balance in K (0 = auto)")
    args = ap.parse_args()

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    prof = pipe.start(cfg)
    dev = prof.get_device()
    depth_sensor = dev.first_depth_sensor()
    for s in dev.query_sensors():
        if s.supports(rs.option.global_time_enabled):
            s.set_option(rs.option.global_time_enabled, 1)
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1)
    color = next(s for s in dev.query_sensors() if "RGB" in s.get_info(rs.camera_info.name))
    color.set_option(rs.option.enable_auto_exposure, 1)
    if args.wb:
        color.set_option(rs.option.enable_auto_white_balance, 0)
        color.set_option(rs.option.white_balance, args.wb)
    else:
        color.set_option(rs.option.enable_auto_white_balance, 1)
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    hello = json.dumps({"intrinsics": {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
                                       "width": intr.width, "height": intr.height},
                        "depth_scale": depth_sensor.get_depth_scale(),
                        "camera": dev.get_info(rs.camera_info.name),
                        "serial": dev.get_info(rs.camera_info.serial_number),
                        "usb": dev.get_info(rs.camera_info.usb_type_descriptor),
                        "mode": [args.width, args.height, args.fps], "white_balance": args.wb}).encode()
    print("camera:", json.loads(hello)["camera"], "USB", json.loads(hello)["usb"], flush=True)
    align = rs.align(rs.stream.color)

    latest = {"frame": None}
    cond = threading.Condition()

    def sender():
        while True:
            sock = connect(args.host, args.port)
            print("connected to WSL receiver", flush=True)
            try:
                sock.sendall(struct.pack("<I", len(hello)) + hello)
                sent, t0 = 0, time.time()
                while True:
                    with cond:
                        while latest["frame"] is None:
                            cond.wait()
                        num, tc, c, d = latest["frame"]
                        latest["frame"] = None
                    hdr = HEADER.pack(MAGIC, num, tc, time.time() * 1000.0, len(c), len(d))
                    sock.sendall(hdr + c + d)
                    sent += 1
                    if time.time() - t0 > 5:
                        print(f"sending {sent / (time.time() - t0):.1f} fps", flush=True)
                        sent, t0 = 0, time.time()
            except OSError as e:
                print("connection lost:", e, flush=True)
                sock.close()

    threading.Thread(target=sender, daemon=True).start()
    try:
        while True:
            fs = align.process(pipe.wait_for_frames())
            c, d = fs.get_color_frame(), fs.get_depth_frame()
            item = (c.get_frame_number(), c.get_timestamp(), bytes(c.get_data()), bytes(d.get_data()))
            with cond:
                latest["frame"] = item  # overwrite: never queue old frames
                cond.notify()
    finally:
        pipe.stop()


if __name__ == "__main__":
    main()
