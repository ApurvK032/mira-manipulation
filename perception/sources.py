"""Frame sources with one interface: RealSense (live) and recorded directories (replay).

read() returns a CameraFrame with BGR color, depth aligned to color (uint16, depth_scale m/unit),
and timestamps in host milliseconds, so capture-to-pose latency can be measured.
"""

import json
import socket
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .camera import Intrinsics


@dataclass
class CameraFrame:
    index: int
    color: np.ndarray
    depth: np.ndarray
    t_capture_ms: float  # sensor timestamp, host clock (RealSense global time)
    t_arrival_ms: float  # when the frame reached this process


class RealSenseSource:
    """D455 / D405 color + depth at the same resolution, depth aligned to color.

    Exposure and white balance run in auto mode for `settle_s`, then are locked so cube colors stay
    constant (auto modes shift hue and brightness as hands and arms move through the image).
    """

    def __init__(self, serial=None, width=848, height=480, fps=60, settle_s=1.5, lock_exposure=True):
        import pyrealsense2 as rs

        self.rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.profile = self.pipe.start(cfg)
        dev = self.profile.get_device()
        self.name = dev.get_info(rs.camera_info.name)
        self.serial = dev.get_info(rs.camera_info.serial_number)
        self.usb = dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else "?"
        depth_sensor = dev.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        for s in dev.query_sensors():
            if s.supports(rs.option.global_time_enabled):
                s.set_option(rs.option.global_time_enabled, 1)
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1)  # the projector gives texture to plain cube faces
        self.align = rs.align(rs.stream.color)
        cs = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = Intrinsics.from_realsense(cs.get_intrinsics())
        self.index = 0

        t_end = time.time() + settle_s
        while time.time() < t_end:
            self.pipe.wait_for_frames()
        self.locked = {}
        if lock_exposure:
            self.locked = self._lock(dev)

    def _lock(self, dev):
        """Freeze the values auto exposure / white balance settled on.

        The exposure *option* does not report what auto exposure is using (it keeps its last manual
        value), so the real values come from frame metadata. Without metadata (e.g. Windows without
        the metadata registry patch) the camera stays in auto mode rather than being locked blind.
        """
        rs = self.rs
        M = rs.frame_metadata_value
        color = next((s for s in dev.query_sensors() if "rgb" in s.get_info(rs.camera_info.name).lower()), None)
        if color is None:  # D405: color comes from the stereo module
            color = dev.first_depth_sensor()
        f = self.pipe.wait_for_frames().get_color_frame()
        locked = {}
        if f.supports_frame_metadata(M.actual_exposure) and f.supports_frame_metadata(M.gain_level):
            exposure, gain = f.get_frame_metadata(M.actual_exposure), f.get_frame_metadata(M.gain_level)
            color.set_option(rs.option.enable_auto_exposure, 0)
            # metadata exposure is in microseconds; the RGB option is in units of 100 us
            color.set_option(rs.option.exposure, max(1.0, round(exposure / 100.0)))
            color.set_option(rs.option.gain, gain)
            locked.update(exposure_us=exposure, gain=gain)
        if f.supports_frame_metadata(M.white_balance):
            wb = f.get_frame_metadata(M.white_balance)
            color.set_option(rs.option.enable_auto_white_balance, 0)
            color.set_option(rs.option.white_balance, wb)
            locked["white_balance"] = wb
        if not locked:
            print("RealSense: no frame metadata, leaving auto exposure and white balance on")
        return locked

    def read(self) -> CameraFrame:
        frames = self.pipe.wait_for_frames()
        # drop anything older that is already queued: always process the newest frame
        while True:
            newer = self.pipe.poll_for_frames()
            if not newer:
                break
            frames = newer
        t_arrival = time.time() * 1000.0
        frames = self.align.process(frames)
        c, d = frames.get_color_frame(), frames.get_depth_frame()
        self.index += 1
        return CameraFrame(c.get_frame_number(), np.asanyarray(c.get_data()).copy(), np.asanyarray(d.get_data()).copy(),
                           c.get_timestamp(), t_arrival)

    def close(self):
        self.pipe.stop()


class DirectorySource:
    """Replays a directory written by scripts/record.py."""

    def __init__(self, path, realtime=False):
        self.path = Path(path)
        meta = json.loads((self.path / "meta.json").read_text())
        self.intrinsics = Intrinsics(**meta["intrinsics"])
        self.depth_scale = meta["depth_scale"]
        self.stamps = meta["frames"]  # list of [index, t_capture_ms, t_arrival_ms]
        self.i = 0
        self.realtime = realtime
        self.t0 = None

    def __len__(self):
        return len(self.stamps)

    def read(self) -> CameraFrame | None:
        if self.i >= len(self.stamps):
            return None
        idx, tc, ta = self.stamps[self.i]
        if self.realtime:
            now = time.time() * 1000.0
            if self.t0 is None:
                self.t0 = (now, tc)
            wait = (tc - self.t0[1]) - (now - self.t0[0])
            if wait > 0:
                time.sleep(wait / 1000.0)
        color = cv2.imread(str(self.path / f"color_{self.i:06d}.png"), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(self.path / f"depth_{self.i:06d}.png"), cv2.IMREAD_UNCHANGED)
        self.i += 1
        return CameraFrame(idx, color, depth, tc, ta)

    def close(self):
        pass


class NetworkSource:
    """Frames streamed by scripts/win_camera_stream.py (camera on the Windows host, pipeline in WSL).

    Keeps only the newest frame, so a slow consumer never builds up latency. If the streamer
    restarts, it reconnects. The constructor blocks until the first streamer connects.
    """

    HEADER = struct.Struct("<4sQddII")

    def __init__(self, host="0.0.0.0", port=5600):
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((host, port))
        self.srv.listen(1)
        self.cond = threading.Condition()
        self.latest = None
        self.info = None
        self.received = 0
        print(f"waiting for the camera streamer on {host}:{port} ...", flush=True)
        self._accept()
        threading.Thread(target=self._reader, daemon=True).start()

    def _accept(self):
        self.conn, addr = self.srv.accept()
        self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        (n,) = struct.unpack("<I", self._recv(4))
        self.info = json.loads(self._recv(n))
        self.intrinsics = Intrinsics(**self.info["intrinsics"])
        self.depth_scale = self.info["depth_scale"]
        print(f"camera streamer connected from {addr[0]}: {self.info['camera']} USB {self.info['usb']} "
              f"{self.info['mode']}", flush=True)

    def _recv(self, n):
        buf = bytearray(n)
        view, got = memoryview(buf), 0
        while got < n:
            k = self.conn.recv_into(view[got:], n - got)
            if k == 0:
                raise ConnectionError("streamer disconnected")
            got += k
        return buf

    def _reader(self):
        w, h = self.intrinsics.width, self.intrinsics.height
        while True:
            try:
                magic, num, tc, _ts, nc, nd = self.HEADER.unpack(self._recv(self.HEADER.size))
                if magic != b"RSF1":
                    raise ConnectionError("bad frame header")
                color = np.frombuffer(self._recv(nc), np.uint8).reshape(h, w, 3)
                depth = np.frombuffer(self._recv(nd), np.uint16).reshape(h, w)
                f = CameraFrame(num, color, depth, tc, time.time() * 1000.0)
                with self.cond:
                    self.latest = f
                    self.received += 1
                    self.cond.notify_all()
            except (ConnectionError, OSError) as e:
                print(f"camera streamer: {e}; waiting for it to reconnect", flush=True)
                self.conn.close()
                self._accept()

    def read(self) -> CameraFrame:
        with self.cond:
            while self.latest is None:
                self.cond.wait()
            f, self.latest = self.latest, None
        return f

    def close(self):
        self.srv.close()


def open_source(spec: str, **kw):
    """'realsense', 'realsense:<serial>', 'tcp:<bind host>:<port>' (Windows streamer) or a recording."""
    if spec == "realsense" or spec.startswith("realsense:"):
        serial = spec.split(":", 1)[1] if ":" in spec else None
        return RealSenseSource(serial=serial, **kw)
    if spec.startswith("tcp:"):
        _, host, port = spec.split(":")
        return NetworkSource(host, int(port))
    return DirectorySource(spec)
