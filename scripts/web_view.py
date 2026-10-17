"""Live cube tracking in a web browser.

Runs the perception pipeline on a live source and serves the overlay as an MJPEG stream plus live
stats on http://127.0.0.1:8080 (localhost only; forward the port over SSH to view it elsewhere).

Usage:  python -m scripts.web_view --cubes red,green
        (with scripts/win_camera_stream.py running on Windows), or --source realsense / a recording.
Views: overlay (tracked cubes), depth, masks (what each color range picks up, for tuning).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from perception.detector import load_colors
from perception.overlay import draw_cube, draw_result
from perception.pipeline import CubePerception
from perception.sources import open_source

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cube tracking</title>
<style>
:root { --bg:#f6f6f4; --fg:#1c1c1c; --muted:#6b6b6b; --card:#ffffff; --line:#e2e2de; --ok:#1f8a3b; --warn:#b7791f; --bad:#c0392b; }
@media (prefers-color-scheme: dark) { :root { --bg:#161616; --fg:#ececec; --muted:#9a9a9a; --card:#202020; --line:#333; --ok:#4cc26b; --warn:#e0a64a; --bad:#ef6a5b; } }
* { box-sizing: border-box; }
body { margin:0; padding:16px; background:var(--bg); color:var(--fg); font:14px/1.4 system-ui, sans-serif; }
h1 { font-size:18px; margin:0 0 12px; font-weight:600; }
.wrap { display:flex; flex-wrap:wrap; gap:16px; align-items:flex-start; }
.video { flex:1 1 640px; max-width:960px; }
.video img { width:100%; height:auto; display:block; border-radius:8px; background:#000; }
.side { flex:0 1 340px; min-width:280px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; margin-bottom:12px; }
.views button { font:inherit; padding:6px 12px; margin-right:6px; border-radius:6px; border:1px solid var(--line);
  background:var(--card); color:var(--fg); cursor:pointer; }
.views button.on { border-color:var(--fg); font-weight:600; }
table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
td, th { text-align:left; padding:3px 4px; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:500; }
.TRACKING { color:var(--ok); font-weight:600; } .OCCLUDED { color:var(--warn); font-weight:600; } .LOST { color:var(--bad); font-weight:600; }
.k { color:var(--muted); }
.arm button { font:inherit; padding:8px 14px; margin:0 6px 6px 0; border-radius:6px; border:1px solid var(--line);
  background:var(--card); color:var(--fg); cursor:pointer; font-weight:600; }
.arm button.rest { border-color:var(--bad); color:var(--bad); }
</style></head><body>
<h1>Cube tracking (live)</h1>
<div id="status" class="card" style="display:none;font-weight:600"></div>
<div class="wrap">
  <div class="video"><img id="v" src="/stream.mjpg" alt="live view"></div>
  <div class="side">
    <div class="card arm">
      <div class="k" style="margin-bottom:6px">Arm (stacking controller)</div>
      <button data-c="start">Start</button><button data-c="pause">Pause</button><button data-c="resume">Resume</button><button data-c="rest" class="rest">Go to rest</button>
      <div class="k" id="cmd"></div>
    </div>
    <div class="card views">
      <button data-v="overlay">Overlay</button><button data-v="depth">Depth</button><button data-v="masks">Color masks</button>
    </div>
    <div class="card"><table id="cubes"><tr><th>Cube</th><th>State</th><th>x, y, z (mm)</th></tr></table></div>
    <div class="card"><table id="perf"></table></div>
    <div class="card k" id="cam"></div>
  </div>
</div>
<script>
const views = document.querySelectorAll('.views button');
function mark(v) { views.forEach(b => b.classList.toggle('on', b.dataset.v === v)); }
views.forEach(b => b.onclick = () => fetch('/view?mode=' + b.dataset.v).then(() => mark(b.dataset.v)));
document.querySelectorAll('.arm button').forEach(b => b.onclick = () =>
  fetch('/command?c=' + b.dataset.c).then(() => { document.getElementById('cmd').textContent = 'sent: ' + b.textContent; }));
async function tick() {
  try {
    const s = await (await fetch('/stats.json', {cache: 'no-store'})).json();
    mark(s.view);
    const st = document.getElementById('status');
    st.style.display = s.status ? 'block' : 'none'; st.textContent = s.status || '';
    let rows = '<tr><th>Cube</th><th>State</th><th>x, y, z (mm)</th></tr>';
    for (const [n, c] of Object.entries(s.cubes)) {
      const p = c.pos ? c.pos.map(v => v.toFixed(1)).join(', ') : '–';
      rows += `<tr><td>${n}</td><td class="${c.state}">${c.state}</td><td>${p}</td></tr>`;
      const k = c.check;
      rows += `<tr><td></td><td colspan="2" class="k">unhidden ${k[0].toFixed(2)} · color ${k[1].toFixed(2)} · depth ${k[2].toFixed(2)}</td></tr>`;
    }
    document.getElementById('cubes').innerHTML = rows;
    const f = (v, d=1) => (v === null || v === undefined) ? '–' : v.toFixed(d);
    document.getElementById('perf').innerHTML =
      `<tr><td class="k">Pipeline rate</td><td>${f(s.fps)} fps</td></tr>` +
      `<tr><td class="k">Compute per frame (p50 / p95)</td><td>${f(s.compute_p50)} / ${f(s.compute_p95)} ms</td></tr>` +
      `<tr><td class="k">Camera capture → pose</td><td>${f(s.latency_p50)} ms</td></tr>` +
      `<tr><td class="k">Frames from camera</td><td>${s.received}</td></tr>`;
    document.getElementById('cam').textContent = s.camera;
  } catch (e) {}
  setTimeout(tick, 250);
}
tick();
</script></body></html>"""


class Viewer:
    def __init__(self, src, per, colors, stream_fps, quality):
        self.src, self.per, self.colors = src, per, colors
        self.view = "overlay"
        self.quality = quality
        self.min_dt = 1.0 / stream_fps
        self.cond = threading.Condition()
        self.jpeg, self.seq = None, 0
        self.raw = None  # latest (color, depth) for /raw.png and /depth.png
        self.status = ""  # banner text set by a running task (GET /status?msg=...)
        self.command = {"seq": 0, "c": None, "t": None}  # last operator button press, read by robot.controller
        self.stats = {"cubes": {}, "view": self.view, "received": 0, "camera": ""}
        self.comp, self.lat, self.times = [], [], []
        self.log = None  # set to a file to record per-frame states and check scores

    def render(self, f, r):
        if self.view == "depth":
            d = cv2.convertScaleAbs(f.depth, alpha=255.0 / (1.2 / self.src.depth_scale))
            img = cv2.applyColorMap(d, cv2.COLORMAP_TURBO)
            img[f.depth == 0] = 0
            for n, e in r.cubes.items():
                if e.T is not None:
                    draw_cube(img, self.src.intrinsics, e.T, (255, 255, 255), 2)
            return img
        if self.view == "masks":
            masks = self.per.detector.color_masks(f.color)
            img = (f.color * 0.35).astype(np.uint8)
            for n, m in masks.items():
                img[m > 0] = self.colors[n].bgr
            return img
        return draw_result(f.color, self.src.intrinsics, r, self.colors)

    def loop(self):
        last_pub = 0.0
        while True:
            f = self.src.read()
            self.raw = (f.color, f.depth)
            r = self.per.process(f.color, f.depth)
            now = time.time()
            self.comp.append(r.timing_ms["total"])
            self.lat.append(now * 1000.0 - f.t_capture_ms)
            self.times.append(now)
            self.comp, self.lat, self.times = self.comp[-240:], self.lat[-240:], self.times[-240:]
            fps = (len(self.times) - 1) / (self.times[-1] - self.times[0]) if len(self.times) > 10 else None
            cubes = {n: {"state": e.state, "pos": (e.T[:3, 3] * 1000).round(1).tolist() if e.T is not None else None,
                         "rot": e.T[:3, :3].round(5).tolist() if e.T is not None else None,
                         "check": [round(e.check.visible, 2), round(e.check.color, 2), round(e.check.depth, 2),
                                   round(e.check.leak, 2)], "source": e.source}
                     for n, e in r.cubes.items()}
            if self.log is not None:
                self.log.write(json.dumps({"t": now, "cubes": cubes}) + "\n")
            self.stats = {"t": now, "cubes": cubes, "view": self.view, "fps": fps, "status": self.status,
                          "command": self.command,
                          "compute_p50": float(np.median(self.comp)), "compute_p95": float(np.percentile(self.comp, 95)),
                          "latency_p50": float(np.median(self.lat)),
                          "received": getattr(self.src, "received", 0),
                          "camera": " · ".join(str(x) for x in (getattr(self.src, "info", None) or {}).values()
                                               if not isinstance(x, dict))}
            if now - last_pub >= self.min_dt:
                ok, buf = cv2.imencode(".jpg", self.render(f, r), [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if ok:
                    with self.cond:
                        self.jpeg, self.seq = buf.tobytes(), self.seq + 1
                        self.cond.notify_all()
                last_pub = now


def make_handler(viewer: Viewer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                return self._send(200, "text/html; charset=utf-8", PAGE.encode())
            if self.path.startswith("/stats.json"):
                return self._send(200, "application/json", json.dumps(viewer.stats).encode())
            if self.path.startswith("/view?mode="):
                mode = self.path.split("=", 1)[1]
                if mode in ("overlay", "depth", "masks"):
                    viewer.view = mode
                return self._send(200, "text/plain", b"ok")
            if self.path.startswith("/raw.png") or self.path.startswith("/depth.png"):
                if viewer.raw is None:
                    return self._send(503, "text/plain", b"no frame yet")
                img = viewer.raw[0] if self.path.startswith("/raw.png") else viewer.raw[1]
                ok, buf = cv2.imencode(".png", img)  # lossless; depth stays uint16 (depth_scale m per unit)
                return self._send(200, "image/png", buf.tobytes())
            if self.path.startswith("/intrinsics.json"):
                info = {"intrinsics": viewer.src.intrinsics.as_dict(), "depth_scale": viewer.src.depth_scale}
                return self._send(200, "application/json", json.dumps(info).encode())
            if self.path.startswith("/status"):
                from urllib.parse import parse_qs, urlparse

                viewer.status = parse_qs(urlparse(self.path).query).get("msg", [""])[0][:200]
                return self._send(200, "text/plain", b"ok")
            if self.path.startswith("/command"):
                from urllib.parse import parse_qs, urlparse

                c = parse_qs(urlparse(self.path).query).get("c", [""])[0]
                if c not in ("start", "pause", "resume", "rest"):
                    return self._send(400, "text/plain", b"unknown command")
                viewer.command = {"seq": viewer.command["seq"] + 1, "c": c, "t": time.time()}
                return self._send(200, "text/plain", b"ok")
            if self.path.startswith("/snapshot.jpg"):
                return self._send(200, "image/jpeg", viewer.jpeg or b"")
            if self.path.startswith("/stream.mjpg"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                seq = -1
                try:
                    while True:
                        with viewer.cond:
                            viewer.cond.wait_for(lambda: viewer.seq != seq, timeout=5)
                            jpg, seq = viewer.jpeg, viewer.seq
                        if jpg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return
            return self._send(404, "text/plain", b"not found")

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="tcp:127.0.0.1:5600")  # localhost only (mirrored WSL networking)
    ap.add_argument("--cubes", default="red,green")
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--stream-fps", type=float, default=30.0, help="browser frame rate (the pipeline runs at camera rate)")
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--no-m3t", action="store_true")
    ap.add_argument("--log", help="JSON-lines file with per-frame states and check scores")
    args = ap.parse_args()

    colors = load_colors()
    names = [n for n in args.cubes.split(",") if n in colors]
    server = ThreadingHTTPServer((args.bind, args.port), None)
    src = open_source(args.source)
    per = CubePerception(src.intrinsics, names, colors={n: colors[n] for n in names},
                         use_m3t=not args.no_m3t, depth_scale=src.depth_scale)
    viewer = Viewer(src, per, colors, args.stream_fps, args.quality)
    if args.log:
        viewer.log = open(args.log, "a", buffering=1)
    server.RequestHandlerClass = make_handler(viewer)
    threading.Thread(target=viewer.loop, daemon=True).start()
    print(f"web view on http://{args.bind}:{args.port}  (cubes: {', '.join(names)})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
