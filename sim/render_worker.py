"""Render scenario frames in a separate process.

M3T and the MuJoCo renderer both use GLFW, and M3T calls glfwTerminate after generating its models,
which would kill MuJoCo's GL context. A separate process keeps them apart; it also renders ahead
while the parent runs perception, like a camera streaming frames.

Preferred: a separate "rlwx-render" conda env with Mesa EGL, rendering headless (no X display needed;
on WSL it lands on the GPU through D3D12). Fallback: a spawned child of this interpreter using GLFW,
which needs a working X display (WSLg).
"""

import multiprocessing as mp
import os
import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RENDER_PY = Path(os.environ.get("RLWX_RENDER_PYTHON", Path.home() / "micromamba/envs/rlwx-render/bin/python"))


def _render(scenario_name: str, seed: int, fps: float, emit):
    from sim.scenarios import make_scenarios
    from sim.scene import BenchScene

    scenario = {s.name: s for s in make_scenarios(seed)}[scenario_name]
    scene = BenchScene(seed=seed)
    n = int(round(scenario.duration * fps))
    for i in range(n):
        emit(scene.render(scenario, i / fps))
    emit(None)


def _worker(scenario_name: str, seed: int, fps: float, q, cpus=None):
    if cpus:
        os.sched_setaffinity(0, cpus)  # keep the software renderer off the perception cores
        os.environ.setdefault("LP_NUM_THREADS", str(len(cpus)))
    _render(scenario_name, seed, fps, q.put)


def _headless_env():
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY", "PYTHONPATH", "MUJOCO_GL")}
    prefix = RENDER_PY.parent.parent
    env.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", EGL_PLATFORM="surfaceless", PYTHONPATH=str(ROOT),
               __EGL_VENDOR_LIBRARY_FILENAMES=str(prefix / "share/glvnd/egl_vendor.d/50_mesa.json"))
    return env


def render_frames(scenario_name: str, seed: int = 0, fps: float = 60.0, prefetch: int = 24, cpus=None):
    """Yield sim.scene.Frame objects for one scenario, rendered in a separate process."""
    if RENDER_PY.exists():
        cmd = [str(RENDER_PY), "-m", "sim.render_worker", scenario_name, str(seed), str(fps)]
        if cpus:
            cmd.append(",".join(str(c) for c in sorted(cpus)))
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, cwd=ROOT, env=_headless_env(), bufsize=1 << 22)
        try:
            while (frame := pickle.load(p.stdout)) is not None:
                yield frame
        finally:
            p.kill()
            p.wait()
        return
    ctx = mp.get_context("spawn")
    q = ctx.Queue(prefetch)
    p = ctx.Process(target=_worker, args=(scenario_name, seed, fps, q, cpus), daemon=True)
    p.start()
    try:
        while (frame := q.get()) is not None:
            yield frame
    finally:
        p.terminate()
        p.join()


if __name__ == "__main__":
    # headless render server: python -m sim.render_worker <scenario> <seed> <fps> [cpu,list]
    name, seed, fps = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    if len(sys.argv) > 4:
        os.sched_setaffinity(0, {int(c) for c in sys.argv[4].split(",")})
    out = sys.stdout.buffer

    def emit(frame):
        pickle.dump(frame, out, protocol=5)
        out.flush()

    _render(name, seed, fps, emit)
    os._exit(0)  # skip MuJoCo's EGL teardown, which raises harmless errors at interpreter exit
