"""Cube perception: global color+depth detector, M3T tracking, consistency checks."""

import os

# M3T's GLEW is built for GLX. GLFW >= 3.4 prefers Wayland when it finds a Wayland socket (WSLg always
# has one), which gives an EGL context and makes glewInit fail. Keep every GLFW user in this process
# (M3T and the MuJoCo renderer share one GLFW) on X11, which is XWayland on WSLg.
os.environ["WAYLAND_DISPLAY"] = "no-wayland"
os.environ.setdefault("DISPLAY", ":0")
