"""Tune the HSV ranges for your cubes under your lab lighting, by clicking on them.

Usage:  python -m scripts.tune_colors [--source realsense|recordings/x]
Keys: 1 red, 2 blue, 3 green, 4 yellow (select color) | left-click a cube face to add samples
      (click all visible faces, in light and shadow) | c clear samples | w write assets/colors.yaml | q quit
The right half shows the mask for the selected color; it should cover the cube and nothing else.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from perception.detector import DEFAULT_COLORS, ColorSpec, load_colors
from perception.sources import open_source

OUT = Path(__file__).resolve().parents[1] / "assets" / "colors.yaml"
KEYS = {ord("1"): "red", ord("2"): "blue", ord("3"): "green", ord("4"): "yellow"}


def ranges_from_samples(hsv: np.ndarray):
    h, s, v = hsv[:, 0].astype(int), hsv[:, 1], hsv[:, 2]
    wraps = (h < 15).any() and (h > 165).any()
    hs = (h + 90) % 180 if wraps else h
    lo_h, hi_h = np.percentile(hs, 1) - 4, np.percentile(hs, 99) + 4
    lo_s, lo_v = max(40, np.percentile(s, 1) - 30), max(25, np.percentile(v, 1) - 40)
    if not wraps:
        return [((int(max(0, lo_h)), int(lo_s), int(lo_v)), (int(min(179, hi_h)), 255, 255))]
    lo_h, hi_h = (lo_h - 90) % 180, (hi_h - 90) % 180  # back to the real hue: range wraps through 0
    return [((0, int(lo_s), int(lo_v)), (int(hi_h), 255, 255)), ((int(lo_h), int(lo_s), int(lo_v)), (179, 255, 255))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="realsense")
    args = ap.parse_args()
    src = open_source(args.source)
    colors = load_colors()
    samples = {n: [] for n in DEFAULT_COLORS}
    sel = "blue"
    state = {"hsv": None}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and state["hsv"] is not None and x < state["hsv"].shape[1]:
            patch = state["hsv"][max(0, y - 3): y + 4, max(0, x - 3): x + 4].reshape(-1, 3)
            samples[sel].extend(patch.tolist())
            colors[sel] = ColorSpec(sel, ranges_from_samples(np.array(samples[sel])), DEFAULT_COLORS[sel].bgr)
            print(sel, colors[sel].hsv_ranges)

    cv2.namedWindow("tune")
    cv2.setMouseCallback("tune", on_mouse)
    while True:
        f = src.read()
        if f is None:
            break
        hsv = cv2.cvtColor(f.color, cv2.COLOR_BGR2HSV)
        state["hsv"] = hsv
        m = None
        for lo, hi in colors[sel].hsv_ranges:
            r = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            m = r if m is None else m | r
        view = np.hstack([f.color, cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)])
        cv2.putText(view, f"selected: {sel}  samples: {len(samples[sel])}  (1-4 select, click, c clear, w write)",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("tune", view)
        k = cv2.waitKey(1) & 0xFF
        if k in KEYS:
            sel = KEYS[k]
        elif k == ord("c"):
            samples[sel] = []
            colors[sel] = DEFAULT_COLORS[sel]
        elif k == ord("w"):
            data = {n: {"hsv_ranges": [[list(lo), list(hi)] for lo, hi in c.hsv_ranges], "bgr": list(c.bgr)}
                    for n, c in colors.items()}
            OUT.write_text(yaml.safe_dump(data, sort_keys=False))
            print("wrote", OUT)
        elif k == ord("q"):
            break
    src.close()


if __name__ == "__main__":
    main()
