"""Run the perception pipeline on every simulated scenario and check the Phase 1 pass bars.

Usage:  python -m scripts.eval_sim [--scenarios a,b] [--no-m3t] [--gif]
Writes docs/poc/results.json, docs/poc/results.md, plots and overlay GIFs.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")  # M3T's OpenMP pool, read when pym3t loads
# small matrices only: one BLAS thread, and OpenMP threads that sleep instead of spinning, so the two
# pools do not fight over cores (that showed up as 50 ms stalls every time the table plane was refit)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import argparse
import gzip
import json
import time
from pathlib import Path

import cv2
import numpy as np

from perception.camera import Intrinsics
from perception.detector import DEFAULT_COLORS
from perception.geometry import rot_error_deg
from perception.overlay import draw_result
from perception.pipeline import CubePerception
from sim.render_worker import render_frames
from sim.scenarios import make_scenarios
from sim.scene import FOVY, HEIGHT, WIDTH

OUT = Path(__file__).resolve().parents[1] / "docs" / "poc"
RUNS = Path(__file__).resolve().parents[1] / "runs"  # per-frame logs (not committed)
FPS = 60.0
WRONG_POS_MM, WRONG_ROT_DEG = 10.0, 10.0
GOOD_POS_MM, GOOD_ROT_DEG = 5.0, 5.0
GIF_SCENARIOS = ("hand_pick_place", "fast_snatch", "teleport", "stack_knock", "clutter_light")


def run_scenario(per, scenario, cpus, gif_path=None, seed=0):
    names = per.names
    rows = []  # one per frame
    gif = []
    for i, f in enumerate(render_frames(scenario.name, seed=seed, fps=FPS, cpus=cpus)):
        r = per.process(f.color, f.depth)
        row = {"i": i, "t": f.t, "timing": r.timing_ms, "cubes": {}}
        for n in names:
            e, g = r.cubes[n], f.gt.get(n)
            c = {"state": e.state, "source": e.source, "vis": f.visibility.get(n, 0.0), "gt": g is not None,
                 "chk": [e.check.visible, e.check.color, e.check.depth, e.check.leak]}
            if g is not None and e.T is not None:
                c["pos_err"] = 1e3 * float(np.linalg.norm(e.T[:3, 3] - g[:3, 3]))
                c["rot_err"] = rot_error_deg(g[:3, :3], e.T[:3, :3])
            if g is not None:
                c["gt_pos"] = g[:3, 3].tolist()
            if e.T is not None:
                c["est_pos"] = e.T[:3, 3].tolist()
            row["cubes"][n] = c
        rows.append(row)
        if gif_path is not None and i % 2 == 0:
            im = draw_result(f.color, per.K, r, DEFAULT_COLORS, gt={k: v for k, v in f.gt.items() if v is not None})
            cv2.putText(im, f"{scenario.name}  t={f.t:5.2f}s", (WIDTH - 260, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            gif.append(cv2.cvtColor(cv2.resize(im, (WIDTH // 2, HEIGHT // 2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    if gif_path is not None and gif:
        import imageio.v2 as imageio

        imageio.mimsave(gif_path, gif, duration=1 / 30, loop=0)
    return rows


def first_good_frame(rows, n, start):
    for r in rows[start:]:
        c = r["cubes"][n]
        if c["state"] == "TRACKING" and c.get("pos_err", 1e9) < GOOD_POS_MM and c.get("rot_err", 1e9) < GOOD_ROT_DEG:
            return r["i"]
    return None


def analyze(scenario, rows):
    names = list(rows[0]["cubes"])
    res = {"frames": len(rows), "cubes": {}, "events": []}
    wrong = []
    for n in names:
        cs = [r["cubes"][n] for r in rows if r["cubes"][n]["gt"]]
        if not cs:
            continue
        trk = [c for c in cs if c["state"] == "TRACKING"]
        pe = np.array([c["pos_err"] for c in trk]) if trk else np.zeros(0)
        re_ = np.array([c["rot_err"] for c in trk]) if trk else np.zeros(0)
        clear = [c for c in cs if c["vis"] >= 0.6]
        st = {}
        for c in cs:
            st[c["state"]] = st.get(c["state"], 0) + 1
        d = {"frames": len(cs), "states": st,
             "tracking_when_clearly_visible": (sum(c["state"] == "TRACKING" for c in clear) / len(clear)) if clear else None}
        if len(pe):
            d.update(pos_med=float(np.median(pe)), pos_p95=float(np.percentile(pe, 95)), pos_max=float(pe.max()),
                     rot_med=float(np.median(re_)), rot_p95=float(np.percentile(re_, 95)), rot_max=float(re_.max()))
        res["cubes"][n] = d
        for r in rows:
            c = r["cubes"][n]
            if c["state"] == "TRACKING" and c["gt"] and (c["pos_err"] > WRONG_POS_MM or c["rot_err"] > WRONG_ROT_DEG):
                wrong.append({"frame": r["i"], "cube": n, "pos_err": c["pos_err"], "rot_err": c["rot_err"], "vis": c["vis"],
                              "source": c["source"]})
    res["wrong_tracking_frames"] = wrong

    for ev in scenario.events:
        n = ev.cube
        e = {"kind": ev.kind, "cube": n, "t": ev.t}
        if ev.kind in ("release", "teleport", "knock"):
            if ev.kind == "release":
                # from the first frame after the cube is put down in which at least half of it is visible
                f0 = int(np.ceil(ev.detail["t_place"] * FPS))
                while f0 < len(rows) and rows[f0]["cubes"][n]["vis"] < 0.5:
                    f0 += 1
                e["peak_speed"] = ev.detail["peak_speed"]
            else:
                f0 = int(np.ceil(ev.t * FPS))
            if f0 >= len(rows):
                continue
            fg = first_good_frame(rows, n, f0)
            e["reacquire_frames"] = None if fg is None else fg - f0
            e["reacquire_ms"] = None if fg is None else 1e3 * (fg - f0) / FPS
        elif ev.kind == "static":
            f0, f1 = int(ev.t * FPS), min(len(rows), int(ev.t_end * FPS))
            P = np.array([rows[k]["cubes"][n]["est_pos"] for k in range(f0, f1)
                          if rows[k]["cubes"][n]["state"] == "TRACKING"])
            if len(P) > 10:
                e["jitter_mm"] = float(1e3 * np.linalg.norm(P.std(axis=0)))
                e["tracking_fraction"] = len(P) / (f1 - f0)
        elif ev.kind == "slide":
            fa, fb = int(ev.t * FPS) - 3, min(len(rows) - 1, int((ev.t_end + 0.3) * FPS))
            ca, cb = rows[fa]["cubes"][n], rows[fb]["cubes"][n]
            if "est_pos" in ca and "est_pos" in cb:
                est = 1e3 * (np.array(cb["est_pos"]) - np.array(ca["est_pos"]))
                gt = 1e3 * (np.array(cb["gt_pos"]) - np.array(ca["gt_pos"]))
                e["commanded_mm"] = float(np.linalg.norm(gt))
                e["measured_mm"] = float(np.linalg.norm(est))
                e["displacement_error_mm"] = float(np.linalg.norm(est - gt))
                e["dyaw_deg"] = ev.detail["dyaw"]
            # lag: frame where the estimate has covered half the move vs the frame the cube has
            if "est_pos" in ca and "est_pos" in cb and np.linalg.norm(gt) > 1.0:
                p0e, p0g = np.array(ca["est_pos"]), np.array(ca["gt_pos"])
                half = 0.5e-3 * np.linalg.norm(gt)
                fg_half = next((r["i"] for r in rows[fa:fb + 1]
                                if np.linalg.norm(np.array(r["cubes"][n]["gt_pos"]) - p0g) >= half), None)
                fe_half = next((r["i"] for r in rows[fa:fb + 1] if "est_pos" in r["cubes"][n]
                                and np.linalg.norm(np.array(r["cubes"][n]["est_pos"]) - p0e) >= half), None)
                if fg_half is not None and fe_half is not None:
                    e["motion_lag_frames"] = fe_half - fg_half
        res["events"].append(e)

    tot = np.array([r["timing"]["total"] for r in rows])
    det = np.array([r["timing"]["detect"] for r in rows])
    trk = np.array([r["timing"]["track"] for r in rows])
    chk = np.array([r["timing"]["check"] for r in rows])
    res["timing_ms"] = {k: {"p50": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(v.max())}
                        for k, v in (("total", tot[5:]), ("detect", det[5:]), ("track", trk[5:]), ("check", chk[5:]))}
    return res


def plot_errors(name, rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in rows[0]["cubes"] if any(r["cubes"][n]["gt"] for r in rows)]
    fig, axes = plt.subplots(len(names), 1, figsize=(11, 1.8 * len(names)), sharex=True)
    axes = np.atleast_1d(axes)
    colors = {"TRACKING": "#2e7d32", "OCCLUDED": "#f9a825", "LOST": "#c62828"}
    for ax, n in zip(axes, names):
        t = np.array([r["t"] for r in rows])
        pe = np.array([r["cubes"][n].get("pos_err", np.nan) if r["cubes"][n]["state"] == "TRACKING" else np.nan for r in rows])
        vis = np.array([r["cubes"][n]["vis"] for r in rows])
        for s, col in colors.items():
            m = np.array([r["cubes"][n]["state"] == s for r in rows])
            ax.fill_between(t, 0, 6, where=m, color=col, alpha=0.15, step="mid", linewidth=0)
        ax.plot(t, pe, color="#1a237e", lw=1, label="position error (mm), TRACKING only")
        ax.plot(t, vis * 5, color="#555", lw=0.8, ls=":", label="visible fraction x5")
        ax.set_ylim(0, 6)
        ax.set_ylabel(n)
    axes[0].legend(loc="upper right", fontsize=7)
    axes[0].set_title(f"{name}: green = TRACKING, amber = OCCLUDED, red = LOST")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--no-m3t", action="store_true")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="new seed = held-out scenarios (random approach directions, spots, stress events)")
    args = ap.parse_args()

    ncpu = os.cpu_count() or 8
    perception_cpus = set(range(0, min(8, ncpu // 2)))
    render_cpus = set(range(min(8, ncpu // 2), ncpu))
    os.sched_setaffinity(0, perception_cpus)
    cv2.setNumThreads(len(perception_cpus))

    OUT.mkdir(parents=True, exist_ok=True)
    K = Intrinsics.from_fov(WIDTH, HEIGHT, FOVY)
    scenarios = make_scenarios(args.seed)
    if args.scenarios:
        keep = set(args.scenarios.split(","))
        scenarios = [s for s in scenarios if s.name in keep]
    tag = ("detector_only" if args.no_m3t else "m3t") + (f"_seed{args.seed}" if args.seed else "")
    results = {"config": {"pipeline": tag, "fps": FPS, "resolution": [WIDTH, HEIGHT], "perception_cpus": len(perception_cpus)}}
    for s in scenarios:
        per = CubePerception(K, ["red", "blue", "green", "yellow"], use_m3t=not args.no_m3t)
        t0 = time.perf_counter()
        gif = OUT / f"{s.name}.gif" if args.gif and s.name in GIF_SCENARIOS else None
        rows = run_scenario(per, s, render_cpus, gif, args.seed)
        RUNS.mkdir(parents=True, exist_ok=True)
        with gzip.open(RUNS / f"rows_{s.name}_{tag}.json.gz", "wt") as fh:
            json.dump(rows, fh)
        res = analyze(s, rows)
        res["wall_s"] = time.perf_counter() - t0
        results[s.name] = res
        plot_errors(s.name, rows, OUT / f"{s.name}_{tag}.png")
        tm = res["timing_ms"]["total"]
        print(f"{s.name:16s} frames={res['frames']:5d} wrong={len(res['wrong_tracking_frames']):3d} "
              f"total p50={tm['p50']:.1f} p95={tm['p95']:.1f} ms", flush=True)
        for n, d in res["cubes"].items():
            if "pos_med" in d:
                print(f"    {n:6s} pos med/p95/max {d['pos_med']:.2f}/{d['pos_p95']:.2f}/{d['pos_max']:.2f} mm  "
                      f"rot {d['rot_med']:.2f}/{d['rot_p95']:.2f}/{d['rot_max']:.2f} deg  states {d['states']}", flush=True)
        for e in res["events"]:
            print("    event", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in e.items()}, flush=True)
    (OUT / f"results_{tag}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
