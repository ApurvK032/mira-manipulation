"""README figures: the real arm against the simulator (system ID), and the policies' training success.

Reads runs/ (not in the repository): the 100 Hz streaming system-ID recording of 2026-10-20 and the progress logs of
the grab and placement training chains. Writes light and dark PNGs to docs/media/.

    pyt -m scripts.plot_figures
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl.sysid_fit import replay_stream

SYSID = Path("runs/sysid/sysid_20261020_173305_stream100.npz")
GRAB, PLACE = "grab13_ppo", "place12_ppo"
OUT = Path("docs/media")

# Colours: the first two categorical slots of the reference palette, validated for CVD and contrast against GitHub's
# light (#ffffff) and dark (#0d1117) pages. Text and chrome use ink tokens, never the series colours.
THEMES = {
    "light": dict(real="#2a78d6", sim="#eb6834", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7"),
    "dark": dict(real="#3987e5", sim="#d95926", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835"),
}


def style(ax, c):
    ax.set_facecolor("none")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(c["axis"])
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(colors=c["muted"], labelcolor=c["muted"], labelsize=8, length=0, pad=4)
    ax.grid(axis="y", color=c["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def header(fig, c, title, subtitle):
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=12.5, fontweight="bold", color=c["ink"])
    fig.text(0.012, 0.945, subtitle, ha="left", va="top", fontsize=8.5, color=c["ink2"], linespacing=1.4)


def sysid_figure(theme):
    c = THEMES[theme]
    d = np.load(SYSID)
    t, u, y = d["t"] - d["t"][0], d["q_target"], d["q_meas"]
    sim = replay_stream(d["t"], u, y[0], float(d["stream"]))
    moving = np.where(np.ptp(u, axis=0) > 0.01)[0]
    rmse = np.degrees(np.sqrt(np.mean((sim[:, moving] - y[:, moving]) ** 2)))
    switch = float(t[np.argmax(d["phase"] == 2)])

    fig = plt.figure(figsize=(8, 6.2), dpi=200)
    fig.patch.set_alpha(0)
    header(fig, c, "The real arm against the simulator",
           f"Joint angles while the arm follows streamed targets at 100 Hz (2026-10-20), and the simulator's replay of the\n"
           f"same targets, with no fitted parameters. RMSE {rmse:.2f}° over {t[-1]:.0f} s across the {len(moving)} joints that moved.")
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.8], hspace=0.45, left=0.09, right=0.985, top=0.83, bottom=0.08)
    names = {0: "Joint 0 (base)", 1: "Joint 1 (shoulder)", 2: "Joint 2 (elbow)"}
    axes = []
    for row, j in enumerate(names):
        ax = fig.add_subplot(gs[row], sharex=axes[0] if axes else None)
        style(ax, c)
        ax.plot(t, np.degrees(y[:, j]), color=c["real"], linewidth=3.2, solid_capstyle="round", label="Real arm (measured)")
        ax.plot(t, np.degrees(sim[:, j]), color=c["sim"], linewidth=1.4, solid_capstyle="round",
                label="Simulator (replay)")
        e = np.degrees(np.sqrt(np.mean((sim[:, j] - y[:, j]) ** 2)))
        ax.set_title(names[j], loc="left", fontsize=9, color=c["ink"], pad=3)
        ax.text(1.0, 1.02, f"RMSE {e:.2f}°", transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                color=c["ink2"])
        ax.set_ylabel("deg", fontsize=8, color=c["muted"])
        ax.axvline(switch, color=c["axis"], linewidth=1)
        plt.setp(ax.get_xticklabels(), visible=False)
        axes.append(ax)
    axes[0].text(switch / 2, 1.02, "steps", transform=axes[0].get_xaxis_transform(), ha="center", va="bottom",
                 fontsize=8, color=c["muted"])
    axes[0].text((switch + t[-1]) / 2, 1.02, "sweep", transform=axes[0].get_xaxis_transform(), ha="center",
                 va="bottom", fontsize=8, color=c["muted"])
    leg = axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.22), ncol=2, frameon=False, fontsize=8.5,
                         handlelength=2.2, borderaxespad=0, columnspacing=1.6)
    for txt in leg.get_texts():
        txt.set_color(c["ink2"])

    ax = fig.add_subplot(gs[3], sharex=axes[0])
    style(ax, c)
    for j in moving:
        ax.plot(t, np.degrees(sim[:, j] - y[:, j]), color=c["muted"], linewidth=0.9, alpha=0.8)
    ax.set_title(f"Simulator − real arm, all {len(moving)} moving joints", loc="left", fontsize=9, color=c["ink"], pad=3)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([-0.5, 0, 0.5])
    ax.set_ylabel("deg", fontsize=8, color=c["muted"])
    ax.axvline(switch, color=c["axis"], linewidth=1)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("time (s)", fontsize=8, color=c["muted"])
    out = OUT / f"sysid_real_vs_sim{'_dark' if theme == 'dark' else ''}.png"
    fig.savefig(out, transparent=True)
    plt.close(fig)
    return out, rmse


def chain(leaf):
    """Runs from the first training run to `leaf`, following each config's "resume", oldest first."""
    runs, r = [], leaf
    while r:
        runs.append(r)
        res = json.load(open(f"runs/rl/{r}/config.json")).get("resume") or ""
        r = res.split("/")[2] if res else None
    return runs[::-1]


def progress(run):
    """(steps in millions, success %) per log line. A step count logged more than once (stack3 logged 20 lines at its
    resume, each from 0-1 episodes) is dropped."""
    pts = []
    for line in open(f"runs/rl/{run}/progress.log"):
        m = re.search(r"steps\s+([\d.]+)M\s+success\s+([\d.]+)%", line)
        if m:
            pts.append((float(m.group(1)), float(m.group(2))))
    pts = np.array(pts)
    steps, counts = np.unique(pts[:, 0], return_counts=True)
    return pts[np.isin(pts[:, 0], steps[counts == 1])]


def lineage(leaf):
    """(steps, success) along the weights' path: each run up to where the next run resumed, and where each run starts."""
    runs = chain(leaf)
    data = [progress(r) for r in runs]
    xs, ys, starts = [], [], {}
    for i, (r, p) in enumerate(zip(runs, data)):
        stop = data[i + 1][0, 0] if i + 1 < len(runs) else np.inf
        keep = p[p[:, 0] < stop - 1e-6]
        xs.append(keep[:, 0]), ys.append(keep[:, 1])
        starts[r] = p[0, 0]
    return np.concatenate(xs), np.concatenate(ys), starts


def training_figure(theme):
    c = THEMES[theme]
    # Where training conditions changed (each a fine-tune that resumed the previous weights)
    marks = {
        GRAB: [("grab12_ppo", 2), ("grab13_ppo", 3)],
        PLACE: [("place1_ppo", 1), ("place11_ppo", 2), ("place12_ppo", 3)],
    }
    fig = plt.figure(figsize=(8, 4.3), dpi=200)
    fig.patch.set_alpha(0)
    header(fig, c, "Training success of the two policies",
           "Share of the last 500 simulated episodes that succeeded, along each policy's chain of fine-tunes. Numbered\n"
           "lines mark the main changes to the training conditions; ② and ③ made it harder, to match the real arm.")
    notes = ["①  placement starts only; before it, full-task runs that never learned to grasp",
             "②  streamed arm model, with calibration, tracker, friction and mass errors",
             "③  the real gripper's speed"]
    for i, note in enumerate(notes):
        fig.text(0.012, 0.835 - i * 0.048, note, ha="left", va="top", fontsize=8, color=c["ink2"])
    gs = fig.add_gridspec(1, 2, wspace=0.18, left=0.07, right=0.975, top=0.6, bottom=0.13)
    for k, (leaf, title) in enumerate(((GRAB, "Grab policy (grab13)"), (PLACE, "Placement policy (place12)"))):
        x, y, starts = lineage(leaf)
        ax = fig.add_subplot(gs[k])
        style(ax, c)
        ax.plot(x, y, color=c["real"], linewidth=2, solid_joinstyle="round", solid_capstyle="round")
        ax.plot(x[-1], y[-1], "o", markersize=6, color=c["real"], markeredgecolor="none")
        ax.text(x[-1], y[-1] + 6, f"{y[-1]:.0f}%", ha="center", va="bottom", fontsize=8.5, color=c["ink"])
        for run, n in marks[leaf]:
            ax.axvline(starts[run], color=c["muted"], linewidth=1)
            ax.text(starts[run], 103, "①②③"[n - 1], ha="center", va="bottom", fontsize=9, color=c["ink2"])
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(["0", "25", "50", "75", "100%"])
        ax.set_xlim(0, x[-1] * 1.04)
        ax.set_xlabel("environment steps (millions)", fontsize=8, color=c["muted"])
        ax.set_title(title, loc="left", fontsize=9.5, color=c["ink"], pad=16)
    out = OUT / f"training_success{'_dark' if theme == 'dark' else ''}.png"
    fig.savefig(out, transparent=True)
    plt.close(fig)
    return out


def main():
    for theme in THEMES:
        out, rmse = sysid_figure(theme)
        print(out, f"RMSE {rmse:.3f} deg")
        print(training_figure(theme))


if __name__ == "__main__":
    main()
