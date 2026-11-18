#!/usr/bin/env python3
"""Figures for the Sonoro input-SNR analysis.  Run sonoro_snr.py first.

Sizes follow IEEE two-column practice: 7.16 in (two columns) or 3.5 in (one column)
wide, 9 pt text, 8 pt tick labels, bold panel labels and titles.  Colours: blue for
the runs with the right speaker at 0.7, orange for 0.6 (checked for colour-vision
deficiencies); routes are told apart by marker shape, not colour.

Usage, from the repository root:
    python snr_analysis/make_figures.py [--root <repo>] [--out <dir with the CSVs>] [--fig <dir>]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # sonoro_snr.py sits beside this file

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import sonoro_snr as S

# ---- style ---------------------------------------------------------------------------
W2, W1 = 7.16, 3.5                      # IEEE double / single column width, inches
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
C07, C06 = "#2a78d6", "#eb6834"         # right level 0.7 / 0.6  (categorical slots 1, 2)
C07_DARK = "#0d366b"                    # A1_11, the one 0.7 run that went left
BLUES = ["#86b6ef", "#2a78d6", "#0d366b"]   # ordinal ramp, light -> dark
GREYS = ["#a9a79f", "#6f6d68", "#0b0b0b"]

have = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ("Liberation Sans", "Arial", "Helvetica") if f in have), "DejaVu Sans")   # TrueType, embeds cleanly
plt.rcParams.update({
    "font.family": FONT, "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "axes.titleweight": "bold", "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold", "mathtext.cal": FONT, "mathtext.sf": FONT, "mathtext.tt": "DejaVu Sans Mono",
    "mathtext.default": "it",
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.2,
    "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def panel(ax, letter, title=None, x=-0.02, y=1.04):
    """Bold panel label and title, the same on every figure."""
    txt = f"({letter})" + (f"  {title}" if title else "")
    ax.text(x, y, txt, transform=ax.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold", color=INK)


def save(fig, figdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"{name}.{ext}"))
    plt.close(fig)


# ---- figure 1: what a sample is, and where the robot went -------------------------------
def fig_input_and_arena(root, C, figdir):
    fig = plt.figure(figsize=(W2, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0], left=0.075, right=0.985, bottom=0.155, top=0.80, wspace=0.2)
    ax = fig.add_subplot(gs[0])
    run = "08"
    ring = S.load_ring(root, run)
    d = C[C.run == run]
    kd = int(d.kdec.iloc[0])
    picks = [(5, "cycle 5 (start)"), (kd - 3, f"cycle {kd - 3} (before turn)"), (kd + 12, f"cycle {kd + 12} (after turn)")]
    for (k, lab), col in zip(picks, BLUES):
        ax.plot(S.THETA, ring["hext"][k] / S.H_SCALE, color=col, label=lab)
        r = d[d.k == k].iloc[0]
        for th, qq, name in [(r.thL, r.qL, "L"), (r.thR, r.qR, "R")]:
            if np.isfinite(th):
                ax.plot([th], [qq], "o", ms=4.5, color=col, mec="white", mew=0.8, zorder=5)
                ax.annotate(name, (th, qq), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                            fontsize=8, color=INK2)
    ax.set_xlim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_ylim(0, 1.16)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("ring angle (deg), positive to the robot's left")
    ax.set_ylabel(r"ring input  $h_i$ / 0.19")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=3, columnspacing=1.0, handlelength=1.3, fontsize=8)
    panel(ax, "a", f"Input to the ring in three cycles of A1_{run}", y=1.13)

    ax = fig.add_subplot(gs[1])
    tr = S.load_traj(root, "03")
    SL = np.nanmedian(tr[["speakerL_x", "speakerL_y"]].to_numpy(float), axis=0) / 1000
    SR = np.nanmedian(tr[["speakerR_x", "speakerR_y"]].to_numpy(float), axis=0) / 1000
    for run in [r for r in S.RUNS if r != "11"] + ["11"]:
        d = C[(C.run == run) & C.tracked]
        if len(d) == 0:
            continue
        col = C07_DARK if run == "11" else (C07 if S.RIGHT_LEVEL[run] == 0.7 else C06)
        ax.plot(d.x / 1000, d.y / 1000, color=col, lw=1.0, alpha=0.9)
    for p, name, va, dy in [(SL, "left speaker", "bottom", 0.05), (SR, "right speaker", "top", -0.05)]:
        ax.plot(*p, "s", ms=6, color=INK2, mec="white", mew=0.8)
        ax.text(p[0] + 0.05, p[1] + dy, name, ha="right", va=va, fontsize=8, color=INK2)
    ax.plot([-0.06], [-0.01], "o", ms=4.5, color=INK2, mec="white", mew=0.8, zorder=6)
    ax.text(-0.06, 0.06, "start", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.plot([], [], color=C07, label="right level 0.7")
    ax.plot([], [], color=C07_DARK, label="A1_11 (0.7)")
    ax.plot([], [], color=C06, label="right level 0.6")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, -0.01), fontsize=8, handlelength=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-0.2, 1.15)
    ax.set_ylim(-0.68, 0.68)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([-0.5, 0, 0.5])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    panel(ax, "b", "Tracked paths", x=-0.14, y=1.13)
    save(fig, figdir, "fig1_input_and_arena")


# ---- figure 2: the sample along the approach ------------------------------------------
def fig_sample(C, V, curve, figdir):
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.85), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.16, top=0.88, wspace=0.06)
    for ax, lev, col, letter in [(axs[0], 0.7, C07, "a"), (axs[1], 0.6, C06, "b")]:
        ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
        for run in S.RUNS:
            if S.RIGHT_LEVEL[run] != lev:
                continue
            d = V[V.run == run].sort_values("k")
            c = C07_DARK if run == "11" else col
            ax.plot(d.alpha, d.o, "-", color=c, lw=0.7, alpha=0.5 if run != "11" else 0.9, zorder=2)
            ax.plot(d.alpha, d.o, "o", color=c, ms=2.3, alpha=0.65 if run != "11" else 0.95, mec="none", zorder=3)
            last = d.iloc[-1]
            ax.plot([last.alpha], [last.o], "o", ms=5, color=c, mec="white", mew=0.8, zorder=6)
            for _, r in C[(C.run == run) & C.dropout].iterrows():
                a = np.interp(r.k, d.k, d.alpha)
                ax.annotate("", (a, 0.445), (a, 0.395), arrowprops=dict(arrowstyle="-|>", color=c, lw=0.9, mutation_scale=7))
        m = curve[curve.group == ("0.7R" if lev == 0.7 else "0.6L")]
        ax.plot(m.alpha, m.dqB, "-", color=col, lw=2.2, zorder=5)
        ax.set_xlim(46, 114)
        ax.set_xticks(np.arange(50, 111, 10))
        ax.set_ylim(-0.27, 0.46)
        ax.set_xlabel(r"separation of the two lobes, $\alpha$ (deg)")
        panel(ax, letter, f"Right level {lev}, left level 1")
    axs[0].set_ylabel(r"sample  $o_n$ = $\hat q_{L,n}$ − $\hat q_{R,n}$")
    a = axs[0]
    a.text(103.5, -0.185, "four runs\nturn right", ha="left", va="center", fontsize=8, color=INK2)
    a.text(99.5, 0.235, "A1_11\nturns left", ha="left", va="center", fontsize=8, color=INK2)
    a.text(56.5, 0.42, "A1_11: right lobe missing in two cycles", ha="left", va="center", fontsize=8, color=INK2)
    b = axs[1]
    b.text(101.5, 0.215, "five runs\nturn left", ha="left", va="center", fontsize=8, color=INK2)
    b.text(60, 0.02, "left lobe taller", ha="left", va="bottom", fontsize=8, color=MUTED)
    b.text(60, -0.02, "right lobe taller", ha="left", va="top", fontsize=8, color=MUTED)
    save(fig, figdir, "fig2_sample_along_approach")


# ---- figure 4: geometry ----------------------------------------------------------------
def fig_geometry(V, par, figdir):
    beta = par.value.to_numpy()
    T = V[V.tracked].copy()
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.7))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.165, top=0.88, wspace=0.25)
    ax = axs[0]
    ax.axhline(0, color=MUTED, lw=0.8)
    for run in S.RUNS:
        d = T[T.run == run].sort_values("k")
        if len(d) == 0:
            continue
        c = C07_DARK if run == "11" else (C07 if S.RIGHT_LEVEL[run] == 0.7 else C06)
        ax.plot(d.alpha, 100 * d.ldr, "-", color=c, lw=1.0, alpha=0.9)
    ax.plot([], [], color=C07, label="right level 0.7")
    ax.plot([], [], color=C07_DARK, label="A1_11 (0.7)")
    ax.plot([], [], color=C06, label="right level 0.6")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.97))
    ax.set_xlim(46, 112)
    ax.set_xticks(np.arange(50, 111, 10))
    ax.set_ylim(-2.4, 7.2)
    ax.set_xlabel(r"separation of the two lobes, $\alpha$ (deg)")
    ax.set_ylabel(r"100 ln($d_L$/$d_R$)")
    ax.text(47, -0.25, "equidistant", ha="left", va="top", fontsize=8, color=MUTED)
    ax.text(111, 4.9, "right speaker\nnearer", ha="right", va="center", fontsize=8, color=MUTED)
    ax.text(111, -1.35, "left speaker\nnearer", ha="right", va="center", fontsize=8, color=MUTED)
    panel(ax, "a", "Distance ratio along the approach", x=-0.09)

    ax = axs[1]
    z = (T.alpha - S.ALPHA_REF) / S.ALPHA_SCALE
    base = np.where(T.level_R == 0.7, beta[0], beta[1]) + beta[2] * z + beta[3] * z ** 2 + beta[4] * z ** 3
    part = T.ell - base                       # what is left for the distance ratio to explain
    ax.axhline(0, color=MUTED, lw=0.8)
    for (lo, hi), col, lab in zip([(46, 60), (60, 80), (80, 113)], GREYS, ["below 60°", "60° to 80°", "above 80°"]):
        m = (T.alpha > lo) & (T.alpha <= hi)
        g = beta[5] + beta[6] * (T.alpha[m].mean() - S.ALPHA_REF) / S.ALPHA_SCALE
        ax.plot(100 * T.ldr[m], part[m], "o", ms=2.6, color=col, mec="none", alpha=0.7)
        xx = np.array([T.ldr[m].min(), T.ldr[m].max()])
        ax.plot(100 * xx, g * xx, "-", color="white", lw=3.0, zorder=4)
        ax.plot(100 * xx, g * xx, "-", color=col, lw=1.6, zorder=5, label=rf"$\alpha$ {lab}:  $\gamma$ = {g:.1f}")
    ax.set_xlim(-2.2, 7.4)
    ax.set_ylim(-0.3, 0.52)
    ax.set_xlabel(r"100 ln($d_L$/$d_R$)")
    ax.set_ylabel(r"ln($\hat q_R$/$\hat q_L$), other terms removed")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.99), handlelength=1.4)
    panel(ax, "b", "Sensitivity to the distance ratio", x=-0.11)
    save(fig, figdir, "fig4_geometry")


# ---- figure 5: the noise ---------------------------------------------------------------
def fig_noise(V, curve, diag, figdir):
    fig = plt.figure(figsize=(W2, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 1.0], left=0.078, right=0.98, bottom=0.17, top=0.87, wspace=0.37)
    ax = fig.add_subplot(gs[0])
    for grp, col in [("0.7R", C07), ("0.6L", C06)]:
        c = curve[curve.group == grp]
        ax.plot(c.alpha, c.sigA, "-", color=col, lw=1.2)
        ax.plot(c.alpha, c.sigA, "s", color="white", mec=col, mew=1.1, ms=4.5)
        ax.plot(c.alpha, c.sigB, "-", color=col, lw=1.2)
        ax.plot(c.alpha, c.sigB, "o", color=col, mec="white", mew=0.7, ms=5)
    ax.plot([], [], "s-", color=INK2, mfc="white", mec=INK2, ms=4.5, lw=1.2, label="Route A")
    ax.plot([], [], "o-", color=INK2, mec="white", ms=5, lw=1.2, label="Route B")
    ax.plot([], [], "-", color=C07, lw=2, label="right 0.7")
    ax.plot([], [], "-", color=C06, lw=2, label="right 0.6")
    ax.legend(loc="upper center", ncol=2, columnspacing=1.2, bbox_to_anchor=(0.5, 1.0))
    ax.set_xlim(46, 100)
    ax.set_xticks(np.arange(50, 101, 10))
    ax.set_ylim(0, 0.085)
    ax.set_xlabel(r"separation of the two lobes, $\alpha$ (deg)")
    ax.set_ylabel(r"noise of the sample, $\sigma_o$")
    panel(ax, "a", "Noise along the approach", x=-0.16)

    ax = fig.add_subplot(gs[1])
    r = V.res_ell.dropna().to_numpy()
    u = (r - np.median(r)) / diag["robust_sd"]
    ax.hist(np.clip(u, -7, 7), bins=np.arange(-7.25, 7.26, 0.5), color=MUTED, alpha=0.55, edgecolor="white", linewidth=0.6)
    xx = np.linspace(-7, 7, 400)
    ax.plot(xx, len(u) * 0.5 * np.exp(-0.5 * xx ** 2) / np.sqrt(2 * np.pi), color=INK, lw=1.2)
    ax.set_xlim(-7.3, 7.3)
    ax.set_xticks([-6, -3, 0, 3, 6])
    ax.set_xlabel("residual / robust s.d.")
    ax.set_ylabel("number of cycles")
    nout = int(np.sum(np.abs(u) > 3))
    ax.text(0.98, 0.62, f"{nout} of {len(u)}\nbeyond\n3 s.d.", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
    ax.text(0.03, 0.62, "normal\ncurve", transform=ax.transAxes, ha="left", va="top", fontsize=8, color=INK2)
    panel(ax, "b", "Residuals of Route A", x=-0.24)

    ax = fig.add_subplot(gs[2])
    d = V[(V.session == 3) & np.isfinite(V.detr_ell)]
    ph = (d.tod % 2.0).to_numpy()
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.plot(ph, d.detr_ell, "o", ms=2.6, color=C07, mec="none", alpha=0.7)
    nb = 8
    b = np.minimum((ph / 2.0 * nb).astype(int), nb - 1)
    med = [np.median(d.detr_ell.to_numpy()[b == j]) if np.any(b == j) else np.nan for j in range(nb)]
    ax.step(np.r_[0, (np.arange(nb) + 1) * 2.0 / nb], np.r_[med[0], med], where="pre", color=INK, lw=1.2)
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.25, 0.2)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xticklabels(["0", "0.5", "1", "1.5", "2"])
    ax.set_xlabel("clock time modulo 2 s (s)")
    ax.set_ylabel(r"detrended ln($\hat q_R$/$\hat q_L$)")
    ax.text(0.97, 0.04, "line: median per\nquarter second", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK2)
    panel(ax, "c", "Chirp timing, A1_08 to A1_11", x=-0.3)
    save(fig, figdir, "fig5_noise")


# ---- figure 3: the SNR -----------------------------------------------------------------
def fig_snr(curve, figdir):
    fig, ax = plt.subplots(figsize=(W1, 2.75))
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.165, top=0.9)
    ax.axhspan(-1, 1, color=MUTED, alpha=0.22, lw=0, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    for grp, col in [("0.7R", C07), ("0.6L", C06)]:
        c = curve[curve.group == grp]
        ax.plot(c.alpha, c.DA, "-", color=col, lw=1.2)
        ax.plot(c.alpha, c.DA, "s", color="white", mec=col, mew=1.1, ms=4.5)
        ax.plot(c.alpha, c.DB, "-", color=col, lw=1.2)
        ax.plot(c.alpha, c.DB, "o", color=col, mec="white", mew=0.7, ms=5)
    ax.plot([], [], "s-", color=INK2, mfc="white", mec=INK2, ms=4.5, lw=1.2, label="Route A")
    ax.plot([], [], "o-", color=INK2, mec="white", ms=5, lw=1.2, label="Route B")
    ax.plot([], [], "-", color=C06, lw=2, label="right level 0.6")
    ax.plot([], [], "-", color=C07, lw=2, label="right level 0.7")
    ax.legend(loc="upper left", ncol=2, columnspacing=1.0, bbox_to_anchor=(0.0, 1.0))
    ax.text(47.5, -1.6, "ICRA 2027 grid, |D| ≤ 1", ha="left", va="top", fontsize=8, color=INK2)
    ax.set_xlim(46, 100)
    ax.set_xticks(np.arange(50, 101, 10))
    ax.set_ylim(-13, 31)
    ax.set_yticks([-10, 0, 10, 20, 30])
    ax.set_xlabel(r"separation of the two lobes, $\alpha$ (deg)")
    ax.set_ylabel(r"per-sample SNR,  $D$")
    ax.text(-0.15, 1.03, "SNR along the approach", transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
            fontweight="bold", color=INK)
    save(fig, figdir, "fig3_snr_along_approach")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.dirname(here))
    ap.add_argument("--out", default=os.path.join(here, "out"))
    ap.add_argument("--fig", default=os.path.join(here, "fig"))
    a = ap.parse_args()
    os.makedirs(a.fig, exist_ok=True)
    C = pd.read_csv(os.path.join(a.out, "cycles.csv"), dtype={"run": str})
    V = pd.read_csv(os.path.join(a.out, "precommit_samples.csv"), dtype={"run": str})
    V = V.merge(C[["run", "k", "tracked", "level_R"]], on=["run", "k"])
    par = pd.read_csv(os.path.join(a.out, "route_a_parameters.csv"))
    summary = json.load(open(os.path.join(a.out, "summary.json")))
    f = os.path.join(a.out, "snr_along_approach.csv")
    if os.path.exists(f):
        curve = pd.read_csv(f, dtype={"group": str})
    else:
        curve = S.approach_curve(V.assign(k_orig=V.k), summary["cycle_period_s"])
    fig_input_and_arena(a.root, C, a.fig)                # fig. 1
    fig_sample(C, V, curve, a.fig)                       # fig. 2
    fig_snr(curve, a.fig)                                # fig. 3
    fig_geometry(V, par, a.fig)                          # fig. 4
    fig_noise(V, curve, summary["diagnostics"], a.fig)   # fig. 5
    print("figures written to", a.fig)


if __name__ == "__main__":
    main()
