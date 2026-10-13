"""@file qa_figs.py
@brief Figures: the build check (before CASTEP) and the relaxation comparison (after).

Colour roles are fixed across both figures: ground truth = ink, P1 (3-D ptycho) = blue,
P2 (2-D guess) = orange. Starting structures are thin dashed, relaxed ones solid.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pvk

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
COL = {"gt": INK, "p1": "#2a78d6", "p2": "#eb6834"}
LABEL = {"gt": "ground truth", "p1": "P1  3-D ptycho", "p2": "P2  2-D guess"}
SHORT = {"gt": "GT", "p1": "P1", "p2": "P2"}
DRAW_ORDER = ("p2", "p1", "gt")      # reference last, so it stays visible where P1 overlaps it
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})


def layer_profile(pos, sites, n_cells, L):
    """Ti off-centring averaged over the B sites of each depth layer k -> (nz, 3)."""
    rows, d = pvk.ti_offcentring(pos, sites, n_cells, L)
    k = sites[rows, 2]
    return np.array([d[k == kk].mean(0) for kk in range(n_cells[2])])


def _depth_panels(axes, curves, a_z, title_extra=""):
    """curves: list of (key, profile (nz,3), dashed?)."""
    for ax, c, name in zip(axes, range(3), ("x  (across the layers)", "y  (along the layers)",
                                             "z  (along the beam)")):
        for key, prof, dashed in curves:
            depth = (np.arange(len(prof)) + 0.5) * a_z
            ax.plot(depth, prof[:, c], color=COL[key],
                    lw=1.0 if dashed else (1.4 if key == "gt" else 1.8),
                    ls=(0, (4, 3)) if dashed else "-", marker=None if dashed else "o",
                    ms=3.5, label=None)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_title(f"Ti off-centre δ{name}{title_extra}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("depth in block (Å)")
    axes[0].set_ylabel("block-layer mean (Å)")


def build_figure(B, cfg, deltas, path):
    """Projected image with P2's fitted columns + the three starting depth profiles."""
    sites, L, nc = B["sites"], B["L"], cfg.n_cells
    fig = plt.figure(figsize=(11, 3.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.1, 1, 1, 1])
    ax0 = fig.add_subplot(gs[0])
    img, al = B["img"], B["al"]
    o = B["origin"]
    pad = 3.0
    xs = np.array([o[0] - pad, o[0] + L[0] + pad])
    ys = np.array([o[1] - pad, o[1] + L[1] + pad])
    r_lo, c_lo, _ = al.site_to_index(xs[0], ys[0], 0.0)
    r_hi, c_hi, _ = al.site_to_index(xs[1], ys[1], 0.0)
    r_lo, r_hi = int(max(r_lo, 0)), int(min(r_hi, img.shape[0]))
    c_lo, c_hi = int(max(c_lo, 0)), int(min(c_hi, img.shape[1]))
    X0, Y0 = al.index_to_site(r_lo, c_lo)
    X1, Y1 = al.index_to_site(r_hi, c_hi)
    ax0.imshow(img[r_lo:r_hi, c_lo:c_hi], cmap="gray", origin="lower",
               extent=(X0, X1, Y0, Y1))
    ax0.add_patch(plt.Rectangle(o[:2], L[0], L[1], fill=False, ec="#eda100", lw=1.2))
    for (i, j, t), (X, Y, ok) in B["xy2d"].items():
        if not (o[0] - 1 < X < o[0] + L[0] + 1 and o[1] - 1 < Y < o[1] + L[1] + 1):
            continue
        ax0.plot(X, Y, marker="o" if t == 0 else ("s" if t == 1 else "^"), ms=4,
                 mfc="none", mec=COL["p2"], mew=1.0)
    ax0.set_title("depth-summed phase + P2 columns", fontsize=9, color=INK, loc="left")
    ax0.set_xlabel("x (Å)"); ax0.set_ylabel("y (Å)"); ax0.grid(False)
    axes = [fig.add_subplot(gs[c]) for c in (1, 2, 3)]
    curves = [(k, layer_profile(B["structs"][k] if k == "gt" else B[f"{k}_unrattled"],
                                sites, nc, L), False) for k in DRAW_ORDER]
    _depth_panels(axes, curves, L[2] / nc[2])
    for key in ("gt", "p1", "p2"):
        axes[0].plot([], [], color=COL[key], lw=1.8, marker="o", ms=3.5, label=LABEL[key])
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{cfg.tag}: starting structures ({len(sites)} atoms; "
                 f"block {nc[0]}×{nc[1]}×{nc[2]} cells)", fontsize=10, color=INK, x=0.01,
                 ha="left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def relax_figure(S, sites, n_cells, L, metrics, path):
    """Depth profiles start (dashed) vs relaxed (solid), and RMS error vs the relaxed GT.

    S: {"gt": (start, relaxed or None), "p1": ..., "p2": ...}
    """
    fig = plt.figure(figsize=(12, 3.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.1])
    axes = [fig.add_subplot(gs[c]) for c in range(3)]
    curves = []
    for key in DRAW_ORDER:
        start, relaxed = S[key]
        curves.append((key, layer_profile(start, sites, n_cells, L), True))
        if relaxed is not None:
            curves.append((key, layer_profile(relaxed, sites, n_cells, L), False))
    _depth_panels(axes, curves, L[2] / n_cells[2])
    for key in S:
        axes[0].plot([], [], color=COL[key], lw=1.8, marker="o", ms=3.5, label=LABEL[key])
    axes[0].plot([], [], color=MUTED, lw=1.0, ls=(0, (4, 3)), label="start (dashed)")
    axes[0].legend(loc="best", fontsize=8)

    ax = fig.add_subplot(gs[3])
    names = [("p1", "start"), ("p1", "relaxed"), ("p2", "start"), ("p2", "relaxed"),
             ("gt", "start")]
    w = 0.16
    for n, (key, stage) in enumerate(names):
        rec = metrics.get(f"{key}_{stage}")
        if rec is None:
            continue
        v = rec["rms_xyz_A"]["all"]
        xs = np.arange(3) + (n - 2) * w
        ax.bar(xs, v, width=w - 0.02, color=COL[key], alpha=1.0 if stage == "relaxed" else 0.35,
               edgecolor="none")
    ax.set_xticks(range(3), ["x", "y", "z (beam)"])
    ax.set_ylabel("RMS distance to relaxed GT (Å)")
    ax.set_title("all atoms, translation removed", fontsize=9, color=INK, loc="left")
    for key, stage, alpha in (("p1", "start", .35), ("p1", "relaxed", 1), ("p2", "start", .35),
                              ("p2", "relaxed", 1), ("gt", "start", .35)):
        ax.bar([0], [0], color=COL[key], alpha=alpha, label=f"{SHORT[key]} {stage}")
    ax.legend(fontsize=7.5, ncol=1, loc="upper left")
    fig.savefig(path, dpi=150)
    plt.close(fig)
