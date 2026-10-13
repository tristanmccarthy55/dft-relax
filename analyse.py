#!/usr/bin/env python
"""@file analyse.py
@brief Score the relaxed P1 / P2 against the relaxed ground truth.

Point it at a directory holding <tag>/block.npz and <tag>/{gt,p1,p2}/<seed>.geom (the unpacked
Blythe tarballs, or runs/<tag> itself once results are copied in):

    ~/hyperspy-bundle/bin/python analyse.py ~/Desktop/dftrelax_b2x2x12 --tag b2x2x12

Reference = the RELAXED GT (same functional, same cell, same boundary conditions). Scores per
structure, before and after relaxation:
  rms_xyz_A          per-axis RMS distance to the reference, per species (translation removed)
  rms_best_shift_A   the same after the best whole-cell cyclic relabelling (so a domain wall
                     that formed one cell away along the beam is not double-counted)
  delta_*            Ti off-centring field vs the reference: RMS error per component, and
                     signal/error = spread of the reference / RMS error (>1 = resolved)
  dE_meV_per_fu      final energy minus the relaxed GT's, per formula unit (same basin ~ 0)
Missing results are skipped, so this also runs on a partial set.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

import pvk


def castep_status(path):
    if not os.path.exists(path):
        return "no .castep"
    txt = open(path, errors="replace").read()
    if re.search(r"Geometry optimi[sz]ation completed successfully", txt, re.I):
        return "converged"
    if re.search(r"Geometry optimi[sz]ation failed to converge", txt, re.I):
        return "max_iter"
    return "unfinished"


def load_relaxed(root, tag, name, start, znum, L):
    """Final .geom frame mapped back to site order, or None if absent.

    The mapping comes from matching the FIRST frame to the built start (CASTEP re-sorts ions).
    """
    seed = f"{tag}_{name}"
    geom = os.path.join(root, tag, name, f"{seed}.geom")
    if not os.path.exists(geom):
        return None
    frames = pvk.read_geom(geom)
    perm = pvk.match_to_start(frames[0], start, znum, L)
    last = frames[-1]
    pos = last["pos"][perm]
    fmax = float(np.linalg.norm(last["force"], axis=1).max()) if len(last["force"]) else np.nan
    return {"pos": pos, "energy_eV": last["energy_eV"], "steps": len(frames) - 1,
            "fmax_eV_A": fmax, "first_energy_eV": frames[0]["energy_eV"],
            "status": castep_status(os.path.join(root, tag, name, f"{seed}.castep"))}


def score(ref, test, sites, n_cells, L, d_ref):
    out = {"rms_xyz_A": pvk.rms_xyz(ref, test, sites, L)}
    best = None
    for shift, perm in pvk.cyclic_perms(sites, n_cells):
        r = pvk.rms_xyz(ref, test[perm], sites, L)["all"]
        tot = float(np.sqrt(np.sum(np.square(r))))
        if best is None or tot < best[0]:
            best = (tot, shift, r)
    out["rms_best_shift_A"] = {"all": best[2], "shift_cells": list(best[1])}
    _, d = pvk.ti_offcentring(test, sites, n_cells, L)
    err = np.sqrt(((d - d_ref) ** 2).mean(0))
    out["delta_rms_err_xyz_A"] = err.round(4).tolist()
    out["delta_signal_to_error_xyz"] = [round(float(sp / e), 2) if e > 1e-6 else None
                                        for sp, e in zip(d_ref.std(0), err)]
    out["delta_corr_xyz"] = [round(float(np.corrcoef(d[:, c], d_ref[:, c])[0, 1]), 3)
                             if d[:, c].std() > 1e-9 else None for c in range(3)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("root", help="directory containing <tag>/block.npz and <tag>/<name>/")
    ap.add_argument("--tag", default=None, help="build tag (default: the only one under root)")
    ap.add_argument("--out", default=None, help="output dir (default: root/<tag>)")
    args = ap.parse_args(argv)
    tag = args.tag or os.path.basename(os.path.dirname(
        sorted(glob.glob(os.path.join(args.root, "*", "block.npz")))[0]))
    b = np.load(os.path.join(args.root, tag, "block.npz"))
    sites, znum, L, order = b["sites"], b["znum"], b["L"], b["order"]
    n_cells = tuple(int(v) for v in b["n_cells"])
    n_fu = int(np.prod(n_cells))
    out_dir = args.out or os.path.join(args.root, tag)

    start = {k: b[f"start_{k}"] for k in ("gt", "p1", "p2")}
    relaxed = {k: load_relaxed(args.root, tag, k, start[k], znum, L) for k in ("gt", "p1", "p2")}
    if relaxed["gt"] is not None:
        ref, ref_name = relaxed["gt"]["pos"], "gt_relaxed"
    else:
        ref, ref_name = start["gt"], "gt_start"
        print("WARNING: no relaxed GT yet -- scoring against the UNRELAXED GT")
    _, d_ref = pvk.ti_offcentring(ref, sites, n_cells, L)

    metrics = {"reference": ref_name, "n_atoms": int(len(sites)), "n_fu": n_fu,
               "delta_ref_spread_xyz_A": d_ref.std(0).round(4).tolist(), "runs": {}}
    for k in ("gt", "p1", "p2"):
        metrics[f"{k}_start"] = score(ref, start[k], sites, n_cells, L, d_ref)
        r = relaxed[k]
        if r is None:
            metrics["runs"][k] = {"status": "missing"}
            continue
        metrics[f"{k}_relaxed"] = score(ref, r["pos"], sites, n_cells, L, d_ref)
        moved = pvk.min_image(r["pos"] - start[k], L)
        moved -= moved.mean(0)
        metrics["runs"][k] = {
            "status": r["status"], "steps": r["steps"], "fmax_eV_A": round(r["fmax_eV_A"], 4),
            "energy_eV": r["energy_eV"],
            "relax_rms_move_A": float(np.sqrt((moved ** 2).sum(1).mean()).round(4)),
            "relax_energy_drop_meV_per_fu": round(1e3 * (r["energy_eV"] - r["first_energy_eV"])
                                                  / n_fu, 1)}
    if relaxed["gt"] is not None:
        e0 = relaxed["gt"]["energy_eV"]
        for k in ("p1", "p2"):
            if relaxed[k] is not None:
                metrics["runs"][k]["dE_meV_per_fu_vs_gt"] = round(
                    1e3 * (relaxed[k]["energy_eV"] - e0) / n_fu, 2)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    import qa_figs
    qa_figs.relax_figure({k: (start[k], None if relaxed[k] is None else relaxed[k]["pos"])
                          for k in ("gt", "p1", "p2")}, sites, n_cells, L, metrics,
                         os.path.join(out_dir, "fig_relax.png"))

    # ---- terminal table
    print(f"\n{tag}: {len(sites)} atoms, reference = {ref_name}; "
          f"reference Ti off-centre spread x/y/z = {metrics['delta_ref_spread_xyz_A']} A\n")
    print("| structure | status | steps | RMS x / y / z (A) | best-shift RMS x / y / z | "
          "δ error x / y / z (A) | δ signal/error x / y / z | ΔE vs GT (meV/f.u.) |")
    print("|---|---|---|---|---|---|---|---|")
    for k in ("gt", "p1", "p2"):
        for stage in ("start", "relaxed"):
            m = metrics.get(f"{k}_{stage}")
            if m is None:
                continue
            run = metrics["runs"][k] if stage == "relaxed" else {}
            print(f"| {k} {stage} | {run.get('status', '')} | {run.get('steps', '')} | "
                  f"{m['rms_xyz_A']['all']} | {m['rms_best_shift_A']['all']} | "
                  f"{m['delta_rms_err_xyz_A']} | {m['delta_signal_to_error_xyz']} | "
                  f"{run.get('dE_meV_per_fu_vs_gt', '')} |")
    print(f"\nwrote {out_dir}/results.json and fig_relax.png")


if __name__ == "__main__":
    main()
