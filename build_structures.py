#!/usr/bin/env python
"""@file build_structures.py
@brief Build the CASTEP relaxation inputs: GT, P1 (3-D ptycho) and P2 (2-D projection guess).

All three are written on one shared block of perovskite sites and one shared fixed cell, so they
differ ONLY in where the atoms sit:

  gt  the reference structure (second-principles snapshot) cut to the block.
  p1  the blind atomfind result (found_atoms.csv) snapped onto the sites. Measured positions are
      kept; a site with no found atom is filled at its column's own mean (x, y) and the ideal
      depth; found atoms that fit no site (spurious, wrong species) are dropped.
  p2  the depth-summed phase used as a 2-D image: each column's (x, y) from a Gaussian fit,
      and every atom of that column stacked at the ideal depth. No depth information at all.

P1 and P2 get the same small seeded rattle so that DFT is free to break a symmetry the start
happens to have (a z-uniform P2 has zero force along z by mirror symmetry otherwise).

Run locally (needs the atomfind data + report.json), then ship runs/<tag>/ to Blythe:
    ~/hyperspy-bundle/bin/python build_structures.py                  # default 2x2x12 block
    ~/hyperspy-bundle/bin/python build_structures.py --preset smoke   # 40-atom pipeline test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict, field

import numpy as np

import pvk

HOME = os.path.expanduser("~")


@dataclass
class Cfg:
    tag: str = "b2x2x12"
    n_cells: tuple = (2, 2, 12)            # block size in pseudo-cubic cells (x, y, z)
    centre_xy: tuple = (40.0, 20.0)        # block centred here in-plane (the scan-window centre)
    z_centre: float = 33.0                 # block centred on this depth (bulk band 10-56 A)
    rattle_A: float = 0.02                 # Gaussian rattle (per component) for p1 and p2
    seed: int = 0
    kspacing: float = 0.07                 # MP spacing, 1/A in CASTEP's 2*pi-free convention
    cutoff_eV: int = 700
    xc: str = "PBESOL"
    geom_max_iter: int = 400
    geom_force_tol: float = 0.02           # eV/A
    tol_xy: float = 0.8                    # site-assignment acceptance (A)
    tol_z: float = 1.3
    fit_r_A: float = 0.9                   # 2-D fit disc: ~half the 1.95 A column spacing. A
                                           # 0.5 A disc locks onto one half of a depth-split
                                           # Pb column (0.17 A worse than a perfect projection)
    gt: str = f"{HOME}/atomfind/atomfind/data/gt_prepared.npz"
    found: str = f"{HOME}/Desktop/atomfind_out/found_atoms.csv"
    phase: str = f"{HOME}/atomfind/atomfind/data/NL70_phase.npz"
    report: str = f"{HOME}/Desktop/atomfind_out/report.json"
    scan_window_A: float = 20.0
    scan_centre_xy: tuple = (40.0, 20.0)
    out_root: str = "runs"
    extra_param: dict = field(default_factory=dict)


PRESETS = {
    "default": {},
    "wide": dict(tag="b3x3x12", n_cells=(3, 3, 12)),                  # 540 atoms: ~5-10x cost
    # No full-beam (18-cell, 70 A) preset: the prepared GT frame dropped the entry-face O3 plane
    # (abtem orthogonalize; 301 atoms lost cell-wide) and P1 has no atoms past z ~ 65 A.
    # 40 atoms around P1's closest Ti-O contact (1.04 A, z ~ 39 A): proves CASTEP survives the
    # worst starting geometry, and exercises cell -> run -> pack -> analyse end to end.
    "smoke": dict(tag="smoke", n_cells=(2, 2, 2), z_centre=40.9, cutoff_eV=500,
                  geom_max_iter=3, kspacing=0.1),
}


# ---------------------------------------------------------------- inputs
def load_found(path):
    f = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    pos = np.stack([f["X_A"], f["Y_A"], f["Z_A"]], 1).astype(float)
    zn = np.array([pvk.ZNUM[e] for e in f["element"]])
    return pos, zn, np.asarray(f["quality"], float)


def load_alignment(cfg):
    """The recon -> GT-frame map atomfind used for found_atoms.csv (from its report.json)."""
    sys.path.insert(0, f"{HOME}/atomfind")
    from atomfind import align
    r = json.load(open(cfg.report))
    a = r["align"]
    X0 = cfg.scan_centre_xy[0] - cfg.scan_window_A / 2
    Y0 = cfg.scan_centre_xy[1] - cfg.scan_window_A / 2
    return align.Alignment(dx=r["dx"], dz=r["dz"], X0=X0, Y0=Y0, SGN=a["SGN"], OFF=a["OFF"],
                           CAL_X=a["CAL_X"], CAL_Y=a["CAL_Y"], corr_depth=a["corr_depth"],
                           mX=a["mX"], bX=a["bX"], mY=a["mY"], bY=a["bY"])


def projected_image(cfg):
    """Depth-summed phase: the 2-D image a projection measurement would give."""
    with np.load(cfg.phase) as d:
        V = d["phase"].astype(float)
    V -= np.median(V, axis=(1, 2), keepdims=True)
    return V.sum(0)


# ---------------------------------------------------------------- 2-D column fit (P2)
def gauss_fit(img, row, col, r_px):
    """Sub-pixel (row, col) of one column: 2-D Gaussian + flat background in a disc."""
    from scipy.optimize import curve_fit
    r0, c0, h = int(round(row)), int(round(col)), int(np.ceil(r_px))
    if not (h <= r0 < img.shape[0] - h and h <= c0 < img.shape[1] - h):
        raise ValueError(f"column at pixel ({row:.0f}, {col:.0f}) is within {h} px of the image "
                         f"edge -- move the block further inside the scan window")
    rr, cc = np.mgrid[r0 - h:r0 + h + 1, c0 - h:c0 + h + 1]
    disc = (rr - row) ** 2 + (cc - col) ** 2 <= r_px ** 2
    y, x, v = rr[disc].astype(float), cc[disc].astype(float), img[rr[disc], cc[disc]]

    def g(xy, A, yc, xc, s, B):
        return B + A * np.exp(-((xy[0] - yc) ** 2 + (xy[1] - xc) ** 2) / (2 * s * s))
    p0 = (np.ptp(v), row, col, r_px / 2.5, v.min())
    lo = (0, row - r_px / 2, col - r_px / 2, 1.0, -np.inf)
    hi = (np.inf, row + r_px / 2, col + r_px / 2, 2.0 * r_px, np.inf)
    try:
        p, _ = curve_fit(g, (y, x), v, p0=p0, bounds=(lo, hi), maxfev=4000)
        return p[1], p[2], True
    except RuntimeError:
        w = np.clip(v - np.median(v), 0, None)
        return (y * w).sum() / w.sum(), (x * w).sum() / w.sum(), False


def p2_columns(cfg, al, img, o_gt, a_gt, block_sites):
    """@brief Column (x, y) for every (i, j, column type) from the 2-D image alone.

    The A-column lattice is found from the image's own brightest peaks, seeded with the
    textbook 3.9 A spacing (no GT); every column is then refined by a Gaussian fit. Column
    indices are tied to the block by the A column nearest the block origin (a choice of
    WHERE to cut, not information about the structure).
    """
    from scipy.ndimage import gaussian_filter, maximum_filter
    sm = gaussian_filter(img, 2)
    pk = (sm == maximum_filter(sm, size=int(1.5 / al.dx))) & (sm > np.quantile(sm, 0.97))
    rows, cols = np.nonzero(pk)
    X, Y = al.index_to_site(rows, cols)
    peaks = np.stack([X, Y, np.zeros_like(X)], 1)
    pick = np.argmin(np.hypot(*(peaks[:, :2] - o_gt[:2]).T))
    o2, a2 = peaks[pick, :2], np.array([3.9, 3.9])
    for tol in (0.8, 0.5, 0.3):        # keep peaks on the A lattice (B-O peaks sit at 1/2,1/2)
        f = (peaks[:, :2] - o2) / a2
        on = np.hypot(*((f - np.round(f)) * a2).T) < tol
        o2, a2 = pvk.fit_lattice(peaks[on, :2], a2, peaks[pick, :2])
    # column types: A -> s 0; B/O3 share one column; O1; O2
    col_off = {0: (0.0, 0.0), 1: (0.5, 0.5), 2: (0.0, 0.5), 3: (0.5, 0.0), 4: (0.5, 0.5)}
    r_px = cfg.fit_r_A / al.dx
    xy = {}
    for (i, j, _, s) in block_sites:
        key = (i, j, 1 if s == 4 else s)
        if key in xy:
            continue
        fx, fy = col_off[key[2]]
        Xp, Yp = o2[0] + (i + fx) * a2[0], o2[1] + (j + fy) * a2[1]
        rp, cp, _ = al.site_to_index(Xp, Yp, 0.0)
        rf, cf, ok = gauss_fit(img, float(rp), float(cp), r_px)
        Xf, Yf = al.index_to_site(rf, cf)
        xy[key] = (float(Xf), float(Yf), ok)
    return xy, (o2, a2)


# ---------------------------------------------------------------- build
def build(cfg):
    rng = np.random.default_rng(cfg.seed)
    nx, ny, nz = cfg.n_cells
    d = np.load(cfg.gt)
    gpos, gz = d["pos"], d["Z"]

    # ---- shared lattice + block from the GT A sites near the window centre
    pb = gpos[gz == 82]
    near = (np.abs(pb[:, 0] - cfg.centre_xy[0]) < 12) & (np.abs(pb[:, 1] - cfg.centre_xy[1]) < 12)
    ref = pb[near][np.argmin(np.hypot(*(pb[near, :2] - cfg.centre_xy).T)
                             + np.abs(pb[near, 2] - cfg.z_centre))]
    o, a = pvk.fit_lattice(pb[near], (4.0, 3.89, 3.89), ref)
    i0 = int(np.round((cfg.centre_xy[0] - o[0]) / a[0] - nx / 2.0))
    j0 = int(np.round((cfg.centre_xy[1] - o[1]) / a[1] - ny / 2.0))
    k0 = int(np.round((cfg.z_centre - o[2]) / a[2] - nz / 2.0))
    # refit on the block's own A sites (local spacing), then the cell is n * a
    nloc = np.round((pb - o) / a)
    inb = ((nloc[:, 0] >= i0) & (nloc[:, 0] <= i0 + nx) & (nloc[:, 1] >= j0)
           & (nloc[:, 1] <= j0 + ny) & (nloc[:, 2] >= k0) & (nloc[:, 2] <= k0 + nz))
    o, a = pvk.fit_lattice(pb[inb], a, ref)
    L = np.array(cfg.n_cells) * a
    sites_ext, inside = pvk.block_sites(cfg.n_cells, (i0, j0, k0), margin=1)
    ideal_ext = pvk.ideal_positions(sites_ext, o, a)
    sites = sites_ext[inside]
    loc = sites.copy()
    loc[:, :3] -= (i0, j0, k0)
    origin = o + np.array([i0, j0, k0]) * a - 0.25 * a     # cell corner; keeps sites off the faces
    to_cell = lambda p: np.mod(p - origin, L)
    znum = pvk.SUB_Z[sites[:, 3]]
    info = {"lattice_gt": {"o": o.tolist(), "a": a.tolist()}, "cell_L": L.tolist(),
            "block_origin_idx": [i0, j0, k0], "n_sites": int(len(sites))}

    # ---- GT
    s_of, a_of = pvk.assign(gpos, gz, sites_ext, ideal_ext, cfg.tol_xy, cfg.tol_z)
    a_in = a_of[inside]
    if (a_in < 0).any():
        raise RuntimeError(f"GT block incomplete: {(a_in < 0).sum()} empty sites")
    p_gt = to_cell(gpos[a_in])
    other = np.isin(gz, (38,)) & (np.abs(gpos - (origin + L / 2)) < L / 2 + 1).all(1)
    info["gt"] = {"sr_in_block": int(other.sum())}

    # ---- P1: found atoms onto the sites
    fpos, fz, fq = load_found(cfg.found)
    s_of, a_of = pvk.assign(fpos, fz, sites_ext, ideal_ext, cfg.tol_xy, cfg.tol_z)
    a_in = a_of[inside]
    measured = a_in >= 0
    p1 = np.zeros((len(sites), 3))
    p1[measured] = to_cell(fpos[a_in[measured]])
    # p1's own lattice (blind) for the fills' depth, and a check on the shared cell
    pb_f = fpos[(fz == 82) & (s_of >= 0) & inside[np.maximum(s_of, 0)]]
    o1, a1 = pvk.fit_lattice(pb_f, a, ref)
    for m in np.where(~measured)[0]:
        i, j, k, s = sites[m]
        same_col = measured & (sites[:, 0] == i) & (sites[:, 1] == j) & (sites[:, 3] == s)
        xy_col = (p1[same_col, :2] if same_col.any() else
                  to_cell(pvk.ideal_positions(sites[m:m + 1], o1, a1))[:, :2])
        xy = pvk.min_image(xy_col - xy_col[0], L[:2]).mean(0) + xy_col[0]
        zc = to_cell(pvk.ideal_positions(sites[m:m + 1], o1, a1))[0, 2]
        p1[m] = np.mod([xy[0], xy[1], zc], L)
    blk = inside[np.maximum(s_of, 0)] & (s_of >= 0)
    near_blk = (np.abs(fpos - (origin + L / 2)) < L / 2).all(1)
    info["p1"] = {"measured": int(measured.sum()), "filled": int((~measured).sum()),
                  "filled_by_species": {pvk.SYMBOL[z]: int(((~measured) & (znum == z)).sum())
                                        for z in pvk.CASTEP_ORDER},
                  "found_in_block_volume": int(near_blk.sum()),
                  "found_dropped_in_block_volume": int((near_blk & ~blk).sum()),
                  "lattice_p1": {"o": o1.tolist(), "a": a1.tolist()}}

    # ---- P2: 2-D columns, ideal depth
    al = load_alignment(cfg)
    img = projected_image(cfg)
    xy, (o2, a2) = p2_columns(cfg, al, img, o, a, sites)
    p2 = np.zeros((len(sites), 3))
    fails = 0
    for m, (i, j, k, s) in enumerate(sites):
        X, Y, ok = xy[(i, j, 1 if s == 4 else s)]
        fails += (not ok)
        zc = to_cell(pvk.ideal_positions(sites[m:m + 1], o, a))[0, 2]
        p2[m] = np.mod([*(np.array([X, Y]) - origin[:2]), zc], L)
    info["p2"] = {"lattice_2d": {"o": o2.tolist(), "a": a2.tolist()},
                  "gauss_fit_fallbacks": int(fails)}

    # ---- the ideal projection of the GT (what a perfect 2-D measurement would give)
    p2_gt = p_gt.copy()
    for key in {(i, j, 1 if s == 4 else s) for (i, j, _, s) in loc}:
        i, j, t = key
        col = (loc[:, 0] == i) & (loc[:, 1] == j) & ((loc[:, 3] == t) | ((t == 1) & (loc[:, 3] == 4)))
        ref_xy = p_gt[col][0, :2]
        p2_gt[col, :2] = pvk.min_image(p_gt[col, :2] - ref_xy, L[:2]).mean(0) + ref_xy

    # ---- rattle the propositions identically
    rat = rng.normal(0, cfg.rattle_A, p1.shape)
    p1r, p2r = np.mod(p1 + rat, L), np.mod(p2 + rat, L)

    structs = {"gt": p_gt, "p1": p1r, "p2": p2r}
    return dict(sites=loc, znum=znum, L=L, structs=structs, measured=measured,
                p2_ideal=p2_gt, p1_unrattled=p1, p2_unrattled=p2, info=info, img=img, al=al,
                origin=origin, xy2d=xy, o=o, a=a)


# ---------------------------------------------------------------- checks + output
def kgrid(L, spacing):
    return [max(1, int(np.ceil(1.0 / (l * spacing) - 1e-9))) for l in L]


def write_param(path, cfg, comment):
    p = {"task": "geometryoptimization", "xc_functional": cfg.xc,
         "cut_off_energy": f"{cfg.cutoff_eV} eV", "spin_polarised": "false",
         "fix_occupancy": "true", "elec_energy_tol": "1e-7 eV", "max_scf_cycles": 200,
         "geom_method": "lbfgs", "geom_max_iter": cfg.geom_max_iter,
         "geom_force_tol": f"{cfg.geom_force_tol} eV/ang", "geom_energy_tol": "2e-5 eV",
         "geom_disp_tol": "1e-3 ang", "calculate_stress": "false",
         "write_cell_structure": "true", "opt_strategy": "speed", "num_dump_cycles": 0,
         "backup_interval": 3600, "iprint": 1}
    p.update(cfg.extra_param)
    with open(path, "w") as f:
        f.write(f"! {comment}\n! run_relax.slurm copies this to <seed>.param and appends run_time /"
                f" continuation\n")
        for k, v in p.items():
            f.write(f"{k:<18s}: {v}\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--preset", default="default", choices=sorted(PRESETS))
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args(argv)
    cfg = Cfg(**PRESETS[args.preset])
    if args.out_root:
        cfg.out_root = args.out_root
    B = build(cfg)
    sites, znum, L = B["sites"], B["znum"], B["L"]
    order = pvk.castep_order(sites)
    kg = kgrid(L, cfg.kspacing)
    out = os.path.join(cfg.out_root, cfg.tag)
    os.makedirs(out, exist_ok=True)

    info = B["info"]
    gt = B["structs"]["gt"]
    info["vs_gt_unrelaxed_rms_xyz_A"] = {
        "p1": pvk.rms_xyz(gt, B["p1_unrattled"], sites, L),
        "p2": pvk.rms_xyz(gt, B["p2_unrattled"], sites, L),
        "p2_ideal_projection_of_gt": pvk.rms_xyz(gt, B["p2_ideal"], sites, L),
        "ideal_lattice": pvk.rms_xyz(gt, np.mod(pvk.ideal_positions(sites, np.zeros(3), L / cfg.n_cells)
                                            + 0.25 * L / cfg.n_cells, L), sites, L)}
    deltas = {}
    for name, p in [("gt", gt), ("p1", B["p1_unrattled"]), ("p2", B["p2_unrattled"])]:
        _, dl = pvk.ti_offcentring(p, sites, cfg.n_cells, L)
        deltas[name] = dl
    info["delta_vs_gt_rms_xyz_A"] = {
        k: np.sqrt(((deltas[k] - deltas["gt"]) ** 2).mean(0)).round(4).tolist() for k in ("p1", "p2")}
    info["delta_gt_spread_xyz_A"] = deltas["gt"].std(0).round(4).tolist()
    info["min_distance_A"] = {k: round(pvk.min_distance(p, L)[0], 3) for k, p in B["structs"].items()}
    nel = {82: 22, 22: 12, 8: 6}
    info["electrons"] = int(sum(nel[int(z)] for z in znum))
    info["kpoint_grid"] = kg
    info["cfg"] = {k: v for k, v in asdict(cfg).items()}

    for name, p in B["structs"].items():
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        seed = f"{cfg.tag}_{name}"
        write_cell(d, seed, p[order], znum[order], L, kg, cfg)
        write_param(os.path.join(d, f"{seed}.param.in"), cfg, f"{seed}: fixed-cell relax, {cfg.xc}")
    np.savez(os.path.join(out, "block.npz"), sites=sites, znum=znum, L=L, order=order,
             n_cells=np.array(cfg.n_cells), measured=B["measured"],
             p2_ideal=B["p2_ideal"], p1_unrattled=B["p1_unrattled"],
             p2_unrattled=B["p2_unrattled"], **{f"start_{k}": v for k, v in B["structs"].items()})
    with open(os.path.join(out, "build_report.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps({k: info[k] for k in info if k != "cfg"}, indent=1))
    if not args.no_fig:
        import qa_figs
        qa_figs.build_figure(B, cfg, deltas, os.path.join(out, "build_qa.png"))
    print(f"\nwrote {out}/{{gt,p1,p2}}/  ({len(sites)} atoms, {info['electrons']} electrons, "
          f"k-grid {kg}, cell {np.round(L, 3).tolist()} A)")


def write_cell(d, seed, pos, zn, L, kg, cfg):
    pvk.write_cell(os.path.join(d, f"{seed}.cell"), pos, zn, L, kg,
                   comment=f"{seed}: shared fixed cell, block {cfg.n_cells} at "
                           f"{cfg.centre_xy}, depth centre {cfg.z_centre} A")


if __name__ == "__main__":
    main()
