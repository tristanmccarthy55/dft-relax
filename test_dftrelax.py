"""@file test_dftrelax.py
@brief Synthetic checks of the site machinery, the CASTEP I/O and the analysis path.

    ~/hyperspy-bundle/bin/python test_dftrelax.py        (or pytest test_dftrelax.py)

The last test needs a built runs/smoke (build_structures.py --preset smoke) and is skipped
without it.
"""
from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np

import pvk

HERE = os.path.dirname(os.path.abspath(__file__))


def _lattice(n=(2, 2, 3), a=(4.1, 3.9, 3.9), seed=1, jitter=0.05):
    sites, _ = pvk.block_sites(n)
    L = np.array(n) * a
    pos = pvk.ideal_positions(sites, np.zeros(3), np.array(a)) + 0.3
    pos += np.random.default_rng(seed).normal(0, jitter, pos.shape)
    return sites, L, pos


def test_assign_recovers_sites_and_flags_gaps():
    sites, L, pos = _lattice()
    ideal = pvk.ideal_positions(sites, np.full(3, 0.3), np.array([4.1, 3.9, 3.9]))
    zn = pvk.SUB_Z[sites[:, 3]].copy()
    keep = np.ones(len(pos), bool)
    keep[[5, 40]] = False                                  # two missing atoms
    extra = pos[7] + [0, 0, 0.3]                           # a duplicate claiming site 7
    wrong = pos[sites[:, 3] == 1][0] + [0, 0, 1.95]        # a Ti put on an O3 depth
    P = np.vstack([pos[keep], extra, wrong])
    Z = np.concatenate([zn[keep], [zn[7]], [22]])
    s_of, a_of = pvk.assign(P, Z, sites, ideal, 0.8, 1.3)
    assert (a_of < 0).sum() == 2 and set(np.where(a_of < 0)[0]) == {5, 40}
    assert s_of[-2] == -1 and s_of[-1] == -1               # duplicate and misplaced Ti rejected
    ok = a_of >= 0
    assert np.allclose(P[a_of[ok]], pos[ok])               # every kept atom on its own site


def test_offcentring_zero_on_ideal_and_exact_across_the_boundary():
    n = (2, 2, 3)
    sites, L, _ = _lattice(n, jitter=0.0)
    pos = pvk.ideal_positions(sites, np.zeros(3), L / n) + 0.3
    _, d = pvk.ti_offcentring(pos, sites, n, L)
    assert np.abs(d).max() < 1e-12
    # Ti in the last cell along z: its O3 cage partner is across the periodic boundary
    r = [m for m, s in enumerate(sites) if tuple(s) == (1, 1, 2, 1)][0]
    pos[r] += [0.1, -0.05, 0.2]
    rows, d = pvk.ti_offcentring(pos, sites, n, L)
    assert np.allclose(d[list(rows).index(r)], [0.1, -0.05, 0.2])


def test_lattice_fit_recovers_spacing():
    sites, L, pos = _lattice(jitter=0.02)
    pb = pos[sites[:, 3] == 0]
    o, a = pvk.fit_lattice(pb, (4.0, 4.0, 4.0), pb[0])
    assert np.allclose(a, [4.1, 3.9, 3.9], atol=0.03)


def test_cell_written_reads_back_in_ase_and_keeps_order():
    import warnings
    from ase.io import read
    warnings.filterwarnings("ignore")           # ASE cannot validate keywords without CASTEP
    sites, L, pos = _lattice()
    order = pvk.castep_order(sites)
    zn = pvk.SUB_Z[sites[:, 3]]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.cell")
        pvk.write_cell(p, pos[order], zn[order], L, (2, 2, 1))
        at = read(p, format="castep-cell")
        assert np.allclose(at.get_positions(), pos[order], atol=1e-6)
        assert (at.get_atomic_numbers() == zn[order]).all()
        assert np.allclose(at.cell.lengths(), L)
        txt = open(p).read()
        assert "FIX_ALL_CELL : true" in txt and "KPOINTS_MP_GRID 2 2 1" in txt


def write_fake_geom(path, frames, species, L):
    """A .geom in CASTEP's layout (atomic units), one block per frame."""
    with open(path, "w") as f:
        f.write(" BEGIN header\n  \n END header\n  \n")
        counter = {}
        idx = []
        for s in species:
            counter[s] = counter.get(s, 0) + 1
            idx.append(counter[s])
        for n, (E, pos, frc) in enumerate(frames):
            f.write(f"{n:>40d}{'':>55s}F   F   F   F      <-- c\n")
            e = E / pvk.HARTREE_EV
            f.write(f"{'':>19s}{e:.16E}    {e:.16E}{'':>29s}<-- E\n")
            for ax in range(3):
                h = [0.0, 0.0, 0.0]
                h[ax] = L[ax] / pvk.BOHR_A
                f.write(f"{'':>19s}" + "    ".join(f"{v:.16E}" for v in h) + "  <-- h\n")
            for s, i, p in zip(species, idx, pos / pvk.BOHR_A):
                f.write(f" {s:<3s}{i:>14d}    " + "    ".join(f"{v:.16E}" for v in p) + "  <-- R\n")
            for s, i, g in zip(species, idx, frc / (pvk.HARTREE_EV / pvk.BOHR_A)):
                f.write(f" {s:<3s}{i:>14d}    " + "    ".join(f"{v:.16E}" for v in g) + "  <-- F\n")
            f.write("  \n")


def test_geom_roundtrip():
    sites, L, pos = _lattice()
    order = pvk.castep_order(sites)
    sp = [pvk.SYMBOL[int(z)] for z in pvk.SUB_Z[sites[order, 3]]]
    frc = np.random.default_rng(3).normal(0, 0.1, pos.shape)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.geom")
        write_fake_geom(p, [(-100.0, pos[order], frc), (-101.5, pos[order] + 0.01, frc / 2)], sp, L)
        fr = pvk.read_geom(p)
        assert len(fr) == 2 and fr[1]["species"] == sp
        assert np.allclose(fr[1]["pos"], pos[order] + 0.01, atol=1e-9)
        assert np.allclose(fr[0]["force"], frc, atol=1e-9)
        assert abs(fr[1]["energy_eV"] + 101.5) < 1e-9


def test_analysis_end_to_end_on_fake_relaxation():
    src = os.path.join(HERE, "runs", "smoke")
    if not os.path.exists(os.path.join(src, "block.npz")):
        print("  (skipped: build runs/smoke first)")
        return
    import analyse
    b = np.load(os.path.join(src, "block.npz"))
    sites, L, order, n = b["sites"], b["L"], b["order"], tuple(b["n_cells"])
    sp = [pvk.SYMBOL[int(z)] for z in b["znum"][order]]
    gt_rel = b["start_gt"] + 0.03                    # 'relaxed GT' = GT moved rigidly
    zero = np.zeros_like(gt_rel)
    with tempfile.TemporaryDirectory() as d:
        shutil.copytree(src, os.path.join(d, "smoke"))
        # P1 relaxes all the way onto the relaxed GT, P2 only halfway
        fin = {"gt": gt_rel, "p1": gt_rel.copy(),
               "p2": b["start_p2"] + 0.5 * pvk.min_image(gt_rel - b["start_p2"], L)}
        for k, E in (("gt", -1000.0), ("p1", -1000.0), ("p2", -999.0)):
            write_fake_geom(os.path.join(d, "smoke", k, f"smoke_{k}.geom"),
                            [(E + 5, b[f"start_{k}"][order], zero), (E, fin[k][order], zero)],
                            sp, L)
        analyse.main([d, "--tag", "smoke"])
        if os.environ.get("KEEP_FAKE_FIG"):
            shutil.copy(os.path.join(d, "smoke", "fig_relax.png"), os.environ["KEEP_FAKE_FIG"])
        import json
        r = json.load(open(os.path.join(d, "smoke", "results.json")))
        assert r["reference"] == "gt_relaxed"
        assert max(r["p1_relaxed"]["rms_xyz_A"]["all"]) < 1e-6        # identical to the ref
        assert r["runs"]["p2"]["dE_meV_per_fu_vs_gt"] == round(1e3 / np.prod(n), 2)
        p2s = np.array(r["p2_start"]["rms_xyz_A"]["all"])
        p2r = np.array(r["p2_relaxed"]["rms_xyz_A"]["all"])
        assert np.allclose(p2r, p2s / 2, atol=2e-4)                     # halfway -> half error
        assert os.path.exists(os.path.join(d, "smoke", "fig_relax.png"))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, HERE)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
