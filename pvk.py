"""@file pvk.py
@brief Perovskite site-block machinery shared by the builder and the analysis.

Every structure in this experiment (ground truth, 3-D ptycho, 2-D guess) is expressed on ONE
set of ABO3 lattice sites, so atom i in one structure is atom i in every other and no matching
is needed after relaxation. Frame: x = stacking (normal to the PTO/STO layers), y = along the
layers, z = beam. All lengths in angstrom.

Sublattices s (fractional offsets inside a pseudo-cubic cell, A at the origin):
    0 A  (Pb)  (0, 0, 0)
    1 B  (Ti)  (1/2, 1/2, 1/2)
    2 O1       (0, 1/2, 1/2)   Ti-O bond along x
    3 O2       (1/2, 0, 1/2)   Ti-O bond along y
    4 O3       (1/2, 1/2, 0)   Ti-O bond along z; shares the Ti column seen down the beam
"""
from __future__ import annotations

import numpy as np

BASIS = np.array([[0.0, 0.0, 0.0],
                  [0.5, 0.5, 0.5],
                  [0.0, 0.5, 0.5],
                  [0.5, 0.0, 0.5],
                  [0.5, 0.5, 0.0]])
SUB_Z = np.array([82, 22, 8, 8, 8])
SUB_NAME = ("A", "B", "O1", "O2", "O3")
SYMBOL = {82: "Pb", 22: "Ti", 8: "O", 38: "Sr"}
ZNUM = {v: k for k, v in SYMBOL.items()}
CASTEP_ORDER = (82, 22, 8)          # species order written to the .cell (CASTEP re-sorts it)
HARTREE_EV = 27.211386245988
BOHR_A = 0.529177210903


# ---------------------------------------------------------------- lattice
def fit_lattice(pb, a0, ref, iters=6):
    """@brief Fit an orthogonal lattice  r = o + a * n  (per axis) to A-site positions.

    @param pb   (N, D) A-site (Pb) positions near the block (D = 2 or 3)
    @param a0   (D,) starting lattice spacing
    @param ref  (D,) a position close to one A site (sets which site is n = 0)
    @return (o, a) -- origin (D,) and spacing (D,)
    """
    o = np.asarray(ref, float).copy()
    a = np.asarray(a0, float).copy()
    for _ in range(iters):
        n = np.round((pb - o) / a)
        for ax in range(pb.shape[1]):
            if np.ptp(n[:, ax]) > 0:
                a[ax], o[ax] = np.polyfit(n[:, ax], pb[:, ax], 1)
            else:
                o[ax] = np.mean(pb[:, ax] - a[ax] * n[:, ax])
    return o, a


def block_sites(n_cells, origin_idx=(0, 0, 0), margin=0):
    """@brief Integer site table (i, j, k, s) of a block, A sites first then by sublattice.

    With margin > 0 the block is padded by that many cells on every side (used to catch atoms
    that belong just outside the block instead of letting them claim a boundary site).
    @return (M, 4) int array and an (M,) bool mask of the sites inside the unpadded block.
    """
    nx, ny, nz = n_cells
    i0, j0, k0 = origin_idx
    rows, inside = [], []
    for s in range(5):
        for i in range(i0 - margin, i0 + nx + margin):
            for j in range(j0 - margin, j0 + ny + margin):
                for k in range(k0 - margin, k0 + nz + margin):
                    rows.append((i, j, k, s))
                    inside.append(i0 <= i < i0 + nx and j0 <= j < j0 + ny and k0 <= k < k0 + nz)
    return np.array(rows, int), np.array(inside, bool)


def ideal_positions(sites, o, a):
    """Ideal Cartesian positions of integer sites (i, j, k, s) on lattice (o, a)."""
    return o + (sites[:, :3] + BASIS[sites[:, 3]]) * a


def assign(pos, znum, sites, ideal, tol_xy, tol_z, score=None):
    """@brief Put labelled atoms onto lattice sites of their own species.

    Each atom goes to the nearest site of its own sublattice family, measured in a metric
    scaled by (tol_xy, tol_xy, tol_z); it is accepted only inside that ellipsoid. When two
    atoms claim one site the closer one wins (or the higher `score`, when given).

    @return (site_of_atom, atom_of_site): site row per atom (-1 = rejected) and atom index per
            site (-1 = empty).
    """
    from scipy.spatial import cKDTree
    scale = np.array([tol_xy, tol_xy, tol_z], float)
    site_of_atom = np.full(len(pos), -1)
    dist = np.full(len(pos), np.inf)
    for zn in np.unique(SUB_Z):
        srows = np.where(SUB_Z[sites[:, 3]] == zn)[0]
        arows = np.where(znum == zn)[0]
        if len(srows) == 0 or len(arows) == 0:
            continue
        d, j = cKDTree(ideal[srows] / scale).query(pos[arows] / scale)
        ok = d <= 1.0
        site_of_atom[arows[ok]] = srows[j[ok]]
        dist[arows[ok]] = d[ok]
    atom_of_site = np.full(len(sites), -1)
    key = dist if score is None else -np.asarray(score, float)
    for a_idx in np.argsort(key):
        s_idx = site_of_atom[a_idx]
        if s_idx < 0:
            continue
        if atom_of_site[s_idx] < 0:
            atom_of_site[s_idx] = a_idx
        else:
            site_of_atom[a_idx] = -1          # lost the conflict -> treated as a spurious atom
    return site_of_atom, atom_of_site


# ---------------------------------------------------------------- periodic geometry
def min_image(d, L):
    """Minimum-image displacement for an orthogonal cell of edge lengths L."""
    return d - L * np.round(d / L)


def site_lookup(sites, n_cells):
    """Map (i, j, k, s) -> row for block-local indices 0 <= i < nx etc."""
    return {tuple(r): m for m, r in enumerate(sites)}


def ti_offcentring(pos, sites, n_cells, L):
    """@brief Ti off-centring delta = r_Ti - centroid(O6), periodic, per B site.

    The O6 cage is taken from the site topology (not nearest neighbours), so a badly placed O
    still counts toward the cage it belongs to. Same proxy as the report (pol_vortex.py).
    @return (rows of the B sites, (nB, 3) delta)
    """
    nx, ny, nz = n_cells
    look = site_lookup(sites, n_cells)
    b_rows = np.where(sites[:, 3] == 1)[0]
    out = np.zeros((len(b_rows), 3))
    for m, r in enumerate(b_rows):
        i, j, k = sites[r, :3]
        cage = [look[(i, j, k, 2)], look[((i + 1) % nx, j, k, 2)],
                look[(i, j, k, 3)], look[(i, (j + 1) % ny, k, 3)],
                look[(i, j, k, 4)], look[(i, j, (k + 1) % nz, 4)]]
        rel = min_image(pos[cage] - pos[r], L)
        out[m] = -rel.mean(0)
    return b_rows, out


def rms_xyz(ref, test, sites, L):
    """@brief Per-axis RMS distance of test from ref (rigid translation removed), per species.

    Sites correspond one-to-one, so no matching is involved. The mean displacement is removed
    because a periodic cell has no absolute origin (and CASTEP keeps each centre of mass).
    """
    d = min_image(test - ref, L)
    d -= d.mean(0)
    out = {}
    for z in CASTEP_ORDER:
        m = SUB_Z[sites[:, 3]] == z
        out[SYMBOL[z]] = np.sqrt((d[m] ** 2).mean(0)).round(4).tolist()
    out["all"] = np.sqrt((d ** 2).mean(0)).round(4).tolist()
    return out


def cyclic_perms(sites, n_cells):
    """Yield (shift, perm) for every whole-cell cyclic relabelling of a periodic block.

    test[perm] puts test's site (i+sx, j+sy, k+sz) where ref's site (i, j, k) is: used to
    score a structure whose features may sit at another place along a periodic axis.
    """
    look = site_lookup(sites, n_cells)
    nx, ny, nz = n_cells
    for sx in range(nx):
        for sy in range(ny):
            for sz in range(nz):
                perm = np.array([look[((i + sx) % nx, (j + sy) % ny, (k + sz) % nz, s)]
                                 for i, j, k, s in sites])
                yield (sx, sy, sz), perm


def min_distance(pos, L):
    """Shortest interatomic distance in a periodic orthogonal cell, and the atom pair."""
    from scipy.spatial import cKDTree
    t = cKDTree(np.mod(pos, L), boxsize=L)
    d, j = t.query(np.mod(pos, L), k=2)
    i = int(np.argmin(d[:, 1]))
    return float(d[i, 1]), (i, int(j[i, 1]))


# ---------------------------------------------------------------- CASTEP I/O
def castep_order(sites):
    """Row order that groups atoms by species (Pb, Ti, O) and keeps site order inside each.

    Only a tidy .cell. CASTEP 24.1 re-sorts species in its output (O, Ti, Pb in the smoke run),
    so results are mapped back to sites by position (match_to_start), never by order.
    """
    zs = SUB_Z[sites[:, 3]]
    return np.concatenate([np.where(zs == z)[0] for z in CASTEP_ORDER])


def write_cell(path, pos, znum, L, kgrid, comment=""):
    """@brief Write a fixed-cell CASTEP .cell (orthogonal lattice, absolute positions)."""
    lines = [f"! {comment}" if comment else "!",
             "%BLOCK LATTICE_CART", "ang",
             f"  {L[0]:.8f}  0.0  0.0", f"  0.0  {L[1]:.8f}  0.0", f"  0.0  0.0  {L[2]:.8f}",
             "%ENDBLOCK LATTICE_CART", "", "%BLOCK POSITIONS_ABS", "ang"]
    lines += [f"{SYMBOL[int(z)]:<3s} {p[0]:14.8f} {p[1]:14.8f} {p[2]:14.8f}"
              for p, z in zip(pos, znum)]
    lines += ["%ENDBLOCK POSITIONS_ABS", "",
              "! the shared fixed cell is the experiment's constraint",
              "FIX_ALL_CELL : true",
              f"KPOINTS_MP_GRID {kgrid[0]} {kgrid[1]} {kgrid[2]}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def match_to_start(frame0, start, znum, L, tol=1e-3):
    """@brief Permutation p with frame0["pos"][p[m]] == start[m] for every site m.

    frame0 is the first .geom frame, i.e. the structure CASTEP was given, in CASTEP's own ion
    order; start is the same structure in site order. Matching is per species, periodic, and
    must be one-to-one within tol (A) -- anything else means the wrong files were paired.
    """
    from scipy.spatial import cKDTree
    sym = np.array(frame0["species"])
    p = np.full(len(start), -1)
    for z in np.unique(znum):
        rows = np.where(znum == z)[0]
        ions = np.where(sym == SYMBOL[int(z)])[0]
        if len(ions) != len(rows):
            raise RuntimeError(f"{SYMBOL[int(z)]}: {len(ions)} ions in the .geom, "
                               f"{len(rows)} sites in the block")
        d, j = cKDTree(np.mod(frame0["pos"][ions], L), boxsize=L).query(np.mod(start[rows], L))
        if d.max() > tol or len(set(j)) != len(j):
            raise RuntimeError(f"{SYMBOL[int(z)]}: .geom start does not match the built start "
                               f"(max offset {d.max():.2e} A, {len(j) - len(set(j))} clashes)")
        p[rows] = ions[j]
    return p


def read_geom(path):
    """@brief Parse a CASTEP .geom trajectory.

    @return list of frames, each {"energy_eV", "pos" (N,3) A, "force" (N,3) eV/A,
            "species" [N]} in CASTEP's ion order (species-grouped, input order within).
    """
    frames, cur = [], None
    with open(path) as f:
        for line in f:
            tag = line.rstrip()[-5:]
            parts = line.split()
            if tag.endswith("<-- c"):
                if cur is not None:
                    frames.append(cur)
                cur = {"energy_eV": np.nan, "pos": [], "force": [], "species": []}
            elif cur is None:
                continue
            elif tag.endswith("<-- E"):
                cur["energy_eV"] = float(parts[0]) * HARTREE_EV
            elif tag.endswith("<-- R"):
                cur["species"].append(parts[0])
                cur["pos"].append([float(v) * BOHR_A for v in parts[2:5]])
            elif tag.endswith("<-- F"):
                cur["force"].append([float(v) * HARTREE_EV / BOHR_A for v in parts[2:5]])
    if cur is not None:
        frames.append(cur)
    for fr in frames:
        fr["pos"] = np.array(fr["pos"])
        fr["force"] = np.array(fr["force"])
    return frames
