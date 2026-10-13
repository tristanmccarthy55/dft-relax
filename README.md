# dft-relax

Does the 3-D ptychography structure relax, under DFT, to something closer to the truth than a
guess made from a 2-D projection?

Three structures go into CASTEP and are relaxed the same way:

| seed | what it is | depth (z, along the beam) |
|---|---|---|
| `gt` | the reference structure cut to the block | true |
| `p1` | the blind atomfind result (`found_atoms.csv`, report_v2 Sec. III) snapped onto lattice sites | measured by ptychography |
| `p2` | the depth-summed phase used as a 2-D image: each column's (x, y) from a Gaussian fit, every atom of the column stacked at the ideal depth | none, uniform |

The GT is relaxed too, and the relaxed GT is the reference. Then the ranking compares like
with like: same functional, same cell, same boundary conditions.

## Design

**One shared block of sites.** All three structures are written on the same 240 perovskite
sites (2 × 2 × 12 pseudo-cubic cells: 48 Pb, 48 Ti, 144 O) in the same fixed cell. Atom *i* is
the same site in every structure, so no matching is needed after relaxation. They differ only in
where the atoms sit.

- Block: the centre of the scan window, pure PbTiO₃, mid-layer. In the prepared GT frame it
  spans x 34.7–43.0, y 14.6–22.3, z 8.8–55.4 Å, which is the report's bulk depth band (10–56 Å).
- Cell: 8.338 × 7.759 × 46.622 Å, from the GT's own lattice spacing in the block. It is held
  fixed (`FIX_ALL_CELL`), standing in for the clamping by the rest of the film. These three
  numbers are the only GT input the propositions share.
- P1: 237 of 240 sites carry a measured atom. The 3 empty sites are oxygens in the Ti columns,
  the report's "overlapped O"; each is filled at its column's own mean (x, y) and the ideal depth.
  Five found atoms in the block volume fit no site (spurious or wrong species) and are dropped.
- P2: column centres come from a 0.9 Å fit disc, about half the 1.95 Å column spacing. A 0.5 Å disc
  locks onto one half of a split Pb column (see below) and is 0.17 Å worse than a perfect projection.
  At 0.9 Å, P2's columns sit 0.055 Å RMS from where a perfect projection of the GT would put them.
- P1 and P2 get the same seeded 0.02 Å rattle. Without it P2 would be mirror-symmetric along z.
  The z forces would then be exactly zero, and DFT could never move it off that symmetry.

**DFT settings** (`build_structures.py` `Cfg`; written into each `.param.in`): CASTEP 24.1,
**PBEsol**, OTFG ultrasoft pseudopotentials (generated for PBEsol on the fly), 700 eV (above the O
"fine" 626 eV), 2 × 2 × 1 k-points (spacing 0.07 Å⁻¹), fixed occupancy, LBFGS to 0.02 eV/Å.
2496 electrons per structure (Pb carries 22 with its semicore shells).

## What the starting structures already show

`runs/b2x2x12/build_report.json` and `build_qa.png`. RMS distance from the unrelaxed GT, per
axis, rigid translation removed:

| | x (across layers) | y (along layers) | z (beam) |
|---|---|---|---|
| P1, 3-D ptycho | 0.013 Å | 0.009 Å | **0.259 Å** |
| P2, 2-D guess | **0.209 Å** | 0.048 Å | 0.045 Å |
| a perfect undistorted lattice | 0.208 Å | 0.048 Å | 0.045 Å |

Ti off-centring δ (the report's polarisation proxy), RMS error against the GT. The GT's own spread
is 0.271 / 0.053 / 0.059 Å:

| | δx | δy | δz |
|---|---|---|---|
| P1 | 0.009 Å | 0.008 Å | 0.185 Å |
| P2 | 0.277 Å | 0.052 Å | 0.060 Å |

- The block holds a domain wall. Pb sits about 0.3 Å one way along x for the first ~25 Å of depth
  and about 0.5 Å the other way after it. P1 follows this almost exactly. In projection the two halves
  overlap, the Pb column shows up split in two, and P2 is no closer to the GT than a perfect lattice.
- In z the order flips. P1's depth noise (0.26 Å) is five times the GT's real deviation from a
  perfect lattice (0.045 Å). So a single all-atom RMS before relaxation would favour P2.
  Report per axis, and report δ.
- P1's δz is noise (error 0.185 against a 0.059 spread), as the report found ("not measured").
  **The sharpest question DFT can answer here: does relaxing P1 recover δz?**

## Running it

Build locally (needs `~/atomfind`, `~/Desktop/atomfind_out/`), then commit `runs/` and push. The
builder is deterministic (seeded), and `runs/` is tracked so Blythe gets the exact inputs by
`git pull`. CASTEP's outputs inside it are git-ignored.

```bash
~/hyperspy-bundle/bin/python build_structures.py                  # runs/b2x2x12  (the experiment)
~/hyperspy-bundle/bin/python build_structures.py --preset smoke   # runs/smoke    (40 atoms)
~/hyperspy-bundle/bin/python test_dftrelax.py                     # 6 checks, ~5 s
```

On Blythe, first time, then whenever the inputs change:

```bash
cd $SHARE/phucrh && git clone https://github.com/tristanmccarthy55/dft-relax.git    # once
cd $SHARE/phucrh/dft-relax && git pull
bash submit_blythe.sh runs/smoke         # FIRST: ~2 h wall, 32 cores
```

The job never edits a tracked file: it copies `<seed>.param.in` to `<seed>.param` and adds the run
time there. So `git pull` works while runs are going.

The smoke block is centred on P1's worst starting contact (a Ti–O pair 1.04 Å apart), so it also
proves CASTEP survives that. Each smoke job should end `RELAX STATUS: MAXITER` (3 steps by
design; check with `grep "RELAX STATUS" runs/smoke/*/relax_*.out`). Pull and check it the same way
as the real run below:

```bash
mkdir -p ~/Desktop/dftrelax_smoke && cd ~/Desktop/dftrelax_smoke
scp -O 'phucrh@blythe.scrtp.warwick.ac.uk:/springbrook/share/physics/phucrh/dftrelax_smoke_*_results_*.tgz' .
for f in dftrelax_*.tgz; do tar xzf "$f"; done
~/hyperspy-bundle/bin/python ~/Library/CloudStorage/OneDrive-UniversityofWarwick/Documents/PtycoShelves/dft-relax/analyse.py . --tag smoke
```

When that parses, submit the experiment:

```bash
bash submit_blythe.sh runs/b2x2x12       # 3 jobs x 96 cores, 48 h links, up to 4 continuations
```

Each job writes `relax_dftrelax_<seed>_<jobid>.out` in its own `runs/<tag>/<name>/`. CASTEP stops
45 min before the wall and the script resubmits itself with `continuation : default`. The last
link packs `$SHARE/phucrh/dftrelax_<seed>_results_<time>.tgz` (no `.check`) and prints
`RELAX STATUS: CONVERGED | MAXITER | STALLED | CHAIN_EXHAUSTED | FAILED`. STALLED means a link
made no geometry step: read the `.castep`.

Pull and analyse (Mac):

```bash
mkdir -p ~/Desktop/dftrelax_b2x2x12 && cd ~/Desktop/dftrelax_b2x2x12
scp -O 'phucrh@blythe.scrtp.warwick.ac.uk:/springbrook/share/physics/phucrh/dftrelax_b2x2x12_gt_results_*.tgz' .
scp -O 'phucrh@blythe.scrtp.warwick.ac.uk:/springbrook/share/physics/phucrh/dftrelax_b2x2x12_p1_results_*.tgz' .
scp -O 'phucrh@blythe.scrtp.warwick.ac.uk:/springbrook/share/physics/phucrh/dftrelax_b2x2x12_p2_results_*.tgz' .
for f in dftrelax_*.tgz; do tar xzf "$f"; done
~/hyperspy-bundle/bin/python ~/Library/CloudStorage/OneDrive-UniversityofWarwick/Documents/PtycoShelves/dft-relax/analyse.py . --tag b2x2x12
```

`analyse.py` writes `results.json` and `fig_relax.png`, and prints a table:

- RMS distance to the relaxed GT, per axis, before and after relaxation;
- the same after the best whole-cell shift, so a wall that settled one cell away is not
  double-counted;
- δ error and signal/error per component (>1 = resolved);
- ΔE against the relaxed GT in meV per formula unit (≈0 means the same minimum).

It runs on a partial set, and scores against the unrelaxed GT (with a warning) until the GT
finishes.

## Smoke run (2026-10-13): passed

All three ran on CASTEP 24.1 with PBEsol and stopped after 3 LBFGS steps, as designed. The cell
was held fixed (6 cell constraints), no symmetry was imposed (1 operation), and the centre of
mass was constrained. The only warning was the intended 3-step stop.

- **P1's 1.04 Å contact is survivable.** Peak force went 136 → 23 → 7.7 → 5.8 eV/Å, and the
  contact opened to 1.73 Å by step 2. P1's first SCF took 78 iterations against 25 for the GT, so
  P1 runs about 3× slower per step early on.
- The smoke GT starts at 23 eV/Å with a 1.41 Å Pb–O contact. That comes from a box only 2 cells
  deep along the beam, which glues the block's ends together. The main block's GT starts at 1.66 Å.
- **CASTEP re-sorts ions in its output (O, Ti, Pb).** `analyse.py` therefore maps the results
  back to sites by matching the first `.geom` frame to the built start, to 1e-3 Å.

## Cost

Measured on the smoke run (40 atoms, 500 eV, 4 k-points, 32 processes): 2.2 s per SCF
iteration for the GT, about 15–25 iterations per SCF, and 1–2 SCFs per LBFGS step. The main
block (240 atoms, 700 eV, 2 k-points, 96 processes) is roughly 25–50× dearer per iteration
after the extra cores. That makes **30–90 min per geometry step**. If P1 needs 100–300 steps,
that is several 48 h links. `MAX_CHAIN=8` caps a structure at about 16 days; it is a ceiling,
and each job stops when converged.

**Memory check.** CASTEP reserves about 1 GB of fixed overhead per process, and the smoke peaked
at 2.3 GB per process, so the main run requests 400 GB. About two minutes after a job starts:

```bash
sed -n '/MEMORY AND SCRATCH/,/Requirements will/p' runs/b2x2x12/p1/b2x2x12_p1.castep
```

Take "Approx. total storage required per process" × 96 × 1.5 (the smoke's peak was 1.5× its
estimate). If that exceeds 400 GB: `scancel`, move the `.castep` aside, and resubmit that
structure with `MEM=600G`.

Partition limits (`scontrol show partition compute`, 2026-10-13): `MaxMemPerCPU=4591` MB,
`MaxTime=2-00:00:00`. 400 GB over 96 tasks is 4.27 GB per CPU, inside the limit. Above
96 × 4.59 ≈ 430 GB, Slurm adds CPUs to match the memory (600G → about 134 CPUs for the same
96 tasks). That still works, but costs more of your CPU allowance.

## Caveats

1. **Size.** The 20 Å window holds ~2000 atoms, far past a CASTEP relaxation. The block is a
   2 × 2-column core, 12 cells deep. `--preset wide` (3 × 3 × 12, 540 atoms) is there and roughly
   5–10× dearer.
2. **The box is not the sample.** Periodic faces join pieces of the texture that do not meet in the
   sample. Measured on the GT: 0.18 / 0.17 Å RMS mismatch at the x / y faces and **0.43 Å at the z
   face**, because the block's two ends sit on opposite sides of the domain wall, so the box adds a
   second wall. There is no SrTiO₃ in the box, so nothing opposes polarisation across the layers.
   The relaxed structures are therefore not the sample's structure. That is why the relaxed GT is
   the reference: the comparison is only fair inside one box.
3. **The GT was not relaxed with DFT.** Its header is `_HIST2Poscar time 346`, a frame from an
   ABINIT/MULTIBINIT history file: a second-principles model at 19,440 atoms. Expect the GT itself to
   move under PBEsol. `gt_relaxed` vs `gt_start` is the size of that model difference. Gaps
   between P1 and the GT smaller than it are not meaningful.
4. **P1 starts with impossible bonds.** 16 of 48 Ti–O pairs along the beam start 1.04–1.5 Å apart
   (real ≈ 1.95 Å), from depth noise. That is below the pseudopotential core radii (~1.53 Å), so the
   first steps carry very large forces. The smoke run tests the worst one.
5. **Relaxation finds the nearest minimum, and lower energy is not the same as closer to the truth.**
   The box holds no SrTiO₃, so nothing makes domain walls favourable, and a single-domain
   structure is likely the lower-energy state in this box. P2 has no wall to keep, so expect it to
   fall into a single domain *and* sit lowest in energy. The smoke run already shows this: P2 is
   56 meV per formula unit below the GT after 3 steps. Judge the result by distance to the relaxed
   GT and by the Ti off-centring. ΔE only says which basin each start fell into.
6. **PBEsol, not the PBE of the EELS work.** PBE overstates PbTiO₃'s tetragonality and polar
   distortion. For a structural comparison PBEsol is the standard choice. Switch with
   `Cfg.xc` if you want consistency with `eels/` instead.
7. **The 2-D proxy is generous to P2.** A depth-summed phase shows oxygen columns; HAADF would not.
8. **Full-beam block not offered.** The prepared GT frame lost the entry-face O plane (abtem
   orthogonalisation dropped 301 atoms cell-wide), and P1 has no atoms past z ≈ 65 Å.

## Files

| file | role |
|---|---|
| `pvk.py` | site lattice, assignment, Ti off-centring, periodic helpers, `.cell` writer, `.geom` reader |
| `build_structures.py` | builds `runs/<tag>/{gt,p1,p2}/<seed>.{cell,param.in}`, `block.npz`, `build_report.json`, `build_qa.png` |
| `run_relax.slurm` | one fixed-cell relax, self-continuing, packs its results |
| `submit_blythe.sh` | submits the three structures of one build |
| `analyse.py` | scores the relaxed structures, `results.json` + `fig_relax.png` |
| `qa_figs.py` | the two figures |
| `test_dftrelax.py` | synthetic checks incl. a fake CASTEP trajectory through the analysis |
