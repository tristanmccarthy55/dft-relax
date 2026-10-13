#!/bin/bash
#SBATCH -J dftrelax
#SBATCH -o relax_%x_%j.out
#SBATCH -e relax_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --time=48:00:00
#SBATCH --mem=192G
#SBATCH -p compute
#
# Fixed-cell CASTEP geometry optimisation of ONE structure (gt, p1 or p2), run from its own
# directory runs/<tag>/<name>/ which holds <SEED>.cell + <SEED>.param.in. Use submit_blythe.sh,
# which sets everything below.
#
# Past the wall: CASTEP is given run_time = wall - 45 min, so it stops cleanly and checkpoints;
# this script then resubmits itself with `continuation : default` (up to MAX_CHAIN links).
# The LAST link packs <SEED>'s results (no .check) into $PACK_DIR/dftrelax_<SEED>_results_<time>.tgz
# and prints  RELAX STATUS: <CONVERGED|MAXITER|STALLED|CHAIN_EXHAUSTED|FAILED>.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
: "${SEED:?set SEED=<tag>_<name>}"
: "${RELAX_SCRIPT:?set RELAX_SCRIPT=<abs path to run_relax.slurm>}"
CHAIN="${CHAIN:-0}"
MAX_CHAIN="${MAX_CHAIN:-4}"
WALL_H="${WALL_H:-48}"
PACK_DIR="${PACK_DIR:-/springbrook/share/physics/phucrh}"
CASTEP_MODULE="${CASTEP_MODULE:-phys/CASTEP/24.1-foss-2023b}"
export OMP_NUM_THREADS=1

module purge 2>/dev/null || true
module load "$CASTEP_MODULE"
BIN="$(command -v castep.mpi || command -v castep.serial || command -v castep || true)"
[ -n "$BIN" ] || { echo "no castep binary after 'module load $CASTEP_MODULE'"; exit 1; }

# ---- param for this link: the tracked template + run_time (+ continuation after the first link).
# The template stays untouched so `git pull` never meets a locally modified file.
cp "${SEED}.param.in" "${SEED}.param"
echo "run_time          : $(( WALL_H * 3600 - 2700 ))" >> "${SEED}.param"
if [ "$CHAIN" -gt 0 ]; then
  [ -f "${SEED}.check" ] || { echo "CHAIN=$CHAIN but no ${SEED}.check to continue from"; exit 1; }
  echo "continuation      : default" >> "${SEED}.param"
fi

nsteps () { local n; n=$(grep -c "<-- c" "${SEED}.geom" 2>/dev/null); echo "${n:-0}"; }
touch ".link_start_${CHAIN}"             # progress test: CASTEP rewrites .geom after every step
echo "[$(date)] $SEED link $CHAIN/$MAX_CHAIN on $SLURM_NTASKS tasks ($BIN), $(nsteps) geometry frames so far"
if [[ "$BIN" == *.mpi ]]; then srun "$BIN" "$SEED"; else "$BIN" "$SEED"; fi
echo "[$(date)] CASTEP exited ($?)"

# ---- what happened?
STATUS=""
if grep -qiE "Geometry optimi[sz]ation completed successfully" "${SEED}.castep" 2>/dev/null; then
  STATUS=CONVERGED
elif grep -qiE "Geometry optimi[sz]ation failed to converge" "${SEED}.castep" 2>/dev/null; then
  STATUS=MAXITER
elif ! [ "${SEED}.geom" -nt ".link_start_${CHAIN}" ]; then
  STATUS=STALLED            # no new geometry step this link: a crash, not the clock -> read the .castep
elif [ -f "${SEED}.check" ] && [ "$CHAIN" -lt "$MAX_CHAIN" ]; then
  NEXT=$(sbatch --parsable -J "$SLURM_JOB_NAME" --time="${WALL_H}:00:00" \
           --ntasks="$SLURM_NTASKS" --mem="${SLURM_MEM_PER_NODE}M" \
           --export=ALL,SEED="$SEED",CHAIN=$((CHAIN + 1)),MAX_CHAIN="$MAX_CHAIN",WALL_H="$WALL_H",RELAX_SCRIPT="$RELAX_SCRIPT",PACK_DIR="$PACK_DIR" \
           "$RELAX_SCRIPT")
  echo "[$(date)] not finished -> continuation link $((CHAIN + 1)) submitted as job $NEXT"
  echo "  geometry frames so far: $(nsteps)"
  exit 0
elif [ -f "${SEED}.check" ]; then
  STATUS=CHAIN_EXHAUSTED
else
  STATUS=FAILED
fi

# ---- final link: pack the small outputs (never the multi-GB .check)
NAME_DIR="$(basename "$PWD")"
TAG_DIR="$(dirname "$PWD")"
OUT="${PACK_DIR}/dftrelax_${SEED}_results_$(date +%Y%m%d_%H%M%S).tgz"
tar czf "$OUT" -C "$(dirname "$TAG_DIR")" \
    --exclude='*.check' --exclude='*.check_bak' --exclude='*.castep_bin' --exclude='*.cst_esp' \
    "$(basename "$TAG_DIR")/${NAME_DIR}" \
    "$(basename "$TAG_DIR")/block.npz" "$(basename "$TAG_DIR")/build_report.json"
echo "geometry frames: $(nsteps)"
grep -E "Final (free )?energy|Final Enthalpy" "${SEED}.castep" 2>/dev/null | tail -1
echo "RELAX STATUS: $STATUS"
echo "PACKED -> $OUT"
