#!/bin/bash
# @file submit_blythe.sh
# @brief Submit the relaxations of one build (runs/<tag>/{gt,p1,p2}) on Blythe, one job each.
#
#   bash submit_blythe.sh runs/smoke          # 40 atoms, 3 geometry steps, 2 h wall: run FIRST
#   bash submit_blythe.sh runs/b2x2x12        # the experiment: 240 atoms each, 48 h links
#   SEEDS="p2" bash submit_blythe.sh runs/b2x2x12      # resubmit one structure
#
# Overrides: NTASKS (96), MEM (192G), WALL_H (48), MAX_CHAIN (4), PACK_DIR ($SHARE/phucrh).
set -euo pipefail
[ $# -eq 1 ] || { echo "usage: bash submit_blythe.sh runs/<tag>"; exit 1; }
TAG_DIR="$(cd "$1" && pwd)"
TAG="$(basename "$TAG_DIR")"
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ "$TAG" = smoke ]; then
  NTASKS="${NTASKS:-32}"; MEM="${MEM:-64G}"; WALL_H="${WALL_H:-2}"; MAX_CHAIN="${MAX_CHAIN:-0}"
fi
NTASKS="${NTASKS:-96}"; MEM="${MEM:-192G}"; WALL_H="${WALL_H:-48}"; MAX_CHAIN="${MAX_CHAIN:-4}"
PACK_DIR="${PACK_DIR:-/springbrook/share/physics/phucrh}"

for name in ${SEEDS:-gt p1 p2}; do
  d="$TAG_DIR/$name"; seed="${TAG}_${name}"
  [ -f "$d/$seed.cell" ] && [ -f "$d/$seed.param.in" ] || { echo "missing $d/$seed.{cell,param.in}"; exit 1; }
  if [ -f "$d/$seed.castep" ]; then
    echo "$d/$seed.castep exists -- move it away first (a fresh start must not append to it)"; exit 1
  fi
  jid=$(cd "$d" && sbatch --parsable -J "dftrelax_${seed}" --time="${WALL_H}:00:00" \
          --ntasks="$NTASKS" --mem="$MEM" \
          --export=ALL,SEED="$seed",CHAIN=0,MAX_CHAIN="$MAX_CHAIN",WALL_H="$WALL_H",RELAX_SCRIPT="$HERE/run_relax.slurm",PACK_DIR="$PACK_DIR" \
          "$HERE/run_relax.slurm")
  echo "$seed  job $jid"
done
squeue -u "$USER"
