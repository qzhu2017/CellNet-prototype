#!/bin/bash
#SBATCH --job-name=score
#SBATCH --cpus-per-task=48
#SBATCH --mem=48G  # measured peak 11-20 GB (rematch, 48 procs)
#SBATCH --time=24:00:00
# Post-hoc scoring of a blind run against two references:
#   1. the experimental crystal                         -> sg_*/rematch_<CODE>_0.7/
#   2. the experiment relaxed with the run's force field -> sg_*/rematch_sagefree_0.7/, rematch_sagefrozen_0.7/
# (2) tells "the search found it, the force field moved it" apart from "the search missed it".
# usage (from the repository root): sbatch EM/score_blind.sh OUTDIR CODE DB TAG [TAG ...]
#   scripts/run_blind.py --submit queues it automatically after the search array.
#   CODE may differ from OUTDIR's name (another polymorph of the same molecule).
set -eo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cellnet
export PATH="$HOME/CHARMM/bin:$PATH" OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-.}"  # the repository root (run_blind.py submits from there)
OUT=$1; CODE=$2; DB=$3; shift 3
python EM/rematch.py "$@" --dir "$OUT" --nproc 48 --cutoff 0.7 --ref "$CODE" --ref-db "$DB"
# A second reference for the same search (e.g. another polymorph) gets its own folder/labels.
if [ "$(basename "$OUT")" = "$CODE" ]; then REF="$OUT/ref_sage"; LAB=sage; else REF="$OUT/ref_sage_$CODE"; LAB=${CODE}_sage; fi
[ -f "$REF/exp_relaxed_box_free.cif" ] || python EM/relax_ref.py "$CODE" --db "$DB" --runs "$OUT" --out "$REF"
for M in free frozen; do
  F="$REF/exp_relaxed_box_$M.cif"
  [ -f "$F" ] && python EM/rematch.py "$@" --dir "$OUT" --nproc 48 --cutoff 0.7 --ref-cif "$F" --label $LAB$M
done
