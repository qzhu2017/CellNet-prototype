#!/bin/bash
#SBATCH --job-name=z12-precompute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=training/joint_z12/precompute_%j.out
# Precompute training caches (lattice targets + molecular graphs) for the joint Z'=1+2
# tables in datasets/joint-z12/, then shard the train graph sidecar (10k per shard) so the
# GPU job can stream it. Resumable: resubmit and finished parts are reused.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cellnet
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
D=datasets/joint-z12
for s in test train; do
  if [ -f $D/${s}_precomputed_graphs.pt ] && [ -f $D/${s}_precomputed.csv ]; then
    echo "$s: cache exists, skipping"; continue
  fi
  python scripts/precompute_lattice_cache.py --input $D/$s.csv --workers 48 --resume --no-progress
done
for s in train test; do
  if [ ! -f $D/${s}_precomputed_graphs_shards.json ]; then
    python scripts/shard_graph_sidecar.py --input $D/${s}_precomputed_graphs.pt --shard-size 10000
  fi
done
python - <<'PY'
import csv
for s in ("train", "test"):
    rows = list(csv.DictReader(open(f"datasets/joint-z12/{s}_precomputed.csv")))
    z = {}
    for r in rows: z[r["zprime"]] = z.get(r["zprime"], 0) + 1
    print(f"{s}: {len(rows):,} precomputed rows, by Z': {z}")
PY
