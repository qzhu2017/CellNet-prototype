#!/usr/bin/env bash
# Slurm array over benchmark/codes.txt, one code per task. Adapt the #SBATCH lines
# (partition, account, environment activation) to your cluster, then:
#
#   mkdir -p outputs/benchmark
#   sbatch --array=0-83 benchmark/submit_slurm.sh
#   python benchmark/summarize.py --root outputs/benchmark --compare benchmark/results.csv
#
#SBATCH --job-name=cellnet-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=outputs/benchmark/slurm_%A_%a.out

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

# Batch shells don't read ~/.bashrc, so load conda's shell hook first.
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cellnet   # <- activate the environment from the README here
set -u

mapfile -t CODES < <(grep -vE '^#|^$' benchmark/codes.txt)
CODE="${CODES[${SLURM_ARRAY_TASK_ID:-0}]}"

export PYTHONUNBUFFERED=1
NPROC="${SLURM_CPUS_PER_TASK:-48}" bash benchmark/run_code.sh "$CODE"
