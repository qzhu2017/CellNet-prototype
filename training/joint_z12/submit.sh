#!/usr/bin/env bash
# Precompute caches (CPU), then train (GPU) once the caches exist.
#   bash training/joint_z12/submit.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
P=$(sbatch --parsable training/joint_z12/precompute.sh)
T=$(sbatch --parsable --dependency=afterok:$P training/joint_z12/train.sh)
echo "joint z12 precompute $P, train $T (after $P)"
