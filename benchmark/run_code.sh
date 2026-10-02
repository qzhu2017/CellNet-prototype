#!/usr/bin/env bash
# Run one benchmark code with the protocol reported in the paper.
#
#   bash benchmark/run_code.sh OBEQUJ                 # -> outputs/benchmark/OBEQUJ/
#   OUTPUT_DIR=/path/to/out NPROC=32 bash benchmark/run_code.sh OBEQUJ
#   MAX_CONF_RUNS=72 bash benchmark/run_code.sh OBEQUJ   # the earlier 72-cell protocol
#
# Protocol: K=96 flow draws (seed 42); lattice QRS with 12 Sobol stages x 1024 points;
# 64 cells chosen by the cell-blind multichannel selection (lambda consistency 6,
# QRS loss 6, density 4, Selling disagreement 4, monoclinic unique-axis alternates 12,
# shape diversity for the rest); one conformational QRS per cell (20 generations x 96)
# relaxed with CHARMM + OpenFF Sage 2.0.0, lattice free; if no cell matches the
# experimental structure, the same 64 cells are searched again with the lattice frozen.
# All 64 cells are searched (no early stop) so hit counts are comparable.
#
# Structures come from datasets/test.db if present, else from the test.db shipped with
# PyXtal. Needs charmm on PATH. One code takes hours on 48 cores (see README).

set -euo pipefail

CODE="${1:?usage: run_code.sh CSD_CODE}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/benchmark}"
NPROC="${NPROC:-48}"
CKPT="${CKPT:-${ROOT}/checkpoints/cellnet_flow/best.pt}"
MAX_CONF_RUNS="${MAX_CONF_RUNS:-64}"

# Parallelism comes from NPROC worker processes. Without these limits every worker
# starts MKL/OpenMP/torch thread pools sized to the whole node (~110 threads each).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "${OUTPUT_DIR}/${CODE}"
cd "$ROOT"

python scripts/run_pipeline.py \
  --csd-code "$CODE" \
  --checkpoint "$CKPT" \
  --output-dir "$OUTPUT_DIR" \
  --sage 2.0 \
  --match-ref \
  --k 96 \
  --seed 42 \
  --qrs-stages 12 \
  --qrs-samples-per-stage 1024 \
  --lattice-qrs-nproc "$(( NPROC > 1 ? NPROC / 2 : 1 ))" \
  --max-lattice-qrs-lambda-ratio 8.0 \
  --conf-ngen 20 \
  --conf-npop 96 \
  --conf-nproc "$NPROC" \
  --max-conf-runs "$MAX_CONF_RUNS" \
  --conf-selection multichannel \
  --conf-channel-lambda 6 \
  --conf-channel-qrs-loss 6 \
  --conf-channel-density 4 \
  --conf-channel-disagreement 4 \
  --conf-channel-axis-alternates 12 \
  --axis-alternate-top 6 \
  --conf-qrs-origin-only \
  --conf-restarts 1 \
  --conf-seed-stride 100000 \
  --cell-blind \
  --relax-lattice \
  --adaptive-relax \
  --no-conf-check-stable \
  2>&1 | tee "${OUTPUT_DIR}/${CODE}/pipeline.log"
