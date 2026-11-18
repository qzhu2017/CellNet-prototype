#!/usr/bin/env bash
# Run one benchmark code with the protocol reported in the paper.
#
#   bash benchmark/run_code.sh OBEQUJ                 # -> outputs/benchmark/OBEQUJ/
#   OUTPUT_DIR=/path/to/out NPROC=32 bash benchmark/run_code.sh OBEQUJ
#   MAX_CONF_RUNS=72 bash benchmark/run_code.sh OBEQUJ   # the earlier 72-cell protocol
#   DB=EM/CTMTNA/CTMTNA.db bash benchmark/run_code.sh CTMTNA   # code from your own PyXtal db
#   SAGE=2.1 bash benchmark/run_code.sh CTMTNA            # other OpenFF Sage version (default 2.0)
#   K=192 SEED=7 MAX_CONF_RUNS=128 bash benchmark/run_code.sh CTMTNA   # more flow draws and cells
#   HALL=39 bash benchmark/run_code.sh OCHTET      # other setting of the same space group
#   SCALE_VOL=6805 FLOW_NPZ=x_flow_k192.npz bash benchmark/run_code.sh X   # cells scaled to V (diagnostic)
#   LADDER=1.0,0.88,0.77,0.68,0.6,0.52,0.45 LADDER_CELLS=16 bash benchmark/run_code.sh X   # blind density ladder
#   SPECIAL=1 bash benchmark/run_code.sh OCHTET    # keep the molecule on its special position (Fdd2 Z′ 0.5)
#   HALL=212 ZPRIME=0.5 WYCKOFF=8a bash benchmark/run_code.sh OCHTET   # chosen site, random template
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
DB="${DB:-}"
SAGE="${SAGE:-2.0}"
K="${K:-96}"
SEED="${SEED:-42}"
HALL="${HALL:-}"
SCALE_VOL="${SCALE_VOL:-}"
FLOW_NPZ="${FLOW_NPZ:-}"
LADDER="${LADDER:-}"
LADDER_CELLS="${LADDER_CELLS:-}"
ZPRIME="${ZPRIME:-}"
SPECIAL="${SPECIAL:-}"
WYCKOFF="${WYCKOFF:-}"

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
  ${DB:+--db "$DB"} \
  ${HALL:+--hall "$HALL"} \
  ${SCALE_VOL:+--scale-cells-volume "$SCALE_VOL"} \
  ${FLOW_NPZ:+--flow-npz "$FLOW_NPZ"} \
  ${LADDER:+--density-ladder "$LADDER"} \
  ${LADDER_CELLS:+--ladder-cells "$LADDER_CELLS"} \
  ${ZPRIME:+--zprime "$ZPRIME"} \
  ${SPECIAL:+--keep-special-sites} \
  ${WYCKOFF:+--wyckoff "$WYCKOFF"} \
  --checkpoint "$CKPT" \
  --output-dir "$OUTPUT_DIR" \
  --sage "$SAGE" \
  --match-ref \
  --k "$K" \
  --seed "$SEED" \
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
