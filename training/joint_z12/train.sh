#!/bin/bash
#SBATCH --job-name=z12-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=training/joint_z12/train_%j.out
# Joint Z'=1+2 CellNet flow, trained FROM SCRATCH (no --init-checkpoint: the shipped
# best.pt saw 90 test-target rows). Same architecture and loss settings as the shipped
# model (outputs args in checkpoints/cellnet_flow/training_history.json): 400 epochs,
# batch 128, lr 3e-4, validation split by molecule, polymorph loss. Z' is a 2-value input.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cellnet
# If the env's torch does not match the GPU driver, activate a matching torch here
# (the shipped models used torch 2.8.0+cu128 layered over the cellnet env).
python -c "import torch; assert torch.cuda.is_available(), torch.__version__; print('torch', torch.__version__, 'GPU', torch.cuda.get_device_name(0))"
export PYTHONUNBUFFERED=1
D=datasets/joint-z12
OUT=${OUT:-outputs/joint_z12/flow_scratch}
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/train_flow.py \
  --train-csv $D/train_precomputed.csv --test-csv $D/test_precomputed.csv \
  --train-graphs $D/train_precomputed_graphs.pt --test-graphs $D/test_precomputed_graphs.pt \
  --output-dir "$OUT" --epochs ${EPOCHS:-400} --batch-size 128 --lr 3e-4 --patience 400 \
  --eager-graphs --graph-hot-shards 4 --skip-test-eval --device cuda
