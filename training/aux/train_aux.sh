#!/bin/bash
#SBATCH --job-name=aux-sg-dens
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=training/aux/train_%j.out
# Auxiliary model: space-group head p(Hall, Z'|molecule) + 3-Gaussian density mixture
# p(log rho | molecule, Hall, Z'), trained on the leak-free joint Z'=1+2 data.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate cellnet
# If the env's torch does not match the GPU driver, activate a matching torch here
# (the shipped models used torch 2.8.0+cu128 layered over the cellnet env).
export PYTHONUNBUFFERED=1
python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
python scripts/train_aux.py --out outputs/aux_sg_density --epochs 60 --patience 8 --device cuda
