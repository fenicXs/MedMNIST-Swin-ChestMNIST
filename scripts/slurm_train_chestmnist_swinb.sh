#!/bin/bash
#SBATCH --job-name=chestmnist_swinb
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# NOTE:
# - Adjust partition / qos / gres to match ASU SOL.
# - For multi-GPU DDP: set --gres=gpu:<N> and use torchrun --nproc_per_node=<N>.

set -euo pipefail

mkdir -p logs

# Activate your environment
# module load cuda/12.1  # (example)
# source ~/.bashrc
# conda activate <env>
# or: source .venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python -m src.train --config configs/chestmnist_swinb_224.yaml
