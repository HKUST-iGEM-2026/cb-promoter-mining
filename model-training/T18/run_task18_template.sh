#!/bin/bash
#SBATCH --job-name=task18_prokbert
#SBATCH --account=igem2026
#SBATCH --partition=preempt
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/task18_%j.out

# Load modules (no --mem/--cpus on GPU jobs — SuperPOD auto-allocates per H800)
module load <conda-module>   # exact name from `module avail` (e.g. miniforge3/anaconda3)
module load cuda/12.1
conda activate /project/igem2026/envs/promoter-mining-hpc4

# Dependencies come from the shared T4 env activated above — no per-job pip install.
# (was: pip install transformers datasets accelerate optuna wandb scikit-learn)

# Run training
python train_prokbert.py

# Save model
tar -czf prokbert_model.tar.gz models/prokbert_promoter/
