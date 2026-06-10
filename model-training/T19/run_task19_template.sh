#!/bin/bash
#SBATCH --job-name=task19_nt_lora
#SBATCH --account=igem2026
#SBATCH --partition=preempt
#SBATCH --gpus-per-node=1
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/task19_%j.out

# Load modules (no --mem/--cpus on GPU jobs — SuperPOD auto-allocates per H800)
module load <conda-module>   # exact name from `module avail` (e.g. miniforge3/anaconda3)
module load cuda/12.1

# Use the shared T4 env (built once on /project) instead of a per-job env.
conda activate /project/igem2026/envs/promoter-mining-hpc4

# Dependencies come from that env. Kept as a version reference — note SuperPOD's H800
# wants CUDA 12.x, so the cu118 torch pin below is superseded by the T4 env (torch + cu121):
#   torch==2.0.1 (cu118)  transformers==4.35.0  peft==0.6.0  bitsandbytes==0.41.0
#   accelerate==0.24.0  datasets optuna wandb scikit-learn pandas pyarrow

# Run training with memory optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
python train_nt_lora.py

# Save model and results
tar -czf nt_lora_model.tar.gz models/nt_lora_promoter/
