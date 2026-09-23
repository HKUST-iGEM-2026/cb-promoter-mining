#!/bin/bash
#SBATCH --job-name=nt3_finetune
#SBATCH --output=logs/nt3_finetune_%j.out
#SBATCH --error=logs/nt3_finetune_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --account=igem2026

mkdir -p logs

echo "Running on host: $(hostname)"
nvidia-smi

# Directly execute using the conda environment's python binary
/project/igem2026/cb-promoter-mining/.conda/bin/python -u model-training/nucleotide-transformer-v3/earlier-iteration/fine_tune_nt3.py
