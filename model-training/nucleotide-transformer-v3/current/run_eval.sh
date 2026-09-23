#!/bin/bash
#SBATCH --job-name=ntv3_eval
#SBATCH --output=logs/ntv3_eval_%j.out
#SBATCH --error=logs/ntv3_eval_%j.err
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --account=igem2026

# Activate your environment
source ~/.bashrc
conda activate ntv3_env

# Run the evaluation script
python -u model-training/nucleotide-transformer-v3/earlier-iteration/evaluate_ntv3.py
