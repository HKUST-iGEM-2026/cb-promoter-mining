#!/bin/bash
#SBATCH --job-name=ntv3_extract
#SBATCH --output=logs/ntv3_extract_%j.out
#SBATCH --error=logs/ntv3_extract_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --account=igem2026

# Activate environment
source ~/.bashrc
conda activate ntv3_env

# Run embedding extraction
python -u scripts/extract_embeddings.py
