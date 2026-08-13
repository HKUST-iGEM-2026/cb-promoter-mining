#!/bin/bash
#SBATCH --job-name=nt3_train
#SBATCH --output=logs/nt3_train_%j.out
#SBATCH --error=logs/nt3_train_%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=igem2026

# Activate conda environment
source ~/.bashrc
conda activate /project/igem2026/cb-promoter-mining/.conda

# Run the training script
python -u scripts/train_classifier.py
