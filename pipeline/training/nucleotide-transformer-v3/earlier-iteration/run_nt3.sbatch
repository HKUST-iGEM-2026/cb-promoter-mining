#!/bin/bash
#SBATCH --job-name=nt3_train
#SBATCH --output=logs/nt3_train_%j.out
#SBATCH --error=logs/nt3_train_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Activate conda environment
source ~/.bashrc
conda activate ${PROMOTER_REPO}/.conda

# Run the training script
python -u pipeline/training/nucleotide-transformer-v3/earlier-iteration/train_classifier.py
