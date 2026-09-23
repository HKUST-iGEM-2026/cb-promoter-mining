#!/bin/bash
#SBATCH --job-name=nt3_smoke_test
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --account=igem2026
#SBATCH --output=logs/nt3_smoke_%j.out
#SBATCH --error=logs/nt3_smoke_%j.err

# Print basic node and environment info for debugging
echo "Running on host: $(hostname)"
echo "Assigned GPU: $CUDA_VISIBLE_DEVICES"

# Activate your project conda environment
conda activate /project/igem2026/cb-promoter-mining/.conda

# Run a simple python smoke test
python -c "import sys; print('Python version:', sys.version); print('Environment active and running!')"