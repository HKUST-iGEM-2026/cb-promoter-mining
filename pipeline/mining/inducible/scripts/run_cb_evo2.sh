#!/bin/bash
#SBATCH --job-name=cb_evo2
#SBATCH --account=igem2026
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=pipeline/mining/inducible/logs/cb_evo2_%j.out
#SBATCH --error=pipeline/mining/inducible/logs/cb_evo2_%j.err

set -eo pipefail

# Initialize Conda
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh

# Activate the Evo2-specific environment
conda activate /project/igem2026/envs/evo2

# Go to the project directory
cd /project/igem2026/cb-promoter-mining

# Hugging Face shared cache
export PYTHONNOUSERSITE=1
export HF_HOME=/project/igem2026/hf_shared
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1

# Input/output paths
GENOME="pipeline/mining/inducible/bakta/bakta_annotation/DSM10702.fna"
REGIONS="pipeline/mining/inducible/data/regions/CB_oxidative_stress_upstream_300bp.bed"
PROBE="workflow/results/regimeC/evo2/evo2_probe_xs.pkl"
OUT="pipeline/mining/inducible/model_outputs/evo2/CB_oxidative_stress_evo2.csv"

echo "Starting Evo2 scoring..."
echo "Python: $(which python)"
echo "Genome: $GENOME"
echo "Regions: $REGIONS"
echo "Probe: $PROBE"
echo "Output: $OUT"

echo "GPU information:"
nvidia-smi

python pipeline/training/Evo2/evo2_scan_restricted.py \
  --genome "$GENOME" \
  --regions "$REGIONS" \
  --probe "$PROBE" \
  --window 81 \
  --step 10 \
  --batch-size 16 \
  --strands "+,-" \
  --align-grid \
  --out "$OUT"

echo "Evo2 scoring complete."

wc -l "$OUT"
head "$OUT"