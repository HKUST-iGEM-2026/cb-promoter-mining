#!/bin/bash
#SBATCH --job-name=cb_evo2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=pipeline/mining/inducible/logs/cb_evo2_%j.out
#SBATCH --error=pipeline/mining/inducible/logs/cb_evo2_%j.err

set -eo pipefail

# Initialize Conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate the Evo2-specific environment
conda activate "${ENV_EVO2:?set it or source env.sh}"

# Go to the project directory
cd "${PROMOTER_REPO:?set it or source env.sh}"

# Hugging Face shared cache
export PYTHONNOUSERSITE=1
if [ -n "${HF_HOME:-}" ]; then export HF_HOME; fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"   # 1 = require a pre-staged cache

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