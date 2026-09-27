#!/bin/bash
#SBATCH --job-name=cb_prokbert
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=inducible_promoters/logs/cb_prokbert_%j.out
#SBATCH --error=inducible_promoters/logs/cb_prokbert_%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_MAIN:?set it or source env.sh}"
cd "${PROMOTER_REPO:?set it or source env.sh}"

export PYTHONNOUSERSITE=1
if [ -n "${HF_HOME:-}" ]; then export HF_HOME; fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"   # 1 = require a pre-staged cache

GENOME="inducible_promoters/bakta/bakta_annotation/DSM10702.fna"
REGIONS="inducible_promoters/data/regions/CB_oxidative_stress_upstream_300bp.bed"
MODEL="infrastructure/T3/snakemake_scaffold/results/T21_xs/prokbert/best_model"
OUT="inducible_promoters/model_outputs/prokbert/CB_oxidative_stress_prokbert.csv"

echo "Starting ProkBERT scoring..."
echo "Genome: $GENOME"
echo "Regions: $REGIONS"
echo "Model: $MODEL"
echo "Output: $OUT"

python evaluation-and-calibration/genome_inference.py \
  --genome "$GENOME" \
  --model-id neuralbioinfo/prokbert-mini \
  --model-path "$MODEL" \
  --model-type transformer \
  --regions "$REGIONS" \
  --window 81 \
  --step 10 \
  --out "$OUT"

echo "Finished ProkBERT scoring."

wc -l "$OUT"
head "$OUT"