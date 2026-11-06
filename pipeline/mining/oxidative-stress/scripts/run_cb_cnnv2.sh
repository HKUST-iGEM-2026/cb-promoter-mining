#!/bin/bash
#SBATCH --job-name=cb_cnnv2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=pipeline/mining/oxidative-stress/logs/cb_cnnv2_%j.out
#SBATCH --error=pipeline/mining/oxidative-stress/logs/cb_cnnv2_%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_MAIN:?set it or source env.sh}"

cd "${PROMOTER_REPO:?set it or source env.sh}"

export PYTHONNOUSERSITE=1

GENOME="pipeline/mining/oxidative-stress/bakta/bakta_annotation/DSM10702.fna"
REGIONS="pipeline/mining/oxidative-stress/data/regions/CB_oxidative_stress_upstream_300bp.bed"
MODEL="workflow/results/T21_xs/cnn_v2/best_model.pt"
OUT="pipeline/mining/oxidative-stress/model_outputs/cnnv2/CB_oxidative_stress_cnnv2.csv"

python pipeline/evaluation/genome_inference.py \
  --genome "$GENOME" \
  --model-path "$MODEL" \
  --model-type cnn_v2 \
  --regions "$REGIONS" \
  --window 81 \
  --step 10 \
  --out "$OUT"

echo "CNNv2 scoring complete"
wc -l "$OUT"
head "$OUT"