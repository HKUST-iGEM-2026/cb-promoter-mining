#!/bin/bash
#SBATCH --job-name=cb_null_prokbert
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=inducible_promoters/logs/cb_null_prokbert_%j.out
#SBATCH --error=inducible_promoters/logs/cb_null_prokbert_%j.err

set -eo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_MAIN:?set it or source env.sh}"

cd "${PROMOTER_REPO:?set it or source env.sh}"

export PYTHONNOUSERSITE=1
if [ -n "${HF_HOME:-}" ]; then export HF_HOME; fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"   # 1 = require a pre-staged cache

python evaluation-and-calibration/genome_inference.py \
  --genome inducible_promoters/model_outputs/fdr/prep/null.fasta \
  --model-path infrastructure/T3/snakemake_scaffold/results/T21_xs/prokbert/best_model \
  --model-id neuralbioinfo/prokbert-mini \
  --batch-size 512 \
  --window 81 \
  --step 81 \
  --strands + \
  --out inducible_promoters/model_outputs/fdr/null_tracks/prokbert.csv