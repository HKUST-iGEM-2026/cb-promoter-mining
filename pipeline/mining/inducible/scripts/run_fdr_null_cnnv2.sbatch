#!/bin/bash
#SBATCH --job-name=cb_null_cnnv2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=pipeline/mining/inducible/logs/cb_null_cnnv2_%j.out
#SBATCH --error=pipeline/mining/inducible/logs/cb_null_cnnv2_%j.err

set -eo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_MAIN:?set it or source env.sh}"

cd "${PROMOTER_REPO:?set it or source env.sh}"

export PYTHONNOUSERSITE=1
if [ -n "${HF_HOME:-}" ]; then export HF_HOME; fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"   # 1 = require a pre-staged cache

python pipeline/evaluation/genome_inference.py \
  --genome pipeline/mining/inducible/model_outputs/fdr/prep/null.fasta \
  --model-path workflow/results/T21_xs/cnn_v2/best_model.pt \
  --model-type cnn_v2 \
  --batch-size 512 \
  --window 81 \
  --step 81 \
  --strands + \
  --out pipeline/mining/inducible/model_outputs/fdr/null_tracks/cnnv2.csv