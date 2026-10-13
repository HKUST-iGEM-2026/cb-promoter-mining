#!/bin/bash
#SBATCH --job-name=cb_null_evo2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=pipeline/mining/inducible/logs/cb_null_evo2_%j.out
#SBATCH --error=pipeline/mining/inducible/logs/cb_null_evo2_%j.err

set -eo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_EVO2:?set it or source env.sh}"

cd "${PROMOTER_REPO:?set it or source env.sh}"

export PYTHONNOUSERSITE=1
if [ -n "${HF_HOME:-}" ]; then export HF_HOME; fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"   # 1 = require a pre-staged cache

python pipeline/training/Evo2/evo2_scan_restricted.py \
  --genome pipeline/mining/inducible/model_outputs/fdr/prep/null.fasta \
  --regions pipeline/mining/inducible/model_outputs/fdr/prep/null_regions.bed \
  --probe workflow/results/regimeC/evo2/evo2_probe_xs.pkl \
  --window 81 \
  --step 81 \
  --batch-size 16 \
  --strands + \
  --align-grid \
  --out pipeline/mining/inducible/model_outputs/fdr/null_tracks/evo2.csv