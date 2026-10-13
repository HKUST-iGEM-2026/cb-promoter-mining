#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=evo2_embed
#SBATCH --output=%x_%j.out
#
# Extract frozen Evo2-7B embeddings for the T15 splits -> shared evo2_embeddings/ (GPU).
# Then train the probe on CPU:  python evo2_promoter_classifier.py
set -eo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_EVO2:?set it or source env.sh}"
if [ -n "${HF_HOME:-}" ]; then export HF_HOME; fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"   # 1 = require a pre-staged cache
export PYTHONNOUSERSITE=1

EVO2="${PROMOTER_REPO:?set it or source env.sh}/pipeline/training/Evo2"
# default --data-dir = T15_xs (the deployed cross-species split, 44,085 rows) and --out-dir = shared
# evo2_embeddings/ are baked into the extractor. Override with --data-dir only for a different split.
# add --sequence-col sequence_context to embed the long window instead of the 81 bp core.
python "$EVO2/extract_embeddings_evo2.py"
