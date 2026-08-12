#!/bin/bash
#SBATCH --account=igem2026
#SBATCH --partition=preempt
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=evo2_embed
#SBATCH --output=%x_%j.out
#
# Extract frozen Evo2-7B embeddings for the T15 splits -> shared evo2_embeddings/ (GPU).
# Then train the probe on CPU:  python evo2_promoter_classifier.py
set -eo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate /project/igem2026/envs/evo2
export HF_HOME=/project/igem2026/hf_shared      # shared team cache (NOT a personal dir); weights pre-staged, offline
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1

EVO2=/project/igem2026/cb-promoter-mining/model-training/Evo2
# default --data-dir (current xs T15_ctx) and --out-dir (shared evo2_embeddings) are baked into the script.
# add --sequence-col sequence_context to embed the long window instead of the 81 bp core.
python "$EVO2/extract_embeddings_evo2.py"
