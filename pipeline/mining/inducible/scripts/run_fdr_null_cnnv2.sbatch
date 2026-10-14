#!/bin/bash
#SBATCH --job-name=cb_null_cnnv2
#SBATCH --account=igem2026
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=pipeline/mining/inducible/logs/cb_null_cnnv2_%j.out
#SBATCH --error=pipeline/mining/inducible/logs/cb_null_cnnv2_%j.err

set -eo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate /project/igem2026/envs/promoter-mining-hpc4

cd /project/igem2026/cb-promoter-mining

export PYTHONNOUSERSITE=1
export HF_HOME=/project/igem2026/hf_shared
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1

python pipeline/evaluation/genome_inference.py \
  --genome pipeline/mining/inducible/model_outputs/fdr/prep/null.fasta \
  --model-path workflow/results/T21_xs/cnn_v2/best_model.pt \
  --model-type cnn_v2 \
  --batch-size 512 \
  --window 81 \
  --step 81 \
  --strands + \
  --out pipeline/mining/inducible/model_outputs/fdr/null_tracks/cnnv2.csv