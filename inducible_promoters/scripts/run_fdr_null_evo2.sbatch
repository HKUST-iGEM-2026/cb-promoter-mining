#!/bin/bash
#SBATCH --job-name=cb_null_evo2
#SBATCH --account=igem2026
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=inducible_promoters/logs/cb_null_evo2_%j.out
#SBATCH --error=inducible_promoters/logs/cb_null_evo2_%j.err

set -eo pipefail

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate /project/igem2026/envs/evo2

cd /project/igem2026/cb-promoter-mining

export PYTHONNOUSERSITE=1
export HF_HOME=/project/igem2026/hf_shared
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1

python model-training/Evo2/evo2_scan_restricted.py \
  --genome inducible_promoters/model_outputs/fdr/prep/null.fasta \
  --regions inducible_promoters/model_outputs/fdr/prep/null_regions.bed \
  --probe infrastructure/T3/snakemake_scaffold/results/regimeC/evo2/evo2_probe_xs.pkl \
  --window 81 \
  --step 81 \
  --batch-size 16 \
  --strands + \
  --align-grid \
  --out inducible_promoters/model_outputs/fdr/null_tracks/evo2.csv