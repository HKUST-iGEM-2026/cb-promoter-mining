#!/bin/bash
#SBATCH --account=igem2026
#SBATCH --partition=preempt
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=extract_embeddings
#SBATCH --output=%x_%j.out

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate /project/igem2026/envs/evo2
export HF_HOME=/project/igem2026/xzhangjj/hf_cache
export HF_HUB_OFFLINE=1

python /project/igem2026/cb-promoter-mining/extract_embeddings.py
