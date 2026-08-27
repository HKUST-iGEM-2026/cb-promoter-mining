#!/bin/bash
#SBATCH --job-name=t27_fdr_mine
#SBATCH --account=igem2026
#SBATCH --partition=preempt
#SBATCH --gpus-per-node=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --time=23:59:00
#SBATCH --output=t27_fdr_mine_%j.log
#
# T27 empirical-FDR promoter mine on the target C. butyricum, BOTH model sets (xs + 0.6).
# This sbatch is JUST the GPU resource request; ALL logic lives in fdr_mine.py (one reproducible
# Python entrypoint). fdr_mine.py subprocess-activates the right conda env per model itself
# (nt-tf4x for NT, the main env for the rest, the bio env for FIMO) — see its header for why one
# Python process cannot host both transformers versions. Resume-safe: a completed per-model track
# is skipped on a preempt requeue.
#
# SUBMIT:  sbatch evaluation-and-calibration/T27/run_fdr_mine.sbatch
set -eo pipefail

REPO=/project/igem2026/cb-promoter-mining
ANN=$REPO/evaluation-and-calibration/T26/c_butyricum_annotation
CONDA_BASE=/cm/shared/apps/Anaconda3/2023.09-0
MAIN_ENV=/project/igem2026/envs/promoter-mining-hpc4

# The orchestrator process runs in the MAIN env (pandas/numpy/Bio + shelling out); the heavy model
# scans and FIMO get their own envs from inside fdr_mine.py.
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$MAIN_ENV"

python "$REPO/evaluation-and-calibration/T27/fdr_mine.py" \
    --genome   "$ANN/C_butyricum.fna" \
    --gff      "$ANN/C_butyricum.gff3" \
    --null-tsv "$ANN/convergent_intergenic_negatives.tsv" "$ANN/../c_acetobutylicum_annotation/convergent_intergenic_negatives.tsv" "$ANN/../c_phytofermentans_annotation/convergent_intergenic_negatives.tsv"\
    --model-root "$REPO/infrastructure/T3/snakemake_scaffold/results" \
    --model-set both \
    --out-dir "$REPO/evaluation-and-calibration/T27/results" \
    --conda-base "$CONDA_BASE" \
    --main-env "$MAIN_ENV" \
    --nt-env  /project/igem2026/envs/nt-tf4x \
    --bio-env /project/igem2026/envs/meme

echo "=== DONE ===  see evaluation-and-calibration/T27/results/cbut_{xs,06}/"
