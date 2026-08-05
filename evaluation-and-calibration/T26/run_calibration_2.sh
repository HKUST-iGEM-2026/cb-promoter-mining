#!/bin/bash
# T26 — calibrate all three ensemble models (ProkBERT, CNN, NT) in one GPU job.
#
# Two envs in one job (infrastructure/T4/NT_ENV.md): ProkBERT + CNN load on the main
# promoter-mining-hpc4 (transformers 5.3.0) env; NT needs nt-tf4x (transformers 4.57). We activate the
# main env, run prokbert + cnn, then switch to nt-tf4x and run nt.
#
# Data = 93 T9 positives (literature_promoters.tsv) + 186 curated genomic negatives
# (130 convergent-intergenic + 56 CDS-interior) from extract_negatives.py, read from this dir by the
# scripts' defaults. Each script selects unregularized vs L2 Platt(+AT) by stratified CV, then refits.
#
# Submit:  sbatch evaluation-and-calibration/T26/run_calibration.sh
#SBATCH --job-name=t26_calib
#SBATCH --account=igem2026
#SBATCH --partition=preempt
#SBATCH --gpus-per-node=1
#SBATCH --requeue
#SBATCH --time=00:30:00
#SBATCH --output=t26_calib_%j.log

set -eo pipefail   # NB not -u: conda's activate scripts reference unbound vars

# --- paths -------------------------------------------------------------------
REPO=/project/igem2026/cb-promoter-mining
CONDA_BASE=/cm/shared/apps/Anaconda3/2023.09-0
MAIN_ENV=/project/igem2026/envs/promoter-mining-hpc4
NT_ENV=/project/igem2026/envs/nt-tf4x

POSITIVES="$REPO/data-acquisition-and-curation/T9/literature_promoters.tsv"
CONV_NEG="$REPO/evaluation-and-calibration/T26/convergent_intergenic_negatives.tsv"
CDS_NEG="$REPO/evaluation-and-calibration/T26/cds_negatives.tsv"

# --- shared HF / hygiene (both envs) -----------------------------------------
export HF_HOME=/project/igem2026/hf_shared
export HF_HUB_OFFLINE=1            # compute nodes are offline — models must be pre-staged
export PYTHONNOUSERSITE=1          # ignore ~/.local so a stray user-site pkg can't shadow the env

cd "$REPO"
source "$CONDA_BASE/etc/profile.d/conda.sh"

activate () {                     # activate a prefix + set torch's libstdc++ path (NT_ENV.md gotcha)
    conda activate "$1"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
}

# =============================================================================
# Main env (transformers 5.3.0): ProkBERT + CNN
# =============================================================================
activate "$MAIN_ENV"
echo "=== env: $CONDA_PREFIX | $(python -c 'import transformers; print("transformers", transformers.__version__)') ==="

echo "=== [1/3] ProkBERT calibration ==="
python "$REPO/evaluation-and-calibration/T26/prokbert_calib.py" \
    --positives "$POSITIVES" --convergent-negatives "$CONV_NEG" --cds-negatives "$CDS_NEG" --out-dir /project/igem2026/cb-promoter-mining/evaluation-and-calibration/T26/calib_2/prokbert

echo "=== [2/3] CNN calibration ==="
python "$REPO/evaluation-and-calibration/T26/cnn_calib.py" \
    --positives "$POSITIVES" --convergent-negatives "$CONV_NEG" --cds-negatives "$CDS_NEG" --out-dir /project/igem2026/cb-promoter-mining/evaluation-and-calibration/T26/calib_2/cnn

conda deactivate

# =============================================================================
# nt-tf4x env (transformers 4.57): NT LoRA calibration
# =============================================================================
activate "$NT_ENV"
echo "=== env: $CONDA_PREFIX | $(python -c 'import transformers; print("transformers", transformers.__version__)') ==="

echo "=== [3/3] NT calibration ==="
python "$REPO/evaluation-and-calibration/T26/nt_calib.py" \
    --positives "$POSITIVES" --convergent-negatives "$CONV_NEG" --cds-negatives "$CDS_NEG" --out-dir /project/igem2026/cb-promoter-mining/evaluation-and-calibration/T26/calib_2/nt

conda deactivate
echo "=== T26 done — see evaluation-and-calibration/T26/out/{prokbert,cnn,nt}/ ==="
