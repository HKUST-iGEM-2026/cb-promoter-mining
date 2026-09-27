#!/bin/bash
# T26 — calibrate all three ensemble models (ProkBERT, CNN, NT) in one GPU job.
#
# Two envs in one job (docs/SETUP.md): ProkBERT + CNN load on the main
# $ENV_MAIN (transformers 5.3.0) env; NT needs $ENV_NT (transformers 4.57). We activate the
# main env, run prokbert + cnn, then switch to $ENV_NT and run nt.
#
# Data = 93 T9 positives (literature_promoters.tsv) + 186 curated genomic negatives
# (130 convergent-intergenic + 56 CDS-interior) from extract_negatives.py, read from this dir by the
# scripts' defaults. Each script selects unregularized vs L2 Platt(+AT) by stratified CV, then refits.
#
# Submit:  sbatch evaluation-and-calibration/calibration/run_calibration_2.sh
#SBATCH --job-name=t26_calib
#SBATCH --gpus-per-node=1
#SBATCH --requeue
#SBATCH --time=00:30:00
#SBATCH --output=t26_calib_%j.log

set -eo pipefail   # NB not -u: conda's activate scripts reference unbound vars

# --- paths -------------------------------------------------------------------
REPO="${PROMOTER_REPO:?set it or source env.sh}"
CONDA_BASE=$(conda info --base)
MAIN_ENV="${ENV_MAIN:?set it or source env.sh}"
NT_ENV="${ENV_NT:?set it or source env.sh}"

POSITIVES="$REPO/data-acquisition-and-curation/synthetic-promoters/literature_promoters.tsv"
CONV_NEG="$REPO/evaluation-and-calibration/calibration/convergent_intergenic_negatives.tsv"
CDS_NEG="$REPO/evaluation-and-calibration/calibration/cds_negatives.tsv"

# --- shared HF / hygiene (both envs) -----------------------------------------
if [ -n "${HF_HOME:-}" ]; then export HF_HOME; fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"   # 1 = require a pre-staged cache
export PYTHONNOUSERSITE=1          # ignore ~/.local so a stray user-site pkg can't shadow the env

cd "$REPO"
source "$CONDA_BASE/etc/profile.d/conda.sh"

activate () {                     # activate a prefix + set torch's libstdc++ path (docs/SETUP.md gotcha)
    conda activate "$1"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
}

# =============================================================================
# Main env (transformers 5.3.0): ProkBERT + CNN
# =============================================================================
activate "$MAIN_ENV"
echo "=== env: $CONDA_PREFIX | $(python -c 'import transformers; print("transformers", transformers.__version__)') ==="

echo "=== [1/3] ProkBERT calibration ==="
python "$REPO/evaluation-and-calibration/calibration/prokbert_calib.py" \
    --positives "$POSITIVES" --convergent-negatives "$CONV_NEG" --cds-negatives "$CDS_NEG" --out-dir ${PROMOTER_REPO}/evaluation-and-calibration/calibration/calib_2/prokbert

echo "=== [2/3] CNN calibration ==="
python "$REPO/evaluation-and-calibration/calibration/cnn_calib.py" \
    --positives "$POSITIVES" --convergent-negatives "$CONV_NEG" --cds-negatives "$CDS_NEG" --out-dir ${PROMOTER_REPO}/evaluation-and-calibration/calibration/calib_2/cnn

conda deactivate

# =============================================================================
# $ENV_NT env (transformers 4.57): NT LoRA calibration
# =============================================================================
activate "$NT_ENV"
echo "=== env: $CONDA_PREFIX | $(python -c 'import transformers; print("transformers", transformers.__version__)') ==="

echo "=== [3/3] NT calibration ==="
python "$REPO/evaluation-and-calibration/calibration/nt_calib.py" \
    --positives "$POSITIVES" --convergent-negatives "$CONV_NEG" --cds-negatives "$CDS_NEG" --out-dir ${PROMOTER_REPO}/evaluation-and-calibration/calibration/calib_2/nt

conda deactivate
echo "=== T26 done — see evaluation-and-calibration/calibration/out/{prokbert,cnn,nt}/ ==="
