#!/bin/bash
# T26 — calibrate all 4 ensemble models (ProkBERT, CNN, CNN-v2, NT) in one GPU job.
# All models are from the 0.6 clustering retrain
#
# Two envs in one job (docs/SETUP.md): ProkBERT + CNN + CNN-v2 load on the main
# $ENV_MAIN (transformers 5.3.0) env; NT needs $ENV_NT (transformers 4.57). We activate the
# main env, run prokbert + cnn, then switch to $ENV_NT and run nt.
#
# Data = 93 T9 positives (literature_promoters.tsv) + 186 curated genomic negatives
# (130 convergent-intergenic + 56 CDS-interior) from extract_negatives.py, read from this dir by the
# scripts' defaults. Each script selects unregularized vs L2 Platt(+AT) by stratified CV, then refits.
#
# Submit:  sbatch pipeline/evaluation/calibration/run_calibration_06.sh
#SBATCH --job-name=t26_calib_3
#SBATCH --gpus-per-node=1
#SBATCH --requeue
#SBATCH --time=00:30:00
#SBATCH --output=t26_0.6_calib_%j.log

set -eo pipefail   # NB not -u: conda's activate scripts reference unbound vars

# --- paths -------------------------------------------------------------------
REPO="${PROMOTER_REPO:?set it or source env.sh}"
CONDA_BASE=$(conda info --base)
MAIN_ENV="${ENV_MAIN:?set it or source env.sh}"
NT_ENV="${ENV_NT:?set it or source env.sh}"

POSITIVES="$REPO/pipeline/acquisition/synthetic-promoters/literature_promoters.tsv"
CONV_NEG="$REPO/pipeline/evaluation/calibration/convergent_intergenic_negatives.tsv"
CDS_NEG="$REPO/pipeline/evaluation/calibration/cds_negatives.tsv"

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
# Main env (transformers 5.3.0): SRS negatives + ProkBERT + CNN
# =============================================================================
activate "$MAIN_ENV"
echo "=== env: $CONDA_PREFIX | $(python -c 'import transformers; print("transformers", transformers.__version__)') ==="

echo "=== [1/4] ProkBERT calibration ==="
python "$REPO/pipeline/evaluation/calibration/prokbert_calib.py" \
    --model-dir ${PROMOTER_REPO}/workflow/results/T21_final_06/prokbert/best_model --positives "$POSITIVES" --convergent-negatives "$CONV_NEG" --cds-negatives "$CDS_NEG" --out-dir ${PROMOTER_REPO}/pipeline/evaluation/calibration/calib_06/prokbert

echo "=== [2/4] CNN calibration ==="
python "$REPO/pipeline/evaluation/calibration/cnn_calib.py" \
    --model-dir ${PROMOTER_REPO}/workflow/results/T21_final_06/cnn --positives "$POSITIVES" --convergent-negatives "$CONV_NEG" --cds-negatives "$CDS_NEG" --out-dir ${PROMOTER_REPO}/pipeline/evaluation/calibration/calib_06/cnn

echo "=== [3/4] CNN-v2 calibration ==="
python "$REPO/pipeline/evaluation/calibration/cnn_v2_calib.py" \
    --model-dir ${PROMOTER_REPO}/workflow/results/T21_final_06/cnn_v2 --positives "$POSITIVES" --convergent-negatives "$CONV_NEG" --cds-negatives "$CDS_NEG" --out-dir ${PROMOTER_REPO}/pipeline/evaluation/calibration/calib_06/cnn_v2

conda deactivate

# =============================================================================
# $ENV_NT env (transformers 4.57): NT LoRA calibration
# =============================================================================
activate "$NT_ENV"
echo "=== env: $CONDA_PREFIX | $(python -c 'import transformers; print("transformers", transformers.__version__)') ==="

echo "=== [4/4] NT calibration ==="
python "$REPO/pipeline/evaluation/calibration/nt_calib.py" \
    --model-dir ${PROMOTER_REPO}/workflow/results/T21_final_06/nt_lora/best_adapter --positives "$POSITIVES" --convergent-negatives "$CONV_NEG" --cds-negatives "$CDS_NEG" --out-dir ${PROMOTER_REPO}/pipeline/evaluation/calibration/calib_06/nt_lora

conda deactivate
echo "=== T26 done — see pipeline/evaluation/calibration/out/{prokbert,cnn,nt}/ ==="
