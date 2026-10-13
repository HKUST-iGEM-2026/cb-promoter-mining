#!/bin/bash
#SBATCH --job-name=ntv3_ctx_finetune
#SBATCH --gpus-per-node=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --time=23:59:00
#SBATCH --output=ntv3_ctx_finetune_%j.log
#
# Full fine-tune of NTv3 (650M) as a PROMOTER CLASSIFIER on the context split, using
# pipeline/training/nucleotide-transformer-v3/current/finetune_ntv3_classifier.py (backbone + mean-pool + linear head, weighted CE on the
# `weight` column, weighted-AUPRC metric). Replaces the degenerate MLM run in results_ntv3_full.
#
# DATA: defaults to T15_ctx (ready today: sequence_context/label/weight all present, ctx=384 bp).
#       Point DATA at T15_xs_ctx once the xs+context set is built to add the out-of-genus tail.
# ENV : the HF/torch env with the NTv3 remote-code deps (NOT ntv3-jax — that's the separate JAX pkg).
#       Defaults to the main env; if AutoModel's trust_remote_code import errors on the transformers
#       version, switch ENV to $ENV_NT and change `eval_strategy`->`evaluation_strategy` is NOT needed
#       (the script uses eval_strategy = transformers 5.x, which the main env has).
#
# PREREQ (login node, has internet — compute nodes are OFFLINE): pre-stage the base weights into the
# HF cache this job reads, or the offline load fails:
#     export HF_HOME=/shared/hf_cache      # optional; defaults to ~/.cache/huggingface
#     hf download InstaDeepAI/NTv3_650M_pre
#
# SUBMIT:  sbatch pipeline/training/nucleotide-transformer-v3/current/finetune_ntv3.sbatch
#   resume after a preempt:  RESUME=1 sbatch pipeline/training/nucleotide-transformer-v3/current/finetune_ntv3.sbatch
set -eo pipefail

REPO="${PROMOTER_REPO:?set it or source env.sh}"
DIR=$REPO/workflow/results
ENV=${ENV:-${ENV_MAIN}}   # HF/torch env (override: ENV=.../$ENV_NT)
CONDA_BASE=$(conda info --base)

# ---- point these two as needed ----
DATA=${DATA:-$DIR/T15_ctx}                 # ready now; swap to $DIR/T15_xs_ctx for the xs composition
OUT=${OUT:-$DIR/ntv3_ctx_classifier}
MAXLEN=${MAXLEN:-384}                       # MUST equal the context length in DATA (T15_ctx = 384)
# -----------------------------------

if [ -n "${HF_HOME:-}" ]; then export HF_HOME; fi   # where NTv3_650M_pre was pre-staged
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-/tmp/hf_notoken_$USER}   # dodge the unreadable shared-cache token file
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 MKL_THREADING_LAYER=GNU
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV"

echo "=== preflight ==="
for f in train val test; do
    [[ -f "$DATA/$f.parquet" ]] || { echo "MISSING $DATA/$f.parquet"; exit 1; }
done
python -c "import torch; assert torch.cuda.is_available(), 'NO GPU'; print('GPU:', torch.cuda.get_device_name(0))"
echo "data=$DATA  out=$OUT  max_length=$MAXLEN  env=$ENV"

# batch 4 x grad-accum 8 = eff. 32; NTv3-650M full fine-tune with all hidden states retained is
# memory-heavy (sized for an 80 GB GPU) — if it OOMs, drop --batch-size to 2 (raise --grad-accum to 16 to keep eff. 32).
python "$REPO/pipeline/training/nucleotide-transformer-v3/current/finetune_ntv3_classifier.py" \
    --data-dir "$DATA" --model-id InstaDeepAI/NTv3_650M_pre \
    --seq-col sequence_context --max-length "$MAXLEN" \
    --out "$OUT" --epochs 3 --batch-size 4 --grad-accum 8 --lr 1e-5 \
    ${RESUME:+--resume}

echo "=== DONE ===  weighted val/test AUPRC printed above; model -> $OUT/final_model"
