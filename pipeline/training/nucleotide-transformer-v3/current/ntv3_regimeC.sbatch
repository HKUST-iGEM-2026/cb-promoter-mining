#!/bin/bash
#SBATCH --job-name=ntv3_regimeC
#SBATCH --gpus-per-node=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --time=11:59:00
#SBATCH --output=ntv3_regimeC_%j.log
#
# Regime C for NTv3-ctx: restricted scans of two benchmark genomes at step 10.
# Published TSS labels are withheld, but both genomes supplied silver/negative
# training windows. This is not wholly unseen-genome or leak-free evaluation.
# NTv3 reads 384 bp; the benchmark uses its reported 81-bp core coordinates.
#
# PREREQ: the Regime-C genome/region/truth files must already be staged in $RC by the
# Evo2 / cnn_v2 runs (prep_cau_regimeC.sh, C_ljung_scan_xs.sh). Needs clj.fna, cau.fna,
# {clj,cau}_candidate_regions.bed, cau_truth.tsv. ljungdahlii truth is committed in $EVAL.
# Scan scripts (ntv3_infer.py, ntv3_genome_scan.py) are read from ~ (scp'd there).
#
# SUBMIT:  sbatch pipeline/training/nucleotide-transformer-v3/current/ntv3_regimeC.sbatch
#   (uses the xs model by default; MODEL=... to point elsewhere, STEP=1 for a denser scan)
set -eo pipefail

REPO="${PROMOTER_REPO:?set it or source env.sh}"
DIR=$REPO/workflow/results
EVAL=$REPO/pipeline/evaluation
RC=$DIR/regimeC
ENV=${ENV:-${ENV_MAIN}}
MODEL=${MODEL:-$DIR/ntv3_xs_ctx_classifier/final_model}
STEP=${STEP:-10}
BS=${BS:-32}
OUT=$RC/ntv3
mkdir -p "$OUT"

if [ -n "${HF_HOME:-}" ]; then export HF_HOME; fi
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-/tmp/hf_notoken_$USER}
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 MKL_THREADING_LAYER=GNU
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

python -c "import torch; assert torch.cuda.is_available(), 'NO GPU'; print('GPU:', torch.cuda.get_device_name(0))"
echo "model=$MODEL  step=$STEP  out=$OUT"

# ---- C. ljungdahlii (published benchmark TSS withheld; truth committed in $EVAL) ----
python ~/ntv3_genome_scan.py --genome "$RC/clj.fna" --model-dir "$MODEL" \
    --regions "$RC/clj_candidate_regions.bed" --window 81 --context 384 --step "$STEP" \
    --batch-size "$BS" --out "$OUT/clj_ntv3_track.csv"
python "$EVAL/genome_fp_benchmark.py" --scores "$OUT/clj_ntv3_track.csv" \
    --truth "$EVAL/ljungdahlii_holdout_tss.tsv" --recall-targets 0.1,0.5,0.7,0.9 \
    --regions "$RC/clj_candidate_regions.bed" --out "$OUT/fp_clj_ntv3_restrict"

# ---- C. autoethanogenum (published benchmark TSS withheld; genome partly training-exposed) ----
python ~/ntv3_genome_scan.py --genome "$RC/cau.fna" --model-dir "$MODEL" \
    --regions "$RC/cau_candidate_regions.bed" --window 81 --context 384 --step "$STEP" \
    --batch-size "$BS" --out "$OUT/cau_ntv3_track.csv"
python "$EVAL/genome_fp_benchmark.py" --scores "$OUT/cau_ntv3_track.csv" \
    --truth "$RC/cau_truth.tsv" --recall-targets 0.1,0.5,0.7,0.9 \
    --regions "$RC/cau_candidate_regions.bed" --out "$OUT/fp_cau_ntv3_restrict"

echo "=== DONE === restricted precision@recall:"
echo "  ljungdahlii -> $OUT/fp_clj_ntv3_restrict/genome_fp_benchmark.md"
echo "  autoethanogenum -> $OUT/fp_cau_ntv3_restrict/genome_fp_benchmark.md"
echo "Compare precision@recall 0.1/0.5 to cnn_v2 (63.1%/48.2% clj, 81.2%/67.1% cau) and Evo2 (71.0%/58.0% clj, 85.8%/71.4% cau)."
