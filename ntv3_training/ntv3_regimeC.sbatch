#!/bin/bash
#SBATCH --job-name=ntv3_regimeC
#SBATCH --account=igem2026
#SBATCH --partition=preempt
#SBATCH --gpus-per-node=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --time=11:59:00
#SBATCH --output=ntv3_regimeC_%j.log
#
# Regime C for NTv3-ctx: restricted genome scan of the two held-out validation genomes
# (C. ljungdahlii = in-genus held-out; C. autoethanogenum = clean, never in training),
# then FP-peak benchmark vs their TSS at matched --step 10. Leak-immune (a different
# genome), so this is the number to compare against cnn_v2 / Evo2 restricted results.
#
# PREREQ: the Regime-C genome/region/truth files must already be staged in $RC by the
# Evo2 / cnn_v2 runs (prep_cau_regimeC.sh, C_ljung_scan_xs.sh). Needs clj.fna, cau.fna,
# {clj,cau}_candidate_regions.bed, cau_truth.tsv. ljungdahlii truth is committed in $EVAL.
# Scan scripts (ntv3_infer.py, ntv3_genome_scan.py) are read from ~ (scp'd there).
#
# SUBMIT:  sbatch ntv3_training/ntv3_regimeC.sbatch
#   (uses the xs model by default; MODEL=... to point elsewhere, STEP=1 for a denser scan)
set -eo pipefail

REPO=/project/igem2026/cb-promoter-mining
DIR=$REPO/infrastructure/T3/snakemake_scaffold/results
EVAL=$REPO/evaluation-and-calibration
RC=$DIR/regimeC
ENV=${ENV:-/project/igem2026/envs/promoter-mining-hpc4}
MODEL=${MODEL:-$DIR/ntv3_xs_ctx_classifier/final_model}
STEP=${STEP:-10}
BS=${BS:-32}
OUT=$RC/ntv3
mkdir -p "$OUT"

export HF_HOME=${HF_HOME:-/project/igem2026/hf_$USER}
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-/tmp/hf_notoken_$USER}
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 MKL_THREADING_LAYER=GNU
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate "$ENV"

python -c "import torch; assert torch.cuda.is_available(), 'NO GPU'; print('GPU:', torch.cuda.get_device_name(0))"
echo "model=$MODEL  step=$STEP  out=$OUT"

# ---- C. ljungdahlii (held-out in-genus validation genome; truth committed in $EVAL) ----
python ~/ntv3_genome_scan.py --genome "$RC/clj.fna" --model-dir "$MODEL" \
    --regions "$RC/clj_candidate_regions.bed" --window 81 --context 384 --step "$STEP" \
    --batch-size "$BS" --out "$OUT/clj_ntv3_track.csv"
python "$EVAL/genome_fp_benchmark.py" --scores "$OUT/clj_ntv3_track.csv" \
    --truth "$EVAL/ljungdahlii_holdout_tss.tsv" --recall-targets 0.1,0.5,0.7,0.9 \
    --regions "$RC/clj_candidate_regions.bed" --out "$OUT/fp_clj_ntv3_restrict"

# ---- C. autoethanogenum (clean; never in training) ----
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
