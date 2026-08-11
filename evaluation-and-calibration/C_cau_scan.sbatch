#!/bin/bash
#SBATCH --job-name=cau_scan_bxs
#SBATCH --account=igem2026
#SBATCH --partition=preempt
#SBATCH --gpus-per-node=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --time=23:59:00
#SBATCH --output=cau_scan_bxs_%j.log
#
# 2nd validation genome (C. autoethanogenum DSM 10061, GCF_000484505.1) — base-vs-xs genome-scale FP,
# a de-noised replicate of the ljungdahlii result for the mining-at-recall-0.1 decision. Scans all FOUR
# models in BOTH variants (base = no cross-species aug; xs = cross-species aug) = 8 tracks, benchmarking
# each RAW at recall 0.1/0.5/0.7/0.9 (full + restricted) right after it scans, cnn_v2 (slow) last so the
# fast models' 0.1 numbers land first. Resume-safe: a completed track is skipped on requeue.
#
# PREREQ:  bash prep_cau_regimeC.sh   (makes cau.fna, cau_truth.tsv, cau_candidate_regions.bed)
# SUBMIT:  sbatch C_cau_scan.sbatch
set -eo pipefail

REPO=/project/igem2026/cb-promoter-mining
DIR=$REPO/infrastructure/T3/snakemake_scaffold/results
EVAL=$REPO/evaluation-and-calibration
OUT=$DIR/regimeC
MAIN_ENV=/project/igem2026/envs/promoter-mining-hpc4
NT_ENV=/project/igem2026/envs/nt-tf4x
CONDA_BASE=/cm/shared/apps/Anaconda3/2023.09-0
FASTA=$OUT/cau.fna
TRUTH=$OUT/cau_truth.tsv
REGIONS=$OUT/cau_candidate_regions.bed
RECALLS=0.1,0.5,0.7,0.9
ANALYSIS=$OUT/cau_base_vs_xs_analysis.md

export HF_HOME=/project/igem2026/hf_shared HF_HUB_OFFLINE=1 HF_TOKEN_PATH=/tmp/hf_notoken_$USER
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ============================ MODEL PATHS — CONFIRM BEFORE SUBMITTING ============================
# These MUST be the SAME checkpoints you scanned for clj_<model>_{base,xs}.csv on ljungdahlii, or
# genome-1 and genome-2 are not comparable. Recover the exact paths from that job:
#     grep -E "model-path" <your ljungdahlii base-vs-xs job>.sh
# Pre-filled: base = the 0.6 models WITHOUT cross-species aug (T21_final_06, as in C_ljung_scan_06.sh);
#             xs   = the cross-species-augmented models (T21_xs). Edit if yours differ.
declare -A MPATH=(
    [prokbert_base]="$DIR/T21_final_06/prokbert/best_model"
    [prokbert_xs]="$DIR/T21_xs/prokbert/best_model"
    [nt_base]="$DIR/T21_final_06/nt_lora/best_adapter"
    [nt_xs]="$DIR/T21_xs/nt_lora/best_adapter"
    [cnn_base]="$DIR/T21_final_06/cnn/best_model.pt"
    [cnn_xs]="$DIR/T21_xs/cnn/best_model.pt"
    [cnn_v2_base]="$DIR/T21_final_06/cnn_v2/best_model.pt"
    [cnn_v2_xs]="$DIR/T21_xs/cnn_v2/best_model.pt"
)
# fast / decision-relevant models first; the two slow cnn_v2 scans last.
COMBOS=(prokbert_base prokbert_xs nt_base nt_xs cnn_base cnn_xs cnn_v2_base cnn_v2_xs)
# ================================================================================================

extra_args() {   # tokenizer/type + batch size, keyed by model family
    case "$1" in
        prokbert_*) echo "--model-id neuralbioinfo/prokbert-mini --batch-size 512" ;;
        cnn_v2_*)   echo "--model-type cnn_v2 --batch-size 512" ;;
        cnn_*)      echo "--model-type cnn --batch-size 512" ;;
        nt_*)       echo "--model-id InstaDeepAI/nucleotide-transformer-v2-250m-multi-species --batch-size 256" ;;
    esac
}
env_for() { case "$1" in nt_*) echo "$NT_ENV" ;; *) echo "$MAIN_ENV" ;; esac; }

echo "=== preflight ==="
for f in "$FASTA" "$TRUTH" "$REGIONS"; do
    [[ -f "$f" ]] || { echo "MISSING $f — run prep_cau_regimeC.sh first"; exit 1; }
done
for C in "${COMBOS[@]}"; do
    [[ -e "${MPATH[$C]}" ]] || { echo "MISSING model path for $C: ${MPATH[$C]} — fix the MPATH table"; exit 1; }
done
echo "FASTA=$FASTA  truth=$(($(wc -l <"$TRUTH")-1)) TSS  regions=$REGIONS"

bench() {   # bench <scores.csv> <out_prefix> [extra args e.g. --regions ...]
    python "$EVAL/genome_fp_benchmark.py" --scores "$1" --truth "$TRUTH" \
        --recall-targets "$RECALLS" --out "$2" "${@:3}"
}

for C in "${COMBOS[@]}"; do
    TRACK=$OUT/cau_${C}.csv
    # 1) scan (skip if already complete; write to .tmp then atomically move so a preempt can't leave a partial)
    if [[ ! -s "$TRACK" ]]; then
        conda activate "$(env_for "$C")"
        echo "=== scan $C  (env $(env_for "$C")) ==="
        python "$EVAL/genome_inference.py" --genome "$FASTA" --model-path "${MPATH[$C]}" $(extra_args "$C") \
            --emit-logit --window 81 --step 1 --out "$TRACK.tmp"
        mv "$TRACK.tmp" "$TRACK"
    else
        echo "=== skip $C (track exists) ==="
    fi
    # 2) benchmark this track (raw; full + restricted), in the MAIN env
    conda activate "$MAIN_ENV"
    bench "$TRACK" "$OUT/fp_cau_${C}_full"
    bench "$TRACK" "$OUT/fp_cau_${C}_restrict" --regions "$REGIONS"
    echo "### $C  (restricted):"
    grep -E "0\.(10|50) \(got" "$OUT/fp_cau_${C}_restrict/genome_fp_benchmark.md" || true
done

# ============================ collate ============================
{
    echo "# C. autoethanogenum (2nd validation genome) — base-vs-xs genome-scale FP"
    echo ""
    echo "Genome: C. autoethanogenum DSM 10061 (GCF_000484505.1), truth = $(($(wc -l <"$TRUTH")-1)) primary TSS."
    echo "Raw (uncalibrated). Restricted = windows in candidate promoter regions (300 up / 50 into gene)."
    echo "~99% ANI to C. ljungdahlii -> a de-noised CONSISTENCY check (2,885 TSS), NOT an independent genome."
    echo "tol 40 bp, FP-peak merge 81 bp, strand-aware (same settings as the ljungdahlii benchmark)."
    echo ""
    for C in "${COMBOS[@]}"; do
        echo "## $C — restricted"
        cat "$OUT/fp_cau_${C}_restrict/genome_fp_benchmark.md"
        echo ""
    done
} > "$ANALYSIS"

echo ""
echo "=== DONE ===  analysis -> $ANALYSIS"
echo "--- recall 0.1 & 0.5, restricted, all 8 ---"
grep -HE "0\.(10|50) \(got" "$OUT"/fp_cau_*_restrict/genome_fp_benchmark.md
