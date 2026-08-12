#!/bin/bash 
#SBATCH --job-name=06_C_ljung_scan
#SBATCH --account=igem2026
#SBATCH --partition=preempt
#SBATCH --gpus-per-node=1
#SBATCH --requeue
#SBATCH --time=08:00:00
#SBATCH --output=06_C_ljung_scan%j.log
#
# Regime C, 0.6 retrain — calibrated genome-scale FP pipeline on the held-out in-genus genome
# C. ljungdahlii (NC_014328.1). For all FOUR models (prokbert, cnn, cnn_v2, nt) it:
#   1. scans the WHOLE genome once with genome_inference --emit-logit  -> track carries
#      promoter_score + `logit` + `at_fraction` (the exact T26-calibrator inputs).
#   2. applies that model's saved Platt(+AT) calibrator with apply_calibrator.py  -> calibrated track
#      (reads logit + at_fraction straight from the track; NO FASTA, NO prob->logit inversion).
#   3. runs genome_fp_benchmark.py FOUR ways per model: {raw, calibrated} x {full, restricted}.
#      NB the scan is NOT run twice: "full" vs "restricted" is a benchmark-time flag (--regions) on
#      the SAME track; the candidate-regions BED is model-independent and reused.
#   4. collates everything (incl. all caveats) into one analysis markdown.
#
# Submit:  sbatch C_ljung_scan_06.sh
set -eo pipefail

# --------------------------------------------------------------------------- paths
REPO=/project/igem2026/cb-promoter-mining
DIR=$REPO/infrastructure/T3/snakemake_scaffold/results        # holds T21_final_06/ + T16/
EVAL=$REPO/evaluation-and-calibration
T26=$EVAL/T26/calib_06                                        # the 0.6 calibrators live here
MAIN_ENV=/project/igem2026/envs/promoter-mining-hpc4          # prokbert + cnn + cnn_v2 + all CPU steps
NT_ENV=/project/igem2026/envs/nt-tf4x                         # NT only (transformers 4.57)
CONDA_BASE=/cm/shared/apps/Anaconda3/2023.09-0

OUTDIR=$DIR/regimeC                                           # tracks + fp_* + analysis land here
TRUTH=$EVAL/ljungdahlii_holdout_tss.tsv                       # 1,245 held-out primary TSS (committed)
REGIONS=$OUTDIR/clj_candidate_regions.bed                     # model-independent; reused
ANALYSIS=$OUTDIR/regimeC_calibrated_analysis_06.md
RECALLS=0.1,0.5,0.7,0.9
mkdir -p "$OUTDIR"

# C. ljungdahlii RefSeq FASTA — record id MUST be NC_014328.1 to match the truth.
# Set CLJ_FASTA in the environment to override; otherwise we try to locate it.
CLJ_FASTA=${CLJ_FASTA:-$OUTDIR/clj.fna}
if [[ ! -f "$CLJ_FASTA" ]]; then
    CLJ_FASTA=$(find "$REPO" /project/igem2026 -name "*.fna" 2>/dev/null \
                | grep -iE "ljung|014328|143685" | head -1 || true)
fi

# --------------------------------------------------------------------------- HF offline cache (shared)
export HF_HOME=/project/igem2026/hf_shared
export HF_HUB_OFFLINE=1
export HF_TOKEN_PATH=/tmp/hf_notoken_$USER

# --------------------------------------------------------------------------- preflight
source "$CONDA_BASE/etc/profile.d/conda.sh"
MODELS=(prokbert cnn cnn_v2 nt)

# genome_inference args per model (model-path + tokenizer/type + batch size)
scan_args() {
    case "$1" in
        prokbert) echo "--model-path $DIR/T21_final_06/prokbert/best_model --model-id neuralbioinfo/prokbert-mini --batch-size 512" ;;
        cnn)      echo "--model-path $DIR/T21_final_06/cnn/best_model.pt --model-type cnn --batch-size 512" ;;
        cnn_v2)   echo "--model-path $DIR/T21_final_06/cnn_v2/best_model.pt --model-type cnn_v2 --batch-size 512" ;;
        nt)       echo "--model-path $DIR/T21_final_06/nt_lora/best_adapter --model-id InstaDeepAI/nucleotide-transformer-v2-250m-multi-species --batch-size 256" ;;
    esac
}
# each model's saved calibrator (note: nt dir is nt_lora but the file is nt_calibrator_at.joblib)
calib_path() {
    case "$1" in
        prokbert) echo "$T26/prokbert/prokbert_calibrator_at.joblib" ;;
        cnn)      echo "$T26/cnn/cnn_calibrator_at.joblib" ;;
        cnn_v2)   echo "$T26/cnn_v2/cnn_v2_calibrator_at.joblib" ;;
        nt)       echo "$T26/nt_lora/nt_calibrator_at.joblib" ;;
    esac
}
env_for() { [[ "$1" == "nt" ]] && echo "$NT_ENV" || echo "$MAIN_ENV"; }

echo "=== preflight ==="
[[ -n "$CLJ_FASTA" && -f "$CLJ_FASTA" ]] || { echo "ERROR: C. ljungdahlii FASTA not found. Set CLJ_FASTA (record NC_014328.1) or stage GCF_000143685.1."; exit 1; }
[[ -f "$TRUTH" ]]   || { echo "ERROR: truth TSS not found: $TRUTH"; exit 1; }
[[ -f "$REGIONS" ]] || { echo "ERROR: candidate-regions BED not found: $REGIONS (build once with make_candidate_regions.py --gff <clj.gff3> --contig NC_014328.1 --up 300 --down 50)."; exit 1; }
for M in "${MODELS[@]}"; do
    [[ -f "$(calib_path "$M")" ]] || { echo "ERROR: calibrator missing for $M: $(calib_path "$M")"; exit 1; }
done

# --- contig-consistency guard (the silent-0%-recall trap) ------------------------------------------
# The scan writes the FASTA's record id into the track's `contig` column, but the truth TSS and the
# candidate-regions BED both use the RefSeq id (e.g. NC_014328.1). A Bakta-renamed FASTA (`contig_1`)
# joins to NEITHER -> genome_fp_benchmark reports 0% recall (full) and aborts (restricted). Detect the
# mismatch here and, for a single-replicon genome, force the emitted contig to the truth id.
TRUTH_CONTIG=$(awk -F'\t' 'NR>1{print $1}' "$TRUTH" | sort -u)
NTC=$(printf '%s\n' "$TRUTH_CONTIG" | grep -c .)
FASTA_IDS=$(grep '^>' "$CLJ_FASTA" | sed 's/^>//; s/[[:space:]].*//')
NFR=$(printf '%s\n' "$FASTA_IDS" | grep -c .)
CONTIG_ARG=""
if [[ "$NTC" -eq 1 ]]; then
    if printf '%s\n' "$FASTA_IDS" | grep -qx "$TRUTH_CONTIG"; then
        echo "contig  : FASTA carries truth id '$TRUTH_CONTIG' — no override."
    elif [[ "$NFR" -eq 1 ]]; then
        echo "contig  : WARNING FASTA id '$FASTA_IDS' != truth '$TRUTH_CONTIG' (single-replicon) -> forcing --contig-name."
        CONTIG_ARG="--contig-name $TRUTH_CONTIG"
    else
        echo "ERROR: FASTA has $NFR records, none match truth contig '$TRUTH_CONTIG' — wrong assembly? ($(printf '%s ' $FASTA_IDS))"; exit 1
    fi
else
    echo "contig  : truth has $NTC contigs; skipping single-replicon override — verify track/truth ids match."
fi

# --- calibrators load in MAIN_ENV (nt calibrator was fit under nt-tf4x -> catch sklearn skew now) ---
conda activate "$MAIN_ENV"
for M in "${MODELS[@]}"; do
    python - "$(calib_path "$M")" <<'PY' || { echo "ERROR: calibrator failed to load in MAIN_ENV (sklearn version skew?): $(calib_path "$M")"; exit 1; }
import sys, joblib
cal = joblib.load(sys.argv[1])
n = getattr(cal, "n_features_in_", None)
assert n in (1, 2), f"unexpected n_features_in_={n} for {sys.argv[1]}"
print(f"calib   : loaded {sys.argv[1]} (n_features_in_={n})")
PY
done

echo "HF_HOME : $HF_HOME"
echo "FASTA   : $CLJ_FASTA"
echo "truth   : $TRUTH"
echo "regions : $REGIONS"
echo "outdir  : $OUTDIR"

# =========================================================================== 1) SCAN (GPU, --emit-logit)
# Full-genome scan once per model. NT runs in its own env; the rest in MAIN_ENV.
# RESUME-SAFE: an existing non-empty track is kept (delete it to force a rescan) so a preempt requeue
# doesn't redo completed GPU scans.
for M in "${MODELS[@]}"; do
    if [[ -s "$OUTDIR/clj_${M}_track.csv" ]]; then
        echo "=== scan $M: track exists — skipping (rm $OUTDIR/clj_${M}_track.csv to force rescan) ==="
        continue
    fi
    conda activate "$(env_for "$M")"
    echo "=== scan $M  (env $(env_for "$M")) ==="
    python "$EVAL/genome_inference.py" --genome "$CLJ_FASTA" $(scan_args "$M") $CONTIG_ARG \
        --emit-logit --window 81 --step 1 --out "$OUTDIR/clj_${M}_track.csv"
done

# =========================================================================== 2+3) CALIBRATE + BENCHMARK (CPU)
conda activate "$MAIN_ENV"
bench() {   # bench <scores.csv> <out_prefix> [--regions ...]
    python "$EVAL/genome_fp_benchmark.py" --scores "$1" --truth "$TRUTH" \
        --recall-targets "$RECALLS" --out "$2" "${@:3}"
}
for M in "${MODELS[@]}"; do
    RAW=$OUTDIR/clj_${M}_track.csv
    CAL=$OUTDIR/clj_${M}_calib_track.csv
    echo "=== calibrate $M ==="
    # apply_calibrator reads logit + at_fraction from the track (no --genome). Log its stdout: it
    # records the calibrator type (score+AT vs score-only) + logit/AT source, quoted in the analysis.
    python "$EVAL/apply_calibrator.py" --track "$RAW" --calibrator "$(calib_path "$M")" \
        --out "$CAL" 2>&1 | tee "$OUTDIR/apply_${M}.log"

    echo "=== benchmark $M  (raw & calibrated, full & restricted) ==="
    bench "$RAW" "$OUTDIR/fp_${M}_raw_full"
    bench "$RAW" "$OUTDIR/fp_${M}_raw_restrict"     --regions "$REGIONS"
    bench "$CAL" "$OUTDIR/fp_${M}_calib_full"
    bench "$CAL" "$OUTDIR/fp_${M}_calib_restrict"   --regions "$REGIONS"
done

# =========================================================================== 4) ANALYSIS (md + tex)
# Collated by the standalone analyzer, which re-reads the apply_*.log + fp_*/ outputs written above.
# It is re-runnable on these outputs WITHOUT a re-scan, and emits BOTH markdown and LaTeX (via its
# built-in md->tex renderer). To regenerate/retune the report later without re-scanning, just run:
#   python "$EVAL/T26/analyze_regimeC.py" --regime-dir "$OUTDIR" --truth "$TRUTH" --label "0.6 retrain"
python "$EVAL/T26/analyze_regimeC.py" --regime-dir "$OUTDIR" --truth "$TRUTH" \
    --models "$(IFS=,; echo "${MODELS[*]}")" --label "0.6 retrain" --recalls "$RECALLS" \
    --out "$ANALYSIS" --format both

echo ""
echo "=== DONE ==="
echo "analysis -> ${ANALYSIS%.md}.{md,tex}"
