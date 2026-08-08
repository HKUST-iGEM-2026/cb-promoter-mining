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

OUTDIR=$DIR/RegimeC                                           # tracks + fp_* + analysis land here
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
echo "HF_HOME : $HF_HOME"
echo "FASTA   : $CLJ_FASTA"
echo "truth   : $TRUTH"
echo "regions : $REGIONS"
echo "outdir  : $OUTDIR"

# =========================================================================== 1) SCAN (GPU, --emit-logit)
# Full-genome scan once per model. NT runs in its own env; the rest in MAIN_ENV.
for M in "${MODELS[@]}"; do
    conda activate "$(env_for "$M")"
    echo "=== scan $M  (env $(env_for "$M")) ==="
    python "$EVAL/genome_inference.py" --genome "$CLJ_FASTA" $(scan_args "$M") \
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

# =========================================================================== 4) ANALYSIS FILE
{
cat <<HDR
# Regime C — calibrated genome-scale FP analysis (0.6 retrain)

Held-out in-genus genome: **C. ljungdahlii** (NC_014328.1), truth = $(wc -l < "$TRUTH" | tr -d ' ') lines of \`ljungdahlii_holdout_tss.tsv\`.
Generated by \`C_ljung_scan_06.sh\` on $(date -u +"%Y-%m-%d %H:%M UTC").
For each of the four models: full-genome scan (\`--emit-logit\`) -> per-model Platt(+AT) calibrator
-> \`genome_fp_benchmark\` at recall $RECALLS, in all four {raw|calibrated} x {full|restricted} cells.

**Headline question:** at matched recall, does AT-aware calibration lower the false-positive PEAK
count vs the raw track? Compare the RAW vs CALIBRATED tables per model below.

---

## Caveats (read before trusting any number)

1. **Genome-scale FP is the metric that matters; balanced AUPRC hides it.** At genome scale promoters
   are ~0.2% of windows, so a strong balanced classifier still emits many FP peaks per scan. These
   tables are the deployment-relevant view; the T22/T25 balanced numbers are not.
2. **Full vs restricted is one scan, two benchmarks.** The scan is always full-genome; \`restricted\`
   only adds \`--regions clj_candidate_regions.bed\` (upstream 300 / into-gene 50 bp windows around
   every gene 5'), which shrinks the FP denominator to plausible promoter neighbourhoods. The BED is
   **model-independent** and identical across all models/regimes, which is what keeps the comparison fair.
3. **Score-only calibrators are a NO-OP on FP counts.** If a model's calibrator chose \`score\` only
   (\`n_features_in_==1\`, see its \`apply_*.log\` below), the transform is monotonic in the score, so
   ranking — and therefore every FP-peak count — is IDENTICAL to raw. A raw-vs-calibrated difference
   for such a model would only be numerical noise. Only score+AT calibrators can move the numbers.
4. **AT length/domain shift.** The calibrators fit \`at_fraction\` (and its StandardScaler mean/scale)
   on the ~52 bp calibration sequences (93 T9 positives + 186 curated negatives); here AT is over the
   81 bp scored window. AT is a fraction (scale-free) so this is the right deployment choice, but the
   absolute calibrated probabilities carry that shift. Because the benchmark is rank-based, only the
   AT-driven RE-RANKING matters, which is valid regardless of the offset — so FP counts are affected
   far less than absolute probabilities.
5. **Logit provenance is exact here.** Tracks were scored with \`--emit-logit\`, so apply_calibrator
   reads the exact \`logit\` (= logit1-logit0) the calibrator was fit on — no prob->logit inversion,
   so the float32-saturation caveat (that only bit the inversion path) does not apply to this run.
6. **Calibration is prevalence-dependent.** Fit at the 93:186 (1:2) calibration operating point, NOT
   genome-wide promoter prevalence (~0.2%). Read calibrated values as recalibrated *scores*, not as
   true genome-wide promoter probabilities. The rank-percentile ensemble (T29) is calibration-invariant.
7. **Truth is incomplete.** The 1,245 TSS are mostly primary / intergenic-sense; deep-CDS and
   antisense promoters are under-catalogued. So a counted "false positive" peak may be a real,
   unannotated promoter — treat FP counts as an upper bound on genuine errors.
8. **genome_fp_benchmark settings:** strand-aware, tol 40 bp (a window within 40 bp of a TSS covers
   it), FP windows merged into peaks within 81 bp. Contig-aware (chromosome-only here).
9. **Minor, model-code:** genome_inference's transformer scorer currently computes softmax after a
   \`.float()\` cast (the byte-identical prob-ordering refinement was lost in a revert). This can shift
   NT's promoter_score by ~1e-7 vs the published pre-calibration run; negligible for ranking, but note
   it if NT raw numbers differ trivially from the earlier report.
10. **Round context (from CLAUDE.md §10):** cnn_v2 is the 0.6 standout and is threshold-robust; NT is
    the round's weak link at genome scale despite 2nd-best balanced AUPRC; ProkBERT had a genome-scale
    regression partly recovered by the 0.6 re-sweep. C. ljungdahlii is the held-out validation genome
    (never train/mine on it — that is T33's circularity rule).

---
HDR

for M in "${MODELS[@]}"; do
    echo ""
    echo "## $M"
    echo ""
    echo "### calibrator (from apply_${M}.log)"
    echo '```'
    grep -E "calibrator|n_features_in_|logit source|AT source|WARNING|raw  prob|calib prob|AT fraction" "$OUTDIR/apply_${M}.log" || cat "$OUTDIR/apply_${M}.log"
    echo '```'
    for VARIANT in raw_full raw_restrict calib_full calib_restrict; do
        echo ""
        echo "### ${VARIANT/_/ · }"
        cat "$OUTDIR/fp_${M}_${VARIANT}/genome_fp_benchmark.md"
    done
done
} > "$ANALYSIS"

echo ""
echo "=== DONE ==="
echo "analysis -> $ANALYSIS"
