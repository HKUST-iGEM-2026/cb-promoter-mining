#!/bin/bash
#SBATCH --job-name=t28_scan_t29_ens
#SBATCH --gpus-per-node=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --time=23:59:00
#SBATCH --output=t28_scan_t29_ens_%j.log
#
# Historical four-model xs scan and mean-rank ensemble; not the deployed three-member mine.
#   1) scans ProkBERT, NT-v2 LoRA, CNN-v1 and CNN-v2 over the whole genome on one window grid,
#   2) averages their percentile ranks computed over that genome-wide pool,
#   3) optionally benchmarks catalogue coverage restricted to candidate regions.
# Strictly increasing score-only transforms preserve ranks; AT-dependent calibration need not.
# This pool and membership differ from mine_3model.sbatch, which includes Evo2 and intersects
# tracks before ranking. Do not substitute these outputs for the deployed target ensemble.
# Resume-safe: a completed per-model track is skipped on requeue.
#
# PREREQ (login node): stage the genome + candidate regions (+ optional truth) — see docs/PIPELINE.md.
# SUBMIT:  sbatch pipeline/mining/genome-scan/scan_ensemble.sbatch
set -eo pipefail

# ==================== CONFIG: override any of these from the submit line ====================
# Each is env-overridable, so you drive it per-genome WITHOUT editing this file (keeps the /project
# clone pull-only). e.g.:
#   TAG=clj FASTA=$RC/clj.fna REGIONS=$RC/clj_candidate_regions.bed TRUTH=$EVAL/ljungdahlii_holdout_tss.tsv \
#     sbatch pipeline/mining/genome-scan/scan_ensemble.sbatch
REPO="${PROMOTER_REPO:?set it or source env.sh}"
TAG=${TAG:-clj}                                  # short genome tag -> goes in every output filename
FASTA=${FASTA:-$REPO/PATH/TO/genome.fna}         # the genome to scan (single- or multi-record FASTA)
REGIONS=${REGIONS:-$REPO/PATH/TO/candidate_regions.bed}  # make_candidate_regions.py output (restrict-scan space)
TRUTH=${TRUTH:-}                                 # OPTIONAL known-TSS (contig,pos,strand, tab-sep).
                                                 #   set a path  -> validation mode (benchmark ON)
                                                 #   leave unset  -> target-mine mode (shortlist only, no benchmark)
STEP=${STEP:-1}                                  # 1 = every position. Bump to 3-5 to speed the scan (cnn_v2 is the slow one).
CONTIG_OVERRIDE=${CONTIG_OVERRIDE:-}             # e.g.  --contig-name NC_014328.1  if Bakta renamed the sole contig; else leave unset
RECALLS=${RECALLS:-0.1,0.5,0.7,0.9}
# ===========================================================================================

DIR=$REPO/workflow/results
EVAL=$REPO/pipeline/evaluation
T29=$REPO/pipeline/mining/ensemble-aggregation/ensemble_rank_percentile.py
OUT=$DIR/regimeC/mine_$TAG ; mkdir -p "$OUT"
MAIN_ENV="${ENV_MAIN:?set it or source env.sh}"
NT_ENV="${ENV_NT:?set it or source env.sh}"
CONDA_BASE=$(conda info --base)
if [ -n "${HF_HOME:-}" ]; then export HF_HOME; fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"   # 1 = require a pre-staged cache
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---- historical four-model xs checkpoints; verify these paths before running ----
# Evo2 is absent. The deployed three-model job is mine_3model.sbatch.
declare -A MPATH=(
    [prokbert]="$DIR/T21_xs/prokbert/best_model"
    [nt]="$DIR/T21_xs/nt_lora/best_adapter"
    [cnn]="$DIR/T21_xs/cnn/best_model.pt"
    [cnn_v2]="$DIR/T21_xs/cnn_v2/best_model.pt"
)
MODELS=(prokbert nt cnn cnn_v2)                   # fast-first; cnn_v2 (slow) last
extra_args() { case "$1" in
    prokbert) echo "--model-id neuralbioinfo/prokbert-mini --batch-size 512" ;;
    nt)       echo "--model-id InstaDeepAI/nucleotide-transformer-v2-250m-multi-species --batch-size 256" ;;
    cnn_v2)   echo "--model-type cnn_v2 --batch-size 512" ;;
    cnn)      echo "--model-type cnn --batch-size 512" ;;
esac; }
env_for() { case "$1" in nt) echo "$NT_ENV" ;; *) echo "$MAIN_ENV" ;; esac; }

echo "=== preflight ==="
[[ -f "$FASTA" ]]   || { echo "MISSING FASTA=$FASTA"; exit 1; }
[[ -f "$REGIONS" ]] || { echo "MISSING REGIONS=$REGIONS  (make it with make_candidate_regions.py)"; exit 1; }
for m in "${MODELS[@]}"; do
    [[ -e "${MPATH[$m]}" ]] || { echo "MISSING model $m: ${MPATH[$m]} — fix the MPATH table"; exit 1; }
done
if [[ -n "$TRUTH" && -f "$TRUTH" ]]; then
    echo "truth: $(($(wc -l <"$TRUTH")-1)) TSS  ->  VALIDATION mode (benchmark ON)"
elif [[ -n "$TRUTH" ]]; then
    echo "TRUTH set but not found: $TRUTH"; exit 1
else
    echo "no truth  ->  TARGET-MINE mode (restricted shortlist only)"
fi
echo "genome=$FASTA  regions=$REGIONS  step=$STEP  out=$OUT"

# ---- 1) scan each xs model, full genome, IDENTICAL grid (same window/step/strands => T29 can align) ----
TRACK_ARGS=()
for m in "${MODELS[@]}"; do
    TRACK=$OUT/${TAG}_${m}_xs.csv
    if [[ ! -s "$TRACK" ]]; then
        conda activate "$(env_for "$m")"
        echo "=== scan $m  (env $(env_for "$m")) ==="
        # write to .tmp then atomically mv, so a preempt can't leave a half-written track that the
        # resume logic would mistake for complete.
        python "$EVAL/genome_inference.py" --genome "$FASTA" --model-path "${MPATH[$m]}" \
            $(extra_args "$m") --window 81 --step "$STEP" $CONTIG_OVERRIDE --out "$TRACK.tmp"
        mv "$TRACK.tmp" "$TRACK"
    else
        echo "=== skip $m (track exists: $TRACK) ==="
    fi
    TRACK_ARGS+=(--track "$m" "$TRACK")
done

# ---- 2) T29 ensemble (mean rank-percentile), main env ----
conda activate "$MAIN_ENV"
ENS=$OUT/${TAG}_ensemble_xs.csv
echo "=== T29 ensemble ==="
python "$T29" "${TRACK_ARGS[@]}" --out "$ENS" \
    --regions "$REGIONS" --shortlist "$OUT/${TAG}_shortlist_restricted.csv"
echo "ensemble track -> $ENS"

# ---- 3) benchmark the ensemble (+ each member) RESTRICTED, only if truth was supplied ----
if [[ -n "$TRUTH" && -f "$TRUTH" ]]; then
    echo "=== restricted benchmark ==="
    python "$EVAL/genome_fp_benchmark.py" --scores "$ENS" --truth "$TRUTH" \
        --recall-targets "$RECALLS" --regions "$REGIONS" --out "$OUT/fp_${TAG}_ensemble_restrict"
    for m in "${MODELS[@]}"; do
        python "$EVAL/genome_fp_benchmark.py" --scores "$OUT/${TAG}_${m}_xs.csv" --truth "$TRUTH" \
            --recall-targets "$RECALLS" --regions "$REGIONS" --out "$OUT/fp_${TAG}_${m}_restrict" >/dev/null
    done
    echo ""
    echo "--- recall 0.1 & 0.5, restricted: ENSEMBLE vs members ---"
    grep -HE "0\.(10|50) \(got" "$OUT"/fp_${TAG}_ensemble_restrict/genome_fp_benchmark.md \
                                "$OUT"/fp_${TAG}_*_restrict/genome_fp_benchmark.md | sort -u
fi

echo ""
echo "=== DONE ===  outputs in $OUT"
echo "  ensemble track      : $ENS"
echo "  restricted shortlist: $OUT/${TAG}_shortlist_restricted.csv"
[[ -n "$TRUTH" && -f "$TRUTH" ]] && echo "  benchmark md        : $OUT/fp_${TAG}_ensemble_restrict/genome_fp_benchmark.md"
