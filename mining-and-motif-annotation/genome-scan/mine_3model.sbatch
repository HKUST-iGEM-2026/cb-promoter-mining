#!/bin/bash
#SBATCH --job-name=mineN
#SBATCH --account=igem2026
#SBATCH --partition=preempt
#SBATCH --gpus-per-node=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --time=23:59:00
#SBATCH --output=mineN_%j.log
#
# N-MODEL ensemble mine (default = the 3-model deployment set: ProkBERT + cnn_v2 + Evo2).
# Scans one genome with the chosen models, reconciles them onto a common window grid, and combines them
# by mean rank-percentile (T29) into ONE ensemble track + a restricted, score-sorted shortlist.
#
# Pick the model set with MODELS (space-separated; evo2 is special-cased to its own scanner + env, all
# others go through genome_inference). Per-model tracks are cached in the same $OUT and reused across
# model-set choices, so a 4-model run after a 3-model run only scans the ONE new model. Ensemble outputs
# are suffixed by RUN (default = "<N>m") so 3m and 4m results coexist.
#   3-model (default): MODELS="prokbert cnn_v2 evo2"
#   4-model (2 CNNs) : MODELS="prokbert cnn cnn_v2 evo2"
#
# WHY the reconcile: Evo2 scans only candidate regions with its own scanner, so its track is a SUBSET of
# the genome_inference full-genome tracks. Evo2 runs with --align-grid (per-strand grid match), then
# align_tracks.py intersects all tracks to the common grid (= Evo2's restricted candidate windows), then
# T29 runs unchanged. The ensemble is ranked over the candidate space — the right universe for a mine.
#
# TWO MODES:
#   * TRUTH set (validation genome, e.g. ljungdahlii) -> benchmarks ensemble + each member, restricted.
#   * TRUTH="" (target mine, e.g. g409755 / C. butyricum) -> writes the score-sorted shortlist, no benchmark.
#
# SUBMIT (validate 3-model on ljungdahlii — built-in DEFAULT):
#   sbatch mining-and-motif-annotation/genome-scan/mine_3model.sbatch
# SUBMIT (validate 4-model, 2 CNNs, on ljungdahlii — reuses the 3 cached tracks, scans only cnn):
#   MODELS="prokbert cnn cnn_v2 evo2" sbatch mining-and-motif-annotation/genome-scan/mine_3model.sbatch
# SUBMIT (mine the target, no truth):
#   TAG=g409755 TRUTH="" MODELS="prokbert cnn_v2 evo2" \
#     FASTA=/abs/g409755.fna REGIONS=/abs/g409755_candidate_regions.bed \
#     sbatch mining-and-motif-annotation/genome-scan/mine_3model.sbatch
# Resume-safe: a completed per-model track is skipped on requeue. NOTE: run only ONE job per TAG at a
# time (finish the first, then submit the next) — same-TAG jobs share the per-model track files, so the
# second reuses them. (Temp files are job-unique, so a concurrent submit won't corrupt, just waste work.)
set -eo pipefail

REPO=/project/igem2026/cb-promoter-mining
DIR=$REPO/infrastructure/T3/snakemake_scaffold/results
EVAL=$REPO/evaluation-and-calibration
EVO2=$REPO/model-training/Evo2
T29DIR=$REPO/mining-and-motif-annotation/ensemble-aggregation
RC=$DIR/regimeC
MAIN_ENV=/project/igem2026/envs/promoter-mining-hpc4
EVO2_ENV=/project/igem2026/envs/evo2
CONDA_BASE=/cm/shared/apps/Anaconda3/2023.09-0

# ---- config (override from the submit line) ----
export RC
TAG=${TAG:-clj}
FASTA=${FASTA:-$RC/clj.fna}
REGIONS=${REGIONS:-$RC/clj_candidate_regions.bed}
TRUTH=${TRUTH-$EVAL/ljungdahlii_holdout_tss.tsv}   # NOTE: '-' not ':-' — default only if UNSET, so TRUTH="" (empty) => target mine, no benchmark
STEP=${STEP:-10}                                   # MUST match the Evo2/validation grid; shared by ALL models
RECALLS=${RECALLS:-0.1,0.5,0.7,0.9}
PROBE=${PROBE:-$RC/evo2/evo2_probe_xs.pkl}         # frozen T15_xs logreg probe (genome-agnostic)
MODELS=${MODELS:-"prokbert cnn_v2 evo2"}           # the ensemble members; add "cnn" for the 4-model run
RUN=${RUN:-$(echo $MODELS | wc -w)m}               # ensemble-output suffix so 3m / 4m coexist
OUT=$RC/mine3_$TAG ; mkdir -p "$OUT"               # per-model tracks live here, shared across model sets

declare -A MPATH=(                                 # genome_inference models (evo2 is separate)
    [prokbert]="$DIR/T21_xs/prokbert/best_model"
    [cnn]="$DIR/T21_xs/cnn/best_model.pt"
    [cnn_v2]="$DIR/T21_xs/cnn_v2/best_model.pt"
)
extra_args() { case "$1" in
    prokbert) echo "--model-id neuralbioinfo/prokbert-mini --batch-size 512" ;;
    cnn_v2)   echo "--model-type cnn_v2 --batch-size 512" ;;
    cnn)      echo "--model-type cnn --batch-size 512" ;;
esac; }

export HF_HOME=/project/igem2026/hf_shared HF_HUB_OFFLINE=1 HF_TOKEN_PATH=/tmp/hf_notoken_$USER
export PYTHONNOUSERSITE=1 MKL_THREADING_LAYER=GNU
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "=== preflight ==="
for f in "$FASTA" "$REGIONS"; do
    [[ -f "$f" ]] || { echo "MISSING $f"; exit 1; }
done
for m in $MODELS; do
    if [[ "$m" == "evo2" ]]; then
        [[ -f "$PROBE" ]] || { echo "MISSING Evo2 probe $PROBE"; exit 1; }
    else
        [[ -n "${MPATH[$m]:-}" ]] || { echo "unknown model '$m' (known: prokbert cnn cnn_v2 evo2)"; exit 1; }
        [[ -e "${MPATH[$m]}" ]]   || { echo "MISSING model $m: ${MPATH[$m]} — fix MPATH"; exit 1; }
    fi
done
# contig-id parity: Evo2 emits the FASTA header as `contig` and can't override it; genome_inference reads
# the SAME FASTA (no --contig-name), so all tracks share ids. Materialise the contig list first (no pipe
# into grep -q, which would SIGPIPE under pipefail).
FID=$(grep -m1 '^>' "$FASTA" | sed 's/^>//; s/[[:space:]].*//')
REG_CONTIGS=$(tail -n +2 "$REGIONS" | cut -f1 | sort -u)
if ! grep -qx "$FID" <<<"$REG_CONTIGS"; then
    echo "ERROR: FASTA record id '$FID' is not among the regions' contigs — fix contig ids"; exit 1
fi
if [[ -n "$TRUTH" && ! -f "$TRUTH" ]]; then echo "TRUTH set but not found: $TRUTH"; exit 1; fi
if [[ -n "$TRUTH" ]]; then
    TCONTIG=$(awk 'NR==2{print $1; exit}' "$TRUTH")        # line-2 field-1; single process, no pipe/SIGPIPE
    [[ "$TCONTIG" == "$FID" ]] || { echo "ERROR: truth contig '$TCONTIG' != genome '$FID'. For a target"\
        " mine set TRUTH=\"\" (no benchmark); for validation point TRUTH at THIS genome's TSS."; exit 1; }
    echo "MODE: validation (truth=$(($(wc -l <"$TRUTH")-1)) TSS)"
else
    echo "MODE: target mine (shortlist only)"
fi
echo "MODELS=[$MODELS]  RUN=$RUN  TAG=$TAG  step=$STEP  out=$OUT"

# ---- 1) scan each model (evo2 -> restricted --align-grid, evo2 env; others -> genome_inference, main env) ----
ALIGN_ARGS=()
for m in $MODELS; do
    TRACK=$OUT/${TAG}_${m}.csv
    TMP="$TRACK.${SLURM_JOB_ID:-$$}.tmp"            # job-unique temp: two concurrent jobs can't race on one .tmp
    if [[ ! -s "$TRACK" ]]; then
        if [[ "$m" == "evo2" ]]; then
            conda activate "$EVO2_ENV"
            echo "=== scan evo2 (evo2_scan_restricted --align-grid, step $STEP) ==="
            python "$EVO2/evo2_scan_restricted.py" --genome "$FASTA" --regions "$REGIONS" --probe "$PROBE" \
                --window 81 --step "$STEP" --align-grid --batch-size 16 --out "$TMP"
        else
            conda activate "$MAIN_ENV"
            echo "=== scan $m (genome_inference, step $STEP) ==="
            python "$EVAL/genome_inference.py" --genome "$FASTA" --model-path "${MPATH[$m]}" \
                $(extra_args "$m") --window 81 --step "$STEP" --out "$TMP"
        fi
        mv -f "$TMP" "$TRACK"
    else echo "=== skip $m (track exists) ==="; fi
    ALIGN_ARGS+=(--track "$m" "$TRACK")
done

# ---- 2) reconcile to a common grid, then T29 mean-rank-percentile (main env) ----
conda activate "$MAIN_ENV"
ALIGNED=$OUT/aligned_$RUN
echo "=== align_tracks ($RUN; intersection = Evo2's restricted candidate windows) ==="
python "$T29DIR/align_tracks.py" "${ALIGN_ARGS[@]}" --out-dir "$ALIGNED"

T29_ARGS=(); for m in $MODELS; do T29_ARGS+=(--track "$m" "$ALIGNED/${m}_aligned.csv"); done
ENS=$OUT/${TAG}_ensemble_${RUN}.csv
SHORT=$OUT/${TAG}_shortlist_${RUN}_restricted.csv
echo "=== T29 ensemble ($MODELS) ==="
python "$T29DIR/ensemble_rank_percentile.py" "${T29_ARGS[@]}" \
    --out "$ENS" --regions "$REGIONS" --shortlist "$SHORT"

# ---- 3) validation mode only: benchmark ensemble + each member, restricted ----
if [[ -n "$TRUTH" && -f "$TRUTH" ]]; then
    echo "=== restricted benchmark (validation) ==="
    python "$EVAL/genome_fp_benchmark.py" --scores "$ENS" --truth "$TRUTH" \
        --recall-targets "$RECALLS" --regions "$REGIONS" --out "$OUT/fp_${TAG}_ensemble_${RUN}_restrict"
    GREP_FILES=("$OUT/fp_${TAG}_ensemble_${RUN}_restrict/genome_fp_benchmark.md")
    for m in $MODELS; do
        python "$EVAL/genome_fp_benchmark.py" --scores "$ALIGNED/${m}_aligned.csv" --truth "$TRUTH" \
            --recall-targets "$RECALLS" --regions "$REGIONS" --out "$OUT/fp_${TAG}_${m}_${RUN}_restrict" >/dev/null
        GREP_FILES+=("$OUT/fp_${TAG}_${m}_${RUN}_restrict/genome_fp_benchmark.md")
    done
    echo ""
    echo "--- recall 0.1 & 0.5, restricted ($RUN): ENSEMBLE vs members ---"
    grep -HE "0\.(10|50) \(got" "${GREP_FILES[@]}"
fi

echo ""
echo "=== DONE ($RUN) ===  outputs in $OUT"
echo "  ensemble track : $ENS"
echo "  shortlist      : $SHORT   <- the mine deliverable (score-sorted candidate windows)"
[[ -n "$TRUTH" && -f "$TRUTH" ]] && echo "  benchmark      : $OUT/fp_${TAG}_ensemble_${RUN}_restrict/genome_fp_benchmark.md"
echo "Next (target): T30 call_peaks.py --track $SHORT --min-score 0.9 --out ${TAG}_peaks.csv  -> T31 -> T32."
