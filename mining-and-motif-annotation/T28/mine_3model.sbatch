#!/bin/bash
#SBATCH --job-name=mine3
#SBATCH --account=igem2026
#SBATCH --partition=preempt
#SBATCH --gpus-per-node=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --time=23:59:00
#SBATCH --output=mine3_%j.log
#
# 3-MODEL ensemble mine  =  ProkBERT + cnn_v2 + Evo2   (Oak's 2026-08 deployment decision; NT & CNN-v1 dropped).
# Scans one genome with the three deployed models, reconciles them onto a common window grid, and combines
# them by mean rank-percentile (T29) into ONE ensemble track + a restricted, score-sorted shortlist.
#
# WHY a dedicated job (not scan_ensemble.sbatch): Evo2 is not a genome_inference model — it scans only the
# candidate regions with its own scanner + env, so its track is a SUBSET of the ProkBERT/cnn_v2 full-genome
# tracks. This job (a) scans Evo2 with --align-grid so its windows land on the global step grid, then
# (b) align_tracks.py intersects the three to a common grid (= Evo2's restricted set), then (c) runs T29
# unchanged. The ensemble is therefore ranked over the CANDIDATE SPACE, which is the right universe for a
# restricted mine.
#
# TWO MODES (same as the runbook):
#   * TRUTH set (validation genome, e.g. ljungdahlii) -> benchmarks the ensemble + each member, restricted.
#     DO THIS FIRST to confirm the ensemble + grid alignment work before the target.
#   * TRUTH="" (target mine, e.g. g409755 / C. butyricum) -> writes the score-sorted shortlist, no benchmark.
#
# SUBMIT (validate on ljungdahlii FIRST — this is the built-in DEFAULT; all clj paths are staged):
#   sbatch mining-and-motif-annotation/T28/mine_3model.sbatch
# SUBMIT (mine the target — absolute paths; TRUTH="" turns the benchmark OFF for the no-truth target):
#   TAG=g409755 TRUTH="" \
#     FASTA=/project/igem2026/cb-promoter-mining/<path>/g409755.fna \
#     REGIONS=/project/igem2026/cb-promoter-mining/<path>/g409755_candidate_regions.bed \
#     sbatch mining-and-motif-annotation/T28/mine_3model.sbatch
# Resume-safe: a completed per-model track is skipped on requeue.
set -eo pipefail

REPO=/project/igem2026/cb-promoter-mining
DIR=$REPO/infrastructure/T3/snakemake_scaffold/results
EVAL=$REPO/evaluation-and-calibration
EVO2=$REPO/model-training/Evo2
T29DIR=$REPO/mining-and-motif-annotation/T29
RC=$DIR/regimeC
MAIN_ENV=/project/igem2026/envs/promoter-mining-hpc4
EVO2_ENV=/project/igem2026/envs/evo2
CONDA_BASE=/cm/shared/apps/Anaconda3/2023.09-0

# ---- config (override from the submit line; RC is exported so the examples above resolve) ----
export RC
TAG=${TAG:-clj}
FASTA=${FASTA:-$RC/clj.fna}
REGIONS=${REGIONS:-$RC/clj_candidate_regions.bed}
TRUTH=${TRUTH:-$EVAL/ljungdahlii_holdout_tss.tsv}  # default = clj validation; set TRUTH="" for a target mine
STEP=${STEP:-10}                                   # MUST be 10 to match the Evo2/validation grid; shared by all 3
RECALLS=${RECALLS:-0.1,0.5,0.7,0.9}
PROBE=${PROBE:-$RC/evo2/evo2_probe_xs.pkl}         # frozen T15_xs logreg probe (genome-agnostic; reused across genomes)
OUT=$RC/mine3_$TAG ; mkdir -p "$OUT"

declare -A MPATH=(
    [prokbert]="$DIR/T21_xs/prokbert/best_model"
    [cnn_v2]="$DIR/T21_xs/cnn_v2/best_model.pt"
)
extra_args() { case "$1" in
    prokbert) echo "--model-id neuralbioinfo/prokbert-mini --batch-size 512" ;;
    cnn_v2)   echo "--model-type cnn_v2 --batch-size 512" ;;
esac; }

export HF_HOME=/project/igem2026/hf_shared HF_HUB_OFFLINE=1 HF_TOKEN_PATH=/tmp/hf_notoken_$USER
export PYTHONNOUSERSITE=1 MKL_THREADING_LAYER=GNU
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "=== preflight ==="
for f in "$FASTA" "$REGIONS"; do
    [[ -f "$f" ]] || { echo "MISSING $f"; exit 1; }
done
[[ -f "$PROBE" ]] || { echo "MISSING Evo2 probe $PROBE (the T15_xs logreg probe from the Evo2 runs)"; exit 1; }
for m in prokbert cnn_v2; do
    [[ -e "${MPATH[$m]}" ]] || { echo "MISSING model $m: ${MPATH[$m]} — fix MPATH"; exit 1; }
done
# contig-id parity: Evo2 emits the FASTA header as `contig` and can't override it, and genome_inference
# reads the SAME FASTA here (no --contig-name), so all three tracks share ids. Guard it anyway.
FID=$(grep -m1 '^>' "$FASTA" | sed 's/^>//; s/[[:space:]].*//')
REG_CONTIGS=$(tail -n +2 "$REGIONS" | cut -f1 | sort -u)   # materialise first: no pipe into grep -q (SIGPIPE+pipefail)
if ! grep -qx "$FID" <<<"$REG_CONTIGS"; then
    echo "ERROR: FASTA record id '$FID' is not among the regions' contigs — fix contig ids"; exit 1
fi
if [[ -n "$TRUTH" && ! -f "$TRUTH" ]]; then echo "TRUTH set but not found: $TRUTH"; exit 1; fi
# guard: a non-empty truth whose contig != this genome means a stale default on a target mine — fail loud.
if [[ -n "$TRUTH" ]]; then
    TCONTIG=$(awk 'NR==2{print $1; exit}' "$TRUTH")        # line-2 field-1; single process, no pipe/SIGPIPE
    [[ "$TCONTIG" == "$FID" ]] || { echo "ERROR: truth contig '$TCONTIG' != genome '$FID'. For a target"\
        " mine set TRUTH=\"\" (no benchmark); for validation point TRUTH at THIS genome's TSS."; exit 1; }
    echo "MODE: validation (truth=$(($(wc -l <"$TRUTH")-1)) TSS)"
else
    echo "MODE: target mine (shortlist only)"
fi
echo "TAG=$TAG  step=$STEP  out=$OUT"

# ---- 1) ProkBERT + cnn_v2: full-genome scan on the global step grid (main env) ----
for m in prokbert cnn_v2; do
    TRACK=$OUT/${TAG}_${m}.csv
    if [[ ! -s "$TRACK" ]]; then
        conda activate "$MAIN_ENV"
        echo "=== scan $m (genome_inference, step $STEP) ==="
        python "$EVAL/genome_inference.py" --genome "$FASTA" --model-path "${MPATH[$m]}" \
            $(extra_args "$m") --window 81 --step "$STEP" --out "$TRACK.tmp"
        mv "$TRACK.tmp" "$TRACK"
    else echo "=== skip $m (exists) ==="; fi
done

# ---- 2) Evo2: restricted scan, snapped to the SAME global grid (evo2 env) ----
ETRACK=$OUT/${TAG}_evo2.csv
if [[ ! -s "$ETRACK" ]]; then
    conda activate "$EVO2_ENV"
    echo "=== scan evo2 (evo2_scan_restricted --align-grid, step $STEP) ==="
    python "$EVO2/evo2_scan_restricted.py" --genome "$FASTA" --regions "$REGIONS" --probe "$PROBE" \
        --window 81 --step "$STEP" --align-grid --batch-size 16 --out "$ETRACK.tmp"
    mv "$ETRACK.tmp" "$ETRACK"
else echo "=== skip evo2 (exists) ==="; fi

# ---- 3) reconcile to a common grid, then T29 mean-rank-percentile (main env) ----
conda activate "$MAIN_ENV"
echo "=== align_tracks (intersection = Evo2's restricted candidate windows) ==="
python "$T29DIR/align_tracks.py" \
    --track prokbert "$OUT/${TAG}_prokbert.csv" \
    --track cnn_v2   "$OUT/${TAG}_cnn_v2.csv" \
    --track evo2     "$ETRACK" \
    --out-dir "$OUT/aligned"

ENS=$OUT/${TAG}_ensemble3.csv
SHORT=$OUT/${TAG}_shortlist3_restricted.csv
echo "=== T29 ensemble (ProkBERT + cnn_v2 + Evo2) ==="
python "$T29DIR/ensemble_rank_percentile.py" \
    --track prokbert "$OUT/aligned/prokbert_aligned.csv" \
    --track cnn_v2   "$OUT/aligned/cnn_v2_aligned.csv" \
    --track evo2     "$OUT/aligned/evo2_aligned.csv" \
    --out "$ENS" --regions "$REGIONS" --shortlist "$SHORT"

# ---- 4) validation mode only: benchmark ensemble + each member, restricted ----
if [[ -n "$TRUTH" && -f "$TRUTH" ]]; then
    echo "=== restricted benchmark (validation) ==="
    python "$EVAL/genome_fp_benchmark.py" --scores "$ENS" --truth "$TRUTH" \
        --recall-targets "$RECALLS" --regions "$REGIONS" --out "$OUT/fp_${TAG}_ensemble3_restrict"
    for m in prokbert cnn_v2 evo2; do
        python "$EVAL/genome_fp_benchmark.py" --scores "$OUT/aligned/${m}_aligned.csv" --truth "$TRUTH" \
            --recall-targets "$RECALLS" --regions "$REGIONS" --out "$OUT/fp_${TAG}_${m}_restrict" >/dev/null
    done
    echo ""
    echo "--- recall 0.1 & 0.5, restricted: ENSEMBLE vs members ---"
    grep -HE "0\.(10|50) \(got" "$OUT"/fp_${TAG}_ensemble3_restrict/genome_fp_benchmark.md \
                                "$OUT"/fp_${TAG}_{prokbert,cnn_v2,evo2}_restrict/genome_fp_benchmark.md
fi

echo ""
echo "=== DONE ===  outputs in $OUT"
echo "  ensemble track : $ENS"
echo "  shortlist      : $SHORT   <- the mine deliverable (score-sorted candidate windows)"
[[ -n "$TRUTH" && -f "$TRUTH" ]] && echo "  benchmark      : $OUT/fp_${TAG}_ensemble3_restrict/genome_fp_benchmark.md"
echo "Next (target): T30 call_peaks.py --track $SHORT --min-score 0.9 --out ${TAG}_peaks.csv  -> T31 -> T32."
