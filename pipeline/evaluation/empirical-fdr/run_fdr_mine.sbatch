#!/bin/bash
#SBATCH --job-name=t27_fdr_mine
#SBATCH --gpus-per-node=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --time=23:59:00
#SBATCH --output=t27_fdr_mine_%j.log
#
# T27 empirical-FDR promoter mine on the target C. butyricum.
#
# ARCHITECTURE (the scan/stats split): this bash owns ALL orchestration — GPU scans, per-model conda-env
# switching, grid alignment, and the motif pass — reusing the repo's standalone tools. fdr_stats.py owns
# ONLY the statistics (pooled rank-percentile ensemble + empirical p/q), pure Python, no subprocess, so
# you can re-run the FDR maths on the login node against cached score tracks without touching a GPU.
#
#   0) fdr_stats.py prep        -> contig preflight + packed null.fasta + null_regions.bed
#   1) make_candidate_regions   -> cbut_candidate_regions.bed (upstream-of-gene windows)
#   2) per-model scan loop       -> cand_<m>.csv (restricted, +/-) and null_<m>.csv (packed nullcat, +, step 81)
#                                  evo2 -> evo2_scan_restricted.py (evo2 env, --align-grid); prokbert/cnn_v2 ->
#                                  genome_inference.py (main env). Resume-safe (cached tracks).
#   3) align_tracks.py x2        -> reconcile candidate tracks and null tracks each onto one common grid
#   4) fdr_stats.py fdr          -> ensemble + empirical p/q -> peaks_fdr.csv + topN_promoters.csv + analysis.md
#   5) annotate_motifs.py        -> FIMO -10/-35 on the top-N (bio env)
#
# MODELS = the DEPLOYMENT 3-model ensemble (ProkBERT + cnn_v2 + Evo2) — NT-LoRA and CNN-v1 are dropped
# (Oak's decision, 2026-08; see the README, "Why three models?"). Only these three are supported here. evo2 is special-cased
# (own scanner + env); prokbert/cnn_v2 go through genome_inference. Raw scores are ranked per model
# over the shared candidate-plus-null pool. This does not calibrate promoter probabilities or establish
# biological FDR control. Subsets of the three members are supported; additional model types are not.
#
# The null TSV(s) MUST contain clean 81 bp sequences; extract_negatives.py defaults to 52.
# Run only ONE job per OUT at a time. Cached tracks are not fingerprinted by assembly/checkpoint:
# use a fresh output directory when those inputs or scan settings change.
#
# Current complete-reference catalog: GCF_014131795.1, 929 / 180 / 45 nested candidates.
# Historical g409755 and cbut_xs outputs use other coordinate frames; do not join them by position.
# The shell defaults below retain historical draft-reference locations and must be overridden for
# the current catalog. See docs/PIPELINE.md for staging, 81 bp null generation and q attachment.
#
# SUBMIT after staging the SAME complete-reference FASTA/GFF and exact candidate BED as the mine:
#   GENOME=/abs/g131795.fna GFF=/abs/g131795.gff3 \
#     OUT=/abs/cbut_g131795_xs MODELSET=xs STEP=10 \
#     NULL_TSVS=/abs/convergent_intergenic_negatives.tsv \
#     sbatch pipeline/evaluation/empirical-fdr/run_fdr_mine.sbatch
#
# Copy the exact mine BED to OUT/cbut_candidate_regions.bed before submission. Existing per-member
# tracks can also be reused as cand_prokbert.csv / cand_cnn_v2.csv / cand_evo2.csv, only when assembly,
# regions, checkpoint, stride and strand grid match the intended FDR run.
set -eo pipefail

REPO="${PROMOTER_REPO:?set it or source env.sh}"
DIR=$REPO/workflow/results
EVAL=$REPO/pipeline/evaluation
EVO2=$REPO/pipeline/training/Evo2
T27=$REPO/pipeline/evaluation/empirical-fdr
T29DIR=$REPO/pipeline/mining/ensemble-aggregation
T31DIR=$REPO/pipeline/mining/motif-annotation
ANN=$REPO/pipeline/evaluation/calibration/c_butyricum_annotation
MAIN_ENV="${ENV_MAIN:?set it or source env.sh}"
EVO2_ENV="${ENV_EVO2:?set it or source env.sh}"
BIO_ENV="${ENV_MEME:?set it or source env.sh}"
CONDA_BASE=$(conda info --base)

# ---- config (override from the submit line) ----
MODELSET=${MODELSET:-xs}                            # xs (T21_xs, deployed) or historical 06 (T21_final_06)
case "$MODELSET" in xs) SUB=T21_xs ;; 06) SUB=T21_final_06 ;; *) echo "bad MODELSET '$MODELSET' (xs|06)"; exit 1 ;; esac
MODELS=${MODELS:-"prokbert cnn_v2 evo2"}            # the deployment 3-model ensemble (only these 3 supported)
STEP=${STEP:-10}                                    # candidate stride; null scan uses isolated 81 bp blocks
PEAK_ALPHA=${PEAK_ALPHA:-0.05}
MERGE=${MERGE:-81}
TOPN=${TOPN:-80}
UP=${UP:-300}; DOWN=${DOWN:-50}                     # candidate region = [gene_start-UP, gene_start+DOWN]
# Historical draft-reference defaults. Current recipes explicitly override GENOME and GFF with
# the complete-reference files used for mining; matching strain names alone is insufficient.
G409755_FASTA=${G409755_FASTA:-${PROMOTER_REPO}/PATH_TO/g409755.fna}   # <-- SET ME
G409755_GFF=${G409755_GFF:-${PROMOTER_REPO}/PATH_TO/g409755.gff3}      # <-- SET ME
GENOME=${GENOME:-$G409755_FASTA}
GFF=${GFF:-$G409755_GFF}
# The null is scored independently in packed blocks and need not share candidate coordinates.
# It remains a putative negative sample, not proven promoter-free or guaranteed composition-matched.
# Record its source assemblies/conditions and regenerate it when that sampling design changes.
NULL_TSVS=${NULL_TSVS:-"$ANN/convergent_intergenic_negatives.tsv $EVAL/calibration/c_acetobutylicum_annotation/convergent_intergenic_negatives.tsv $EVAL/calibration/c_phytofermentans_annotation/convergent_intergenic_negatives.tsv"}
PROBE=${PROBE:-$DIR/regimeC/evo2/evo2_probe_xs.pkl}   # frozen T15_xs logreg probe (genome-agnostic)
OUT=${OUT:-$T27/results/cbut_g409755_${MODELSET}} ; mkdir -p "$OUT"   # fresh dir — NOT the stale 4-contig cbut_xs

declare -A MPATH=(                                  # genome_inference models (evo2 is separate)
    [prokbert]="$DIR/$SUB/prokbert/best_model"
    [cnn_v2]="$DIR/$SUB/cnn_v2/best_model.pt"
)
extra_args() { case "$1" in
    prokbert) echo "--model-id neuralbioinfo/prokbert-mini --batch-size 512" ;;
    cnn_v2)   echo "--model-type cnn_v2 --batch-size 512" ;;
esac; }
env_for() { case "$1" in evo2) echo "$EVO2_ENV" ;; *) echo "$MAIN_ENV" ;; esac; }

if [ -n "${HF_HOME:-}" ]; then export HF_HOME; fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"   # 1 = require a pre-staged cache
export PYTHONNOUSERSITE=1 MKL_THREADING_LAYER=GNU
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "=== preflight ==="
for f in "$GENOME" "$GFF"; do [[ -f "$f" ]] || { echo "MISSING $f"; exit 1; }; done
for m in $MODELS; do
    if [[ "$m" == evo2 ]]; then
        [[ -f "$PROBE" ]] || { echo "MISSING Evo2 probe $PROBE"; exit 1; }
    else
        [[ -n "${MPATH[$m]:-}" ]] || { echo "unknown model '$m' (known: prokbert cnn_v2 evo2)"; exit 1; }
        [[ -e "${MPATH[$m]}" ]]   || { echo "MISSING model $m: ${MPATH[$m]}"; exit 1; }
    fi
done
echo "MODELS=[$MODELS]  MODELSET=$MODELSET ($SUB)  step=$STEP  out=$OUT"

# ---- 0) prep: contig preflight + packed null.fasta + null_regions.bed (pure python, main env) ----
conda activate "$MAIN_ENV"
echo "=== prep (contig preflight + packed null) ==="
python "$T27/fdr_stats.py" prep --genome "$GENOME" --gff "$GFF" --null-tsv $NULL_TSVS --out-dir "$OUT"
NULL_FASTA=$OUT/null.fasta
NULL_REGIONS=$OUT/null_regions.bed
for f in "$NULL_FASTA" "$NULL_REGIONS"; do
    [[ -s "$f" ]] || { echo "FATAL: prep did not produce $f (deleted results/? stale clone?)"; exit 1; }
done

# ---- 1) candidate regions BED (main env; contig ids come straight from the GFF) ----
# Cheap, deterministic output derived from the GFF — safe to delete, always regenerable. Rebuild it
# whenever it's missing/empty, then HARD-ASSERT it exists so a deleted results/ can never resurface as
# an opaque FileNotFoundError inside a scanner further down.
REGIONS=$OUT/cbut_candidate_regions.bed
if [[ ! -s "$REGIONS" ]]; then
    echo "=== make_candidate_regions (up $UP / down $DOWN) ==="
    [[ -f "$EVAL/make_candidate_regions.py" ]] || { echo "FATAL: missing $EVAL/make_candidate_regions.py — is the clone up to date? (git pull)"; exit 1; }
    python "$EVAL/make_candidate_regions.py" --gff "$GFF" --up "$UP" --down "$DOWN" --out "$REGIONS"
else echo "=== skip candidate regions (exists) ==="; fi
[[ -s "$REGIONS" ]] || { echo "FATAL: candidate regions BED not created at $REGIONS — check GFF $GFF"; exit 1; }

# ---- 2) per-model scans: candidate (restricted, +/-) + null (packed nullcat, +, step 81), cached ----
CAND_ALIGN=(); NULL_ALIGN=()
for m in $MODELS; do
    E=$(env_for "$m")
    CT=$OUT/cand_${m}.csv ; NULLTRACK=$OUT/null_${m}.csv
    # candidate scan (restricted to the upstream regions; both strands)
    if [[ ! -s "$CT" ]]; then
        TMP="$CT.${SLURM_JOB_ID:-$$}.tmp" ; conda activate "$E"
        echo "=== scan candidate $m (step $STEP, restricted) ==="
        if [[ "$m" == evo2 ]]; then
            python "$EVO2/evo2_scan_restricted.py" --genome "$GENOME" --regions "$REGIONS" --probe "$PROBE" \
                --window 81 --step "$STEP" --strands +,- --align-grid --batch-size 16 --out "$TMP"
        else
            python "$EVAL/genome_inference.py" --genome "$GENOME" --model-path "${MPATH[$m]}" $(extra_args "$m") \
                --window 81 --step "$STEP" --strands +,- --regions "$REGIONS" --out "$TMP"
        fi
        mv -f "$TMP" "$CT"
    else echo "=== skip candidate $m (track exists) ==="; fi
    # null scan (the packed nullcat contig; + only, step 81 so window == block; evo2 needs the whole-contig BED)
    if [[ ! -s "$NULLTRACK" ]]; then
        TMP="$NULLTRACK.${SLURM_JOB_ID:-$$}.tmp" ; conda activate "$E"
        echo "=== scan null $m (packed nullcat, step 81, +) ==="
        if [[ "$m" == evo2 ]]; then
            python "$EVO2/evo2_scan_restricted.py" --genome "$NULL_FASTA" --regions "$NULL_REGIONS" --probe "$PROBE" \
                --window 81 --step 81 --strands + --align-grid --batch-size 16 --out "$TMP"
        else
            python "$EVAL/genome_inference.py" --genome "$NULL_FASTA" --model-path "${MPATH[$m]}" $(extra_args "$m") \
                --window 81 --step 81 --strands + --out "$TMP"
        fi
        mv -f "$TMP" "$NULLTRACK"
    else echo "=== skip null $m (track exists) ==="; fi
    CAND_ALIGN+=(--track "$m" "$CT") ; NULL_ALIGN+=(--track "$m" "$NULLTRACK")
done

# ---- 3) reconcile candidate tracks and null tracks each onto one common grid (main env) ----
conda activate "$MAIN_ENV"
echo "=== align_tracks (candidates) ==="
python "$T29DIR/align_tracks.py" "${CAND_ALIGN[@]}" --out-dir "$OUT/aligned_cand"
echo "=== align_tracks (null) ==="
python "$T29DIR/align_tracks.py" "${NULL_ALIGN[@]}" --out-dir "$OUT/aligned_null"

# ---- 4) FDR statistics: ensemble + empirical p/q (main env, pure python, no GPU) ----
echo "=== fdr_stats (ensemble + p/q) ==="
python "$T27/fdr_stats.py" fdr --models $MODELS \
    --cand-dir "$OUT/aligned_cand" --null-dir "$OUT/aligned_null" \
    --genome "$GENOME" --peak-alpha "$PEAK_ALPHA" --merge "$MERGE" --top-n "$TOPN" --out-dir "$OUT"

# ---- 5) motif detection on the top-N (bio env; non-fatal if MEME/fimo unavailable) ----
TOP_CSV=$OUT/top${TOPN}_promoters.csv
echo "=== annotate_motifs (FIMO -10/-35 on top-$TOPN) ==="
conda activate "$BIO_ENV"
python "$T31DIR/annotate_motifs.py" --peaks "$TOP_CSV" --genome "$GENOME" \
    --motif "$T31DIR/sigma_a_promoter.meme" --window 81 --fimo-thresh 1e-3 \
    --out "$OUT/top${TOPN}_promoters_motif.csv" \
    || echo "[motif] FIMO step failed (bio env / meme?). Top-N table without motif columns is still written."

echo ""
echo "=== DONE ===  outputs in $OUT/"
echo "  peaks_fdr.csv               <- full FDR table (every locus, p/q)"
echo "  top${TOPN}_promoters.csv        <- the mine deliverable (ranked promoters + summit sequence)"
echo "  top${TOPN}_promoters_motif.csv  <- + FIMO -10/-35 architecture"
echo "  analysis.md                 <- summary report"
