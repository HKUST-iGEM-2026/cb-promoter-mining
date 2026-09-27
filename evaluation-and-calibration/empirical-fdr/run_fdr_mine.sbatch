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
# (own scanner + env); prokbert/cnn_v2 go through genome_inference. T29 rank-percentile is
# calibration-invariant, so members combine on raw scores. Per-model tracks are cached in $OUT and reused.
# (You can restrict to a subset, e.g. MODELS="prokbert evo2", but not add other models.)
#
# ⚠️ The null TSV(s) MUST be 81 bp (extract_negatives.py defaults to --window 52 — regenerate at 81, or
# fdr_stats.py prep fails loud). Run only ONE job per $OUT at a time (same-$OUT jobs share track files).
#
# ⚠️ ASSEMBLY MUST MATCH THE DELIVERABLE. The candidate list (g409755_candidates.tsv + the 912/306/100
# tiers) was mined on the 96-contig RefSeq WGS assembly g409755 = GCF_000409755.1 (contig ids
# NZ_AQQF01000…). This FDR run MUST use the SAME assembly, or its peaks won't share coordinates with the
# candidate table and can't be joined. The earlier results/cbut_xs/ run used T26's C_butyricum.fna — a
# DIFFERENT 4-contig assembly (contig_1..4) — so those q-values do NOT map to the deliverable.
# Set G409755_FASTA / G409755_GFF below to the SAME files the mine used.
#
# SUBMIT (deployment 3-model, xs checkpoints — the default), after setting the two g409755 paths:
#   G409755_FASTA=/abs/g409755.fna G409755_GFF=/abs/g409755.gff3 \
#     sbatch evaluation-and-calibration/empirical-fdr/run_fdr_mine.sbatch
#   (leakage-safe 0.6 checkpoints instead of xs: add  MODELSET=06)
#
# To guarantee the FDR windows are byte-identical to the mine's (recommended), copy the mine's regions BED
# into the output dir BEFORE submitting so T27 reuses it instead of rebuilding from the GFF:
#   OUT=evaluation-and-calibration/empirical-fdr/results/cbut_g409755_xs ; mkdir -p "$OUT"
#   cp <mine>/g409755_candidate_regions.bed  "$OUT/cbut_candidate_regions.bed"
# (Optional, skips the GPU candidate scan entirely: also drop the mine's per-model g409755 tracks there as
#  cand_prokbert.csv / cand_cnn_v2.csv / cand_evo2.csv — T27 is resume-safe and reuses existing tracks.)
set -eo pipefail

REPO="${PROMOTER_REPO:?set it or source env.sh}"
DIR=$REPO/infrastructure/T3/snakemake_scaffold/results
EVAL=$REPO/evaluation-and-calibration
EVO2=$REPO/model-training/Evo2
T27=$REPO/evaluation-and-calibration/empirical-fdr
T29DIR=$REPO/mining-and-motif-annotation/ensemble-aggregation
T31DIR=$REPO/mining-and-motif-annotation/motif-annotation
ANN=$REPO/evaluation-and-calibration/calibration/c_butyricum_annotation
MAIN_ENV="${ENV_MAIN:?set it or source env.sh}"
EVO2_ENV="${ENV_EVO2:?set it or source env.sh}"
BIO_ENV="${ENV_MEME:?set it or source env.sh}"
CONDA_BASE=$(conda info --base)

# ---- config (override from the submit line) ----
MODELSET=${MODELSET:-xs}                            # xs (T21_xs, DEPLOYED) or 06 (T21_final_06, leakage-safe)
case "$MODELSET" in xs) SUB=T21_xs ;; 06) SUB=T21_final_06 ;; *) echo "bad MODELSET '$MODELSET' (xs|06)"; exit 1 ;; esac
MODELS=${MODELS:-"prokbert cnn_v2 evo2"}            # the deployment 3-model ensemble (only these 3 supported)
STEP=${STEP:-10}                                    # candidate scan stride (10 = the validated regimeC grid; 1 = finest but ~10x evo2 cost). Null is always step 81.
PEAK_ALPHA=${PEAK_ALPHA:-0.05}
MERGE=${MERGE:-81}
TOPN=${TOPN:-80}
UP=${UP:-300}; DOWN=${DOWN:-50}                     # candidate region = [gene_start-UP, gene_start+DOWN]
# TARGET ASSEMBLY — set to the SAME g409755 files the mine used (mine_3model.sbatch FASTA + the GFF the
# candidate-regions BED was built from). RefSeq WGS, contig ids NZ_AQQF01000…  ⚠️ NOT T26's 4-contig
# C_butyricum.fna — that mismatch is why the old cbut_xs q-values don't map to the deliverable.
G409755_FASTA=${G409755_FASTA:-${PROMOTER_REPO}/PATH_TO/g409755.fna}   # <-- SET ME
G409755_GFF=${G409755_GFF:-${PROMOTER_REPO}/PATH_TO/g409755.gff3}      # <-- SET ME
GENOME=${GENOME:-$G409755_FASTA}
GFF=${GFF:-$G409755_GFF}
# Null pool stays as-is: it is scored INDEPENDENTLY (packed nullcat) and never shares coordinates with the
# candidates, so promoter-free C. butyricum intergenic sequence is a valid composition-matched null whatever
# assembly it came from. (Regenerating the cbut null from g409755 is optional, not required for correctness.)
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
