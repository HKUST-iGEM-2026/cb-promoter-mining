#!/bin/bash
#SBATCH --job-name=t27_fdr_mine
#SBATCH --account=igem2026
#SBATCH --partition=preempt
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
#                                  evo2 -> evo2_scan_restricted.py (evo2 env, --align-grid); others ->
#                                  genome_inference.py (main env; nt -> nt-tf4x). Resume-safe (cached tracks).
#   3) align_tracks.py x2        -> reconcile candidate tracks and null tracks each onto one common grid
#   4) fdr_stats.py fdr          -> ensemble + empirical p/q -> peaks_fdr.csv + topN_promoters.csv + analysis.md
#   5) annotate_motifs.py        -> FIMO -10/-35 on the top-N (bio env)
#
# MODELS = the deployment 3-model ensemble by default (ProkBERT + cnn_v2 + Evo2). Tunable: any space-
# separated subset of {prokbert cnn cnn_v2 nt evo2}. evo2 is special-cased (own scanner + env); the rest
# go through genome_inference. T29 rank-percentile is calibration-invariant, so members combine on raw
# scores. Per-model tracks are cached in $OUT and reused, so re-running with a different MODELS set only
# scans the NEW model.
#
# ⚠️ The null TSV(s) MUST be 81 bp (extract_negatives.py defaults to --window 52 — regenerate at 81, or
# fdr_stats.py prep fails loud). Run only ONE job per $OUT at a time (same-$OUT jobs share track files).
#
# SUBMIT (deployment 3-model, xs checkpoints — the default):
#   sbatch evaluation-and-calibration/T27/run_fdr_mine.sbatch
# SUBMIT (4-model, add CNN-v1; reuses the 3 cached tracks, scans only cnn):
#   MODELS="prokbert cnn cnn_v2 evo2" sbatch evaluation-and-calibration/T27/run_fdr_mine.sbatch
# SUBMIT (the leakage-safe 0.6 checkpoints instead of xs):
#   MODELSET=06 sbatch evaluation-and-calibration/T27/run_fdr_mine.sbatch
set -eo pipefail

REPO=/project/igem2026/cb-promoter-mining
DIR=$REPO/infrastructure/T3/snakemake_scaffold/results
EVAL=$REPO/evaluation-and-calibration
EVO2=$REPO/model-training/Evo2
T27=$REPO/evaluation-and-calibration/T27
T29DIR=$REPO/mining-and-motif-annotation/T29
T31DIR=$REPO/mining-and-motif-annotation/T31
ANN=$REPO/evaluation-and-calibration/T26/c_butyricum_annotation
MAIN_ENV=/project/igem2026/envs/promoter-mining-hpc4
NT_ENV=/project/igem2026/envs/nt-tf4x
EVO2_ENV=/project/igem2026/envs/evo2
BIO_ENV=/project/igem2026/envs/meme
CONDA_BASE=/cm/shared/apps/Anaconda3/2023.09-0

# ---- config (override from the submit line) ----
MODELSET=${MODELSET:-xs}                            # xs (T21_xs, DEPLOYED) or 06 (T21_final_06, leakage-safe)
case "$MODELSET" in xs) SUB=T21_xs ;; 06) SUB=T21_final_06 ;; *) echo "bad MODELSET '$MODELSET' (xs|06)"; exit 1 ;; esac
MODELS=${MODELS:-"prokbert cnn_v2 evo2"}            # the deployment 3-model ensemble
STEP=${STEP:-10}                                    # candidate scan stride (10 = the validated regimeC grid; 1 = finest but ~10x evo2 cost). Null is always step 81.
PEAK_ALPHA=${PEAK_ALPHA:-0.05}
MERGE=${MERGE:-81}
TOPN=${TOPN:-80}
UP=${UP:-300}; DOWN=${DOWN:-50}                     # candidate region = [gene_start-UP, gene_start+DOWN]
GENOME=${GENOME:-$ANN/C_butyricum.fna}
GFF=${GFF:-$ANN/C_butyricum.gff3}
NULL_TSVS=${NULL_TSVS:-"$ANN/convergent_intergenic_negatives.tsv $EVAL/T26/c_acetobutylicum_annotation/convergent_intergenic_negatives.tsv $EVAL/T26/c_phytofermentans_annotation/convergent_intergenic_negatives.tsv"}
PROBE=${PROBE:-$DIR/regimeC/evo2/evo2_probe_xs.pkl}   # frozen T15_xs logreg probe (genome-agnostic)
OUT=${OUT:-$T27/results/cbut_${MODELSET}} ; mkdir -p "$OUT"

declare -A MPATH=(                                  # genome_inference models (evo2 is separate)
    [prokbert]="$DIR/$SUB/prokbert/best_model"
    [cnn]="$DIR/$SUB/cnn/best_model.pt"
    [cnn_v2]="$DIR/$SUB/cnn_v2/best_model.pt"
    [nt]="$DIR/$SUB/nt_lora/best_adapter"
)
extra_args() { case "$1" in
    prokbert) echo "--model-id neuralbioinfo/prokbert-mini --batch-size 512" ;;
    cnn_v2)   echo "--model-type cnn_v2 --batch-size 512" ;;
    cnn)      echo "--model-type cnn --batch-size 512" ;;
    nt)       echo "--model-id InstaDeepAI/nucleotide-transformer-v2-250m-multi-species --batch-size 256" ;;
esac; }
env_for() { case "$1" in nt) echo "$NT_ENV" ;; evo2) echo "$EVO2_ENV" ;; *) echo "$MAIN_ENV" ;; esac; }

export HF_HOME=/project/igem2026/hf_shared HF_HUB_OFFLINE=1 HF_TOKEN_PATH=/tmp/hf_notoken_$USER
export PYTHONNOUSERSITE=1 MKL_THREADING_LAYER=GNU
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "=== preflight ==="
for f in "$GENOME" "$GFF"; do [[ -f "$f" ]] || { echo "MISSING $f"; exit 1; }; done
for m in $MODELS; do
    if [[ "$m" == evo2 ]]; then
        [[ -f "$PROBE" ]] || { echo "MISSING Evo2 probe $PROBE"; exit 1; }
    else
        [[ -n "${MPATH[$m]:-}" ]] || { echo "unknown model '$m' (known: prokbert cnn cnn_v2 nt evo2)"; exit 1; }
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

# ---- 1) candidate regions BED (main env; contig ids come straight from the GFF) ----
REGIONS=$OUT/cbut_candidate_regions.bed
if [[ ! -s "$REGIONS" ]]; then
    echo "=== make_candidate_regions (up $UP / down $DOWN) ==="
    python "$EVAL/make_candidate_regions.py" --gff "$GFF" --up "$UP" --down "$DOWN" --out "$REGIONS"
else echo "=== skip candidate regions (exists) ==="; fi

# ---- 2) per-model scans: candidate (restricted, +/-) + null (packed nullcat, +, step 81), cached ----
CAND_ALIGN=(); NULL_ALIGN=()
for m in $MODELS; do
    E=$(env_for "$m")
    CT=$OUT/cand_${m}.csv ; NT=$OUT/null_${m}.csv
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
    if [[ ! -s "$NT" ]]; then
        TMP="$NT.${SLURM_JOB_ID:-$$}.tmp" ; conda activate "$E"
        echo "=== scan null $m (packed nullcat, step 81, +) ==="
        if [[ "$m" == evo2 ]]; then
            python "$EVO2/evo2_scan_restricted.py" --genome "$NULL_FASTA" --regions "$NULL_REGIONS" --probe "$PROBE" \
                --window 81 --step 81 --strands + --align-grid --batch-size 16 --out "$TMP"
        else
            python "$EVAL/genome_inference.py" --genome "$NULL_FASTA" --model-path "${MPATH[$m]}" $(extra_args "$m") \
                --window 81 --step 81 --strands + --out "$TMP"
        fi
        mv -f "$TMP" "$NT"
    else echo "=== skip null $m (track exists) ==="; fi
    CAND_ALIGN+=(--track "$m" "$CT") ; NULL_ALIGN+=(--track "$m" "$NT")
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
