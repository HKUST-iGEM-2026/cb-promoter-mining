#!/bin/bash
#SBATCH --job-name=evo2_clj_regimeC
#SBATCH --gpus-per-node=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --time=12:00:00
#SBATCH --output=evo2_clj_regimeC_%j.log
#
# COMPLETENESS RUN — the one axis Evo2 was never tested on: genome-scale (Regime C) restricted scan on
# the held-out C. ljungdahlii genome, through the SAME restricted benchmark as the four ensemble models.
# So the writeup can say "evaluated at genome scale too", with a real number instead of an assumption.
#
# Stated up front (not a yes-man): this is expected to RECONFIRM the drop. Evo2's balanced AUPRC (0.70)
# is already below the weakest ensemble member (CNN 0.80), it is a FROZEN model (never learned -10/-35),
# and genome scale is strictly harder (~0.2% base rate). The observed balanced->genome pattern in this
# project is models *dropping*, never rising from last place. This run tests that; it does not assume it.
#
# Pipeline:
#   0. (CPU) retrain the probe from the cached T15_xs embeddings with the FIXED classifier
#      (weighted-AUPRC selection) -> provenance is T15_xs, NOT the committed pkl (likely Sine's stale
#      T15_ctx run). Eyeball its printout: ~44k train rows and TEST AUPRC ~0.70 = the right probe.
#      If you see ~33.7k train / AUPRC via AUROC-looking numbers, the embeddings are stale T15_ctx —
#      re-run extract_embeddings.sbatch first.
#   1. (GPU, evo2 env) evo2_scan_restricted.py: score candidate-region windows (step 10) -> track.
#   2. (CPU, main env) genome_fp_benchmark.py restricted + full vs the 1,245 ljungdahlii TSS.
#
# Reuses clj.fna + clj_candidate_regions.bed already staged by C_ljung_scan_xs.sh (the xs 4-model scan).
# Submit:  sbatch model-training/Evo2/C_evo2_ljung_scan.sbatch
set -eo pipefail

REPO="${PROMOTER_REPO:?set it or source env.sh}"
DIR=$REPO/infrastructure/T3/snakemake_scaffold/results
EVAL=$REPO/evaluation-and-calibration
EVO2=$REPO/model-training/Evo2
OUT=$DIR/regimeC/evo2 ; mkdir -p "$OUT"
EMB=$REPO/evo2_embeddings                        # cached T15_xs embeddings (from extract_embeddings.sbatch)
PROBE=$OUT/evo2_probe_xs.pkl
EVO2_ENV="${ENV_EVO2:?set it or source env.sh}"
MAIN_ENV="${ENV_MAIN:?set it or source env.sh}"
CONDA_BASE=$(conda info --base)

FASTA=${CLJ_FASTA:-$DIR/regimeC/clj.fna}         # staged by the xs ljung scan (record MUST be NC_014328.1)
REGIONS=$DIR/regimeC/clj_candidate_regions.bed   # model-independent; reused
TRUTH=$EVAL/ljungdahlii_holdout_tss.tsv          # 1,245 held-out primary TSS (committed; contig,pos,strand)
STEP=10                                          # 7B model over ~1.4 Mb restricted -> ~280k windows. Feasible.
RECALLS=0.1,0.5,0.7,0.9
TRACK=$OUT/clj_evo2_track.csv

if [ -n "${HF_HOME:-}" ]; then export HF_HOME; fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"   # 1 = require a pre-staged cache
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "=== preflight ==="
for f in "$FASTA" "$REGIONS" "$TRUTH"; do
    [[ -f "$f" ]] || { echo "MISSING $f"; exit 1; }
done
[[ -s "$PROBE" || -d "$EMB" ]] || { echo "MISSING embeddings dir $EMB — run extract_embeddings.sbatch first"; exit 1; }

# contig-id guard: evo2_scan_restricted.py emits the FASTA header as `contig` and (unlike
# genome_inference.py) has NO --contig-name override — it only scores windows whose (contig,strand) is
# in the regions BED. If the FASTA id != the regions/truth id (NC_014328.1, e.g. a Bakta-renamed
# `contig_1`), it silently scores ZERO windows. Catch that here.
FID=$(grep -m1 '^>' "$FASTA" | sed 's/^>//; s/[[:space:]].*//')
REG_CONTIGS=$(tail -n +2 "$REGIONS" | cut -f1 | sort -u)
if ! printf '%s\n' "$REG_CONTIGS" | grep -qx "$FID"; then
    echo "ERROR: FASTA record id '$FID' is not among the regions' contigs: $REG_CONTIGS"
    echo "  The Evo2 scanner writes the FASTA header as 'contig' and can't override it, so a mismatch"
    echo "  => empty track. Stage a FASTA whose header is the RefSeq id NC_014328.1, or rebuild the"
    echo "  regions+truth with id '$FID'."
    exit 1
fi
echo "FASTA=$FASTA (id $FID)  regions=$REGIONS  truth=$(($(wc -l <"$TRUTH")-1)) TSS  step=$STEP"

# --- 0) retrain probe from cached T15_xs embeddings (CPU; evo2 env has sklearn) ---
conda activate "$EVO2_ENV"
if [[ ! -s "$PROBE" ]]; then
    echo "=== retrain probe from $EMB  (want ~44k train, TEST AUPRC ~0.70 => T15_xs provenance) ==="
    python "$EVO2/evo2_promoter_classifier.py" --emb-dir "$EMB" --out "$PROBE"
else
    echo "=== probe exists: $PROBE (rm to force retrain) ==="
fi

# --- 1) restricted Evo2-7B scan (GPU) — resume-safe (.tmp then atomic mv) ---
if [[ ! -s "$TRACK" ]]; then
    echo "=== scan (Evo2-7B, step $STEP, both strands) ==="
    python "$EVO2/evo2_scan_restricted.py" --genome "$FASTA" --regions "$REGIONS" \
        --probe "$PROBE" --window 81 --step "$STEP" --batch-size 16 --out "$TRACK.tmp"
    mv "$TRACK.tmp" "$TRACK"
else
    echo "=== track exists: $TRACK (rm to force rescan) ==="
fi

# --- 2) benchmark restricted + full vs truth (CPU, main env) ---
conda activate "$MAIN_ENV"
echo "=== benchmark (restricted + full) ==="
python "$EVAL/genome_fp_benchmark.py" --scores "$TRACK" --truth "$TRUTH" \
    --recall-targets "$RECALLS" --regions "$REGIONS" --out "$OUT/fp_evo2_restrict"
python "$EVAL/genome_fp_benchmark.py" --scores "$TRACK" --truth "$TRUTH" \
    --recall-targets "$RECALLS" --out "$OUT/fp_evo2_full"

echo ""
echo "=== DONE ==="
echo "--- Evo2 RESTRICTED, recall 0.1 & 0.5 (compare to cnn_v2-xs / the ensemble on THIS genome) ---"
grep -E "0\.(10|50) \(got" "$OUT/fp_evo2_restrict/genome_fp_benchmark.md" || true
echo ""
echo "track     -> $TRACK"
echo "benchmark -> $OUT/fp_evo2_restrict/genome_fp_benchmark.md"
echo "CAVEAT: this Evo2 track is step-$STEP (restricted); the four ensemble tracks are step-1. Recall"
echo "  ceiling is preserved (tol 40 bp >> step $STEP), but read FP-peaks/precision as approximate when"
echo "  comparing across the step difference."
