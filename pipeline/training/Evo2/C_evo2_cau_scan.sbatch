#!/bin/bash
#SBATCH --job-name=evo2_cau_regimeC
#SBATCH --gpus-per-node=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --time=12:00:00
#SBATCH --output=evo2_cau_regimeC_%j.log
#
# Historical restricted benchmark on C. autoethanogenum (GCF_000484505.1, NC_022592.1).
# Its published TSS labels are withheld, but the genome supplied silver/negative
# training windows. This is not wholly unseen-genome evaluation.
#
# Reuses the same T15_xs logistic probe as the ljungdahlii scan and compares at step 10.
# Matching step and benchmark settings control those choices, not genome independence,
# catalogue completeness or exact reverse-strand grid alignment. The deployed ensemble
# uses evo2_scan_restricted.py --align-grid and intersects tracks before aggregation.
# This historical standalone script does not use that alignment flag.
# Benchmark settings: tol 40, merge 81, window 81, strain-specific truth and regions.
#
# Reuses cau.fna / cau_candidate_regions.bed / cau_truth.tsv staged by prep_cau_regimeC.sh, and the
# cnn_v2-xs cau track from C_cau_scan.sbatch. Requires the flash-attn ABI fix already applied to the
# evo2 env (torch2.7-matched wheel).
# Submit:  sbatch pipeline/training/Evo2/C_evo2_cau_scan.sbatch
set -eo pipefail

REPO="${PROMOTER_REPO:?set it or source env.sh}"
DIR=$REPO/workflow/results
EVAL=$REPO/pipeline/evaluation
EVO2=$REPO/pipeline/training/Evo2
RC=$DIR/regimeC
OUT=$RC/evo2_cau ; mkdir -p "$OUT"
EMB=$REPO/evo2_embeddings                    # cached T15_xs embeddings (only used if the probe must be rebuilt)
PROBE=$RC/evo2/evo2_probe_xs.pkl             # REUSE the ljung-trained T15_xs probe -> identical probe on both genomes
EVO2_ENV="${ENV_EVO2:?set it or source env.sh}"
MAIN_ENV="${ENV_MAIN:?set it or source env.sh}"
CONDA_BASE=$(conda info --base)

FASTA=$RC/cau.fna                            # staged by prep_cau_regimeC.sh (record NC_022592.1)
REGIONS=$RC/cau_candidate_regions.bed
TRUTH=$RC/cau_truth.tsv                      # contig,pos,strand (2,885 primary TSS)
CNN_TRACK=$RC/cau_cnn_v2_xs.csv              # cnn_v2-xs step-1 track from the cau base-vs-xs scan
STEP=10
RECALLS=0.1,0.5,0.7,0.9
TRACK=$OUT/cau_evo2_track.csv

if [ -n "${HF_HOME:-}" ]; then export HF_HOME; fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"   # 1 = require a pre-staged cache
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "=== preflight ==="
for f in "$FASTA" "$REGIONS" "$TRUTH"; do
    [[ -f "$f" ]] || { echo "MISSING $f"; exit 1; }
done
# contig-id guard: the Evo2 scanner emits the FASTA header as `contig` and can't override it, so the
# FASTA id must match the regions/truth id (NC_022592.1) or it silently scores zero windows.
FID=$(grep -m1 '^>' "$FASTA" | sed 's/^>//; s/[[:space:]].*//')
REG_CONTIGS=$(tail -n +2 "$REGIONS" | cut -f1 | sort -u)
if ! printf '%s\n' "$REG_CONTIGS" | grep -qx "$FID"; then
    echo "ERROR: FASTA record id '$FID' is not among the regions' contigs: $REG_CONTIGS"; exit 1
fi
echo "FASTA id=$FID  truth=$(($(wc -l <"$TRUTH")-1)) TSS  step=$STEP"

# --- probe: reuse the ljung T15_xs probe; rebuild only if it is somehow missing ---
conda activate "$EVO2_ENV"
if [[ ! -s "$PROBE" ]]; then
    echo "=== probe missing — rebuild from $EMB (want ~44k train, TEST AUPRC ~0.70 => T15_xs) ==="
    [[ -d "$EMB" ]] || { echo "MISSING embeddings dir $EMB"; exit 1; }
    python "$EVO2/evo2_promoter_classifier.py" --emb-dir "$EMB" --out "$PROBE"
else
    echo "=== reusing probe: $PROBE (identical to the ljung run) ==="
fi

# --- 1) Evo2 restricted scan (GPU); resume-safe ---
if [[ ! -s "$TRACK" ]]; then
    echo "=== Evo2 scan (7B, step $STEP, both strands) ==="
    python "$EVO2/evo2_scan_restricted.py" --genome "$FASTA" --regions "$REGIONS" \
        --probe "$PROBE" --window 81 --step "$STEP" --batch-size 16 --out "$TRACK.tmp"
    mv "$TRACK.tmp" "$TRACK"
else
    echo "=== track exists: $TRACK (rm to force rescan) ==="
fi

# --- 2) benchmark Evo2 restricted (main env) ---
conda activate "$MAIN_ENV"
python "$EVAL/genome_fp_benchmark.py" --scores "$TRACK" --truth "$TRUTH" \
    --recall-targets "$RECALLS" --regions "$REGIONS" --out "$OUT/fp_cau_evo2_restrict"

# --- 3) matched step-10 cnn_v2-xs fairness benchmark on cau ---
if [[ -s "$CNN_TRACK" ]]; then
    S10=$OUT/cau_cnn_v2_xs_step10.csv
    [[ -s "$S10" ]] || awk -F, 'NR==1{print;next} ($2%10)==0' "$CNN_TRACK" > "$S10"
    python "$EVAL/genome_fp_benchmark.py" --scores "$S10" --truth "$TRUTH" \
        --recall-targets "$RECALLS" --regions "$REGIONS" --out "$OUT/fp_cau_cnn_v2_step10_restrict"
else
    echo "WARN: $CNN_TRACK not found — skipping cnn_v2 step-10 fairness (subsample+benchmark it later)."
fi

# --- 4) head-to-head ---
echo ""
echo "=== DONE — C. autoethanogenum head-to-head (restricted, recall 0.1 & 0.5) ==="
echo "--- Evo2 @step10 ---"
grep -E "0\.(10|50) \(got" "$OUT/fp_cau_evo2_restrict/genome_fp_benchmark.md" || true
echo "--- cnn_v2-xs @step10 (matched) ---"
grep -E "0\.(10|50) \(got" "$OUT/fp_cau_cnn_v2_step10_restrict/genome_fp_benchmark.md" 2>/dev/null || echo "(cnn_v2 step10 not run)"
echo ""
echo "compare against ljungdahlii: Evo2 71.0%/58.0% vs cnn_v2-xs@step10 63.1%/48.2% at recall 0.1/0.5."
