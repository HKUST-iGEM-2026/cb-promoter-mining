#!/bin/bash
#SBATCH --job-name=rc_s10_all
#SBATCH --account=igem2026
#SBATCH --partition=preempt
#SBATCH --gpus-per-node=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --time=23:59:00
#SBATCH --output=rc_s10_all_%j.log
#
# Unified STEP-10 Regime C — the missing rows for the six-model grid.
# Scans the deployed xs ProkBERT / CNN / NT-LoRA (+ cnn_v2 as a control) across BOTH held-out
# validation genomes (C. ljungdahlii, C. autoethanogenum) at step 10, then benchmarks each RESTRICTED
# through the SAME genome_fp_benchmark.py as the Evo2 / cnn_v2-xs / NTv3-ctx runs. Result: every model
# finally sits in ONE apples-to-apples table (same step, same xs lineage, same benchmark, both genomes).
#
# WHY THIS EXISTS: the species-report four-model Regime C is step-1 and C. ljungdahlii ONLY; Evo2 /
# cnn_v2-xs / NTv3-ctx are step-10 on clj + cau. So ProkBERT / CNN / NT-LoRA had (a) no step-10 numbers
# and (b) no autoethanogenum scan at all. This job produces both. cnn_v2_xs is re-scanned here as a
# POSITIVE CONTROL: it must reproduce the Evo2 report's cnn_v2-xs numbers (clj 63.1%/48.2%,
# cau 81.2%/67.1% @ recall 0.1/0.5); if it does, the three new rows are trustworthy.
#
# PREREQ: $RC already holds clj.fna, cau.fna, {clj,cau}_candidate_regions.bed, cau_truth.tsv (staged by
#   the Evo2 / NTv3 runs; the NTv3 regimeC job used all of them). ljungdahlii truth is committed in $EVAL.
#   Confirm the T21_xs model paths below (same set as scan_ensemble.sbatch / C_cau_scan.sbatch).
# SUBMIT: sbatch ~/regimeC_step10_allmodels.sbatch
#   overrides (no edit needed): STEP=1  GENOMES=clj  MODELS="prokbert cnn nt"  RECALLS=0.1,0.5,0.7,0.9
# Resume-safe: a completed per-(genome,model) track is skipped on requeue.
set -eo pipefail

REPO=/project/igem2026/cb-promoter-mining
DIR=$REPO/infrastructure/T3/snakemake_scaffold/results
EVAL=$REPO/evaluation-and-calibration
RC=$DIR/regimeC
OUT=$RC/step10_allmodels ; mkdir -p "$OUT"
MAIN_ENV=/project/igem2026/envs/promoter-mining-hpc4
NT_ENV=/project/igem2026/envs/nt-tf4x
CONDA_BASE=/cm/shared/apps/Anaconda3/2023.09-0
STEP=${STEP:-10}
RECALLS=${RECALLS:-0.1,0.5,0.7,0.9}
GENOMES=${GENOMES:-clj cau}
MODELS=${MODELS:-prokbert nt cnn cnn_v2}          # fast-first; cnn_v2 (slow) last; cnn_v2 = control

export HF_HOME=/project/igem2026/hf_shared HF_HUB_OFFLINE=1 HF_TOKEN_PATH=/tmp/hf_notoken_$USER
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---- deployed xs (cross-species-augmented) checkpoints — same set scan_ensemble.sbatch deploys ----
declare -A MPATH=(
    [prokbert]="$DIR/T21_xs/prokbert/best_model"
    [nt]="$DIR/T21_xs/nt_lora/best_adapter"
    [cnn]="$DIR/T21_xs/cnn/best_model.pt"
    [cnn_v2]="$DIR/T21_xs/cnn_v2/best_model.pt"
)
# per-genome FASTA / candidate-region BED / truth (contig ids in the FASTA already match the truth)
declare -A FASTA=(   [clj]="$RC/clj.fna"                       [cau]="$RC/cau.fna" )
declare -A REGIONS=( [clj]="$RC/clj_candidate_regions.bed"     [cau]="$RC/cau_candidate_regions.bed" )
declare -A TRUTH=(   [clj]="$EVAL/ljungdahlii_holdout_tss.tsv" [cau]="$RC/cau_truth.tsv" )

extra_args() { case "$1" in
    prokbert) echo "--model-id neuralbioinfo/prokbert-mini --batch-size 512" ;;
    nt)       echo "--model-id InstaDeepAI/nucleotide-transformer-v2-250m-multi-species --batch-size 256" ;;
    cnn_v2)   echo "--model-type cnn_v2 --batch-size 512" ;;
    cnn)      echo "--model-type cnn --batch-size 512" ;;
esac; }
env_for() { case "$1" in nt) echo "$NT_ENV" ;; *) echo "$MAIN_ENV" ;; esac; }

echo "=== preflight ==="
for g in $GENOMES; do
    for f in "${FASTA[$g]}" "${REGIONS[$g]}" "${TRUTH[$g]}"; do
        [[ -f "$f" ]] || { echo "MISSING $f  (stage via prep_cau_regimeC.sh / the ljung prep)"; exit 1; }
    done
done
for m in $MODELS; do
    [[ -e "${MPATH[$m]}" ]] || { echo "MISSING model $m: ${MPATH[$m]} — fix the MPATH table"; exit 1; }
done
echo "genomes=[$GENOMES]  models=[$MODELS]  step=$STEP  out=$OUT"

for g in $GENOMES; do
    for m in $MODELS; do
        TRACK=$OUT/s${STEP}_${g}_${m}_xs.csv
        # 1) scan (skip if complete; .tmp -> atomic mv so a preempt can't leave a partial track)
        if [[ ! -s "$TRACK" ]]; then
            conda activate "$(env_for "$m")"
            echo "=== scan $g/$m  (env $(env_for "$m"), step $STEP) ==="
            python "$EVAL/genome_inference.py" --genome "${FASTA[$g]}" --model-path "${MPATH[$m]}" \
                $(extra_args "$m") --emit-logit --window 81 --step "$STEP" --out "$TRACK.tmp"
            mv "$TRACK.tmp" "$TRACK"
        else
            echo "=== skip $g/$m (track exists: $TRACK) ==="
        fi
        # 2) benchmark RESTRICTED (the deployment-relevant view), in the MAIN env
        conda activate "$MAIN_ENV"
        python "$EVAL/genome_fp_benchmark.py" --scores "$TRACK" --truth "${TRUTH[$g]}" \
            --recall-targets "$RECALLS" --regions "${REGIONS[$g]}" --out "$OUT/fp_${g}_${m}_restrict"
        echo "### $g/$m (restricted):"
        grep -E "0\.(10|50) \(got" "$OUT/fp_${g}_${m}_restrict/genome_fp_benchmark.md" || true
    done
done

# ---------------- collate: one grid file for every (genome, model) this job produced ----------------
GRID=$OUT/regimeC_step10_grid.md
{
    echo "# Unified step-$STEP Regime C — xs checkpoints, restricted, both held-out genomes"
    echo ""
    echo "Same benchmark/geometry as the Evo2 & NTv3-ctx runs (restricted, step $STEP, tol 40 bp, FP-merge 81 bp)."
    echo "CONTROL: cnn_v2 should reproduce the Evo2 report (clj 63.1/48.2, cau 81.2/67.1 @ recall 0.1/0.5)."
    echo ""
    for g in $GENOMES; do
        echo "## $g"
        for m in $MODELS; do
            echo "### $g / $m — restricted"
            cat "$OUT/fp_${g}_${m}_restrict/genome_fp_benchmark.md"
            echo ""
        done
    done
} > "$GRID"

echo ""
echo "=== DONE ===  grid -> $GRID"
echo "--- recall 0.1 & 0.5, restricted, everything this job scanned ---"
grep -HE "0\.(10|50) \(got" "$OUT"/fp_*_restrict/genome_fp_benchmark.md | sort
echo ""
echo "Slot these existing numbers in to complete the SIX-model grid (@ recall 0.1/0.5, restricted, step 10):"
echo "  Evo2      : clj 71.0/58.0   cau 85.8/71.4   (EVO2_GENOME_SCALE_REPORT.md)"
echo "  NTv3-ctx  : clj 35.7/28.0   cau 58.5/48.3   (ntv3 regimeC job 526023)"
