# Regenerate the evaluation on the 0.6-clustered models

After the CD-HIT/MMseqs2 identity threshold moved 0.9 → 0.6 and all three models were
retrained (`results/T21_final/{prokbert,cnn,nt_lora}`), every published eval number is on the
OLD models. This runbook re-derives all three regimes on the new checkpoints.

- **Regime A** — separability from in-distribution heterogeneous negatives (per-class AUROC).
- **Regime B** — the composition-matched dinucleotide-shuffle (SRS) null; chance = 0.5.
- **Regime C** — genome-scale false positives on the held-out in-genus genome *C. ljungdahlii*.

All commands use **absolute paths through variables you set once in Step 0**, so nothing depends
on which `results/` tree you are standing in.

Two envs (see `infrastructure/T4/NT_ENV.md`): `promoter-mining-hpc4` for ProkBERT + CNN,
`nt-tf4x` for NT. HF must be offline-cached (the retrain already populated `~/hf_cache`).

```bash
# ---- env activation (run once per shell; see infrastructure/T4/DEPLOY.md for the base path) ----
source <conda-base>/etc/profile.d/conda.sh
MAIN=/project/igem2026/envs/promoter-mining-hpc4     # ProkBERT + CNN
NTENV=/project/igem2026/envs/nt-tf4x                 # NT only
# HF offline (avoids the token-permission + LocalEntryNotFound saga)
export HF_HOME=$HOME/hf_cache
export HF_HUB_OFFLINE=1
export HF_TOKEN_PATH=/tmp/hf_notoken_$USER
```

---

## Step 0 — locate inputs and set the variables (do this first)

```bash
REPO=/project/igem2026/cb-promoter-mining
EVAL=$REPO/evaluation-and-calibration

# find the results/ tree that actually holds the fresh (Aug-1) checkpoints + tokenised data
find $REPO -maxdepth 4 -path "*T21_final*/metrics.json" -newermt 2026-08-01 2>/dev/null
find $REPO -maxdepth 4 -type d -name T16 2>/dev/null
```

Set `RES` to the `results/` directory that contains **both** `T21_final/` and `T16/` (they are
siblings — same pipeline tree), then define the rest:

```bash
RES=<the results dir from the find above>        # e.g. $REPO/infrastructure/T3/snakemake_scaffold/results
T15DIR=$RES/T15
T16DIR=$RES/T16
CKPT_PROKBERT=$RES/T21_final/prokbert/best_model
CKPT_NT=$RES/T21_final/nt_lora/best_adapter
CKPT_CNN=$RES/T21_final/cnn/best_model.pt        # NEW cnn — NOT results/T20
mkdir -p $RES/T22 $RES/T25b $RES/regimeC

# verify everything the runbook needs actually exists
ls -d  $CKPT_PROKBERT $CKPT_NT && ls -l $CKPT_CNN
ls -d  $T16DIR/prokbert $T16DIR/nt $T16DIR/cnn
ls -l  $T15DIR/test.parquet
ls -l  $EVAL/ljungdahlii_holdout_tss.tsv         # truth (committed, model-independent)

# C. ljungdahlii RefSeq FASTA (contig must be NC_014328.1 to match the truth) + a GFF for regions
find $REPO /project/igem2026 -name "*.fna" 2>/dev/null | grep -iE "ljung|014328|143685" | head
find $REPO /project/igem2026 -name "*.gff3" 2>/dev/null | grep -iE "ljung|014328|143685" | head
```

Set the last two (from the prior Regime-C run; fetch fallback in the appendix if missing):

```bash
CLJ_FASTA=<C. ljungdahlii RefSeq FASTA, record id NC_014328.1>
CLJ_GFF=<C. ljungdahlii Bakta GFF3>              # only needed to rebuild candidate regions
```

> Everything below assumes these variables are set in the current shell.

---

## Phase 1 — Regimes A + B (test-set breakdown). Fast; small GPU.

Re-score the held-out test set with each new checkpoint, then break the scores down by negative
class. This regenerates the negatives-by-class table and the SRS shuffled-null column.

```bash
# --- ProkBERT + CNN in the main env ---
conda activate $MAIN
python $EVAL/T22/predict_and_score.py --model prokbert \
    --model-path $CKPT_PROKBERT --data-dir $T16DIR/prokbert --out $RES/T22
python $EVAL/T22/predict_and_score.py --model cnn \
    --model-path $CKPT_CNN      --data-dir $T16DIR/cnn      --out $RES/T22

# --- NT in nt-tf4x ---
conda activate $NTENV
python $EVAL/T22/predict_and_score.py --model nt \
    --model-path $CKPT_NT --data-dir $T16DIR/nt --out $RES/T22

# --- breakdown (CPU; main env) ---
conda activate $MAIN
python $EVAL/modular_eval.py --t15-dir $T15DIR --t22-dir $RES/T22 --out $RES/T25b
cat $RES/T25b/modular_eval.md
```

**Produces:** `$RES/T22/{prokbert,nt,cnn}_test_predictions.csv` + `_metrics.json`, and
`$RES/T25b/modular_eval.{md,csv}` (Regime A per-class AUROC + Regime B = the SRS row).

---

## Phase 2 — Regime C (genome-scale FP on *C. ljungdahlii*). Heavy; GPU SLURM.

Slide each model across the held-out genome, then count false-positive peaks against the known
TSS at several recall targets — whole-genome and restricted to candidate promoter regions.

### 2a. Candidate regions (skip if the BED from the prior run still exists)

```bash
conda activate $MAIN
python $EVAL/make_candidate_regions.py \
    --gff $CLJ_GFF --contig NC_014328.1 --up 300 --down 50 \
    --out $RES/regimeC/clj_candidate_regions.bed
```

### 2b. Score tracks — submit as GPU jobs

ProkBERT + CNN (main env):

```bash
sbatch <<'EOF'
#!/bin/bash
#SBATCH --account=igem2026
#SBATCH --partition=preempt
#SBATCH --gpus-per-node=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --job-name=clj_scan_main
#SBATCH --output=clj_scan_main_%j.out
set -e
source <conda-base>/etc/profile.d/conda.sh
conda activate /project/igem2026/envs/promoter-mining-hpc4
export HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1 HF_TOKEN_PATH=/tmp/hf_notoken_$USER
EVAL=/project/igem2026/cb-promoter-mining/evaluation-and-calibration
RES=__RES__                                        # <-- paste your $RES value here
CLJ_FASTA=__CLJ_FASTA__                            # <-- paste your $CLJ_FASTA value here
python $EVAL/genome_inference.py --genome $CLJ_FASTA \
    --model-path $RES/T21_final/prokbert/best_model --model-id neuralbioinfo/prokbert-mini \
    --out $RES/regimeC/clj_prokbert_track.csv --window 81 --step 1 --batch-size 512
python $EVAL/genome_inference.py --genome $CLJ_FASTA \
    --model-path $RES/T21_final/cnn/best_model.pt --model-type cnn \
    --out $RES/regimeC/clj_cnn_track.csv --window 81 --step 1 --batch-size 512
EOF
```

NT (nt-tf4x env):

```bash
sbatch <<'EOF'
#!/bin/bash
#SBATCH --account=igem2026
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1
#SBATCH --job-name=clj_scan_nt
#SBATCH --output=clj_scan_nt_%j.out
set -e
source <conda-base>/etc/profile.d/conda.sh
conda activate /project/igem2026/envs/nt-tf4x
export HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1 HF_TOKEN_PATH=/tmp/hf_notoken_$USER
EVAL=/project/igem2026/cb-promoter-mining/evaluation-and-calibration
RES=__RES__
CLJ_FASTA=__CLJ_FASTA__
python $EVAL/genome_inference.py --genome $CLJ_FASTA \
    --model-path $RES/T21_final/nt_lora/best_adapter \
    --model-id InstaDeepAI/nucleotide-transformer-v2-250m-multi-species \
    --out $RES/regimeC/clj_nt_track.csv --window 81 --step 1 --batch-size 256
EOF
```

### 2c. FP benchmark — full genome + restricted, at recall 0.1/0.5/0.7/0.9

```bash
conda activate $MAIN
for M in prokbert cnn nt; do
  python $EVAL/genome_fp_benchmark.py \
      --scores $RES/regimeC/clj_${M}_track.csv \
      --truth  $EVAL/ljungdahlii_holdout_tss.tsv \
      --recall-targets 0.1,0.5,0.7,0.9 \
      --out    $RES/regimeC/fp_${M}_full
  python $EVAL/genome_fp_benchmark.py \
      --scores  $RES/regimeC/clj_${M}_track.csv \
      --truth   $EVAL/ljungdahlii_holdout_tss.tsv \
      --regions $RES/regimeC/clj_candidate_regions.bed \
      --recall-targets 0.1,0.5,0.7,0.9 \
      --out     $RES/regimeC/fp_${M}_restrict
done
grep -H "" $RES/regimeC/fp_*_restrict/genome_fp_benchmark.md | head -80
```

---

## Phase 3 — update the report (Oak commits from the Mac)

Paste the Phase 1 (`modular_eval.md`) and Phase 2 (`fp_*_{full,restrict}`) outputs back into the
chat. Claude collates them into `species_metrics_report.tex` + `negatives_by_class.csv`, diffs old
vs new (the key question: did 0.6 lower FP peaks at matched recall?), and Oak commits.

---

## Appendix — fetch *C. ljungdahlii* if it is not staged

```bash
conda activate $MAIN                 # has ncbi-datasets-cli
datasets download genome accession GCF_000143685.1 --include genome,gff3
unzip -o ncbi_dataset.zip -d clj_dl
# FASTA record id will be NC_014328.1 (matches the truth); GFF for make_candidate_regions.
```
Bakta is only needed if you want a Bakta GFF instead of the RefSeq GFF — the RefSeq GFF works for
`make_candidate_regions.py` (pass `--contig NC_014328.1`).
