# T28 restricted scan + T29 rank-percentile ensemble — run guide

> **Cluster setup (login, VPN, env, SLURM):** `infrastructure/HPC4_GUIDE.md`.
> **How each model loads / where the checkpoints are:** `LOADING_MODELS.md`.

Scan one genome with the four **deployed pre-T26 xs models** and combine them into one ensemble
track by **mean rank-percentile** (T29). Two files do the work, plus the two evaluation scripts they
reuse unchanged:

```
mining-and-motif-annotation/
├── T28/scan_ensemble.sbatch                 # 1) scan 4 xs models  2) call T29  3) benchmark (if truth)
└── T29/ensemble_rank_percentile.py          # the aggregator (mean rank-percentile)
evaluation-and-calibration/
├── genome_inference.py                      # the per-model scanner (reused as-is)
├── genome_fp_benchmark.py                   # restricted benchmark (reused as-is, --regions)
└── make_candidate_regions.py                # builds the restrict-scan search space
```

You only ever **edit the block at the top of `scan_ensemble.sbatch`** and run it. Nothing else changes.

---

## What "restricted" and "pre-T26" mean here

- **Restricted scan** = the repo's *restrict-scan* pattern: each model scans the **whole** genome, and
  the benchmark counts only windows inside candidate promoter regions (`--regions`). That keeps the
  rank-percentiles **genome-wide** (a window at the 99.9th percentile is 99.9th vs the *whole* genome,
  which is the honest read) while the false-positive denominator is restricted to where promoters
  actually live. Same approach as `C_cau_scan.sbatch`.
- **Pre-T26 / uncalibrated** = raw softmax scores, no T26 calibrator. This is **correct** for T29:
  mean rank-percentile depends only on each model's within-genome rank order, so it is
  **calibration-invariant** — the optimistic CNN-v2 and the other three combine on equal footing with
  no calibration step. (Calibration only matters for a fixed 0.5 threshold, which is T30/T32, not T29.)
- **Primary set = xs** (`T21_xs/*`): the cross-species-augmented models, which won recall-0.1
  genome-scale on both held-out validation genomes (ljungdahlii + autoethanogenum). The paths are
  pre-filled in the sbatch `MPATH` table.

---

## Two modes

**A. Validation genome (has known TSS) — recommended first run.** Point `TRUTH` at a TSS table and the
job benchmarks the ensemble *and each member* restricted, at recall 0.1/0.5/0.7/0.9. Use this to
confirm the ensemble is doing what you want (ideally: fewer FP peaks at recall 0.1 than cnn_v2-xs
alone) before you trust it on the target. Held-out genomes with truth already prepped: **C. ljungdahlii**
(`evaluation-and-calibration/ljungdahlii_holdout_tss.tsv`, 1,245 TSS) and **C. autoethanogenum**
(`data-acquisition-and-curation/TSS_results/autoethanogenum_tss.tsv`, 2,885 TSS — prep via
`prep_cau_regimeC.sh`).

**B. Target mine (no truth).** Leave `TRUTH=""`. The job scans + ensembles and writes the
**restricted, score-sorted shortlist** (`<tag>_shortlist_restricted.csv`) — the ranked candidate list
for *C. butyricum*. No benchmark (there are no known TSS to score against). This is the real T28→T29
deliverable; peak-calling/motif annotation is T30/T31.

> Do a validation run (A) first. It's the cheap way to catch a wrong path / contig-id mismatch before
> the long target scan.

---

## Step 1 — stage the genome (login node, has internet)

You need a **FASTA**, a **candidate-regions BED**, and (mode A only) a **truth TSV**. The
autoethanogenum prep script is a working template — copy it and change the accession:

```bash
source /project/igem2026/cb-promoter-mining/env.sh
conda activate /project/igem2026/envs/promoter-mining-hpc4      # has ncbi-datasets-cli

# fetch genome + gff, then build candidate regions (300 bp upstream / 50 bp into gene, per strand):
datasets download genome accession <ACC> --include genome,gff3 --filename g.zip
unzip -o g.zip -d g_dl
cp "$(find g_dl -name '*.fna'  | head -1)" genome.fna
cp "$(find g_dl -name '*.gff*' | head -1)" genome.gff3
REC=$(grep '^>' genome.fna | head -1 | sed 's/^>//; s/ .*//')     # the FASTA record id
python evaluation-and-calibration/make_candidate_regions.py \
    --gff genome.gff3 --contig "$REC" --up 300 --down 50 --out candidate_regions.bed
```

**Contig-id parity is the #1 failure.** The score track's `contig`, the regions' `contig`, and the
truth's `contig` must be the SAME string. If Bakta renamed the sole contig (`contig_1`) but your regions/
truth use the RefSeq id (`NC_...`), set `CONTIG_OVERRIDE="--contig-name NC_..."` in the sbatch so the
scan emits the RefSeq id. `make_candidate_regions.py --contig <REC>` already stamps the right id above.

Mode A truth (`contig  pos  strand`, tab-separated). For the two prepped genomes the TSS files are
`pos,strand` only — stamp the contig in (see `prep_cau_regimeC.sh` step 2 for the one-liner), or point
`TRUTH` at an already-stamped `*_truth.tsv`.

## Step 2 — edit the top block of `scan_ensemble.sbatch`

```
TAG=clj                       # short name for output files
FASTA=$REPO/.../genome.fna
REGIONS=$REPO/.../candidate_regions.bed
TRUTH=$REPO/.../clj_truth.tsv # a path = validation (benchmark); ""  = target mine (shortlist only)
STEP=1                        # 3-5 speeds the target mine; keep 1 for validation
CONTIG_OVERRIDE=""            # or "--contig-name NC_014328.1"
```

Confirm the `MPATH` table points at the `T21_xs` checkpoints (pre-filled). If `T21_xs` was moved,
`grep -m1 model-path evaluation-and-calibration/T26/C_ljung_scan_xs.sh` shows the paths the xs
ljungdahlii scan used.

## Step 3 — submit

```bash
sbatch mining-and-motif-annotation/T28/scan_ensemble.sbatch
squeue --me
tail -f t28_scan_t29_ens_*.log
```

Outputs land in `.../results/regimeC/mine_<TAG>/`:

| file | what |
|---|---|
| `<tag>_{prokbert,nt,cnn,cnn_v2}_xs.csv` | per-model score tracks |
| `<tag>_ensemble_xs.csv` | **ensemble track** (genome order) — `promoter_score` = mean rank-percentile |
| `<tag>_shortlist_restricted.csv` | ensemble RESTRICTED to candidate regions, score-sorted (the mine output) |
| `fp_<tag>_ensemble_restrict/…md` | mode A only: restricted FP/precision at each recall, ensemble + members |

---

## Gotchas

- **Identical grid.** All four scans MUST use the same `--window`/`--step`/`--strands` — the sbatch
  enforces one `STEP`. T29 aligns tracks by `(contig,start,strand)` and **fails loud** if they don't
  match (that error means one model was scanned differently — re-scan it).
- **NT env.** NT-LoRA loads ONLY in `nt-tf4x`; the sbatch switches env per model automatically. Don't
  merge NT into the main-env loop.
- **Cost.** cnn_v2 is the slow scan (per-window Python featurisation). On a ~4 Mb genome at `STEP=1`
  the four scans fit the 24 h wall comfortably (cau did 8 tracks); for the target, `STEP=3` roughly
  thirds the cnn_v2 time at a small recall cost. Resume-safe on preempt.
- **`git` while an env is active** breaks over SSH — run git as `LD_LIBRARY_PATH= git …`. The shared
  `/project` clone is **pull-only**; commit + push from Oak's Mac.
- **This mines the target only — never trains on it.** T28/T29 are read-only inference; `C. butyricum`
  stays out of every training set.

## Commit (from Oak's Mac, surgical)

```bash
git add mining-and-motif-annotation/T28/scan_ensemble.sbatch \
        mining-and-motif-annotation/T29/ensemble_rank_percentile.py \
        mining-and-motif-annotation/T28_T29_RUN_CLAUDE.md
git commit -m "T28/T29: restricted scan + mean-rank-percentile ensemble (xs models)"
git push
```
Do **not** commit `results/` (tracks/shortlists are regenerated on the shared clone).
