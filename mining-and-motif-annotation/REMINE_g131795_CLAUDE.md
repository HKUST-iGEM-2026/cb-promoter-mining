# Re-mine the target on the COMPLETE assembly — GCF_014131795.1 (DSM 10702) — run-book

> HKUST iGEM 2026 Dry Lab. Runs on **HPC4** end-to-end (GPU scan + login-node post-processing).
> Cluster setup: `infrastructure/HPC4_GUIDE.md`. This reuses the deployed 3-model mine unchanged —
> you only stage a new genome and point the job at it.

## What this is and why

`GCF_014131795.1` (`ASM1413179v1`) is the **same target strain** as the current mine — *Clostridium
butyricum* **DSM 10702** — but a **complete, reference-grade** assembly (Flye 2.4; PacBio + Nanopore +
Illumina MiSeq, 361×; "assembly from type material"; RefSeq **Reference Genome**), with four replicons:

| RefSeq record | molecule | length |
|---|---|---|
| `NZ_CP040626.1` | chromosome 1 | 3,921,278 bp |
| `NZ_CP040627.1` | chromosome 2 | 770,199 bp |
| `NZ_CP040628.1` | plasmid pCB_1 | 6,059 bp |
| `NZ_CP040629.1` | plasmid pCB_2 | 8,060 bp |

The mine so far used **`g409755` = `GCF_000409755.1`**, a 96-contig `NZ_AQQF01000…` **draft WGS**. Same
organism, different (better) assembly. Re-mining here gives the definitive DSM 10702 candidate list on a
complete genome, and — because it's complete, reference-grade, and from type material — it's the natural
**canonical target assembly** to standardise T26/T27/T28 on (the draft-vs-4-contig split is exactly the
`CLAUDE.md` §9 assembly gotcha that stopped the T27 q-values from joining the candidate tiers).

Nothing trains on the target — T28→T32 are read-only inference. The models are genome-agnostic (they
score any 81 bp; the Evo2 probe is reused across genomes), so only the **FASTA + coordinates** must be
self-consistent, which the staging step guarantees.

This is the same machinery as the `g409755` mine and the `MINE_UPSTREAM_3MODEL_FOR_SINE.md` inducible
mine — just a whole-genome restrict-scan of a new assembly.

---

## Step 0 — the deployment ensemble (unchanged)

3 models, combined by mean rank-percentile (T29, calibration-invariant): **ProkBERT** (`T21_xs/prokbert`)
+ **cnn_v2** (`T21_xs/cnn_v2`) + **Evo2** (frozen-embedding logreg probe `evo2_probe_xs.pkl`). Paths are
pre-filled in `mine_3model.sbatch`; you don't touch them.

## Step 1 — stage the complete genome (login node, has internet)

```bash
source /project/igem2026/cb-promoter-mining/env.sh
conda activate /project/igem2026/envs/promoter-mining-hpc4     # has ncbi-datasets-cli
bash mining-and-motif-annotation/T28/prep_g131795.sh
```

This fetches the assembly + GFF, writes `g131795.fna` and `g131795_candidate_regions.bed` into
`…/results/regimeC/mine3_g131795/`, and **fails loud on any contig-id mismatch** (the #1 silent-miss
failure). It's multi-replicon safe — `make_candidate_regions.py` keys on the GFF seqids, which already
equal the RefSeq FASTA headers, so **no `--contig` override** is used (that flag is only for single-contig
Bakta renames). Expect ~4,700 kb / 4 records and a few thousand merged candidate regions across the two
chromosomes (+ a handful on the plasmids).

> Chromosome-only? If you don't want the two tiny plasmids in the mine, drop them from the BED after
> staging: `grep -vE '^(NZ_CP040628|NZ_CP040629)\.1' g131795_candidate_regions.bed > tmp && mv tmp g131795_candidate_regions.bed`.
> Harmless either way — they add only a handful of candidate regions.

## Step 2 — submit the 3-model mine (GPU, target-mine mode)

`TRUTH=""` (empty, **not** unset) is what disables the benchmark — the target has no known TSS. `STEP=10`
is the deployment grid Evo2's `--align-grid` and `align_tracks.py` expect; don't change it. Don't edit the
sbatch — pass everything on the submit line:

```bash
cd /project/igem2026/cb-promoter-mining
OUT=infrastructure/T3/snakemake_scaffold/results/regimeC/mine3_g131795

TAG=g131795 TRUTH="" MODELS="prokbert cnn_v2 evo2" STEP=10 \
  FASTA=$PWD/$OUT/g131795.fna \
  REGIONS=$PWD/$OUT/g131795_candidate_regions.bed \
  sbatch mining-and-motif-annotation/T28/mine_3model.sbatch

squeue --me ; tail -f mineN_*.log
```

The job scans each model (auto-switching to the `evo2` env for the Evo2 member and back to the main env
for T29), reconciles the tracks to a common grid, and writes the ensemble + shortlist. Run **one job per
TAG** at a time (per-model tracks are cached by TAG). Resume-safe on preempt. The `g409755` whole-genome
mine took ~17 min; a complete ~4.7 Mb genome is the same order.

**Output** in `…/results/regimeC/mine3_g131795/`:

| file | what |
|---|---|
| `g131795_{prokbert,cnn_v2,evo2}.csv` | per-model score tracks (cached) |
| `g131795_ensemble_3m.csv` | ensemble track (`promoter_score` = mean rank-percentile) |
| `g131795_shortlist_3m_restricted.csv` | **the mine deliverable** — restricted, score-sorted candidate windows |

## Step 3 — T30 peak calling (login node, pure pandas)

Collapse the per-window shortlist into one call per locus (same thresholds as `g409755`):

```bash
python mining-and-motif-annotation/T30/call_peaks.py \
    --track $OUT/g131795_shortlist_3m_restricted.csv --min-score 0.9 --merge 81 \
    --out $OUT/g131795_peaks.csv
```

Prints `… windows -> … >= 0.9 -> … promoter calls` and how many have ≥5 supporting windows (the confident
set). For reference, `g409755` gave 10,933 windows ≥0.9 → 3,167 calls → **912 with ≥5 windows**.

## Step 4 — T31 motif annotation (−10/−35)

Flow B (the one that produced the `g409755` candidate table). Extract summit sequences (keeps ≥5-window
calls, assigns `candidate_id`), then MEME de-novo + FIMO **in the MEME env**:

```bash
# 4a. extract summit windows (main env: needs pandas)
python mining-and-motif-annotation/T31/extract_call_sequences.py \
    --calls $OUT/g131795_peaks.csv --genome $OUT/g131795.fna \
    --min-windows 5 --top-n 500 \
    --out-fasta $OUT/g131795_call_seqs.fasta \
    --out-top   $OUT/g131795_call_seqs_top.fasta \
    --out-calls $OUT/g131795_calls_with_id.tsv

# 4b. MEME + FIMO (MEME env: has meme/fimo, NOT pandas)
conda activate /project/igem2026/envs/meme
bash mining-and-motif-annotation/T31/annotate_motifs.sh \
    $OUT/g131795_call_seqs_top.fasta $OUT/g131795_call_seqs.fasta $OUT/t31_g131795
conda activate /project/igem2026/envs/promoter-mining-hpc4
```

> Optional architecture summary over **all** calls (fixed σA −10/−35, the `g409755` "51 canonical /
> 1,746 minus10_only / …" breakdown): `python mining-and-motif-annotation/T31/annotate_motifs.py
> --peaks $OUT/g131795_peaks.csv --genome $OUT/g131795.fna --out $OUT/g131795_architecture.tsv`. Not
> needed for the candidate table.

## Step 5 — T32 final candidate table (PT6 schema)

```bash
python mining-and-motif-annotation/T32/build_candidate_table.py \
    --calls $OUT/g131795_calls_with_id.tsv --fimo $OUT/t31_g131795/fimo/fimo.tsv \
    --out $OUT/g131795_candidates.tsv --jsonl $OUT/g131795_candidates.jsonl
```

`g131795_candidates.tsv` — the wet-lab shortlist, one row per confident candidate in the PT6 schema
(`candidate_id, chromosome, strand, tss_position, summit_tss, region_start, region_end, peak_score,
mean_score, n_windows, …, has_promoter_architecture, has_minus10, minus10_offset, …, has_minus35, …`).
`*_known*`/`is_validated` stay blank (target mine, no known TSS). It prints how many have a placed −10
(promoter architecture) and −10 AND −35 (canonical). `g409755`, for comparison: **912 candidates, 306
with a placed −10, 100 canonical**.

---

## Compare to g409755 (sanity, once it's done)

Because it's the same strain, the two candidate lists should broadly agree in *number* and motif
breakdown. Differences to expect: (a) a complete genome recovers upstreams that were **truncated at
contig ends** in the 96-contig draft (a modest bump in candidates near old boundaries); (b) coordinates
are on `NZ_CP0406xx.1`, not `NZ_AQQF01000…`, so the two tables are **not** row-joinable by position — map
by sequence/orthology if you need a crosswalk. To line the biology up with the draft's 912/306/100 tiers,
re-run `split_candidate_tiers.py` on `g131795_candidates.tsv`.

## Gotchas (don't repeat old ones)

- **Contig-id parity** is enforced in `prep_g131795.sh` (step 4) and again in the sbatch preflight. If it
  ever complains, the FASTA header ids and the BED/GFF seqids disagree — re-stage, don't hand-edit scores.
- **`STEP=10` for all models** — the tracks won't reconcile otherwise (Evo2 `--align-grid` +
  `align_tracks.py`).
- **One job per TAG** at a time (shared cached tracks).
- **`git` while an env is active** breaks over SSH — run git as `LD_LIBRARY_PATH= git …`; the shared
  `/project` clone is **pull-only**; commit + push from Oak's Mac. Don't commit `results/` (regenerated).
- **This mines the target only — never trains on it.** DSM 10702 stays out of every training set;
  re-mining a *different assembly of the same strain* is still pure inference.

## One-screen quick reference

```bash
# 1) stage complete genome + candidate regions (login node)
bash mining-and-motif-annotation/T28/prep_g131795.sh
OUT=infrastructure/T3/snakemake_scaffold/results/regimeC/mine3_g131795

# 2) 3-model mine (GPU, target-mine mode)
TAG=g131795 TRUTH="" MODELS="prokbert cnn_v2 evo2" STEP=10 \
  FASTA=$PWD/$OUT/g131795.fna REGIONS=$PWD/$OUT/g131795_candidate_regions.bed \
  sbatch mining-and-motif-annotation/T28/mine_3model.sbatch

# 3) peaks -> 4) motif -> 5) table
python mining-and-motif-annotation/T30/call_peaks.py --track $OUT/g131795_shortlist_3m_restricted.csv \
  --min-score 0.9 --merge 81 --out $OUT/g131795_peaks.csv
python mining-and-motif-annotation/T31/extract_call_sequences.py --calls $OUT/g131795_peaks.csv \
  --genome $OUT/g131795.fna --min-windows 5 --top-n 500 \
  --out-fasta $OUT/g131795_call_seqs.fasta --out-top $OUT/g131795_call_seqs_top.fasta \
  --out-calls $OUT/g131795_calls_with_id.tsv
conda activate /project/igem2026/envs/meme
bash mining-and-motif-annotation/T31/annotate_motifs.sh \
  $OUT/g131795_call_seqs_top.fasta $OUT/g131795_call_seqs.fasta $OUT/t31_g131795
conda activate /project/igem2026/envs/promoter-mining-hpc4
python mining-and-motif-annotation/T32/build_candidate_table.py --calls $OUT/g131795_calls_with_id.tsv \
  --fimo $OUT/t31_g131795/fimo/fimo.tsv --out $OUT/g131795_candidates.tsv --jsonl $OUT/g131795_candidates.jsonl
```
