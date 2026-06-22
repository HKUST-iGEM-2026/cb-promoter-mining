# HPC4 Runbook — what to actually run (and where it stops)

Companion to `DEPLOY.md` (which covers building the conda envs). This is the **after the env is
built** runbook: pull data, stage inputs, run what's live, and understand the wall.

> **Honest state of the pipeline (verified 2026-06, AI-council, on-disk).**
> The pipeline does **NOT** run end-to-end yet. Today on HPC4 you can stage the Phase A inputs,
> run **Phase B data construction (T11–T14) for real**, and run the live pre-tasks (PT3/PT4/PT5).
> **Everything from T15 onward is blocked** because `dataset-construction/T15/cdhit_splits.py` and
> `T16/tokenise.py` **do not exist yet** — so **no model training can start**, regardless of the GPU
> env. A plain `snakemake --workflow-profile profiles/slurm` will run Phase B + the live pre-tasks,
> then **fail at `t15_cdhit_splits`**. That failure is expected, not a misconfiguration.

---

## TL;DR — the sequence

0. **(From your Mac)** commit + push the latest, so HPC4 can pull it.
1. **Verify the env is the real cu128 GPU build** — don't assume.
2. **`git pull && git lfs pull`** — materialise the LFS zips/CSV (without this, Phase B can't read Bakta).
3. **Pre-stage HuggingFace weights** on the login node (compute nodes are offline).
4. **Stage Phase A `inputs/`** with `local_prep.smk` (they're 0-byte placeholders in git by design).
5. **Run what's live:** Phase B (T11–T14) + PT3/PT4 on the login node; PT5 as a GPU job.
6. **The wall:** write `T15` then `T16`. Until then the cluster can't train anything — this is the
   single highest-priority task and it's **code to write, not a job to submit**.
7. **(Optional, independent track)** T10 / TSSpredator.

All commands below run from `infrastructure/T3/snakemake_scaffold/` unless noted.

---

## 0. From your Mac (before HPC4)

Commits/pushes happen from your machine, not the cluster. Make sure `main` has the latest —
including the T8 4,336 matrix (commit `82d9bbb`) and the T7b graceful-skip fix to
`data-acquisition-and-curation/T7/build_published_tss.sh`:

```bash
git add data-acquisition-and-curation/T7/build_published_tss.sh
git commit -m "fix(T7b): skip Git-LFS pointers / missing inputs instead of crashing local_prep"
git push
```

## 1. Verify the env is actually the GPU build (don't trust that it "installed")

The committed `environment.lock.yml` is **invalid** (mixes CUDA-12.1 conda with CUDA-13 wheels →
silent CPU fallback). Confirm your env was built from `environment.yml` and that torch is the
`cu128` build, **inside a GPU job** (the login node has no GPU):

```bash
conda activate /project/igem2026/envs/promoter-mining-hpc4
srun --account=igem2026 --partition=preempt --gpus-per-node=1 --nodes=1 --time=00:05:00 \
  python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# EXPECT:  True 12.8     (if False or 12.x≠12.8 → rebuild from environment.yml, NOT the lock)
```

Also confirm Snakemake 8 + the SLURM executor plugin are present (the profile needs them):

```bash
snakemake --version                                   # expect >= 8.0
python -c "import snakemake_executor_plugin_slurm; print('slurm plugin OK')"
```

> **Gotcha:** the profile uses Snakemake-8 syntax (`executor: slurm`). On a stock Snakemake 7 the
> profile command dies with `error: unrecognized arguments: --executor=slurm`. You **must** have the
> deployed env active.

## 2. Pull the repo + LFS data

```bash
cd /project/igem2026/cb-promoter-mining
git pull
git lfs pull          # turns the ~130-byte pointers into the real binaries
# sanity: these must be real sizes, not ~130 bytes:
ls -lh data-acquisition-and-curation/T6/all_bakta_output.zip   # ~456 MB
ls -lh data-acquisition-and-curation/T5/Clostridium_genomes_v2_166.zip  # ~196 MB
```

Without `git lfs pull`, `config[bakta_dir]` resolves to a pointer and `extract_bakta` fails. (The
T7b fix from step 0 now degrades gracefully on un-pulled pointers, but Phase B still needs the real
Bakta zip — so `git lfs pull` is mandatory.)

## 3. Pre-stage HuggingFace weights (login node — it has internet; compute nodes don't)

```bash
export HF_HOME=/project/igem2026/hf_cache
huggingface-cli download neuralbioinfo/prokbert-mini-promoter                 # PT5 (sliding-window)
huggingface-cli download neuralbioinfo/prokbert-mini                          # PT1
huggingface-cli download InstaDeepAI/nucleotide-transformer-v2-250m-multi-species   # T19 (when training)
```

Then every GPU job preamble sets `export HF_HOME=/project/igem2026/hf_cache` and
`export HF_HUB_OFFLINE=1` (fail-fast instead of hanging on a cache miss).

> **Flag — reconcile model IDs before you cache for training.** The training templates and the
> docs disagree: `T18` currently hard-codes `biomedia-ai/prokbert-mini-long` (**wrong org** — should
> be `neuralbioinfo/prokbert-mini-long`). Fix the script's ID and cache the correct one when you get
> to T18; otherwise the offline job will miss the cache and fail.

## 4. Stage the Phase A `inputs/` (they're 0-byte placeholders in git, by design)

The main Snakefile never builds Phase A — it consumes pre-staged files in `inputs/`. Generate them
on the login node with `local_prep.smk` (dry-run first to see what it'll do):

```bash
cd infrastructure/T3/snakemake_scaffold
snakemake -s local_prep.smk -n                       # wiring check
snakemake -s local_prep.smk --cores 4                # produces the inputs/ files
```

This populates `inputs/`: `reference_promoters.parquet` (T8, ~3,963 RegulonDB + 373 DBTBS),
`literature_promoters.tsv` (T9), `literature_promoters_train.parquet` (T9-split),
`published_tss_promoters.parquet` (T7b), and the SRR list.

- **T6 (Bakta) regeneration is opt-in** — the committed `all_bakta_output.zip` is canonical; you do
  **not** need a Bakta DB. If a `local_prep` target tries to re-run Bakta, skip it.
- **T7b in-genus positives are optional.** Without the journal supplement tables (see
  `data-acquisition-and-curation/T7/PUBLISHED_TSS_README.md`) it writes an **empty but schema-valid**
  parquet and the pipeline still runs — you just get 0 verified in-genus positives (the §7 open
  decision). To get them, hand-download the Hocq/Al-Bassam tables into `T7/published_tss/` first.
- **`tss_sites.gff3` (T10)** is produced by the separate TSSpredator track (step 7), not here. It
  isn't needed until T15 exists, so you can leave the placeholder for now.

## 5. Run what's LIVE today

Phase B and the CPU pre-tasks are `localrules` — they run on the login node, no SLURM needed:

```bash
# Phase B end-to-end (extract_bakta → collect → T11 → T12 → T13 → T14). Minutes.
snakemake --cores 4 results/T14/negative_set.tsv
#   Expect ~22,760 housekeeping genes → 14,659 operon leaders → 14,553 positive upstream windows
#   → ~52,933 four-class negatives (SRS / CDS / Downstream / Intergenic).

# Live pre-tasks (CPU): metrics CV + calibration/FDR harness
snakemake --cores 4 results/PT3.done results/PT4.done
```

GPU pre-task (PT5, E. coli sliding-window inference — reused by T28) submits via the profile:

```bash
snakemake --workflow-profile profiles/slurm results/PT5.done
```

> If you run the **default** target (`snakemake --workflow-profile profiles/slurm` with no target),
> it will complete Phase B + the live pre-tasks and then **stop at `t15_cdhit_splits`** with
> `can't open file '.../T15/cdhit_splits.py'`. That is the expected wall, not a bug.

## 6. The wall — what must be written next (this is the real blocker)

The cluster cannot do anything downstream until these exist:

| Order | Task | File to write | Blocks |
|------:|------|---------------|--------|
| 1 | **T15** CD-HIT clustering + cluster-aware 80/10/10 split | `dataset-construction/T15/cdhit_splits.py` | **everything** (keystone — all leaves depend on it) |
| 2 | **T16** per-model tokenisation | `dataset-construction/T16/tokenise.py` | all training (T18/19/20) |
| 3 | fix training templates | `model-training/T18,T19,T20` | T18 wrong HF org · T19 no `__main__` guard (runs nothing) · T20 reads a hard-coded path T16 was meant to make |

Phase B output (from step 5) feeds **straight into T15**, so T15 is genuinely unblocked to write
today. **Before wiring T15, resolve the §7 open decision** the Snakefile flags in the T15 rule
comment: as currently wired the training split has *no verified in-genus Clostridium positive*
(T8 is cross-species; T10 is out-of-genus C. difficile; T13 upstreams are inferred). The fix is to
feed T9-split's in-genus positives (`literature_promoters_train.parquet`, already produced in step 4)
and/or the published in-genus TSS into T15.

Also still unwritten (after T15/T16): PT2, T22, T25, T26, T27, T29, T30, T31, T32. Templates with
known defects: T21, T23, T24, T28.

## 7. (Optional, independent) T10 / TSSpredator

Not on the critical path to a first run, and its data is *Clostridioides difficile* — **out of genus
1485** (cross-species-adjacent, not in-genus positives). Run it only if you want the empirical TSS
track. It needs the **bio env** (`openjdk`, `sra-tools`, `segemehl`) and manual prep on the login
node:

```bash
conda activate /project/igem2026/envs/promoter-mining-bio
# stages 1–2 (login node, internet): download the TSSpredator JAR + the ~20 dRNA-seq SRA runs
bash data-acquisition-and-curation/T10/run_tss_pipeline.sh 1
bash data-acquisition-and-curation/T10/run_tss_pipeline.sh 2
# then submit stages 3–6 as a CPU job:
snakemake -s tss_prep.smk --workflow-profile profiles/slurm
```

Confirm the `WORK=` dir in the script header and the `.conf` paths match before submitting; the D4
adapter is still flagged "CONFIRM" in the script header.

---

## Quick reference — verified gotchas

- **Activate the deployed env or the profile errors** (`--executor=slurm` needs Snakemake 8 +
  `snakemake-executor-plugin-slurm`; both are pinned in `environment.yml`).
- **`inputs/` are 0-byte stubs in git** — run `local_prep.smk` first (step 4) or every downstream
  rule ingests empty files.
- **`environment.lock.yml` is invalid** — build/verify from `environment.yml` (torch `2.11.0+cu128`,
  `torch.version.cuda=='12.8'`).
- **HF model IDs disagree** between docs and scripts (T18 wrong org) — reconcile before the offline
  cache step for training.
- **`MasterTable.tsv` in `.gitattributes`** is a TSSpredator *output*, not a committed input — there's
  nothing to `git lfs pull` for it; ignore it when staging.
- **`snakemake -n` (dry-run) is green even though scripts are missing** — it validates wiring only and
  executes nothing. Don't read a clean dry-run as "the pipeline runs."
