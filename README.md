# Promoter Mining — *Clostridium butyricum* DSM 10702

**HKUST iGEM 2026 Dry Lab.** We computationally mine the genome of *Clostridium butyricum*
DSM 10702 (the *target* — deliberately excluded from training) for **promoters**: short DNA
stretches that switch a gene on. We train 3 genomic language
models to spot them, then combine an ensemble of models' votes into a ranked, statistically
filtered candidate list for experimental follow-up.

Stage 1's scope is **constitutive promoters**.

## Why this is hard

*Clostridium* genomes are **AT-rich** (GC ~28–32%), and the promoter motifs we're looking for
(−10/−35 boxes) are *also* AT-rich. A naive model learns "AT-rich = promoter" and drowns in
false positives. We counter this with AT-rich cross-species training data, shuffled-sequence
negatives and empirical false-discovery-rate control. The case for
why this approach is sound (and why the deliverable is an *enriched candidate list*, not a precise
genome-wide map) is written up in **[`WORKFLOW_RATIONALE.md`](WORKFLOW_RATIONALE.md)**.

## How the project works

```mermaid
flowchart TB
  subgraph IN[" Step 1 · Gather the raw data "]
    direction LR
    A["Clostridium DNA<br/>the genome we want to search"]
    B["Known promoters<br/>from other, well-studied bacteria"]
    C["Lab experiment data<br/>showing where genes switch on"]
  end

  D["Step 2 · Build a training set<br/>turn the raw data into clear examples:<br/>real promoters vs. look-alike decoys"]
  E["Step 3 · Train the AI<br/>teach several models to recognise<br/>what a promoter looks like"]
  F["Step 4 · Check the models work<br/>test their accuracy and make sure<br/>their confidence scores are trustworthy"]
  G["Step 5 · Scan the target genome<br/>slide across the DNA and score<br/>every position for 'is this a promoter?'"]
  H["Step 6 · Final result<br/>a ranked, quality-checked list<br/>of candidate promoters"]

  A --> D
  B --> D
  C --> D
  D --> E --> F --> G --> H

  classDef input fill:#eaf2ff,stroke:#3b6fb5,color:#111;
  classDef goal fill:#d9f2d9,stroke:#27ae60,color:#111;
  class A,B,C input;
  class H goal;
```

### Why an ensemble of models?

Several models score every position along the DNA independently, and we combine their votes by
**mean rank-percentile** — if they agree, we're far more confident a spot is a real promoter. The
project explored **four** trained candidates plus an off-the-shelf foundation model:

| Model | What it is | Role |
|---|---|---|
| **ProkBERT-mini** | prokaryote-pretrained transformer, fully fine-tuned | ensemble member (deployed) |
| **CNN-v2** | custom CNN from scratch, multi-encoding fusion (one-hot + ENAC + k-mer) | ensemble member (deployed) |
| **Evo2-7B** | frozen genomic foundation model + logistic-regression probe | ensemble member (deployed) |

**Why the deployment set differs from the original 3-model design:** balanced-test accuracy
(precision/recall on a held-out curated set) and genome-scale accuracy (false positives per real
promoter when scanning a whole chromosome) turned out to rank models *differently* — NT-LoRA in
particular has strong balanced AUPRC but collapses at genome scale. Section 10 of
**[`CLAUDE.md`](CLAUDE.md)** has the full evidence trail; the model-selection reasoning (why three
inductive biases, why fine-tune differently, why Evo2 counts despite "only" returning embeddings)
is in **[`MODEL_CHOICE_RATIONALE.md`](MODEL_CHOICE_RATIONALE.md)**.

We also run experimental/statistical checks — cluster-aware cross-validation, AT-content-aware
calibration, and an empirical FDR against a convergent-intergenic null — so the final list is
reliable and not just a stack of confident guesses.


## Repo layout

```
cb-promoter-mining/
├── CLAUDE.md                       # full project context, task table, decisions, gotchas
├── WORKFLOW_RATIONALE.md           # why this approach is sound (published best practice)
├── MODEL_CHOICE_RATIONALE.md       # why an ensemble, why these models, how each is validated
├── LOADING_MODELS.md               # how to load & score with each trained model
├── infrastructure/
│   ├── HPC4_GUIDE.md                # start here for the HKUST SuperPOD ("HPC4")
│   ├── T3/snakemake_scaffold/        # the Snakemake pipeline (Phase A–E)
│   └── T4/                           # shared conda env + deploy runbook
├── data-acquisition-and-curation/   # Phase A — genomes, annotations, curated + empirical TSS
├── dataset-construction/            # Phase B — housekeeping genes, operons, windows, negatives, splits
├── model-training/                  # Phase C — ProkBERT, NT-LoRA, CNN, CNN-v2, Evo2, sweeps
├── evaluation-and-calibration/      # Phase D — metrics, CV, cross-species holdout, calibration, FDR
├── mining-and-motif-annotation/     # Phase E — genome scan, ensemble, peak calling, motifs, final output
└── pre-tasks/                       # onboarding code (fine-tuning loop, metrics/CV stack, sliding-window inference)
```

## Getting started

This is an HPC-first project — almost everything runs on the **HKUST SuperPOD ("HPC4")**, not a
laptop. New to the cluster? **[`infrastructure/HPC4_GUIDE.md`](infrastructure/HPC4_GUIDE.md)**
covers login/VPN, activating the env, submitting SLURM jobs, and storage — read that first.

### 1. Clone the repo + pull the large files

Genomes, Bakta annotations, and a few large result files are stored via **Git-LFS**, not in git
proper — a plain clone only gets ~130-byte pointers.

```bash
git clone https://github.com/HKUST-iGEM-2026/cb-promoter-mining.git
cd cb-promoter-mining
git lfs install
git lfs pull
```

On HPC4, clone (or pull) straight into `/project/igem2026/cb-promoter-mining` — that's the shared,
persistent copy everyone works from.

### 2. Get the conda environment

There's already a shared, GPU-verified environment on HPC4 — most people just activate it, they
don't build it:

```bash
source /project/igem2026/cb-promoter-mining/env.sh          # shared offline HF cache, etc.
conda activate /project/igem2026/envs/promoter-mining-hpc4  # the main env (transformers 5.3.0)
```

If you need to build the env from scratch (new cluster, or a from-scratch check), the full
recipe — CUDA-build verification, the conda/pip split, and six env gotchas that cost the team
real hours — is in **[`infrastructure/T4/DEPLOY.md`](infrastructure/T4/DEPLOY.md)**
(`infrastructure/T4/environment.yml` is the spec).


Cluster settings (account `igem2026`, partition `preempt`, GPU flags) live in one place:
`infrastructure/T3/snakemake_scaffold/profiles/slurm/config.yaml` — don't set them per-job.

### 4. Just want promoter scores from an already-trained model?

You don't need to run the whole pipeline for that — the trained checkpoints are already on HPC4
at `results/models_deployed/`. **[`LOADING_MODELS.md`](LOADING_MODELS.md)** shows how to load each
one and get `P(promoter)` for a list of 81 bp sequences in a few lines, reusing
`evaluation-and-calibration/genome_inference.py`'s scorers instead of reimplementing the loading
logic (and its gotchas — window length, per-model env, tokenizer quirks).

## Reference

- **[`CLAUDE.md`](CLAUDE.md)** — full project context, task-by-task status and owners, locked
  conventions, and every gotcha found so far. Read it before picking up any task, and keep it
  current when you finish one.
- **[`WORKFLOW_RATIONALE.md`](WORKFLOW_RATIONALE.md)** — why the approach is sound.
- **[`MODEL_CHOICE_RATIONALE.md`](MODEL_CHOICE_RATIONALE.md)** — why an ensemble, why these models.
- **[`LOADING_MODELS.md`](LOADING_MODELS.md)** — score sequences with a trained model.
- **[`infrastructure/HPC4_GUIDE.md`](infrastructure/HPC4_GUIDE.md)** — cluster basics.
- **[`infrastructure/T4/DEPLOY.md`](infrastructure/T4/DEPLOY.md)** / **[`NT_ENV.md`](infrastructure/T4/NT_ENV.md)** — build/lock the conda envs.
