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
.


## RegulonDB data licensing

RegulonDB's [official License and Terms of Use, Version 1.0, April 2025](https://github.com/regulondbunam/regulondb-docs/blob/main/00_about_policies/license.md)
permits academic/noncommercial **sharing and adaptation under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)**,
with attribution to RegulonDB and original publications, license links and disclosure of changes.
Commercial data use requires a separate agreement. See [`REGULONDB_LICENSE.md`](REGULONDB_LICENSE.md)
for the source citation, retained export provenance and processing changes. This permission is separate
from the repository's source-code and fitted-model licenses and does not relicense other data sources.

## Recovery and checking status

- **Hugging Face artifact recovery: complete (user-reported).** The three deployment repositories under
  [CloBiT](https://huggingface.co/CloBiT) remain private; authenticated access and inventories were verified.
- **Remaining reproducibility checks: ongoing.** Recovery does not establish historical training/inference
  reproduction or completion of the outstanding scientific checks.
