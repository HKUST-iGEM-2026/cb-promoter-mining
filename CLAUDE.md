# CLAUDE.md — Promoter Mining (HKUST iGEM 2026 Dry Lab)

Context for any Claude Code instance helping a teammate work on this repo. Read this top
to bottom before coding. Behavioral rules live in `prompt-guideline/CLAUDE.md` — follow
both.

> **Keep this file current.** Whenever you finish a task or a meaningful change, edit this
> `CLAUDE.md`: flip the status in the §6 task table, record any new decision (§7) or gotcha
> (§9), fix the repo layout (§3) if files moved — then re-read the whole file to catch
> anything you made stale. It's the team's shared source of truth; an out-of-date line here
> misleads the next person's Claude.

---

## 0. How to work here (read first)

Hold to these:
# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.


---

## 1. The project in one page

- **HKUST iGEM 2026 Dry Lab — Promoter Mining.** 9-person team, 7-week sprint
  (1 Jun → 17 Jul 2026), led by **Oak** (dry-lab module lead).
- **Goal:** computationally mine the genome of **_Clostridium butyricum_ DSM 10702** (the
  *target*) for **promoters** (Stage 1 = constitutive). A promoter is a short stretch of DNA
  that switches a gene on; finding them by hand is slow, so we train models to spot them.
- **Approach:** an **ensemble of 3 DNA models** scores every position, then we combine their
  votes (agreement = confidence) and run statistical/experimental checks:
  - **ProkBERT-mini-long** (fine-tuned) — `github.com/nbrg-ppcu/prokbert`
  - **Nucleotide Transformer v2 250M multispecies** (fine-tuned with **LoRA**)
  - **custom 1D CNN** (trained from scratch — the control)
  - combined via **mean rank-percentile**.
- **Pipeline phases:** A data acquisition → B dataset construction → C training →
  D evaluation/calibration → E mining/annotation.
- **Why it's hard (the core biological problem):** *Clostridium* genomes are **AT-rich**
  (GC ~28–32%), and promoter motifs (−10/−35) are *also* AT-rich, so naive models produce
  **false positives**. We counter this with AT-rich cross-species training data (B. subtilis),
  shuffled-sequence negatives (SRS), AT-content-aware calibration, and empirical FDR.

---

## 2. The data, in brief

- **Target (what we mine):** _C. butyricum_ DSM 10702. **Deliberately excluded** from training.
- **Training genomes (T5):** 166 other *Clostridium* (genus, NCBI taxon **1485**) RefSeq
  genomes — the regions where promoters live.
- **Positive promoters come from two places:**
  - **Curated cross-species (T8):** RegulonDB (*E. coli*) + DBTBS (*B. subtilis*). We borrow
    *E. coli* so the model generalises and *B. subtilis* because it's AT-rich like Clostridium.
  - **Empirical TSS (T10):** transcription start sites called from published dRNA-seq with
    TSSpredator.
- **Negatives (T14), four classes:** coding-sequence windows, intergenic non-promoter,
  downstream/3′, and **SRS** = Shuffled Real Sequence (dinucleotide-preserving shuffle of real
  promoters — keeps base composition, destroys the −10/−35 signal). SRS is also the null for
  empirical FDR (T27).

---

## 3. Repo layout

```
cb-promoter-mining/
├── CLAUDE.md                       # this file
├── README.md                       # project blurb + plain-language flowchart
├── prompt-guideline/CLAUDE.md      # behavioral rules (see §0)
├── infrastructure/
│   ├── T3/snakemake_scaffold/      # the pipeline scaffold (see §5)
│   │   ├── Snakefile               #   Phase B–E (HPC4)
│   │   ├── local_prep.smk          #   Phase A: T5–T9 (local)
│   │   ├── tss_prep.smk            #   Phase A: T10/TSSpredator (HPC4)
│   │   ├── config/config.yaml      #   placeholder paths/params
│   │   └── profiles/slurm/config.yaml  # the ONE place for cluster settings
│   └── T4/                         # conda env + deploy runbook
│       ├── environment.yml
│       └── DEPLOY.md
├── data-acquisition-and-curation/
│   ├── T5/                         # 166 Clostridium genomes (zip, Git-LFS) + provenance
│   ├── T7/                         # dRNA-seq/TSS literature search (README + run lists)
│   └── T8/                         # RegulonDB+DBTBS -> Parquet (run_t8_pipeline.py)
├── dataset-construction/           # T11–T16 land here (currently README only)
├── model-training/
│   ├── T18/train_prokbert_template.py  + run_task18_template.sh
│   ├── T19/train_nt_lora_template.py   + run_task19_template.sh
│   ├── T20/train_cnn_template.py
│   └── T21/hyperparameter_sweep_template.py
├── evaluation-and-calibration/
│   ├── T23/cluster_cv_template.py
│   └── T24/evaluate_external_template.py
├── mining-and-motif-annotation/
│   └── T28/sliding_window_engine_template.py
└── pre-tasks/
    ├── PT1/prokbert_finetune.py, regulonDB_treating.py
    └── PT3/metrics_cv.py
```

T6 (Bakta annotations, ~8.9 GB) lives in a **separate** repo/folder `bakta_outputs/`, not here
(only `.gff3` + `.fna` + `.tsv` per genome matter downstream). The `*_template.py` files are
**starter drafts**, not finished code — owners replace them.

---

## 4. Compute — HPC4 = HKUST SuperPOD

"HPC4" in our docs **is** the HKUST SuperPOD. Key facts:

| Thing | Value |
|---|---|
| Login | `ssh <user>@superpod.ust.hk` (off-campus: HKUST VPN first) |
| Slurm account | **`igem2026`** — required on *every* job (`--account=igem2026`) |
| Partitions | `normal`, `preempt`, `cpu` — we use **`preempt`** (5× cheaper, requeue/checkpoint) |
| GPUs | **NVIDIA H800 (Hopper)** → want **CUDA 12.x**. Request with `--gpus-per-node=1` |
| GPU job rule | 1 GPU : 28 threads : 8 GB RAM/thread — **don't** set `--mem`/CPU flags on GPU jobs |
| Storage | `/home` 200 GB/user · **`/project/igem2026`** 5 TB group (persistent — put the env here) · `/scratch/igem2026` 5 TB Lustre (**files inactive >30 days are deleted** — job I/O only) |

**The shared conda env (T4):** `infrastructure/T4/environment.yml`, deployed at
`/project/igem2026/envs/promoter-mining-hpc4`. Build/verify/lock steps are in
`infrastructure/T4/DEPLOY.md`. Everyone `conda activate`s that one prefix.
- Channels: `pytorch, nvidia, conda-forge, bioconda`; PyTorch pinned `>=2.1,<2.5` with
  `pytorch-cuda=12.1` (**confirm against `module avail cuda`** on SuperPOD before building).
- **TSSpredator is a Java JAR, not a conda package** — the env ships `openjdk`; download the
  JAR separately and run `java -jar TSSpredator.jar <config>`.

**Big-data rule:** raw reads (~36 GB), genomes, bakta bulk **never go in git** — commit an
accession list + a download script, or only small processed outputs. Genomes/zip use Git-LFS.

---

## 5. The pipeline (Snakemake scaffold)

It's a **runnable mock**: every rule prints a message and `touch`es a `results/<task>.done`
sentinel. The point is to prove the DAG wires up and submits to SLURM **before** real code
exists. Each rule carries a `# real:` comment showing the actual command to drop in.

**Three tiers — where each task runs:**
1. **Phase A local (T5–T9)** → `local_prep.smk`, run on your machine: `snakemake -s local_prep.smk --cores 4`
2. **Phase A on HPC4 (T10/TSSpredator)** → `tss_prep.smk`, CPU job:
   `snakemake -s tss_prep.smk --workflow-profile profiles/slurm`
3. **Phase B–E (T11–T32)** → main `Snakefile` on HPC4:
   `snakemake --workflow-profile profiles/slurm` (locally `snakemake --cores 4` to test wiring)

Phase A outputs are uploaded to HPC4 as pre-staged `inputs/` files (`annotations.gff3`,
`reference_promoters.parquet`, `literature_promoters.tsv`, `tss_sites.gff3`); the main
Snakefile consumes them and never rebuilds them.

**Cluster settings live in ONE file:** `profiles/slurm/config.yaml` (account `igem2026`,
partition `preempt`, preempt hygiene `--requeue --signal=B:USR1@120`, no default `mem_mb`).
GPU rules add `slurm_extra=GPU_EXTRA` (`--gpus-per-node=1 …`). 9 rules run on the
cluster (PT1, PT5, T18–T21, T23, T24, T28); the other 19 are in `localrules:` and run on the
login node.

**To implement your task:** find your rule in the `Snakefile`, replace the placeholder
`shell:` with the `# real:` command, and for rules that pass real data give the rule a real
`output` filename and point the next rule's `input` at it (the mock wires by `.done` sentinels).
PT1–PP6 are kept as mock rules for the DAG even though conceptually they're imported code.

---

## 6. Tasks (owner · depends-on · status · where)

Status: ✅ done/verified · 🟡 in progress / has known issue · ⬜ not started ·
📄 starter template ready. Owners are from the task brief; the handoff notes an "equal-6"
rebalance, so confirm current owners with Oak.

**Infrastructure**
- T1 HPC4 access — Oak — ✅
- T2 GitHub repo — Oak — ✅
- T3 Snakemake scaffold + SLURM — Swasti — ✅ (`infrastructure/T3/snakemake_scaffold/`)
- T4 Conda env + deploy — Swasti — 🟡 spec ✅, deploy on HPC4 pending (`infrastructure/T4/`)

**Pre-tasks (learning + reusable code)**
- PT1 Fine-tune loop on RegulonDB/DBTBS — Sine — 🟡 (check it runs on **GPU** not CPU)
- PT2 Ship ProkBERT to HuggingFace — Sine — ⬜ (dep PT1)
- PT3 Metrics + CV stack — Seonyu — ✅ `pre-tasks/PT3/metrics_cv.py` (reused by T22/T23)
- PT4 Calibration + FDR harness — Seonyu — ⬜ (dep PT3; reused by T26/T27)
- PT5 Sliding-window inference on E. coli — Ishi — ⬜ (can run locally; reused by T28)
- PT6 Validate pipeline + lock output schema — Swasti — ⬜ (dep PT5; locks T32 schema)

**Phase A — data acquisition & curation**
- T5 Download Clostridium genomes — Salimah — ✅ 166 RefSeq genomes (taxon 1485)
- T6 Re-annotate with Bakta — Leo — ✅ 166/166, Bakta 1.11.4 + DB v6.0 **light** (in `bakta_outputs/`)
- T7 dRNA-seq literature search — Salimah — ✅ (see §8; 20 dRNA-seq vs 27 published TSS)
- T8 Normalise RegulonDB+DBTBS → Parquet — Leo — 🟡 **has a real bug** (see §9) `data-acquisition-and-curation/T8/`
- T9 Curate synthetic promoters → `literature_promoters.tsv` — Salimah — ⬜ (used by T24/T26)
- T10 Re-process dRNA-seq w/ TSSpredator — Sine — ⬜ runs on HPC4 (`tss_prep.smk`; dep T6,T7)

**Phase B — dataset construction** (land in `dataset-construction/`)
- T11 Housekeeping gene set — Sine — ⬜ (dep T6; **select by product, not gene symbol** — §9)
- T12 Operon detection (+ contig-end exclusion) — Seonyu — ⬜ (dep T6,T11)
- T13 Upstream-region extraction (100–300 bp, + contig-end exclusion) — Seonyu — ⬜ (dep T11,T12; reused by T28)
- T14 Four-class negatives incl. SRS — Seonyu — ⬜ (dep T6,T13)
- T15 CD-HIT clustering + cluster-aware 80/10/10 splits — Sine — ⬜ (dep T8,T10,T13,T14)
- T16 Per-model tokenisation (ProkBERT/NT/CNN) — Leo — ⬜ (dep T15)

**Phase C — model training**
- T17 Finalise fine-tuning recipe — Swasti+Oak — ⬜ (dep PT2)
- T18 ProkBERT-mini-long fine-tune — Sanaa+Oak — 📄 `model-training/T18/` (dep T4,T16,T17)
- T19 NT v2 250M LoRA fine-tune — Sanaa — 📄 `model-training/T19/` (dep T4,T16,T17)
- T20 Custom 1D CNN from scratch — Yany — 📄 `model-training/T20/` (dep T4,T16,T17)
- T21 Hyperparameter sweep — Sanaa+Oak — 📄 `model-training/T21/` (dep T18,T19,T20)

**Phase D — evaluation & calibration**
- T22 Metrics suite (AUROC/AUPRC/MCC/F1) — Yany — ⬜ (dep PT3,T18,T19,T20)
- T23 Cluster-level 5-fold CV — Yany — 📄 `evaluation-and-calibration/T23/` (dep T15,T18-20)
- T24 Cross-species external holdout — Yany — 📄 `evaluation-and-calibration/T24/` (dep T9,T18-20)
- T25 Sigma-factor-stratified eval — Seonyu+Oak — ⬜ (dep T10,T22)
- T26 Calibration regressor (AT covariate) — Ishi — ⬜ (dep PT4,T9,T22)
- T27 Empirical FDR via SRS null → q-values — Ishi — ⬜ (dep PT4,T14,T26)

**Phase E — mining & motif annotation**
- T28 Sliding-window inference on Clostridium — Swasti — 📄 `mining-and-motif-annotation/T28/` (dep PT6,T13,T20,T21)
- T29 Ensemble aggregation (mean rank-percentile) — Swasti/Salimah — ⬜ (dep T28)
- T30 Locus-level peak calling — Ishi — ⬜ (dep T29)
- T31 Motif annotation FIMO/MEME (−10/−35 + sigma) — Leo — ⬜ (dep T30; FIMO alone if short on time)
- T32 FINAL candidate output (TSV/JSONL, PT6 schema) — Ishi+Oak — ⬜ (dep T26,T27,T30,T31)

---

## 7. Conventions & decisions locked

- **Phase A is prep, uploaded as `inputs/`.** The HPC4 pipeline starts at Phase B. T1–T4
  (infra) and PT1–PT6 (learning) are not real data-pipeline steps.
- **SLURM hygiene in one place** (the profile): `--account=igem2026`, `--partition=preempt`,
  `--requeue --signal=B:USR1@120`, `--gpus-per-node=1` for GPU; **never `--exclusive`**
  (8× cost); **no `--mem` on GPU jobs**. The SIGUSR1 checkpoint trap goes in each training script.
- **AUPRC must use `sklearn.metrics.average_precision_score`**, never `auc(recall, precision)`
  (the latter inflates the no-skill baseline from prevalence toward 0.5). Reused in T21/T22/T23.
- **Cluster-aware CV** (T23): `StratifiedGroupKFold` on CD-HIT cluster IDs, *not* plain
  StratifiedKFold — otherwise near-identical sequences leak across train/test and inflate scores.
- **Splits ~50/50 positive/negative**, cluster-disjoint (T15).
- **Contig-end exclusion** (T12/T13): the target assembly isn't a complete genome, so drop
  loci whose upstream region is truncated by a contig boundary.
- **TSSpredator only fits the 20 dRNA-seq runs** (paired TEX±). The 27 Capp-Switch / 5′-RNA-seq
  runs use the papers' **published TSS tables** — do NOT push them through TSSpredator (§8).

---

## 8. Key technical reference

**Reference genome assemblies** (confirm version against each paper's Methods before mapping):
- *C. difficile* 630 → `GCF_000009205.2` (NC_009089.1) · 630Δerm → use 630
- *C. acetobutylicum* ATCC 824 → `GCF_000008765.1` (NC_003030.1 + pSOL1)
- *C. beijerinckii* NCIMB 8052 → `GCF_000016965.1` · DSM 6423 → `GCF_900010805.1`
- *C. ljungdahlii* ATCC 55383 (=DSM 13528) → `GCF_000143685.1` (NC_014328.1)

**T7 dRNA-seq/TSS data split** (see `data-acquisition-and-curation/T7/README.md`):
- 47 SRA runs, **3 assay types**. Only **20 dRNA-seq** runs (Soutourina 2020 TAP-based ×2 +
  Fuchs 2021 TEX ×18, all *Clostridioides difficile*) go through **TSSpredator** — use
  `SRR_dRNAseq_for_TSSpredator.txt`. The **27** Capp-Switch (Hocq 2022) + 5′-RNA-seq
  (Al-Bassam 2018) runs → use the papers' published TSS tables (in-genus species; preferred).
- For the Soutourina pair, the enriched library is **TAP+ = `SRR11573852`**, control =
  `SRR11573851`.

**T6 annotation:** Bakta 1.11.4, DB v6.0 **light**. ~27% of CDSs are hypothetical and gene
*symbols* are sparse, but **products** are present for the housekeeping genes — so T11 should
select by **product description**, not the gene-symbol field.

---

## 9. Gotchas / mistakes already found (don't repeat)

- **Taxonomy trap:** *C. difficile* was reclassified to ***Clostridioides*** difficile in 2016 —
  it is **NOT** in genus *Clostridium* (taxon 1485) and **not** in T5's 166 genomes. The
  in-genus species (*C. acetobutylicum/beijerinckii/ljungdahlii*) **are**. Map each dataset to
  its *own* genome.
- **`tsspredator` is not a conda package** (bioconda 404). It's a Java JAR; the env provides
  `openjdk`, download the JAR separately.
- **`pytorch-cuda`** only has versions **11.8 / 12.1 / 12.4** and lives on the **`pytorch`**
  channel (not conda-forge). H800 wants 12.x → we pin `12.1`.
- **T8 bug (open):** `data-acquisition-and-curation/T8/run_t8_pipeline.py` reads the wrong
  RegulonDB columns (`sigma`=strand, `seq`=pmids) so it silently falls back to a 2-row
  placeholder instead of ~4,055 promoters. Columns are: (3) strand, (5) sigmaFactor,
  (6) pmSequence. Fix before relying on its parquet.
- **PT1 (open):** fine-tuning reportedly took ~6 h on ~9.5k examples → almost certainly ran on
  **CPU**. Confirm `torch.cuda.is_available()`, `nvidia-smi`, and `--gpus-per-node=1`.
- **No secrets in code.** A HuggingFace token was once hard-coded in `prokbert_finetune.py`
  (`neuralbioinfo/prokbert-mini` is public — no token needed). Never commit tokens/keys.

---

## 10. Status snapshot

**Solid:** T1, T2, T3 scaffold, T4 env spec, T5, T6, T7, PT3.
**Needs attention:** T4 deploy on HPC4 · T8 column bug · PT1 GPU check · most of Phase B–E
(starter templates exist for T18–T21, T23, T24, T28; everything else is unimplemented).
**When in doubt:** confirm owners/decisions with Oak; verify tool usage and accessions against
the literature; keep changes surgical and within your task.
