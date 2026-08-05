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
  - **ProkBERT-mini** (fine-tuned) — `github.com/nbrg-ppcu/prokbert`
  - **Nucleotide Transformer v2 250M multispecies** (fine-tuned with **LoRA**)
  - **custom 1D CNN** (trained from scratch — the control)
  - combined via **mean rank-percentile**.
- **Pipeline phases:** A data acquisition → B dataset construction → C training →
  D evaluation/calibration → E mining/annotation.
- **Why it's hard (the core biological problem):** *Clostridium* genomes are **AT-rich**
  (GC ~28–32%), and promoter motifs (−10/−35) are *also* AT-rich, so naive models produce
  **false positives**. We counter this with AT-rich cross-species training data (B. subtilis),
  shuffled-sequence negatives (SRS), AT-content-aware calibration, and empirical FDR. **Why this
  approach is sound (published best practice), the base-rate ceiling, and the "enriched candidate list,
  not a precise map" framing: see `WORKFLOW_RATIONALE.md`.**

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
│   ├── T18/                            (template removed 2026-06 — owner writes fresh)
│   ├── T19/                            (template removed 2026-06 — owner writes fresh)
│   ├── T20/train_cnn_template.py
│   └── T21/hyperparameter_sweep_template.py
├── evaluation-and-calibration/
│   ├── T23/cluster_cv_template.py
│   └── T24/evaluate_external_template.py
├── mining-and-motif-annotation/
│   └── T28/sliding_window_engine_template.py
└── pre-tasks/
    ├── PT1/prokbert_finetune.py, regulonDB_treating.py
    ├── PT3/metrics_cv.py
    └── PT5/sliding_window_inference.py  # E. coli (MG1655) sliding-window inference; reused by T28
```

T6 (Bakta annotations, ~8.9 GB) lives in a **separate** repo/folder `bakta_outputs/`, not here
(only `.gff3` + `.fna` + `.tsv` per genome matter downstream). The `*_template.py` files are
**starter drafts**, not finished code — owners replace them.

---

## 4. Compute — HPC4 = HKUST SuperPOD

"HPC4" in our docs **is** the HKUST SuperPOD. **New to the cluster? Start with the step-by-step guide
[`infrastructure/HPC4_GUIDE.md`](infrastructure/HPC4_GUIDE.md)** (login/VPN, env activation, SLURM jobs,
storage, gotchas). Key facts:

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

**TWO envs (2026-07):** the main `promoter-mining-hpc4` (transformers **5.3.0**) runs everything EXCEPT the
Nucleotide Transformer (**T19**), which needs a **second** env `nt-tf4x` (transformers **4.57.6**) — NT's
remote code is v4-era and breaks on v5 in ≥6 ways (one *silent*). Recipe + rationale:
**`infrastructure/T4/NT_ENV.md`**. Only T19 (standalone + the T21 NT sweep) uses `nt-tf4x`; T18/T20/all data
tasks stay on the 5.3.0 env.
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
- T3 Snakemake scaffold + SLURM — Swasti — ✅ (`infrastructure/T3/snakemake_scaffold/`). **Rewired 2026-06**: Phase B (`collect_bakta_tsv`, T11–T14) now runs FOR REAL (config-driven `bakta_dir`, real `.tsv` outputs; T11 fed by a flat-`.tsv` staging rule that symlinks the nested Bakta dirs); T15–T32 + pre-tasks stay mock but every `# real:` now names the actual script + exact args (fixed the bogus `scripts/*.py {input} {output}` refs, the PT6 `validate_lock_schema.py`→`validate_promoters.py` name, PT5 `cd`, T18/T19 `.sh` filename bug). Validated by AI council: `snakemake -n` builds the full 30-job DAG; real run produced T11 (22,760 rows/166 genomes) → T12 (14,659 leaders). `config.yaml` now has real paths/params. **Real-shell pass (2026-06)**: `bakta_dir` may be a `.zip` (HPC4: `all_bakta_output.zip`) → new `extract_bakta` rule unzips it (collect symlinks made absolute). Phase A now carries REAL shell too: T5 (unzip genomes), T6 (bakta loop, opt-in regen — needs a bakta env + DB), T7/T8/T9 (cp/run+cp; T8 re-run gives the real 3,963 RegulonDB rows), T7b still a stub (its published-TSS tables aren't staged). T10 mock→ real 6-stage `data-acquisition-and-curation/T10/run_tss_pipeline.sh` (datasets→prefetch/fasterq-dump→segemehl→bedtools 5′-wiggle→TSSpredator×2→merge; 8 open decisions D1–D8 in its header). Council-validated: 31-job DAG builds; T5/T8/T9 + Phase-B-from-zip run for real; all T10 stage CLIs audited correct. **Main Snakefile now carries REAL commands in EVERY rule** (echo/touch mocks removed; `results/<task>.done` sentinels kept so the DAG order holds, `&& touch {output}` appended). LIVE today: T11–T14 + PT3/PT4/PT5/PT6; TEMPLATE-with-defects: PT1/T18–T21/T23/T24/T28; SCRIPT-not-yet-written (rule fails until built): PT2/T15/T16/T22/T25–27/T29–32. Validated: dry-run 31 jobs + real PT3 (wrote pt3_cv_results.csv) + real T12 (14,659 leaders). NB `snakemake --cores 4` now executes real commands and stops at the first unwritten script; `snakemake -n` is the wiring test.
- T4 Conda env + deploy — Swasti — ✅ **deployed + GPU-verified on HPC4 (2026-06)**: built from
  `environment.yml`; a GPU job confirms `torch.cuda.is_available()`=True + `torch.version.cuda`=12.8
  (clean single **cu128** stack — the old 12.1-conda / cu13-pip frankenstein was rebuilt out).
  `environment.lock.yml` regenerated (re-add the two pip sources `conda env export` drops: the cu128
  `--extra-index-url` and `prokbert @ git+…`). The 6 env gotchas that cost us hours are in DEPLOY.md:
  `source <base>/etc/profile.d/conda.sh` **before** activate; `conda env create` (no `mamba` on base);
  `chmod -R g+rX … 2>/dev/null`; `PYTHONNOUSERSITE=1` + `MKL_THREADING_LAYER=GNU` baked into the env's
  `activate.d`; the lock pip-sources. Lock + env edits commit from the Mac. (`infrastructure/T4/`)

**Pre-tasks (learning + reusable code)**
- PT1 Fine-tune loop on RegulonDB/DBTBS — Sine — 🟡 (check it runs on **GPU** not CPU)
- PT2 Ship ProkBERT to HuggingFace — Sine — ⬜ (dep PT1)
- PT3 Metrics + CV stack — Seonyu — ✅ `pre-tasks/PT3/metrics_cv.py` (reused by T22/T23)
- PT4 Calibration + FDR harness — Seonyu — 🟡 implemented `pre-tasks/PT4/calibration_fdr.py` (Snakefile
  marks it LIVE; not independently re-verified this session). (dep PT3; reused by T26/T27)
- PT5 Sliding-window inference on E. coli — Ishi — ✅ code fixed (batch-scoring); CSV regenerated → `mg1655_promoter_scores.csv` (242 MB, Git-LFS via the `pre-tasks/PT5/*.csv` rule; script now writes that exact name). Reused by T28. **2026-06:** re-run on HPC4 (SLURM GPU). Model = `neuralbioinfo/prokbert-mini-promoter` — the promoter **classifier**, NOT base `prokbert-mini` and NOT `…-long-promoter` (DEPLOY §7 mis-states "-long-promoter"; cache the ID the code actually names). Input `MG1655.fna` (E. coli K-12, `GCF_000005845.2`/`U00096.3`) is **not committed** — stage it in `pre-tasks/PT5/` first or it `FileNotFoundError`s; script is cwd-relative (run from that dir). NB the HPC4 `sliding_window_inference.py` is currently modified/uncommitted — review the diff before committing.
- PT6 Validate pipeline + lock output schema — Swasti — 🟡 implemented `pre-tasks/PT6/validate_promoters.py` (coord logic incl. reverse strand verified — matched dist median 4 bp, signed median 0; locks the candidate-table schema for T32). Baseline finding: off-the-shelf `prokbert-mini-long-promoter` is weak genome-wide (recall 0.37, precision 0.036, ~2× enrichment over random) but 4 bp-accurate when it hits — the bar to beat after fine-tuning. (dep PT5)

**Phase A — data acquisition & curation**
- T5 Download Clostridium genomes — Salimah — ✅ 166 RefSeq genomes (taxon 1485)
- T6 Re-annotate with Bakta — Leo — ✅ 166/166, Bakta 1.11.4 + DB v6.0 **light** (in `bakta_outputs/`)
- T7 dRNA-seq literature search — Salimah — ✅ (see §8; 20 dRNA-seq vs 27 published TSS)
- T7b Published in-genus TSS → `published_tss_promoters.parquet` — Salimah/Leo/Oak — ✅ **5,759 verified in-genus
  positives** (EXTENDED 2026-07): Hocq 2022 `intergenic_sense` TSS, 81 bp, from **three** strains in the one
  committed workbook — *C. acetobutylicum* ATCC 824 (1,434) + *C. beijerinckii* NCIMB 8052 (2,482) + DSM 6423
  (1,843). Run on HPC4 via `build_published_tss.sh` + `extract_published_tss.py`. Genomes auto-fetched via
  `ncbi-datasets-cli` (installed into the env 2026-07); `build_published_tss.sh` now **stages each genome +
  auto-derives the Locus→record contig-map from the FASTA headers** (chromosome = longest record; DSM 6423's
  12 phage/plasmid TSS drop as unmapped, <1%). Needs the Hocq `.xlsx` (Git-LFS), `openpyxl`, and internet on
  the login node for the genome fetch. Skips LFS pointers / missing inputs gracefully. **In-genus training-positive
  source (§7).** Al-Bassam 2018 *C. ljungdahlii* TSS have since been downloaded + extracted (1,245 primary TSS,
  `evaluation-and-calibration/ljungdahlii_holdout_tss.tsv`) but are used as the **held-out** in-genus validation
  genome (Regime C / T33 validation), deliberately NOT added as training positives. (`data-acquisition-and-curation/T7/`)
- T8 Normalise RegulonDB+DBTBS → Parquet — Leo — ✅ **4,336** (3,963 RegulonDB + 373 DBTBS), landed on
  `main` (commit `82d9bbb`): `run_t8_pipeline.py` reads the committed `DBTBS_prom.fa` (Sine's 373 σA
  set, 81 bp) + `PromoterS_Set.txt`, asserts both sources non-empty, and reproducibly regenerates
  `curated_positive_promoters.parquet`. `data-acquisition-and-curation/T8/`
- T9 Curate synthetic promoters → `literature_promoters.tsv` — Salimah — ✅ 93 promoters merged + cleaned into one `literature_promoters.tsv` via re-runnable `build_literature_promoters.py`: C. acetobutylicum 120 bp cloning scaffold trimmed → ~50 bp promoter, 7 alignment-gap hyphens stripped → clean ACGT, strengths kept verbatim with `strength_unit`, `original_sequence` retained lossless. Verified 93/93 clean ACGT, reconstruction exact. NB C. phytofermentans (25) is out-of-genus (→ *Lachnoclostridium*), flagged `in_genus=False`. Held out → eval T24/T26.
- T10 Re-process dRNA-seq w/ TSSpredator — Sine — ✅ **re-run + verified on HPC4 (2026-06): over-call
  fixed → 1,746 enriched primary TSS** (was 170,766; the merged `inputs/tss_sites.gff3`). Root cause was
  the replicate filter, not thresholds: `fuchs_config_hpc.conf` (3×3, `numReplicates=3`/`minNumRepMatches=2`)
  + `soutourina_tap_config.conf` (TAP 1×1), `minNormalHeight 0→0.2`. Verified by re-running TSSpredator off
  the **committed `wig_chrom_only/` tracks** (no SRA re-map needed) → MasterTable 73,116 candidates → 1,746
  filtered primary. Gotchas: stage-5 must `mkdir -p` the config output dirs; the configs hard-code
  `/project/igem2026/xzhangjj/cdiff_tss` (repoint to the repo, or stage there); `cdiff_chrom_only.fna` must
  be fetched. NB C. difficile = **out-of-genus** → cross-species-adjacent support, not the in-genus signal
  (and not passed to T15's `--positives`). (`tss_prep.smk` / `run_tss_pipeline.sh`; dep T6,T7)

**Phase B — dataset construction** (land in `dataset-construction/`)
- T11 Housekeeping gene set — Sine — ✅ ran on HPC4 → **22,760** housekeeping genes / 166 genomes (`dataset-construction/T11/define_housekeeping.py`; selects by product per §9). (dep T6)
- T12 Operon detection (+ contig-end exclusion) — Seonyu — ✅ **fixed + verified 2026-06**: (a) nested Bakta path (`<genome>_genomic_output/<genome>_genomic.gff3`); (b) **reverse-strand leader bug** — now clusters then picks the most-upstream gene strand-aware (min-start on `+`, **max-stop on `−`**). Ran on 166 genomes → 22,760 loci → 14,659 leaders; `+` leaders 7064/7064 and `−` leaders independently verified correct (`dataset-construction/T12/operon_detection.py`). (dep T6,T11)
- T13 Upstream-region extraction (100–300 bp, + contig-end exclusion) — Seonyu — ✅ **fixed + verified 2026-06**: nested Bakta `.fna` path; ran end-to-end (236/236 windows, all 300 bp clean ACGT, ~72% AT). Reused by T28. Its windows are INFERRED positives → T15 treats them as **silver / down-weighted** (`--silver-weight 0.3`); §7 open decision now **resolved** (verified in-genus positives come from T7b). (`dataset-construction/T13/upstream_extraction.py`) (dep T11,T12)
- T14 Four-class negatives incl. SRS — Seonyu — ✅ **fixed + verified 2026-06**: nested Bakta `.fna`/`.gff3` path; ran end-to-end → all four classes generate (SRS/CDS/Downstream/Intergenic). Minor: `random` is unseeded (non-reproducible negatives) — add a `--seed` before locking the dataset. (`dataset-construction/T14/negative_set_generation.py`) (dep T6,T13)
- T15 CD-HIT clustering + cluster-aware splits — Sine — ✅ **run + verified on HPC4 (2026-07)**:
  `dataset-construction/T15/cdhit_splits.py` harmonises the 5 sources, re-windows to 81 bp (T13 → gene-PROXIMAL
  crop; T14 SRS **regenerated** as 81 bp dinucleotide shuffles of the re-windowed positives), `cd-hit-est`
  clusters, then splits cluster-disjoint. **Eval is VERIFIED-ONLY:** T13 silver clusters are pinned to TRAIN
  (`StratifiedGroupKFold` over eligible-only), so val/test hold only verified positives (~87/6.4/6.4, NOT
  80/10/10). **`weight` = label-confidence (silver 0.3) × class-balance (step 6b rescales train negatives so
  train is effectively 50/50 — verified negatives land at ~0.633).** Ran on the extended data → **47,088 rows:
  train 40,442 / val 3,323 / test 3,323; train pos_mass==neg_mass==11,919 (50/50); val/test verified-only ~29%
  positive; leakage check PASS.** Decisions: window=81, silver-weight 0.3, T9 held out (`t9_train_fraction=0`). (dep T8,T7b,T13,T14)
- T16 Per-model tokenisation (ProkBERT/NT/CNN) — Leo — ✅ **run on HPC4 (2026-07)**: `dataset-construction/T16/tokenise.py`
  → `results/T16/{prokbert,nt,cnn}/{train,val,test}` (ProkBERT 78 tok, NT **67 tok**, CNN one-hot (N,81,4)).
  ⚠️ NT=67 tok is higher than a typical 6-mer NT tokenisation (~15–18) — worth a sanity-check before the heavy T19 run. (dep T15)

**Phase C — model training**
- T17 Finalise fine-tuning recipe — **Sanaa** — 🟡 via the T21 sweep (template defaults = starting recipe; sweep tunes them). T19's slight overfit-after-epoch-2 ⇒ its LR has room to drop.
- T18 ProkBERT-mini fine-tune — **Sanaa** — ✅ **run + verified on HPC4 (2026-07): val AUPRC 0.876 / test 0.872, weighted, full 40,442-row train.** `model-training/T18/train_prokbert.py`. Session bug-fixes: v5 `warmup_ratio` (feature-detect), `remove_unused_columns=False` (the `weight` KeyError). (dep T4,T16)
- T19 NT v2 250M LoRA fine-tune — Sanaa — ✅ **run + verified on HPC4 (2026-07): val AUPRC 0.853 / test 0.841, weighted, LoRA 0.507% trainable.** Runs on the **`nt-tf4x`** (transformers 4.57) env, NOT 5.3.0 (`infrastructure/T4/NT_ENV.md`); `--no-grad-checkpoint`. `model-training/T19/train_nt_lora.py`. (dep T4,T16)
- T20 Custom 1D CNN from scratch — Yany — ✅ **run + verified on HPC4: 0.9 val AUPRC 0.811; 0.6 retrain val 0.784 / test 0.804.** `model-training/T20/train_cnn.py` (weighted-mean loss on T16 `weight`; BatchNorm on; `--data-dir` defaults to `results/T16/cnn`). **+ CNN-v2 (2026-08): multi-encoding fusion (one-hot+ENAC+di/tri-nuc) `model-training/T20/train_cnn_v2.py`, 0.6 val 0.827 / test 0.848 — a 4th ensemble model; reads T16/cnn one-hot (reconstructs seqs), wired through T22/modular_eval/genome_inference/T24; scores are optimistic → needs T26 calibration.** (dep T4,T16)
- T21 Hyperparameter sweep — Sanaa+Oak — ✅ **run on 0.6 for all 4 models (2026-08)** incl. cnn_v2 (now in `SWEEP_CONFIG`; `--data-dir` via extra_args). `results/T21_06/{model}/best_params.json`, 15 trials × `--sweep-epochs 4` proxy, two-env split. Winners retrained at full epochs → `results/T21_final_06`. **Key result: the earlier sweep was on 0.9 data, so T21_final ran 0.9-tuned knobs; re-sweeping recovered ~⅔ of ProkBERT's genome-scale drop (see §10) → adopt `T21_final_06/prokbert`.** (`model-training/T21/hyperparameter_sweep.py`; dep T18,T19,T20)

**Phase D — evaluation & calibration**
- T22 Metrics suite (AUROC/AUPRC/MCC/F1) — Yany — 🟡 **4-model test predictions + metrics run at 0.6** (`T22/predict_and_score.py` → `results/T22`), plus `modular_eval.py` negatives-by-class → `results/T25b`; consolidated into the 3-regime `evaluation-and-calibration/species_metrics_report.{tex,pdf}` + `negatives_by_class.csv` (council-audited). (dep PT3,T18,T19,T20)
- T23 Cluster-level 5-fold CV — Yany — 📄 `evaluation-and-calibration/T23/` (dep T15,T18-20)
- T24 Cross-species external holdout — Yany — 🟡 **cnn_v2 wired in (`--models …,cnn_v2`, `--cnn-v2-model-path`); 4-model 0.6 refresh of the T9 + out-of-genus *C. phytofermentans* composition-matched holdout queued on HPC4** (the report's Regime-B out-of-genus numbers are still pre-0.6 until this runs). `evaluation-and-calibration/T24/` (dep T9,T18-20)
- T25 Sigma-factor-stratified eval — Seonyu+Oak — ⬜ (dep T10,T22)
- T26 Calibration regressor (AT covariate) — Ishi — 🟡 **three per-model Platt(+AT) calibration scripts
  wired for CV model-selection (not yet run on HPC4)**: `calibrate_cnn.py` / `prokbert_calib.py` /
  `nt_calib.py` now calibrate on **93 T9 positives + 186 curated genomic negatives** (130
  convergent-intergenic + 56 CDS-interior from `extract_negatives.py`, read from the T26 dir by default;
  counts/paths are `--n-convergent`/`--n-cds`/`--convergent-negatives`/`--cds-negatives`). The single
  train/test split is replaced by **stratified K-fold CV that picks unregularized vs L2 Platt+AT on OOF
  log-loss+Brier (simpler wins unless L2 beats by >`--tol`=0.005; `--l2-C`=1.0), then refits the chosen
  calibrator on the full set** (StandardScaler→LogisticRegression pipeline; one artifact
  `<model>_calibrator_at.joblib`). `run_calibration.sh` updated to the new flags (SRS-negative step
  dropped). Not run yet — needs the trained models + the two negative TSVs staged on HPC4. (dep PT4,T9,T22)
- T27 Empirical FDR via SRS null → q-values — Ishi — ⬜ (dep PT4,T14,T26)

**Phase E — mining & motif annotation**
- T28 Sliding-window inference on Clostridium — Swasti — 📄 `mining-and-motif-annotation/T28/` (dep PT6,T13,T20,T21)
- T29 Ensemble aggregation (mean rank-percentile) — Swasti/Salimah — ⬜ (dep T28)
- T30 Locus-level peak calling — Ishi — ⬜ (dep T29)
- T31 Motif annotation FIMO/MEME (−10/−35 + sigma) — Leo — ⬜ (dep T30; FIMO alone if short on time)
- T32 FINAL candidate output (TSV/JSONL, PT6 schema) — Ishi+Oak — ⬜ (dep T26,T27,T30,T31)
- T33 Iterative hard-negative mining (v2 refinement) — Oak+? — ⬜ **mine confident false positives where labels EXIST, and keep the mining and validation genomes DISJOINT.** ⚠️ Do NOT relabel the target's (*C. butyricum*) top T28 scores as negatives: with no target TSS a high score is ambiguous (may be a *real* promoter) and would train the model to reject real promoters. **Mine from:** (a) high-scoring windows on the in-genus genomes already in TRAINING that we have TSS for (*C. acetobutylicum/beijerinckii*, Hocq), far from their known TSS; and (b) high-scoring windows deep in annotated CDS far from gene 5′ ends — annotation (not TSS) gives the label, so this class is mineable **even on the target itself** and is the most transfer-safe. Add as a 5th T14 negative class → regenerate T15/T16 → retrain T18–T20. **Do NOT mine from *C. ljungdahlii*** — keep it untouched as the held-out validation genome (we have its 1,245 primary TSS; it is already the Regime-C benchmark). **Validate there:** the retrain must lower FP peaks on *C. ljungdahlii* while holding recall on its TSS (mining AND validating the same genome would be circular). Council caveats: TSS catalogues are incomplete (mostly primary / intergenic-sense) and deep CDS still holds sparse internal/antisense promoters, so a mined "FP" can be a real unannotated promoter. **Load-bearing mitigation = DOWN-WEIGHT mined negatives** (T15 silver-weight style, ~0.3) + keep to 1–2 rounds; do NOT filter them by "has a clean −10/−35" (self-defeating: AT-trap negatives look promoter-like by design), and note "far from a 5′ end" does nothing for mid-gene internal/antisense promoters (better: exclude windows overlapping any dRNA-seq-called internal/antisense TSS where available). Transfer to the more AT-rich target (GC ~28–29% vs relatives ~30–31%) is imperfect — another reason to lean on target-side deep-CDS negatives. Strongest FP-reduction lever (DeePromoter-style); a 2nd pass, not a new DAG rule. See `WORKFLOW_RATIONALE.md`. (dep T28 + a mining-genome scan; validation = held-out *C. ljungdahlii*)

---

## 7. Conventions & decisions locked

> ✅ **RESOLVED (2026-06).** The "no _verified_ in-genus positive in training" problem (raised
> 2026-06-19) is fixed. The T15 positive set is now **T8 (cross-species, verified) + T7b (in-genus,
> verified) + T13 (in-genus, _silver_)**, with T9 and T10 held out:
> 1. **T7b supplies 5,759 verified in-genus positives** (Hocq 2022 empirical TSS: _C. acetobutylicum_ 1,434
>    + _C. beijerinckii_ NCIMB 8052 / DSM 6423 = 4,325; extended 2026-07) — built + run on HPC4, feeds T15 `--positives`.
> 2. **T13 housekeeping upstreams = silver/weak labels** — T15 down-weights them (`--silver-weight 0.3`),
>    never mixed at equal weight with verified positives; the SRS null is **regenerated per-positive at
>    81 bp** (not a crop of the 300 bp shuffle).
> 3. **T9 held fully out** for eval (`config t9_train_fraction=0`) — the "split T9 into training"
>    workaround is no longer needed now that T7b exists, and T9's ~51 bp would need poor N-padding.
> 4. **T10 (_C. difficile_) stays out-of-genus** — re-run + verified (1,746 TSS) but treated as
>    cross-species-adjacent support, and **not** passed to T15's `--positives`.
> Remaining: the Al-Bassam 2018 _C. ljungdahlii_ TSS are downloaded (1,245 primary TSS) but held OUT of
> training on purpose — they are the clean in-genus validation genome (Regime C, and the T33 validation
> genome). Adding them as training positives would forfeit that held-out set.

- **ProkBERT base = `neuralbioinfo/prokbert-mini` (NOT `-mini-long`)** — decided 2026-06: the ensemble's
  ProkBERT member (T18) fine-tunes plain **prokbert-mini**. The 81 bp T15 windows are short, so the
  long-context `-mini-long` variant buys nothing — mini is the right size; T16 tokenises with mini's
  tokenizer to match. *Fine-tune base only* — the off-the-shelf **baselines keep their own IDs** (PT5 =
  `prokbert-mini-promoter`, PT6 = `prokbert-mini-long-promoter`); don't rename those. ⚠️ The T18 template
  still hard-codes `max_length=300` (leftover from the old 300 bp windows) — set it to the 81 bp T15 window.
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
- **T15 split hyperparameters — sane defaults, NOT derived optima; sweepable by the T15 owner:**
  - `cdhit_identity = 0.6` (config, **changed from 0.9 → 0.6 on a reviewer's rec, 2026-08**) — clustering identity;
    seqs ≥ this share a cluster and whole clusters stay on one split side (the anti-leakage guard). **CD-HIT-est
    cannot cluster DNA below ~0.8 (word-size floor), so `cdhit_splits.py` auto-routes <0.8 to MMseqs2**
    (`easy-cluster --min-seq-id 0.6 -c 0.8 --cov-mode 1`; `mmseqs2` added to the env). At 0.6 the 47,088-row set
    formed 39,278 clusters; val/test stayed healthy and the leakage check passed. The stricter threshold barely
    moved balanced AUPRC vs 0.9 (so the old scores were not leakage-inflated) but did shift the genome-scale ranking
    (see §10 / the species report). Validate further by sweeping 0.6–0.9 + checking CV stability.
  - `silver_weight = 0.3` — T13 inferred positives weighted 0.3 vs 1.0 for verified. T13 is inferred through
    3 stacked assumptions and the −10-crop check found ~⅓ of windows don't frame the promoter, so at full
    weight the ~14k AT-rich windows would teach "AT-rich = promoter" (the §1 trap). 0.3 balances the
    *effective* weight: 13,838 × 0.3 ≈ 4,150 silver vs 5,634 verified → verified ≈ 58% / silver ≈ 42%
    (0 = drop T13; 1.0 = silver dominates at 71%). Set properly by sweeping {0,0.1,0.3,0.5,1.0} on validation AUPRC.
  - **`weight` now also class-balances TRAIN to effective 50/50** (cdhit_splits.py step 6b rescales train
    negative weights so pos-mass == neg-mass; down-weighting silver alone had skewed train to ~36% positive).
    So `weight` = label-confidence (silver 0.3) × class-balance; val/test left at natural prevalence.
  - ⚠️ **CONFIRMED INERT in current code (2026-06):** no trainer applies it. T18/T19 are unwritten; **T20's
    template (L538–544) builds a `WeightedRandomSampler` from inverse class-frequency — it ignores AND
    conflicts with the T15 weight** (re-balances by class but leaves silver at ~75% of positives → the
    AT-trap). Trainers MUST apply `weight` as a per-sample loss weight (T20 fix+run: `model-training/T20/YANY_T20_RUN_CLAUDE.md`);
    until then T13 trains at full weight and training is negative-skewed.
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
- **T8 RegulonDB parse — FIXED:** `run_t8_pipeline.py` now reads the right columns
  (5=sigmaFactor, 6=pmSequence), loads the committed local `PromoterS_Set.txt`, skips the
  header row, drops sequenceless "None" rows, and keeps `confidence_level` → **3,963**
  E. coli promoters. The committed `curated_positive_promoters.parquet` has since been **regenerated**
  to the real 3,963 rows (no longer the old 2-row placeholder; verified byte-identical on re-run). **Still open — DBTBS:** the
  `promoter.dat.gz` bulk file in `parse_dbtbs()` does **not exist** (no web record; Wayback has
  no snapshot) and its flat-file format was guessed; DBTBS only offers sequence-search +
  browsing. Pick a real B. subtilis source (scrape DBTBS pages, the Sierro 2008 NAR supplement,
  or Nicolas 2012 genome-wide TSS) and rewrite `parse_dbtbs()` to match.
- **PT1 (open):** fine-tuning reportedly took ~6 h on ~9.5k examples → almost certainly ran on
  **CPU**. Confirm `torch.cuda.is_available()`, `nvidia-smi`, and `--gpus-per-node=1`.
- **No secrets in code.** A HuggingFace token was once hard-coded in `prokbert_finetune.py`
  (`neuralbioinfo/prokbert-mini` is public — no token needed). Never commit tokens/keys.
- **PT5 batch-scoring bug — FIXED:** `predict_batch` reassigned `inputs` inside a `for` loop, so
  only the **last** window of each batch was tokenised, then scored as a fake batch-of-1 — emitting
  **one row per 64-window batch** with the score attached to the **wrong** window (exactly 29,010 of
  ~1.86M windows). Now tokenises/stacks/scores the whole batch in one pass (window also 80→81 bp to
  match the model's TSS-at-+61 framing). The committed `mg1655_promoter_predictions.csv` predates the
  fix — **regenerate it** (needs `prokbert` + GPU) before trusting any numbers.
- **torch/CUDA — RESOLVED (2026-06-19):** the SuperPOD H800 driver maxes at **CUDA 12.8** (not 12.1 as first
  assumed), and prokbert needs `transformers>=4.56` which needs a newer torch than conda ships. Default PyPI
  torch is a **CUDA-13** build the 12.8 driver can't run → silent **CPU fallback**. Fix: T4 installs
  **torch==2.11.0+cu128 / torchvision==0.26.0+cu128 via pip** from `download.pytorch.org/whl/cu128` (exact
  `+cu128` pins, listed before prokbert/transformers); the conda `pytorch`/`pytorch-cuda` pins and the
  `pytorch`/`nvidia` channels are gone. GPU env builds from `environment.yml`; the CPU bio CLI
  (meme/sra-tools/segemehl) from `environment-bio.yml`. Verify `torch.cuda.is_available()` +
  `torch.version.cuda=='12.8'` **inside a GPU job**. Never run a bare `pip install torch` — it pulls the cu13 build.
- **`git` over SSH breaks while a training env is active (2026-07):** the env's `activate.d` prepends
  `$CONDA_PREFIX/lib` to `LD_LIBRARY_PATH` (needed for torch's libstdc++), which makes the system `ssh` load the
  env's OpenSSL → `OpenSSL version mismatch … Could not read from remote repository`. Run git as
  **`LD_LIBRARY_PATH= git …`** (or in a shell without the env active). The commit/push flow lives on the Mac;
  the shared `/project` clone should be **pull-only** (a stray local commit there, e.g. `ceb9783`, diverges it).
- **`~/.local` shadows the env (2026-07):** a stale user-site `pandas`/`numpy` (built for numpy 1.x) on
  `~/.local` shadows the env and crashes with `_ARRAY_API not found` under numpy 2.x. The envs set
  `PYTHONNOUSERSITE=1` (so **jobs are immune**); for interactive checks, activate the env or prefix
  `PYTHONNOUSERSITE=1` (do NOT add `python -S` — that also hides the env's own packages).

---

## 10. Status snapshot

**Solid (done + run/verified on HPC4):** T1–T16 + PT3/PT5, **all Phase-C models retrained at the 0.6 clustering
(MMseqs2, reviewer rec), and Phase-D evaluation regenerated for four models across three regimes.**
- **0.6 four-model balanced numbers** (weighted, full 40,451-row train / 3,318 verified-only test): **ProkBERT
  val 0.878 / test AUPRC 0.884**, **NT-LoRA 0.855 / 0.854** (`nt-tf4x` env), **CNN 0.784 / 0.804**, and a **new
  4th model CNN-v2** (multi-encoding fusion, `model-training/T20/train_cnn_v2.py`) **val 0.827 / test 0.848**.
  0.6 barely moved the balanced numbers vs the old 0.9 splits → the earlier scores were not leakage-inflated.
- **CNN-v2 is the round's standout:** best composition-matched SRS AUROC (0.924, cleanest motif-learning signal)
  AND best restricted genome-scale precision at recall 0.5–0.9. It reads the T16/cnn one-hot data (reconstructs
  sequences) and is wired into T22/modular_eval/genome_inference/T24. **Ensemble decision: keep all FOUR** (Oak,
  2026-08); the two CNNs are correlated, flagged for T29 weighting.
- **Deployment finding flip (see `evaluation-and-calibration/species_metrics_report.{tex,pdf}` + `negatives_by_class.csv`,
  regenerated + council-audited):** at genome scale (Regime C, held-out *C. ljungdahlii*, restricted scan) the **CNN
  family leads, CNN-v2 best at practical recall**; **NT is weakest at genome scale despite 2nd-best balanced AUPRC**;
  and **ProkBERT's restricted precision@0.5 dropped 45%→33% vs the pre-0.6 model (T18, val 0.876)** while its balanced
  AUPRC held — a genome-scale regression now **CONFIRMED** (re-scanned T18 through the identical pipeline → reproduced
  45.0% / 760 FP@0.5). Confound: T18 differs from the 0.6 model in both clustering threshold AND swept hyperparams, so it's
  "the retrain as a whole," not 0.6 alone. **RESOLVED (2026-08): cnn_v2 is threshold-robust** — retrained at 0.9 its
  genome-scale precision is ~unchanged (40% @0.5 vs 0.6's 44%; balanced 0.838/0.836 vs 0.827/0.848), i.e. UNLIKE ProkBERT
  it pays no genome-scale penalty for 0.6 (it reads real motif signal, not near-duplicates). cnn_v2@0.6 already matches
  old 0.9 ProkBERT genome-scale (44%≈45%) with no leakage exposure → **deploy cnn_v2@0.6, don't revert ProkBERT to 0.9.**
- **Regime B refreshed on 0.6 (T24, in+out-genus):** in-genus composition-matched AUROC ProkBERT 0.736 > cnn_v2 0.659 >
  CNN 0.629 > **NT 0.440 (collapsed to chance from its pre-0.6 best 0.71)**; out-of-genus all fail (ProkBERT anti-ranks 0.17).
  **NT is the 0.6-round weak link on every deployment axis** (genome-scale worst AND in-genus motif at chance) despite 2nd-best
  balanced AUPRC — the CNN family carries the ensemble.
- **0.6 hyperparameter sweep + swept-ProkBERT recovery (2026-08):** the T21 sweep had been run on 0.9 data; the 0.6 models
  (T21_final) inherited 0.9-tuned knobs. Re-swept on 0.6 (all 4 models incl. cnn_v2, `results/T21_06/*/best_params.json`) +
  retrained (`results/T21_final_06`). Balanced barely moved (ProkBERT 0.884→0.887; cnn_v2/NT slightly DOWN = proxy noise, keep
  their current versions). **Decisive: the 0.6-swept ProkBERT's genome-scale restricted precision@0.5 recovered 33%→40.8%
  (905 FP), i.e. ~⅔ of the 45%→33% drop was STALE HYPERPARAMETERS, ~⅓ (40.8 vs 45) is intrinsic to 0.6 clustering.**
  → **ADOPT the 0.6-swept ProkBERT (`T21_final_06/prokbert`)** over T21_final's; it beats it on both balanced AND genome scale,
  and gets most of the 0.9 benefit with the clean leakage-safe protocol (kills the "revert to 0.9" question for good).
**Frontier (Phase D→E):** evaluation regeneration COMPLETE (all regimes, 4 models, 0.6 + the 0.9 robustness check).
Next: the rest of Phase D (T25 sigma-strat,
T26 calibration — note CNN-v2 needs it, T27 FDR) + Phase E (T28 mining on the target, T29 four-model rank-percentile
ensemble which is calibration-invariant, T30–T32). Runbook: `evaluation-and-calibration/REGEN_EVAL_0.6.md`.
**Two envs (`infrastructure/T4/NT_ENV.md`):** 5.3.0 for everything; 4.57 (`nt-tf4x`) for NT only.
**Needs attention:** PT2 (unwritten) · PT1 GPU re-run · PT6 needs a RegulonDB TSS table · Regime B out-of-genus numbers
in the report are still pre-0.6 (refresh queued) · CNN-v2 genome scan is slow (per-window Python featurisation; bump
`--step` or vectorise if needed) · keep the shared `/project` clone **pull-only**.
**When in doubt:** confirm owners/decisions with Oak; verify tool usage and accessions against
the literature; keep changes surgical and within your task.
