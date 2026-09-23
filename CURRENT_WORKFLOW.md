# Current workflow — HKUST iGEM 2026 Promoter Mining (Phases A–E)

*Written 2026-07-15. Ground truth = the Snakemake DAG (`infrastructure/T3/snakemake_scaffold/Snakefile`).
This describes the pipeline as it is actually wired. CLAUDE.md §6/§10 is the owner/status table and currently
lags this file (see "Status note" at the end). For **why** this approach is sound, see `WORKFLOW_RATIONALE.md`.*

## The three models (read first — the naming trips people up)

The ensemble we train and mine with is **three fine-tuned models**:

- **ProkBERT** — base **`neuralbioinfo/prokbert-mini`**, fine-tuned in **T18** (81 bp windows). This is *our* ProkBERT.
- **Nucleotide Transformer (NT)** — `InstaDeepAI/nucleotide-transformer-v2-250m-multi-species` + LoRA, **T19**.
- **CNN** — a 1-D convnet trained from scratch, **T20**.

Do **not** confuse `prokbert-mini` (the base we fine-tune) with **`prokbert-mini-promoter`** and
**`prokbert-mini-long-promoter`** — those are *off-the-shelf* promoter classifiers used **only as baselines**
(PT5 / PT6, "the bar to beat"). They are **not** in the pipeline or the ensemble. (CLAUDE.md §7 locks this.)

## Pipeline by phase

Status key: ✅ done/verified · 🟡 code exists, run/verify pending · ⬜ not written.

**Phase A — data acquisition & curation (local + HPC4): ✅**
166 *Clostridium* genomes (T5), Bakta re-annotation (T6), literature/TSS curation (T7/T7b/T8/T9),
TSSpredator empirical TSS (T10).

**Phase B — dataset construction: ✅**
Housekeeping genes (T11) → operon leaders (T12) → upstream positive windows (T13) → four-class negatives
incl. SRS (T14) → CD-HIT cluster-disjoint splits (T15: 47,088 rows = 40,442 train / 3,323 val / 3,323 test)
→ per-model tokenisation (T16).

**Phase C — model training: ✅ (this sprint)**
T18 ProkBERT (val AUPRC 0.876), T19 NT-LoRA (0.853), T20 CNN (0.811), all verified. T21 = Optuna
hyperparameter sweep over the three (optimising val AUPRC), run this sprint with the retrained winners at
`results/T21_final/`.

**Phase D — evaluation & calibration: partial**
- **T22** metrics suite — 🟡 code exists (`predict_and_score.py`).
- **T23** cluster-disjoint 5-fold CV — ✅ **run this sprint** (numbers below).
- **T24** cross-species external holdout — ✅ **run this sprint** (numbers below).
- **T25** sigma-factor-stratified eval — ⬜.
- **T26** calibration regressor (AT-content covariate) — 🟡 starter code (`prokbert_calib.py`).
- **T27** empirical FDR via the SRS null → q-values — ⬜.

**Phase E — mining & motif annotation: ⬜ (templates only)**
T28 sliding-window inference across *C. butyricum* → T29 ensemble (mean rank-percentile) → T30 locus
peak-calling → T31 FIMO/MEME −10/−35 motif annotation → T32 final tiered candidate table.
T33 = iterative hard-negative mining (an upstream retraining loop, not a DAG rule).

## How the tasks wire together (it is a fork, not a line)

Phase C converges at the **T21** sweep. From T21 the graph splits into **two branches that meet only at T32**:

- **Evaluation/calibration branch (Phase D):** T22 (metrics) feeds T25 (sigma eval) and T26 (calibration);
  T26 → T27 (FDR). **T23, T24 and T25 are terminal leaves** — evaluation reports that feed nothing downstream.
- **Mining branch (Phase E):** T28 (genome scan) → T29 (ensemble) → T30 (peaks) → T31 (motifs). **T28 depends
  only on the trained models (T21), PT6 and T13 — it does *not* wait on calibration**, so mining and
  calibration run in parallel.
- The branches **converge at T32**, the final candidate table, which joins **four inputs: T26 + T27 + T30 + T31**
  (T30 feeds T32 directly as well as via T31).
- **T33** adds mined hard negatives to T14 → T15/T16 → T18–T20 and retrains — a second pass, and the only
  step that raises the models' intrinsic discrimination. Mine confident false positives where labels exist
  (high-scoring windows on TSS-annotated in-genus relatives far from their TSS, and deep-CDS windows far
  from gene 5′ ends — mineable even on the target), **not** the target's top T28 scores (no target TSS, so
  a high score may be a real promoter). Validate on the held-out *C. ljungdahlii* benchmark, which must not
  itself be mined. See `CLAUDE.md` T33 / `WORKFLOW_RATIONALE.md`.

Exact edges (from the Snakefile): T22←PT3,T21 · T23←T15,T21 · T24←T9,T21 · T25←T10,T22 · T26←PT4,T9,T22 ·
T27←PT4,T14,T26 · T28←PT6,T13,T21 · T29←T28 · T30←T29 · T31←T30 · T32←T26,T27,T30,T31. Graph leaves: T23, T24, T25, T32.

## Results so far (run on HPC4 this sprint; `results/` is git-ignored, so not committed)

- **T23 — cluster CV (in-distribution, stable):** AUPRC 0.845±0.011 (ProkBERT) / 0.783±0.014 (NT) /
  0.769±0.009 (CNN). The tight std means the single-split training numbers weren't a lucky partition.
- **T24 — cross-species holdout (out-of-distribution, hard):** against per-promoter dinucleotide-shuffled
  (composition-matched) negatives, balanced 1:1 (chance AUROC 0.5) → AUROC 0.62 / 0.55 / 0.50. Only ProkBERT
  keeps a modest structural edge; NT is at chance. This *measures* the AT-composition confound — the models
  partly ride base composition rather than −10/−35 motif structure.

## What the downstream stack does about the T24 weakness (short version)

Only **T33** (hard-negative mining) can raise the T24 number — it retrains against the model's own
composition-confounded false positives, so the −10/−35 arrangement becomes the deciding feature. Everything
else *manages* a limited model rather than curing it: **T26** calibration reweights by AT-content (cuts
deployed false positives but can't change a composition-matched score), **T27** empirical FDR against the SRS
null filters composition-driven candidates (and honestly passes few if the only signal is composition),
**T29** ensembling cancels each model's idiosyncratic error but not the composition bias they share, **T30**
peak-calling drops sporadic hits, **T31** motif annotation adds independent −35+spacing evidence (the
composition-orthogonal part), and **T25** stratifies performance by sigma factor to show where the model is
trustworthy. The deliverable stays an **enriched, FDR-controlled candidate list for wet-lab validation — not a
precise genome map** (rationale + citations in `WORKFLOW_RATIONALE.md`).

## Status note

CLAUDE.md §6 task table and §10 snapshot currently lag this file — they still mark T22–T27 as
not-started/template. They should read: T18–T20 verified; T23/T24 run (numbers above); T22/T26 code exists;
T25/T27 and all of Phase E (T28–T32) + T33 pending.
