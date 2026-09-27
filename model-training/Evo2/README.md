# Evo2 — the genome-scale leader we found late (2026-08)

**Status: NOT dropped.** A frozen Evo2-7B embedding + logistic-regression probe has the *worst*
balanced AUPRC of any model (0.70; cnn_v2 scores 0.726 on the same split), but on the genome-scale false-positive benchmark (Regime C) it
beats cnn_v2-xs — our best CNN and prior genome-scale leader — at every recall on **both**
validation genomes. Full write-up with method, audit and numbers: **`EVO2_GENOME_SCALE_REPORT.md`**.

> This reverses the earlier "evaluated and NOT used, below all four" call, which was made on the
> balanced metric alone. It is the sharpest balanced≠deployment case in the project (opposite direction
> to NT, which is balanced-strong / genome-weak).

## What was tested
Frozen **Evo2-7B** (`evo2_7b`, StripedHyena 2), mean-pooled embeddings from layer `blocks.28.mlp.l3`
(4096-d), on the 81 bp core of the deployed cross-species split (`T15_xs`). A weighted scikit-learn
probe is trained on those embeddings and the better of logreg / RF is auto-picked on weighted val
AUPRC — **logreg wins** (val 0.7007 / test 0.7026; RF 0.61). The backbone is **frozen, never
fine-tuned.** The same probe (`evo2_probe_xs.pkl`) is reused across both genomes.

## The result

Balanced test — Evo2 is last. The four fine-tuned numbers are the plain 0.6-split retrains
(`T21_final`); Evo2 was measured on `T15_xs`, where the deployed cnn_v2 scores 0.726:

| | Test AUPRC |
|---|---|
| ProkBERT / NT / cnn_v2 / CNN (fine-tuned) | 0.884 / 0.854 / 0.848 / 0.804 |
| **Evo2 frozen + logreg** | **0.703** |

Genome scale (Regime C, validation genomes, **matched step 10**, restricted) — Evo2 wins every row:

| | ljungdahlii r=0.1 | r=0.5 | autoethanogenum r=0.1 | r=0.5 |
|---|---|---|---|---|
| **Evo2** | **71.0%** (51 FP) | **58.0%** | **85.8%** (48 FP) | **71.4%** |
| cnn_v2-xs @step10 | 63.1% (73 FP) | 48.2% | 81.2% (67 FP) | 67.1% |

(Both models detect the same number of TSS at each recall; the gap is entirely fewer false-positive
peaks. Absolute precision isn't comparable across genomes — cau has denser truth — only the
within-genome Evo2-vs-cnn_v2 delta is, and it favors Evo2 on both.)

## Why it's not just an artifact
The comparison code was audited (see the report): minus-strand windows get identical forward
coordinates + reading-strand sequence in the Evo2 and CNN scanners; the restricted scan covers exactly
the benchmarked window set; embeddings are re-scattered to original order; the probe's layer/pool/window
match the extractor. Neither genome's published start sites are in T15_xs, though both strains are
among the 166 genomes that supplied its silver positives and negatives (`docs/PIPELINE.md`). The
step-10 comparison is fair because cnn_v2 is re-benchmarked at the same step 10, not its native step 1.

## Files (all reproducible, and now candidates for the mine — not just the archive)
- `extract_embeddings_evo2.py` + `extract_embeddings.sbatch` — build embeddings (`evo2_embeddings/`).
- `evo2_promoter_classifier.py` — the weighted logreg/RF probe (reports weighted AUPRC). Run it with
  `--out infrastructure/T3/snakemake_scaffold/results/regimeC/evo2/evo2_probe_xs.pkl` to build the deployed probe
  (commands in `docs/PIPELINE.md`, Phase C).
- `evo2_promoter_classifier.pkl` — the deployed probe, a copy of `evo2_probe_xs.pkl`: logistic regression
  refit on the `T15_xs` train and validation rows (44,085 + 3,052 = 47,137), saved with scikit-learn 1.9.0.
- `evo2_common.py` — batched embedding; `evo2_scan_restricted.py` — the deployable restricted scanner.
- `C_evo2_ljung_scan.sbatch` / `C_evo2_cau_scan.sbatch` — the two genome-scale runs above.
- `docs/SETUP.md` — building the Evo2 env (note: flash-attn must match the env's torch, cu12/torch2.7/cxx11abiTRUE).

## Recommendation
Consider Evo2 for the C. butyricum target mine — as the lead genome-scale caller, or as a 5th member of
the T29 rank-percentile ensemble (its probability slots in cleanly since rank-percentile is
calibration-invariant; run all members at step 10 so the tracks share a grid). Cost is ~1 GPU-hour per
restricted genome scan — feasible for a one-time mine. See `EVO2_GENOME_SCALE_REPORT.md`.
