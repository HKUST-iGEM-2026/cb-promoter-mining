# T26 — Calibration Notes

A guide to the calibration scripts, the rounds they were run on, and what each writes.

## What T26 does

Each model outputs a raw promoter score. T26 fits a **Platt calibrator with an AT-content
covariate** (`Platt+AT`) so scores become probabilities that account for the AT-rich false-positive
trap. Each script picks **unregularized vs L2** Platt+AT by stratified K-fold CV (OOF log-loss + Brier;
simpler wins unless L2 beats by `--tol` = 0.005, `--l2-C` = 1.0), then **refits the chosen calibrator on
the full set**. The mine does not use calibration: its mean rank-percentile ensemble is
calibration-invariant.

**Calibration data (shared by all scripts):** 93 T9 positives
(`data-acquisition-and-curation/synthetic-promoters/literature_promoters.tsv`) + 186 curated genomic
negatives = 130 convergent-intergenic + 56 CDS-interior, sampled from the output of
`extract_negatives.py` (see `NEGATIVES_NOTES.md`).

## The four scripts

| Script | Model | Env |
|---|---|---|
| `prokbert_calib.py` | ProkBERT-mini | `$ENV_MAIN` (transformers 5.3.0) |
| `cnn_calib.py` | CNN-v1 | `$ENV_MAIN` |
| `cnn_v2_calib.py` | CNN-v2 (fusion) | `$ENV_MAIN` |
| `nt_calib.py` | NT v2 250M LoRA | `$ENV_NT` (transformers 4.57) |

All take the same flags (`--model-dir`, `--positives`, `--convergent-negatives`, `--cds-negatives`,
`--n-convergent` 130, `--n-cds` 56, `--out-dir`, `--cv-folds` 5, `--cv-repeats` 5, `--l2-C`, `--tol`,
`--seed` 42).

## The four rounds

Each driver is one GPU job that runs the main-env models, then switches to `$ENV_NT` for NT.

| Driver | Checkpoints | Output folder | Committed |
|---|---|---|---|
| `run_calibration_2.sh` | 0.9-sweep trial checkpoints (`results/T21/<model>/trial_*`) | `calib_2/` | calibrators, metrics, predictions, report |
| `run_calibration_3.sh` | `results/T21_final/` | `calib_3/` | the same |
| `run_calibration_06.sh` | `results/T21_final_06/` | `calib_06/` | the same |
| `run_calibration_xs.sh` | `results/T21_xs/`, the **deployed** set | `calib_xs/` | reports only |

`run_calibration_2.sh`, `_3.sh` and `_06.sh` read the negatives from this folder;
`run_calibration_xs.sh` reads them from `negative_generation/`. Rebuild them with the commands in
`docs/PIPELINE.md` (Phase D, *Calibration*).

## What each model writes

In `<out-dir>/<model>/`:

- **`<model>_calibrator_at.joblib`** — the fitted calibrator (StandardScaler → LogisticRegression
  pipeline, refit on the full set). `evaluation-and-calibration/apply_calibrator.py` applies it to a
  genome score track.
- **`<model>_calibration_metrics.json`** — sample counts, CV settings, `chosen` (unregularized vs L2) +
  `selection_reason`, seed, calibration metrics (log-loss, Brier, ECE), and the input/output `paths`.
- **`<model>_calibration_predictions.csv`** — one row per sample with `sequence`, `label`, `source`,
  `at_fraction`, `raw_logit`, `raw_prob`, and out-of-fold `calib_prob_at`.

## Reports

- `calib_*/calibration_analysis.md` (+ `.tex`) — per-round consolidated report, from `analyze_calibration.py`.
- `calib_06/regimeC_calibrated_analysis_06.md`, `calib_xs/regimeC_calibrated_analysis_xs.md` — raw vs
  calibrated genome-scale benchmarks on *C. ljungdahlii* at step 1, from `C_ljung_scan_06.sh` /
  `C_ljung_scan_xs.sh` and `analyze_regimeC.py`.
- `CALIBRATION_ANALYSIS_06_vs_3.md` — the `calib_06` vs `calib_3` comparison.

## Other files in this folder

- `extract_negatives.py` → `convergent_intergenic_negatives.tsv` + `cds_negatives.tsv` (not committed).
- `C_acetobutylicum.gff3`, `C_phytofermentans.gff3` — the two source annotations, genome sequences
  embedded after `##FASTA`.
- `region_counts.tsv` — convergent/CDS region + window counts per source genome (CACE, CPHY).
- `NEGATIVES_NOTES.md` — how the genomic negatives are defined.
