# T26 — Calibration Notes

Brief guide to the three calibration scripts and what they write on HPC4.

## What T26 does

Each ensemble model outputs a raw promoter score. T26 fits a **Platt calibrator with an AT-content
covariate** (`Platt+AT`) so scores become probabilities that account for the AT-rich false-positive
trap. Each script picks **unregularized vs L2** Platt+AT by stratified K-fold CV (OOF log-loss + Brier;
simpler wins unless L2 beats by `--tol` = 0.005, `--l2-C` = 1.0), then **refits the chosen calibrator on
the full set**.

**Calibration data (shared by all three):** 93 T9 positives (`literature_promoters.tsv`) + 186 curated
genomic negatives = 130 convergent-intergenic + 56 CDS-interior (from `extract_negatives.py`; see
`NEGATIVES_NOTES.md`).

## The three scripts

| Script | Model | Env |
|---|---|---|
| `prokbert_calib.py` | ProkBERT-mini (T18) | `promoter-mining-hpc4` (transformers 5.3.0) |
| `cnn_calib.py` | 1D CNN (T20) | `promoter-mining-hpc4` |
| `nt_calib.py` | NT v2 250M LoRA (T19) | `nt-tf4x` (transformers 4.57) |

All three take the same flags (`--positives`, `--convergent-negatives`, `--cds-negatives`,
`--n-convergent` 130, `--n-cds` 56, `--out-dir`, `--cv-folds` 5, `--cv-repeats` 5, `--l2-C`, `--tol`,
`--seed` 42). Driver: `run_calibration.sh` (one GPU job — runs prokbert + cnn on the main env, switches
to `nt-tf4x` for nt).

## Outputs stored on HPC4

Written to `--out-dir` (`run_calibration.sh` uses `.../T26/calib_2/{prokbert,cnn,nt}/`). Each model
writes **3 files**, prefixed by model name (`<model>` = `prokbert` / `cnn` / `nt`):

- **`<model>_calibrator_at.joblib`** — the fitted calibrator artifact (StandardScaler → LogisticRegression
  pipeline, refit on the full set). This is what mining/T32 loads to calibrate raw scores.
- **`<model>_calibration_metrics.json`** — sample counts (n_total / n_positive / n_negative / n_convergent /
  n_cds), CV settings, `chosen` (unregularized vs L2) + `selection_reason`, seed, and calibration metrics
  (log-loss, Brier, ECE), plus absolute input/output `paths`.
- **`<model>_calibration_predictions.csv`** — one row per sample with `sequence`, `label`, `source`,
  `at_fraction`, `raw_logit`, `raw_prob`, and out-of-fold `calib_prob_at`.

## Other files in this dir

- `extract_negatives.py` → `convergent_intergenic_negatives.tsv` + `cds_negatives.tsv` (the negatives).
- `region_counts.tsv` — bookkeeping: convergent/CDS region + window counts per source genome (CACE, CPHY).
- `NEGATIVES_NOTES.md` — how the genomic negatives are defined.

**Status:** scripts wired but not yet run on HPC4 — needs the trained T18/T19/T20 models + the two
negative TSVs staged.
