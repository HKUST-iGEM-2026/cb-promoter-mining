# Optional calibration

T26 compares score-only and score+AT Platt maps, each unregularized or L2, using StandardScaler→LogisticRegression and 5×5 stratified CV. Candidates are considered in that order; selection requires mean per-repeat OOF log-loss improvement greater than tol=0.005, with L2 C=1.0. Brier/ECE are reported, not tie-breaks. The selected map is refit on the full cohort; calib_prob_at averages held-out predictions over repeats.

## Cohort and scope

The shared mixture has 93 T9 positives and 186 putative genomic negatives: 130 convergent/56 CDS. Stored calib_2/3/06 CSVs contain 49–52-bp positives and 52-bp negatives. Padding does not recreate 81-bp context. calib_xs predictions/maps are absent; its header is not input-length evidence. Repeated rounds reuse one mixture, not independent validation.

All recorded rounds chose unregularized score+AT. Its AT feature can reorder windows; only strictly increasing score-only transformations preserve ranks/ties. **The deployed trio uses raw ranks and no calibrator.** These sampled-mixture outputs are not target-genome promoter probabilities.

## Scripts

| Script | Model | Env |
|---|---|---|
| `prokbert_calib.py` | ProkBERT-mini | `$ENV_MAIN` (transformers 5.3.0) |
| `cnn_calib.py` | CNN-v1 | `$ENV_MAIN` |
| `cnn_v2_calib.py` | CNN-v2 (fusion) | `$ENV_MAIN` |
| `nt_calib.py` | NT v2 250M LoRA | `$ENV_NT` (transformers 4.57) |


Common flags: `--model-dir`, `--positives`, `--convergent-negatives`, `--cds-negatives`, `--n-convergent` 130, `--n-cds` 56, `--out-dir`, `--cv-folds` 5, `--cv-repeats` 5, `--l2-C`, `--tol` and `--seed` 42.

## Rounds

| Driver | Checkpoints | Output folder | Committed |
|---|---|---|---|
| `run_calibration_2.sh` | 0.9-sweep trial checkpoints (`results/T21/<model>/trial_*`) | `calib_2/` | three-model calibrators, metrics, predictions, report |
| `run_calibration_3.sh` | `results/T21_final/` | `calib_3/` | four-model calibrators, metrics, predictions, report |
| `run_calibration_06.sh` | `results/T21_final_06/` | `calib_06/` | four-model calibrators, metrics, predictions, report |
| `run_calibration_xs.sh` | `results/T21_xs/`, historical xs calibration round (four models; not the deployed three-member ensemble) | `calib_xs/` | reports only |


Drivers switch from ENV_MAIN to ENV_NT for the historical NT member. The xs driver reads negatives under negative_generation; others read this directory. [Pipeline recipe](../../../docs/PIPELINE.md#calibration-optional-the-mine-does-not-use-it) gives extraction/scoring. Matching calibrator/checkpoint/cohort is required; older maps are not replacements.

## Outputs

Each model directory holds prefix_calibrator_at.joblib, prefix_calibration_metrics.json and prefix_calibration_predictions.csv. Prefixes are prokbert/cnn/cnn_v2/**nt**, even in directory nt_lora. Saved feature count identifies score-only versus score+AT. Metrics record selection/CV/settings/paths; predictions retain DNA, label, source, AT fraction, raw logit/score and OOF output. [apply_calibrator.py](../apply_calibrator.py) applies a trusted map to a compatible score track.

analyze_calibration.py writes consolidated Markdown/TeX. analyze_regimeC.py compares historical step-1 raw/calibrated tracks in full and restricted scope; these are not the step-10 trio benchmark. The negative annotations embed FASTA after ##FASTA; [negative notes](NEGATIVES_NOTES.md) define masks and boundaries.
