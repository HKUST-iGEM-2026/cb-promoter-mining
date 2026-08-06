# T26 — Calibration Analysis (all models)

Consolidated from `/project/igem2026/cb-promoter-mining/evaluation-and-calibration/T26/calib_3` — 4 model(s): cnn, cnn_v2, nt_lora, prokbert. Each model's raw promoter score is recalibrated with an AT-content-aware Platt regressor (`StandardScaler -> LogisticRegression([logit, AT])`), selected by 5x5 stratified CV.

**Eval set:** 279 sequences — 93 positives, 186 negatives (130 convergent-intergenic + 56 CDS-interior). Window 81 bp.

---
## 1. Provenance — which checkpoint each calibrator was fit on

| Model | chosen calibrator | features | checkpoint / adapter | n_conv / n_cds |
|---|---|---|---|---|
| cnn | score+AT / unregularized | logit, at_fraction | `/project/igem2026/cb-promoter-mining/infrastructure/T3/snakemake_scaffold/results/T21_final/cnn/best_model.pt` | 130 / 56 |
| cnn_v2 | score+AT / unregularized | logit, at_fraction | `/project/igem2026/cb-promoter-mining/infrastructure/T3/snakemake_scaffold/results/T21_final/cnn_v2/best_model.pt` | 130 / 56 |
| nt_lora | score+AT / unregularized | logit, at_fraction | `/project/igem2026/cb-promoter-mining/infrastructure/T3/snakemake_scaffold/results/T21_final/nt_lora/best_adapter` | 130 / 56 |
| prokbert | score+AT / unregularized | logit, at_fraction | `/project/igem2026/cb-promoter-mining/infrastructure/T3/snakemake_scaffold/results/T21_final/prokbert/best_model` | 130 / 56 |

> If any checkpoint path is under `results/T21/<model>/trial_*` (sweep trials) while the deployed/reported model is `T21_final_06/<model>`, that calibrator is fit on the wrong checkpoint and should be re-run with the matching `--model-dir`/`--trial-dir`/`--adapter-dir`.

---
## 2. Metrics — raw vs OOF-calibrated

Calibrated = `calibrated_logit_at_oof` (the out-of-fold deployed behaviour). Lower is better for ECE / Brier / log-loss; higher for AUROC / AUPRC. Arrow = calibration helped.

| Metric | cnn raw | cnn_v2 raw | nt_lora raw | prokbert raw | cnn calib | cnn_v2 calib | nt_lora calib | prokbert calib |
|---|---|---|---|---|---|---|---|---|
| ECE | 0.148 | 0.165 | 0.344 | 0.292 | **0.029** ▲ | **0.059** ▲ | **0.044** ▲ | **0.056** ▲ |
| Brier | 0.218 | 0.217 | 0.345 | 0.285 | **0.156** ▲ | **0.134** ▲ | **0.196** ▲ | **0.172** ▲ |
| Log-loss | 0.636 | 0.633 | 1.616 | 1.278 | **0.480** ▲ | **0.421** ▲ | **0.583** ▲ | **0.516** ▲ |
| AUROC | 0.715 | 0.775 | 0.470 | 0.691 | **0.829** ▲ | **0.872** ▲ | **0.712** ▲ | **0.796** ▲ |
| AUPRC | 0.591 | 0.633 | 0.337 | 0.560 | **0.687** ▲ | **0.769** ▲ | **0.562** ▲ | **0.630** ▲ |

---
## 3. Does the AT covariate earn its place? (score-only vs score+AT)

Recomputed from each CSV with a repeated-stratified-KFold logistic. `+AT` is worth keeping only if it lowers OOF log-loss; a **positive AT coef** means high-AT sequences are pushed *up* (AT-rich promoters rescued), negative means pushed *down*.

| Model | OOF log-loss score-only | OOF log-loss +AT | AT helps? | AT coef | mean P(positives) score-only -> +AT |
|---|---|---|---|---|---|
| cnn | 0.5679 | 0.4826 | **yes** | +1.127 | 0.427 -> 0.533 |
| cnn_v2 | 0.5283 | 0.4215 | **yes** | +1.351 | 0.472 -> 0.599 |
| nt_lora | 0.6408 | 0.5807 | **yes** | +0.878 | 0.330 -> 0.416 |
| prokbert | 0.5900 | 0.5175 | **yes** | +0.994 | 0.401 -> 0.488 |

*Decision rule for the calibrator: keep +AT only where it lowers OOF log-loss; otherwise score-only (simpler) is the honest choice.*

---
## 4. Where the calibrator messed up

| Model | mean \|calib-label\| | mean \|raw-label\| | rows made worse | right->wrong flips |
|---|---|---|---|---|
| cnn | 0.310 | 0.398 | 84/279 | 17 |
| cnn_v2 | 0.267 | 0.366 | 84/279 | 15 |
| nt_lora | 0.390 | 0.374 | 171/279 ⚠️ | 26 |
| prokbert | 0.339 | 0.310 | 187/279 ⚠️ | 12 |

### cnn

**Calibration error by source class:**

| Source | n | raw err | calib err | Δ |
|---|---|---|---|---|
| CDS_interior | 56 | 0.358 | 0.143 | -0.215 |
| T9_positive | 93 | 0.403 | 0.466 | +0.063 |
| convergent_intergenic | 130 | 0.412 | 0.270 | -0.142 |

**Worst 8 rows** (highest `|calib - label|`):

| label | source | AT | raw | calib | err | Δ vs raw |
|---|---|---|---|---|---|---|
| NEG | convergent_intergenic | 0.77 | 0.983 | 0.974 | **0.974** | -0.009 |
| POS | T9_positive | 0.58 | 0.312 | 0.029 | **0.971** | +0.283 |
| POS | T9_positive | 0.67 | 0.154 | 0.049 | **0.951** | +0.105 |
| POS | T9_positive | 0.61 | 0.385 | 0.058 | **0.942** | +0.327 |
| NEG | convergent_intergenic | 0.77 | 0.945 | 0.917 | **0.917** | -0.028 |
| POS | T9_positive | 0.75 | 0.131 | 0.124 | **0.876** | +0.008 |
| NEG | convergent_intergenic | 0.83 | 0.785 | 0.868 | **0.868** | +0.083 |
| POS | T9_positive | 0.73 | 0.210 | 0.150 | **0.850** | +0.060 |

### cnn_v2

**Calibration error by source class:**

| Source | n | raw err | calib err | Δ |
|---|---|---|---|---|
| CDS_interior | 56 | 0.331 | 0.095 | -0.236 |
| T9_positive | 93 | 0.310 | 0.401 | +0.091 |
| convergent_intergenic | 130 | 0.421 | 0.244 | -0.177 |

**Worst 8 rows** (highest `|calib - label|`):

| label | source | AT | raw | calib | err | Δ vs raw |
|---|---|---|---|---|---|---|
| POS | T9_positive | 0.67 | 0.129 | 0.025 | **0.975** | +0.104 |
| POS | T9_positive | 0.69 | 0.117 | 0.025 | **0.975** | +0.093 |
| POS | T9_positive | 0.73 | 0.080 | 0.036 | **0.964** | +0.044 |
| NEG | convergent_intergenic | 0.77 | 0.947 | 0.948 | **0.948** | +0.001 |
| NEG | convergent_intergenic | 0.87 | 0.672 | 0.918 | **0.918** | +0.246 |
| NEG | convergent_intergenic | 0.88 | 0.570 | 0.908 | **0.908** | +0.338 |
| NEG | convergent_intergenic | 0.75 | 0.930 | 0.889 | **0.889** | -0.040 |
| POS | T9_positive | 0.75 | 0.203 | 0.145 | **0.855** | +0.058 |

### nt_lora

**Calibration error by source class:**

| Source | n | raw err | calib err | Δ |
|---|---|---|---|---|
| CDS_interior | 56 | 0.180 | 0.233 | +0.053 |
| T9_positive | 93 | 0.864 | 0.584 | -0.280 |
| convergent_intergenic | 130 | 0.108 | 0.318 | +0.210 |

**Worst 8 rows** (highest `|calib - label|`):

| label | source | AT | raw | calib | err | Δ vs raw |
|---|---|---|---|---|---|---|
| POS | T9_positive | 0.47 | 0.977 | 0.036 | **0.964** | +0.942 |
| POS | T9_positive | 0.53 | 0.393 | 0.064 | **0.936** | +0.329 |
| POS | T9_positive | 0.51 | 0.991 | 0.069 | **0.931** | +0.922 |
| POS | T9_positive | 0.59 | 0.016 | 0.108 | **0.892** | -0.092 |
| POS | T9_positive | 0.58 | 0.821 | 0.110 | **0.890** | +0.711 |
| POS | T9_positive | 0.61 | 0.027 | 0.116 | **0.884** | -0.089 |
| NEG | convergent_intergenic | 0.88 | 0.974 | 0.875 | **0.875** | -0.098 |
| POS | T9_positive | 0.60 | 0.034 | 0.127 | **0.873** | -0.093 |

### prokbert

**Calibration error by source class:**

| Source | n | raw err | calib err | Δ |
|---|---|---|---|---|
| CDS_interior | 56 | 0.116 | 0.171 | +0.055 |
| T9_positive | 93 | 0.560 | 0.509 | -0.051 |
| convergent_intergenic | 130 | 0.215 | 0.291 | +0.076 |

**Worst 8 rows** (highest `|calib - label|`):

| label | source | AT | raw | calib | err | Δ vs raw |
|---|---|---|---|---|---|---|
| NEG | convergent_intergenic | 0.88 | 0.976 | 0.946 | **0.946** | -0.030 |
| POS | T9_positive | 0.61 | 0.007 | 0.058 | **0.942** | -0.051 |
| POS | T9_positive | 0.47 | 0.996 | 0.080 | **0.920** | +0.916 |
| NEG | convergent_intergenic | 0.83 | 0.993 | 0.906 | **0.906** | -0.087 |
| POS | T9_positive | 0.67 | 0.005 | 0.121 | **0.879** | -0.116 |
| POS | T9_positive | 0.69 | 0.006 | 0.149 | **0.851** | -0.143 |
| POS | T9_positive | 0.69 | 0.006 | 0.153 | **0.847** | -0.147 |
| POS | T9_positive | 0.51 | 0.996 | 0.154 | **0.846** | +0.842 |

---
## 5. Takeaways

- AT **helps** (lower OOF log-loss, positive coef = rescues AT-rich positives) for: cnn, cnn_v2, nt_lora, prokbert.
- The calibrator made the majority of rows worse than raw for: nt_lora, prokbert — that is the base-rate/intercept level-shift, not necessarily the AT term (see §3).
- Ensemble aggregation (T29) is rank-percentile and calibration-invariant, so these distortions matter only for per-model probabilities / fixed thresholds, not the ensemble ranking.

---
<sub>`err = |probability - label|`; `Δ vs raw = calib_err - raw_err` (positive => calibration made the row worse). §3 ablation is recomputed from raw_logit + at_fraction + label with a stdlib logistic (5x5 stratified OOF).</sub>