# T26 · `calib_2` — Calibration Analysis

Row-level analysis of the AT-content-aware Platt calibration for the three ensemble models (ProkBERT-mini, NT-v2 250M LoRA, 1D CNN). Each raw promoter score is recalibrated with a `StandardScaler → LogisticRegression([logit, AT_fraction])` map, selected by 5×5 stratified CV (unregularized vs L2, tol 0.005).

**Eval set:** 279 sequences — 93 T9 positives, 186 negatives (130 convergent-intergenic + 56 CDS-interior). 81 bp windows. All three models selected the **unregularized** Platt+AT calibrator.

---

## 1. Metrics (one page)

Raw model output vs 5×5 out-of-fold calibrated. Lower is better for ECE / Brier / log-loss; higher for AUROC / AUPRC. Δ is calibrated − raw (the arrow marks whether calibration helped).

| Metric | ProkBERT-mini raw | NT-v2 250M LoRA raw | 1D CNN raw | ProkBERT-mini calib | NT-v2 250M LoRA calib | 1D CNN calib |
|---|---|---|---|---|---|---|
| ECE | 0.248 | 0.338 | 0.201 | **0.059** ±0.004 ▲ | **0.052** ±0.007 ▲ | **0.048** ±0.010 ▲ |
| Brier | 0.252 | 0.339 | 0.219 | **0.159** ±0.001 ▲ | **0.191** ±0.001 ▲ | **0.146** ±0.001 ▲ |
| Log-loss | 1.054 | 1.413 | 0.635 | **0.483** ±0.005 ▲ | **0.563** ±0.004 ▲ | **0.447** ±0.005 ▲ |
| AUROC | 0.790 | 0.563 | 0.785 | **0.827** ±0.004 ▲ | **0.731** ±0.005 ▲ | **0.853** ±0.003 ▲ |
| AUPRC | 0.647 | 0.376 | 0.658 | **0.704** ±0.008 ▲ | **0.556** ±0.006 ▲ | **0.726** ±0.006 ▲ |

**Read-out.** Calibration does its job on *probability quality* — ECE collapses everywhere (ProkBERT 0.248→0.059, NT 0.338→0.052, CNN 0.201→0.048), and Brier / log-loss fall with it. Rank metrics (AUROC/AUPRC) barely move because a monotone Platt map can't reorder scores, so the gain is *trustworthy probabilities, not better discrimination*. NT stays the weakest discriminator (CV AUROC 0.73 vs ProkBERT 0.83, CNN 0.85).

---

## 2. Where the calibrator messed up

Ranking all 279 rows by calibrated error `|calib − label|` exposes a consistent failure mode: the AT covariate learns *high-AT ⇒ negative* from the AT-rich convergent-intergenic negatives, then **punishes the genuinely AT-rich T9 promoters** — the exact AT-trap this project exists to avoid.

### Summary

| Model | mean \|calib−label\| | mean \|raw−label\| | rows made worse | right→wrong flips |
|---|---|---|---|---|
| ProkBERT-mini | 0.311 | 0.280 | 195/279 ⚠️ | 14 |
| NT-v2 250M LoRA | 0.378 | 0.378 | 170/279 ⚠️ | 30 |
| 1D CNN | 0.286 | 0.391 | 85/279 | 19 |

*rows made worse* = calibrated error exceeds raw error. *flips* = raw put it on the correct side of 0.5 but the calibrator pushed it to the wrong side. For **ProkBERT and NT the calibrator degrades the majority of rows** (195/279, 170/279) even while fixing global ECE; **CNN is the exception** — its raw scores were badly miscalibrated (mean raw error 0.391) so calibration is a net win there (only 85/279 worsened).

### ProkBERT-mini

**Calibration error by class:**

| Source | n | raw err | calib err | Δ |
|---|---|---|---|---|
| T9 positive | 93 | 0.726 | 0.465 | -0.260 |
| convergent intergenic | 130 | 0.061 | 0.265 | +0.204 |
| CDS interior | 56 | 0.047 | 0.159 | +0.113 |

**Worst 8 rows** (highest `|calib − label|`):

| label | source | AT | raw | calib | err | Δ vs raw |
|---|---|---|---|---|---|---|
| NEG | convergent intergenic | 0.77 | 0.951 | 0.963 | **0.963** | +0.012 |
| POS | T9 positive | 0.61 | 0.009 | 0.051 | **0.949** | -0.043 |
| NEG | convergent intergenic | 0.65 | 0.992 | 0.940 | **0.940** | -0.052 |
| NEG | CDS interior | 0.65 | 0.981 | 0.898 | **0.898** | -0.083 |
| POS | T9 positive | 0.60 | 0.042 | 0.103 | **0.897** | -0.061 |
| NEG | convergent intergenic | 0.88 | 0.071 | 0.897 | **0.897** | +0.825 |
| POS | T9 positive | 0.67 | 0.008 | 0.112 | **0.888** | -0.104 |
| NEG | CDS interior | 0.65 | 0.974 | 0.868 | **0.868** | -0.106 |

### NT-v2 250M LoRA

**Calibration error by class:**

| Source | n | raw err | calib err | Δ |
|---|---|---|---|---|
| T9 positive | 93 | 0.849 | 0.565 | -0.284 |
| convergent intergenic | 130 | 0.115 | 0.307 | +0.192 |
| CDS interior | 56 | 0.206 | 0.229 | +0.023 |

**Worst 8 rows** (highest `|calib − label|`):

| label | source | AT | raw | calib | err | Δ vs raw |
|---|---|---|---|---|---|---|
| POS | T9 positive | 0.47 | 0.987 | 0.065 | **0.935** | +0.922 |
| POS | T9 positive | 0.61 | 0.008 | 0.068 | **0.932** | -0.061 |
| POS | T9 positive | 0.58 | 0.142 | 0.083 | **0.917** | +0.059 |
| POS | T9 positive | 0.51 | 0.985 | 0.119 | **0.881** | +0.867 |
| NEG | convergent intergenic | 0.88 | 0.160 | 0.875 | **0.875** | +0.714 |
| POS | T9 positive | 0.53 | 0.994 | 0.154 | **0.846** | +0.840 |
| POS | T9 positive | 0.59 | 0.679 | 0.160 | **0.840** | +0.520 |
| POS | T9 positive | 0.60 | 0.384 | 0.168 | **0.832** | +0.215 |

### 1D CNN

**Calibration error by class:**

| Source | n | raw err | calib err | Δ |
|---|---|---|---|---|
| T9 positive | 93 | 0.285 | 0.430 | +0.145 |
| convergent intergenic | 130 | 0.472 | 0.254 | -0.218 |
| CDS interior | 56 | 0.378 | 0.121 | -0.257 |

**Worst 8 rows** (highest `|calib − label|`):

| label | source | AT | raw | calib | err | Δ vs raw |
|---|---|---|---|---|---|---|
| POS | T9 positive | 0.69 | 0.234 | 0.057 | **0.943** | +0.177 |
| NEG | convergent intergenic | 0.77 | 0.962 | 0.939 | **0.939** | -0.023 |
| POS | T9 positive | 0.67 | 0.295 | 0.062 | **0.938** | +0.233 |
| NEG | convergent intergenic | 0.87 | 0.835 | 0.918 | **0.918** | +0.083 |
| POS | T9 positive | 0.63 | 0.478 | 0.086 | **0.914** | +0.392 |
| POS | T9 positive | 0.61 | 0.601 | 0.089 | **0.911** | +0.512 |
| POS | T9 positive | 0.58 | 0.728 | 0.096 | **0.904** | +0.632 |
| NEG | convergent intergenic | 0.83 | 0.896 | 0.903 | **0.903** | +0.006 |

---

## 3. Takeaway

- The worst rows are overwhelmingly **AT-rich T9 positives driven toward 0** by the AT term, plus a handful of high-AT convergent negatives left high. The AT covariate is double-counting: it treats AT-richness as evidence *against* being a promoter, but real Clostridium promoters are AT-rich.

- **ProkBERT / NT:** the Platt+AT map trades per-row correctness for global ECE — good for reported calibration, bad for recall on genuine promoters. Consider dropping or down-weighting the AT feature, or fitting on a class-balanced / AT-matched negative set so AT stops proxying the label.

- **CNN:** raw output was the most miscalibrated, so here the calibrator is clearly beneficial (net error down, 85/279 worsened, best final ECE 0.048).

- Because ensemble aggregation (T29) uses **rank-percentile**, it is invariant to any monotone calibration — these AT-driven distortions matter for the *per-model probability outputs and any probability threshold*, not for the rank-based ensemble.

---
<sub>Generated from `calib_2/<model>/{model}_calibration_metrics.json` + `_predictions.csv`. `err = |probability − label|`; `Δ vs raw = calib_err − raw_err` (positive ⇒ calibration made the row worse). Calibrator selection: OOF log-loss, tol 0.005, L2 C=1.0.</sub>