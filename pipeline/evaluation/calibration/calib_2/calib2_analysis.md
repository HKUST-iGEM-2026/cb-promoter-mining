# Calibration: calib_2

Historical three-model calibration, not the deployed ProkBERT/CNN-v2/Evo2 ensemble. The 279-row mixture has 93 T9 positives and 186 putative negatives (130 convergent/56 CDS). Stored inputs are 49–52-bp positives and 52-bp negatives. Padding does not supply 81-bp biological context.

## Method

StandardScaler→LogisticRegression compares score-only/score+AT, unregularized/L2, with 5×5 stratified OOF selection on log-loss (tol 0.005, L2 C=1). This round chose unregularized score+AT. Lower ECE/Brier/log-loss is better; higher AUROC/AP is better. ± values are repeated-CV standard deviations, not confidence intervals.

## Metrics

| Metric | ProkBERT-mini raw | NT-v2 250M LoRA raw | 1D CNN raw | ProkBERT-mini calib | NT-v2 250M LoRA calib | 1D CNN calib |
|---|---|---|---|---|---|---|
| ECE | 0.248 | 0.338 | 0.201 | **0.059** ±0.004  | **0.052** ±0.007  | **0.048** ±0.010  |
| Brier | 0.252 | 0.339 | 0.219 | **0.159** ±0.001  | **0.191** ±0.001  | **0.146** ±0.001  |
| Log-loss | 1.054 | 1.413 | 0.635 | **0.483** ±0.005  | **0.563** ±0.004  | **0.447** ±0.005  |
| AUROC | 0.790 | 0.563 | 0.785 | **0.827** ±0.004  | **0.731** ±0.005  | **0.853** ±0.003  |
| AUPRC | 0.647 | 0.376 | 0.658 | **0.704** ±0.008  | **0.556** ±0.006  | **0.726** ±0.006  |


## Row-error summary

| Model | mean \|calib−label\| | mean \|raw−label\| | rows made worse | right→wrong flips |
|---|---|---|---|---|
| ProkBERT-mini | 0.311 | 0.280 | 195/279  | 14 |
| NT-v2 250M LoRA | 0.378 | 0.378 | 170/279  | 30 |
| 1D CNN | 0.286 | 0.391 | 85/279 | 19 |


## ProkBERT-mini — source errors

| Source | n | raw err | calib err | Δ |
|---|---|---|---|---|
| T9 positive | 93 | 0.726 | 0.465 | -0.260 |
| convergent intergenic | 130 | 0.061 | 0.265 | +0.204 |
| CDS interior | 56 | 0.047 | 0.159 | +0.113 |


## ProkBERT-mini — largest calibrated errors

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


## NT-v2 LoRA — source errors

| Source | n | raw err | calib err | Δ |
|---|---|---|---|---|
| T9 positive | 93 | 0.849 | 0.565 | -0.284 |
| convergent intergenic | 130 | 0.115 | 0.307 | +0.192 |
| CDS interior | 56 | 0.206 | 0.229 | +0.023 |


## NT-v2 LoRA — largest calibrated errors

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


## CNN-v1 — source errors

| Source | n | raw err | calib err | Δ |
|---|---|---|---|---|
| T9 positive | 93 | 0.285 | 0.430 | +0.145 |
| convergent intergenic | 130 | 0.472 | 0.254 | -0.218 |
| CDS interior | 56 | 0.378 | 0.121 | -0.257 |


## CNN-v1 — largest calibrated errors

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


## Interpretation

ECE/log-loss improve while many ProkBERT/NT rows worsen in absolute error. CNN has fewer worsened rows. Intercept, score coefficient and AT coefficient change jointly; row errors do not isolate a causal AT effect. err is |score−label|; Δ is calibrated error minus raw error, and a flip crosses the wrong side of 0.5.

The sampled prevalence and short context limit transfer. AT-dependent maps can change rank ensembles; the deployed mine uses raw ranks, not these maps. Reused OOF cohorts are sensitivity evidence, not independent final tests. [Calibration notes](../CALIBRATION_NOTES.md) identify scripts and output prefixes.
