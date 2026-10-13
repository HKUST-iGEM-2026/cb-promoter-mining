# Calibration: calib_3

Historical four-model calibration, not the deployed ProkBERT/CNN-v2/Evo2 ensemble. The 279-row mixture has 93 T9 positives and 186 putative negatives (130 convergent/56 CDS). Stored inputs are 49–52-bp positives and 52-bp negatives. Padding does not supply 81-bp biological context.

## Method

StandardScaler→LogisticRegression compares score-only/score+AT, unregularized/L2, with 5×5 stratified OOF selection on log-loss (tol 0.005, L2 C=1). This round chose unregularized score+AT. Lower ECE/Brier/log-loss is better; higher AUROC/AP is better. ± values are repeated-CV standard deviations, not confidence intervals.

## Checkpoint provenance

| Model | chosen calibrator | features | checkpoint / adapter | n_conv / n_cds |
|---|---|---|---|---|
| cnn | score+AT / unregularized | logit, at_fraction | `workflow/results/T21_final/cnn/best_model.pt` | 130 / 56 |
| cnn_v2 | score+AT / unregularized | logit, at_fraction | `workflow/results/T21_final/cnn_v2/best_model.pt` | 130 / 56 |
| nt_lora | score+AT / unregularized | logit, at_fraction | `workflow/results/T21_final/nt_lora/best_adapter` | 130 / 56 |
| prokbert | score+AT / unregularized | logit, at_fraction | `workflow/results/T21_final/prokbert/best_model` | 130 / 56 |


## Metrics

| Metric | cnn raw | cnn_v2 raw | nt_lora raw | prokbert raw | cnn calib | cnn_v2 calib | nt_lora calib | prokbert calib |
|---|---|---|---|---|---|---|---|---|
| ECE | 0.148 | 0.165 | 0.344 | 0.292 | **0.029**  | **0.059**  | **0.044**  | **0.056**  |
| Brier | 0.218 | 0.217 | 0.345 | 0.285 | **0.156**  | **0.134**  | **0.196**  | **0.172**  |
| Log-loss | 0.636 | 0.633 | 1.616 | 1.278 | **0.480**  | **0.421**  | **0.583**  | **0.516**  |
| AUROC | 0.715 | 0.775 | 0.470 | 0.691 | **0.829**  | **0.872**  | **0.712**  | **0.796**  |
| AUPRC | 0.591 | 0.633 | 0.337 | 0.560 | **0.687**  | **0.769**  | **0.562**  | **0.630**  |


## Score-only versus AT

| Model | OOF log-loss score-only | OOF log-loss +AT | AT helps? | AT coef | mean P(positives) score-only -> +AT |
|---|---|---|---|---|---|
| cnn | 0.5679 | 0.4826 | **yes** | +1.127 | 0.427 -> 0.533 |
| cnn_v2 | 0.5283 | 0.4215 | **yes** | +1.351 | 0.472 -> 0.599 |
| nt_lora | 0.6408 | 0.5807 | **yes** | +0.878 | 0.330 -> 0.416 |
| prokbert | 0.5900 | 0.5175 | **yes** | +0.994 | 0.401 -> 0.488 |


## Row-error summary

| Model | mean \|calib-label\| | mean \|raw-label\| | rows made worse | right->wrong flips |
|---|---|---|---|---|
| cnn | 0.310 | 0.398 | 84/279 | 17 |
| cnn_v2 | 0.267 | 0.366 | 84/279 | 15 |
| nt_lora | 0.390 | 0.374 | 171/279  | 26 |
| prokbert | 0.339 | 0.310 | 187/279  | 12 |


## CNN-v1 — source errors

| Source | n | raw err | calib err | Δ |
|---|---|---|---|---|
| CDS_interior | 56 | 0.358 | 0.143 | -0.215 |
| T9_positive | 93 | 0.403 | 0.466 | +0.063 |
| convergent_intergenic | 130 | 0.412 | 0.270 | -0.142 |


## CNN-v1 — largest calibrated errors

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


## CNN-v2 — source errors

| Source | n | raw err | calib err | Δ |
|---|---|---|---|---|
| CDS_interior | 56 | 0.331 | 0.095 | -0.236 |
| T9_positive | 93 | 0.310 | 0.401 | +0.091 |
| convergent_intergenic | 130 | 0.421 | 0.244 | -0.177 |


## CNN-v2 — largest calibrated errors

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


## NT-LoRA — source errors

| Source | n | raw err | calib err | Δ |
|---|---|---|---|---|
| CDS_interior | 56 | 0.180 | 0.233 | +0.053 |
| T9_positive | 93 | 0.864 | 0.584 | -0.280 |
| convergent_intergenic | 130 | 0.108 | 0.318 | +0.210 |


## NT-LoRA — largest calibrated errors

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


## ProkBERT — source errors

| Source | n | raw err | calib err | Δ |
|---|---|---|---|---|
| CDS_interior | 56 | 0.116 | 0.171 | +0.055 |
| T9_positive | 93 | 0.560 | 0.509 | -0.051 |
| convergent_intergenic | 130 | 0.215 | 0.291 | +0.076 |


## ProkBERT — largest calibrated errors

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


## Interpretation

The AT ablation lowers recorded OOF log-loss on this mixture. Aggregate improvements coexist with more rows worsening for some members. Intercept, score coefficient and AT coefficient change jointly; row errors do not isolate a causal AT effect. err is |score−label|; Δ is calibrated error minus raw error, and a flip crosses the wrong side of 0.5.

The sampled prevalence and short context limit transfer. AT-dependent maps can change rank ensembles; the deployed mine uses raw ranks, not these maps. Reused OOF cohorts are sensitivity evidence, not independent final tests. [Calibration notes](../CALIBRATION_NOTES.md) identify scripts and output prefixes.
