# Calibration: calib_06

Historical four-model calibration, not the deployed ProkBERT/CNN-v2/Evo2 ensemble. The 279-row mixture has 93 T9 positives and 186 putative negatives (130 convergent/56 CDS). Stored inputs are 49–52-bp positives and 52-bp negatives. Padding does not supply 81-bp biological context.

## Method

StandardScaler→LogisticRegression compares score-only/score+AT, unregularized/L2, with 5×5 stratified OOF selection on log-loss (tol 0.005, L2 C=1). This round chose unregularized score+AT. Lower ECE/Brier/log-loss is better; higher AUROC/AP is better. ± values are repeated-CV standard deviations, not confidence intervals.

## Checkpoint provenance

| Model | chosen calibrator | features | checkpoint / adapter | n_conv / n_cds |
|---|---|---|---|---|
| cnn | score+AT / unregularized | logit, at_fraction | `workflow/results/T21_final_06/cnn/best_model.pt` | 130 / 56 |
| cnn_v2 | score+AT / unregularized | logit, at_fraction | `workflow/results/T21_final_06/cnn_v2/best_model.pt` | 130 / 56 |
| nt_lora | score+AT / unregularized | logit, at_fraction | `workflow/results/T21_final_06/nt_lora/best_adapter` | 130 / 56 |
| prokbert | score+AT / unregularized | logit, at_fraction | `workflow/results/T21_final_06/prokbert/best_model` | 130 / 56 |


## Metrics

| Metric | cnn raw | cnn_v2 raw | nt_lora raw | prokbert raw | cnn calib | cnn_v2 calib | nt_lora calib | prokbert calib |
|---|---|---|---|---|---|---|---|---|
| ECE | 0.180 | 0.196 | 0.302 | 0.262 | **0.064**  | **0.081**  | **0.063**  | **0.065**  |
| Brier | 0.236 | 0.238 | 0.302 | 0.265 | **0.154**  | **0.154**  | **0.180**  | **0.171**  |
| Log-loss | 0.699 | 0.732 | 1.520 | 1.212 | **0.469**  | **0.472**  | **0.536**  | **0.517**  |
| AUROC | 0.750 | 0.720 | 0.683 | 0.708 | **0.833**  | **0.834**  | **0.769**  | **0.799**  |
| AUPRC | 0.596 | 0.586 | 0.504 | 0.572 | **0.701**  | **0.692**  | **0.617**  | **0.643**  |


## Score-only versus AT

| Model | OOF log-loss score-only | OOF log-loss +AT | AT helps? | AT coef | mean P(positives) score-only -> +AT |
|---|---|---|---|---|---|
| cnn | 0.5490 | 0.4678 | **yes** | +1.076 | 0.445 -> 0.543 |
| cnn_v2 | 0.5736 | 0.4752 | **yes** | +1.230 | 0.422 -> 0.541 |
| nt_lora | 0.6093 | 0.5355 | **yes** | +0.994 | 0.374 -> 0.469 |
| prokbert | 0.5819 | 0.5169 | **yes** | +0.959 | 0.411 -> 0.490 |


## Row-error summary

| Model | mean \|calib-label\| | mean \|raw-label\| | rows made worse | right->wrong flips |
|---|---|---|---|---|
| cnn | 0.304 | 0.379 | 102/279 | 18 |
| cnn_v2 | 0.304 | 0.380 | 110/279 | 22 |
| nt_lora | 0.355 | 0.313 | 192/279  | 17 |
| prokbert | 0.338 | 0.288 | 197/279  | 14 |


## CNN-v1 — source errors

| Source | n | raw err | calib err | Δ |
|---|---|---|---|---|
| CDS_interior | 56 | 0.345 | 0.125 | -0.219 |
| T9_positive | 93 | 0.312 | 0.456 | +0.144 |
| convergent_intergenic | 130 | 0.442 | 0.272 | -0.170 |


## CNN-v1 — largest calibrated errors

| label | source | AT | raw | calib | err | Δ vs raw |
|---|---|---|---|---|---|---|
| POS | T9_positive | 0.69 | 0.082 | 0.040 | **0.960** | +0.043 |
| POS | T9_positive | 0.58 | 0.514 | 0.045 | **0.955** | +0.469 |
| POS | T9_positive | 0.73 | 0.092 | 0.075 | **0.925** | +0.016 |
| POS | T9_positive | 0.61 | 0.558 | 0.080 | **0.920** | +0.478 |
| NEG | convergent_intergenic | 0.77 | 0.966 | 0.899 | **0.899** | -0.067 |
| NEG | convergent_intergenic | 0.83 | 0.913 | 0.889 | **0.889** | -0.025 |
| NEG | convergent_intergenic | 0.87 | 0.772 | 0.883 | **0.883** | +0.111 |
| POS | T9_positive | 0.77 | 0.092 | 0.142 | **0.858** | -0.051 |


## CNN-v2 — source errors

| Source | n | raw err | calib err | Δ |
|---|---|---|---|---|
| CDS_interior | 56 | 0.356 | 0.132 | -0.224 |
| T9_positive | 93 | 0.347 | 0.457 | +0.111 |
| convergent_intergenic | 130 | 0.414 | 0.268 | -0.146 |


## CNN-v2 — largest calibrated errors

| label | source | AT | raw | calib | err | Δ vs raw |
|---|---|---|---|---|---|---|
| POS | T9_positive | 0.67 | 0.017 | 0.014 | **0.986** | +0.003 |
| NEG | convergent_intergenic | 0.77 | 0.978 | 0.933 | **0.933** | -0.045 |
| POS | T9_positive | 0.47 | 0.982 | 0.085 | **0.915** | +0.898 |
| POS | T9_positive | 0.53 | 0.930 | 0.105 | **0.895** | +0.825 |
| POS | T9_positive | 0.75 | 0.058 | 0.112 | **0.888** | -0.054 |
| NEG | convergent_intergenic | 0.83 | 0.842 | 0.874 | **0.874** | +0.033 |
| POS | T9_positive | 0.60 | 0.738 | 0.127 | **0.873** | +0.611 |
| POS | T9_positive | 0.75 | 0.096 | 0.131 | **0.869** | -0.035 |


## NT-LoRA — source errors

| Source | n | raw err | calib err | Δ |
|---|---|---|---|---|
| CDS_interior | 56 | 0.076 | 0.186 | +0.111 |
| T9_positive | 93 | 0.760 | 0.531 | -0.228 |
| convergent_intergenic | 130 | 0.095 | 0.301 | +0.205 |


## NT-LoRA — largest calibrated errors

| label | source | AT | raw | calib | err | Δ vs raw |
|---|---|---|---|---|---|---|
| POS | T9_positive | 0.61 | 0.003 | 0.072 | **0.928** | -0.069 |
| POS | T9_positive | 0.60 | 0.005 | 0.078 | **0.922** | -0.072 |
| POS | T9_positive | 0.47 | 0.995 | 0.108 | **0.892** | +0.886 |
| POS | T9_positive | 0.63 | 0.005 | 0.114 | **0.886** | -0.109 |
| POS | T9_positive | 0.59 | 0.062 | 0.117 | **0.883** | -0.055 |
| NEG | convergent_intergenic | 0.77 | 0.988 | 0.880 | **0.880** | -0.109 |
| POS | T9_positive | 0.67 | 0.003 | 0.149 | **0.851** | -0.146 |
| POS | T9_positive | 0.53 | 0.963 | 0.157 | **0.843** | +0.806 |


## ProkBERT — source errors

| Source | n | raw err | calib err | Δ |
|---|---|---|---|---|
| CDS_interior | 56 | 0.059 | 0.176 | +0.117 |
| T9_positive | 93 | 0.675 | 0.508 | -0.167 |
| convergent_intergenic | 130 | 0.110 | 0.287 | +0.177 |


## ProkBERT — largest calibrated errors

| label | source | AT | raw | calib | err | Δ vs raw |
|---|---|---|---|---|---|---|
| POS | T9_positive | 0.58 | 0.022 | 0.063 | **0.937** | -0.042 |
| NEG | convergent_intergenic | 0.83 | 0.983 | 0.934 | **0.934** | -0.049 |
| POS | T9_positive | 0.61 | 0.006 | 0.068 | **0.932** | -0.062 |
| NEG | convergent_intergenic | 0.88 | 0.690 | 0.928 | **0.928** | +0.238 |
| POS | T9_positive | 0.60 | 0.007 | 0.073 | **0.927** | -0.066 |
| NEG | convergent_intergenic | 0.77 | 0.969 | 0.867 | **0.867** | -0.102 |
| POS | T9_positive | 0.67 | 0.006 | 0.143 | **0.857** | -0.137 |
| POS | T9_positive | 0.47 | 0.995 | 0.147 | **0.853** | +0.848 |


## Interpretation

The AT ablation lowers recorded OOF log-loss on this mixture. Aggregate improvements coexist with more rows worsening for some members. Intercept, score coefficient and AT coefficient change jointly; row errors do not isolate a causal AT effect. err is |score−label|; Δ is calibrated error minus raw error, and a flip crosses the wrong side of 0.5.

The sampled prevalence and short context limit transfer. AT-dependent maps can change rank ensembles; the deployed mine uses raw ranks, not these maps. Reused OOF cohorts are sensitivity evidence, not independent final tests. [Calibration notes](../CALIBRATION_NOTES.md) identify scripts and output prefixes.
