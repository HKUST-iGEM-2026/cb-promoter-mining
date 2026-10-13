# Calibration: calib_xs

Historical four-model calibration, not the deployed ProkBERT/CNN-v2/Evo2 ensemble. The 279-row mixture has 93 T9 positives and 186 putative negatives (130 convergent/56 CDS). Original prediction CSVs/maps are absent; the T9 positives are 49–52 bp and the extractor defaults to 52 bp. Padding does not supply 81-bp biological context.

## Method

StandardScaler→LogisticRegression compares score-only/score+AT, unregularized/L2, with 5×5 stratified OOF selection on log-loss (tol 0.005, L2 C=1). This round chose unregularized score+AT. Lower ECE/Brier/log-loss is better; higher AUROC/AP is better. ± values are repeated-CV standard deviations, not confidence intervals.

## Checkpoint provenance

| Model | chosen calibrator | features | checkpoint / adapter | n_conv / n_cds |
|---|---|---|---|---|
| cnn | score+AT / unregularized | logit, at_fraction | `workflow/results/T21_xs/cnn/best_model.pt` | 130 / 56 |
| cnn_v2 | score+AT / unregularized | logit, at_fraction | `workflow/results/T21_xs/cnn_v2/best_model.pt` | 130 / 56 |
| nt_lora | score+AT / unregularized | logit, at_fraction | `workflow/results/T21_xs/nt_lora/best_adapter` | 130 / 56 |
| prokbert | score+AT / unregularized | logit, at_fraction | `workflow/results/T21_xs/prokbert/best_model` | 130 / 56 |


## Metrics

| Metric | cnn raw | cnn_v2 raw | nt_lora raw | prokbert raw | cnn calib | cnn_v2 calib | nt_lora calib | prokbert calib |
|---|---|---|---|---|---|---|---|---|
| ECE | 0.242 | 0.165 | 0.347 | 0.175 | **0.051**  | **0.064**  | **0.037**  | **0.040**  |
| Brier | 0.257 | 0.206 | 0.348 | 0.205 | **0.152**  | **0.152**  | **0.194**  | **0.148**  |
| Log-loss | 0.762 | 0.633 | 1.308 | 0.703 | **0.467**  | **0.467**  | **0.579**  | **0.455**  |
| AUROC | 0.769 | 0.771 | 0.568 | 0.795 | **0.839**  | **0.846**  | **0.718**  | **0.843**  |
| AUPRC | 0.605 | 0.675 | 0.378 | 0.687 | **0.682**  | **0.713**  | **0.566**  | **0.744**  |


## Score-only versus AT

| Model | OOF log-loss score-only | OOF log-loss +AT | AT helps? | AT coef | mean P(positives) score-only -> +AT |
|---|---|---|---|---|---|
| cnn | 0.5351 | 0.4689 | **yes** | +0.962 | 0.461 -> 0.547 |
| cnn_v2 | 0.5273 | 0.4684 | **yes** | +0.950 | 0.477 -> 0.548 |
| nt_lora | 0.6401 | 0.5791 | **yes** | +0.869 | 0.332 -> 0.419 |
| prokbert | 0.5198 | 0.4552 | **yes** | +1.028 | 0.487 -> 0.563 |


## Row-error summary

| Model | mean \|calib-label\| | mean \|raw-label\| | rows made worse | right->wrong flips |
|---|---|---|---|---|
| cnn | 0.300 | 0.396 | 102/279 | 22 |
| cnn_v2 | 0.299 | 0.311 | 166/279  | 12 |
| nt_lora | 0.387 | 0.391 | 165/279  | 25 |
| prokbert | 0.290 | 0.267 | 190/279  | 13 |


## CNN-v1 — source errors

| Source | n | raw err | calib err | Δ |
|---|---|---|---|---|
| CDS_interior | 56 | 0.412 | 0.126 | -0.286 |
| T9_positive | 93 | 0.230 | 0.451 | +0.221 |
| convergent_intergenic | 130 | 0.507 | 0.267 | -0.240 |


## CNN-v1 — largest calibrated errors

| label | source | AT | raw | calib | err | Δ vs raw |
|---|---|---|---|---|---|---|
| NEG | convergent_intergenic | 0.77 | 0.993 | 0.956 | **0.956** | -0.037 |
| POS | T9_positive | 0.67 | 0.161 | 0.045 | **0.955** | +0.116 |
| POS | T9_positive | 0.73 | 0.150 | 0.077 | **0.923** | +0.073 |
| NEG | convergent_intergenic | 0.87 | 0.907 | 0.911 | **0.911** | +0.004 |
| POS | T9_positive | 0.58 | 0.767 | 0.098 | **0.902** | +0.669 |
| POS | T9_positive | 0.61 | 0.687 | 0.108 | **0.892** | +0.580 |
| NEG | convergent_intergenic | 0.83 | 0.945 | 0.882 | **0.882** | -0.063 |
| POS | T9_positive | 0.84 | 0.052 | 0.121 | **0.879** | -0.069 |


## CNN-v2 — source errors

| Source | n | raw err | calib err | Δ |
|---|---|---|---|---|
| CDS_interior | 56 | 0.174 | 0.126 | -0.048 |
| T9_positive | 93 | 0.405 | 0.450 | +0.045 |
| convergent_intergenic | 130 | 0.304 | 0.266 | -0.038 |


## CNN-v2 — largest calibrated errors

| label | source | AT | raw | calib | err | Δ vs raw |
|---|---|---|---|---|---|---|
| POS | T9_positive | 0.53 | 0.299 | 0.039 | **0.961** | +0.260 |
| NEG | convergent_intergenic | 0.77 | 0.989 | 0.941 | **0.941** | -0.049 |
| POS | T9_positive | 0.67 | 0.038 | 0.062 | **0.938** | -0.024 |
| POS | T9_positive | 0.58 | 0.354 | 0.072 | **0.928** | +0.282 |
| NEG | convergent_intergenic | 0.87 | 0.883 | 0.924 | **0.924** | +0.041 |
| NEG | convergent_intergenic | 0.83 | 0.934 | 0.922 | **0.922** | -0.012 |
| NEG | convergent_intergenic | 0.83 | 0.880 | 0.865 | **0.865** | -0.016 |
| POS | T9_positive | 0.73 | 0.045 | 0.143 | **0.857** | -0.098 |


## NT-LoRA — source errors

| Source | n | raw err | calib err | Δ |
|---|---|---|---|---|
| CDS_interior | 56 | 0.201 | 0.229 | +0.027 |
| T9_positive | 93 | 0.833 | 0.581 | -0.252 |
| convergent_intergenic | 130 | 0.156 | 0.316 | +0.160 |


## NT-LoRA — largest calibrated errors

| label | source | AT | raw | calib | err | Δ vs raw |
|---|---|---|---|---|---|---|
| POS | T9_positive | 0.47 | 0.958 | 0.038 | **0.962** | +0.920 |
| POS | T9_positive | 0.51 | 0.974 | 0.071 | **0.929** | +0.903 |
| POS | T9_positive | 0.53 | 0.835 | 0.075 | **0.925** | +0.760 |
| POS | T9_positive | 0.58 | 0.041 | 0.079 | **0.921** | -0.038 |
| POS | T9_positive | 0.60 | 0.010 | 0.103 | **0.897** | -0.093 |
| POS | T9_positive | 0.59 | 0.037 | 0.106 | **0.894** | -0.069 |
| POS | T9_positive | 0.61 | 0.047 | 0.116 | **0.884** | -0.068 |
| NEG | convergent_intergenic | 0.88 | 0.870 | 0.859 | **0.859** | -0.011 |


## ProkBERT — source errors

| Source | n | raw err | calib err | Δ |
|---|---|---|---|---|
| CDS_interior | 56 | 0.092 | 0.130 | +0.038 |
| T9_positive | 93 | 0.548 | 0.436 | -0.112 |
| convergent_intergenic | 130 | 0.142 | 0.255 | +0.113 |


## ProkBERT — largest calibrated errors

| label | source | AT | raw | calib | err | Δ vs raw |
|---|---|---|---|---|---|---|
| POS | T9_positive | 0.61 | 0.024 | 0.042 | **0.958** | -0.017 |
| POS | T9_positive | 0.60 | 0.048 | 0.055 | **0.945** | -0.008 |
| NEG | convergent_intergenic | 0.77 | 0.974 | 0.942 | **0.942** | -0.032 |
| POS | T9_positive | 0.67 | 0.019 | 0.084 | **0.916** | -0.064 |
| POS | T9_positive | 0.53 | 0.512 | 0.089 | **0.911** | +0.423 |
| POS | T9_positive | 0.73 | 0.010 | 0.112 | **0.888** | -0.102 |
| NEG | convergent_intergenic | 0.75 | 0.928 | 0.886 | **0.886** | -0.042 |
| POS | T9_positive | 0.69 | 0.031 | 0.125 | **0.875** | -0.094 |


## Interpretation

The AT ablation lowers recorded OOF log-loss on this mixture. Aggregate improvements coexist with more rows worsening for some members. Intercept, score coefficient and AT coefficient change jointly; row errors do not isolate a causal AT effect. err is |score−label|; Δ is calibrated error minus raw error, and a flip crosses the wrong side of 0.5.

The sampled prevalence and short context limit transfer. AT-dependent maps can change rank ensembles; the deployed mine uses raw ranks, not these maps. Reused OOF cohorts are sensitivity evidence, not independent final tests. [Calibration notes](../CALIBRATION_NOTES.md) identify scripts and output prefixes.
