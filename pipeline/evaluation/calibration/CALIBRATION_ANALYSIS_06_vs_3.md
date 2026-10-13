# Calibration checkpoint comparison

calib_06 uses T21_final_06 with 0.6-swept hyperparameters; calib_3 uses T21_final with older 0.9-swept hyperparameters. **Both were trained on 0.6-clustered data.** They share four models and the same 279-row mixture, so this tests checkpoint sensitivity, not clustering or independent validation. Short positive/negative context and sampled prevalence limit genomic transfer.

## Historical checkpoint preferences

| Model | Best checkpoint (by calibrated discrimination) | One-line verdict |
|---|---|---|
| **cnn_v2** | **`calib_3` (0.9-swept)** — cal AUROC **0.872**, AUPRC **0.769** | Best model in the whole comparison. The 0.6 re-sweep *regressed* it (→0.834/0.692). **Keep the T21_final checkpoint.** |
| **prokbert** | **`calib_06` (0.6-swept)** — cal AUROC 0.799, AUPRC 0.643 | Effectively a tie (0.796/0.630 in calib_3); 0.6 wins on raw too. **Adopt T21_final_06**, consistent with the genome-scale finding. |
| **nt_lora** | **`calib_06` (0.6-swept)** — cal AUROC 0.769, AUPRC 0.617 | The 0.9-swept nt raw ranker **collapsed below chance** (raw AUROC 0.470). 0.6 is decisively healthier, but nt is still the weakest member either way. |
| **cnn** | ~tie (calib_06 0.833/0.701 vs calib_3 0.829/0.687) | The most *honest* calibrator (fewest rows harmed) in both runs; middling discrimination. Stable, not a leader. |


## Raw discrimination

| Metric | model | `calib_06` raw (0.6-swept) | `calib_3` raw (0.9-swept) | Δ (06 − 3) |
|---|---|---|---|---|
| AUROC | cnn | 0.750 | 0.715 | +0.035 |
| AUROC | cnn_v2 | 0.720 | **0.775** | −0.055 |
| AUROC | nt_lora | 0.683 | **0.470**  | +0.213 |
| AUROC | prokbert | 0.708 | 0.691 | +0.017 |
| AUPRC | cnn | 0.596 | 0.591 | +0.005 |
| AUPRC | cnn_v2 | 0.586 | **0.633** | −0.047 |
| AUPRC | nt_lora | 0.504 | **0.337** | +0.167 |
| AUPRC | prokbert | 0.572 | 0.560 | +0.012 |


## OOF discrimination and reliability

| Metric | model | `calib_06` (0.6-swept) | `calib_3` (0.9-swept) | better run |
|---|---|---|---|---|
| **cal AUROC** | cnn | 0.833 | 0.829 | ~tie |
| | cnn_v2 | 0.834 | **0.872** | calib_3 |
| | nt_lora | **0.769** | 0.712 | calib_06 |
| | prokbert | 0.799 | 0.796 | ~tie |
| **cal AUPRC** | cnn | **0.701** | 0.687 | calib_06 |
| | cnn_v2 | 0.692 | **0.769** | calib_3 |
| | nt_lora | **0.617** | 0.562 | calib_06 |
| | prokbert | **0.643** | 0.630 | calib_06 |
| **ECE** | cnn | 0.064 | **0.029** | calib_3 |
| | cnn_v2 | 0.081 | **0.059** | calib_3 |
| | nt_lora | 0.063 | **0.044** | calib_3 |
| | prokbert | 0.065 | **0.056** | calib_3 |
| **log-loss** | cnn | 0.469 | 0.480 | calib_06 |
| | cnn_v2 | 0.472 | **0.421** | calib_3 |
| | nt_lora | **0.536** | 0.583 | calib_06 |
| | prokbert | 0.517 | 0.516 | ~tie |


## AT ablation

| Model | raw→cal AUROC, `calib_06` | raw→cal AUROC, `calib_3` | AT coef (06 / 3) |
|---|---|---|---|
| cnn | 0.750 → 0.833 (**+0.083**) | 0.715 → 0.829 (**+0.114**) | +1.076 / +1.127 |
| cnn_v2 | 0.720 → 0.834 (**+0.114**) | 0.775 → 0.872 (**+0.097**) | +1.230 / +1.351 |
| nt_lora | 0.683 → 0.769 (**+0.086**) | 0.470 → 0.712 (**+0.242** ) | +0.994 / +0.878 |
| prokbert | 0.708 → 0.799 (**+0.091**) | 0.691 → 0.796 (**+0.105**) | +0.959 / +0.994 |


## Source-error directions

| Family | on genomic **negatives** (CDS + convergent) | on real **positives** (T9) |
|---|---|---|
| **CNN / cnn_v2** (liberal) | raw err **high** (CDS ~0.33–0.36, conv ~0.41–0.44) → scores putative negatives higher | raw err **low** (~0.31–0.40) |
| **nt_lora / prokbert** (conservative) | raw err **low** (CDS ~0.06–0.18, conv ~0.10–0.22) → clean on negatives | raw err **high** (~0.56–0.86) → misses real promoters |


## Interpretation

The re-sweep is not uniformly better: NT raw AUROC changes 0.470→0.683, while CNN-v2 changes 0.775→0.720. calib_3 gives lower ECE across all four members, but ECE is not a ranking score. These point estimates on reused data do not establish statistically reliable target-genome gains.

AT lowers recorded OOF log-loss in all eight model×round cells and can reorder scores. Low-AT positives include severe raw→calibrated drops: calib_3 NT at AT 0.47, 0.977→0.036; ProkBERT, 0.996→0.080. Joint coefficient/intercept changes prevent attributing each error to AT alone.

Many NT/ProkBERT rows worsen (192/197 in calib_06; 171/187 in calib_3) despite improved proper scores. Error magnitudes and row counts measure different things; class-level differences motivate testing an ensemble, not proof of anticorrelated sequence errors.

Checkpoint preferences above are historical, not current deployment instructions. The trio uses raw ranks without calibration. See [calib_06](calib_06/calibration_analysis.md), [calib_3](calib_3/calibration_analysis.md) and [method notes](CALIBRATION_NOTES.md).
