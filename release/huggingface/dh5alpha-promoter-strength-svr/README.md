---
license: mit
language:
- en
tags:
- biology
- genomics
- promoter-strength
- support-vector-regression
- escherichia-coli
---

# DH5α promoter-strength SVR

Our genome-mining ensemble ranks promoter-like DNA, but its scores do not predict expression strength. To explore activity in an *Escherichia coli* DH5α reporter context, we reproduced the RBF support-vector regression procedure from [Wang et al. (2020)](https://doi.org/10.1093/nar/gkaa325). This package contains the fitted numeric model, CPU inference code and the first-ten candidate results. It is a separate activity model, not a fourth member of the Clostridium promoter-classification ensemble.

## Building the model

We fitted the source's 88 eligible records after removing six short controls from its 94-row table and stripping the cloning overhangs. Each 50-bp promoter core becomes 200 position-major T/C/G/A one-hot features; the target is log₂ of the published activity label. The seeded five-fold R² search selected C=10 and gamma=0.01 for an RBF SVR with epsilon=0.1, then refitted all 88 rows.

The model is stored as `model.npz` and `model_metadata.json`. Loading uses `allow_pickle=False` and checks the recorded weight checksum. Inference needs NumPy only; scikit-learn and SciPy are needed only to reproduce the fit. No GPU, remote inference service, pretrained language model or executable pickle is needed.

The published training assay used DH5α, modified pFAB217/p15A, sfGFP, a fixed B0030 5′UTR and LB/kanamycin. A different host, construct, leader or condition can alter reporter activity. The source table's AU have no physical-unit header and should not be interpreted as RFU, validated RPU or transcription-initiation rates.

## Download and predict

The project repositories are private. Download with an account authorised for `CloBiT/dh5alpha-promoter-strength-svr`; the download CLI requires `huggingface_hub`.

```bash
hf download CloBiT/dh5alpha-promoter-strength-svr --local-dir dh5alpha_svr
uv run --no-project --with numpy==2.5.3 python \
  dh5alpha_svr/predict_dh5alpha_strength.py \
  --model dh5alpha_svr \
  --reference dh5alpha_svr/pptb_fU_reference.json \
  --input dh5alpha_svr/canonical_candidates.tsv \
  --top-n 10 --tss-offset 60 \
  --out-prefix dh5alpha_first10
```

The input TSV must contain `candidate_id`, `tier_label`, `summit_seq`, `chromosome`, `strand` and `tss_position`. `summit_seq` is already transcription-oriented 81-bp DNA; no padding or second reverse complement is applied. The runner selects the first ten canonical records in file order and scores their `[10:60]` 50-bp cores. It writes TSV, JSON and CSV with predictions, reference ratios, DNA, coordinates and assumptions. The output ranking is by predicted activity, not the input catalogue's mining rank.

To reproduce the fit instead of loading it:

```bash
uv run --no-project --with numpy==2.5.3 --with scipy==1.18.1 \
  --with scikit-learn==1.9.1 python \
  dh5alpha_svr/predict_dh5alpha_strength.py \
  --reference dh5alpha_svr/pptb_fU_reference.json \
  --input dh5alpha_svr/canonical_candidates.tsv \
  --top-n 10 --tss-offset 60 \
  --export-model reproduced_svr --out-prefix reproduced_first10
```

## First-ten results

For the primary framing, the same-model `pptb_fU`/DL10 reference prediction is **6.927284194385597 AU**, defined as 100%. The candidate start is assumed at offset 60; the exact 149-bp reference uses offset 113 and `[63:113]`. These starts are inferred, not experimentally mapped DH5α TSS.

| DH5α prediction rank | Candidate | Predicted activity, AU | % of predicted primary reference | % across five reference-start scenarios |
|---:|---|---:|---:|---:|
| 1 | `cand_00098` | 32.8751 | 474.57 | 93.67–474.57 |
| 2 | `cand_00101` | 17.9027 | 258.44 | 51.01–258.44 |
| 3 | `cand_00029` | 12.0296 | 173.66 | 34.28–173.66 |
| 4 | `cand_00089` | 11.3169 | 163.37 | 32.24–163.37 |
| 5 | `cand_00072` | 10.1352 | 146.31 | 28.88–146.31 |
| 6 | `cand_00035` | 9.92639 | 143.29 | 28.28–143.29 |
| 7 | `cand_00094` | 9.79309 | 141.37 | 27.90–141.37 |
| 8 | `cand_00070` | 7.16201 | 103.39 | 20.41–103.39 |
| 9 | `cand_00032` | 5.91377 | 85.37 | 16.85–85.37 |
| 10 | `cand_00048` | 3.97501 | 57.38 | 11.33–57.38 |

Eight candidates exceed the primary predicted reference, with `cand_00098` and `cand_00101` leading. However, five reference-start scenarios span 6.927284–35.097146 AU, changing the leading ratio from 4.7457× to 0.9367×. These are framing-scenario ranges, not confidence intervals. Candidate-offset sensitivity also changes the leader: `cand_00048` leads at 58, while `cand_00098` leads at 59–62. `cand_00048` falls below the fitted labels' minimum; that extrapolation is not evidence of inactivity.

## Evaluation and experimental limits

The reproduced fit's selection-CV R² is **0.266090**, not independent candidate accuracy. Wang et al.'s held-out Pearson correlation **0.57** refers to a different split and fit. `verification.json` records numerical checks of this package, not biological validation. `first10_predictions.csv` and `.tsv` retain the original full-precision conditional results.

These predictions apply to the framed 50-bp cores, not whole 81-bp candidate inserts or the full 149-bp reference. The ratios divide same-model linear predictions, not notebook fluorescence. Promoter Calculator supplied conditional reference framing; its initiation-rate AU never enter the SVR normalization. The team's separately reported E. coli GFP assays do not provide candidate-resolved quantitative validation or establish that the reporter strain was DH5α. Native C. butyricum activity remains unvalidated.

## Sources and licences

- [Project predictor and data on GitHub](https://github.com/HKUST-iGEM-2026/cb-promoter-mining/tree/reorganise/pipeline/mining/strength-prediction), with [full method and interpretation](https://github.com/HKUST-iGEM-2026/cb-promoter-mining/blob/reorganise/docs/ECOLI_STRENGTH_PREDICTIONS.md).
- Wang et al., “Synthetic promoter design in Escherichia coli based on a deep generative network,” *Nucleic Acids Research* 48:6403–6412, 2020, [doi:10.1093/nar/gkaa325](https://doi.org/10.1093/nar/gkaa325).
- Pinned upstream procedure/data: [HaochenW/Deep_promoter, c0e16c8218a983fbb07bf46d59c44109d4371b0e](https://github.com/HaochenW/Deep_promoter/tree/c0e16c8218a983fbb07bf46d59c44109d4371b0e). The original MIT notice is retained in `UPSTREAM_LICENSE`.
- Project inference code and fitted numeric artifact: MIT (`LICENSE`). The package uses Wang's small published table, not RegulonDB exports. The genomic ensemble's separate data/weight licences do not describe this SVR.
- Conditional reference framing: [LaFleur et al. (2022)](https://doi.org/10.1038/s41467-022-32829-5); the GPL-3.0-or-later Promoter Calculator implementation is not bundled or needed for cached inference.
