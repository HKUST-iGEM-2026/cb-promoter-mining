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

An RBF support-vector regressor reproduced from [Wang et al. (2020)](https://doi.org/10.1093/nar/gkaa325) to explore reporter activity in an *Escherichia coli* DH5α context. This public, ungated package contains fitted numeric parameters, NumPy inference code and real first-ten candidate examples. Downloads need no login/token; it is not a hosted prediction service.

**This activity model is separate from the three-model Clostridium promoter-classification ensemble.** Classifier scores rank promoter-like DNA, not strength; this SVR is neither a fourth ensemble member nor a conversion of ensemble scores.

- **Input to the SVR:** a framed 50-bp promoter core, encoded as 200 position-major T/C/G/A one-hot values. The candidate runner takes already transcription-oriented 81-bp `summit_seq` DNA and uses `[10:60]` at the primary candidate TSS offset 60.
- **Output:** predicted log₂ activity, converted by `2 ** prediction` to the source table's unspecified **arbitrary units (AU)**. AU are not RFU, validated RPU or physical transcription-initiation rates.
- **Reference comparison:** divide each linear prediction by the same SVR's prediction for a framed `pptb_fU` core, not measured notebook fluorescence.

## Results: first ten candidates

The primary 149-bp `pptb_fU` reference uses inferred TSS offset **113**, hence core `[63:113]`, and predicts **6.927284194385597 AU**, defined as 100%. Candidate starts are assumed at offset 60; these are not experimentally mapped DH5α TSS. **Shortlist DL10 is `cand_00101` / `P_prfB`; `pptb_fU` is an independent reference**, despite its older DL10 notebook alias.

| SVR prediction rank | Candidate (shortlist name) | Predicted activity, AU | % of primary predicted `pptb_fU` core | % across five reference-start scenarios |
|---:|---|---:|---:|---:|
| 1 | `cand_00098` (DL9) | 32.8751 | 474.57 | 93.67–474.57 |
| 2 | `cand_00101` (DL10) | 17.9027 | 258.44 | 51.01–258.44 |
| 3 | `cand_00029` (DL1) | 12.0296 | 173.66 | 34.28–173.66 |
| 4 | `cand_00089` (DL7) | 11.3169 | 163.37 | 32.24–163.37 |
| 5 | `cand_00072` (DL5) | 10.1352 | 146.31 | 28.88–146.31 |
| 6 | `cand_00035` (DL3) | 9.92639 | 143.29 | 28.28–143.29 |
| 7 | `cand_00094` (DL8) | 9.79309 | 141.37 | 27.90–141.37 |
| 8 | `cand_00070` (DL6) | 7.16201 | 103.39 | 20.41–103.39 |
| 9 | `cand_00032` (DL2) | 5.91377 | 85.37 | 16.85–85.37 |
| 10 | `cand_00048` (DL4) | 3.97501 | 57.38 | 11.33–57.38 |

Eight candidates exceed the **primary model-predicted reference**, not a measured control. The five reference-start scenarios give:

| Reference TSS offset (zero-based) | Predicted reference activity, AU |
|---:|---:|
| 113 (primary) | 6.927284 |
| 109 | 30.008978 |
| 111 | 12.589095 |
| 112 | 21.106966 |
| 110 | 35.097146 |

With candidate framing fixed, the leading DL9/reference ratio changes from **4.7457× to 0.9367×**. These are **framing-scenario ranges, not confidence intervals or measured fold changes**. Candidate framing matters too: `cand_00048` leads at offset 58, whereas `cand_00098` leads at offsets 59–62. The primary `cand_00048` prediction is below the training labels' minimum (4.590844374 AU); that extrapolation does not establish inactivity.

| Fit evidence | Value | Meaning |
|---|---:|---|
| Reproduced hyperparameter-selection CV R² | **0.2660902338588789** | Five-fold selection on log₂ labels, not independent candidate accuracy |
| Wang et al. held-out Pearson correlation | **0.57** | Original 66/22 split and different fit, not this all-88-record refit |

Predictions apply to framed **50-bp cores**, not whole 81-bp inserts or the full 149-bp reference. The published training assay used DH5α, modified pFAB217/p15A, sfGFP, fixed B0030 5′UTR and LB/kanamycin; host, construct, leader and condition changes can alter activity. Promoter Calculator supplied conditional reference framing only: its MG1655-labelled setting is not DH5α-calibrated, and its initiation-rate AU never enter SVR normalization. The team's separately reported *E. coli* GFP tests do not supply candidate-resolved quantitative validation or establish DH5α as the reporter strain. Native *C. butyricum* activity remains unvalidated.

### Downloads and reuse

The public [results and downloads index](https://github.com/HKUST-iGEM-2026/cb-promoter-mining/blob/main/results/README.md) links full-precision prediction CSV/TSV/JSON, framing-sensitivity results, reference DNA/provenance, named first-ten FASTA/TSV and gene-context files. It also provides the separate mining catalogue: **929 retained regions, 180 placed −10 hits and 45 both-hit candidates** on GCF_014131795.1. The first-ten analysis selects the first ten rows of that 45-candidate subset in catalogue order; the table above reorders them by predicted activity, not mining rank. Motif support and nearby genes are not biological validation.

## Download and predict without refitting

Install `huggingface_hub` for the `hf` CLI. The pinned public package supplies the predictor, fitted NPZ/metadata, exact reference and catalogue; inference requires **NumPy only**, not scikit-learn, a GPU, a language model or executable pickle.

```bash
hf download CloBiT/dh5alpha-promoter-strength-svr \
  --revision 0c5202892f4255eb8eb17493bed8d51e04f8cfc4 --local-dir dh5alpha_svr
uv run --no-project --with numpy==2.5.3 python \
  dh5alpha_svr/predict_dh5alpha_strength.py \
  --model dh5alpha_svr \
  --reference dh5alpha_svr/pptb_fU_reference.json \
  --input dh5alpha_svr/canonical_candidates.tsv \
  --top-n 10 --tss-offset 60 --out-prefix dh5alpha_first10
```

The TSV must contain `candidate_id`, `tier_label`, `summit_seq`, `chromosome`, `strand` and `tss_position`. Input `summit_seq` is already transcription-oriented: **do not reverse-complement it again**. No padding is applied. The runner selects the first ten `canonical` (both positional motifs) rows in file order, then writes activity-ranked TSV/JSON/CSV including DNA, cores, coordinates, reference ratios and assumptions.

The fitted NPZ/metadata are also committed in the [GitHub release directory](https://github.com/HKUST-iGEM-2026/cb-promoter-mining/tree/main/release/huggingface/dh5alpha-promoter-strength-svr); the [method guide](https://github.com/HKUST-iGEM-2026/cb-promoter-mining/blob/main/docs/ECOLI_STRENGTH_PREDICTIONS.md) gives local-checkout commands. Loading uses `allow_pickle=False` and verifies the recorded weight checksum. The fitted NPZ SHA-256 is `153f41e1d051d432a2172b21b6e94f07799fa2229a45d795ae974fc8140130a8`. Historical publication checks matched all 14 payload files and reproduced the ten predictions using the downloaded NumPy-only package; `verification.json` is numerical provenance, not biological validation.

## Training and optional fit reproduction

The pinned Wang source has 94 rows. Excluding six short controls and stripping cloning overhangs with `[5:-1]` leaves **88 eligible records: 83 generated promoters and five random sequences**. Features are float32, position-major T/C/G/A one-hot (50 × 4), target log₂ of the activity label. A seed-2 permutation precedes unshuffled five-fold R² selection over C=[1,10,100,1000] and gamma=[0.01,0.1,1,10,100]. Selected RBF SVR: **C=10, gamma=0.01, epsilon=0.1**, refitted on all 88 rows. The adapter uses the raw SVR/inverse-log₂ output, not upstream maximum-normalized files or swapped CNN/SVR assignments.

To reproduce the fit rather than load the retained parameters:

```bash
uv run --no-project --with numpy==2.5.3 --with scipy==1.18.1 \
  --with scikit-learn==1.9.1 python \
  dh5alpha_svr/predict_dh5alpha_strength.py \
  --reference dh5alpha_svr/pptb_fU_reference.json \
  --input dh5alpha_svr/canonical_candidates.tsv \
  --top-n 10 --tss-offset 60 \
  --export-model reproduced_svr --out-prefix reproduced_first10
```

This optional command fits a new local artifact; it is not required for downloading, publishing cards or using the saved model. The original fitting environment was Python 3.13.5, NumPy 2.5.3, SciPy 1.18.1 and scikit-learn 1.9.1. `model_metadata.json` retains source/data hashes, preprocessing, settings and assay context.

## Sources and licences

- Wang et al., “Synthetic promoter design in Escherichia coli based on a deep generative network,” *Nucleic Acids Research* 48:6403–6412 (2020), [doi:10.1093/nar/gkaa325](https://doi.org/10.1093/nar/gkaa325).
- Pinned procedure/data: [HaochenW/Deep_promoter, c0e16c8218a983fbb07bf46d59c44109d4371b0e](https://github.com/HaochenW/Deep_promoter/tree/c0e16c8218a983fbb07bf46d59c44109d4371b0e). Preserve the bundled original MIT `UPSTREAM_LICENSE`.
- Project inference code and fitted numeric artifact: **MIT** (`LICENSE`); cite this HKUST iGEM 2026 project. This package uses Wang's small published table, **not RegulonDB exports**. The genomic ensemble's separate data/weight licences do not describe this SVR.
- Conditional reference framing: LaFleur et al. (2022), [doi:10.1038/s41467-022-32829-5](https://doi.org/10.1038/s41467-022-32829-5). The GPL-3.0-or-later Promoter Calculator implementation is not bundled or required for cached inference.
- [Project predictor/data](https://github.com/HKUST-iGEM-2026/cb-promoter-mining/tree/main/pipeline/mining/strength-prediction) and [full method/interpretation](https://github.com/HKUST-iGEM-2026/cb-promoter-mining/blob/main/docs/ECOLI_STRENGTH_PREDICTIONS.md).
