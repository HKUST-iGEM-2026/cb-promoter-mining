---
license: cc-by-4.0
tags:
  - biology
  - genomics
  - dna
  - promoter-prediction
  - clostridium
  - cnn
  - pytorch
  - igem
pipeline_tag: text-classification
---

# Fusion CNN promoter classifier

HKUST iGEM 2026 from-scratch CNN-v2, deployed with ProkBERT-mini and the frozen-Evo2 probe. The artifact declares **CC-BY-4.0**; source-data terms remain separate.

## Architecture and loading

One-hot and ENAC branches use kernels 7/5/3, filters 64/128/256, batch normalization and pooling. Dinucleotide/trinucleotide frequencies supply 16/64-d MLP branches. Features concatenate into fusion dimension 256. The helper center-crops/N-pads to 81 bp; supply transcription-oriented 81-bp DNA to retain the nominal offset-60 frame. ENAC width is 5.

The private package requires authorised access and includes its feature/model code:

```python
import json, sys, torch
from huggingface_hub import snapshot_download

path = snapshot_download("CloBiT/cnn-fusion-clostridium-promoter",
                         revision="191fc88fffeb52ecbf4bdb27702a734365545815")
sys.path.insert(0, path)
from train_cnn_v2 import build_model_from_config, predict_proba

cfg = json.load(open(f"{path}/config.json"))
model = build_model_from_config(cfg, device="cpu")
model.load_state_dict(torch.load(f"{path}/best_model.pt", map_location="cpu", weights_only=True))

seqs = ["A" * 81]  # your 81 bp windows, TSS at offset 60
scores = predict_proba(model, seqs, device="cpu", enac_window=cfg.get("enac_window", 5))
print(scores)  # raw positive-class scores
```

## Files

| File | What it is |
|---|---|
| `best_model.pt` | PyTorch `state_dict` of the deployed checkpoint |
| `config.json` | architecture + active encodings; `build_model_from_config` reads this |
| `train_cnn_v2.py` | model class, feature pipeline and `predict_proba` — needed to load and run |
| `metrics.json` | the deployed run's validation and test scores |
| `history.json` | per-epoch training history |
| `SOURCE_CODE_LICENSE` | root MIT copyright/license notice for bundled project source; included by the upload script, separate from model-weight licensing |
| `REGULONDB_LICENSE.md` | training-source data license, attribution and changes; separate from model-weight licensing |
| `verification.json` | model/data hashes, alignment checks, saved-feature evaluation and remaining limits |
| `recovery_SHA256SUMS` | original relative-path SHA-256 receipt identifying all 159 recovered files; inputs themselves remain local |


## Training data

**Silver positives** are putative promoter-positive upstream DNA labelled from predicted housekeeping-gene annotations and genomic position, not mapped TSS or experimentally measured promoters.

The deployed T15_xs cohort has 50,190 81-bp windows, 25,095 labels of each class. Recovered MMseqs2 memberships keep clusters within one split. The retained recipe uses identity 0.6; the original invocation/version is not recovered. Train/validation/test are 44,085/3,052/3,053; positive counts 24,001/547/547. Silver positives and cross-species positives are training-only, leaving a test no-skill AP near 0.18.

| Source | Windows | Species | Kind |
|---|---:|---|---|
| Hocq et al. 2022 published TSS | 5,506 | *C. acetobutylicum* ATCC 824, *C. beijerinckii* NCIMB 8052 / DSM 6423 | in-genus, verified |
| Housekeeping-gene upstreams | 13,838 | in-genus, across the 166 genomes | **silver**: annotation/position-derived putative positives, no mapped TSS or measured promoter activity; training-only, sample-confidence weight 0.3 (not biological probability) |
| RegulonDB | 3,827 | *E. coli* K-12 | cross-species, curated |
| *Bacillus licheniformis* published TSS | 888 | *B. licheniformis* | cross-species — **see erratum** |
| *Ruminiclostridium cellulolyticum* H10 published TSS | 663 | ex-*Clostridium* relative | cross-species |
| DBTBS | 373 | *B. subtilis* | cross-species, curated |


Only Hocq supplies held-out positives (4,412/547/547); 19,344 positives are in-genus. Bakta products identify housekeeping loci: 22,760 selected loci reduce to 14,659 proximity-derived representatives among selected loci, not a full operon map. Silver crops lack mapped starts, receive sample-confidence weight 0.3 for deployed training loss (not biological probability) and do not prove constitutive expression.

Four putative-negative classes are CDS, intergenic, downstream and training-only SRS shuffles of silver cores. The custom shuffle has a fallback; no class is experimentally guaranteed promoter-free. All 166 annotations record Bakta v1.11.4/database v6.0 light.

**Erratum:** 888 B. licheniformis positives use GCF_000011645.1/NC_006270.3 instead of the source-compatible GCF_000008425.1/NC_006322.1. They are training-only but remain in deployed weights. Correcting documentation does not retrain the artifact. Target C. butyricum is excluded from supervised construction; backbone pretraining exclusion is unknown. Benchmark strains supplied silver (annotation/position-derived putative positive) and putative-negative windows, though their published TSS were withheld.

## Training

AdamW with cosine-annealing warm restarts uses per-sample loss weights and unweighted validation AP for early stopping.

| | |
|---|---|
| learning rate | 0.0003748224111698842 |
| batch size | 32 |
| dropout | 0.10382857322260322 |
| weight decay | 0.00024777455052193346 |
| max epochs / patience | 30 / 8 |
| seed | 42 |
| encodings | one-hot, dinuc, trinuc, ENAC (window 5) |


## Evaluation

| Regime | Metric | Score |
|---|---|---|
| **A** · in-genus test (`T15_xs`, unweighted; no-skill ≈0.18) | AUPRC | 0.726 |
| **A** · same test set | AUROC / accuracy / F1 / MCC | 0.898 / 0.834 / 0.627 / 0.543 |
| **B** · composition-matched SRS discrimination\* | AUROC | **0.924** (highest of four task-trained models tested; not proof of motif mechanism) |
| **C** · genome scale, *C. ljungdahlii* (restricted, step 10) | precision @ recall 0.1 / 0.5 | 64.4% / 47.2% |
| **C** · genome scale, *C. autoethanogenum* (restricted, step 10) | precision @ recall 0.1 / 0.5 | 81.0% / 66.0% |


Original-run validation/test AP is 0.7380835045313733/0.7255828423854476; weighted:true denotes loss, not AP. The SRS result uses an earlier T21_final/plain-0.6 checkpoint, not the deployed xs model. Historical standalone matched-step values 63.1/48.2% and 81.2/67.1%, and step-1 C. ljungdahlii 71.6% at recall 0.1, are separate runs from the table.

## Benchmark interpretation

Restricted step-10 regions span 300 bp upstream/50 bp into annotated genes. Same-contig/strand 81-bp windows expand ±40 bp to cover published TSS; unmatched starts merge at ≤81 bp. Precision proxy is covered TSS/(covered TSS+unmatched peaks), not the fraction of calls matching or measured biological purity. Truth is 1,245 C. ljungdahlii starts (GCF_000143685.1/NC_014328.1) and 2,885 C. autoethanogenum starts (GCF_000484505.1/NC_022592.1). Compare within a genome; truth coverage differs.

Deployment intersects member coordinates before averaging raw-score percentiles. Mean rank is not probability, strength or unanimity; no calibrator is used. Labels/motifs do not establish activity, sigma use, constitutive expression or downstream transcripts. No experimental validation accompanies the release.

## Recovery checks

The recovered classifier previously scored all 45 promoters with both −10 and −35 motifs on macOS CPU (Python 3.13.5, torch 2.11.0, NumPy 2.5.3); helper/CLI agreed within six-decimal rounding. All original 50,190 rows/encodings align, and 159 recovered files passed source hashes. These checks do not remeasure CNN held-out metrics. verification.json records scope; large inputs are not packaged.

## Source credit and data terms

Cite the model base, this project and the data used:

- Hocq et al., 2022, in-genus training TSS, [doi:10.1128/spectrum.02288-21](https://doi.org/10.1128/spectrum.02288-21).
- Al-Bassam et al., 2018, C. ljungdahlii benchmark, [doi:10.1038/s41467-018-06993-6](https://doi.org/10.1038/s41467-018-06993-6).
- RegulonDB export 13.5 (2025-01-28); database v12.0 paper, [doi:10.1093/nar/gkad1072](https://doi.org/10.1093/nar/gkad1072).
- [DBTBS](https://dbtbs.hgc.jp), B. subtilis promoters; export release unrecorded.
- Xu et al., 2015, R. cellulolyticum TSS, [doi:10.1038/ncomms7900](https://doi.org/10.1038/ncomms7900), PRJNA247408.
- Wiegand et al., 2013, B. licheniformis TSS, [doi:10.1186/1471-2164-14-667](https://doi.org/10.1186/1471-2164-14-667), PRJNA188930.
- de Souza Pinto Lemgruber et al., 2019, C. autoethanogenum benchmark, [doi:10.3389/fmicb.2019.02549](https://doi.org/10.3389/fmicb.2019.02549), PRJNA428323.
- NCBI RefSeq genomes, Bakta annotation and MMseqs2 clustering.

The bundled REGULONDB_LICENSE.md records [official April 2025 CC-BY-NC-4.0 terms](https://github.com/regulondbunam/regulondb-docs/blob/main/00_about_policies/license.md). Preserve attribution/licence links and disclose sanitising, extraction, deduplication/clustering and mixed-label processing. Academic/noncommercial sharing/adaptation is permitted; commercial data use needs an agreement. Model/source-code licences remain separate. This package contains fitted parameters/helpers, not raw database exports.
