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

# Multi-encoding fusion CNN — Clostridium promoter classifier (HKUST iGEM 2026)

A 1D CNN **trained from scratch** that scores whether an 81 bp DNA window is a σ<sup>A</sup>
constitutive promoter. It fuses four DNA representations, and was the strongest CNN in the HKUST
iGEM 2026 promoter-mining ensemble, alongside
[`CloBiT/prokbert-mini-clostridium-promoter`](https://huggingface.co/CloBiT/prokbert-mini-clostridium-promoter)
and
[`CloBiT/evo2-logreg-clostridium-promoter-probe`](https://huggingface.co/CloBiT/evo2-logreg-clostridium-promoter-probe).

Because it is trained from scratch, the weights are entirely the team's own work — this repo is
released under a permissive **CC-BY-4.0** license (attribution only). Attribute the training data
sources (below) if you build on it.

## Architecture

Four parallel encoder branches whose features are concatenated and passed to a fusion head
(`MultiEncodingPromoterCNN` in `train_cnn_v2.py`):

- **one-hot** (4×L) → 3-layer 1D CNN (kernels 7/5/3, BatchNorm, MaxPool, AdaptiveMaxPool)
- **ENAC** positional composition (4×L, window 5) → 3-layer 1D CNN
- **dinucleotide** frequency (16-d) → MLP
- **trinucleotide** frequency (64-d) → MLP

Deployed configuration: filters 64/128/256, fusion dim 256, BatchNorm on, all four encodings active.
The convolutional branches use `AdaptiveMaxPool1d(1)`, so the model accepts any window length, but it
was trained and is deployed on **81 bp** windows.

## Intended use & input convention

Binary `P(promoter)` for a fixed **81 bp** A/C/G/T window, with the +1 TSS framed at 0-based offset 60
(1-based 61) — the same framing as the rest of the ensemble. Designed for sliding-window genome mining
that produces an enriched, ranked candidate list for experimental validation, **not** a precise
promoter map (see *Limitations*).

## Files in this repo

| File | What it is |
|---|---|
| `best_model.pt` | PyTorch `state_dict` of the deployed checkpoint |
| `config.json` | architecture + active encodings; `build_model_from_config` reads this |
| `train_cnn_v2.py` | model class, feature pipeline and `predict_proba` — needed to load and run |
| `metrics.json` | the deployed run's validation and test scores |
| `history.json` | per-epoch training history |
| `SOURCE_CODE_LICENSE` | root MIT copyright/license notice for bundled project source, separate from model-weight licensing |
| `REGULONDB_LICENSE.md` | training-source data license, attribution and processing changes; separate from model-weight licensing |

Because the feature pipeline lives in `train_cnn_v2.py`, download the repo and import from it rather
than loading the weights alone:

```python
import json, sys, torch
from huggingface_hub import snapshot_download

path = snapshot_download("CloBiT/cnn-fusion-clostridium-promoter")
sys.path.insert(0, path)
from train_cnn_v2 import build_model_from_config, predict_proba

cfg = json.load(open(f"{path}/config.json"))
model = build_model_from_config(cfg, device="cpu")
model.load_state_dict(torch.load(f"{path}/best_model.pt", map_location="cpu"))

seqs = ["A" * 81]  # your 81 bp windows, TSS at offset 60
proba = predict_proba(model, seqs, device="cpu")  # feature pipeline is byte-identical to training
print(proba)
```

## Training data

Trained on the project's cross-species-augmented split (`T15_xs`): **50,190 windows, ~50/50
positive/negative**, split **44,085 train / 3,052 validation / 3,053 test**. Sequences were clustered
with MMseqs2 at **0.6 identity** and the split is cluster-disjoint, so near-duplicates never straddle
train/val/test. Cross-species sources are **pinned to train**, so validation and test are in-genus only.

### Where the 25,095 positives come from

| Source | Windows | Species | Kind |
|---|---:|---|---|
| Hocq et al. 2022 published TSS | 5,506 | *C. acetobutylicum* ATCC 824, *C. beijerinckii* NCIMB 8052 / DSM 6423 | in-genus, verified |
| Housekeeping-gene upstreams | 13,838 | in-genus, across the 166 genomes | **silver** (inferred), weight 0.3 |
| RegulonDB | 3,827 | *E. coli* K-12 | cross-species, curated |
| *Bacillus licheniformis* published TSS | 888 | *B. licheniformis* | cross-species — **see erratum** |
| *Ruminiclostridium cellulolyticum* H10 published TSS | 663 | ex-*Clostridium* relative | cross-species |
| DBTBS | 373 | *B. subtilis* | cross-species, curated |

19,344 of the positives are in-genus. The counts above were read from the split itself, grouped by
`source`; only the Hocq set appears in validation and test (4,412 / 547 / 547), every other source is
train-only.

**Silver positives** are upstream regions of housekeeping genes, inferred rather than measured: a −10-crop check found about a third do not frame a promoter, so they
carry weight 0.3 against 1.0 for verified promoters. At full weight they would teach the "AT-rich
means promoter" shortcut this project exists to avoid.

Housekeeping genes were selected by Bakta **product description** (not gene symbol, which the light
Bakta database leaves sparse), excluding regulated genes such as sigma factors, heat-shock chaperones
and RecA. About 22,760 such genes across 166 genomes collapse to 14,659 operon leaders; the 300 bp
immediately 5′ of each leader is taken, and the split keeps the 81 bp nearest the gene.

> **Erratum.** The 888 *B. licheniformis* windows were cut from assembly `GCF_000011645.1`
> (`NC_006270.3`), which the published TSS coordinates do not fit — they match `GCF_000008425.1`
> (`NC_006322.1`), an offset of 147–196 bp. Those windows are therefore effectively random genomic
> sequence labelled positive, not promoters. They are 3.5% of the positives, are pinned to train, and
> appear in none of the evaluation data below.

### The negatives

Four classes, all windows of the same 81 bp geometry: **coding-sequence** interiors, **intergenic
non-promoter**, **downstream / 3′**, and **SRS** — dinucleotide-preserving shuffles of real promoters,
which keep base composition but destroy the −10/−35 signal. Composition-matched negatives are the
main defence against the AT-rich false-positive trap (*Clostridium* GC ~28–32%). SRS is training-only.

### Genomes

166 *Clostridium* RefSeq reference assemblies (NCBI taxon 1485), annotated with Bakta 1.11.4
(DB v6.0 light), supply the silver positives and the negative windows.

- **Target excluded:** *C. butyricum* DSM 10702 is in no training data, and no *C. butyricum* strain
  is among the 166.
- **Validation genomes:** the published start sites of *C. ljungdahlii* and *C. autoethanogenum* were
  never training positives, but both strains **are** among the 166, so they contributed silver
  positives and negative windows. The genome-scale numbers below are therefore not fully independent
  of training.

## Training procedure

Each window's `weight` column (silver 0.3 × class-balance) is applied as a **per-sample loss weight**;
the selection metric is weighted validation AUPRC, with early stopping on it. Optimizer AdamW with
cosine-annealing warm restarts. Hyperparameters come from an Optuna sweep and are the `config.json`
shipped here:

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

Three regimes, because a balanced test score does not predict genome-scale behaviour. Regime A is
this exact checkpoint's `metrics.json`.

| Regime | Metric | Score |
|---|---|---|
| **A** · in-genus test (`T15_xs`, weighted; no-skill ≈0.18) | AUPRC | 0.726 |
| **A** · same test set | AUROC / accuracy / F1 / MCC | 0.898 / 0.834 / 0.627 / 0.543 |
| **B** · composition-matched SRS discrimination\* | AUROC | **0.924** (best of the four fine-tuned models tested — cleanest motif signal) |
| **C** · genome scale, *C. ljungdahlii* (restricted, step 10) | precision @ recall 0.1 / 0.5 | 64.4% / 47.2% |
| **C** · genome scale, *C. autoethanogenum* (restricted, step 10) | precision @ recall 0.1 / 0.5 | 81.0% / 66.0% |

\* Measured on the earlier `T21_final` checkpoint (plain 0.6 split, swept hyperparameters, without
the two cross-species TSS sets). The Evo2 probe was never measured in this regime, so "best" covers
the four fine-tuned models only.
**Read the test score against its baseline.** Validation and test hold **only** verified in-genus
positives (547 of 3,052 and 547 of 3,053 rows, ~18%), because the silver and cross-species positives
are all pinned to train. The whole split is ~50/50, but val and test are not, and their weights are
left at natural prevalence. The no-skill AUPRC on this test set is therefore **≈0.18**, not 0.5.


**How Regime C is measured.** The model scores every window of a validation genome at step 10, but
only windows inside candidate promoter regions (300 bp upstream to 50 bp into each annotated gene, on
the gene's strand) count toward the result. A window within 40 bp of a published TSS on the same
contig and strand is a hit; recall is the fraction of that genome's published TSS hit at a given
threshold. Consecutive non-hit windows within 81 bp collapse into one false-positive **peak**, and
precision is hits / (hits + FP peaks). Truth sets: *C. ljungdahlii* DSM 13528 `GCF_000143685.1`
(`NC_014328.1`), 1,245 TSS from Al-Bassam et al. 2018; *C. autoethanogenum* DSM 10061
`GCF_000484505.1` (`NC_022592.1`), 2,885 TSS.

Absolute precision is **not** comparable between the two genomes — *C. autoethanogenum* has a denser
truth catalogue, so every model reads higher there. Only within-genome differences between models are
meaningful. At step 1 on *C. ljungdahlii* this model reaches 71.6% at recall 0.1. In the deployed
3-model ensemble (mean rank-percentile) the same scan reaches 74.4% / 61.6% and 87.0% / 76.6%.

Balanced-test accuracy far overstates genome-scale precision, because real promoters are ~0.2% of
genomic windows; judge deployment by the *number* of false-positive peaks, not the balanced score.

## Limitations

- No bacterial promoter predictor achieves high genome-wide precision in AT-rich DNA (*Clostridium*
  GC ~28–32%); outputs need wet-lab confirmation.
- Trained for σ<sup>A</sup> promoters at the 81 bp / TSS-at-60 framing; other σ factors, other window
  sizes or other framings are out of distribution.
- Verified in-genus training and evaluation data is thin, so in-genus performance carries wide error
  bars.
- 3.5% of the training positives are mislabelled (see the erratum) and the validation genomes
  contributed negatives and silver positives, so the reported genome-scale numbers are a slight
  over-estimate of performance on a truly unseen genome.
- No wet-lab validation of the predictions had been completed when this card was written.

## Citation & attribution

HKUST iGEM 2026 dry-lab promoter-mining project. Training and evaluation data — please cite these if
you build on this model:

- **Hocq R, Jagtap S, Boutard M, Tolonen AC, Duval L, Pirayre A, Lopes Ferreira N, Wasels F.**
  Genome-Wide TSS Distribution in Three Related Clostridia with Normalized Capp-Switch Sequencing.
  *Microbiol Spectr.* 2022;10(2):e0228821. doi:[10.1128/spectrum.02288-21](https://doi.org/10.1128/spectrum.02288-21)
  — the in-genus training positives.
- **Al-Bassam MM, Kim JN, Zaramela LS, Kellman BP, Zuniga C, Wozniak JM, Gonzalez DJ, Zengler K.**
  Optimization of carbon and energy utilization through differential translational efficiency.
  *Nat Commun.* 2018;9(1):4474. doi:[10.1038/s41467-018-06993-6](https://doi.org/10.1038/s41467-018-06993-6)
  — the *C. ljungdahlii* benchmark truth.
- **RegulonDB** v13.5 (Salgado H, Gama-Castro S, et al., *RegulonDB v12.0*, *Nucleic Acids Research*
  2023, doi:[10.1093/nar/gkad1072](https://doi.org/10.1093/nar/gkad1072)) — the
  [official April 2025 policy](https://github.com/regulondbunam/regulondb-docs/blob/main/00_about_policies/license.md)
  permits academic/noncommercial sharing and adaptation under
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), with attribution, license links
  and disclosure of changes.
- **DBTBS**, the *B. subtilis* transcriptional regulation database ([dbtbs.hgc.jp](https://dbtbs.hgc.jp)).
- **Xu C, Huang R, Teng L, et al.** Cellulosome stoichiometry in *Clostridium cellulolyticum* is
  regulated by selective RNA processing and stabilization. *Nat Commun.* 2015;6:6900.
  doi:[10.1038/ncomms7900](https://doi.org/10.1038/ncomms7900) — the *R. cellulolyticum* H10 TSS
  table (dRNA-seq, BioProject [PRJNA247408](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA247408)).
- **Wiegand S, Dietrich S, Hertel R, Bongaerts J, et al.** RNA-Seq of *Bacillus licheniformis*: active
  regulatory RNA features expressed within a productive fermentation. *BMC Genomics.* 2013;14:667.
  doi:[10.1186/1471-2164-14-667](https://doi.org/10.1186/1471-2164-14-667) — the *B. licheniformis*
  DSM 13 TSS table (supplementary Table S5; dRNA-seq BioProject
  [PRJNA188930](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA188930)). This is the `Wiegand2013`
  source tag in the split.
- **de Souza Pinto Lemgruber R, Valgepea K, Gonzalez Garcia RA, de Bakker C, et al.** A TetR-Family
  Protein (CAETHG_0459) Activates Transcription From a New Promoter Motif Associated With Essential
  Genes for Autotrophic Growth in Acetogens. *Front Microbiol.* 2019;10:2549.
  doi:[10.3389/fmicb.2019.02549](https://doi.org/10.3389/fmicb.2019.02549) — the *C. autoethanogenum*
  benchmark truth (dRNA-seq BioProject
  [PRJNA428323](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA428323)); evaluation only, never training.
- Genome assemblies from **NCBI RefSeq**; annotation with **Bakta** (Schwengers et al.); clustering
  with **MMseqs2** (Steinegger & Söding).

## RegulonDB data licensing

See the [bundled data notice](https://huggingface.co/CloBiT/cnn-fusion-clostridium-promoter/blob/main/REGULONDB_LICENSE.md)
and [official policy](https://github.com/regulondbunam/regulondb-docs/blob/main/00_about_policies/license.md).
Noncommercial reuse, sharing and adaptation of RegulonDB content are permitted under **CC BY-NC 4.0**;
commercial data use requires a separate agreement. Attribute RegulonDB and the original publications,
link the license and identify changes. Training preparation normalised/filtered sequences, prepared
81 bp windows, deduplicated/clustered them and combined sources and labels. This package contains fitted
parameters/helpers, not database exports or training tables. The fitted-weight **CC-BY-4.0** and
separate source-code MIT licenses remain unchanged; they do not relicense the upstream data or
establish unrestricted commercial clearance.
