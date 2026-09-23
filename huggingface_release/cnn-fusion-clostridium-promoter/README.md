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
iGEM 2026 promoter-mining ensemble (with a fine-tuned ProkBERT-mini and an Evo2-7B embedding probe).

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

Default filters 64/128/256, fusion dim 256, dropout 0.3, BatchNorm on. The convolutional branches use
`AdaptiveMaxPool1d(1)`, so the model accepts any window length, but it was trained and is deployed on
**81 bp** windows.

## Intended use & input convention

Binary `P(promoter)` for a fixed **81 bp** A/C/G/T window, with the +1 TSS framed at 0-based offset 60
(1-based 61) — the same framing as the rest of the ensemble. Designed for sliding-window genome mining
that produces an enriched, ranked candidate list for experimental validation, **not** a precise
promoter map (see *Limitations*).

## Files in this repo

- `best_model.pt` — PyTorch `state_dict`
- `config.json` — architecture / active encodings (`build_model_from_config` reads this)
- `train_cnn_v2.py` — model class + feature pipeline + `predict_proba` (needed to load and run)

```python
import json, torch
from train_cnn_v2 import build_model_from_config, predict_proba

cfg = json.load(open("config.json"))
model = build_model_from_config(cfg, device="cpu")
model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))

seqs = ["A" * 81]  # your 81 bp windows, TSS at offset 60
proba = predict_proba(model, seqs, device="cpu")  # feature pipeline is byte-identical to training
print(proba)
```

## Training data

Trained on the project's cross-species-augmented split (`T15_xs`): **50,190 windows, ~50/50
positive/negative**, MMseqs2-clustered at 0.6 identity and split cluster-disjoint.

- **Positives (25,095):** in-genus empirical TSS (*C. acetobutylicum*, *C. beijerinckii*; Hocq et al.
  2022) + down-weighted housekeeping upstreams; cross-species *E. coli* RegulonDB (3,827),
  *B. subtilis* DBTBS (373), *B. licheniformis* (888), *Ruminiclostridium cellulolyticum* (663).
- **Negatives (four classes):** CDS, intergenic non-promoter, downstream/3′, and SRS
  (dinucleotide-shuffled real promoters).
- **Held out entirely:** target *C. butyricum* DSM 10702; validation genomes *C. ljungdahlii*,
  *C. autoethanogenum*.

Each window's training `weight` (silver 0.3 × class-balance) is applied as a per-sample loss weight;
selection metric is weighted AUPRC. Optimizer AdamW, cosine-annealing-warm-restarts, early stopping on
val AUPRC.

## Evaluation

| Regime | Metric | Score |
|---|---|---|
| Balanced 81 bp test (`T15_xs`, weighted AUPRC) | AUPRC | 0.848 |
| Composition-matched SRS discrimination | AUROC | **0.924** (best in ensemble — cleanest motif signal) |
| Genome scale — held-out *C. ljungdahlii* (restricted, step 10) | precision @ recall 0.1 / 0.5 | 64.4% / 47.2% |
| Genome scale — held-out *C. autoethanogenum* (restricted, step 10) | precision @ recall 0.1 / 0.5 | 81.0% / 66.0% |

## Limitations

- No bacterial promoter predictor achieves high genome-wide precision in AT-rich DNA (*Clostridium*
  GC ~28–32%); outputs need wet-lab confirmation.
- Trained for σ<sup>A</sup> promoters at the 81 bp / TSS-at-60 framing; other σ factors or framings are
  out of distribution.
- Verified in-genus evaluation data is thin, so in-genus performance carries wide error bars.

## Citation & attribution

HKUST iGEM 2026 dry-lab promoter-mining project. Training data: RegulonDB (CC-BY), DBTBS, and published
TSS datasets (Hocq et al. 2022 and others) — please cite those sources.
