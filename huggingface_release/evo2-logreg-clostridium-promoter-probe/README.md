---
license: cc-by-4.0
base_model: arcinstitute/evo2_7b
tags:
  - biology
  - genomics
  - dna
  - promoter-prediction
  - clostridium
  - evo2
  - probe
  - igem
---

# Evo2-7B embedding probe — Clostridium promoter classifier (HKUST iGEM 2026)

A lightweight **logistic-regression probe on frozen [Evo2-7B](https://github.com/ArcInstitute/evo2)
embeddings** that scores whether an 81 bp DNA window is a σ<sup>A</sup> constitutive promoter. The
Evo2 backbone is **never fine-tuned** — this repo publishes only the ~116 KB scikit-learn probe the
team trained on top of Evo2's embeddings. It was the strongest **genome-scale** member of the HKUST
iGEM 2026 ensemble (fewest false-positive peaks), despite the lowest balanced score.

> This repo does **not** contain Evo2 weights. Evo2 (Apache-2.0, Arc Institute) is a multi-GB model
> distributed separately on [HuggingFace](https://huggingface.co/arcinstitute) / NVIDIA NGC — install
> it yourself. The probe here is CC-BY-4.0; using it requires Evo2 under its own Apache-2.0 license.

## How it works

1. Take an **81 bp** window (A/C/G/T, +1 TSS framed at 0-based offset 60).
2. Run it through frozen **Evo2-7B** (`evo2_7b`, StripedHyena-2) and **mean-pool** the per-nucleotide
   embeddings from layer **`blocks.28.mlp.l3`** → a **4096-d** vector.
3. Apply this probe (a scikit-learn `Pipeline` of `StandardScaler` → `LogisticRegression`) to get
   `P(promoter)`.

`apply_probe.py` in this repo does all three steps. Minimal version:

```python
import joblib, numpy as np
pipe = joblib.load("evo2_promoter_classifier.pkl")   # StandardScaler -> LogisticRegression, expects (N, 4096)

# X = mean-pooled Evo2-7B blocks.28.mlp.l3 embeddings for your 81 bp windows, shape (N, 4096)
proba = pipe.predict_proba(X)[:, 1]
```

**Reproduce the embeddings** exactly (layer / pool / dtype) with the team's helper:

```python
from evo2 import Evo2
from evo2_common import embed_sequences   # bundled in this repo
model = Evo2("evo2_7b")
X = embed_sequences(model, seqs, layer="blocks.28.mlp.l3", pool="mean")  # (N, 4096)
```

## Files in this repo

- `evo2_promoter_classifier.pkl` — the trained probe (scikit-learn Pipeline, ~116 KB)
- `evo2_common.py` — batched Evo2 embedding helper (`embed_sequences`, `revcomp`)
- `apply_probe.py` — end-to-end: sequences → Evo2 embeddings → `P(promoter)`

> **Pin scikit-learn.** The probe was saved with scikit-learn **1.9.0**; load it with a matching
> version to avoid `InconsistentVersionWarning` / silent breakage.

## Training data

The probe is fit on frozen Evo2 embeddings of the project's cross-species split (`T15_xs`): **50,190
windows, ~50/50**, MMseqs2-clustered at 0.6 identity, cluster-disjoint splits. Positives = in-genus
empirical TSS (*C. acetobutylicum* / *C. beijerinckii*, Hocq et al. 2022) + down-weighted housekeeping
upstreams + cross-species RegulonDB / DBTBS / *B. licheniformis* / *Ruminiclostridium*. Negatives =
CDS / intergenic / downstream / SRS. Target *C. butyricum* DSM 10702 and validation genomes
*C. ljungdahlii* / *C. autoethanogenum* held out. The probe is fit with the T15 per-sample `weight`
(silver 0.3 × class-balance), auto-selected (logreg vs random forest) on weighted validation AUPRC —
**logreg won**.

## Evaluation

| Regime | Metric | Score |
|---|---|---|
| Balanced 81 bp test (`T15_xs`, weighted AUPRC) | AUPRC | 0.703 *(lowest in ensemble)* |
| Genome scale — held-out *C. ljungdahlii* (restricted, step 10) | precision @ recall 0.1 / 0.5 | **71.0% / 58.0%** |
| Genome scale — held-out *C. autoethanogenum* (restricted, step 10) | precision @ recall 0.1 / 0.5 | **85.8% / 71.4%** |

This is the project's sharpest "balanced ≠ deployment" case: a frozen foundation-model embedding is
worst on the balanced test yet **best at genome scale** (fewest false-positive peaks) on both held-out
genomes. Judge by the number of false positives at genome scale, not the balanced AUPRC.

## Limitations

- **Requires Evo2-7B at inference** — you must install Evo2 and reproduce the exact embedding recipe
  (layer `blocks.28.mlp.l3`, mean pool, 81 bp). A different layer/pool/window will not match this
  probe. Evo2 inference needs a capable GPU; flash-attn must match the env's torch build.
- No bacterial promoter predictor achieves high genome-wide precision in AT-rich DNA; treat outputs as
  an enriched candidate list for wet-lab validation.
- σ<sup>A</sup> promoters, 81 bp / TSS-at-60 framing only.

## Citation & attribution

Cite Evo2 (Brixi et al., Arc Institute; Apache-2.0) and this project. Training data: RegulonDB (CC-BY),
DBTBS, and published TSS datasets. HKUST iGEM 2026 dry-lab promoter-mining project.
