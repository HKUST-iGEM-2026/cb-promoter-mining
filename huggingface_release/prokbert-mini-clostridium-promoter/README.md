---
license: cc-by-nc-4.0
base_model: neuralbioinfo/prokbert-mini
tags:
  - biology
  - genomics
  - dna
  - promoter-prediction
  - clostridium
  - prokbert
  - igem
pipeline_tag: text-classification
library_name: transformers
---

# ProkBERT-mini — Clostridium promoter classifier (HKUST iGEM 2026)

A fine-tune of [`neuralbioinfo/prokbert-mini`](https://huggingface.co/neuralbioinfo/prokbert-mini)
that scores whether an 81 bp DNA window is a **σ<sup>A</sup> constitutive promoter**. Built for the
HKUST iGEM 2026 dry-lab project to mine promoters from the AT-rich genome of *Clostridium butyricum*
DSM 10702, and one of the three members of that project's deployment ensemble (with a multi-encoding
fusion CNN and an Evo2-7B embedding probe).

> ⚠️ **License is CC-BY-NC-4.0 (non-commercial).** This is inherited from the base model
> `prokbert-mini`, which is CC-BY-NC-4.0. A fine-tune is a derivative work, so it carries the same
> terms. Fine for academic / iGEM use; **not** for commercial use.

## Intended use

Binary classification of a fixed **81 bp** DNA window: `P(promoter)` vs background. It is designed to
be slid across a genome (with per-model calibration and empirical FDR downstream) to produce an
**enriched, ranked candidate list for experimental validation — not a precise genome-wide promoter
map.** See *Limitations*.

**Input convention.** 81 bp, A/C/G/T. The window is framed so the transcription start site (+1) sits
at 0-based offset 60 (1-based position 61); the −10 box then falls around offset ~50 and the −35 around
offset ~25. Match this framing at inference or scores will not mean what they did in training.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

REPO = "<HF_OWNER>/prokbert-mini-clostridium-promoter"
# The tokenizer is unchanged from the base model, so load it from there (this repo ships the
# fine-tuned weights + custom architecture, not a copy of the base's tokenizer):
tok = AutoTokenizer.from_pretrained("neuralbioinfo/prokbert-mini", trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(REPO, trust_remote_code=True).eval()

seq = "A" * 81  # your 81 bp window, TSS at offset 60
inputs = tok(seq, return_tensors="pt")
with torch.no_grad():
    p = torch.softmax(model(**inputs).logits, dim=1)[0, 1].item()
print(f"P(promoter) = {p:.3f}")
```

## Training data

Trained on the project's cross-species-augmented split (`T15_xs`): **50,190 windows, ~50/50
positive/negative**, clustered with MMseqs2 at 0.6 identity and split cluster-disjoint (near-duplicate
sequences never straddle train/val/test).

- **Positives (25,095):** in-genus empirical TSS from *C. acetobutylicum* and *C. beijerinckii*
  (Hocq et al. 2022) plus down-weighted (0.3) housekeeping-gene upstreams; and cross-species curated /
  empirical promoters — *E. coli* RegulonDB (3,827), *Bacillus subtilis* DBTBS (373),
  *B. licheniformis* (888), *Ruminiclostridium cellulolyticum* (663).
- **Negatives (four classes):** coding-sequence windows, intergenic non-promoter, downstream/3′, and
  **SRS** (dinucleotide-preserving shuffles of real promoters — same base composition, no −10/−35
  signal). Realistic negatives are the key defence against the AT-rich false-positive trap.
- **Held out entirely:** the target *C. butyricum* DSM 10702, and the validation genomes
  *C. ljungdahlii* and *C. autoethanogenum*.

Fine-tuning applies each window's training `weight` (silver 0.3 × class-balance) as a per-sample loss
weight. Base model `prokbert-mini`: MegatronBERT, 20.6 M params, 6-mer/shift-1 tokenizer.

## Evaluation

| Regime | Metric | Score |
|---|---|---|
| Balanced 81 bp test (`T15_xs`, weighted AUPRC) | AUPRC | **0.884** |
| Genome scale — held-out *C. ljungdahlii* (restricted scan, step 10) | precision @ recall 0.1 / 0.5 | 64.4% / 47.7% |
| Genome scale — held-out *C. autoethanogenum* (restricted scan, step 10) | precision @ recall 0.1 / 0.5 | 76.7% / 64.6% |

Balanced-test accuracy far overstates genome-scale precision because real promoters are ~0.2% of
genomic windows; judge deployment by the *number* of false-positive peaks, not the balanced score.

## Limitations

- **No bacterial promoter predictor achieves high genome-wide precision** — the −10/−35 motif is too
  low-information in AT-rich DNA (*Clostridium* GC ~28–32%). Treat outputs as an enriched candidate
  list requiring wet-lab (e.g. dRNA-seq TSS) confirmation.
- Verified in-genus evaluation data is thin (~hundreds of promoters), so in-genus gains are measured
  with wide error bars.
- Trained for σ<sup>A</sup> constitutive promoters and the 81 bp / TSS-at-60 framing; other σ factors,
  other window sizes, or other framings are out of distribution.

## Citation & attribution

Cite the base model and this project:

```bibtex
@article{ligeti2024prokbert,
  title={ProkBERT family: genomic language models for microbiome applications},
  author={Ligeti, Bal\'azs and Szepesi-Nagy, Istv\'an and Bodn\'ar, Babett and Ligeti-Nagy, No\'emi and Juh\'asz, J\'anos},
  journal={Frontiers in Microbiology}, volume={14}, year={2024}, doi={10.3389/fmicb.2023.1331233}
}
```

Training data: RegulonDB (CC-BY), DBTBS, and published TSS datasets (Hocq et al. 2022 and others) —
please cite those sources if you build on this model. HKUST iGEM 2026 dry-lab, promoter-mining project.
