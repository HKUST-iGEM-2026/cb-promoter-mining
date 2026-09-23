# Design

> **Standard URL:** `2026.igem.wiki/hkust/design`

This page explains **how we designed** our promoter-mining solution and the reasoning behind each
choice. The full technical implementation is on the [Model](model.md) page; the iteration history is on
[Engineering](engineering.md).

## Design principle: don't trust one model — vote

No single DNA model is reliable enough to call promoters in AT-rich DNA on its own. So we built an
**ensemble of three deliberately different models** and combine their scores. Where independent models
*agree*, we have far more confidence than any one model alone. The three members are:

1. **A fine-tuned genomic language model (ProkBERT-mini).** A transformer pre-trained on prokaryotic
   DNA, fine-tuned by us to classify promoters. Brings learned "grammar" of bacterial sequence.
2. **A custom neural network built from scratch (a multi-encoding fusion CNN).** Reads the DNA through
   several complementary lenses (one-hot bases, local composition, di-/tri-nucleotide frequencies) and
   fuses them. Learns the motif directly and is our cleanest motif detector.
3. **A foundation-model probe (Evo2-7B).** A lightweight classifier on top of a frozen 7-billion-
   parameter DNA foundation model. Surprisingly, it is our **best model at genome scale** (see below).

We combine them by **mean rank-percentile**: each model ranks every window, we average the ranks, and
agreement rises to the top. Ranking is **calibration-invariant**, so the models combine cleanly without
needing identical score scales.

## Designing against the "AT-trap"

The central design challenge (see [Description](description.md)) is that AT-rich background sequence
mimics the promoter signal. Our defences, all evidence-based and drawn from the promoter-prediction
literature:

- **Realistic negatives, not random ones.** The single biggest lever. We train the models to reject the
  exact things that fool naive scanners:
  - **Shuffled real promoters (SRS):** real promoters with their nucleotides shuffled to preserve base
    composition but destroy the −10/−35 signal. This forces the model to learn *motif*, not *AT
    content*.
  - **Real genomic background:** coding-sequence, intergenic non-promoter, and downstream windows.
- **AT-rich cross-species training data.** We borrow verified promoters from AT-rich relatives
  (*Bacillus subtilis*, *B. licheniformis*) and from *E. coli* (RegulonDB) so the model generalizes,
  plus in-genus *Clostridium* transcription start sites.
- **Calibration for the genomic prior.** Balanced-test scores are misleading at genome scale; we
  calibrate scores against realistic AT content so a "high" score means something on the real genome.
- **Empirical false-discovery control.** We compare candidate scores against a null built from
  non-promoter genomic sequence and assign **q-values**, so the final list has a controlled
  false-discovery rate rather than an arbitrary threshold.
- **Motif confirmation.** Finally we look for the actual −10/−35 boxes in each candidate, so the
  shortlist is not just "high-scoring" but structurally plausible.

## Honest framing: an enriched candidate list, not a perfect map

A design decision as important as any model choice: **we frame the deliverable correctly.** No method
achieves high genome-wide *precision* for bacterial promoters — the −10/−35 motif is too
low-information in AT-rich DNA for that to be possible. Success is therefore defined as **strong
enrichment over random + a ranked, FDR-controlled candidate list + a high experimental hit-rate**, not
a flawless genome-wide promoter map. This keeps expectations honest and the wet-lab hand-off realistic.

## Keeping the evaluation honest

Good design is worthless if the evaluation lies to you. Two safeguards:

- **Cluster-aware data splits.** Near-identical sequences are grouped (by sequence-identity clustering)
  and whole groups are kept on one side of the train/validation/test split. Without this, near-
  duplicates leak across the split and inflate scores.
- **Held-out genomes.** The target *C. butyricum* is **never** in training. Two additional in-genus
  genomes (*C. ljungdahlii*, *C. autoethanogenum*) are held out entirely and used to measure real
  **genome-scale** performance — which is very different from balanced-test performance.

This distinction produced our most important scientific finding: **the model that looks best on a
balanced test is not the model that performs best on a real genome.** See [Model](model.md) and
[Engineering](engineering.md).

## Wet-lab validation design

The computational shortlist is designed to be **directly testable**. The intended experimental loop:

1. Select a spread of candidates across predicted strengths (including canonical −10/−35 hits and some
   controls).
2. Clone each candidate upstream of a reporter (e.g. a fluorescent protein) in a *C. butyricum*-
   compatible vector.
3. Measure expression to obtain real promoter strengths, and feed the results back as new training
   labels (closing the design–build–test–learn loop).

> **Placeholder — wet-lab team.** Vector choice, transformation protocol, reporter, and measurement
> method to be documented here and in the [Notebook](notebook.md).
