# Proof of Concept

> **Standard URL:** `2026.igem.wiki/hkust/proof-of-concept`

Does the pipeline actually work? We built our evidence in layers, from computational validation to the
(planned) experimental test.

## Layer 1 — The method recovers known promoters on genomes it has never seen

The strongest computational evidence: on **two in-genus genomes held completely out of training**
(*C. ljungdahlii*, *C. autoethanogenum*), the ensemble recovers the genomes' real, experimentally-mapped
transcription start sites while producing **fewer false-positive peaks than any single model**. Because
these genomes were never in training, this is a genuine test of generalization — and the ensemble is the
best caller on **both**. See [Results](results.md).

## Layer 2 — The ensemble beats its own members

The 3-model ensemble outperforms every individual model at genome scale on both held-out genomes. This
confirms the core design hypothesis — that **combining diverse models by agreement** yields a better,
more robust caller than any one model — rather than us simply picking a lucky single model.

## Layer 3 — The target mine produces biologically plausible promoters

Run on *C. butyricum*, the pipeline yields **100 canonical candidates** carrying the correct σ<sup>A</sup>
**−10/−35 architecture**, plus a broader confident set. The distribution of motif types is exactly what
promoter biology predicts for an AT-rich genome (the conserved −10 box dominates), which is a sanity
check that the models are reading **real motif signal**, not AT-content noise.

## Layer 4 — Honest false-discovery control

Rather than an arbitrary cutoff, candidates are compared against a null built from non-promoter genomic
sequence and assigned **q-values**, so the shortlist has a controlled, reported false-discovery rate.
This is the standard the field requires, because in any genome some sequences resemble promoters by
chance.

## Layer 5 — Experimental validation (planned)

The final proof of concept is wet-lab characterization: cloning a spread of candidates upstream of a
reporter in *C. butyricum* and measuring real expression strength. A high experimental hit-rate on the
canonical shortlist would close the loop and turn predicted candidates into characterized parts.

> **Placeholder — wet-lab team.** Experimental design is outlined on the [Design](design.md) page;
> results and protocols to be documented here and in the [Notebook](notebook.md).

---

**Summary.** Computationally, the concept is proven: a validated ensemble generalizes to unseen genomes,
beats its members, and yields motif-plausible, FDR-controlled candidates. The experimental validation is
the next milestone.
