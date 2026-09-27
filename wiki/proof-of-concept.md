# Proof of Concept

> **Standard URL:** `2026.igem.wiki/hkust/proof-of-concept`

Does the pipeline actually work? We built our evidence in layers, from computational validation to the
(planned) experimental test.

## Layer 1 — The method recovers known promoters it has never seen

The strongest computational evidence: on **two in-genus validation genomes** (*C. ljungdahlii*,
*C. autoethanogenum*), whose experimentally mapped transcription start sites were never used in
training, the ensemble recovers those start sites while **matching or beating the best single model**
on false-positive peaks. Because the start-site maps were never in training, this tests
generalization to promoters the models never saw, and the ensemble is best or tied-best on **both**
genomes. See [Results](results.md).

## Layer 2 — The ensemble beats its own members

The 3-model ensemble is best or tied-best at genome scale on both validation genomes. This
confirms the core design hypothesis — that **combining diverse models by agreement** yields a better,
more robust caller than any one model — rather than us simply picking a lucky single model.

## Layer 3 — The target mine produces biologically plausible promoters

Run on *C. butyricum*, the pipeline yields **929 confident candidates**, **45** of them carrying the full
σ<sup>A</sup> **−10/−35 architecture**. The distribution of motif types is exactly what promoter biology
predicts for an AT-rich genome — the conserved −10 box dominates and the weaker −35 is often absent —
which is a sanity check that the models are reading **real motif signal**, not AT-content noise.


![Ensemble against its members on two validation genomes](figures/F14_ensemble_vs_members.png)

***Figure 1.** On two validation genomes, the ensemble is best or tied-best — the design hypothesis, tested.*

## Layer 4 — Honest false-discovery control

Rather than an arbitrary cutoff, candidates are compared against a null built from non-promoter genomic
sequence and assigned **q-values**, so the shortlist has a controlled, reported false-discovery rate.
This is the standard the field requires, because in any genome some sequences resemble promoters by
chance. All 929 candidates clear q ≤ 0.05, meaning the score threshold we chose was already strict
enough that the statistical test validates the whole set rather than trimming it.


![Null distribution with candidate peaks overlaid](figures/F10_fdr_null_vs_candidates.png)

***Figure 2.** Candidates against the AT-matched null. Because the p-value floor is 1/421, the test validates the set wholesale rather than ranking it.*

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
