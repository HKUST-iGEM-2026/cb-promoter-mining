# Project Description

> **Standard URL:** `2026.igem.wiki/hkust/description`
> **Medal relevance:** Bronze (Project Description)

## Why promoters, and why they matter

A **promoter** is the short DNA sequence just upstream of a gene that the cell's RNA polymerase
recognizes as "start transcribing here." It sets *whether* and *how strongly* a gene is expressed. In
synthetic biology, promoters are the volume knobs: a well-stocked collection of promoters of different
strengths is what lets you tune a pathway, balance a genetic circuit, or express a therapeutic protein
at the right level.

In bacteria, the most common promoters are read by the housekeeping sigma factor **σ<sup>A</sup>**
(σ<sup>70</sup> in *E. coli*). They are marked by two short conserved boxes upstream of the
transcription start site (TSS): a **−35 box** (consensus TTGACA) and a **−10 box** (consensus TATAAT),
separated by a spacer. Recognizing that two-box architecture is the biological problem at the heart of
our project.

## Why *Clostridium butyricum*

*Clostridium butyricum* is a Gram-positive, spore-forming, strictly anaerobic bacterium that is
genuinely useful:

- **Medicine:** it is a **probiotic** (marketed strains are used for gut-health and to treat
  antibiotic-associated diarrhoea) and is being explored as a living therapeutic and tumour-targeting
  delivery chassis.
- **Industry:** it is a natural **butyrate and butanol producer**, of interest for green chemical and
  biofuel production.

But engineering *C. butyricum* is held back by a **lack of characterized genetic parts**. Unlike
*E. coli*, it has essentially no curated promoter collection. Every team that wants to build circuits
in this organism has to rediscover its basic parts from scratch. We chose it as our **target** precisely
because closing this gap has real downstream value — and because doing so exposes a hard, general
computational problem worth solving.

We work with the type strain **DSM 10702**, mining the complete, reference-grade assembly
`GCF_014131795.1` — two chromosomes and two small plasmids, about 4.7 Mb in total.

## The core challenge: the "AT-trap"

*Clostridium* genomes are strikingly **AT-rich** — only about **28–32% G+C**. The promoter motifs
(−10 = TATAAT, −35 = TTGACA) are themselves AT-rich. That means AT-rich junk sequence *looks like* a
promoter to a naive scanner, so simple methods produce enormous numbers of **false positives**. A model
that scores 90% specificity on a balanced test can still predict *hundreds of thousands* of promoters
that do not exist when slid across a real genome, because real promoters are only ~0.2% of the genome.

Beating this trap — separating genuine promoters from AT-rich background — is what our whole design is
built around. See [Design](design.md).


![AT content of the genome, the null, and the promoter motifs](figures/F12_at_trap.png)

***Figure 1. The AT-trap.** The *C. butyricum* genome is about 71% A+T, and the promoter motifs are more AT-rich still — 85% for the −35 box and 88% for the −10. Motif and background overlap almost completely, so a scanner that keys on composition rather than motif drowns in false positives. Beating this is what the rest of the design is for.*

![Phylogeny of the 166 Clostridium training genomes](figures/F1_phylogeny.png)

***Figure 2.** The 166 *Clostridium* genomes behind the in-genus training data.*

## Our goals

1. **Stage 1 (this project): constitutive promoters.** Computationally mine the *C. butyricum* genome
   for σ<sup>A</sup> constitutive promoters and deliver a **ranked, statistically filtered candidate
   list** with the −10/−35 architecture annotated — ready for wet-lab characterization.
2. **Reusable method.** Build the pipeline so it generalizes to any AT-rich, non-model bacterium, and
   publish the models and code so future iGEM teams can reuse them.
3. **Wet-lab validation.** Characterize a shortlist of candidates experimentally to measure real
   promoter strength and close the design–build–test loop. *(Wet-lab plan — see [Notebook](notebook.md);
   content to be completed by the wet-lab team.)*
4. **Stage 2 (roadmap): inducible promoters.** Extend the approach from always-on (constitutive) to
   switchable (inducible) promoters, which are needed for controllable circuits. *(Early-stage.)*

## Project plan & roadmap

We ran the dry lab as a five-phase computational pipeline (a 7-week core sprint, then iterative
refinement). Each phase feeds the next:

| Phase | What it produces | Status |
|---|---|---|
| **A. Data acquisition & curation** | Genomes, annotations, and known promoters/TSS from *Clostridium* and AT-rich relatives | ✅ Complete |
| **B. Dataset construction** | Balanced, leakage-controlled training windows (positives + realistic negatives) | ✅ Complete |
| **C. Model training** | Three trained promoter classifiers | ✅ Complete |
| **D. Evaluation & calibration** | Honest performance across three test regimes; calibration; false-discovery control | 🟡 In progress |
| **E. Mining & annotation** | Genome-wide scan → peaks → motif-annotated candidate table | ✅ Delivered on the target |

Downstream of the dry lab:

- **Wet-lab validation** of the top candidates *(planned — wet-lab team)*
- **Stage 2: inducible promoters** *(roadmap)*
- **Iterative hard-negative mining** to further cut false positives *(planned refinement)*

The methodology, the biological rationale, and the honest limits are described in [Design](design.md)
and [Model](model.md); the deliverables in [Results](results.md).
