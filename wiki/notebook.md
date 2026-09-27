# Notebook

> **Standard URL:** `2026.igem.wiki/hkust/notebook` (experiments/protocols may live at
> `.../experiments`)

## Dry-lab timeline

The computational work ran as a phased sprint followed by iterative refinement. High-level chronology:

| Period | Milestone |
|---|---|
| Setup | HPC access, shared software environments, and the workflow scaffold |
| Phase A | Collected 166 *Clostridium* genomes, re-annotated them, and curated promoter/TSS data from in-genus and AT-rich cross-species sources |
| Phase B | Built the balanced, leakage-controlled 81 bp training windows (positives + four negative classes incl. SRS) |
| Phase C | Trained the three models (fine-tuned ProkBERT, fusion CNN, Evo2 probe) |
| Phase D | Three-regime evaluation on validation genomes; discovered the balanced-vs-genome-scale gap; calibration & FDR |
| Refinement | Tightened data-leakage control (0.6 clustering), re-tuned hyperparameters, added the fusion CNN, selected the 3-model ensemble |
| Phase E | Mined *C. butyricum* → peak calling → motif annotation → final candidate table |
| Re-mine | Repeated the mine on the complete reference assembly and re-ran the false-discovery step on the same coordinates → the delivered 929 / 180 / 45 tiers |

> **Placeholder.** Add specific dates and per-week detail from the team's records.

## Wet-lab notebook

> **Placeholder — wet-lab team.** Day-by-day experimental log: strains, media, growth conditions,
> cloning steps, transformations, measurements, and outcomes (including what didn't work).

### Protocols
> **Placeholder.** Standard protocols to document, e.g.:
> - *C. butyricum* culturing under anaerobic conditions
> - Vector construction (promoter → reporter fusions)
> - Transformation / conjugation into *C. butyricum*
> - Reporter assay and promoter-strength measurement
> - Controls and replicates

## Design–Build–Test–Learn record

The iteration history (what we tried, measured, and changed) is written up narratively on the
[Engineering](engineering.md) page.
