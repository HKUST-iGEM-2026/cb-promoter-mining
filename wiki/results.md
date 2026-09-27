# Results

> **Standard URL:** `2026.igem.wiki/hkust/results`

Two kinds of result: (1) how well the models perform, measured honestly, and (2) the actual promoter
candidates we mined from *C. butyricum*.

---

## 1. Model performance


![Balanced AUPRC against genome-scale precision for six models](figures/F13_balanced_vs_genome.png)

***Figure 1. The project's central finding.** Balanced-test accuracy is close to an exact inversion of genome-scale precision. NTv3 has the best balanced score of anything we trained and the worst deployment score; the Evo2 probe is the reverse. Rank deployment models on the genome-scale metric, never on the balanced one. (NTv3 reads 384 bp of context, so its balanced number is not strictly comparable to the others.)*

### Balanced test (Regime A)

On the standard balanced test (weighted AUPRC, higher is better):

| Model | Balanced test AUPRC |
|---|---|
| ProkBERT-mini (fine-tuned) | 0.884\* |
| Fusion CNN | 0.726 |
| Evo2-7B probe | 0.703 |

\* Measured on the previous ProkBERT checkpoint, trained without the two cross-species TSS sets. No
balanced test score was saved for the deployed checkpoint.

Taken alone this would rank Evo2 last — but the balanced test is **not** how the models behave on a real
genome.

### Genome scale (Regime C) — the metric that matters

Slid across a **held-out** in-genus genome, we measure **precision at a fixed recall** (of the true
transcription start sites). Higher precision = fewer false-positive peaks for the same number of real
promoters recovered. Two validation genomes:

**Validation genome 1 (*C. ljungdahlii*)** — precision at recall 0.1 / 0.5, all from one run:

| Model | @ recall 0.1 | @ recall 0.5 |
|---|---|---|
| **3-model ensemble** | **74.4%** | **61.6%** |
| Evo2 probe | 69.4% | 58.4% |
| ProkBERT | 64.4% | 47.7% |
| Fusion CNN | 64.4% | 47.2% |

**Validation genome 2 (*C. autoethanogenum*)** — precision at recall 0.1 / 0.5, all from one run:

| Model | @ recall 0.1 | @ recall 0.5 |
|---|---|---|
| **3-model ensemble** | 87.0% | **76.6%** |
| Evo2 probe | **87.6%** | 72.5% |
| Fusion CNN | 81.0% | 66.0% |
| ProkBERT | 76.7% | 64.6% |

![Ensemble against its three members on two validation genomes](figures/F14_ensemble_vs_members.png)

***Figure 2.** The ensemble is best or tied-best on both validation genomes. Absolute precision is not comparable between the two panels — *C. autoethanogenum* has a denser truth catalogue — so only the within-panel ordering carries meaning.*

**Key results:**

- The **3-model ensemble is the strongest caller at 50% recall on both validation genomes**, and within
  a point of the best single model at 10% recall. Combining diverse models works.
- The **Evo2 probe (worst on the balanced test) is the best single model at genome scale.** This is our
  headline scientific finding: *judge deployment models by the genome-scale metric.*
- We confirmed that adding a weaker, correlated model **dilutes** the ensemble, which is why we deploy
  exactly these three.

*(The validation genomes' mapped start sites were never used in training, though both genomes are
among the 166 that supplied inferred training examples. Coordinate/strand handling was audited for
correctness — see [Model](model.md).)*

---

## 2. The *C. butyricum* promoter candidates

We ran the validated 3-model ensemble on the target genome, *C. butyricum* DSM 10702, using the
complete reference-grade assembly `GCF_014131795.1`. Scanning its candidate regions, aggregating by
rank percentile, calling peaks and annotating motifs produced three nested tiers:

![Funnel from 179,172 scored windows to 45 canonical candidates](figures/F6_candidate_funnel.png)

***Figure 3. From genome to shortlist.** 179,172 scored windows narrow to 929 confident candidates. Note where the line falls: everything below 929 is a filter on *motif architecture*, not on score. That is why the robust deliverable is the 929 and its ranking, not the 45.*

| Tier | Count | What it is |
|---|---|---|
| **High-confidence candidates** | **929** | scored ≥ 0.9 by the ensemble with ≥ 5 supporting windows, and statistically significant against the null |
| …with a placed **−10 box** | 180 | promoter architecture confirmed by motif search |
| …**canonical** (both **−10 and −35**) | **45** | **the primary wet-lab shortlist** |

Every one of the 929 overlaps a significant peak at q ≤ 0.05, so the false-discovery step **validates
the whole set** rather than filtering it down. Because the top tier sits at the null's resolution
floor, the candidates are ranked by ensemble score, not by q-value.

Each candidate carries genomic coordinates, strand, predicted transcription start site, ensemble score,
number of supporting windows, the −10/−35 offsets and spacer, its q-value, and the 81 bp summit
sequence, in a fixed schema (TSV + JSONL).

> **On an earlier count.** A first mine used the older 96-contig draft assembly of the same strain and
> reported 912 / 306 / 100. The numbers above supersede it: same organism, better genome, and the two
> coordinate frames are not interchangeable.


![Distribution of ensemble peak scores across the 929 candidates](figures/F7_score_distribution.png)

***Figure 4.** The confident set does not hug the threshold: mean peak score is 0.980 against a 0.9 gate. Canonical calls are spread across the range rather than concentrated at the top, which is another reason canonical status is not a quality ranking.*

### Reading the motif results

Among the confident calls, a **−10-only** architecture is far more common than a canonical pair. This is
**expected and biologically sensible** in an AT-rich genome: the −10 box is the more conserved,
information-rich element, while the −35 is weaker and more variable. A follow-up motif audit confirmed
this directly — the −10 boxes carry strong, clean signal, while the −35 boxes are barely distinguishable
from random AT-rich DNA.


![Sequence logos for the -10 and -35 boxes](figures/F8_motif_logos.png)

***Figure 5.** The −10 box recovers a crisp `TATAAT` (7.75 bits of information); the −35 is visibly degenerate (5.19 bits). We have not cleaned this up — the contrast *is* the result.*

![Motif log-odds against a matched AT-rich null](figures/F2_signal_vs_null.png)

***Figure 6.** Scored against a matched AT-rich null, the −10 boxes sit at median percentile 98 while the −35 boxes sit at 52 — indistinguishable from noise.*

![Motif architecture of the 929 and spacer geometry of the 45](figures/F9_architecture_spacer.png)

***Figure 7.** Left: only 45 of the 929 carry both boxes. Right: those 45 have −35→−10 spacers from 5 to 34 bp around a median of 18. A median of 18 rules out a gross coordinate offset; the spread is why canonical status is a soft annotation.*

So we treat "canonical" as a **soft annotation, not a quality gate**: the 45 canonical candidates are the
top-priority picks, but the robust deliverable is the full 929 and its score ranking, which never depends
on the −35 call.

---

## What the results mean

We set out to produce, from genome sequence alone, an **enriched, ranked, statistically filtered and
motif-annotated shortlist of promoter candidates** for a non-model, AT-rich chassis — and we did, with a
validated ensemble that matches or beats every individual model on both validation genomes. The final proof is
experimental: characterizing these candidates in the lab (see [Proof of Concept](proof-of-concept.md)
and [Notebook](notebook.md)).
