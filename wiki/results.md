# Results

> **Standard URL:** `2026.igem.wiki/hkust/results`

Two kinds of result: (1) how well the models perform, measured honestly, and (2) the actual promoter
candidates we mined from *C. butyricum*.

---

## 1. Model performance

### Balanced test (Regime A)

On the standard balanced test (weighted AUPRC, higher is better):

| Model | Balanced test AUPRC |
|---|---|
| ProkBERT-mini (fine-tuned) | 0.884 |
| Fusion CNN | 0.848 |
| Evo2-7B probe | 0.703 |

Taken alone this would rank Evo2 last — but the balanced test is **not** how the models behave on a real
genome.

### Genome scale (Regime C) — the metric that matters

Slid across a **held-out** in-genus genome, we measure **precision at a fixed recall** (of the true
transcription start sites). Higher precision = fewer false-positive peaks for the same number of real
promoters recovered. Two independent held-out genomes:

**Held-out genome 1 (*C. ljungdahlii*)** — precision at recall 0.1 / 0.5:

| Model | @ recall 0.1 | @ recall 0.5 |
|---|---|---|
| **3-model ensemble** | **74.4%** | **61.6%** |
| Evo2 probe | 71.0% | 58.0% |
| ProkBERT | 64.4% | 47.7% |
| Fusion CNN | 64.4% | 47.2% |

**Held-out genome 2 (*C. autoethanogenum*)** — precision at recall 0.1 / 0.5:

| Model | @ recall 0.1 | @ recall 0.5 |
|---|---|---|
| **3-model ensemble** | **87.0%** | **76.6%** |
| Evo2 probe | 85.8% | 71.4% |
| Fusion CNN | 81.0% | 66.0% |
| ProkBERT | 76.7% | 64.6% |

**Key results:**

- The **3-model ensemble is the strongest caller on both held-out genomes** — better than any single
  model. Combining diverse models works.
- The **Evo2 probe (worst on the balanced test) is the best single model at genome scale.** This is our
  headline scientific finding: *judge deployment models by the genome-scale metric.*
- We confirmed that adding a weaker, correlated model **dilutes** the ensemble, which is why we deploy
  exactly these three.

*(All genome-scale numbers are on genomes held completely out of training, with coordinate/strand
handling audited for correctness — see [Model](model.md).)*

---

## 2. The *C. butyricum* promoter candidates

Running the validated 3-model ensemble on the target genome (*C. butyricum* DSM 10702), the funnel from
raw scan to shortlist:

| Stage | Count |
|---|---|
| Candidate windows scored | ~173,900 |
| Windows above the confidence threshold | 10,933 |
| Distinct promoter calls (peaks) | 3,167 |
| **High-confidence candidates** (≥5 supporting windows) | **912** |
| …with a correctly placed **−10 box** | 306 |
| …**canonical** (both **−10 and −35** boxes) | **100** |

These nested tiers are the deliverable:

- **912 high-confidence candidates** — the full confident set.
- **306 with promoter architecture** — those with a placed −10 box.
- **100 canonical candidates** — both −10 and −35 present: **the primary wet-lab shortlist.**

Each candidate carries genomic coordinates, strand, predicted transcription start site, ensemble score,
number of supporting windows, and the −10/−35 offsets and spacer, in a fixed schema (TSV + JSONL).

### Reading the motif results

Among the confident calls, a **−10-only** architecture is the most common. This is **expected and
biologically sensible** in an AT-rich genome: the −10 box is the more conserved, information-rich element,
while the −35 is weaker and more variable. The subset with **both** boxes correctly placed and spaced is
the most promoter-like, and is why the 100 canonical candidates form the top-priority list.

---

## What the results mean

We set out to produce, from genome sequence alone, an **enriched, ranked, statistically filtered and
motif-annotated shortlist of promoter candidates** for a non-model, AT-rich chassis — and we did, with a
validated ensemble that beats every individual model on held-out genomes. The final proof is
experimental: characterizing these candidates in the lab (see [Proof of Concept](proof-of-concept.md)
and [Notebook](notebook.md)).
