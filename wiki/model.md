# Modeling

> **Standard URL:** `2026.igem.wiki/hkust/model`

This is the core of our dry lab: a five-phase computational pipeline that turns raw genomes into a
motif-annotated list of promoter candidates. Everything here is reproducible from our open code and
[published models](contribution.md).

---

## Overview: the pipeline

```
A. Data           B. Dataset          C. Training        D. Evaluation        E. Mining
   acquisition  →    construction   →    (3 models)   →    & calibration   →    & annotation
   genomes,          balanced,           ProkBERT,         3 test regimes,      scan → peaks →
   TSS, promoters    leakage-safe        fusion CNN,       calibration,         motif → ranked
                     windows             Evo2 probe        empirical FDR        candidate table
```

The target genome, ***C. butyricum* DSM 10702, is deliberately excluded from every training step** so
that mining it is a genuine prediction.

---

## Phase A — Data acquisition & curation

We gathered the raw material a promoter model needs:

- **Genomes.** 166 *Clostridium* genus genomes (the regions where in-genus promoters live), re-annotated
  consistently so gene boundaries are comparable.
- **Verified promoters / transcription start sites (TSS):**
  - *In-genus* (most valuable): empirical TSS from *C. acetobutylicum* and *C. beijerinckii*.
  - *Cross-species* (for generalization and AT-rich coverage): *E. coli* (RegulonDB), *Bacillus
    subtilis* (DBTBS), *B. licheniformis*, and *Ruminiclostridium cellulolyticum*.
- **Held-out data** for honest evaluation: synthetic promoters, an out-of-genus dRNA-seq dataset, and
  the in-genus genomes *C. ljungdahlii* and *C. autoethanogenum* (kept entirely out of training).

## Phase B — Dataset construction

We converted raw data into **balanced, leakage-controlled 81-base-pair windows**, each framed so the
transcription start site sits at a fixed position:

- **Positives:** verified promoters/TSS (above), plus down-weighted "silver" positives from
  housekeeping-gene upstream regions (weighted 0.3 so they can't dominate).
- **Negatives (four classes):** coding-sequence, intergenic non-promoter, downstream/3′, and **SRS**
  (shuffled real promoters — same composition, no motif). This is the key anti-AT-trap ingredient.
- **Leakage control:** sequences are clustered by identity and whole clusters are assigned to a single
  split, so near-duplicates never straddle train/validation/test. We tightened the clustering threshold
  to 0.6 identity to be conservative.
- **Result:** a deployed training set of ~50,000 windows, roughly 50/50 positive/negative, with the
  target and validation genomes held out.

## Phase C — The three models

| Model | Type | Role in the ensemble |
|---|---|---|
| **ProkBERT-mini (fine-tuned)** | Prokaryotic DNA language model (transformer, ~20 M params) | Learned sequence "grammar"; strong all-rounder |
| **Fusion CNN** | Custom neural net, trained from scratch | Cleanest motif detector; fuses 4 DNA encodings |
| **Evo2-7B probe** | Lightweight classifier on a *frozen* 7-billion-parameter DNA foundation model | **Best at genome scale** — fewest false positives |

Each model scores an 81 bp window for `P(promoter)`. We train all three on the same windows, applying
each window's confidence weight so the AT-trap guard (SRS + down-weighted silver labels) is respected.

### How they combine (ensemble)

We aggregate the three tracks by **mean rank-percentile**: every model ranks all windows, we average the
per-model ranks, and windows where the models *agree* rise to the top. Because ranking ignores absolute
score scale, the models combine cleanly without shared calibration — and we verified that adding a
weaker, correlated model actually *hurts*, which is why we settled on exactly these three.

## Phase D — Evaluation & calibration

We evaluate in **three regimes**, because a model can look very different in each:

1. **Balanced (Regime A):** the standard 50/50 test. Good for sanity, **misleading for deployment**.
2. **Composition-matched / cross-species (Regime B):** can the model tell a promoter from an AT-matched
   non-promoter, including on species it never trained on?
3. **Genome-scale (Regime C):** slide across a **held-out genome** and count real false positives at a
   given recall — the metric that actually predicts deployment behaviour.

We then **calibrate** scores against AT content (so a "high" score is meaningful on the real, ~0.2%-
prevalence genome) and apply **empirical false-discovery control**: candidate scores are compared to a
null built from non-promoter genomic windows, giving each candidate a **q-value**.

### The key modeling finding: balanced ≠ deployment

Our sharpest result is that **balanced-test ranking does not predict genome-scale ranking**:

- The **Evo2 probe** has the *worst* balanced score but the **best** genome-scale precision on *both*
  held-out genomes (fewest false-positive peaks).
- A **Nucleotide-Transformer** model has a near-top balanced score but **collapses** at genome scale.
- **ProkBERT** and the **fusion CNN** are strong co-leaders on the deployment metric.

This is why we choose deployment models by the **genome-scale** metric, and it is a transferable lesson
for anyone doing genome-wide prediction. (Illustrative genome-scale numbers are on the
[Results](results.md) page.)

## Phase E — Mining & annotation

Finally we point the finished ensemble at the target genome:

1. **Scan** the candidate regions of *C. butyricum* with all three models.
2. **Aggregate** by mean rank-percentile into one ensemble score per window.
3. **Call peaks:** collapse clusters of high-scoring windows into discrete promoter calls, keeping those
   with several supporting windows as "confident."
4. **Annotate motifs:** search each call for the σ<sup>A</sup> **−10 and −35 boxes** and record their
   positions and spacer, classifying the architecture (canonical / partial / none).
5. **Output** a ranked candidate table in a fixed schema (genomic coordinates, strand, predicted TSS,
   scores, and motif annotation) as TSV/JSONL.

See [Results](results.md) for what came out, and [Contribution](contribution.md) for the reusable code
and models.

---

## Reproducibility

The pipeline is orchestrated as a workflow (Snakemake) with pinned software environments, and every
model is trained from documented scripts. The three trained models are **published on Hugging Face** so
anyone can reproduce or reuse them — see [Contribution](contribution.md).
