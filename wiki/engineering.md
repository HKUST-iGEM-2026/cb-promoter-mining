# Engineering Success

> **Standard URL:** `2026.igem.wiki/hkust/engineering`
> **Medal relevance:** Silver (Engineering Success)

Our project is a computational engineering project, and we advanced it through repeated
**Design–Build–Test–Learn (DBTL)** cycles. Each cycle produced a concrete measurement that changed the
next design. The most important cycles:

---

## Cycle 1 — "Why does an off-the-shelf promoter model fail on a real genome?"

- **Design.** Take an existing, published promoter classifier and slide it across a genome to find
  promoters. Simple, and it should just work.
- **Build.** We ran a pre-trained ProkBERT promoter classifier over the *E. coli* genome as a sanity
  check.
- **Test.** On a **balanced** test it worked well, but slid across the genome it scored **essentially at
  random** (recall ~0.24, precision ~0.02 — barely above chance).
- **Learn.** This is expected, not a bug. Genome-wide mining is a *different task* from balanced
  classification: real promoters are ~0.2% of windows, and the model had **never seen realistic
  negatives**. Two design requirements fell out of this: (1) train on realistic, promoter-like
  negatives, and (2) always judge by **genome-scale** metrics, never balanced accuracy.

## Cycle 2 — Engineering the training set to beat the AT-trap

- **Design.** Build a dataset that teaches "motif, not AT content."
- **Build.** We assembled positives from in-genus *Clostridium* TSS plus AT-rich cross-species
  promoters, and four classes of negatives — coding, intergenic, downstream, and **shuffled real
  promoters (SRS)** that keep base composition but destroy the −10/−35 signal. We fine-tuned our models
  on this set.
- **Test.** Balanced test performance was strong (AUPRC up to ~0.88).
- **Learn.** Necessary but not sufficient: a strong balanced score does not prove genome-scale
  usefulness. We needed a genome-scale test on **validation genomes**.


![Balanced AUPRC against genome-scale precision](figures/F13_balanced_vs_genome.png)

***Figure 1.** The measurement that forced Cycle 3: the balanced ranking and the genome-scale ranking are close to inverses of each other.*

## Cycle 3 — Discovering that "best on the test" ≠ "best on the genome"

- **Design.** Evaluate every model on held-out in-genus genomes (*C. ljungdahlii*,
  *C. autoethanogenum*) by sliding it across the genome and counting real false positives.
- **Build & Test.** The rankings **flipped** depending on the metric:
  - A Nucleotide-Transformer model had the 2nd-best balanced score but **collapsed at genome scale**.
  - A frozen **Evo2-7B** foundation-model probe had the **worst** balanced score yet was the **best at
    genome scale** on both validation genomes.
- **Learn.** This is our central scientific lesson: **select deployment models by the genome-scale
  metric, not the balanced one.** It reshaped our ensemble choice (see Cycle 5).

## Cycle 4 — Killing data leakage, then re-tuning

- **Design.** A reviewer flagged that our sequence-clustering threshold (0.9 identity) might let near-
  duplicate sequences leak across train/test and inflate scores. Tighten it to 0.6.
- **Build.** Re-clustered at 0.6 identity (switching to a tool that clusters DNA that divergent),
  rebuilt the splits, retrained all models, and **re-ran the hyperparameter sweep on the new data**.
- **Test.** Balanced scores barely moved (so the earlier numbers were *not* leakage-inflated), but the
  genome-scale ranking shifted. A ProkBERT genome-scale dip turned out to be **mostly stale
  hyperparameters**, not the stricter clustering — re-tuning recovered most of it.
- **Learn.** Leakage control and honest re-tuning matter more than chasing the balanced number. Always
  re-tune after changing the data.

## Cycle 5 — A better CNN, and choosing the final ensemble

- **Design.** Improve the from-scratch model and decide which models actually go into deployment.
- **Build.** We engineered a **multi-encoding fusion CNN** (one-hot + positional composition + di-/tri-
  nucleotide branches) and benchmarked all candidate models on the **same** genome-scale test.
- **Test.** The fusion CNN became our best CNN and cleanest motif detector. On a unified genome-scale
  benchmark the stable ranking was **Evo2 > ProkBERT ≈ fusion-CNN > (older CNN, NT models)**.
- **Learn.** Keep three **diverse, individually strong** models — a foundation-model probe, a
  transformer, and a fusion CNN — and drop the weaker, correlated ones. Adding a weak correlated model
  *diluted* the ensemble, which we confirmed empirically.

## Cycle 6 — Mining the target and closing to a shortlist

- **Design.** Run the finished 3-model ensemble on *C. butyricum* and produce a wet-lab-ready list.
- **Build.** Scanned the genome → aggregated by rank-percentile → called peaks → confirmed −10/−35
  motifs → wrote a candidate table.
- **Test.** On our first pass, over a fragmented draft assembly, ~174,000 candidate windows collapsed to
  912 high-confidence candidates. We then re-ran the entire mine on the **complete reference assembly**
  of the same strain, which is the delivered result: **929 high-confidence candidates**, **45** of them
  with the full canonical two-box architecture.
- **Learn.** Comparing the two mines was itself informative. The confident set barely moved
  (912 → 929), but the motif tiers shifted sharply (306 → 180 with a −10 box; 100 → 45 canonical).
  The **ensemble score ranking is robust**; the motif labels depend on a motif rediscovered from
  scratch on each run. So we rank on score and treat "canonical" as a soft annotation, not a filter.
- **Learn.** The end-to-end pipeline works and generalizes; the remaining validation is experimental
  (wet lab), and the next refinement is **iterative hard-negative mining** to cut false positives
  further. See [Results](results.md).

---

**Engineering takeaway.** The project's success came from *measurement-driven iteration*: every design
change was forced by a concrete test result, and the biggest wins (realistic negatives, genome-scale
evaluation, leakage control, deployment-metric model selection) came from **testing honestly rather than
optimizing the easy number.**
