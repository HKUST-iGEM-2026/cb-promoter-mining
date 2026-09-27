# Results and findings

What we measured, what it means, and where the evidence lives.

---

## 1. The headline finding: balanced accuracy does not predict deployment

This is the project's sharpest result and the one that generalises. A model's rank on a balanced test
set tells you almost nothing about how it behaves scanning a real genome, because the two tasks
differ in base rate and in what the negatives look like.

We evaluate in **three regimes**, and a model can look very different in each.

| Regime | What it asks | Verdict it supports |
|---|---|---|
| **A · In-distribution** | separate promoters from curated negatives on the held-out split | sanity only, misleading for deployment |
| **B · Composition-matched** | separate promoters from AT-matched non-promoters, including unseen species | does it read the motif or the composition? |
| **C · Genome-scale** | slide across a validation genome, count real false positives at fixed recall | **the one that predicts deployment** |

### Regime A — balanced test

Weighted test AUPRC. The table mixes two splits, so read the ordering rather than small gaps:

- The four fine-tuned 81 bp models are the plain 0.6-split retrains (`T21_final`, 3,318-row test,
  967 positives, so a no-skill AUPRC of ≈0.29).
- Evo2 and NTv3 were measured on the deployed `xs` split, NTv3 on its 384 bp context build. That
  test set is 547 positives in 3,053 rows, a no-skill AUPRC of ≈**0.18**.

The two groups therefore sit against **different baselines**, which is most of why the `xs` numbers
look lower. Neither test set is 50/50: the whole split is, but validation and test hold only verified
in-genus positives, because silver and cross-species positives are pinned to train.

On the `xs` split the deployed CNN-v2 scores 0.726, per its `metrics.json`.

| Model | Balanced AUPRC |
|---|---|
| NTv3 with 384 bp context | 0.960 |
| ProkBERT-mini, fine-tuned | 0.884 |
| NT v2 250M LoRA | 0.854 |
| CNN-v2, fusion | 0.848 |
| CNN-v1 | 0.804 |
| **Evo2-7B frozen probe** | **0.703** |

Read alone, this ranks Evo2 last and NTv3 first. Both conclusions are wrong for deployment.

### Regime C — genome scale, the metric that matters

Precision at recall 0.1 / 0.5 on two validation genomes, restricted scan, matched step 10. Higher
precision means fewer false-positive peaks for the same number of real promoters recovered. Their
published start sites were never used in training, but both strains are among the 166 genomes that
supplied silver positives and negatives, so the benchmark is not fully independent of training
([PIPELINE.md](PIPELINE.md)). Every model shares this, so comparisons between them hold.

The ensemble rows come from the ensemble's own run, where Evo2 scored 69.4% / 58.4% on
*C. ljungdahlii* and 87.6% / 72.5% on *C. autoethanogenum*. The other rows come from separate
per-model scans. Within its own run, the ensemble is best at recall 0.5 on both genomes and 0.6
points behind Evo2 at recall 0.1 on *C. autoethanogenum*.

**Validation genome: *C. ljungdahlii***

| Model | @ recall 0.1 | @ recall 0.5 |
|---|---|---|
| **3-model ensemble** | **74.4%** | **61.6%** |
| Evo2 probe | 71.0% | 58.0% |
| ProkBERT | 64.4% | 47.7% |
| CNN-v2 | 64.4% | 47.2% |
| CNN-v1 | 49.0% | 40.3% |
| NT-LoRA | 43.9% | 36.1% |
| NTv3 context | 35.7% | 28.0% |

**Validation genome: *C. autoethanogenum***

| Model | @ recall 0.1 | @ recall 0.5 |
|---|---|---|
| **3-model ensemble** | **87.0%** | **76.6%** |
| Evo2 probe | 85.8% | 71.4% |
| CNN-v2 | 81.0% | 66.0% |
| ProkBERT | 76.7% | 64.6% |
| CNN-v1 | 72.6% | 63.0% |
| NT-LoRA | 64.9% | 58.7% |
| NTv3 context | 58.5% | 48.3% |

The ranking is **stable across both genomes**: Evo2 > ProkBERT ≈ CNN-v2 > CNN-v1 > NT-LoRA > NTv3.
That is close to an exact inversion of the balanced ordering at both ends.

Absolute precision is **not comparable across the two genomes** — *C. autoethanogenum* has a denser
truth catalogue (2,885 TSS against 1,245), so every model reads higher there. Only the within-genome
difference between models is meaningful, and it favours Evo2 on both.

One nuance worth stating: CNN-v2 can afford a step-1 scan, where it reaches 71.6% at recall 0.1 on
*C. ljungdahlii*, effectively tying Evo2's step-10 71.0% at that single operating point. Evo2's
advantage is clearest at higher recall and per unit of compute. The ensemble (74.4%) beats both.

### Regime B — composition-matched

Measured on the earlier checkpoints trained on the plain 0.6 split, like the fine-tuned rows of
Regime A. In-genus composition-matched AUROC: ProkBERT 0.736 > CNN-v2 0.659 > CNN-v1 0.629 >
**NT 0.440**, which is at chance (90% CI 0.36–0.53). Out-of-genus, all members fail. CNN-v2 has the best SRS discrimination of the four at AUROC
0.924 (the Evo2 probe was never measured in this regime), meaning it carries the cleanest motif signal rather than a composition shortcut.

### What this cost us, and what it bought

NT looked like a reasonable second member on the balanced test and was near-worst everywhere that
mattered. NTv3 was the most extreme case: best balanced score of anything we trained, worst genome
scale. Adding CNN-v1 as a fourth ensemble member **diluted** the result on both validation genomes
(74.4%/61.6% → 71.4%/58.3% and 87.0%/76.6% → 86.5%/74.5%; see [MODELS.md](MODELS.md)). The deployment
ensemble is therefore exactly three members, chosen on Regime C.

---

## 2. The delivered candidates

Mined from the complete reference assembly `GCF_014131795.1` of *C. butyricum* DSM 10702.

| Tier | Count | What it is |
|---|---|---|
| **High-confidence** | **929** | ensemble ≥ 0.9, ≥ 5 supporting windows, significant against the null |
| …with a placed −10 box | 180 | promoter architecture confirmed by motif search |
| …**canonical**, −10 and −35 | **45** | the primary wet-lab shortlist |

All 929 overlap an FDR-significant peak at q ≤ 0.05 (largest q 0.032).

| File in [`final-candidates/`](../final-candidates/) | Rows | What it is |
|---|---|---|
| `g131795_ranked_seq.tsv` | 929 | every confident candidate, ranked, with tier, q-value, and the summit, −10 and −35 sequences |
| `g131795_tier1_canonical_seq.tsv` | 45 | the canonical tier of the same table |
| `g131795_wetlab_shortlist.xlsx` | 45 | the canonical tier for the wet lab, with a Notes tab |
| `g131795_final_wetlab.tsv`, `g131795_dropped.tsv` | 899, 30 | `filter_and_rank_wetlab.py`: 30 internal duplicates removed |
| `g131795_wetlab_final_ranked.tsv`, `g131795_wetlab_final_novelty.xlsx` | 45 | the canonical tier after a novelty check against RegulonDB, DBTBS and the 93 literature promoters
(0 matched). That check could **not** include the ~5,759 in-genus Hocq sequences, the set most likely
to share a conserved promoter, so "novel" means novel against those three references only |
| `g409755_*` | 912, 306, 100, … | the superseded mine on the 96-contig draft assembly |

`prioritize_candidates.py` writes `g131795_ranked.tsv` and the tier files without the three
sequence columns. The `_seq` versions add them in a step that is not in the repo; the sequences are
in the mine's `g131795_call_seqs.fasta` and FIMO's `fimo.tsv`.

### Reading these honestly

- **This is an enriched candidate list, not a precise map.** At genome scale the base rate of real
  promoters is tiny, which caps achievable precision no matter how good the model is.
- **The FDR step validates, it does not rank.** Every candidate clears q ≤ 0.05 and the top tier sits
  at the null's resolution floor: a smallest achievable p of 5.79×10⁻⁴ from 1,726 null samples, and a
  0.9 score cutoff already at the null's 95th percentile. Those two figures are quoted from the
  committed `empirical-fdr/results/cbut_xs/analysis.md`, an earlier run on a different 4-contig
  assembly; the delivered q-values come from the `g131795` run, whose report is not committed. The
  shape of the argument holds, so this is a wholesale validation
  rather than a filter. **Rank by score, never by q-value.**
- **"Canonical" is a motif convenience, not a biological verdict.** De-novo motif discovery runs per
  mine and is not perfectly stable between runs.
- **A missing −10 is not a missing promoter.** On an AT-rich genome the −10 box is the conserved,
  information-rich signal and the −35 often is not, so "−10 only" dominating is expected.

### The assembly change, and what it showed

An earlier mine used a 96-contig draft of the same strain and reported 912 / 306 / 100. Re-mining on
the complete assembly gave 929 / 180 / 45. The confident set barely moved, **+1.9%**, while the motif
tiers fell by 41% and 55%.

That difference is the useful part. The **ensemble score ranking is robust**; the motif labels depend
on a motif rediscovered from scratch on each run. It is why we rank on score and treat "canonical" as
a soft annotation.

The two assemblies share **no contig identifiers**, so their tables cannot be joined by position. Map
by sequence if you need a crosswalk.

### The −35 audit

A follow-up asked whether a −35 windowing bug explained the drop. It did not. Re-running the check on
the real files reproduced the original numbers exactly, but the cause is biological rather than a
coordinate error: the −35 boxes carry **almost no canonical `TTGACA` signal above random AT-rich
DNA**, while the −10 boxes are strong and clean. Coverage across the 929: both boxes 45, −10 only
135, −35 only 288, neither 461.

So do **not** re-extract the −35 boxes to "fix" this; the signal is not there to recover and a
re-extraction would chase noise. If an architecture call is genuinely wanted for this organism, learn
a *Clostridium*-specific −35 model rather than using the *E. coli* consensus, and report −35 strength
as a p-value against the null. Evidence: [`../mining-and-motif-annotation/motif_qc/REANALYSIS.md`](../mining-and-motif-annotation/motif_qc/REANALYSIS.md).

---

## 3. Calibration

Raw scores are uncalibrated, and CNN-v2 in particular runs optimistic. Calibration fits a Platt
regressor with an **AT-content covariate** so a score becomes a probability that accounts for the
AT-rich trap. Unregularised versus L2 is chosen by 5×5 stratified cross-validation on out-of-fold
log-loss and Brier score; the simpler model wins unless L2 beats it by a margin.

The evaluation set is 279 sequences: 93 positives plus 186 curated genomic negatives, being 130
convergent-intergenic and 56 coding-interior windows.

**Calibration is not needed for the mine.** Rank percentile is scale-invariant, so the ensemble
combines raw scores correctly. It matters only for a fixed threshold.

Two things the calibration analysis of the earlier checkpoints (`T21_final`, `T21_final_06`)
confirmed. The deployed checkpoints were calibrated too, by `run_calibration_xs.sh`; their reports
are in `calib_xs/`, and the calibrator files themselves are not committed.

- **The two model families fail in opposite directions.** The CNNs show the AT-trap failure mode,
  scoring AT-rich non-promoters as promoters. The transformers are the opposite, so wary of the trap
  that they reject genuine AT-rich promoters. A liberal CNN vote and a conservative transformer vote
  average toward the truth, which is direct per-class evidence that the ensemble is well-motivated
  rather than redundant.
- **NT's instability reproduces here.** With the older hyperparameters its calibrated AUROC is
  sub-chance on this genomic set, matching its Regime-B collapse. Independent confirmation from a
  different measurement.

Reports live in [`../evaluation-and-calibration/calibration/`](../evaluation-and-calibration/calibration/).

### A caveat on the negatives

The whole 279-sequence calibration set is ~52 bp: the genomic negatives were extracted at **52 bp**
and the 93 T9 positives are 49–52 bp while the models were trained and evaluated at
**81 bp**. Scoring shorter windows feeds the models less context than they saw in training, which can
shift the very distribution you are calibrating. Regenerate at 81 bp before trusting calibrated
q-values; the extractor takes a `--window` flag, so it is a one-flag change.

Two further honest limits. The convergent-intergenic set is *putatively*, not confirmed,
promoter-free: convergent orientation makes a promoter unlikely, but antisense and internal start
sites exist. And one source genome is out-of-genus with a different AT bias, so it belongs in a
separate stratum rather than pooled.

---

## 4. Known gaps

- **The deployed models trained on 888 misplaced *B. licheniformis* positives.** The `xs` split took
  them from the wrong assembly (`NC_006270.3` rather than `NC_006322.1`), so they are effectively random
  windows. Only 5% carry a −10 box and 52% start on a purine, against 63% and 95% from the right
  genome. They are 3.5% of the positives and all in train; no validation, test or genome-scale
  evaluation contains them. [PIPELINE.md](PIPELINE.md), under *Genome assemblies*, has the check and
  the assembly the table does fit. Scanned identically, the deployed ProkBERT still made a third fewer
  false-positive peaks than its predecessor at recall 0.5 (604 against 905). The `xs` split changed
  several things at once, so which change produced that was not isolated.
- **Iterative hard-negative mining** is the strongest remaining lever on false positives and was
  scoped but never run. The design matters: mine confident false positives only where labels exist,
  keep the mining and validation genomes disjoint, and **down-weight** mined negatives the way silver
  positives are down-weighted. Do not relabel the target's own top scores as negatives — with no
  target TSS a high score is ambiguous, and training on that teaches the model to reject real
  promoters.
- **The stratified evaluation** (by source, species, GC band and sigma factor) was run on an earlier
  round of ProkBERT, NT-LoRA and CNN-v1 (`evaluation-and-calibration/sigma-factor-stratified/
  T25_report.tex`), not on the deployed models.
- **Cluster-disjoint cross-validation** scripts exist (`evaluation-and-calibration/cross-validation/`)
  but were not run.
- **Verified in-genus evaluation data is thin**, a few hundred promoters, so in-genus performance
  carries wide error bars.
