# PT6 findings — why the E. coli baseline "only mines ~800" (and why that's expected)

**Status:** resolved 2026-06. **Verdict: not a bug.** `prokbert-mini-promoter` is the correct
off-the-shelf model, it demonstrably has real promoter signal, and its near-random *genome-wide*
behaviour is the predicted consequence of prior-shift + the AT-trap. It is the **baseline to beat**,
not something to fix. Don't re-litigate — read this first.

---

## The numbers, read correctly

The committed run (`metrics.json`) is **not** "800 validated vs 30,000 known." It is:

| quantity | value | what it is |
|---|---|---|
| `n_known_tss` | **3,964** | RegulonDB E. coli promoters (= T8's curated set, not 30k) |
| `recall` | **0.24** | known TSS recovered within ±20 bp (954 / 3,964) |
| validated candidates | **818** | candidates within ±20 bp of a known TSS — this is the *precision* numerator |
| `precision_vs_known` | 0.0206 | 818 / 39,805 candidates |
| `enrichment_over_random` | **1.17×** | barely above chance |

The "~800 / ~30,000 → 2.7%" framing divided the **818 precision count** by a wrong denominator. The
real recall is **24%**. (If a future run swaps in RegulonDB's ~30k *high-throughput* TSS collection,
that's an unfair denominator — many condition-specific/antisense/internal TSS an off-the-shelf σ⁷⁰
model was never meant to call. Evaluate against the ~4k curated promoters.)

## The model works — proof from our own data

756 known RegulonDB promoters (T8) were located inside the committed PT5 scan by exact 81 bp match and
compared to background:

| | median score | % ≥ 0.9 |
|---|---|---|
| **Known RegulonDB promoters** | **0.202** | **36.2%** |
| Genome background | 0.008 | 18.7% |

**AUC P(known > background) = 0.624, Mann–Whitney p = 1.7×10⁻³⁷.** Known promoters score ~25× higher
(median) than background — overwhelmingly significant **real signal**. A base `prokbert-mini` with a
random head would give AUC ≈ 0.5, so a *trained* promoter head was used: **the model was loaded
correctly** (it is `prokbert-mini-promoter`, not base `prokbert-mini`). Class index is also correct
(promoter = index 1, per the authors' config + inference notebook), so `softmax(...)[:,1]` is right.

## So why does it look near-random genome-wide? (two compounding, expected effects)

1. **Specificity collapse on real negatives (the AT-trap).** It fires on **18.7% of all background
   windows** — promoters aren't 1/5 of a genome. It was fine-tuned against easy/balanced negatives;
   real AT-rich intergenic/CDS sequence is out-of-distribution, so it lights up on composition.
2. **Base rate.** Promoters are ~0.21% of windows. Even a Se=0.95/Sp=0.95 classifier gives only
   **3.9% genome-wide precision**; reaching 50% precision needs specificity **0.998** (<1 false
   positive per 520 negatives). Observed ~2% precision is exactly this regime.

This is "**right model, wrong task**": `prokbert-mini-promoter` is built + validated for *balanced
window classification* (and it works at that — 36% of known promoters score ≥0.9), but *genome-wide
mining* is an extreme-imbalance, OOD-negative, localization task. Both are true at once.

## What is NOT the problem
- **Code** — coordinates aligned (matched signed-distance median ≈ 1 bp; no off-by-one), strand/match
  logic correct, score extraction correct.
- **Model choice** — correct and working (evidence above).
- **Dataset** — RegulonDB ~3,964 is the right reference.
- **Threshold** — irrelevant here: the score distribution is saturated/bimodal, so the sweep is flat
  (recall 0.225–0.242 from 0.5→0.99). Tuning the threshold won't help; **calibration** is the lever.

## What actually moves the needle (the project plan, confirmed)
- **Fine-tune our own models (T18–T20)** on T15's **realistic negatives** (T14: dinucleotide-shuffled
  SRS + CDS + intergenic) — this teaches the model to *not* fire on AT-rich background, attacking the
  AT-trap directly. Target: AUC 0.62 → 0.8–0.9, and a far lower false-positive rate.
- **Calibrate (PT4 / T26)** for the genomic prior (AT-content-aware), then **empirical FDR via the SRS
  null (T27)**.
- Treat this off-the-shelf result as the **bar to beat**, and report it as such.

## Minor cleanups noted (not the cause)
- PT5 writes `mg1655_promoter_scores1.csv` (stray `1`) instead of the consumed `...scores.csv` — a
  rerun silently won't overwrite the file PT6 reads. Fix the output name.
- Always report **recall and precision separately** — conflating the 818 precision count with recall
  is what produced the "2.7%" scare.
- `merge_gap=40` caps recall ~12 points by collapsing TSS <40 bp apart into one candidate (a known
  ceiling, secondary).

## How to reproduce the signal test
Match T8's `curated_positive_promoters.parquet` (RegulonDB, 81 bp) sequences against the `sequence`
column of `pre-tasks/PT5/mg1655_promoter_scores.csv`; compare `promoter_score` of the matches vs a
background sample (AUC / Mann–Whitney). ~750 grid-aligned matches suffice for p ≪ 10⁻³⁰.

> The 0.624 AUC is a *lower bound* — it's on grid-aligned matches and the "background" contains
> unannotated TSS, both of which depress it. The model's true balanced discrimination is higher, which
> only strengthens the "real signal, wrong task" conclusion.
