# Workflow rationale — is fine-tuned ProkBERT the right approach for genome-wide promoter mining?

*Resolved 2026-06. Read this before re-litigating "the model scores at random, are we doing it wrong?".*

**Question.** The off-the-shelf `prokbert-mini-promoter` scores ~at-random when slid across the
E. coli genome (recall 0.24, precision ~0.02, ~1.17× over random) — even though it is a *fine-tuned
promoter classifier*. So: are we wrong to use ProkBERT this way? Will fine-tuning our own models fix
it? Is the workflow correct?

**Verdict.** The approach is **sound and aligned with published best practice.** Fine-tuning on
realistic negatives directly fixes the failure we measured. The workflow is correct — **provided the
deliverable is framed as an enriched, FDR-controlled candidate list for experimental validation, not a
precise genome-wide promoter map.**

---

## 1. Why the off-the-shelf model looked random (expected, not a bug)

"Right model, wrong task." `prokbert-mini-promoter` was trained/validated on **balanced** 81 bp window
classification, where it genuinely works — on our own data, known RegulonDB promoters score ~25× higher
than background (AUC 0.624, Mann–Whitney p = 1.7×10⁻³⁷): real signal, correctly loaded. **Genome-wide
mining is a different task**: ~0.2% prevalence + out-of-distribution negatives (real AT-rich
intergenic/CDS) + localization. The field quantifies this: a 90%-specificity model applied to the
E. coli genome predicts **~460,000 promoters that don't exist** [BioSystems 2022]. Balanced-test
accuracy is misleading at genome scale — judge by the *number*, not the proportion, of false positives.

## 2. Why fine-tuning *our* models should fix it

The off-the-shelf model failed because it **never saw realistic negatives.** Our T14 set is the
established fix, by name:
- **SRS (dinucleotide-shuffled real promoters):** the literature uses this exact technique — SRS
  "result[s] in less detection of background noise in real genomic data," and promoter-derived (not
  random-genomic) negatives "significantly reduce false positive predictions" [DeePromoter 2019;
  Bioinformatics 2025].
- **+ CDS / intergenic / downstream negatives:** teach rejection of real genomic background.

This is the single biggest lever, and it is already in the design. Fine-tuned DNA LMs also demonstrably
learn real promoter motifs after fine-tuning (attention concentrates on upstream −10/−35 regions)
[Brief. Bioinform. 2025].

## 3. The workflow is standard practice

Negative engineering → calibration for the genomic prior (PT4/T26) → empirical FDR vs the SRS null
(T27) → 3-model ensemble (T29) → peak calling (T30) → motif confirmation (T31) → experimental
validation (dRNA-seq TSS). FDR control + experimental validation are exactly what the field requires,
because "in the whole genome of E. coli there must be several sequences that by chance resemble
promoters" [PLOS ONE]. Our pipeline is, if anything, more rigorous than many published predictors.

## 4. Honest limits — set expectations accordingly

- **No method achieves high genome-wide *precision*** for bacterial promoters; the −10/−35 motif is
  too low-information in AT-rich DNA. **Success = enrichment over random + experimental hit-rate + a
  ranked candidate list — not a precise map.** If anyone expects a clean genome-wide promoter map, the
  expectation is wrong, not the method.
- **Verified in-genus eval is thin (~270 promoters).** We measure cross-species performance well but
  cannot yet strongly confirm in-genus gains; a real improvement and noise look similar at that size.

## 5. Recommended upgrades (evidence-based)

1. **Iterative hard-negative mining (task T33).** The strongest FP-reduction trick in the literature:
   run the model on the genome, take its highest-scoring false positives, add them as a 5th T14
   negative class, regenerate T15/T16, retrain T18–T20. We currently use **one-shot** negatives; this
   directly targets the AT-trap. Implement as a *second pass* of the existing rules with an augmented
   T14 (it's a loop, so not a new acyclic Snakemake rule). [DeePromoter; Bioinformatics 2025]
2. **Restrict the scan region.** Score intergenic + gene-upstream windows rather than *every* window
   (incl. deep CDS) — cuts the search space and the FP denominator, complementing the CDS negatives.
3. **Grow verified in-genus data** (e.g. Al-Bassam 2018 *C. ljungdahlii* TSS) so in-genus evaluation
   becomes trustworthy — higher value than another model tweak.

## Sources
- Genome-scale prediction of bacterial promoters — BioSystems 2022: https://www.sciencedirect.com/science/article/abs/pii/S0303264722001526
- Negative dataset selection impacts ML-based bacterial promoter predictors — Bioinformatics 2025: https://academic.oup.com/bioinformatics/article/41/4/btaf135/8098048
- DeePromoter: Robust Promoter Predictor Using Deep Learning — Front. Genet. 2019: https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2019.00286/full
- Genomic language models decode bacterial genomes — Brief. Bioinform. 2025: https://academic.oup.com/bib/article/26/4/bbaf311/8182749
- Genome-wide TSS/promoter identification in E. coli — PLOS ONE: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0007526
- Genome-wide TSS mapping & sigma-factor promoter assignment in C. difficile — Front. Microbiol. 2020: https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2020.01939/full
