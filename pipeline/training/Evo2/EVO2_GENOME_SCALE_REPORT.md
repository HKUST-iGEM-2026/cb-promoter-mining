# Evo2 restricted-scan benchmark

**Historical August 2026 run.** The frozen Evo2-7B probe produced fewer benchmark-unmatched peaks than the compared CNN at all listed recall targets. This reversed the earlier AUPRC-only decision to drop Evo2; it now belongs to the deployed trio.

## Method

The weighted probe selects logreg/RF by validation AP, refits train+validation (47,137 rows) and evaluates test. Historical logreg validation/test AP is 0.7007/0.7026. Saved-feature recovery later reproduced weighted test AP 0.70258010/AUROC 0.89534660 on 3,053 rows/547 positives; it did not rerun extraction or the scans below.

The current frozen recipe uses 81-bp windows, mean-pooled blocks.28.mlp.l3 and 4096 float32 features. Exact historical source revision/layer remains unresolved. [Model documentation](../../../docs/MODELS.md#verified-original-data-and-encodings) identifies the recovered data.

Restricted regions span 300 bp upstream/50 bp into annotated genes, by strand. Step-10 windows overlap these regions. Same-contig/strand windows expanded ±40 bp cover catalogue TSS; unmatched starts merge at gaps ≤81 bp. Precision proxy is covered TSS/(covered TSS+unmatched peaks), not matched-call precision. Historical “FP” means unmatched, not experimentally false.

**Silver positives** are putative promoter-positive upstream DNA labelled from predicted housekeeping-gene annotations and genomic position, not mapped TSS or experimentally measured promoters. Putative-negative windows are not experimentally established promoter-free DNA.

The CNN step-1 track was subsampled to step 10. Equal stride does not prove equal grids; the standalone Evo2 jobs omit `--align-grid`, which deployment adds. Both benchmark strains supplied silver positives and putative-negative training windows while published TSS labels were withheld.

## Curated classification

| Model | Test AUPRC |
|---|---|
| ProkBERT (fine-tuned) | 0.884 |
| NT-LoRA (fine-tuned) | 0.854 |
| CNN-v2 (from scratch) | 0.848 |
| CNN-v1 (from scratch) | 0.804 |
| **Evo2 frozen + logreg** | **0.703** |


Task-trained rows are earlier T21_final/plain-0.6 checkpoints; Evo2 uses T15_xs, where deployed CNN-v2 AP is 0.726. These numbers do not establish a controlled ranking. Test prevalence is about 18%, not 50%.

## C. ljungdahlii

GCF_000143685.1/NC_014328.1, 1,245 published TSS; historical restricted-window coverage baseline 0.70%, not promoter prevalence.

| Recall | Evo2 (step 10) | cnn_v2-xs (step 10) | Δ precision | cnn_v2-xs (step 1, ref) |
|---|---|---|---|---|
| 0.10 | **71.0%** (51 FP) | 63.1% (73 FP) | +7.9 | 71.6% |
| 0.50 | **58.0%** (451 FP) | 48.2% (670 FP) | +9.8 | 51.9% |
| 0.70 | **44.8%** (1,074 FP) | 38.3% (1,403 FP) | +6.5 | — |
| 0.90 | **27.1%** (3,012 FP) | 24.8% (3,391 FP) | +2.3 | — |


## C. autoethanogenum

GCF_000484505.1/NC_022592.1, 2,885 published TSS; historical coverage baseline 1.70%.

| Recall | Evo2 (step 10) | cnn_v2-xs (step 10) | Δ precision |
|---|---|---|---|
| 0.10 | **85.8%** (48 FP) | 81.2% (67 FP) | +4.6 |
| 0.50 | **71.4%** (581 FP) | 67.1% (708 FP) | +4.3 |
| 0.70 | **59.7%** (1,361 FP) | 57.3% (1,507 FP) | +2.4 |
| 0.90 | **46.9%** (2,939 FP) | 40.2% (3,864 FP) | +6.7 |


## Interpretation

Compare members within a genome. Catalogue density/completeness differs; absolute precision is not transferable target accuracy. The step-1 CNN reference and later paired-grid numbers are separate runs: paired Evo2 gives 69.4/58.4% and 87.6/72.5%, while the trio gives 74.4/61.6% and 87.0/76.6%.

These results motivate Evo2's inclusion, not a causal pretraining explanation or promoter confirmation. Original run checks are not preserved as a new audit; source inspection alone does not replace execution. The probe retains the 888 wrong-assembly training positives. Historical runtime around one GPU-hour depends on hardware/batching/coverage. Removing `--regions` from a benchmark cannot reconstruct unscored windows.

[Pipeline](../../../docs/PIPELINE.md) gives reproduction prerequisites; [interpretation](../../../docs/RESULTS.md#interpreting-candidates-and-downstream-units) separates ranks/motifs from biological claims.
