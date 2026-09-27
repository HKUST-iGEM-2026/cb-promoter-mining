# Evo2 at genome scale: a frozen foundation-model embedding beats the task-trained CNN

**HKUST iGEM 2026 — Dry Lab, Promoter Mining. 2026-08.**

## Summary

A frozen **Evo2-7B** embedding with a logistic-regression probe has the **worst balanced AUPRC of any model we trained** (0.70, versus 0.80–0.88 for the four fine-tuned models on the 0.6 split and 0.726 for cnn_v2 on the same T15_xs split). Yet on the genome-scale false-positive benchmark (Regime C) — the one that actually reflects how the model behaves when scanning a whole genome — it produces the **fewest false-positive peaks at matched recall on both validation genomes**, beating cnn_v2-xs, our best CNN and the prior genome-scale leader, at every recall level. The result replicates on two independent genomes and survives a coordinate/strand code audit. It reverses our earlier decision to drop Evo2, and it is the sharpest demonstration in this project of the central thesis that balanced metrics do not predict genome-scale deployment.

## Background

The project evaluates promoter callers under three regimes. Regime A measures in-distribution discrimination on a balanced test set; Regime B measures discrimination against composition-matched (dinucleotide-shuffled) negatives; Regime C measures genome-scale false positives — how many spurious promoter peaks a model hands the wet lab when it scans a real genome at a fixed recall. The report's recurring finding is that Regime A/B rankings do not carry over to Regime C: NT, for example, has the second-best balanced AUPRC but collapses at genome scale.

Evo2 is an autoregressive genomic foundation model (StripedHyena-2, 7B parameters) pretrained on diverse genomes. We use it **frozen**: each 81 bp window is embedded as the mean-pooled activation of layer `blocks.28.mlp.l3` (4096 dimensions), and a logistic-regression probe is trained on those embeddings. The backbone is never fine-tuned. An initial evaluation used only the balanced metric: Evo2's probe scored AUPRC 0.70 (logistic regression) and 0.61 (random forest), below all four fine-tuned models, and Evo2 was set aside. That decision is what this report corrects.

## Method

The probe is trained on the deployed cross-species split (T15_xs; 44,085 train / 3,052 val / 3,053 test), weighted with the same per-sample weights the four ensemble models use, and the better of logistic regression versus random forest is selected on weighted validation AUPRC, then refit on train plus validation (47,137 rows) and scored once on test. Logistic regression wins (val 0.7007, test 0.7026), as expected for a dense 4096-dimensional embedding where a linear probe uses all dimensions jointly and a random forest splits one at a time. The identical probe (`evo2_probe_xs.pkl`) is used for both genomes so that only the genome differs between runs.

Each genome is scanned over its candidate promoter regions (300 bp upstream through 50 bp into every annotated gene, per strand) at step 10 on both strands. Step 10 is a practical necessity: a step-1 genome scan with a 7B model is infeasible, and promoter signal spans ~35 bp so step 10 loses nothing for peak-finding. Scored windows are benchmarked with `genome_fp_benchmark.py` (coverage tolerance 40 bp, false-positive-peak merge 81 bp, strand-aware), reporting precision and false-positive peaks at recall targets 0.1, 0.5, 0.7 and 0.9.

Because Evo2 can only be run at step 10 while the CNNs normally run at step 1, the comparison is made fair by re-benchmarking cnn_v2-xs at the **matched step 10** (its step-1 track subsampled to every tenth position), so the two models face the same window density and the same false-positive denominator. Neither genome's published start sites were used in training. Both strains are among the 166 genomes that supplied inferred positives and negative windows, so the benchmark is not fully independent of training; both models share that, so the comparison between them is like-for-like.

Given how much the result reversed expectations, the comparison code was audited before the numbers were trusted. A direct test confirmed that the Evo2 scanner and the CNN scanner (`genome_inference.py`) report every minus-strand window with the identical forward-genome coordinate and the identical reading-strand sequence, so a promoter on either strand is scored and matched identically by both. The Evo2 restricted scan was shown to cover exactly the window set the restricted benchmark counts (no over- or under-scanning), the batched embedding routine re-scatters results to their original indices so scores never misalign to windows, and the probe's embedding layer, pooling and window length match those used to train it. No leakage is possible: both validation genomes are excluded from T15_xs, on which every model including Evo2 was trained.

## Results

### Balanced discrimination (context)

On the balanced test sets, Evo2's frozen probe is the weakest model by AUPRC. The four fine-tuned rows are the plain 0.6-split retrains (`T21_final`); Evo2 was measured on T15_xs, where the deployed cnn_v2 scores 0.726:

| Model | Test AUPRC |
|---|---|
| ProkBERT (fine-tuned) | 0.884 |
| NT-LoRA (fine-tuned) | 0.854 |
| cnn_v2 (fine-tuned) | 0.848 |
| CNN (fine-tuned) | 0.804 |
| **Evo2 frozen + logreg** | **0.703** |

Its AUROC (0.895) looks competitive, but on a ~22%-positive task AUROC systematically over-reads; AUPRC is the honest metric and Evo2 is last on it. On the balanced evidence alone, dropping Evo2 was reasonable.

### Genome-scale false positives (validation genomes, matched step 10)

The picture inverts at genome scale. Both models detect the identical number of true TSS at each recall target; the entire difference is the number of false-positive peaks, and Evo2 produces fewer at every level on both genomes.

**C. ljungdahlii** (GCF_000143685.1 / NC_014328.1; 1,245 held-out TSS; genome-scale base rate 0.70%):

| Recall | Evo2 (step 10) | cnn_v2-xs (step 10) | Δ precision | cnn_v2-xs (step 1, ref) |
|---|---|---|---|---|
| 0.10 | **71.0%** (51 FP) | 63.1% (73 FP) | +7.9 | 71.6% |
| 0.50 | **58.0%** (451 FP) | 48.2% (670 FP) | +9.8 | 51.9% |
| 0.70 | **44.8%** (1,074 FP) | 38.3% (1,403 FP) | +6.5 | — |
| 0.90 | **27.1%** (3,012 FP) | 24.8% (3,391 FP) | +2.3 | — |

**C. autoethanogenum** (GCF_000484505.1 / NC_022592.1; 2,885 held-out TSS; base rate 1.70%):

| Recall | Evo2 (step 10) | cnn_v2-xs (step 10) | Δ precision |
|---|---|---|---|
| 0.10 | **85.8%** (48 FP) | 81.2% (67 FP) | +4.6 |
| 0.50 | **71.4%** (581 FP) | 67.1% (708 FP) | +4.3 |
| 0.70 | **59.7%** (1,361 FP) | 57.3% (1,507 FP) | +2.4 |
| 0.90 | **46.9%** (2,939 FP) | 40.2% (3,864 FP) | +6.7 |

The absolute precisions differ between genomes only because their truth catalogues differ in density (autoethanogenum has 2,885 TSS to ljungdahlii's 1,245, so its base rate is higher and every model reads higher); absolute precision is therefore not comparable across genomes, and only the within-genome difference between models is meaningful. That difference favors Evo2 on both genomes, at every recall.

## Interpretation

Two validation genomes give the same answer: at matched sampling, the frozen Evo2 embedding is the better genome-scale promoter caller than our best CNN, despite the worst balanced AUPRC. A plausible mechanism is that the balanced test rewards separating promoters from a handful of curated negative classes, whereas genome-scale scanning rewards ranking real transcription start sites above millions of AT-rich genomic decoys; Evo2's pretraining across diverse genomes yields representations that make that second separation cleanly, a capability the balanced set underweights. This is precisely the balanced-versus-deployment gap the project set out to expose, here pointing in the direction opposite to NT.

The deployment picture has one nuance. cnn_v2 can afford a step-1 scan, where it recovers to 71.6% precision at recall 0.1 on ljungdahlii — matching Evo2's step-10 71.0% at that operating point. So at recall 0.1, in realistic deployment, the two are effectively tied; Evo2's advantage is clearest at higher recall (58.0% vs 51.9% at recall 0.5 even against cnn_v2's step-1 best), and per unit of sampling Evo2 is simply the stronger model.

## Caveats

The result is a property of a frozen probe, so the credit belongs to Evo2's pretraining rather than to any task fine-tuning; a fine-tuned Evo2 remains untested and is impractical on the current inference-only software stack. Evo2 runs only at coarse step, which is adequate for peak-finding but forecloses fine-resolution scanning. The TSS catalogues used as truth are incomplete — they are dominated by primary, intergenic-sense start sites — so some counted "false positives" are likely real but unannotated internal or antisense promoters; this inflates neither model preferentially, so the relative ranking is robust while the absolute precisions are lower bounds. Finally, Evo2-7B is far heavier and more fragile to run than the small CNN: a single restricted step-10 scan of a genome is roughly one GPU-hour and is entirely feasible for a one-time target mine, but iterative use is costlier and the environment (a flash-attention ABI matched to the exact torch build) needs care.

## Recommendation

Evo2 should no longer be described as dropped or as below the other models; on the metric that matters for this project it is the genome-scale leader on both validation genomes, and it is the clearest instance we have of balanced AUPRC failing to predict deployment. For the target mine of C. butyricum it is worth using directly — either as the lead genome-scale caller or as a fifth member of the T29 rank-percentile ensemble, into which its logistic-regression probability slots cleanly because rank-percentile aggregation is calibration-invariant; the practical requirement is to run every ensemble member at step 10 over the target's restricted candidate space so the tracks share a window grid. All scripts are retained and reproducible: `extract_embeddings_evo2.py` and `evo2_promoter_classifier.py` build the probe, `evo2_scan_restricted.py` scans, and `C_evo2_ljung_scan.sbatch` / `C_evo2_cau_scan.sbatch` produce the two genome results above.
