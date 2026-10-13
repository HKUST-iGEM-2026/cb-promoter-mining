# Results

The numerical results below are computational evaluations, rankings and predictions, not measured promoter activity. Keep checkpoint, split, weighting and benchmark protocol attached to each number. Separately, the team reports ordering plasmids after mining, performing transformation and testing GFP in *E. coli*. Time limits prevented proceeding to Cb. Tested candidate identities, reporter strain and quantitative assay results were not supplied; the reported work does not characterise the entire 929/180/45 catalogue or validate genotype-specific DH5α SVR predictions.

## Evaluation regimes

| Regime | Measurement | Scope |
|---|---|---|
| **A · In-distribution** | held-out window classification | split prevalence, source selection and checkpoint matter |
| **B · Composition-matched** | discrimination against shuffled or sampled genomic negatives | a control for composition shortcuts, not proof of a particular motif mechanism |
| **C · Genome-scale benchmark** | published-TSS recovery versus benchmark-unmatched peaks at fixed recall | restricted candidate regions, incomplete truth and partially training-exposed strains |


### Regime A — historical held-out scores

| Model | Reported test AUPRC |
|---|---|
| NTv3, 384 bp context | 0.960 |
| ProkBERT-mini | 0.884 |
| NT v2 250M LoRA | 0.854 |
| CNN-v2 fusion | 0.848 |
| CNN-v1 | 0.804 |
| Evo2-7B frozen probe | 0.703 |


These historical comparisons mix checkpoints/splits and NTv3's 384-bp context. Earlier plain-0.6 test prevalence is 967/3,318, versus deployed xs 547/3,053 = 0.179168. The latter is the no-skill AP reference, not a model score; 0.5 is the no-skill AUROC reference.

Deployed ProkBERT validation/test AP is 0.8080683612857302/0.8249372466065321, and CNN-v2 is 0.7380835045313733/0.7255828423854476. Their trainers weight loss, not AP. The recovered Evo2 probe reproduced test AP 0.70258010/AUROC 0.89534660 on saved features, supplying sample weights. Every recovered test weight is 1, so weighted and unweighted test AP/AUROC coincide; the evaluation records have different origins. It refits on train+validation, so validation is not a final held-out cohort. [Model evidence](MODELS.md#verified-original-data-and-encodings) identifies the recovery checks.

### Regime C — restricted published-TSS benchmark

**Silver positives** are putative promoter-positive upstream DNA labelled from predicted housekeeping-gene annotations and genomic position, not mapped TSS or experimentally measured promoters. Putative-negative windows are not experimentally established promoter-free DNA.

Same-contig/strand 81-bp windows are expanded ±40 bp. Recall counts covered catalogue TSS; precision proxy is covered TSS/(covered TSS+merged unmatched peaks), not the fraction of calls matching. Above-threshold unmatched starts at gaps ≤81 bp merge. The genomes supplied silver positives and putative-negative training windows; only published benchmark TSS were withheld. Incomplete truth does not make unmatched peaks confirmed false promoters.

| Model | *C. ljungdahlii* @0.1 / @0.5 | *C. autoethanogenum* @0.1 / @0.5 |
|---|---|---|
| **3-model ensemble** | **74.4% / 61.6%** | **87.0% / 76.6%** |
| Evo2, standalone run | 71.0% / 58.0% | 85.8% / 71.4% |
| ProkBERT | 64.4% / 47.7% | 76.7% / 64.6% |
| CNN-v2 | 64.4% / 47.2% | 81.0% / 66.0% |
| CNN-v1 | 49.0% / 40.3% | 72.6% / 63.0% |
| NT-LoRA | 43.9% / 36.1% | 64.9% / 58.7% |
| NTv3 context | 35.7% / 28.0% | 58.5% / 48.3% |


The Evo2 row is the standalone run. In the paired trio run, Evo2 is **69.4/58.4%** on C. ljungdahlii and **87.6/72.5%** on C. autoethanogenum. The ensemble's 74.4/61.6% and 87.0/76.6% lead at recall 0.5, not everywhere. Adding CNN-v1 gave 71.4/58.3% and 86.5/74.5%. A separate step-1 CNN-v2 scan gave 71.6/51.9% on C. ljungdahlii; differing grids/runs are not controlled comparisons.

CPU recomputation from retained aligned tracks and published-TSS tables reproduced the paired trio: C. ljungdahlii 125/(125+43) = 74.4048% at recall 0.1 and 623/(623+388) = 61.6222% at recall 0.5; C. autoethanogenum 289/(289+43) = 87.0482% and 1443/(1443+442) = 76.5517%. This verifies the saved-score benchmark, not a new DNA scan.

### Regime B — composition controls

Historical plain-0.6 CNN-v2 real-versus-SRS AUROC is 0.924. Shuffled controls test sequence-order information under the supplied sampling, not a unique −10/−35 mechanism. Evo2 was not measured in this regime. Species/sigma panels use earlier checkpoints, sparse/confounded labels and are not candidate sigma assignments.

## Delivered candidates

Current frame is complete DSM 10702 assembly **GCF_014131795.1**. Ordinary mining uses common-window average rank percentiles, a ≥0.9 gate and ≥5 supporting windows.

| Nested set | Count | Operational meaning |
|---|---|---|
| **Confident candidates** | **929** | retained score-thresholded peaks with at least 5 supporting windows |
| With a placed −10 hit | 180 | positional motif annotation, not confirmed promoter activity |
| **Promoters with both −10 and −35 motifs** | **45** | both positional hits; primary motif-based shortlist, not characterised promoters |


Exclusive labels are 749 `confident`, 135 `has_minus10` and 45 promoters with both −10 and −35 motifs, stored under the historical machine tier `canonical`. Motif combinations are both 45, −10-only 135, −35-only 288 and neither 461. Priority is motif tier, descending score, support and best motif p-value. The actual prioritizer preserved all 929 IDs/order in the recorded check. Internal filenames/tier codes remain unchanged to preserve artifact provenance.

| File in [results/candidates/](../results/candidates) | Rows | Scope |
|---|---|---|
| `g131795_ranked_seq.tsv` | 929 | ranked catalogue with motif tiers and summit/box sequences |
| `g131795_tier1_canonical_seq.tsv` | 45 | both-box subset |
| `g131795_wetlab_shortlist.xlsx` | 45 | spreadsheet hand-off of promoters with both −10 and −35 motifs |
| `g131795_final_wetlab.tsv`, `g131795_dropped.tsv` | 899, 30 | 20-mer redundancy/known-reference filtering outputs |
| `g131795_wetlab_final_ranked.tsv`, `g131795_wetlab_final_novelty.xlsx` | 45 | later two-motif hand-off; sequence-screen annotations do not confirm biological novelty |
| `g409755_*` | 912 confident / 306 −10 / 100 with both motifs | **historical**, superseded `GCF_000409755.1` draft-assembly mine; `top_candidates.tsv` has 82 rows |


Sequence/spreadsheet enrichment is supplied, but its packaging generator is not bundled. Earlier GCF_000409755.1 counts 912/306/100 are superseded; no sequence crosswalk establishes locus/rank stability across assemblies.

### Coordinates and motif labels

All 929 summit sequences match oriented 81-bp slices of the checksum-verified complete reference; inferred starts lie inside stored regions. Scan coordinates are 0-based half-open, candidate regions 1-based inclusive. Inferred TSS is summit+61 forward/summit+21 reverse, where summit is the forward-genome window start. It is not RNA mapping.

Recovered `t31_g131795/meme/meme.txt` and `fimo/fimo.xml` identify MEME/FIMO 5.3.0 and five de-novo motifs learned from the top 500 summit sequences, with widths 14/7/8/18/10 bp. FIMO scanned 929 oriented windows with `--norc`, emitted p≤0.0001 and used NRDB background A/C/G/T = 0.282/0.222/0.229/0.267. Builder p≤0.001 filters emitted hits, then labels centres 3–18 bp upstream as −10 and 22–42 as −35 without checking motif ID. The labels are positional, not verified σA grammar. The supplied width-six `sigma_a_promoter.meme` is a separate QC reference, not this original scan's input.

All 180 −10 and 333 −35 contexts equal their selected original FIMO matched sequences, and the recovered hits reproduce the builder fields. Of the −35 selections, 288 use the same 14-bp MEME-1 motif also used for 170 of the −10 selections; a positional label is therefore not a motif-specific assignment. `spacer_bp` is centre-to-centre distance, not an edge spacer. Separate width-six QC cores give TATAAT/TTGATA and matched-null median percentiles 98/52; this is selected-context QC, not independent activity validation. Recovered target tracks reproduce the funnel 179,172→11,302→3,267→929.

## Interpreting candidates and downstream units

A promoter is regulatory DNA, not a protein-coding gene. High mean rank does not require unanimity and is not probability or strength. Housekeeping selection, motif support and a nearby oxidative-stress gene do not prove constitutive/inducible behavior.

A separate map assigns the nearest downstream same-strand CDS and groups adjacent CDSs at next_start−previous_end≤100 bp. With inclusive coordinates this is one bp greater than the intervening gap. single_gene_unit and putative_multigene_operon are proximity-defined hypotheses, not measured transcripts. T12 groups only input housekeeping loci, not all CDSs.

The first-ten map uses PGAP annotation GCF_014131795.1-RS_2026_04_15. Nine units are single CDSs; cand_00089 has two members with a reported 44-bp CDS gap. The full map/transcript evidence is not committed.

| Candidate | Inferred downstream members, in transcription order | Annotated product(s) |
|---|---|---|
| `cand_00029` | `FF104_RS03955` | Transglycosylase domain-containing protein |
| `cand_00032` | `FF104_RS17285` | Glutaredoxin domain-containing protein |
| `cand_00035` | `FF104_RS16265` | Hypothetical protein |
| `cand_00048` | `FF104_RS19570` | Methyl-accepting chemotaxis protein |
| `cand_00072` | `FF104_RS12975` | Hypothetical protein |
| `cand_00070` | `FF104_RS05165` | Hypothetical protein |
| `cand_00089` | `FF104_RS17655` (*nrdG*) → `FF104_RS17650` | Anaerobic ribonucleoside-triphosphate reductase activating protein → aldose 1-epimerase family protein |
| `cand_00094` | `FF104_RS03625` | ComEC/Rec2 family competence protein |
| `cand_00098` | `FF104_RS20510` | Hypothetical protein |
| `cand_00101` | `FF104_RS15625` (*prfB*) | Peptide chain release factor 2 |


Use stable locus tags, optional gene symbols and predicted products—not invented operon names. A hypothetical protein lacks a reliable specific functional assignment; it is not necessarily inactive or useless. Cotranscription requires junction-spanning transcript evidence.

## DH5α strength predictions for the first ten promoters with both −10 and −35 motifs

The independent DH5α RBF SVR uses 50-bp cores at summit_seq[10:60], not ensemble ranks or whole-insert expression. cand_00098 gives 32.8751 AU (4.7457× primary predicted pptb_fU), cand_00101 17.9027 AU (2.5844×). Five reference-start scenarios give 6.927284–35.097146 AU and a leading ratio of 0.9367–4.7457×, not a confidence interval. At candidate offset 58 cand_00048 leads; at 59–62 cand_00098 leads.

The 88-record fit has selection-CV R² 0.266090; the source paper reports held-out Pearson 0.57. [Strength method](ECOLI_STRENGTH_PREDICTIONS.md) provides full tables, reference DNA, reproduction and the separate generic initiation comparison. The two models' AU are not interchangeable.

## Calibration and FDR

Optional historical Platt+AT calibration uses 279 short sequences, including 93 synthetic positives/186 putative negatives, not Evo2 or the trio. Sampled prevalence/context limits genomic transfer; AT-dependent maps can reorder ranks. Original calib_xs maps/cohort are unavailable, and older maps are not substitutes. The deployed mine needs no calibrator.

FDR tracks pool candidate+null member ranks and use selected minimum window p-values for peaks. They are different from ordinary target ranks and do not establish calibrated biological peak FDR≤5%. [T27 report](../pipeline/evaluation/empirical-fdr/results/cbut_xs/analysis.md) gives screening counts.

## Remaining evidence

The 888 wrong-assembly B. licheniformis labels persist. The original MMseqs command/version requires source-host evidence; [recovery commands](PIPELINE.md#recover-the-original-clustering-command-and-version) collect it without rerunning clustering. Exact historical Evo2 extraction revision/layer, the imported six-base QC PWMs' scientific source, the sequence/spreadsheet packaging generator and NTv3 context-overlap remain unresolved; the original mining MEME/FIMO records are recovered. Cluster-disjoint CV/hard-negative mining code is not proof of completed experiments. Full rescanning needs identified assembly/annotation/grid and compatible Evo2 GPU inference. Native Cb activity, mapped TSS and transcript/operon confirmation remain unestablished; candidate-specific GFP measurements and assay details were not supplied.

### Source-backed audit after merge reconciliation

The current audit checks claims against source code, saved artifacts, pinned source methods, provider policies and the team's supplied account. It does not replace historical metrics with new training or GPU inference. The predictor add/add conflicts contained identical code on both sides and differed only in executable mode; the reconciled commands preserve the retained outputs.

| Claim family | Evidence status | Interpretation boundary |
|---|---|---|
| Sources, splits and weights | Recovered split/encoding records and source metadata; [pipeline](PIPELINE.md) | Putative labels, cluster-disjoint partitions and supervised target exclusion do not prove biological independence or absence from backbone pretraining. |
| Model architecture and classification | Saved configs/probe and original metric records; [models](MODELS.md) | Neural AP records and Evo2 saved-feature replay have different origins; all-one test weights make weighted/unweighted test metrics coincide. |
| Genomic benchmarks and ensemble | Saved aligned tracks, published-TSS tables and benchmark code | Restricted covered-TSS/merged-unmatched-peak proxy, not biological precision; standalone and paired runs remain distinct. |
| Catalogue, motifs and coordinates | Recovered MEME/FIMO records and reproduced builder fields | Positional labels and oriented reference matches do not establish sigma usage, promoter activity, constitutive behavior or mapped starts. |
| Downstream units and novelty | Annotation/sequence-screen records | Proximity and reference-match rules do not establish cotranscription, operon identity or biological novelty; the packaging generator remains absent. |
| DH5α and generic initiation estimates | Both reconciled CLI paths exercised on all ten retained candidates | Same-model SVR reference ratios and framing scenarios are not measured fold-changes or confidence intervals; the generic initiation model is a separate unvalidated host-transfer proxy. |
| Calibration and exploratory q metadata | Historical sampled-cohort reports and saved-rank reconstruction | Neither is a deployed target-genome calibrator or proof of biological peak-FDR control. |
| Experimental, safety and personal claims | Team-supplied account only | Reporter strain, candidate-linked measurements, approvals, contributor confirmations and native Cb validation are not inferred. |
| Access and licences | Authenticated private-package records, pinned licences and current RegulonDB policy | Component existence is not public access or a packaged service; project MIT and artifact declarations do not waive third-party data/model terms. |
