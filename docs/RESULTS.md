# Results

This page reports retained computational measurements and predictions, with their protocols attached. **[Results and downloads](../results/README.md)** is the public artifact index. The team separately reports ordering plasmids, transformation and GFP testing in *E. coli*; candidate-linked quantitative measurements and reporter strain were not supplied here. Native *C. butyricum* validation was not reached. No computational ratio below is a measured expression improvement.

## 1. Target catalogue and sequence downloads

The current target is complete DSM 10702 assembly **GCF_014131795.1**. Common-window average rank percentiles from ProkBERT, CNN-v2 and Evo2 are averaged, gated at ≥0.9 and converted to peaks with ≥5 supporting windows. Recovered tracks reproduce **179,172 aligned windows → 11,302 gated windows → 3,267 peaks → 929 supported regions**.

| Nested set | Count | Meaning |
|---|---:|---|
| Retained regions | **929** | score/support-filtered promoter-like DNA |
| With a placed −10 hit | **180** | positional motif annotation |
| With both placed −10 and −35 hits | **45** | primary motif-based shortlist; historical tier `canonical` |

Exclusive labels are 749 `confident`, 135 `has_minus10` and 45 `canonical`. Motif combinations are 45 both-hit, 135 −10-only, 288 −35-only and 461 neither. Catalogue priority is motif tier, descending score, support and best motif p-value—not activity. The historical draft-assembly GCF_000409755.1 mine (912/306/100) is superseded; no sequence crosswalk establishes rank/locus stability across assemblies.

### Downloads

| Artifact | Contents |
|---|---|
| [Ranked TSV](../results/candidates/g131795_ranked_seq.tsv) | all 929 regions with motif tiers and summit/box sequences |
| [Both-hit TSV](../results/candidates/g131795_tier1_canonical_seq.tsv) | all 45 both-hit candidates |
| [Named first-ten TSV](../results/candidates/g131795_first10_canonical_named.tsv) / [FASTA](../results/candidates/g131795_first10_canonical_named.fasta) | DL1–DL10, proposed promoter names, catalogue IDs and exact oriented 81-bp DNA |
| [Reporter DNA FASTA](../results/candidates/sfGFP_pptb_reporter_sequences.fasta) | supplied 717-bp sfGFP and 1,017-bp `pptb-sfGFP-B1010` circuit, preserving case |
| [Gene-context CSV](../results/candidates/g131795_first10_canonical_gene_context.csv) / [JSON](../results/candidates/g131795_first10_canonical_gene_context.json) | downstream associations and method/input hashes |
| [Full download index](../results/README.md) | additional spreadsheets, filtering hand-offs, predictions and provenance |

The naming export preserves first-ten catalogue order; it is not a new ranking or validation. **DL10 = `cand_00101` / `P_prfB`; `pptb_fU` is a separate 149-bp reference**, despite its older notebook DL10 alias. The circuit contains the supplied sfGFP and exact reference once each; registry identity `BBa_26MAY2X6` is team-supplied metadata, not independently authenticated here. Sequence-screen annotations in spreadsheet hand-offs do not establish biological novelty. The original sequence/spreadsheet packaging generator is not bundled.

### Coordinates and motif evidence

All 929 summit sequences match oriented 81-bp slices of the checksum-verified complete reference. Scan coordinates are 0-based half-open; candidate regions are 1-based inclusive. With summit defined as forward-genome window start, inferred TSS is summit+61 forward / summit+21 reverse. These are frame-derived starts, not RNA mapping.

Recovered MEME/FIMO 5.3.0 records identify five de-novo motifs learned from the top 500 summits (widths 14/7/8/18/10 bp). FIMO scanned all 929 oriented windows with `--norc`, emitted p≤0.0001 and used NRDB A/C/G/T background 0.282/0.222/0.229/0.267. The builder filters emitted hits at p≤0.001 and labels centres 3–18 bp upstream as −10 and 22–42 bp upstream as −35 **without checking motif ID**. These are positional labels, not verified σA grammar.

All 180 selected −10 and 333 selected −35 contexts reproduce the original FIMO fields. Of the −35 selections, 288 use the same 14-bp MEME-1 motif used for 170 −10 selections. `spacer_bp` is centre-to-centre distance, not an edge spacer. Separate width-six reference-PWM QC gives TATAAT/TTGATA cores and AT-rich-null median percentiles 98/52; selected-context QC is not independent activity validation. The width-six reference file was not the original discovery scan input.

### Downstream genes and putative units

A promoter is regulatory DNA, not a protein-coding gene. Nearest downstream same-strand annotation helps interpret a locus but does not establish regulation. The first-ten map uses PGAP **GCF_014131795.1-RS_2026_04_15**, merging multipart CDS segments of the same contig/locus into one gene. Adjacent same-strand loci are grouped when they have ≤100 intervening bases: `next.start − previous.end − 1` on +, or `previous.start − next.end − 1` in − transcriptional order. No circular-origin wrap or measured transcript is inferred. T12 instead groups only its input housekeeping loci.

| Candidate | Downstream locus/loci, transcription order | Annotated product(s) |
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

Nine units have one gene; `cand_00089` has two with **44 intervening bases**. The programmed-frameshift CDS segments of *prfB* remain one gene. Use locus tags and annotated products rather than invented operon names; cotranscription needs transcript evidence. “Hypothetical” means an uncertain specific function, not inactivity.

## 2. Classifier evaluation

### Regime A — held-out labelled windows

The deployed dataset contains **50,190 windows** from a 166-assembly cohort with no supervised Cb target. Train/validation/test counts are **44,085/3,052/3,053**, with **24,001/547/547 positive labels**. Silver positives are annotation/position-derived putative promoter DNA, weighted 0.3 for training loss and confined to training. Putative negatives are not proven promoter-free. Cluster-ID intersections across splits are zero, not proof of biological independence or unknown backbone pretraining exclusion. See the [training manifest](../results/training_sources.json) and [pipeline provenance](PIPELINE.md#genome-assemblies-and-coordinate-provenance).

Negative-source counts from checksum-matched retained T15_xs Parquets show **SRS train/validation/test = 5,505/652/701**; see the [12-row negative-source CSV](https://github.com/HKUST-iGEM-2026/cb-promoter-mining/blob/main/results/training_negative_sources.csv) and [full count table](../results/README.md#training-source-manifest). The training-only restriction applies to silver/cross-species **positive** labels, not their SRS negatives. Earlier documentation stating otherwise was incorrect. SRS intends a dinucleotide shuffle of 81-bp silver cores, with a plain-shuffle fallback: exact preservation and motif destruction are not guaranteed. CNN k-mer frequencies are features of the same labelled windows, not replacement negative generation; Evo2 embeds those windows and refits its probe on train+validation. No split, fitted artifact or numerical result changes with this correction.

| Deployed member | Validation AP | Test AP | Test AUROC |
|---|---:|---:|---:|
| ProkBERT-mini | 0.8080683613 | **0.8249372466** | — |
| CNN-v2 | 0.7380835045 | **0.7255828424** | — |
| Evo2-7B frozen probe | selection cohort later used in refit | **0.70258010** | **0.89534660** |

ProkBERT/CNN trainers weight loss, not AP. Evo2's retained saved-feature replay supplies sample weights; all test weights are 1, so weighted/unweighted AP and AUROC coincide. Evo2 refits on train+validation; validation is not a final held-out cohort. The test prevalence **547/3,053 = 0.179168** is the no-skill AP reference; no-skill AUROC is 0.5. Dashes mean not reported here, not zero.

Historical comparisons reported test AP NTv3 (384-bp context) **0.960**, ProkBERT **0.884**, NT-v2 LoRA **0.854**, CNN-v2 **0.848**, CNN-v1 **0.804**, Evo2 probe **0.703**. These mix checkpoints/splits and input contexts and must not replace the deployed table. Earlier plain-0.6 prevalence was 967/3,318. NTv3 context-overlap/provenance remains unresolved. [Models](MODELS.md) records architectures, input frames and recovery evidence.

### Regime B — composition controls

Historical plain-0.6 CNN-v2 real-versus-SRS AUROC is **0.924**. Shuffled controls test sequence-order information under their sampling, not a unique −10/−35 mechanism. Evo2 was not measured in this regime. Sparse/confounded species/sigma panels do not assign candidate sigma factors.

### Regime C — restricted genomic published-TSS recovery

Same-contig/strand 81-bp windows expand ±40 bp. Recall counts covered catalogue TSS. **Precision proxy = covered TSS / (covered TSS + merged unmatched peaks)**, not the fraction of called peaks matching. Above-threshold unmatched starts merge at gaps ≤81 bp. Restricted candidate regions and incomplete TSS truth limit interpretation: an unmatched peak is not a confirmed false promoter. Both benchmark genomes supplied silver/negative training windows; only published benchmark TSS truth was withheld.

| Model/run | *C. ljungdahlii* @ recall 0.1 / 0.5 | *C. autoethanogenum* @ recall 0.1 / 0.5 |
|---|---|---|
| **Paired 3-model ensemble** | **74.4% / 61.6%** | **87.0% / 76.6%** |
| Evo2, standalone six-model scan | 71.0% / 58.0% | 85.8% / 71.4% |
| ProkBERT | 64.4% / 47.7% | 76.7% / 64.6% |
| CNN-v2 | 64.4% / 47.2% | 81.0% / 66.0% |
| CNN-v1 | 49.0% / 40.3% | 72.6% / 63.0% |
| NT-LoRA | 43.9% / 36.1% | 64.9% / 58.7% |
| NTv3 context | 35.7% / 28.0% | 58.5% / 48.3% |

<a id="regime-c--restricted-published-tss-benchmark"></a>

Download the [reported six-model comparison CSV](../results/benchmarks/reported_six_model_genomic_comparison.csv). The ensemble is from the paired trio comparison, not an extra controlled member of the historical six-model scan. **Paired Evo2** gives **69.4/58.4%** and **87.6/72.5%**, respectively: the ensemble leads at recall 0.5 but not everywhere. Adding CNN-v1 gives 71.4/58.3% and 86.5/74.5%. A separate step-1 CNN-v2 scan gives 71.6/51.9% on *C. ljungdahlii*; different grids/runs are not controlled comparisons.

Retained aligned-score recomputation reproduced paired counts: *C. ljungdahlii* 125/(125+43)=74.4048% and 623/(623+388)=61.6222%; *C. autoethanogenum* 289/(289+43)=87.0482% and 1443/(1443+442)=76.5517%. This verifies saved-score arithmetic, not a new DNA scan.

## 3. Independent DH5α reporter-core estimates

The 88-record RBF SVR scores **50-bp `summit_seq[10:60]` cores**, not ensemble ranks or whole constructs. `cand_00098` predicts **32.8751 AU / 474.57%** of primary predicted `pptb_fU`; `cand_00101` predicts **17.9027 AU / 258.44%**. The primary reference predicts **6.927284 AU**. These AU are the external training table's unspecified numerical activity scale—not RFU or validated RPU.

Five reference-start scenarios predict **6.927284–35.097146 AU**, moving the leading ratio from **0.9367× to 4.7457×**. This is reference-framing sensitivity, not a confidence interval. Candidate-start sensitivity also matters: `cand_00048` leads at offset 58; `cand_00098` at 59–62. Selection-CV R² is **0.266090**; the source paper reports held-out Pearson **0.57**, neither a candidate-specific accuracy estimate.

Download [CSV](../results/candidates/g131795_first10_canonical_dh5alpha_strength.csv), [TSV](../results/candidates/g131795_first10_canonical_dh5alpha_strength.tsv), [JSON](../results/candidates/g131795_first10_canonical_dh5alpha_strength.json), [framing sensitivity](../results/candidates/g131795_first10_canonical_dh5alpha_tss_sensitivity.json) and [reference DNA/provenance](../results/candidates/pptb_fU_reference.json). The [strength guide](ECOLI_STRENGTH_PREDICTIONS.md) gives all ten estimates, reproduction and the separate generic σ70 initiation comparison; their AU are not interchangeable.

## 4. Oxidative-stress-associated candidates

The separate [source workflow](../pipeline/mining/inducible/) selects stress-associated gene neighbourhoods. Its target list has **31 entries / 30 distinct loci / 27 regions**, producing **24 peaks** from separate **1,026 candidate / 420 null windows** on GCF_014131795.1. It is not a subset filter on the ordinary 929-region score pool.

Download the [24-row final CSV](../results/oxidative-stress/CB_oxidative_stress_final_promoters.csv) and [24-sequence FASTA](../results/oxidative-stress/top80_promoters_summits.fasta); `top80` is a historical filename, not the sequence count. Stored six-base reference FIMO annotations give **2 both-hit / 16 −10-only / 1 −35-only / 5 neither**. Both-hit `cbut_0002`/`cbut_0005` have 15/17-bp edge gaps, distinct from ordinary catalogue centre distances.

[Per-candidate motif QC](../results/oxidative-stress/motif_qc/candidate_motif_qc.csv), [summary JSON](../results/oxidative-stress/motif_qc/summary.json) and [signal-versus-shuffled SVG](../results/oxidative-stress/motif_qc/signal_vs_shuffled.svg) compare full-window maximum reference-PWM scores with **2,000 mononucleotide-count-preserving shuffles per candidate**. Median empirical p-values are **0.3483 / 0.4450** for −10/−35-like scores; **none passes q≤0.05 across 48 tests**. This is selected-sequence QC, not a calibrated biological FDR or activity result.

No activity, constitutivity, inducibility or stress protection is validated. The separate [codon optimiser](https://github.com/HKUST-iGEM-2026/codon-optimiser) addresses coding sequences, not promoter strength. Neither it nor these stress candidates was used in experimental constructs. Its codon table uses draft GCF_000409755.1; future use must reconcile assembly/context rather than treating model outputs as tested constructs.

## 5. Calibration, provenance and remaining limits

Optional historical Platt+AT calibration uses 279 short sequences (93 synthetic positives/186 putative negatives), not Evo2 or the trio. Sampled context/prevalence limits transfer; AT-dependent maps can reorder ranks. Original `calib_xs` maps/cohort are unavailable; older maps are not substitutes. The ordinary deployed mine uses no calibrator.

Exploratory FDR tracks pool candidate+null member ranks and select minimum window p-values per peak. They differ from ordinary target ranks and do not establish biological peak-FDR≤5%; see the [retained screening report](../pipeline/evaluation/empirical-fdr/results/cbut_xs/analysis.md).

**All deployed members retain 888 wrong-assembly *B. licheniformis* positive windows.** Correction requires re-extraction, new splits and retraining, not relabelling the retained metrics. Other unresolved provenance includes the exact historical Evo2 backbone extraction revision/layer, NTv3 context-overlap, the scientific source of imported six-base QC PWMs and the original packaging generator. The clustering wrapper and installed MMseqs2 package record are recovered, but do not establish the exact historical executable/runtime linkage; [pipeline recovery notes](PIPELINE.md#recover-the-original-clustering-command-and-version) distinguish these claims.

Public availability is not a licence waiver or hosted service. Project MIT, model declarations, executable upstream code and source-data restrictions remain distinct; see [model loading/licences](MODELS.md#public-hugging-face-repositories), [release licences](../release/huggingface/README.md#licences) and [RegulonDB terms](../REGULONDB_LICENSE.md).
