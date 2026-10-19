# Results and downloadable sequences

This directory contains retained computational outputs for HKUST iGEM 2026 promoter mining in *Clostridium butyricum* DSM 10702. It is a download index, not a collection of experimentally validated promoters. The complete-reference catalogue uses **GCF_014131795.1**. [Methods and interpretation](../docs/RESULTS.md), [model loading](../docs/MODELS.md) and the [DH5α strength guide](../docs/ECOLI_STRENGTH_PREDICTIONS.md) give the evaluation context.

## Promoter discovery

| Nested candidate set | Regions | Download |
|---|---:|---|
| Confident ranked catalogue | 929 | [TSV with oriented DNA](candidates/g131795_ranked_seq.tsv) |
| With a placed −10 motif label | 180 | Included in the full TSV; filter `has_minus10` |
| With both positional motif labels | 45 | [Both-hit TSV](candidates/g131795_tier1_canonical_seq.tsv) · [Shortlist XLSX](candidates/g131795_wetlab_shortlist.xlsx) |

These sets are nested, not additive. `canonical` is the machine label for both positional motif hits, not confirmed σA grammar, constitutive expression or biological activity. Mean percentile ranks are neither promoter probabilities nor expression estimates.

### DL1–DL10 and reporter sequences

| File | Contents and interpretation |
|---|---|
| [Named-candidate TSV](candidates/g131795_first10_canonical_named.tsv) | All ten names, candidate IDs, suggested promoter names, downstream annotations, coordinates and exact 81-bp transcription-oriented DNA |
| [Named-candidate FASTA](candidates/g131795_first10_canonical_named.fasta) | The same ten 5′→3′ sequences; names follow catalogue priority, not predicted activity |
| [Gene-context CSV](candidates/g131795_first10_canonical_gene_context.csv) · [JSON](candidates/g131795_first10_canonical_gene_context.json) | Ten candidate associations and eleven gene-member records; proximity does not establish regulation or cotranscription |
| [Supplied reporter FASTA](candidates/sfGFP_pptb_reporter_sequences.fasta) | The team's 717-bp sfGFP (`BBa_26MAY2X6`) and 1,017-bp unregistered `pptb-sfGFP-B1010` reference circuit, preserving supplied bases and case |

**DL10 is `cand_00101` / `P_prfB`.** The independent reference `pptb_fU` is not a mined shortlist candidate, despite its older notebook alias. Reporter registry metadata is team-supplied, not independently authenticated here.

## Classifier evaluation

The deployed test split contains 547 positive labels among 3,053 windows; the no-skill average-precision reference is 0.1792. Confidence weights affect training loss, not the ProkBERT/CNN AP values below. All Evo2 test weights are one, so weighted and unweighted test metrics coincide.

| Deployed member | Test average precision | Evidence |
|---|---:|---|
| ProkBERT-mini | 0.8249372466 | [Recovered metrics](../release/huggingface/prokbert-mini-clostridium-promoter/metrics.json) |
| Fusion CNN-v2 | 0.7255828424 | Historical deployed-run record; [evaluation provenance](../docs/RESULTS.md) |
| Frozen-Evo2 logistic probe | 0.70258010 | Saved-feature evaluation; [verification record](../release/huggingface/verification.json) |

Held-out label classification and genomic ranking are different tasks. In paired restricted scans, the retained three-model ensemble gave:

| Published-TSS benchmark | Precision proxy at recall 0.1 | Precision proxy at recall 0.5 |
|---|---:|---:|
| *C. ljungdahlii* | 74.4% | 61.6% |
| *C. autoethanogenum* | 87.0% | 76.6% |

The [six-model comparison CSV](benchmarks/reported_six_model_genomic_comparison.csv) retains the separate individual-model scan values. Those runs used differing pools/contexts and are not a matched-grid ablation. The paired Evo2 constituent led the ensemble at recall 0.1 in *C. autoethanogenum*; ensemble superiority is not universal.

The precision proxy is covered catalogue TSS / (covered catalogue TSS + unmatched peaks), not measured promoter purity. An 81-bp same-strand window is expanded ±40 bp for matching; unmatched starts separated by at most 81 bp merge into peaks. Truth catalogues are incomplete. Both benchmark genomes supplied silver positives and putative negatives during training, although their published TSS labels were withheld.

## Training-source manifest

[training_sources.json](training_sources.json) records all **166 assembly accessions**, recorded taxon names, per-assembly label counts, published/curated label sources, split counts and retained hashes. Its inventory path is repository-root-relative. It is public data, not part of the separately published wiki source.

| Partition | Windows | Positive labels | Putative negative labels |
|---|---:|---:|---:|
| Training | 44,085 | 24,001 | 20,084 |
| Validation | 3,052 | 547 | 2,505 |
| Test | 3,053 | 547 | 2,506 |

The same 166 assemblies supply silver-positive and genomic-negative windows. Silver positives are gene-anchored, annotation-derived upstream DNA, not mapped TSS or validated promoters; they are training-only with confidence/loss weight 0.3. The deployed dataset still includes **888 wrong-assembly *B. licheniformis* training positives**. Target DSM 10702 was excluded from project supervised training; foundation-model pretraining exposure is not established. Publishing this manifest does not repair or retrain the models.

## Independent DH5α activity estimates

The fitted RBF SVR uses 88 published DH5α reporter records and predicts activity from 50-bp cores, independently of classifier ranks. Its AU scale is inherited from the source table, which specifies no physical unit.

- [First-ten predictions: TSV](candidates/g131795_first10_canonical_dh5alpha_strength.tsv) · [CSV](candidates/g131795_first10_canonical_dh5alpha_strength.csv) · [JSON](candidates/g131795_first10_canonical_dh5alpha_strength.json).
- [Exact `pptb_fU` reference and alternative starts](candidates/pptb_fU_reference.json).
- [Candidate-start sensitivity](candidates/g131795_first10_canonical_dh5alpha_tss_sensitivity.json).
- [All ten predictions, normalization and loading instructions](../docs/ECOLI_STRENGTH_PREDICTIONS.md).

The primary reference prediction is 6.927284 AU. DL9 (`cand_00098`) has the highest main-window estimate, but alternative reference windows change its relative estimate from below to above the reference, and a two-base candidate-start shift changes the top-ranked candidate. These are conditional sequence-model comparisons, not measured fluorescence, confidence intervals or native *C. butyricum* strength.

## Oxidative-stress-associated candidates

This separate screen starts with 31 target entries representing 30 distinct loci, collapses them into 27 upstream regions, and selects 24 peaks. It is not another tier of the 929-region catalogue.

| Download | Contents |
|---|---|
| [24-candidate CSV](oxidative-stress/CB_oxidative_stress_final_promoters.csv) | Retained candidate/gene associations, DNA and stored motif annotations |
| [Summit FASTA](oxidative-stress/top80_promoters_summits.fasta) | The 24 oriented 81-bp sequences; `top80` is a historical filename, not a record count |
| [Reference motif file](oxidative-stress/sigma_a_promoter_fixed.meme) | The supplied six-base reference PWMs; scientific source is not recorded in that file |
| [Per-candidate motif QC](oxidative-stress/motif_qc/candidate_motif_qc.csv) | Candidate × reference-motif scores and composition-preserving shuffle comparisons |
| [QC summary](oxidative-stress/motif_qc/summary.json) · [SVG](oxidative-stress/motif_qc/signal_vs_shuffled.svg) | Method, source hashes, counts, adjusted comparisons and figure |

Stored motif annotations comprise two both-hit, sixteen −10-only, one −35-only and five neither-hit windows. Separate reference-PWM QC used 2,000 mononucleotide shuffles per candidate and corrected across 48 candidate × motif comparisons; none passed q ≤ 0.05. This does not prove inactivity, but it does not validate motif enrichment or constitutive expression either. Time constraints prevented using this shortlist in experimental constructs; stress protection remains unmeasured.

These downloads preserve the retained source files byte-for-byte. Paths inside the QC summary identify the historical analysis inputs (`pipeline/mining/oxidative-stress/...`); the corresponding candidate table and motif file are now downloadable here with the same hashes. The source tree also retains historical pipeline inputs under `pipeline/mining/inducible/`; their presence does not imply that every historical script reproduces the later QC.

## Reuse and evidence boundary

The [four public model packages](../release/huggingface/README.md) are downloadable components, not a hosted promoter service or a packaged ensemble. Project MIT terms, individual model-artifact terms, external executable-code licences and [RegulonDB data terms](../REGULONDB_LICENSE.md) remain separate.

The team reports *E. coli* reporter work; this directory does not add candidate-linked experimental measurements or native-host validation. No new training, genome scan or biological assay was performed to prepare these downloads.
