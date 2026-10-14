# Promoter candidates for oxidative-stress-associated genes

This workflow prepares **putative constitutive-promoter candidates for future constructs**. Constitutive expression is the intended use, not a measured property. The team did not use this shortlist or the separate [codon optimiser](https://github.com/HKUST-iGEM-2026/codon-optimiser) in experimental constructs because of time constraints. Gene function, sequence rank and motif resemblance establish neither basal activity nor stress induction/protection.

The neutral `oxidative-stress` directory replaces the misleading `inducible` source-directory name. Scripts, score tracks and existing results remain together; paths were migrated without changing the retained result values. The codon optimiser stays in its own private repository rather than being duplicated here.

## What was produced

| Artifact | Contents |
|---|---|
| [Target list](CB_oxidative_stress_genes.csv) | 31 annotated stress-associated target records |
| [Final promoter table](model_outputs/fdr/results/CB_oxidative_stress_final_promoters.csv) | 24 peaks, oriented 81-bp summit DNA, target/leader annotations and stored motif labels |
| [Summit FASTA](model_outputs/fdr/results/top80_promoters_summits.fasta) | The same 24 sequences; `top80` is a filename limit, not the delivered count |
| [Screening description](model_outputs/fdr/results/analysis.md) | Candidate/null pooling and selected-peak p/q conventions |
| [New per-candidate motif QC](../../../results/oxidative-stress/motif_qc/candidate_motif_qc.csv) | Best six-base cores, scores, candidate-specific shuffle p-values and 48-comparison BH adjustment |
| [New numerical summary](../../../results/oxidative-stress/motif_qc/summary.json) | Input hashes, method and aggregate motif results |
| [PNG](../../../results/oxidative-stress/motif_qc/signal_vs_shuffled.png) / [SVG](../../../results/oxidative-stress/motif_qc/signal_vs_shuffled.svg) | Labelled probability-density comparison with base-count-preserving controls |

The target list supplied available upstream neighbourhoods. The aligned candidate/null pools contain **1,026/420 windows**; pooled member percentiles differ from the ordinary 929-candidate catalogue's target-only ranking. The selected peaks have reported q≤0.05, but selection and minimum window p-values do **not** establish calibrated biological peak FDR. The final table associates 24 upstream regions with 26 retained target-label identifiers; those annotations are not remapped to a later PGAP release.

All 24 stored sequences exactly match transcription-oriented slices of complete assembly **GCF_014131795.1**. `summit` is the 0-based forward-genome start of an 81-bp window on either strand, not an experimentally mapped TSS. Bakta `contig_1`–`contig_4` correspond to `NZ_CP040626.1`–`NZ_CP040629.1` respectively. The checked NCBI FASTA SHA-256 is `a3f45eb7e8dcdd33383a35c1fd2200bd93fc2c44f9bf63a6e9f6a4b94efdbbe2`.

## Motif analysis

The stored FIMO annotation uses six-base reference patterns, not the five de-novo motifs used for the ordinary catalogue. It contains **2 both-hit**, **16 −10-only**, **1 −35-only** and **5 neither** windows. The two both-hit candidates are `cbut_0002` (flavodoxin-associated) and `cbut_0005` (thioredoxin-associated), with **15/17-bp edge gaps** between their six-base hits. This `spacer` convention differs from the ordinary catalogue's centre-to-centre `spacer_bp`.

The new analysis rescored every six-base window in all 24 oriented sequences using the supplied TATAAT/TTGACA reference PWMs and their declared A/T=0.35, C/G=0.15 background. The reference PWMs' scientific source is not recorded. No new MEME discovery, GPU scoring or classifier fitting was performed.

For each candidate, **2,000 independent mononucleotide permutations**, seed 7, preserve its exact length and A/C/G/T counts—not dinucleotide counts. Both observations and controls use the maximum score across the full 81-bp window. Empirical p-values count ties conservatively: `(1 + count(null_max ≥ observed_max))/(1 + 2000)`. Benjamini–Hochberg adjustment covers all **48 candidate × motif comparisons together**, following the existing empirical-FDR code convention.

The windows average **80.50% AT**. Median shuffle p-values are **0.3483 for −10-like** and **0.4450 for −35-like** scores. Three −35 comparisons have unadjusted p≤0.05, but **none of the 48 comparisons passes adjusted q≤0.05**. Short motif resemblance therefore does not separate these windows convincingly from their composition-preserving controls under this analysis. This is conditional, selected-sequence QC—not proof of inactivity or a biological FDR guarantee. Reporter measurements across basal and stress conditions remain necessary.

## Reproduce the new analysis

From the main repository root, using the committed tables and motif file:

```bash
uv run --no-project --python 3.13.5 \
  --with matplotlib==3.11.2 --with numpy==2.5.4 \
  python pipeline/mining/oxidative-stress/scripts/analyze_oxidative_motifs.py
```

Outputs are written to `results/oxidative-stress/motif_qc/`. The result summary pins the final-table SHA-256 `497c694a837ab9e51629c40302a0322119b3ae5683fcf4230ead5baff47e0809` and motif-file SHA-256 `d2d48ec23dc03348467eebb101e8a308fe8ad23f7fe3b01f05653ba11d8a2ab2`.

Re-running original mining requires the matching genome/Bakta annotations, upstream-region inputs, checkpoints and aligned candidate **and** null tracks. These prerequisites are not replaced by the final peak table. Future constructs must also reconcile the codon optimiser's current **GCF_000409755.1** table source with this complete-reference promoter frame and verify the intended gene associations.
