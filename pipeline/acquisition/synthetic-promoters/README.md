# Synthetic-promoter evaluation set

T9 merges **93 literature promoters** for T24 discrimination and T26 calibration. They are held out of deployed training. Run from the repository root:

```bash
python pipeline/acquisition/synthetic-promoters/build_literature_promoters.py
```

The standard-library builder reads three committed TSVs, not the workbook, and writes beside its source regardless of working directory. The workflow stages the result to `workflow/inputs/literature_promoters.tsv`.

| File | Study | Rows |
|---|---|---:|
| `literature_promoters - C. acetobutylicum - 1.tsv` | Yang 2017, *ACS Synthetic Biology*, PMID 28602076 | 35 |
| `literature_promoters - C. acetobutylicum - 2.tsv` | Mordaka 2018, *ACS Synthetic Biology*, PMID 29320851 | 33 |
| `literature_promoters - C. phytofermentans.tsv` | Rostain 2022, *ACS Synthetic Biology*, PMID 36427328 | 25 |


## Cleaning

For all 68 acetobutylicum constructs, the builder asserts and removes the constant 120-bp 3′ reporter scaffold. It also removes alignment gaps from seven sequences. Phytofermentans sequences already lack that scaffold. No rows are dropped; `original_sequence` preserves the curated input.

Cleaned lengths are **49–52 bp**. Padding does not recreate 81-bp genomic context or locate a TSS. Strength values retain study-specific units; relative_to_Pthl and mean_AU are not a common quantitative target.

## Schema

```text
  promoter_id          <source>_<original name>, unique
  original_name        name as curated (e.g. 100-1, pPM36-79%-18*, Pcphy1)
  species              Clostridium acetobutylicum | Clostridium phytofermentans
  in_genus             True (acetobutylicum) | False (phytofermentans, see note)
  sequence             cleaned, model-ready promoter (ACGT only, ~49-52 bp)
  seq_len              length of `sequence`
  strength             curated value, verbatim; blank where the study reported N/A
  strength_unit        relative_to_Pthl (Yang/Mordaka) | mean_AU (Rostain)
  n_gaps_removed       alignment-gap '-' chars removed (0 for all but 7 rows)
  scaffold_trimmed_bp  120 for aceto, 0 for phytofermentans
  source               Yang2017 | Mordaka2018 | Rostain2022
  label                1 (all positives)
  taxonomy_note        non-empty only for phytofermentans (see below)
  citation             full literature citation
  original_sequence    untouched curated input string (lossless provenance)
```

Phytofermentans is reclassified as *Lachnoclostridium phytofermentans*, outside Clostridium sensu stricto; its 25 rows have `in_genus=False`. Acetobutylicum rows have `in_genus=True`.

The historical optional T9 training split is not the deployed recipe. If used in a new experiment, evaluate its held-out output, not all 93 rows. Reusing this set across calibration rounds is sensitivity evidence, not independent replication. These source-assay strengths are separate from the [DH5α SVR](../../../docs/ECOLI_STRENGTH_PREDICTIONS.md).
