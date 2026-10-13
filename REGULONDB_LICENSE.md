# RegulonDB data licence

RegulonDB's [official policy](https://github.com/regulondbunam/regulondb-docs/blob/main/00_about_policies/license.md), version 1.0 (April 2025), permits academic/research/noncommercial sharing and adaptation under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Commercial use requires a separate agreement: `regulondb@ccg.unam.mx`.

## Conditions

Attribute RegulonDB and the original supporting publications, link the licence/policy, identify processing changes and preserve identifiers/notices. The provider requests links to corresponding [RegulonDB objects](https://regulondb.ccg.unam.mx/). Do not imply endorsement.

## Project data

[PromoterS_Set.txt](pipeline/acquisition/curated-promoters/PromoterS_Set.txt) retains its original header and records: release **13.5**, dated **28 January 2025**. The export cites Salgado, Gama-Castro et al., “RegulonDB v12.0: a comprehensive resource of transcriptional regulation in E. coli K-12,” [doi:10.1093/nar/gkad1072](https://doi.org/10.1093/nar/gkad1072). The paper was published online in 2023 and appears in the 2024 *Nucleic Acids Research* issue; the raw export citation is unchanged. Paper version, journal year and export release are distinct.

Processing sanitises sequences, filters unsuitable records and adds metadata. The curated table combines 3,963 RegulonDB and 373 DBTBS records; deployed splits retain 3,827 RegulonDB positive windows after preparation/deduplication. [Archived prototypes](archive/prototypes/prokbert-finetune/) also contain raw/adapted exports and gene-derived negatives. These transformations do not create experimentally verified labels.

This notice covers RegulonDB portions of mixed datasets, not DBTBS, RefSeq or other inputs. Project software/documentation remains MIT; ProkBERT artifacts declare CC-BY-NC-4.0 and CNN/Evo2 probe artifacts CC-BY-4.0. These declarations do not waive base-model or data terms. Hub packages contain fitted parameters/helpers, not raw RegulonDB exports or training tables, and bundle this notice for source attribution.
