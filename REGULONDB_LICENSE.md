# RegulonDB data license and attribution

RegulonDB's [official License and Terms of Use](https://github.com/regulondbunam/regulondb-docs/blob/main/00_about_policies/license.md), **Version 1.0, April 2025**, licenses all curated RegulonDB content for academic, research and noncommercial use under **[Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/)**.

## Permitted reuse and conditions

Noncommercial users may **share** (copy and redistribute) and **adapt** (remix, transform and build upon) the data. This published permission covers noncommercial reuse and sharing; it is not an unresolved requirement to obtain a separate permission email for those permitted uses.

When reusing or redistributing RegulonDB content:

- Attribute **RegulonDB** and the original publications supporting the reused data.
- Link to [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) and the [official policy](https://github.com/regulondbunam/regulondb-docs/blob/main/00_about_policies/license.md).
- Indicate changes, including filtering, sequence normalisation, window extraction, relabelling or integration with other sources.
- Preserve source identifiers and provenance. The provider also requests direct links to corresponding objects in the [RegulonDB interface](https://regulondb.ccg.unam.mx/) when integrating or redistributing records.
- Do not imply endorsement by RegulonDB or remove retained upstream notices.

**Commercial use of RegulonDB content requires a separate licensing agreement** with the provider: `regulondb@ccg.unam.mx`.

## Data used by this project

The reorganised source checkout (branch `reorganise`) stores the promoter-table source at `data-acquisition-and-curation/curated-promoters/PromoterS_Set.txt`; older branches may retain the corresponding export in a different directory. Its original header identifies **RegulonDB release 13.5**, dated **28 January 2025**. This is a promoter-table export, not the entire database. The raw header and records are retained unchanged, including the older notice and original source citation; the current reuse policy linked above provides the noncommercial sharing/adaptation permission.

The export cites:

> Salgado H, Gama-Castro S, et al. **RegulonDB v12.0: a comprehensive resource of transcriptional regulation in E. coli K-12.** *Nucleic Acids Research*, 2023. [doi:10.1093/nar/gkad1072](https://doi.org/10.1093/nar/gkad1072).

The paper's v12.0 title and the export's v13.5 release identify different things. Retain citations to the original publications attached to individual reused records as well as the database citation.

Project changes to RegulonDB-derived data include:

- The curated positive table normalises/sanitises DNA sequences, filters unsuitable records, adds source/species/label metadata and combines **3,963 RegulonDB records with 373 DBTBS records**.
- Model-input preparation uses 81 bp windows and sequence deduplication/clustering. The historical deployed split contains **3,827 RegulonDB-derived positive windows**.
- On the same reorganised branch, `prototypes/prokbert-finetune/` retains promoter and gene-sequence exports, sequence-only/labelled promoter tables and gene-derived 81 bp negative examples. These are raw or adapted data, not newly authored RegulonDB records.

These are descriptions of processing and provenance, not claims that labels, model outputs or individual promoters have been experimentally validated.

## Scope: data, source code and fitted models

This notice applies to **RegulonDB content and its data adaptations**, including the RegulonDB portions of mixed datasets. It does not relicense DBTBS, RefSeq, published TSS datasets or other third-party inputs; their terms and citations remain separate.

The project's root `LICENSE` remains **MIT for project software/documentation**. Model/probe artifacts retain their separate declared licenses: **CC-BY-NC-4.0 for ProkBERT** and **CC-BY-4.0 for the CNN and Evo2 probe**. This data notice does not change those artifact declarations or waive base-model licenses, and does not claim unrestricted commercial clearance.

The three Hugging Face model packages contain fitted parameters, configurations, helper code and documentation, **not the raw RegulonDB exports or training datasets**. This notice is bundled to preserve training-source licensing and attribution.
