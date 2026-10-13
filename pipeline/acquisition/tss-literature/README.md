# TSS sources

The 47-run catalogue combines four studies, not one compatible assay. Deployed labels use published tables; the optional raw-read track is separate. [Pipeline](../../../docs/PIPELINE.md) gives the downstream construction.

## Published tables

- **Hocq 2022**, [doi:10.1128/spectrum.02288-21](https://doi.org/10.1128/spectrum.02288-21): C. acetobutylicum ATCC 824 and C. beijerinckii NCIMB 8052/DSM 6423. [Workbook](published_tss/hocq2022_cac_tss.xlsx) uses Git LFS. The builder selects three 25RPM sheets and intergenic_sense records. Historical pre-split output is 5,759; deployed xs retains 5,506 after deduplication/splitting.
- **Al-Bassam 2018**, [doi:10.1038/s41467-018-06993-6](https://doi.org/10.1038/s41467-018-06993-6): 1,245 primary C. ljungdahlii starts in the [benchmark table](../../evaluation/ljungdahlii_holdout_tss.tsv). Keep this out of `published_tss/`, which the builder ingests as training positives.

From the repository root:

```bash
git lfs pull
bash pipeline/acquisition/tss-literature/build_published_tss.sh
```

This needs pandas, openpyxl, a parquet engine and matching FASTAs. Automatic staging also needs datasets, unzip and network access. Inspect drop messages and output rows: missing inputs can yield partial/empty outputs despite a successful exit.

## Optional C. difficile reads

Only 20 runs enter the prepared TSSpredator path: 18 Fuchs TEX libraries and two Soutourina TAP libraries. The other 27 Capp-Switch/5′ RNA-seq runs lack TEX controls and use the papers' tables instead.

[Bio environment](../../../environments/environment-bio.yml) supplies SRA tools. Download on the machine that will process the reads:

```bash
source env.sh             # from the repository root; retains PROMOTER_REPO after changing directory
conda activate promoter-mining-bio
cd /path/to/work          # replace with a work volume with tens of GB free

# Fetch only the 20 TSSpredator-compatible runs.
prefetch --option-file "$PROMOTER_REPO/pipeline/acquisition/tss-literature/SRR_dRNAseq_for_TSSpredator.txt"
for srr in $(cat "$PROMOTER_REPO/pipeline/acquisition/tss-literature/SRR_dRNAseq_for_TSSpredator.txt"); do
    fasterq-dump --split-spot --skip-technical "$srr" -O . && gzip -f "${srr}.fastq"
done
```

The [driver](../cdifficile-drnaseq/run_tss_pipeline.sh) also downloads in stage 2, so this loop is optional. Its default is strain 630, GCF_000009205.2/NC_009089.1, with one reference/annotation pair for both studies. Check study-specific strain compatibility, including 630Δerm, before running. This prepared-only track supplies no deployed training positives.
