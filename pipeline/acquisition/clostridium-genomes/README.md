# Historical genome cohort

Downloaded **2026-06-04 at 12:25** through [NCBI Datasets](https://www.ncbi.nlm.nih.gov/datasets/genome/?taxon=1485&reference_only=true), searching Clostridium with the RefSeq reference filter. DSM 10702 was excluded.

`Clostridium_genomes_v2_166.zip` contains 166 assemblies and uses Git LFS. [genome_list.tsv](genome_list.tsv) is a headerless two-column accession/species inventory with 166 data rows. Preserve these versions; today's live taxon query may return a different cohort.

From the repository root:

```bash
git lfs pull
unzip -q pipeline/acquisition/clostridium-genomes/Clostridium_genomes_v2_166.zip \
  -d pipeline/acquisition/clostridium-genomes/genome_dataset
```

**Silver positives** are putative promoter-positive upstream DNA labelled from predicted housekeeping-gene annotations and genomic position, not mapped TSS or experimentally measured promoters. Putative-negative windows are not experimentally established promoter-free DNA.

The cohort includes C. ljungdahlii GCF_000143685.1 and C. autoethanogenum GCF_001484725.1. The latter benchmark used a different assembly, GCF_000484505.1. Both strains supplied silver positives and putative-negative training windows, but not their published benchmark TSS. No C. butyricum strain is in the archive. Sequence provenance does not establish backbone pretraining exclusion or measured operons; [pipeline documentation](../../../docs/PIPELINE.md) records label construction.
