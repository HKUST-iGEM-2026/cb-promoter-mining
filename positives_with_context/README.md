# Positives with genomic context (for the NTv3 long-context path)

**Canonical builder: `build_positives_context.py`.** The older `create_positives.py` is DEPRECATED
(it loaded the wrong T7 file — a 47-row literature catalog instead of the 5,759 Hocq positives — and
its single global `{contig: seq}` dict collided across the 166 Bakta genomes, so T13 context was
pulled from random genomes). Do not use it.

## What the builder does

Combines the three positive sources into one context-carrying set, each handled correctly:

| Source | rows | context | how |
|---|---|---|---|
| T7b (Hocq/Al-Bassam published TSS) | 5,759 | **real** | coords parsed from `promoter_id` (`{src}_{contig}_{pos}_{strand}`), carved from the RefSeq FASTAs in `data-acquisition-and-curation/tss-literature/ref/` using the exact `extract_published_tss` framing (81 bp, +1 at offset 60). 100% round-trip verified. |
| T13 (housekeeping upstreams, silver) | 14,602 | **real** | loaded PER GENOME (its `genome` column → that genome's own Bakta FASTA) and located by sequence search, so the genome and framing are both correct (no `contig_1` collision). |
| T8 (RegulonDB/DBTBS, cross-species) | 4,336 | sequence-only | cross-species (E. coli / B. subtilis) — no coordinates for our genomes, so `has_genomic_context=False`, core fit to 81 bp, context N-padded. |

Round-trip guarantee: the carved 81 bp core must equal the stored sequence. T7b hard-fails on >1%
mismatch; T13 is located by search so it matches by construction.

## Run it

```bash
source /project/igem2026/cb-promoter-mining/env.sh
conda activate /project/igem2026/envs/promoter-mining-hpc4      # NOT base — ~/.local shadows it
python /project/igem2026/cb-promoter-mining/positives_with_context/build_positives_context.py
```

Optional: `--length 384` (context bp, multiple of 128), `--out <path>`.

## Output

```
/project/igem2026/cb-promoter-mining/positives_with_context/output/positives_with_context.parquet
```

24,697 rows. Columns: `sequence` (81 bp core), `sequence_context` (384 bp, multiple of 128),
`core_offset` (index of the core inside the context), `has_genomic_context`, `source_type`,
`label`, `species`, plus `promoter_id`/`genome`/`contig` where known. 20,361 rows (T7b + T13) carry
real context; T8's 4,336 are sequence-only.

## Downstream: wiring into the NTv3 training path

The two-column design (`sequence` = 81 bp core for ProkBERT/CNN, `sequence_context` = long window
for NTv3) plugs into the existing T15 → T16 flow:

1. **T15 clustering/splits** — `dataset-construction/clustering-and-splits/cdhit_splits.py` clusters on the 81 bp
   `sequence` core (leakage-safe) and carries `sequence_context`/`core_offset` through the splits.
2. **T16 / encode** — `dataset-construction/negative-tokenization/encode.py --model nt
   --sequence-col sequence_context` tokenizes the long window for NTv3 (single-base, N-pad to a
   multiple of 128); ProkBERT/CNN keep `--sequence-col sequence` (the 81 bp core).

**Prerequisite before this can run end-to-end (important):** `cdhit_splits.py` enforces
context on ALL sources or none (the all-or-nothing guard). So the **T14 negatives must also be given
`sequence_context`** before the NTv3 splits can be built — the four negative classes (CDS,
intergenic, downstream) have genome coordinates and can go through `build_context_windows.py`; SRS
is regenerated flank-preserving inside `cdhit_splits.py`. Until the negatives carry context, the
context columns can't flow through T15. This is the remaining task to actually train NTv3 on context.

Note: this is a SEPARATE dataset from the deployed 81 bp pipeline. Do not repoint the current T15
rule at these positives — the deployed models train on the 81 bp cores. This path is additive, for
the NTv3 long-context experiment.
