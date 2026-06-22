# T7b — staging the published in-genus TSS tables

**Why:** the fine-tune's training positives contain no *verified in-genus* Clostridium promoter
(CLAUDE.md §7). The genuine in-genus signal is the published TSS from two in-genus 5′-end studies.
This step turns those TSS into 81 bp promoter positives (`inputs/published_tss_promoters.parquet`)
that T15 concatenates with the T8 cross-species positives.

**Status:** the ingester (`extract_published_tss.py`) and runner (`build_published_tss.sh`) are
production-ready and the C. ljungdahlii reference is auto-staged from the T5 zip. The only thing
missing is the **TSS coordinate tables**, which are journal *supplementary files* (binary `.xlsx`)
that must be **downloaded by hand** — they can't be fetched programmatically.

## The two sources

| Study | Species (genome) | Paper | TSS table |
|---|---|---|---|
| **Al-Bassam 2018** | *C. ljungdahlii* DSM 13528 — **NC_014328.1** (single replicon; auto-staged from the T5 zip) | PMC6203783 / doi:10.1186/s12934-018-1008-6 | supplementary TSS table |
| **Hocq 2022** | *C. acetobutylicum* ATCC 824 — **NC_003030.1** + pSOL1 **NC_001988.2** (2 replicons; **not** in the T5 zip) | PMC9045289 / doi:10.1128/spectrum.02288-21 | supplementary TSS table (≈12,114 TSS across 3 clostridia) |

## How to populate it

1. **Download** the supplementary TSS table from the paper (PMC links above → "Supplementary
   material" / journal site). Open the `.xlsx` and **save the TSS sheet as TSV**.
2. **Place it** under `data-acquisition-and-curation/T7/published_tss/` with the expected name:
   - Al-Bassam → `published_tss/albassam2018_clj_tss.tsv`
   - Hocq → `published_tss/hocq2022_cac_tss.tsv`
3. **Inspect the columns** and set the matching env vars (the script defaults are placeholders):
   - Al-Bassam: `CLJ_POS_COL`, `CLJ_STRAND_COL`, optional `CLJ_SIGMA_COL`.
   - Hocq: `CAC_POS_COL`, `CAC_STRAND_COL`, `CAC_CONTIG_COL` (REQUIRED — 2 replicons), optional `CAC_SIGMA_COL`,
     plus `CAC_FNA=<path to GCF_000008765.1 FASTA>` (download it from NCBI; it's not in the T5 zip).
4. **Run** (either directly or via the Snakemake `t7b_published_tss` rule):
   ```bash
   CLJ_POS_COL=<pos col> CLJ_STRAND_COL=<strand col> \
   bash data-acquisition-and-curation/T7/build_published_tss.sh
   ```

## Coordinate-parity warning (the #1 silent-failure mode)

The table's contig/replicon values must match the FASTA headers **exactly**:
- C. ljungdahlii FASTA header is `>NC_014328.1` — if the table uses `NC_014328` (no `.1`) or `CP001666.1`,
  every row is dropped. The ingester now **asserts a non-zero result** and prints the drop reasons, so a
  parity mismatch fails loudly instead of emitting an empty table.
- C. acetobutylicum is 2 replicons → `--contig-col` is **required**; the ingester hard-errors otherwise
  (so pSOL1 TSS can't be silently sliced from the chromosome).

Coordinates are treated as **1-based**, TSS at offset 60 of the 81 bp [−60,+20] window (RegulonDB/PT5/PT6
framing). Confirm each paper's coordinate convention before trusting downstream scores.

## Alternative that needs NO download (for the §7 goal)

If the goal is simply "get verified in-genus positives into training," the repo already has 68 verified
in-genus *C. acetobutylicum* promoters in T9 (`literature_promoters.tsv`, `in_genus=True`). Splitting T9
(train on some, hold out the rest) achieves that today without any external file — a complementary option
to discuss with Oak alongside the §7 decision.
