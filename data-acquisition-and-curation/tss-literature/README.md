# T7 — Clostridium TSS datasets (literature search)

**Owner:** Salimah · **Feeds:** T10 (empirical TSS calling)

A catalogue of published TSS-mapping datasets for *Clostridium* and related species, with
SRA run accessions and citations. The literature search itself is solid; the notes below
make it safe to hand to T10.

## The one thing T10 must know: the 47 runs are NOT interchangeable

They span **three different assay types**, and only one *kind* is compatible with
**TSSpredator** (the T10 tool). TSSpredator needs a **differential 5′-end pair**: a library
where primary transcripts are enriched (**treated, "+"**) vs an untreated control (**"−"**),
and it calls a TSS from the +/− enrichment. The 20 runs that provide this are **18 TEX
dRNA-seq** (Fuchs, +TEX/−TEX) plus **2 TAP 5′-end RNA-seq** (Soutourina, TAP+/TAP− — see
"library assignment" below). The other two methods (Capp-Switch, plain 5′ RNA-seq) have
**no −/control library**, so TSSpredator cannot process them; those studies already published
TSS tables, which should be used directly.

| Method | Study | Species / strain | In genus *Clostridium* (taxon 1485)? | In T5's 166? | Runs | T10 path |
|---|---|---|---|---|---|---|
| 5′-end RNA-seq (TAP±) | Soutourina 2020 | *Clostridioides difficile* 630Δerm | **No** (Clostridioides) | No | 2 | **TSSpredator** (enriched = TAP+) |
| dRNA-seq (TEX±) | Fuchs 2021 | *Clostridioides difficile* 630 | **No** (Clostridioides) | No | 18 | **TSSpredator** (enriched = TEX+) |
| Capp-Switch | Hocq 2022 | *C. acetobutylicum* 824; *C. beijerinckii* NCIMB 8052 / DSM 6423 | Yes | Yes | 11 | Published TSS table — **do not** run TSSpredator |
| 5′ RNA-seq | Al-Bassam 2018 | *C. ljungdahlii* ATCC 55383 | Yes | Yes | 16 | Published TSS table — **do not** run TSSpredator |

**20 runs → TSSpredator · 27 runs → published tables · 47 total.**

### Taxonomy trap
The only TSSpredator-ready data is *Clostridioides difficile* — reclassified **out of** genus
*Clostridium* in 2016, so it is **not** among T5's 166 genomes. The three species that *are*
in-genus and in T5 (*acetobutylicum*, *beijerinckii*, *ljungdahlii*) are exactly the ones
whose data needs the published-table path. Using *C. difficile* as cross-species positives is
fine (AT-rich, related — like *B. subtilis* in T8), but it must be mapped to its **own**
genome, never a T5 assembly.

## Reference genome assembly per dataset (verified on NCBI)
TSS calling and lift-over need the exact assembly each paper used. Confirmed RefSeq assemblies
(checked against NCBI Datasets / ENA):

| Strain | RefSeq assembly | Chromosome (+ plasmids) |
|---|---|---|
| *C. difficile* 630 | **GCF_000009205.2** (ASM920v2) | NC_009089.1 (AM180355); plasmid pCD630 NC_008226.1 (AM180356) |
| *C. difficile* 630Δerm | use **GCF_000009205.2** (630) | near-identical derivative — a dedicated van Eijk 2015 630Δerm assembly exists if exact 630Δerm coordinates are needed |
| *C. acetobutylicum* ATCC 824 | **GCF_000008765.1** (ASM876v1) | NC_003030.1 (AE001437); megaplasmid pSOL1 NC_001988.2 (AE001438) |
| *C. beijerinckii* NCIMB 8052 | **GCF_000016965.1** (ASM1696v1) | NC_009617.1 (CP000721) |
| *C. beijerinckii* DSM 6423 | **GCF_900010805.1** | Wasels 2018 genome (chromosome + phage φ6423 + plasmids pNF1/pNF2) |
| *C. ljungdahlii* ATCC 55383 (= DSM 13528) | **GCF_000143685.1** (ASM14368v1) | NC_014328.1 (CP001666) |

## TSSpredator library assignment — Soutourina pair (RESOLVED)
Checked the paper (Soutourina 2020, *Front. Microbiol.*, Methods → "RNA-Seq and 5′-End
RNA-Seq"): those 2 runs are **TAP-based 5′-end RNA-seq** — *"two strand-specific cDNA
libraries … treated or not with Tobacco Acid Pyrophosphatase (TAP +/−)"* (Wurtzel 2010 /
Soutourina 2013 method), **not** TEX dRNA-seq. TAP converts primary-transcript 5′-PPP ends so
they become ligatable, so **TAP+ is the enriched/primary library and TAP− is the control** —
the *same* direction as +TEX, so there's no inversion. For TSSpredator set:

- **enriched ("+TEX" slot) → `SRR11573852` (TAP+)**
- **control ("−TEX" slot) → `SRR11573851` (TAP−)**

(The Fuchs 18 runs are standard TEX dRNA-seq: enriched = TEX+, control = TEX−.)

## 5′ RNA-seq run selection — the SRR6286684–693 gap (RESOLVED)
Checked the SRA run table for **PRJNA418190** (ENA filereport): the project holds 26 runs —
16 **5′ RNA-seq (TSS)** libraries plus 10 others. The skipped **SRR6286684–693** are exactly
those 10 non-TSS runs — 4 standard RNA-seq (`RNA_*`, selection = cDNA: SRR6286684/685/688/689)
and 6 ribosome-profiling (`Ribo_*`: SRR6286686/687/690/691/692/693). Excluding them is
**correct**: T7 keeps precisely the 16 `TSS_*` libraries. The gap is intentional, not an error.

## ✅ Verify items — all resolved
The three review flags are now closed: the Soutourina library assignment (TAP+/−), the
SRR6286684–693 selection, and the genome assemblies (verified on NCBI) — all above. Standing
good-practice check at T10 time: confirm each paper mapped to the same assembly version listed
(esp. *C. difficile* 630 vs 630Δerm).

## Files in this folder
- **`T7_master_curated.csv`** — authoritative clean table (one row per run: method, study,
  genus/T5 flags, library role, BioProject, SRA run, reference assembly, T10 path, citation).
  Fixes the old `TEX / Assay Type` column conflation and trailing-space cells.
- **`SRR_dRNAseq_for_TSSpredator.txt`** — the 20 TSSpredator-compatible runs to fetch for T10
  (18 TEX dRNA-seq + 2 TAP 5′-end).
- **`Download instructions`** — corrected two-path workflow.
- Per-method CSVs + `T7.xlsx` — original literature records, kept as provenance.
- `SRR_Acc_List.txt` (flat 47) — **superseded**: do not feed all 47 to one pipeline.
