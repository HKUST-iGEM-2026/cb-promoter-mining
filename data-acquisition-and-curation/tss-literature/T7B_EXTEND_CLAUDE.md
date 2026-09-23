# CLAUDE.md — extend T7b: more verified in-genus positives (for Sine)

> **Cluster setup (login, env, SLURM):** see `infrastructure/HPC4_GUIDE.md` — do that first.

You're helping **Sine** grow T7b's verified in-genus *Clostridium* promoter set. Today T7b =
**1,434** (Hocq 2022 *C. acetobutylicum*, `intergenic_sense` only). Goal: add the two **unused
*C. beijerinckii* strains already in the committed Hocq workbook**, and add **Al-Bassam 2018
*C. ljungdahlii***. The extractor (`extract_published_tss.py`) is general and needs **no code change** —
this is mostly more invocations in `build_published_tss.sh` + getting one supplement file.

## Why this matters
Verified in-genus is the only data allowed into val/test (silver T13 is train-only). Right now val/test
hold only ~268 in-genus promoters because T7b is small. Growing T7b ~4× lifts the in-genus eval slice to
~1,000+, finally enough to measure in-genus performance honestly. These are **verified**, so under the
verified-only split ~20% flows to val/test automatically.

## Correction to an earlier note
There is **no "primary" TSS class** in Hocq. The `Class` column is `{intergenic_sense, intragenic_sense,
intergenic_antisense, intragenic_antisense}`. The promoter-bearing class is **`intergenic_sense`** (TSS in
an intergenic region, sense strand → a real upstream promoter). Keep using `intergenic_sense` only;
`intragenic_sense` is *internal* TSS (often processing/internal, riskier) and the antisense classes are
not promoters for the annotated gene. So the win is **more strains, same class** — not more classes.

---

## Part A — Hocq *C. beijerinckii* (the big, easy win; data already committed)

The workbook `published_tss/hocq2022_cac_tss.xlsx` has 7 sheets; you currently read only `ATCC_824_25RPM`.
Add `DSM_6423_25RPM` and `NCIMB_8052_25RPM` (both *C. beijerinckii*). Keep the **25 RPM** sheets (Hocq's
higher-confidence threshold; the 10 RPM sheets ~triple the count but are noisier — only add them if you
want a lower-confidence tier, and tag them so).

Genomes (both in T5's 166 → stage from the T5 zip like *C. ljungdahlii*, or `datasets download`):
- *C. beijerinckii* **NCIMB 8052** → `GCF_000016965.1` (chromosome **NC_009617.1**; confirm single-replicon).
- *C. beijerinckii* **DSM 6423** → `GCF_900010805.1` (chromosome + phage φ6423 + plasmids pNF1/pNF2 →
  **multiple replicons**, so `--contig-col Locus` + a `--contig-map` are REQUIRED).

**Step A1 — the `Locus` labels per strain (already inspected for you — the contig-map must cover EVERY
label, or those TSS drop as `unmapped_contig` and the extractor's zero-row `assert` may fire):**
- **NCIMB 8052** — `Locus` = {`Chromosome`} → single replicon → `--contig-map "Chromosome=NC_009617.1"`.
- **DSM 6423** — `Locus` = {`Chromosome`, `Phi_6423`, `pNF1`, `pNF2`} → **4 replicons**. Stage the genome,
  read its 4 FASTA record ids, and map each `Locus` label to the matching accession:
  ```bash
  grep "^>" "$REF_DIR/GCF_900010805.1.fna"      # -> 4 record ids (NZ_… for this RefSeq assembly)
  ```
  Build `--contig-map "Chromosome=<chr>,Phi_6423=<phage>,pNF1=<p1>,pNF2=<p2>"` — all four (most TSS are on
  the Chromosome, but the phage/plasmid TSS are real in-genus promoters too; omitting a label drops them).
  (Accession parity matters: match the exact record id incl. the version suffix.)

**Step A2 — refactor the Hocq stanza in `build_published_tss.sh` into a per-strain loop** (extractor
unchanged; columns are identical across sheets: `Genome_position`, `Strand`, `Locus`, `Class`):
```bash
# sheet : genome accession : species : Locus->accession contig-map  (FILL the two C. beijerinckii maps)
run_hocq () {  # $1 sheet  $2 acc  $3 species  $4 contigmap
  local fna="$REF_DIR/$2.fna"
  stage_genome "$2" "$fna" || { echo "[T7b] $2 genome missing — skip $1"; return; }
  python "$ING" --tss-table "$HOCQ" --sheet "$1" --genome "$fna" \
    --species "$3" --source Hocq2022 \
    --pos-col Genome_position --strand-col Strand --contig-col Locus --contig-map "$4" \
    --filter-col Class --filter-vals intergenic_sense \
    --out "$REF_DIR/hocq_$1.parquet" && PARTS+=("$REF_DIR/hocq_$1.parquet")
}
run_hocq ATCC_824_25RPM   GCF_000008765.1 "Clostridium acetobutylicum" "Chromosome=NC_003030.1,pSOL=NC_001988.2"
run_hocq DSM_6423_25RPM   GCF_900010805.1 "Clostridium beijerinckii"   "Chromosome=<chr>,Phi_6423=<phage>,pNF1=<p1>,pNF2=<p2>"   # fill the 4 ids: grep ">" the staged .fna (Step A1)
run_hocq NCIMB_8052_25RPM GCF_000016965.1 "Clostridium beijerinckii"   "Chromosome=NC_009617.1"
```
(`stage_genome` = factor out the existing unzip-from-T5 / `datasets download` logic. `promoter_id` already
includes source+contig+pos+strand, so no cross-strain id collisions; the final concat dedups.)

Expected add: ~1,855 (DSM 6423) + ~2,482 (NCIMB 8052) ≈ **+4,337** before contig-end/non-ACGT drops.

## Part B — Al-Bassam 2018 *C. ljungdahlii* (second species; needs the supplement)

The runner already has an Al-Bassam stanza + can stage the genome (`GCF_000143685.1`, single replicon
`NC_014328.1`). The blocker is the **TSS table itself isn't in `published_tss/`**.
1. Download the **correct** Al-Bassam 2018 supplementary TSS list (NB the README warning: "Supplementary
   Data 2" is the gene/TE table, *not* the TSS list — get the TSS supplement). Save as
   `published_tss/albassam2018_clj_tss.{tsv,csv,xlsx}` (commit via Git-LFS).
2. Inspect its columns and set the env vars the runner passes: `CLJ_POS_COL`, `CLJ_STRAND_COL`
   (+ `CLJ_SHEET` if xlsx, `CLJ_SIGMA_COL` if present). Single replicon → no `--contig-col` needed.
3. Re-run; the stanza ingests it automatically.

## Validate (after each part)
```python
import pandas as pd
d = pd.read_parquet("infrastructure/T3/snakemake_scaffold/inputs/published_tss_promoters.parquet")
print(len(d), d.species.value_counts().to_dict(), d.source_db.value_counts().to_dict())
assert (d.label==1).all() and d.in_genus.all()
assert (d.sequence.str.len()==81).all() and d.sequence.str.fullmatch("[ACGT]+").all()
assert d.promoter_id.is_unique
```
Sanity: a chunk of windows should show a `TATAAT`-like −10 around offset ~40–46 (TSS at offset 60). Then
**re-run T15** and confirm the val/test verified-in-genus count rises (T15 summary's
`verified_in_genus` per split).

## Caveats
- **Keep the framing identical** (81 bp, TSS at offset 60) — it's hard-coded in the extractor; don't change it.
- **Accession parity** is the #1 failure (zero-row assert) — match `Locus` labels to the exact FASTA record ids incl. the `.1`/`.2` version suffix.
- **Threshold consistency**: use the 25 RPM sheets for all strains (don't mix 25 + 10 RPM silently).
- These are still **3 in-genus species, not the target** *C. butyricum* (still excluded) — this improves in-genus *coverage*, not target leakage.

## What to commit / push
- **`build_published_tss.sh`** — your per-strain-loop edit + the 3 `run_hocq` lines. This is the tracked,
  reproducible source and **must go back** (hand the diff to Oak if HPC4 push is blocked).
- **The Al-Bassam supplement** (only if you download it) — via **Git-LFS**. `published_tss/` has its own
  `.gitignore`, so force-add it: `git add -f published_tss/albassam2018_clj_tss.*`. The Hocq workbook is
  **already committed**, so the *C. beijerinckii* half needs no new data commit — just the script edit.
- **Do NOT commit:** the regenerated `published_tss_promoters.parquet`, the staged genomes, or the T15
  re-run outputs — all git-ignored (regenerated on the shared clone).

## One-line summary
Add the two unused *C. beijerinckii* sheets from the Hocq workbook (≈ +4,337) and the Al-Bassam
*C. ljungdahlii* table → verified in-genus ~1,434 → ~5,700+, which flows into val/test and ~4×'s the
in-genus eval slice. Same extractor, same `intergenic_sense` class, just more invocations.
