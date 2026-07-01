#!/usr/bin/env bash
# =====================================================================
# T7b — build in-genus published-TSS promoter POSITIVES -> $OUT (parquet).
#
# Stages reference genome(s), runs extract_published_tss.py on the published TSS tables in
# T7/published_tss/, and concatenates. If no usable table is present — including when the T5 zip or
# the supplements are still Git-LFS pointers (no `git lfs pull`) — it writes an EMPTY but schema-valid
# parquet so the pipeline still runs. See PUBLISHED_TSS_README.md for sources.
#
# Confirmed table layouts (inspected 2026-06):
#   Hocq 2022  hocq2022_cac_tss.xlsx  -> sheet ATCC_824_25RPM, cols Genome_position/Strand/Locus,
#              Locus in {Chromosome,pSOL} -> remap to NC_003030.1 / NC_001988.2; keep Class=intergenic_sense
#              (canonical promoter TSS; 1436 of 3441). Genome GCF_000008765.1 (2 replicons) auto-fetched.
#   Al-Bassam  albassam2018_clj_tss.*  -> C. ljungdahlii (NC_014328.1, single replicon, staged from T5 zip).
#              NB: the paper's "Supplementary Data 2" is the gene/TE table, NOT the TSS list — use the
#              correct Supplementary Data file and set CLJ_POS_COL / CLJ_STRAND_COL from it.
# =====================================================================
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
GENOMES_ZIP="${GENOMES_ZIP:-$REPO/data-acquisition-and-curation/T5/Clostridium_genomes_v2_166.zip}"
OUT="${OUT:-$REPO/infrastructure/T3/snakemake_scaffold/inputs/published_tss_promoters.parquet}"
T7="$REPO/data-acquisition-and-curation/T7"
TBL_DIR="$T7/published_tss"
REF_DIR="$T7/ref"
mkdir -p "$REF_DIR" "$(dirname "$OUT")"
ING="$T7/extract_published_tss.py"
PARTS=()

# A file is USABLE only if it exists, is non-empty, and is NOT an unmaterialised Git-LFS pointer
# (a ~130-byte text stub whose first line is "version https://git-lfs..."). On a checkout without
# `git lfs pull` the T5 zip and the TSS supplements are pointers — skip them gracefully, never crash.
is_real() {
  [ -s "$1" ] || return 1
  local first=""; read -r first < "$1" 2>/dev/null || true
  case "$first" in "version https://git-lfs"*) return 1 ;; *) return 0 ;; esac
}

# --- Al-Bassam 2018 — C. ljungdahlii (genome staged from the T5 zip; single replicon NC_014328.1) ---
CLJ_FNA="$REF_DIR/GCF_000143685.1_ASM14368v1_genomic.fna"
if [ ! -s "$CLJ_FNA" ]; then
  if is_real "$GENOMES_ZIP"; then
    unzip -p "$GENOMES_ZIP" 'ncbi_dataset/data/GCF_000143685.1/*_genomic.fna' > "$CLJ_FNA" \
      || { echo "[T7b] could not extract C. ljungdahlii ref from $GENOMES_ZIP — skipping Al-Bassam."; rm -f "$CLJ_FNA"; }
  else
    echo "[T7b] T5 genomes zip is a Git-LFS pointer or missing ($GENOMES_ZIP) — run 'git lfs pull'. Skipping Al-Bassam."
  fi
fi
ALB=$(ls "$TBL_DIR"/albassam2018_clj_tss.* 2>/dev/null | head -1 || true)
if [ -n "$ALB" ] && is_real "$ALB" && [ -s "$CLJ_FNA" ]; then
  python "$ING" --tss-table "$ALB" --sheet "${CLJ_SHEET:-0}" \
    --genome "$CLJ_FNA" --species "Clostridium ljungdahlii" --source AlBassam2018 \
    --pos-col "${CLJ_POS_COL:-TSS_position}" --strand-col "${CLJ_STRAND_COL:-strand}" \
    ${CLJ_CONTIG_COL:+--contig-col "$CLJ_CONTIG_COL"} ${CLJ_CONTIG_MAP:+--contig-map "$CLJ_CONTIG_MAP"} \
    ${CLJ_SIGMA_COL:+--sigma-col "$CLJ_SIGMA_COL"} --out "$REF_DIR/albassam.parquet" \
    && PARTS+=("$REF_DIR/albassam.parquet") || echo "[T7b] Al-Bassam ingest failed — check CLJ_* cols vs the table."
elif [ -n "$ALB" ]; then
  echo "[T7b] Al-Bassam: table present but unusable (Git-LFS pointer, or C. ljungdahlii genome not staged) — skipping."
fi

# --- Hocq 2022 — C. acetobutylicum ATCC 824 (VALIDATED config) ---
HOCQ=$(ls "$TBL_DIR"/hocq2022_cac_tss.* 2>/dev/null | head -1 || true)
if [ -n "$HOCQ" ] && ! is_real "$HOCQ"; then
  echo "[T7b] Hocq table is a Git-LFS pointer ($HOCQ) — run 'git lfs pull'. Skipping."
elif [ -n "$HOCQ" ]; then
  # GCF_000008765.1 (NC_003030.1 chromosome + pSOL1 NC_001988.2) is NOT in the T5 zip -> auto-fetch.
  CAC_FNA="${CAC_FNA:-$REF_DIR/GCF_000008765.1.fna}"
  if [ ! -s "$CAC_FNA" ] && command -v datasets >/dev/null 2>&1; then
    ( cd "$REF_DIR" && datasets download genome accession GCF_000008765.1 --include genome --filename cac.zip \
      && unzip -o -q cac.zip -d cac && cat cac/ncbi_dataset/data/GCF_000008765.1/*_genomic.fna > "$CAC_FNA" ) \
      || echo "[T7b] Hocq genome auto-fetch failed — set CAC_FNA=<GCF_000008765.1 fasta> manually."
  fi
  if [ -s "$CAC_FNA" ]; then
    python "$ING" --tss-table "$HOCQ" --sheet "${CAC_SHEET:-ATCC_824_25RPM}" \
      --genome "$CAC_FNA" --species "Clostridium acetobutylicum" --source Hocq2022 \
      --pos-col Genome_position --strand-col Strand --contig-col Locus \
      --contig-map "Chromosome=NC_003030.1,pSOL=NC_001988.2" \
      --filter-col Class --filter-vals "${CAC_CLASSES:-intergenic_sense}" \
      --out "$REF_DIR/hocq.parquet" && PARTS+=("$REF_DIR/hocq.parquet") \
      || echo "[T7b] Hocq ingest failed — check the sheet/cols vs the table."
  else
    echo "[T7b] Hocq: need the C. acetobutylicum genome — set CAC_FNA=<GCF_000008765.1 fasta> or install ncbi-datasets-cli."
  fi
fi

# --- Hocq 2022 — C. beijerinckii (two MORE strains in the SAME workbook; +~4,337 in-genus positives) ---
# Same extractor, same intergenic_sense class — just the two unused 25RPM sheets. Genomes are staged from
# the T5 zip (datasets fallback), and each contig-map is DERIVED from the genome's own FASTA headers
# (chromosome = longest record; plasmids/phage matched by name) so no accession is hard-coded. Any genome
# that can't be staged/mapped is skipped gracefully (fewer positives, never a crash).
_stage_genome() {                     # $1 accession -> prints path to .fna (empty on failure)
  local acc="$1" fna="$REF_DIR/${acc}.fna"
  if [ ! -s "$fna" ]; then
    if is_real "$GENOMES_ZIP" && unzip -p "$GENOMES_ZIP" "ncbi_dataset/data/${acc}/*_genomic.fna" > "$fna" 2>/dev/null && [ -s "$fna" ]; then
      :
    elif command -v datasets >/dev/null 2>&1; then
      ( cd "$REF_DIR" && datasets download genome accession "$acc" --include genome --filename "${acc}.zip" \
        && unzip -o -q "${acc}.zip" -d "${acc}_d" \
        && cat "${acc}_d"/ncbi_dataset/data/"${acc}"/*_genomic.fna > "$fna" ) 2>/dev/null || rm -f "$fna"
    fi
  fi
  [ -s "$fna" ] && echo "$fna" || true
}
_auto_contig_map() {                  # $1 fna  $2 comma-list of Locus labels -> "Label=recordID,..."
  python - "$1" "$2" <<'PY'
import sys
fna, labels = sys.argv[1], sys.argv[2].split(",")
recs=[]; rid=None; desc=""; ln=0
def flush():
    if rid is not None: recs.append([rid, desc.lower(), ln])
for line in open(fna):
    if line.startswith(">"):
        flush(); h=line[1:].strip(); rid=h.split()[0]; desc=h; ln=0
    else:
        ln += len(line.strip())
flush()
def pick(label):
    L=label.lower()
    keys={"chromosome":["chromosome"], "pnf1":["pnf1"], "pnf2":["pnf2"],
          "phi_6423":["phage","prophage"]}.get(L, [L])   # NOT "6423" — it's in the strain name of every record
    for rid_, d, _ in recs:
        if any(k in d for k in keys): return rid_
    if L=="chromosome" and recs:                 # fallback: chromosome = the LONGEST record
        return max(recs, key=lambda r: r[2])[0]
    return ""
pairs=[(lab, pick(lab)) for lab in labels]
print(",".join(f"{lab}={r}" for lab, r in pairs if r))
PY
}
_run_hocq_beij() {                    # $1 sheet  $2 accession  $3 comma-list of Locus labels
  local fna cmap
  fna="$(_stage_genome "$2")"
  if [ -z "$fna" ]; then echo "[T7b] C. beijerinckii $2 genome not staged — skipping $1 (need 'git lfs pull' for the T5 zip, or ncbi-datasets-cli + internet)."; return 0; fi
  cmap="$(_auto_contig_map "$fna" "$3")"
  if [ -z "$cmap" ]; then echo "[T7b] could not derive Locus->record map for $2 — skipping $1."; return 0; fi
  echo "[T7b] $1: contig-map = $cmap"
  python "$ING" --tss-table "$HOCQ" --sheet "$1" --genome "$fna" \
    --species "Clostridium beijerinckii" --source Hocq2022 \
    --pos-col Genome_position --strand-col Strand --contig-col Locus --contig-map "$cmap" \
    --filter-col Class --filter-vals "${CAC_CLASSES:-intergenic_sense}" \
    --out "$REF_DIR/hocq_$1.parquet" && PARTS+=("$REF_DIR/hocq_$1.parquet") \
    || echo "[T7b] $1 ingest failed — check the sheet/cols vs the table."
}
if [ -n "$HOCQ" ] && is_real "$HOCQ"; then
  _run_hocq_beij NCIMB_8052_25RPM GCF_000016965.1 "Chromosome"
  _run_hocq_beij DSM_6423_25RPM   GCF_900010805.1 "Chromosome,Phi_6423,pNF1,pNF2"
fi

if [ ${#PARTS[@]} -eq 0 ]; then
  echo "[T7b] no usable published TSS tables -> writing an EMPTY (schema-valid) parquet. See $T7/PUBLISHED_TSS_README.md"
  python - "$OUT" <<'PY'
import sys, pandas as pd
cols=["species","promoter_id","sequence","strand","sigma_factor","source_db","in_genus","label"]
pd.DataFrame(columns=cols).to_parquet(sys.argv[1], index=False)
PY
else
  python - "$OUT" "${PARTS[@]}" <<'PY'
import sys, pandas as pd
out, parts = sys.argv[1], sys.argv[2:]
df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).drop_duplicates(subset=["promoter_id"])
df.to_parquet(out, index=False)
print(f"[T7b] wrote {len(df)} verified in-genus positives -> {out}")
PY
fi
