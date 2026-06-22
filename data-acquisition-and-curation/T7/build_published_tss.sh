#!/usr/bin/env bash
# =====================================================================
# T7b — build in-genus published-TSS promoter POSITIVES -> $OUT (parquet).
#
# Stages reference genome(s), runs extract_published_tss.py on the published TSS tables in
# T7/published_tss/, and concatenates. If no usable table is present it writes an EMPTY but
# schema-valid parquet so the pipeline still runs. See PUBLISHED_TSS_README.md for sources.
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

# --- Al-Bassam 2018 — C. ljungdahlii (genome staged from the T5 zip; single replicon NC_014328.1) ---
CLJ_FNA="$REF_DIR/GCF_000143685.1_ASM14368v1_genomic.fna"
[ -f "$CLJ_FNA" ] || unzip -p "$GENOMES_ZIP" 'ncbi_dataset/data/GCF_000143685.1/*_genomic.fna' > "$CLJ_FNA"
ALB=$(ls "$TBL_DIR"/albassam2018_clj_tss.* 2>/dev/null | head -1 || true)
if [ -n "$ALB" ]; then
  python "$ING" --tss-table "$ALB" --sheet "${CLJ_SHEET:-0}" \
    --genome "$CLJ_FNA" --species "Clostridium ljungdahlii" --source AlBassam2018 \
    --pos-col "${CLJ_POS_COL:-TSS_position}" --strand-col "${CLJ_STRAND_COL:-strand}" \
    ${CLJ_CONTIG_COL:+--contig-col "$CLJ_CONTIG_COL"} ${CLJ_CONTIG_MAP:+--contig-map "$CLJ_CONTIG_MAP"} \
    ${CLJ_SIGMA_COL:+--sigma-col "$CLJ_SIGMA_COL"} --out "$REF_DIR/albassam.parquet" \
    && PARTS+=("$REF_DIR/albassam.parquet") || echo "[T7b] Al-Bassam ingest failed — check CLJ_* cols vs the table."
fi

# --- Hocq 2022 — C. acetobutylicum ATCC 824 (VALIDATED config) ---
HOCQ=$(ls "$TBL_DIR"/hocq2022_cac_tss.* 2>/dev/null | head -1 || true)
if [ -n "$HOCQ" ]; then
  # GCF_000008765.1 (NC_003030.1 chromosome + pSOL1 NC_001988.2) is NOT in the T5 zip -> auto-fetch.
  CAC_FNA="${CAC_FNA:-$REF_DIR/GCF_000008765.1.fna}"
  if [ ! -f "$CAC_FNA" ] && command -v datasets >/dev/null 2>&1; then
    ( cd "$REF_DIR" && datasets download genome accession GCF_000008765.1 --include genome --filename cac.zip \
      && unzip -o -q cac.zip -d cac && cat cac/ncbi_dataset/data/GCF_000008765.1/*_genomic.fna > "$CAC_FNA" )
  fi
  if [ -f "$CAC_FNA" ]; then
    python "$ING" --tss-table "$HOCQ" --sheet "${CAC_SHEET:-ATCC_824_25RPM}" \
      --genome "$CAC_FNA" --species "Clostridium acetobutylicum" --source Hocq2022 \
      --pos-col Genome_position --strand-col Strand --contig-col Locus \
      --contig-map "Chromosome=NC_003030.1,pSOL=NC_001988.2" \
      --filter-col Class --filter-vals "${CAC_CLASSES:-intergenic_sense}" \
      --out "$REF_DIR/hocq.parquet" && PARTS+=("$REF_DIR/hocq.parquet")
  else
    echo "[T7b] Hocq: need the C. acetobutylicum genome — set CAC_FNA=<GCF_000008765.1 fasta> or install ncbi-datasets-cli."
  fi
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
