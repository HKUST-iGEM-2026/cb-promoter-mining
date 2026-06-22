#!/usr/bin/env bash
# =====================================================================
# T7b — build in-genus published-TSS promoter POSITIVES -> $OUT (parquet).
#
# Stages the reference genome(s) from the T5 zip, runs extract_published_tss.py on any
# published TSS table(s) staged under T7/published_tss/, and concatenates the result.
# If NO table is staged it writes an EMPTY but schema-valid parquet, so the pipeline still
# runs (T7b just contributes 0 verified in-genus positives until the tables are added).
#
# The published TSS coordinate tables are paper SUPPLEMENTS (binary xlsx) that must be
# downloaded by hand — see PUBLISHED_TSS_README.md for the exact sources, file names, and
# the column-name env vars to set.
# =====================================================================
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
GENOMES_ZIP="${GENOMES_ZIP:-$REPO/data-acquisition-and-curation/T5/Clostridium_genomes_v2_166.zip}"
OUT="${OUT:-$REPO/infrastructure/T3/snakemake_scaffold/inputs/published_tss_promoters.parquet}"
T7="$REPO/data-acquisition-and-curation/T7"
TBL_DIR="$T7/published_tss"          # drop the downloaded TSS tables here (*.tsv)
REF_DIR="$T7/ref"
mkdir -p "$REF_DIR" "$(dirname "$OUT")"

# Stage the C. ljungdahlii reference (single replicon NC_014328.1) from the T5 zip.
CLJ_FNA="$REF_DIR/GCF_000143685.1_ASM14368v1_genomic.fna"
[ -f "$CLJ_FNA" ] || unzip -p "$GENOMES_ZIP" 'ncbi_dataset/data/GCF_000143685.1/*_genomic.fna' > "$CLJ_FNA"

PARTS=()

# Al-Bassam 2018 — C. ljungdahlii (genome staged above; single replicon -> no --contig-col needed).
# Set CLJ_POS_COL / CLJ_STRAND_COL / CLJ_SIGMA_COL from the paper's table (see README).
if [ -f "$TBL_DIR/albassam2018_clj_tss.tsv" ]; then
  python "$T7/extract_published_tss.py" --tss-table "$TBL_DIR/albassam2018_clj_tss.tsv" \
    --genome "$CLJ_FNA" --species "Clostridium ljungdahlii" --source AlBassam2018 \
    --pos-col "${CLJ_POS_COL:-TSS_position}" --strand-col "${CLJ_STRAND_COL:-strand}" \
    ${CLJ_SIGMA_COL:+--sigma-col "$CLJ_SIGMA_COL"} --out "$REF_DIR/albassam.parquet"
  PARTS+=("$REF_DIR/albassam.parquet")
fi

# Hocq 2022 — C. acetobutylicum ATCC 824 (2 replicons: NC_003030.1 + pSOL1 NC_001988.2).
# NOT in the T5 zip -> provide its FASTA via CAC_FNA, and --contig-col is REQUIRED.
if [ -f "$TBL_DIR/hocq2022_cac_tss.tsv" ] && [ -n "${CAC_FNA:-}" ]; then
  python "$T7/extract_published_tss.py" --tss-table "$TBL_DIR/hocq2022_cac_tss.tsv" \
    --genome "$CAC_FNA" --species "Clostridium acetobutylicum" --source Hocq2022 \
    --pos-col "${CAC_POS_COL:-TSS_position}" --strand-col "${CAC_STRAND_COL:-strand}" \
    --contig-col "${CAC_CONTIG_COL:-replicon}" ${CAC_SIGMA_COL:+--sigma-col "$CAC_SIGMA_COL"} \
    --out "$REF_DIR/hocq.parquet"
  PARTS+=("$REF_DIR/hocq.parquet")
fi

if [ ${#PARTS[@]} -eq 0 ]; then
  echo "[T7b] no published TSS tables in $TBL_DIR — writing an EMPTY (schema-valid) parquet."
  echo "      Populate with REAL in-genus positives: see $T7/PUBLISHED_TSS_README.md"
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
