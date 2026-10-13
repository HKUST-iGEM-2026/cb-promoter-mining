#!/usr/bin/env python3
"""
tss_to_truth.py — build a ground-truth TSS table for genome_fp_benchmark.py from a published
in-genus TSS table (Hocq 2022 for C. acetobutylicum / C. beijerinckii; Al-Bassam 2018 for
C. ljungdahlii; or any paper's table).

Why this exists: T7b's positives parquet keeps only the extracted 81 bp SEQUENCE and drops the
genomic coordinate, so the benchmark's ground truth is rebuilt here from the same source table T7b
ingests. This mirrors extract_published_tss.py's filtering/remapping so the truth set matches the
verified positives T15 was trained/evaluated on.

Emits a TSV with columns: contig, pos, strand   (pos = 1-based TSS / +1 site).

IMPORTANT (matches build_published_tss.sh for the Hocq workbook):
  * --filter-col Class --filter-vals intergenic_sense   keep only canonical promoter TSS
    (1,436 of 3,441 rows for C. acetobutylicum) — the same rows T7b calls positives.
  * --contig-map "Chromosome=NC_003030.1,pSOL=NC_001988.2"   the Hocq Locus column holds
    "Chromosome"/"pSOL", not accessions; remap so contigs match the genome FASTA / score track.
  * --contig NC_003030.1   then keep just the chromosome (>99% of TSS); run the plasmid separately.

  python tss_to_truth.py --tss-table hocq2022_cac_tss.xlsx --sheet ATCC_824_25RPM \
      --pos-col Genome_position --strand-col Strand --contig-col Locus \
      --contig-map "Chromosome=NC_003030.1,pSOL=NC_001988.2" \
      --filter-col Class --filter-vals intergenic_sense --contig NC_003030.1 --out cac_truth.tsv
"""
import argparse
import pandas as pd

PLUS = {"+", "1", "plus", "F", "fwd", "forward", "+1"}
MINUS = {"-", "-1", "minus", "R", "rev", "reverse"}


def norm_strand(s):
    s = str(s).strip()
    return "+" if s in PLUS else "-" if s in MINUS else s


def parse_map(spec):
    """'A=x,B=y' -> {'A':'x','B':'y'}."""
    out = {}
    for pair in (spec or "").split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def coerce_sheet(s):
    try:
        return int(s)          # a numeric sheet index from the CLI arrives as str
    except (TypeError, ValueError):
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tss-table", required=True, help="published TSS table (CSV/TSV/XLSX)")
    ap.add_argument("--pos-col", required=True, help="column with the genomic TSS position (1-based)")
    ap.add_argument("--strand-col", required=True)
    ap.add_argument("--contig-col", default=None, help="column with the replicon/contig id")
    ap.add_argument("--contig-map", default=None, help="'Chromosome=NC_003030.1,pSOL=NC_001988.2' remap of contig ids")
    ap.add_argument("--filter-col", default=None, help="keep only rows whose value is in --filter-vals (e.g. Class)")
    ap.add_argument("--filter-vals", default=None, help="comma list of values to keep (e.g. intergenic_sense)")
    ap.add_argument("--contig", default=None, help="keep only rows on this (remapped) contig id")
    ap.add_argument("--sep", default="\t")
    ap.add_argument("--sheet", default=0, help="xlsx sheet name or index (e.g. ATCC_824_25RPM); needs openpyxl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if str(a.tss_table).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(a.tss_table, sheet_name=coerce_sheet(a.sheet))
    else:
        df = pd.read_csv(a.tss_table, sep=a.sep)
    for c in (a.pos_col, a.strand_col):
        if c not in df.columns:
            raise SystemExit(f"missing column '{c}'; table has {list(df.columns)}")

    if a.filter_col:
        if a.filter_col not in df.columns:
            raise SystemExit(f"--filter-col '{a.filter_col}' not in {list(df.columns)}")
        keep = set((a.filter_vals or "").split(","))
        df = df[df[a.filter_col].astype(str).isin(keep)]

    contig = (df[a.contig_col].astype(str) if (a.contig_col and a.contig_col in df.columns)
              else pd.Series(["."] * len(df), index=df.index))
    cmap = parse_map(a.contig_map)
    if cmap:
        contig = contig.map(lambda v: cmap.get(v, v))

    out = pd.DataFrame({
        "contig": contig.to_numpy(),
        "pos": pd.to_numeric(df[a.pos_col], errors="coerce").astype("Int64"),
        "strand": df[a.strand_col].map(norm_strand).to_numpy(),
    }).dropna(subset=["pos"])
    if a.contig:
        out = out[out["contig"] == a.contig]
    out = out[["contig", "pos", "strand"]].drop_duplicates().sort_values(["contig", "pos"])
    if len(out) == 0:
        raise SystemExit("0 TSS after filtering — check --filter-vals / --contig / --contig-map against the table.")
    out.to_csv(a.out, sep="\t", index=False)
    print(f"wrote {len(out)} TSS -> {a.out}")
    print("  strands:", out["strand"].value_counts().to_dict(), "| contigs:", out["contig"].value_counts().to_dict())


if __name__ == "__main__":
    main()
