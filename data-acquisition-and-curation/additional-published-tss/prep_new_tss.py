#!/usr/bin/env python3
"""
prep_new_tss.py — turn the three downloaded published-TSS workbooks into clean 2-column TSVs
(pos, strand) that extract_published_tss.py / tss_to_truth.py consume. Handles each workbook's
offset header + filters to canonical promoter TSS.

  Ruminiclostridium cellulolyticum (Sheet1)  -> gTS (gene-associated)   [CROSS-SPECIES positive]
  Bacillus licheniformis            (S5)      -> Class == P (primary)    [CROSS-SPECIES positive]
  Clostridium autoethanogenum       (Table S1)-> Class == P, deduped     [VALIDATION genome only]

Run from this folder (data-acquisition-and-curation/additional-published-tss).
"""
import pandas as pd

def clean(df, pos, strand, out, name):
    d = df[[pos, strand]].dropna()
    d.columns = ["pos", "strand"]
    d["pos"] = d["pos"].astype(float).astype(int)
    d["strand"] = d["strand"].astype(str).str.strip()
    d = d[d["strand"].isin(["+", "-"])].drop_duplicates().sort_values("pos")
    d.to_csv(out, sep="\t", index=False)
    print(f"{name}: {len(d)} TSS -> {out}")

# 1. Ruminiclostridium (out-of-genus cross-species): gene-associated TSS
r = pd.read_excel("Ruminiclostridium cellulolyticum_TSS .xlsx", sheet_name="Sheet1", header=0)
clean(r[r["Classification"] == "gTS"], "Location", "Strand",
      "ruminiclostridium_tss.tsv", "Ruminiclostridium gTS")

# 2. B. licheniformis (out-of-genus cross-species): primary TSS (header on row index 1)
b = pd.read_excel("B. licheniformis_TSS.xls", sheet_name="S5", header=1)
clean(b[b["Class"] == "P"], "Start", "Strand",
      "blicheniformis_tss.tsv", "B. licheniformis primary")

# 3. C. autoethanogenum (VALIDATION only — do NOT train): primary TSS, dedup across gas+fructose
a = pd.read_excel("C. autoethanogenum_TSS.xlsx", sheet_name="Table S1", header=3)
clean(a[a["Class"] == "P"], "Position", "Strand",
      "autoethanogenum_tss.tsv", "C. autoethanogenum primary")
