#!/usr/bin/env python3
"""
prep_new_tss.py — turn the three downloaded published-TSS workbooks into clean 2-column TSVs
(pos, strand) that extract_published_tss.py / tss_to_truth.py consume. Handles each workbook's
offset header + filters to canonical promoter TSS.

  Ruminiclostridium cellulolyticum (Sheet1)  -> gTS (gene-associated)   [CROSS-SPECIES positive]
  Bacillus licheniformis            (S5)      -> Class == P (primary)    [CROSS-SPECIES positive]
  Clostridium autoethanogenum       (Table S1)-> Class == P, deduped     [VALIDATION genome only]

Run from this folder (pipeline/acquisition/additional-published-tss).

WHERE EACH WORKBOOK COMES FROM (cite these; the model cards in release/huggingface/ repeat them):
  R. cellulolyticum H10  Xu C, Huang R, Teng L, et al. Cellulosome stoichiometry in Clostridium
                         cellulolyticum is regulated by selective RNA processing and stabilization.
                         Nat Commun. 2015;6:6900. doi:10.1038/ncomms7900
                         dRNA-seq BioProject PRJNA247408 (Exo/Hyb libraries — the same Exo/Hyb split
                         named in Sheet1's own header).
  B. licheniformis DSM13 Wiegand S, Dietrich S, Hertel R, Bongaerts J, et al. RNA-Seq of Bacillus
                         licheniformis: active regulatory RNA features expressed within a productive
                         fermentation. BMC Genomics. 2013;14:667. doi:10.1186/1471-2164-14-667
                         Supplementary Table S5; dRNA-seq BioProject PRJNA188930 (SRR747881-747900).
  C. autoethanogenum     de Souza Pinto Lemgruber R, Valgepea K, Gonzalez Garcia RA, de Bakker C,
                         et al. A TetR-Family Protein (CAETHG_0459) Activates Transcription From a
                         New Promoter Motif Associated With Essential Genes for Autotrophic Growth
                         in Acetogens. Front Microbiol. 2019;10:2549. doi:10.3389/fmicb.2019.02549
                         Supplementary File 1, Table S1; BioProject PRJNA428323. VALIDATION truth
                         only, never training.

The deployed split tags these sources Wiegand2013 and Xu2015 (not the --source values an older
draft of docs/PIPELINE.md used); Hocq is Hocq2022, and T8 is T8_RegulonDB / T8_DBTBS.
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
