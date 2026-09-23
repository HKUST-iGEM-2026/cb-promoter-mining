#!/usr/bin/env python3
"""Split the T32 candidate table (final-candidates/g409755_candidates.tsv) into three nested TSV lists:
   all confident (912) -> has a placed -10 (306) -> canonical -10 AND -35 (100).

Usage:
   python mining-and-motif-annotation/split_candidate_tiers.py                      # reads ./final-candidates/g409755_candidates.tsv
   python mining-and-motif-annotation/split_candidate_tiers.py /path/to/candidates.tsv
"""
import sys
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "final-candidates/g409755_candidates.tsv"
df = pd.read_csv(SRC, sep="\t")

# booleans may read back as the strings "True"/"False" — compare case-insensitively
def truthy(col):
    return df[col].astype(str).str.lower().eq("true")

tiers = {
    "final-candidates/g409755_all_confident_912.tsv": df,                                         # all >=0.9, >=5-window calls
    "final-candidates/g409755_has_minus10_306.tsv":   df[truthy("has_promoter_architecture")],    # canonically-placed -10
    "final-candidates/g409755_canonical_100.tsv":     df[truthy("has_minus10") & truthy("has_minus35")],  # -10 AND -35
}

for path, sub in tiers.items():
    sub.to_csv(path, sep="\t", index=False)
    print(f"{len(sub):4d} rows -> {path}")
