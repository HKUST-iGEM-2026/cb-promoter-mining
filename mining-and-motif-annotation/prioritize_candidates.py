#!/usr/bin/env python3
"""Rank a T32 candidate table into a prioritized wet-lab shortlist.

Unlike split_candidate_tiers.py (which only slices by motif architecture), this ranks the
candidates for hand-off: tier by promoter architecture, then within tier by ensemble
peak_score, then supporting-window count, then motif p-value. FDR-aware: if the table
carries the T27 join columns (fdr_q_value ...) they are surfaced, and the run reports how
many candidates are FDR-significant. The FDR VALIDATES the shortlist (a candidate with no
overlapping FDR peak is flagged) but does not re-order it — every confident call already
sits above the convergent-intergenic null, so score/architecture do the ranking.

Tiers (nested, most confident first):
  1  canonical    has_minus10 AND has_minus35   (the -10/-35 wet-lab pick)
  2  has_minus10  a canonically-placed -10       (superset of tier 1)
  3  confident    >=0.9, >=5 windows             (all rows)

Usage:
  python mining-and-motif-annotation/prioritize_candidates.py g131795_candidates_fdr.tsv
  python mining-and-motif-annotation/prioritize_candidates.py g131795_candidates_fdr.tsv --tag g131795 --outdir .
"""
import argparse
import os
import sys

import pandas as pd

# columns pulled to the front of every output so the wet-lab view leads with what matters
LEAD_COLS = [
    "priority_rank", "tier", "tier_label", "candidate_id", "chromosome", "strand",
    "tss_position", "peak_score", "mean_score", "n_windows",
    "has_minus10", "minus10_offset", "has_minus35", "minus35_offset", "spacer_bp",
    "best_motif_pvalue", "fdr_q_value", "fdr_significant_q05",
]


def truthy(s):
    """Booleans may round-trip through TSV as the strings 'True'/'False'."""
    return s.astype(str).str.lower().eq("true")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidates", help="T32 table (…_candidates.tsv or …_candidates_fdr.tsv)")
    ap.add_argument("--tag", default=None, help="output prefix (default: inferred from filename)")
    ap.add_argument("--outdir", default=None, help="output dir (default: alongside the input)")
    a = ap.parse_args()

    df = pd.read_csv(a.candidates, sep="\t")
    for c in ("candidate_id", "peak_score", "n_windows", "has_minus10", "has_minus35"):
        if c not in df.columns:
            sys.exit(f"{a.candidates}: missing required column '{c}'; has {list(df.columns)}")

    tag = a.tag or os.path.basename(a.candidates).split("_candidates")[0]
    outdir = a.outdir or (os.path.dirname(os.path.abspath(a.candidates)) or ".")
    os.makedirs(outdir, exist_ok=True)

    m10, m35 = truthy(df["has_minus10"]), truthy(df["has_minus35"])
    df["tier"] = 3
    df.loc[m10, "tier"] = 2
    df.loc[m10 & m35, "tier"] = 1
    df["tier_label"] = df["tier"].map({1: "canonical", 2: "has_minus10", 3: "confident"})

    # rank within tier: strongest ensemble peak, then most supporting windows, then best motif
    sort_cols, asc = ["tier", "peak_score", "n_windows"], [True, False, False]
    if "best_motif_pvalue" in df.columns:
        sort_cols.append("best_motif_pvalue"); asc.append(True)  # lower p = better; NaN sorts last
    df = df.sort_values(sort_cols, ascending=asc, na_position="last").reset_index(drop=True)
    df.insert(0, "priority_rank", range(1, len(df) + 1))

    lead = [c for c in LEAD_COLS if c in df.columns]
    df = df[lead + [c for c in df.columns if c not in lead]]

    outputs = {
        f"{tag}_ranked.tsv": df,                       # master: every confident call, ranked
        f"{tag}_tier2_has_minus10.tsv": df[df["tier"] <= 2],
        f"{tag}_tier1_canonical.tsv": df[df["tier"] == 1],  # the wet-lab pick
    }
    for name, sub in outputs.items():
        sub.to_csv(os.path.join(outdir, name), sep="\t", index=False)

    n = len(df)
    n1 = int((df["tier"] == 1).sum())
    n2 = int((df["tier"] <= 2).sum())
    print(f"{n} confident candidates  ->  {n2} with a placed -10  ->  {n1} canonical (-10 AND -35)")
    for name in outputs:
        print(f"  {os.path.join(outdir, name)}")

    if "fdr_q_value" in df.columns:
        q = pd.to_numeric(df["fdr_q_value"], errors="coerce")
        print(f"\nFDR: {int((q <= 0.05).sum())}/{n} candidates overlap an FDR-significant peak (q<=0.05); "
              f"{int(q.isna().sum())} unmatched (no FDR call).")

    show = [c for c in ("priority_rank", "candidate_id", "chromosome", "strand", "tss_position",
                        "peak_score", "n_windows", "spacer_bp", "best_motif_pvalue", "fdr_q_value")
            if c in df.columns]
    print(f"\nTop 15 canonical (wet-lab pick):\n{df[df['tier'] == 1].head(15)[show].to_string(index=False)}")


if __name__ == "__main__":
    main()
