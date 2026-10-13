#!/usr/bin/env python3
"""
filter_and_rank_wetlab.py — novelty-filter + de-duplicate the g131795 mined promoters, then
re-rank the survivors by confidence for the wet lab.

Two filters, both by EXACT shared 20-mer (at k=20 a chance collision is ~1e-2 over these set
sizes, so any hit is a real shared stretch, not coincidence):

  1. NOVELTY  — drop a candidate whose 81 bp window shares a 20-mer (either strand) with a KNOWN
     promoter (RegulonDB E. coli + DBTBS B. subtilis, 81 bp; + T9 literature). i.e. it is
     "a sequence of an already-existing promoter", not a novel find.
  2. REDUNDANCY — collapse candidates that share a 20-mer with EACH OTHER (repeat-driven, e.g.
     mobile-element / multi-copy promoters); keep the highest-peak_score representative per group.

Survivors are ordered by the existing confidence rank (tier: canonical > has-10 > confident, then
peak_score, then n_windows, then motif p) and re-numbered.

NB the in-genus Hocq TSS set (C. acetobutylicum / C. beijerinckii) is NOT included — its parquet is
an unfetched Git-LFS pointer locally. Those relatives are the most likely to share sequence, so add
them (git lfs pull, or re-extract) and re-run for a stricter novelty screen.

  python pipeline/mining/filter_and_rank_wetlab.py
"""
import sys
import pandas as pd

K = 20
CANDIDATES = "results/candidates/g131795_ranked_seq.tsv"
KNOWN = [
    ("pipeline/acquisition/curated-promoters/curated_positive_promoters.parquet", "sequence"),  # RegulonDB+DBTBS
    ("pipeline/acquisition/synthetic-promoters/literature_promoters.tsv", "sequence"),            # T9 literature
]
_RC = str.maketrans("ACGT", "TGCA")


def rc(s):
    return s.translate(_RC)[::-1]


def canon_kmers(seq, k=K):
    """Canonical k-mers (min of each k-mer and its reverse complement) — strand-agnostic."""
    s = "".join(c for c in seq.upper() if c in "ACGT")
    out = set()
    for i in range(len(s) - k + 1):
        km = s[i:i + k]
        out.add(min(km, rc(km)))
    return out


def load_known():
    kmers = set()
    n = 0
    for path, col in KNOWN:
        d = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, sep="\t")
        for s in d[col].dropna().astype(str):
            kmers |= canon_kmers(s)
            n += 1
    return kmers, n


def main():
    cand = pd.read_csv(CANDIDATES, sep="\t")
    cand["summit_seq"] = cand["summit_seq"].astype(str)
    known_kmers, n_known = load_known()
    print(f"known promoters: {n_known}  ({len(known_kmers):,} canonical {K}-mers)")
    print(f"candidates: {len(cand)}")

    ckmers = [canon_kmers(s) for s in cand["summit_seq"]]

    # ---- 1) novelty: candidate shares a k-mer with a known promoter ----
    cand["matches_known"] = [bool(km & known_kmers) for km in ckmers]

    # ---- 2) internal redundancy: union candidates that share a k-mer ----
    parent = list(range(len(cand)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[max(ra, rb)] = min(ra, rb)
    seen = {}
    for i, km in enumerate(ckmers):
        for k in km:
            if k in seen: union(seen[k], i)
            else: seen[k] = i
    cand["_grp"] = [find(i) for i in range(len(cand))]
    # within each group keep the best peak_score (ties -> better priority_rank) as representative
    cand["_ord"] = cand.sort_values(["peak_score", "priority_rank"], ascending=[False, True]).groupby("_grp").cumcount().reindex(cand.index).fillna(0)
    order = cand.sort_values(["_grp", "peak_score", "priority_rank"], ascending=[True, False, True])
    rep_idx = set(order.groupby("_grp").head(1).index)
    cand["redundant_dup"] = [i not in rep_idx for i in cand.index]
    grp_sizes = cand["_grp"].map(cand["_grp"].value_counts())
    cand["dup_group_size"] = grp_sizes

    # ---- survivors ----
    keep = cand[~cand["matches_known"] & ~cand["redundant_dup"]].copy()
    keep = keep.sort_values("priority_rank").reset_index(drop=True)   # existing confidence order
    keep.insert(0, "wetlab_rank", range(1, len(keep) + 1))

    removed_known = int(cand["matches_known"].sum())
    removed_dup = int((~cand["matches_known"] & cand["redundant_dup"]).sum())
    print(f"\nremoved as already-existing (matches known promoter DB): {removed_known}")
    print(f"removed as internal duplicate (repeat-driven):           {removed_dup}")
    print(f"SURVIVORS (novel, de-duplicated): {len(keep)}")
    print(f"  of which canonical (-10 & -35): {int((keep['tier']==1).sum())}   "
          f"placed -10: {int((keep['tier']<=2).sum())}")

    cols = ["wetlab_rank", "candidate_id", "chromosome", "strand", "tss_position", "peak_score",
            "n_windows", "tier_label", "has_minus10", "has_minus35", "minus35_offset", "spacer_bp",
            "minus10_offset", "minus35_seq", "minus10_seq", "fdr_q_value", "summit_seq",
            "dup_group_size"]
    cols = [c for c in cols if c in keep.columns]
    keep[cols].to_csv("results/candidates/g131795_final_wetlab.tsv", sep="\t", index=False)
    # audit trail of what was dropped
    dropped = cand[cand["matches_known"] | cand["redundant_dup"]].copy()
    dropped["drop_reason"] = ["matches_known_db" if mk else "internal_duplicate"
                              for mk in dropped["matches_known"]]
    dropped.sort_values("priority_rank")[["candidate_id", "chromosome", "strand", "tss_position",
        "peak_score", "drop_reason", "dup_group_size"]].to_csv("results/candidates/g131795_dropped.tsv", sep="\t", index=False)
    print("\nwrote results/candidates/g131795_final_wetlab.tsv  +  results/candidates/g131795_dropped.tsv (audit)")
    print("\nTop 15 (final wet-lab order):")
    show = ["wetlab_rank", "candidate_id", "chromosome", "strand", "tss_position", "peak_score", "tier_label"]
    print(keep.head(15)[show].to_string(index=False))


if __name__ == "__main__":
    main()
