#!/usr/bin/env python3
"""
call_peaks.py — T30 locus-level peak calling. Collapse the per-window ensemble track (T29) into one
promoter CALL per locus.

A real promoter is covered by many adjacent high-scoring windows: the sliding window steps 1 bp and the
promoter signal spans ~35 bp, so the raw shortlist carries tens of near-identical windows per promoter
(e.g. the six windows at contig-1:15230-15236 that all scored >0.998). This groups windows that pass a
score threshold and sit within a merge gap on the same (contig, strand) into a single peak, takes the
top-scoring window as the summit, and writes ONE row per call, score-sorted. That call list is the input
to T31 (FIMO/MEME -10/-35 motif annotation) and T32 (final candidate table).

Input: a per-window track with columns contig, start, strand, promoter_score (+ end optional) — the T29
ensemble track or its restricted shortlist, or any single-model genome_inference track. If you pass the
full (unrestricted) ensemble track, add --regions to peak-call only inside candidate promoter regions.

Output: contig, peak_start, peak_end, summit, strand, peak_score, mean_score, n_windows  (0-based;
`summit` = start of the peak's highest-scoring window; `peak_score` = that window's score).

  # from the restricted shortlist (already region-limited):
  python call_peaks.py --track g409755_shortlist_restricted.csv --min-score 0.9 --out g409755_peaks.csv
  # or from the full ensemble track, restricting on the fly:
  python call_peaks.py --track g409755_ensemble_xs.csv --regions g409755_candidate_regions.bed \
      --min-score 0.9 --out g409755_peaks.csv
"""
import argparse
import numpy as np
import pandas as pd


def region_mask(wctg, wstart, wstr, window, reg):
    """Windows overlapping a candidate region on their (contig, strand). Mirrors
    genome_fp_benchmark.in_region_mask / the T29 restriction."""
    reg = reg.copy()
    reg["contig"] = reg["contig"].astype(str); reg["strand"] = reg["strand"].astype(str)
    keep = np.zeros(len(wstart), dtype=bool); we = wstart + window
    for (c, s), g in reg.groupby(["contig", "strand"]):
        gi = np.where((wctg == c) & (wstr == s))[0]
        if len(gi) == 0:
            continue
        g = g.sort_values("start"); rs, re = g["start"].to_numpy(int), g["end"].to_numpy(int)
        idx = np.searchsorted(rs, we[gi], "right") - 1; ok = idx >= 0
        ov = np.zeros(len(gi), dtype=bool); ov[ok] = re[idx[ok]] > wstart[gi][ok]
        keep[gi[ov]] = True
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, help="per-window track (contig,start,strand,promoter_score[,end])")
    ap.add_argument("--out", required=True, help="one row per promoter call, score-sorted")
    ap.add_argument("--min-score", type=float, default=0.9, help="keep only windows scoring >= this before merging")
    ap.add_argument("--merge", type=int, default=81,
                    help="bp; windows whose gap (next.start - prev.end) is <= this, same contig+strand, join one peak")
    ap.add_argument("--score-col", default="promoter_score")
    ap.add_argument("--window", type=int, default=81, help="window length (fills a missing 'end' col and the --regions test)")
    ap.add_argument("--regions", default=None, help="optional BED (contig,start,end,strand): peak-call only inside these")
    a = ap.parse_args()

    df = pd.read_csv(a.track, usecols=lambda c: c in ("contig", "start", "end", "strand", a.score_col))
    for c in ("contig", "start", "strand", a.score_col):
        if c not in df.columns:
            raise SystemExit(f"{a.track}: missing column '{c}'; has {list(df.columns)}")
    df["contig"] = df["contig"].astype(str); df["strand"] = df["strand"].astype(str)
    if "end" not in df.columns:
        df["end"] = df["start"] + a.window
    n0 = len(df)

    if a.regions:
        reg = pd.read_csv(a.regions, sep="\t")
        m = region_mask(df["contig"].to_numpy().astype(str), df["start"].to_numpy(int),
                        df["strand"].to_numpy().astype(str), a.window, reg)
        df = df[m]

    df = df[df[a.score_col] >= a.min_score]
    if df.empty:
        raise SystemExit(f"no windows scored >= {a.min_score} (of {n0:,}). Lower --min-score.")
    df = df.sort_values(["contig", "strand", "start"], kind="mergesort").reset_index(drop=True)

    start = df["start"].to_numpy(int); end = df["end"].to_numpy(int)
    contig = df["contig"].to_numpy(); strand = df["strand"].to_numpy()
    score = df[a.score_col].to_numpy(float)
    # a new peak starts where contig or strand changes, or the gap to the previous window exceeds --merge
    breaks = (contig[1:] != contig[:-1]) | (strand[1:] != strand[:-1]) | ((start[1:] - end[:-1]) > a.merge)
    pid = np.concatenate([[0], np.cumsum(breaks)])

    g = pd.DataFrame({"pid": pid, "contig": contig, "strand": strand, "start": start, "end": end, "score": score})
    agg = g.groupby("pid", sort=False).agg(
        contig=("contig", "first"), strand=("strand", "first"),
        peak_start=("start", "min"), peak_end=("end", "max"),
        peak_score=("score", "max"), mean_score=("score", "mean"), n_windows=("score", "size"))
    summit_idx = g.groupby("pid", sort=False)["score"].idxmax()
    agg["summit"] = g.loc[summit_idx.to_numpy(), "start"].to_numpy()

    out = agg[["contig", "peak_start", "peak_end", "summit", "strand",
               "peak_score", "mean_score", "n_windows"]].copy()
    out["mean_score"] = out["mean_score"].round(4); out["peak_score"] = out["peak_score"].round(4)
    out = out.sort_values("peak_score", ascending=False, kind="mergesort").reset_index(drop=True)
    out.to_csv(a.out, index=False)
    print(f"{n0:,} windows -> {len(df):,} >= {a.min_score} -> {len(out):,} promoter calls -> {a.out}")
    print(f"  peaks with >=5 supporting windows: {(out.n_windows >= 5).sum():,}  (single-window peaks: {(out.n_windows == 1).sum():,})")


if __name__ == "__main__":
    main()
