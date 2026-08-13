#!/usr/bin/env python3
"""
ensemble_rank_percentile.py — T29 ensemble aggregation by MEAN RANK-PERCENTILE.

Combines the per-window score tracks of the deployed models (genome_inference.py output) into ONE
ensemble track. For each model, every window's raw `promoter_score` is converted to its percentile
rank in [0,1] among ALL of that model's windows; the ensemble score is the MEAN of those percentiles
across models. Models agreeing that a window is high-ranked -> high ensemble percentile.

WHY rank-percentile (not mean probability): it is CALIBRATION-INVARIANT. It depends only on each
model's within-genome rank ORDER, so the notoriously optimistic CNN-v2 and the differently-scaled
ProkBERT / NT / CNN combine on equal footing WITHOUT applying T26 calibration first. This is exactly
why T29 can (and should) run on the RAW, pre-T26 scores — the point of "the primary set before T26".

The output carries a `promoter_score` column (= the mean rank-percentile) so the ensemble track is a
DROP-IN for genome_fp_benchmark.py — benchmark it restricted exactly like a single model:
    python genome_fp_benchmark.py --scores <ensemble>.csv --truth truth.tsv --regions regions.bed ...

INPUTS: two or more per-model tracks, each a CSV from genome_inference.py with columns
    contig, start, end, strand, promoter_score      (extra columns e.g. logit/at_fraction are ignored).
All tracks MUST be over the IDENTICAL window grid — the same genome scanned with the same --window,
--step and --strands. Tracks are aligned by (contig, start, strand); if the keys do not match across
files the run FAILS LOUD (a silent misalignment would scramble the ensemble, so we never guess).

  python ensemble_rank_percentile.py \
      --track prokbert clj_prokbert_xs.csv --track nt     clj_nt_xs.csv \
      --track cnn      clj_cnn_xs.csv      --track cnn_v2 clj_cnn_v2_xs.csv \
      --out clj_ensemble_xs.csv \
      [--regions clj_candidate_regions.bed --shortlist clj_shortlist_restricted.csv]
"""
import argparse
import sys

import numpy as np
import pandas as pd

KEYS = ["contig", "start", "strand"]          # a window is identified by these across tracks


def load_track(path):
    """Read a genome_inference.py track, keep only what we need, sort deterministically by KEYS so
    two independently-written tracks line up row-for-row."""
    df = pd.read_csv(path, usecols=lambda c: c in ("contig", "start", "end", "strand", "promoter_score"))
    for c in ("contig", "start", "strand", "promoter_score"):
        if c not in df.columns:
            raise SystemExit(f"{path}: missing column '{c}'; has {list(df.columns)}")
    if "end" not in df.columns:
        df["end"] = df["start"] + 81       # genome_inference always writes it; tolerate an older track
    df["contig"] = df["contig"].astype(str)
    df["strand"] = df["strand"].astype(str)
    return df.sort_values(KEYS, kind="mergesort").reset_index(drop=True)


def restrict_mask(wctg, wstart, wstr, window, reg):
    """Boolean mask: window [start, start+window) overlaps a candidate region on its (contig, strand).
    Mirrors evaluation-and-calibration/genome_fp_benchmark.py:in_region_mask (regions are the merged,
    non-overlapping output of make_candidate_regions.py, so the single region with the largest start
    below the window end is the only one that can overlap)."""
    reg = reg.copy()
    reg["contig"] = reg["contig"].astype(str)
    reg["strand"] = reg["strand"].astype(str)
    keep = np.zeros(len(wstart), dtype=bool)
    we = wstart + window
    for (c, s), g in reg.groupby(["contig", "strand"]):
        gi = np.where((wctg == c) & (wstr == s))[0]
        if len(gi) == 0:
            continue
        g = g.sort_values("start")
        rs, re = g["start"].to_numpy(int), g["end"].to_numpy(int)
        idx = np.searchsorted(rs, we[gi], side="right") - 1     # region with largest start < window end
        ok = idx >= 0
        ov = np.zeros(len(gi), dtype=bool)
        ov[ok] = re[idx[ok]] > wstart[gi][ok]                    # ...and it ends after the window starts
        keep[gi[ov]] = True
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", nargs=2, action="append", metavar=("NAME", "CSV"), required=True,
                    help="a model name and its genome_inference.py score CSV. Repeat once per model (>=2).")
    ap.add_argument("--out", required=True,
                    help="ensemble track CSV (genome order): contig,start,end,strand,<name>_pct...,promoter_score")
    ap.add_argument("--regions", default=None,
                    help="optional candidate-regions BED (make_candidate_regions.py). If given, also writes "
                         "--shortlist: the ensemble RESTRICTED to candidate windows, score-sorted — the "
                         "target-mine shortlist when there is no truth table to benchmark against.")
    ap.add_argument("--shortlist", default=None, help="output path for the restricted score-sorted shortlist (needs --regions)")
    ap.add_argument("--window", type=int, default=81, help="window length, for the --regions overlap test")
    a = ap.parse_args()

    names = [n for n, _ in a.track]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate --track names: {names}")
    if len(names) < 2:
        raise SystemExit("need >=2 tracks to ensemble.")
    if a.regions and not a.shortlist:
        raise SystemExit("--regions given but --shortlist path missing.")

    base, keyref, pct = None, None, {}
    for name, path in a.track:
        df = load_track(path)
        if base is None:
            base = df[["contig", "start", "end", "strand"]].copy()
            keyref = df[KEYS]
        elif len(df) != len(base) or not df[KEYS].equals(keyref):
            raise SystemExit(
                f"track '{name}' ({path}) is NOT on the same window grid as '{names[0]}' "
                f"({len(df):,} vs {len(base):,} rows, or a key mismatch). Every track must be scanned "
                f"with identical --window/--step/--strands over the same genome. Re-scan the odd one out.")
        # percentile rank of the raw score (ties averaged) — the calibration-invariant transform
        pct[name] = df["promoter_score"].rank(pct=True, method="average").to_numpy()
        print(f"  {name:8s}: {len(df):>10,} windows  raw score "
              f"[{df.promoter_score.min():.3f}, {df.promoter_score.max():.3f}]", file=sys.stderr)
        del df

    ens = np.mean(np.column_stack([pct[n] for n in names]), axis=1)
    for n in names:
        base[f"{n}_pct"] = np.round(pct[n], 6)
    base["promoter_score"] = np.round(ens, 6)          # mean rank-percentile; drop-in for genome_fp_benchmark
    base.to_csv(a.out, index=False)                    # genome order (sorted by KEYS) = a proper track for T30
    top = np.sort(ens)[::-1]
    print(f"Wrote {len(base):,} ensemble windows -> {a.out}  "
          f"(ensemble>=0.99: {int((ens >= 0.99).sum()):,}; max {top[0]:.4f})")

    if a.regions:
        reg = pd.read_csv(a.regions, sep="\t")
        keep = restrict_mask(base["contig"].to_numpy().astype(str), base["start"].to_numpy(int),
                             base["strand"].to_numpy().astype(str), a.window, reg)
        sl = base[keep].sort_values("promoter_score", ascending=False, kind="mergesort").reset_index(drop=True)
        sl.to_csv(a.shortlist, index=False)
        print(f"Restricted shortlist: {len(sl):,}/{len(base):,} windows in candidate regions "
              f"({keep.mean():.1%}), score-sorted -> {a.shortlist}")


if __name__ == "__main__":
    main()
