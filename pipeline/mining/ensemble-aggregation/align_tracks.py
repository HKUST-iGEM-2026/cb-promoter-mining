#!/usr/bin/env python3
"""
align_tracks.py — reconcile per-model score tracks onto a COMMON window grid for the T29 ensemble.

WHY THIS EXISTS: T29 (ensemble_rank_percentile.py) requires every track to be on the IDENTICAL window
grid and fails loud otherwise. That holds for genome_inference.py tracks scanned with the same
--window/--step/--strands. But the Evo2 caller (evo2_scan_restricted.py) scores ONLY candidate-region
windows, so its track is a SUBSET of a full-genome genome_inference track. Even with Evo2's --align-grid
(which puts Evo2 on the global step grid), the ProkBERT / cnn_v2 full-genome tracks still have MORE rows.
This tool takes the INTERSECTION of the tracks on (contig, start, strand) and writes one aligned CSV per
track — all with the identical row set and order — ready to hand to T29 unchanged.

In the default three-model mine (ProkBERT/CNN-v2 full-genome, Evo2 restricted with --align-grid),
the common grid is Evo2's candidate-window set. Percentiles are computed within that shared pool,
not each model's original scan pool. The historical four-model scan_ensemble.sbatch instead
uses full-genome tracks and ranks genome-wide; its pool and model membership differ.

  python align_tracks.py --track prokbert clj_prokbert.csv --track cnn_v2 clj_cnn_v2.csv \
      --track evo2 clj_evo2.csv --out-dir aligned/
  # -> aligned/{prokbert,cnn_v2,evo2}_aligned.csv  (identical grids)
  # then: ensemble_rank_percentile.py --track prokbert aligned/prokbert_aligned.csv --track ...
"""
import argparse
import os
import sys

import pandas as pd

KEYS = ["contig", "start", "strand"]


def load(path):
    df = pd.read_csv(path, usecols=lambda c: c in ("contig", "start", "end", "strand", "promoter_score"))
    for c in ("contig", "start", "strand", "promoter_score"):
        if c not in df.columns:
            raise SystemExit(f"{path}: missing column '{c}'; has {list(df.columns)}")
    if "end" not in df.columns:
        df["end"] = df["start"] + 81
    df["contig"] = df["contig"].astype(str)
    df["strand"] = df["strand"].astype(str)
    df["start"] = df["start"].astype(int)
    if df.duplicated(KEYS).any():
        raise SystemExit(f"{path}: {int(df.duplicated(KEYS).sum())} duplicate (contig,start,strand) keys "
                         f"— a track must be unique per window.")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", nargs=2, action="append", metavar=("NAME", "CSV"), required=True,
                    help="model name + its score CSV (genome_inference.py or evo2_scan_restricted.py). "
                         "Repeat once per model (>=2).")
    ap.add_argument("--out-dir", required=True, help="dir for the <NAME>_aligned.csv outputs")
    ap.add_argument("--min-overlap", type=float, default=0.5,
                    help="abort if the intersection is < this fraction of the SMALLEST track — guards a "
                         "silent grid mismatch (wrong --step, or Evo2 scanned without --align-grid). Default 0.5.")
    a = ap.parse_args()

    names = [n for n, _ in a.track]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate --track names: {names}")
    if len(names) < 2:
        raise SystemExit("need >=2 tracks to align.")
    os.makedirs(a.out_dir, exist_ok=True)

    dfs, key_sets = {}, []
    for name, path in a.track:
        df = load(path)
        dfs[name] = df
        key_sets.append(set(map(tuple, df[KEYS].to_numpy())))
        print(f"  {name:10s}: {len(df):>12,} windows  ({path})", file=sys.stderr)

    inter = set.intersection(*key_sets)
    smallest = min(len(s) for s in key_sets)
    if not inter:
        raise SystemExit("EMPTY intersection — tracks share no (contig,start,strand). Scan Evo2 with "
                         "--align-grid and use the SAME --step for every model.")
    frac = len(inter) / smallest
    print(f"  intersection: {len(inter):,} windows  ({frac:.1%} of the smallest track)", file=sys.stderr)
    if frac < a.min_overlap:
        raise SystemExit(f"intersection {frac:.1%} < --min-overlap {a.min_overlap:.0%} — likely a grid "
                         f"mismatch (different --step, or Evo2 not scanned with --align-grid). Aborting "
                         f"rather than silently ensembling a shrunken window set.")

    inter_df = pd.DataFrame(sorted(inter), columns=KEYS)
    inter_df["start"] = inter_df["start"].astype(int)
    inter_df["contig"] = inter_df["contig"].astype(str)
    inter_df["strand"] = inter_df["strand"].astype(str)

    outs = []
    for name in names:
        m = (dfs[name].merge(inter_df, on=KEYS, how="inner")
             .sort_values(KEYS, kind="mergesort").reset_index(drop=True))
        if len(m) != len(inter_df):
            raise SystemExit(f"{name}: merge produced {len(m):,} rows, expected {len(inter_df):,} — aborting.")
        out = os.path.join(a.out_dir, f"{name}_aligned.csv")
        m[["contig", "start", "end", "strand", "promoter_score"]].to_csv(out, index=False)
        outs.append((name, out))

    # cross-check: every aligned track has byte-identical KEYS in the same order (what T29 demands)
    ref = pd.read_csv(outs[0][1], usecols=KEYS)
    ref["contig"] = ref["contig"].astype(str); ref["strand"] = ref["strand"].astype(str)
    ref = ref.reset_index(drop=True)
    for name, out in outs[1:]:
        d = pd.read_csv(out, usecols=KEYS)
        d["contig"] = d["contig"].astype(str); d["strand"] = d["strand"].astype(str)
        if not d.reset_index(drop=True).equals(ref):
            raise SystemExit(f"{name}: aligned keys differ from {outs[0][0]} — reconcile failed.")

    print(f"Wrote {len(outs)} aligned tracks ({len(inter_df):,} windows each) -> {a.out_dir}")
    print("Feed to T29:")
    print("  python ensemble_rank_percentile.py \\")
    print(" ".join(f"--track {n} {o}" for n, o in outs), "\\")
    print("      --out <ensemble>.csv --regions <regions>.bed --shortlist <shortlist>.csv")


if __name__ == "__main__":
    main()
