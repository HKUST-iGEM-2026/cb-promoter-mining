#!/usr/bin/env python3
"""
PT6 - Validate the E. coli sliding-window promoter pipeline against RegulonDB
       and emit the locked candidate-table schema.

Inputs
------
1. PT5 output CSV (per-window scores): columns
   chromosome,start,end,strand,sequence,promoter_score
   - `start`/`end` are 0-based, end-exclusive genome coordinates (Python slice
     convention), as written by sliding_window_inference.py.

2. RegulonDB TSS / promoter table (TSV). Column names are configurable via
   the --reg-* flags. Must give, per known site: a genome position (the +1 / TSS)
   and a strand (+/- or forward/reverse).

Coordinate logic (must match the model's training window)
---------------------------------------------------------
The model was trained on 81 bp RegulonDB windows with the TSS (+1) at position 61
(1-based) -> offset 60 from the window's 5' end.

  forward strand : predicted_TSS_0based = start + 60
  reverse strand : predicted_TSS_0based = end   - 61
  (1-based for RegulonDB comparison = 0based + 1)

Reverse derivation: PT5 scans the reverse-complement and stores
start = len-end_rc, end = len-start_rc. The window's 5' base sits at the highest
forward coordinate (end-1); the TSS is 60 nt into the 3' direction -> end-1-60.

Usage
-----
python validate_promoters.py \
    --pred mg1655_promoter_predictions.csv \
    --regulondb regulondb_tss.tsv \
    --genome MG1655.fna \
    --threshold 0.9 --tolerance 20 \
    --outdir pt6_out
"""

import argparse
import json
import os
import numpy as np
import pandas as pd

TSS_OFFSET = 60          # 0-based offset of the +1 inside an 81 bp window
WINDOW_SIZE = 81


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def genome_length_from_fasta(path):
    n = 0
    with open(path) as fh:
        for line in fh:
            if not line.startswith(">"):
                n += len(line.strip())
    return n


def norm_strand(s):
    s = str(s).strip().lower()
    if s in {"+", "1", "f", "forward", "plus", "fwd"}:
        return "+"
    if s in {"-", "-1", "r", "reverse", "minus", "rev"}:
        return "-"
    return s  # leave as-is; will simply not match


def load_predictions(path):
    df = pd.read_csv(path)
    df["strand"] = df["strand"].map(norm_strand)
    # Predicted TSS, 1-based, in forward-genome coordinates.
    fwd = df["strand"] == "+"
    df["pred_tss"] = np.where(
        fwd,
        df["start"] + TSS_OFFSET + 1,
        df["end"] - TSS_OFFSET - 1 + 1,   # = end - 60
    )
    return df


def load_regulondb(path, pos_col, strand_col, id_col, sep):
    reg = pd.read_csv(path, sep=sep, comment="#", dtype=str)
    if pos_col not in reg.columns:
        raise SystemExit(
            f"--reg-pos-col '{pos_col}' not found. Columns are: {list(reg.columns)}"
        )
    out = pd.DataFrame()
    out["known_tss"] = pd.to_numeric(reg[pos_col], errors="coerce")
    out["strand"] = reg[strand_col].map(norm_strand) if strand_col in reg.columns else "+"
    out["known_id"] = reg[id_col] if id_col in reg.columns else [
        f"tss_{i}" for i in range(len(reg))
    ]
    out = out.dropna(subset=["known_tss"]).reset_index(drop=True)
    out["known_tss"] = out["known_tss"].astype(int)
    return out


# --------------------------------------------------------------------------- #
# Consolidation: merge overlapping high-scoring windows into one candidate
# --------------------------------------------------------------------------- #
def consolidate(df, threshold, merge_gap):
    """Cluster above-threshold windows (per strand) into candidate promoters.

    A cluster breaks when consecutive predicted-TSS positions are more than
    `merge_gap` bp apart. Consensus TSS = score-weighted mean of member TSS
    (the 'average of positions' approach). Summit TSS (argmax score) is also
    reported for reference.
    """
    hits = df[df["promoter_score"] >= threshold].copy()
    candidates = []
    for strand, grp in hits.groupby("strand"):
        grp = grp.sort_values("pred_tss")
        cluster = []
        prev = None
        for _, row in grp.iterrows():
            if prev is not None and row["pred_tss"] - prev > merge_gap:
                candidates.append(_finalize(cluster, strand))
                cluster = []
            cluster.append(row)
            prev = row["pred_tss"]
        if cluster:
            candidates.append(_finalize(cluster, strand))
    cand = pd.DataFrame(candidates)
    if not cand.empty:
        cand = cand.sort_values("peak_score", ascending=False).reset_index(drop=True)
        cand.insert(0, "candidate_id", [f"cand_{i+1:05d}" for i in range(len(cand))])
    return cand


def _finalize(cluster, strand):
    c = pd.DataFrame(cluster)
    w = c["promoter_score"].to_numpy()
    pos = c["pred_tss"].to_numpy()
    summit = c.loc[c["promoter_score"].idxmax()]
    return {
        "chromosome": c["chromosome"].iloc[0],
        "strand": strand,
        "tss_position": int(round(np.average(pos, weights=w))),  # weighted mean
        "summit_tss": int(summit["pred_tss"]),
        "region_start": int(c["start"].min() + 1),   # 1-based, inclusive
        "region_end": int(c["end"].max()),           # 1-based, inclusive
        "peak_score": float(w.max()),
        "mean_score": float(w.mean()),
        "n_windows": int(len(c)),
    }


# --------------------------------------------------------------------------- #
# Matching predicted vs known
# --------------------------------------------------------------------------- #
def match(cand, reg, tolerance):
    """Annotate each candidate with its nearest known TSS (same strand) and
    compute recall over the known set."""
    if cand.empty:
        cand["nearest_known_id"] = []
        cand["distance_to_known"] = []
        cand["is_validated"] = []
        return cand, 0.0, np.array([])

    nearest_id, nearest_dist, validated = [], [], []
    for _, row in cand.iterrows():
        same = reg[reg["strand"] == row["strand"]]
        if same.empty:
            nearest_id.append(None); nearest_dist.append(np.nan); validated.append(False)
            continue
        # Use summit_tss (argmax score) rather than tss_position (weighted-mean centroid).
        # For wide clusters the weighted mean drifts toward the cluster centre and can be
        # hundreds of bp from the true TSS; the score summit is a sharper estimate.
        d = same["known_tss"].to_numpy() - row["summit_tss"]   # signed (known - pred)
        j = np.argmin(np.abs(d))
        signed = -int(d[j])  # report as predicted - known
        nearest_id.append(same["known_id"].iloc[j])
        nearest_dist.append(signed)
        validated.append(abs(signed) <= tolerance)
    cand = cand.copy()
    cand["nearest_known_id"] = nearest_id
    cand["distance_to_known"] = nearest_dist
    cand["is_validated"] = validated

    # Recall: known TSS recovered by >=1 candidate within tolerance (per strand).
    recovered = 0
    matched_distances = []
    cand_by_strand = {s: g["summit_tss"].to_numpy() for s, g in cand.groupby("strand")}
    for _, k in reg.iterrows():
        preds = cand_by_strand.get(k["strand"])
        if preds is None or len(preds) == 0:
            continue
        dd = preds - k["known_tss"]
        if np.min(np.abs(dd)) <= tolerance:
            recovered += 1
            matched_distances.append(int(dd[np.argmin(np.abs(dd))]))
    recall = recovered / len(reg) if len(reg) else 0.0
    return cand, recall, np.array(matched_distances)


def threshold_sweep(df, reg, tolerance, merge_gap, thresholds):
    rows = []
    for t in thresholds:
        cand = consolidate(df, t, merge_gap)
        cand, recall, md = match(cand, reg, tolerance)
        precision = (cand["is_validated"].mean() if not cand.empty else 0.0)
        rows.append({
            "threshold": t,
            "n_candidates": int(len(cand)),
            "recall": round(recall, 4),
            "precision": round(float(precision), 4),
            "median_abs_dist": (round(float(np.median(np.abs(md))), 1) if len(md) else None),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="PT5 predictions CSV")
    ap.add_argument("--regulondb", required=True, help="RegulonDB TSS table (TSV)")
    ap.add_argument("--genome", help="genome FASTA (for length / background null)")
    ap.add_argument("--genome-length", type=int, help="override genome length (bp)")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--tolerance", type=int, default=20, help="match window, +/- bp")
    ap.add_argument("--merge-gap", type=int, default=40,
                    help="max bp gap between window TSS to stay in one cluster")
    ap.add_argument("--reg-pos-col", default="4)posTSS")
    ap.add_argument("--reg-strand-col", default="3)strand")
    ap.add_argument("--reg-id-col", default="2)name")
    ap.add_argument("--reg-sep", default="\t")
    ap.add_argument("--outdir", default="pt6_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    glen = (args.genome_length
            or (genome_length_from_fasta(args.genome) if args.genome else None))

    df = load_predictions(args.pred)
    reg = load_regulondb(args.regulondb, args.reg_pos_col, args.reg_strand_col,
                         args.reg_id_col, args.reg_sep)
    print(f"Loaded {len(df):,} windows and {len(reg):,} known TSS.")

    cand = consolidate(df, args.threshold, args.merge_gap)
    cand, recall, matched_d = match(cand, reg, args.tolerance)

    # --- metrics ---------------------------------------------------------- #
    precision = float(cand["is_validated"].mean()) if not cand.empty else 0.0
    # analytic random baseline: chance a position lands within +/-tol of a known TSS
    rand_precision = None
    if glen:
        per_strand = reg.groupby("strand").size().to_dict()
        # expected hit prob averaged over the candidates' strands
        if not cand.empty:
            probs = []
            for s, g in cand.groupby("strand"):
                k = per_strand.get(s, 0)
                probs.extend([min(1.0, k * (2 * args.tolerance + 1) / glen)] * len(g))
            rand_precision = float(np.mean(probs))

    metrics = {
        "params": {
            "threshold": args.threshold, "tolerance_bp": args.tolerance,
            "merge_gap_bp": args.merge_gap, "window_size": WINDOW_SIZE,
            "tss_offset": TSS_OFFSET, "genome_length": glen,
        },
        "n_windows": int(len(df)),
        "n_known_tss": int(len(reg)),
        "n_candidates": int(len(cand)),
        "recall": round(recall, 4),
        "precision_vs_known": round(precision, 4),
        "random_precision_baseline": (round(rand_precision, 4)
                                      if rand_precision is not None else None),
        "enrichment_over_random": (round(precision / rand_precision, 2)
                                   if rand_precision else None),
        "matched_abs_dist_median": (float(np.median(np.abs(matched_d)))
                                    if len(matched_d) else None),
        "matched_abs_dist_mean": (float(np.mean(np.abs(matched_d)))
                                  if len(matched_d) else None),
        "matched_signed_dist_median": (float(np.median(matched_d))
                                       if len(matched_d) else None),
    }

    sweep = threshold_sweep(df, reg, args.tolerance, args.merge_gap,
                            thresholds=[0.5, 0.7, 0.8, 0.9, 0.95, 0.99])

    # --- write outputs ---------------------------------------------------- #
    cand_cols = ["candidate_id", "chromosome", "strand", "tss_position",
                 "summit_tss", "region_start", "region_end", "peak_score",
                 "mean_score", "n_windows", "nearest_known_id",
                 "distance_to_known", "is_validated"]
    cand_path = os.path.join(args.outdir, "candidate_promoters.tsv")
    cand.reindex(columns=cand_cols).to_csv(cand_path, sep="\t", index=False)

    # BED (0-based half-open) for genome-browser sanity checks in IGV
    bed_path = os.path.join(args.outdir, "candidate_promoters.bed")
    if not cand.empty:
        bed = pd.DataFrame({
            "chrom": cand["chromosome"],
            "start": cand["tss_position"] - 1,
            "end": cand["tss_position"],
            "name": cand["candidate_id"],
            "score": (cand["peak_score"] * 1000).round().astype(int).clip(0, 1000),
            "strand": cand["strand"],
        })
        bed.to_csv(bed_path, sep="\t", header=False, index=False)

    sweep.to_csv(os.path.join(args.outdir, "threshold_sweep.tsv"), sep="\t", index=False)
    if len(matched_d):
        pd.DataFrame({"signed_dist": matched_d}).to_csv(
            os.path.join(args.outdir, "matched_distances.csv"), index=False)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    # --- console summary -------------------------------------------------- #
    print("\n=== PT6 validation summary ===")
    print(json.dumps(metrics, indent=2))
    print("\nThreshold sweep:")
    print(sweep.to_string(index=False))
    print(f"\nWrote: {cand_path}")
    print(f"       {bed_path}")
    print(f"       {os.path.join(args.outdir, 'threshold_sweep.tsv')}")
    print(f"       {os.path.join(args.outdir, 'metrics.json')}")


if __name__ == "__main__":
    main()
