#!/usr/bin/env python3
"""
Genome-scale false-positive benchmark — the metric Umarov's review said actually matters.

The balanced test set (T22/T25/modular_eval) can't show this: at genome scale positives are ~0.2%
of windows, so even a strong classifier throws many false positives per scan. This slides a model's
score track across a genome WITH known promoters and reports, at several recall targets, how many
false-positive PEAKS the model produces genome-wide, per Mb, the genome-scale precision, and the
enrichment over random.

Inputs:
  --scores    per-window score track from genome_inference.py (or PT5): columns
              contig, start, strand, promoter_score
  --truth     TSS table from tss_to_truth.py: columns contig, pos, strand

CONTIG-AWARE: coverage and FP-peak merging happen WITHIN (contig[, strand]), so multi-replicon
Clostridium genomes (chromosome + plasmid) are handled correctly. STRAND-AWARE by default (a + window
can't cover a - TSS); pass --ignore-strand to disable. Matching: a window [start, start+window] (± tol)
containing a known TSS (same contig & strand) is a HIT; recall(t) = fraction of ALL known TSS hit by
some window scoring >= t. A false positive is a window scoring >= t hitting no TSS; consecutive FP
windows (gap <= merge, same contig & strand) collapse into one FP PEAK.

To benchmark YOUR models: make a track with genome_inference.py using the fine-tuned checkpoint
(ProkBERT results/T18/best_model, or NT results/T19/best_adapter), then point --scores here.

Outputs:  <out>/genome_fp_benchmark.md  and  .csv
  python genome_fp_benchmark.py --scores cac_t18_track.csv --truth cac_truth.tsv \
      --genome-length 3940880 --out results/fp_bench_cac
"""
import argparse
import os
import numpy as np
import pandas as pd


def load_track(path, contig_col, start_col, strand_col, score_col):
    df = pd.read_csv(path)
    for c in (start_col, score_col):
        if c not in df.columns:
            raise SystemExit(f"score track missing '{c}'; has {list(df.columns)}")
    contig = df[contig_col].astype(str).to_numpy() if contig_col in df.columns else np.array(["."] * len(df))
    strand = df[strand_col].astype(str).to_numpy() if strand_col in df.columns else np.array(["."] * len(df))
    return contig, df[start_col].to_numpy(int), strand, df[score_col].to_numpy(float)


def load_truth(path, contig_col, pos_col, strand_col, sep):
    df = pd.read_csv(path, sep=sep)
    if pos_col not in df.columns:
        raise SystemExit(f"truth table missing position column '{pos_col}'; has {list(df.columns)}")
    contig = df[contig_col].astype(str).to_numpy() if contig_col in df.columns else np.array(["."] * len(df))
    strand = df[strand_col].astype(str).to_numpy() if strand_col in df.columns else np.array(["."] * len(df))
    return contig, df[pos_col].to_numpy(int), strand


def load_regions(path):
    """BED (contig,start,end,strand) -> {(contig,strand): (sorted_starts, sorted_ends)} (non-overlapping)."""
    df = pd.read_csv(path, sep="\t")
    out = {}
    for (c, st), g in df.groupby(["contig", "strand"]):
        g = g.sort_values("start")
        out[(str(c), str(st))] = (g["start"].to_numpy(int), g["end"].to_numpy(int))
    return out


def in_region_mask(wctg, wstart, wstr, window, regions):
    """Boolean mask: window [start, start+window) overlaps some region on its (contig, strand)."""
    keep = np.zeros(len(wstart), dtype=bool)
    we = wstart + window
    for (contig, strand), (rs, re) in regions.items():
        gi = np.where((wctg == contig) & (wstr == strand))[0]
        if len(gi) == 0:
            continue
        idx = np.searchsorted(rs, we[gi], side="right") - 1     # region with largest start < window end
        ok = idx >= 0
        ov = np.zeros(len(gi), dtype=bool)
        ov[ok] = re[idx[ok]] > wstart[gi][ok]                    # ...and it ends after the window starts
        keep[gi[ov]] = True
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out", default="results/fp_bench")
    ap.add_argument("--genome-length", type=int, default=0, help="bp; 0 = infer from max window end")
    ap.add_argument("--window", type=int, default=81)
    ap.add_argument("--tol", type=int, default=40, help="bp; a window within tol of a TSS covers it")
    ap.add_argument("--merge", type=int, default=81, help="bp; collapse FP windows within this gap into one peak")
    ap.add_argument("--recall-targets", default="0.5,0.7,0.9")
    ap.add_argument("--regions", default=None,
                    help="restrict-scan: BED (contig,start,end,strand) from make_candidate_regions.py; "
                         "only windows overlapping a region are scored (shrinks the FP denominator).")
    ap.add_argument("--ignore-strand", action="store_true", help="match regardless of strand (default: strand-aware)")
    ap.add_argument("--contig-col", default="contig")
    ap.add_argument("--start-col", default="start")
    ap.add_argument("--strand-col", default="strand")
    ap.add_argument("--score-col", default="promoter_score")
    ap.add_argument("--truth-contig-col", default="contig")
    ap.add_argument("--truth-pos-col", default="pos")
    ap.add_argument("--truth-strand-col", default="strand")
    ap.add_argument("--truth-sep", default="\t")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    use_strand = not a.ignore_strand

    wctg, wstart, wstr, wsc = load_track(a.scores, a.contig_col, a.start_col, a.strand_col, a.score_col)
    n_full = len(wsc)
    region_note = ""
    if a.regions:
        keep = in_region_mask(wctg, wstart, wstr, a.window, load_regions(a.regions))
        wctg, wstart, wstr, wsc = wctg[keep], wstart[keep], wstr[keep], wsc[keep]
        region_note = (f" **Restrict-scan:** {int(keep.sum()):,}/{n_full:,} windows "
                       f"({keep.mean():.1%}) in candidate regions.")
    tctg, tpos, tstr = load_truth(a.truth, a.truth_contig_col, a.truth_pos_col, a.truth_strand_col, a.truth_sep)
    n_win, n_truth = len(wsc), len(tpos)
    if n_win == 0 or n_truth == 0:
        raise SystemExit("empty score track or truth table.")
    glen = a.genome_length or int((wstart + a.window).max())

    # group windows by contig (+ strand); keep each group's starts sorted for fast coverage lookup
    keyw = np.char.add(np.char.add(wctg, "|"), wstr) if use_strand else wctg.astype(str)
    wdf = pd.DataFrame({"key": keyw, "start": wstart, "score": wsc, "idx": np.arange(n_win)})
    groups = {k: g.sort_values("start") for k, g in wdf.groupby("key", sort=False)}

    truth_best = np.full(n_truth, -np.inf)     # best covering-window score per TSS
    matched = np.zeros(n_win, dtype=bool)       # window covers some TSS
    for k in range(n_truth):
        key = f"{tctg[k]}|{tstr[k]}" if use_strand else str(tctg[k])
        g = groups.get(key)
        if g is None:
            continue
        starts = g["start"].to_numpy()
        p = tpos[k]
        i0 = np.searchsorted(starts, p - a.window - a.tol, "left")   # window covers p iff start in [p-window-tol, p+tol]
        i1 = np.searchsorted(starts, p + a.tol, "right")
        if i1 > i0:
            sub = g.iloc[i0:i1]
            truth_best[k] = float(sub["score"].max())
            matched[sub["idx"].to_numpy()] = True

    def fp_peaks(thr):
        m = (~matched) & (wsc >= thr)
        if not m.any():
            return 0
        d = pd.DataFrame({"key": keyw[m], "start": wstart[m]}).sort_values(["key", "start"])
        peaks = 0
        for _, g in d.groupby("key", sort=False):
            s = g["start"].to_numpy()
            peaks += 1 + int((np.diff(s) > a.merge).sum())
        return peaks

    finite = np.sort(truth_best[np.isfinite(truth_best)])[::-1]     # detectable TSS best-scores, descending
    detectable = len(finite)
    if detectable == 0:
        # 0 TSS covered by any window is never a real result — it means the track and truth don't
        # join (contig id / strand / coordinate mismatch). Fail loud instead of writing a nan table.
        raise SystemExit(
            f"0 of {n_truth:,} known TSS are covered by any scored window — the score track and truth "
            f"table do not join. Check that their contig ids and strands match.\n"
            f"  track contigs : {sorted(set(wctg))[:5]}\n"
            f"  truth contigs : {sorted(set(tctg))[:5]}\n"
            f"  strand-aware  : {use_strand} (track strands {sorted(set(wstr))}, truth strands {sorted(set(tstr))})")
    base_rate = n_truth / n_win

    rows, md = [], []
    md.append("# Genome-scale false-positive benchmark\n")
    md.append(f"Genome length {glen:,} bp - windows scored {n_win:,} - known TSS {n_truth:,} "
              f"({detectable:,} covered by >=1 window). Genome-scale base rate (TSS / windows) = {base_rate:.4%}. "
              f"Contigs in track: {len(set(wctg))}. strand-aware={use_strand}.{region_note}\n")
    md.append(f"tol {a.tol} bp - FP-peak merge {a.merge} bp.\n")
    md.append("| Recall target | threshold | TSS detected | FP peaks | FP per Mb | precision | enrichment vs random |")
    md.append("|---|---|---|---|---|---|---|")
    for r in [float(x) for x in a.recall_targets.split(",")]:
        k = int(np.ceil(r * n_truth))                          # detect this many of ALL TSS
        thr = float(finite[k - 1]) if (0 < k <= detectable) else (float(finite[-1]) if detectable else np.inf)
        tp = int((truth_best >= thr).sum())
        fp = fp_peaks(thr)
        rec = tp / n_truth
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        fp_mb = fp / (glen / 1e6)
        enr = (prec / base_rate) if (base_rate and np.isfinite(prec)) else float("nan")
        md.append(f"| {r:.2f} (got {rec:.2f}) | {thr:.3f} | {tp:,} | {fp:,} | {fp_mb:,.0f} | {prec:.3%} | {enr:.1f}x |")
        rows.append({"recall_target": r, "recall_actual": round(rec, 4), "threshold": round(thr, 4),
                     "tss_detected": tp, "fp_peaks": fp, "fp_per_mb": round(fp_mb, 1),
                     "precision": round(prec, 5), "enrichment": round(enr, 2)})

    md.append(f"\n_Max achievable recall = {detectable / n_truth:.2%} "
              f"({n_truth - detectable:,} TSS have no covering window in the track)._\n")
    md.append("**Read as:** at each recall, `FP peaks` is how many false promoters the model would hand "
              "the wet lab genome-wide; `precision` is the fraction of its calls that are real; `enrichment` "
              "is how much better than random. F1/AUPRC on the balanced set hide all of this.\n")

    pd.DataFrame(rows).to_csv(os.path.join(a.out, "genome_fp_benchmark.csv"), index=False)
    text = "\n".join(md) + "\n"
    with open(os.path.join(a.out, "genome_fp_benchmark.md"), "w") as f:
        f.write(text)
    print(text)
    print("Wrote", os.path.join(a.out, "genome_fp_benchmark.{md,csv}"))


if __name__ == "__main__":
    main()
