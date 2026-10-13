#!/usr/bin/env python3
"""
Interpretability check - sequence logos from each model's TOP-CONFIDENCE predictions.

WHY: T22/T25/modular_eval tell you HOW WELL each model separates classes, but not WHAT it
learned. This takes each model's highest-scoring TRUE POSITIVES and highest-scoring negatives
("false positives") from the T22 test predictions, aligns them on the TSS, and builds a
sequence logo for each, plus a model-independent reference logo from known positives.

WHAT TO LOOK FOR:
  - Top true positives should show a real sigma70 signal: a -10 box (~TATAAT-like) and often a
    -35 box (~TTGACA-like), with a flatter spacer between them.
  - Top-scoring negatives that ALSO show a clean -10/-35 box mean the model found a real
    promoter-like sequence in a negative region (biologically interesting, not a bug).
    Top-scoring negatives that look like generic AT-rich noise with NO motif structure mean
    the model is leaning on base composition (the AT-confound from T25's GC-quartile dip /
    T26's calibration covariate) rather than regulatory signal.

IMPORTANT - two things that will silently give you a WRONG answer if unchecked:

  1. STRAND. If T13/T16 stored windows on the genomic + strand regardless of promoter
     orientation, half your sequences carry a reverse-complemented -10 box and every logo
     flattens into AT-rich mush - which looks exactly like "the model learned composition".
     Pass --strand-col if your parquet has strand info; otherwise verify upstream that
     windows are already sense-oriented.

  2. BACKGROUND. Information content against a uniform 25% background makes ordinary AT-rich
     composition look like signal at every position. Since separating motif from composition
     is the whole point here, --background negatives (default) computes the background from
     the label-0 windows so a position only rises above baseline if it is enriched beyond
     what local composition already predicts. Use --background uniform to get the classic
     textbook-style logo for comparison.

INPUT ASSUMPTIONS (adjust with flags if your schema differs):
  - results/T15/test.parquet has: label, source, and a raw-sequence column (default "sequence").
  - results/T22/{model}_test_predictions.csv has: label, prob (from predict_and_score.py) and
    is ROW-ALIGNED with T15/test.parquet. This is now VERIFIED against the label column, not
    just assumed from row count.

Requires: pandas, numpy, matplotlib; logomaker for the PNGs
(pip install logomaker --break-system-packages). Without logomaker it still writes the
frequency matrices and manifests as CSV, so a run never silently produces nothing.

Run from the scaffold dir (same convention as your other T22 scripts):
  python pipeline/evaluation/topN_sequence_logos.py \
      --t15-dir results/T15 --t22-dir results/T22 --out results/logos --top-n 200
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

MODELS = ["prokbert", "nt", "cnn", "cnn_v2"]
BASES = ["A", "C", "G", "T"]
_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
PSEUDOCOUNT = 0.25  # keeps all-N positions from producing a zero row


def revcomp(s):
    return s.translate(_COMP)[::-1]


def load_preds(t22_dir, key, labels):
    """Return prob array for `key`, or None. Verifies row-alignment via the label column."""
    path = os.path.join(t22_dir, f"{key}_test_predictions.csv")
    if not os.path.exists(path):
        print(f"  no predictions file at {path} - skipping.")
        return None
    df = pd.read_csv(path)
    if len(df) != len(labels):
        print(f"  !! {key}: {len(df)} preds != {len(labels)} T15 test rows - skipping.")
        return None
    if "prob" not in df.columns:
        print(f"  !! {key}: no 'prob' column in {path} - skipping.")
        return None
    # The real alignment check. Matching row COUNT is not matching row ORDER; a shuffled
    # DataLoader writes a same-length file whose rows correspond to different sequences.
    if "label" in df.columns:
        mismatch = int((df["label"].astype(int).to_numpy() != labels).sum())
        if mismatch:
            print(f"  !! {key}: {mismatch}/{len(labels)} label mismatches vs test.parquet - "
                  f"predictions are NOT row-aligned. Skipping (logos would be meaningless).")
            return None
    else:
        print(f"  ~  {key}: no 'label' column, cannot verify row alignment - trusting order.")
    probs = df["prob"].to_numpy(float)
    n_nan = int(np.isnan(probs).sum())
    if n_nan:
        # np.argsort sorts NaN LAST; reversing for descending order would put them FIRST and
        # your entire top-N would be NaN rows. Mask them out instead.
        print(f"  ~  {key}: {n_nan} NaN probs, excluded from ranking.")
    return probs


def freq_matrix(seqs, positions):
    """Position-wise base frequency matrix (index=TSS-relative position, cols=A/C/G/T)."""
    L = len(positions)
    counts = np.full((L, 4), PSEUDOCOUNT, dtype=float)
    base_idx = {b: i for i, b in enumerate(BASES)}
    for s in seqs:
        for pos, ch in enumerate(s.upper()):
            j = base_idx.get(ch)
            if j is not None:  # N and other degenerates are simply not counted
                counts[pos, j] += 1
    freqs = counts / counts.sum(axis=1, keepdims=True)
    return pd.DataFrame(freqs, columns=BASES, index=pd.Index(positions, name="position"))


def background_freqs(seqs):
    """Overall base composition across a set of sequences -> length-4 array."""
    counts = np.full(4, PSEUDOCOUNT, dtype=float)
    base_idx = {b: i for i, b in enumerate(BASES)}
    for s in seqs:
        for ch in s.upper():
            j = base_idx.get(ch)
            if j is not None:
                counts[j] += 1
    return counts / counts.sum()


def prepare_seqs(seqs, strands, tss_index):
    """Orient to sense strand, verify equal length, return (seqs, relative positions)."""
    if strands is not None:
        seqs = np.array([revcomp(s) if str(st) in ("-", "-1", "minus") else s
                         for s, st in zip(seqs, strands)])
    lens = {len(s) for s in seqs}
    if len(lens) != 1:
        raise SystemExit(f"sequences are not all the same length: {sorted(lens)} - "
                         f"trim/pad upstream before logo-building.")
    L = lens.pop()
    if tss_index is None:
        tss_index = L // 2
    if not 0 <= tss_index < L:
        raise SystemExit(f"--tss-index {tss_index} outside window of length {L}")
    positions = np.arange(L) - tss_index
    return seqs, positions, tss_index


def dedupe(seqs, idx):
    """Drop exact duplicate sequences (paralogs / overlapping tiles fake a crisp motif)."""
    seen, keep = set(), []
    for i, s in zip(idx, seqs):
        if s not in seen:
            seen.add(s)
            keep.append(i)
    return np.array(keep, dtype=int)


def top_n_indices(mask, probs, n, rng=None, sample=False):
    """Row indices of the n highest-scoring rows within `mask` (NaN probs excluded)."""
    valid = mask & ~np.isnan(probs)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return idx
    if sample:
        return rng.permutation(idx)[:n]
    order = idx[np.argsort(probs[idx], kind="stable")][::-1]
    return order[:n]


def render(freqs, title, subtitle, png_path, csv_path, background):
    freqs.to_csv(csv_path)
    try:
        import matplotlib
        matplotlib.use("Agg")  # HPC nodes have no display
        import matplotlib.pyplot as plt
        import logomaker
    except ImportError as e:
        print(f"  {e.name} not installed - wrote frequency table only: {csv_path}")
        return None
    info = logomaker.transform_matrix(freqs, from_type="probability",
                                      to_type="information", background=background)
    fig, ax = plt.subplots(figsize=(max(6, len(freqs) * 0.14), 2.6))
    logomaker.Logo(info, ax=ax)
    ax.axvline(0, color="0.4", lw=0.8, ls="--", zorder=0)
    for box in (-10, -35):
        ax.axvspan(box - 3, box + 3, color="0.9", zorder=0)
    ax.set_title(f"{title}\n{subtitle}", fontsize=9)
    ax.set_xlabel("position relative to TSS (0 = TSS; shaded = expected -10 / -35)")
    ax.set_ylabel("bits")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {png_path}")
    return info


def combined_panel(panels, out_path):
    """One stacked figure of every logo, which is what side-by-side comparison needs."""
    if not panels:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import logomaker
    except ImportError:
        return
    n = len(panels)
    width = max(6, len(panels[0][1]) * 0.14)
    fig, axes = plt.subplots(n, 1, figsize=(width, 2.2 * n), sharex=True)
    axes = np.atleast_1d(axes)
    ymax = max(float(np.nansum(info.clip(lower=0).to_numpy(), axis=1).max())
               for _, info in panels)
    for ax, (label, info) in zip(axes, panels):
        logomaker.Logo(info, ax=ax)
        ax.axvline(0, color="0.4", lw=0.8, ls="--", zorder=0)
        ax.set_ylim(0, ymax * 1.05)  # shared scale, or the comparison is misleading
        ax.set_ylabel("bits", fontsize=8)
        ax.set_title(label, fontsize=8, loc="left")
    axes[-1].set_xlabel("position relative to TSS")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nwrote combined panel {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Sequence logos from top-confidence T22 predictions")
    ap.add_argument("--t15-dir", default="results/T15")
    ap.add_argument("--t22-dir", default="results/T22")
    ap.add_argument("--out", default="results/logos")
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--seq-col", default="sequence", help="raw-sequence column in test.parquet")
    ap.add_argument("--strand-col", default=None,
                    help="strand column; rows marked '-' are reverse-complemented first")
    ap.add_argument("--tss-index", type=int, default=None,
                    help="0-based index of the TSS within the window (default: window centre)")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--ref-sources", nargs="+", default=["RegulonDB", "DBTBS", "Hocq2022"],
                    help="source labels treated as the known-true-positive reference logo")
    ap.add_argument("--background", choices=["negatives", "dataset", "uniform"],
                    default="negatives",
                    help="composition baseline for information content (see module docstring)")
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    meta = pd.read_parquet(os.path.join(args.t15_dir, "test.parquet"))
    if args.seq_col not in meta.columns:
        raise SystemExit(f"'{args.seq_col}' not found in test.parquet columns: {list(meta.columns)}. "
                         f"Pass --seq-col to point at the raw sequence field.")
    raw = meta[args.seq_col].astype(str).to_numpy()
    y = meta["label"].astype(int).to_numpy()
    src = (meta["source"].astype(str).to_numpy() if "source" in meta.columns
           else np.array([""] * len(y)))
    strands = meta[args.strand_col].to_numpy() if args.strand_col else None
    if strands is None:
        print("NOTE: no --strand-col given. If T15 windows are NOT already sense-oriented, "
              "every logo below is invalid (half the motifs are reverse-complemented).")

    seqs, positions, tss_index = prepare_seqs(raw, strands, args.tss_index)
    print(f"{len(seqs)} test windows, length {len(positions)}, TSS at index {tss_index} "
          f"(positions {positions[0]}..{positions[-1]})")

    if args.background == "uniform":
        bg = np.array([0.25] * 4)
    elif args.background == "dataset":
        bg = background_freqs(seqs)
    else:
        bg = background_freqs(seqs[y == 0])
    print(f"background ({args.background}): " +
          ", ".join(f"{b}={v:.3f}" for b, v in zip(BASES, bg)))

    panels, manifest = [], []

    def build(idx, name, title, subtitle, probs=None):
        if len(idx) < 10:
            print(f"  skipping {name} - only {len(idx)} sequences")
            return
        sub = seqs[idx]
        info = render(freq_matrix(sub, positions), title, subtitle,
                      os.path.join(args.out, f"{name}_logo.png"),
                      os.path.join(args.out, f"{name}_freqs.csv"), bg)
        if info is not None:
            panels.append((f"{title} - {subtitle}", info))
        rows = pd.DataFrame({"logo": name, "row_index": idx, "label": y[idx], "source": src[idx]})
        if probs is not None:
            rows["prob"] = probs[idx]
        manifest.append(rows)

    # --- reference logo: known true positives, independent of any model -------------------
    ref_mask = (y == 1) & np.isin(src, args.ref_sources)
    if ref_mask.sum() < 10:
        print(f"skipping reference logo - only {int(ref_mask.sum())} rows match "
              f"{args.ref_sources}. Sources present: {sorted(set(src))[:20]}")
    else:
        # random sample, NOT head() - test.parquet is usually grouped by source/organism,
        # so the first N rows would be one organism rather than a representative set
        idx = top_n_indices(ref_mask, np.zeros(len(y)), args.top_n, rng=rng, sample=True)
        if not args.no_dedupe:
            idx = dedupe(seqs[idx], idx)
        build(idx, "reference_known_positives", "Reference: known positives",
              f"{', '.join(args.ref_sources)}, n={len(idx)} (random sample)")

    # --- per-model logos ------------------------------------------------------------------
    top_sets = {}
    for key in args.models:
        print(f"{key}:")
        probs = load_preds(args.t22_dir, key, y)
        if probs is None:
            continue

        for label_val, tag, human in ((1, "top_true_positives", "true positives"),
                                      (0, "top_false_positives", "scored-high negatives")):
            idx = top_n_indices(y == label_val, probs, args.top_n)
            if not args.no_dedupe:
                idx = dedupe(seqs[idx], idx)
            if len(idx):
                p = probs[idx]
                # the prob range matters: if the "false positives" all sit at p=0.2 they
                # aren't false positives at all, and the logo means something different
                build(idx, f"{key}_{tag}", f"{key}: top {len(idx)} {human}",
                      f"prob {p.min():.3f}-{p.max():.3f}", probs=probs)
                if label_val == 1:
                    top_sets[key] = set(idx.tolist())

    # --- how much do the models' top sets actually differ? --------------------------------
    keys = sorted(top_sets)
    if len(keys) > 1:
        print("\nTop-N true-positive overlap (Jaccard) - near 1.0 means the models are ranking "
              "the same windows and their logos cannot differ much:")
        overlap = {}
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                j = len(top_sets[a] & top_sets[b]) / len(top_sets[a] | top_sets[b])
                overlap[f"{a}|{b}"] = round(j, 3)
                print(f"  {a} vs {b}: {j:.3f}")
        with open(os.path.join(args.out, "topn_overlap.json"), "w") as fh:
            json.dump(overlap, fh, indent=2)

    if manifest:
        pd.concat(manifest, ignore_index=True).to_csv(
            os.path.join(args.out, "logo_manifest.csv"), index=False)
    combined_panel(panels, os.path.join(args.out, "all_logos_panel.png"))
    print(f"\nDone. Logos, frequency tables and logo_manifest.csv in {args.out}/")


if __name__ == "__main__":
    main()
