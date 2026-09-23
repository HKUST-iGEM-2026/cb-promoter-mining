#!/usr/bin/env python3
"""
context_leakage_check.py — how much of NTv3's val/test CONTEXT was already in TRAIN?

WHY
---
T15 clusters/splits on the 81 bp core (`sequence`); the long `sequence_context`
(384 bp for NTv3) rides along into whatever split its core lands in
(cdhit_splits.py step 5 + build_context_windows.py header). So the anti-leakage
guarantee is enforced on the 81 bp core ONLY. Two windows with DIFFERENT cores
(-> different clusters -> can split across train/test) but OVERLAPPING 384 bp
context — because their cores sit <~300 bp apart on the same contig, or because
the flank is a repeat/paralog, or because an SRS negative keeps the REAL flanks
of its (train-pinned) T13 source — are never co-clustered, so their shared real
genomic sequence straddles the train/test boundary. NTv3 reads that context, so
it can see val/test sequence during training. This script measures how often.

WHAT IT MEASURES (model-free, no GPU)
-------------------------------------
For every val and test row it reports how many bp of its `sequence_context`
recur VERBATIM (either strand) in ANY training row's `sequence_context`:

  * k-mer method (always): exact 2-bit-packed k-mers (default k=31, collision-
    free for k<=31), indexed on BOTH strands of every train context. For each
    val/test context we find the LONGEST run of consecutive train-present k-mers
    -> `shared_bp = run + k - 1`. A physically overlapping / repeated / copied
    flank shows a long contiguous run; an unrelated window shows ~0. Low-
    complexity k-mers (<3 distinct bases, i.e. poly-A/T that is everywhere in an
    AT-rich Clostridium genome) are masked by default so AT tracts don't inflate
    the number — this is the conservative, defensible metric.

  * coordinate method (only if genome/contig/start/end columns survive into the
    parquet): exact genomic-interval overlap of the 384 bp windows. This is the
    ground truth for the "physically overlapping windows" leak but misses cross-
    contig repeats/paralogs (which the k-mer method catches). Reported when
    available; skipped with a note otherwise.

OUTPUT
------
A printed summary (leak rate by split x label x source, at several bp cutoffs)
and a per-row CSV so you can slice it (e.g. how much is the SRS-negative channel).

INTERPRETATION
--------------
`shared_bp` is "how many bp of this val/test window NTv3 already saw in training."
High recurrence on val/test => the reported balanced AUPRC is optimistic by that
much, and the true generalisation number is lower. This does NOT by itself say
the model is bad — it says re-measure after decorrelating context across splits
(and confirm at genome scale, Regime C).

USAGE (HPC4)
------------
  conda activate /project/igem2026/envs/promoter-mining-hpc4   # pandas+pyarrow+numpy
  python dataset-construction/clustering-and-splits/context_leakage_check.py \
      --data-dir /project/igem2026/cb-promoter-mining/infrastructure/T3/snakemake_scaffold/results/T15_xs_ctx \
      --seq-col sequence_context --core-col sequence \
      --out results/T15_xs_ctx_context_leakage.csv

  # sanity-check the tool itself first (no data needed):
  python dataset-construction/clustering-and-splits/context_leakage_check.py --self-test
"""
import argparse
import os
import sys

import numpy as np

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s: str) -> str:
    return s.translate(COMPLEMENT)[::-1]


# ---- 2-bit k-mer encoding ---------------------------------------------------
_LUT = np.full(256, 255, dtype=np.uint8)
for _b, _c in zip(b"ACGTacgt", [0, 1, 2, 3, 0, 1, 2, 3]):
    _LUT[_b] = _c


def forward_kmer_codes(seq: str, k: int, mask_lowcomplexity: bool):
    """Return an int64 array of length len(seq)-k+1 of forward k-mer codes,
    with -1 where the k-mer contains a non-ACGT base (or, if masking, has <3
    distinct bases). Codes are 2-bit-packed and thus collision-free for k<=31."""
    if len(seq) < k:
        return np.empty(0, dtype=np.int64)
    codes = _LUT[np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)]
    n = codes.shape[0]
    valid_base = codes != 255
    sw = np.lib.stride_tricks.sliding_window_view(codes.astype(np.int64), k)
    valid = np.lib.stride_tricks.sliding_window_view(valid_base, k).all(axis=1)
    weights = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)
    kcodes = (sw * weights).sum(axis=1)
    if mask_lowcomplexity:
        # distinct bases present in each window (mask windows dominated by <=2 bases)
        distinct = np.zeros(kcodes.shape[0], dtype=np.int64)
        for b in range(4):
            distinct += np.lib.stride_tricks.sliding_window_view(codes == b, k).any(axis=1)
        valid &= (distinct >= 3)
    kcodes[~valid] = -1
    return kcodes


def build_train_index(seqs, k, mask_lc):
    """Sorted unique int64 array of k-mer codes over BOTH strands of all train contexts."""
    chunks = []
    for s in seqs:
        s = str(s).upper()
        f = forward_kmer_codes(s, k, mask_lc)
        r = forward_kmer_codes(revcomp(s), k, mask_lc)
        chunks.append(f[f >= 0])
        chunks.append(r[r >= 0])
    if not chunks:
        return np.empty(0, dtype=np.int64)
    allc = np.concatenate(chunks)
    return np.unique(allc)


def longest_shared_bp(seq, k, train_sorted, mask_lc):
    """Longest run of consecutive train-present forward k-mers in `seq`,
    expressed in bp (run + k - 1); 0 if none."""
    q = forward_kmer_codes(str(seq).upper(), k, mask_lc)
    if q.size == 0:
        return 0
    member = np.zeros(q.shape[0], dtype=bool)
    valid = q >= 0
    if train_sorted.size and valid.any():
        qi = q[valid]
        pos = np.searchsorted(train_sorted, qi)
        pos = np.clip(pos, 0, train_sorted.size - 1)
        hit = train_sorted[pos] == qi
        member[np.where(valid)[0]] = hit
    best = cur = 0
    for v in member.tolist():
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return best + k - 1 if best > 0 else 0


# ---- coordinate (exact interval) method -------------------------------------
def detect_coord_cols(cols):
    g = next((c for c in ("genome", "genome_id", "assembly") if c in cols), None)
    ctg = next((c for c in ("contig", "replicon", "seqid") if c in cols), None)
    s = next((c for c in ("position_start", "start", "core_start") if c in cols), None)
    e = next((c for c in ("position_end", "end", "core_end") if c in cols), None)
    return (g, ctg, s, e) if all((g, ctg, s, e)) else None


def context_intervals(df, coord, seq_col, core_col):
    """Genomic interval [gstart, gend) spanned by each row's context window.
    up=(L-core)//2, down=L-core-up; genomic span is strand-independent.
    Returns (keys, gstart, gend, valid) where `valid` is False for rows without
    REAL coordinates (has_genomic_context==False, or NaN/placeholder start/end) —
    e.g. cross-species T8, whose context is N-padded, not a genomic window. Those
    must be excluded or they manufacture fake overlaps among identical placeholders."""
    g, ctg, s, e = coord
    L = df[seq_col].astype(str).str.len().to_numpy()
    core = df[core_col].astype(str).str.len().to_numpy()
    up = (L - core) // 2
    down = L - core - up
    start = df[s].to_numpy()
    end = df[e].to_numpy()
    valid = ~(np.isnan(start.astype(float)) | np.isnan(end.astype(float)))
    valid &= (end.astype(float) > 0)
    if "has_genomic_context" in df.columns:
        valid &= df["has_genomic_context"].fillna(False).to_numpy().astype(bool)
    gstart = start.astype(np.int64) - up
    gend = end.astype(np.int64) + down
    keys = (df[g].astype(str) + "\t" + df[ctg].astype(str)).to_numpy()
    return keys, np.maximum(0, gstart), gend, valid


def max_coord_overlap(val_keys, vs, ve, train_by_key):
    out = np.zeros(vs.shape[0], dtype=np.int64)
    for i in range(vs.shape[0]):
        ivals = train_by_key.get(val_keys[i])
        if not ivals:
            continue
        a, b = vs[i], ve[i]
        best = 0
        for (ta, tb) in ivals:              # small per (genome,contig); linear is fine
            ov = min(b, tb) - max(a, ta)
            if ov > best:
                best = ov
        out[i] = max(0, best)
    return out


# ---- reporting --------------------------------------------------------------
def summarize(name, sub, cutoffs, col):
    n = len(sub)
    if n == 0:
        print(f"    {name:<26} n=0")
        return
    vals = sub[col].to_numpy()
    frac = " ".join(f">={c}bp {100*np.mean(vals >= c):5.1f}%" for c in cutoffs)
    print(f"    {name:<26} n={n:<6} median={int(np.median(vals)):<4} max={int(vals.max()):<4} | {frac}")


def report(df, cutoffs, col, title):
    print(f"\n=== {title} ===")
    for split in ("val", "test"):
        s = df[df.split == split]
        if s.empty:
            continue
        print(f"  [{split}]  n={len(s)}")
        summarize("all", s, cutoffs, col)
        for lab, nm in ((1, "positives"), (0, "negatives")):
            summarize(nm, s[s.label == lab], cutoffs, col)
        if "source" in df.columns:
            for src in sorted(s.source.astype(str).unique()):
                summarize(f"src={src}", s[s.source.astype(str) == src], cutoffs, col)


# ---- self test --------------------------------------------------------------
def self_test():
    import pandas as pd
    rng = np.random.default_rng(0)

    def rseq(n):
        return "".join(rng.choice(list("ACGT"), n))

    k = 31
    # train context; a val row that shares a 200 bp genomic overlap with it;
    # a val row that shares nothing; a val row equal to revcomp of a train chunk.
    train_ctx = rseq(384)
    overlap = train_ctx[120:320]                       # 200 bp real overlap
    val_overlap = rseq(90) + overlap + rseq(384 - 90 - 200)
    val_clean = rseq(384)
    val_rc = rseq(80) + revcomp(train_ctx[50:270]) + rseq(384 - 80 - 220)
    train = pd.DataFrame({"sequence_context": [train_ctx, rseq(384)],
                          "sequence": [train_ctx[151:232], rseq(81)],
                          "split": "train", "label": 1, "source": "x"})
    val = pd.DataFrame({"sequence_context": [val_overlap, val_clean, val_rc],
                        "sequence": [val_overlap[151:232], val_clean[151:232], val_rc[151:232]],
                        "split": "val", "label": [1, 0, 1], "source": "x"})
    idx = build_train_index(train["sequence_context"], k, mask_lc=True)
    got = [longest_shared_bp(s, k, idx, True) for s in val["sequence_context"]]
    print("self-test shared_bp:", got, "(expect ~200, ~0, ~220)")
    assert got[0] >= 190, got
    assert got[1] < k, got                              # clean row: no long run
    assert got[2] >= 210, got                           # revcomp overlap caught
    print("SELF-TEST PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", help="dir with train/val/test.parquet")
    ap.add_argument("--seq-col", default="sequence_context")
    ap.add_argument("--core-col", default="sequence")
    ap.add_argument("--k", type=int, default=31, help="k-mer size (<=31, collision-free)")
    ap.add_argument("--cutoffs", default="50,100,150", help="bp thresholds to report")
    ap.add_argument("--no-lowcomplexity-mask", action="store_true",
                    help="do NOT mask <3-distinct-base k-mers (poly-A/T). Default masks them.")
    ap.add_argument("--out", default="context_leakage.csv")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        self_test()
        return
    if not a.data_dir:
        sys.exit("ERROR: --data-dir required (or use --self-test).")
    if a.k > 31:
        sys.exit("ERROR: --k must be <=31 (2-bit packing into int64).")

    import pandas as pd
    frames = {}
    for split in ("train", "val", "test"):
        p = os.path.join(a.data_dir, f"{split}.parquet")
        if not os.path.exists(p):
            sys.exit(f"ERROR: missing {p}")
        d = pd.read_parquet(p)
        if a.seq_col not in d.columns:
            sys.exit(f"ERROR: '{a.seq_col}' not in {split}.parquet. Have: {list(d.columns)}")
        d["split"] = split
        frames[split] = d
    train, val, test = frames["train"], frames["val"], frames["test"]
    mask_lc = not a.no_lowcomplexity_mask
    cutoffs = [int(x) for x in a.cutoffs.split(",")]
    print(f"rows: train={len(train)} val={len(val)} test={len(test)} | "
          f"k={a.k} lowcomplexity_mask={mask_lc}")

    # k-mer recurrence -------------------------------------------------------
    print("building both-strand train k-mer index ...")
    idx = build_train_index(train[a.seq_col].tolist(), a.k, mask_lc)
    print(f"  {idx.size:,} unique train k-mers")
    q = pd.concat([val, test], ignore_index=True)
    q["kmer_shared_bp"] = [longest_shared_bp(s, a.k, idx, mask_lc) for s in q[a.seq_col]]
    for c in ("label", "source"):
        if c not in q.columns:
            q[c] = -1 if c == "label" else "?"
    report(q, cutoffs, "kmer_shared_bp",
           "CONTEXT RECURRENCE (bp of val/test context found verbatim in TRAIN, either strand)")

    # coordinate overlap (if coords survived) --------------------------------
    coord = detect_coord_cols(set(train.columns))
    if coord:
        print(f"\ncoordinate columns present {coord} -> exact interval-overlap too")
        tk, ts, te, tv = context_intervals(train, coord, a.seq_col, a.core_col)
        tby = {}
        for key, s0, e0, ok in zip(tk, ts, te, tv):
            if ok:
                tby.setdefault(key, []).append((s0, e0))
        vk, vs, ve, vv = context_intervals(q, coord, a.seq_col, a.core_col)
        ov = max_coord_overlap(vk, vs, ve, tby)
        ov[~vv] = -1                       # rows without real coords: not measurable
        q["coord_overlap_bp"] = ov
        n_skip = int((~vv).sum())
        print(f"  {tv.sum()}/{len(train)} train and {vv.sum()}/{len(q)} val+test rows have "
              f"real coordinates; {n_skip} val/test rows skipped (coord_overlap_bp=-1).")
        report(q[q.coord_overlap_bp >= 0], [1] + cutoffs, "coord_overlap_bp",
               "GENOMIC-INTERVAL OVERLAP of val/test context with any TRAIN context (real-coord rows)")
    else:
        print("\n[note] no usable genome/contig/start/end columns in the parquet — "
              "coordinate method skipped (k-mer recurrence above is the metric). "
              "cdhit_splits.py does not carry coordinates through; to enable the exact "
              "check, add contig/position_start/position_end/genome to load_source's output.")

    keep = [c for c in ("split", "label", "source", "in_genus", "promoter_id",
                        "kmer_shared_bp", "coord_overlap_bp") if c in q.columns]
    q[keep].to_csv(a.out, index=False)
    print(f"\nwrote per-row detail -> {a.out}")
    print("HEADLINE: the '>=100bp' row-fractions above are the context-leak rate; "
          "if non-trivial on val/test, NTv3's balanced AUPRC is optimistic by ~that "
          "much. Decisive follow-up = genome-scale Regime C.")


if __name__ == "__main__":
    main()
