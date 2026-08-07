#!/usr/bin/env python3
"""
apply_calibrator.py — apply a saved T26 Platt(+AT) calibrator to a genome_inference score track.

This is PURE post-processing: no model forward pass. It reads a per-window score track from
genome_inference.py, feeds each window's [logit, at_fraction] to a saved calibrator, and writes a
track with promoter_score REPLACED by the calibrated probability. The schema is otherwise unchanged,
so genome_fp_benchmark.py --scores <this output> consumes it with no other changes.

What the calibrator consumes (and where each input comes from)
--------------------------------------------------------------
The T26 calibrators were fit on up to TWO features, IN THIS ORDER:
    [0] logit  = logits[:,1] - logits[:,0]        (always)
    [1] at_fraction                               (only for a score+AT calibrator)
This script sources each from the track when present, so no work is redone:
  * logit  — prefer the exact `logit` column (genome_inference --emit-logit). If absent, recover it
             from promoter_score via logit = log(p/(1-p)); this is exact for every model because
             promoter_score = softmax[:,1] = sigmoid(logit1-logit0), except where float32 saturated
             the prob to 0/1 (the eps clip compresses those top-confidence windows).
  * at_fraction — prefer the track's `at_fraction` column (also written by --emit-logit). If absent,
             re-derive it from --genome. A score-ONLY calibrator ignores AT entirely, so neither the
             column nor --genome is needed then.

So --genome is OPTIONAL: required only for a score+AT calibrator whose track lacks at_fraction.

One script for all four models
------------------------------
A score-only calibrator is a 1-feature pipeline; a score+AT one is 2-feature (cv_select refits on the
chosen columns). We read pipeline.n_features_in_ and pass [logit] or [logit, at] accordingly — no
per-model branching and no dependence on the sibling *_calibration_metrics.json.

AT is strand-invariant (the reverse complement has the same A+T count), so strand is ignored for it.

Run (CPU is fine — a few sklearn matmuls over the track):
  # track scored WITH genome_inference --emit-logit (has logit + at_fraction): no FASTA needed
  python apply_calibrator.py \
      --track  results/regimeC/clj_cnn_v2_track.csv \
      --calibrator evaluation-and-calibration/T26/calib_06/cnn_v2/cnn_v2_calibrator_at.joblib \
      --out    results/regimeC/clj_cnn_v2_calib_track.csv
  # older track with only promoter_score + a score+AT calibrator: add --genome to supply AT
  #   ... --genome C_ljungdahlii.fna
"""
import argparse
import sys
import numpy as np
import pandas as pd


def die(msg):
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def read_fasta(path):
    """Minimal FASTA reader -> {record_id: sequence}. record_id = header up to first whitespace."""
    seqs, name, chunks = {}, None, []
    try:
        f = open(path)
    except OSError as e:
        die(f"could not open genome FASTA {path}: {e}")
    with f:
        for line in f:
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(chunks)
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
    if name is not None:
        seqs[name] = "".join(chunks)
    if not seqs:
        die(f"no sequences parsed from {path}")
    return seqs


def at_cumsum(seq):
    """Prefix-sum of A/T indicator (length L+1): cum[e]-cum[s] = #AT in seq[s:e]."""
    arr = np.frombuffer(seq.upper().encode("ascii", "replace"), dtype=np.uint8)
    is_at = (arr == ord("A")) | (arr == ord("T"))
    cum = np.zeros(len(arr) + 1, dtype=np.int64)
    np.cumsum(is_at, out=cum[1:])
    return cum


def main():
    ap = argparse.ArgumentParser(description="Apply a saved Platt(+AT) calibrator to a score track.")
    ap.add_argument("--track", required=True, help="genome_inference.py CSV (contig,start,end,strand,promoter_score)")
    ap.add_argument("--calibrator", required=True, help="<model>_calibrator_at.joblib from T26")
    ap.add_argument("--out", required=True, help="output CSV (same schema; promoter_score = calibrated prob)")
    ap.add_argument("--genome", default=None,
                    help="genome FASTA — ONLY needed for a score+AT calibrator whose track has no "
                         "'at_fraction' column (i.e. was scored WITHOUT genome_inference --emit-logit). "
                         "Record ids must match the track's contig column. Unused for score-only "
                         "calibrators or when the track already carries AT.")
    ap.add_argument("--score-col", default="promoter_score")
    ap.add_argument("--logit-col", default="logit",
                    help="if this column exists in the track, use it as the calibrator's score feature "
                         "DIRECTLY (exact); otherwise fall back to inverting --score-col.")
    ap.add_argument("--at-col", default="at_fraction",
                    help="if this column exists in the track (genome_inference --emit-logit writes it), "
                         "use it as the AT covariate DIRECTLY; otherwise re-derive it from --genome.")
    ap.add_argument("--window", type=int, default=81, help="fallback window length if the track has no 'end' column")
    ap.add_argument("--eps", type=float, default=1e-7, help="clip prob to [eps, 1-eps] before the logit inversion")
    a = ap.parse_args()

    import joblib

    cal = joblib.load(a.calibrator)
    n_feat = getattr(cal, "n_features_in_", None)
    if n_feat not in (1, 2):
        die(f"calibrator reports n_features_in_={n_feat}; expected 1 (score) or 2 (score+AT). "
            f"Is {a.calibrator} really a T26 Platt(+AT) pipeline?")

    df = pd.read_csv(a.track)
    have_logit = a.logit_col in df.columns
    have_score = a.score_col in df.columns
    if "start" not in df.columns or "contig" not in df.columns or not (have_logit or have_score):
        die(f"track missing required columns; has {list(df.columns)} "
            f"(need contig, start, and one of {a.logit_col!r}/{a.score_col!r}; end optional).")
    df = df.reset_index(drop=True)
    start = df["start"].to_numpy(np.int64)
    end = df["end"].to_numpy(np.int64) if "end" in df.columns else start + a.window

    # 1) get the logit the calibrator was fit on (logit1 - logit0).
    #    Prefer an exact `logit` column from genome_inference --emit-logit; else invert the softmax
    #    prob (exact except where float32 saturated the prob to 0/1 -> eps clip compresses it).
    if have_logit:
        logit = df[a.logit_col].to_numpy(float)
        raw_prob = (df[a.score_col].to_numpy(float) if have_score
                    else 1.0 / (1.0 + np.exp(-logit)))          # for reporting only
        logit_source = f"'{a.logit_col}' column (exact)"
    else:
        p = np.clip(df[a.score_col].to_numpy(float), a.eps, 1.0 - a.eps)
        logit = np.log(p / (1.0 - p))
        raw_prob = p
        logit_source = f"inverted from '{a.score_col}' (clip eps={a.eps:g})"

    # 2) AT covariate — ONLY when the calibrator uses it (n_features_in_ == 2). Prefer the track's
    #    at_fraction column (genome_inference --emit-logit); else re-derive from the FASTA.
    at = None
    at_source = "not used (score-only calibrator)"
    if n_feat == 2:
        if a.at_col in df.columns:
            at = df[a.at_col].to_numpy(float)
            at_source = f"'{a.at_col}' column"
        elif a.genome:
            at = np.empty(len(df), dtype=float)
            genome = read_fasta(a.genome)
            clipped = 0
            for contig, g in df.groupby("contig", sort=False):     # each contig appears once
                cid = str(contig)
                if cid not in genome:
                    die(f"track contig '{cid}' not in FASTA (records: {list(genome)[:5]}...). "
                        f"Do the score track and FASTA use the same contig ids (e.g. NC_014328.1)?")
                cum = at_cumsum(genome[cid])
                L = len(cum) - 1
                idx = g.index.to_numpy()
                s = np.clip(start[idx], 0, L)
                e = np.clip(end[idx], 0, L)
                clipped += int((s != start[idx]).sum() + (e != end[idx]).sum())
                span = np.maximum(e - s, 1)
                at[idx] = (cum[e] - cum[s]) / span
            at_source = f"re-derived from FASTA {a.genome}"
            if clipped:
                print(f"NOTE: clipped {clipped} window bound(s) to contig length (near-contig-end windows).")
        else:
            die(f"this is a score+AT calibrator (n_features_in_=2) but the track has no '{a.at_col}' "
                f"column and no --genome was given. Either re-score with genome_inference --emit-logit "
                f"(writes at_fraction) or pass --genome to re-derive AT.")

    # 3) apply the calibrator (feature order = [logit, at]; only pass AT if the pipeline expects it)
    X = np.column_stack([logit, at]) if n_feat == 2 else logit.reshape(-1, 1)
    calib_prob = cal.predict_proba(X)[:, 1]

    # 4) write the track with promoter_score replaced (schema preserved for genome_fp_benchmark.py)
    out = df.copy()
    out["raw_promoter_score"] = raw_prob
    if at is not None and a.at_col not in out.columns:
        out[a.at_col] = at
    out[a.score_col] = calib_prob
    out.to_csv(a.out, index=False)

    print(f"calibrator   : {a.calibrator}  (n_features_in_={n_feat}: "
          f"{'score+AT' if n_feat == 2 else 'score-only'})")
    print(f"logit source : {logit_source}")
    print(f"AT source    : {at_source}")
    print(f"windows      : {len(df):,}   contigs: {sorted(set(df['contig'].astype(str)))}")
    print(f"raw  prob    : mean {raw_prob.mean():.4f}  min {raw_prob.min():.4f}  max {raw_prob.max():.4f}")
    print(f"calib prob   : mean {calib_prob.mean():.4f}  min {calib_prob.min():.4f}  max {calib_prob.max():.4f}")
    if at is not None:
        print(f"AT fraction  : mean {at.mean():.4f}  (window AT of scored positions)")
    if n_feat == 1:
        print("WARNING: this calibrator is score-only (monotonic) -> genome_fp_benchmark rankings, "
              "and therefore FP-peak counts, are UNCHANGED vs the raw track. Re-running the benchmark "
              "on this output is only useful if you need calibrated thresholds, not new FP numbers.")
    print(f"wrote        : {a.out}")


if __name__ == "__main__":
    main()
