#!/usr/bin/env python3
"""
ntv3_rescore_clean.py — NTv3-ctx balanced AUPRC on val/test, FULL vs de-leaked.

The context-leak check showed ~39% of val/test rows have >=100 bp of their 384 bp
context verbatim in a training row (positives ~45%), so the run's balanced AUPRC is
optimistic. This scores the trained model and recomputes AUPRC on the subset of rows
whose context recurrence with train is below a cutoff — an honest, de-leaked number
(and, since low-recurrence rows are genuinely unseen context, a fair one).

The per-row recurrence comes from context_leakage_check.py's CSV. Rows are aligned to
the parquets POSITIONALLY within each split (that CSV is written as concat(val,test) in
parquet order) and the alignment is verified against the `label` column — it aborts on
any mismatch, so a silent reorder cannot corrupt the join.

  python ntv3_rescore_clean.py \
     --model-dir <...>/results/ntv3_xs_ctx_classifier/final_model \
     --data-dir  <...>/results/T15_xs_ctx \
     --leak-csv  ~/T15_xs_ctx_leak.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import ntv3_infer


def block(df, split):
    print(f"\n[{split}]  (context-recurrence cutoffs on the SAME scored rows)")
    print(f"    {'subset':<12}{'n':>7}{'pos':>7}{'prev':>8}{'wAUPRC':>9}{'uAUPRC':>9}{'AUROC':>8}")
    base = None
    for name, mask in (("full", df.kmer_shared_bp >= 0),
                       ("<150bp", df.kmer_shared_bp < 150),
                       ("<100bp", df.kmer_shared_bp < 100),
                       ("<50bp", df.kmer_shared_bp < 50)):
        s = df[mask]
        n, pos = len(s), int(s.label.sum())
        if s.label.nunique() < 2:
            print(f"    {name:<12}{n:>7}{pos:>7}{'--':>8}   (one class — skip)")
            continue
        y, p, w = s.label.values, s.prob.values, s.weight.values
        wap = average_precision_score(y, p, sample_weight=w)
        uap = average_precision_score(y, p)
        auc = roc_auc_score(y, p)
        tag = "  <- full" if name == "full" else ""
        print(f"    {name:<12}{n:>7}{pos:>7}{pos/n:>8.1%}{wap:>9.4f}{uap:>9.4f}{auc:>8.4f}{tag}")
        if name == "full":
            base = wap
        elif base is not None and name == "<100bp":
            print(f"    -> de-leaked (<100bp) weighted AUPRC {wap:.4f} vs full {base:.4f} "
                  f"= {wap - base:+.4f}  (negative => the full number was inflated)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="<out>/final_model")
    ap.add_argument("--data-dir", required=True, help="T15_xs_ctx dir")
    ap.add_argument("--leak-csv", required=True, help="context_leakage_check.py output CSV")
    ap.add_argument("--seq-col", default="sequence_context")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="ntv3_xs_ctx_clean_rescore.csv")
    a = ap.parse_args()

    leak = pd.read_csv(a.leak_csv)
    if "kmer_shared_bp" not in leak.columns:
        raise SystemExit(f"{a.leak_csv} has no kmer_shared_bp column (cols {list(leak.columns)})")
    tok, backbone, head, meta, device = ntv3_infer.load_ntv3(a.model_dir)
    print(f"loaded NTv3 on {device} | max_length={meta['max_length']} seq_col={a.seq_col}")

    allrows = []
    for split in ("val", "test"):
        d = pd.read_parquet(os.path.join(a.data_dir, f"{split}.parquet")).reset_index(drop=True)
        if "weight" not in d.columns:
            d["weight"] = 1.0
        lk = leak[leak.split == split].reset_index(drop=True)
        if len(lk) != len(d):
            raise SystemExit(f"{split}: leak CSV has {len(lk)} rows but parquet has {len(d)} — "
                             f"regenerate the CSV from THIS T15_xs_ctx.")
        if not (lk["label"].to_numpy() == d["label"].to_numpy()).all():
            raise SystemExit(f"{split}: label order mismatch between leak CSV and parquet — "
                             f"positional join unsafe, aborting.")
        prob, _ = ntv3_infer.score_seqs(d[a.seq_col].tolist(), tok, backbone, head, meta,
                                        device, a.batch_size)
        d = d.assign(prob=prob, kmer_shared_bp=lk["kmer_shared_bp"].to_numpy(),
                     source=lk.get("source", "?"), split=split)
        block(d, split)
        allrows.append(d[["split", "label", "source", "weight", "prob", "kmer_shared_bp"]])

    pd.concat(allrows, ignore_index=True).to_csv(a.out, index=False)
    print(f"\nwrote per-row scores -> {a.out}")
    print("READ: compare the <100bp (de-leaked) weighted AUPRC to full. The gap is how much "
          "context leakage inflated the run's number. The genome-scale Regime C scan is the "
          "independent, leak-immune verdict.")


if __name__ == "__main__":
    main()
