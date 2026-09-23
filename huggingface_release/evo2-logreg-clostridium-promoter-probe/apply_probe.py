#!/usr/bin/env python3
"""
apply_probe.py — score DNA windows with the frozen-Evo2-7B + logistic-regression promoter probe.

Pipeline: sequences -> mean-pooled Evo2-7B `blocks.28.mlp.l3` embeddings (4096-d) -> probe -> P(promoter).
The embedding recipe MUST match how the probe was trained (layer / pool / 81 bp window), so this reuses
the project's `evo2_common.embed_sequences`.

Requirements: the `evo2` package + a GPU (see github.com/ArcInstitute/evo2), scikit-learn **1.9.0**
(match the version the probe was pickled with), joblib, numpy. `evo2_common.py` must be alongside this
file (it is, in this repo).

Usage:
  # FASTA of 81 bp windows (TSS framed at 0-based offset 60):
  python apply_probe.py --fasta windows.fna --out scores.csv
  # or one sequence per line:
  python apply_probe.py --seqs windows.txt --out scores.csv
  # point at a different probe / layer if you re-trained:
  python apply_probe.py --fasta windows.fna --probe evo2_promoter_classifier.pkl --layer blocks.28.mlp.l3
"""
import argparse
import csv
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evo2_common import embed_sequences  # noqa: E402


def read_fasta(path):
    ids, seqs, cur, buf = [], [], None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if cur is not None:
                    ids.append(cur); seqs.append("".join(buf))
                cur = line[1:].split()[0]; buf = []
            else:
                buf.append(line.strip())
        if cur is not None:
            ids.append(cur); seqs.append("".join(buf))
    return ids, seqs


def read_lines(path):
    seqs = [l.strip() for l in open(path) if l.strip()]
    return [f"seq_{i}" for i in range(len(seqs))], seqs


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fasta", help="FASTA of DNA windows")
    src.add_argument("--seqs", help="text file, one sequence per line")
    ap.add_argument("--probe", default="evo2_promoter_classifier.pkl",
                    help="joblib-pickled scikit-learn Pipeline (StandardScaler -> LogisticRegression)")
    ap.add_argument("--model", default="evo2_7b", help="Evo2 model name")
    ap.add_argument("--layer", default="blocks.28.mlp.l3",
                    help="Evo2 layer to pool (must match how the probe was trained)")
    ap.add_argument("--pool", default="mean", choices=["mean", "last", "max"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="scores.csv")
    a = ap.parse_args()

    ids, seqs = read_fasta(a.fasta) if a.fasta else read_lines(a.seqs)
    lengths = {len(s) for s in seqs}
    if lengths != {81}:
        print(f"  NOTE: probe was trained on 81 bp windows; got lengths {sorted(lengths)}. "
              f"Scores are only comparable to training if windows are 81 bp with the TSS at offset 60.",
              file=sys.stderr)

    print(f"Loading Evo2 '{a.model}' (frozen) ...", flush=True)
    from evo2 import Evo2
    evo2_model = Evo2(a.model)

    print(f"Embedding {len(seqs)} windows @ {a.layer} ({a.pool}-pool) ...", flush=True)
    X = embed_sequences(evo2_model, seqs, layer=a.layer, pool=a.pool, batch_size=a.batch_size)  # (N, 4096)

    pipe = joblib.load(a.probe)
    proba = pipe.predict_proba(X)[:, 1]

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "p_promoter"])
        for i, p in zip(ids, proba):
            w.writerow([i, f"{p:.6f}"])
    print(f"wrote {len(proba)} scores -> {a.out}  (mean P={float(np.mean(proba)):.3f})")


if __name__ == "__main__":
    main()
