#!/usr/bin/env python3
"""
extract_embeddings_evo2.py — mean-pool Evo2 embeddings for the T15 splits, for a frozen-embedding
sklearn probe (evo2_promoter_classifier.py). NOT fine-tuning: the 7B backbone is frozen.

WHY THIS FILE CHANGED (2026-08): the committed version wrote to a PERSONAL dir
($PROMOTER_RESULTS/evo2_embeddings) while the classifier read from the SHARED
$PROMOTER_REPO/evo2_embeddings — so the two never lined up, and it dropped the
T15 `weight` column that every other model (T18/T19/T20) trains on. Both are fixed here; paths are
now flags with shared defaults, and `weight` is carried through so the probe trains on the SAME
distribution as the ensemble.

Reads {train,val,test}.parquet with columns: `sequence` (81 bp core) or `sequence_context` (long
window), `label`, and `weight`. Emits {split}_{X,y,w}.npy to --out-dir.

  # 81 bp core (apples-to-apples with ProkBERT/CNN/NT — the default):
  python extract_embeddings_evo2.py --data-dir <T15_ctx> --out-dir <shared>/evo2_embeddings
  # or exploit Evo2's long context:
  python extract_embeddings_evo2.py --sequence-col sequence_context ...
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evo2_common import embed_sequences

# Portable paths: override with PROMOTER_REPO / PROMOTER_RESULTS (or `source env.sh`).
# The fallbacks resolve from this file, so a plain clone works with no configuration.
PROMOTER_REPO = os.environ.get("PROMOTER_REPO") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PROMOTER_RESULTS = os.environ.get("PROMOTER_RESULTS") or os.path.join(
    PROMOTER_REPO, "infrastructure", "T3", "snakemake_scaffold", "results")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",
                    default=os.path.join(PROMOTER_RESULTS, "T15_xs"),
                    help="dir with {train,val,test}.parquet. Default T15_xs (44,085 rows) = the "
                         "cross-species-augmented split the DEPLOYED T21_xs models trained on, so the probe "
                         "is comparable to the deployed ensemble. (NOT T15_ctx=stale, T15=0.6 non-xs, T15_09=0.9.)")
    ap.add_argument("--out-dir", default=os.path.join(PROMOTER_REPO, "evo2_embeddings"),
                    help="SHARED output dir the classifier reads (must match evo2_promoter_classifier.py --emb-dir).")
    ap.add_argument("--model", default="evo2_7b")
    ap.add_argument("--layer", default="blocks.28.mlp.l3",
                    help="7B intermediate layer (upstream default). Re-index if you change model size.")
    ap.add_argument("--sequence-col", default="sequence",
                    help="'sequence' = 81 bp core (comparable to the other 4 models); "
                         "'sequence_context' = long window (uses Evo2's long-context strength).")
    ap.add_argument("--pool", default="mean", choices=["mean", "last", "max"],
                    help="pool per-nucleotide embeddings to one vector. 'last' = final position "
                         "(has seen the whole causal context); worth benchmarking vs mean.")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="windows per Evo2 forward (auto-falls back to 1 if Vortex rejects B>1).")
    a = ap.parse_args()

    from evo2 import Evo2
    evo2_model = Evo2(a.model)
    os.makedirs(a.out_dir, exist_ok=True)

    for split in ("train", "val", "test"):
        df = pd.read_parquet(f"{a.data_dir}/{split}.parquet")
        if a.sequence_col not in df.columns:
            raise SystemExit(f"{split}: no column '{a.sequence_col}'. Have: {list(df.columns)}")
        seqs = df[a.sequence_col].astype(str).tolist()
        y = df["label"].to_numpy()
        if "weight" in df.columns:
            w = df["weight"].to_numpy()
        else:
            print(f"  WARNING: no 'weight' column in {split} — probe will train UNWEIGHTED "
                  f"(silver T13 at full weight = AT-trap risk; not comparable to T18/T19/T20).")
            w = np.ones(len(y))

        print(f"\n{split}: {len(seqs)} seqs, col='{a.sequence_col}', pool='{a.pool}', batch={a.batch_size}")
        X = embed_sequences(evo2_model, seqs, a.layer, a.pool, a.batch_size)

        np.save(f"{a.out_dir}/{split}_X.npy", X)
        np.save(f"{a.out_dir}/{split}_y.npy", y)
        np.save(f"{a.out_dir}/{split}_w.npy", w)
        print(f"  wrote {split}_{{X,y,w}}.npy  X={X.shape}  pos={int(y.sum())}/{len(y)}")

    print(f"\nDone -> {a.out_dir}  (next: evo2_promoter_classifier.py --emb-dir {a.out_dir})")


if __name__ == "__main__":
    main()
