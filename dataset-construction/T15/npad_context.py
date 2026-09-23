#!/usr/bin/env python3
"""
npad_context.py — give an 81 bp-core source parquet an N-padded `sequence_context`.

For the CROSS-SPECIES, out-of-genus sources (T8 E.coli/B.subtilis, B.licheniformis,
Ruminiclostridium) that have no coordinates in *our* genomes, we cannot carve a real
384 bp flank. They are train-pinned (out-of-genus never enters val/test), so an
N-padded context is the safe, consistent choice — it is exactly how T8 is already
handled, and since these rows never appear in val/test it cannot inflate the eval
metric (an N-flank shortcut would only be learnable in train, where it would hurt,
not help, generalisation to the real-flank val/test).

Each core (81 bp) is centred in a length-L window of Ns:
    sequence_context = 'N'*up + core + 'N'*down,  up=(L-81)//2,  core_offset=up
`sequence` is uppercased and asserted 81 bp so it stays in sync with the context
(cdhit_splits re-windows the core with anchor='center', a no-op at 81). All original
columns are preserved so cdhit_splits.load_source still detects the source schema.

  python npad_context.py --inputs A.parquet B.parquet --out-dir <dir> [--length 384]
"""
import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="81 bp-core source parquets (must have a `sequence` column)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--length", type=int, default=384,
                    help="context window bp; MUST be a multiple of 128 for NTv3 (default 384)")
    ap.add_argument("--core", type=int, default=81, help="core length (default 81)")
    a = ap.parse_args()
    if a.length % 128:
        raise SystemExit(f"--length {a.length} must be a multiple of 128 for NTv3")
    if a.core > a.length:
        raise SystemExit("--core cannot exceed --length")
    up = (a.length - a.core) // 2
    down = a.length - a.core - up
    os.makedirs(a.out_dir, exist_ok=True)

    for p in a.inputs:
        d = pd.read_parquet(p)
        if "sequence" not in d.columns:
            raise SystemExit(f"{p}: no `sequence` column (have {list(d.columns)})")
        s = d["sequence"].astype(str).str.upper()
        bad = int((s.str.len() != a.core).sum())
        if bad:
            raise SystemExit(f"{p}: {bad}/{len(d)} rows are not {a.core} bp — "
                             f"re-window to {a.core} before N-padding")
        d = d.copy()
        d["sequence"] = s
        d["sequence_context"] = "N" * up + s + "N" * down
        d["core_offset"] = up
        d["has_genomic_context"] = False
        # sanity: every context is exactly L and holds the core at core_offset
        assert (d["sequence_context"].str.len() == a.length).all()
        assert (d.apply(lambda r: r["sequence_context"][up:up + a.core] == r["sequence"],
                        axis=1)).all()
        out = os.path.join(a.out_dir,
                           os.path.basename(p).replace(".parquet", "_ctx.parquet"))
        d.to_parquet(out, index=False)
        print(f"{os.path.basename(p):40s} {len(d):>6} rows -> {out}  "
              f"(ctx={a.length}, core_offset={up})")


if __name__ == "__main__":
    main()
