#!/usr/bin/env python3
"""
T9-split (CLAUDE.md §7 option b): make a portion of T9's VERIFIED in-genus C. acetobutylicum
promoters available to TRAINING (T15) instead of holding out 100% for evaluation. The split is
FAMILY-AWARE and seeded, so near-identical promoter-library variants don't straddle train/holdout
(a first-pass leakage guard; T15's CD-HIT step does the rigorous cluster-aware split downstream).

  in-genus (C. acetobutylicum)               -> split: train-frac of FAMILIES -> train, rest -> holdout
  out-of-genus (C. phytofermentans -> Lachno) -> ALWAYS holdout (kept for cross-species eval only)

Outputs:
  --train-out   parquet, T15-positive schema (species, promoter_id, sequence, strand, sigma_factor,
                source_db, in_genus, label) — concatenated with the T8/T7b positives in T15.
  --holdout-out tsv, full T9 schema — the cross-species external eval set for T24/T26.

NB: enabling this means T24/T26 must evaluate on the HOLDOUT file, not the full literature_promoters.tsv,
or the model would be tested on promoters it trained on.
"""
import argparse
import csv
import os
import random

OUT_COLS = ["species", "promoter_id", "sequence", "strand",
            "sigma_factor", "source_db", "in_genus", "label"]


def family(name):
    # group library variants so they don't straddle the split:
    #   "100-1" -> "100"   |   "pPM36-79%-18" -> "pPM36-79%"
    return name.rsplit("-", 1)[0] if "-" in name else name


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default=os.path.join(here, "literature_promoters.tsv"))
    ap.add_argument("--train-frac", type=float, default=0.5, help="fraction of in-genus FAMILIES -> train")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-out", required=True, help="parquet of in-genus training positives (-> T15)")
    ap.add_argument("--holdout-out", required=True, help="tsv of held-out promoters (-> T24/T26 eval)")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.inp), delimiter="\t"))
    if not rows:
        raise SystemExit(f"empty / missing T9 file: {a.inp}")
    ingenus = [r for r in rows if str(r["in_genus"]).strip().lower() == "true"]
    outgenus = [r for r in rows if str(r["in_genus"]).strip().lower() != "true"]

    fams = sorted({family(r["original_name"]) for r in ingenus})
    rnd = random.Random(a.seed)
    rnd.shuffle(fams)
    n_train_fams = int(round(len(fams) * a.train_frac))
    train_fams = set(fams[:n_train_fams])
    train = [r for r in ingenus if family(r["original_name"]) in train_fams]
    holdout_ingenus = [r for r in ingenus if family(r["original_name"]) not in train_fams]

    # train parquet — harmonized T15-positive schema (strand unknown for synthetic promoters)
    import pandas as pd
    tr = pd.DataFrame([{
        "species": r["species"],
        "promoter_id": f'T9_{r["promoter_id"]}',
        "sequence": r["sequence"],
        "strand": ".",
        "sigma_factor": "Unknown",
        "source_db": f'T9_{r["source"]}',
        "in_genus": True,
        "label": 1,
    } for r in train], columns=OUT_COLS)
    os.makedirs(os.path.dirname(os.path.abspath(a.train_out)), exist_ok=True)
    tr.to_parquet(a.train_out, index=False)

    # holdout tsv — full T9 schema: in-genus remainder + all out-of-genus
    holdout = holdout_ingenus + outgenus
    os.makedirs(os.path.dirname(os.path.abspath(a.holdout_out)), exist_ok=True)
    with open(a.holdout_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(holdout)

    # leakage guard: no family in both sets
    assert not (train_fams & {family(r["original_name"]) for r in holdout_ingenus}), \
        "family leak across train/holdout"
    print(f"[T9-split] seed={a.seed} frac={a.train_frac}: {len(ingenus)} in-genus -> "
          f"{len(train)} train ({len(train_fams)}/{len(fams)} families) / {len(holdout_ingenus)} holdout; "
          f"+{len(outgenus)} out-of-genus -> holdout. "
          f"train={os.path.basename(a.train_out)} holdout={os.path.basename(a.holdout_out)}")


if __name__ == "__main__":
    main()
