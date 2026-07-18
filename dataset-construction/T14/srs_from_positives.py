"""
SRS-only negatives from a positives TSV (e.g. T9 literature promoters).

Companion to negative_set_generation.py. The full T14 script builds four negative classes, but three
of them (CDS / Intergenic / Downstream) sample from the genome and need each row's genome/contig/strand
to open the Bakta files. T9 literature promoters have no genomic coordinates (and many are out-of-genus),
so only the SRS class is derivable from them.

SRS = Shuffled Real Sequence: a dinucleotide-preserving (Eulerian) shuffle of a real promoter — keeps
base/dinucleotide composition (so AT-content is matched), destroys the -10/-35 motif. This is exactly
the negative class the T26 calibrators consume (--neg-classes SRS), so the output is drop-in for
calibrate_cnn.py / nt_calib.py --negatives.

Usage:
    python srs_from_positives.py \
        --positives-tsv ../../data-acquisition-and-curation/T9/literature_promoters.tsv \
        --out-tsv out/srs_from_t9.tsv
"""

import os
import random
import argparse
import collections
import pandas as pd

# reuse T14's exact SRS algorithm so the SRS definition is identical project-wide
from negative_set_generation import dinucleotide_shuffle


def main():
    parser = argparse.ArgumentParser(description="Generate SRS-only negatives from a positives TSV")
    parser.add_argument("--positives-tsv", required=True,
                        help="Positives TSV with a sequence column (e.g. T9 literature_promoters.tsv)")
    parser.add_argument("--out-tsv", required=True, help="Path to save the SRS negative_set.tsv")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for the reproducible dinucleotide shuffle")
    args = parser.parse_args()

    random.seed(args.seed)

    pos_df = pd.read_csv(args.positives_tsv, sep="\t")
    seq_col = next((c for c in ("sequence", "Sequence", "seq") if c in pos_df.columns), None)
    if seq_col is None:
        raise SystemExit(f"{args.positives_tsv} has no sequence column "
                         f"(looked for sequence/Sequence/seq); got {list(pos_df.columns)}")

    src_name = os.path.basename(args.positives_tsv)
    records = []
    for _, row in pos_df.iterrows():
        seq = str(row[seq_col]).upper()
        records.append({
            "sequence": dinucleotide_shuffle(seq),
            "negative_class": "SRS",
            "label": 0,
            "source": src_name,
        })

    out_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.out_tsv) or ".", exist_ok=True)
    out_df.to_csv(args.out_tsv, sep="\t", index=False)

    print(f"Shuffled {len(out_df)} SRS negatives from {seq_col!r} in {src_name}")
    print(f"Wrote {args.out_tsv}")


if __name__ == "__main__":
    main()
