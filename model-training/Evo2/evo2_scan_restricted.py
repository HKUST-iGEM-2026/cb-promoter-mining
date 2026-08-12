#!/usr/bin/env python3
"""
evo2_scan_restricted.py — the DEPLOYABLE Evo2 promoter caller. Scores ONLY the windows inside candidate
promoter regions (make_candidate_regions.py BED) with frozen Evo2 + the trained sklearn probe, and emits
a track compatible with genome_fp_benchmark.py / peak-calling.

Full-genome step-1 Evo2 is infeasible (~8.7M windows through a 7B model). The restricted candidate space
is ~1.4 Mb; at --step 10 that is ~280k windows — a feasible GPU job. Promoter signal (-10/-35) spans
~35 bp, so step 5-10 loses nothing for peak-calling and is 5-10x cheaper than step 1.

Windows are the reading-strand w-mers whose forward span overlaps a candidate region (matching
genome_fp_benchmark's --regions restrict, so validation numbers line up with the other four models).
Forward-genome coordinates are reported, exactly like genome_inference.py.

The pooled embedding MUST use the SAME --layer/--pool/--window the probe was trained on, or the features
are out of distribution and the scores are meaningless.

  python evo2_scan_restricted.py --genome cau.fna --regions cau_candidate_regions.bed \
      --probe evo2_promoter_classifier.pkl --step 10 --batch-size 16 --out cau_evo2_track.csv
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evo2_common import embed_sequences, revcomp


def load_fasta(path):
    seqs, cid, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if cid is not None:
                    seqs[cid] = "".join(buf).upper()
                cid = line[1:].strip().split()[0]; buf = []
            else:
                buf.append(line.strip())
        if cid is not None:
            seqs[cid] = "".join(buf).upper()
    return seqs


def load_regions(path):
    df = pd.read_csv(path, sep="\t")
    out = {}
    for (c, st), g in df.groupby(["contig", "strand"]):
        g = g.sort_values("start")
        out[(str(c), str(st))] = list(zip(g["start"].astype(int), g["end"].astype(int)))
    return out


def region_starts(intervals, window, step, contig_len):
    """All window starts whose forward span [s, s+window) OVERLAPS some region [rs, re) and fits the
    contig — i.e. s in [rs-window+1, re-1] — matching genome_fp_benchmark's in_region_mask."""
    starts = set()
    for rs, re in intervals:
        lo = max(0, rs - window + 1)
        hi = min(contig_len - window, re - 1)
        s = lo
        while s <= hi:
            starts.add(s)
            s += step
    return sorted(starts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", required=True)
    ap.add_argument("--regions", required=True, help="candidate-regions BED (contig,start,end,strand)")
    ap.add_argument("--probe", required=True, help="trained sklearn probe (.pkl from evo2_promoter_classifier.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="evo2_7b")
    ap.add_argument("--layer", default="blocks.28.mlp.l3", help="MUST match the probe's extractor")
    ap.add_argument("--pool", default="mean", choices=["mean", "last", "max"], help="MUST match the probe's extractor")
    ap.add_argument("--window", type=int, default=81, help="MUST match the probe's training window")
    ap.add_argument("--step", type=int, default=10, help="stride; 5-10 is plenty for peak-calling")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--strands", default="+,-")
    a = ap.parse_args()

    from evo2 import Evo2
    evo2_model = Evo2(a.model)
    probe = joblib.load(a.probe)
    genome = load_fasta(a.genome)
    regions = load_regions(a.regions)
    strands = [s for s in a.strands.split(",") if s]

    total, header = 0, True
    with open(a.out, "w") as fh:
        for contig, seq in genome.items():
            L = len(seq)
            for strand in strands:
                ivs = regions.get((contig, strand))
                if not ivs:
                    continue
                starts = region_starts(ivs, a.window, a.step, L)
                if not starts:
                    continue
                wins = [seq[s:s + a.window] if strand == "+" else revcomp(seq[s:s + a.window]) for s in starts]
                X = embed_sequences(evo2_model, wins, a.layer, a.pool, a.batch_size)
                scores = probe.predict_proba(X)[:, 1]
                pd.DataFrame({
                    "contig": contig,
                    "start": starts,
                    "end": [s + a.window for s in starts],
                    "strand": strand,
                    "promoter_score": scores,
                }).to_csv(fh, header=header, index=False)
                header = False
                total += len(starts)
                print(f"  {contig} {strand}: {len(starts):,} windows scored", flush=True)
    print(f"Wrote {total:,} scored windows -> {a.out}")
    print(f"Benchmark it:  python evaluation-and-calibration/genome_fp_benchmark.py "
          f"--scores {a.out} --truth <truth.tsv> --regions {a.regions} --recall-targets 0.1,0.5,0.7,0.9 --out <dir>")


if __name__ == "__main__":
    main()
