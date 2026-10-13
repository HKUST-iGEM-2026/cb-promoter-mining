#!/usr/bin/env python3
"""
ntv3_genome_scan.py — restricted (or full) genome scan for the NTv3-ctx classifier.

Regime C scores candidate windows on benchmark genomes, then evaluates catalogue coverage
with pipeline/evaluation/genome_fp_benchmark.py. Both genomes supplied silver/negative
training windows; only the published benchmark TSS labels were withheld. NTv3 reads a 384-bp
context but reports 81-bp core coordinates. That makes its track compatible with --window 81,
not a matched-input model comparison or wholly independent evaluation.

Scan geometry mirrors genome_inference.py (forward start grid, --step, both strands,
forward coordinates). Per 81 bp window at forward [s, s+81] it carves the centered 384 bp
context genome[s-151 : s+233], N-padded at contig ends, and reverse-complements it on the
'-' strand — EXACTLY the framing build_context_windows.py / npad_context.py used in
training (core_offset 151 on '+', 152 on '-'). Output columns: contig,start,end,strand,
promoter_score,logit (genome_fp_benchmark defaults + the calibration logit).

  python ntv3_genome_scan.py --genome clj.fna --model-dir <...>/final_model \
      --regions clj_candidate_regions.bed --step 10 --out clj_ntv3_track.csv
  # self-check the coordinate/context math (no model, no GPU):
  python ntv3_genome_scan.py --self-test
"""
import argparse
import os

import pandas as pd

COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s):
    return s.translate(COMP)[::-1]


def load_regions(path):
    """BED (contig,start,end,strand) -> {(contig,strand): sorted [(start,end)]}."""
    d = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.replace(",", " ").split()
            if len(f) < 4:
                continue
            try:
                s, e = int(f[1]), int(f[2])
            except ValueError:
                continue                     # header row (contig start end strand) or malformed
            d.setdefault((f[0], f[3]), []).append((s, e))
    for k in d:
        d[k].sort()
    return d


def starts_for(L, window, step, ivals):
    """Forward starts s in [0, L-window], step-aligned, whose [s,s+window) overlaps a region.
    ivals=None -> full scan."""
    if ivals is None:
        return list(range(0, L - window + 1, step))
    hi = L - window
    out = set()
    for (a, b) in ivals:
        s0 = max(0, a - window + 1)
        s1 = min(hi, b - 1)
        if s1 < s0:
            continue
        s0 = ((s0 + step - 1) // step) * step        # snap up to the step grid
        out.update(range(s0, s1 + 1, step))
    return sorted(out)


def carve(seq, s, window, up, down):
    """Length up+window+down context around core [s,s+window); N-pad at contig ends."""
    a, b = s - up, s + window + down
    return ("N" * max(0, -a)) + seq[max(0, a):min(len(seq), b)] + ("N" * max(0, b - len(seq)))


def self_test():
    seq = "".join("ACGT"[i % 4] for i in range(1000))
    up, down = (384 - 81) // 2, 384 - 81 - (384 - 81) // 2
    assert (up, down) == (151, 152)
    # interior window: full 384, core exactly recovered at offset 151
    c = carve(seq, 300, 81, up, down)
    assert len(c) == 384 and c[151:232] == seq[300:381], "interior carve/centering wrong"
    # contig-start window: left N-padded, still 384, core intact
    c0 = carve(seq, 10, 81, up, down)
    assert len(c0) == 384 and c0[:141] == "N" * 141 and c0[151:232] == seq[10:91], "left N-pad wrong"
    # region-restricted starts overlap the interval and sit on the step grid
    st = starts_for(1000, 81, 10, [(400, 450)])
    assert all(s % 10 == 0 and s < 450 and s + 81 > 400 for s in st) and st, "region starts wrong"
    # minus strand context = revcomp of the forward carve (reading-strand, core at offset 152)
    cm = revcomp(carve(seq, 300, 81, up, down))
    assert len(cm) == 384 and cm[152:233] == revcomp(seq[300:381]), "minus-strand framing wrong"
    print("SELF-TEST PASS (centering, N-pad, region grid, minus-strand framing)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome")
    ap.add_argument("--model-dir", help="<out>/final_model")
    ap.add_argument("--out")
    ap.add_argument("--regions", default=None,
                    help="candidate-region BED (contig,start,end,strand) -> restricted scan")
    ap.add_argument("--window", type=int, default=81)
    ap.add_argument("--context", type=int, default=384, help="NTv3 context bp (multiple of 128)")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--strands", default="+,-")
    ap.add_argument("--contigs", default=None, help="comma list; keep only these record ids")
    ap.add_argument("--contig-name", default=None, help="override emitted contig id (single-record FASTA)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    for req in ("genome", "model_dir", "out"):
        if getattr(a, req) is None:
            raise SystemExit(f"--{req.replace('_','-')} is required (or use --self-test)")
    if a.context % 128:
        raise SystemExit("--context must be a multiple of 128")
    up = (a.context - a.window) // 2
    down = a.context - a.window - up
    strands = a.strands.split(",")
    regions = load_regions(a.regions) if a.regions else None

    from Bio import SeqIO
    import ntv3_infer
    tok, backbone, head, meta, device = ntv3_infer.load_ntv3(a.model_dir)
    if meta["max_length"] != a.context:
        print(f"WARNING: model max_length={meta['max_length']} != --context {a.context}")
    print(f"NTv3 on {device} | window={a.window} context={a.context} step={a.step} "
          f"{'restricted' if regions else 'FULL'} scan")

    recs = list(SeqIO.parse(a.genome, "fasta"))
    if a.contigs:
        keep = set(a.contigs.split(","))
        recs = [r for r in recs if r.id in keep]
    if a.contig_name:
        if len(recs) != 1:
            raise SystemExit("--contig-name requires a single-record FASTA")
        recs[0].id = a.contig_name
    if not recs:
        raise SystemExit("no records to scan (check --genome / --contigs)")

    outdir = os.path.dirname(os.path.abspath(a.out))
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    wrote_header, total = False, 0
    for rec in recs:
        seq = str(rec.seq).upper()
        L = len(seq)
        for strand in strands:
            ivals = regions.get((rec.id, strand)) if regions else None
            if regions and not ivals:
                continue
            starts = starts_for(L, a.window, a.step, ivals)
            for i in range(0, len(starts), a.batch_size):
                chunk = starts[i:i + a.batch_size]
                ctxs = [carve(seq, s, a.window, up, down) for s in chunk]
                if strand == "-":
                    ctxs = [revcomp(c) for c in ctxs]
                prob, logit = ntv3_infer.score_seqs(ctxs, tok, backbone, head, meta,
                                                    device, batch_size=len(ctxs))
                pd.DataFrame({"contig": rec.id, "start": chunk,
                              "end": [s + a.window for s in chunk], "strand": strand,
                              "promoter_score": prob, "logit": logit}
                             ).to_csv(a.out, mode="a", header=not wrote_header, index=False)
                wrote_header = True
                total += len(chunk)
        print(f"  {rec.id} ({L:,} bp): {total:,} windows scanned")
    print(f"wrote {total:,} windows -> {a.out}")


if __name__ == "__main__":
    main()
