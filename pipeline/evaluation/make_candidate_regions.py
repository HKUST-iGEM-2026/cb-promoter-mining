#!/usr/bin/env python3
"""
make_candidate_regions.py — build the restrict-scan search space from a genome annotation.

Genome-scale scanning scores every window, but real promoters live in a small fraction of the
genome (upstream of genes / intergenic), so most scored windows are deep coding sequence that can
only add false positives. This emits a BED of candidate promoter regions -- the region just
upstream of each gene start, on the gene's strand -- so genome_fp_benchmark.py --regions can
restrict to them.

Input:  a GFF3 (gene/CDS features). If the annotation's seqid differs from the score-track contig
        (e.g. Bakta renames the sole contig to `contig_1` while the raw FASTA is `NC_014328.1`),
        pass --contig to override the emitted contig for a single-replicon genome.
Output: BED-like TSV: contig, start, end, strand   (0-based, end-exclusive; merged per strand).

  python make_candidate_regions.py --gff clj.gff3 --contig NC_014328.1 \
      --up 300 --down 50 --out clj_candidate_regions.bed
"""
import argparse


def parse_gff_genes(path):
    """Yield (seqid, start1, end1, strand) for gene/CDS features (1-based inclusive from GFF)."""
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 8:
                continue
            if p[2] not in ("gene", "CDS"):
                continue
            yield p[0], int(p[3]), int(p[4]), p[6]


def merge(intervals):
    """Merge overlapping [start,end) intervals (already same contig+strand)."""
    intervals = sorted(intervals)
    out = []
    for s, e in intervals:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gff", required=True, help="GFF3 with gene/CDS features")
    ap.add_argument("--out", required=True, help="output BED-like TSV (contig,start,end,strand)")
    ap.add_argument("--up", type=int, default=300, help="bp upstream of the gene start to include")
    ap.add_argument("--down", type=int, default=50, help="bp into the gene (past the start) to include")
    ap.add_argument("--contig", default=None, help="override the emitted contig id (single-replicon genomes)")
    ap.add_argument("--feature-prefer-gene", action="store_true",
                    help="if both gene and CDS present, this still dedups by interval so it's harmless")
    a = ap.parse_args()

    # collect candidate windows per (contig, strand), dedup by interval to avoid gene+CDS double count
    by_key = {}
    n_genes = 0
    for seqid, s1, e1, strand in parse_gff_genes(a.gff):
        n_genes += 1
        # gene START is 5' end: s1 on '+', e1 on '-'
        if strand == "+":
            start0, end0 = (s1 - 1) - a.up, (s1 - 1) + a.down       # 0-based [start-up, start+down)
        elif strand == "-":
            start0, end0 = e1 - a.down, e1 + a.up                    # 5' end is e1 (1-based) -> 0-based e1-1+1
        else:
            continue
        start0 = max(0, start0)
        contig = a.contig if a.contig else seqid
        by_key.setdefault((contig, strand), set()).add((start0, end0))

    rows = []
    for (contig, strand), ivs in by_key.items():
        for s, e in merge([list(x) for x in ivs]):
            rows.append((contig, s, e, strand))
    rows.sort()

    with open(a.out, "w") as f:
        f.write("contig\tstart\tend\tstrand\n")
        for contig, s, e, strand in rows:
            f.write(f"{contig}\t{s}\t{e}\t{strand}\n")

    total_bp = sum(e - s for _, s, e, _ in rows)
    print(f"{n_genes} gene/CDS features -> {len(rows)} merged candidate regions -> {a.out}")
    print(f"  strands: " + ", ".join(f"{k[1]}:{sum(1 for r in rows if r[3]==k[1])}" for k in sorted(by_key)))
    print(f"  total candidate span: {total_bp:,} bp")


if __name__ == "__main__":
    main()
