"""
T12: Operon detection
Owner: Seonyu
Due: 15 Jun

Clusters the supplied housekeeping loci by proximity and retains one putative
leader per cluster for upstream extraction. This is not a whole-genome operon
map or evidence of cotranscription; internal genes can have their own promoters.
The contig-end filter avoids truncated upstream windows.

Design:
  * Proximity clustering: Groups input housekeeping loci by genome, contig,
    and strand. Adjacent sorted loci with next_start - previous_stop <= max_dist
    (default 100) share a cluster. Only its most-upstream input locus is retained:
    lowest start on '+' and highest stop on '-'. These are inferred leaders,
    not experimentally established transcript boundaries.
  * Contig-End Exclusion: Because target assemblies are often incomplete, operon 
    leaders whose upstream window (default 300 bp) spills over the start (+ strand) 
    or end (- strand) of a contig are dropped.
  * Contig lengths are parsed directly from the `##sequence-region` headers of 
    the T6 `.gff3` files to accurately calculate distances for the - strand.

Usage:
    python pipeline/datasets/operon-detection/operon_detection.py \
        --hk-tsv <path_to_T11_housekeeping_genes.tsv> \
        --bakta-dir <dir_containing_T6_gff3_files> \
        --out-tsv <path_to_output_operon_leaders.tsv> \
        --max-dist 100 \
        --window 300
"""

import os
import glob
import argparse
import pandas as pd

def get_contig_lengths(gff_file):
    """Extract contig lengths from Bakta GFF3 ##sequence-region headers."""
    lengths = {}
    if not os.path.exists(gff_file):
        return lengths
        
    with open(gff_file, 'r') as f:
        for line in f:
            if line.startswith("##sequence-region"):
                parts = line.strip().split()
                if len(parts) >= 4:
                    lengths[parts[1]] = int(parts[3])
            elif not line.startswith("#"):
                break # Stop reading once we hit the feature annotations
    return lengths

def main():
    parser = argparse.ArgumentParser(description="T12: Operon detection and contig-end exclusion")
    parser.add_argument("--hk-tsv", required=True, help="Path to T11 housekeeping_genes.tsv")
    parser.add_argument("--bakta-dir", required=True, help="Directory containing T6 Bakta .gff3 files")
    parser.add_argument("--out-tsv", required=True, help="Path to output operon_leaders.tsv")
    parser.add_argument("--max-dist", type=int, default=100, help="Max distance (bp) to consider genes in the same operon")
    parser.add_argument("--window", type=int, default=300, help="Upstream window size (bp) required for exclusion filter")
    args = parser.parse_args()

    # 1. Load the housekeeping gene set
    df = pd.read_csv(args.hk_tsv, sep='\t')
    
    leaders = []

    # 2. Process each genome individually to handle operon collapsing and contig lengths
    for genome, group in df.groupby('genome'):
        # Parse contig lengths for this specific genome. Bakta writes a nested
        # per-genome dir: <bakta-dir>/<genome>_genomic_output/<genome>_genomic.gff3
        gff_path = os.path.join(args.bakta_dir, f"{genome}_genomic_output",
                                f"{genome}_genomic.gff3")
        contig_lengths = get_contig_lengths(gff_path)
        if not contig_lengths:
            print(f"Warning: no contig lengths for {genome} at {gff_path}; "
                  f"minus-strand contig-end exclusion will be skipped for it.")

        # 3. Build operon clusters within each (contig, strand), then keep the
        #    most-upstream gene per cluster. Genes are clustered by genomic gap
        #    (strand-independent); the LEADER is strand-aware:
        #      '+' -> lowest-coordinate gene (min start)
        #      '-' -> highest-coordinate gene (max stop), since transcription runs high->low
        for (contig, strand), sub in group.groupby(['contig', 'strand'], sort=False):
            sub = sub.sort_values('start').reset_index(drop=True)
            clusters, cluster, prev_stop = [], [], None
            for _, row in sub.iterrows():
                if prev_stop is not None and (row['start'] - prev_stop) > args.max_dist:
                    clusters.append(cluster)
                    cluster = []
                cluster.append(row)
                prev_stop = row['stop']
            if cluster:
                clusters.append(cluster)

            # 4. Pick the leader per cluster + contig-end exclusion filter
            for cl in clusters:
                if strand == '+':
                    leader = min(cl, key=lambda r: r['start'])   # 5' end = lowest start
                    if leader['start'] <= args.window:           # not enough upstream room
                        continue
                else:
                    leader = max(cl, key=lambda r: r['stop'])    # 5' end = highest stop
                    contig_len = contig_lengths.get(contig, 0)
                    if contig_len > 0 and (leader['stop'] + args.window) > contig_len:
                        continue
                leaders.append(leader)

    # 5. Save the output
    out_df = pd.DataFrame(leaders)
    os.makedirs(os.path.dirname(args.out_tsv) or '.', exist_ok=True)
    out_df.to_csv(args.out_tsv, sep='\t', index=False)
    
    print(f"Original loci: {len(df)}")
    print(f"Operon leaders retained: {len(out_df)}")
    print(f"Dropped {len(df) - len(out_df)} internal/truncated loci.")

if __name__ == "__main__":
    main()