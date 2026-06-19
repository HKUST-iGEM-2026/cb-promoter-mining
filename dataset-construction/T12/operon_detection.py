"""
T12: Operon detection
Owner: Seonyu
Due: 15 Jun

Collapses grouped loci into operon leaders so we only extract the promoter in front 
of the first gene, rather than internal genes that lack their own promoter. 
Implements a contig-end exclusion filter to prevent extracting truncated or empty 
upstream regions from genes sitting near assembly boundaries.

Design:
  * Operon Collapsing: Groups loci by genome, contig, and strand. Consecutive 
    genes within a defined distance (default 100 bp) are collapsed, and only 
    the first gene (the leader) is retained.
  * Contig-End Exclusion: Because target assemblies are often incomplete, operon 
    leaders whose upstream window (default 300 bp) spills over the start (+ strand) 
    or end (- strand) of a contig are dropped.
  * Contig lengths are parsed directly from the `##sequence-region` headers of 
    the T6 `.gff3` files to accurately calculate distances for the - strand.

Usage:
    python detect_operons.py \
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
    
    # 2. Group by genome, contig, and strand, then sort by start coordinate
    df = df.sort_values(by=['genome', 'contig', 'strand', 'start']).reset_index(drop=True)
    
    leaders = []
    
    # 3. Process each genome individually to handle operon collapsing and contig lengths
    for genome, group in df.groupby('genome'):
        # Parse contig lengths for this specific genome
        gff_path = os.path.join(args.bakta_dir, f"{genome}.gff3")
        contig_lengths = get_contig_lengths(gff_path)
        
        # Operon collapsing logic
        prev_contig = None
        prev_strand = None
        prev_stop = -1
        
        for _, row in group.iterrows():
            is_leader = False
            
            # If contig or strand changes, or distance from previous gene is too large, it's a new operon
            if row['contig'] != prev_contig or row['strand'] != prev_strand:
                is_leader = True
            else:
                dist = row['start'] - prev_stop
                if dist > args.max_dist:
                    is_leader = True
            
            # Update tracking variables
            prev_contig = row['contig']
            prev_strand = row['strand']
            prev_stop = row['stop']
            
            if not is_leader:
                continue # Skip internal operon genes
                
            # 4. Contig-end exclusion filter
            contig_len = contig_lengths.get(row['contig'], 0)
            
            if row['strand'] == '+':
                # + strand: promoter is 5', before 'start'
                if row['start'] <= args.window:
                    continue
            else:
                # - strand: promoter is 3', after 'stop'
                if contig_len > 0 and (row['stop'] + args.window) > contig_len:
                    continue
                    
            leaders.append(row)

    # 5. Save the output
    out_df = pd.DataFrame(leaders)
    os.makedirs(os.path.dirname(args.out_tsv), exist_ok=True)
    out_df.to_csv(args.out_tsv, sep='\t', index=False)
    
    print(f"Original loci: {len(df)}")
    print(f"Operon leaders retained: {len(out_df)}")
    print(f"Dropped {len(df) - len(out_df)} internal/truncated loci.")

if __name__ == "__main__":
    main()