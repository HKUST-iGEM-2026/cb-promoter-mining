"""
T14: Four-class negative-set generation including Shuffled Real Sequence (SRS)
Owner: Seonyu
Due: 17 Jun

Builds a balanced four-class negative control dataset to train classification 
models, ensuring compositional features are preserved while biological promoter 
signals are systematically destroyed.

Design:
  * Class 1 (SRS): Uses an Eulerian graph algorithm to perform a dinucleotide-
    preserving shuffle of the true positive upstream sequences from T13.
  * Class 2 (CDS): Samples windows from the structural cores of active coding 
    regions, avoiding the sequence edges.
  * Class 3 (Intergenic Non-Promoter): Targets convergent intergenic regions 
    (3' to 3' gene boundaries) where no promoter traffic exists.
  * Class 4 (Downstream 3'): Extracts windows immediately following gene 
    stop boundaries (transcription terminators).

Usage (from the repository root):
    python pipeline/datasets/negative-sets/negative_set_generation.py \
        --upstream-tsv pipeline/datasets/upstream-extraction/out/upstream_regions.tsv \
        --bakta-dir ../bakta_outputs \
        --out-tsv pipeline/datasets/negative-sets/out/negative_set.tsv \
        --window 300
"""

import os
import random
import argparse
import collections
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

def dinucleotide_shuffle(seq):
    """
    Performs a strict dinucleotide-preserving shuffle of a sequence string
    using an Eulerian path approach to ensure identical background frequencies.
    """
    if len(seq) < 2:
        return seq
    
    # Build transition graph
    edges = collections.defaultdict(list)
    for i in range(len(seq) - 1):
        edges[seq[i]].append(seq[i+1])
        
    # Shuffle edges for each node, keeping the last edge fixed to maintain Eulerian property
    for node, edge_list in edges.items():
        if len(edge_list) > 1:
            last_edge = edge_list[-1]
            rest = edge_list[:-1]
            random.shuffle(rest)
            edges[node] = rest + [last_edge]
            
    # Reconstruct sequence path
    shuffled = []
    curr = seq[0]
    shuffled.append(curr)
    
    while edges[curr]:
        nxt = edges[curr].pop(0)
        shuffled.append(nxt)
        curr = nxt
        
    # Fallback to standard shuffle if the graph traversal gets trapped early
    if len(shuffled) < len(seq):
        seq_list = list(seq)
        random.shuffle(seq_list)
        return "".join(seq_list)
        
    return "".join(shuffled)

def parse_gff_features(gff_path):
    """Parses GFF3 file to return a dataframe of clean CDS boundaries."""
    features = []
    with open(gff_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9 or parts[2] != 'CDS':
                continue
            features.append({
                'contig': parts[0],
                'start': int(parts[3]),
                'stop': int(parts[4]),
                'strand': parts[6]
            })
    return pd.DataFrame(features)

def main():
    parser = argparse.ArgumentParser(description="T14: Generate Four-Class Negative Set")
    parser.add_argument("--upstream-tsv", required=True, help="Path to T13 upstream_regions.tsv")
    parser.add_argument("--bakta-dir", required=True, help="Directory containing T6 outputs")
    parser.add_argument("--out-tsv", required=True, help="Path to save output negative_set.tsv")
    parser.add_argument("--window", type=int, default=300, help="Window size for sequence extraction")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducible negatives (SRS shuffle + CDS/intergenic sampling)")
    args = parser.parse_args()

    random.seed(args.seed)   # make the negative set reproducible (was unseeded)

    # Load positive references
    pos_df = pd.read_csv(args.upstream_tsv, sep='\t')
    target_count_per_class = len(pos_df)
    
    negative_records = []

    # --- CLASS 1: Shuffled Real Sequences (SRS) ---
    print("Generating Class 1: Shuffled Real Sequences...")
    for _, row in pos_df.iterrows():
        shuffled_seq = dinucleotide_shuffle(row['upstream_sequence'])
        negative_records.append({
            'genome': row['genome'], 'contig': row['contig'], 'negative_class': 'SRS',
            'sequence': shuffled_seq, 'strand': row['strand']
        })

    # Group processing per genome for the remaining physical classes
    genomes = pos_df['genome'].unique()
    
    class_counts = {'CDS': 0, 'Intergenic_NonPromoter': 0, 'Downstream_3Prime': 0}

    for genome in genomes:
        # Bakta writes a nested per-genome dir: <bakta-dir>/<genome>_genomic_output/<genome>_genomic.{fna,gff3}
        gdir = f"{genome}_genomic_output"
        fna_path = os.path.join(args.bakta_dir, gdir, f"{genome}_genomic.fna")
        gff_path = os.path.join(args.bakta_dir, gdir, f"{genome}_genomic.gff3")

        if not (os.path.exists(fna_path) and os.path.exists(gff_path)):
            print(f"Warning: Bakta outputs not found for {genome} at {gdir}/. Skipping.")
            continue
            
        contig_dict = {rec.id: str(rec.seq) for rec in SeqIO.parse(fna_path, "fasta")}
        gff_df = parse_gff_features(gff_path)
        
        for contig_id, group in gff_df.groupby('contig'):
            if contig_id not in contig_dict:
                continue
            contig_seq = contig_dict[contig_id]
            seq_len = len(contig_seq)
            sorted_genes = group.sort_values(by='start').to_dict('records')
            
            for i, gene in enumerate(sorted_genes):
                # --- CLASS 2: Coding Sequence (CDS) Negatives ---
                gene_len = gene['stop'] - gene['start'] + 1
                if gene_len > (args.window + 200) and class_counts['CDS'] < target_count_per_class:
                    sample_start = gene['start'] + 100
                    sample_end = gene['stop'] - 100 - args.window
                    if sample_end > sample_start:
                        idx = random.randint(sample_start, sample_end) - 1
                        seq_fragment = contig_seq[idx : idx + args.window]
                        if gene['strand'] == '-':
                            seq_fragment = str(Seq(seq_fragment).reverse_complement())
                        
                        negative_records.append({
                            'genome': genome, 'contig': contig_id, 'negative_class': 'CDS',
                            'sequence': seq_fragment, 'strand': gene['strand']
                        })
                        class_counts['CDS'] += 1

                # --- CLASS 3: Intergenic Non-Promoter (Convergent Pairs) ---
                if i < len(sorted_genes) - 1:
                    next_gene = sorted_genes[i+1]
                    if gene['strand'] == '+' and next_gene['strand'] == '-':
                        # Convergent space found between their 3' ends
                        space_start = gene['stop'] # 1-based
                        space_end = next_gene['start'] - 1
                        if (space_end - space_start) >= args.window and class_counts['Intergenic_NonPromoter'] < target_count_per_class:
                            idx = random.randint(space_start, space_end - args.window)
                            seq_fragment = contig_seq[idx : idx + args.window]
                            negative_records.append({
                                'genome': genome, 'contig': contig_id, 'negative_class': 'Intergenic_NonPromoter',
                                'sequence': seq_fragment, 'strand': '+'
                            })
                            class_counts['Intergenic_NonPromoter'] += 1

                # --- CLASS 4: Downstream / 3' Negatives ---
                if class_counts['Downstream_3Prime'] < target_count_per_class:
                    if gene['strand'] == '+' and (gene['stop'] + args.window) <= seq_len:
                        idx = gene['stop']
                        seq_fragment = contig_seq[idx : idx + args.window]
                        negative_records.append({
                            'genome': genome, 'contig': contig_id, 'negative_class': 'Downstream_3Prime',
                            'sequence': seq_fragment, 'strand': '+'
                        })
                        class_counts['Downstream_3Prime'] += 1
                    elif gene['strand'] == '-' and (gene['start'] - 1 - args.window) >= 0:
                        idx = gene['start'] - 1 - args.window
                        seq_fragment = contig_seq[idx : idx + args.window]
                        seq_fragment = str(Seq(seq_fragment).reverse_complement())
                        negative_records.append({
                            'genome': genome, 'contig': contig_id, 'negative_class': 'Downstream_3Prime',
                            'sequence': seq_fragment, 'strand': '-'
                        })
                        class_counts['Downstream_3Prime'] += 1

    # Output assembly
    out_df = pd.DataFrame(negative_records)
    os.makedirs(os.path.dirname(args.out_tsv) or '.', exist_ok=True)
    out_df.to_csv(args.out_tsv, sep='\t', index=False)
    
    print("\n--- Generation Summary ---")
    print(out_df['negative_class'].value_counts())
    print(f"Total negative control sequences exported: {len(out_df)}")

if __name__ == "__main__":
    main()