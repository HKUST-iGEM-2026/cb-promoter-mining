"""
T13: Upstream-region extraction
Owner: Seonyu
Due: 16 Jun

Extracts the upstream DNA sequence in front of each operon leader. 
Provides a reusable extraction function with strict contig-end bounds checking, 
which is reused by the T28 sliding-window mining engine.

Design:
  * Parses biological 1-based inclusive coordinates to Python 0-based exclusive.
  * Extacts [start - window: start] for '+' strand.
  * Extracts [stop : stop + window] for '-' strand, then reverse-complements.
  * Drops loci where the requested window exceeds contig boundaries.

Usage:
    python extract_upstream.py \
        --leaders-tsv <path_to_T12_operon_leaders.tsv> \
        --bakta-dir <dir_containing_T6_fna_files> \
        --out-tsv <path_to_output_upstream_regions.tsv> \
        --window 300
"""

import os
import argparse
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

def extract_upstream_sequence(sequence_str, start, stop, strand, window=300):
    """
    Extracts the upstream window from a 0-based sequence string.
    Assumes `start` and `stop` are 1-based inclusive biological coordinates.
    Returns None if the window exceeds contig boundaries.
    """
    seq_len = len(sequence_str)
    
    if strand == '+':
        # 1-based start 100 -> 0-based index 99
        start_idx = start - 1
        extract_start = start_idx - window
        extract_end = start_idx
        
        if extract_start < 0:
            return None
            
        return str(sequence_str[extract_start:extract_end])
        
    elif strand == '-':
        # 1-based stop 200 -> 0-based index 200 is the first base AFTER the gene
        extract_start = stop
        extract_end = stop + window
        
        if extract_end > seq_len:
            return None
            
        raw_seq = str(sequence_str[extract_start:extract_end])
        return str(Seq(raw_seq).reverse_complement())
        
    return None

def main():
    parser = argparse.ArgumentParser(description="T13: Extract upstream promoter regions")
    parser.add_argument("--leaders-tsv", required=True, help="Path to T12 operon_leaders.tsv")
    parser.add_argument("--bakta-dir", required=True, help="Directory containing T6 Bakta .fna files")
    parser.add_argument("--out-tsv", required=True, help="Path to output upstream_regions.tsv")
    parser.add_argument("--window", type=int, default=300, help="Upstream window size (bp) to extract")
    args = parser.parse_args()

    # 1. Load the operon leaders
    df = pd.read_csv(args.leaders_tsv, sep='\t')
    extracted_data = []
    dropped_count = 0

    # 2. Process genome by genome to avoid loading all FASTAs into RAM at once
    for genome, group in df.groupby('genome'):
        # Bakta writes a nested per-genome dir: <bakta-dir>/<genome>_genomic_output/<genome>_genomic.fna
        fasta_path = os.path.join(args.bakta_dir, f"{genome}_genomic_output",
                                  f"{genome}_genomic.fna")

        if not os.path.exists(fasta_path):
            print(f"Warning: FASTA not found for {genome} at {fasta_path}. Skipping.")
            continue
            
        # Load contigs into a dictionary: {contig_id: sequence_string}
        contig_dict = {rec.id: str(rec.seq) for rec in SeqIO.parse(fasta_path, "fasta")}
        
        for _, row in group.iterrows():
            contig_id = row['contig']
            if contig_id not in contig_dict:
                dropped_count += 1
                continue
                
            sequence = extract_upstream_sequence(
                sequence_str=contig_dict[contig_id],
                start=row['start'],
                stop=row['stop'],
                strand=row['strand'],
                window=args.window
            )
            
            if sequence:
                row_dict = row.to_dict()
                row_dict['upstream_sequence'] = sequence
                row_dict['window_size'] = args.window
                extracted_data.append(row_dict)
            else:
                dropped_count += 1

    # 3. Save the output
    out_df = pd.DataFrame(extracted_data)
    out_dir = os.path.dirname(args.out_tsv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(args.out_tsv, sep='\t', index=False)
    
    print(f"Original operon leaders: {len(df)}")
    print(f"Successfully extracted: {len(out_df)}")
    print(f"Dropped due to contig boundaries (or missing contigs): {dropped_count}")

if __name__ == "__main__":
    main()