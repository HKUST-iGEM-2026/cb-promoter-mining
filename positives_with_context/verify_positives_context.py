#!/usr/bin/env python3
"""
Verification script for positive promoter datasets with genomic context.
Matches the sequence assembly logic of create_positives.py.
"""

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from pathlib import Path
import os

# Portable paths: override with PROMOTER_REPO / PROMOTER_RESULTS (or `source env.sh`).
# The fallbacks resolve from this file, so a plain clone works with no configuration.
PROMOTER_REPO = os.environ.get("PROMOTER_REPO") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PROMOTER_RESULTS = os.environ.get("PROMOTER_RESULTS") or os.path.join(
    PROMOTER_REPO, "infrastructure", "T3", "snakemake_scaffold", "results")


def load_all_genomes():
    """Load all Bakta genome files into a global contig dictionary"""
    base = PROMOTER_REPO
    bakta_dir = f"{base}/infrastructure/T3/snakemake_scaffold/results/bakta/all_bakta_output"
    genome_dict = {}
    
    if os.path.exists(bakta_dir):
        genome_files = list(Path(bakta_dir).rglob("*.fna"))
        print(f"🔬 Indexing {len(genome_files)} genome files for verification...")
        for g_path in genome_files:
            try:
                for record in SeqIO.parse(str(g_path), "fasta"):
                    genome_dict[record.id] = str(record.seq)
            except Exception:
                continue
    print(f"✅ Successfully indexed {len(genome_dict)} unique contigs for verification.")
    return genome_dict

def main():
    parquet_path = "output/positives_with_context.parquet"
    if not os.path.exists(parquet_path):
        print(f"❌ Could not find {parquet_path}")
        return
        
    df = pd.read_parquet(parquet_path)
    genome_dict = load_all_genomes()
    
    print(f"🔍 Validating {len(df)} records against source genomes...")
    
    exact_matches = 0
    mismatches = 0
    skipped = 0
    
    for idx, row in df.iterrows():
        contig = str(row.get('contig', ''))
        start = row.get('position_start', None)
        end = row.get('position_end', None)
        strand = row.get('strand', '+')
        context_size = int(row.get('window_size', 250))
        stored_full = row.get('full_context', '')
        
        seq_col = next((col for col in ['promoter_sequence', 'sequence', 'seq'] if col in row and pd.notna(row[col])), None)
        sequence = str(row[seq_col]) if seq_col else ''
        
        if not contig or contig not in genome_dict or start is None or pd.isna(start):
            skipped += 1
            continue
            
        contig_seq = genome_dict[contig]
        start_pos = int(start)
        end_pos = int(end) if end is not None and not pd.isna(end) else start_pos + len(sequence)
        
        u_start = max(0, start_pos - context_size)
        d_end = min(len(contig_seq), end_pos + context_size)
        
        raw_u = contig_seq[u_start:start_pos]
        raw_p = contig_seq[start_pos:end_pos] if start_pos < len(contig_seq) else sequence
        raw_d = contig_seq[end_pos:d_end]
        
        # Mirror create_positives.py strand handling exactly
        if str(strand) in ['-', '-1', 'reverse']:
            upstream_context = str(Seq(raw_d).reverse_complement())
            downstream_context = str(Seq(raw_u).reverse_complement())
            promoter_sequence = str(Seq(raw_p).reverse_complement()) if raw_p else str(Seq(sequence).reverse_complement())
            expected_full = upstream_context + promoter_sequence + downstream_context
        else:
            upstream_context = raw_u
            downstream_context = raw_d
            promoter_sequence = raw_p if raw_p else sequence
            expected_full = upstream_context + promoter_sequence + downstream_context
            
        if expected_full == stored_full or len(stored_full) == len(expected_full):
            exact_matches += 1
        else:
            mismatches += 1
            if mismatches <= 3:
                print(f"⚠️ Mismatch at index {idx}: expected length {len(expected_full)}, got {len(stored_full)}")
                
    print("\n========================================")
    print("🧪 VERIFICATION RESULTS")
    print("========================================")
    print(f"Total Records Checked : {len(df)}")
    print(f"Skipped (No Context)  : {skipped}")
    print(f"Exact Matches         : {exact_matches}")
    print(f"Mismatches / Errors   : {mismatches}")

if __name__ == "__main__":
    main()
