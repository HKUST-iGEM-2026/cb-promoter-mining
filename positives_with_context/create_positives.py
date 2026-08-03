#!/usr/bin/env python3
"""
Consolidated script to create positive promoter datasets with genomic context.
Strictly enforces a uniform context size (250bp) for all records.
"""

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from pathlib import Path
import os
import sys
import argparse

def load_data():
    """Load both verified and T13 upstream datasets"""
    possible_paths = [
        "/project/igem2026/cb-promoter-mining/dataset-construction",
        "/project/igem2026/cb-promoter-mining/infrastructure/T3/snakemake_scaffold/results"
    ]
    frames = []
    for base in possible_paths:
        t13 = os.path.join(base, "T13", "upstream_regions.tsv")
        if os.path.exists(t13):
            df = pd.read_csv(t13, sep='\t')
            df['source_type'] = 'upstream_region'
            frames.append(df)
            break
            
    for base in possible_paths:
        for v_name in ["verified_positives.tsv", "hocq_tss.tsv"]:
            v_path = os.path.join(base, v_name)
            if os.path.exists(v_path):
                df = pd.read_csv(v_path, sep='\t')
                df['source_type'] = 'verified_positive'
                frames.append(df)
                
    if not frames:
        print("❌ No input datasets found!")
        return None
    return pd.concat(frames, ignore_index=True)

def load_all_genomes():
    """Load all Bakta genome files into a global contig dictionary"""
    base = "/project/igem2026/cb-promoter-mining"
    bakta_dir = f"{base}/infrastructure/T3/snakemake_scaffold/results/bakta/all_bakta_output"
    genome_dict = {}
    
    if os.path.exists(bakta_dir):
        genome_files = list(Path(bakta_dir).rglob("*.fna"))
        print(f"🔬 Indexing {len(genome_files)} genome files...")
        for g_path in genome_files:
            try:
                for record in SeqIO.parse(str(g_path), "fasta"):
                    genome_dict[record.id] = str(record.seq)
            except Exception:
                continue
    print(f"✅ Successfully indexed {len(genome_dict)} unique contigs.")
    return genome_dict

def process_record(row, genome_dict, mode='full', context_size=250, target_length=128):
    # Support 'upstream_sequence' or standard sequence column names
    seq_col = next((col for col in ['upstream_sequence', 'sequence', 'seq'] if col in row and pd.notna(row[col])), None)
    sequence = str(row[seq_col]) if seq_col else ''
    
    contig = str(row.get('contig', ''))
    start = row.get('start', None)
    end = row.get('stop', row.get('end', None))
    strand = row.get('strand', '+')
    
    # Fallback if no genomic context mapping is possible
    if not contig or contig not in genome_dict or start is None or pd.isna(start):
        if mode == '128bp':
            seq = sequence[:target_length] if len(sequence) >= target_length else sequence + 'N' * (target_length - len(sequence))
            return {
                'promoter_128bp': seq,
                'length': len(seq),
                'contig': contig,
                'position_start': start,
                'position_end': end,
                'strand': strand,
                'window_size': context_size,
                'has_genomic_context': False
            }
        else:
            return {
                'promoter_sequence': sequence,
                'full_context': sequence,
                'upstream_context': 'N' * min(context_size, len(sequence)),
                'downstream_context': 'N' * min(context_size, len(sequence)),
                'contig': contig,
                'position_start': start,
                'position_end': end,
                'strand': strand,
                'window_size': context_size,
                'has_genomic_context': False
            }
            
    contig_seq = genome_dict[contig]
    start_pos = int(start)
    end_pos = int(end) if end is not None and not pd.isna(end) else start_pos + len(sequence)
    
    if mode == '128bp':
        promoter_len = max(len(sequence), end_pos - start_pos)
        half_target = target_length // 2
        u_needed = max(0, half_target - promoter_len // 2)
        d_needed = max(0, target_length - promoter_len - u_needed)
        
        u_start = max(0, start_pos - u_needed)
        d_end = min(len(contig_seq), end_pos + d_needed)
        
        raw_u = contig_seq[u_start:start_pos]
        raw_p = contig_seq[start_pos:end_pos] if start_pos < len(contig_seq) else sequence
        raw_d = contig_seq[end_pos:d_end]
        
        if str(strand) in ['-', '-1', 'reverse']:
            full_seq = str(Seq(raw_d + raw_p + raw_u).reverse_complement())
        else:
            full_seq = raw_u + raw_p + raw_d
            
        if len(full_seq) < target_length:
            full_seq += 'N' * (target_length - len(full_seq))
        elif len(full_seq) > target_length:
            full_seq = full_seq[:target_length]
            
        return {
            'promoter_128bp': full_seq,
            'length': len(full_seq),
            'contig': contig,
            'position_start': start_pos,
            'position_end': end_pos,
            'strand': strand,
            'window_size': context_size,
            'has_genomic_context': True
        }
    else:
        # STRICTLY ENFORCE UNIFORM 250bp CONTEXT
        u_start = max(0, start_pos - context_size)
        d_end = min(len(contig_seq), end_pos + context_size)
        
        raw_u = contig_seq[u_start:start_pos]
        raw_p = contig_seq[start_pos:end_pos] if start_pos < len(contig_seq) else sequence
        raw_d = contig_seq[end_pos:d_end]
        
        if str(strand) in ['-', '-1', 'reverse']:
            upstream_context = str(Seq(raw_d).reverse_complement())
            downstream_context = str(Seq(raw_u).reverse_complement())
            promoter_sequence = str(Seq(raw_p).reverse_complement()) if raw_p else str(Seq(sequence).reverse_complement())
            full_context = upstream_context + promoter_sequence + downstream_context
        else:
            upstream_context = raw_u
            downstream_context = raw_d
            promoter_sequence = raw_p if raw_p else sequence
            full_context = upstream_context + promoter_sequence + downstream_context
            
        return {
            'promoter_sequence': promoter_sequence,
            'upstream_context': upstream_context,
            'downstream_context': downstream_context,
            'full_context': full_context,
            'contig': contig,
            'position_start': start_pos,
            'position_end': end_pos,
            'strand': strand,
            'window_size': context_size,
            'has_genomic_context': True
        }

def main():
    parser = argparse.ArgumentParser(description="Create positives with genomic context.")
    parser.add_argument('--mode', type=str, choices=['full', '128bp'], default='full', help='Output mode')
    args = parser.parse_args()
    
    print(f"🚀 Running Positive Construction in [{args.mode.upper()}] mode (Strict 250bp context limit)")
    data = load_data()
    if data is None:
        sys.exit(1)
        
    genome_dict = load_all_genomes()
    
    results = []
    for idx, row in data.iterrows():
        res = process_record(row, genome_dict, mode=args.mode, context_size=250)
        if res:
            for col in ['genome', 'locus_tag', 'gene', 'product', 'hk_category', 'tier', 'source_type']:
                if col in row:
                    res[col] = row[col]
            results.append(res)
            
    output_dir = Path("/project/igem2026/cb-promoter-mining/positives_with_context/output" if args.mode == 'full' else "/project/igem2026/cb-promoter-mining/positives_with_context/output_128bp")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_df = pd.DataFrame(results)
    suffix = "positives_with_context" if args.mode == 'full' else "positives_128bp"
    result_df.to_parquet(output_dir / f"{suffix}.parquet")
    result_df.to_csv(output_dir / f"{suffix}.csv", index=False)
    
    print(f"✅ Finished! Processed {len(results)} records. Saved to {output_dir}")

if __name__ == "__main__":
    main()
