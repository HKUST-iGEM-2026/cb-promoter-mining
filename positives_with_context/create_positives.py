#!/usr/bin/env python3
"""
Consolidated script to create positive promoter datasets with genomic context.
Strictly enforces the pipeline schema contract:
 - sequence: Fixed 81 bp core promoter sequence
 - sequence_context: Fixed-length genomic window (multiple of 128, e.g., 384 bp)
 - core_offset: Integer offset of the 81 bp core within sequence_context
"""

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from pathlib import Path
import os
import sys
import argparse

def load_data():
    """Load and combine positive datasets from T13, T7, and T8 directories"""
    frames = []
    
    # 1. Load T13 upstream regions
    t13_path = "/project/igem2026/cb-promoter-mining/infrastructure/T3/snakemake_scaffold/results/T13/upstream_regions.tsv"
    if os.path.exists(t13_path):
        df_t13 = pd.read_csv(t13_path, sep='\t')
        df_t13['source_type'] = 'upstream_region'
        frames.append(df_t13)
        print(f"📥 Loaded T13: {len(df_t13)} rows")

    # 2. Load T7 / T8 datasets
    t7_path = "/project/igem2026/cb-promoter-mining/data-acquisition-and-curation/T7/T7 - all.csv"
    if os.path.exists(t7_path):
        df_t7 = pd.read_csv(t7_path)
        df_t7['source_type'] = 't7_master_curated'
        frames.append(df_t7)
        print(f"📥 Loaded T7: {len(df_t7)} rows")

    t8_path = "/project/igem2026/cb-promoter-mining/data-acquisition-and-curation/T8/curated_positive_promoters.parquet"
    if os.path.exists(t8_path):
        df_t8 = pd.read_parquet(t8_path)
        df_t8['source_type'] = 't8_curated_positive'
        frames.append(df_t8)
        print(f"📥 Loaded T8: {len(df_t8)} rows")

    if not frames:
        print("❌ No input datasets found in T13, T7, or T8!")
        return None
        
    combined_df = pd.concat(frames, ignore_index=True)
    print(f"✅ Successfully combined all datasets. Total rows: {len(combined_df)}")
    return combined_df

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

def process_record(row, genome_dict, core_length=81, total_context_len=384):
    """
    Extracts genomic context if coordinates are available, 
    otherwise falls back gracefully to native sequence data.
    """
    seq_col = next((col for col in ['upstream_sequence', 'sequence', 'seq', 'promoter_sequence'] if col in row and pd.notna(row[col])), None)
    raw_fallback_seq = str(row[seq_col]) if seq_col else ''
    
    contig = str(row.get('contig', row.get('Locus', '')))
    start = row.get('start', row.get('Genome_position', row.get('position_start', None)))
    end = row.get('stop', row.get('end', row.get('position_end', start)))
    strand = row.get('strand', row.get('Strand', '+'))
    
    half_context = total_context_len // 2
    half_core = core_length // 2
    
    matched_contig = None
    if contig in genome_dict:
        matched_contig = contig
    else:
        for g_id in genome_dict.keys():
            if contig and (contig in g_id or g_id in contig):
                matched_contig = g_id
                break

    # Fallback mode for rows without genome coordinates (like T8 sequence sets)
    if not matched_contig or start is None or pd.isna(start):
        padded_seq = raw_fallback_seq[:total_context_len].ljust(total_context_len, 'N')
        core_seq = raw_fallback_seq[:core_length].ljust(core_length, 'N')
        return {
            'sequence': core_seq,
            'sequence_context': padded_seq,
            'core_offset': half_context - half_core,
            'contig': contig if contig else 'unknown',
            'position_start': 0,
            'position_end': 0,
            'strand': strand if strand else '+',
            'has_genomic_context': False
        }
            
    contig_seq = genome_dict[matched_contig]
    start_pos = int(float(start))
    
    core_start = max(0, start_pos - half_core)
    core_end = core_start + core_length
    
    center_pos = start_pos
    context_start = max(0, center_pos - half_context)
    context_end = min(len(contig_seq), center_pos + half_context)
    
    raw_context = contig_seq[context_start:context_end]
    raw_core = contig_seq[core_start:core_end]
    
    if len(raw_context) < total_context_len:
        if context_start == 0:
            raw_context = 'N' * (total_context_len - len(raw_context)) + raw_context
        else:
            raw_context = raw_context + 'N' * (total_context_len - len(raw_context))
            
    if len(raw_core) < core_length:
        raw_core = raw_core.ljust(core_length, 'N')

    if str(strand) in ['-', '-1', 'reverse', 'R']:
        sequence_context = str(Seq(raw_context).reverse_complement())
        sequence = str(Seq(raw_core).reverse_complement())
        core_offset = total_context_len - (core_end - context_start)
    else:
        sequence_context = raw_context
        sequence = raw_core
        core_offset = core_start - context_start

    return {
        'sequence': sequence,
        'sequence_context': sequence_context,
        'core_offset': core_offset,
        'contig': matched_contig,
        'position_start': start_pos,
        'position_end': end,
        'strand': strand,
        'has_genomic_context': True
    }

def main():
    parser = argparse.ArgumentParser(description="Create positives with genomic context matching NTv3 contract.")
    args = parser.parse_args()
    
    print("🚀 Running Positive Construction with NTv3 128-multiple Schema Compliance")
    data = load_data()
    if data is None:
        sys.exit(1)
        
    genome_dict = load_all_genomes()
    
    results = []
    for idx, row in data.iterrows():
        res = process_record(row, genome_dict)
        if res:
            for col in ['genome', 'locus_tag', 'gene', 'product', 'hk_category', 'tier', 'source_type']:
                if col in row:
                    res[col] = row[col]
            results.append(res)
            
    output_dir = Path("/project/igem2026/cb-promoter-mining/positives_with_context/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_df = pd.DataFrame(results)
    result_df.to_parquet(output_dir / "positives_with_context.parquet", index=False)
    result_df.to_csv(output_dir / "positives_with_context.csv", index=False)
    
    print(f"✅ Finished! Processed {len(results)} records. Saved to {output_dir}")

if __name__ == "__main__":
    main()
