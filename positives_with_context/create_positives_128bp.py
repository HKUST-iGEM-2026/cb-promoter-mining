#!/usr/bin/env python3
"""
Create positive promoter examples with 128bp sequences for NT v3.
NT v3 requires sequences >= 120bp and multiples of 128 tokens.
"""

import pandas as pd
from Bio import SeqIO
from pathlib import Path
import os
import sys

def load_upstream_regions():
    """Load the upstream regions from T13"""
    possible_paths = [
        "/project/igem2026/cb-promoter-mining/infrastructure/T3/snakemake_scaffold/results/T13/upstream_regions.tsv",
        "/project/igem2026/cb-promoter-mining/dataset-construction/T13/upstream_regions.tsv"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found upstream regions: {path}")
            return pd.read_csv(path, sep='\t')
    print("❌ No upstream regions found!")
    return None

def find_genome_files():
    """Find genome files for context extraction"""
    base = "/project/igem2026/cb-promoter-mining"
    genome_files = []
    bakta_dir = f"{base}/infrastructure/T3/snakemake_scaffold/results/bakta/all_bakta_output"
    if os.path.exists(bakta_dir):
        for root, dirs, files in os.walk(bakta_dir):
            for file in files:
                if file.endswith('.fna'):
                    genome_files.append(os.path.join(root, file))
    return genome_files

def extract_128bp_promoter(row, genome, target_length=128):
    """
    Extract a 128bp promoter sequence with genomic context.
    
    For NT v3:
    - Minimum sequence length: 120bp (128 tokens)
    - Takes upstream region + promoter + downstream context
    """
    # Try to get sequence and location info
    seq_col = None
    for col in row.index:
        if 'sequence' in col.lower() or 'seq' in col.lower():
            seq_col = col
            break
    
    if seq_col is None:
        return None
    
    sequence = str(row[seq_col]) if pd.notna(row[seq_col]) else ''
    if not sequence:
        return None
    
    contig = str(row.get('contig', ''))
    start = row.get('start', None)
    end = row.get('end', None)
    
    # Default: just use the sequence and pad if needed
    if not contig or contig not in genome or start is None:
        # If no genome context, pad sequence to target length
        if len(sequence) < target_length:
            sequence = sequence + 'N' * (target_length - len(sequence))
        elif len(sequence) > target_length:
            sequence = sequence[:target_length]
        return {
            'promoter_128bp': sequence,
            'length': len(sequence),
            'has_genomic_context': False
        }
    
    # Extract 128bp with genomic context
    contig_seq = str(genome[contig].seq)
    start_pos = int(start)
    end_pos = int(end) if end is not None else start_pos + len(sequence)
    
    # Center the 128bp window around the promoter
    promoter_length = len(sequence)
    half_target = target_length // 2
    
    # Calculate upstream and downstream needed
    upstream_needed = max(0, half_target - promoter_length // 2)
    downstream_needed = max(0, target_length - promoter_length - upstream_needed)
    
    # Get upstream context
    upstream_start = max(0, start_pos - upstream_needed)
    upstream_context = contig_seq[upstream_start:start_pos]
    
    # Get downstream context
    downstream_end = min(len(contig_seq), end_pos + downstream_needed)
    downstream_context = contig_seq[end_pos:downstream_end]
    
    # Build the full 128bp sequence
    full_sequence = upstream_context + sequence + downstream_context
    
    # Pad with N if still too short
    if len(full_sequence) < target_length:
        full_sequence = full_sequence + 'N' * (target_length - len(full_sequence))
    elif len(full_sequence) > target_length:
        full_sequence = full_sequence[:target_length]
    
    return {
        'promoter_128bp': full_sequence,
        'length': len(full_sequence),
        'contig': contig,
        'position_start': start_pos,
        'position_end': end_pos,
        'has_genomic_context': True,
        'upstream_used': len(upstream_context),
        'downstream_used': len(downstream_context)
    }

def main():
    print("=" * 60)
    print("CREATING 128bp POSITIVES FOR NT v3")
    print("=" * 60)
    
    # Load data
    upstream = load_upstream_regions()
    if upstream is None:
        print("Error: Could not load upstream regions!")
        sys.exit(1)
    
    print(f"Loaded {len(upstream)} upstream records")
    
    # Find genomes
    genome_files = find_genome_files()
    print(f"Found {len(genome_files)} genome files")
    
    # Load first genome
    genome = None
    if genome_files:
        print(f"Loading: {os.path.basename(genome_files[0])}")
        try:
            genome = SeqIO.to_dict(SeqIO.parse(genome_files[0], "fasta"))
            print(f"Loaded {len(genome)} contigs")
        except Exception as e:
            print(f"Error loading genome: {e}")
    
    # Process all records
    results = []
    for idx, row in upstream.iterrows():
        result = extract_128bp_promoter(row, genome, target_length=128)
        if result:
            # Add metadata
            for col in ['locus_tag', 'gene', 'product', 'hk_category', 'tier']:
                if col in row:
                    result[col] = row[col]
            results.append(result)
    
    # Save results
    output_dir = Path("/project/igem2026/cb-promoter-mining/positives_with_context/output_128bp")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_df = pd.DataFrame(results)
    
    # Save in multiple formats
    result_df.to_parquet(output_dir / "positives_128bp.parquet")
    result_df.to_csv(output_dir / "positives_128bp.csv", index=False)
    
    # Create summary
    print("\n📊 SUMMARY")
    print("-" * 40)
    print(f"Total sequences: {len(results)}")
    print(f"With genomic context: {sum(r.get('has_genomic_context', False) for r in results)}")
    
    if 'promoter_128bp' in result_df.columns:
        lengths = result_df['promoter_128bp'].str.len()
        print(f"Sequence lengths: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.0f}")
    
    print(f"\n✅ Files saved to: {output_dir}")
    for f in output_dir.iterdir():
        size = f.stat().st_size / 1024
        print(f"   - {f.name} ({size:.1f} KB)")

if __name__ == "__main__":
    main()
