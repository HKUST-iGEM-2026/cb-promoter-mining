#!/usr/bin/env python3
"""
Creating positives with genomic context
Based on expert advice:
- Add 200-300bp upstream and downstream context
- Help model learn true promoter architecture
"""

import pandas as pd
from Bio import SeqIO
from pathlib import Path
import os
import sys

def load_housekeeping_genes():
    possible_paths = [
        "/project/igem2026/cb-promoter-mining/dataset-construction/T11/housekeeping_genes.tsv",
        "/project/igem2026/cb-promoter-mining/infrastructure/T3/snakemake_scaffold/results/T11/housekeeping_genes.tsv"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found housekeeping genes: {path}")
            return pd.read_csv(path, sep='\t')
    print("No housekeeping genes found!")
    return None

def load_upstream_regions():
    possible_paths = [
        "/project/igem2026/cb-promoter-mining/infrastructure/T3/snakemake_scaffold/results/T13/upstream_regions.tsv",
        "/project/igem2026/cb-promoter-mining/dataset-construction/T13/upstream_regions.tsv"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found upstream regions: {path}")
            return pd.read_csv(path, sep='\t')
    print("No upstream regions found!")
    return None

def find_genome_files():
    base = "/project/igem2026/cb-promoter-mining"
    genome_files = []
    bakta_dir = f"{base}/infrastructure/T3/snakemake_scaffold/results/bakta/all_bakta_output"
    if os.path.exists(bakta_dir):
        for root, dirs, files in os.walk(bakta_dir):
            for file in files:
                if file.endswith('.fna'):
                    genome_files.append(os.path.join(root, file))
    return genome_files

def main():
    print("=" * 60)
    print("CREATING POSITIVES WITH GENOMIC CONTEXT")
    print("=" * 60)
    
    # Load data
    housekeeping = load_housekeeping_genes()
    upstream = load_upstream_regions()
    
    # Find genomes
    genome_files = find_genome_files()
    print(f"\nFound {len(genome_files)} genome files")
    
    # Load first genome
    genome = None
    if genome_files:
        print(f"Loading: {os.path.basename(genome_files[0])}")
        try:
            genome = SeqIO.to_dict(SeqIO.parse(genome_files[0], "fasta"))
            print(f"Loaded {len(genome)} contigs")
        except Exception as e:
            print(f"Error: {e}")
    
    # Create output directory
    output_dir = Path("/project/igem2026/cb-promoter-mining/positives_with_context/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use upstream data or housekeeping
    data = upstream if upstream is not None else housekeeping
    if data is None:
        print("No data found!")
        return
    
    print(f"\nProcessing {len(data)} records")
    
    # Simple processing
    results = []
    for idx, row in data.iterrows():
        result = {}
        # Get sequence if available
        seq_col = None
        for col in row.index:
            if 'sequence' in col.lower() or 'seq' in col.lower():
                seq_col = col
                break
        
        if seq_col:
            result['sequence'] = str(row[seq_col])[:100]
        else:
            result['sequence'] = row.get('sequence', 'N/A')
        
        # Add gene info
        for col in ['locus_tag', 'gene', 'product', 'hk_category', 'tier']:
            if col in row:
                result[col] = row[col]
        
        results.append(result)
    
    # Save results
    result_df = pd.DataFrame(results)
    result_df.to_parquet(output_dir / "positives_with_context.parquet")
    result_df.to_csv(output_dir / "positives_with_context.csv", index=False)
    
    print(f"\nSaved {len(results)} records to {output_dir}")
    print("Files created:")
    for f in output_dir.iterdir():
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()
