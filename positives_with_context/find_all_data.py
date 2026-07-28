#!/usr/bin/env python3
"""
Find and describe all data files for the positive promoter with genomic context task
"""
import os
import subprocess
from pathlib import Path

def find_files(directory, pattern):
    """Find files matching pattern in directory"""
    try:
        result = subprocess.run(
            f"find {directory} -name '{pattern}' 2>/dev/null | head -30",
            shell=True, capture_output=True, text=True
        )
        lines = result.stdout.strip().split('\n')
        return [f for f in lines if f]
    except:
        return []

def main():
    directories = [
        "/project/igem2026/cb-promoter-mining",
        "/scratch/igem2026",
    ]
    
    patterns = [
        "*T13*", "*T11*", "*T12*", "*operon*", "*housekeeping*",
        "*.parquet", "*.tsv", "*.csv", "*promoter*",
        "*.gbff", "*.gff", "*.fna", "*.fasta", "*.genbank",
        "*positive*", "*neg*", "*dataset*"
    ]
    
    print("=" * 80)
    print("FINDING DATA FILES FOR POSITIVES WITH GENOMIC CONTEXT")
    print("=" * 80)
    
    for directory in directories:
        if not os.path.exists(directory):
            continue
        print(f"\nSearching in: {directory}")
        print("-" * 40)
        
        for pattern in patterns:
            files = find_files(directory, pattern)
            if files:
                print(f"\n  Pattern: {pattern}")
                for f in files[:10]:
                    try:
                        size = os.path.getsize(f) / (1024 * 1024)
                        print(f"    {f} ({size:.2f} MB)")
                    except:
                        print(f"    {f}")
                if len(files) > 10:
                    print(f"    ... and {len(files) - 10} more files")

if __name__ == "__main__":
    main()
