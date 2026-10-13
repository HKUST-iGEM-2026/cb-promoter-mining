#!/usr/bin/env python3
"""
build_context_windows.py  —  TEMPLATE / starting point (not yet wired into the DAG)

Purpose
-------
Produce ONE canonical dataset that serves every model at its native length, so
the 81 bp models (ProkBERT/CNN) and the long-context models (NTv3/Evo2) train and
evaluate on the SAME rows and the SAME splits, cropped to different lengths.

It does this by emitting TWO sequence columns per record:

    sequence          the 81 bp promoter core          (ProkBERT / CNN)
    sequence_context  the long window, e.g. 768 bp      (NTv3 / Evo2)
    core_offset       index where `sequence` starts inside `sequence_context`

Downstream, `encode.py --from-parquet` picks a column with `--sequence-col`:
    CNN/ProkBERT :  --sequence-col sequence          (default)
    NTv3         :  --sequence-col sequence_context

Why two columns in ONE file (not two files): the rows can never fall out of
sync, and cropping is a pure slice — no genome access, no re-split.

What this script does (and does NOT do)
---------------------------------------
DOES: for every input record (a promoter core with genomic coordinates), pull the
real flanking sequence from the genome to build `sequence_context` of a fixed
length L (a multiple of 128 for NTv3), strand-aware, padding with 'N' ONLY where a
contig end is hit; keep the 81 bp core at a known `core_offset`; and round-trip
verify that the core it carved equals the genome at those coordinates.

DOES NOT: cluster or split. After this, run the EXISTING T15 step
(pipeline/datasets/clustering-and-splits/cdhit_splits.py) and cluster/split on the `sequence`
(81 bp core) column — clustering on the core, not the long window, preserves the
anti-leakage guarantee (90% identity over 768 bp is a much weaker guard than over
81 bp). Both columns then inherit whatever split each row lands in.

Expected input schema (all column names are configurable via flags)
-------------------------------------------------------------------
A parquet or TSV with, per row:
  contig   str   contig / replicon id WITHIN its genome
  start    int   core start, 0-based, half-open, FORWARD-strand genomic coord
  end      int   core end,   0-based, half-open, FORWARD-strand genomic coord
  strand   str   '+' or '-'
  genome   str   genome id that selects the FASTA (see --genome-dir). REQUIRED
                 because Bakta renames contigs to `contig_1`, so `contig` alone
                 is NOT unique across the 166 genomes — you must know the genome.
  (optional) sequence  str  the known 81 bp core, used ONLY to round-trip verify.
  ... plus any metadata (label, weight, source, species, sigma_factor, ...) which
      is carried through UNCHANGED and row-aligned.

NOTE ON COORDINATES: if your source is a GFF (1-based, inclusive), convert to
0-based half-open first: start0 = gff_start - 1, end0 = gff_end.

  python build_context_windows.py --input cores.parquet --genome-dir <bakta_fna_dir> \
      --length 768 --out results/T15_inputs/context_windows.parquet
"""
import argparse
import os
import sys
from pathlib import Path

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def find_genome_fastas(genome_dir):
    """Map genome-id -> fasta path. Genome id = the .fna/.fasta file stem.
    Walks subdirectories (Bakta nests one dir per genome)."""
    paths = {}
    for root, _dirs, files in os.walk(genome_dir):
        for f in files:
            if f.endswith((".fna", ".fasta", ".fa")):
                stem = f
                for ext in (".fna", ".fasta", ".fa"):
                    if stem.endswith(ext):
                        stem = stem[: -len(ext)]
                        break
                paths.setdefault(stem, os.path.join(root, f))
                # Bakta/NCBI files are "<accession>_genomic.fna", but upstream tables
                # (e.g. T13) put the BARE accession in the `genome` column — register
                # that too so lookup by accession works.
                for suf in ("_genomic", "_genomic_output"):
                    if stem.endswith(suf):
                        paths.setdefault(stem[: -len(suf)], os.path.join(root, f))
    return paths


def load_contigs(fasta_path):
    """Read a FASTA into {contig_id: sequence} (uppercase, first whitespace token
    of the header is the id)."""
    contigs, cid, buf = {}, None, []
    with open(fasta_path) as fh:
        for line in fh:
            if line.startswith(">"):
                if cid is not None:
                    contigs[cid] = "".join(buf).upper()
                cid = line[1:].strip().split()[0]
                buf = []
            else:
                buf.append(line.strip())
        if cid is not None:
            contigs[cid] = "".join(buf).upper()
    return contigs


def extract_context_window(contig_seq, start, end, strand, L):
    """
    Build a length-L window around the core [start, end) (0-based, half-open,
    forward-strand), strand-aware, padding with 'N' only at contig ends.

    Returns (context, core_offset, core):
      context      length-L string on the READING strand (revcomp'd if '-')
      core_offset  index where the core begins inside `context`
      core         context[core_offset : core_offset + core_len]
    """
    core_len = end - start
    if core_len <= 0:
        raise ValueError(f"non-positive core length: start={start} end={end}")
    if core_len > L:
        raise ValueError(f"core ({core_len} bp) longer than window L={L}")

    up = (L - core_len) // 2          # bases of context on the upstream side
    down = L - core_len - up          # remainder downstream (handles odd L-core_len)
    w_start, w_end = start - up, end + down     # forward-strand window, length L

    # Clip to the contig and remember how much fell off each end (to N-pad).
    left_pad = max(0, -w_start)
    right_pad = max(0, w_end - len(contig_seq))
    frag = contig_seq[max(0, w_start): min(len(contig_seq), w_end)]
    frag = "N" * left_pad + frag + "N" * right_pad
    assert len(frag) == L, f"internal: fragment {len(frag)} != L {L}"

    if strand == "+":
        return frag, up, frag[up: up + core_len]
    elif strand == "-":
        ctx = revcomp(frag)
        # After reverse-complementing, the core [up, up+core_len) maps to
        # [L-up-core_len, L-up) = [down, down+core_len).
        return ctx, down, ctx[down: down + core_len]
    else:
        raise ValueError(f"strand must be '+' or '-', got {strand!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="parquet or TSV of cores+coords+metadata (see docstring).")
    ap.add_argument("--genome-dir", required=True,
                    help="directory of per-genome FASTAs (walked recursively).")
    ap.add_argument("--out", required=True, help="output parquet.")
    ap.add_argument("--length", type=int, default=768,
                    help="context window length L in bp. MUST be a multiple of 128 "
                    "for NTv3 (default 768).")
    ap.add_argument("--contig-col", default="contig")
    ap.add_argument("--start-col", default="start")
    ap.add_argument("--end-col", default="end")
    ap.add_argument("--strand-col", default="strand")
    ap.add_argument("--genome-col", default="genome")
    ap.add_argument("--core-col", default="sequence",
                    help="optional known-core column for round-trip verification; "
                    "if absent, verification is skipped.")
    ap.add_argument("--max-unmapped-frac", type=float, default=0.01,
                    help="abort if more than this fraction of rows can't be mapped "
                    "to a genome/contig (default 0.01). Catches the 'only loaded "
                    "one genome' bug loudly instead of silently N-padding.")
    a = ap.parse_args()

    import pandas as pd

    if a.length % 128 != 0:
        sys.exit(f"ERROR: --length {a.length} must be a multiple of 128 for NTv3.")

    df = (pd.read_parquet(a.input) if a.input.endswith(".parquet")
          else pd.read_csv(a.input, sep="\t")).reset_index(drop=True)
    n = len(df)
    for c in (a.contig_col, a.start_col, a.end_col, a.strand_col, a.genome_col):
        if c not in df.columns:
            sys.exit(f"ERROR: input missing required column '{c}'. "
                     f"Have: {list(df.columns)}")

    genome_paths = find_genome_fastas(a.genome_dir)
    print(f"Found {len(genome_paths)} genome FASTAs under {a.genome_dir}")

    genome_cache = {}                       # genome_id -> {contig: seq}, loaded lazily
    contexts, cores, offsets = [], [], []
    unmapped, npad_rows, rt_ok, rt_fail = 0, 0, 0, 0

    for i, row in df.iterrows():
        gid = str(row[a.genome_col])
        contig = str(row[a.contig_col])
        strand = str(row[a.strand_col])
        start, end = int(row[a.start_col]), int(row[a.end_col])

        if gid not in genome_cache:
            path = genome_paths.get(gid)
            genome_cache[gid] = load_contigs(path) if path else {}
        contigs = genome_cache[gid]

        if contig not in contigs:
            unmapped += 1
            contexts.append(None); cores.append(None); offsets.append(-1)
            continue

        ctx, off, core = extract_context_window(contigs[contig], start, end,
                                                strand, a.length)
        contexts.append(ctx); cores.append(core); offsets.append(off)
        if "N" in ctx:
            npad_rows += 1

        # Round-trip check: the carved core must equal the known core (if given).
        if a.core_col in df.columns and pd.notna(row[a.core_col]):
            known = str(row[a.core_col]).upper()
            if core == known:
                rt_ok += 1
            else:
                rt_fail += 1

    if unmapped:
        frac = unmapped / n
        msg = (f"{unmapped}/{n} rows ({frac:.1%}) could not be mapped to a "
               f"genome/contig — check --genome-dir and that every row's "
               f"'{a.genome_col}' matches a FASTA stem (Bakta contig ids are "
               f"'contig_1' etc., unique only within a genome).")
        if frac > a.max_unmapped_frac:
            sys.exit("ERROR: " + msg + " Aborting (exceeds --max-unmapped-frac).")
        print("WARNING: " + msg)

    df = df.copy()
    df["sequence_context"] = contexts
    df["sequence"] = cores               # 81 bp core (overwrites any input core col)
    df["core_offset"] = offsets
    before = len(df)
    df = df[df["sequence_context"].notna()].reset_index(drop=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)

    print(f"Wrote {len(df)}/{before} rows to {a.out}")
    print(f"  context length: {a.length} bp (multiple of 128)")
    print(f"  rows with any N (context reached a contig end): {npad_rows}")
    if a.core_col in df.columns:
        total = rt_ok + rt_fail
        print(f"  round-trip core check: {rt_ok} ok, {rt_fail} FAIL")
        if total and rt_fail / total > 0.01:
            sys.exit(f"ERROR: {rt_fail}/{total} carved cores disagree with the genome — a "
                     f"coordinate/strand bug (this check exists to catch exactly that). Refusing "
                     f"to write a corrupt dataset. Verify the coordinate convention (0-based, "
                     f"half-open, forward-strand) and the strand column, then re-run.")
    print("\nNEXT: cluster + split on the `sequence` (81 bp core) column with the "
          "existing pipeline/datasets/clustering-and-splits/cdhit_splits.py, so val/test stay "
          "leakage-safe; both `sequence` and `sequence_context` ride the split.\n"
          "Then tokenise: CNN/ProkBERT read `sequence`; NTv3 reads "
          "`--sequence-col sequence_context` (encode.py).")


if __name__ == "__main__":
    main()
