#!/usr/bin/env python3
"""
T7 (published-TSS branch) — turn published in-genus *Clostridium* TSS tables into 81 bp
promoter POSITIVES that feed the training split (T15).

WHY THIS EXISTS (see CLAUDE.md §7 open decision):
The fine-tune's training positives (T15 <- T8, T10, T13, T14) currently contain NO verified
in-genus Clostridium promoter — T8 is cross-species, T10 is *Clostridioides difficile* (out of
taxon 1485) and unbuilt, T13 is inferred. The genuine in-genus signal we already have but don't
use for training is the PUBLISHED TSS from in-genus 5'-end studies (CLAUDE.md §8): Hocq 2022
(*C. acetobutylicum*) and Al-Bassam 2018 (*C. ljungdahlii*). This script ingests those tables and
emits verified in-genus positives in the SAME schema as T8's parquet, so T15 concatenates them.

INPUT : a published TSS table (CSV/TSV) with at least a genome position and a strand, plus the
        matching reference-genome FASTA. Column names are configurable (papers differ).
OUTPUT: a parquet with columns
          species, promoter_id, sequence, strand, sigma_factor, source_db, in_genus, label(=1)
        matching data-acquisition-and-curation/T8/curated_positive_promoters.parquet.

WINDOW FRAMING (must match RegulonDB / PT5 / PT6): 81 bp, TSS (+1) at 0-based offset 60
(1-based position 61) -> genomic window [-60,+20] around the TSS. RE-CONFIRM this offset parity
against PT5/PT6 before trusting downstream scores — an off-by-one here silently degrades the model.

REFERENCE GENOMES (CLAUDE.md §8 — confirm the accession against each paper's Methods):
  C. acetobutylicum ATCC 824  -> GCF_000008765.1 (NC_003030.1 + pSOL1 NC_001988.2)
  C. ljungdahlii    DSM 13528 -> GCF_000143685.1 (NC_014328.1)

EXAMPLE
  python extract_published_tss_template.py \
    --tss-table hocq2022_cac_tss.tsv --sep '\t' \
    --genome GCF_000008765.1.fna --contig-col replicon \
    --species "Clostridium acetobutylicum" --source Hocq2022 \
    --pos-col TSS_position --strand-col strand --sigma-col sigma \
    --out inputs/published_tss_promoters.parquet
"""
import argparse
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

WINDOW = 81
TSS_OFFSET = 60          # 0-based offset of the +1 inside the 81 bp window (RegulonDB / PT5 framing)


def norm_strand(s):
    s = str(s).strip().lower()
    if s in {"+", "1", "f", "fwd", "forward", "plus", "sense"}:        return "+"
    if s in {"-", "-1", "r", "rev", "reverse", "minus", "antisense"}:  return "-"
    return s  # leave as-is; row will be dropped


def load_genome(path):
    """contig_id -> uppercase sequence."""
    return {rec.id: str(rec.seq).upper() for rec in SeqIO.parse(path, "fasta")}


def window_for(seq, pos1, strand):
    """81 bp window around a 1-based TSS, TSS at offset 60. None if it runs off the contig end."""
    i = pos1 - 1                                              # 0-based TSS index
    if strand == "+":
        start, end = i - TSS_OFFSET, i + (WINDOW - TSS_OFFSET)        # [i-60, i+21)
        if start < 0 or end > len(seq):
            return None
        return seq[start:end]
    if strand == "-":
        # take the genomic span, then reverse-complement so the TSS lands back at offset 60
        start, end = i - (WINDOW - 1 - TSS_OFFSET), i + TSS_OFFSET + 1  # [i-20, i+61)
        if start < 0 or end > len(seq):
            return None
        return str(Seq(seq[start:end]).reverse_complement())
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tss-table", required=True, help="published TSS table (CSV/TSV)")
    ap.add_argument("--genome", required=True, help="reference FASTA the TSS coordinates map to")
    ap.add_argument("--species", required=True, help='e.g. "Clostridium acetobutylicum"')
    ap.add_argument("--source", required=True, help="citation tag, e.g. Hocq2022 / AlBassam2018")
    ap.add_argument("--sep", default="\t")
    ap.add_argument("--pos-col", default="TSS_position")
    ap.add_argument("--strand-col", default="strand")
    ap.add_argument("--contig-col", default=None,
                    help="column naming the contig/replicon; if omitted, uses the first FASTA record")
    ap.add_argument("--sigma-col", default=None, help="optional sigma-factor column")
    ap.add_argument("--out", required=True, help="output parquet")
    args = ap.parse_args()

    tss = pd.read_csv(args.tss_table, sep=args.sep, comment="#")
    genome = load_genome(args.genome)
    default_contig = next(iter(genome))

    rows, dropped = [], 0
    for _, r in tss.iterrows():
        contig = str(r[args.contig_col]) if args.contig_col else default_contig
        strand = norm_strand(r[args.strand_col])
        try:
            pos = int(r[args.pos_col])
        except (ValueError, TypeError, KeyError):
            dropped += 1
            continue
        if contig not in genome:
            dropped += 1
            continue
        seq = window_for(genome[contig], pos, strand)
        if seq is None or len(seq) != WINDOW:        # contig-end exclusion (§7 convention)
            dropped += 1
            continue
        rows.append({
            "species": args.species,
            "promoter_id": f"{args.source}_{contig}_{pos}_{strand}",
            "sequence": seq,
            "strand": strand,
            "sigma_factor": (str(r[args.sigma_col])
                             if args.sigma_col and args.sigma_col in tss.columns else "Unknown"),
            "source_db": args.source,
            "in_genus": True,           # genus Clostridium (taxon 1485) — the whole point
            "label": 1,
        })

    out = pd.DataFrame(rows)
    out.to_parquet(args.out, index=False)
    print(f"{args.species}: wrote {len(out)} verified in-genus promoter positives -> {args.out} "
          f"({dropped} rows dropped: unmapped contig / contig-end / bad strand). "
          f"Concatenate with the T8 parquet in T15.")


if __name__ == "__main__":
    main()
