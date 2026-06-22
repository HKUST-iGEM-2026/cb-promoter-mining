#!/usr/bin/env python3
"""
T7b — turn a published in-genus *Clostridium* TSS table into 81 bp promoter POSITIVES
that feed the training split (T15). Production version (was extract_published_tss_template.py).

WHY (CLAUDE.md §7): the fine-tune's training positives currently contain NO verified in-genus
Clostridium promoter. The genuine in-genus signal we have but don't use is the PUBLISHED TSS
from in-genus 5'-end studies: Hocq 2022 (*C. acetobutylicum*) and Al-Bassam 2018
(*C. ljungdahlii*). This ingests those tables -> verified in-genus positives in the SAME schema
as T8's parquet, so T15 concatenates them. See PUBLISHED_TSS_README.md for where to get the tables.

INPUT : a published TSS table (CSV/TSV) with a genomic position + strand column, plus the matching
        reference-genome FASTA. Column names are configurable (papers differ).
OUTPUT: a parquet with columns
          species, promoter_id, sequence, strand, sigma_factor, source_db, in_genus, label(=1)

WINDOW FRAMING (must match RegulonDB / PT5 / PT6): 81 bp, TSS (+1) at 0-based offset 60
(1-based position 61) -> genomic window [-60,+20] around the TSS.

REFERENCE GENOMES (CLAUDE.md §8):
  C. acetobutylicum ATCC 824  -> GCF_000008765.1 (chromosome NC_003030.1 + pSOL1 NC_001988.2)  [2 replicons -> --contig-col REQUIRED]
  C. ljungdahlii    DSM 13528 -> GCF_000143685.1 (NC_014328.1)                                  [single replicon]

EXAMPLE
  python extract_published_tss.py \
    --tss-table albassam2018_clj_tss.tsv --genome GCF_000143685.1_ASM14368v1_genomic.fna \
    --species "Clostridium ljungdahlii" --source AlBassam2018 \
    --pos-col TSS_position --strand-col strand --out published_tss_promoters.parquet
"""
import argparse
import sys
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

WINDOW = 81
TSS_OFFSET = 60          # 0-based offset of the +1 inside the 81 bp window (RegulonDB / PT5 framing)
OUT_COLS = ["species", "promoter_id", "sequence", "strand", "sigma_factor",
            "source_db", "in_genus", "label"]


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
    ap.add_argument("--species", required=True, help='e.g. "Clostridium ljungdahlii"')
    ap.add_argument("--source", required=True, help="citation tag, e.g. Hocq2022 / AlBassam2018")
    ap.add_argument("--sep", default="\t")
    ap.add_argument("--pos-col", default="TSS_position")
    ap.add_argument("--strand-col", default="strand")
    ap.add_argument("--contig-col", default=None,
                    help="column naming the contig/replicon; REQUIRED if the genome has >1 record")
    ap.add_argument("--sigma-col", default=None, help="optional sigma-factor column")
    ap.add_argument("--out", required=True, help="output parquet")
    args = ap.parse_args()

    tss = pd.read_csv(args.tss_table, sep=args.sep, comment="#")
    genome = load_genome(args.genome)
    default_contig = next(iter(genome))

    # GUARD (fixes the silent multi-replicon bug): if the genome has >1 replicon, a contig column
    # is mandatory — otherwise every TSS would be sliced out of the first replicon only.
    if len(genome) > 1 and not args.contig_col:
        sys.exit(f"ERROR: genome has {len(genome)} replicons ({', '.join(genome)}) — "
                 f"--contig-col is REQUIRED so each TSS maps to the right replicon.")
    for col in (args.pos_col, args.strand_col):
        if col not in tss.columns:
            sys.exit(f"ERROR: column '{col}' not in the TSS table. Columns are: {list(tss.columns)}")

    rows, dropped = [], {"bad_pos": 0, "unmapped_contig": 0, "contig_end": 0, "bad_strand": 0, "non_acgt": 0}
    for _, r in tss.iterrows():
        contig = str(r[args.contig_col]) if args.contig_col else default_contig
        strand = norm_strand(r[args.strand_col])
        if strand not in {"+", "-"}:
            dropped["bad_strand"] += 1; continue
        try:
            pos = int(r[args.pos_col])
        except (ValueError, TypeError, KeyError):
            dropped["bad_pos"] += 1; continue
        if contig not in genome:
            dropped["unmapped_contig"] += 1; continue
        seq = window_for(genome[contig], pos, strand)
        if seq is None or len(seq) != WINDOW:        # contig-end exclusion
            dropped["contig_end"] += 1; continue
        if set(seq) - set("ACGT"):                   # drop windows spanning assembly gaps (N etc.)
            dropped["non_acgt"] += 1; continue
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

    out = pd.DataFrame(rows, columns=OUT_COLS)
    before = len(out)
    out = out.drop_duplicates(subset=["promoter_id"]).reset_index(drop=True)   # dedupe per-condition repeats
    # fail LOUD on a zero-row result (the classic contig-id-parity failure, e.g. NC_014328 vs NC_014328.1)
    assert len(out) > 0, (
        f"no TSS mapped to a clean 81 bp window — check --contig-col / accession parity "
        f"(genome replicons: {list(genome)}); drop reasons: {dropped}")
    out.to_parquet(args.out, index=False)
    print(f"{args.species}: wrote {len(out)} verified in-genus positives -> {args.out} "
          f"({before - len(out)} duplicate ids removed; dropped {dropped}). Concatenate with the T8 parquet in T15.")


if __name__ == "__main__":
    main()
