#!/usr/bin/env python3
"""
build_positives_context.py — correct genomic-context extraction for the positive set.

Replaces create_positives.py's context step, fixing two real bugs:

  1. T7b (Hocq/Al-Bassam published TSS) got N-padded. Its coordinates are ENCODED in
     promoter_id ("{source}_{contig}_{pos}_{strand}") and its genome is the RefSeq assembly
     (e.g. NC_003030.1), which is NOT in the Bakta set. Here T7b is carved from the staged
     RefSeq FASTAs (pipeline/acquisition/tss-literature/ref/) using the EXACT extract_published_tss
     framing (81 bp, +1 at 0-based offset 60), so the core reproduces the stored sequence.

  2. Bakta names every contig contig_1..contig_N, so one global {contig: seq} dict COLLIDES
     across the 166 genomes and T13 context was pulled from a random genome. Here T13 is loaded
     PER GENOME (its `genome` column -> that genome's own FASTA) and the stored sequence is
     LOCATED by search in that genome, so both the genome and the framing are correct.

Round-trip: the carved core must equal the stored sequence. T7b hard-fails on >1% mismatch
(it must be exact); T13 is located by search so it matches by construction.

T8 is cross-species (E. coli / B. subtilis) with no coordinates for our genomes, so it stays
sequence-only (has_genomic_context = False), core fit to 81 bp, context N-padded.

Output: positives_with_context.parquet with columns
  sequence (81 bp core), sequence_context (L, multiple of 128), core_offset,
  has_genomic_context, source_type, label, species (+ promoter_id / genome / contig where known).
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Portable paths: override with PROMOTER_REPO / PROMOTER_RESULTS (or `source env.sh`).
# The fallbacks resolve from this file, so a plain clone works with no configuration.
PROMOTER_REPO = os.environ.get("PROMOTER_REPO") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PROMOTER_RESULTS = os.environ.get("PROMOTER_RESULTS") or os.path.join(PROMOTER_REPO, "workflow", "results")


REPO = PROMOTER_REPO
SCAF = f"{REPO}/workflow"
T7B_PARQUET = f"{SCAF}/inputs/published_tss_promoters.parquet"
T13_TSV = f"{SCAF}/results/T13/upstream_regions.tsv"
T8_PARQUET = f"{REPO}/pipeline/acquisition/curated-promoters/curated_positive_promoters.parquet"
T7_REF_DIR = f"{REPO}/pipeline/acquisition/tss-literature/ref"
BAKTA_DIR = f"{SCAF}/results/bakta/all_bakta_output"

CORE_LEN = 81
TSS_OFFSET = 60          # must match extract_published_tss.py
COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s):
    return s.translate(COMPLEMENT)[::-1]


def load_contigs(fasta_path):
    contigs, cid, buf = {}, None, []
    with open(fasta_path) as fh:
        for line in fh:
            if line.startswith(">"):
                if cid is not None:
                    contigs[cid] = "".join(buf).upper()
                cid = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if cid is not None:
        contigs[cid] = "".join(buf).upper()
    return contigs


def extract_context(contig_seq, start, end, strand, L):
    """A length-L window around the forward-strand core [start, end), strand-aware,
    N-padded only at contig ends. Returns (context, core_offset, core) on the reading strand."""
    core_len = end - start
    up = (L - core_len) // 2
    down = L - core_len - up
    w_start, w_end = start - up, end + down
    left_pad = max(0, -w_start)
    right_pad = max(0, w_end - len(contig_seq))
    frag = contig_seq[max(0, w_start): min(len(contig_seq), w_end)]
    frag = "N" * left_pad + frag + "N" * right_pad
    if len(frag) != L:                      # numerical guard; shouldn't trigger
        frag = (frag + "N" * L)[:L]
    if str(strand) in ("+", "+1", "plus", "1"):
        return frag, up, frag[up: up + core_len]
    ctx = revcomp(frag)
    return ctx, down, ctx[down: down + core_len]


def find_bakta_fastas(bakta_dir):
    """genome-id -> fasta path for the Bakta outputs (registers both the full file stem and the
    stem with a trailing _genomic stripped, so a `genome` column either way matches)."""
    paths = {}
    for root, _dirs, files in os.walk(bakta_dir):
        for f in files:
            if f.endswith(".fna"):
                stem = f[:-4]
                paths.setdefault(stem, os.path.join(root, f))
                if stem.endswith("_genomic"):
                    paths.setdefault(stem[:-len("_genomic")], os.path.join(root, f))
    return paths


def fit_core(seq, n=CORE_LEN):
    s = str(seq).upper()
    if len(s) < n:
        pad = n - len(s)
        return "N" * (pad // 2) + s + "N" * (pad - pad // 2)
    off = (len(s) - n) // 2
    return s[off: off + n]


# ─────────────────────────────────────────────────────────────────────
def process_t7b(L):
    if not os.path.exists(T7B_PARQUET):
        print("T7b: parquet missing, skipping"); return []
    df = pd.read_parquet(T7B_PARQUET)

    def parse_pid(pid):
        p = str(pid).split("_")
        if len(p) < 4:
            return "", None, None
        return "_".join(p[1:-2]), p[-2], p[-1]

    # one contig dict across the RefSeq genomes — their accessions are unique, so NO collision
    cd = {}
    for f in sorted(Path(T7_REF_DIR).glob("*.fna")):
        cd.update(load_contigs(str(f)))
    print(f"T7b: {len(df)} rows; indexed {len(cd)} RefSeq contigs from {T7_REF_DIR}")

    rows, rt_ok, rt_fail, unmapped = [], 0, 0, 0
    unmapped_contigs = {}
    for _, r in df.iterrows():
        contig, pos, strand = parse_pid(r["promoter_id"])
        meta = {"source_type": "t7b_published_tss", "label": int(r.get("label", 1)),
                "species": r.get("species", ""), "promoter_id": r.get("promoter_id", ""),
                "contig": contig, "strand": strand}
        if not contig or pos is None or contig not in cd:
            unmapped += 1
            unmapped_contigs[contig] = unmapped_contigs.get(contig, 0) + 1
            core = fit_core(r.get("sequence", ""))
            rows.append({**meta, "sequence": core, "sequence_context": (core + "N" * L)[:L],
                         "core_offset": (L - CORE_LEN) // 2, "has_genomic_context": False})
            continue
        i = int(pos) - 1
        if strand == "+":
            cstart, cend = i - TSS_OFFSET, i + (CORE_LEN - TSS_OFFSET)     # [i-60, i+21)
        else:
            cstart, cend = i - (CORE_LEN - 1 - TSS_OFFSET), i + TSS_OFFSET + 1  # [i-20, i+61)
        ctx, off, core = extract_context(cd[contig], cstart, cend, strand, L)
        if core == str(r.get("sequence", "")).upper():
            rt_ok += 1
        else:
            rt_fail += 1
        rows.append({**meta, "sequence": core, "sequence_context": ctx,
                     "core_offset": off, "has_genomic_context": True})
    total_rt = rt_ok + rt_fail
    print(f"T7b: mapped {total_rt}, unmapped {unmapped}; round-trip {rt_ok} ok / {rt_fail} FAIL")
    if unmapped_contigs:
        print(f"     unmapped contigs (need their genome in {T7_REF_DIR}): "
              f"{dict(sorted(unmapped_contigs.items(), key=lambda x:-x[1])[:8])}")
    if total_rt and rt_fail / total_rt > 0.01:
        sys.exit(f"ERROR: T7b round-trip failure {rt_fail}/{total_rt} (>1%). The carved core does not "
                 f"match the stored sequence — framing or genome mismatch. Refusing to write.")
    return rows


def process_t13(L):
    if not os.path.exists(T13_TSV):
        print("T13: tsv missing, skipping"); return []
    df = pd.read_csv(T13_TSV, sep="\t")
    gcol = next((c for c in ["genome", "genome_id", "assembly"] if c in df.columns), None)
    scol = next((c for c in ["upstream_sequence", "sequence", "seq"] if c in df.columns), None)
    print(f"T13: {len(df)} rows; genome-col={gcol!r} seq-col={scol!r} cols={list(df.columns)}")
    if gcol is None or scol is None:
        print("T13: cannot find genome/sequence columns — leaving T13 sequence-only.")
        return [{"source_type": "upstream_region", "label": 1, "species": "",
                 "sequence": fit_core(r[scol] if scol else ""),
                 "sequence_context": (fit_core(r[scol] if scol else "") + "N" * L)[:L],
                 "core_offset": (L - CORE_LEN) // 2, "has_genomic_context": False}
                for _, r in df.iterrows()]

    fastas = find_bakta_fastas(BAKTA_DIR)
    rows, found, missing = [], 0, 0
    # process grouped by genome so each FASTA is loaded once and freed (bounded memory)
    for gid, g in df.groupby(gcol):
        path = fastas.get(str(gid))
        contigs = load_contigs(path) if path else {}
        for _, r in g.iterrows():
            seq = str(r[scol]).upper()
            meta = {"source_type": "upstream_region", "label": 1,
                    "species": r.get("species", ""), "genome": gid}
            loc = None
            for cid, cseq in contigs.items():
                p = cseq.find(seq)
                if p >= 0:
                    loc = (cid, p, "+"); break
                p = cseq.find(revcomp(seq))
                if p >= 0:
                    loc = (cid, p, "-"); break
            if loc is None:
                missing += 1
                core = fit_core(seq)
                rows.append({**meta, "sequence": core, "sequence_context": (core + "N" * L)[:L],
                             "core_offset": (L - CORE_LEN) // 2, "has_genomic_context": False})
                continue
            found += 1
            cid, p, strand = loc
            center = p + len(seq) // 2                       # 81 bp core at the centre of the located region
            cstart, cend = center - CORE_LEN // 2, center - CORE_LEN // 2 + CORE_LEN
            ctx, off, core = extract_context(contigs[cid], cstart, cend, strand, L)
            rows.append({**meta, "contig": cid, "strand": strand, "sequence": core,
                         "sequence_context": ctx, "core_offset": off, "has_genomic_context": True})
    print(f"T13: located {found} in-genome / {missing} not found (kept sequence-only)")
    return rows


def process_t8(L):
    if not os.path.exists(T8_PARQUET):
        print("T8: parquet missing, skipping"); return []
    df = pd.read_parquet(T8_PARQUET)
    print(f"T8: {len(df)} rows (cross-species, sequence-only — no context)")
    scol = next((c for c in ["sequence", "seq"] if c in df.columns), None)
    rows = []
    for _, r in df.iterrows():
        core = fit_core(r.get(scol, "") if scol else "")
        rows.append({"source_type": "t8_curated_positive", "label": int(r.get("label", 1)),
                     "species": r.get("species", ""), "sequence": core,
                     "sequence_context": (core + "N" * L)[:L],
                     "core_offset": (L - CORE_LEN) // 2, "has_genomic_context": False})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--length", type=int, default=384, help="context window bp (multiple of 128).")
    ap.add_argument("--out", default=f"{REPO}/pipeline/datasets/positives-with-context/output/positives_with_context.parquet")
    a = ap.parse_args()
    if a.length % 128 != 0:
        sys.exit(f"--length {a.length} must be a multiple of 128 for NTv3.")

    rows = process_t7b(a.length) + process_t13(a.length) + process_t8(a.length)
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    df.to_csv(a.out.replace(".parquet", ".csv"), index=False)

    print("\n===== SUMMARY =====")
    print(f"total rows: {len(df)}")
    print(df.groupby("source_type")["has_genomic_context"].value_counts())
    ctx_len = df["sequence_context"].astype(str).str.len()
    print(f"context length: min={ctx_len.min()} max={ctx_len.max()} (should all == {a.length})")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
