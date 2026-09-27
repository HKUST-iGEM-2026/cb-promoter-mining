#!/usr/bin/env python3
"""
build_context_sources.py — per-source context files for the wide-context (NTv3 / CNN-v2-384)
experiment. Each output = the ORIGINAL source columns (so cdhit_splits.load_source still detects
it) PLUS sequence_context (L bp, multiple of 128) and core_offset (index of the 81 bp core inside
the context). Rows whose context can't be recovered are dropped (reported), so every emitted row
carries REAL context and cdhit_splits' all-or-nothing guard passes.

Outputs (into --out):
  published_tss_ctx.parquet     T7b — exact extract_published_tss framing, carved from the RefSeq
                                FASTAs (T7/ref/). Detected by cdhit_splits via `in_genus`+`sequence`.
  upstream_regions_ctx.parquet  T13 — located by search in the row's own Bakta genome (per-genome,
                                no contig_1 collision). Keeps `upstream_sequence` (T13 detection).
  negatives_struct_ctx.parquet  T14 CDS/intergenic/downstream — located by search per-genome. Keeps
                                `negative_class`. SRS is EXCLUDED (cdhit_splits regenerates it
                                flank-preserving from the positives' context).

Next:
  python dataset-construction/clustering-and-splits/cdhit_splits.py \
      --positives <out>/published_tss_ctx.parquet <out>/upstream_regions_ctx.parquet \
      --negatives <out>/negatives_struct_ctx.parquet \
      --identity 0.6 --window 81 --out results/T15_ctx
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Portable paths: override with PROMOTER_REPO / PROMOTER_RESULTS (or `source env.sh`).
# The fallbacks resolve from this file, so a plain clone works with no configuration.
PROMOTER_REPO = os.environ.get("PROMOTER_REPO") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PROMOTER_RESULTS = os.environ.get("PROMOTER_RESULTS") or os.path.join(
    PROMOTER_REPO, "infrastructure", "T3", "snakemake_scaffold", "results")


REPO = PROMOTER_REPO
SCAF = f"{REPO}/infrastructure/T3/snakemake_scaffold"
T7B_PARQUET = f"{SCAF}/inputs/published_tss_promoters.parquet"
T13_TSV = f"{SCAF}/results/T13/upstream_regions.tsv"
T14_TSV = f"{SCAF}/results/T14/negative_set.tsv"
T7_REF_DIR = f"{REPO}/data-acquisition-and-curation/tss-literature/ref"
BAKTA_DIR = f"{SCAF}/results/bakta/all_bakta_output"

CORE_LEN = 81
TSS_OFFSET = 60
COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s):
    return s.translate(COMPLEMENT)[::-1]


def load_contigs(path):
    contigs, cid, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if cid is not None:
                    contigs[cid] = "".join(buf).upper()
                cid = line[1:].split()[0]; buf = []
            else:
                buf.append(line.strip())
    if cid is not None:
        contigs[cid] = "".join(buf).upper()
    return contigs


def extract_context(contig_seq, start, end, strand, L):
    core_len = end - start
    up = (L - core_len) // 2
    down = L - core_len - up
    w_start, w_end = start - up, end + down
    left_pad = max(0, -w_start)
    right_pad = max(0, w_end - len(contig_seq))
    frag = contig_seq[max(0, w_start): min(len(contig_seq), w_end)]
    frag = "N" * left_pad + frag + "N" * right_pad
    if len(frag) != L:
        frag = (frag + "N" * L)[:L]
    if str(strand) in ("+", "+1", "plus", "1"):
        return frag, up, frag[up: up + core_len]
    ctx = revcomp(frag)
    return ctx, down, ctx[down: down + core_len]


def find_bakta_fastas(bakta_dir):
    paths = {}
    for root, _d, files in os.walk(bakta_dir):
        for f in files:
            if f.endswith(".fna"):
                stem = f[:-4]
                paths.setdefault(stem, os.path.join(root, f))
                if stem.endswith("_genomic"):
                    paths.setdefault(stem[:-len("_genomic")], os.path.join(root, f))
    return paths


def locate(seq, contigs):
    """Find seq (reading strand) in any contig -> (contig, forward_start, strand)."""
    for cid, cseq in contigs.items():
        p = cseq.find(seq)
        if p >= 0:
            return cid, p, "+"
        p = cseq.find(revcomp(seq))
        if p >= 0:
            return cid, p, "-"
    return None


# ── T7b: exact framing, RefSeq genomes ────────────────────────────────
def build_t7b(L):
    df = pd.read_parquet(T7B_PARQUET).reset_index(drop=True)
    cd = {}
    for f in sorted(Path(T7_REF_DIR).glob("*.fna")):
        cd.update(load_contigs(str(f)))
    ctx_col, off_col, keep, rt_ok, rt_fail = [], [], [], 0, 0
    for _, r in df.iterrows():
        p = str(r["promoter_id"]).split("_")
        contig, pos, strand = "_".join(p[1:-2]), int(p[-2]), p[-1]
        if contig not in cd:
            ctx_col.append(None); off_col.append(-1); keep.append(False); continue
        i = pos - 1
        cs, ce = (i - TSS_OFFSET, i + (CORE_LEN - TSS_OFFSET)) if strand == "+" \
            else (i - (CORE_LEN - 1 - TSS_OFFSET), i + TSS_OFFSET + 1)
        ctx, off, core = extract_context(cd[contig], cs, ce, strand, L)
        rt_ok += core == str(r["sequence"]).upper()
        rt_fail += core != str(r["sequence"]).upper()
        ctx_col.append(ctx); off_col.append(off); keep.append(True)
    df["sequence_context"] = ctx_col; df["core_offset"] = off_col
    out = df[keep].reset_index(drop=True)
    print(f"T7b: {len(out)}/{len(df)} kept; round-trip {rt_ok} ok / {rt_fail} FAIL")
    if rt_ok + rt_fail and rt_fail / (rt_ok + rt_fail) > 0.01:
        sys.exit("ERROR: T7b round-trip >1% mismatch — refusing.")
    return out


# ── T13 / T14: locate-by-search, per genome (no contig_1 collision) ───
def build_by_search(path, seq_col, L, exclude_srs=False):
    df = (pd.read_csv(path, sep="\t") if path.endswith((".tsv", ".txt"))
          else pd.read_parquet(path)).reset_index(drop=True)
    if exclude_srs and "negative_class" in df.columns:
        df = df[df["negative_class"] != "SRS"].reset_index(drop=True)
    fastas = find_bakta_fastas(BAKTA_DIR)
    ctx_col = [None] * len(df); off_col = [-1] * len(df); keep = [False] * len(df)
    found = 0
    for gid, g in df.groupby("genome"):
        contigs = load_contigs(fastas[str(gid)]) if str(gid) in fastas else {}
        for idx, r in g.iterrows():
            seq = str(r[seq_col]).upper()
            loc = locate(seq, contigs) if contigs else None
            if loc is None:
                continue
            cid, p, strand = loc
            center = p + len(seq) // 2
            cs, ce = center - CORE_LEN // 2, center - CORE_LEN // 2 + CORE_LEN
            ctx, off, _core = extract_context(contigs[cid], cs, ce, strand, L)
            ctx_col[idx] = ctx; off_col[idx] = off; keep[idx] = True; found += 1
    df["sequence_context"] = ctx_col; df["core_offset"] = off_col
    out = df[keep].reset_index(drop=True)
    print(f"{os.path.basename(path)}: located {found}/{len(df)} "
          f"({'excl. SRS' if exclude_srs else 'positives'})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--length", type=int, default=384)
    ap.add_argument("--out", default=f"{REPO}/positives_with_context/context_sources")
    a = ap.parse_args()
    if a.length % 128:
        sys.exit(f"--length {a.length} must be a multiple of 128.")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    build_t7b(a.length).to_parquet(out / "published_tss_ctx.parquet", index=False)
    build_by_search(T13_TSV, "upstream_sequence", a.length).to_parquet(out / "upstream_regions_ctx.parquet", index=False)
    build_by_search(T14_TSV, "sequence", a.length, exclude_srs=True).to_parquet(out / "negatives_struct_ctx.parquet", index=False)
    print(f"\nwrote 3 per-source context files to {out}")
    print("next: cdhit_splits.py --positives published_tss_ctx.parquet upstream_regions_ctx.parquet "
          "--negatives negatives_struct_ctx.parquet --identity 0.6 --window 81 --out results/T15_ctx")


if __name__ == "__main__":
    main()
