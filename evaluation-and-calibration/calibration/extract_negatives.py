#!/usr/bin/env python3
"""
T26 negative-set extraction: convergent-intergenic and CDS-interior negatives.

Purpose
-------
Build two putative-negative sequence sets from paired genome FASTA + Bakta/NCBI GFF3
files, for use as a background / empirical-FDR null in T26 calibration (and reusable by
T27). Both sets are label=0. Convergent-intergenic windows are *putative* promoter-free
(convergent 3'-to-3' gene boundaries carry no promoter traffic BY ORIENTATION), not
confirmed negatives -- see NEGATIVES_NOTES.md.

Coordinate convention
---------------------
All coordinates in the output TSVs are **1-based, inclusive on both ends** -- identical to
the GFF3 convention. A 52-bp window reported as start=551, end=602 spans bases 551..602
inclusive (602 - 551 + 1 == 52). Sequences are extracted from the FASTA with the usual
0-based Python slice `seq[start-1:end]` internally; only 1-based values are ever written.

Window logic (per usable interval)
----------------------------------
Within each usable interval we place the MAXIMUM number of strictly non-overlapping W-bp
windows (floor(L / W)); a single random phase offset in [0, L mod W] is drawn (seeded) so
the tiling start is reproducibly randomized. No maximum is imposed by default: the goal is
to quantify all available negative data. Duplicate sequences are dropped within each output.

Two negative types
------------------
1. convergent_intergenic (CIG): between an upstream '+' CDS and a downstream '-' CDS
   (CDS+ ->  intergenic  <- CDS-). The raw gap is trimmed by `--trim` bp at each
   CDS-facing boundary, then any sub-interval overlapping ANY annotated feature
   (gene/CDS/pseudogene/rRNA/tRNA/tmRNA/ncRNA/...) is removed before windowing. The stored
   sequence is the reference (+ strand) as-is; there is no defined transcription direction.
2. CDS_interior (CDS): windows fully inside a CDS after trimming `--trim` bp off both ends,
   oriented in the annotated CDS direction ('+' -> reference; '-' -> reverse complement).

Outputs (written to --output-dir)
--------------------------------
  convergent_intergenic_negatives.tsv
  cds_negatives.tsv
  region_counts.tsv                 (# convergent regions & # CDS regions per genome)

Run `--dry-run` first to validate + preview without writing. Existing outputs are not
overwritten unless `--force` is given.
"""

import argparse
import csv
import os
import random
import sys
import urllib.parse
from collections import Counter, defaultdict

# Portable paths: override with PROMOTER_REPO / PROMOTER_RESULTS (or `source env.sh`).
# The fallbacks resolve from this file, so a plain clone works with no configuration.
PROMOTER_REPO = os.environ.get("PROMOTER_REPO") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PROMOTER_RESULTS = os.environ.get("PROMOTER_RESULTS") or os.path.join(
    PROMOTER_REPO, "infrastructure", "T3", "snakemake_scaffold", "results")


# ---- genome registry: filenames are retained even though C. phytofermentans is now
# ---- classified outside Clostridium (Lachnoclostridium); source_strain label per request.
GENOMES = [
    {"prefix": "CACE", "fna": "C_acetobutylicum.fna", "gff3": "C_acetobutylicum.gff3",
     "species": "Clostridium acetobutylicum", "strain": "acetobutylicum"},
    {"prefix": "CPHY", "fna": "C_phytofermentans.fna", "gff3": "C_phytofermentans.gff3",
     "species": "Lachnoclostridium phytofermentans", "strain": "phytofermentans"},
]

DEFAULT_DIR = os.path.join(PROMOTER_REPO, "evaluation-and-calibration/calibration")

VALID_BASES = set("ACGT")
_COMP = str.maketrans("ACGT", "TGCA")

# Feature types that do NOT occupy sequence for masking purposes (the replicon-spanning record).
NON_OCCUPYING = {"region", "databank_entry", "source"}


def revcomp(s):
    return s.translate(_COMP)[::-1]


def at_fraction(seq, window):
    return (seq.count("A") + seq.count("T")) / window


# --------------------------------------------------------------------------- parsing

def parse_fasta(path):
    """Return {record_id: (upper_sequence, description)}. record_id = first header token."""
    records = {}
    rid, desc, chunks = None, "", []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if rid is not None:
                    records[rid] = ("".join(chunks).upper(), desc)
                header = line[1:].strip()
                parts = header.split(None, 1)
                rid = parts[0]
                desc = parts[1] if len(parts) > 1 else ""
                chunks = []
            else:
                chunks.append(line.strip())
    if rid is not None:
        records[rid] = ("".join(chunks).upper(), desc)
    return records


def _attr(attrs, key):
    for field in attrs.split(";"):
        if "=" in field:
            k, v = field.split("=", 1)
            if k == key:
                return urllib.parse.unquote(v)
    return ""


def parse_gff3(path, skips):
    """Parse a GFF3. Return (features, cds_features).

    features   : list of (seqid, ftype, start, end)   -- everything occupying sequence.
    cds_features: list of dicts with seqid/start/end/strand/id/gene/locus_tag/product.
    Malformed lines are counted in `skips` and skipped.
    """
    features = []
    cds = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("##FASTA"):
                break  # GFF3 sequence section: everything after is FASTA, not features
            if line.startswith("#") or not line.strip():
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9:
                skips["gff_malformed"] += 1
                continue
            seqid, _src, ftype, s, e, _score, strand, _phase, attrs = p[:9]
            try:
                start, end = int(s), int(e)
            except ValueError:
                skips["gff_malformed"] += 1
                continue
            if start > end:
                skips["gff_malformed"] += 1
                continue
            if ftype not in NON_OCCUPYING:
                features.append((seqid, ftype, start, end))
            if ftype == "CDS":
                cds.append({
                    "seqid": seqid, "start": start, "end": end, "strand": strand,
                    "id": _attr(attrs, "locus_tag") or _attr(attrs, "ID"),
                    "gene": _attr(attrs, "gene"),
                    "locus_tag": _attr(attrs, "locus_tag"),
                    "product": _attr(attrs, "product"),
                })
    return features, cds


def replicon_type(description):
    d = description.lower()
    if "plasmid" in d:
        return "plasmid"
    if "chromosome" in d:
        return "chromosome"
    return ""


# --------------------------------------------------------------------------- interval math

def subtract(a, b, forbidden):
    """Free sub-intervals of [a,b] (1-based inclusive) after removing `forbidden` intervals."""
    free = []
    cur = a
    for s, e in forbidden:            # forbidden must be sorted by start
        if e < cur:
            continue
        if s > b:
            break
        if s > cur:
            free.append((cur, min(s - 1, b)))
        cur = max(cur, e + 1)
        if cur > b:
            break
    if cur <= b:
        free.append((cur, b))
    return [(s, e) for s, e in free if e >= s]


def place_windows(fs, fe, window, rng):
    """Max non-overlapping W-bp windows in [fs,fe] (1-based incl) with a random phase."""
    length = fe - fs + 1
    n = length // window
    if n == 0:
        return []
    offset = rng.randint(0, length - n * window)
    return [(fs + offset + k * window, fs + offset + k * window + window - 1) for k in range(n)]


# --------------------------------------------------------------------------- core

def process_genome(g, input_dir, window, trim, rng, skips):
    fna = os.path.join(input_dir, g["fna"])
    gff = os.path.join(input_dir, g["gff3"])
    for pth in (fna, gff):
        if not os.path.exists(pth):
            print(f"  [SKIP GENOME] missing input: {pth}", file=sys.stderr)
            return None

    replicons = parse_fasta(fna)
    features, cds_all = parse_gff3(gff, skips)

    fa_ids = set(replicons)
    gff_ids = {f[0] for f in features} | {c["seqid"] for c in cds_all}
    matched = fa_ids & gff_ids
    print(f"  replicon-id check: FASTA={len(fa_ids)} GFF3={len(gff_ids)} matched={len(matched)}")
    for rid in sorted(gff_ids - fa_ids):
        print(f"    [WARN] GFF3 seqid not in FASTA (features skipped): {rid}")
    for rid in sorted(fa_ids - gff_ids):
        print(f"    [note] FASTA replicon has no GFF3 features: {rid}")

    occ_by_rep = defaultdict(list)
    for seqid, _ftype, s, e in features:
        occ_by_rep[seqid].append((s, e))
    for rid in occ_by_rep:
        occ_by_rep[rid].sort()

    cds_by_rep = defaultdict(list)
    for c in cds_all:
        cds_by_rep[c["seqid"]].append(c)
    for rid in cds_by_rep:
        cds_by_rep[rid].sort(key=lambda c: (c["start"], c["end"]))

    conv_rows, cds_rows = [], []
    conv_seen, cds_seen = set(), set()
    counts = Counter()  # per-genome tallies

    for rid in sorted(fa_ids):
        seq, desc = replicons[rid]
        rtype = replicon_type(desc)
        occ = occ_by_rep.get(rid, [])
        cds_list = cds_by_rep.get(rid, [])

        # ---- convergent intergenic ----
        for i in range(len(cds_list) - 1):
            L, R = cds_list[i], cds_list[i + 1]
            if not (L["strand"] == "+" and R["strand"] == "-"):
                continue
            counts["conv_pairs_found"] += 1
            gap_s, gap_e = L["end"] + 1, R["start"] - 1
            if gap_e < gap_s:
                skips["convergent_no_gap"] += 1
                continue
            ts, te = gap_s + trim, gap_e - trim
            if te - ts + 1 < window:
                skips["convergent_gap_too_short_after_trim"] += 1
                continue
            frags = subtract(ts, te, occ)
            got = 0
            for fs, fe in frags:
                for ws, we in place_windows(fs, fe, window, rng):
                    sub = seq[ws - 1:we]
                    if len(sub) != window or not set(sub) <= VALID_BASES:
                        skips["convergent_unsupported_char"] += 1
                        continue
                    if sub in conv_seen:
                        skips["convergent_duplicate"] += 1
                        continue
                    conv_seen.add(sub)
                    conv_rows.append({
                        "sequence": sub, "label": 0, "negative_type": "convergent_intergenic",
                        "source_species": g["species"], "source_strain": g["strain"],
                        "replicon_id": rid, "replicon_type": rtype,
                        "start": ws, "end": we, "orientation": "convergent(+/-)",
                        "left_CDS_id": L["id"], "left_CDS_product": L["product"], "left_CDS_strand": L["strand"],
                        "right_CDS_id": R["id"], "right_CDS_product": R["product"], "right_CDS_strand": R["strand"],
                        "original_intergenic_start": gap_s, "original_intergenic_end": gap_e,
                        "original_intergenic_length": gap_e - gap_s + 1,
                        "trimmed_intergenic_start": ts, "trimmed_intergenic_end": te,
                        "AT_fraction": round(at_fraction(sub, window), 4),
                    })
                    got += 1
            if got:
                counts["conv_valid_regions"] += 1
                counts["conv_windows"] += got
            else:
                skips["convergent_region_no_window"] += 1

        # ---- CDS interior ----
        for c in cds_list:
            counts["cds_total"] += 1
            cs, ce = c["start"] + trim, c["end"] - trim
            if ce - cs + 1 < window:
                skips["cds_too_short_after_trim"] += 1
                continue
            windows = place_windows(cs, ce, window, rng)
            got = 0
            for ws, we in windows:
                sub = seq[ws - 1:we]
                if len(sub) != window or not set(sub) <= VALID_BASES:
                    skips["cds_unsupported_char"] += 1
                    continue
                oriented = sub if c["strand"] == "+" else revcomp(sub)
                if oriented in cds_seen:
                    skips["cds_duplicate"] += 1
                    continue
                cds_seen.add(oriented)
                cds_rows.append({
                    "sequence": oriented, "label": 0, "negative_type": "CDS_interior",
                    "source_species": g["species"], "source_strain": g["strain"],
                    "replicon_id": rid, "replicon_type": rtype,
                    "start": ws, "end": we, "strand": c["strand"],
                    "CDS_id": c["id"], "gene": c["gene"], "locus_tag": c["locus_tag"],
                    "product": c["product"], "CDS_start": c["start"], "CDS_end": c["end"],
                    "AT_fraction": round(at_fraction(oriented, window), 4),
                })
                got += 1
            if got:
                counts["cds_usable"] += 1
                counts["cds_windows"] += got

    # assign sequential IDs (sorted by replicon, position) after dedup
    conv_rows.sort(key=lambda r: (r["replicon_id"], r["start"], r["end"]))
    for n, r in enumerate(conv_rows, 1):
        r["sequence_id"] = f"{g['prefix']}_CIG_{n:06d}"
    cds_rows.sort(key=lambda r: (r["replicon_id"], r["start"], r["end"]))
    for n, r in enumerate(cds_rows, 1):
        r["sequence_id"] = f"{g['prefix']}_CDS_{n:06d}"

    return {"prefix": g["prefix"], "strain": g["strain"],
            "conv_rows": conv_rows, "cds_rows": cds_rows, "counts": counts}


# --------------------------------------------------------------------------- output

CONV_COLS = ["sequence_id", "sequence", "label", "negative_type", "source_species",
             "source_strain", "replicon_id", "replicon_type", "start", "end", "orientation",
             "left_CDS_id", "left_CDS_product", "left_CDS_strand", "right_CDS_id",
             "right_CDS_product", "right_CDS_strand", "original_intergenic_start",
             "original_intergenic_end", "original_intergenic_length",
             "trimmed_intergenic_start", "trimmed_intergenic_end", "AT_fraction"]

CDS_COLS = ["sequence_id", "sequence", "label", "negative_type", "source_species",
            "source_strain", "replicon_id", "replicon_type", "start", "end", "strand",
            "CDS_id", "gene", "locus_tag", "product", "CDS_start", "CDS_end", "AT_fraction"]


def write_tsv(path, cols, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def validate_rows(rows, window):
    for r in rows:
        assert len(r["sequence"]) == window, f"{r['sequence_id']}: len {len(r['sequence'])} != {window}"
        assert set(r["sequence"]) <= VALID_BASES, f"{r['sequence_id']}: non-ACGT"
        assert r["end"] - r["start"] + 1 == window, f"{r['sequence_id']}: coord span != {window}"


def preview(rows, cols, n=3):
    show = [c for c in cols if c not in ("sequence",)][:8]
    print("    " + " | ".join(show))
    for r in rows[:n]:
        print("    " + " | ".join(str(r.get(c, "")) for c in show))
        print(f"       seq={r['sequence']}  AT={r['AT_fraction']}")


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="T26 convergent-intergenic + CDS-interior negatives")
    ap.add_argument("--input-dir", default=DEFAULT_DIR, help="dir holding the .fna/.gff3 pairs")
    ap.add_argument("--output-dir", default=DEFAULT_DIR, help="dir to write the TSVs")
    # Single-genome override: point at ONE genome (e.g. the C. butyricum target) by giving its
    # filenames explicitly. When both are set, the hardcoded two-genome GENOMES list is replaced.
    ap.add_argument("--fna", help="single-genome FASTA filename (relative to --input-dir)")
    ap.add_argument("--gff3", help="single-genome GFF3 filename (relative to --input-dir)")
    ap.add_argument("--prefix", default="CBUT", help="sequence-id prefix for --fna/--gff3 mode")
    ap.add_argument("--species", default="Clostridium butyricum",
                    help="source_species label for --fna/--gff3 mode")
    ap.add_argument("--strain", default="butyricum",
                    help="source_strain label for --fna/--gff3 mode")
    ap.add_argument("--window", type=int, default=52, help="window length (bp)")
    ap.add_argument("--trim", type=int, default=50, help="bp trimmed from each CDS-facing boundary")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (shared across both classes)")
    ap.add_argument("--max-windows", type=int, default=None,
                    help="optional cap on windows per interval (default: no cap)")
    ap.add_argument("--dry-run", action="store_true", help="validate + preview, do not write TSVs")
    ap.add_argument("--force", action="store_true", help="overwrite existing output TSVs")
    ap.add_argument("--conv-out", default="convergent_intergenic_negatives.tsv",
                    help="filename for the convergent-intergenic negatives (the FDR null pool). "
                         "Relative names are placed under --output-dir; an absolute path is used "
                         "as-is. Set it per run so different genomes don't overwrite each other.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    skips = Counter()

    if bool(args.fna) ^ bool(args.gff3):
        sys.exit("ERROR: --fna and --gff3 must be given together.")
    genomes = GENOMES
    if args.fna and args.gff3:
        genomes = [{"prefix": args.prefix, "fna": args.fna, "gff3": args.gff3,
                    "species": args.species, "strain": args.strain}]

    conv_out = args.conv_out if os.path.isabs(args.conv_out) else os.path.join(args.output_dir, args.conv_out)
    cds_out = os.path.join(args.output_dir, "cds_negatives.tsv")
    counts_out = os.path.join(args.output_dir, "region_counts.tsv")

    if not args.dry_run and not args.force:
        for p in (conv_out, cds_out, counts_out):
            if os.path.exists(p):
                sys.exit(f"ERROR: {p} exists. Use --force to overwrite (or --dry-run).")

    print(f"window={args.window}bp trim={args.trim}bp seed={args.seed} "
          f"max_windows={args.max_windows} input_dir={args.input_dir}")

    all_conv, all_cds, region_rows = [], [], []
    genomes_processed = 0
    for g in genomes:
        print(f"\n[{g['prefix']}] {g['species']} (strain label: {g['strain']})")
        res = process_genome(g, args.input_dir, args.window, args.trim, rng, skips)
        if res is None:
            continue
        genomes_processed += 1
        c = res["counts"]
        # apply optional per-genome cap (post-hoc, keeps determinism of selection order)
        conv_rows, cds_rows = res["conv_rows"], res["cds_rows"]
        if args.max_windows is not None:
            conv_rows = conv_rows[:args.max_windows]
            cds_rows = cds_rows[:args.max_windows]
        validate_rows(conv_rows, args.window)
        validate_rows(cds_rows, args.window)
        all_conv += conv_rows
        all_cds += cds_rows
        region_rows.append({
            "genome_prefix": g["prefix"], "source_strain": g["strain"],
            "convergent_pairs_found": c["conv_pairs_found"],
            "convergent_valid_regions": c["conv_valid_regions"],
            "convergent_windows": len(conv_rows),
            "cds_total": c["cds_total"], "cds_usable": c["cds_usable"],
            "cds_windows": len(cds_rows),
        })
        print(f"  convergent: pairs_found={c['conv_pairs_found']} "
              f"valid_regions={c['conv_valid_regions']} windows={len(conv_rows)}")
        print(f"  CDS:        total={c['cds_total']} usable={c['cds_usable']} "
              f"windows={len(cds_rows)}")
        print("  preview convergent_intergenic:")
        preview(conv_rows, CONV_COLS)
        print("  preview CDS_interior:")
        preview(cds_rows, CDS_COLS)

    print("\n================ SUMMARY ================")
    print(f"genomes processed           : {genomes_processed}")
    print(f"convergent regions (valid)  : {sum(r['convergent_valid_regions'] for r in region_rows)}")
    print(f"convergent windows total    : {len(all_conv)}")
    print(f"usable CDSs                 : {sum(r['cds_usable'] for r in region_rows)}")
    print(f"CDS windows total           : {len(all_cds)}")
    print("per-genome convergent windows: " +
          ", ".join(f"{r['genome_prefix']}={r['convergent_windows']}" for r in region_rows))
    print("per-genome CDS windows       : " +
          ", ".join(f"{r['genome_prefix']}={r['cds_windows']}" for r in region_rows))
    print("skipped intervals (reason -> count):")
    for reason, n in sorted(skips.items()):
        print(f"    {reason:38s} {n}")

    if args.dry_run:
        print("\n[dry-run] no files written.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    write_tsv(conv_out, CONV_COLS, all_conv)
    write_tsv(cds_out, CDS_COLS, all_cds)
    with open(counts_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "genome_prefix", "source_strain", "convergent_pairs_found",
            "convergent_valid_regions", "convergent_windows", "cds_total",
            "cds_usable", "cds_windows"], delimiter="\t")
        w.writeheader()
        w.writerows(region_rows)
    print(f"\nwrote:\n  {conv_out}\n  {cds_out}\n  {counts_out}")


if __name__ == "__main__":
    main()
