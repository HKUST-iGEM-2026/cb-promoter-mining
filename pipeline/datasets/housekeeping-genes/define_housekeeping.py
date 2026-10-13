#!/usr/bin/env python3
"""
T11 - Housekeeping gene set definition.
Owner: Seonyu
Due: 12 Jun

Selects conserved housekeeping-family product annotations from Bakta (T6).
Their upstream regions provide weaker silver promoter labels, not demonstrated
constitutive-promoter examples (consumed by T12 clustering and T13 extraction).

Design (see README.md for full rationale):
  * Selection is by PRODUCT description, not the gene-symbol field. The Bakta
    v6.0 *light* DB leaves gene symbols sparse (~27% of CDS are hypothetical),
    but products are populated for the conserved housekeeping families.
  * Curated product-family catalogue for housekeeping-associated silver labels.
    Sigma factors, heat-shock chaperones and SOS RecA are deliberately excluded;
    this selection does not establish condition-independent expression.
  * Only CDS features are considered (rRNA/tRNA promoters are atypical; see README).

Usage:
    python define_housekeeping.py --tsv-dir <dir of *.tsv> --out-dir <dir>
"""

import argparse
import glob
import os
import re
import csv
from collections import defaultdict, Counter

# ---------------------------------------------------------------------------
# Housekeeping catalog: (category, tier, allow-regex) applied to the Product
# field. A match is kept only if it does NOT also hit GLOBAL_DENY.
# All regexes are case-insensitive.
# ---------------------------------------------------------------------------

# Global deny list - excludes modifying enzymes, fragments, and unsure calls
# that would otherwise be caught by a family pattern (e.g. the ribosomal-protein
# methyltransferases / acetylases, or "...-binding protein" partners).
GLOBAL_DENY = re.compile(
    r"(?i)("
    r"domain-containing|"
    r"\bputative\b|hypothetical|"
    r"methyltransferase|methylthiotransferase|"
    r"acetyltransferase|acetylase|"
    r"-binding protein|binding protein\b|"
    r"family protein\b|[Ff]lagell|"
    r"\bRim[OI]\b|maturation|biogenesis|assembly|"
    r"pseudouridine|glutamate ligase|"
    r"\(ATP\)"
    r")"
)

CATALOG = [
    # ---- Tier 1: core housekeeping-associated product families, not expression measurements
    # Structural ribosomal proteins. The light DB mostly uses the modern
    # (Ban et al. 2014) "<Small|Large> ribosomal subunit protein <u/b/e><S/L>N"
    # nomenclature (~50 per genome); the legacy "30S/50S ribosomal protein SN"
    # forms appear in a minority of genomes. Modifying enzymes / biogenesis /
    # assembly / -binding / domain-containing variants are removed by GLOBAL_DENY.
    ("ribosomal_protein", 1, r"(?i)^(?:Small|Large) ribosomal subunit protein [ube][SL]\d"),
    ("ribosomal_protein", 1, r"(?i)\b(?:30S|50S) ribosomal protein [SL]\d"),
    ("ribosomal_protein", 1, r"(?i)^Ribosomal protein [SL]\d+[A-Za-z]?$"),
    # Translation elongation / initiation factors (core set).
    ("translation_factor", 1, r"(?i)^Elongation factor (?:Tu|G|Ts|P)\b"),
    ("translation_factor", 1, r"(?i)^Translation initiation factor IF-[123]\b"),
    # RNA polymerase core subunits (alpha/beta/beta'/omega) - NOT sigma factors.
    ("rnap_core", 1, r"(?i)^DNA-directed RNA polymerase subunit (?:alpha|beta'?|omega)$"),
    # Aminoacyl-tRNA synthetases (Bakta names them "<aa>--tRNA ligase").
    ("aminoacyl_trna_synthetase", 1, r"(?i)--tRNA ligase"),
    # ATP synthase (F-type and V/A-type); energy-metabolism annotation, not proven constitutive regulation.
    ("atp_synthase", 1, r"(?i)(?:^|\bV-type (?:sodium )?)?ATP synthase\b"),

    # ---- Tier 2: additional housekeeping-associated families
    # Central glycolysis (the core fermentative energy pathway in Clostridium).
    ("glycolysis", 2, r"(?i)^Glyceraldehyde-3-phosphate dehydrogenase"),
    ("glycolysis", 2, r"(?i)^Enolase$"),
    ("glycolysis", 2, r"(?i)^Phosphoglycerate kinase$"),
    ("glycolysis", 2, r"(?i)^Triosephosphate isomerase$"),
    ("glycolysis", 2, r"(?i)phosphoglycerate mutase"),
    ("glycolysis", 2, r"(?i)^Pyruvate kinase$"),
    ("glycolysis", 2, r"(?i)6-phosphofructokinase$"),
    ("glycolysis", 2, r"(?i)^Glucose-6-phosphate isomerase$"),
    ("glycolysis", 2, r"(?i)Fructose-bisphosphate aldolase$"),
    # DNA replication & topology core.
    ("dna_replication", 2, r"(?i)^Chromosomal replication initiator protein DnaA$"),
    ("dna_replication", 2, r"(?i)^DNA gyrase subunit [AB]$"),
    ("dna_replication", 2, r"(?i)^DNA polymerase III subunit"),
    ("dna_replication", 2, r"(?i)^Single-stranded DNA-binding protein"),
    ("dna_replication", 2, r"(?i)^DNA primase$"),
    ("dna_replication", 2, r"(?i)^Replicative DNA helicase$"),
    ("dna_replication", 2, r"(?i)^DNA topoisomerase"),
    ("dna_replication", 2, r"(?i)^DNA ligase$"),
    # Core transcription elongation / termination factors.
    ("transcription_factor", 2, r"(?i)^Transcription termination/antitermination protein Nus[AG]$"),
    ("transcription_factor", 2, r"(?i)^Transcription termination factor Rho$"),
    ("transcription_factor", 2, r"(?i)Antitermination protein NusB$"),
    ("transcription_factor", 2, r"(?i)^Transcription elongation factor GreA$"),
]

# Note for ssb / dna_replication: GLOBAL_DENY contains "-binding protein", which
# would wrongly drop "Single-stranded DNA-binding protein". That family is
# whitelisted explicitly below so the deny list does not remove it.
SSB_WHITELIST = re.compile(r"(?i)^Single-stranded DNA-binding protein")

COMPILED = [(cat, tier, re.compile(pat)) for cat, tier, pat in CATALOG]


def classify(product: str):
    """Return (category, tier) if the product is a housekeeping gene, else None.

    First matching catalog entry wins. The global deny list is applied, except
    that the SSB family is whitelisted (the deny list's '-binding protein' rule
    would otherwise remove it).
    """
    product = product.strip()
    if not product:
        return None
    denied = GLOBAL_DENY.search(product) is not None
    for cat, tier, rx in COMPILED:
        if rx.search(product):
            if denied and not SSB_WHITELIST.match(product):
                return None
            return cat, tier
    return None


def genome_id_from_path(path: str) -> str:
    """'GCF_000143685.1_ASM14368v1_genomic.tsv' -> 'GCF_000143685.1_ASM14368v1'."""
    base = os.path.basename(path)
    base = re.sub(r"\.tsv$", "", base)
    base = re.sub(r"_genomic$", "", base)
    return base


def parse_tsv(path):
    """Yield CDS rows from a Bakta TSV as dicts."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 8 or c[1] != "cds":
                continue
            yield {
                "contig": c[0],
                "start": c[2],
                "stop": c[3],
                "strand": c[4],
                "locus_tag": c[5],
                "gene": c[6],
                "product": c[7],
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tsvs = sorted(glob.glob(os.path.join(args.tsv_dir, "*.tsv")))
    if not tsvs:
        raise SystemExit(f"No *.tsv found in {args.tsv_dir}")

    out_rows = []
    cat_counts = Counter()
    tier_counts = Counter()
    cat_genome_presence = defaultdict(set)   # category -> set(genomes)
    per_genome_total = Counter()
    per_genome_cds = Counter()

    for path in tsvs:
        gid = genome_id_from_path(path)
        for row in parse_tsv(path):
            per_genome_cds[gid] += 1
            hit = classify(row["product"])
            if hit is None:
                continue
            cat, tier = hit
            cat_counts[cat] += 1
            tier_counts[tier] += 1
            cat_genome_presence[cat].add(gid)
            per_genome_total[gid] += 1
            out_rows.append({
                "genome": gid,
                "contig": row["contig"],
                "locus_tag": row["locus_tag"],
                "gene": row["gene"],
                "product": row["product"],
                "start": int(row["start"]),
                "stop": int(row["stop"]),
                "strand": row["strand"],
                "hk_category": cat,
                "tier": tier,
            })

    n_genomes = len(tsvs)

    # ---- main per-locus table -------------------------------------------------
    main_path = os.path.join(args.out_dir, "housekeeping_genes.tsv")
    fields = ["genome", "contig", "locus_tag", "gene", "product",
              "start", "stop", "strand", "hk_category", "tier"]
    with open(main_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in sorted(out_rows, key=lambda r: (r["genome"], r["contig"], r["start"])):
            w.writerow(r)

    # ---- category summary (with genome prevalence = conservation check) -------
    summary_path = os.path.join(args.out_dir, "housekeeping_summary.tsv")
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["hk_category", "tier", "n_loci", "n_genomes_present",
                    "pct_genomes_present", "mean_copies_per_genome"])
        for cat, _, _ in CATALOG:
            pass  # keep order via dict below
        seen = []
        order = []
        for cat, tier, _ in CATALOG:
            if cat not in seen:
                seen.append(cat)
                order.append((cat, tier))
        for cat, tier in order:
            n = cat_counts[cat]
            g = len(cat_genome_presence[cat])
            w.writerow([cat, tier, n, g, f"{100*g/n_genomes:.1f}",
                        f"{n/n_genomes:.2f}"])
        w.writerow([])
        w.writerow(["TOTAL", "", len(out_rows), n_genomes, "100.0",
                    f"{len(out_rows)/n_genomes:.2f}"])
        w.writerow(["tier1_loci", "", tier_counts[1], "", "", ""])
        w.writerow(["tier2_loci", "", tier_counts[2], "", "", ""])

    # ---- per-genome counts ----------------------------------------------------
    pg_path = os.path.join(args.out_dir, "per_genome_counts.tsv")
    with open(pg_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["genome", "n_cds", "n_housekeeping", "pct_housekeeping"])
        for gid in sorted(per_genome_cds):
            ncds = per_genome_cds[gid]
            nhk = per_genome_total[gid]
            w.writerow([gid, ncds, nhk, f"{100*nhk/ncds:.2f}" if ncds else "0"])

    # ---- console report -------------------------------------------------------
    print(f"Genomes processed         : {n_genomes}")
    print(f"Total CDS scanned         : {sum(per_genome_cds.values()):,}")
    print(f"Housekeeping loci selected: {len(out_rows):,} "
          f"({100*len(out_rows)/sum(per_genome_cds.values()):.2f}% of CDS)")
    print(f"  tier 1 loci             : {tier_counts[1]:,}")
    print(f"  tier 2 loci             : {tier_counts[2]:,}")
    print("\nPer-category (n_loci | %genomes present | mean copies/genome):")
    for cat, tier in order:
        g = len(cat_genome_presence[cat])
        print(f"  [{tier}] {cat:<28} {cat_counts[cat]:>6}  "
              f"{100*g/n_genomes:5.1f}%  {cat_counts[cat]/n_genomes:5.2f}")

    # ---- sanity checks --------------------------------------------------------
    print("\nSanity checks:")
    zero = [g for g in per_genome_cds if per_genome_total[g] == 0]
    print(f"  genomes with zero housekeeping genes: {len(zero)}"
          + (f"  -> {zero}" if zero else "  (none - good)"))
    # Core families that must be ~universal:
    for core in ["rnap_core", "aminoacyl_trna_synthetase", "ribosomal_protein",
                 "translation_factor"]:
        g = len(cat_genome_presence[core])
        flag = "" if g >= 0.95 * n_genomes else "  <-- LOW, investigate"
        print(f"  {core:<28} present in {g}/{n_genomes} genomes{flag}")

    print(f"\nWrote:\n  {main_path}\n  {summary_path}\n  {pg_path}")


if __name__ == "__main__":
    main()