#!/usr/bin/env python3
"""
T9 — merge the three curated synthetic-promoter TSVs into ONE model-ready file
(`literature_promoters.tsv`) that the pipeline consumes (held out -> eval, T24/T26).

The three raw curation files differ in three ways that make them unusable as-is:

  1. NOT MERGED. The pipeline expects a single `inputs/literature_promoters.tsv`;
     curation shipped three separate per-study TSVs + an .xlsx.

  2. CLONING SCAFFOLD ON THE C. acetobutylicum SEQUENCES. Every Yang-2017 and
     Mordaka-2018 construct is ~171 bp but carries a CONSTANT 120 bp 3' tail
     (5'UTR + RBS `AGGAGG` + SalI/`GTCGAC` + XbaI/`TCTAGA` + BamHI/`GGATCC`) that is
     byte-identical across all 68 constructs — i.e. ~70 % of each "promoter" is the
     reporter vector, not the promoter. A DNA model would learn the constant tail.
     We trim it, leaving the ~50 bp variable promoter (the -35..-10 region; the
     `TATAAT` -10 box is retained). Phytofermentans seqs are already ~51 bp, no tail.

  3. ALIGNMENT GAPS. 7 aceto sequences contain '-' characters — gaps the source
     alignment inserted around the -35/-10 to keep the -10 box in register. The true
     sequence of a variant is the ungapped string, so we strip them to clean ACGT.

Nothing curated is altered or discarded: strengths are preserved verbatim in their
original units (tagged via `strength_unit`), and the untouched input string is kept
in `original_sequence` for full provenance. Raw per-study TSVs stay in this folder.

Run:  python build_literature_promoters.py
Out:  literature_promoters.tsv   (stage to inputs/ for the pipeline)
"""
import csv
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# The constant 120 bp downstream reporter scaffold shared by ALL C. acetobutylicum
# constructs (Yang 2017 + Mordaka 2018). Computed as the longest common 3' suffix
# across all 68 sequences; asserted below so a curation change can't silently mistrim.
SCAFFOLD = ("GTTAGAGAAAACGTATAAATTAGGGATAAACTATGGAACTTATGAAATAGATTGAAATGG"
            "TTTATCTGTTACCCCGTATCAAAATTTAGGAGGTTAGTTAGAGTCGACTCTAGAGGATCC")

# Per-study metadata keyed by a substring of the source filename.
STUDIES = {
    "acetobutylicum - 1": dict(
        source="Yang2017", species="Clostridium acetobutylicum",
        in_genus=True, strength_unit="relative_to_Pthl", has_scaffold=True,
        taxonomy_note=""),
    "acetobutylicum - 2": dict(
        source="Mordaka2018", species="Clostridium acetobutylicum",
        in_genus=True, strength_unit="relative_to_Pthl", has_scaffold=True,
        taxonomy_note=""),
    "phytofermentans": dict(
        source="Rostain2022", species="Clostridium phytofermentans",
        in_genus=False, strength_unit="mean_AU", has_scaffold=False,
        taxonomy_note=("reclassified to Lachnoclostridium phytofermentans; "
                       "outside Clostridium sensu stricto (genus Clostridium, taxon 1485)")),
}

OUT_FIELDS = [
    "promoter_id", "original_name", "species", "in_genus", "sequence", "seq_len",
    "strength", "strength_unit", "n_gaps_removed", "scaffold_trimmed_bp",
    "source", "label", "taxonomy_note", "citation", "original_sequence",
]


def study_for(filename):
    for key, meta in STUDIES.items():
        if key in filename:
            return meta
    raise SystemExit(f"unrecognised T9 source file (no study mapping): {filename}")


def parse_strength(raw):
    """Preserve curated strength verbatim; blank for N/A. Returns string."""
    raw = (raw or "").strip()
    if raw == "" or raw.upper() in {"N/A", "NA", "NAN", "-"}:
        return ""
    return raw  # keep the exact curated value (float or int), do not re-round


def clean_sequence(raw, has_scaffold):
    """raw curated string -> (clean_promoter, n_gaps_removed, scaffold_trimmed_bp)."""
    n_gaps = raw.count("-")
    seq = raw.upper().replace("-", "")               # strip alignment gaps -> clean ACGT
    trimmed = 0
    if has_scaffold:
        if not seq.endswith(SCAFFOLD):               # fail loud, never mistrim silently
            raise SystemExit(
                f"expected 120 bp scaffold not found at 3' end of a sequence "
                f"(curation changed?): ...{seq[-40:]}")
        seq = seq[:-len(SCAFFOLD)]
        trimmed = len(SCAFFOLD)
    if not re.fullmatch(r"[ACGT]+", seq):
        raise SystemExit(f"non-ACGT character left in cleaned sequence: {seq}")
    return seq, n_gaps, trimmed


def main():
    files = sorted(glob.glob(os.path.join(HERE, "literature_promoters - *.tsv")))
    if not files:
        raise SystemExit("no raw 'literature_promoters - *.tsv' source files found")

    out_rows, n_in = [], 0
    for path in files:
        meta = study_for(os.path.basename(path))
        with open(path, encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter="\t")
            header = next(reader)
            # columns: Promoter_Name | Sequence (5'->3') | Host_Species | <strength> | Literature citation
            for row in reader:
                if not row or not any(c.strip() for c in row):
                    continue
                n_in += 1
                name = row[0].strip()
                raw_seq = row[1].strip()
                strength = parse_strength(row[3] if len(row) > 3 else "")
                citation = (row[4].strip() if len(row) > 4 else "")
                seq, n_gaps, trimmed = clean_sequence(raw_seq, meta["has_scaffold"])
                out_rows.append({
                    "promoter_id": f'{meta["source"]}_{name}',
                    "original_name": name,
                    "species": meta["species"],
                    "in_genus": meta["in_genus"],
                    "sequence": seq,
                    "seq_len": len(seq),
                    "strength": strength,
                    "strength_unit": meta["strength_unit"],
                    "n_gaps_removed": n_gaps,
                    "scaffold_trimmed_bp": trimmed,
                    "source": meta["source"],
                    "label": 1,
                    "taxonomy_note": meta["taxonomy_note"],
                    "citation": citation,
                    "original_sequence": raw_seq,
                })

    # guard: nothing dropped, ids unique
    assert len(out_rows) == n_in, f"row count drift: in {n_in}, out {len(out_rows)}"
    ids = [r["promoter_id"] for r in out_rows]
    assert len(ids) == len(set(ids)), "duplicate promoter_id after merge"

    out_path = os.path.join(HERE, "literature_promoters.tsv")
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)

    # summary
    from collections import Counter
    by_src = Counter(r["source"] for r in out_rows)
    lens = [r["seq_len"] for r in out_rows]
    print(f"[T9] merged {len(out_rows)} promoters -> {os.path.basename(out_path)}")
    for s, n in by_src.items():
        print(f"      {s}: {n}")
    print(f"      seq_len min/max: {min(lens)}/{max(lens)} bp")
    print(f"      rows with gaps removed: {sum(1 for r in out_rows if r['n_gaps_removed'])}")
    print(f"      in_genus True/False: "
          f"{sum(r['in_genus'] for r in out_rows)}/{sum(not r['in_genus'] for r in out_rows)}")
    print(f"      with numeric strength: {sum(1 for r in out_rows if r['strength'])}")


if __name__ == "__main__":
    main()
