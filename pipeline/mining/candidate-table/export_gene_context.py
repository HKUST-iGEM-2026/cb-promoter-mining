#!/usr/bin/env python3
"""Export inferred downstream coding-locus associations for a ranked shortlist.

GFF and candidate TSS coordinates are 1-based inclusive. Multipart CDS records
with the same locus tag describe one coding locus, not multiple unit members.
Associations use linear contig coordinates; no circular-origin wrap is inferred.
They do not establish promoter activity, transcriptional starts or operons.
"""
import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote


@dataclass(frozen=True)
class CodingLocus:
    contig: str
    strand: str
    start: int
    end: int
    locus_tag: str
    gene: str
    product: str
    protein_ids: tuple
    cds_segments: tuple


def load_coding_loci(path):
    groups = {}
    metadata = {}
    with Path(path).open() as source:
        for line in source:
            if line.startswith("#!"):
                key, value = line[2:].strip().split(" ", 1)
                metadata[key] = value
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "CDS":
                continue
            attributes = dict(item.split("=", 1) for item in fields[8].split(";") if "=" in item)
            attributes = {key: unquote(value) for key, value in attributes.items()}
            locus = attributes["locus_tag"]
            key = fields[0], locus
            group = groups.setdefault(key, {
                "contig": fields[0], "strand": fields[6], "locus_tag": locus,
                "gene": attributes.get("gene", ""), "product": attributes.get("product", ""),
                "protein_ids": set(), "cds_segments": set(),
            })
            if group["strand"] != fields[6]:
                raise ValueError(f"CDS segments for {locus} have inconsistent strands")
            if attributes.get("protein_id"):
                group["protein_ids"].add(attributes["protein_id"])
            group["cds_segments"].add((int(fields[3]), int(fields[4])))
    by_contig = {}
    for group in groups.values():
        segments = tuple(sorted(group.pop("cds_segments")))
        group["protein_ids"] = tuple(sorted(group["protein_ids"]))
        coding_locus = CodingLocus(
            **group, start=min(segment[0] for segment in segments),
            end=max(segment[1] for segment in segments), cds_segments=segments,
        )
        by_contig.setdefault(coding_locus.contig, []).append(coding_locus)
    for loci in by_contig.values():
        loci.sort(key=lambda locus: (locus.start, locus.end, locus.locus_tag))
    return by_contig, metadata


def downstream_unit(loci, tss, strand, max_gap_bp):
    """Nearest downstream same-strand locus, followed by adjacent coding loci."""
    eligible = (
        (index, locus) for index, locus in enumerate(loci)
        if locus.strand == strand and (locus.start >= tss if strand == "+" else locus.end <= tss)
    )
    nearest = min(eligible, key=lambda item: item[1].start - tss if strand == "+" else tss - item[1].end,
                  default=None)
    if nearest is None:
        return []
    index, first = nearest
    members = [(first, None)]
    step = 1 if strand == "+" else -1
    while 0 <= index + step < len(loci):
        following = loci[index + step]
        previous = members[-1][0]
        gap = following.start - previous.end - 1 if strand == "+" else previous.start - following.end - 1
        if following.strand != strand or gap > max_gap_bp:
            break
        members.append((following, gap))
        index += step
    return members


def build_report(candidates_path, gff_path, limit, max_gap_bp):
    with Path(candidates_path).open(newline="") as source:
        candidates = sorted(csv.DictReader(source, delimiter="\t"), key=lambda row: int(row["priority_rank"]))[:limit]
    by_contig, annotation = load_coding_loci(gff_path)
    associations = []
    for candidate in candidates:
        tss = int(candidate["tss_position"])
        strand = candidate["strand"]
        members = downstream_unit(by_contig.get(candidate["chromosome"], []), tss, strand, max_gap_bp)
        associations.append({
            "candidate_id": candidate["candidate_id"], "priority_rank": int(candidate["priority_rank"]),
            "contig": candidate["chromosome"], "strand": strand, "inferred_tss_1based": tss,
            "mapping_status": "associated" if members else "no_downstream_coding_locus",
            "member_count": len(members),
            "members": [dict(
                asdict(locus), member_order=order, gap_from_previous_member_bp=gap,
                tss_to_cds_5prime_distance_bp=(locus.start - tss if strand == "+" else tss - locus.end),
            ) for order, (locus, gap) in enumerate(members, 1)],
        })
    return {
        "inputs": {
            "candidates": Path(candidates_path).name,
            "candidates_sha256": hashlib.sha256(Path(candidates_path).read_bytes()).hexdigest(),
            "annotation": Path(gff_path).name,
            "annotation_sha256": hashlib.sha256(Path(gff_path).read_bytes()).hexdigest(),
            "annotation_metadata": annotation,
        },
        "method": {
            "candidate_selection": f"First {limit} rows by ascending priority_rank",
            "coordinates": "1-based inclusive; candidate TSS is inferred, not experimentally mapped",
            "association": "Nearest downstream same-strand coding locus; extend through adjacent same-strand coding loci",
            "max_intervening_bases": max_gap_bp,
            "gap_formula": "+ strand: next.start - previous.end - 1; - strand: previous.start - next.end - 1",
            "multipart_cds": "Group CDS segments by contig and locus_tag; retain segment coordinates",
            "circular_origin_wrap_inferred": False,
            "biological_status": "Sequence- and proximity-based hypotheses, not validated promoter targets or operons",
        },
        "candidate_count": len(associations), "member_rows": sum(row["member_count"] for row in associations),
        "associations": associations,
    }


def write_report(report, csv_path, json_path):
    columns = [
        "candidate_id", "priority_rank", "contig", "strand", "inferred_tss_1based", "mapping_status",
        "member_count", "member_order", "locus_tag", "gene", "product", "protein_ids", "cds_start_1based",
        "cds_end_1based", "cds_segments_1based", "tss_to_cds_5prime_distance_bp", "gap_from_previous_member_bp",
    ]
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(csv_path).open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        for association in report["associations"]:
            common = {key: association[key] for key in columns[:7]}
            if not association["members"]:
                writer.writerow(common)
            for member in association["members"]:
                writer.writerow(dict(
                    common, member_order=member["member_order"], locus_tag=member["locus_tag"],
                    gene=member["gene"], product=member["product"], protein_ids=";".join(member["protein_ids"]),
                    cds_start_1based=member["start"], cds_end_1based=member["end"],
                    cds_segments_1based=";".join(f"{start}-{end}" for start, end in member["cds_segments"]),
                    tss_to_cds_5prime_distance_bp=member["tss_to_cds_5prime_distance_bp"],
                    gap_from_previous_member_bp=member["gap_from_previous_member_bp"],
                ))
    Path(json_path).write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="ranked canonical candidate TSV with priority_rank")
    parser.add_argument("--gff", required=True, help="the assembly's exact PGAP GFF annotation")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-gap-bp", type=int, default=100, help="maximum intervening bases between adjacent unit members")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--json", required=True)
    arguments = parser.parse_args()
    report = build_report(arguments.candidates, arguments.gff, arguments.limit, arguments.max_gap_bp)
    write_report(report, arguments.csv, arguments.json)
    print(f"{report['candidate_count']} candidates; {report['member_rows']} coding-locus rows -> {arguments.csv}, {arguments.json}")


if __name__ == "__main__":
    main()
