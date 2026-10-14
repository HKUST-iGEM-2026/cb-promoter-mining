import csv
import re

targets_file = "pipeline/mining/inducible/data/genes/CB_oxidative_stress_genes.csv"
bakta_file = "pipeline/mining/inducible/bakta/bakta_annotation/DSM10702.tsv"
output_file = "pipeline/mining/inducible/data/genes/CB_oxidative_stress_genes_mapped.csv"

seq_map = {
    "NZ_CP040626.1": "contig_1",
    "NZ_CP040627.1": "contig_2",
    "NZ_CP040628.1": "contig_3",
    "NZ_CP040629.1": "contig_4",
}

bakta_cds = []

with open(bakta_file, newline="") as f:
    for line in f:
        if line.startswith("#"):
            continue

        fields = line.rstrip("\n").split("\t")

        if len(fields) < 8 or fields[1] != "cds":
            continue

        bakta_cds.append({
            "seq_id": fields[0],
            "start": int(fields[2]),
            "stop": int(fields[3]),
            "strand": fields[4],
            "locus_tag": fields[5],
            "gene": fields[6],
            "product": fields[7],
        })


def overlap_fraction(a_start, a_stop, b_start, b_stop):
    overlap = max(
        0,
        min(a_stop, b_stop) - max(a_start, b_start) + 1
    )

    target_length = a_stop - a_start + 1

    return overlap / target_length


with open(targets_file, newline="", encoding="utf-8-sig") as infile, \
     open(output_file, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)

    extra_fields = [
        "Bakta sequence",
        "Bakta start",
        "Bakta stop",
        "Bakta strand",
        "Bakta locus tag",
        "Bakta gene",
        "Bakta product",
        "Overlap fraction",
        "Mapping status",
    ]

    writer = csv.DictWriter(
        outfile,
        fieldnames=reader.fieldnames + extra_fields
    )

    writer.writeheader()

    counts = {
        "exact": 0,
        "boundary mismatch": 0,
        "not annotated by Bakta": 0,
        "other": 0,
    }

    for row in reader:

        location = row["Genomic location"]

        m = re.match(r"([^:]+):(\d+)-(\d+)", location)

        if not m:
            row["Mapping status"] = "invalid location"
            writer.writerow(row)
            counts["other"] += 1
            continue

        accession = m.group(1)
        start = int(m.group(2))
        stop = int(m.group(3))

        orientation = row["Orientation"].strip().lower()

        strand = {
            "plus": "+",
            "minus": "-"
        }.get(orientation, orientation)

        bakta_seq = seq_map.get(accession)

        row["Bakta sequence"] = bakta_seq or ""

        candidates = [
            x for x in bakta_cds
            if x["seq_id"] == bakta_seq
            and x["strand"] == strand
        ]

        # 1. Exact match
        exact = [
            x for x in candidates
            if x["start"] == start and x["stop"] == stop
        ]

        if exact:
            hit = exact[0]
            status = "exact"
            frac = 1.0

        else:
            # 2. Best overlapping CDS on same strand
            overlaps = []

            for x in candidates:
                frac = overlap_fraction(
                    start, stop,
                    x["start"], x["stop"]
                )

                if frac > 0:
                    overlaps.append((frac, x))

            overlaps.sort(
                key=lambda z: z[0],
                reverse=True
            )

            if overlaps and overlaps[0][0] >= 0.8:
                frac, hit = overlaps[0]
                status = "boundary mismatch"

            else:
                hit = None
                frac = 0
                status = "not annotated by Bakta"

        if hit:
            row["Bakta start"] = hit["start"]
            row["Bakta stop"] = hit["stop"]
            row["Bakta strand"] = hit["strand"]
            row["Bakta locus tag"] = hit["locus_tag"]
            row["Bakta gene"] = hit["gene"]
            row["Bakta product"] = hit["product"]
            row["Overlap fraction"] = f"{frac:.3f}"

        else:
            row["Bakta start"] = ""
            row["Bakta stop"] = ""
            row["Bakta strand"] = ""
            row["Bakta locus tag"] = ""
            row["Bakta gene"] = ""
            row["Bakta product"] = ""
            row["Overlap fraction"] = "0"

        row["Mapping status"] = status
        writer.writerow(row)

        counts[status] += 1


print("Mapping summary:")
for k, v in counts.items():
    print(f"  {k}: {v}")

print(f"\nOutput written to: {output_file}")
