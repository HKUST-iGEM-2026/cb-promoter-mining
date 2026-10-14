import csv
import re
from collections import defaultdict

mapped_file = "pipeline/mining/oxidative-stress/data/genes/CB_oxidative_stress_genes_mapped.csv"
bakta_file = "pipeline/mining/oxidative-stress/bakta/bakta_annotation/DSM10702.tsv"
output_file = "pipeline/mining/oxidative-stress/data/operons/target_operon_neighbors.csv"

# How many genes to inspect upstream of each target
MAX_UPSTREAM_GENES = 8


# ------------------------------------------------------------
# 1. Load all Bakta CDS annotations
# ------------------------------------------------------------

cds_by_contig = defaultdict(list)

with open(bakta_file, newline="", encoding="utf-8") as f:
    for line in f:

        if line.startswith("#"):
            continue

        fields = line.rstrip("\n").split("\t")

        if len(fields) < 8:
            continue

        if fields[1] != "cds":
            continue

        feature = {
            "contig": fields[0],
            "start": int(fields[2]),
            "stop": int(fields[3]),
            "strand": fields[4],
            "locus_tag": fields[5],
            "gene": fields[6],
            "product": fields[7],
        }

        cds_by_contig[feature["contig"]].append(feature)


# Sort every contig by genomic position
for contig in cds_by_contig:
    cds_by_contig[contig].sort(key=lambda x: x["start"])


# ------------------------------------------------------------
# 2. Utility function
# ------------------------------------------------------------

def genomic_gap(left_gene, right_gene):
    """
    Signed number of bases between two CDSs.

    positive = intergenic bases
    0        = directly adjacent
    negative = CDSs overlap
    """

    return right_gene["start"] - left_gene["stop"] - 1


# ------------------------------------------------------------
# 3. Examine every target
# ------------------------------------------------------------

output_rows = []

with open(mapped_file, newline="", encoding="utf-8-sig") as f:

    reader = csv.DictReader(f)

    for target in reader:

        target_locus = target["Locus tag"]
        target_function = target["Function"]
        target_bakta_locus = target["Bakta locus tag"]
        target_contig = target["Bakta sequence"]

        orientation = target["Orientation"].strip().lower()

        strand = {
            "plus": "+",
            "minus": "-"
        }.get(orientation, orientation)


        # ----------------------------------------------------
        # Determine target coordinates
        #
        # Prefer Bakta coordinates when available.
        # For FF104_RS04850, which Bakta did not annotate,
        # fall back to RefSeq coordinates.
        # ----------------------------------------------------

        if target["Bakta start"] and target["Bakta stop"]:

            target_start = int(target["Bakta start"])
            target_stop = int(target["Bakta stop"])

        else:

            m = re.match(
                r"([^:]+):(\d+)-(\d+)",
                target["Genomic location"]
            )

            if not m:
                print(f"Could not parse location for {target_locus}")
                continue

            target_start = int(m.group(2))
            target_stop = int(m.group(3))


        genes = cds_by_contig[target_contig]


        # ----------------------------------------------------
        # Find target position in genomic ordering
        # ----------------------------------------------------

        target_index = None

        if target_bakta_locus:

            for i, gene in enumerate(genes):

                if gene["locus_tag"] == target_bakta_locus:
                    target_index = i
                    break


        # ----------------------------------------------------
        # Special case:
        # target was not annotated by Bakta
        #
        # Find where it would sit in the ordered gene list.
        # ----------------------------------------------------

        if target_index is None:

            insertion_index = 0

            while (
                insertion_index < len(genes)
                and genes[insertion_index]["start"] < target_start
            ):
                insertion_index += 1

            # We use a pseudo-feature representing the RefSeq target
            target_feature = {
                "contig": target_contig,
                "start": target_start,
                "stop": target_stop,
                "strand": strand,
                "locus_tag": target_locus + "_RefSeq_only",
                "gene": target["Gene symbol"],
                "product": target_function,
            }

        else:

            target_feature = genes[target_index]


        # ----------------------------------------------------
        # 4. Search transcriptionally upstream
        # ----------------------------------------------------

        upstream_features = []

        if strand == "+":

            # Smaller genomic coordinates

            if target_index is not None:
                candidate_indices = range(
                    target_index - 1,
                    max(-1, target_index - MAX_UPSTREAM_GENES - 1),
                    -1
                )
            else:
                candidate_indices = range(
                    insertion_index - 1,
                    max(-1, insertion_index - MAX_UPSTREAM_GENES - 1),
                    -1
                )

        else:

            # Minus strand:
            # transcriptional upstream is toward larger coordinates

            if target_index is not None:
                candidate_indices = range(
                    target_index + 1,
                    min(len(genes), target_index + MAX_UPSTREAM_GENES + 1)
                )
            else:
                candidate_indices = range(
                    insertion_index,
                    min(len(genes), insertion_index + MAX_UPSTREAM_GENES)
                )


        previous_feature = target_feature

        rank = 1

        for idx in candidate_indices:

            neighbor = genes[idx]

            # Calculate physical genomic gap between
            # the current downstream feature and its upstream neighbor

            if strand == "+":
                gap = genomic_gap(neighbor, previous_feature)
            else:
                gap = genomic_gap(previous_feature, neighbor)


            same_strand = neighbor["strand"] == strand

            output_rows.append({
                "Target RefSeq locus": target_locus,
                "Target function": target_function,
                "Target Bakta locus": target_bakta_locus,
                "Target contig": target_contig,
                "Target strand": strand,
                "Target start": target_start,
                "Target stop": target_stop,

                "Upstream rank": rank,

                "Neighbor Bakta locus": neighbor["locus_tag"],
                "Neighbor gene": neighbor["gene"],
                "Neighbor product": neighbor["product"],
                "Neighbor start": neighbor["start"],
                "Neighbor stop": neighbor["stop"],
                "Neighbor strand": neighbor["strand"],

                "Same strand": same_strand,
                "Gap to next downstream gene": gap,

                "Potential operon member":
                    "YES" if same_strand else "NO - opposite strand"
            })


            # Opposite strand is a strong operon boundary.
            # Stop walking further upstream.
            if not same_strand:
                break

            previous_feature = neighbor
            rank += 1


# ------------------------------------------------------------
# 5. Write results
# ------------------------------------------------------------

fieldnames = [
    "Target RefSeq locus",
    "Target function",
    "Target Bakta locus",
    "Target contig",
    "Target strand",
    "Target start",
    "Target stop",

    "Upstream rank",

    "Neighbor Bakta locus",
    "Neighbor gene",
    "Neighbor product",
    "Neighbor start",
    "Neighbor stop",
    "Neighbor strand",

    "Same strand",
    "Gap to next downstream gene",
    "Potential operon member",
]

with open(output_file, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(output_rows)


print(f"Targets analyzed: {len(set(r['Target RefSeq locus'] for r in output_rows))}")
print(f"Neighborhood rows written: {len(output_rows)}")
print(f"Output: {output_file}")
