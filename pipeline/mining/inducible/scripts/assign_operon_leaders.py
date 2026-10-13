import csv
from collections import defaultdict

BAKTA_FILE = "pipeline/mining/inducible/bakta_annotation/DSM10702.tsv"
TARGET_FILE = "pipeline/mining/inducible/CB_oxidative_stress_genes_mapped.csv"
OUTPUT_FILE = "pipeline/mining/inducible/CB_oxidative_stress_operon_leaders.csv"

MAX_DIST = 100


# ============================================================
# 1. Load all Bakta CDSs
# ============================================================

cds_by_group = defaultdict(list)
cds_by_locus = {}

with open(BAKTA_FILE, newline="", encoding="utf-8") as f:
    for line in f:
        if line.startswith("#"):
            continue

        fields = line.rstrip("\n").split("\t")

        if len(fields) < 8 or fields[1] != "cds":
            continue

        gene = {
            "contig": fields[0],
            "start": int(fields[2]),
            "stop": int(fields[3]),
            "strand": fields[4],
            "locus": fields[5],
            "gene": fields[6],
            "product": fields[7],
        }

        cds_by_group[(gene["contig"], gene["strand"])].append(gene)
        cds_by_locus[gene["locus"]] = gene


# ============================================================
# 2. Cluster all CDSs using T12's 100-bp rule
# ============================================================

locus_to_cluster = {}
cluster_number = 0

for (contig, strand), genes in cds_by_group.items():

    genes.sort(key=lambda x: x["start"])

    clusters = []
    current_cluster = []
    prev_stop = None

    for gene in genes:

        if (
            prev_stop is not None
            and (gene["start"] - prev_stop) > MAX_DIST
        ):
            clusters.append(current_cluster)
            current_cluster = []

        current_cluster.append(gene)
        prev_stop = gene["stop"]

    if current_cluster:
        clusters.append(current_cluster)


    # ========================================================
    # 3. Determine leader of every cluster
    # ========================================================

    for cluster in clusters:

        cluster_number += 1

        if strand == "+":
            leader = min(cluster, key=lambda x: x["start"])
        else:
            leader = max(cluster, key=lambda x: x["stop"])

        cluster_id = f"operon_{cluster_number:05d}"

        for gene in cluster:
            locus_to_cluster[gene["locus"]] = {
                "cluster_id": cluster_id,
                "leader": leader,
                "members": cluster,
            }


# ============================================================
# 4. Load oxidative-stress targets and assign their leaders
# ============================================================

output_rows = []

with open(TARGET_FILE, newline="", encoding="utf-8-sig") as f:

    reader = csv.DictReader(f)

    for row in reader:

        bakta_locus = row["Bakta locus tag"]

        # Special case:
        # target exists in RefSeq but was not annotated by Bakta
        if not bakta_locus:

            row["Operon ID"] = ""
            row["Operon size"] = ""
            row["Operon leader Bakta locus"] = ""
            row["Operon leader gene"] = ""
            row["Operon leader product"] = ""
            row["Operon leader start"] = ""
            row["Operon leader stop"] = ""
            row["Operon leader strand"] = ""
            row["Is target operon leader"] = ""
            row["Operon members"] = ""
            row["Operon assignment status"] = "requires manual handling"

            output_rows.append(row)
            continue


        cluster_info = locus_to_cluster.get(bakta_locus)

        if cluster_info is None:

            row["Operon ID"] = ""
            row["Operon size"] = ""
            row["Operon leader Bakta locus"] = ""
            row["Operon leader gene"] = ""
            row["Operon leader product"] = ""
            row["Operon leader start"] = ""
            row["Operon leader stop"] = ""
            row["Operon leader strand"] = ""
            row["Is target operon leader"] = ""
            row["Operon members"] = ""
            row["Operon assignment status"] = "Bakta locus not found"

            output_rows.append(row)
            continue


        leader = cluster_info["leader"]
        members = cluster_info["members"]

        row["Operon ID"] = cluster_info["cluster_id"]
        row["Operon size"] = len(members)

        row["Operon leader Bakta locus"] = leader["locus"]
        row["Operon leader gene"] = leader["gene"]
        row["Operon leader product"] = leader["product"]
        row["Operon leader start"] = leader["start"]
        row["Operon leader stop"] = leader["stop"]
        row["Operon leader strand"] = leader["strand"]

        row["Is target operon leader"] = (
            "YES" if bakta_locus == leader["locus"] else "NO"
        )

        # Store members in transcription order
        if leader["strand"] == "+":
            ordered_members = sorted(
                members,
                key=lambda x: x["start"]
            )
        else:
            ordered_members = sorted(
                members,
                key=lambda x: x["stop"],
                reverse=True
            )

        row["Operon members"] = ";".join(
            x["locus"] for x in ordered_members
        )

        row["Operon assignment status"] = "assigned"

        output_rows.append(row)


# ============================================================
# 5. Write output
# ============================================================

extra_fields = [
    "Operon ID",
    "Operon size",
    "Operon leader Bakta locus",
    "Operon leader gene",
    "Operon leader product",
    "Operon leader start",
    "Operon leader stop",
    "Operon leader strand",
    "Is target operon leader",
    "Operon members",
    "Operon assignment status",
]

with open(TARGET_FILE, newline="", encoding="utf-8-sig") as f:
    original_fields = csv.DictReader(f).fieldnames

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=original_fields + extra_fields
    )

    writer.writeheader()
    writer.writerows(output_rows)


# ============================================================
# 6. Summary
# ============================================================

assigned = sum(
    r["Operon assignment status"] == "assigned"
    for r in output_rows
)

manual = sum(
    r["Operon assignment status"] == "requires manual handling"
    for r in output_rows
)

leaders = sum(
    r["Is target operon leader"] == "YES"
    for r in output_rows
)

internal = sum(
    r["Is target operon leader"] == "NO"
    for r in output_rows
)

print("Operon assignment summary:")
print(f"  Total targets: {len(output_rows)}")
print(f"  Assigned: {assigned}")
print(f"  Requires manual handling: {manual}")
print(f"  Targets that are operon leaders: {leaders}")
print(f"  Targets internal to operons: {internal}")
print()
print(f"Output: {OUTPUT_FILE}")
