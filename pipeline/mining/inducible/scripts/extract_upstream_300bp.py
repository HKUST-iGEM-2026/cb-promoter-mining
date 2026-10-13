import csv
import re
from collections import defaultdict

# ============================================================
# Input / output
# ============================================================

OPERON_FILE = (
    "pipeline/mining/inducible/"
    "CB_oxidative_stress_operon_leaders.csv"
)

GENOME_FASTA = (
    "pipeline/mining/inducible/"
    "bakta_annotation/"
    "DSM10702.fna"
)

OUTPUT_FASTA = (
    "pipeline/mining/inducible/"
    "CB_oxidative_stress_upstream_300bp.fasta"
)

OUTPUT_CSV = (
    "pipeline/mining/inducible/"
    "CB_oxidative_stress_upstream_300bp.csv"
)

WINDOW = 300


# ============================================================
# RefSeq accession -> Bakta contig mapping
#
# Needed only for the target not annotated by Bakta.
# ============================================================

ACCESSION_TO_BAKTA = {
    "NZ_CP040626.1": "contig_1",
    "NZ_CP040627.1": "contig_2",
    "NZ_CP040628.1": "contig_3",
    "NZ_CP040629.1": "contig_4",
}


# ============================================================
# Utilities
# ============================================================

def reverse_complement(seq):
    table = str.maketrans(
        "ACGTNacgtn",
        "TGCANtgcan"
    )
    return seq.translate(table)[::-1]


def load_fasta(path):
    sequences = {}

    current_id = None
    chunks = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(chunks).upper()

                current_id = line[1:].split()[0]
                chunks = []

            else:
                chunks.append(line)

    if current_id is not None:
        sequences[current_id] = "".join(chunks).upper()

    return sequences


def extract_upstream(seq, start, stop, strand, window=300):
    """
    Coordinates:
        start, stop = 1-based inclusive genomic coordinates

    Output:
        upstream sequence in 5' -> 3' transcriptional orientation

    Circular wrap-around is supported.
    """

    L = len(seq)

    if strand == "+":

        # Python index of first CDS base
        cds_start_0 = start - 1

        if cds_start_0 >= window:

            upstream = seq[
                cds_start_0 - window:
                cds_start_0
            ]

            wrapped = False

            genomic_description = (
                f"{start-window}-{start-1}"
            )

        else:

            # Circular chromosome/plasmid:
            # wrap through the end of the molecule

            missing = window - cds_start_0

            upstream = (
                seq[L-missing:]
                +
                seq[:cds_start_0]
            )

            wrapped = True

            genomic_description = (
                f"circular:{L-missing+1}-{L}"
                f"+1-{start-1}"
            )


    elif strand == "-":

        # stop is 1-based inclusive.
        # seq[stop] is therefore the first base AFTER the CDS.

        available_after = L - stop

        if available_after >= window:

            genomic_seq = seq[
                stop:
                stop + window
            ]

            wrapped = False

            genomic_description = (
                f"{stop+1}-{stop+window}"
            )

        else:

            missing = window - available_after

            genomic_seq = (
                seq[stop:]
                +
                seq[:missing]
            )

            wrapped = True

            genomic_description = (
                f"circular:{stop+1}-{L}"
                f"+1-{missing}"
            )

        # Put sequence into transcriptional 5' -> 3' orientation
        upstream = reverse_complement(genomic_seq)

    else:
        raise ValueError(f"Unknown strand: {strand}")

    if len(upstream) != window:
        raise ValueError(
            f"Expected {window} bp but extracted "
            f"{len(upstream)} bp"
        )

    return upstream, wrapped, genomic_description


# ============================================================
# 1. Load genome
# ============================================================

genome = load_fasta(GENOME_FASTA)

print("Loaded genome sequences:")

for contig, seq in genome.items():
    print(f"  {contig}: {len(seq):,} bp")


# ============================================================
# 2. Load target / operon assignments
# ============================================================

with open(
    OPERON_FILE,
    newline="",
    encoding="utf-8-sig"
) as f:
    targets = list(csv.DictReader(f))


# ============================================================
# 3. Collapse multiple targets onto unique operon leaders
# ============================================================

leader_groups = defaultdict(list)
manual_targets = []

for row in targets:

    status = row["Operon assignment status"]

    if status == "assigned":

        leader_locus = row["Operon leader Bakta locus"]

        leader_groups[leader_locus].append(row)

    else:

        manual_targets.append(row)


print()
print(
    "Unique assigned operon leaders:",
    len(leader_groups)
)

print(
    "Targets requiring manual handling:",
    len(manual_targets)
)


# ============================================================
# 4. Extract 300 bp upstream of assigned operon leaders
# ============================================================

output_rows = []
fasta_records = []


for leader_locus, rows in leader_groups.items():

    first = rows[0]

    contig = first["Bakta sequence"]
    strand = first["Operon leader strand"]

    start = int(first["Operon leader start"])
    stop = int(first["Operon leader stop"])

    if contig not in genome:
        raise ValueError(
            f"{contig} not found in genome FASTA"
        )

    sequence, wrapped, region = extract_upstream(
        genome[contig],
        start,
        stop,
        strand,
        WINDOW
    )

    target_loci = [
        r["Locus tag"]
        for r in rows
    ]

    target_functions = [
        r["Function"]
        for r in rows
    ]

    target_bakta_loci = [
        r["Bakta locus tag"]
        for r in rows
    ]

    operon_id = first["Operon ID"]

    fasta_id = (
        f"{leader_locus}"
        f"|{contig}"
        f"|{strand}"
        f"|upstream_{WINDOW}bp"
    )

    fasta_description = (
        f"targets={','.join(target_loci)} "
        f"operon={operon_id} "
        f"region={region} "
        f"circular_wrap={'yes' if wrapped else 'no'}"
    )

    fasta_records.append(
        (
            fasta_id,
            fasta_description,
            sequence
        )
    )

    output_rows.append({
        "Region ID": fasta_id,
        "Operon ID": operon_id,

        "Leader Bakta locus": leader_locus,
        "Leader gene": first["Operon leader gene"],
        "Leader product": first["Operon leader product"],

        "Contig": contig,
        "Leader start": start,
        "Leader stop": stop,
        "Strand": strand,

        "Upstream length": len(sequence),
        "Genomic upstream region": region,
        "Circular wrap": (
            "YES" if wrapped else "NO"
        ),

        "Target RefSeq loci": ";".join(target_loci),
        "Target Bakta loci": ";".join(target_bakta_loci),
        "Target functions": ";".join(target_functions),

        "Sequence": sequence,

        "Status": "operon leader",
    })


# ============================================================
# 5. Handle RefSeq-only target
#
# This target was not annotated by Bakta, so we do NOT claim
# that its operon has been identified.
#
# We extract its direct 300-bp upstream region as a provisional
# sequence and clearly flag it.
# ============================================================

for row in manual_targets:

    location = row["Genomic location"]

    m = re.match(
        r"([^:]+):(\d+)-(\d+)",
        location
    )

    if not m:
        print(
            "WARNING: could not parse manual target:",
            row["Locus tag"],
            location
        )
        continue

    accession = m.group(1)
    start = int(m.group(2))
    stop = int(m.group(3))

    strand_text = row["Orientation"].strip().lower()

    strand = {
        "plus": "+",
        "minus": "-",
        "+": "+",
        "-": "-",
    }[strand_text]

    contig = ACCESSION_TO_BAKTA.get(accession)

    if contig is None:
        raise ValueError(
            f"No Bakta contig mapping for {accession}"
        )

    sequence, wrapped, region = extract_upstream(
        genome[contig],
        start,
        stop,
        strand,
        WINDOW
    )

    target_locus = row["Locus tag"]

    fasta_id = (
        f"{target_locus}"
        f"|RefSeq_only"
        f"|{contig}"
        f"|{strand}"
        f"|upstream_{WINDOW}bp"
    )

    fasta_description = (
        f"target={target_locus} "
        f"region={region} "
        f"status=provisional_direct_upstream "
        f"circular_wrap={'yes' if wrapped else 'no'}"
    )

    fasta_records.append(
        (
            fasta_id,
            fasta_description,
            sequence
        )
    )

    output_rows.append({
        "Region ID": fasta_id,
        "Operon ID": "",

        "Leader Bakta locus": "",
        "Leader gene": row["Gene symbol"],
        "Leader product": row["Function"],

        "Contig": contig,
        "Leader start": start,
        "Leader stop": stop,
        "Strand": strand,

        "Upstream length": len(sequence),
        "Genomic upstream region": region,
        "Circular wrap": (
            "YES" if wrapped else "NO"
        ),

        "Target RefSeq loci": target_locus,
        "Target Bakta loci": "",
        "Target functions": row["Function"],

        "Sequence": sequence,

        "Status": (
            "provisional direct upstream; "
            "operon not validated by Bakta/T12"
        ),
    })


# ============================================================
# 6. Write FASTA
# ============================================================

with open(
    OUTPUT_FASTA,
    "w",
    encoding="utf-8"
) as f:

    for fasta_id, description, sequence in fasta_records:

        f.write(
            f">{fasta_id} {description}\n"
        )

        # 60-bp FASTA wrapping
        for i in range(0, len(sequence), 60):
            f.write(sequence[i:i+60] + "\n")


# ============================================================
# 7. Write metadata CSV
# ============================================================

fieldnames = [
    "Region ID",
    "Operon ID",

    "Leader Bakta locus",
    "Leader gene",
    "Leader product",

    "Contig",
    "Leader start",
    "Leader stop",
    "Strand",

    "Upstream length",
    "Genomic upstream region",
    "Circular wrap",

    "Target RefSeq loci",
    "Target Bakta loci",
    "Target functions",

    "Sequence",
    "Status",
]


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(output_rows)


# ============================================================
# 8. Final checks
# ============================================================

bad_lengths = [
    r for r in output_rows
    if len(r["Sequence"]) != WINDOW
]

wrapped_count = sum(
    r["Circular wrap"] == "YES"
    for r in output_rows
)

represented_targets = []

for r in output_rows:
    represented_targets.extend(
        x for x in r["Target RefSeq loci"].split(";")
        if x
    )


print()
print("Extraction summary:")
print(
    f"  Assigned operon-leader regions: "
    f"{len(leader_groups)}"
)

print(
    f"  Provisional RefSeq-only regions: "
    f"{len(manual_targets)}"
)

print(
    f"  Total upstream regions: "
    f"{len(output_rows)}"
)

print(
    f"  Original targets represented: "
    f"{len(represented_targets)}"
)

print(
    f"  Circular wrap-around regions: "
    f"{wrapped_count}"
)

print(
    f"  Regions not exactly {WINDOW} bp: "
    f"{len(bad_lengths)}"
)

print()
print(f"FASTA: {OUTPUT_FASTA}")
print(f"CSV:   {OUTPUT_CSV}")
