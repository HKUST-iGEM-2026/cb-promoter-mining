import csv

INPUT_FILE = "pipeline/mining/oxidative-stress/CB_oxidative_stress_upstream_300bp.csv"

OUTPUT_CSV = "pipeline/mining/oxidative-stress/CB_oxidative_stress_81bp_windows.csv"
OUTPUT_FASTA = "pipeline/mining/oxidative-stress/CB_oxidative_stress_81bp_windows.fasta"

WINDOW = 81
STRIDE = 1


rows_out = []
fasta_records = []


with open(INPUT_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for region in reader:

        seq = region["Sequence"].strip().upper()

        if len(seq) != 300:
            raise ValueError(
                f'{region["Region ID"]}: expected 300 bp, got {len(seq)}'
            )

        strand = region["Strand"]
        leader_start = int(region["Leader start"])
        leader_stop = int(region["Leader stop"])

        for i in range(0, len(seq) - WINDOW + 1, STRIDE):

            window_seq = seq[i:i + WINDOW]

            # Positions within the 300-bp upstream sequence
            # 1-based, inclusive
            window_start_in_region = i + 1
            window_end_in_region = i + WINDOW

            # ------------------------------------------------
            # Convert back to genomic coordinates
            #
            # The 300-bp sequence is already stored in
            # transcriptional 5' -> 3' orientation.
            # ------------------------------------------------

            if strand == "+":

                # Whole upstream region:
                # leader_start-300 ... leader_start-1

                genomic_start = leader_start - 300 + i
                genomic_end = genomic_start + WINDOW - 1

            elif strand == "-":

                # Whole genomic region before reverse-complement:
                # leader_stop+1 ... leader_stop+300
                #
                # Since our stored sequence is reverse-complemented,
                # sequence position 1 corresponds to genomic
                # leader_stop+300.

                genomic_end = leader_stop + 300 - i
                genomic_start = genomic_end - WINDOW + 1

            else:
                raise ValueError(
                    f'Unknown strand: {strand}'
                )

            window_number = i + 1

            window_id = (
                f'{region["Leader Bakta locus"] or region["Target RefSeq loci"]}'
                f'|w{window_number:03d}'
                f'|{strand}'
                f'|{genomic_start}-{genomic_end}'
            )

            row = {
                "Window ID": window_id,
                "Region ID": region["Region ID"],

                "Window number": window_number,
                "Window start in 300bp region": window_start_in_region,
                "Window end in 300bp region": window_end_in_region,

                "Genomic start": genomic_start,
                "Genomic end": genomic_end,
                "Strand": strand,

                "Operon ID": region["Operon ID"],
                "Leader Bakta locus": region["Leader Bakta locus"],
                "Leader gene": region["Leader gene"],
                "Leader product": region["Leader product"],

                "Target RefSeq loci": region["Target RefSeq loci"],
                "Target functions": region["Target functions"],

                "Region status": region["Status"],

                "Sequence": window_seq,
            }

            rows_out.append(row)

            fasta_description = (
                f'region={region["Region ID"]} '
                f'window={window_number} '
                f'genomic={genomic_start}-{genomic_end} '
                f'strand={strand} '
                f'targets={region["Target RefSeq loci"]}'
            )

            fasta_records.append(
                (window_id, fasta_description, window_seq)
            )


# ============================================================
# Write CSV
# ============================================================

fieldnames = [
    "Window ID",
    "Region ID",

    "Window number",
    "Window start in 300bp region",
    "Window end in 300bp region",

    "Genomic start",
    "Genomic end",
    "Strand",

    "Operon ID",
    "Leader Bakta locus",
    "Leader gene",
    "Leader product",

    "Target RefSeq loci",
    "Target functions",

    "Region status",

    "Sequence",
]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_out)


# ============================================================
# Write FASTA
# ============================================================

with open(OUTPUT_FASTA, "w", encoding="utf-8") as f:

    for window_id, description, sequence in fasta_records:
        f.write(f">{window_id} {description}\n")
        f.write(sequence + "\n")


# ============================================================
# Checks
# ============================================================

bad = [
    r for r in rows_out
    if len(r["Sequence"]) != WINDOW
]

print("Window generation summary:")
print(f"  Input promoter regions: {len(set(r['Region ID'] for r in rows_out))}")
print(f"  Windows per region: {300 - WINDOW + 1}")
print(f"  Total windows: {len(rows_out)}")
print(f"  Expected total: {27 * (300 - WINDOW + 1)}")
print(f"  Bad-length windows: {len(bad)}")
print()
print(f"CSV:   {OUTPUT_CSV}")
print(f"FASTA: {OUTPUT_FASTA}")
