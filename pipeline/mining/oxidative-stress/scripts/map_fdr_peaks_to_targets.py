import pandas as pd
from pathlib import Path

REGION_BED = Path(
    "pipeline/mining/oxidative-stress/data/regions/"
    "CB_oxidative_stress_upstream_300bp.bed"
)

REGION_CSV = Path(
    "pipeline/mining/oxidative-stress/data/regions/"
    "CB_oxidative_stress_upstream_300bp.csv"
)

PEAKS_CSV = Path(
    "pipeline/mining/oxidative-stress/model_outputs/fdr/results/peaks_fdr.csv"
)

OUT_DIR = Path(
    "pipeline/mining/oxidative-stress/model_outputs/fdr/results"
)

OUT_MAPPING = OUT_DIR / "CB_oxidative_stress_promoter_mapping.csv"
OUT_SUMMARY = OUT_DIR / "CB_oxidative_stress_region_summary.csv"


# load the 27 upstream regions
bed = pd.read_csv(REGION_BED, sep="\t")

regions = pd.read_csv(REGION_CSV)

if len(bed) != len(regions):
    raise ValueError(
        f"BED has {len(bed)} regions but metadata CSV has "
        f"{len(regions)} regions."
    )

# The BED and metadata CSV were generated together and have the same row order. Add the exact 0-based BED coordinates to the metadata table.
regions = regions.copy()

regions["bed_contig"] = bed["contig"].values
regions["bed_start"] = bed["start"].values
regions["bed_end"] = bed["end"].values
regions["bed_strand"] = bed["strand"].values


# Safety check: biological metadata and BED should agree.
bad = regions[
    (regions["Contig"] != regions["bed_contig"])
    | (regions["Strand"] != regions["bed_strand"])
]

if len(bad) > 0:
    raise ValueError(
        "BED and metadata CSV do not have matching contig/strand "
        "row order."
    )


# load FDR-significant promoter peaks
peaks = pd.read_csv(PEAKS_CSV)

# peaks_fdr.csv currently contains q <= 0.05 discoveries, but explicitly enforce the threshold for reproducibility.
peaks = peaks[peaks["q_value"] <= 0.05].copy()

# Give each peak the same candidate IDs used in the ranked output.
peaks = peaks.sort_values(
    "ensemble_score", ascending=False
).reset_index(drop=True)

peaks["candidate_id"] = [
    f"cbut_{i:04d}" for i in range(1, len(peaks) + 1)
]


# ------------------------------------------------------------
# 3. Map peaks to upstream regions
#
# BED intervals use 0-based half-open coordinates:
#
#     [region_start, region_end)
#
# Two intervals overlap when:
#
#     peak_start < region_end
#     AND
#     peak_end > region_start
#
# We additionally require same contig + same strand.
# ------------------------------------------------------------

mapping_rows = []

for _, peak in peaks.iterrows():

    matches = regions[
        (regions["bed_contig"] == peak["contig"])
        & (regions["bed_strand"] == peak["strand"])
        & (peak["peak_start"] < regions["bed_end"])
        & (peak["peak_end"] > regions["bed_start"])
    ]

    if len(matches) == 0:
        print(
            "WARNING: no upstream region found for "
            f"{peak['candidate_id']} "
            f"{peak['contig']}:{peak['peak_start']}-"
            f"{peak['peak_end']}({peak['strand']})"
        )
        continue

    for _, region in matches.iterrows():

        overlap_start = max(
            int(peak["peak_start"]),
            int(region["bed_start"])
        )

        overlap_end = min(
            int(peak["peak_end"]),
            int(region["bed_end"])
        )

        mapping_rows.append({
            "candidate_id": peak["candidate_id"],

            "contig": peak["contig"],
            "strand": peak["strand"],
            "peak_start": int(peak["peak_start"]),
            "peak_end": int(peak["peak_end"]),
            "summit": int(peak["summit"]),

            "ensemble_score": peak["ensemble_score"],
            "p_value": peak["p_value"],
            "q_value": peak["q_value"],
            "n_windows": int(peak["n_windows"]),

            "overlap_bp": overlap_end - overlap_start,

            "Region ID": region["Region ID"],
            "Operon ID": region["Operon ID"],
            "Leader Bakta locus": region["Leader Bakta locus"],
            "Leader gene": region["Leader gene"],
            "Leader product": region["Leader product"],

            "Target RefSeq loci": region["Target RefSeq loci"],
            "Target Bakta loci": region["Target Bakta loci"],
            "Target functions": region["Target functions"],

            "upstream_start": int(region["bed_start"]),
            "upstream_end": int(region["bed_end"]),
        })


mapping = pd.DataFrame(mapping_rows)

mapping.to_csv(OUT_MAPPING, index=False)


# make a region-level summary
# This lets us see which of the 27 oxidative-stress upstream regions contain >=1 significant promoter peak.
summary_rows = []

for _, region in regions.iterrows():

    if len(mapping) > 0:
        m = mapping[mapping["Region ID"] == region["Region ID"]]
    else:
        m = pd.DataFrame()

    summary_rows.append({
        "Region ID": region["Region ID"],
        "Operon ID": region["Operon ID"],
        "Leader Bakta locus": region["Leader Bakta locus"],
        "Leader gene": region["Leader gene"],
        "Leader product": region["Leader product"],
        "Target RefSeq loci": region["Target RefSeq loci"],
        "Target Bakta loci": region["Target Bakta loci"],
        "Target functions": region["Target functions"],

        "n_significant_peaks": len(m),

        "has_significant_promoter":
            "YES" if len(m) > 0 else "NO",

        "best_candidate_id":
            m.iloc[0]["candidate_id"] if len(m) > 0 else "",

        "best_ensemble_score":
            m["ensemble_score"].max() if len(m) > 0 else "",

        "best_q_value":
            m["q_value"].min() if len(m) > 0 else "",
    })


summary = pd.DataFrame(summary_rows)

summary.to_csv(OUT_SUMMARY, index=False)


# print useful summary
print()
print("=== Mapping complete ===")
print(f"Significant FDR peaks: {len(peaks)}")
print(f"Peak-region mappings: {len(mapping)}")

n_with = (summary["has_significant_promoter"] == "YES").sum()
n_without = (summary["has_significant_promoter"] == "NO").sum()

print(f"Regions with >=1 significant peak: {n_with}")
print(f"Regions with no significant peak:  {n_without}")

print()
print("Outputs:")
print(f"  {OUT_MAPPING}")
print(f"  {OUT_SUMMARY}")

print()
print("Regions without a significant promoter:")
no_peak = summary[
    summary["has_significant_promoter"] == "NO"
]

if len(no_peak) == 0:
    print("  None")
else:
    for _, row in no_peak.iterrows():
        print(
            f"  {row['Leader Bakta locus']} | "
            f"{row['Leader product']} | "
            f"{row['Target functions']}"
        )