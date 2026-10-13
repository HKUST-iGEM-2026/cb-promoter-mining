import pandas as pd

mapping_file = (
    "pipeline/mining/inducible/model_outputs/fdr/results/"
    "CB_oxidative_stress_promoter_mapping.csv"
)

motif_file = (
    "pipeline/mining/inducible/model_outputs/fdr/results/"
    "top80_promoters_motif.csv"
)

out_file = (
    "pipeline/mining/inducible/model_outputs/fdr/results/"
    "CB_oxidative_stress_final_promoters.csv"
)

mapping = pd.read_csv(mapping_file)
motif = pd.read_csv(motif_file)

# Motif-specific columns only; FDR/model columns already exist in mapping.
motif_cols = [
    "candidate_id",
    "sequence",
    "has_minus10",
    "minus10_start",
    "minus10_pval",
    "has_minus35",
    "minus35_start",
    "minus35_pval",
    "spacer",
    "architecture",
]

final = mapping.merge(
    motif[motif_cols],
    on="candidate_id",
    how="left",
    validate="one_to_one"
)

# Useful presentation order
columns = [
    "candidate_id",
    "Leader Bakta locus",
    "Leader gene",
    "Leader product",
    "Target RefSeq loci",
    "Target Bakta loci",
    "Target functions",
    "Operon ID",
    "contig",
    "strand",
    "peak_start",
    "peak_end",
    "summit",
    "ensemble_score",
    "p_value",
    "q_value",
    "n_windows",
    "architecture",
    "has_minus10",
    "minus10_start",
    "minus10_pval",
    "has_minus35",
    "minus35_start",
    "minus35_pval",
    "spacer",
    "sequence",
    "Region ID",
    "upstream_start",
    "upstream_end",
    "overlap_bp",
]

final = final[columns]

# Keep strongest ensemble candidates first
final = final.sort_values(
    ["ensemble_score", "q_value"],
    ascending=[False, True]
)

final.to_csv(out_file, index=False)

print(f"Final candidates: {len(final)}")
print(f"Canonical: {(final['architecture'] == 'canonical').sum()}")
print(f"q <= 0.05: {(final['q_value'] <= 0.05).sum()}")

print("\nCanonical promoter candidates:")
print(
    final.loc[
        final["architecture"] == "canonical",
        [
            "candidate_id",
            "Leader Bakta locus",
            "Leader product",
            "Target functions",
            "ensemble_score",
            "q_value",
            "n_windows",
            "minus35_start",
            "minus10_start",
            "spacer",
        ],
    ].to_string(index=False)
)

print(f"\nSaved: {out_file}")
