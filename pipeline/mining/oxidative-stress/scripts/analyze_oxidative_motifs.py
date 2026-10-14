#!/usr/bin/env python3
"""Reference-PWM QC of the produced oxidative-stress promoter shortlist.

Scan the already oriented summit sequences, not gene CDSs. Each null replicate
is an independent permutation of that candidate's bases: length and all four
base counts are preserved, but dinucleotide counts are not. Compare maximum
window scores with maximum null-window scores, counting ties conservatively.
This is descriptive sequence QC, not a test of promoter activity or regulation.
"""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[4]
WORKFLOW = Path(__file__).resolve().parents[1]
BASES = "ACGT"


def load_reference(path):
    source = REPO / "pipeline/mining/motif_qc/reanalyze_motif_calls.py"
    spec = importlib.util.spec_from_file_location("reference_motif_qc", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    matrices, background = module.parse_meme(path)
    return {name: np.log2((np.asarray(matrix) + 1e-12) / background)
            for name, matrix in matrices.items()}, matrices, background


def window_scores(encoded, log_odds):
    sequences = np.atleast_2d(encoded)
    width = len(log_odds)
    scores = np.zeros((len(sequences), sequences.shape[1] - width + 1))
    for position in range(width):
        scores += log_odds[position, sequences[:, position:position + scores.shape[1]]]
    return scores


def shuffled_sequences(encoded, replicates, rng):
    return rng.permuted(np.broadcast_to(encoded, (replicates, len(encoded))), axis=1)


def empirical_p(observed, null_scores):
    return float((1 + np.count_nonzero(null_scores >= observed)) / (1 + len(null_scores)))


def bh_qvalues(p_values):
    """Use the existing empirical-fdr workflow's Benjamini-Hochberg convention."""
    p_values = np.asarray(p_values, float)
    count = len(p_values)
    order = np.argsort(p_values)
    q_values = p_values[order] * count / np.arange(1, count + 1)
    q_values = np.minimum.accumulate(q_values[::-1])[::-1]
    result = np.empty(count)
    result[order] = np.clip(q_values, 0, 1)
    return result


def analyze(candidates_path, motif_path, replicates, seed):
    with Path(candidates_path).open(newline="") as source:
        candidates = list(csv.DictReader(source))
    log_odds, matrices, background = load_reference(motif_path)
    rng = np.random.default_rng(seed)
    rows = []
    null_distributions = {name: [] for name in ("minus10", "minus35")}
    for candidate in candidates:
        sequence = candidate["sequence"].upper()
        if len(sequence) < 6 or set(sequence) - set(BASES):
            raise ValueError(f"{candidate['candidate_id']}: expected unambiguous summit DNA of at least six bases")
        encoded = np.array([BASES.index(base) for base in sequence], dtype=np.uint8)
        null_sequences = shuffled_sequences(encoded, replicates, rng)
        row = {
            "candidate_id": candidate["candidate_id"], "contig": candidate["contig"],
            "strand": candidate["strand"], "summit_window_start_0based": int(candidate["summit"]),
            "sequence": sequence, "length_bp": len(sequence),
            "at_percent": 100 * (sequence.count("A") + sequence.count("T")) / len(sequence),
            "target_refseq_loci": candidate["Target RefSeq loci"], "target_functions": candidate["Target functions"],
            "stored_fimo_architecture": candidate["architecture"],
            "stored_fimo_edge_gap_bp": int(candidate["spacer"]) if int(candidate["spacer"]) >= 0 else None,
        }
        for name in ("minus10", "minus35"):
            observed_scores = window_scores(encoded, log_odds[name])[0]
            start = int(np.argmax(observed_scores))
            observed = float(observed_scores[start])
            null_scores = window_scores(null_sequences, log_odds[name]).max(axis=1)
            null_distributions[name].append(null_scores)
            row.update({
                f"{name}_best_start_1based": start + 1,
                f"{name}_best_core": sequence[start:start + 6],
                f"{name}_best_score_bits": observed,
                f"{name}_shuffle_null_95th_bits": float(np.quantile(null_scores, 0.95)),
                f"{name}_shuffle_p": empirical_p(observed, null_scores),
                f"{name}_fraction_null_strictly_below": float(np.mean(null_scores < observed)),
            })
        rows.append(row)
    comparisons = [(row, name) for row in rows for name in ("minus10", "minus35")]
    adjusted = bh_qvalues([row[f"{name}_shuffle_p"] for row, name in comparisons])
    for (row, name), q_value in zip(comparisons, adjusted):
        row[f"{name}_shuffle_bh_q"] = float(q_value)
    summary = {
        "inputs": {
            "candidate_table": str(Path(candidates_path).resolve().relative_to(REPO)),
            "candidate_table_sha256": hashlib.sha256(Path(candidates_path).read_bytes()).hexdigest(),
            "reference_motifs": str(Path(motif_path).resolve().relative_to(REPO)),
            "reference_motifs_sha256": hashlib.sha256(Path(motif_path).read_bytes()).hexdigest(),
            "assembly_accession": "GCF_014131795.1",
        },
        "method": {
            "analysis": "Reference-PWM rescoring, not de-novo motif discovery or a new classifier scan",
            "reference_pwm_width_bp": 6,
            "declared_background_ACGT": background,
            "reference_pwm_consensus": {name: "".join(BASES[int(np.argmax(position))] for position in matrix)
                                        for name, matrix in matrices.items()},
            "reference_pwm_scientific_source": "Not recorded in the supplied motif file",
            "search": "Every six-base window on the stored transcription-oriented sequence; no reverse-complement rescan",
            "null": "Independent per-candidate mononucleotide permutations; exact length and A/C/G/T counts, not dinucleotides",
            "null_replicates_per_candidate": replicates, "seed": seed,
            "empirical_p": "(1 + number of null maximum scores >= observed maximum score) / (1 + replicates)",
            "multiple_comparisons": "Benjamini-Hochberg across all candidate × motif comparisons together",
            "interpretation": "Conditional descriptive QC after model selection; no calibrated biological FDR or activity/constitutivity claim",
        },
        "candidate_count": len(rows),
        "sequence_lengths_bp": dict(Counter(row["length_bp"] for row in rows)),
        "mean_at_percent": float(np.mean([row["at_percent"] for row in rows])),
        "stored_fimo_architecture_counts": dict(Counter(row["stored_fimo_architecture"] for row in rows)),
        "stored_fimo_two_hit_candidates": [row["candidate_id"] for row in rows if row["stored_fimo_architecture"] == "canonical"],
        "stored_fimo_edge_gaps_bp": [row["stored_fimo_edge_gap_bp"] for row in rows if row["stored_fimo_architecture"] == "canonical"],
        "reference_pwm_qc": {
            name: {
                "median_max_score_bits": float(np.median([row[f"{name}_best_score_bits"] for row in rows])),
                "median_shuffle_p": float(np.median([row[f"{name}_shuffle_p"] for row in rows])),
                "windows_with_shuffle_p_le_0.05": sum(row[f"{name}_shuffle_p"] <= 0.05 for row in rows),
                "windows_with_shuffle_bh_q_le_0.05": sum(row[f"{name}_shuffle_bh_q"] <= 0.05 for row in rows),
                "best_core_counts": dict(Counter(row[f"{name}_best_core"] for row in rows)),
            } for name in ("minus10", "minus35")
        },
    }
    return rows, summary, null_distributions


def plot_scores(rows, null_distributions, destination):
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True, layout="constrained")
    for axis, name, title, colour in zip(axes, ("minus10", "minus35"), ("−10-like reference", "−35-like reference"), ("#009E73", "#D55E00")):
        observed = np.array([row[f"{name}_best_score_bits"] for row in rows])
        null = np.concatenate(null_distributions[name])
        bins = np.linspace(min(null.min(), observed.min()) - 0.1, max(null.max(), observed.max()) + 0.1, 18)
        axis.hist(null, bins=bins, density=True, color="#999999", alpha=0.5, label="Base-count-preserving shuffles")
        axis.hist(observed, bins=bins, density=True, histtype="step", linewidth=2, color=colour, label="Candidate windows")
        axis.set_title(title)
        axis.set_xlabel("Best six-base PWM score (bits)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Probability density")
    axes[0].legend(fontsize=8, loc="upper left")
    figure.savefig(destination / "signal_vs_shuffled.png", dpi=170)
    figure.savefig(destination / "signal_vs_shuffled.svg")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=WORKFLOW / "model_outputs/fdr/results/CB_oxidative_stress_final_promoters.csv")
    parser.add_argument("--motifs", type=Path, default=WORKFLOW / "model_outputs/fdr/sigma_a_promoter_fixed.meme")
    parser.add_argument("--outdir", type=Path, default=REPO / "results/oxidative-stress/motif_qc")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args()
    if arguments.replicates < 1:
        parser.error("--replicates must be positive")
    rows, summary, null_distributions = analyze(arguments.candidates, arguments.motifs, arguments.replicates, arguments.seed)
    arguments.outdir.mkdir(parents=True, exist_ok=True)
    with (arguments.outdir / "candidate_motif_qc.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (arguments.outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_scores(rows, null_distributions, arguments.outdir)
    print(f"{len(rows)} candidate windows; {arguments.replicates} shuffles per candidate -> {arguments.outdir}")
    print(json.dumps(summary["reference_pwm_qc"], indent=2))


if __name__ == "__main__":
    main()
