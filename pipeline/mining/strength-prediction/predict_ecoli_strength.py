#!/usr/bin/env python3
"""Estimate sigma70 strength for ten 81 bp inserts intended for E. coli DH5alpha.

Use the published Promoter Calculator, not the Clostridium classifier score. Summit sequences are
already transcription-oriented: never reverse-complement them again based on genomic strand.
Only fully contained forward-strand configurations are scored; no cloning flanks are invented.
The published in-vivo settings are labelled MG1655, not calibrated specifically for DH5alpha.
See docs/ECOLI_STRENGTH_PREDICTIONS.md for the model, units and short-fragment limitations.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import platform

from promoter_calculator.promoter_calculator import Promoter_Calculator

MODEL_REVISION = "0fe5f6f0fb2befc087556afa575f7241afd98fbf"
MODEL_SOURCE = f"https://github.com/barricklab/promoter-calculator/tree/{MODEL_REVISION}"
ORGANISM = "Escherichia coli str. K-12 substr. MG1655"
TARGET_HOST = "Escherichia coli DH5alpha"
HOST_TRANSFER_STATUS = "Unvalidated DH5alpha proxy; MG1655-labelled in-vivo parameters; no DH5alpha-specific fit"
NATIVE_ANCHOR = 60


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_candidates(path, count):
    selected = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["tier_label"] == "canonical":
                sequence = row["summit_seq"].upper()
                if len(sequence) != 81 or not set(sequence) <= set("ACGT"):
                    raise ValueError(f"{row['candidate_id']}: expected clean, transcription-oriented 81 bp DNA")
                selected.append((row, sequence))
                if len(selected) == count:
                    break
    if len(selected) != count:
        raise ValueError(f"expected {count} canonical candidates; found {len(selected)}")
    return selected


def prediction_record(prediction):
    record = {
        "tss_offset_0based": int(prediction["TSS"]),
        "tx_rate_au": float(prediction["Tx_rate"]),
        "dg_total_model_scale": float(prediction["dG_total"]),
    }
    for key in ("UP", "hex35", "spacer", "hex10", "disc", "ITR", "promoter_sequence"):
        record[key] = prediction[key]
    for key in ("UP_position", "hex35_position", "spacer_position", "hex10_position", "disc_position"):
        record[key + "_0based_halfopen"] = prediction[key]
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="canonical/ranked sequence TSV; preserve file order")
    parser.add_argument("--out-prefix", type=Path, required=True, help="write PREFIX.tsv and PREFIX.json")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()
    if args.top_n < 1:
        parser.error("--top-n must be positive")

    distribution = importlib.metadata.distribution("promotercalculator")
    expected_url = f"https://github.com/barricklab/promoter-calculator/archive/{MODEL_REVISION}.zip"
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    if distribution.version != "1.2.4" or direct.get("url") != expected_url:
        raise RuntimeError("install the pinned requirements.txt; this run requires the reviewed model revision")

    selected = select_candidates(args.input, args.top_n)
    calculator = Promoter_Calculator(organism=ORGANISM, threads=1, verbosity=0)
    rows, profiles = [], []
    for input_order, (source, sequence) in enumerate(selected, 1):
        forward, _ = calculator.predict(sequence, [0, len(sequence)], callback=lambda: None)
        if not forward or NATIVE_ANCHOR not in forward:
            raise RuntimeError(f"{source['candidate_id']}: no fully contained prediction at the native anchor")
        best = min(forward.values(), key=lambda finding: (float(finding["dG_total"]), int(finding["TSS"])))
        rate = float(best["Tx_rate"])
        anchor_rate = float(forward[NATIVE_ANCHOR]["Tx_rate"])
        tss = int(best["TSS"])
        rows.append({
            "input_order": input_order,
            "candidate_id": source["candidate_id"],
            "ecoli_strength_rank": 0,
            "target_host": TARGET_HOST,
            "model_organism": ORGANISM,
            "host_transfer_status": HOST_TRANSFER_STATUS,
            "predicted_tx_rate_au": rate,
            "relative_strength_max_100": 0.0,
            "log10_predicted_tx_rate": math.log10(rate),
            "model_best_tss_offset_0based": tss,
            "model_best_tss_position_1based": tss + 1,
            "native_tss_anchor_offset_0based": NATIVE_ANCHOR,
            "predicted_tx_at_native_anchor_au": anchor_rate,
            "best_to_anchor_fold_change": rate / anchor_rate,
            "model_dg_total": float(best["dG_total"]),
            "model_hex35": best["hex35"],
            "model_spacer": best["spacer"],
            "model_spacer_length_bp": len(best["spacer"]),
            "model_hex10": best["hex10"],
            "model_discriminator": best["disc"],
            "model_up": best["UP"],
            "model_itr": best["ITR"],
            "valid_forward_tss_count": len(forward),
            "source_chromosome": source["chromosome"],
            "source_genomic_strand": source["strand"],
            "source_native_tss_1based": source["tss_position"],
            "source_clostridium_peak_score": source["peak_score"],
            "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
            "sequence_5to3": sequence,
            "interpretation": "DH5alpha transfer estimate for the 81 bp fragment; cloning context absent; not measured expression",
        })
        profiles.append({
            "candidate_id": source["candidate_id"],
            "sequence_5to3": sequence,
            "forward_predictions": [prediction_record(forward[position]) for position in sorted(forward)],
        })

    rows.sort(key=lambda row: (-row["predicted_tx_rate_au"], row["input_order"]))
    maximum = rows[0]["predicted_tx_rate_au"]
    for rank, row in enumerate(rows, 1):
        row["ecoli_strength_rank"] = rank
        row["relative_strength_max_100"] = 100.0 * row["predicted_tx_rate_au"] / maximum

    model_dir = Path(inspect.getfile(Promoter_Calculator)).parent
    manifest = {
        "model": "Promoter Calculator v1.0; Barrick Lab packaging 1.2.4",
        "model_source": MODEL_SOURCE,
        "model_direct_url": direct,
        "model_files_sha256": {name: sha256(model_dir / name) for name in (
            "promoter_calculator.py", "util.py", "free_energy_coeffs.npy", "model_intercept.npy")},
        "organism": calculator.organism,
        "target_host": TARGET_HOST,
        "host_transfer_status": HOST_TRANSFER_STATUS,
        "K": calculator.K,
        "beta": calculator.BETA,
        "input_file": str(args.input),
        "input_sha256": sha256(args.input),
        "selection": f"first {args.top_n} canonical rows in file order",
        "sequence_orientation": "supplied summit_seq, already transcription-oriented, forward predictions only",
        "strength_definition": "maximum fully contained forward-TSS Tx_rate; strongest of selected set = 100",
        "native_anchor_offset_0based": NATIVE_ANCHOR,
        "units": "model transcription-initiation arbitrary units, not RPU, fluorescence, mRNA abundance or molecules/time",
        "context": "exact 81 bp fragments; no extra flanks, padding, circularization or sequence changes",
        "python": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name) for name in (
            "promotercalculator", "numpy", "scipy", "scikit-learn", "biopython")},
        "predictions": profiles,
    }
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(args.out_prefix) + ".tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    Path(str(args.out_prefix) + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("E. coli sigma70 strength estimates for exact 81 bp inserts (model AU; strongest = 100)")
    print(f"Target host: {TARGET_HOST}; model setting: {ORGANISM}; {HOST_TRANSFER_STATUS}")
    for row in rows:
        print(f"{row['ecoli_strength_rank']:2d}  {row['candidate_id']}  "
              f"Tx={row['predicted_tx_rate_au']:.6g}  relative={row['relative_strength_max_100']:.4g}  "
              f"TSS={row['model_best_tss_position_1based']}")
    print(f"Wrote {args.out_prefix}.tsv and {args.out_prefix}.json")


if __name__ == "__main__":
    main()
