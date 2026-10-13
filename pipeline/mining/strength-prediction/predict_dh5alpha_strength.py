#!/usr/bin/env python3
"""Predict DH5alpha 50-bp-core activity with Wang et al.'s reproduced RBF SVR.

SVR procedure adapted from HaochenW/Deep_promoter, predictor.py at
c0e16c8218a983fbb07bf46d59c44109d4371b0e (MIT; notice in wang2020-dh5alpha/LICENSE).
See docs/ECOLI_STRENGTH_PREDICTIONS.md for input framing, units and limitations.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np


HOST = "Escherichia coli DH5alpha"
REVISION = "c0e16c8218a983fbb07bf46d59c44109d4371b0e"
SOURCE_URL = f"https://github.com/HaochenW/Deep_promoter/blob/{REVISION}/predictor.py"
SOURCE_SHA256 = "37912a418ec1e98b8f935c38731f5365712f6c1e8fe26159c4a366d28d834e68"
DATA = Path(__file__).parent / "wang2020-dh5alpha" / "seq_exp_94.txt"
DATA_SHA256 = "e81b9316059886846c4e367dd6ae1a0bb196953fcd1a1bc6ac634ed4a1394976"
CHANNELS = {"T": 0, "C": 1, "G": 2, "A": 3}
REFERENCE_PROMOTER = "pptb_fU"
REFERENCE = (Path(__file__).resolve().parent / "../../../results/candidates/pptb_fU_reference.json").resolve()
REFERENCE_SHA256 = "8550c923680a174104cb5782161a4d6c565ca8d5182600e8d95357f0c5983a16"
MODEL_FORMAT = "dh5alpha-rbf-svr-1"


class FittedSVR:
    """Inference from numeric support vectors, without pickle or scikit-learn."""

    def __init__(self, support_vectors, dual_coef, intercept, gamma):
        self.support_vectors = support_vectors
        self.dual_coef = dual_coef
        self.intercept = intercept
        self.gamma = gamma
        self.support_norms = np.einsum("ij,ij->i", support_vectors, support_vectors)

    def predict(self, features):
        features = np.asarray(features, dtype=np.float64)
        squared_distances = (
            np.einsum("ij,ij->i", features, features)[:, None]
            + self.support_norms[None, :]
            - 2 * features @ self.support_vectors.T
        )
        np.maximum(squared_distances, 0, out=squared_distances)
        squared_distances *= -self.gamma
        np.exp(squared_distances, out=squared_distances)
        return squared_distances @ self.dual_coef + self.intercept


def export_model(model, model_info, directory):
    """Save the fitted estimator and its provenance as NPZ/JSON, never pickle."""
    directory.mkdir(parents=True, exist_ok=True)
    fitted = model.best_estimator_
    weights = directory / "model.npz"
    np.savez_compressed(
        weights,
        support_vectors=fitted.support_vectors_,
        dual_coef=fitted.dual_coef_[0],
        intercept=fitted.intercept_[0],
        gamma=fitted._gamma,
    )
    metadata = {
        "format": MODEL_FORMAT,
        "model_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "model_info": model_info,
        "fit_environment": {
            "python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")},
        },
    }
    (directory / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_model(directory):
    metadata = json.loads((directory / "model_metadata.json").read_text())
    weights = directory / "model.npz"
    if metadata["format"] != MODEL_FORMAT:
        raise ValueError("Unsupported DH5alpha model format")
    if hashlib.sha256(weights.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("The fitted model differs from its metadata checksum")
    with np.load(weights, allow_pickle=False) as arrays:
        support = arrays["support_vectors"]
        dual = arrays["dual_coef"]
        intercept = float(arrays["intercept"])
        gamma = float(arrays["gamma"])
    if support.ndim != 2 or support.shape[1] != 200 or dual.shape != (len(support),):
        raise ValueError("The fitted model requires 200 position-major T/C/G/A features")
    if not np.isfinite(support).all() or not np.isfinite(dual).all() or not np.isfinite([intercept, gamma]).all() or gamma <= 0:
        raise ValueError("The fitted SVR contains invalid parameters")
    return FittedSVR(support, dual, intercept, gamma), metadata["model_info"]


def encode(sequences):
    """Position-major float32 one-hot features, in the source's T/C/G/A order."""
    data = np.zeros((len(sequences), 50, 4), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        if len(sequence) != 50 or set(sequence) - CHANNELS.keys():
            raise ValueError("The DH5alpha SVR requires exactly 50 clean A/C/G/T bases")
        for position, base in enumerate(sequence):
            data[row, position, CHANNELS[base]] = 1
    return data.reshape(len(sequences), 200)


def train():
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVR

    raw_data = DATA.read_bytes()
    if hashlib.sha256(raw_data).hexdigest() != DATA_SHA256:
        raise ValueError("The bundled training table differs from the pinned upstream data")
    records = [line.split() for line in raw_data.decode().splitlines()]
    # The upstream procedure excludes six short positive controls and strips
    # AATTC / terminal T cloning overhangs from the remaining 88 sequences.
    sequences = [record[0][5:-1] for record in records[6:]]
    activities = np.array([float(record[1]) for record in records[6:]])
    features = encode(sequences)
    labels = np.log2(activities)
    indices = np.arange(len(sequences))
    np.random.RandomState(2).shuffle(indices)
    model = GridSearchCV(
        SVR(kernel="rbf", gamma=0.1),
        cv=5,
        param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)},
    )
    model.fit(features[indices], labels[indices])
    return model, {
        "source_url": SOURCE_URL,
        "source_commit": REVISION,
        "source_predictor_sha256": SOURCE_SHA256,
        "training_data_url": f"https://github.com/HaochenW/Deep_promoter/blob/{REVISION}/seq/seq_exp_94.txt",
        "training_data_sha256": DATA_SHA256,
        "training_host": HOST,
        "training_records": len(sequences),
        "excluded_records": 6,
        "training_composition": "83 first-round generated promoters and 5 random sequences",
        "training_target": "log2 of the numerical activity labels in seq_exp_94.txt",
        "activity_units": "Upstream training-table arbitrary units; no unit header; not RPU or physical transcription rates",
        "training_activity_range_au": [float(activities.min()), float(activities.max())],
        "encoding": "50 positions, T/C/G/A channels, float32, flattened to 200 features",
        "shuffle_seed": 2,
        "parameter_grid": {"C": [1.0, 10.0, 100.0, 1000.0], "gamma": np.logspace(-2, 2, 5).tolist()},
        "cross_validation": "5-fold unshuffled KFold after seeded permutation; R2 on log2 labels; refit all 88 records",
        "best_parameters": model.best_params_,
        "best_selection_cv_mean_r2": float(model.best_score_),
        "cv_score_warning": "Hyperparameter-selection CV, not an independent test of these candidates",
        "paper": "Wang et al., Nucleic Acids Research 48:6403-6412 (2020), doi:10.1093/nar/gkaa325",
        "paper_reported_test_pcc": 0.57,
        "paper_reported_test_split": "66 training/validation records, 22 test records; not the full-data refit used here",
        "published_assay_context": "DH5alpha, modified pFAB217/p15A, sfGFP, fixed B0030 5-prime UTR, LB plus kanamycin; final published methods",
    }


def load_reference(path=REFERENCE):
    raw_reference = path.read_bytes()
    if hashlib.sha256(raw_reference).hexdigest() != REFERENCE_SHA256:
        raise ValueError("The pptb_fU reference differs from the supplied DNA and documented framing")
    return json.loads(raw_reference)


def select_candidates(path, count):
    with path.open(newline="") as handle:
        candidates = [row for row in csv.DictReader(handle, delimiter="\t") if row["tier_label"] == "canonical"][:count]
    if len(candidates) != count:
        raise ValueError(f"Requested {count} canonical records, found {len(candidates)}")
    for row in candidates:
        sequence = row["summit_seq"]
        if len(sequence) != 81 or set(sequence) - CHANNELS.keys():
            raise ValueError(f"{row['candidate_id']}: expected 81 clean, already-oriented A/C/G/T bases")
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--tss-offset", type=int, default=60, help="Assumed TSS offset, zero-based within each 81-bp fragment; default is the inferred native anchor")
    parser.add_argument("--out-prefix", type=Path, required=True, help="Output prefix for ranked .tsv, provenance .json and relative-strength .csv; any existing extension is replaced")
    parser.add_argument("--model", type=Path, help="Load a fitted model directory; needs NumPy only")
    parser.add_argument("--export-model", type=Path, help="Save the reproduced fit as model.npz and model_metadata.json")
    parser.add_argument("--reference", type=Path, default=REFERENCE, help="Exact pptb_fU reference JSON; supply this path for a standalone download")
    args = parser.parse_args()
    if args.top_n < 1:
        parser.error("--top-n must be positive")
    if not 50 <= args.tss_offset < 81:
        parser.error("--tss-offset must be 50 through 80 to contain both the 50-bp core and assumed TSS")
    if args.model and args.export_model:
        parser.error("Use --export-model when fitting, or --model when loading, not both")
    candidates = select_candidates(args.input, args.top_n)
    core_start = args.tss_offset - 50
    cores = [row["summit_seq"][core_start:args.tss_offset] for row in candidates]
    model, model_info = load_model(args.model) if args.model else train()
    if args.export_model:
        export_model(model, model_info, args.export_model)
    reference = load_reference(args.reference)
    reference_sequence = reference["sequence_5to3"]
    reference_configurations = reference["forward_tss_profile"][:5]
    reference_cores = [
        reference_sequence[config["tss_offset_0based"] - 50:config["tss_offset_0based"]]
        for config in reference_configurations
    ]
    all_log2_predictions = model.predict(encode(cores + reference_cores))
    log2_predictions = all_log2_predictions[:len(cores)]
    reference_log2_predictions = all_log2_predictions[len(cores):]
    # Invert the trained log2 target before comparing linear activity. The
    # upstream full script instead divides log predictions and swaps filenames.
    activities = np.exp2(log2_predictions)
    reference_activities = np.exp2(reference_log2_predictions)
    reference_activity_min = reference_activities.min()
    reference_activity_max = reference_activities.max()
    reference_activity = float(reference_activities[0])
    relative = activities / reference_activity
    reference_info = {
        **reference,
        "reference_file": str(args.reference),
        "reference_file_sha256": REFERENCE_SHA256,
        "value_type": "Prediction from the same DH5alpha SVR and fitted model as the candidates; not a notebook fluorescence measurement",
        "predicted_log2_activity": float(reference_log2_predictions[0]),
        "predicted_activity_training_table_au": reference_activity,
        "normalization": "Predicted linear candidate activity divided by predicted pptb_fU core activity; reference = 1.0 or 100 percent",
        "top5_framing_tss_sensitivity": [
            {
                "assumed_tss_offset_0based": config["tss_offset_0based"],
                "framing_model_dG_total": config["dG_total"],
                "core_sequence_50bp": core,
                "predicted_log2_activity": float(log2_value),
                "predicted_activity_training_table_au": float(activity),
            }
            for config, core, log2_value, activity in zip(
                reference_configurations, reference_cores, reference_log2_predictions, reference_activities
            )
        ],
        "sensitivity_warning": "Scenario range over the five lowest-energy reference TSS configurations, selected before SVR scoring; not a confidence interval or a maximum-SVR primary prediction",
    }
    lower, upper = model_info["training_activity_range_au"]
    results = []
    for input_order, (row, core, log2_value, activity, normalized) in enumerate(zip(candidates, cores, log2_predictions, activities, relative), 1):
        sequence = row["summit_seq"]
        percent_min = float(100 * activity / reference_activity_max)
        percent_max = float(100 * activity / reference_activity_min)
        results.append({
            "candidate_id": row["candidate_id"],
            "input_order": input_order,
            "target_host": HOST,
            "model_training_host": HOST,
            "predicted_log2_activity": float(log2_value),
            "predicted_activity_training_table_au": float(activity),
            "reference_promoter": REFERENCE_PROMOTER,
            "reference_predicted_activity_training_table_au": reference_activity,
            "reference_assumed_tss_offset_0based": reference["assumed_tss_offset_0based"],
            "reference_core_sequence_50bp": reference_cores[0],
            "predicted_fraction_of_reference": float(normalized),
            "predicted_percent_of_reference": float(100 * normalized),
            "predicted_percent_of_reference_top5_tss_min": percent_min,
            "predicted_percent_of_reference_top5_tss_max": percent_max,
            "predicted_fraction_of_reference_top5_tss_min": percent_min / 100,
            "predicted_fraction_of_reference_top5_tss_max": percent_max / 100,
            "outside_training_activity_range": bool(activity < lower or activity > upper),
            "assumed_tss_offset_0based": args.tss_offset,
            "core_start_offset_0based": core_start,
            "core_end_offset_0based_exclusive": args.tss_offset,
            "core_sequence_50bp": core,
            "source_sequence_81bp": sequence,
            "source_sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "source_chromosome": row["chromosome"],
            "source_genomic_strand": row["strand"],
            "source_tss_position": row["tss_position"],
        })
    results.sort(key=lambda row: (-row["predicted_activity_training_table_au"], row["input_order"]))
    results = [{"dh5alpha_strength_rank": rank, **row} for rank, row in enumerate(results, 1)]
    manifest = {
        "model": "Wang2020 DH5alpha-trained RBF SVR; " + ("numeric fitted-model inference" if args.model else "CPU reproduction of published training procedure"),
        "target_host": HOST,
        "model_info": model_info,
        "reference_info": reference_info,
        "input_file": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "input_selection": f"First {args.top_n} canonical records in file order; no reverse-complementing or padding",
        "assumed_tss_offset_0based": args.tss_offset,
        "tss_warning": "Native inferred anchor, not an experimentally mapped DH5alpha TSS",
        "output_transformation": "activity = 2 ** predicted_log2_activity; fraction = activity / predicted_pptb_fU_core_activity; percent = 100 * fraction",
        "interpretation": "Conditional 50-bp-core reporter-activity estimates relative to the same-model predicted pptb_fU core; not measured fold-changes, whole-insert expression, validated RPU or probabilities; reference TSS uncertainty is material",
        "environment": {
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            **({name: importlib.metadata.version(name) for name in ("scipy", "scikit-learn")} if not args.model else {}),
        },
        "results": results,
    }
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with args.out_prefix.with_suffix(".tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(results)
    args.out_prefix.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    csv_rows = [
        {
            "Rank": row["dh5alpha_strength_rank"],
            "Candidate": row["candidate_id"],
            "Predicted_activity_training_table_AU": row["predicted_activity_training_table_au"],
            "Reference_promoter": row["reference_promoter"],
            "Predicted_reference_activity_training_table_AU": row["reference_predicted_activity_training_table_au"],
            "Primary_relative_strength_pptb_fU_equals_1": row["predicted_fraction_of_reference"],
            "Relative_strength_range_pptb_fU_equals_1": f"{row['predicted_fraction_of_reference_top5_tss_min']:.3f} to {row['predicted_fraction_of_reference_top5_tss_max']:.3f}",
            "Relative_strength_min_reference_TSS_scenarios": row["predicted_fraction_of_reference_top5_tss_min"],
            "Relative_strength_max_reference_TSS_scenarios": row["predicted_fraction_of_reference_top5_tss_max"],
            "Percent_of_predicted_pptb_fU": row["predicted_percent_of_reference"],
            "Percent_min_reference_TSS_scenarios": row["predicted_percent_of_reference_top5_tss_min"],
            "Percent_max_reference_TSS_scenarios": row["predicted_percent_of_reference_top5_tss_max"],
            "Candidate_TSS_offset_0based": row["assumed_tss_offset_0based"],
            "Reference_TSS_offset_0based": row["reference_assumed_tss_offset_0based"],
            "Outside_training_activity_range": str(row["outside_training_activity_range"]).lower(),
            "Model": manifest["model"],
            "Interpretation": manifest["interpretation"],
            "Scenario_range_note": reference_info["sensitivity_warning"],
        }
        for row in results
    ]
    with args.out_prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"DH5alpha SVR: {model_info['training_records']} training records; {model_info['best_parameters']}; selection-CV R2={model_info['best_selection_cv_mean_r2']:.6f}")
    print(f"Reference: predicted {REFERENCE_PROMOTER} = {reference_activity:.6f} training-table AU (1.0x / 100 percent); inferred TSS offset={reference['assumed_tss_offset_0based']}")
    print(f"Reference top-five-TSS activity range: {reference_activity_min:.6f} to {reference_activity_max:.6f} AU (scenario range, not a confidence interval)")
    for row in results:
        print(f"{row['dh5alpha_strength_rank']:2d} {row['candidate_id']} activity_AU={row['predicted_activity_training_table_au']:.6f} {REFERENCE_PROMOTER}_fraction={row['predicted_fraction_of_reference']:.6f} percent={row['predicted_percent_of_reference']:.2f}")
    print("Wrote", *(args.out_prefix.with_suffix(suffix) for suffix in (".tsv", ".json", ".csv")),
          "(primary ratio and reference-TSS scenario range)")


if __name__ == "__main__":
    main()
