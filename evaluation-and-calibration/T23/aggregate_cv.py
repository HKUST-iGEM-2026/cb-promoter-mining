#!/usr/bin/env python3
"""
T23 — aggregate the per-fold scores into cross-validation mean +/- std per model.

This is the actual T23 deliverable: it turns 5 folds of numbers into
"CV AUPRC = 0.xx +/- 0.0y" per model, so we report STABILITY across cluster-disjoint
splits rather than a single lucky test set.

Reads the per-fold metric files written by predict_and_score.py (one per fold, per model):
    <scores-dir>/fold_XX/<model>_metrics.json      (model in {prokbert, nt, cnn})
Writes:
    <out>/cv_summary.json    nested: model -> metric -> {mean, std, n, folds}
    <out>/cv_summary.csv     one row per (model, metric)

Pure standard library + numpy, so it runs in ANY env (even base).

Run:
    python evaluation-and-calibration/T23/aggregate_cv.py \
      --scores-dir results/T23/scores --out results/T23
(run_t23.sh submits this automatically after all fold jobs succeed.)
"""
import argparse
import glob
import json
import os

import numpy as np

METRICS = ["auprc", "auroc", "mcc", "f1"]
MODELS = ["prokbert", "nt", "cnn"]


def main():
    ap = argparse.ArgumentParser(description="Aggregate T23 per-fold scores into CV mean +/- std.")
    ap.add_argument("--scores-dir", default="results/T23/scores",
                    help="dir containing fold_XX/<model>_metrics.json (from predict_and_score.py)")
    ap.add_argument("--out", default="results/T23")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    summary = {}
    csv_rows = ["model,metric,mean,std,n_folds,per_fold"]
    for model in MODELS:
        files = sorted(glob.glob(os.path.join(args.scores_dir, "*", f"{model}_metrics.json")))
        if not files:
            continue
        collected = {m: [] for m in METRICS}
        for f in files:
            with open(f) as fh:
                d = json.load(fh)
            for m in METRICS:
                v = d.get(m)
                if isinstance(v, (int, float)):
                    collected[m].append(float(v))

        summary[model] = {"n_folds_found": len(files)}
        for m in METRICS:
            vals = collected[m]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            summary[model][m] = {"mean": mean, "std": std, "n": len(arr), "folds": vals}
            csv_rows.append(f'{model},{m},{mean:.4f},{std:.4f},{len(arr)},'
                            f'"{[round(v, 4) for v in vals]}"')

        printable = "  ".join(
            f"{m}={summary[model][m]['mean']:.4f}+/-{summary[model][m]['std']:.4f}"
            for m in METRICS if m in summary[model])
        print(f"{model:9s} ({len(files)} folds):  {printable}")

    if not summary:
        raise SystemExit(
            f"No per-fold metrics found under {args.scores_dir}/*/<model>_metrics.json.\n"
            "Run the fold trainings + predict_and_score first (bash results/T23/run_t23.sh).")

    with open(os.path.join(args.out, "cv_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(args.out, "cv_summary.csv"), "w") as fh:
        fh.write("\n".join(csv_rows) + "\n")
    print(f"\nWrote {args.out}/cv_summary.json and {args.out}/cv_summary.csv")


if __name__ == "__main__":
    main()
