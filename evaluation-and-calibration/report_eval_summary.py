#!/usr/bin/env python3
"""
Collate the T23 (cluster-CV) and T24 (cross-species holdout) results into report-ready tables.

Reads (defaults relative to the scaffold dir where results/ lives):
  results/T23/cv_summary.csv        — written by aggregate_cv.py   (model,metric,mean,std,n_folds,per_fold)
  results/T24/metrics_summary.json  — written by evaluate_external_holdout.py  ({model:{auprc,auroc,mcc,f1,...}})
Writes results/eval_summary_for_report.md (two markdown tables) and prints the same to the console.

Pure standard library — no env / GPU / pandas needed. Run from the scaffold dir:
  python evaluation-and-calibration/report_eval_summary.py
"""
import argparse
import csv
import json
import os

MET = ["auprc", "auroc", "mcc", "f1"]
ORDER = [("prokbert", "ProkBERT"), ("nt", "NT-LoRA"), ("cnn", "CNN")]


def norm(m):
    # T23 uses "nt"; T24 uses "nt_lora" — collapse to one key.
    return "nt" if m in ("nt", "nt_lora") else m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t23", default="results/T23/cv_summary.csv")
    ap.add_argument("--t24", default="results/T24/metrics_summary.json")
    ap.add_argument("--out", default="results/eval_summary_for_report.md")
    a = ap.parse_args()
    for path in (a.t23, a.t24):
        if not os.path.exists(path):
            raise SystemExit(f"Not found: {path} — run T23/T24 first (from the scaffold dir).")

    t23 = {}
    for r in csv.DictReader(open(a.t23)):
        t23.setdefault(norm(r["model"]), {})[r["metric"]] = (float(r["mean"]), float(r["std"]))
    t24 = {norm(k): v for k, v in json.load(open(a.t24)).items()}

    out = []
    def p(s=""):
        out.append(s)
        print(s)

    p("## T23 — cluster-disjoint 5-fold CV (in-distribution; mean ± std over 5 folds)\n")
    p("| Model | AUPRC | AUROC | MCC | F1 |")
    p("|---|---|---|---|---|")
    for k, name in ORDER:
        d = t23.get(k, {})
        c = [f"{d[m][0]:.3f} ± {d[m][1]:.3f}" if m in d else "—" for m in MET]
        p(f"| {name} | {c[0]} | {c[1]} | {c[2]} | {c[3]} |")

    p("\n## T24 — cross-species external holdout (out-of-distribution; balanced 1:1, chance = 0.5)\n")
    p("| Model | AUPRC | AUROC | MCC | F1 |")
    p("|---|---|---|---|---|")
    for k, name in ORDER:
        d = t24.get(k, {})
        c = [f"{d[m]:.3f}" if m in d else "—" for m in MET]
        p(f"| {name} | {c[0]} | {c[1]} | {c[2]} | {c[3]} |")

    with open(a.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
