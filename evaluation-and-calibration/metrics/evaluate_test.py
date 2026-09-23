#!/usr/bin/env python3
"""
T22 — evaluate a FINAL model on the T16 TEST set: full metric suite (AUROC / AUPRC / MCC / F1 /
precision / recall / accuracy) AND save the raw per-example predictions.

The trainers only stored aggregate metrics.json (no per-example scores), so MCC — which is built from
the confusion matrix at a threshold — can't be computed from them. This re-runs inference to get the
predictions, computes everything, and writes predictions.csv so MCC (at any threshold), T26 calibration,
and T27 FDR all have what they need.

Run PER MODEL in the matching env (see infrastructure/conda-environments/NT_ENV.md):
  prokbert / cnn -> promoter-mining-hpc4  (transformers 5.3.0)
  nt             -> nt-tf4x               (transformers 4.57)   [+ export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"]

Examples (run from .../infrastructure/T3/snakemake_scaffold so results/... resolves):
  python ../../evaluation-and-calibration/metrics/evaluate_test.py --model prokbert \
      --model-path results/T21_final/prokbert/best_model --test results/T16/prokbert/test.parquet --out results/T22/prokbert
  python ../../evaluation-and-calibration/metrics/evaluate_test.py --model cnn \
      --model-path results/T20/best_model.pt            --test results/T16/cnn/test.npz     --out results/T22/cnn
  python ../../evaluation-and-calibration/metrics/evaluate_test.py --model nt \
      --model-path results/T21_final/nt_lora/best_adapter --test results/T16/nt/test.parquet  --out results/T22/nt
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)

NT_ID = "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species"


def _probs_transformer(model, ids, mask, device, batch=64):
    model.eval().to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(ids), batch):
            bi = torch.tensor(ids[i:i + batch], dtype=torch.long, device=device)
            bm = torch.tensor(mask[i:i + batch], dtype=torch.long, device=device)
            logits = model(input_ids=bi, attention_mask=bm).logits.float()
            out.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def _probs_cnn(model, X, device, batch=256):
    model.eval().to(device)
    X = np.transpose(X.astype(np.float32), (0, 2, 1))       # (N,81,4) -> (N,4,81) for Conv1d
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch], device=device)
            logits = model(xb).float()
            out.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def load_and_predict(kind, model_path, test_path, device):
    if kind == "cnn":
        import sys
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(here, "../../model-training/cnn"))
        from train_cnn import PromoterCNN
        z = np.load(test_path)
        X, labels = z["X"], z["label"].astype(int)
        sd = torch.load(model_path, map_location=device)
        has_bn = any("running_mean" in k for k in sd.keys())   # rebuild the exact trained architecture
        model = PromoterCNN(input_channels=4, sequence_length=81, use_batch_norm=has_bn)
        model.load_state_dict(sd)
        return labels, _probs_cnn(model, X, device)

    import pandas as pd
    from transformers import AutoModelForSequenceClassification
    df = pd.read_parquet(test_path)
    ids = [list(map(int, x)) for x in df["input_ids"]]
    mask = [list(map(int, x)) for x in df["attention_mask"]]
    labels = df["label"].astype(int).to_numpy()

    if kind == "prokbert":
        model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
    elif kind == "nt":
        from peft import PeftModel
        base = AutoModelForSequenceClassification.from_pretrained(NT_ID, num_labels=2, trust_remote_code=True)
        model = PeftModel.from_pretrained(base, model_path)    # LoRA adapter incl. the saved classifier head
    else:
        raise SystemExit(f"unknown model kind: {kind}")
    return labels, _probs_transformer(model, ids, mask, device)


def full_metrics(labels, probs, thr=0.5):
    preds = (probs >= thr).astype(int)
    return {
        "auroc": float(roc_auc_score(labels, probs)),
        "auprc": float(average_precision_score(labels, probs)),   # §7: average_precision, NOT auc(recall, precision)
        "mcc": float(matthews_corrcoef(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "threshold": float(thr),
        "n": int(len(labels)),
        "n_pos": int(labels.sum()),
        "prevalence": float(labels.mean()),
    }


def main():
    ap = argparse.ArgumentParser(description="T22: full metric suite + raw predictions on the test set")
    ap.add_argument("--model", required=True, choices=["prokbert", "nt", "cnn"])
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--test", required=True, help="T16 test file: parquet (prokbert/nt) or npz (cnn)")
    ap.add_argument("--out", required=True, help="output dir for metrics.json + predictions.csv")
    ap.add_argument("--threshold", type=float, default=0.5, help="decision threshold for MCC/F1 (default 0.5)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels, probs = load_and_predict(args.model, args.model_path, args.test, device)
    m = full_metrics(labels, probs, args.threshold)
    m["model"] = args.model

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    with open(os.path.join(args.out, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "prob"])
        for y, p in zip(labels, probs):
            w.writerow([int(y), float(p)])

    print(f"{args.model}: AUROC {m['auroc']:.4f} | AUPRC {m['auprc']:.4f} | MCC {m['mcc']:.4f} | "
          f"F1 {m['f1']:.4f} | precision {m['precision']:.4f} | recall {m['recall']:.4f} "
          f"| n={m['n']} pos={m['n_pos']} ({m['prevalence']:.3f})")
    print(f"saved: {args.out}/metrics.json + predictions.csv")


if __name__ == "__main__":
    main()
