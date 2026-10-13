#!/usr/bin/env python3
"""
T22 — regenerate test-set predictions for a FINAL ensemble model and score the full metric suite.

WHY THIS EXISTS: the trainers (T18/T19/T20) only saved AGGREGATE metrics.json (AUPRC/AUROC/F1), NOT
per-example probabilities. MCC (and any threshold-based metric) needs the confusion matrix, i.e. the
raw predictions — so this script re-runs inference to produce them. The saved predictions ALSO feed
T26 (calibration) and T27 (empirical FDR), so this is the reusable prediction backbone, not a one-off.

AUPRC uses sklearn.average_precision_score, never auc(recall, precision).

TWO-ENV SPLIT — run ONE model at a time in the right env:
  # in $ENV_MAIN (transformers 5.3.0):
  python predict_and_score.py --model prokbert \
     --model-path <SCAF>/results/T21_final/prokbert/best_model \
     --data-dir  <SCAF>/results/T16/prokbert --out results/T22
  python predict_and_score.py --model cnn \
     --model-path <SCAF>/results/T20/best_model.pt \
     --data-dir  <SCAF>/results/T16/cnn --out results/T22
  # in $ENV_NT (transformers 4.57) — NT only:
  python predict_and_score.py --model nt \
     --model-path <SCAF>/results/T21_final/nt_lora/best_adapter \
     --data-dir  <SCAF>/results/T16/nt --out results/T22
  (<SCAF> = workflow)

Writes  <out>/<model>_test_predictions.csv  (label, prob, pred)  and  <out>/<model>_metrics.json.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef, f1_score)

NT_ID = "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species"


def _load_v2_module():
    """Import pipeline/training/cnn/train_cnn_v2.py as a module (for its model +
    inference helpers)."""
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "..", "training", "cnn", "train_cnn_v2.py")
    spec = importlib.util.spec_from_file_location("train_cnn_v2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_test(data_dir, model):
    """Return (encoded_inputs, y_true) for the T16 test split of this model."""
    if model == "cnn":
        z = np.load(f"{data_dir}/test.npz", allow_pickle=False)
        if z["X"].shape[1:] != (81, 4):
            raise ValueError(f"expected CNN X (N,81,4), got {z['X'].shape}")
        X = np.transpose(z["X"], (0, 2, 1)).astype("float32")   # (N,81,4) -> (N,4,81) for Conv1d
        return {"X": X}, z["label"].astype(int)
    if model == "cnn_v2":
        # v2 needs raw sequences (it computes one-hot/ENAC/k-mer features itself).
        # Reuse the SAME T16/cnn one-hot npz as the prod CNN, reconstructed to strings.
        z = np.load(f"{data_dir}/test.npz", allow_pickle=False)
        seqs = _load_v2_module().reconstruct_sequences_from_onehot(z["X"])
        return {"seqs": seqs}, z["label"].astype(int)
    df = pd.read_parquet(f"{data_dir}/test.parquet")
    ids = torch.tensor([list(map(int, x)) for x in df["input_ids"]], dtype=torch.long)
    am = torch.tensor([list(map(int, x)) for x in df["attention_mask"]], dtype=torch.long)
    return {"input_ids": ids, "attention_mask": am}, df["label"].astype(int).values


@torch.no_grad()
def probs_transformer(model, enc, device, bs=64):
    out = []
    n = len(enc["input_ids"])
    for i in range(0, n, bs):
        o = model(input_ids=enc["input_ids"][i:i + bs].to(device),
                  attention_mask=enc["attention_mask"][i:i + bs].to(device))
        out.append(torch.softmax(o.logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def probs_cnn(model, X, device, bs=256):
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        out.append(torch.softmax(model(xb).float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def build_cnn(model_path, device):
    """Rebuild PromoterCNN with the SAME config it was trained with (read from its metrics.json,
    which stores the full config) so the state_dict loads — the standalone CNN was trained with
    BatchNorm OFF, so we must not assume the current default."""
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "train_cnn", os.path.join(here, "..", "..", "training", "cnn", "train_cnn.py"))
    tcnn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tcnn)
    # find the config next to the checkpoint (best_model.pt -> metrics.json in the same dir)
    cfg = {}
    mpath = os.path.join(os.path.dirname(model_path), "metrics.json")
    if os.path.exists(mpath):
        cfg = json.load(open(mpath)).get("config", {})
    keys = ("conv1_filters", "conv2_filters", "conv3_filters", "kernel_size",
            "dropout_rate", "use_batch_norm")
    kw = {k: cfg[k] for k in keys if k in cfg}
    model = tcnn.PromoterCNN(input_channels=4, sequence_length=81, **kw).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.eval()


def main():
    ap = argparse.ArgumentParser(description="T22 inference + full metric suite for one final model")
    ap.add_argument("--model", required=True, choices=["prokbert", "nt", "cnn", "cnn_v2"])
    ap.add_argument("--model-path", required=True, help="HF dir (prokbert) / adapter dir (nt) / best_model.pt (cnn, cnn_v2)")
    ap.add_argument("--base-model", default=NT_ID, help="base checkpoint for the NT LoRA adapter")
    ap.add_argument("--data-dir", required=True, help="T16 dir for this model (has test.parquet or test.npz)")
    ap.add_argument("--out", default="results/T22")
    ap.add_argument("--threshold", type=float, default=0.5, help="decision threshold for MCC/F1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc, y = load_test(args.data_dir, args.model)

    if args.model == "cnn":
        model = build_cnn(args.model_path, device)
        probs = probs_cnn(model, enc["X"], device)
    elif args.model == "cnn_v2":
        v2 = _load_v2_module()
        cfg = {}
        cfg_path = os.path.join(os.path.dirname(args.model_path), "config.json")
        if os.path.exists(cfg_path):
            cfg = json.load(open(cfg_path))
        model = v2.build_model_from_config(cfg, device=str(device))
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        probs = v2.predict_proba(model, enc["seqs"], str(device),
                                 enac_window=int(cfg.get("enac_window", 5)))
    elif args.model == "prokbert":
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path, trust_remote_code=True).to(device).eval()
        probs = probs_transformer(model, enc, device)
    else:  # nt — run this in the $ENV_NT env
        from transformers import AutoModelForSequenceClassification
        from peft import PeftModel
        base = AutoModelForSequenceClassification.from_pretrained(
            args.base_model, num_labels=2, trust_remote_code=True)
        model = PeftModel.from_pretrained(base, args.model_path).to(device).eval()
        probs = probs_transformer(model, enc, device)

    preds = (probs >= args.threshold).astype(int)
    pd.DataFrame({"label": y, "prob": probs, "pred": preds}).to_csv(
        f"{args.out}/{args.model}_test_predictions.csv", index=False)

    metrics = {
        "model": args.model, "n": int(len(y)), "n_pos": int(y.sum()),
        "prevalence": float(y.mean()), "threshold": args.threshold,
        "auroc": float(roc_auc_score(y, probs)),
        "auprc": float(average_precision_score(y, probs)),   # use average_precision, NOT auc(recall, precision)
        "mcc":   float(matthews_corrcoef(y, preds)),
        "f1":    float(f1_score(y, preds, zero_division=0)),
    }
    json.dump(metrics, open(f"{args.out}/{args.model}_metrics.json", "w"), indent=2)
    print(f"{args.model}: " + " | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                                         for k, v in metrics.items() if k not in ("model",)))
    print(f"  -> {args.out}/{args.model}_test_predictions.csv  (+ {args.model}_metrics.json)")


if __name__ == "__main__":
    main()
