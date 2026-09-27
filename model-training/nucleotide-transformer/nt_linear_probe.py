#!/usr/bin/env python3
"""
Frozen-NT linear probe — does Nucleotide Transformer's PRETRAINED embeddings already
separate promoters from non-promoters, before any fine-tuning?

This is the BASELINE/ablation that answers "is the signal already in the embeddings?"
(it is NOT the primary NT validation — that's the T19 LoRA fine-tune evaluated with AUPRC).
It reuses the EXACT T16 NT splits so the number is apples-to-apples with the T19 fine-tune:
the gap (LoRA AUPRC - probe AUPRC) = what fine-tuning buys.

Method: frozen NT -> mean-pool the last hidden state over real tokens -> logistic regression -> AUPRC.
This mirrors the Nucleotide Transformer paper's own "probing" evaluation.

Run on a cluster (needs the NT checkpoint pre-staged + a GPU):
    python nt_linear_probe.py --data-dir results/T16/nt
"""
import argparse
import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score, roc_auc_score

NT_ID = "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species"


def embed(path, model, device, batch=32):
    """Mean-pooled frozen embedding per row of a T16 NT parquet (input_ids/attention_mask/label)."""
    df = pd.read_parquet(path)
    ids = [list(map(int, x)) for x in df["input_ids"]]
    am = [list(map(int, x)) for x in df["attention_mask"]]
    y = df["label"].astype(int).to_numpy()
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ids), batch):
            I = torch.tensor(ids[i:i + batch], device=device)
            A = torch.tensor(am[i:i + batch], device=device)
            h = model(input_ids=I, attention_mask=A).last_hidden_state    # (B, L, H)
            m = A.unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)               # mean over real tokens
            out.append(pooled.float().cpu().numpy())
    return np.concatenate(out), y


def main():
    ap = argparse.ArgumentParser(description="Frozen-NT linear probe (baseline for T19)")
    ap.add_argument("--data-dir", default="results/T16/nt", help="dir with train/val/test.parquet")
    ap.add_argument("--model-id", default=NT_ID)
    ap.add_argument("--save-head", default=None,
                    help="joblib path to save the fitted head -> reuse in nt_sliding_window.py (PT5/PT6-style genome scan)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # base model (NOT ...ForSequenceClassification) — we want raw embeddings, frozen.
    model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True).to(dev)

    Xtr, ytr = embed(f"{args.data_dir}/train.parquet", model, dev)
    Xte, yte = embed(f"{args.data_dir}/test.parquet", model, dev)

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]

    auprc = average_precision_score(yte, p)
    auroc = roc_auc_score(yte, p)
    print(f"FROZEN-NT linear probe  ->  test AUPRC {auprc:.3f} | AUROC {auroc:.3f}")
    print(f"no-skill AUPRC baseline = test prevalence = {yte.mean():.3f}")
    print("Compare to the T19 LoRA fine-tuned AUPRC: (LoRA - probe) = the value of fine-tuning.")
    if args.save_head:
        joblib.dump(clf, args.save_head)
        print(f"saved probe head -> {args.save_head}  (feed it to nt_sliding_window.py for the genome-wide PT5/PT6 test)")
    # Optional: NT paper found MIDDLE layers probe best -> rerun with output_hidden_states=True
    #           and pool a chosen hidden_states[k] instead of last_hidden_state.


if __name__ == "__main__":
    main()
