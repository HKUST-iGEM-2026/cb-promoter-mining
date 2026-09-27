#!/usr/bin/env python3
"""
finetune_ntv3_classifier.py — full fine-tune of NTv3 (650M) as a PROMOTER CLASSIFIER on the
xs + context dataset (T15_xs_ctx).

Why this exists: the old `fine_tune_ntv3.py` loaded the masked-LM head and set `labels = input_ids`
with no masking / no MLM collator, so it trained a degenerate copy task (eval_loss ~1e-7) and learned
nothing about promoters. It is NOT a classifier and must not be used. This script replaces it.

What it does:
  * Loads the real NTv3 backbone via AutoModelForMaskedLM (the only class the remote-code auto_map
    registers), trust_remote_code=True, and taps its final full-length hidden state
    (forward(output_hidden_states=True) -> MaskedLMOutput.hidden_states[-1], shape (B, L, embed_dim)).
  * Mean-pools that over the sequence (attention-mask aware) and adds a linear classification head.
    The WHOLE model is trained (backbone + head) = full fine-tune.
  * Trains on the promoter LABEL, tokenizing SEQUENCE_CONTEXT (the long window), with the per-sample
    WEIGHT column applied as a weighted cross-entropy — the same label-confidence x class-balance
    weighting every other ensemble model uses (T15).
  * Selects the best epoch on, and reports, WEIGHTED AUPRC (sklearn.average_precision_score) — the
    project's metric convention — computed on val AND test with the weight column.

NTv3 specifics: single-base tokenization; sequence length must be a
multiple of 128 (7 downsamples). T15_xs_ctx's `sequence_context` is built at a fixed 384 bp (=3x128),
N-padded only at contig ends, so every row is already a valid fixed length — no [PAD] needed, and the
attention mask is all-ones. Keep --max-length equal to the context length the dataset was built at.

ENV: the HF/torch env the NTv3 remote code runs in (the one that produced results_ntv3_full) — NOT the
`ntv3-jax` env (that's the separate InstaDeep JAX package). trust_remote_code pulls the bundled
modeling_ntv3_pretrained.py from the HF cache (set HF_HOME / HF_HUB_OFFLINE on the cluster).

  python finetune_ntv3_classifier.py \
      --data-dir $PROMOTER_RESULTS/T15_xs_ctx \
      --model-id InstaDeepAI/NTv3_650M_pre --seq-col sequence_context --max-length 384 \
      --out ./results_ntv3_xs_ctx --epochs 3 --batch-size 8 --grad-accum 4 --lr 1e-5
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import (AutoModelForMaskedLM, AutoTokenizer, Trainer,
                          TrainingArguments, default_data_collator)


class NTv3Classifier(nn.Module):
    """NTv3 backbone (full fine-tune) + mean-pool + linear head. Returns {'loss','logits'} so a
    stock Trainer can drive it; loss is a per-sample-WEIGHTED cross-entropy."""

    def __init__(self, model_id, num_labels=2, dropout=0.1):
        super().__init__()
        self.backbone = AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True)
        hidden = getattr(self.backbone.config, "embed_dim", None) or self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, labels=None, weight=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)
        h = out.hidden_states[-1]                       # (B, L, embed_dim): final full-length repr
        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)     # mask-aware mean pool
        else:
            pooled = h.mean(1)
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            ce = F.cross_entropy(logits, labels, reduction="none")
            if weight is not None:
                loss = (ce * weight).sum() / weight.sum().clamp(min=1e-8)   # weighted mean
            else:
                loss = ce.mean()
        return {"loss": loss, "logits": logits}


def load_split(data_dir, name, seq_col):
    df = pd.read_parquet(os.path.join(data_dir, f"{name}.parquet"))
    for c in (seq_col, "label"):
        if c not in df.columns:
            raise SystemExit(f"{name}.parquet missing '{c}'; has {list(df.columns)}")
    if "weight" not in df.columns:
        df["weight"] = 1.0
    return df[[seq_col, "label", "weight"]].rename(columns={seq_col: "seq"})


def weighted_auprc(y, p, w):
    return (average_precision_score(y, p, sample_weight=w),
            roc_auc_score(y, p, sample_weight=w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="T15_xs_ctx dir with train/val/test.parquet")
    ap.add_argument("--model-id", default="InstaDeepAI/NTv3_650M_pre")
    ap.add_argument("--seq-col", default="sequence_context", help="the long-window column NTv3 reads")
    ap.add_argument("--max-length", type=int, default=384, help="MUST be a multiple of 128 and match the context length")
    ap.add_argument("--out", default="./results_ntv3_xs_ctx")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--resume", action="store_true", help="resume from the last checkpoint in --out (preempt-safe)")
    a = ap.parse_args()
    if a.max_length % 128 != 0:
        raise SystemExit(f"--max-length {a.max_length} must be a multiple of 128 for NTv3.")

    tok = AutoTokenizer.from_pretrained(a.model_id, trust_remote_code=True)
    splits = {n: load_split(a.data_dir, n, a.seq_col) for n in ("train", "val", "test")}
    for n, d in splits.items():
        print(f"{n}: {len(d)} rows, {int(d.label.sum())} pos ({d.label.mean():.1%})")

    def encode(batch):
        enc = tok(batch["seq"], add_special_tokens=False, padding="max_length",
                  max_length=a.max_length, truncation=True)
        enc["labels"] = batch["label"]
        enc["weight"] = [float(w) for w in batch["weight"]]
        return enc

    ds = {}
    for n, d in splits.items():
        hf = Dataset.from_pandas(d, preserve_index=False)
        ds[n] = hf.map(encode, batched=True, remove_columns=hf.column_names)

    model = NTv3Classifier(a.model_id, num_labels=2, dropout=a.dropout)

    def compute_metrics(p):
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        z = logits - logits.max(axis=1, keepdims=True)
        prob1 = np.exp(z)[:, 1] / np.exp(z).sum(axis=1)
        y = p.label_ids
        return {"auprc": float(average_precision_score(y, prob1)),
                "auroc": float(roc_auc_score(y, prob1))}

    args = TrainingArguments(
        output_dir=a.out,
        learning_rate=a.lr,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=a.epochs,
        weight_decay=a.weight_decay,
        warmup_ratio=a.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auprc",
        greater_is_better=True,
        logging_steps=20,
        bf16=True,                       # H800; NTv3 was trained in bf16
        remove_unused_columns=False,     # keep the `weight` column so forward() can use it
        report_to="none",
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["val"],
                      data_collator=default_data_collator, compute_metrics=compute_metrics)
    trainer.train(resume_from_checkpoint=a.resume)

    # ---- final WEIGHTED AUPRC on val + test (the project metric), using the weight column ----
    os.makedirs(a.out, exist_ok=True)
    report = {}
    for n in ("val", "test"):
        pred = trainer.predict(ds[n])
        logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        z = logits - logits.max(axis=1, keepdims=True)
        prob1 = np.exp(z)[:, 1] / np.exp(z).sum(axis=1)
        y = splits[n]["label"].to_numpy()
        w = splits[n]["weight"].to_numpy()
        ap_w, auc_w = weighted_auprc(y, prob1, w)
        ap_u = float(average_precision_score(y, prob1))
        report[n] = {"weighted_auprc": ap_w, "weighted_auroc": auc_w, "unweighted_auprc": ap_u, "n": int(len(y))}
        print(f"{n}: weighted AUPRC {ap_w:.4f} | weighted AUROC {auc_w:.4f} | unweighted AUPRC {ap_u:.4f}")

    # save backbone (save_pretrained -> reloadable with trust_remote_code) + the head + metrics
    self_dir = os.path.join(a.out, "final_model")
    model.backbone.save_pretrained(os.path.join(self_dir, "backbone"))
    tok.save_pretrained(os.path.join(self_dir, "backbone"))
    torch.save(model.classifier.state_dict(), os.path.join(self_dir, "classifier_head.pt"))
    json.dump({"model_id": a.model_id, "seq_col": a.seq_col, "max_length": a.max_length,
               "embed_dim": model.classifier.in_features, "num_labels": model.num_labels,
               "metrics": report}, open(os.path.join(self_dir, "meta.json"), "w"), indent=2)
    print(f"\nsaved -> {self_dir}   (backbone/ + classifier_head.pt + meta.json)")


if __name__ == "__main__":
    main()
