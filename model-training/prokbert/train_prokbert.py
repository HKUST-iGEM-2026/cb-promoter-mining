#!/usr/bin/env python3
"""
T18 — fine-tune ProkBERT-mini for promoter classification on the T16 tokens.
Owner: Sanaa + Oak.

Reads the PRE-TOKENISED T16 output (results/T16/prokbert/{train,val,test}.parquet — columns
input_ids, attention_mask, label, weight, …), fine-tunes `neuralbioinfo/prokbert-mini` with a
2-class head, and writes a model + metrics.json.

TWO CONTRACTS THIS SCRIPT HONORS (do not break them — T21 + T15/T16 depend on them):
  1. SAMPLE WEIGHT. Each row's `weight` (from T15: silver 0.3 + the 50/50 class-balance factor) is
     applied as a PER-SAMPLE LOSS WEIGHT via WeightedTrainer. Without this the AT-trap guard is inert
     and the model trains negative-skewed + silver-dominated. `--no-weight` disables it (ablation only).
  2. SWEEP CLI. Accepts `--sweep-trial`, `--output-dir`, and every hyperparameter T21 sweeps as a CLI
     flag; writes `<output-dir>/metrics.json` with `val_auprc` (the value T21 maximises) + prints
     "Validation AUPRC: <x>". Selection metric is **AUPRC** (average_precision_score), never F1/accuracy.

Base model is plain prokbert-mini (NOT -mini-long, NOT -promoter): the 81 bp windows are short. The
2-class head is randomly initialised on top of the pretrained backbone and learned here.

Run (standalone):
  python train_prokbert.py --data-dir results/T16/prokbert --output-dir results/T18
Run (driven by the T21 sweep): T21 appends --sweep-trial N --output-dir <trial dir> --learning-rate … etc.
NB on offline compute nodes, pre-stage the tokenizer/model first (HF_HOME + HF_HUB_OFFLINE=1).
"""
import argparse
import inspect
import json
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoModelForSequenceClassification, Trainer,
                          TrainingArguments, EarlyStoppingCallback, set_seed)
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

PROKBERT_ID = "neuralbioinfo/prokbert-mini"


def _warmup_kwargs(ratio=None, steps=None):
    """v5 DEPRECATED `warmup_ratio` (still present, but emits a warning; `warmup_steps` now accepts a
    float in [0,1) = ratio). v4 keeps `warmup_ratio` as the ratio arg (its `warmup_steps` is int-only).
    Feature-detect by the `warmup_steps` annotation so this runs CLEAN (no deprecation warning) on BOTH."""
    params = inspect.signature(TrainingArguments.__init__).parameters
    if ratio is not None:
        ws = params.get("warmup_steps")
        if ws is not None and ws.annotation in (float, "float"):   # v5: warmup_steps takes a float ratio
            return {"warmup_steps": ratio}
        if "warmup_ratio" in params:                                # v4: the dedicated ratio arg
            return {"warmup_ratio": ratio}
        if ws is not None:
            return {"warmup_steps": ratio}
    if steps is not None:
        if "warmup_steps" in params:
            return {"warmup_steps": steps}
    return {}


# --------------------------------------------------------------------------- data
class TokenDataset(Dataset):
    """A T16 parquet of PRE-TOKENISED rows -> dicts the Trainer consumes. Inputs are fixed-length
    (T16 padded per model), so no dynamic padding/collator is needed."""
    def __init__(self, path):
        df = pd.read_parquet(path)
        for c in ("input_ids", "attention_mask", "label", "weight"):
            if c not in df.columns:
                raise SystemExit(f"{path}: missing column '{c}' — is this a T16 output?")
        self.input_ids = [list(map(int, x)) for x in df["input_ids"]]
        self.attention_mask = [list(map(int, x)) for x in df["attention_mask"]]
        self.labels = df["label"].astype(int).tolist()
        self.weights = df["weight"].astype(float).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": torch.tensor(self.input_ids[i], dtype=torch.long),
                "attention_mask": torch.tensor(self.attention_mask[i], dtype=torch.long),
                "labels": torch.tensor(self.labels[i], dtype=torch.long),
                "weight": torch.tensor(self.weights[i], dtype=torch.float)}


# --------------------------------------------------------------------------- weighted loss
class WeightedTrainer(Trainer):
    """Apply the T15 per-sample `weight` to the loss. Pops `weight` before the forward (the model
    never sees it); keeps `labels` in the batch so eval/compute_metrics still find them."""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weight")
        outputs = model(**inputs)                      # inputs still carries input_ids/attention_mask/labels
        logits = outputs.logits
        labels = inputs["labels"]
        per = nn.functional.cross_entropy(logits, labels, reduction="none")
        loss = (per * weights).sum() / weights.sum().clamp(min=1e-8)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits = getattr(eval_pred, "predictions", None)      # v5 EvalPrediction: attribute access, not tuple-unpack
    labels = getattr(eval_pred, "label_ids", None)
    if logits is None:                                    # tuple-style fallback (older/edge cases)
        logits, labels = eval_pred[0], eval_pred[1]
    if isinstance(logits, (tuple, list)):                 # some models wrap logits in a tuple
        logits = logits[0]
    logits = np.asarray(logits, dtype=np.float32)
    probs = torch.softmax(torch.from_numpy(logits), dim=1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)
    return {"auprc": float(average_precision_score(labels, probs)),       # the selection metric
            "auroc": float(roc_auc_score(labels, probs)),
            "f1": float(f1_score(labels, preds, zero_division=0))}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="T18 ProkBERT-mini fine-tune")
    ap.add_argument("--data-dir", default="results/T16/prokbert", help="dir with train/val/test.parquet")
    ap.add_argument("--output-dir", default="results/T18")
    ap.add_argument("--model-id", default=PROKBERT_ID)
    ap.add_argument("--sweep-trial", type=int, default=-1, help="set by T21; -1 = standalone run")
    # --- hyperparameters T21 sweeps (defaults are sane starting points) ---
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-epochs", type=int, default=10)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--dropout-rate", type=float, default=0.1)
    ap.add_argument("--no-weight", action="store_true", help="ABLATION: ignore the T15 sample weight")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train = TokenDataset(f"{args.data_dir}/train.parquet")
    val = TokenDataset(f"{args.data_dir}/val.parquet")
    test = TokenDataset(f"{args.data_dir}/test.parquet")

    # dropout is set via the config (the custom ProkBERT config attr names vary -> best-effort).
    config = AutoConfig.from_pretrained(args.model_id, num_labels=2, trust_remote_code=True)
    for attr in ("hidden_dropout_prob", "classifier_dropout", "dropout"):
        if hasattr(config, attr):
            setattr(config, attr, args.dropout_rate)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, config=config, trust_remote_code=True)   # fresh 2-class head on the pretrained backbone

    targs = TrainingArguments(
        output_dir=f"{args.output_dir}/hf",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        num_train_epochs=args.num_epochs,
        **_warmup_kwargs(ratio=args.warmup_ratio),   # v5-safe: warmup_ratio was removed (see _warmup_kwargs)
        weight_decay=args.weight_decay,
        eval_strategy="epoch",                 # NB transformers>=4.56: eval_strategy, NOT evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_auprc",
        greater_is_better=True,
        save_total_limit=1,
        remove_unused_columns=False,   # CRITICAL: keep the per-sample `weight` column. The Trainer strips
                                       # non-model-input columns by default -> drops `weight` before
                                       # compute_loss -> KeyError. The AT-trap guard depends on this.
        seed=args.seed,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),      # H800 -> bf16 (safer than fp16)
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to=[],
        logging_steps=50,
    )
    TrainerCls = Trainer if args.no_weight else WeightedTrainer
    trainer = TrainerCls(model=model, args=targs, train_dataset=train, eval_dataset=val,
                         compute_metrics=compute_metrics,
                         callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    trainer.train()

    val_m = trainer.evaluate(val)
    test_m = trainer.evaluate(test)
    metrics = {
        "model": "prokbert", "trial": args.sweep_trial, "weighted": (not args.no_weight),
        "val_auprc": val_m["eval_auprc"], "val_auroc": val_m["eval_auroc"], "val_f1": val_m["eval_f1"],
        "test_auprc": test_m["eval_auprc"], "test_auroc": test_m["eval_auroc"], "test_f1": test_m["eval_f1"],
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    trainer.save_model(os.path.join(args.output_dir, "best_model"))
    print(f"Validation AUPRC: {metrics['val_auprc']:.4f}")     # T21 parses this (or metrics.json)
    print(f"Test AUPRC: {metrics['test_auprc']:.4f}")


if __name__ == "__main__":
    main()
