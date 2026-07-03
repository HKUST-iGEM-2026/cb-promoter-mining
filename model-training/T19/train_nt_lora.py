#!/usr/bin/env python3
"""
T19 — fine-tune Nucleotide Transformer v2 250M (multi-species) with LoRA for promoter classification.
Owner: Sanaa.

Reads the PRE-TOKENISED T16 output (results/T16/nt/{train,val,test}.parquet — columns input_ids,
attention_mask, label, weight, …), wraps `InstaDeepAI/nucleotide-transformer-v2-250m-multi-species`
(an ESM-architecture model) in a LoRA adapter, and writes a model + metrics.json.

Same TWO CONTRACTS as T18 (keep them identical so T21 can drive all three models the same way):
  1. SAMPLE WEIGHT applied as a per-sample loss weight (WeightedTrainer). `--no-weight` = ablation.
  2. SWEEP CLI: `--sweep-trial`, `--output-dir`, every hyperparameter as a flag; write metrics.json
     with `val_auprc` + print "Validation AUPRC: <x>". Selection metric = AUPRC (CLAUDE.md §7).

LoRA: NT is an ESM model, so the attention projections are named query/key/value; we adapt
target_modules=["query","value"] (the standard SEQ_CLS LoRA target) and keep the `classifier` head
trainable via modules_to_save. Backbone stays frozen -> trains ~1% of params, fits a single GPU.

Run (standalone):  python train_nt_lora.py --data-dir results/T16/nt --output-dir results/T19
Driven by T21:     T21 appends --sweep-trial N --output-dir <trial dir> --lora-r … --learning-rate … etc.
NB pre-stage NT on the login node (HF_HOME + HF_HUB_OFFLINE=1); it ships pytorch_model.bin only (no safetensors).
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
from transformers import (AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments,
                          EarlyStoppingCallback, set_seed)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

# --- transformers v5 compat shim -----------------------------------------------------------------
# NT's cached remote code (modeling_esm.py, loaded via trust_remote_code) imports two head-pruning
# utilities that transformers v5 REMOVED. They're only imported (not called) during load/train, so
# re-supplying them here (before from_pretrained runs) lets NT load on transformers 5.x.
import transformers.pytorch_utils as _tpu
if not hasattr(_tpu, "find_pruneable_heads_and_indices"):
    def _fphi(heads, n_heads, head_size, already_pruned_heads):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for h in heads:
            h = h - sum(1 if p < h else 0 for p in already_pruned_heads)
            mask[h] = 0
        mask = mask.view(-1).contiguous().eq(1)
        return heads, torch.arange(len(mask))[mask].long()
    _tpu.find_pruneable_heads_and_indices = _fphi
if not hasattr(_tpu, "prune_linear_layer"):
    def _pll(layer, index, dim=0):
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()
        b = None
        if layer.bias is not None:
            b = layer.bias.clone().detach() if dim == 1 else layer.bias[index].clone().detach()
        size = list(layer.weight.size()); size[dim] = len(index)
        new = nn.Linear(size[1], size[0], bias=layer.bias is not None).to(layer.weight.device)
        new.weight.requires_grad = False; new.weight.copy_(W.contiguous()); new.weight.requires_grad = True
        if b is not None:
            new.bias.requires_grad = False; new.bias.copy_(b.contiguous()); new.bias.requires_grad = True
        return new
    _tpu.prune_linear_layer = _pll
# -------------------------------------------------------------------------------------------------

NT_ID = "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species"


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


# --------------------------------------------------------------------------- data (identical to T18)
class TokenDataset(Dataset):
    """A T16 parquet of PRE-TOKENISED rows -> dicts the Trainer consumes (fixed length; no collator)."""
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


# --------------------------------------------------------------------------- weighted loss (identical to T18)
class WeightedTrainer(Trainer):
    """Per-sample `weight` on the loss; pop weight before the forward, keep labels for eval."""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weight")
        outputs = model(**inputs)
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
    return {"auprc": float(average_precision_score(labels, probs)),
            "auroc": float(roc_auc_score(labels, probs)),
            "f1": float(f1_score(labels, preds, zero_division=0))}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="T19 Nucleotide Transformer v2 250M LoRA fine-tune")
    ap.add_argument("--data-dir", default="results/T16/nt", help="dir with train/val/test.parquet")
    ap.add_argument("--output-dir", default="results/T19")
    ap.add_argument("--model-id", default=NT_ID)
    ap.add_argument("--sweep-trial", type=int, default=-1, help="set by T21; -1 = standalone run")
    # --- hyperparameters T21 sweeps ---
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-epochs", type=int, default=10)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--no-weight", action="store_true", help="ABLATION: ignore the T15 sample weight")
    ap.add_argument("--no-grad-checkpoint", action="store_true", help="disable gradient checkpointing (uses more VRAM)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train = TokenDataset(f"{args.data_dir}/train.parquet")
    val = TokenDataset(f"{args.data_dir}/val.parquet")
    test = TokenDataset(f"{args.data_dir}/test.parquet")

    # NT's remote code is v4-era: it reads config attributes (is_decoder, add_cross_attention, ...)
    # that transformers v5 dropped as base-config defaults. Load the config first and backfill the
    # missing encoder/decoder defaults (all False/0 -> NT is encoder-only) before building the model.
    config = AutoConfig.from_pretrained(args.model_id, num_labels=2, trust_remote_code=True)
    for _attr, _default in (("is_decoder", False), ("is_encoder_decoder", False),
                            ("add_cross_attention", False), ("tie_encoder_decoder", False),
                            ("chunk_size_feed_forward", 0), ("use_cache", False),
                            ("output_attentions", False), ("output_hidden_states", False)):
        if not hasattr(config, _attr):
            setattr(config, _attr, _default)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, config=config, trust_remote_code=True)
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["query", "value"],     # ESM/NT attention projections
        modules_to_save=["classifier"],        # keep the SequenceClassification head trainable
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    if not args.no_grad_checkpoint:
        model.enable_input_require_grads()      # peft + grad-checkpointing: inputs must require grad (Trainer enables the checkpointing itself via the arg below)

    targs = TrainingArguments(
        output_dir=f"{args.output_dir}/hf",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32,
        num_train_epochs=args.num_epochs,
        **_warmup_kwargs(steps=args.warmup_steps),   # v5-safe: warmup arg name feature-detected
        weight_decay=args.weight_decay,
        eval_strategy="epoch",                  # NB transformers>=4.56: eval_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_auprc",
        greater_is_better=True,
        save_total_limit=1,
        remove_unused_columns=False,   # CRITICAL: keep the per-sample `weight` column (same as T18) —
                                       # the Trainer strips non-model-input columns by default, which
                                       # would drop `weight` before compute_loss (KeyError).
        seed=args.seed,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),      # H800 -> bf16 (safer than fp16)
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=(not args.no_grad_checkpoint),
        gradient_checkpointing_kwargs={"use_reentrant": False},
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
        "model": "nt_lora", "trial": args.sweep_trial, "weighted": (not args.no_weight),
        "val_auprc": val_m["eval_auprc"], "val_auroc": val_m["eval_auroc"], "val_f1": val_m["eval_f1"],
        "test_auprc": test_m["eval_auprc"], "test_auroc": test_m["eval_auroc"], "test_f1": test_m["eval_f1"],
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    model.save_pretrained(os.path.join(args.output_dir, "best_adapter"))   # LoRA adapter only
    print(f"Validation AUPRC: {metrics['val_auprc']:.4f}")
    print(f"Test AUPRC: {metrics['test_auprc']:.4f}")


if __name__ == "__main__":
    main()
