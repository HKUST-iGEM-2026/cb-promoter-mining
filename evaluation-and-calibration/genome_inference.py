#!/usr/bin/env python3
"""
genome_inference.py — slide a FINE-TUNED model across a genome and write a per-window score track
for genome_fp_benchmark.py. Fine-tuned counterpart of PT5, and the seed of T28.

Scans PER CONTIG (multi-replicon safe) and per strand, streaming batches to CSV. Output columns:
    contig, start, end, strand, promoter_score   (0-based genome coords; reverse windows mapped back).

Handles all three ensemble members:
  * ProkBERT (T18)  — full HF model.        --model-path results/T18/best_model  --model-id neuralbioinfo/prokbert-mini
  * NT (T19)        — LoRA adapter (auto).   --model-path results/T19/best_adapter --model-id InstaDeepAI/nucleotide-transformer-v2-250m-multi-species  (nt-tf4x env)
  * CNN (T20)       — from-scratch, one-hot. --model-path results/T20/best_model.pt  --model-type cnn
Model type is auto-detected (a `.pt` path => cnn; an `adapter_config.json` => LoRA; else full HF).

HPC4: export HF_HOME=/project/igem2026/hf_cache ; HF_HUB_OFFLINE=1 ; request a GPU.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from Bio.Seq import Seq

BASE = {"A": 0, "C": 1, "G": 2, "T": 3}


# --------------------------------------------------------------------------- transformer path
def load_transformer(model_id, model_path, device):
    from transformers import AutoModelForSequenceClassification
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        from peft import PeftModel
        base = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2, trust_remote_code=True)
        model = PeftModel.from_pretrained(base, model_path)
        print(f"[info] LoRA adapter {model_path} on base {model_id}", file=sys.stderr)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
        print(f"[info] full model {model_path}", file=sys.stderr)
    return model.to(device).eval()


def transformer_scorer(tok, model, device, max_length):
    def score(wins):
        enc = tok([w.upper() for w in wins], padding="max_length", truncation=True,
                  max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        return torch.softmax(logits, dim=1)[:, 1].float().cpu().numpy()
    return score


# --------------------------------------------------------------------------- CNN path (one-hot)
def load_cnn(model_path, device):
    import importlib.util
    repo = Path(__file__).resolve().parents[1]
    t20 = repo / "model-training" / "T20" / "train_cnn.py"
    spec = importlib.util.spec_from_file_location("train_cnn", str(t20))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = {}
    metrics = Path(model_path).parent / "metrics.json"
    if metrics.exists():
        cfg = json.loads(metrics.read_text()).get("config", {})
    keys = ("conv1_filters", "conv2_filters", "conv3_filters", "kernel_size", "dropout_rate", "use_batch_norm")
    kwargs = {k: cfg[k] for k in keys if k in cfg}
    model = mod.PromoterCNN(input_channels=4, sequence_length=81, **kwargs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[info] CNN {model_path}", file=sys.stderr)
    return model


def cnn_scorer(model, device, window):
    def score(wins):
        X = np.zeros((len(wins), 4, window), dtype=np.float32)
        for i, s in enumerate(wins):
            for j, b in enumerate(s.upper()[:window]):
                k = BASE.get(b)
                if k is not None:
                    X[i, k, j] = 1.0
        with torch.no_grad():
            logits = model(torch.from_numpy(X).to(device))
        return torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
    return score


# --------------------------------------------------------------------------- CNN-v2 (multi-encoding)
def load_cnn_v2(model_path, device):
    """Load the multi-branch v2 CNN (train_cnn_v2.MultiEncodingPromoterCNN). Returns
    (module, model, enac_window) — the module carries predict_proba, enac_window is
    the feature window the model was trained with (read from config.json)."""
    import importlib.util
    repo = Path(__file__).resolve().parents[1]
    t20v2 = repo / "model-training" / "T20" / "train_cnn_v2.py"
    spec = importlib.util.spec_from_file_location("train_cnn_v2", str(t20v2))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = {}
    cfg_path = Path(model_path).parent / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
    model = mod.build_model_from_config(cfg, device=str(device))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[info] CNN-v2 {model_path}  encodings={cfg.get('encodings', cfg.get('active_encodings'))}",
          file=sys.stderr)
    return mod, model, int(cfg.get("enac_window", 5))


def cnn_v2_scorer(mod, model, device, enac_window):
    def score(wins):
        return mod.predict_proba(model, list(wins), str(device),
                                 enac_window=enac_window, batch_size=len(wins))
    return score


# --------------------------------------------------------------------------- scan
def scan_contig(rec_id, seq, strand, window, step, scorer, batch_size, out_handle, write_header):
    L = len(seq)
    scan = seq if strand == "+" else str(Seq(seq).reverse_complement())
    starts = range(0, L - window + 1, step)
    n = 0
    for i in range(0, len(starts), batch_size):
        bs = starts[i:i + batch_size]
        scores = scorer([scan[s:s + window] for s in bs])
        recs = []
        for s, sc in zip(bs, scores):
            g0, g1 = (s, s + window) if strand == "+" else (L - (s + window), L - s)
            recs.append((rec_id, g0, g1, strand, float(sc)))
        pd.DataFrame(recs, columns=["contig", "start", "end", "strand", "promoter_score"]).to_csv(
            out_handle, header=write_header, index=False)
        write_header = False
        n += len(recs)
    return write_header, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", required=True)
    ap.add_argument("--model-id", default=None, help="BASE id for the tokenizer / LoRA base (transformers)")
    ap.add_argument("--model-path", required=True, help="checkpoint dir (HF) or .pt (CNN)")
    ap.add_argument("--model-type", default="auto", choices=["auto", "transformer", "cnn", "cnn_v2"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=81)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=81)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--contigs", default=None)
    ap.add_argument("--strands", default="+,-")
    a = ap.parse_args()

    mtype = a.model_type
    if mtype == "auto":
        # both CNNs are .pt; auto picks the prod CNN. For the multi-branch model
        # pass --model-type cnn_v2 explicitly (it can't be told apart by extension).
        mtype = "cnn" if str(a.model_path).endswith(".pt") else "transformer"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn] no GPU detected — a genome-scale scan on CPU is very slow.", file=sys.stderr)

    if mtype == "cnn":
        scorer = cnn_scorer(load_cnn(a.model_path, device), device, a.window)
    elif mtype == "cnn_v2":
        mod, model, enac_w = load_cnn_v2(a.model_path, device)
        scorer = cnn_v2_scorer(mod, model, device, enac_w)
    else:
        if not a.model_id:
            raise SystemExit("--model-id is required for transformer models (tokenizer / LoRA base).")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model_id, trust_remote_code=True)
        scorer = transformer_scorer(tok, load_transformer(a.model_id, a.model_path, device), device, a.max_length)

    keep = set(a.contigs.split(",")) if a.contigs else None
    strands = [s for s in a.strands.split(",") if s]
    write_header, total = True, 0
    with open(a.out, "w") as out_handle:
        for rec in SeqIO.parse(a.genome, "fasta"):
            if keep and rec.id not in keep:
                continue
            seq = str(rec.seq)
            for strand in strands:
                write_header, n = scan_contig(rec.id, seq, strand, a.window, a.step, scorer,
                                              a.batch_size, out_handle, write_header)
                total += n
                print(f"  {rec.id} {strand}: {n:,} windows", file=sys.stderr)
    print(f"Wrote {total:,} scored windows -> {a.out}")


if __name__ == "__main__":
    main()
