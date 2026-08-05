#!/usr/bin/env python3
"""
score_promoters.py — load one trained model and score sequences for P(promoter).

Single hand-off entry point. Run it in the RIGHT conda env (NT is the exception):

  source /project/igem2026/cb-promoter-mining/env.sh

  # ProkBERT / CNN / CNN-v2  -> promoter-mining-hpc4 env:
  conda activate /project/igem2026/envs/promoter-mining-hpc4
  python score_promoters.py --model prokbert --fasta seqs.fa --out scores.csv
  python score_promoters.py --model cnn      --fasta seqs.fa --out scores.csv
  python score_promoters.py --model cnn_v2   --fasta seqs.fa --out scores.csv

  # NT-LoRA  -> nt-tf4x env ONLY (transformers 4.57):
  conda activate /project/igem2026/envs/nt-tf4x
  python score_promoters.py --model nt --fasta seqs.fa --out scores.csv

Input : a FASTA of ~81 bp sequences. Output: CSV with columns id,prob (prob = P(promoter),
uncalibrated softmax[:,1]). CNN-v2 auto-fits length to 81 bp; the others truncate/pad to 81.
Defaults point at the deployed 0.6 models (T21_final); use --models-dir to point elsewhere
(e.g. .../results/T21_final_06 for the re-swept set, or .../results/T18 for the old 0.9 ProkBERT).
"""
import argparse
import csv
import importlib.util
import json
import os

import numpy as np

# Repo root = this file's directory, so it works from any clone.
REPO = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODELS_DIR = "/project/igem2026/cb-promoter-mining/infrastructure/T3/snakemake_scaffold/results/T21_final"
NT_BASE = "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species"


def read_fasta(path):
    ids, seqs, cur, buf = [], [], None, []
    with open(path) as fh:
        for ln in fh:
            ln = ln.rstrip()
            if ln.startswith(">"):
                if cur is not None:
                    ids.append(cur); seqs.append("".join(buf))
                cur, buf = ln[1:].split()[0] if ln[1:].strip() else ln[1:], []
            elif ln:
                buf.append(ln)
        if cur is not None:
            ids.append(cur); seqs.append("".join(buf))
    return ids, [s.upper() for s in seqs]


def _load_module(pyfile, name):
    spec = importlib.util.spec_from_file_location(name, pyfile)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _score_transformer(model, tok, seqs, device, bs=64):
    import torch
    out = []
    for i in range(0, len(seqs), bs):
        enc = tok(seqs[i:i + bs], padding="max_length", truncation=True,
                  max_length=81, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out.append(torch.softmax(model(**enc).logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["prokbert", "nt", "cnn", "cnn_v2"])
    ap.add_argument("--fasta", required=True, help="FASTA of ~81 bp sequences")
    ap.add_argument("--out", required=True, help="output CSV (id,prob)")
    ap.add_argument("--models-dir", default=DEFAULT_MODELS_DIR,
                    help="dir holding prokbert/nt_lora/cnn/cnn_v2 checkpoints (default: deployed 0.6 models)")
    a = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids, seqs = read_fasta(a.fasta)
    print(f"[{a.model}] loaded {len(seqs)} sequences; device={device}")

    if a.model == "prokbert":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("neuralbioinfo/prokbert-mini", trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            f"{a.models_dir}/prokbert/best_model", trust_remote_code=True).to(device).eval()
        probs = _score_transformer(model, tok, seqs, device)

    elif a.model == "nt":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained(NT_BASE, trust_remote_code=True)
        base = AutoModelForSequenceClassification.from_pretrained(NT_BASE, num_labels=2, trust_remote_code=True)
        model = PeftModel.from_pretrained(base, f"{a.models_dir}/nt_lora/best_adapter").to(device).eval()
        probs = _score_transformer(model, tok, seqs, device)

    elif a.model == "cnn":
        tcnn = _load_module(f"{REPO}/model-training/T20/train_cnn.py", "train_cnn")
        cfg = json.load(open(f"{a.models_dir}/cnn/metrics.json")).get("config", {})
        keys = ("conv1_filters", "conv2_filters", "conv3_filters", "kernel_size", "dropout_rate", "use_batch_norm")
        model = tcnn.PromoterCNN(input_channels=4, sequence_length=81,
                                 **{k: cfg[k] for k in keys if k in cfg}).to(device)
        model.load_state_dict(torch.load(f"{a.models_dir}/cnn/best_model.pt", map_location=device))
        model.eval()
        base_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
        X = np.zeros((len(seqs), 4, 81), np.float32)
        for i, s in enumerate(seqs):
            for j, b in enumerate(s[:81]):
                if b in base_idx:
                    X[i, base_idx[b], j] = 1.0
        with torch.no_grad():
            probs = torch.softmax(model(torch.from_numpy(X).to(device)).float(), dim=1)[:, 1].cpu().numpy()

    else:  # cnn_v2
        v2 = _load_module(f"{REPO}/model-training/T20/train_cnn_v2.py", "train_cnn_v2")
        cfg = json.load(open(f"{a.models_dir}/cnn_v2/config.json"))
        model = v2.build_model_from_config(cfg, device=device)
        model.load_state_dict(torch.load(f"{a.models_dir}/cnn_v2/best_model.pt", map_location=device))
        probs = v2.predict_proba(model, seqs, device, enac_window=int(cfg.get("enac_window", 5)))

    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "prob"])
        for i, p in zip(ids, probs):
            w.writerow([i, f"{float(p):.6f}"])
    print(f"[{a.model}] wrote {len(ids)} scores to {a.out}")


if __name__ == "__main__":
    main()
