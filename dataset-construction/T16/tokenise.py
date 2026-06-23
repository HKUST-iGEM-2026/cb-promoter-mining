#!/usr/bin/env python3
"""
T16 — per-model tokenisation of the T15 splits.   [FIRST DRAFT for Leo to review]

Reads results/T15/{train,val,test}.parquet and writes, per model, the tokenised tensors each model
trains on — carrying label / weight / cluster_id through UNCHANGED. T16 is a pure format conversion:
it must NOT re-split, re-window, re-balance, or reorder rows.

  results/T16/prokbert/{split}.parquet   input_ids, attention_mask + metadata   (ProkBERT-mini-long, T18)
  results/T16/nt/{split}.parquet         input_ids, attention_mask + metadata   (NT-v2-250m, T19)
  results/T16/cnn/{split}.npz            X (N,81,4) one-hot + label/weight/cluster_id   (CNN, T20)
  results/T16/cnn/{split}_meta.parquet   full metadata for the CNN rows (row-aligned with the .npz)

Snakefile calls:  python tokenise.py results/T15/ results/T16/

>>> Two things to settle with the training owners before a real run:
    1. CONFIRM the checkpoint IDs (the T18 template hard-codes the WRONG org — see PROKBERT_ID).
       Tokens MUST be produced by the exact tokenizer each model fine-tunes, or training learns garbage.
    2. Agree the output format (this writes parquet for the LM models, .npz for the CNN). <<<

HPC4: compute nodes are offline -> pre-stage the tokenizers on the login node first
  export HF_HOME=/project/igem2026/hf_cache
  huggingface-cli download neuralbioinfo/prokbert-mini-long
  huggingface-cli download InstaDeepAI/nucleotide-transformer-v2-250m-multi-species
then run with HF_HOME set + HF_HUB_OFFLINE=1. Tokenisation is CPU-only.
"""
import os
import argparse
import numpy as np
import pandas as pd

# Metadata carried through to every output (training needs label+weight; CV needs cluster_id).
CARRY = ["label", "weight", "cluster_id", "split", "in_genus", "source", "promoter_id"]

# DECISION: confirm with the T18/T19 owners. The T18 template wrongly uses `biomedia-ai/...`.
PROKBERT_ID = "neuralbioinfo/prokbert-mini-long"
NT_ID = "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species"

BASE = {"A": 0, "C": 1, "G": 2, "T": 3}


# --------------------------------------------------------------------------- HuggingFace tokenisers
def hf_tokenise(seqs, model_id, max_length=None):
    """Tokenise with the model's OWN tokenizer so tokens match its pre-training.
    Returns (input_ids, attention_mask, max_length_used, vocab_size)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)  # trust_remote_code: ProkBERT ships a custom tokenizer
    seqs = list(seqs)
    if max_length is None:                                   # fit the pad length to the data once
        max_length = max(len(x) for x in tok(seqs)["input_ids"])
    enc = tok(seqs, padding="max_length", truncation=True, max_length=max_length)
    return enc["input_ids"], enc["attention_mask"], max_length, tok.vocab_size


# --------------------------------------------------------------------------- CNN one-hot
def one_hot(seq, L):
    a = np.zeros((L, 4), dtype=np.float32)
    for i, b in enumerate(seq[:L]):
        j = BASE.get(b)            # DECISION: N (rare — T9 is held out) -> all-zeros row; switch to 0.25 each if you prefer
        if j is not None:
            a[i, j] = 1.0
    return a


def write_hf(df, ids, mask, path, vocab_size):
    if ids:
        assert max(max(x) for x in ids) < vocab_size, "token id >= vocab_size — wrong tokenizer/vocab"
    out = df[CARRY].reset_index(drop=True).copy()
    out["input_ids"] = list(ids)
    out["attention_mask"] = list(mask)
    out.to_parquet(path, index=False)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="T16 per-model tokenisation of the T15 splits")
    ap.add_argument("t15_dir", help="dir with train/val/test.parquet from T15")
    ap.add_argument("out_dir", help="output dir (per-model subdirs are created)")
    ap.add_argument("--prokbert-id", default=PROKBERT_ID)
    ap.add_argument("--nt-id", default=NT_ID)
    ap.add_argument("--window", type=int, default=81)
    ap.add_argument("--models", default="prokbert,nt,cnn", help="comma list of: prokbert,nt,cnn")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    splits = {s: pd.read_parquet(os.path.join(args.t15_dir, f"{s}.parquet"))
              for s in ("train", "val", "test")}
    for m in models:
        os.makedirs(os.path.join(args.out_dir, m), exist_ok=True)

    pb_len = nt_len = None        # fit padding length on train, reuse for val/test (consistent shapes)
    for name in ("train", "val", "test"):
        df = splits[name].reset_index(drop=True)
        seqs = df["sequence"].astype(str).str.upper().tolist()
        n0 = len(df)

        if "prokbert" in models:
            ids, mask, pb_len, vsz = hf_tokenise(seqs, args.prokbert_id, pb_len)
            write_hf(df, ids, mask, f"{args.out_dir}/prokbert/{name}.parquet", vsz)
        if "nt" in models:
            ids, mask, nt_len, vsz = hf_tokenise(seqs, args.nt_id, nt_len)
            write_hf(df, ids, mask, f"{args.out_dir}/nt/{name}.parquet", vsz)
        if "cnn" in models:
            X = (np.stack([one_hot(s, args.window) for s in seqs]) if n0
                 else np.empty((0, args.window, 4), np.float32))
            assert X.shape == (n0, args.window, 4)
            np.savez_compressed(f"{args.out_dir}/cnn/{name}.npz", X=X,
                                label=df["label"].to_numpy(),
                                weight=df["weight"].to_numpy(),
                                cluster_id=df["cluster_id"].to_numpy())
            df[CARRY].to_parquet(f"{args.out_dir}/cnn/{name}_meta.parquet", index=False)

        # row-count guard: tokenisation must never add/drop/reorder rows
        for m in models:
            f = f"{args.out_dir}/cnn/{name}.npz" if m == "cnn" else f"{args.out_dir}/{m}/{name}.parquet"
            got = np.load(f)["X"].shape[0] if m == "cnn" else len(pd.read_parquet(f))
            assert got == n0, f"{m}/{name}: {got} rows != {n0} from T15"
        print(f"{name}: {n0} rows -> {', '.join(models)}"
              + (f"  (prokbert_len={pb_len}" if 'prokbert' in models else "")
              + (f" nt_len={nt_len}" if 'nt' in models else "")
              + (")" if {'prokbert', 'nt'} & set(models) else ""))

    print("T16 done — outputs under", args.out_dir)


if __name__ == "__main__":
    main()
