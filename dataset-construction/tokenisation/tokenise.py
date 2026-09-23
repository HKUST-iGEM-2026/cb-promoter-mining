#!/usr/bin/env python3
"""
T16 — per-model tokenisation of the T15 splits.

Reads results/T15/{train,val,test}.parquet and writes, per model, the tokenised tensors each model
trains on — carrying EVERY T15 metadata column (everything except `sequence`) through UNCHANGED.
T16 is a pure format conversion: it must NOT re-split, re-window, re-balance, or reorder rows.

  results/T16/prokbert/{split}.parquet   input_ids, attention_mask + metadata   (ProkBERT-mini, T18)
  results/T16/nt/{split}.parquet         input_ids, attention_mask + metadata   (NT v2 250M, T19)
  results/T16/cnn/{split}.npz            X (N,81,4) one-hot + label/weight/cluster_id   (CNN, T20)
  results/T16/cnn/{split}_meta.parquet   full metadata for the CNN rows (row-aligned with the .npz)

Snakefile calls:  python tokenise.py results/T15/ results/T16/

Model bases are LOCKED (no per-run decision, no one to ask):
  ProkBERT = neuralbioinfo/prokbert-mini
             custom LCATokenizer -> needs trust_remote_code=True; 6-mer, shift 1 (overlapping);
             an 81 bp window -> 78 tokens ([CLS] + 76 6-mers + [SEP]); vocab 4101.
  NT       = InstaDeepAI/nucleotide-transformer-v2-250m-multi-species
             built-in EsmTokenizer; 6-mer, non-overlapping; 81 bp -> ~18 tokens; vocab 4107.
The two token streams are DIFFERENT lengths on purpose (78 vs ~18) — they are not interchangeable,
which is why each model's pad width is fitted independently below. Both tokenizers require UPPERCASE
input (lowercase silently becomes all-UNK) — handled here via .str.upper().

HPC4: compute nodes are offline -> pre-stage the tokenizers on the login node first
  export HF_HOME=/project/igem2026/hf_cache
  huggingface-cli download neuralbioinfo/prokbert-mini
  huggingface-cli download InstaDeepAI/nucleotide-transformer-v2-250m-multi-species
then run with HF_HOME set + HF_HUB_OFFLINE=1. Tokenisation is CPU-only.
"""
import os
import argparse
import numpy as np
import pandas as pd

PROKBERT_ID = "neuralbioinfo/prokbert-mini"
NT_ID = "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species"

BASE = {"A": 0, "C": 1, "G": 2, "T": 3}
SPLITS = ("train", "val", "test")


def meta_cols(df):
    """Carry every T15 column EXCEPT the raw `sequence` (label, weight, cluster_id, split, in_genus,
    strand, source, species, sigma_factor, promoter_id, …). Carrying all-but-sequence guarantees
    nothing T15 produced is silently dropped — e.g. sigma_factor, which T25 needs downstream."""
    return [c for c in df.columns if c != "sequence"]


# --------------------------------------------------------------------------- HuggingFace tokenisers
def load_tokenizer(model_id):
    from transformers import AutoTokenizer
    # trust_remote_code: ProkBERT ships a custom LCATokenizer; harmless for NT's built-in EsmTokenizer.
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def token_len(tok, seqs):
    """True (un-padded) token length of the longest sequence — used to fit ONE pad width over all splits."""
    return max((len(x) for x in tok(list(seqs))["input_ids"]), default=0)


def hf_encode(tok, seqs, max_length):
    """Encode to a FIXED width. Raises rather than silently truncating if anything is longer than
    max_length. (max_length is fitted over train+val+test up front, so this never fires in practice —
    it's the guard against the old train-only-fit-then-truncate-val/test bug.)"""
    seqs = list(seqs)
    true_max = token_len(tok, seqs)
    if true_max > max_length:
        raise ValueError(f"a sequence tokenises to {true_max} tokens > max_length {max_length}; "
                         f"would truncate — refit max_length over all splits.")
    enc = tok(seqs, padding="max_length", truncation=True, max_length=max_length)
    return enc["input_ids"], enc["attention_mask"]


# --------------------------------------------------------------------------- CNN one-hot
def one_hot(seq, L):
    a = np.zeros((L, 4), dtype=np.float32)
    for i, b in enumerate(seq[:L]):
        j = BASE.get(b)            # N (rare — T9 held out) -> all-zeros row (LOCKED). Indistinguishable from pad to a CNN.
        if j is not None:
            a[i, j] = 1.0
    return a


def write_hf(df, ids, mask, path, vocab_size):
    if ids:
        assert max(max(x) for x in ids) < vocab_size, "token id >= vocab_size — wrong tokenizer/vocab"
    out = df[meta_cols(df)].reset_index(drop=True).copy()
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

    splits = {s: pd.read_parquet(os.path.join(args.t15_dir, f"{s}.parquet")).reset_index(drop=True)
              for s in SPLITS}
    seqs_upper = {s: splits[s]["sequence"].astype(str).str.upper().tolist() for s in SPLITS}
    for m in models:
        os.makedirs(os.path.join(args.out_dir, m), exist_ok=True)

    # Load each HF tokenizer ONCE and fit a single pad width over ALL splits (train+val+test).
    # Fitting on train alone then truncating val/test would silently drop tokens (the B1 bug).
    hf = {}
    for m, mid in (("prokbert", args.prokbert_id), ("nt", args.nt_id)):
        if m in models:
            tok = load_tokenizer(mid)
            union = [s for name in SPLITS for s in seqs_upper[name]]
            hf[m] = (tok, token_len(tok, union))

    for name in SPLITS:
        df = splits[name]
        seqs = seqs_upper[name]
        n0 = len(df)

        if "prokbert" in models:
            tok, L = hf["prokbert"]
            ids, mask = hf_encode(tok, seqs, L)
            write_hf(df, ids, mask, f"{args.out_dir}/prokbert/{name}.parquet", tok.vocab_size)
        if "nt" in models:
            tok, L = hf["nt"]
            ids, mask = hf_encode(tok, seqs, L)
            write_hf(df, ids, mask, f"{args.out_dir}/nt/{name}.parquet", tok.vocab_size)
        if "cnn" in models:
            X = (np.stack([one_hot(s, args.window) for s in seqs]) if n0
                 else np.empty((0, args.window, 4), np.float32))
            assert X.shape == (n0, args.window, 4)
            np.savez_compressed(f"{args.out_dir}/cnn/{name}.npz", X=X,
                                label=df["label"].to_numpy(),
                                weight=df["weight"].to_numpy(),
                                cluster_id=np.array(df["cluster_id"].astype(str).tolist(), dtype="U"))  # fixed-width unicode -> loads w/o allow_pickle
            df[meta_cols(df)].to_parquet(f"{args.out_dir}/cnn/{name}_meta.parquet", index=False)

        # row-count guard: tokenisation must never add/drop/reorder rows
        for m in models:
            f = f"{args.out_dir}/cnn/{name}.npz" if m == "cnn" else f"{args.out_dir}/{m}/{name}.parquet"
            got = np.load(f)["X"].shape[0] if m == "cnn" else len(pd.read_parquet(f))
            assert got == n0, f"{m}/{name}: {got} rows != {n0} from T15"
        print(f"{name}: {n0} rows -> {', '.join(models)}"
              + (f"  (prokbert_len={hf['prokbert'][1]}" if 'prokbert' in models else "")
              + (f" nt_len={hf['nt'][1]}" if 'nt' in models else "")
              + (")" if {'prokbert', 'nt'} & set(models) else ""))

    print("T16 done — outputs under", args.out_dir)


if __name__ == "__main__":
    main()
