#!/usr/bin/env python3
"""
NT genome-wide sliding-window scoring — the PT5/PT6-equivalent for Nucleotide Transformer.

A drop-in twin of prototypes/ecoli-sliding-window/sliding_window_inference.py (same 81 bp window, step 5, both strands,
same output columns), so the PT6 validation runs UNCHANGED and the numbers sit next to the
prokbert-mini-promoter PT5/PT6 results.

The only difference vs PT5: NT has no off-the-shelf promoter head, so the scorer is
    FROZEN NT mean-pooled embedding  ->  a tiny probe head (trained + saved by nt_linear_probe.py --save-head)
instead of a ready-made classifier.

  # 1) make the head (also prints the Tier-1 classification AUPRC):
  python nt_linear_probe.py --data-dir results/T16/nt --save-head nt_probe_head.joblib
  # 2) scan the genome (coarsen --step for a cheaper first pass — NT-250M is heavy):
  python nt_sliding_window.py --fasta MG1655.fna --head nt_probe_head.joblib --step 10
"""
from Bio import SeqIO
from Bio.Seq import Seq
import argparse
import joblib
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

NT_ID = "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species"


def main():
    ap = argparse.ArgumentParser(description="NT frozen-embedding + probe-head genome scan (PT5/PT6 twin)")
    ap.add_argument("--fasta", default="MG1655.fna")
    ap.add_argument("--head", default="nt_probe_head.joblib", help="from nt_linear_probe.py --save-head")
    ap.add_argument("--model-id", default=NT_ID)
    ap.add_argument("--window", type=int, default=81)        # matches PT5 / the model window
    ap.add_argument("--step", type=int, default=5)           # raise to 10-20 for a cheaper first pass
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default="mg1655_nt_probe_scores.csv")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    nt = AutoModel.from_pretrained(args.model_id, trust_remote_code=True).to(dev).eval()
    head = joblib.load(args.head)                            # sklearn pipeline (scaler + logreg)

    rec = next(SeqIO.parse(args.fasta, "fasta"))
    genome = str(rec.seq).upper()
    chrom = rec.id
    results = []

    @torch.no_grad()
    def score(seqs):
        enc = tok(seqs, return_tensors="pt", padding=True).to(dev)
        h = nt(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        emb = ((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy()   # mean-pool over real tokens
        return head.predict_proba(emb)[:, 1]

    def scan(seq, strand):
        batch, meta = [], []
        for s in range(0, len(seq) - args.window + 1, args.step):
            w = seq[s:s + args.window]
            os_, oe = ((s, s + args.window) if strand == "+"
                       else (len(genome) - (s + args.window), len(genome) - s))
            batch.append(w); meta.append((os_, oe))
            if len(batch) == args.batch_size:
                for (a, b), p in zip(meta, score(batch)):
                    results.append({"chromosome": chrom, "start": a, "end": b,
                                    "strand": strand, "promoter_score": float(p)})
                batch, meta = [], []
        if batch:
            for (a, b), p in zip(meta, score(batch)):
                results.append({"chromosome": chrom, "start": a, "end": b,
                                "strand": strand, "promoter_score": float(p)})

    print("Scanning forward strand"); scan(genome, "+")
    print("Scanning reverse strand"); scan(str(Seq(genome).reverse_complement()), "-")
    pd.DataFrame(results).sort_values("promoter_score", ascending=False).to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(results)} windows) — now run the PT6 validation against RegulonDB.")


if __name__ == "__main__":
    main()
