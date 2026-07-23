#!/usr/bin/env python3
"""
genome_inference.py — slide a FINE-TUNED model across a genome and write a per-window score track
for genome_fp_benchmark.py. The fine-tuned counterpart of PT5 (off-the-shelf), and the seed of T28.

Scans PER CONTIG (so multi-replicon Clostridium genomes work) and per strand, streaming batches to
the CSV so a whole genome fits in memory. Output columns (consumed by genome_fp_benchmark):
    contig, start, end, strand, promoter_score      (0-based genome coords; reverse-strand windows
    are mapped back to genome coordinates so they align with the TSS truth).

Loads an HF sequence-classifier. ProkBERT (T18) is a FULL model; NT (T19) is a LoRA ADAPTER, so this
detects `adapter_config.json` and loads base(--model-id) + PeftModel (mirrors T22/T24). Tokenizer is
always loaded from --model-id (the base) because a fine-tuned dir may lack a usable tokenizer.

  # ProkBERT (T18), main env — the final model is saved at results/T18/best_model (NOT hf/):
  python genome_inference.py --genome GCF_000008765.1.fna \
      --model-id neuralbioinfo/prokbert-mini --model-path results/T18/best_model \
      --out cac_t18_track.csv --contigs NC_003030.1 --step 1 --batch-size 512

  # NT (T19), nt-tf4x env — saved as a LoRA adapter at results/T19/best_adapter:
  python genome_inference.py --genome <fna> \
      --model-id InstaDeepAI/nucleotide-transformer-v2-250m-multi-species \
      --model-path results/T19/best_adapter --out <track>.csv --contigs <id> ...

The from-scratch CNN (T20) needs a one-hot encoder rather than a tokenizer and is not handled here.
HPC4: export HF_HOME=/project/igem2026/hf_cache ; HF_HUB_OFFLINE=1 ; request a GPU.
"""
import argparse
import os
import sys
import pandas as pd
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_model(model_id, model_path, device):
    """Full model (ProkBERT) or base+adapter (NT LoRA), auto-detected."""
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        from peft import PeftModel
        base = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2, trust_remote_code=True)
        model = PeftModel.from_pretrained(base, model_path)
        print(f"[info] loaded LoRA adapter {model_path} on base {model_id}", file=sys.stderr)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
        print(f"[info] loaded full model {model_path}", file=sys.stderr)
    return model.to(device).eval()


def score_batch(wins, tok, model, device, max_length):
    enc = tok([w.upper() for w in wins], padding="max_length", truncation=True,
              max_length=max_length, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    return torch.softmax(logits, dim=1)[:, 1].float().cpu().numpy()


def scan_contig(rec_id, seq, strand, window, step, tok, model, device, max_length, batch_size,
                out_handle, write_header):
    """Stream windows of one contig/strand to the open out_handle; returns (write_header_next, n)."""
    L = len(seq)
    scan = seq if strand == "+" else str(Seq(seq).reverse_complement())
    starts = range(0, L - window + 1, step)          # range slices lazily; no list() materialisation
    n = 0
    for i in range(0, len(starts), batch_size):
        bs = starts[i:i + batch_size]
        scores = score_batch([scan[s:s + window] for s in bs], tok, model, device, max_length)
        recs = []
        for s, sc in zip(bs, scores):
            if strand == "+":
                g0, g1 = s, s + window
            else:                                    # reverse window -> genome coords (PT5 convention)
                g0, g1 = L - (s + window), L - s
            recs.append((rec_id, g0, g1, strand, float(sc)))
        pd.DataFrame(recs, columns=["contig", "start", "end", "strand", "promoter_score"]).to_csv(
            out_handle, header=write_header, index=False)
        write_header = False
        n += len(recs)
    return write_header, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", required=True, help="genome FASTA (one or more contigs)")
    ap.add_argument("--model-id", required=True, help="BASE model id (for the tokenizer / LoRA base)")
    ap.add_argument("--model-path", required=True, help="fine-tuned checkpoint dir (full model or LoRA adapter)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=81)
    ap.add_argument("--step", type=int, default=1, help="1 = every position (heavy); raise to subsample")
    ap.add_argument("--max-length", type=int, default=81)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--contigs", default=None, help="comma list to restrict to (default: all)")
    ap.add_argument("--strands", default="+,-")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn] no GPU detected — a genome-scale scan on CPU is very slow.", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(a.model_id, trust_remote_code=True)
    model = load_model(a.model_id, a.model_path, device)

    keep = set(a.contigs.split(",")) if a.contigs else None
    strands = [s for s in a.strands.split(",") if s]
    write_header, total = True, 0
    with open(a.out, "w") as out_handle:
        for rec in SeqIO.parse(a.genome, "fasta"):
            if keep and rec.id not in keep:
                continue
            seq = str(rec.seq)
            for strand in strands:
                write_header, n = scan_contig(rec.id, seq, strand, a.window, a.step, tok, model, device,
                                              a.max_length, a.batch_size, out_handle, write_header)
                total += n
                print(f"  {rec.id} {strand}: {n:,} windows", file=sys.stderr)
    print(f"Wrote {total:,} scored windows -> {a.out}")


if __name__ == "__main__":
    main()
