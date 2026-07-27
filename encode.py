#!/usr/bin/env python3
"""
Unified sequence encoder for CNN, NT v3, and ProkBERT models.
Supports NT v3 minimum input length (>= 120 bp) and variable-length genomic context.

Fixed bugs from prior version:
  1. IUPAC ambiguity counting was dead code (indented inside unreachable block)
  2. CNN crashed on variable-length sequences (torch.tensor on jagged lists)
  3. CNN produced no attention_mask
  4. NT padding=True when --max-tokens omitted → pos/neg shape drift
  5. No headers saved for traceability
"""

import argparse
import sys
import warnings
from pathlib import Path
from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence

VALID_DNA = set("ACGTN")
IUPAC_AMBIG = set("RYSWKMBDHV")


def load_fasta(fasta_path):
    """Load a FASTA file, returning (headers, sequences). All bases uppercased."""
    names, sequences = [], []
    current_name = None
    current_seq = []
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    names.append(current_name)
                    sequences.append("".join(current_seq).upper())
                current_name = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_name is not None:
            names.append(current_name)
            sequences.append("".join(current_seq).upper())
    return names, sequences


def check_sequences(sequences, min_length, label):
    """
    Validate sequences: minimum length, valid alphabet, IUPAC ambiguity codes.
    """
    n_ambig_total = 0
    for i, seq in enumerate(sequences):
        if len(seq) < min_length:
            print(
                f"ERROR: {label} sequence {i} has length {len(seq)} bp, "
                f"which is shorter than the required minimum of {min_length} bp. Aborting.",
                file=sys.stderr,
            )
            sys.exit(1)

        ambig = [b for b in seq if b in IUPAC_AMBIG]
        unexpected = [b for b in seq if b not in VALID_DNA and b not in IUPAC_AMBIG]
        if unexpected:
            print(
                f"ERROR: {label} sequence {i} contains unexpected characters: "
                f"{set(unexpected)}. Aborting.",
                file=sys.stderr,
            )
            sys.exit(1)

        # FIX 1: This was previously dead code — indented inside the
        # 'if unexpected:' block after sys.exit(1). Now correctly at
        # the outer loop level so ambiguity codes are always counted.
        n_ambig_total += len(ambig)

    if n_ambig_total > 0:
        warnings.warn(
            f"{label}: {n_ambig_total} IUPAC ambiguity codes found across "
            f"{len(sequences)} sequences. These will be treated as N. "
            f"A dinucleotide shuffler should not produce any — check upstream."
        )
    return True


def dinucleotide_freq(seq):
    return Counter(seq[i : i + 2] for i in range(len(seq) - 1))


def assert_composition_match(pos_seqs, neg_seqs):
    """Assert each negative has identical mono- and dinucleotide composition to paired positive."""
    if len(pos_seqs) != len(neg_seqs):
        print(
            f"ERROR: positives ({len(pos_seqs)}) and negatives ({len(neg_seqs)}) "
            f"have different counts. Cannot verify pairing.",
            file=sys.stderr,
        )
        sys.exit(1)

    mismatches = []
    for i, (p, n) in enumerate(zip(pos_seqs, neg_seqs)):
        if Counter(p) != Counter(n):
            mismatches.append((i, "mononucleotide"))
            continue
        if dinucleotide_freq(p) != dinucleotide_freq(n):
            mismatches.append((i, "dinucleotide"))

    if mismatches:
        for idx, kind in mismatches[:10]:
            print(f"  Pair {idx}: {kind} mismatch", file=sys.stderr)
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more", file=sys.stderr)
        print(
            f"ERROR: {len(mismatches)} pairs fail composition matching. Check the shuffler.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  Composition-match assertion passed for all {len(pos_seqs)} pairs.")


# ─────────────────────────────────────────────────────────────────────
# CNN encoder
# ─────────────────────────────────────────────────────────────────────

DNA_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
CNN_PAD_VALUE = 4  # pad with N-index; change if your CNN uses a different pad token


def encode_cnn(sequences):
    """
    Integer-encode variable-length sequences for CNN baseline.

    FIX 2: Uses pad_sequence instead of torch.tensor on jagged lists.
    FIX 3: Produces an attention_mask (1 = real base, 0 = padding).
    """
    tensors = []
    for seq in sequences:
        cleaned = "".join(b if b in DNA_TO_INT else "N" for b in seq)
        numeric = [DNA_TO_INT[b] for b in cleaned]
        tensors.append(torch.tensor(numeric, dtype=torch.long))

    # pad_sequence pads shorter sequences to the length of the longest
    padded_ids = pad_sequence(tensors, batch_first=True, padding_value=CNN_PAD_VALUE)

    # Build attention mask: 1 where real, 0 where padded
    attention_mask = torch.zeros_like(padded_ids)
    for i, t in enumerate(tensors):
        attention_mask[i, : len(t)] = 1

    return {"input_ids": padded_ids, "attention_mask": attention_mask}


# ─────────────────────────────────────────────────────────────────────
# NT (Nucleotide Transformer v3) encoder
# ─────────────────────────────────────────────────────────────────────


def encode_nt(sequences, model_repo, max_length):
    """
    Tokenize sequences for Nucleotide Transformer v3 (character-level, U-Net
    architecture). NTv3 requires sequence length to be a multiple of 128
    tokens — this is a structural constraint from the downsampling tower,
    not a padding preference.

    FIX 4: --max-tokens is required and MUST be a multiple of 128. Using the
    same explicit value for both positives and negatives (rather than
    padding=True's "pad to longest in this file") prevents pos/neg shape
    drift across separate encoding runs.
    """
    if max_length % 128 != 0:
        print(
            f"ERROR: --max-tokens ({max_length}) must be a multiple of 128 "
            f"for NTv3 (its U-Net downsampling tower requires this). "
            f"Round up, e.g. {((max_length // 128) + 1) * 128}.",
            file=sys.stderr,
        )
        sys.exit(1)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)

    # NTv3 uses single-base tokenization (vocab size 11: A/T/C/G/N + specials).
    # add_special_tokens=False matches the model card's usage example;
    # pad_to_multiple_of=128 is belt-and-suspenders alongside the explicit
    # max_length, since max_length is already validated as a multiple of 128.
    encoded = tokenizer(
        sequences,
        add_special_tokens=False,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        pad_to_multiple_of=128,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


# ─────────────────────────────────────────────────────────────────────
# ProkBERT encoder
# ─────────────────────────────────────────────────────────────────────


def encode_prokbert(sequences, model_repo, max_length, kmer, shift):
    """Tokenize sequences for ProkBERT using its native LCA k-mer tokenizer."""
    try:
        from prokbert.prokbert_tokenizer import ProkBERTTokenizer
    except ImportError:
        print(
            "ERROR: `prokbert` package not installed. Install with: pip install prokbert",
            file=sys.stderr,
        )
        sys.exit(1)

    tok_params = {"kmer": kmer, "shift": shift}
    tokenizer = ProkBERTTokenizer(tokenization_params=tok_params)

    all_input_ids = []
    all_attention_masks = []

    for seq in sequences:
        encoded = tokenizer(seq, return_tensors="pt")
        all_input_ids.append(encoded["input_ids"].squeeze(0))
        all_attention_masks.append(encoded["attention_mask"].squeeze(0))

    max_len = max_length if max_length else max(t.size(0) for t in all_input_ids)
    padded_ids = torch.zeros(len(all_input_ids), max_len, dtype=torch.long)
    padded_mask = torch.zeros(len(all_attention_masks), max_len, dtype=torch.long)

    for i, (ids, mask) in enumerate(zip(all_input_ids, all_attention_masks)):
        length = min(ids.size(0), max_len)
        padded_ids[i, :length] = ids[:length]
        padded_mask[i, :length] = mask[:length]

    return {"input_ids": padded_ids, "attention_mask": padded_mask}


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Encode FASTA sequences for CNN / NT v3 / ProkBERT models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Encode T24 negatives for CNN (variable-length, >=120 bp)
  python encode_fixed.py --model cnn --input neg.fasta --output neg_cnn.pt \\
      --positives pos.fasta --min-length 120

  # Encode for NT v3 (must specify --max-tokens, same value for pos and neg)
  python encode_fixed.py --model nt --input pos.fasta --output pos_nt.pt \\
      --label pos --max-tokens 256

  # Encode for ProkBERT (mini-c, kmer=1, shift=1)
  python encode_fixed.py --model prokbert --input neg.fasta --output neg_prokbert.pt \\
      --positives pos.fasta --prokbert-kmer 1 --prokbert-shift 1 --max-tokens 128
""",
    )

    parser.add_argument("--model", required=True, choices=["cnn", "nt", "prokbert"])
    parser.add_argument("--input", required=True, type=str, help="Path to input FASTA file.")
    parser.add_argument("--output", required=True, type=str, help="Path to save encoded output (.pt).")
    parser.add_argument("--label", choices=["pos", "neg"], default="neg")
    parser.add_argument(
        "--min-length", type=int, default=120,
        help="Minimum required sequence length in bp (default: 120, the "
        "practical floor below which NTv3 inference is not meaningful). "
        "Note: this is separate from --max-tokens, which must independently "
        "be a multiple of 128 for NT due to its U-Net architecture.",
    )
    parser.add_argument(
        "--positives", type=str, default=None,
        help="Path to paired positive FASTA for T24 composition check.",
    )
    parser.add_argument(
        "--expected-count", type=int, default=None,
        help="Expected number of sequences (e.g. 93 for T24). Aborts if mismatched.",
    )

    # NT-specific
    parser.add_argument(
        "--nt-repo", type=str,
        default="InstaDeepAI/NTv3_650M_pre",
        help="NTv3 checkpoint repo. Default is the largest published "
        "pre-trained checkpoint (650M). Do NOT use *_8kb variants — those "
        "are intermediate checkpoints, not the recommended models.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Max token length for transformer tokenizers. For NT, this MUST "
        "be a multiple of 128 (NTv3's U-Net architecture requirement) and "
        "MUST be the same value for both positives and negatives. For a "
        "681 bp max sequence, use 768.",
    )

    # ProkBERT-specific
    parser.add_argument("--prokbert-repo", type=str, default="neuralbioinfo/prokbert-mini")
    parser.add_argument("--prokbert-kmer", type=int, default=6)
    parser.add_argument("--prokbert-shift", type=int, default=1)

    args = parser.parse_args()

    # ── Load and validate ─────────────────────────────────────────
    headers, sequences = load_fasta(args.input)
    n = len(sequences)
    print(f"Loaded {n} {args.label} sequences from {args.input}")

    if args.expected_count is not None and n != args.expected_count:
        print(
            f"ERROR: expected {args.expected_count} sequences, got {n}.",
            file=sys.stderr,
        )
        sys.exit(1)

    check_sequences(sequences, args.min_length, args.label)

    # Print length distribution for awareness
    lengths = [len(s) for s in sequences]
    print(f"  Lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={sum(lengths)/len(lengths):.0f} bp")

    # ── Composition-match check (T24) ─────────────────────────────
    if args.positives is not None:
        print(f"Checking composition match against {args.positives} ...")
        _, pos_seqs = load_fasta(args.positives)
        check_sequences(pos_seqs, args.min_length, "positives")
        assert_composition_match(pos_seqs, sequences)

    # ── Encode ────────────────────────────────────────────────────
    if args.model == "cnn":
        result = encode_cnn(sequences)

    elif args.model == "nt":
        # FIX 4: Require --max-tokens for NT to prevent pos/neg shape drift
        if args.max_tokens is None:
            print(
                "ERROR: --max-tokens is required for NT. Use the same value "
                "for both positives and negatives to ensure matching tensor shapes.",
                file=sys.stderr,
            )
            sys.exit(1)
        result = encode_nt(sequences, args.nt_repo, args.max_tokens)

    elif args.model == "prokbert":
        result = encode_prokbert(
            sequences,
            args.prokbert_repo,
            args.max_tokens,
            args.prokbert_kmer,
            args.prokbert_shift,
        )

    # ── Save ──────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)

    print(f"Saved to {output_path}:")
    for key, tensor in result.items():
        print(f"  {key}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")

    # FIX 5: Save headers for traceability and pairing verification
    header_path = output_path.with_suffix(".headers.txt")
    with open(header_path, "w") as f:
        for h in headers:
            f.write(h + "\n")
    print(f"  Headers saved to {header_path}")


if __name__ == "__main__":
    main()
