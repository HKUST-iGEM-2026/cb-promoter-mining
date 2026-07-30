#!/usr/bin/env python3
"""
Unified sequence encoder for CNN, NTv3, and ProkBERT models.
Supports minimum input length (>= 128 bp) and variable-length genomic context.

Model backends:
  - CNN:      integer encoding (torch), variable length + attention mask
  - NTv3:     JAX package `nucleotide_transformer_v3` via batch_np_tokenize
              (NOT HuggingFace); saves tokens as .npz; tokenizer loaded via
              get_ntv3_tokenizer() (no weights needed)
  - ProkBERT: HuggingFace LCA k-mer tokenizer (torch)

Fixed bugs from prior versions:
  1. IUPAC ambiguity counting was dead code (indented inside unreachable block)
  2. CNN crashed on variable-length sequences (torch.tensor on jagged lists)
  3. CNN produced no attention_mask
  4. NT padding drift → resolved: raw sequences are N-padded to a fixed
     multiple of the downsample factor (128) before tokenizing
  5. No headers saved for traceability
  6. NT was wired to HuggingFace AutoTokenizer; NTv3 actually ships as JAX
  7. batch_np_tokenize does NOT auto-round to /128 — caller must N-pad the
     raw string first (per NTV3_ENV.md smoke test); now handled here
  8. NT path used get_pretrained_ntv3_model (needs HF weights); replaced with
     get_ntv3_tokenizer() which runs locally, no weights or HF needed
  9. NT had no attention_mask; now produces all-ones mask because N-padded
     positions are real bases (model was never trained on [PAD])
  10. --min-length defaulted to 120 (not a multiple of 128); fixed to 128
  11. NT can now read a T15 split parquet (--from-parquet) instead of loose
      FASTA, carrying label/weight/source/species and all other T15 metadata
      through row-aligned (matching T16), so training integration isn't broken
  12. CNN output was variable-length integer IDs (incompatible with the T20
      trainer, which hard-requires (N,81,4) one-hot). Now emits fixed-window
      one-hot .npz (X/label/weight) matching T16.one_hot and T20 load_split.
  13. Composition check was applied to ALL negatives and falsely failed on
      real genomic negatives (CDS/intergenic/downstream). Now source-aware:
      only SRS shuffle negatives are checked; others are skipped and counted.

Two input modes:
  --input <fasta>         FASTA mode (standalone; e.g. T24 composition shuffle)
  --from-parquet <pq>     T15-split mode: reads results/T15/{split}.parquet,
                          carries every metadata column through UNCHANGED and
                          row-aligned. --model nt writes a token parquet;
                          --model cnn writes a T20-compatible (N,81,4) .npz
                          (X/label/weight) plus a metadata sidecar parquet.
"""

import argparse
import sys
import warnings
from pathlib import Path
from collections import Counter

# NOTE: torch is intentionally NOT imported at module top. The NTv3 path runs
# in the `ntv3-jax` env, which has no torch installed (JAX and torch are kept
# separate on purpose — see NTV3_ENV.md). torch is imported lazily inside the
# CNN/ProkBERT paths only, so `--model nt` never requires it.

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


def assert_composition_match(pos_seqs, neg_seqs, neg_sources=None):
    """
    Verify that SRS (dinucleotide-shuffle) negatives preserve the mono- and
    dinucleotide composition of their paired positive.

    IMPORTANT — this check is ONLY meaningful for SRS negatives. SRS
    ("Shuffled Real Sequence") negatives are dinucleotide-preserving shuffles
    of a positive, so by construction they must match its mono/dinucleotide
    composition. The OTHER T14/T15 negative classes — CDS, intergenic,
    downstream — are REAL genomic sequence, not shuffles of anything; they
    have no paired positive and would (correctly, by their nature) fail a
    composition-match test. Running the check against them is a false alarm.

    So:
      - If `neg_sources` is given, only rows whose source indicates SRS are
        checked; all other negatives are skipped (and counted).
      - If `neg_sources` is None (e.g. the standalone T24 FASTA path, where
        every negative is an SRS shuffle by construction), ALL pairs are
        checked — that is the all-SRS eval case.

    A source counts as SRS if its (case-insensitive) value contains "srs".
    This matches both T24 ("SRS") and T15's regenerated null (source "SRS")
    and T14's class label ("T14_SRS").
    """
    if len(pos_seqs) != len(neg_seqs):
        print(
            f"ERROR: positives ({len(pos_seqs)}) and negatives ({len(neg_seqs)}) "
            f"have different counts. Cannot verify pairing.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Decide which indices are SRS (checkable) vs skipped.
    if neg_sources is None:
        srs_idx = list(range(len(neg_seqs)))
        n_skipped = 0
    else:
        if len(neg_sources) != len(neg_seqs):
            print(
                f"ERROR: neg_sources ({len(neg_sources)}) length != negatives "
                f"({len(neg_seqs)}).",
                file=sys.stderr,
            )
            sys.exit(1)
        srs_idx = [i for i, s in enumerate(neg_sources)
                   if "srs" in str(s).lower()]
        n_skipped = len(neg_seqs) - len(srs_idx)

    if not srs_idx:
        print(
            f"  Composition check: no SRS negatives present "
            f"({n_skipped} non-SRS negatives skipped); nothing to verify."
        )
        return

    mismatches = []
    for i in srs_idx:
        p, n = pos_seqs[i], neg_seqs[i]
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
            f"ERROR: {len(mismatches)} SRS pair(s) fail composition matching. "
            f"Check the shuffler.",
            file=sys.stderr,
        )
        sys.exit(1)

    msg = f"  Composition-match assertion passed for all {len(srs_idx)} SRS pairs"
    if n_skipped:
        msg += f" ({n_skipped} non-SRS negatives skipped — CDS/intergenic/downstream)"
    print(msg + ".")


# ─────────────────────────────────────────────────────────────────────
# CNN encoder — fixed-length one-hot, matching T16 (tokenise.py) and the
# T20 trainer (train_cnn.py), which hard-requires X shape (N, window, 4).
# ─────────────────────────────────────────────────────────────────────

# One-hot channel order — MUST match T16's one_hot(): A=0, C=1, G=2, T=3.
# N (and any other non-ACGT base) becomes an ALL-ZERO row, exactly as T16
# does it. To the CNN, an N row and a would-be pad row are identical, which
# is why the CNN uses a FIXED window (no separate padding channel/mask).
CNN_BASE = {"A": 0, "C": 1, "G": 2, "T": 3}


def _one_hot(seq, window):
    """
    One-hot a single sequence to shape (window, 4), float32.
    - Longer than `window`: truncated to the first `window` bases (matches
      T16's one_hot, which indexes seq[:L]).
    - Shorter than `window`: the tail rows stay all-zero (implicit padding).
    - N / non-ACGT: all-zero row (matches T16 exactly).
    This is a byte-for-byte reimplementation of T16.one_hot(seq, window).
    """
    import numpy as np
    a = np.zeros((window, 4), dtype=np.float32)
    for i, b in enumerate(seq[:window]):
        j = CNN_BASE.get(b)
        if j is not None:
            a[i, j] = 1.0
    return a


def encode_cnn(sequences, window=81):
    """
    Fixed-length one-hot encode for the T20 CNN.

    Returns X of shape (N, window, 4) float32 — the on-disk layout T20's
    load_split() requires (it asserts X.shape[1:] == (window, 4), then
    transposes to (N, 4, window) itself). This deliberately does NOT produce
    variable-length integer IDs: T20 would reject that on its shape check.

    The CNN operates on the fixed 81 bp promoter window; only the NTv3 path
    uses long genomic context (300-681 bp). Different models, different
    input lengths — by design, not by accident.
    """
    import numpy as np

    # Warn (do not silently truncate) if sequences exceed the window: for the
    # CNN, anything past `window` bp is dropped, so a caller feeding 300 bp
    # here probably meant to re-window upstream (T15 does this) or to use NTv3.
    over = [len(s) for s in sequences if len(s) > window]
    if over:
        warnings.warn(
            f"CNN: {len(over)} sequence(s) longer than window={window} bp will "
            f"be truncated to the first {window} bp (max seen: {max(over)}). "
            f"If you intended long genomic context, that is the NTv3 path, not "
            f"the CNN — the T20 CNN is fixed at {window} bp."
        )

    X = (np.stack([_one_hot(s, window) for s in sequences])
         if sequences else np.empty((0, window, 4), dtype=np.float32))
    assert X.shape == (len(sequences), window, 4), \
        f"internal error: built X {X.shape}, expected {(len(sequences), window, 4)}"
    return {"X": X, "_format": "cnn_onehot", "window": window}


# ─────────────────────────────────────────────────────────────────────
# NT (Nucleotide Transformer v3) encoder
# ─────────────────────────────────────────────────────────────────────


def _nt_pad_to_multiple(sequences, multiple, pad_length=None):
    """
    Pad each raw DNA string with 'N' up to a common length that is a multiple
    of `multiple`, per NTv3's U-Net downsampling requirement.

    InstaDeep's env doc (and its own smoke test, which does
    "SEQ".ljust(128, 'N')) is explicit: you pad the RAW nucleotide string
    with 'N' before tokenizing. You must NOT pad with [PAD] — the models were
    not trained on it. batch_np_tokenize does not do this rounding for you.

    All sequences are padded to the SAME target length so a positives run and
    a negatives run produce identical token dimensions (required for the T24
    1:1 pairing). Since each T24 negative is a shuffle of its paired positive
    (equal raw length), they receive the same number of appended 'N's and the
    pairing is preserved in token space.
    """
    max_len = max(len(s) for s in sequences)

    if pad_length is not None:
        if pad_length % multiple != 0:
            print(
                f"ERROR: --nt-pad-length ({pad_length}) must be a multiple of "
                f"{multiple} (NTv3 downsampling requirement).",
                file=sys.stderr,
            )
            sys.exit(1)
        if max_len > pad_length:
            print(
                f"ERROR: longest sequence ({max_len} bp) exceeds --nt-pad-length "
                f"({pad_length} bp). Increase --nt-pad-length or crop upstream; "
                f"refusing to silently truncate a promoter.",
                file=sys.stderr,
            )
            sys.exit(1)
        target = pad_length
    else:
        # Round the batch max up to the next multiple.
        target = ((max_len + multiple - 1) // multiple) * multiple

    padded = [s.ljust(target, "N") for s in sequences]
    return padded, target


def encode_nt(sequences, model_name, pad_multiple=128, pad_length=None,
              embeddings_layer=None):
    """
    Tokenize sequences for Nucleotide Transformer v3 using its native JAX
    package (`nucleotide_transformer_v3`), NOT HuggingFace.

    NTv3 uses single-base tokenization (vocab size 11). The tokenizer is
    obtained via get_ntv3_tokenizer() (no model weights required) and exposes
    batch_np_tokenize(sequences).

    PADDING SEMANTICS (critical):
      - Sequences are right-padded with 'N' (not [PAD]) to a multiple of the
        downsample factor (128 for 7-downsample models). This is required
        because the model was NEVER trained on [PAD] tokens.
      - N is a real base in NTv3's vocabulary. The model attends to N-padded
        positions the same way it attends to any other base. Therefore, the
        attention mask is ALL ONES — every position, including appended N's,
        has mask=1. This is not padding in the "ignore this" sense; it is
        genuine input the model processes.

    Args:
        model_name:       NTv3 checkpoint NAME (only used if loading via the
                          full model loader; tokenization-only runs can ignore
                          this since get_ntv3_tokenizer() needs no weights).
        pad_multiple:     128 for the main 7-downsample models; 32 for the
                          592M 5-downsample experimental variants.
        pad_length:       optional explicit target length (multiple of
                          pad_multiple). Set the SAME value for the positives
                          and negatives runs to guarantee matching shapes.
        embeddings_layer: only relevant if you later run the model; unused
                          for tokenization.

    Returns {"tokens": np.ndarray, "attention_mask": np.ndarray,
             "real_lengths": list[int], "_format": "ntv3_jax"}.
    """
    # Pad raw sequences with 'N' to a multiple of the downsample factor FIRST,
    # so pad-length validation fails fast even outside the ntv3-jax env.
    real_lengths = [len(s) for s in sequences]
    padded, target = _nt_pad_to_multiple(sequences, pad_multiple, pad_length)
    print(f"  NTv3: padded {len(padded)} sequences with 'N' to {target} bp "
          f"(multiple of {pad_multiple}).")

    try:
        import numpy as np
        from nucleotide_transformer_v3.tokenizers import get_ntv3_tokenizer
    except ImportError:
        print(
            "ERROR: `nucleotide_transformer_v3` (JAX) not installed. This must "
            "run inside the ntv3-jax environment (see NTV3_ENV.md).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build tokenizer directly — no model weights needed, no HF contact.
    tokenizer = get_ntv3_tokenizer()

    tokens = np.asarray(tokenizer.batch_np_tokenize(padded))

    # FIX (Issue 2): N is a real base in NTv3's vocabulary, NOT padding.
    # The model was never trained on [PAD] — we pad with N precisely because
    # the model treats it as legitimate input. Therefore the attention mask
    # is ALL ONES: the model should attend to every position, including the
    # appended N's.
    attention_mask = np.ones_like(tokens, dtype=np.int32)

    # Sanity: token length should itself be a multiple of pad_multiple.
    tok_len = tokens.shape[1] if tokens.ndim > 1 else tokens.shape[0]
    if tok_len % pad_multiple != 0:
        warnings.warn(
            f"NTv3 token length {tok_len} is not a multiple of {pad_multiple}. "
            f"If the model rejects this batch, check that the raw sequence "
            f"length (not the token count) was rounded to a multiple of "
            f"{pad_multiple}."
        )

    return {
        "tokens": tokens,
        "attention_mask": attention_mask,
        "real_lengths": real_lengths,
        "_format": "ntv3_jax",
    }


# ─────────────────────────────────────────────────────────────────────
# ProkBERT encoder
# ─────────────────────────────────────────────────────────────────────


def encode_prokbert(sequences, model_repo, max_length, kmer, shift):
    """Tokenize sequences for ProkBERT using its native LCA k-mer tokenizer."""
    import torch

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
# CNN from T15 parquet split → T20-compatible .npz (X, label, weight, ...)
# ─────────────────────────────────────────────────────────────────────


def encode_cnn_from_parquet(parquet_path, output_path, window=81, min_length=1):
    """
    One-hot encode a T15 split parquet for the T20 CNN, writing a .npz whose
    layout is EXACTLY what train_cnn.py's load_split() consumes:

        X       float32 (N, window, 4)   one-hot, channels-last
        label   int                      per-row class label
        weight  float                    T15 per-sample weight (silver-aware)
        cluster_id  unicode              carried for traceability (as in T16)

    load_split() asserts X.shape[1:] == (window, 4) and reads X/label/weight
    with allow_pickle=False, so those three must be plain arrays. This mirrors
    T16's CNN branch exactly (same one_hot, same npz keys), so encode.py and
    T16 produce interchangeable CNN inputs.

    Pure conversion: no re-split/re-window beyond the fixed-window one-hot,
    no reorder. Row i out == row i in. A row-count guard fires before writing.
    """
    import pandas as pd
    import numpy as np

    df = pd.read_parquet(parquet_path).reset_index(drop=True)
    n_in = len(df)
    print(f"Loaded {n_in} rows from {parquet_path}")

    for required in ("sequence", "label", "weight"):
        if required not in df.columns:
            print(
                f"ERROR: {parquet_path} missing required column '{required}' "
                f"for the CNN. Columns: {list(df.columns)}",
                file=sys.stderr,
            )
            sys.exit(1)

    sequences = df["sequence"].astype(str).str.upper().tolist()
    check_sequences(sequences, min_length, "parquet")

    X = (np.stack([_one_hot(s, window) for s in sequences])
         if n_in else np.empty((0, window, 4), dtype=np.float32))

    # Row-count guard (the T16 invariant).
    if X.shape[0] != n_in:
        print(
            f"ERROR: built {X.shape[0]} rows from {n_in} input rows — row "
            f"alignment broken. Aborting.",
            file=sys.stderr,
        )
        sys.exit(1)
    assert X.shape == (n_in, window, 4), \
        f"internal error: X {X.shape} != {(n_in, window, 4)}"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path = output_path.with_suffix(".npz")

    # Match T16's CNN npz keys: X, label, weight, cluster_id. cluster_id is
    # fixed-width unicode so the file loads with allow_pickle=False.
    save_kwargs = dict(
        X=X,
        label=df["label"].to_numpy(),
        weight=df["weight"].to_numpy(),
    )
    if "cluster_id" in df.columns:
        save_kwargs["cluster_id"] = np.array(
            df["cluster_id"].astype(str).tolist(), dtype="U"
        )
    np.savez_compressed(npz_path, **save_kwargs)

    # Also write a full metadata sidecar (T16 does this: {split}_meta.parquet),
    # so label/weight/source/species/etc. are all recoverable row-aligned.
    meta_cols = [c for c in df.columns if c != "sequence"]
    meta_path = npz_path.with_name(npz_path.stem + "_meta.parquet")
    df[meta_cols].to_parquet(meta_path, index=False)

    print(f"Saved {n_in} rows to {npz_path}:")
    print(f"  X:      shape={X.shape}, dtype={X.dtype}  (T20 load_split ok)")
    print(f"  label:  {len(df)} values")
    print(f"  weight: {len(df)} values (T15 silver-aware)")
    print(f"  metadata sidecar: {meta_path}  ({len(meta_cols)} columns)")


# ─────────────────────────────────────────────────────────────────────
# NTv3 from T15 parquet split (row-aligned, carries all metadata)
# ─────────────────────────────────────────────────────────────────────


def encode_nt_from_parquet(parquet_path, output_path, pad_multiple=128,
                           pad_length=None, min_length=128):
    """
    Tokenize a T15 split parquet for NTv3, carrying EVERY T15 metadata column
    (label, weight, source, species, sigma_factor, cluster_id, promoter_id,
    split, in_genus, strand, ...) through UNCHANGED and row-aligned.

    This mirrors T16 (tokenise.py) semantics deliberately:
      - reads results/T15/{split}.parquet
      - carries all columns EXCEPT `sequence` (the T16 meta_cols() contract)
      - is a pure format conversion: NO re-split, re-window, re-balance, or
        reorder. Output row i corresponds to input row i, always.
      - a row-count guard asserts len(out) == len(in) before writing.

    Output is a parquet with the carried metadata columns plus:
      - input_ids:      list[int] per row (the NTv3 token IDs)
      - attention_mask: list[int] per row (ALL ONES — N is a real base in
                        NTv3's vocabulary, not padding to be ignored)
      - real_length:    pre-padding sequence length (int), for traceability

    Run this inside the ntv3-jax environment.
    """
    import pandas as pd
    import numpy as np

    df = pd.read_parquet(parquet_path).reset_index(drop=True)
    n_in = len(df)
    print(f"Loaded {n_in} rows from {parquet_path}")

    if "sequence" not in df.columns:
        print(
            f"ERROR: {parquet_path} has no 'sequence' column. Columns: "
            f"{list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # meta_cols contract, identical to T16: carry everything EXCEPT sequence.
    meta_cols = [c for c in df.columns if c != "sequence"]

    # Uppercase (NTv3, like the HF tokenizers in T16, needs uppercase input).
    sequences = df["sequence"].astype(str).str.upper().tolist()

    # Validate (length floor + alphabet). Same checks as the FASTA path.
    check_sequences(sequences, min_length, "parquet")

    lengths = [len(s) for s in sequences]
    print(f"  Lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={sum(lengths)/len(lengths):.0f} bp")

    # Encode via the shared NTv3 tokenizer path (N-pad + all-ones mask).
    result = encode_nt(
        sequences,
        model_name=None,          # tokenizer needs no weights
        pad_multiple=pad_multiple,
        pad_length=pad_length,
    )
    tokens = result["tokens"]              # (n_in, target_len) numpy array
    attention_mask = result["attention_mask"]
    real_lengths = result["real_lengths"]

    # Row-count guard BEFORE writing — the T16 invariant. Tokenisation must
    # never add, drop, or reorder rows.
    if tokens.shape[0] != n_in:
        print(
            f"ERROR: tokeniser produced {tokens.shape[0]} rows from {n_in} "
            f"input rows — row alignment broken. Aborting.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build the output frame: metadata carried through UNCHANGED, in the SAME
    # row order, with token columns attached as list-valued cells (matching
    # T16's nt/prokbert parquet layout).
    out = df[meta_cols].reset_index(drop=True).copy()
    out["input_ids"] = list(tokens.tolist())
    out["attention_mask"] = list(attention_mask.tolist())
    out["real_length"] = list(real_lengths)

    # Final alignment assertion: same length, and every carried metadata
    # column is byte-identical to the input (nothing silently mutated).
    assert len(out) == n_in, "row-count guard failed after assembly"
    for c in meta_cols:
        assert out[c].reset_index(drop=True).equals(
            df[c].reset_index(drop=True)
        ), f"metadata column '{c}' was altered — alignment broken"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    print(f"Saved {len(out)} rows to {output_path}")
    print(f"  token width: {tokens.shape[1]}  (multiple of {pad_multiple})")
    print(f"  carried metadata columns: {meta_cols}")
    print(f"  + input_ids, attention_mask (all ones), real_length")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Encode FASTA sequences for CNN / NT v3 / ProkBERT models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Encode T24 negatives for CNN (variable-length, >=128 bp)
  python encode.py --model cnn --input neg.fasta --output neg_cnn.pt \\
      --positives pos.fasta --min-length 128

  # NTv3 from a T15 split parquet (TRAINING path; run in ntv3-jax env).
  # Carries label/weight/source/species + all T15 metadata through,
  # row-aligned with T16. Use the SAME --nt-pad-length across splits.
  python encode.py --model nt --from-parquet results/T15/train.parquet \\
      --output results/T16/nt3/train.parquet --nt-pad-length 768
  python encode.py --model nt --from-parquet results/T15/val.parquet \\
      --output results/T16/nt3/val.parquet --nt-pad-length 768

  # NTv3 from loose FASTA (standalone; e.g. T24 composition-matched shuffle).
  python encode.py --model nt --input neg.fasta --output neg_nt.pt \\
      --positives pos.fasta --nt-pad-length 768

  # CNN from a T15 split parquet → T20-compatible (N,81,4) .npz + metadata.
  python encode.py --model cnn --from-parquet results/T15/train.parquet \\
      --output results/T16/cnn/train.npz --window 81

  # ProkBERT from loose FASTA. Base neuralbioinfo/prokbert-mini is k6s1
  # (6-mer, shift 1) — this MUST match T16/T18 training tokenisation.
  # (Other variants differ: mini-long=k6s2; -promoter/-c-promoter=k1s1.)
  python encode.py --model prokbert --input neg.fasta --output neg_prokbert.pt \\
      --positives pos.fasta --prokbert-kmer 6 --prokbert-shift 1 --max-tokens 128
""",
    )

    parser.add_argument("--model", required=True, choices=["cnn", "nt", "prokbert"])
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input FASTA file (FASTA mode). Mutually "
                        "exclusive with --from-parquet.")
    parser.add_argument("--from-parquet", type=str, default=None,
                        help="Path to a T15 split parquet (e.g. "
                        "results/T15/train.parquet). Carries all T15 metadata "
                        "(label, weight, source, species, ...) through "
                        "row-aligned, matching T16. Supported for --model nt "
                        "(writes token parquet) and --model cnn (writes "
                        "(N,window,4) .npz with label/weight for T20).")
    parser.add_argument("--output", required=True, type=str, help="Path to save encoded output.")
    parser.add_argument("--label", choices=["pos", "neg"], default="neg")
    parser.add_argument(
        "--window", type=int, default=81,
        help="CNN fixed one-hot window in bp (default: 81). The T20 CNN "
        "hard-requires (N, window, 4); do not change unless T20 changes.",
    )
    parser.add_argument(
        "--min-length", type=int, default=128,
        help="Minimum required sequence length in bp (default: 128, the "
        "smallest multiple of 128 that NTv3's U-Net architecture accepts). "
        "For CNN-only runs on shorter sequences, override with e.g. "
        "--min-length 81.",
    )
    parser.add_argument(
        "--positives", type=str, default=None,
        help="Path to paired positive FASTA for T24 composition check.",
    )
    parser.add_argument(
        "--expected-count", type=int, default=None,
        help="Expected number of sequences (e.g. 93 for T24). Aborts if mismatched.",
    )

    # NT-specific (NTv3 JAX package — NOT HuggingFace). Requires ntv3-jax env.
    parser.add_argument(
        "--nt-model", type=str,
        default="NTv3_650M_pre",
        help="NTv3 checkpoint NAME for the JAX package (e.g. NTv3_650M_pre, "
        "NTv3_100M_pre, NTv3_8M_pre). Bare name, NOT an 'InstaDeepAI/...' HF "
        "repo path. Avoid *_8kb variants. Tokenization is identical across "
        "sizes, so NTv3_8M_pre tokenizes cheaply even if you'll run 650M. "
        "(NTV3_ENV.md pre-stages 100M by default — pre-stage 650M if you keep "
        "this default.)",
    )
    parser.add_argument(
        "--nt-pad-multiple", type=int, default=128,
        help="Round sequence length up to a multiple of this with 'N' padding. "
        "128 for the main 7-downsample models; 32 for the 592M 5-downsample "
        "experimental variants.",
    )
    parser.add_argument(
        "--nt-pad-length", type=int, default=None,
        help="Optional explicit target length (multiple of --nt-pad-multiple). "
        "Set the SAME value for the positives and negatives runs to guarantee "
        "matching token dimensions for T24. If omitted, uses the batch max "
        "rounded up.",
    )
    parser.add_argument(
        "--nt-embeddings-layer", type=int, default=None,
        help="Optional NTv3 layer index for embeddings_layers_to_save. Only "
        "relevant if you later run the model; not needed for tokenization.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Max token length. Used by ProkBERT only.",
    )

    # ProkBERT-specific. Defaults (k6s1) match base neuralbioinfo/prokbert-mini,
    # which is what T16 (tokenise.py) and T18 (train_prokbert.py) use — eval
    # tokenisation MUST match training. Per-variant values from the model cards:
    #   prokbert-mini / prokbert-mini-phage        -> kmer=6, shift=1 (k6s1)
    #   prokbert-mini-long                          -> kmer=6, shift=2 (k6s2)
    #   prokbert-mini-promoter / -c-promoter        -> kmer=1, shift=1 (k1s1)
    # Only change these if you switch --prokbert-repo to a non-base variant.
    parser.add_argument("--prokbert-repo", type=str, default="neuralbioinfo/prokbert-mini")
    parser.add_argument("--prokbert-kmer", type=int, default=6,
                        help="ProkBERT k-mer size. Default 6 = base prokbert-mini "
                        "(matches T16/T18). Use 1 only for the -promoter variants.")
    parser.add_argument("--prokbert-shift", type=int, default=1,
                        help="ProkBERT shift. Default 1 = base prokbert-mini "
                        "(matches T16/T18). Use 2 only for prokbert-mini-long.")

    args = parser.parse_args()

    # ── Input-source validation ───────────────────────────────────
    if args.from_parquet and args.input:
        print(
            "ERROR: pass either --input (FASTA) or --from-parquet, not both.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not args.from_parquet and not args.input:
        print(
            "ERROR: one of --input (FASTA) or --from-parquet is required.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Parquet mode (T15 split → row-aligned, metadata-carrying) ──
    # This is the training-integration path: reads a T15 split, carries
    # label/weight/source/species/... through unchanged, writes output
    # row-aligned with T16.
    if args.from_parquet:
        if args.model == "nt":
            encode_nt_from_parquet(
                args.from_parquet,
                args.output,
                pad_multiple=args.nt_pad_multiple,
                pad_length=args.nt_pad_length,
                min_length=args.min_length,
            )
            return
        elif args.model == "cnn":
            # CNN uses a fixed window; min_length only needs to be >=1 here
            # (short sequences are zero-padded within the window, matching T16).
            encode_cnn_from_parquet(
                args.from_parquet,
                args.output,
                window=args.window,
                min_length=1,
            )
            return
        else:
            print(
                f"ERROR: --from-parquet supports --model nt and --model cnn "
                f"(got --model {args.model}). ProkBERT tokenisation from T15 "
                f"parquet is handled by T16 (tokenise.py).",
                file=sys.stderr,
            )
            sys.exit(1)

    # ── FASTA mode (standalone; e.g. T24 composition-matched shuffle) ──
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

    # ── Composition-match check (T24 all-SRS case) ────────────────
    # In FASTA mode every negative is an SRS shuffle by construction (there is
    # no source column), so neg_sources=None → all pairs checked. For mixed
    # T15 negatives, use --from-parquet, where the check reads the source
    # column and skips CDS/intergenic/downstream automatically.
    if args.positives is not None:
        print(f"Checking composition match against {args.positives} ...")
        _, pos_seqs = load_fasta(args.positives)
        check_sequences(pos_seqs, args.min_length, "positives")
        assert_composition_match(pos_seqs, sequences, neg_sources=None)

    # ── Encode ────────────────────────────────────────────────────
    if args.model == "cnn":
        result = encode_cnn(sequences, window=args.window)

    elif args.model == "nt":
        # NTv3 (JAX): we pad raw sequences with 'N' to a multiple of the
        # downsample factor here (batch_np_tokenize does NOT do this).
        result = encode_nt(
            sequences,
            args.nt_model,
            pad_multiple=args.nt_pad_multiple,
            pad_length=args.nt_pad_length,
            embeddings_layer=args.nt_embeddings_layer,
        )

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

    if result.get("_format") == "ntv3_jax":
        # NTv3 tokens are a numpy array from the JAX package, not torch
        # tensors. Save as .npz to keep the pipeline JAX-native and avoid a
        # spurious torch dependency / dtype coupling for this path.
        import numpy as np
        npz_path = output_path.with_suffix(".npz")
        np.savez_compressed(
            npz_path,
            tokens=result["tokens"],
            attention_mask=result["attention_mask"],
            real_lengths=np.array(result["real_lengths"], dtype=np.int32),
        )
        print(f"Saved to {npz_path}:")
        print(f"  tokens:         shape={result['tokens'].shape}, "
              f"dtype={result['tokens'].dtype}")
        print(f"  attention_mask:  shape={result['attention_mask'].shape} "
              f"(all ones — N is a real base)")
        print(f"  real_lengths:    {len(result['real_lengths'])} entries "
              f"(pre-padding lengths for traceability)")
        output_path = npz_path  # for the header sidecar below

    elif result.get("_format") == "cnn_onehot":
        # CNN one-hot: write the T20 .npz layout (X, label, weight). In FASTA
        # mode we have no labels/weights, so derive a label from --label and
        # use weight=1.0 (the T24/standalone case). For real training data,
        # use --from-parquet, which carries the true T15 label/weight.
        import numpy as np
        npz_path = output_path.with_suffix(".npz")
        X = result["X"]
        n = X.shape[0]
        label_val = 1 if args.label == "pos" else 0
        np.savez_compressed(
            npz_path,
            X=X,
            label=np.full(n, label_val, dtype=np.int64),
            weight=np.ones(n, dtype=np.float32),
        )
        print(f"Saved to {npz_path}:")
        print(f"  X:      shape={X.shape}, dtype={X.dtype}  (T20 load_split ok)")
        print(f"  label:  all {label_val} (from --label; use --from-parquet "
              f"for real T15 labels)")
        print(f"  weight: all 1.0 (use --from-parquet for T15 silver weights)")
        output_path = npz_path

    else:
        import torch
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