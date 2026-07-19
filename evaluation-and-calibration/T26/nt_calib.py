#!/usr/bin/env python3
"""
T26 — Logistic (Platt) calibration of the trained Nucleotide Transformer promoter model (T21 trial 14).

Owner: Ishi.  NT analog of calibrate_cnn.py — same calibration logic, different model loading.

What it does
------------
1. Loads the NT backbone `InstaDeepAI/nucleotide-transformer-v2-250m-multi-species` and layers the
   TRAINED LoRA adapter (results/T21/nt_lora/trial_14/best_adapter — what T19 saved via
   model.save_pretrained) back on with PeftModel. The adapter also carries the classifier head
   (T19 used modules_to_save=["classifier"]), so this reconstructs the exact fine-tuned classifier.
   Uses the SAME v5-compat shim + config backfill as train_nt_lora.py so it loads identically.
2. Builds the calibration dataset = T9 positives (label 1) + a balanced sample of T14 negatives
   (label 0). Default 1:1, SRS class. Both the ratio (--neg-per-pos) and classes (--neg-classes) are flags.
3. Tokenises with the NT tokenizer (attention-masked padding handles variable length — no fixed
   one-hot window needed), scores on GPU, and takes the positive-class logit difference
   (logits[:,1] - logits[:,0]) as the 1-D calibration feature.
4. Splits stratified (fixed seed) into a calibration-fit subset and a held-out test subset, fits a
   1-D LogisticRegression (Platt scaling), applies it ONLY to the held-out logits, and compares raw
   vs calibrated probabilities (Brier / log-loss / AUROC / AUPRC / ECE). A second calibrator adds the
   AT covariate (logit + AT) and is reported alongside.

NOTE: run this on the `nt-tf4x` env (transformers 4.57), NOT the 5.3.0 env — NT's remote code is
v4-era (infrastructure/T4/NT_ENV.md). The compat shims below are guarded so it ALSO loads on v5, but
NT was trained + validated on 4.57. Calibration is prevalence-dependent: a 1:1 set calibrates to a
balanced operating point, not genome-wide promoter prevalence (set --neg-per-pos to the deployment
base rate if needed).

HPC4 adapter path: /project/igem2026/cb-promoter-mining/results/T21/nt_lora/trial_14/best_adapter
Run (one GPU on HPC4): see the sbatch snippet at the bottom of this file.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

NT_ID = "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species"


# ------------------------------------------------------------------ helpers
def die(msg: str) -> "NoReturn":
    """Print a clear error and exit non-zero (keeps the traceback out of the user's face)."""
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def at_fraction(seq: str) -> float:
    seq = seq.upper()
    return (seq.count("A") + seq.count("T")) / max(len(seq), 1)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width-binning ECE = sum_b (|bin|/N) * |acc_b - conf_b|.

    acc_b  = fraction of positives among predictions in the bin
    conf_b = mean predicted probability in the bin
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # last bin is closed on the right so prob == 1.0 is counted
        mask = (y_prob >= lo) & (y_prob <= hi) if i == n_bins - 1 else (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def score_block(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> dict:
    from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                                  roc_auc_score)
    return {
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob, n_bins),
    }


# ------------------------------------------------------------------ data loading (identical to calibrate_cnn.py)
def load_positives(path: Path) -> pd.DataFrame:
    if not path.is_file():
        die(f"positives file not found: {path}  (--positives)")
    df = pd.read_csv(path, sep="\t")
    seq_col = next((c for c in ("sequence", "Sequence", "seq") if c in df.columns), None)
    if seq_col is None:
        die(f"positives file {path} has no sequence column (looked for sequence/Sequence/seq); "
            f"got {list(df.columns)}")
    lab_col = next((c for c in ("label", "Target", "y") if c in df.columns), None)
    out = pd.DataFrame({"sequence": df[seq_col].astype(str)})
    out["label"] = df[lab_col].astype(int) if lab_col else 1  # T9 is positives-only
    out["source"] = "T9_positive"
    return out


def load_negatives(path: Path, neg_classes: str, n_needed: int, seed: int) -> pd.DataFrame:
    if not path.is_file():
        die(f"negatives file not found: {path}  (--negatives; expected T14 results/T14/negative_set.tsv)")
    df = pd.read_csv(path, sep="\t")
    if "sequence" not in df.columns:
        die(f"negatives file {path} has no 'sequence' column; got {list(df.columns)}")
    wanted = [c.strip() for c in neg_classes.split(",") if c.strip()]
    if "all" not in [w.lower() for w in wanted]:
        if "negative_class" not in df.columns:
            die(f"negatives file {path} has no 'negative_class' column but --neg-classes={neg_classes} "
                f"was requested; got {list(df.columns)}")
        avail = set(df["negative_class"].unique())
        missing = [w for w in wanted if w not in avail]
        if missing:
            die(f"requested negative class(es) {missing} not in {path}; available: {sorted(avail)}")
        df = df[df["negative_class"].isin(wanted)]
    if len(df) == 0:
        die(f"no negatives left after filtering to classes {wanted} in {path}")
    if len(df) < n_needed:
        print(f"WARNING: only {len(df)} negatives available for classes {wanted}, "
              f"fewer than the {n_needed} requested — using all of them.")
        sample = df
    else:
        sample = df.sample(n=n_needed, random_state=seed)
    out = pd.DataFrame({"sequence": sample["sequence"].astype(str)})
    out["label"] = 0
    out["source"] = sample.get("negative_class", pd.Series(["negative"] * len(sample))).values
    return out.reset_index(drop=True)


def validate_sequences(seqs: pd.Series) -> None:
    """Strict ACGT-only check (the NT tokenizer handles length via padding/truncation, so — unlike the
    CNN's fixed one-hot window — no length harmonisation is needed here)."""
    upper = seqs.str.upper()
    bad = upper.str.contains(r"[^ACGT]", regex=True)
    if bad.any():
        examples = upper[bad].head(3).tolist()
        die(f"{int(bad.sum())} sequence(s) contain non-ACGT characters (e.g. {examples}). "
            f"Clean or remove them before calibrating.")
    lengths = upper.str.len()
    print(f"All {len(upper)} sequences are ACGT-only; length range {int(lengths.min())}–{int(lengths.max())} bp "
          f"(tokenizer pads/truncates to --max-length).")


# ------------------------------------------------------------------ NT model loading
def load_nt_model(base_id: str, adapter_dir: Path, device):
    """Rebuild the trained NT classifier: base backbone + T19's LoRA adapter (which also carries the
    classifier head via modules_to_save). Mirrors train_nt_lora.py's load path exactly."""
    import torch
    from torch import nn
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    from peft import PeftModel

    if not adapter_dir.is_dir():
        die(f"NT adapter dir not found: {adapter_dir}  (--adapter-dir)\n"
            f"       expected the LoRA adapter T19 saved via model.save_pretrained(.../best_adapter)")
    if not (adapter_dir / "adapter_config.json").is_file():
        die(f"{adapter_dir} has no adapter_config.json — not a PEFT/LoRA adapter dir. If this is a full "
            f"merged model, load it with AutoModelForSequenceClassification instead of PeftModel.")

    # --- v5-compat shim: NT's cached remote code imports two head-pruning utils that transformers v5
    #     REMOVED. Only imported (not called) at load, so re-supplying them lets NT load on v5. Guarded,
    #     so it's inert on the v4.57 nt-tf4x env NT actually runs on. (copied from train_nt_lora.py) ---
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

    # --- config backfill: NT's remote code reads encoder/decoder attrs v5 dropped as base defaults
    #     (all False/0 -> NT is encoder-only). Inert on v4. (copied from train_nt_lora.py) ---
    config = AutoConfig.from_pretrained(base_id, num_labels=2, trust_remote_code=True)
    for _attr, _default in (("is_decoder", False), ("is_encoder_decoder", False),
                            ("add_cross_attention", False), ("tie_encoder_decoder", False),
                            ("chunk_size_feed_forward", 0), ("use_cache", False),
                            ("output_attentions", False), ("output_hidden_states", False)):
        if not hasattr(config, _attr):
            setattr(config, _attr, _default)

    try:
        base = AutoModelForSequenceClassification.from_pretrained(
            base_id, config=config, trust_remote_code=True)
        model = PeftModel.from_pretrained(base, str(adapter_dir))
        tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001 - surface any load-time failure clearly
        die(f"could not load NT base '{base_id}' + adapter '{adapter_dir}': {e}\n"
            f"       (pre-stage NT on the login node: HF_HOME set + HF_HUB_OFFLINE=1)")
    model.to(device).eval()
    return model, tokenizer


# ------------------------------------------------------------------ main
def main() -> None:
    repo_default = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="T26 NT Platt calibration (T21 trial 14)")
    ap.add_argument("--repo-root", type=Path, default=repo_default)
    ap.add_argument("--base-id", default=NT_ID, help=f"NT backbone HF id (default: {NT_ID})")
    ap.add_argument("--adapter-dir", type=Path, default=None,
                    help="trained LoRA adapter dir (default: <repo>/results/T21/nt_lora/trial_14/best_adapter)")
    ap.add_argument("--positives", type=Path, default=None,
                    help="T9 positives tsv (default: <repo>/data-acquisition-and-curation/T9/literature_promoters.tsv)")
    ap.add_argument("--negatives", type=Path, default=None,
                    help="T14 negatives tsv (default: <repo>/results/T14/negative_set.tsv)")
    ap.add_argument("--neg-classes", default="SRS",
                    help="comma list of T14 negative_class values, or 'all' (default: SRS)")
    ap.add_argument("--neg-per-pos", type=float, default=1.0,
                    help="negatives sampled per positive (default 1.0 = balanced 50/50)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output dir (default: <repo>/evaluation-and-calibration/T26/out/nt)")
    ap.add_argument("--max-length", type=int, default=128,
                    help="tokenizer max_length cap in TOKENS (default 128 — safely above the ~52–81 bp "
                         "windows so nothing real is truncated; padding is attention-masked)")
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--ece-bins", type=int, default=10)
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    adapter_dir = (args.adapter_dir or repo / "results/T21/nt_lora/trial_14/best_adapter").resolve()
    positives_path = (args.positives or repo / "data-acquisition-and-curation/T9/literature_promoters.tsv").resolve()
    negatives_path = (args.negatives or repo / "results/T14/negative_set.tsv").resolve()
    out_dir = (args.out_dir or repo / "evaluation-and-calibration/T26/out/nt").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch  # deferred: torch/CUDA init is slow; do it after arg/path validation, not for --help
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    import joblib

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- model (base backbone + trained LoRA adapter) ---
    model, tokenizer = load_nt_model(args.base_id, adapter_dir, device)

    # --- dataset ---
    pos = load_positives(positives_path)
    n_needed = int(round(args.neg_per_pos * len(pos)))
    neg = load_negatives(negatives_path, args.neg_classes, n_needed, args.seed)
    data = pd.concat([pos, neg], ignore_index=True)
    data["at_fraction"] = data["sequence"].map(at_fraction)

    if data["label"].nunique() < 2:
        die(f"calibration dataset has only one class (labels present: {sorted(data['label'].unique())}). "
            f"Need both positives and negatives.")
    validate_sequences(data["sequence"])

    # --- tokenise + batched GPU inference ---
    seqs = data["sequence"].str.upper().tolist()
    logit0, logit1 = [], []
    out_cols = None
    with torch.no_grad():
        for i in range(0, len(seqs), args.batch_size):
            batch = seqs[i:i + args.batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=args.max_length,
                            return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc).logits
            if out.ndim != 2 or out.shape[1] != 2:
                die(f"unexpected model output shape {tuple(out.shape)}; expected (batch, 2).")
            out_cols = int(out.shape[1])
            logit0.extend(out[:, 0].cpu().tolist())
            logit1.extend(out[:, 1].cpu().tolist())

    logit0 = np.asarray(logit0)
    logit1 = np.asarray(logit1)
    logit_diff = logit1 - logit0                                    # 1-D calibration feature
    raw_prob = 1.0 / (1.0 + np.exp(-logit_diff))                    # == softmax(...)[:,1]
    data["raw_logit"] = logit_diff
    data["raw_prob"] = raw_prob

    # --- stratified calib/test split (fixed seed) ---
    idx = np.arange(len(data))
    calib_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=data["label"].values)
    data["split"] = "calib"
    data.loc[test_idx, "split"] = "test"

    y_calib = data["label"].values[calib_idx]
    y_test = data["label"].values[test_idx]
    if len(set(y_calib)) < 2 or len(set(y_test)) < 2:
        die("stratified split produced a single-class calib or test subset — "
            "increase the dataset size or adjust --test-size/--neg-per-pos.")

    # --- fit calibrators on the CALIB subset only ---
    Xc_logit = logit_diff[calib_idx].reshape(-1, 1)                 # (N, 1) per spec
    Xc_logit_at = np.column_stack([logit_diff[calib_idx], data["at_fraction"].values[calib_idx]])
    # penalty=None -> unregularized logistic regression (Platt scaling); the supported way to fit a
    # plain logistic calibrator (kept identical across the three T26 scripts).
    cal_logit = LogisticRegression(penalty=None).fit(Xc_logit, y_calib)
    cal_logit_at = LogisticRegression(penalty=None).fit(Xc_logit_at, y_calib)

    # --- apply to the HELD-OUT test logits only ---
    Xt_logit = logit_diff[test_idx].reshape(-1, 1)
    Xt_logit_at = np.column_stack([logit_diff[test_idx], data["at_fraction"].values[test_idx]])
    calib_prob = cal_logit.predict_proba(Xt_logit)[:, 1]
    calib_prob_at = cal_logit_at.predict_proba(Xt_logit_at)[:, 1]

    metrics = {
        "raw": score_block(y_test, raw_prob[test_idx], args.ece_bins),
        "calibrated_logit": score_block(y_test, calib_prob, args.ece_bins),
        "calibrated_logit_at": score_block(y_test, calib_prob_at, args.ece_bins),
    }

    # --- write CSV (test rows get calibrated probs; calib rows NaN, per "apply only to held-out") ---
    data["calib_prob"] = np.nan
    data["calib_prob_at"] = np.nan
    data.loc[test_idx, "calib_prob"] = calib_prob
    data.loc[test_idx, "calib_prob_at"] = calib_prob_at
    csv_path = out_dir / "nt_calibration_predictions.csv"
    data[["sequence", "label", "source", "at_fraction",
          "raw_logit", "raw_prob", "calib_prob", "calib_prob_at", "split"]].to_csv(csv_path, index=False)

    # --- save calibrators + metrics ---
    cal_path = out_dir / "nt_calibrator.joblib"
    cal_at_path = out_dir / "nt_calibrator_at.joblib"
    joblib.dump(cal_logit, cal_path)
    joblib.dump(cal_logit_at, cal_at_path)
    metrics_path = out_dir / "nt_calibration_metrics.json"
    metrics_path.write_text(json.dumps({
        "device": str(device),
        "n_total": int(len(data)),
        "n_positive": int((data["label"] == 1).sum()),
        "n_negative": int((data["label"] == 0).sum()),
        "neg_classes": args.neg_classes,
        "neg_per_pos": args.neg_per_pos,
        "n_calib": int(len(calib_idx)),
        "n_test": int(len(test_idx)),
        "test_size": args.test_size,
        "seed": args.seed,
        "max_length": args.max_length,
        "model_output_shape": [int(len(data)), out_cols],
        "base_id": args.base_id,
        "metrics": metrics,
        "paths": {
            "base_id": args.base_id,
            "adapter_dir": str(adapter_dir),
            "positives": str(positives_path),
            "negatives": str(negatives_path),
            "calibrator": str(cal_path),
            "calibrator_at": str(cal_at_path),
            "predictions_csv": str(csv_path),
            "metrics_json": str(metrics_path),
        },
    }, indent=2))

    # --- diagnostics ---
    print("\n===== T26 NT calibration diagnostics =====")
    print(f"base backbone    : {args.base_id}")
    print(f"adapter          : {adapter_dir}")
    print(f"device           : {device}")
    print(f"sequences        : {len(data)}  (pos={int((data['label']==1).sum())}, "
          f"neg={int((data['label']==0).sum())} from {args.neg_classes})")
    print(f"model output     : (batch, {out_cols})")
    print(f"calib / test     : {len(calib_idx)} / {len(test_idx)}")
    print("\nmetric        raw        cal(logit)   cal(logit+AT)")
    for m in ("brier", "log_loss", "auroc", "auprc", "ece"):
        print(f"{m:<12} {metrics['raw'][m]:>8.4f}   {metrics['calibrated_logit'][m]:>8.4f}   "
              f"{metrics['calibrated_logit_at'][m]:>8.4f}")
    print("\noutputs:")
    for p in (cal_path, cal_at_path, csv_path, metrics_path):
        print(f"  {p}")


if __name__ == "__main__":
    main()


# --- run on one GPU (HPC4 / SuperPOD) -------------------------------------------------------------
# NB use the nt-tf4x env (transformers 4.57), NOT the 5.3.0 env (infrastructure/T4/NT_ENV.md).
# srun --account=igem2026 --partition=preempt --gpus-per-node=1 --requeue \
#      python evaluation-and-calibration/T26/nt_calib.py
