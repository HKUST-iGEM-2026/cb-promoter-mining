#!/usr/bin/env python3
"""
T26 — Logistic (Platt) calibration of the trained ProkBERT-mini promoter model (T21 trial 12).

Owner: Ishi.  ProkBERT analog of calibrate_cnn.py / nt_calib.py — same calibration logic, different
model loading.

What it does
------------
1. Loads the FULL fine-tuned ProkBERT classifier T18/T21 saved (results/T21/prokbert/trial_12/best_model
   — a whole AutoModelForSequenceClassification dir, NOT a LoRA adapter like NT). The tokenizer loads
   from the BASE id `neuralbioinfo/prokbert-mini` (T18 trains plain prokbert-mini and its save dir does
   NOT ship tokenizer files — same as the T24 convention "load ProkBERT tokenizer from base").
2. Builds the calibration dataset = T9 positives (label 1) + a balanced sample of T14 negatives
   (label 0). Default 1:1, SRS class. Both the ratio (--neg-per-pos) and classes (--neg-classes) are flags.
3. Tokenises with the ProkBERT tokenizer (attention-masked padding handles variable length), scores on
   GPU, and takes the positive-class logit difference (logits[:,1] - logits[:,0]) as the 1-D
   calibration feature. (softmax(...)[:,1] is kept only for the raw before/after report — the LOGIT,
   not the squashed prob, is what feeds the calibrator.)
4. Splits stratified (fixed seed) into a calibration-fit subset and a held-out test subset, fits a
   1-D LogisticRegression (Platt scaling, penalty=None), applies it ONLY to the held-out logits, and
   compares raw vs calibrated probabilities (Brier / log-loss / AUROC / AUPRC / ECE). A second
   calibrator adds the AT covariate (logit + AT) and is reported alongside.

NOTE: run on the main `promoter-mining-hpc4` env (transformers 5.3.0) — ProkBERT loads clean there
(unlike NT, which needs the nt-tf4x env). Calibration is prevalence-dependent: a 1:1 set calibrates to
a balanced operating point, not genome-wide promoter prevalence (set --neg-per-pos to the deployment
base rate if needed).

HPC4 model path: /project/igem2026/cb-promoter-mining/results/T21/prokbert/trial_12/best_model
Run (one GPU on HPC4): see the sbatch snippet at the bottom of this file.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROKBERT_ID = "neuralbioinfo/prokbert-mini"


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


# ------------------------------------------------------------------ data loading (identical to nt_calib.py)
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
    """Strict ACGT-only check (the tokenizer handles length via padding/truncation, so — unlike the
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


# ------------------------------------------------------------------ ProkBERT model loading
def load_prokbert_model(tokenizer_id: str, model_dir: Path, device):
    """Load the FULL fine-tuned ProkBERT classifier (T18/T21 saved via trainer.save_model) + the base
    tokenizer. No PEFT/adapter (unlike NT) and no v5 shim (ProkBERT loads clean on the 5.3.0 env)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not model_dir.is_dir():
        die(f"ProkBERT model dir not found: {model_dir}  (--model-dir)\n"
            f"       expected the full model T18/T21 saved via trainer.save_model(.../best_model)")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), trust_remote_code=True)
    except Exception as e:  # noqa: BLE001 - surface any load-time failure clearly
        die(f"could not load ProkBERT model '{model_dir}' + tokenizer '{tokenizer_id}': {e}\n"
            f"       (pre-stage prokbert-mini on the login node: HF_HOME set + HF_HUB_OFFLINE=1)")
    model.to(device).eval()
    return model, tokenizer


# ------------------------------------------------------------------ main
def main() -> None:
    repo_default = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="T26 ProkBERT Platt calibration (T21 trial 12)")
    ap.add_argument("--repo-root", type=Path, default=repo_default)
    ap.add_argument("--tokenizer-id", default=PROKBERT_ID,
                    help=f"ProkBERT tokenizer HF id — the BASE, not the fine-tuned dir (default: {PROKBERT_ID})")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="trained full-model dir (default: <repo>/results/T21/prokbert/trial_12/best_model)")
    ap.add_argument("--positives", type=Path, default=None,
                    help="T9 positives tsv (default: <repo>/data-acquisition-and-curation/T9/literature_promoters.tsv)")
    ap.add_argument("--negatives", type=Path, default=None,
                    help="T14 negatives tsv (default: <repo>/results/T14/negative_set.tsv)")
    ap.add_argument("--neg-classes", default="SRS",
                    help="comma list of T14 negative_class values, or 'all' (default: SRS)")
    ap.add_argument("--neg-per-pos", type=float, default=1.0,
                    help="negatives sampled per positive (default 1.0 = balanced 50/50)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output dir (default: <repo>/evaluation-and-calibration/T26/out/prokbert)")
    ap.add_argument("--max-length", type=int, default=128,
                    help="tokenizer max_length cap in TOKENS (default 128 — safely above the ~52–81 bp "
                         "windows so nothing real is truncated; padding is attention-masked)")
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--ece-bins", type=int, default=10)
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    model_dir = (args.model_dir or repo / "results/T21/prokbert/trial_12/best_model").resolve()
    positives_path = (args.positives or repo / "data-acquisition-and-curation/T9/literature_promoters.tsv").resolve()
    negatives_path = (args.negatives or repo / "results/T14/negative_set.tsv").resolve()
    out_dir = (args.out_dir or repo / "evaluation-and-calibration/T26/out/prokbert").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch  # deferred: torch/CUDA init is slow; do it after arg/path validation, not for --help
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    import joblib

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- model (full fine-tuned ProkBERT + base tokenizer) ---
    model, tokenizer = load_prokbert_model(args.tokenizer_id, model_dir, device)

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
    csv_path = out_dir / "prokbert_calibration_predictions.csv"
    data[["sequence", "label", "source", "at_fraction",
          "raw_logit", "raw_prob", "calib_prob", "calib_prob_at", "split"]].to_csv(csv_path, index=False)

    # --- save calibrators + metrics ---
    cal_path = out_dir / "prokbert_calibrator.joblib"
    cal_at_path = out_dir / "prokbert_calibrator_at.joblib"
    joblib.dump(cal_logit, cal_path)
    joblib.dump(cal_logit_at, cal_at_path)
    metrics_path = out_dir / "prokbert_calibration_metrics.json"
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
        "tokenizer_id": args.tokenizer_id,
        "metrics": metrics,
        "paths": {
            "tokenizer_id": args.tokenizer_id,
            "model_dir": str(model_dir),
            "positives": str(positives_path),
            "negatives": str(negatives_path),
            "calibrator": str(cal_path),
            "calibrator_at": str(cal_at_path),
            "predictions_csv": str(csv_path),
            "metrics_json": str(metrics_path),
        },
    }, indent=2))

    # --- diagnostics ---
    print("\n===== T26 ProkBERT calibration diagnostics =====")
    print(f"model dir        : {model_dir}")
    print(f"tokenizer        : {args.tokenizer_id}")
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
# NB the main promoter-mining-hpc4 env (transformers 5.3.0) — ProkBERT loads clean there (NOT nt-tf4x).
# srun --account=igem2026 --partition=preempt --gpus-per-node=1 --requeue \
#      python evaluation-and-calibration/T26/prokbert_calib.py
