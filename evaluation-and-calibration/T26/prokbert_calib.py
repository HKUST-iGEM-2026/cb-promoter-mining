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
2. Builds the calibration dataset = 93 T9 positives (label 1) + 186 curated genomic negatives
   (label 0): 130 convergent-intergenic + 56 CDS-interior windows from extract_negatives.py (this
   dir). Counts (--n-convergent / --n-cds) and paths are flags.
3. Tokenises with the ProkBERT tokenizer (attention-masked padding handles variable length), scores on
   GPU, and takes the positive-class logit difference (logits[:,1] - logits[:,0]) as the calibration
   score; AT fraction is the second covariate. (softmax(...)[:,1] is kept only for the raw
   before/after report — the LOGIT, not the squashed prob, is what feeds the calibrator.)
4. Platt scaling WITH the AT covariate, selecting unregularized vs L2 by stratified K-fold
   cross-validation: every sample gets an out-of-fold (OOF) calibrated prediction, the two settings
   are compared on OOF log loss + Brier, the simpler/better one is chosen (L2 must beat unregularized
   by > --tol log loss, else the simpler model wins), then REFIT on the full calibration set. Raw vs
   calibrated (Brier / log-loss / AUROC / AUPRC / ECE) is reported.

NOTE: run on the main `promoter-mining-hpc4` env (transformers 5.3.0) — ProkBERT loads clean there
(unlike NT, which needs the nt-tf4x env). Calibration is prevalence-dependent: this 93:186 (1:2) set
calibrates to that operating point, not genome-wide promoter prevalence.

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


def cv_select(X, y, n_splits, seed, l2_C, tol, n_bins):
    """Pick unregularized vs L2 Platt calibrator by stratified-K-fold out-of-fold (OOF) predictions.

    For each candidate the data is split into `n_splits` stratified folds; the calibrator is fit on
    the training folds and predicts the held-out fold, until every sample has one OOF prediction.
    Candidates are compared on OOF log loss (primary) and Brier score; the L2 model must beat the
    unregularized one by more than `tol` in log loss to be chosen, otherwise the simpler unregularized
    model wins. The chosen calibrator is then REFIT on the full (X, y) and returned ready to persist.

    Each candidate is StandardScaler -> LogisticRegression so the L2 penalty acts on a common scale
    (the scaler is a no-op for the unregularized MLE but keeps the two artifacts identical in form).

    Returns (chosen_name, refit_estimator, oof_prob_of_chosen, report_dict).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def make(name):
        lr = (LogisticRegression(penalty="l2", C=l2_C, max_iter=1000) if name == "l2"
              else LogisticRegression(penalty=None, max_iter=1000))
        return make_pipeline(StandardScaler(), lr)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof, report = {}, {}
    for name in ("unregularized", "l2"):
        p = np.zeros(len(y), dtype=float)
        for tr, te in skf.split(X, y):
            p[te] = make(name).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        oof[name] = p
        report[name] = score_block(y, p, n_bins)   # carries log_loss + brier (+ auroc/auprc/ece)

    ll_u, ll_r = report["unregularized"]["log_loss"], report["l2"]["log_loss"]
    if ll_r < ll_u - tol:
        chosen = "l2"
        reason = f"L2 OOF log_loss {ll_r:.4f} beats unregularized {ll_u:.4f} by > tol {tol}"
    else:
        chosen = "unregularized"
        reason = (f"unregularized chosen (simpler): OOF log_loss {ll_u:.4f} vs L2 {ll_r:.4f} "
                  f"(L2 not better by tol {tol})")
    refit = make(chosen).fit(X, y)
    return chosen, refit, oof[chosen], {"candidates": report, "chosen": chosen, "reason": reason}


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


def _sample_negatives(path: Path, n: int, flag: str, seed: int) -> pd.DataFrame:
    """Sample `n` rows from one extract_negatives.py TSV (label=0). Uses all rows if fewer exist."""
    if not path.is_file():
        die(f"negatives file not found: {path}  ({flag})")
    df = pd.read_csv(path, sep="\t")
    if "sequence" not in df.columns:
        die(f"negatives file {path} has no 'sequence' column; got {list(df.columns)}")
    if len(df) < n:
        print(f"WARNING: only {len(df)} rows in {path}, fewer than the {n} requested — using all of them.")
        sample = df
    else:
        sample = df.sample(n=n, random_state=seed)
    src = sample["negative_type"] if "negative_type" in sample.columns else pd.Series([path.stem] * len(sample))
    return pd.DataFrame({"sequence": sample["sequence"].astype(str).values,
                         "label": 0,
                         "source": src.astype(str).values}).reset_index(drop=True)


def load_negatives(convergent_path: Path, cds_path: Path, n_convergent: int, n_cds: int,
                   seed: int) -> pd.DataFrame:
    """Curated genomic negatives: n_convergent convergent-intergenic + n_cds CDS-interior windows."""
    conv = _sample_negatives(convergent_path, n_convergent, "--convergent-negatives", seed)
    cds = _sample_negatives(cds_path, n_cds, "--cds-negatives", seed + 1)
    return pd.concat([conv, cds], ignore_index=True)


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
    ap.add_argument("--convergent-negatives", type=Path, default=None,
                    help="convergent-intergenic negatives tsv "
                         "(default: <repo>/evaluation-and-calibration/T26/convergent_intergenic_negatives.tsv)")
    ap.add_argument("--cds-negatives", type=Path, default=None,
                    help="CDS-interior negatives tsv "
                         "(default: <repo>/evaluation-and-calibration/T26/cds_negatives.tsv)")
    ap.add_argument("--n-convergent", type=int, default=130,
                    help="convergent-intergenic negatives to sample (default 130)")
    ap.add_argument("--n-cds", type=int, default=56, help="CDS-interior negatives to sample (default 56)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output dir (default: <repo>/evaluation-and-calibration/T26/out/prokbert)")
    ap.add_argument("--max-length", type=int, default=128,
                    help="tokenizer max_length cap in TOKENS (default 128 — safely above the ~52–81 bp "
                         "windows so nothing real is truncated; padding is attention-masked)")
    ap.add_argument("--cv-folds", type=int, default=5, help="stratified CV folds for calibrator selection")
    ap.add_argument("--l2-C", type=float, default=1.0, help="inverse L2 strength for the regularized calibrator")
    ap.add_argument("--tol", type=float, default=0.005,
                    help="L2 must beat unregularized by > tol in OOF log loss to be chosen (else simpler wins)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--ece-bins", type=int, default=10)
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    model_dir = (args.model_dir or repo / "results/T21/prokbert/trial_12/best_model").resolve()
    positives_path = (args.positives or repo / "data-acquisition-and-curation/T9/literature_promoters.tsv").resolve()
    convergent_path = (args.convergent_negatives
                       or repo / "evaluation-and-calibration/T26/convergent_intergenic_negatives.tsv").resolve()
    cds_path = (args.cds_negatives or repo / "evaluation-and-calibration/T26/cds_negatives.tsv").resolve()
    out_dir = (args.out_dir or repo / "evaluation-and-calibration/T26/out/prokbert").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch  # deferred: torch/CUDA init is slow; do it after arg/path validation, not for --help
    import joblib

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- model (full fine-tuned ProkBERT + base tokenizer) ---
    model, tokenizer = load_prokbert_model(args.tokenizer_id, model_dir, device)

    # --- dataset ---
    pos = load_positives(positives_path)
    neg = load_negatives(convergent_path, cds_path, args.n_convergent, args.n_cds, args.seed)
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

    # --- Platt scaling with AT covariate: select unregularized vs L2 by stratified-CV OOF ---
    y = data["label"].values
    X_logit_at = np.column_stack([logit_diff, data["at_fraction"].values])   # [score, AT covariate]
    if int(min(np.bincount(y))) < args.cv_folds:
        die(f"a class has fewer than --cv-folds={args.cv_folds} samples "
            f"(pos={int((y==1).sum())}, neg={int((y==0).sum())}); lower --cv-folds or add data.")
    chosen, calibrator, oof_prob, selection = cv_select(
        X_logit_at, y, args.cv_folds, args.seed, args.l2_C, args.tol, args.ece_bins)

    metrics = {
        "raw": score_block(y, raw_prob, args.ece_bins),
        "cv_unregularized": selection["candidates"]["unregularized"],
        "cv_l2": selection["candidates"]["l2"],
        "calibrated_logit_at_oof": score_block(y, oof_prob, args.ece_bins),  # == chosen candidate's OOF
    }

    # --- write CSV (every row carries its out-of-fold calibrated prob) ---
    data["calib_prob_at"] = oof_prob
    csv_path = out_dir / "prokbert_calibration_predictions.csv"
    data[["sequence", "label", "source", "at_fraction",
          "raw_logit", "raw_prob", "calib_prob_at"]].to_csv(csv_path, index=False)

    # --- save the selected calibrator (refit on the FULL set) + metrics ---
    cal_at_path = out_dir / "prokbert_calibrator_at.joblib"
    joblib.dump(calibrator, cal_at_path)
    metrics_path = out_dir / "prokbert_calibration_metrics.json"
    metrics_path.write_text(json.dumps({
        "device": str(device),
        "n_total": int(len(data)),
        "n_positive": int((data["label"] == 1).sum()),
        "n_negative": int((data["label"] == 0).sum()),
        "n_convergent": args.n_convergent,
        "n_cds": args.n_cds,
        "cv_folds": args.cv_folds,
        "l2_C": args.l2_C,
        "tol": args.tol,
        "chosen": chosen,
        "selection_reason": selection["reason"],
        "seed": args.seed,
        "max_length": args.max_length,
        "model_output_shape": [int(len(data)), out_cols],
        "tokenizer_id": args.tokenizer_id,
        "metrics": metrics,
        "paths": {
            "tokenizer_id": args.tokenizer_id,
            "model_dir": str(model_dir),
            "positives": str(positives_path),
            "convergent_negatives": str(convergent_path),
            "cds_negatives": str(cds_path),
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
          f"neg={int((data['label']==0).sum())}: {args.n_convergent} convergent + {args.n_cds} CDS)")
    print(f"model output     : (batch, {out_cols})")
    print(f"CV               : {args.cv_folds}-fold stratified | chosen: {chosen} ({selection['reason']})")
    print("\nPlatt+AT calibrator selection (out-of-fold):")
    print(f"metric        unregularized     L2(C={args.l2_C:g})")
    for m in ("log_loss", "brier", "auroc", "auprc", "ece"):
        print(f"{m:<12} {selection['candidates']['unregularized'][m]:>12.4f}   "
              f"{selection['candidates']['l2'][m]:>12.4f}")
    print("\nraw vs calibrated (OOF, chosen calibrator):")
    print("metric        raw          cal(logit+AT)")
    for m in ("brier", "log_loss", "auroc", "auprc", "ece"):
        print(f"{m:<12} {metrics['raw'][m]:>8.4f}   {metrics['calibrated_logit_at_oof'][m]:>12.4f}")
    print("\noutputs:")
    for p in (cal_at_path, csv_path, metrics_path):
        print(f"  {p}")


if __name__ == "__main__":
    main()


# --- run on one GPU (HPC4 / SuperPOD) -------------------------------------------------------------
# NB the main promoter-mining-hpc4 env (transformers 5.3.0) — ProkBERT loads clean there (NOT nt-tf4x).
# srun --account=igem2026 --partition=preempt --gpus-per-node=1 --requeue \
#      python evaluation-and-calibration/T26/prokbert_calib.py
