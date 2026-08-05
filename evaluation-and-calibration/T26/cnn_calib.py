#!/usr/bin/env python3
"""
T26 — Logistic (Platt) calibration of the trained custom CNN promoter model (T21 trial 12).

Owner: Ishi.  Companion to prokbert_calib.py / cnn_calib.py in this dir; this is the hardened,
runnable version.

What it does
------------
1. Imports PromoterCNN straight from model-training/T20/train_cnn.py (by file path — the dir is
   hyphenated and not importable as a package).
2. Loads results/T21/cnn/trial_12/params.json (the SWEPT hyperparams: learning-rate, batch-size,
   dropout-rate) and reconstructs the *architecture* by INFERRING it from the checkpoint tensor
   shapes — because the CNN sweep never varied conv filters / kernel_size / batch-norm, params.json
   does NOT carry them (train_cnn.py defaults were used). load_state_dict(strict=True) then confirms
   the reconstruction is exact.
3. Builds the calibration dataset = 93 T9 positives (label 1) + 186 curated genomic negatives
   (label 0): 130 convergent-intergenic + 56 CDS-interior windows from extract_negatives.py (this
   dir). Counts (--n-convergent / --n-cds) and paths are flags.
4. One-hot encodes with T16's EXACT semantics (A,C,G,T = 0,1,2,3; short seq -> tail-zero, long ->
   crop, N -> zero column), scores on GPU, and takes the positive-class logit difference
   (logits[:,1] - logits[:,0]) as the calibration score; AT fraction is the second covariate.
5. Platt scaling WITH the AT covariate, selecting unregularized vs L2 by repeated stratified K-fold
   cross-validation: every sample gets an out-of-fold (OOF) calibrated prediction, the two settings
   are compared on OOF log loss + Brier, the simpler/better one is chosen (L2 must beat unregularized
   by > --tol log loss, else the simpler model wins), then REFIT on the full calibration set. Raw vs
   calibrated (Brier / log-loss / AUROC / AUPRC / ECE) is reported.

NOTE on data length: T9 positives are ~52 bp and the genomic negatives are 52 bp — neither is the
81 bp the CNN was trained on. They are harmonised to 81 bp (pad/crop) via the T16 one-hot; pass
--strict-length to hard-fail instead. Calibration is prevalence-dependent: this 93:186 (1:2) set
calibrates to that operating point, not genome-wide promoter prevalence.

Run (one GPU on HPC4): see the sbatch snippet at the bottom of this file.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd


# T16 one-hot channel order (dataset-construction/T16/tokenise.py). DO NOT reorder.
BASE = {"A": 0, "C": 1, "G": 2, "T": 3}


# ------------------------------------------------------------------ helpers
def die(msg: str) -> "NoReturn":
    """Print a clear error and exit non-zero (keeps the traceback out of the user's face)."""
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def import_promoter_cnn(train_cnn_path: Path):
    """Import PromoterCNN by file path (model-training/T20 is hyphenated -> not importable normally)."""
    if not train_cnn_path.is_file():
        die(f"training script not found: {train_cnn_path}\n"
            f"       point --train-cnn at model-training/T20/train_cnn.py")
    try:
        spec = importlib.util.spec_from_file_location("t20_train_cnn", train_cnn_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:  # noqa: BLE001 - surface any import-time failure clearly
        die(f"could not import train_cnn.py ({train_cnn_path}): {e}")
    if not hasattr(module, "PromoterCNN"):
        die(f"PromoterCNN not found in {train_cnn_path} — wrong file or the class was renamed.")
    return module.PromoterCNN, module.__file__


def infer_architecture(state_dict: dict) -> dict:
    """Recover the PromoterCNN architecture straight from the checkpoint tensor shapes.

    params.json only holds swept hyperparams (lr/batch/dropout), so the architecture must come from
    the weights. Layout (train_cnn.py): features[0/5/10]=Conv1d, features[1]=BatchNorm|Identity,
    classifier[1]=Linear(conv3,128), classifier[4]=Linear(128,2).
    """
    try:
        w0 = state_dict["features.0.weight"]    # (conv1, in_ch, kernel)
        w5 = state_dict["features.5.weight"]    # (conv2, conv1, 5)
        w10 = state_dict["features.10.weight"]  # (conv3, conv2, 3)
        head = state_dict["classifier.4.weight"]  # (out_classes, 128)
    except KeyError as e:
        die(f"checkpoint is missing expected CNN layer {e} — not a PromoterCNN state_dict "
            f"(or the architecture changed).")
    out_classes = int(head.shape[0])
    if out_classes != 2:
        die(f"checkpoint head outputs {out_classes} classes, expected 2 (binary logits).")
    return {
        "input_channels": int(w0.shape[1]),
        "conv1_filters": int(w0.shape[0]),
        "conv2_filters": int(w5.shape[0]),
        "conv3_filters": int(w10.shape[0]),
        "kernel_size": int(w0.shape[2]),
        "use_batch_norm": "features.1.weight" in state_dict,
    }


def at_fraction(seq: str) -> float:
    seq = seq.upper()
    return (seq.count("A") + seq.count("T")) / max(len(seq), 1)


def one_hot(seq: str, length: int) -> np.ndarray:
    """(length, 4) one-hot in A,C,G,T order — identical to T16: short -> tail-zero, long -> crop,
    N/unknown -> all-zero row."""
    a = np.zeros((length, 4), dtype=np.float32)
    for i, b in enumerate(seq[:length]):
        j = BASE.get(b)
        if j is not None:
            a[i, j] = 1.0
    return a


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


def cv_select(X, y, feature_sets, n_splits, seed, l2_C, tol, n_bins, n_repeats):
    """Pick unregularized vs L2 Platt calibrator by REPEATED stratified-K-fold out-of-fold (OOF) preds.

    For each candidate, `n_repeats` independent stratified K-fold passes (seeds seed, seed+1, ...) each
    give every sample one held-out prediction; the per-pass OOF metrics are averaged (mean +/- std) so
    the choice does not hinge on a single fold split. Candidates are compared on MEAN OOF log loss
    (primary) and Brier; the L2 model must beat unregularized by more than `tol` in mean log loss to be
    chosen, otherwise the simpler unregularized model wins. The chosen calibrator is REFIT on all (X, y).

    Each candidate is StandardScaler -> LogisticRegression so the L2 penalty acts on a common scale
    (a no-op for the unregularized MLE, but keeps both artifacts identical in form). The returned OOF
    probabilities are averaged across the repeats.

    Returns (chosen_name, chosen_cols, refit_estimator, mean_oof_prob_of_chosen, report_dict).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def make(penalty):
        lr = (LogisticRegression(penalty="l2", C=l2_C, max_iter=1000) if penalty == "l2"
              else LogisticRegression(penalty=None, max_iter=1000))
        return make_pipeline(StandardScaler(), lr)  # StandardScaler -> L2 penalises STANDARDIZED coefs

    order, cand_meta = [], {}
    for fname, cols in feature_sets:                # simplest -> richest, then unreg before L2
        for penalty in ("unregularized", "l2"):
            name = f"{fname} / {penalty}"
            order.append(name)
            cand_meta[name] = (tuple(cols), penalty)

    oof, report = {}, {}
    for name in order:
        cols, penalty = cand_meta[name]
        Xc = X[:, list(cols)]
        oof_sum = np.zeros(len(y), dtype=float)
        per_pass = []
        for r in range(n_repeats):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
            p = np.zeros(len(y), dtype=float)
            for tr, te in skf.split(Xc, y):
                p[te] = make(penalty).fit(Xc[tr], y[tr]).predict_proba(Xc[te])[:, 1]
            per_pass.append(score_block(y, p, n_bins))
            oof_sum += p
        oof[name] = oof_sum / n_repeats             # mean OOF prob across repeats
        agg = {}
        for m in per_pass[0]:                       # mean +/- std of each metric across repeats
            vals = [d[m] for d in per_pass]
            agg[m] = float(np.mean(vals))
            agg[m + "_std"] = float(np.std(vals))
        agg["features"] = [int(c) for c in cols]
        agg["penalty"] = penalty
        report[name] = agg

    # simpler-wins walk: adopt a richer candidate only if it beats the incumbent by > tol OOF log loss
    incumbent, steps = order[0], []
    for name in order[1:]:
        ll_inc, ll_cand = report[incumbent]["log_loss"], report[name]["log_loss"]
        if ll_cand < ll_inc - tol:
            steps.append(f"{name} {ll_cand:.4f} beats {incumbent} {ll_inc:.4f} by > tol {tol} -> adopt")
            incumbent = name
        else:
            steps.append(f"{name} {ll_cand:.4f} not better than {incumbent} {ll_inc:.4f} by tol {tol} -> keep simpler")
    chosen = incumbent
    chosen_cols = list(cand_meta[chosen][0])
    refit = make(cand_meta[chosen][1]).fit(X[:, chosen_cols], y)
    return (chosen, chosen_cols, refit, oof[chosen],
            {"candidates": report, "chosen": chosen, "reason": "; ".join(steps),
             "order": order, "n_repeats": n_repeats, "n_splits": n_splits})


# ------------------------------------------------------------------ data loading
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


def validate_and_report_lengths(seqs: pd.Series, window: int, strict: bool) -> None:
    """Strict ACGT-only check (always raises on non-ACGT). Length is harmonised to `window` by the
    one-hot unless --strict-length, in which case any seq != window raises."""
    upper = seqs.str.upper()
    bad = upper.str.contains(r"[^ACGT]", regex=True)
    if bad.any():
        examples = upper[bad].head(3).tolist()
        die(f"{int(bad.sum())} sequence(s) contain non-ACGT characters (e.g. {examples}). "
            f"Clean or remove them before calibrating.")
    lengths = upper.str.len()
    n_exact = int((lengths == window).sum())
    n_short = int((lengths < window).sum())
    n_long = int((lengths > window).sum())
    if strict and (n_short or n_long):
        die(f"--strict-length: {n_short} seq(s) < {window} bp and {n_long} > {window} bp. "
            f"Re-window to exactly {window} bp or drop --strict-length to pad/crop.")
    if n_short or n_long:
        print(f"NOTE: harmonising to {window} bp — exact:{n_exact}  padded(<{window}):{n_short}  "
              f"cropped(>{window}):{n_long}  (T16 one-hot: short->tail-zero, long->first-{window} crop)")
    else:
        print(f"All {n_exact} sequences are exactly {window} bp.")


# ------------------------------------------------------------------ main
def main() -> None:
    repo_default = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="T26 CNN Platt calibration (T21 trial 12)")
    ap.add_argument("--repo-root", type=Path, default=repo_default)
    ap.add_argument("--trial-dir", type=Path, default=None,
                    help="dir with best_model.pt + params.json (default: <repo>/results/T21/cnn/trial_12)")
    ap.add_argument("--train-cnn", type=Path, default=None,
                    help="path to train_cnn.py (default: <repo>/model-training/T20/train_cnn.py)")
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
                    help="output dir (default: <repo>/evaluation-and-calibration/T26/out/cnn)")
    ap.add_argument("--window", type=int, default=81)
    ap.add_argument("--cv-folds", type=int, default=5, help="stratified CV folds for calibrator selection")
    ap.add_argument("--cv-repeats", type=int, default=5,
                    help="repeated-CV passes with seeds seed..seed+R-1; decision uses mean OOF (1 = single split)")
    ap.add_argument("--l2-C", type=float, default=1.0, help="inverse L2 strength for the regularized calibrator")
    ap.add_argument("--tol", type=float, default=0.005,
                    help="L2 must beat unregularized by > tol in OOF log loss to be chosen (else simpler wins)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--ece-bins", type=int, default=10)
    ap.add_argument("--strict-length", action="store_true",
                    help="hard-fail on any sequence != --window bp (default: pad/crop via T16 one-hot)")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    trial_dir = (args.trial_dir or repo / "results/T21/cnn/trial_12").resolve()
    train_cnn_path = (args.train_cnn or repo / "model-training/T20/train_cnn.py").resolve()
    positives_path = (args.positives or repo / "data-acquisition-and-curation/T9/literature_promoters.tsv").resolve()
    convergent_path = (args.convergent_negatives
                       or repo / "evaluation-and-calibration/T26/convergent_intergenic_negatives.tsv").resolve()
    cds_path = (args.cds_negatives or repo / "evaluation-and-calibration/T26/cds_negatives.tsv").resolve()
    out_dir = (args.out_dir or repo / "evaluation-and-calibration/T26/out/cnn").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch  # deferred: torch/CUDA init is slow; do it after arg/path validation, not for --help
    from torch.utils.data import DataLoader, TensorDataset
    import joblib

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- model class ---
    PromoterCNN, model_class_path = import_promoter_cnn(train_cnn_path)

    # --- params.json (swept hyperparams only) ---
    params_path = trial_dir / "params.json"
    if not params_path.is_file():
        die(f"params.json not found: {params_path}  (--trial-dir)")
    try:
        params = json.loads(params_path.read_text())
    except json.JSONDecodeError as e:
        die(f"params.json is not valid JSON ({params_path}): {e}")
    dropout = float(params.get("dropout-rate", params.get("dropout_rate", 0.3)))

    # --- checkpoint ---
    ckpt_path = trial_dir / "best_model.pt"
    if not ckpt_path.is_file():
        die(f"checkpoint not found: {ckpt_path}  (--trial-dir)")
    try:
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception as e:  # noqa: BLE001 - corrupt/unreadable/pickled-model file
        die(f"could not read checkpoint {ckpt_path}: {e}\n"
            f"       (expected a state_dict saved by torch.save(model.state_dict(), ...))")
    if not isinstance(state_dict, dict) or not all(isinstance(k, str) for k in state_dict):
        die(f"{ckpt_path} is not a state_dict (got {type(state_dict).__name__}); "
            f"this script expects model.state_dict(), not a pickled model.")

    arch = infer_architecture(state_dict)
    model = PromoterCNN(sequence_length=args.window, dropout_rate=dropout, **arch)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        die(f"checkpoint/architecture mismatch loading {ckpt_path} into PromoterCNN{arch}: {e}")
    model.to(device).eval()

    # --- dataset ---
    pos = load_positives(positives_path)
    neg = load_negatives(convergent_path, cds_path, args.n_convergent, args.n_cds, args.seed)
    data = pd.concat([pos, neg], ignore_index=True)
    data["at_fraction"] = data["sequence"].map(at_fraction)

    if data["label"].nunique() < 2:
        die(f"calibration dataset has only one class (labels present: {sorted(data['label'].unique())}). "
            f"Need both positives and negatives.")
    validate_and_report_lengths(data["sequence"], args.window, args.strict_length)

    # --- one-hot + batched GPU inference ---
    X = np.stack([one_hot(s.upper(), args.window) for s in data["sequence"]])  # (N, L, 4)
    X = np.transpose(X, (0, 2, 1))                                             # (N, 4, L) channels-first
    input_shape = tuple(X.shape)
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=args.batch_size, shuffle=False)

    logit0, logit1 = [], []
    out_cols = None
    with torch.no_grad():
        for (xb,) in loader:
            out = model(xb.to(device))
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
    X_all = np.column_stack([logit_diff, data["at_fraction"].values])   # col 0 = score (logit), col 1 = AT
    feature_sets = [("score", [0]), ("score+AT", [0, 1])]               # ordered simplest -> richest
    if int(min(np.bincount(y))) < args.cv_folds:
        die(f"a class has fewer than --cv-folds={args.cv_folds} samples "
            f"(pos={int((y==1).sum())}, neg={int((y==0).sum())}); lower --cv-folds or add data.")
    chosen, chosen_cols, calibrator, oof_prob, selection = cv_select(
        X_all, y, feature_sets, args.cv_folds, args.seed, args.l2_C, args.tol, args.ece_bins, args.cv_repeats)
    chosen_feature_names = [["logit", "at_fraction"][c] for c in chosen_cols]

    metrics = {
        "raw": score_block(y, raw_prob, args.ece_bins),
        "candidates": selection["candidates"],
        "calibrated_oof": score_block(y, oof_prob, args.ece_bins),           # chosen calibrator's OOF (deployed)
        "calibrated_logit_at_oof": score_block(y, oof_prob, args.ece_bins),  # back-compat alias == calibrated_oof
    }

    # --- write CSV (every row carries its out-of-fold calibrated prob) ---
    data["calib_prob_at"] = oof_prob
    csv_path = out_dir / "cnn_calibration_predictions.csv"
    data[["sequence", "label", "source", "at_fraction",
          "raw_logit", "raw_prob", "calib_prob_at"]].to_csv(csv_path, index=False)

    # --- save the selected calibrator (refit on the FULL set) + metrics ---
    cal_at_path = out_dir / "cnn_calibrator_at.joblib"
    joblib.dump(calibrator, cal_at_path)
    metrics_path = out_dir / "cnn_calibration_metrics.json"
    metrics_path.write_text(json.dumps({
        "device": str(device),
        "n_total": int(len(data)),
        "n_positive": int((data["label"] == 1).sum()),
        "n_negative": int((data["label"] == 0).sum()),
        "n_convergent": args.n_convergent,
        "n_cds": args.n_cds,
        "cv_folds": args.cv_folds,
        "cv_repeats": args.cv_repeats,
        "l2_C": args.l2_C,
        "tol": args.tol,
        "chosen": chosen,
        "chosen_features": chosen_feature_names,
        "selection_reason": selection["reason"],
        "seed": args.seed,
        "window": args.window,
        "input_tensor_shape": list(input_shape),
        "model_output_shape": [int(len(data)), out_cols],
        "architecture": arch,
        "dropout_rate": dropout,
        "metrics": metrics,
        "paths": {
            "model_class": model_class_path,
            "checkpoint": str(ckpt_path),
            "params_json": str(params_path),
            "positives": str(positives_path),
            "convergent_negatives": str(convergent_path),
            "cds_negatives": str(cds_path),
            "calibrator_at": str(cal_at_path),
            "predictions_csv": str(csv_path),
            "metrics_json": str(metrics_path),
        },
    }, indent=2))

    # --- diagnostics ---
    print("\n===== T26 CNN calibration diagnostics =====")
    print(f"model class      : PromoterCNN from {model_class_path}")
    print(f"checkpoint       : {ckpt_path}")
    print(f"architecture     : {arch}  (dropout={dropout})")
    print(f"device           : {device}")
    print(f"sequences        : {len(data)}  (pos={int((data['label']==1).sum())}, "
          f"neg={int((data['label']==0).sum())}: {args.n_convergent} convergent + {args.n_cds} CDS)")
    print(f"input tensor     : {input_shape}")
    print(f"model output     : (batch, {out_cols})")
    print(f"CV               : {args.cv_folds}-fold x {args.cv_repeats} repeats stratified")
    print(f"chosen           : {chosen}  (features={chosen_feature_names})")
    print(f"selection        : {selection['reason']}")
    print("\ncandidate selection (out-of-fold, mean over repeats):")
    print(f"{'candidate':<26}{'log_loss':>10}{'brier':>9}{'auroc':>9}{'auprc':>9}{'ece':>9}")
    for name in selection["order"]:
        c = selection["candidates"][name]
        mark = "  <-- chosen" if name == chosen else ""
        print(f"{name:<26}{c['log_loss']:>10.4f}{c['brier']:>9.4f}{c['auroc']:>9.4f}"
              f"{c['auprc']:>9.4f}{c['ece']:>9.4f}{mark}")
    print("\nraw vs calibrated (OOF, chosen calibrator):")
    print("metric        raw          calibrated")
    for m in ("brier", "log_loss", "auroc", "auprc", "ece"):
        print(f"{m:<12} {metrics['raw'][m]:>8.4f}   {metrics['calibrated_oof'][m]:>12.4f}")
    print("\noutputs:")
    for p in (cal_at_path, csv_path, metrics_path):
        print(f"  {p}")


if __name__ == "__main__":
    main()


# --- run on one GPU (HPC4 / SuperPOD) -------------------------------------------------------------
# srun --account=igem2026 --partition=preempt --gpus-per-node=1 --requeue \
#      python evaluation-and-calibration/T26/calibrate_cnn.py
