#!/usr/bin/env python3
"""
T26 — Logistic (Platt) calibration of the trained multi-encoding CNN-v2 promoter model.

Owner: Ishi.  Fourth sibling of prokbert_calib.py / nt_calib.py / cnn_calib.py in this dir.
CNN-v2 is the multi-encoding fusion CNN (model-training/cnn/train_cnn_v2.py): one-hot + ENAC
positional composition + di/tri-nucleotide frequency branches. The doc flags it as the most
miscalibrated of the four (optimistic raw scores), so a Platt+AT calibrator is needed before any
fixed-threshold use; the rank-percentile ensemble (T29) is calibration-invariant and does not need it.

What it does (identical protocol to cnn_calib.py — only model load + featurisation differ)
------------------------------------------------------------------------------------------
1. Imports MultiEncodingPromoterCNN + build_model_from_config + _featurize_batch straight from
   model-training/cnn/train_cnn_v2.py (by file path — the dir is hyphenated, not importable).
2. Rebuilds the exact architecture from the checkpoint's config.json (active_encodings, conv
   filters, dropout, fusion_dim, enac_window live there — NOT in the tensor shapes, unlike the
   one-branch CNN), then load_state_dict(strict=True) confirms the reconstruction.
3. Builds the SAME calibration dataset as the other three: 93 T9 positives (label 1) + 186 curated
   genomic negatives (label 0) = 130 convergent-intergenic + 56 CDS-interior (extract_negatives.py).
4. Featurises with v2's OWN _featurize_batch (center-pad/crop to --window, one-hot + dinuc + trinuc
   + ENAC) — byte-identical to how T22 / genome_inference score v2 — scores on GPU, and takes the
   positive-class logit difference (logits[:,1] - logits[:,0]) as the calibration score. AT fraction
   (of the ORIGINAL, unpadded sequence) is the second covariate.
5. Platt scaling WITH the AT covariate, selecting unregularized vs L2 by repeated stratified K-fold
   OOF cross-validation (L2 must beat unregularized by > --tol OOF log loss, else the simpler model
   wins), then REFIT on the full set. Raw vs calibrated (Brier/log-loss/AUROC/AUPRC/ECE) is reported.

Outputs (evaluation-and-calibration/calibration/out/cnn_v2/ by default):
  cnn_v2_calibrator_at.joblib · cnn_v2_calibration_predictions.csv · cnn_v2_calibration_metrics.json

NOTE on length: T9 positives are ~51 bp and the genomic negatives 52 bp — neither is the 81 bp v2
was trained on. v2's _featurize_batch center-pads with 'N' / center-crops to --window (the trained
convention), so short windows get a lot of N padding; pass --strict-length to hard-fail instead.
Calibration is prevalence-dependent: this 93:186 (1:2) set calibrates to that operating point, not
genome-wide promoter prevalence.

Run (one GPU): see the sbatch snippet at the bottom of this file.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ helpers
def die(msg: str) -> "NoReturn":
    """Print a clear error and exit non-zero (keeps the traceback out of the user's face)."""
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def import_train_cnn_v2(train_cnn_v2_path: Path):
    """Import the v2 model class + inference helpers by file path (dir is hyphenated)."""
    if not train_cnn_v2_path.is_file():
        die(f"training script not found: {train_cnn_v2_path}\n"
            f"       point --train-cnn-v2 at model-training/cnn/train_cnn_v2.py")
    try:
        spec = importlib.util.spec_from_file_location("t20_train_cnn_v2", train_cnn_v2_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:  # noqa: BLE001 - surface any import-time failure clearly
        die(f"could not import train_cnn_v2.py ({train_cnn_v2_path}): {e}")
    for name in ("MultiEncodingPromoterCNN", "build_model_from_config", "_featurize_batch"):
        if not hasattr(module, name):
            die(f"{name} not found in {train_cnn_v2_path} — wrong file or the API changed.")
    return module


def at_fraction(seq: str) -> float:
    seq = seq.upper()
    return (seq.count("A") + seq.count("T")) / max(len(seq), 1)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width-binning ECE = sum_b (|bin|/N) * |acc_b - conf_b|."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
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
    """Pick unregularized vs L2 Platt calibrator by REPEATED stratified-K-fold out-of-fold preds.

    Identical selection logic to the other three T26 scripts: for each candidate, `n_repeats`
    stratified K-fold passes (seeds seed..seed+R-1) give every sample a held-out prediction; the
    per-pass OOF metrics are averaged; L2 must beat unregularized by > `tol` in mean OOF log loss to
    be chosen, else the simpler model wins; the chosen calibrator is REFIT on all (X, y).
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
    """Strict ACGT-only check (always raises on non-ACGT). Length is harmonised to `window` by v2's
    _featurize_batch (center-pad with 'N' / center-crop) unless --strict-length."""
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
            f"Re-window to exactly {window} bp or drop --strict-length to center-pad/crop.")
    if n_short or n_long:
        print(f"NOTE: harmonising to {window} bp — exact:{n_exact}  padded(<{window}):{n_short}  "
              f"cropped(>{window}):{n_long}  (v2 _featurize_batch: center-pad 'N' / center-crop)")
    else:
        print(f"All {n_exact} sequences are exactly {window} bp.")


# ------------------------------------------------------------------ main
def main() -> None:
    repo_default = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="T26 CNN-v2 (multi-encoding) Platt calibration")
    ap.add_argument("--repo-root", type=Path, default=repo_default)
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="dir with best_model.pt + config.json "
                         "(default: <repo>/infrastructure/T3/snakemake_scaffold/results/T21_final_06/cnn_v2 "
                         "— the 0.6-clustering final retrain)")
    ap.add_argument("--train-cnn-v2", type=Path, default=None,
                    help="path to train_cnn_v2.py (default: <repo>/model-training/cnn/train_cnn_v2.py)")
    ap.add_argument("--positives", type=Path, default=None,
                    help="T9 positives tsv (default: <repo>/data-acquisition-and-curation/synthetic-promoters/literature_promoters.tsv)")
    ap.add_argument("--convergent-negatives", type=Path, default=None,
                    help="convergent-intergenic negatives tsv "
                         "(default: <repo>/evaluation-and-calibration/calibration/convergent_intergenic_negatives.tsv)")
    ap.add_argument("--cds-negatives", type=Path, default=None,
                    help="CDS-interior negatives tsv "
                         "(default: <repo>/evaluation-and-calibration/calibration/cds_negatives.tsv)")
    ap.add_argument("--n-convergent", type=int, default=130,
                    help="convergent-intergenic negatives to sample (default 130)")
    ap.add_argument("--n-cds", type=int, default=56, help="CDS-interior negatives to sample (default 56)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output dir (default: <repo>/evaluation-and-calibration/calibration/out/cnn_v2)")
    ap.add_argument("--window", type=int, default=81)
    ap.add_argument("--enac-window", type=int, default=None,
                    help="ENAC sliding window (default: read from config.json, else 5)")
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
                    help="hard-fail on any sequence != --window bp (default: center-pad/crop via v2 featurizer)")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    model_dir = (args.model_dir
                 or repo / "infrastructure/T3/snakemake_scaffold/results/T21_final_06/cnn_v2").resolve()
    train_cnn_v2_path = (args.train_cnn_v2 or repo / "model-training/cnn/train_cnn_v2.py").resolve()
    positives_path = (args.positives or repo / "data-acquisition-and-curation/synthetic-promoters/literature_promoters.tsv").resolve()
    convergent_path = (args.convergent_negatives
                       or repo / "evaluation-and-calibration/calibration/convergent_intergenic_negatives.tsv").resolve()
    cds_path = (args.cds_negatives or repo / "evaluation-and-calibration/calibration/cds_negatives.tsv").resolve()
    out_dir = (args.out_dir or repo / "evaluation-and-calibration/calibration/out/cnn_v2").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch  # deferred: torch/CUDA init is slow; do it after arg/path validation, not for --help
    import joblib

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- v2 model class + inference helpers ---
    v2 = import_train_cnn_v2(train_cnn_v2_path)

    # --- config.json (architecture: active_encodings, conv filters, dropout, fusion_dim, enac_window) ---
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        die(f"config.json not found: {config_path}  (--model-dir)\n"
            f"       train_cnn_v2.py writes config.json next to best_model.pt; point --model-dir at that dir.")
    try:
        cfg = json.loads(config_path.read_text())
    except json.JSONDecodeError as e:
        die(f"config.json is not valid JSON ({config_path}): {e}")
    enac_window = int(args.enac_window if args.enac_window is not None else cfg.get("enac_window", 5))

    # --- checkpoint ---
    ckpt_path = model_dir / "best_model.pt"
    if not ckpt_path.is_file():
        die(f"checkpoint not found: {ckpt_path}  (--model-dir)")
    try:
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception as e:  # noqa: BLE001 - corrupt/unreadable/pickled-model file
        die(f"could not read checkpoint {ckpt_path}: {e}\n"
            f"       (expected a state_dict saved by torch.save(model.state_dict(), ...))")
    if not isinstance(state_dict, dict) or not all(isinstance(k, str) for k in state_dict):
        die(f"{ckpt_path} is not a state_dict (got {type(state_dict).__name__}); "
            f"this script expects model.state_dict(), not a pickled model.")

    model = v2.build_model_from_config(cfg, device=str(device))
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        die(f"checkpoint/architecture mismatch loading {ckpt_path} into MultiEncodingPromoterCNN "
            f"(encodings={sorted(model.active_encodings)}): {e}\n"
            f"       config.json must match the checkpoint (active_encodings + conv filters + fusion_dim).")
    model.to(device).eval()
    active_encodings = sorted(model.active_encodings)

    # --- dataset (identical to the other three T26 scripts) ---
    pos = load_positives(positives_path)
    neg = load_negatives(convergent_path, cds_path, args.n_convergent, args.n_cds, args.seed)
    data = pd.concat([pos, neg], ignore_index=True)
    data["at_fraction"] = data["sequence"].map(at_fraction)  # AT of the ORIGINAL (unpadded) seq

    if data["label"].nunique() < 2:
        die(f"calibration dataset has only one class (labels present: {sorted(data['label'].unique())}). "
            f"Need both positives and negatives.")
    validate_and_report_lengths(data["sequence"], args.window, args.strict_length)

    # --- v2 multi-encoding featurisation + batched GPU inference (raw logits) ---
    seqs = [s.upper() for s in data["sequence"].tolist()]
    logit0, logit1 = [], []
    out_cols = None
    with torch.no_grad():
        for i in range(0, len(seqs), args.batch_size):
            batch = seqs[i:i + args.batch_size]
            x1, x2, x3, x4 = v2._featurize_batch(batch, enac_window, window=args.window)
            out = model(x1.to(device), x2.to(device), x3.to(device), x4.to(device))
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
    csv_path = out_dir / "cnn_v2_calibration_predictions.csv"
    data[["sequence", "label", "source", "at_fraction",
          "raw_logit", "raw_prob", "calib_prob_at"]].to_csv(csv_path, index=False)

    # --- save the selected calibrator (refit on the FULL set) + metrics ---
    cal_at_path = out_dir / "cnn_v2_calibrator_at.joblib"
    joblib.dump(calibrator, cal_at_path)
    metrics_path = out_dir / "cnn_v2_calibration_metrics.json"
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
        "enac_window": enac_window,
        "active_encodings": active_encodings,
        "model_output_shape": [int(len(data)), out_cols],
        "architecture": {
            "active_encodings": active_encodings,
            "conv1_filters": int(cfg.get("conv1_filters", 64)),
            "conv2_filters": int(cfg.get("conv2_filters", 128)),
            "conv3_filters": int(cfg.get("conv3_filters", 256)),
            "fusion_dim": int(cfg.get("fusion_dim", 256)),
            "dropout_rate": float(cfg.get("dropout_rate", 0.3)),
            "use_batch_norm": bool(cfg.get("use_batch_norm", True)),
        },
        "metrics": metrics,
        "paths": {
            "model_class": v2.__file__,
            "checkpoint": str(ckpt_path),
            "config_json": str(config_path),
            "positives": str(positives_path),
            "convergent_negatives": str(convergent_path),
            "cds_negatives": str(cds_path),
            "calibrator_at": str(cal_at_path),
            "predictions_csv": str(csv_path),
            "metrics_json": str(metrics_path),
        },
    }, indent=2))

    # --- diagnostics ---
    print("\n===== T26 CNN-v2 calibration diagnostics =====")
    print(f"model class      : MultiEncodingPromoterCNN from {v2.__file__}")
    print(f"checkpoint       : {ckpt_path}")
    print(f"active encodings : {active_encodings}  (enac_window={enac_window}, dropout={cfg.get('dropout_rate')})")
    print(f"device           : {device}")
    print(f"sequences        : {len(data)}  (pos={int((data['label']==1).sum())}, "
          f"neg={int((data['label']==0).sum())}: {args.n_convergent} convergent + {args.n_cds} CDS)")
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


# --- run on one GPU (Slurm) -------------------------------------------------------------
# srun --gpus-per-node=1 --requeue \
#      python evaluation-and-calibration/calibration/cnn_v2_calib.py \
#        --model-dir infrastructure/T3/snakemake_scaffold/results/T21_final_06/cnn_v2
