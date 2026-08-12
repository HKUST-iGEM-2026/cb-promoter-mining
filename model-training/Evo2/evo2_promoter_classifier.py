#!/usr/bin/env python3
"""
evo2_promoter_classifier.py — frozen-Evo2-embedding probe (logreg vs random forest).

WHY THIS FILE CHANGED (2026-08):
  1. It reported AUROC only. On this ~22%-positive set AUROC massively over-reads vs the metric the
     rest of the project ranks on — weighted AUPRC (sklearn.average_precision_score). The other four
     models report TEST AUPRC (ProkBERT 0.884 / NT 0.854 / CNN 0.804 / cnn_v2 0.848); an AUROC of
     0.90 is NOT comparable to those and is systematically higher. AUPRC is now the selection + report
     metric; AUROC is kept as a secondary.
  2. It ignored the T15 `weight` column, so the probe trained on a different distribution than
     T18/T19/T20 (silver T13 at full weight = the AT-trap, and no 50/50 class rebalance). Fits now use
     sample_weight, matching the ensemble.
  3. `best = lin` was hard-coded; now auto-picked on WEIGHTED val AUPRC.
Paths are flags with shared defaults (was a personal dir / a top-level pkl).

  python evo2_promoter_classifier.py --emb-dir /project/igem2026/cb-promoter-mining/evo2_embeddings
"""
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import joblib


def load(emb_dir, split):
    X = np.load(f"{emb_dir}/{split}_X.npy")
    y = np.load(f"{emb_dir}/{split}_y.npy")
    try:
        w = np.load(f"{emb_dir}/{split}_w.npy")
    except FileNotFoundError:
        print(f"  WARNING: {split}_w.npy missing — training UNWEIGHTED (re-extract; not comparable to T18-20).")
        w = np.ones(len(y))
    return X, y, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", default="/project/igem2026/cb-promoter-mining/evo2_embeddings")
    ap.add_argument("--out", default="/project/igem2026/cb-promoter-mining/model-training/Evo2/evo2_promoter_classifier.pkl")
    a = ap.parse_args()

    X_tr, y_tr, w_tr = load(a.emb_dir, "train")
    X_val, y_val, w_val = load(a.emb_dir, "val")
    X_te, y_te, w_te = load(a.emb_dir, "test")
    print(f"Train {X_tr.shape} | Val {X_val.shape} | Test {X_te.shape} "
          f"(val pos {int(y_val.sum())}/{len(y_val)}, test pos {int(y_te.sum())}/{len(y_te)})")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "rf": RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
    }
    fit_key = {"logreg": "logisticregression__sample_weight", "rf": "sample_weight"}

    # rough unweighted CV screen on train (NB not cluster-aware; val/test below are the real, weighted read)
    for name, clf in models.items():
        s = cross_val_score(clf, X_tr, y_tr, cv=cv, scoring="average_precision")
        print(f"{name} cv AUPRC (unweighted screen): {s.mean():.4f} ± {s.std():.4f}")

    # real evaluation: fit WEIGHTED on train, pick on WEIGHTED val AUPRC
    print("\n--- validation (weighted fit) ---")
    val = {}
    for name, clf in models.items():
        clf.fit(X_tr, y_tr, **{fit_key[name]: w_tr})
        p = clf.predict_proba(X_val)[:, 1]
        ap_w = average_precision_score(y_val, p, sample_weight=w_val)
        auroc_w = roc_auc_score(y_val, p, sample_weight=w_val)
        val[name] = ap_w
        print(f"{name}: val AUPRC {ap_w:.4f} | val AUROC {auroc_w:.4f}")
        print(classification_report(y_val, clf.predict(X_val), sample_weight=w_val, zero_division=0))

    best_name = max(val, key=val.get)
    print(f"picked '{best_name}' on weighted val AUPRC ({val[best_name]:.4f})")

    # refit best on train+val (weighted), report on the untouched test split
    print("\n--- test (best refit on train+val) ---")
    best = models[best_name]
    Xtv = np.concatenate([X_tr, X_val]); ytv = np.concatenate([y_tr, y_val]); wtv = np.concatenate([w_tr, w_val])
    best.fit(Xtv, ytv, **{fit_key[best_name]: wtv})
    p_te = best.predict_proba(X_te)[:, 1]
    ap_te = average_precision_score(y_te, p_te, sample_weight=w_te)
    auroc_te = roc_auc_score(y_te, p_te, sample_weight=w_te)
    print(classification_report(y_te, best.predict(X_te), sample_weight=w_te, zero_division=0))
    print(f"TEST AUPRC {ap_te:.4f} | TEST AUROC {auroc_te:.4f}   [compare AUPRC to ProkBERT 0.884 / NT 0.854 / CNN 0.804 / cnn_v2 0.848]")

    joblib.dump(best, a.out)
    print(f"saved {best_name} -> {a.out}")


if __name__ == "__main__":
    main()
