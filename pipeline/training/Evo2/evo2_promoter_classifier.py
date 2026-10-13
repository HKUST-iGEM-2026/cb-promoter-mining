#!/usr/bin/env python3
"""
evo2_promoter_classifier.py — frozen-Evo2-embedding probe (logreg vs random forest).

Probe selection uses weighted validation average precision and refits the selected
classifier on training plus validation embeddings. Sample weights are used when
present; missing weights fall back to ones. The quick stratified CV is unweighted
and not cluster-aware, so it is exploratory rather than the primary deployment test.
Historical split/checkpoint comparisons are documented in README.md and the Evo2
report; they are not controlled comparisons with every deployed model.
The backbone remains frozen. Neither classifier scores nor gene neighborhood
establish experimental promoter activity or constitutive expression.

  python evo2_promoter_classifier.py --emb-dir $PROMOTER_REPO/evo2_embeddings
"""
import os
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import joblib

# Portable paths: override with PROMOTER_REPO / PROMOTER_RESULTS (or `source env.sh`).
# The fallbacks resolve from this file, so a plain clone works with no configuration.
PROMOTER_REPO = os.environ.get("PROMOTER_REPO") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PROMOTER_RESULTS = os.environ.get("PROMOTER_RESULTS") or os.path.join(PROMOTER_REPO, "workflow", "results")



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
    ap.add_argument("--emb-dir", default=os.path.join(PROMOTER_REPO, "evo2_embeddings"))
    ap.add_argument("--out", default=os.path.join(PROMOTER_REPO, "pipeline/training/Evo2/evo2_promoter_classifier.pkl"))
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
    print(f"TEST AUPRC {ap_te:.4f} | TEST AUROC {auroc_te:.4f}")
    print("Historical context only: earlier ProkBERT 0.884 / NT 0.854 / CNN 0.804 / cnn_v2 0.848 "
          "use a different split/checkpoint round; this is not a controlled all-model ranking. "
          "The recorded deployed CNN-v2 T15_xs test AUPRC is 0.726; match inputs and weighting before comparison.")

    joblib.dump(best, a.out)
    print(f"saved {best_name} -> {a.out}")


if __name__ == "__main__":
    main()
