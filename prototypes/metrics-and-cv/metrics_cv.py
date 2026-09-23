"""
PT3: Metrics + Cross Validation stack on dummy classifier
Owner: Seonyu
Due: 5 Jun

Reusable metric + CV harness for T21 / T22 / T23. Validated against a dummy
classifier whose metrics are known in advance (see validate_known_answers).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)
import logging
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HarnessConfig:
    n_samples: int = 2000
    n_features: int = 15
    positive_ratio: float = 0.05  # 5% promoters
    n_splits: int = 5
    seed: int = 42
    output_dir: str = "results/pt3_metrics"
    dummy_strategy: str = "most_frequent"

def compute_evaluation_stack(y_true, y_pred, y_proba):
    # AUPRC: use average_precision_score. For a no-skill classifier this
    # correctly returns the positive prevalence (0.05 here) — that is the
    # value you should see. Do NOT use auc(recall, precision): it interpolates
    # the PR curve and would wrongly inflate that 0.05 to (1+prevalence)/2 = 0.525.
    # accuracy is reported for reference only — on imbalanced data a do-nothing
    # classifier scores ~1-prevalence, so judge models by AUPRC/MCC, not this.
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auroc": roc_auc_score(y_true, y_proba),
        "auprc": average_precision_score(y_true, y_proba),
    }

def validate_known_answers(mean_results, prevalence, strategy):
    """Sanity-check the metric stack against a baseline whose answer we know.

    For the 'most_frequent' dummy (constant prediction, constant probability):
        accuracy = 1 - prevalence,  F1 = 0,  MCC = 0,  AUROC = 0.5,
        AUPRC = prevalence.
    A failing assert means the *metric implementation* is wrong, not the model.
    """
    if strategy != "most_frequent":
        # Known answers below only hold for the constant 'most_frequent'
        # baseline; for other strategies skip the checks rather than crash.
        logger.warning(
            f"Skipping known-answer checks: no known values defined for "
            f"strategy '{strategy}' (only 'most_frequent' is validated)."
        )
        return
    checks = {
        "accuracy": (mean_results["accuracy"], 1.0 - prevalence, 0.02),
        "f1_score": (mean_results["f1_score"], 0.0, 1e-6),
        "mcc": (mean_results["mcc"], 0.0, 1e-6),
        "auroc": (mean_results["auroc"], 0.5, 1e-6),
        "auprc": (mean_results["auprc"], prevalence, 0.02),
    }
    for name, (got, expected, tol) in checks.items():
        assert np.isclose(got, expected, atol=tol), (
            f"{name}: got {got:.4f}, expected ~{expected:.4f} (tol {tol}) — "
            "metric implementation is wrong"
        )
    logger.info("All known-answer checks passed.")

def run_cv_harness(config: HarnessConfig):
    np.random.seed(config.seed)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Generate random data and imbalanced labels
    X = np.random.randn(config.n_samples, config.n_features)
    y = np.zeros(config.n_samples)
    n_positives = int(config.n_samples * config.positive_ratio)
    y[:n_positives] = 1
    np.random.shuffle(y)
    prevalence = float(y.mean())

    logger.info(f"Data generated. Promoters: {int(y.sum())}, Non-promoters: {int((y==0).sum())}")

    # NOTE for T23: this stratifies by label ONLY. T23 (cluster-level 5-fold CV)
    # needs cluster-aware splitting so near-identical CD-HIT clusters (from T15)
    # never span train and test — otherwise scores are inflated by leakage.
    # Swap this for StratifiedGroupKFold keyed on cluster IDs; the metric stack
    # (compute_evaluation_stack) carries over unchanged.
    cv = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    model = DummyClassifier(strategy=config.dummy_strategy, random_state=config.seed)
    fold_metrics = []

    logger.info(f"Running {config.n_splits}-Fold CV using baseline strategy: {config.dummy_strategy}")
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = compute_evaluation_stack(y_test, y_pred, y_proba)
        metrics["fold"] = fold + 1
        fold_metrics.append(metrics)

    df_results = pd.DataFrame(fold_metrics)
    mean_results = df_results.drop(columns="fold").mean().to_dict()

    logger.info("Baseline Summary Metrics:")
    for metric, score in mean_results.items():
        logger.info(f"  - Mean {metric.upper()}: {score:.4f}")

    # The actual point of PT3: confirm the metrics match the known baseline.
    validate_known_answers(mean_results, prevalence, config.dummy_strategy)

    df_results.to_csv(f"{config.output_dir}/pt3_cv_results.csv", index=False)

def main():
    config = HarnessConfig()
    run_cv_harness(config)

if __name__ == "__main__":
    main()
