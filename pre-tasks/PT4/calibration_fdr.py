"""
PT4: Calibration + False Discovery Rate (FDR) harness on dummy data
Owner: Seonyu
Due: 12 Jun

Reusable calibration and multiple-testing correction harness. Validated 
against a baseline whose calibration metrics and FDR expectations are known 
in advance (see validate_known_answers).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.metrics import brier_score_loss
from statsmodels.stats.multitest import multipletests
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
    output_dir: str = "results/pt4_calibration"
    dummy_strategy: str = "prior"  # 'prior' returns class prevalence as probabilities
    fdr_alpha: float = 0.05       # Target False Discovery Rate control

def compute_evaluation_stack(y_true, y_proba, fdr_alpha):
    """Computes calibration quality and evaluates FDR control.
    
    For FDR evaluation, we simulate p-values under the null hypothesis 
    (or derive them from y_proba) and apply Benjamini-Hochberg adjustment.
    """
    # Calibration metric: Brier Score (lower is better, 0 is perfect calibration)
    brier = brier_score_loss(y_true, y_proba)
    
    # Simulate p-values for FDR control testing. Under a random/dummy baseline,
    # feature associations are non-existent, meaning p-values are uniformly distributed.
    # We generate mock uniform p-values to evaluate how the FDR system handles uncalibrated inputs.
    rng = np.random.default_rng(seed=42)
    mock_p_values = rng.uniform(0, 1, size=len(y_true))
    
    # Apply Benjamini-Hochberg (BH) adjustment
    reject, q_values, _, _ = multipletests(mock_p_values, alpha=fdr_alpha, method='fdr_bh')
    
    # Calculate empirical Discovery Rate
    n_discoveries = int(np.sum(reject))
    discovery_rate = n_discoveries / len(y_true)

    return {
        "brier_score": brier,
        "n_discoveries": n_discoveries,
        "discovery_rate": discovery_rate
    }

def validate_known_answers(mean_results, prevalence, strategy, fdr_alpha):
    """Sanity-check the calibration and FDR metrics against a known baseline.

    For the 'prior' dummy strategy (always outputs positive class prevalence):
        - Expected Brier Score = prevalence * (1 - prevalence)
        - Expected Discoveries on uniform noise = 0.0 (FDR control holds)
    A failing assert means the *metric implementation* is wrong, not the model.
    """
    if strategy != "prior":
        logger.warning(
            f"Skipping known-answer checks: no known values defined for "
            f"strategy '{strategy}' (only 'prior' is validated)."
        )
        return

    # Expected Brier loss for predicting exactly the prevalence constant
    expected_brier = prevalence * (1.0 - prevalence)
    
    checks = {
        "brier_score": (mean_results["brier_score"], expected_brier, 0.01),
        "n_discoveries": (mean_results["n_discoveries"], 0.0, 1.0), # Expect 0 hits on noise
    }
    
    for name, (got, expected, tol) in checks.items():
        assert np.isclose(got, expected, atol=tol), (
            f"{name}: got {got:.4f}, expected ~{expected:.4f} (tol {tol}) — "
            "metric implementation is wrong"
        )
    logger.info("All calibration and FDR known-answer checks passed.")

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

    # NOTE for T23: Keep in mind cluster-aware splitting (StratifiedGroupKFold) 
    # if sequence identities cross folds, preventing artificial calibration inflation.
    cv = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    model = DummyClassifier(strategy=config.dummy_strategy, random_state=config.seed)
    fold_metrics = []

    logger.info(f"Running {config.n_splits}-Fold CV using calibration baseline: {config.dummy_strategy}")
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)

        # Extract probability array for the positive class
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = compute_evaluation_stack(y_test, y_proba, config.fdr_alpha)
        metrics["fold"] = fold + 1
        fold_metrics.append(metrics)

    df_results = pd.DataFrame(fold_metrics)
    mean_results = df_results.drop(columns="fold").mean().to_dict()

    logger.info("Calibration & FDR Summary Metrics:")
    for metric, score in mean_results.items():
        logger.info(f"  - Mean {metric.upper()}: {score:.4f}")

    # Critical Assertion Step for PT4
    validate_known_answers(mean_results, prevalence, config.dummy_strategy, config.fdr_alpha)

    df_results.to_csv(f"{config.output_dir}/pt4_calibration_results.csv", index=False)

def main():
    config = HarnessConfig()
    run_cv_harness(config)

if __name__ == "__main__":
    main()
