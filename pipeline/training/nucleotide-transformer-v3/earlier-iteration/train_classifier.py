import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Portable paths: override with PROMOTER_REPO / PROMOTER_RESULTS (or `source env.sh`).
# The fallbacks resolve from this file, so a plain clone works with no configuration.
PROMOTER_REPO = os.environ.get("PROMOTER_REPO") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
PROMOTER_RESULTS = os.environ.get("PROMOTER_RESULTS") or os.path.join(PROMOTER_REPO, "workflow", "results")


def main():
    print("=== Training Scaled Classifier on NTv3 Embeddings ===", flush=True)
    data_dir = os.path.join(PROMOTER_REPO, "nt3_embeddings")

    # Load data
    X_train = np.load(f"{data_dir}/train_X.npy")
    y_train = np.load(f"{data_dir}/train_y.npy")
    X_val = np.load(f"{data_dir}/val_X.npy")
    y_val = np.load(f"{data_dir}/val_y.npy")
    X_test = np.load(f"{data_dir}/test_X.npy")
    y_test = np.load(f"{data_dir}/test_y.npy")

    print(f"Train shape: {X_train.shape}, Labels: {y_train.shape}", flush=True)
    print(f"Val shape:   {X_val.shape}, Labels: {y_val.shape}", flush=True)
    print(f"Test shape:  {X_test.shape}, Labels: {y_test.shape}", flush=True)

    # Scale features (crucial for high-dimensional transformer embeddings)
    print("\nScaling features...", flush=True)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression with higher max_iter
    print("Training Logistic Regression model...", flush=True)
    clf = LogisticRegression(max_iter=5000, C=1.0, solver='lbfgs')
    clf.fit(X_train_scaled, y_train)

    # Evaluate on Validation Set
    val_preds = clf.predict(X_val_scaled)
    val_probs = clf.predict_proba(X_val_scaled)[:, 1]
    
    print("\n--- Validation Results ---", flush=True)
    print(f"Accuracy: {accuracy_score(y_val, val_preds):.4f}", flush=True)
    print(f"ROC-AUC:  {roc_auc_score(y_val, val_probs):.4f}", flush=True)

    # Evaluate on Test Set
    test_preds = clf.predict(X_test_scaled)
    test_probs = clf.predict_proba(X_test_scaled)[:, 1]
    
    print("\n--- Test Results ---", flush=True)
    print(f"Accuracy: {accuracy_score(y_test, test_preds):.4f}", flush=True)
    print(f"ROC-AUC:  {roc_auc_score(y_test, test_probs):.4f}", flush=True)
    print("\nClassification Report (Test):", flush=True)
    print(classification_report(y_test, test_preds), flush=True)

if __name__ == "__main__":
    main()
