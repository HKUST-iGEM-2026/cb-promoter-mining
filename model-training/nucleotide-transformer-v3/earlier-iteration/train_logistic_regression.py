import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def main():
    print("Loading extracted embeddings...")
    train_X = np.load("train_X.npy")
    train_y = np.load("train_y.npy")
    val_X = np.load("val_X.npy")
    val_y = np.load("val_y.npy")
    test_X = np.load("test_X.npy")
    test_y = np.load("test_y.npy")
    
    print("Scaling features...")
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)
    
    print("Training Logistic Regression classifier with increased iterations...")
    clf = LogisticRegression(max_iter=5000, C=1.0, random_state=42)
    clf.fit(train_X, train_y)
    
    print("\n--- Validation Set Evaluation ---")
    val_preds = clf.predict(val_X)
    val_probs = clf.predict_proba(val_X)[:, 1]
    print(f"Validation Accuracy: {accuracy_score(val_y, val_preds):.4f}")
    print(f"Validation ROC-AUC: {roc_auc_score(val_y, val_probs):.4f}")
    
    print("\n--- Test Set Evaluation ---")
    test_preds = clf.predict(test_X)
    test_probs = clf.predict_proba(test_X)[:, 1]
    print(f"Test Accuracy: {accuracy_score(test_y, test_preds):.4f}")
    print(f"Test ROC-AUC: {roc_auc_score(test_y, test_probs):.4f}")
    print("\nClassification Report (Test):")
    print(classification_report(test_y, test_preds))

if __name__ == "__main__":
    main()
