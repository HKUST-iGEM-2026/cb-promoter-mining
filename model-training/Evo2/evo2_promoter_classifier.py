import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Load embeddings
emb_dir = '/project/igem2026/cb-promoter-mining/evo2_embeddings'
X_tr = np.load(f'{emb_dir}/train_X.npy')
y_tr = np.load(f'{emb_dir}/train_y.npy')
X_val = np.load(f'{emb_dir}/val_X.npy')
y_val = np.load(f'{emb_dir}/val_y.npy')
X_te = np.load(f'{emb_dir}/test_X.npy')
y_te = np.load(f'{emb_dir}/test_y.npy')

print(f'Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_te.shape}')

# Define classifiers
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf  = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
lin = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))

# Benchmark both on training data
for name, clf in [('logreg', lin), ('rf', rf)]:
    scores = cross_val_score(clf, X_tr, y_tr, cv=cv, scoring='roc_auc')
    print(f'{name} cv AUROC: {scores.mean():.4f} ± {scores.std():.4f}')

# Evaluate both on validation set
for name, clf in [('logreg', lin), ('rf', rf)]:
    clf.fit(X_tr, y_tr)
    val_auroc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
    print(f'\n{name} val AUROC: {val_auroc:.4f}')
    print(classification_report(y_val, clf.predict(X_val)))

# Train best model on train+val, evaluate on test
print('\n--- Final evaluation on test set ---')
X_trainval = np.concatenate([X_tr, X_val])
y_trainval = np.concatenate([y_tr, y_val])

best = lin  # change to rf if rf was better on val
best.fit(X_trainval, y_trainval)
test_auroc = roc_auc_score(y_te, best.predict_proba(X_te)[:, 1])
print(classification_report(y_te, best.predict(X_te)))
print(f'Test AUROC: {test_auroc:.4f}')

# Save the trained classifier
joblib.dump(best, '/project/igem2026/cb-promoter-mining/promoter_classifier.pkl')
print('Classifier saved.')