#!/usr/bin/env python3
"""
Side-by-side comparison of each ensemble model's STANDALONE run (default hyperparameters) vs its
RETRAIN (best swept hyperparameters, full epochs), with the delta — so you see both at once.

Pure standard library (json only) — runs in ANY env (even base); no torch/optuna/pandas needed.

Run:  python pipeline/evaluation/show_model_stats.py
      REPO=/some/other/clone python pipeline/evaluation/show_model_stats.py   # override repo path
"""
import json
import os

# Portable paths: override with PROMOTER_REPO / PROMOTER_RESULTS (or `source env.sh`).
# The fallbacks resolve from this file, so a plain clone works with no configuration.
PROMOTER_REPO = os.environ.get("PROMOTER_REPO") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PROMOTER_RESULTS = os.environ.get("PROMOTER_RESULTS") or os.path.join(PROMOTER_REPO, "workflow", "results")


BASE = os.environ.get("REPO", PROMOTER_REPO)
SCAF = f"{BASE}/workflow/results"

# name -> (standalone dir, retrain dir)
MODELS = {
    "ProkBERT": (f"{SCAF}/T18", f"{SCAF}/T21_final/prokbert"),
    "NT-LoRA":  (f"{SCAF}/T19", f"{SCAF}/T21_final/nt_lora"),
    "CNN":      (f"{SCAF}/T20", f"{SCAF}/T21_final/cnn"),
}
METRICS = [
    ("val AUPRC", "val_auprc"), ("test AUPRC", "test_auprc"),
    ("val AUROC", "val_auroc"), ("test AUROC", "test_auroc"),
    ("val F1",    "val_f1"),    ("test F1",    "test_f1"),
]


def load(d):
    try:
        with open(f"{d}/metrics.json") as f:
            return json.load(f)
    except Exception:
        return {}


def cell(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else "  —   "


print("=" * 70)
print("  STANDALONE (default hparams)   vs   RETRAIN (best hparams, full epochs)")
print("=" * 70)

for name, (sdir, rdir) in MODELS.items():
    std, ret = load(sdir), load(rdir)
    print(f"\n{name}   (weighted: standalone={std.get('weighted')}, retrain={ret.get('weighted')})")
    print(f"  {'metric':11}{'standalone':>13}{'retrain':>13}{'Δ (ret-std)':>15}")
    print("  " + "-" * 50)
    for label, key in METRICS:
        s, r = std.get(key), ret.get(key)
        both = isinstance(s, (int, float)) and isinstance(r, (int, float))
        delta = f"{r - s:+.4f}" if both else "  —   "
        print(f"  {label:11}{cell(s):>13}{cell(r):>13}{delta:>15}")
    sv, rv = std.get("val_auprc"), ret.get("val_auprc")
    if isinstance(sv, (int, float)) and isinstance(rv, (int, float)):
        chosen = "RETRAIN" if rv >= sv else "STANDALONE"
        why = "sweep improved it" if rv >= sv else "sweep did NOT beat the default"
        print(f"  -> chosen for the ensemble: {chosen}   ({why}; selection = higher val AUPRC)")
print()
