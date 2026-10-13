#!/bin/bash
# verify_env.sh — detailed correctness check of the clone before Phase-C training.
# Complements check_state.sh (which reports *what's run*); this checks *whether it's correct*.
# Run on the LOGIN node (section E may download prokbert-mini if not cached):
#     bash workflow/verify_env.sh
# GPU (section D) is printed as a separate srun to run yourself.
set -u
REPO="${PROMOTER_REPO:?set it or source env.sh}"
SC="$REPO/workflow"
cd "$REPO" || { echo "repo not found at $REPO"; exit 1; }

echo "===== A. REPO / GIT ====="
echo "branch=$(git branch --show-current)  HEAD=$(git rev-parse --short HEAD)  origin/main=$(git rev-parse --short origin/main)"
git fetch -q origin 2>/dev/null || true
echo "sync: $(git status -sb 2>/dev/null | head -1)"
echo "tracked local modifications (expect none, or known WIP):"
git status --short 2>/dev/null | grep -v '^??' | sed 's/^/  /' || echo "  (none)"

echo; echo "===== B. ENVIRONMENT ====="
python3 - <<'PY'
import importlib
for m in ["torch","transformers","peft","sklearn","scipy","pandas","numpy","optuna"]:
    try:
        mod = importlib.import_module(m)
        print(f"  OK   {m} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  FAIL {m}: {type(e).__name__}: {e}")
import torch
print("  torch.version.cuda =", torch.version.cuda, "(GPU tested in section D)")
PY
command -v cd-hit-est >/dev/null && echo "  OK   cd-hit-est" || echo "  FAIL cd-hit-est missing"
F="$CONDA_PREFIX/etc/conda/activate.d/zz_env_fixes.sh"
grep -q "CONDA_PREFIX/lib" "$F" 2>/dev/null && echo "  OK   libstdc++ LD_LIBRARY_PATH fix present in activate.d" || echo "  WARN libstdc++ fix NOT in activate.d (imports may break in fresh shells)"

echo; echo "===== C. DATA INTEGRITY (T7b -> T15 -> T16) ====="
python3 - "$SC" <<'PY'
import sys, pandas as pd, numpy as np
SC = sys.argv[1]; ok = True
try:
    d = pd.read_parquet(f"{SC}/inputs/published_tss_promoters.parquet")
    print(f"  T7b positives: {len(d)}  species={d.species.value_counts().to_dict()}")
    t15 = {}
    for s in ["train","val","test"]:
        df = pd.read_parquet(f"{SC}/results/T15/{s}.parquet"); t15[s] = len(df)
        pos = int(df.label.eq(1).sum()); sil = int((df.weight.round(3) == 0.3).sum())
        need = {"sequence","label","weight","cluster_id"}
        print(f"  T15/{s}: {len(df)} rows | {pos} pos ({pos/len(df):.0%}) | silver(0.3)={sil} | cols_ok={need.issubset(df.columns)}")
        if s in ("val","test") and sil != 0:
            print(f"    !! {s} contains silver (0.3) — must be verified-only"); ok = False
    for m in ["prokbert","nt"]:
        for s in ["train","val","test"]:
            t = pd.read_parquet(f"{SC}/results/T16/{m}/{s}.parquet")
            cols = [c for c in ["input_ids","attention_mask","label","weight"] if c in t.columns]
            match = (len(t) == t15[s])
            print(f"  T16/{m}/{s}: {len(t)} rows (==T15: {match}) | cols={cols}")
            if not match or len(cols) < 4: ok = False
    for s in ["train","val","test"]:
        z = np.load(f"{SC}/results/T16/cnn/{s}.npz")
        match = (z['X'].shape[0] == t15[s]); shape_ok = tuple(z['X'].shape[1:]) == (81,4)
        print(f"  T16/cnn/{s}: X{z['X'].shape} label{z['label'].shape} weight{z['weight'].shape} (==T15: {match}, shape_ok: {shape_ok})")
        if not match or not shape_ok: ok = False
except Exception as e:
    import traceback; print("  FAIL:", type(e).__name__, e); traceback.print_exc(); ok = False
print("  ==> DATA:", "PASS" if ok else "PROBLEMS ABOVE (see !!)")
PY

echo; echo "===== D. GPU — run this separately (a 5-min job) ====="
echo "  srun --gpus-per-node=1 --time=00:05:00 \\"
echo "    python3 -c \"import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\""

echo; echo "===== E. TRAINING-CODE SMOKE: prokbert-mini seq-clf loads + weighted loss on REAL T16 data ====="
if [ -n "${HF_HOME:-}" ]; then export HF_HOME; fi
python3 - "$SC" <<'PY'
import sys, torch, pandas as pd
SC = sys.argv[1]
try:
    from transformers import AutoModelForSequenceClassification, AutoConfig
    df = pd.read_parquet(f"{SC}/results/T16/prokbert/train.parquet").head(8)
    cfg = AutoConfig.from_pretrained("neuralbioinfo/prokbert-mini", num_labels=2, trust_remote_code=True)
    m = AutoModelForSequenceClassification.from_pretrained("neuralbioinfo/prokbert-mini", config=cfg, trust_remote_code=True)
    ids = torch.tensor([list(map(int, x)) for x in df["input_ids"]])
    am  = torch.tensor([list(map(int, x)) for x in df["attention_mask"]])
    y = torch.tensor(df["label"].astype(int).tolist()); w = torch.tensor(df["weight"].astype(float).tolist())
    out = m(input_ids=ids, attention_mask=am)
    per = torch.nn.functional.cross_entropy(out.logits, y, reduction="none"); loss = (per * w).sum() / w.sum()
    print(f"  OK   prokbert-mini SEQ-CLF loads + forward: logits{tuple(out.logits.shape)}  weighted_loss={float(loss):.4f}")
    print("       (this is exactly what T18's first step does — if it's OK here, T18 will load fine)")
except Exception as e:
    import traceback; print("  FAIL:", type(e).__name__, e); traceback.print_exc()
PY
echo; echo "===== done — paste this whole output ====="
