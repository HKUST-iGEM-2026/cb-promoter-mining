# Loading the trained models

How to load each of the four ensemble models and get `P(promoter)` for a list of
81 bp sequences. All four output a promoter probability via `softmax(logits)[:, 1]`.

## Before anything

```bash
source /project/igem2026/cb-promoter-mining/env.sh          # shared offline HF cache
conda activate /project/igem2026/envs/promoter-mining-hpc4  # NT is the exception -> see below
```

Two hard rules:

1. **NT-LoRA loads ONLY in the `nt-tf4x` env** (transformers 4.57). The other three load in
   `promoter-mining-hpc4` (transformers 5.3). Loading NT in the 5.3 env fails in several ways,
   one silent. See `infrastructure/conda-environments/NT_ENV.md`.
2. **Sequences must be the 81 bp training window.** ProkBERT/NT/CNN expect 81 bp; CNN-v2's
   `predict_proba` will center-pad/crop to 81 for you, but feed 81 bp when you can.

## Where the checkpoints are

```
RES=/project/igem2026/cb-promoter-mining/infrastructure/T3/snakemake_scaffold/results
$RES/T21_final/prokbert/best_model     # ProkBERT  (HF model dir)
$RES/T21_final/nt_lora/best_adapter    # NT-LoRA   (PEFT adapter dir)
$RES/T21_final/cnn/best_model.pt       # CNN       (+ metrics.json in same dir)
$RES/T21_final/cnn_v2/best_model.pt    # CNN-v2    (+ config.json in same dir)
```

**Use `$RES/models_deployed/` — that is the best-of-each set** (symlinks): ProkBERT points at the
0.6-swept model (`T21_final_06/prokbert`, better on balanced AND genome scale), while NT/CNN/CNN-v2
point at the current 0.6 models (`T21_final/...`, where the re-sweep did not help). Do NOT point at a
single raw folder: `T21_final` has the older ProkBERT, `T21_final_06` has worse NT/CNN/CNN-v2. Also
present for reference: `T18/best_model` (the pre-0.6 / 0.9 ProkBERT). Same loading code, different path.

## Fastest path: reuse the repo's scorers (don't reinvent)

`evaluation-and-calibration/genome_inference.py` already loads every model correctly and exposes
a `scorer(list_of_seqs) -> probs` for each. Import it and let it handle the gotchas:

```python
import importlib.util
def _load(pyfile, name):
    spec = importlib.util.spec_from_file_location(name, pyfile)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
gi = _load("/project/igem2026/cb-promoter-mining/evaluation-and-calibration/genome_inference.py", "gi")

# CNN (main env):
score = gi.cnn_scorer(gi.load_cnn(f"{RES}/T21_final/cnn/best_model.pt", "cpu"), "cpu", 81)
probs = score(["ACGT...81bp...", ...])          # numpy array of P(promoter)

# CNN-v2 (main env):
mod, model, ew = gi.load_cnn_v2(f"{RES}/T21_final/cnn_v2/best_model.pt", "cpu")
probs = gi.cnn_v2_scorer(mod, model, "cpu", ew)(["ACGT...", ...])
```

Use `"cuda"` instead of `"cpu"` inside a GPU job.

## Or load each directly

### ProkBERT (env: promoter-mining-hpc4)
Tokenizer comes from the BASE repo, not the fine-tuned dir. `trust_remote_code=True` is required.

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tok = AutoTokenizer.from_pretrained("neuralbioinfo/prokbert-mini", trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    f"{RES}/T21_final/prokbert/best_model", trust_remote_code=True).eval()
enc = tok(seqs, padding="max_length", truncation=True, max_length=81, return_tensors="pt")
with torch.no_grad():
    probs = torch.softmax(model(**enc).logits, dim=1)[:, 1]     # P(promoter)
```

### NT-LoRA (env: nt-tf4x ONLY)
A LoRA adapter on top of the base NT model.

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
NT = "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species"
tok  = AutoTokenizer.from_pretrained(NT, trust_remote_code=True)
base = AutoModelForSequenceClassification.from_pretrained(NT, num_labels=2, trust_remote_code=True)
model = PeftModel.from_pretrained(base, f"{RES}/T21_final/nt_lora/best_adapter").eval()
enc = tok(seqs, padding="max_length", truncation=True, max_length=81, return_tensors="pt")
with torch.no_grad():
    probs = torch.softmax(model(**enc).logits, dim=1)[:, 1]
```

### CNN (env: promoter-mining-hpc4)
Rebuild the architecture from the saved config, then load the state_dict. Input is one-hot `(N, 4, 81)`.

```python
import importlib.util, json, torch, numpy as np
tcnn = _load(f"{REPO}/model-training/cnn/train_cnn.py", "train_cnn")   # REPO = repo root
cfg  = json.load(open(f"{RES}/T21_final/cnn/metrics.json")).get("config", {})
keys = ("conv1_filters","conv2_filters","conv3_filters","kernel_size","dropout_rate","use_batch_norm")
model = tcnn.PromoterCNN(input_channels=4, sequence_length=81, **{k: cfg[k] for k in keys if k in cfg})
model.load_state_dict(torch.load(f"{RES}/T21_final/cnn/best_model.pt", map_location="cpu")); model.eval()

B = {"A":0,"C":1,"G":2,"T":3}
X = np.zeros((len(seqs), 4, 81), np.float32)
for i, s in enumerate(seqs):
    for j, b in enumerate(s.upper()[:81]):
        if b in B: X[i, B[b], j] = 1.0
with torch.no_grad():
    probs = torch.softmax(model(torch.from_numpy(X)).float(), dim=1)[:, 1]
```

### CNN-v2 (env: promoter-mining-hpc4)
Multi-encoding fusion. `build_model_from_config` + `predict_proba` do everything (featurization,
length-fitting, batching).

```python
import importlib.util, json, torch
v2  = _load(f"{REPO}/model-training/cnn/train_cnn_v2.py", "train_cnn_v2")
cfg = json.load(open(f"{RES}/T21_final/cnn_v2/config.json"))
model = v2.build_model_from_config(cfg, device="cpu")
model.load_state_dict(torch.load(f"{RES}/T21_final/cnn_v2/best_model.pt", map_location="cpu"))
probs = v2.predict_proba(model, seqs, "cpu", enac_window=int(cfg.get("enac_window", 5)))  # numpy array
```

## Notes

- Raw scores are uncalibrated (CNN-v2 especially runs optimistic). For rank/ensemble use that is
  fine (rank-percentile is calibration-invariant); for a fixed 0.5 threshold, apply T26 calibration first.
- To reproduce a whole-genome scan or the test-set metrics, don't script it by hand — run
  `evaluation-and-calibration/genome_inference.py` and `evaluation-and-calibration/metrics/predict_and_score.py`
  (see `evaluation-and-calibration/REGEN_EVAL_0.6.md`).
