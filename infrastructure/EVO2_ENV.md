# EVO2_ENV.md — the `evo2` environment (Evo 2, Arc Institute)

## TL;DR
Evo 2 is an **autoregressive** DNA language model — single-nucleotide
resolution, up to 1 Mb context, **StripedHyena 2** architecture, pretrained
autoregressively on **OpenGenome2** (8.8T tokens, all domains of life). Paper:
*Genome modelling and design across all domains of life with Evo 2*, Nature
2026, doi `10.1038/s41586-026-10176-5`.

Different beast from both T19 (NTv2, masked encoder) and NTv3 (masked U-Net):
Evo 2 is causal, so it gives you **sequence likelihoods for free** — zero-shot
variant effect scoring without training anything. That's its main draw for you
alongside embeddings.

Needs its own env (`evo2`): it depends on the **Vortex** inference engine,
optionally **Transformer Engine**, and a compiled **flash-attn**. None of that
belongs anywhere near `nt-tf4x` or `promoter-mining-hpc4`.

Code: https://github.com/ArcInstitute/evo2
Weights: https://huggingface.co/arcinstitute (Apache 2.0, **not gated**)
Dataset: https://huggingface.co/datasets/arcinstitute/opengenome2

### ⚠️ Four corrections to the notes this doc was written from

1. **There is a 20B model now.** Released in v0.5.0 as "40B-level performance
   with double the speed". The variant list is not 1B/7B/40B.
2. **The FP8 requirement is backwards in the notes.** It is *not* "Evo 2 needs
   FP8, so you need compute capability ≥ 8.9". Per upstream: **the 40B, 20B and
   1B models require FP8 via Transformer Engine and a Hopper GPU. The 7B models
   run in bfloat16 without Transformer Engine on any supported GPU.** So
   `evo2_1b_base` — the one that *sounds* like the easy starter model — has a
   *stricter* hardware requirement than `evo2_7b`. Start with 7B.
3. **No submodules.** `git clone --recurse-submodules git@github.com:...` is
   stale guidance. Vortex is now a normal pip dependency. Current upstream
   install is `pip install evo2`, or `git clone` (HTTPS) + `pip install -e .`.
   The `git@github.com:` SSH form will also just hang on a login node without
   deploy keys — use HTTPS.
4. **Python 3.11 or 3.12 only**, not "3.11 or higher". 3.13 is not supported.

### Checkpoints and FP8 requirements

| Checkpoint | Params | Context | FP8 / Transformer Engine required? |
|---|---|---|---|
| `evo2_7b` | 7B | 1M | **No** |
| `evo2_7b_262k` | 7B | 262k | **No** |
| `evo2_7b_base` | 7B | 8k | **No** |
| `evo2_1b_base` | 1B | 8k | **Yes** (Hopper) |
| `evo2_20b` | 20B | 1M | **Yes** (Hopper) |
| `evo2_40b` | 40B | 1M | **Yes** (Hopper) |
| `evo2_40b_base` | 40B | 8k | **Yes** (Hopper) |
| `evo2_7b_microviridae` | 7B | — | **No** (7B base fine-tuned on Microviridae) |

The 40B model requires **multiple** H100-class GPUs; Vortex handles device
placement by splitting across available CUDA devices. On this cluster that
means `--gpus-per-node=2` or more, and a real memory plan — don't attempt it
casually.

7B in bf16 is roughly 14–15 GB of weights. The "16 GB VRAM minimum" figure
floating around community threads is a *floor at short context only*; KV/state
memory grows with sequence length, so at 262k or 1M context you need far more.
H800s (80 GB) are comfortable for 7B at long context and are Hopper, so FP8 is
available if you later move to 20B/40B.

---

## 1. Prerequisites
Same HPC4 basics as everywhere else (see `setup guide`, `T4/DEPLOY.md`,
`working-on-HPC4`): account active, `igem2026` Slurm account, VPN if
off-campus, conda initialised via
```bash
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
```
before any `conda activate` — otherwise `which python` silently stays on base
Anaconda.

Env lives in `/project/igem2026` (persistent, shared), never `/scratch`
(30-day Lustre eviction).

Upstream system requirements:

- Linux (official); WSL2 has limited support — irrelevant here
- **CUDA 12.1+** with compatible drivers, **cuDNN 9.3+**
- **GCC 9+ or Clang 10+ with C++17** — needed to build flash-attn
- **Python 3.11 or 3.12**
- Torch **2.6.x or 2.7.x** recommended

## 2. CUDA and the flash-attn build (read before you start)
H800s, driver ceiling **CUDA 12.8** (`nvidia-smi`, run inside a GPU job — see
`working-on-HPC4` §5). Same rule as the torch and NTv3 envs: **plain
`pip install torch` from PyPI gives you a CUDA-13 build.** Use the cu128 index
explicitly:
```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

⚠️ **flash-attn is the hard part of this install, not Evo 2 itself.**

- `flash-attn` must be installed **after** torch, with `--no-build-isolation`
  (that flag is what lets its build see the already-installed torch; without
  it, pip builds in a clean env, doesn't find torch, and fails).
- It **compiles from source** and is memory- and time-hungry — this can take
  30+ minutes and several GB of RAM. If the login node kills it or it thrashes,
  run the install inside an interactive job instead (§3, note at the end).
- Needs a C++17-capable compiler on PATH. `gcc --version` before you start; if
  the default is ancient, `module load` a newer GCC first.

⚠️ **Transformer Engine is only needed if you use 1B/20B/40B.** For 7B, skip it
entirely — that's the "light install" path and it removes the single most
fragile dependency from the build. Do the light install first; add TE later
only if you actually move off 7B.

---

## 3. Build (login node, ~30–60 min mostly flash-attn)

### 3A. Light install — 7B models only (recommended starting point)
```bash
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda create -y -p /project/igem2026/envs/evo2 python=3.11 pip
conda activate /project/igem2026/envs/evo2
export PYTHONNOUSERSITE=1        # same rationale as nt-tf4x: stop ~/.local shadowing the env

# 1. torch FIRST, from the cu128 index (NOT plain PyPI — see §2)
pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# 2. flash-attn, compiled, after torch, with build isolation off
pip install --no-cache-dir flash-attn==2.8.0.post2 --no-build-isolation

# 3. Evo 2 (pulls Vortex as a normal dependency — no submodules)
pip install --no-cache-dir evo2

# 4. downstream analysis deps for the embeddings → classifier workflow
pip install --no-cache-dir scikit-learn numpy pandas matplotlib joblib
```

### 3B. Full install — adds FP8 / Transformer Engine (needed for 1B, 20B, 40B)
Everything in 3A, but install TE **before** Evo 2. Upstream recommends conda
for TE:
```bash
conda install -y -c nvidia cuda-nvcc cuda-cudart-dev
conda install -y -c conda-forge transformer-engine-torch=2.3.0
pip install --no-cache-dir flash-attn==2.8.0.post2 --no-build-isolation
pip install --no-cache-dir evo2
```

### From source instead (if you need to patch Vortex or read the code)
```bash
cd /project/igem2026
git clone https://github.com/ArcInstitute/evo2.git evo2-repo   # HTTPS, no --recurse-submodules
cd evo2-repo
pip install --no-cache-dir -e .
```
`-e` matters here: you will end up reading Vortex internals when something
about device placement or context length misbehaves.

**If the login node can't handle the flash-attn build**, do the whole thing in
an interactive job (login nodes have internet; check whether compute nodes on
your partition do — if not, this won't work and you must build on the login
node):
```bash
srun --account=igem2026 --partition=preempt --gpus-per-node=1 --nodes=1 \
     --time=02:00:00 --pty bash
```

### Freeze
```bash
pip freeze > /project/igem2026/envs/evo2-requirements.lock.txt
chmod -R g+rX /project/igem2026/envs/evo2 2>/dev/null || true
```
Pin hard here. `evo2` → `vortex` → `flash-attn` → `torch` is a four-layer stack
where every layer is version-sensitive, and flash-attn in particular is
expensive to rebuild. Note the lock file records `flash-attn`'s *version*, not
its compiled artifact — a rebuild recompiles from scratch.

---

## 4. Verify it works
Upstream ships a self-test. Run it on a **GPU node** — it loads real weights:
```bash
srun --account=igem2026 --partition=preempt --gpus-per-node=1 --nodes=1 --time=00:30:00 \
  bash -c 'source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
           conda activate /project/igem2026/envs/evo2
           export HF_HOME=/project/igem2026/hf_cache
           python -m evo2.test.test_evo2_generation --model_name evo2_7b'
```
Quicker sanity check before that, no weights needed:
```bash
python -c "
import torch, evo2
print('torch', torch.__version__, '| cuda build:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
import flash_attn; print('flash-attn', flash_attn.__version__)
"
```
(Don't add `--mem` or CPU flags to GPU jobs — same SuperPOD auto-allocation as
everywhere else. `--account=igem2026` required.)

## 5. Pre-stage the model weights (compute nodes are offline)
Same pattern as `DEPLOY.md` §7. Unlike NTv3, **Evo 2 weights are Apache 2.0 and
ungated** — no HF login, no terms to accept, no token needed. You still want to
pre-stage, because compute nodes have no network.

```bash
export HF_HOME=/project/igem2026/hf_cache      # reuse the shared team cache
huggingface-cli download arcinstitute/evo2_7b
# 8k-context variant, much smaller footprint if you don't need long range:
# huggingface-cli download arcinstitute/evo2_7b_base
# only if you have a Hopper + TE build and a memory plan:
# huggingface-cli download arcinstitute/evo2_40b
```
Check free space before pulling 40B — it is very large, and `/project` is
shared.

Then in every SBATCH preamble:
```bash
export HF_HOME=/project/igem2026/hf_cache
export HF_HUB_OFFLINE=1
```

---

## 6. Smoke tests (GPU job)

### Forward pass / likelihoods
```bash
cat > evo2_smoke.sbatch <<'EOF'
#!/bin/bash
#SBATCH --account=igem2026
#SBATCH --partition=preempt
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --job-name=evo2_smoke
#SBATCH --output=%x_%j.out

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate /project/igem2026/envs/evo2
export HF_HOME=/project/igem2026/hf_cache
export HF_HUB_OFFLINE=1

python - <<'PY'
import torch
from evo2 import Evo2
assert torch.cuda.is_available(), "NO GPU — aborting"

evo2_model = Evo2('evo2_7b')

sequence = 'ACGT'
input_ids = torch.tensor(
    evo2_model.tokenizer.tokenize(sequence),
    dtype=torch.int,
).unsqueeze(0).to('cuda:0')

outputs, _ = evo2_model(input_ids)
logits = outputs[0]
print('Shape (batch, length, vocab):', logits.shape)
PY
EOF
sbatch evo2_smoke.sbatch
```

### Embeddings
```python
layer_name = 'blocks.28.mlp.l3'
outputs, embeddings = evo2_model(
    input_ids, return_embeddings=True, layer_names=[layer_name]
)
print('Embeddings shape:', embeddings[layer_name].shape)   # (1, L, D)
```

⚠️ **`blocks.28.mlp.l3` is specific to the 7B models** (32 layers), and it is
what upstream uses in their own example — but it is not automatically the right
layer for 1B (fewer blocks), 20B (50 layers), or 40B. If you switch model size,
index 28 will either not exist or land somewhere very different in relative
depth. Enumerate the available layer names rather than assuming (see §8).

---

## 7. Embeddings → classifier (your downstream plan)
Your notes on this are sound. Consolidating, with additions:

**Pooling is mandatory.** Evo 2 returns per-nucleotide embeddings of shape
`(batch, length, dim)`. Random forests and logistic regression need a fixed-size
row per sequence, so mean-pool over the length axis:
```python
vec = embeddings[layer_name].float().mean(dim=1).squeeze(0).cpu().numpy()  # (D,)
```
Mean is the default; also worth trying max-pool and mean-of-last-k-positions.
For an autoregressive model, the *final* position's embedding has seen the whole
sequence and is sometimes a better single-vector summary than the mean — cheap
to benchmark, so benchmark it.

**Intermediate > final layer.** Upstream states directly that intermediate
embeddings work better than final embeddings for downstream tasks. Don't reach
for the last block.

**Benchmark logistic regression against the random forest.** Your note is
right and worth acting on rather than filing: on high-dimensional dense
embeddings, linear probes usually match or beat tree ensembles, train in
seconds, and are far less prone to overfitting. Standardise features first
(`StandardScaler`) for the linear model; trees don't need it.

**Guard against overfitting properly.** A single 80/20 split is thin for the
sample sizes you'll have. Use stratified k-fold cross-validation, tune
`n_estimators` / `max_depth` / `max_features` inside the folds, and report on a
held-out split you never touched. Report AUROC/AUPRC alongside accuracy —
accuracy alone hides class imbalance, which genomic label sets usually have.

**Sequence-level splits, not window-level.** If you generate multiple windows
per gene/locus, random splitting leaks near-duplicate sequence between train and
test and inflates every number. Group by locus.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# X: (n_sequences, D) mean-pooled Evo 2 embeddings; y: labels
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf  = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
lin = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))

for name, clf in [("rf", rf), ("logreg", lin)]:
    print(name, "cv AUROC:", cross_val_score(X_tr, y_tr, estimator=clf,
                                             cv=cv, scoring="roc_auc").mean())

best = lin  # pick on CV, not on the test set
best.fit(X_tr, y_tr)
print(classification_report(y_te, best.predict(X_te)))
print("test AUROC:", roc_auc_score(y_te, best.predict_proba(X_te)[:, 1]))
```

**Two upstream notebooks are directly relevant** and worth reading before
writing your own:
- `notebooks/exon_classifier/exon_classifier.ipynb` — this *is* the
  embeddings-plus-classifier pattern you're describing, already working.
- `notebooks/brca1/brca1_zero_shot_vep.ipynb` — zero-shot variant effect
  prediction by comparing reference vs alternative allele likelihoods, including
  ref/alt normalisation. This is the "alternative/complementary approach" from
  your notes, and it needs no training at all. Run it before you invest in the
  supervised route: if zero-shot likelihood scoring already separates your
  classes, the classifier may be unnecessary.

---

## 8. Gotchas
- **Don't start with 1B.** It needs FP8 + Transformer Engine + Hopper. 7B does
  not. The small model is the harder install.
- **Long sequences via Vortex are slow.** Upstream notes forward passes on very
  long sequences may currently be slow, and points to Savanna or NVIDIA BioNeMo
  for embedding long sequences. Budget accordingly before planning a 1 Mb sweep.
- **Autoregressive, so context is one-directional.** Unlike NTv3/NTv2, a
  nucleotide's embedding only sees upstream sequence. Relevant if you're
  comparing embeddings across the three models — they are not the same kind of
  object.
- **`--recurse-submodules` and the SSH clone URL are both wrong** for the
  current repo. HTTPS, no submodules.
- **Torch must go in before flash-attn.** Reinstalling or upgrading torch later
  will likely break the compiled flash-attn and force a rebuild. Pin torch and
  leave it alone.
- **Validate after any hardware or config change.** Upstream explicitly says to
  re-run the tests after configuration changes or on different hardware — with
  FP8 paths and mixed precision this is not boilerplate advice.
- **Separate env, on purpose.** Don't `pip install evo2` into `nt-tf4x`,
  `ntv3-torch`, or `promoter-mining-hpc4`.
- **Not wired into your Snakemake graph.** Same status as NTv3 — this doc builds
  and verifies the env only.

## 9. Still to verify before writing task code
- **Layer names for non-7B models.** Enumerate rather than guess. Something
  along these lines after loading (the exact attribute path holding the
  `nn.Module` may differ — check `dir(evo2_model)` first):
  ```python
  print([n for n, _ in evo2_model.model.named_modules() if n.endswith('mlp.l3')])
  ```
- **Whether compute nodes on `preempt` have outbound network.** Determines
  whether the flash-attn build can be done in a job or must happen on the login
  node.
- **Whether `conda install -c conda-forge transformer-engine-torch` resolves
  behind the cluster proxy.** Only matters if you go past 7B.
- **Actual peak VRAM at your target context length**, measured with
  `torch.cuda.max_memory_allocated()` on a representative sequence — the 16 GB
  figure is a short-context floor, not a planning number.
