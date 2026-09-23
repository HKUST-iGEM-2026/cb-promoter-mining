# NT_ENV.md — the `nt-tf4x` environment (transformers 4.x, for T19 / Nucleotide Transformer only)

**TL;DR.** The Nucleotide Transformer member of the ensemble (**T19**, `InstaDeepAI/nucleotide-transformer-v2-250m-multi-species`) ships **custom remote code written for transformers 4.x**. It does **NOT** run on the main env's transformers 5.3.0 — v5 rewrote model construction/loading and broke NT's code in ≥6 independent places (one of them *silent*). So NT gets its **own** isolated env, `nt-tf4x` (transformers **4.57.6**), reusing the same torch 2.11+cu128. Everything else (ProkBERT/T18, CNN/T20, all data tasks) stays on the main **`promoter-mining-hpc4`** (5.3.0) env.

Built + verified on HPC4 (2026-07): T19 runs clean on `nt-tf4x` → **val AUPRC 0.853** (test 0.841), LoRA 0.507% trainable.

---

## Why NT needs 4.x (don't try to "fix" it on 5.3.0)

We chased v5 incompatibilities one at a time and it was whack-a-mole. NT's `modeling_esm.py` (a self-contained ESM copy, `transformers_version: 4.32.0.dev0`) breaks on transformers 5.3.0 in at least these ways:

1. imports `find_pruneable_heads_and_indices` / `prune_linear_layer` from `transformers.pytorch_utils` — **removed in v5** (head-pruning dropped).
2. reads `config.is_decoder` / `add_cross_attention` / … — v5 **dropped these base-config defaults**.
3. its `EsmForSequenceClassification.__init__` calls the legacy `init_weights()` not v5's `post_init()`, so `all_tied_weights_keys` is never created → `AttributeError` in `_finalize_model_loading` (and `tie_word_embeddings=False` does **not** avoid it — the attribute is accessed unconditionally).
4. imports from `transformers.file_utils` (a long-dead shim) → `ImportError` on v5.
5. legacy gradient-checkpointing hooks that v5 no longer honors.
6. **the dangerous one — a *silent* `_init_weights`/`_is_hf_initialized` mismatch**: the model "loads" but a subset of weights can be re-randomized, i.e. it would train *garbage* for hours and look like a bad model, never crashing.

Loading NT via transformers' **native** `EsmForSequenceClassification` (no `trust_remote_code`) is **also** impossible: NT-v2 uses a **SwiGLU (gated, no-bias) FFN** — its `intermediate.dense` is `[6144,768]`, native ESM expects `[3072,768]`. Hard shape mismatch; native ESM has no GLU path.

The sound fix is to run NT on the transformers version its code was written for. `4.57.6` is the sweet spot: new enough for `eval_strategy` + the modern `Trainer`/`compute_loss(**kwargs)` API, old enough that NT's remote code loads with **zero shims**. (It also satisfies prokbert's `transformers>=4.56` floor, but we keep prokbert on 5.3.0 since it's already verified there.)

---

## Build recipe (one-time, ~10 min, login node — pip only, git untouched)

```bash
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda create -y -p /project/igem2026/envs/nt-tf4x python=3.11 pip
conda activate /project/igem2026/envs/nt-tf4x
export PYTHONNOUSERSITE=1                       # ignore ~/.local so a stray user-site pkg can't shadow the env

# torch FIRST, from the cu128 index, exact +cu128 pins (the same wheels the main env runs):
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.11.0+cu128" "torchvision==0.26.0+cu128"

# freeze torch so the next install can't swap in a CUDA-13 build (the silent-CPU-fallback trap):
printf 'torch==2.11.0+cu128\ntorchvision==0.26.0+cu128\n' > /tmp/con-nt.txt

# transformers 4.x + peft + training deps, isolated from the 5.3 env's hub 1.x / tokenizers:
pip install --no-cache-dir -c /tmp/con-nt.txt \
  "transformers==4.57.6" "peft==0.17.1" \
  "huggingface_hub>=0.34.0,<1.0" "tokenizers>=0.22.0,<=0.23.0" \
  "accelerate>=0.26.0,<1.11" "safetensors>=0.4.3" \
  "scikit-learn>=1.3" "pandas>=2.0" "pyarrow>=14" "numpy>=1.26,<2.3" \
  "optuna>=4.0"                                 # needed so T21's NT sweep runs in THIS env
```

**Verify** (import check on login; run inside a GPU job for `cuda: True`):
```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python - <<'PY'
import torch, transformers, peft, optuna
print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__, "| optuna", optuna.__version__)
from transformers.pytorch_utils import find_pruneable_heads_and_indices   # the symbol v5 removed
from transformers import TrainingArguments; TrainingArguments(output_dir="/tmp/_p", eval_strategy="epoch")
print("OK — NT's 4.x symbols + eval_strategy present")
PY
```
Expect `transformers 4.57.6 | peft 0.17.1 | optuna 4.x` and `OK`.

---

## How the pipeline uses it

- **T19 standalone** and **the T21 NT sweep** activate `nt-tf4x`; **everything else** uses `promoter-mining-hpc4`.
- Both envs share the same HF cache: `export HF_HOME=/project/igem2026/hf_cache HF_HUB_OFFLINE=1`.
- **`--no-grad-checkpoint`** is passed to NT (standalone sbatch + auto-injected by T21 via `SWEEP_CONFIG["nt_lora"]["extra_args"]`): the 67-token windows are far too short to need checkpointing, and it sidesteps NT's legacy checkpointing hooks.
- **LoRA** targets `query`/`value` + keeps the `classifier` head trainable — binds cleanly on peft 0.17.

## Gotchas (env-specific)

- **`LD_LIBRARY_PATH="$CONDA_PREFIX/lib:…"` is required** for torch (libstdc++), same as the main env — set it in the sbatch, NOT `activate.d`, because it also **breaks `git` over SSH** (the env's OpenSSL shadows the system one → `OpenSSL version mismatch`). Run any `git` that talks to GitHub with **`LD_LIBRARY_PATH= git …`** or in a shell where the env isn't active.
- **Cross-env conflict is `huggingface_hub`/`tokenizers`, not torch** — 4.57 needs `hub<1.0`, the 5.3 env has hub 1.x. That's exactly why the envs are separate; once isolated it's a non-issue. torch 2.11 satisfies 4.57's `torch>=2.2` (no upper cap) and is never downgraded (installed first + frozen).
- **`PYTHONNOUSERSITE=1`** — keep it set (a stray `~/.local` numpy/pandas built for numpy 1.x will shadow the env and crash with `_ARRAY_API not found`).
