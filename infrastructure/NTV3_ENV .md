# NTV3_ENV.md — the `ntv3-jax` environment (JAX, for Nucleotide Transformer v3 only)

## TL;DR
NTv3 (InstaDeep, released Dec 2025) is **not** the model your pipeline currently
uses. T19 targets `nucleotide-transformer-v2-250m-multi-species` (ESM-style,
6-mer tokenization, PyTorch/`transformers` via `nt-tf4x`). NTv3 is an
architecturally different model — a U-Net-style network with **single-base
tokenization**, up to 1 Mb context, and it ships as a **JAX** package
(`nucleotide_transformer_v3`), not a HuggingFace `transformers` model class.
So it needs a **third**, independent env: `ntv3-jax`. It is not a drop-in
replacement for T19 — nothing in your Snakemake graph currently calls it.

Model card / code: https://github.com/instadeepai/nucleotide-transformer/blob/main/docs/nucleotide_transformer_v3.md
Paper: https://www.biorxiv.org/content/10.64898/2025.12.22.695963v1
Weights: `InstaDeepAI/NTv3_8M_pre`, `NTv3_100M_pre`, `NTv3_650M_pre` (pre-trained,
MLM only) and `NTv3_100M_post`, `NTv3_650M_post` (post-trained, add functional-track
+ genome-annotation heads, need species conditioning).

---

## 1. Prerequisites
Same HPC4 basics as the main env (see `setup guide` and `T4/DEPLOY.md`):
account active, `igem2026` Slurm account, VPN if off-campus, conda initialised
via
```bash
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
```
before any `conda activate` — otherwise `which python` silently stays on base
Anaconda, same trap as everywhere else on this cluster.

Put the env in `/project/igem2026` (persistent, shared), never `/scratch`
(30-day Lustre eviction would silently delete the env).

## 2. Verify the CUDA build before building (same story as torch)
H800 GPUs, driver max **CUDA 12.8** (`nvidia-smi`, run inside a GPU job — see
`working-on-HPC4` §5). JAX's CUDA-12 pip wheels bundle their own CUDA/cuDNN
runtime the same way `pip install torch --index cu128` does, so you don't need
a system `module load cuda` — but you do need to install the `cuda12` JAX
extra, not plain `jax` (plain `pip install jax` is CPU-only and will silently
run on CPU, same failure mode as the "PyPI torch is CUDA-13" trap in
`DEPLOY.md`).

⚠️ **JAX's own docs now default to recommending the CUDA-13 wheels** ("We
recommend migrating to the CUDA 13 wheels"). Do **not** follow that default
here — the H800 driver's ceiling is 12.8, so a bare `pip install -U jax` or
`jax[cuda13]` would either fail to find a compatible driver or silently
degrade. Always install `jax[cuda12]` explicitly.

## 3. Build (login node, ~10 min)
```bash
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda create -y -p /project/igem2026/envs/ntv3-jax python=3.11 pip
conda activate /project/igem2026/envs/ntv3-jax
export PYTHONNOUSERSITE=1        # same rationale as nt-tf4x: stop ~/.local shadowing the env

# JAX with CUDA 12 support (bundled runtime — matches the H800 driver's 12.8 ceiling).
# Deliberately unpinned on the FIRST install so pip can resolve a mutually
# compatible {jax, jax-cuda12-plugin, jax-cuda12-pjrt} set (these three are
# versioned somewhat independently upstream) — then we pin below.
pip install --no-cache-dir "jax[cuda12]"

# Clone InstaDeep's repo and install the NTv3 package + inference deps:
cd /project/igem2026
git clone https://github.com/instadeepai/nucleotide-transformer.git ntv3-repo
cd ntv3-repo
pip install --no-cache-dir .          # installs nucleotide_transformer_v3 + its listed deps

# extras used by the notebooks / typical inference scripts:
pip install --no-cache-dir huggingface_hub numpy pandas
```

**Freeze immediately after §4's GPU check passes** — same rationale as
`nt-tf4x` pinning torch first and freezing it: JAX resolves fast and without
a pin, two people building this a month apart can land on different
`jax`/`jax-cuda12-plugin`/`jax-cuda12-pjrt` versions with possibly different
behavior on the same driver.

Unlike torch — where `torch==2.11.0+cu128` bakes the CUDA build directly into
one package's version string, so there's nothing separate to drift — JAX's
CUDA runtime ships as its own set of PyPI packages (`nvidia-cudnn-cu12`,
`nvidia-cublas-cu12`, `nvidia-cuda-nvcc-cu12`, etc.), pulled in transitively
by the `cuda12` extra. A grep that only matches `jax`/`jaxlib` prefixes would
pin the JAX-side versions but silently leave the underlying cuDNN/cuBLAS
build free to resolve differently on a rebuild — exactly the drift this step
exists to prevent. The grep below catches both:
```bash
pip freeze | grep -E '^(jax|jaxlib|nvidia-)' > /project/igem2026/envs/ntv3-jax-requirements.lock.txt
```
To rebuild from the lock later (rather than re-resolving):
```bash
pip install --no-cache-dir -r /project/igem2026/envs/ntv3-jax-requirements.lock.txt
```

Group-readability, same hygiene as the main env build:
```bash
chmod -R g+rX /project/igem2026/envs/ntv3-jax 2>/dev/null || true
```

## 4. Verify it works
**Login node — CPU-only import check** (no GPU there, expected):
```bash
python -c "
import jax, nucleotide_transformer_v3
print('jax', jax.__version__)
print('devices (login node, CPU expected):', jax.devices())
"
```

**GPU node — the real check:**
```bash
srun --account=igem2026 --partition=preempt --gpus-per-node=1 --nodes=1 --time=00:10:00 \
  python -c "
import jax
print('backend:', jax.default_backend())      # EXPECT: gpu
print('devices:', jax.devices())              # EXPECT: [CudaDevice(id=0)] or similar
assert jax.default_backend() == 'gpu', 'JAX fell back to CPU — check the cuda12 extra installed cleanly'
"
```
(Don't add `--mem` or CPU flags to the GPU job — same SuperPOD auto-allocation
as every other GPU job here. `--account=igem2026` required.)

## 5. Pre-stage the model weights (compute nodes are offline)
Same pattern as the main env's HF pre-staging (`DEPLOY.md` §7 /
`working-on-HPC4` §7) — fetch on the login node (has internet), cache once,
run jobs offline:
```bash
export HF_HOME=/project/igem2026/hf_cache      # reuse the same shared cache as the rest of the team
huggingface-cli download InstaDeepAI/NTv3_100M_pre
huggingface-cli download InstaDeepAI/NTv3_100M_post     # only if you need functional-track/annotation heads
# larger variant, only if you actually need it (650M):
# huggingface-cli download InstaDeepAI/NTv3_650M_pre
# huggingface-cli download InstaDeepAI/NTv3_650M_post
```
Then in every SBATCH preamble that uses NTv3:
```bash
export HF_HOME=/project/igem2026/hf_cache
export HF_HUB_OFFLINE=1
```

## 6. Minimal inference smoke test (GPU job)
NTv3's own loader (`nucleotide_transformer_v3.pretrained`) pulls straight from
the HF cache once `HF_HOME`/`HF_HUB_OFFLINE` are set — you don't hand it a
local path.
```bash
cat > ntv3_smoke.sbatch <<'EOF'
#!/bin/bash
#SBATCH --account=igem2026
#SBATCH --partition=preempt
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --job-name=ntv3_smoke
#SBATCH --output=%x_%j.out

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate /project/igem2026/envs/ntv3-jax
export HF_HOME=/project/igem2026/hf_cache
export HF_HUB_OFFLINE=1

python -c "
import jax
assert jax.default_backend() == 'gpu', 'NO GPU — aborting'

from nucleotide_transformer_v3.pretrained import get_pretrained_ntv3_model

model, tokenizer, config = get_pretrained_ntv3_model(
    model_name='NTv3_100M_pre',
    embeddings_layers_to_save=(10,),
)

sequences = ['ATTCCGATTCCGATTCCG'.ljust(128, 'N')]   # pad to exactly 128 (7-downsample models need a multiple of 128)
tokens = tokenizer.batch_np_tokenize(sequences)
outs = model(tokens)
print('embeddings_10 shape:', outs['embeddings_10'].shape)
"
EOF
sbatch ntv3_smoke.sbatch
```

## 7. Gotchas specific to NTv3 (don't relearn these the hard way)
- **Sequence length must be divisible by `2^num_downsamples`.** For the main
  8M/100M/650M models that's 7 downsamples → length must be a multiple of
  **128**. The 592M "5-downsample" experimental variants need a multiple of
  32 instead. Pad with `N` (never `[PAD]` — the models weren't trained on it)
  or crop to the nearest valid length.
- **Post-trained models need a species token.** `get_posttrained_ntv3_model`
  requires `species_tokens = model.encode_species('human')` (or whichever
  species) — passing tokens alone without species conditioning is a different
  code path (pre-trained/MLM-only) and won't give you functional-track or
  annotation outputs.
- **Output cropping for post-trained heads.** `bigwig_tracks_logits` and
  `bed_tracks_logits` are cropped to the middle 37.5% of the sequence
  (Borzoi/AlphaGenome-style); the plain MLM `logits` are not cropped. Don't
  compare shapes across these two without accounting for that.
- **bfloat16.** NTv3 was trained end-to-end in bfloat16; both loader functions
  take `use_bfloat16=True` to cut memory. Differences vs full precision are
  reported as small — worth it on H800 if you're memory-constrained at 1 Mb
  context.
- **This is a separate env from `nt-tf4x` on purpose** — JAX and the
  `transformers`/torch stack don't need to coexist and you don't want JAX's
  CUDA wheel bumping anything in the torch envs. Don't try to `pip install`
  this into `promoter-mining-hpc4` or `nt-tf4x`.
- **Not wired into your Snakemake graph.** If/when you decide to actually use
  NTv3 for a task (e.g. as an alternative to T19, or for a long-range
  regulatory-context feature), that's new task code to write — this doc only
  gets the env itself built and verified.
