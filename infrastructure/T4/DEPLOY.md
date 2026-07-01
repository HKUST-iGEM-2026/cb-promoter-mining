# T4 — Deploy the conda/mamba env on HPC4

The spec is [`environment.yml`](./environment.yml) in this folder. This runbook is the
**"+ deploy on HPC4"** half of T4: how to build the env on the cluster, verify it, and
lock it so the whole team runs an identical stack.

**Scope:** everything that runs on HPC4 — Phase B–E, T10 (TSSpredator), and the two GPU
pre-tasks on the cluster, **PT1** (ProkBERT fine-tune) and **PT5** (E. coli sliding-window
inference). PT1/PT5 need no extra packages — they share the torch/ProkBERT stack used by
T18/T28. Phase A (T5 genome download, T6 Bakta) runs locally and is deliberately not in
this env.

## 1. Prerequisites
- HPC4 (= **HKUST SuperPOD**) account active (T1); Slurm account `igem2026`. Log in with
  `ssh <username>@superpod.ust.hk` (off-campus needs the HKUST VPN first).
- **Get the repo + its LFS data.** The genome/Bakta zips, the PT5 CSV, the MasterTable, and the
  published-TSS supplements are Git-LFS objects — a plain `git pull` only fetches ~130-byte pointers.
  On HPC4 you MUST materialise them:
  ```bash
  git lfs version || module load git-lfs   # git-lfs must be available
  cd /project/igem2026/cb-promoter-mining   # the repo clone
  git pull
  git lfs pull                              # turns LFS pointers into the real zips/csv/xlsx
  ```
  Without `git lfs pull`, `config[bakta_dir]` resolves to a 130-byte pointer and `extract_bakta` fails.
  (`config[bakta_dir]` is repo-relative, so it resolves correctly wherever the repo is cloned.)
- A conda front-end. On SuperPOD the base is `/cm/shared/apps/Anaconda3/2023.09-0` (confirm with
  `which conda` / `module avail`). **You must initialise conda in the shell before `conda activate`
  will work** — otherwise it only sets the prompt and leaves `python`/`pip` pointing at the *base*
  Anaconda. This is the #1 trap here: the prompt shows `(promoter-mining-hpc4)` but `which python`
  is the base, so `import torch` → `ModuleNotFoundError` even though the env has torch.
  ```bash
  source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh   # THEN `conda activate …` works
  which python      # must be …/promoter-mining-hpc4/bin/python, not /cm/shared/apps/Anaconda3/…
  ```
  Put that `source` line in your `~/.bashrc` and at the top of every SBATCH preamble. Note **`mamba`
  is not installed on the base Anaconda here** — use `conda env create` (§3) if `mamba: command not found`.
  Containers (Enroot/Singularity) and Spack are the cluster-blessed alternatives if conda is awkward.
- Put the env in **`/project/igem2026`** — the 5 TB group NFS share: persistent and shared
  by all team accounts, so you build once and everyone activates it. Do **NOT** use
  `/scratch` — it's Lustre scratch and **files inactive >30 days are deleted**, which would
  silently break the env. `/home` (200 GB/user) works but isn't shared, so each person would
  rebuild their own copy.

## 2. Verify the CUDA build *before* building
SuperPOD's GPUs are **H800 (Hopper)**; the driver maxes at **CUDA 12.8** (confirmed via `nvidia-smi`).
torch is installed **via pip from the `cu128` index** (`environment.yml` pins `torch==2.11.0+cu128` /
`torchvision==0.26.0+cu128`) — there is **no conda `pytorch-cuda` pin** any more. Confirm the driver first:

```bash
nvidia-smi             # driver's max CUDA (run inside a GPU job — see §4). Must be >= 12.8 for cu128.
module avail cuda      # informational only; torch ships its own CUDA runtime, the system module need not match
```

If the driver maxes below 12.8, repin to a lower index (cu126/cu124) — but **never** install the default
PyPI torch, which is a CUDA-13 build the driver can't run (silent CPU fallback). See CLAUDE.md §9.

## 3. Build (shared `/project` location)
```bash
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh   # init conda (§1) — REQUIRED
conda config --set channel_priority strict          # cleaner solve; retry 'flexible' if it conflicts
ENV=/project/igem2026/envs/promoter-mining-hpc4
# mamba isn't on the base Anaconda here -> use conda with the libmamba solver (fast). If you DO have
# mamba, `mamba env create -f environment.yml -p "$ENV"` also works.
conda env create --solver=libmamba -f environment.yml -p "$ENV"
chmod -R g+rX "$ENV" 2>/dev/null || true   # group read/exec. The `2>/dev/null || true` is REQUIRED:
                                           # conda hardlinks some files from whoever first populated the
                                           # package cache, so "Permission denied" on those .pyc files is
                                           # EXPECTED and harmless (they're already group-readable).
conda activate "$ENV"

# Post-build hygiene — bake two env vars in so EVERY activation (and SBATCH job) gets them. These are
# NOT captured by the lock, so you MUST recreate this file after any rebuild:
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/zz_env_fixes.sh" <<'EOF'
export PYTHONNOUSERSITE=1        # ignore ~/.local so a stray user-site torch/numpy can't shadow the env
export MKL_THREADING_LAYER=GNU   # fixes "Intel oneMKL FATAL ERROR: Cannot load libtorch_cpu.so"
                                 # (conda's MKL/Intel-OpenMP vs the pip-torch GNU-OpenMP clash on import)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"   # env libstdc++ FIRST — else the older SYSTEM
                                 # /lib/.../libstdc++.so.6 is loaded and scipy's _distance_pybind fails with
                                 # "CXXABI_1.3.15 not found" -> breaks sklearn -> transformers -> peft import
EOF
conda deactivate && conda activate "$ENV"   # re-activate so the two vars take effect

# Guard: confirm torch is the cu128 build (CUDA 12.8 = the H800 driver max) AND imports cleanly. The pip
# section installs it; do NOT later run a bare `pip install torch` — PyPI's default is a CUDA-13 build the
# driver can't use (silent CPU fallback). NB run `conda env create` (not pip) so torch lands in the ENV,
# not ~/.local. Verify before going further:
python -c "import torch; assert torch.version.cuda=='12.8', torch.version.cuda; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
```

`-p <prefix>` puts the env in the shared project path; the `name:` field is ignored when
you use `-p`. Tell teammates to `conda activate /project/igem2026/envs/promoter-mining-hpc4`.

## 4. Verify it works
On the login node the imports work but `torch.cuda.is_available()` is `False` — that is
expected (no GPU on the login node). Test the GPU **inside a job**:

```bash
# CPU smoke test (login node)
python -c "import torch, transformers, snakemake; print('torch', torch.__version__)"
which cd-hit meme fimo segemehl java snakemake   # TSSpredator itself is a separate JAR
snakemake --version

# GPU test (must run on a GPU node)
srun --account=igem2026 --partition=preempt --gpus-per-node=1 --nodes=1 --time=00:05:00 \
  python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| built for', torch.version.cuda)"
```

(Don't add `--mem` or CPU flags to GPU jobs — SuperPOD auto-allocates ~28 threads and
8 GB/thread per H800. `--account=igem2026` is required on every job.)

Then dry-run the T3 scaffold against the env to confirm wiring:

```bash
cd ../T3/snakemake_scaffold
snakemake -n --workflow-profile profiles/slurm
```

## 5. Lock it for reproducibility (the whole point of T4)

> ⚠️ **Always build from `environment.yml`, never install straight from `environment.lock.yml`.** A
> `conda env export` lock is only a version *snapshot* — it silently **drops two pip sources** this env
> needs, so a bare lock fails to rebuild. After exporting you MUST patch them back (sed below): the cu128
> index (`torch==2.11.0+cu128` lives only there, not on PyPI) and the prokbert git URL. Regenerate the
> lock only from a freshly built env that imports torch cleanly and shows `True 12.8` on a GPU node.

Once it solves cleanly, freeze versions, patch the two dropped sources, and verify it's clean before committing:

```bash
conda env export --no-builds > environment.lock.yml     # or: conda-lock -f environment.yml (handles pip sources properly)
LOCK=environment.lock.yml
# 1. re-add the cu128 index (else a rebuild can't find torch==2.11.0+cu128):
grep -q extra-index "$LOCK" || sed -i '/- pip:/a\    - --extra-index-url https://download.pytorch.org/whl/cu128' "$LOCK"
# 2. restore prokbert's git source (export records it as a bare ==0.0.50 that isn't on PyPI):
sed -i 's#prokbert==0.0.50#prokbert @ git+https://github.com/nbrg-ppcu/prokbert.git#' "$LOCK"
# 3. sanity: a clean lock has NO pytorch-cuda line (that = the old CUDA-12.1 frankenstein -> rebuild):
grep -q pytorch-cuda "$LOCK" && echo "STILL DIRTY — rebuild from environment.yml" || echo "lock clean"
git add environment.yml environment.lock.yml DEPLOY.md
```

The committed `environment.yml` stays loosely pinned (readable); `environment.lock.yml`
is the version snapshot (with the two pip sources patched back in).

## 6. Wire into the pipeline
The T3 scaffold assumes the env is already **active** before
`snakemake --workflow-profile profiles/slurm`. Add

```bash
conda activate /project/igem2026/envs/promoter-mining-hpc4
```

to your submit step / SBATCH preamble. (torch ships its own cu128 runtime, so a matching `module load cuda` is optional.)

**T3 profile — already reconciled to SuperPOD (done):** `slurm_account: igem2026` added, the
GPU rules' `mem_mb` removed (SuperPOD auto-allocates RAM per H800), and the per-task SBATCH
headers fixed to `cuda/12.8` + `--partition=preempt`. Those per-task scripts now live with
their owners (`model-training/T18–T21`, `evaluation-and-calibration/T23–T24`,
`mining-and-motif-annotation/T28`), not in T3.

## 7. Pre-stage the model weights (offline compute nodes)
SuperPOD compute nodes usually have **no internet**, but ProkBERT and Nucleotide Transformer
pull their checkpoints from HuggingFace at run time — so a GPU job will hang or fail trying to
download mid-run. Fetch them once on the **login node** (which has internet) into a shared
cache, then run jobs offline:

```bash
# once, on the login node (use the exact model IDs your scripts reference):
export HF_HOME=/project/igem2026/hf_cache
huggingface-cli download neuralbioinfo/prokbert-mini                     # base for T18 (PT5 uses prokbert-mini-promoter)
huggingface-cli download InstaDeepAI/nucleotide-transformer-v2-250m-multi-species   # base for T19
```

Then in every SBATCH preamble:
```bash
export HF_HOME=/project/igem2026/hf_cache
export HF_HUB_OFFLINE=1     # fail fast instead of hanging if a file is missing from the cache
```

The custom **CNN has nothing to download** — it's trained from scratch on the cluster. Save the
fine-tuned ProkBERT/NT checkpoints (T18/T19) and the trained CNN (T20) to `/project/igem2026`
too, so every node can read them.

## Open decisions / TODO
- **CUDA build** — torch is `2.11.0+cu128` (pip, cu128 index) for the H800 driver (max CUDA 12.8);
  no conda `pytorch-cuda`. Confirm the driver with `nvidia-smi` (§2); drop to cu126/cu124 only if it's lower.
- **T10 tooling** — `segemehl` / `sra-tools` / `meme` moved to the companion `environment-bio.yml`
  (they conflict with the ML stack on icu/libcurl). **TSSpredator is a JAR, not a conda package**:
  download it and run `java -jar TSSpredator.jar <config>` (the bio env ships `openjdk`).
- **Lock** — regenerate per §5 from a freshly built env (don't trust an old committed lock); always
  re-apply the two pip-source patches (cu128 index + prokbert git URL) and confirm no `pytorch-cuda` line.
- **Activate.d hygiene is not in the lock** — after any rebuild, recreate
  `$CONDA_PREFIX/etc/conda/activate.d/zz_env_fixes.sh` (`PYTHONNOUSERSITE=1` + `MKL_THREADING_LAYER=GNU` +
  `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:…`, §3), or torch fails to import / `~/.local` shadows the env / scipy hits
  **`CXXABI_1.3.15 not found`** (the system libstdc++ shadows the env's newer one — surfaced 2026-07 after a
  `conda install` into the shared env; corollary: **don't install one-off CLIs like `datasets` into the ML env** —
  use a throwaway env).
