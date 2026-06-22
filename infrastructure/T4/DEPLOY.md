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
- A conda/mamba front-end. The welcome email uses `source activate <env>`, so a conda base
  is available — check `module avail` for the exact module name (don't assume `miniforge3`).
  Containers (Enroot/Singularity) and Spack are the cluster-blessed alternatives if conda
  is awkward.
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
module load <conda-module>          # exact name from `module avail` (e.g. miniforge3 / anaconda3)
conda config --set channel_priority strict          # cleaner solve; retry 'flexible' if it conflicts
ENV=/project/igem2026/envs/promoter-mining-hpc4
mamba env create -f environment.yml -p "$ENV"
chmod -R g+rX "$ENV"                # let the whole igem2026 group read/execute it
conda activate "$ENV"
# Guard: confirm torch is the cu128 build (CUDA 12.8 = the H800 driver max). The pip section installs
# it; do NOT later run a bare `pip install torch` — PyPI's default is a CUDA-13 build the driver can't
# use (silent CPU fallback). Verify before going further:
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

> ⚠️ **The currently committed `environment.lock.yml` is INVALID — do not install from it.** It mixes a
> CUDA 12.1 conda stack with CUDA-13 pip wheels (`nvidia-*-cu13`, `cuda-toolkit==13.0.2`) plus
> `torch 2.12` / `torchvision 0.27`. That torch needs a newer driver than the H800 nodes have, so it
> silently falls back to CPU. Build from `environment.yml`, then regenerate this lock from the clean env.

Once it solves cleanly, freeze exact versions so every teammate gets an identical env,
and commit the lock:

```bash
conda env export --no-builds > environment.lock.yml     # or: conda-lock -f environment.yml
git add environment.yml environment.lock.yml DEPLOY.md
```

The committed `environment.yml` stays loosely pinned (readable); `environment.lock.yml`
is the exact, reproducible resolve everyone installs from.

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
huggingface-cli download neuralbioinfo/prokbert-mini-long                     # base for T18 (PT5 uses …-long-promoter)
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
- **Regenerate the lock** — `environment.lock.yml` is a placeholder; build on-cluster from
  `environment.yml`, then `conda env export --no-builds > environment.lock.yml` to capture the real lock.
