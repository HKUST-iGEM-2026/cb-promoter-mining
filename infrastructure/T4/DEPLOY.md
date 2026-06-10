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
- A conda/mamba front-end. The welcome email uses `source activate <env>`, so a conda base
  is available — check `module avail` for the exact module name (don't assume `miniforge3`).
  Containers (Enroot/Singularity) and Spack are the cluster-blessed alternatives if conda
  is awkward.
- Put the env in **`/project/igem2026`** — the 5 TB group NFS share: persistent and shared
  by all team accounts, so you build once and everyone activates it. Do **NOT** use
  `/scratch` — it's Lustre scratch and **files inactive >30 days are deleted**, which would
  silently break the env. `/home` (200 GB/user) works but isn't shared, so each person would
  rebuild their own copy.

## 2. Verify the CUDA pin *before* building
SuperPOD's GPUs are **H800 (Hopper)**, which want **CUDA 12.x**, so `environment.yml` pins
`pytorch-cuda=12.1`. Confirm SuperPOD actually offers a matching CUDA module:

```bash
module avail cuda      # which CUDA modules exist
nvidia-smi             # driver's max supported CUDA (run inside a GPU job — see §4)
```

Repin if needed — only 11.8 / 12.1 / 12.4 exist as `pytorch-cuda` versions (the old `12.3`
did not). 11.8 runs on H800 but isn't optimal for Hopper; prefer the highest 12.x SuperPOD
provides.

## 3. Build (shared `/project` location)
```bash
module load <conda-module>          # exact name from `module avail` (e.g. miniforge3 / anaconda3)
conda config --set channel_priority strict          # cleaner solve; retry 'flexible' if it conflicts
ENV=/project/igem2026/envs/promoter-mining-hpc4
mamba env create -f environment.yml -p "$ENV"
chmod -R g+rX "$ENV"                # let the whole igem2026 group read/execute it
conda activate "$ENV"
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

to your submit step / SBATCH preamble, alongside `module load cuda` (matching the pinned 12.1).

**T3 profile — already reconciled to SuperPOD (done):** `slurm_account: igem2026` added, the
GPU rules' `mem_mb` removed (SuperPOD auto-allocates RAM per H800), and the per-task SBATCH
headers fixed to `cuda/12.1` + `--partition=preempt`. Those per-task scripts now live with
their owners (`model-training/T18–T21`, `evaluation-and-calibration/T23–T24`,
`mining-and-motif-annotation/T28`), not in T3.

## Open decisions / TODO
- **CUDA pin** — set to `12.1` for the H800 (Hopper) GPUs; confirm SuperPOD's exact CUDA
  module with `module avail cuda` (§2) and repin if it differs.
- **T10 tooling** — the env ships `segemehl` (aligner — confirm with Sine, swap for bowtie2/bwa
  if preferred) and `openjdk` for TSSpredator. **TSSpredator itself is a JAR, not a conda
  package**: download it from the project site and run `java -jar TSSpredator.jar <config>`.
- **torch cap** — pinned `<2.5` because the `pytorch-cuda` metapackage path ends there.
  For newer torch, install from conda-forge or pip wheels instead.
