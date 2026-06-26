# Working on HPC4 (HKUST SuperPOD) — setup guide for everyone

HPC4 **is** the HKUST SuperPOD. This is the day-to-day "how do I get on and run something" guide for any
team member. (Building the shared conda env is a separate one-time job — see [`T4/DEPLOY.md`](./T4/DEPLOY.md).)

**Golden rule:** the login node is for editing, git, and submitting jobs — **never for training**. It has
no GPU, so `torch.cuda.is_available()` is `False` there (expected, not a bug). All real compute goes
through Slurm (§5).

---

## 0. One-time, before your first login

- **Account:** your username is your **HKUST ITSC account** (the part before `@connect`/`@ust`). Oak/Swasti request the SuperPOD + `igem2026` account for you via ITSO; you'll get an email when it's active. If `ssh` is refused, it's most likely not provisioned yet — ask Oak/Swasti.
- **Off campus?** Connect the **HKUST VPN** (GlobalProtect, installed from ITSO) **before** `ssh` — `superpod.ust.hk` isn't reachable from outside without it.
- **Passwordless login (optional, *after* your first password login):** from your laptop run `ssh-keygen -t ed25519` (if you don't already have a key), then `ssh-copy-id <user>@superpod.ust.hk`. After that you won't type a password each time. (You can't do this before your first login — the first one uses your password.)

## 1. Log in

```bash
ssh <your-username>@superpod.ust.hk      # VPN first if off-campus
```

You land on a **login node** (editing, git, job submission, tiny tests only).

## 2. Get the code + data

Everyone uses **one shared clone** on the persistent group share:

```bash
cd /project/igem2026/cb-promoter-mining
git pull                                  # latest code
git lfs version || module load git-lfs    # ensure git-lfs is available
git lfs pull                              # turn LFS pointers into the real zips/csv/xlsx
```

⚠️ **Without `git lfs pull`**, the genome/Bakta zips, the PT5 CSV, and the published-TSS supplements are
just **130-byte pointer files** and anything reading them fails. Always `git lfs pull` after `git pull`.

> **The clone already exists** at `/project/igem2026/cb-promoter-mining` (Oak set it up). If that `cd`
> fails, tell Oak — do **not** `git clone` a second personal copy.
> `module load git-lfs` uses SuperPOD's **Lmod** system: `module avail` lists installable tools and
> `module load <name>` adds one to your session. You rarely need it (the §3 env has your tools) — git-lfs is one case.

## 3. Activate the shared environment — the #1 trap

Everyone shares one prebuilt env. You must **initialise conda first**, or activation silently half-works
(the prompt shows `(promoter-mining-hpc4)` but `python` still points at base Anaconda → `import torch` fails):

```bash
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh    # REQUIRED before activate
conda activate /project/igem2026/envs/promoter-mining-hpc4
which python          # MUST be …/promoter-mining-hpc4/bin/python  (NOT /cm/shared/apps/Anaconda3/…)
```

Confirm it's the real env:

```bash
python -c "import torch, transformers; print('torch', torch.__version__, '| cuda build', torch.version.cuda)"
```

## 4. Make it automatic (do once)

Add the conda-init line to your `~/.bashrc` so every login is ready:

```bash
echo 'source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh' >> ~/.bashrc
```

(You still `conda activate …` per session.)

## 5. Running work — through Slurm, never on the login node

**Editing files on the cluster** (you need this to create the SBATCH script below): easiest is
`nano my_job.sbatch` (arrow keys to move; `Ctrl-O` then Enter to save, `Ctrl-X` to exit). `vim` also exists
(if you get trapped in it, press `Esc` then type `:q!` and Enter to bail). Most comfortable: **VS Code
"Remote - SSH"** — connect to `<user>@superpod.ust.hk` and edit/browse files in a normal GUI. Its terminal
runs on the *login* node, so edit and `sbatch` there, but never run training in the VS Code terminal.

### (a) Quick interactive test (a few minutes)

```bash
srun --account=igem2026 --partition=preempt --gpus-per-node=1 --nodes=1 --time=00:10:00 \
  python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| built for', torch.version.cuda)"
```

### (b) A real job — write an SBATCH script

```bash
cat > my_job.sbatch <<'EOF'
#!/bin/bash
#SBATCH --account=igem2026          # REQUIRED — nothing submits without it
#SBATCH --partition=preempt         # 5x cheaper than 'normal'
#SBATCH --gpus-per-node=1           # GPU jobs only; omit this line for CPU-only work
#SBATCH --time=08:00:00
#SBATCH --job-name=my_job
#SBATCH --output=%x_%j.out          # logs to <jobname>_<jobid>.out

source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate /project/igem2026/envs/promoter-mining-hpc4
export HF_HOME=/project/igem2026/hf_cache HF_HUB_OFFLINE=1     # for GPU/model jobs (see §7)
cd /project/igem2026/cb-promoter-mining

python -c "import torch; assert torch.cuda.is_available(), 'NO GPU — aborting'"   # GPU jobs: fail fast
python your_script.py ...
EOF

sbatch my_job.sbatch                 # submit
```

### (c) Watch / manage it

```bash
squeue --me                          # your queued/running jobs
tail -f my_job_<jobid>.out           # live log (output lands in the dir you ran sbatch from; stderr is merged in)
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,MaxRSS    # after it runs
scancel <jobid>                      # cancel
```

**Reading the State:** `COMPLETED` with `ExitCode 0:0` = finished cleanly. `FAILED` / `OUT_OF_MEMORY` /
`NODE_FAIL` = it broke — read the `.out`. `CANCELLED` = killed by you/admin. `PREEMPTED` / `REQUEUED` =
the preempt partition bumped it (§6). `%x`=job name, `%j`=job ID, so `--output=%x_%j.out` → `my_job_<jobid>.out`.

## 6. Slurm rules (memorise these)

- **`--account=igem2026` on every job** — no exceptions; nothing submits without it.
- **Use `--partition=preempt`** — the cheaper, preemptible partition (use it for checkpointable work; the
  repo profile cites ~5x cheaper / HK$2.4 vs HK$12 per GPU-hr, but SuperPOD pricing was **updated Jan 2026**
  — check the ITSO charging page below for current rates). It can be interrupted: your job
  goes back to PENDING and Slurm **auto-resubmits it** (you don't resubmit), but it **restarts from the
  beginning** unless your script checkpoints — which is why training scripts save/resume checkpoints (the
  `--signal=B:USR1@120` trap gives ~2 min warning to save). For a hand-written `my_job.sbatch` doing long
  checkpointable work, add `#SBATCH --requeue` and `#SBATCH --signal=B:USR1@120`. If you truly can't
  checkpoint, use `--partition=normal`.
- **GPU jobs:** request `--gpus-per-node=1`. Do **NOT** set `--mem` or CPU flags — SuperPOD auto-allocates
  ~28 threads + 8 GB/thread per H800. **Never `--exclusive`** (whole node = 8x cost).
- GPUs are **H800 (Hopper)**, driver max **CUDA 12.8** — the env's torch is built to match.

## 7. GPU + model jobs: pre-stage the checkpoints

Compute nodes have **no internet**, so a job that downloads a HuggingFace model mid-run hangs. Fetch once
on the **login node** (it has internet), then run offline:

```bash
# once, on the login node:
export HF_HOME=/project/igem2026/hf_cache
huggingface-cli download neuralbioinfo/prokbert-mini
huggingface-cli download InstaDeepAI/nucleotide-transformer-v2-250m-multi-species
```

⚠️ When **downloading** (login node), do NOT have `HF_HUB_OFFLINE=1` set in your shell — that would make
the download fail. `HF_HUB_OFFLINE=1` is only for the *job*. (`huggingface-cli` also works as the newer `hf download`.)

Then in every SBATCH preamble: `export HF_HOME=/project/igem2026/hf_cache HF_HUB_OFFLINE=1`.

## 8. Where to put files (storage map)

| Path | Size | Lifetime | Use for |
|---|---|---|---|
| `/project/igem2026` | 5 TB, **shared** | **persistent** | the repo clone, the env, `hf_cache`, **trained models / outputs the team needs** |
| `/home/<user>` | 200 GB | persistent, **private** | your personal scratch; teammates can't see it |
| `/scratch/igem2026` | 5 TB Lustre | **deleted after >30 days idle** | big temporary job I/O only — never the env or anything you want to keep |

⚠️ Put anything the **team or downstream tasks** need (e.g. trained models for T28) under
`/project/igem2026`, **not** `/home` — otherwise only you can see it.

Check your usage with `du -sh ~` (home) and `du -sh /project/igem2026/<your-folder>`; for exact quota
limits see the ITSO quota page (linked at the bottom). A full `/home` surfaces as a cryptic
"Disk quota exceeded" with no other hint.

## 9. Common gotchas (every one has bitten us)

- **Prompt says the env is active but `import torch` fails** → you skipped `source …conda.sh` (§3); `which python` will show base Anaconda.
- **`FileNotFoundError` / 130-byte files** → you skipped `git lfs pull` (§2).
- **Training is mysteriously slow** → it fell back to CPU. Assert `torch.cuda.is_available()` at the top of every GPU job (PT1 once ran ~6 h on CPU this way).
- **Never run a bare `pip install torch`** — PyPI's default is a CUDA-13 build the driver can't use (silent CPU fallback). The env already has the correct `cu128` build.
- **Login node ≠ compute node** — no GPU on login; always test GPU via `srun`/`sbatch`.
- **`mamba: command not found`** — base Anaconda here has no mamba; use `conda` (only relevant if you're rebuilding the env).

## 10. Every-session quickstart (copy-paste)

```bash
ssh <user>@superpod.ust.hk                                              # VPN first if off-campus
cd /project/igem2026/cb-promoter-mining && git pull && git lfs pull
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate /project/igem2026/envs/promoter-mining-hpc4
which python      # sanity: …/promoter-mining-hpc4/bin/python
# …then srun (quick test) or sbatch (real job).
```

---

## Official HKUST docs (when you're stuck beyond this guide)

- **HPC Handbook — start here** (Getting Started, SLURM, software env): https://hkust-hpc-docs.readthedocs.io/latest/index.html
- Logging in: https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod/usage-tips/login
- Submit your first SLURM job: https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod/submit-first-job
- Using SLURM (full reference): https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod/slurm
- Partition & resource quota: https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod/partition
- File & storage (quota limits): https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod/fileandstorage
- Software modules (Lmod): https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod/software-module-system
- Charging / pricing — **updated Jan 2026**: https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod/charging
- VS Code on a compute node (official tip): https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod/usage-tips/vscode

*Building or rebuilding the env, locking it, the CUDA/torch details → see [`T4/DEPLOY.md`](./T4/DEPLOY.md).
Cluster facts (accounts, partitions, storage) are also in the repo root `CLAUDE.md` §4.*
