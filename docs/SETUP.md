# Setup

What you need to install, how the environments are split, and how to run the pipeline on your own
machine or cluster. Nothing here assumes a particular site.

**If you only want to score sequences**, you do not need any of this. The three trained models are on
Hugging Face and load in a few lines; see [MODELS.md](MODELS.md).

---

## 1. Get the code and set the environment variables

Large inputs are stored with **Git LFS**, so install it before you clone.

```bash
git lfs install
git clone <this repo> && cd cb-promoter-mining
git lfs pull
source env.sh
```

`env.sh` sets a handful of variables that every script reads. Export your own values before sourcing
it and yours win.

| Variable | Default | What it is |
|---|---|---|
| `PROMOTER_REPO` | this clone | repo root, resolved from `env.sh`'s own location |
| `PROMOTER_RESULTS` | `$PROMOTER_REPO/infrastructure/T3/snakemake_scaffold/results` | where the Python helpers look for checkpoints and outputs. The pipeline itself always writes to that `results/` folder, so leave this alone unless you moved it |
| `ENV_MAIN` | `promoter-mining` | the main conda env |
| `ENV_EVO2` | `promoter-evo2` | the Evo2 probe |
| `ENV_MEME` | `promoter-meme` | MEME suite / FIMO |
| `ENV_NT` | `promoter-nt` | Nucleotide Transformer only, which is **not** in the deployed ensemble |
| `HF_HOME` | Hugging Face default | set it to share one model cache |
| `HF_HUB_OFFLINE` | `0` | set to `1` to require a pre-staged cache |

The env variables hold **names** by default, so `conda activate "$ENV_MAIN"` works anywhere. Set them
to absolute prefixes instead if you install into a shared location:

```bash
export ENV_MAIN=/shared/envs/promoter-mining
```

Source `env.sh` in every new shell and before every `sbatch`. The shell and job scripts stop with
`set it or source env.sh` if you forget. Python scripts fall back to paths resolved from their own
location, so they also work without it.

## 2. Install the environments

The stack is split because the members need mutually incompatible library versions. Do not try to
merge the envs. You need the **main** env for almost everything; add the others only for the parts you
actually run.

The main env pins CUDA 12.8 builds of torch, which are published for Linux only. They also run on a
Linux machine with no GPU.

```bash
conda env create -f infrastructure/conda-environments/environment.yml -n promoter-mining
conda activate promoter-mining
```

Then bake in three variables so every activation gets them. They are not captured by a lock file, so
recreate this after any rebuild:

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/zz_env_fixes.sh" <<'EOF'
export PYTHONNOUSERSITE=1        # ignore ~/.local so a user-site torch/numpy can't shadow the env
export MKL_THREADING_LAYER=GNU   # fixes the Intel oneMKL "Cannot load libtorch_cpu.so" import error
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"   # env libstdc++ first, else scipy
                                 # fails with "CXXABI_1.3.15 not found", which breaks sklearn,
                                 # then transformers, then peft
EOF
conda deactivate && conda activate promoter-mining
```

### GPU build

**Check your driver first.** `nvidia-smi` prints the highest CUDA version the driver supports. If it is
below 12.8, change the two `+cu128` pins and the `cu128` index URL in `environment.yml` to a build
your driver supports, then rebuild.

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Never run a bare `pip install torch` into this env. The default PyPI wheel targets a newer CUDA than
many drivers support, and it fails over to CPU **silently** rather than erroring.

Training and genome scanning want a GPU. Everything else runs on CPU.

### The Evo2 environment

Only needed to run the Evo2 ensemble member. flash-attn is the hard part, not Evo2: it compiles from
source, can take 30+ minutes and several GB of RAM, and needs a C++17 compiler on PATH. Install it
**after** torch with `--no-build-isolation`, which is what lets its build see the installed torch.

```bash
conda create -y -n promoter-evo2 python=3.11 pip
conda activate promoter-evo2
export PYTHONNOUSERSITE=1

pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir flash-attn==2.8.0.post2 --no-build-isolation
pip install --no-cache-dir evo2
pip install --no-cache-dir scikit-learn numpy pandas matplotlib joblib
pip freeze > evo2-requirements.lock.txt
```

Transformer Engine is only needed for the 1B/20B/40B models. For the 7B we use, skip it entirely;
that removes the most fragile dependency. Evo2 weights are Apache-2.0 and ungated, so no login is
needed. Pin hard: `evo2` → `vortex` → `flash-attn` → `torch` is a four-layer stack where every layer
is version-sensitive and the bottom one is expensive to rebuild.

### The MEME environment

FIMO and MEME come from bioconda and conflict with the ML stack, so they live apart:

```bash
conda create -y -n promoter-meme -c bioconda -c conda-forge meme
```

This env has **no pandas**. Run FIMO there and parse the output in the main env.

### The TSS environment (optional)

Only needed to regenerate the *C. difficile* TSS calls from raw dRNA-seq reads
(`data-acquisition-and-curation/cdifficile-drnaseq/run_tss_pipeline.sh`). The trained models do not
need it. It holds sra-tools, segemehl, cutadapt and a Java runtime, and it also carries MEME, so you
can point `ENV_MEME` at it instead of building both.

```bash
conda env create -f infrastructure/conda-environments/environment-bio.yml   # -> promoter-mining-bio
```

TSSpredator itself is a Java JAR, not a conda package. Stage 5 of `run_tss_pipeline.sh` prints the
download command if the JAR is missing.

### The Nucleotide Transformer environment

Only needed to reproduce the NT result, which is **not** in the deployment ensemble. NT ships custom
remote code written for transformers 4.x and does not run on the main env's 5.x: the v5 rewrite
breaks it in at least six places, one of them **silent** — the model appears to load but a subset of
weights is re-randomised, so it trains garbage for hours without ever crashing. Running NT on the
version its code was written for is the sound fix.

```bash
conda create -y -n promoter-nt python=3.11 pip
conda activate promoter-nt
export PYTHONNOUSERSITE=1
pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
printf 'torch==2.11.0\ntorchvision==0.26.0\n' > /tmp/con-nt.txt   # freeze torch
pip install --no-cache-dir -c /tmp/con-nt.txt \
  "transformers==4.57.6" "peft==0.17.1" "huggingface_hub>=0.34.0,<1.0" \
  "tokenizers>=0.22.0,<=0.23.0" "accelerate>=0.26.0,<1.11" "safetensors>=0.4.3" \
  "scikit-learn>=1.3" "pandas>=2.0" "pyarrow>=14" "numpy>=1.26,<2.3" "optuna>=4.0"
```

The cross-env conflict is `huggingface_hub` and `tokenizers`, not torch: 4.57 needs hub <1.0 while
the main env has hub 1.x. Loading NT through transformers' native `EsmForSequenceClassification` is
also impossible, because NT-v2 uses a gated SwiGLU feed-forward whose shapes native ESM does not
expect.

The NTv3 experiment, also benched, fine-tunes through the main env. Only
`dataset-construction/negative-tokenization/encode.py`'s NTv3 path needs InstaDeep's separate JAX
package; install it from the
[nucleotide-transformer repo](https://github.com/instadeepai/nucleotide-transformer) into its own env.

### Locking the main env

Build from `environment.yml`, **never** install straight from a lock file. A `conda env export` lock
is only a version snapshot and silently drops two pip sources this env needs. If you regenerate one,
patch both back: the torch index URL, and the `prokbert` git source that export records as a bare
version that is not on PyPI.

## 3. Check the data

Without `git lfs pull` you get ~130-byte pointer files and everything that reads them fails. Sanity
check: the Bakta annotation archive is a few hundred MB, not 130 bytes.

Two scripts report where you stand:

```bash
bash infrastructure/T3/snakemake_scaffold/check_state.sh   # what has run, what is missing
bash infrastructure/T3/snakemake_scaffold/verify_env.sh    # env, data integrity, a training smoke test
```

Genome assemblies are fetched from NCBI by the staging scripts, which need `ncbi-datasets-cli` (in
the main env) and internet access: Phase A downloads three, Phase B two, Phase D one (*C. autoethanogenum*, `prep_cau_regimeC.sh`), and the mining prep one.
Every published TSS table the pipeline uses is committed, so nothing is downloaded by hand.

## 4. Running it

### Without a scheduler

Every step is a plain Python or shell script, so you can run the pipeline directly. Snakemake
coordinates Phase A and the start of Phase B; the rest are the commands in
[PIPELINE.md](PIPELINE.md). Always name a target:

```bash
cd infrastructure/T3/snakemake_scaffold
snakemake -n -s local_prep.smk                        # dry run: plans Phase A, runs nothing
snakemake --cores 4 -s local_prep.smk                 # Phase A
snakemake --cores 8 results/T14/negative_set.tsv      # Phase B up to the negatives
```

A clean dry run validates **wiring only** and executes nothing. Do not read it as "the pipeline
works". Don't run the default target (plain `snakemake`): it needs the optional *C. difficile*
track's output and reaches old training-template rules.

The `.sbatch` job scripts are ordinary bash scripts, so on a single GPU machine run them with `bash`
instead of `sbatch`, passing the same environment variables.

The full 166-genome dataset build is happier with a few cores and a lot of RAM.

### With Slurm

Source `env.sh` in the shell you submit from. Slurm hands the submitting shell's environment to the
job by default, which is how the job scripts find the repo and the envs.

The `.sbatch` scripts carry no site-specific directives. If your site needs an account or a
partition, set Slurm's own input variables once and every `sbatch` picks them up:

```bash
export SBATCH_ACCOUNT=my-allocation SBATCH_PARTITION=gpu
sbatch mining-and-motif-annotation/genome-scan/mine_3model.sbatch
```

Passing `--account` and `--partition` on the submit line works too.

For the Snakemake profile, fill in the two site fields in
`infrastructure/T3/snakemake_scaffold/profiles/slurm/config.yaml`, then:

```bash
snakemake --workflow-profile profiles/slurm results/T14/negative_set.tsv
```

Both fields default to empty, which works on a cluster that needs neither.

The GPU job scripts request `--requeue --signal=B:USR1@120`, so on a preemptible queue a killed job is
resubmitted and gets two minutes' warning. The scan and sweep steps are resume-safe. On a queue that
never preempts, those flags do nothing.

### Offline compute nodes

Many clusters give compute nodes no internet, so a job that downloads a model mid-run hangs. Fetch
once from a node that has internet into a shared cache, then make the job fail fast on a cache miss:

```bash
export HF_HOME=/shared/hf_cache
hf download neuralbioinfo/prokbert-mini
hf download arcinstitute/evo2_7b          # only if you run the Evo2 member

# then, in the job:
export HF_HOME=/shared/hf_cache HF_HUB_OFFLINE=1
```

Do not set `HF_HUB_OFFLINE=1` while downloading; that makes the download itself fail.

### Results on a small disk

Everything the pipeline produces lands in `infrastructure/T3/snakemake_scaffold/results/`: datasets,
checkpoints, score tracks. If the clone sits on a small disk, make that folder a symlink to a bigger
volume before the first run.

## 5. Gotchas worth knowing

Each of these cost us real time.

- **Silent CPU fallback.** A mismatched CUDA build does not error, it just runs ~50× slower. Assert
  `torch.cuda.is_available()` at the top of every GPU job. One early run burned about six hours this
  way.
- **No GPU on the node you are typing on.** Many clusters have login nodes without GPUs, where
  `torch.cuda.is_available()` is `False`. That is expected. Test CUDA inside a GPU job.
- **A stale `~/.local` shadows the env.** A leftover user-site numpy or torch takes precedence and
  crashes with `_ARRAY_API not found`. `PYTHONNOUSERSITE=1` is the fix and `env.sh` sets it. Do not
  use `python -S`, which also hides the env's own packages.
- **Conda activation that half-works.** On systems where conda is not initialised in your shell, the
  prompt changes but `python` still points at the base install. Run
  `source "$(conda info --base)/etc/profile.d/conda.sh"` first and check `which python`.
- **Git LFS pointers.** A `FileNotFoundError` on a file that clearly exists usually means you skipped
  `git lfs pull`.
- **`git` over SSH can break inside an active env.** Prepending the env's `lib` to
  `LD_LIBRARY_PATH` makes the system `ssh` load the env's OpenSSL, giving `OpenSSL version mismatch`.
  Run git as `LD_LIBRARY_PATH= git …`, or from a shell with no env active.
- **Do not install one-off tools into the ML env.** That is what triggered the `CXXABI_1.3.15`
  breakage for us. Use a throwaway env.
