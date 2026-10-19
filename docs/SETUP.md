# Setup

Use [env.sh](../env.sh) and the [conda recipes](../environments/) for the deployed workflow. Root `uv sync` installs the Python ≥3.13 pandas/tabular helper environment, not the Linux/CUDA/bioinformatics stack. CPU classifier scoring needs compatible packages and recovered weights; Evo2 DNA inference also needs its NVIDIA/CUDA runtime and backbone. The [DH5α SVR](ECOLI_STRENGTH_PREDICTIONS.md#reproduction) has a separate pinned CPU environment.

If you only want sequences, tables or reported numerical results, start with the [public download index](../results/README.md); no ML environment or login is needed. All four project Hub packages are public and ungated. [Model loading](MODELS.md#public-hugging-face-repositories) pins retained package revisions and explains remote-code/pickle safety. Public download does not waive model, upstream-code or data licence restrictions.

## Paths

```bash
git lfs install
git clone https://github.com/HKUST-iGEM-2026/cb-promoter-mining.git
cd cb-promoter-mining
git lfs pull
source env.sh
```

| Variable | Default | Contract |
|---|---|---|
| `PROMOTER_REPO` | directory containing `env.sh` | repository root |
| `PROMOTER_RESULTS` | `$PROMOTER_REPO/workflow/results` | default for helpers that read it; several job scripts hard-code the workflow result root, and Snakemake uses relative `results/` |
| `ENV_MAIN` | `promoter-mining` | main ML/data environment |
| `ENV_EVO2` | `promoter-evo2` | frozen Evo2 embedding/probe environment |
| `ENV_MEME` | `promoter-meme` | MEME/FIMO environment |
| `ENV_NT` | `promoter-nt` | historical NT-v2 only |
| `HF_HOME` | Hub default unless already set | shared model cache |
| `PYTHONNOUSERSITE` | `1` | avoid user-site packages shadowing the environment |


Source `env.sh` in each shell and before submission. It preserves overrides; `HF_HUB_OFFLINE` is not set automatically. Environment variables accept names or installed prefixes. Snakemake still uses relative `workflow/results/`; overriding PROMOTER_RESULTS alone does not move every job output.

## Main environment

The main recipe targets Linux/Python 3.11, with CUDA 12.8 torch 2.11.0+cu128 and torchvision 0.26.0+cu128. Transformers is a range ≥4.56,≤5.3.0, not an exact pin. It is not a native macOS/Apple-GPU recipe.

```bash
conda env create -f environments/environment.yml -n promoter-mining
conda activate promoter-mining
```

Check GPU availability inside the allocation:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

If changing PyTorch builds, change both pins and the wheel-index URL. For documented Linux library conflicts, this optional activation hook avoids user-site shadowing:

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/zz_env_fixes.sh" <<'EOF'
export PYTHONNOUSERSITE=1
export MKL_THREADING_LAYER=GNU
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
EOF
conda deactivate
conda activate promoter-mining
```

Prioritising conda libraries can affect SSH/OpenSSL; use an unactivated shell if needed. `python -S` also hides required packages and is not equivalent to PYTHONNOUSERSITE.

## Additional environments

### Evo2

Install torch before building FlashAttention. Match the CUDA compiler, driver and GPU to upstream requirements.

```bash
conda create -y -n promoter-evo2 python=3.11 pip
conda activate promoter-evo2
export PYTHONNOUSERSITE=1
pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir flash-attn==2.8.0.post2 --no-build-isolation
pip install --no-cache-dir evo2
pip install --no-cache-dir scikit-learn==1.9.0 numpy pandas matplotlib joblib
pip freeze > evo2-requirements.lock.txt
```

The trusted committed probe records scikit-learn 1.9.0. Evo2/Vortex and other dependencies are not fully pinned here; retain the solved stack and backbone revision. Follow [upstream installation](https://github.com/ArcInstitute/evo2) for the selected backbone. The project package contains the probe, not foundation weights.

### MEME/FIMO

```bash
conda create -y -n promoter-meme -c conda-forge -c bioconda meme
```

Motif commands run here; pandas/numpy parsing runs in ENV_MAIN.

### Optional raw-TSS track

```bash
conda env create -f environments/environment-bio.yml
```

This creates promoter-mining-bio with SRA tools, segemehl, cutadapt, Java and MEME. TSSpredator's JAR is separately staged; see the [driver](../pipeline/acquisition/cdifficile-drnaseq/run_tss_pipeline.sh). It is not needed for deployed positive-table reconstruction or scoring.

### Historical NT-v2

Keep this non-deployed Transformers 4.x path separate from the main stack:

```bash
conda create -y -n promoter-nt python=3.11 pip
conda activate promoter-nt
export PYTHONNOUSERSITE=1
pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
printf 'torch==2.11.0\ntorchvision==0.26.0\n' > /tmp/con-nt.txt
pip install --no-cache-dir -c /tmp/con-nt.txt \
  "transformers==4.57.6" "peft==0.17.1" "huggingface_hub>=0.34.0,<1.0" \
  "tokenizers>=0.22.0,<=0.23.0" "accelerate>=0.26.0,<1.11" "safetensors>=0.4.3" \
  "scikit-learn>=1.3" "pandas>=2.0" "pyarrow>=14" "numpy>=1.26,<2.3" "optuna>=4.0"
```

NTv3 uses a separate main-stack supervised path; optional JAX encoding needs [upstream setup](https://github.com/instadeepai/nucleotide-transformer).

## Stage inputs

Materialise LFS archives, not pointer files. Zero-byte workflow inputs/annotation TSVs are placeholders, not prepared datasets. [Pipeline](PIPELINE.md#phase-a--acquire-and-curate) gives explicit staging and split commands; do not use the legacy default DAG as an end-to-end recipe.

```bash
bash workflow/check_state.sh
bash workflow/verify_env.sh
```

check_state.sh reports sizes/missing outputs but still suggests an obsolete T15.done target. verify_env.sh performs integrity/model smoke work and is not read-only. Inspect its prerequisites before running.

Recovered checkpoints under `workflow/results/T21_xs/` are ignored by git. A fresh clone can stage the public Hub packages anonymously using the [checkpoint commands](MODELS.md#local-checkpoints). Large original splits/encodings/tracks live separately under `~/cb-promoter-original-run/` and are not supplied by a public model download. The [166-assembly manifest](../results/training_sources.json) and [public output index](../results/README.md) expose provenance and retained outputs, not all reconstruction inputs. [Model evidence](MODELS.md#verified-original-data-and-encodings) records historical alignment and saved-feature evaluation. Optional original calibration inputs and T22_xs predictions remain unavailable; neither is needed for raw ranks.

Retain exact tokenizer/code revisions, solved dependencies and source licences. Do not load untrusted pickles or executable remote code. Recovery does not repair the 888 wrong-assembly training labels.

## Run jobs

```bash
cd "$PROMOTER_REPO/workflow"
conda activate "$ENV_MAIN"
snakemake -n -s local_prep.smk
snakemake --cores 4 -s local_prep.smk
snakemake --cores 8 results/T14/negative_set.tsv
```

T11–T14 are localrules even with the Slurm profile; launch from a site-approved CPU host/allocation:

```bash
snakemake --workflow-profile profiles/slurm results/T14/negative_set.tsv
```

Submit GPU jobs separately, or use bash on a compatible Linux GPU machine:

```bash
export SBATCH_ACCOUNT=my-allocation SBATCH_PARTITION=gpu
sbatch pipeline/mining/genome-scan/mine_3model.sbatch
```

Submission is not completion. Keep one mine job per tag; use fresh paths when changing assembly, checkpoints or grid. Requeue support depends on scheduler policy, and cached files still need content checks.

## Offline jobs

```bash
export HF_HOME=/shared/hf_cache
hf download neuralbioinfo/prokbert-mini
hf download arcinstitute/evo2_7b
# In the job, after staging (not during download):
export HF_HOME=/shared/hf_cache HF_HUB_OFFLINE=1
```

Also stage the public project classifiers at the [documented pinned revisions](MODELS.md#public-hugging-face-repositories), reviewed remote code/dependencies and genomic inputs. Separate upstream-backbone conditions still apply. Offline mode cannot supply missing cache entries. Keep FASTA, annotation, BED and truth contig/version identifiers identical.

## Documents

Public source documentation and retained downloads live in `docs/` and `results/`; local wiki sources/exports are separate and are not reconstruction prerequisites. Do not link public repository docs into ignored local directories. [Source publication instructions](../README.md#publish-source-updates) describe scoped staging from the clean documentation branch and a PR to `main`; Hugging Face cards require a separate upload.

Compile TeX from each source directory; the exercised toolchain was Tectonic 0.17.0 with this bundle:

```bash
cd "$PROMOTER_REPO/pipeline/evaluation"
tectonic --untrusted --bundle https://data1b.fullyjustified.net/tlextras-2022.0r0.tar species_metrics_report.tex
tectonic --untrusted --bundle https://data1b.fullyjustified.net/tlextras-2022.0r0.tar xs_evaluation_report.tex
```

First use needs network to populate the TeX cache. Compile other reports to scratch unless updating a delivered PDF. Compilation does not rerun scientific evaluations; new reporting cohorts need their own provenance.
