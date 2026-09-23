# CLAUDE.md — run T17 + T18 + T19 (ProkBERT + NT-LoRA fine-tunes), step by step (for Sanaa)

> **Cluster setup (login, env, SLURM):** see `infrastructure/HPC4_GUIDE.md`. The env is **verified working on
> HPC4 (2026-07)**: torch 2.11+cu128, transformers/peft/sklearn import cleanly, the `libstdc++` fix is baked
> into `activate.d`, and the **H800 GPU is visible** (`CUDA True`). Re-confirm any time with
> `bash infrastructure/T3/snakemake_scaffold/verify_hpc4.sh`.

You own **T17** (finalise the fine-tuning recipe), **T18** (ProkBERT-mini), and **T19** (NT v2 250M + LoRA).
The two training scripts are **written, contract-validated, and their model-loading path is verified on HPC4**
— your job is to **pre-stage the checkpoints, run both on a GPU, verify, then hand to the T21 sweep** (which
is *how* you finalise the T17 recipe). Everything runs on the current T16 — the **extended, weight-corrected**
dataset: T7b is **5,759** in-genus positives, val/test are **verified-only**, train is balanced **~50/50**
(train ≈ 40,442 rows; val/test ≈ 3,323 each; val prevalence ≈ 29%).

## The two contracts the scripts honor (don't break them)
- **Weighted loss.** A `WeightedTrainer` applies the T15 per-sample `weight` (silver 0.3 × the 50/50
  class-balance factor) as a per-sample loss weight — the AT-trap guard. Both scripts do this.
- **Selection on validation AUPRC** (CLAUDE.md §7) + the **sweep CLI** (`--sweep-trial`/`--output-dir`/`--<hparam>`)
  so T21 can drive them. They write `metrics.json{val_auprc,…}` and print `Validation AUPRC: <x>`.

## Step 1 — Code + env
```bash
cd /project/igem2026/cb-promoter-mining && git pull
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate /project/igem2026/envs/promoter-mining-hpc4   # just works — the libstdc++ fix is in activate.d
```
(Optional pre-flight: `bash infrastructure/T3/snakemake_scaffold/verify_hpc4.sh` — checks env + data integrity + a real prokbert load.)

## Step 2 — Pre-stage the HF checkpoints on the LOGIN node (compute nodes are offline)
```bash
export HF_HOME=/project/igem2026/hf_cache
huggingface-cli download neuralbioinfo/prokbert-mini                                # likely already cached from the verify run
huggingface-cli download InstaDeepAI/nucleotide-transformer-v2-250m-multi-species   # needed for T19
```
NB NT ships `pytorch_model.bin` only (no safetensors) — the env's torch 2.x loads it fine. prokbert-mini
ships custom code → the script loads it with `trust_remote_code=True` (**already coded + verified loading on HPC4**).

## Step 3 — Read the two scripts (own them)
`model-training/T18/train_prokbert.py`, `T19/train_nt_lora.py`. Note the `WeightedTrainer`, `compute_metrics`
(AUPRC/AUROC/F1), the CLI flags, and where each saves (`best_model/` for T18; `best_adapter/` LoRA for T19).

## Step 4 — Run T18 (ProkBERT) as a GPU job
Smaller/faster model. Submit a SLURM GPU job (**no `--mem`/CPU flags on GPU jobs**):
```bash
cat > run_t18.sbatch <<'EOF'
#!/bin/bash
#SBATCH --account=igem2026
#SBATCH --partition=preempt
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=t18_prokbert
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate /project/igem2026/envs/promoter-mining-hpc4
export HF_HOME=/project/igem2026/hf_cache HF_HUB_OFFLINE=1
cd /project/igem2026/cb-promoter-mining/infrastructure/T3/snakemake_scaffold
python -c "import torch; assert torch.cuda.is_available(), 'NO GPU'"
python ../../../model-training/T18/train_prokbert.py --data-dir results/T16/prokbert --output-dir results/T18
EOF
sbatch run_t18.sbatch
```
Expect per-epoch eval lines, then `Validation AUPRC: 0.xx`, `results/T18/metrics.json`, `results/T18/best_model/`.
(Outputs land under the scaffold's `results/` on the shared `/project` clone — same place as T16.)

## Step 5 — Run T19 (NT-LoRA) as a GPU job
NT-250M is bigger → LoRA + gradient checkpointing ON, batch 8, runs **longer (a few hours)**. NB T16 tokenizes
NT to **67 tokens** per 81 bp window (vs ProkBERT's 78) — higher than a typical 6-mer NT tokenization (~15–18);
if it looks off, check with Leo before this long run (it's a ~4× compute factor, not a correctness bug — the
template handles any length). Same header, more `--time`:
```bash
# run_t19.sbatch — identical header but: #SBATCH --time=12:00:00  / --job-name=t19_nt
python ../../../model-training/T19/train_nt_lora.py --data-dir results/T16/nt --output-dir results/T19
```
Expect `print_trainable_parameters()` ~1% trainable, then `Validation AUPRC: …`, `results/T19/metrics.json`,
`results/T19/best_adapter/` (the LoRA adapter only — to use it, load base NT + this adapter).

## Step 6 — VERIFY both (don't skip)
```python
import json
for t in ("T18", "T19"):
    m = json.load(open(f"results/{t}/metrics.json"))
    print(t, "val AUPRC", round(m["val_auprc"], 3), "| weighted:", m["weighted"], "| test AUPRC", round(m["test_auprc"], 3))
```
Pass criteria:
- **`weighted == True`** — the silver guard ran (NOT an `--no-weight` ablation).
- **val AUPRC well above ~0.29** — the no-skill baseline = the val positive prevalence (969/3323 ≈ 29%).
  At/near 0.29 means it didn't learn — investigate.
- It should clearly **beat the weak off-the-shelf baseline** (PT5/PT6 ≈ chance — see `WORKFLOW_RATIONALE.md`).
- Training loss decreased across epochs (check the SLURM `.out` log).

## Step 7 — Hand to T21 (this IS finalising the T17 recipe)
Once both run clean standalone, T21 (`model-training/T21/hyperparameter_sweep.py`) drives them — it appends
`--sweep-trial N --output-dir <dir> --<hparam> …` and reads `val_auprc` back to maximise. The template defaults
are the **starting** recipe (T17); the sweep tunes them to the **final** recipe. You don't modify T18/T19 for
the sweep — they already honor the CLI contract. (T20/CNN is Yany's; she's aligning it to the same contract.)

## Gotchas (read these — current as of 2026-07)
- **transformers 5.3.0 — the breaking changes are already found + fixed + validated (2026-07).** The templates
  were written against 4.x; the v5 migration guide was checked line-by-line and the real issue caught: `warmup_ratio`
  was **removed** in v5 → the scripts now feature-detect the surviving warmup arg (`_warmup_kwargs`); `compute_metrics`
  is now robust to v5's `EvalPrediction`; precision is **bf16** on the H800. Every other `TrainingArguments`/`Trainer`
  arg was confirmed against the v5 removed-args list. So run #1 should NOT throw on a Trainer arg — if it does, note it + tell Oak.
- **T19's ONE untested path: NT + LoRA binding on transformers 5.3.0.** `verify_hpc4.sh` §E only exercised *prokbert*
  loading. NT's LoRA targets (`query`/`value`) + `modules_to_save=["classifier"]` are stable ESM names so it should
  bind, but that's settled only on the first real T19 run. If `get_peft_model` errors, dump
  `[n for n,_ in model.named_modules()]` and adjust the target/save names (~2-min fix). A short `--sweep-epochs 1`-style
  run surfaces it in minutes.
- **ProkBERT head — VERIFIED (2026-07).** T18 loads prokbert-mini via `AutoModelForSequenceClassification` +
  `trust_remote_code=True` (fresh 2-class head); `verify_hpc4.sh` proved it loads, emits `(N, 2)` logits, and
  computes the weighted loss on real T16 rows. (Only if a future model-cache change breaks the load: the
  `prokbert` package's own sequence-classification class is the fallback.)
- **Preempt + no auto-resume.** Templates checkpoint per epoch but don't auto-resume; on `preempt` a requeue
  restarts from scratch. T18 is short (fine on preempt); for the longer **T19, prefer `--partition=normal`**
  OR add `trainer.train(resume_from_checkpoint=True)` — don't silently lose 6 h.
- **AUPRC is the selection metric** (§7), never F1/accuracy — already coded; don't switch `metric_for_best_model`.
- **GPU-job rules:** no `--mem`/CPU flags; never `--exclusive` (8× cost). The job asserts CUDA is available
  (PT1 once silently ran on CPU for ~6 h — that check prevents a repeat).
- **Don't commit** `results/T18|T19/` (the trained models — git-ignored). The *templates* are committed code.

## What to commit / push
- **Code: nothing — unless you edit a template.** `train_prokbert.py`/`train_nt_lora.py` are committed; you
  `git pull` + run. If you change a script (e.g. a transformers-5.x fix, or T19 resume), hand the diff to Oak
  to commit (HPC4 push is blocked for us).
- **Models don't go in git** (git-ignored, and binaries don't belong there). Keep `--output-dir` on the
  **shared `/project` clone** (as in Step 4), NOT a `/home/<user>` clone, so T28 + the team can read them off
  the filesystem. HF upload is **post-T21**, to a **private** team-org repo (Oak).
- **Optional:** commit `run_t18.sbatch` / `run_t19.sbatch` if you want them in the repo.

## One-line summary
Env + data + prokbert-loading are all **verified on HPC4**. Pre-stage the two checkpoints → run T18 then T19
as GPU jobs on the current T16 (extended to 5,759 in-genus, verified-only eval, train 50/50) → verify each has
`weighted=True` and val AUPRC well above ~0.29 → hand to the T21 sweep (= finalising the T17 recipe). The
transformers-5.x issues are **fixed + validated**; the one thing run #1 settles is T19's NT-LoRA binding. Keep
models on shared `/project` storage; HF upload is post-T21.
