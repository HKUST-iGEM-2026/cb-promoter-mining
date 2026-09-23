# CLAUDE.md — run T21, the hyperparameter sweep, step by step (for Sanaa/Oak)

> **Cluster setup first:** [`infrastructure/HPC4_GUIDE.md`](../../infrastructure/HPC4_GUIDE.md) — login,
> env activation, SLURM. Then come back here.

T21 (`model-training/hyperparameter-sweep/hyperparameter_sweep.py`) is an **orchestrator**, not a trainer. Per model it runs
an Optuna study that launches that model's training script (T18 / T19 / T20) as a **subprocess** once per
trial, passes sampled hyperparameters as CLI flags, and **maximises validation AUPRC** (read from each
trial's `metrics.json`). Output is a `best_params.json` per model.

## Prereqs (all three must hold)
1. **T18, T19, T20 each run standalone** on the current T16 — do `SANAA_T18_T19_RUN_CLAUDE.md` and
   `T20/YANY_T20_RUN_CLAUDE.md` first (at least one clean run each). T21 can't sweep a model whose script errors.
2. **HF checkpoints pre-staged** (HPC4 guide §7) — every prokbert/nt trial reloads them.
3. **Run from the scaffold dir** so the training scripts' default `results/T16/<model>/` paths resolve:
   `cd /project/igem2026/cb-promoter-mining/infrastructure/T3/snakemake_scaffold`.

## What it costs (read before launching)
One FULL GPU training per trial, and **NT dominates**. Rough H800 per-trial: prokbert ~15–40 min, NT-LoRA
~2–4 h, CNN ~15 min → a naive 25×3 sweep ≈ **~4 days of GPU** (mostly NT). Cut it down:
- **Prove the wiring small first:** `--models prokbert --n-trials 3`.
- **Cheap proxy (biggest lever):** `--sweep-epochs 4` forces every trial to 4 epochs (overrides the trainer's
  own epochs) so trials are cheap — then **retrain the winner at full epochs** afterward.
- **Fewer trials:** `--n-trials 12` — TPE finds good configs without 25.
- **One model per job, in parallel** on separate GPUs → wall-clock ≈ the NT sweep alone, not the sum.
- Per-trial **timeouts are automatic per model** (prokbert 1.5 h, NT 6 h, CNN 1 h); `--timeout <sec>` overrides.
  A trial that times out or errors is skipped (not fatal) and the sweep moves on.

## Run
```bash
# inside a GPU job. RESUMABLE: the study is a sqlite db per model, so if the job is preempted/killed, just
# re-submit the SAME command — it continues where it stopped (runs only up to --n-trials total). preempt is fine.
export HF_HOME=/project/igem2026/hf_cache HF_HUB_OFFLINE=1
python ../../../model-training/hyperparameter-sweep/hyperparameter_sweep.py \
  --models prokbert,nt_lora,cnn \
  --n-trials 15 --sweep-epochs 4 \
  --out /project/igem2026/cb-promoter-mining/results/T21
```
(or one `--models <x>` per job). Each trial's training output lands under `results/T21/<model>/trial_<n>/`;
the winner is `results/T21/<model>/best_params.json`.

## Read the result
```python
import json
for m in ("prokbert", "nt_lora", "cnn"):
    b = json.load(open(f"results/T21/{m}/best_params.json"))
    print(m, "best val AUPRC", round(b["best_val_auprc"], 3), "| params", b["best_params"])
```
Then **retrain each model once at its best params** (the per-trial checkpoints are throwaways) to produce
the **final** models → those feed T28 mining and the post-T21 HF upload.

## Gotchas (non-yes-man)
- **Selects on `val_auprc`, never test** — no leak. Don't add test to the objective.
- **CNN won't sweep until `train_cnn.py` exists** with the CLI contract (Yany — `T20/YANY_T20_RUN_CLAUDE.md`).
  T21 **skips** any model whose script is missing (prints `[skip]`), so prokbert/nt can sweep before the CNN is ready.
- **Cost discipline:** don't launch 25×3 blind. Prove 3 trials first; `--n-trials` + `--models` are the throttles.
- **Reuse-or-resweep:** if the data changed only a little since a prior sweep, reuse the prior best params and
  just re-validate instead of paying for a full sweep.
- **Retrain the winner at full epochs** if you swept with `--sweep-epochs` — the cheap-proxy sweep only *ranks*
  configs; the final model must be trained at full epochs on the best params.
- **No pruner (by design):** subprocess trials can't report intermediate scores to Optuna, so bad configs run
  to completion — the TPE sampler does the smart search. `--sweep-epochs` is how you keep those runs cheap.

## What to commit / push
- **Nothing from `results/T21/`** (git-ignored — best_params, trial outputs, models); it lives on shared
  `/project` storage. If you want the winning params in the repo, copy the small `best_params.json` values
  into a committed note — the trial artefacts stay out of git.

## One-line summary
Confirm T18/T19/T20 run standalone + checkpoints staged → `hyperparameter_sweep.py --models … --n-trials …
--sweep-epochs 4 --out …` from the scaffold dir (prove 3 trials first; NT dominates the ~4-day full cost, so
use `--sweep-epochs` + fewer trials + parallel jobs) → read `best_params.json` per model → **retrain each at
its best params, full epochs**, for the final ensemble. Selects on val AUPRC; resumable (sqlite); commits nothing.
