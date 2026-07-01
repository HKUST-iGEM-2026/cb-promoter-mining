#!/usr/bin/env python3
"""
T21 — hyperparameter sweep orchestration for the three models (ProkBERT / NT-LoRA / CNN).
Owner: Sanaa + Oak.

ORCHESTRATOR, not a trainer. Per model it runs an Optuna study that LAUNCHES the model's training script
(T18 / T19 / T20) as a SUBPROCESS once per trial, passes sampled hyperparameters as CLI flags, and reads
back the VALIDATION AUPRC from that trial's metrics.json to maximise. Output is best_params.json per model.

CONTRACT with T18/T19/T20 (they already honor it — keep aligned if you edit either side):
  * each script accepts `--sweep-trial N --output-dir <dir>` plus the hyperparameter flags listed in
    SWEEP_CONFIG[model]["params"], and writes `<dir>/metrics.json` containing **"val_auprc"**.
  * the param keys below are the EXACT CLI flag names (dash form) the scripts expose.
  * `--sweep-epochs` (below) passes `--num-epochs`, so each trainer must accept that flag.

EXPENSIVE step — one full GPU training per trial; NT dominates (~hours/trial). Make it tractable:
  * `--sweep-epochs 4`  force cheap trials (overrides each trainer's epochs); retrain the winner at full epochs after.
  * `--n-trials 12`     fewer trials.
  * one model per job, in parallel:  `--models nt_lora`   (etc.) on separate GPUs.
RESUMABLE: each study is a sqlite db at <out>/<model>/study.db, so a preempted/killed job RESUMES where it
stopped (runs only up to n_trials TOTAL). NB there is no Optuna pruner: subprocess trials can't report
intermediate scores back, so pruning can't fire — the TPE sampler does the smart search instead.

  python hyperparameter_sweep.py --models prokbert,nt_lora --n-trials 15 --sweep-epochs 4 --out results/T21
  # run from the scaffold dir so the trainers' default results/T16/<model>/ paths resolve.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import optuna
from optuna.samplers import TPESampler

REPO = Path(__file__).resolve().parents[2]            # model-training/T21/.. .. -> repo root

SWEEP_CONFIG = {
    "prokbert": {
        "script": str(REPO / "model-training/T18/train_prokbert.py"),
        "params": {
            "learning-rate": {"min": 1e-6, "max": 5e-5, "log": True},
            "batch-size":    {"choices": [8, 16, 32]},
            "warmup-ratio":  {"min": 0.05, "max": 0.15},
            "weight-decay":  {"min": 0.0, "max": 0.1},
            "dropout-rate":  {"min": 0.05, "max": 0.3},
            "num-epochs":    {"choices": [5, 10, 15]},
        },
        "n_trials": 25,
        "timeout": 5400,       # 1.5 h/trial — prokbert-mini is small
    },
    "nt_lora": {
        "script": str(REPO / "model-training/T19/train_nt_lora.py"),
        "params": {
            "lora-r":        {"choices": [4, 8, 16, 32]},
            "lora-alpha":    {"choices": [16, 32, 64]},
            "lora-dropout":  {"min": 0.05, "max": 0.3},
            "learning-rate": {"min": 1e-5, "max": 5e-4, "log": True},
            "batch-size":    {"choices": [4, 8]},
            "warmup-steps":  {"min": 100, "max": 1000},
            "weight-decay":  {"min": 0.0, "max": 0.1},
        },
        "n_trials": 25,
        "timeout": 21600,      # 6 h/trial — NT-250M is heavy (raise if a full-epoch trial needs longer)
    },
    "cnn": {
        # Yany: align these keys with train_cnn.py's CLI flags (+ it must accept --num-epochs for --sweep-epochs).
        # List-valued architecture params (num_filters, kernel_sizes) -> pass a string like "32,64,128" and
        # parse it inside train_cnn.py (Optuna/CLI can't pass a Python list directly).
        "script": str(REPO / "model-training/T20/train_cnn.py"),
        "params": {
            "learning-rate": {"min": 1e-4, "max": 1e-2, "log": True},
            "batch-size":    {"choices": [32, 64, 128]},
            "dropout-rate":  {"min": 0.1, "max": 0.5},
        },
        "n_trials": 25,
        "timeout": 3600,       # 1 h/trial — CNN trains from scratch, fast
    },
}


def suggest(trial, name, spec):
    if "choices" in spec:
        return trial.suggest_categorical(name, spec["choices"])
    if spec.get("log"):
        return trial.suggest_float(name, spec["min"], spec["max"], log=True)
    if isinstance(spec["min"], int) and isinstance(spec["max"], int):
        return trial.suggest_int(name, spec["min"], spec["max"])
    return trial.suggest_float(name, spec["min"], spec["max"])


def run_trial(script, params, trial_dir, trial_num, timeout):
    """Launch the training script; return its val_auprc (None on failure/timeout -> trial pruned)."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, script, "--sweep-trial", str(trial_num), "--output-dir", str(trial_dir)]
    for k, v in params.items():
        cmd += [f"--{k}", str(v)]
    (trial_dir / "params.json").write_text(json.dumps(params, indent=2))
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        (trial_dir / "error.log").write_text(f"TIMEOUT after {timeout}s")
        return None
    if res.returncode != 0:
        (trial_dir / "error.log").write_text((res.stderr or "")[-4000:])
        return None
    mfile = trial_dir / "metrics.json"
    if mfile.exists():
        return json.loads(mfile.read_text()).get("val_auprc")
    for line in res.stdout.splitlines():                       # fallback: parse stdout
        if line.startswith("Validation AUPRC:"):
            try:
                return float(line.split(":", 1)[1])
            except ValueError:
                pass
    return None


def sweep_model(name, cfg, out_dir, n_trials, timeout, sweep_epochs):
    print(f"\n===== sweeping {name} ({n_trials} trials, timeout {timeout}s"
          f"{', epochs=' + str(sweep_epochs) if sweep_epochs else ''}) =====")
    model_out = out_dir / name
    model_out.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        # sample all params — but NOT num-epochs when --sweep-epochs forces a fixed cheap value
        to_sample = {k: v for k, v in cfg["params"].items() if not (sweep_epochs and k == "num-epochs")}
        params = {k: suggest(trial, k, spec) for k, spec in to_sample.items()}
        if sweep_epochs:
            params["num-epochs"] = sweep_epochs
        auprc = run_trial(cfg["script"], params, model_out / f"trial_{trial.number}", trial.number, timeout)
        if auprc is None:
            raise optuna.exceptions.TrialPruned()
        return auprc

    # RESUMABLE: sqlite-backed study -> a preempted/killed job continues where it stopped.
    study = optuna.create_study(
        study_name=name, direction="maximize", sampler=TPESampler(seed=42),
        storage=f"sqlite:///{(model_out / 'study.db').resolve()}", load_if_exists=True)
    remaining = max(0, n_trials - len(study.trials))
    print(f"  {len(study.trials)} trial(s) already recorded; running {remaining} more (target {n_trials}).")
    if remaining:
        study.optimize(objective, n_trials=remaining)

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        print(f"  {name}: NO successful trial yet — check <trial_dir>/error.log.")
        return {"model": name, "best_val_auprc": None, "n_completed": 0}
    best = {"model": name, "best_val_auprc": study.best_value,
            "best_params": study.best_params, "n_completed": len(done)}
    (model_out / "best_params.json").write_text(json.dumps(best, indent=2))
    print(f"  {name}: best val AUPRC = {study.best_value:.4f}  params = {study.best_params}")
    return best


def main():
    ap = argparse.ArgumentParser(description="T21 hyperparameter sweep (Optuna) over T18/T19/T20")
    ap.add_argument("--models", default="prokbert,nt_lora,cnn", help="comma list: prokbert,nt_lora,cnn")
    ap.add_argument("--out", default="results/T21")
    ap.add_argument("--n-trials", type=int, default=0, help="override per-model n_trials (0 = use config)")
    ap.add_argument("--timeout", type=int, default=0, help="per-trial timeout seconds (0 = per-model config)")
    ap.add_argument("--sweep-epochs", type=int, default=0,
                    help="force this many epochs per trial (cheap proxy); 0 = use each trainer's own epochs")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    summary = []
    for m in models:
        if m not in SWEEP_CONFIG:
            sys.exit(f"unknown model '{m}' — choose from {list(SWEEP_CONFIG)}")
        if not os.path.exists(SWEEP_CONFIG[m]["script"]):
            print(f"[skip] {m}: training script not found ({SWEEP_CONFIG[m]['script']}) — write it first.")
            continue
        n = args.n_trials or SWEEP_CONFIG[m]["n_trials"]
        to = args.timeout or SWEEP_CONFIG[m].get("timeout", 7200)
        summary.append(sweep_model(m, SWEEP_CONFIG[m], out_dir, n, to, args.sweep_epochs))

    (out_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nT21 done — best params per model under", out_dir)


if __name__ == "__main__":
    main()
