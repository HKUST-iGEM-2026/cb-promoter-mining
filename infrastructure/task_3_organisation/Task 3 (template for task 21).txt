"""
Hyperparameter sweep orchestration for all models
Owner: Sanaa + Oak
Due: 1 Jul
"""

import json
import subprocess
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml
from datetime import datetime
import wandb
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================
SWEEP_CONFIG = {
    "prokbert": {
        "enabled": True,
        "script": "task18_prokbert_finetune/train_prokbert.py",
        "params": {
            "learning_rate": {"min": 1e-6, "max": 5e-5, "log": True},
            "batch_size": {"choices": [8, 16, 32]},
            "warmup_ratio": {"min": 0.05, "max": 0.15},
            "weight_decay": {"min": 0.0, "max": 0.1},
            "dropout_rate": {"min": 0.05, "max": 0.3},
            "num_epochs": {"choices": [5, 10, 15]}
        },
        "n_trials": 50,
        "resources": {"gpu": 1, "cpu": 8, "mem": "32G"}
    },
    "nt_lora": {
        "enabled": True,
        "script": "task19_nt_lora_finetune/train_nt_lora.py",
        "params": {
            "lora_r": {"choices": [4, 8, 16, 32]},
            "lora_alpha": {"choices": [16, 32, 64]},
            "lora_dropout": {"min": 0.05, "max": 0.3},
            "learning_rate": {"min": 1e-5, "max": 5e-4, "log": True},
            "batch_size": {"choices": [4, 8]},
            "warmup_steps": {"min": 100, "max": 1000},
        },
        "n_trials": 50,
        "resources": {"gpu": 1, "cpu": 12, "mem": "64G"}
    },
    "cnn": {
        "enabled": True,
        "script": "task20_cnn_scratch/train_cnn.py",
        "params": {
            "num_filters": {"choices": [[32,64,128], [64,128,256], [128,256,512]]},
            "kernel_sizes": {"choices": [[3,3,3], [5,3,3], [7,5,3]]},
            "dropout_rate": {"min": 0.1, "max": 0.5},
            "learning_rate": {"min": 1e-4, "max": 1e-2, "log": True},
            "batch_size": {"choices": [32, 64, 128]},
            "use_batch_norm": {"choices": [True, False]},
            "use_attention": {"choices": [True, False]}
        },
        "n_trials": 50,
        "resources": {"gpu": 1, "cpu": 8, "mem": "32G"}
    }
}

class HyperparameterSweepOrchestrator:
    """Orchestrate hyperparameter sweeps across all models"""

    def __init__(self, config, output_dir="sweep_results"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize wandb for sweep tracking (optional, can be moved into each trial)
        wandb.init(project="promoter_sweep", name="master_orchestrator")

    def run_single_trial(self, model_name, trial_params, trial_num):
        """Run a single trial for a specific model"""

        model_config = self.config[model_name]

        # Create trial directory
        trial_dir = self.output_dir / model_name / f"trial_{trial_num}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Save trial parameters
        with open(trial_dir / "params.json", "w") as f:
            json.dump(trial_params, f, indent=2)

        # Prepare command
        cmd = [
            "python", model_config["script"],
            "--sweep_trial", str(trial_num),
            "--output_dir", str(trial_dir)
        ]

        # Add parameters as command line arguments
        for key, value in trial_params.items():
            cmd.extend([f"--{key}", str(value)])

        try:
            # Run training
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )

            # Parse results
            if result.returncode == 0:
                # Load metrics from output
                metrics_file = trial_dir / "metrics.json"
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                else:
                    # Parse from stdout
                    metrics = self._parse_metrics_from_output(result.stdout)

                # Save results
                with open(trial_dir / "results.json", "w") as f:
                    json.dump({
                        "params": trial_params,
                        "metrics": metrics,
                        "status": "success"
                    }, f, indent=2)

                return {
                    "trial": trial_num,
                    "model": model_name,
                    "params": trial_params,
                    "metrics": metrics,
                    "status": "success"
                }
            else:
                # Log error
                with open(trial_dir / "error.log", "w") as f:
                    f.write(result.stderr)

                logger.error(f"Trial {trial_num} for {model_name} failed with error:\n{result.stderr}")

                return {
                    "trial": trial_num,
                    "model": model_name,
                    "status": "failed",
                    "error": result.stderr[:500]
                }

        except subprocess.TimeoutExpired:
            logger.error(f"Trial {trial_num} for {model_name} timed out after 2 hours")
            return {
                "trial": trial_num,
                "model": model_name,
                "status": "timeout",
                "error": "Training timed out after 2 hours"
            }

    def _parse_metrics_from_output(self, stdout):
        """Parse metrics from stdout output"""
        metrics = {}
        for line in stdout.split('\n'):
            if 'Test F1:' in line:
                try:
                    metrics['test_f1'] = float(line.split(':')[1].strip())
                except ValueError:
                    continue
            elif 'Test AUROC:' in line:
                try:
                    metrics['test_auroc'] = float(line.split(':')[1].strip())
                except ValueError:
                    continue
            elif 'Validation F1:' in line:
                try:
                    metrics['val_f1'] = float(line.split(':')[1].strip())
                except ValueError:
                    continue
        return metrics

    def run_optuna_sweep(self, model_name):
        """Run Optuna-based hyperparameter sweep for a model"""

        model_config = self.config[model_name]
        if not model_config.get("enabled", True):
            logger.info(f"Skipping {model_name} (disabled)")
            return None

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting hyperparameter sweep for {model_name}")
        logger.info(f"{'='*60}")

        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_range in model_config["params"].items():
                if "choices" in param_range:
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_range["choices"]
                    )
                elif "min" in param_range:
                    if param_range.get("log", False):
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_range["min"],
                            param_range["max"],
                            log=True
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_range["min"],
                            param_range["max"]
                        )
                elif "values" in param_range:
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_range["values"]
                    )

            # Run trial
            result = self.run_single_trial(model_name, params, trial.number)

            if result and result["status"] == "success":
                val_f1 = result["metrics"].get("val_f1")
                if val_f1 is not None:
                    # Optuna minimizes the objective by default, so return negative if you want to maximize.
                    return val_f1
                else:
                    raise optuna.exceptions.TrialPruned()
            else:
                raise optuna.exceptions.TrialPruned()

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
        )

        study.optimize(objective, n_trials=model_config["n_trials"])

        # Save best trial info
        best_trial = study.best_trial
        best_params_file = self.output_dir / model_name / "best_params.json"
        best_params_file.parent.mkdir(exist_ok=True, parents=True)
        with open(best_params_file, "w") as f:
            json.dump({
                "params": best_trial.params,
                "value": best_trial.value,
                "trial_number": best_trial.number
            }, f, indent=2)

        logger.info(f"Best trial for {model_name}: {best_trial.number} with value: {best_trial.value}")
        return study

    def run_all_sweeps(self):
        for model_name in self.config.keys():
            if self.config[model_name].get("enabled", True):
                try:
                    self.run_optuna_sweep(model_name)
                except Exception as e:
                    logger.error(f"Error running sweep for {model_name}: {e}", exc_info=True)


if __name__ == "__main__":
    orchestrator = HyperparameterSweepOrchestrator(SWEEP_CONFIG)
    orchestrator.run_all_sweeps()
