#!/usr/bin/env python3
"""
T23 cluster-level 5-fold cross-validation runner (new implementation).

This file is intentionally separate from cluster_cv_template.py.
It prepares fold-safe datasets for T18/T19/T20 + T22 without touching the template.

What it does
------------
1. Builds a CV pool from T15 train+val (keeps T15 test untouched).
2. Creates 5 cluster-disjoint folds using StratifiedGroupKFold.
3. Pins silver positives (T13_inferred clusters) to training-only, matching T15 policy.
4. Materializes each fold as trainer-ready train/val/test for all three model formats from T16
   (test = the held-out fold; val = the next fold, for early stopping; train = the rest + silver):
   - prokbert: parquet with input_ids/attention_mask + metadata
   - nt:       parquet with input_ids/attention_mask + metadata
   - cnn:      npz + aligned meta parquet
5. Writes fold manifests and a summary table for auditability.
6. Writes run_t23.sh — a runnable SLURM submitter that trains + scores all folds x models in the
   right envs, then runs aggregate_cv.py (mean +/- std per model). Nothing is left to fill in.

Run
---
# 1) prepare folds + emit run_t23.sh:
python evaluation-and-calibration/T23/cluster_cv_runner.py \
  --t15-dir results/T15 --t16-dir results/T16 --out results/T23
# 2) launch the whole CV (submits jobs, aggregates when they finish):
bash results/T23/run_t23.sh
# -> results/T23/cv_summary.csv  (CV AUPRC/AUROC/MCC/F1 = mean +/- std per model)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DEFAULT_N_FOLDS = 5
DEFAULT_SEED = 42
SILVER_SOURCE = "T13_inferred"

T15_SPLIT_COLS = ["split", "label", "source", "cluster_id", "promoter_id"]


@dataclass
class FoldStats:
    fold: int
    train_rows: int
    val_rows: int
    test_rows: int
    train_pos: int
    val_pos: int
    test_pos: int
    train_clusters: int
    val_clusters: int
    test_clusters: int


def _canonical_value(v) -> str:
    if pd.isna(v):
        return "__NA__"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.12g}"
    return str(v)


def _add_key_and_occ(df: pd.DataFrame, key_cols: Sequence[str], key_name: str) -> pd.DataFrame:
    out = df.copy()
    out[key_name] = out[key_cols].apply(lambda r: "|".join(_canonical_value(v) for v in r), axis=1)
    out["__occ"] = out.groupby(key_name).cumcount()
    return out


def _ensure_columns(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def load_t15_cv_pool(t15_dir: Path) -> pd.DataFrame:
    train = pd.read_parquet(t15_dir / "train.parquet").copy()
    val = pd.read_parquet(t15_dir / "val.parquet").copy()

    _ensure_columns(train, T15_SPLIT_COLS, "T15 train")
    _ensure_columns(val, T15_SPLIT_COLS, "T15 val")

    cv = pd.concat([train, val], ignore_index=True)
    cv["t23_row_id"] = np.arange(len(cv), dtype=np.int64)
    return cv


def assign_cluster_folds(cv_df: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    _ensure_columns(cv_df, ["label", "cluster_id", "source", "t23_row_id"], "T15 CV pool")

    # Match T15 policy: any cluster containing silver positives is pinned to training only.
    silver_clusters = set(cv_df.loc[(cv_df["label"] == 1) & (cv_df["source"] == SILVER_SOURCE), "cluster_id"])

    out = cv_df.copy()
    out["fold"] = -1
    out["cv_role"] = "train_only"

    eligible_mask = ~out["cluster_id"].isin(silver_clusters)
    elig = out.loc[eligible_mask].copy()

    if elig.empty:
        raise ValueError("No eligible rows left for fold creation after excluding silver clusters.")

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold_idx, (_tr, te) in enumerate(sgkf.split(elig, elig["label"], groups=elig["cluster_id"])):
        te_ids = elig.iloc[te]["t23_row_id"].values
        out.loc[out["t23_row_id"].isin(te_ids), "fold"] = fold_idx
        out.loc[out["t23_row_id"].isin(te_ids), "cv_role"] = "eval"

    # Rows in eligible set but not selected as eval for a given fold remain train for that fold.
    out.loc[(out["fold"] >= 0) & (out["cv_role"] != "eval"), "cv_role"] = "pool"

    return out


def fold_tvt_sets(assign_df: pd.DataFrame, fold_idx: int, n_folds: int) -> Tuple[set, set, set]:
    """CV fold k: test = fold k (held out for scoring), val = fold (k+1) % n (early-stopping
    set, verified-only), train = everything else (the other folds' verified rows + the silver
    train_only rows). All three are cluster-disjoint by construction, so no cluster leaks across
    train/val/test."""
    test_rows = set(assign_df.loc[assign_df["fold"] == fold_idx, "t23_row_id"].tolist())
    val_fold = (fold_idx + 1) % n_folds
    val_rows = set(assign_df.loc[assign_df["fold"] == val_fold, "t23_row_id"].tolist())
    train_rows = set(assign_df["t23_row_id"].tolist()) - test_rows - val_rows
    return train_rows, val_rows, test_rows


def _build_assignment_keys(assign_df: pd.DataFrame, model_meta_cols: Sequence[str]) -> pd.DataFrame:
    key_cols = [c for c in model_meta_cols if c in assign_df.columns]
    if not key_cols:
        raise ValueError("No overlapping metadata columns found to map T15 rows to T16 rows.")

    a = _add_key_and_occ(assign_df, key_cols, "__key")
    return a[["t23_row_id", "__key", "__occ"]]


def _map_rows_to_t23(model_df: pd.DataFrame, assignment_keys: pd.DataFrame, key_cols: Sequence[str], model_name: str) -> pd.DataFrame:
    m = _add_key_and_occ(model_df, key_cols, "__key")
    merged = m.merge(assignment_keys, on=["__key", "__occ"], how="left")

    missing = int(merged["t23_row_id"].isna().sum())
    if missing:
        raise ValueError(f"{model_name}: {missing} rows could not be mapped from T16 back to T15 assignment keys.")

    merged["t23_row_id"] = merged["t23_row_id"].astype(np.int64)
    return merged.drop(columns=["__key", "__occ"])


def materialize_hf_model(
    model_name: str,
    model_dir: Path,
    assign_df: pd.DataFrame,
    out_dir: Path,
    n_folds: int,
) -> None:
    train_df = pd.read_parquet(model_dir / "train.parquet")
    val_df = pd.read_parquet(model_dir / "val.parquet")
    cv_df = pd.concat([train_df, val_df], ignore_index=True)

    metadata_cols = [c for c in assign_df.columns if c in cv_df.columns and c not in {"t23_row_id", "fold", "cv_role"}]
    assignment_keys = _build_assignment_keys(assign_df, metadata_cols)
    mapped = _map_rows_to_t23(cv_df, assignment_keys, metadata_cols, model_name)

    model_out = out_dir / model_name
    model_out.mkdir(parents=True, exist_ok=True)

    for k in range(n_folds):
        fold_path = model_out / f"fold_{k + 1:02d}"
        fold_path.mkdir(parents=True, exist_ok=True)

        tr_ids, va_ids, te_ids = fold_tvt_sets(assign_df, k, n_folds)
        mapped[mapped["t23_row_id"].isin(tr_ids)].to_parquet(fold_path / "train.parquet", index=False)
        mapped[mapped["t23_row_id"].isin(va_ids)].to_parquet(fold_path / "val.parquet", index=False)
        mapped[mapped["t23_row_id"].isin(te_ids)].to_parquet(fold_path / "test.parquet", index=False)


def materialize_cnn_model(
    cnn_dir: Path,
    assign_df: pd.DataFrame,
    out_dir: Path,
    n_folds: int,
) -> None:
    z_train = np.load(cnn_dir / "train.npz", allow_pickle=False)
    z_val = np.load(cnn_dir / "val.npz", allow_pickle=False)

    X = np.concatenate([z_train["X"], z_val["X"]], axis=0)
    y = np.concatenate([z_train["label"], z_val["label"]], axis=0)
    w = np.concatenate([z_train["weight"], z_val["weight"]], axis=0)

    meta_train = pd.read_parquet(cnn_dir / "train_meta.parquet")
    meta_val = pd.read_parquet(cnn_dir / "val_meta.parquet")
    meta = pd.concat([meta_train, meta_val], ignore_index=True)

    metadata_cols = [c for c in assign_df.columns if c in meta.columns and c not in {"t23_row_id", "fold", "cv_role"}]
    assignment_keys = _build_assignment_keys(assign_df, metadata_cols)
    mapped_meta = _map_rows_to_t23(meta, assignment_keys, metadata_cols, "cnn")

    if len(mapped_meta) != len(X):
        raise ValueError(f"cnn: metadata rows ({len(mapped_meta)}) do not match tensor rows ({len(X)}).")

    model_out = out_dir / "cnn"
    model_out.mkdir(parents=True, exist_ok=True)

    for k in range(n_folds):
        fold_path = model_out / f"fold_{k + 1:02d}"
        fold_path.mkdir(parents=True, exist_ok=True)

        tr_ids, va_ids, te_ids = fold_tvt_sets(assign_df, k, n_folds)
        for name, ids in (("train", tr_ids), ("val", va_ids), ("test", te_ids)):
            mask = mapped_meta["t23_row_id"].isin(ids).to_numpy()
            np.savez_compressed(fold_path / f"{name}.npz", X=X[mask], label=y[mask], weight=w[mask])
            mapped_meta.loc[mask].to_parquet(fold_path / f"{name}_meta.parquet", index=False)


def compute_fold_stats(assign_df: pd.DataFrame, n_folds: int) -> List[FoldStats]:
    rows: List[FoldStats] = []
    for k in range(n_folds):
        tr_ids, va_ids, te_ids = fold_tvt_sets(assign_df, k, n_folds)
        tr = assign_df[assign_df["t23_row_id"].isin(tr_ids)]
        va = assign_df[assign_df["t23_row_id"].isin(va_ids)]
        te = assign_df[assign_df["t23_row_id"].isin(te_ids)]

        tr_c = set(tr["cluster_id"].tolist())
        va_c = set(va["cluster_id"].tolist())
        te_c = set(te["cluster_id"].tolist())
        # test must be cluster-disjoint from BOTH train and val; val disjoint from train.
        if te_c & (tr_c | va_c):
            raise ValueError(f"Fold {k + 1}: leakage — test clusters overlap train/val.")
        if va_c & tr_c:
            raise ValueError(f"Fold {k + 1}: leakage — val clusters overlap train.")

        rows.append(
            FoldStats(
                fold=k + 1,
                train_rows=len(tr),
                val_rows=len(va),
                test_rows=len(te),
                train_pos=int(tr["label"].sum()),
                val_pos=int(va["label"].sum()),
                test_pos=int(te["label"].sum()),
                train_clusters=int(tr["cluster_id"].nunique()),
                val_clusters=int(va["cluster_id"].nunique()),
                test_clusters=int(te["cluster_id"].nunique()),
            )
        )
    return rows


def write_run_script(out_dir: Path, n_folds: int) -> None:
    """Write a REAL, runnable orchestration script. For each fold it sbatch-submits the
    ProkBERT+CNN training/scoring (promoter-mining-hpc4 env) and the NT training/scoring
    (nt-tf4x env), then submits an aggregation job that runs after ALL folds succeed
    (afterok dependency). Nothing is left to fill in — just `bash run_t23.sh` on HPC4."""
    folds = " ".join(f"{k:02d}" for k in range(1, n_folds + 1))
    script = f'''#!/usr/bin/env bash
# ============================================================================
# T23 — cluster-level {n_folds}-fold CV: train + score every fold x model, then aggregate.
# Auto-generated by cluster_cv_runner.py. Submits SLURM GPU jobs, then one CPU
# aggregation job that runs automatically after every fold job succeeds.
#   ProkBERT + CNN -> promoter-mining-hpc4 env (transformers 5.3.0)
#   NT-LoRA        -> nt-tf4x env             (transformers 4.57)
#
# Override these before running if your cluster paths differ:
#   export REPO=/project/igem2026/cb-promoter-mining
#   export ENV_MAIN=/project/igem2026/envs/promoter-mining-hpc4
#   export ENV_NT=/project/igem2026/envs/nt-tf4x
#   export CONDA_SH=<conda base>/etc/profile.d/conda.sh   # auto-detected via `conda info --base`
#
# NB the CNN uses --no-batch-norm to match the standalone T20 production model
# (see predict_and_score.py). If your chosen CNN was trained WITH BatchNorm,
# drop that flag on the train_cnn line below so the CV matches production.
# ============================================================================
set -euo pipefail

REPO="${{REPO:-/project/igem2026/cb-promoter-mining}}"
ENV_MAIN="${{ENV_MAIN:-/project/igem2026/envs/promoter-mining-hpc4}}"
ENV_NT="${{ENV_NT:-/project/igem2026/envs/nt-tf4x}}"
CONDA_SH="${{CONDA_SH:-$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh}}"
SCAF="$REPO/infrastructure/T3/snakemake_scaffold"

cd "$SCAF"
mkdir -p logs results/T23/runs results/T23/scores
JOBIDS=""

for k in {folds}; do
  # ---- ProkBERT + CNN (promoter-mining-hpc4) ----
  JID=$(sbatch --parsable --account=igem2026 --partition=preempt --requeue --signal=B:USR1@120 \\
        --gpus-per-node=1 --time=06:00:00 --job-name="t23_main_$k" \\
        --output="logs/t23_main_${{k}}_%j.out" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "$CONDA_SH"; conda activate "$ENV_MAIN"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
cd "$SCAF"
python "$REPO/model-training/T18/train_prokbert.py" \\
  --data-dir results/T23/folds/prokbert/fold_$k --output-dir results/T23/runs/prokbert/fold_$k
python "$REPO/evaluation-and-calibration/T22/predict_and_score.py" --model prokbert \\
  --model-path results/T23/runs/prokbert/fold_$k/best_model \\
  --data-dir results/T23/folds/prokbert/fold_$k --out results/T23/scores/fold_$k
python "$REPO/model-training/T20/train_cnn.py" --no-batch-norm \\
  --data-dir results/T23/folds/cnn/fold_$k --output-dir results/T23/runs/cnn/fold_$k
python "$REPO/evaluation-and-calibration/T22/predict_and_score.py" --model cnn \\
  --model-path results/T23/runs/cnn/fold_$k/best_model.pt \\
  --data-dir results/T23/folds/cnn/fold_$k --out results/T23/scores/fold_$k
EOF
)
  echo "fold $k main -> job $JID"; JOBIDS="$JOBIDS:$JID"

  # ---- NT-LoRA (nt-tf4x) ----
  JID=$(sbatch --parsable --account=igem2026 --partition=preempt --requeue --signal=B:USR1@120 \\
        --gpus-per-node=1 --time=08:00:00 --job-name="t23_nt_$k" \\
        --output="logs/t23_nt_${{k}}_%j.out" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "$CONDA_SH"; conda activate "$ENV_NT"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH="\\$CONDA_PREFIX/lib:\\${{LD_LIBRARY_PATH:-}}"
cd "$SCAF"
python "$REPO/model-training/T19/train_nt_lora.py" --no-grad-checkpoint \\
  --data-dir results/T23/folds/nt/fold_$k --output-dir results/T23/runs/nt/fold_$k
python "$REPO/evaluation-and-calibration/T22/predict_and_score.py" --model nt \\
  --model-path results/T23/runs/nt/fold_$k/best_adapter \\
  --data-dir results/T23/folds/nt/fold_$k --out results/T23/scores/fold_$k
EOF
)
  echo "fold $k nt   -> job $JID"; JOBIDS="$JOBIDS:$JID"
done

# ---- aggregate mean +/- std across folds, after every fold job succeeds ----
sbatch --account=igem2026 --partition=cpu --time=00:20:00 --job-name="t23_agg" \\
  --dependency="afterok${{JOBIDS}}" --output="logs/t23_agg_%j.out" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "$CONDA_SH"; conda activate "$ENV_MAIN"
export PYTHONNOUSERSITE=1
cd "$SCAF"
python "$REPO/evaluation-and-calibration/T23/aggregate_cv.py" \\
  --scores-dir results/T23/scores --out results/T23
EOF
echo "aggregation queued (runs afterok all {n_folds} folds x 3 models)."
echo "watch:  squeue -u \\$USER      final table:  $SCAF/results/T23/cv_summary.csv"
'''
    p = out_dir / "run_t23.sh"
    p.write_text(script)
    p.chmod(0o755)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare cluster-level 5-fold CV datasets for T23.")
    p.add_argument("--t15-dir", default="results/T15", help="Directory with T15 train/val/test.parquet")
    p.add_argument("--t16-dir", default="results/T16", help="Directory with T16 model subdirs")
    p.add_argument("--out", default="results/T23", help="Output directory")
    p.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--models", default="prokbert,nt,cnn",
                   help="Comma-separated subset of models to materialize: prokbert, nt, cnn")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    valid = {"prokbert", "nt", "cnn"}
    bad = [m for m in models if m not in valid]
    if bad:
        raise ValueError(f"Unknown model(s): {bad}. Choose from {sorted(valid)}")

    t15_dir = Path(args.t15_dir)
    t16_dir = Path(args.t16_dir)
    out_dir = Path(args.out)
    folds_out = out_dir / "folds"

    folds_out.mkdir(parents=True, exist_ok=True)

    cv = load_t15_cv_pool(t15_dir)
    assigned = assign_cluster_folds(cv, n_folds=args.n_folds, seed=args.seed)

    assigned.to_parquet(out_dir / "t23_fold_assignment.parquet", index=False)
    assigned[["t23_row_id", "label", "source", "cluster_id", "fold", "cv_role"]].to_csv(
        out_dir / "t23_fold_assignment.csv", index=False
    )

    if "prokbert" in models:
        materialize_hf_model("prokbert", t16_dir / "prokbert", assigned, folds_out, args.n_folds)
    if "nt" in models:
        materialize_hf_model("nt", t16_dir / "nt", assigned, folds_out, args.n_folds)
    if "cnn" in models:
        materialize_cnn_model(t16_dir / "cnn", assigned, folds_out, args.n_folds)

    stats = compute_fold_stats(assigned, args.n_folds)
    stats_df = pd.DataFrame([s.__dict__ for s in stats])
    stats_df.to_csv(out_dir / "fold_summary.csv", index=False)

    manifest = {
        "models": models,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "input": {
            "t15_dir": str(t15_dir),
            "t16_dir": str(t16_dir),
        },
        "policy": {
            "cluster_cv": "StratifiedGroupKFold",
            "silver_source": SILVER_SOURCE,
            "silver_clusters": "training_only",
            "test_set": "T15 test excluded from T23 fold generation",
        },
        "rows": {
            "cv_pool": int(len(assigned)),
            "eval_pool": int((assigned["fold"] >= 0).sum()),
            "train_only_pool": int((assigned["fold"] < 0).sum()),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    write_run_script(out_dir, args.n_folds)

    print(f"T23 fold datasets prepared at: {out_dir}")
    print(f"Fold summary: {out_dir / 'fold_summary.csv'}")
    print(f"Run the CV:   bash {out_dir / 'run_t23.sh'}   (submits all folds, then aggregates)")


if __name__ == "__main__":
    main()
