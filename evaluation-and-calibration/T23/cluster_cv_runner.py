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
4. Materializes fold datasets for all three model formats from T16:
   - prokbert: parquet with input_ids/attention_mask + metadata
   - nt:       parquet with input_ids/attention_mask + metadata
   - cnn:      npz + aligned meta parquet
5. Writes fold manifests and a summary table for auditability.

Run
---
python evaluation-and-calibration/T23/cluster_cv_runner.py \
  --t15-dir results/T15 \
  --t16-dir results/T16 \
  --out results/T23
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
    eval_rows: int
    train_pos: int
    eval_pos: int
    train_clusters: int
    eval_clusters: int


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


def fold_row_sets(assign_df: pd.DataFrame, fold_idx: int) -> Tuple[set, set]:
    eval_rows = set(assign_df.loc[assign_df["fold"] == fold_idx, "t23_row_id"].tolist())
    train_rows = set(assign_df["t23_row_id"].tolist()) - eval_rows
    return train_rows, eval_rows


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

        tr_ids, ev_ids = fold_row_sets(assign_df, k)
        fold_train = mapped[mapped["t23_row_id"].isin(tr_ids)].copy()
        fold_eval = mapped[mapped["t23_row_id"].isin(ev_ids)].copy()

        fold_train.to_parquet(fold_path / "train.parquet", index=False)
        fold_eval.to_parquet(fold_path / "eval.parquet", index=False)


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

        tr_ids, ev_ids = fold_row_sets(assign_df, k)
        tr_mask = mapped_meta["t23_row_id"].isin(tr_ids).to_numpy()
        ev_mask = mapped_meta["t23_row_id"].isin(ev_ids).to_numpy()

        np.savez_compressed(fold_path / "train.npz", X=X[tr_mask], label=y[tr_mask], weight=w[tr_mask])
        np.savez_compressed(fold_path / "eval.npz", X=X[ev_mask], label=y[ev_mask], weight=w[ev_mask])

        mapped_meta.loc[tr_mask].to_parquet(fold_path / "train_meta.parquet", index=False)
        mapped_meta.loc[ev_mask].to_parquet(fold_path / "eval_meta.parquet", index=False)


def compute_fold_stats(assign_df: pd.DataFrame, n_folds: int) -> List[FoldStats]:
    rows: List[FoldStats] = []
    for k in range(n_folds):
        tr_ids, ev_ids = fold_row_sets(assign_df, k)
        tr = assign_df[assign_df["t23_row_id"].isin(tr_ids)]
        ev = assign_df[assign_df["t23_row_id"].isin(ev_ids)]

        overlap = set(tr["cluster_id"].tolist()) & set(ev["cluster_id"].tolist())
        if overlap:
            raise ValueError(f"Fold {k + 1}: leakage detected ({len(overlap)} overlapping clusters).")

        rows.append(
            FoldStats(
                fold=k + 1,
                train_rows=len(tr),
                eval_rows=len(ev),
                train_pos=int(tr["label"].sum()),
                eval_pos=int(ev["label"].sum()),
                train_clusters=int(tr["cluster_id"].nunique()),
                eval_clusters=int(ev["cluster_id"].nunique()),
            )
        )
    return rows


def write_command_stub(out_dir: Path, n_folds: int) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# T23 command stub: fill model-paths for each fold and run in the correct env.",
        "# ProkBERT/CNN -> promoter-mining-hpc4 env; NT -> nt-tf4x env.",
        "",
    ]

    for k in range(1, n_folds + 1):
        fold = f"fold_{k:02d}"
        lines.extend([
            f"# ==== {fold} ====",
            f"# python model-training/T18/train_prokbert.py --data-dir results/T23/folds/prokbert/{fold} --output-dir results/T23/runs/prokbert/{fold}",
            f"# python model-training/T19/train_nt_lora.py --data-dir results/T23/folds/nt/{fold} --output-dir results/T23/runs/nt/{fold}",
            f"# python model-training/T20/train_cnn.py --data-dir results/T23/folds/cnn/{fold} --output-dir results/T23/runs/cnn/{fold}",
            "",
            "# Score eval fold with T22-style script after each model finishes:",
            f"# python evaluation-and-calibration/T22/predict_and_score.py --model prokbert --model-path results/T23/runs/prokbert/{fold}/best_model --data-dir results/T23/folds/prokbert/{fold} --out results/T23/scores/{fold}",
            f"# python evaluation-and-calibration/T22/predict_and_score.py --model nt --model-path results/T23/runs/nt/{fold}/best_adapter --data-dir results/T23/folds/nt/{fold} --out results/T23/scores/{fold}",
            f"# python evaluation-and-calibration/T22/predict_and_score.py --model cnn --model-path results/T23/runs/cnn/{fold}/best_model.pt --data-dir results/T23/folds/cnn/{fold} --out results/T23/scores/{fold}",
            "",
        ])

    (out_dir / "run_t23_commands.sh").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare cluster-level 5-fold CV datasets for T23.")
    p.add_argument("--t15-dir", default="results/T15", help="Directory with T15 train/val/test.parquet")
    p.add_argument("--t16-dir", default="results/T16", help="Directory with T16 model subdirs")
    p.add_argument("--out", default="results/T23", help="Output directory")
    p.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--models", default="prokbert,nt,cnn",
                   help="Comma-separated subset of models to materialize: prokbert, nt, cnn")
    p.add_argument("--write-command-stub", action="store_true", help="Write run_t23_commands.sh helper")
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

    if args.write_command_stub:
        write_command_stub(out_dir, args.n_folds)

    print(f"T23 fold datasets prepared at: {out_dir}")
    print(f"Fold summary: {out_dir / 'fold_summary.csv'}")


if __name__ == "__main__":
    main()
