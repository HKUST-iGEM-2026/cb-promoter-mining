#!/usr/bin/env python3
"""
T24 — external holdout evaluation (cross-species), new implementation.

This script is intentionally separate from evaluate_external_template.py.
It evaluates trained models on out-of-genus T9 promoters and reports generalization metrics.

Design choices
--------------
- Positives: T9 holdout promoters (default: out-of-genus rows where in_genus == False).
- Negatives: per-positive dinucleotide-shuffled sequences (matched composition null).
- Balanced eval set: 1:1 positive/negative to make threshold metrics interpretable.
- Models: ProkBERT (T18), NT-LoRA (T19), CNN (T20).

Notes
-----
- ProkBERT/CNN should run in promoter-mining-hpc4 env.
- NT adapter should run in nt-tf4x env for strict compatibility.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score

NT_ID = "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species"
# Load the ProkBERT tokenizer from the BASE repo (identical vocab) rather than the fine-tuned
# best_model dir: trainer.save_model can leave a slow tokenizer transformers can't rebuild, and
# fine-tuning never changes the tokenizer. The fine-tuned MODEL still comes from --prokbert-model-path.
PROKBERT_BASE = "neuralbioinfo/prokbert-mini"
BOOT_SEED = 0      # deterministic bootstrap for the per-species CIs
NBOOT = 200


def dinucleotide_shuffle(seq: str, rng: random.Random) -> str:
    seq = seq.upper().replace("N", "")
    if len(seq) < 2:
        return seq
    edges = collections.defaultdict(list)
    for a, b in zip(seq, seq[1:]):
        edges[a].append(b)
    for node, arr in edges.items():
        if len(arr) > 1:
            core = arr[:-1]
            rng.shuffle(core)
            edges[node] = core + [arr[-1]]
    out = [seq[0]]
    cur = seq[0]
    while edges[cur]:
        nxt = edges[cur].pop(0)
        out.append(nxt)
        cur = nxt
    if len(out) < len(seq):
        arr = list(seq)
        rng.shuffle(arr)
        return "".join(arr)
    return "".join(out)


def one_hot_81(seq: str) -> np.ndarray:
    base = {"A": 0, "C": 1, "G": 2, "T": 3}
    s = seq.upper()
    if len(s) < 81:
        pad = 81 - len(s)
        s = ("N" * (pad // 2)) + s + ("N" * (pad - pad // 2))
    elif len(s) > 81:
        off = (len(s) - 81) // 2
        s = s[off:off + 81]

    x = np.zeros((4, 81), dtype=np.float32)
    for i, b in enumerate(s):
        j = base.get(b)
        if j is not None:
            x[j, i] = 1.0
    return x


@dataclass
class ExternalDataset:
    df: pd.DataFrame
    positives: int
    negatives: int


def load_t9_holdout(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"T9 holdout file not found: {path}")
    if path.suffix.lower() in {".tsv", ".txt"}:
        df = pd.read_csv(path, sep="\t")
    elif path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported holdout format: {path.suffix}")

    if "sequence" not in df.columns:
        raise ValueError("T9 holdout is missing required column 'sequence'.")
    if "species" not in df.columns:
        df["species"] = "unknown"
    if "in_genus" not in df.columns:
        df["in_genus"] = False

    df = df.copy()
    df["sequence"] = df["sequence"].astype(str).str.upper()
    df = df[df["sequence"].str.fullmatch(r"[ACGTN]+", na=False)].reset_index(drop=True)
    return df


def build_external_dataset(
    holdout_df: pd.DataFrame,
    out_of_genus_only: bool,
    seed: int,
) -> ExternalDataset:
    src = holdout_df.copy()
    if out_of_genus_only:
        src = src[src["in_genus"].astype(str).str.lower() != "true"].reset_index(drop=True)

    if src.empty:
        raise ValueError("No rows available after holdout filtering.")

    rng = random.Random(seed)
    pos = src[["species", "sequence"]].copy()
    pos["label"] = 1
    pos["origin"] = "t9_holdout_positive"

    neg = pos.copy()
    neg["sequence"] = [dinucleotide_shuffle(s, rng) for s in pos["sequence"].tolist()]
    neg["label"] = 0
    neg["origin"] = "t9_dinuc_shuffle_negative"

    ds = pd.concat([pos, neg], ignore_index=True)
    ds = ds.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return ExternalDataset(df=ds, positives=len(pos), negatives=len(neg))


@torch.no_grad()
def predict_transformer(model, tokenizer, sequences: List[str], device: torch.device, batch_size: int = 64) -> np.ndarray:
    out: List[np.ndarray] = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = torch.softmax(logits.float(), dim=1)[:, 1].detach().cpu().numpy()
        out.append(probs)
    return np.concatenate(out)


@torch.no_grad()
def predict_cnn(model, sequences: List[str], device: torch.device, batch_size: int = 256) -> np.ndarray:
    out: List[np.ndarray] = []
    X = np.stack([one_hot_81(s) for s in sequences]).astype(np.float32)
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        logits = model(xb)
        probs = torch.softmax(logits.float(), dim=1)[:, 1].detach().cpu().numpy()
        out.append(probs)
    return np.concatenate(out)


def load_cnn_model(cnn_model_path: Path, device: torch.device):
    import importlib.util

    repo = Path(__file__).resolve().parents[2]
    t20_path = (repo / "model-training" / "T20" / "train_cnn.py").resolve()

    spec = importlib.util.spec_from_file_location("train_cnn", str(t20_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cfg = {}
    metrics_path = cnn_model_path.parent / "metrics.json"
    if metrics_path.exists():
        cfg = json.loads(metrics_path.read_text()).get("config", {})

    keys = ("conv1_filters", "conv2_filters", "conv3_filters", "kernel_size", "dropout_rate", "use_batch_norm")
    kwargs = {k: cfg[k] for k in keys if k in cfg}

    model = mod.PromoterCNN(input_channels=4, sequence_length=81, **kwargs).to(device)
    model.load_state_dict(torch.load(cnn_model_path, map_location=device))
    model.eval()
    return model


def load_cnn_v2_model(cnn_v2_model_path: Path, device: torch.device):
    """Load the multi-encoding CNN-v2. Returns (module, model, enac_window); the module
    exposes predict_proba, which takes raw sequences (v2 computes its own features)."""
    import importlib.util

    repo = Path(__file__).resolve().parents[2]
    path = (repo / "model-training" / "T20" / "train_cnn_v2.py").resolve()
    spec = importlib.util.spec_from_file_location("train_cnn_v2", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cfg = {}
    cfg_path = cnn_v2_model_path.parent / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())

    model = mod.build_model_from_config(cfg, device=str(device))
    model.load_state_dict(torch.load(cnn_v2_model_path, map_location=device))
    model.eval()
    return mod, model, int(cfg.get("enac_window", 5))


def metric_dict(y: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    m = {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "prevalence": float(y.mean()),
        "threshold": float(threshold),
        "auprc": float(average_precision_score(y, prob)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
    }
    if len(np.unique(y)) >= 2:
        m["auroc"] = float(roc_auc_score(y, prob))
    else:
        m["auroc"] = float("nan")
    return m


def bootstrap_ci(y: np.ndarray, prob: np.ndarray, seed: int = BOOT_SEED, nboot: int = NBOOT) -> Dict[str, float]:
    """90% CI for AUROC and AUPRC, resampling positives and negatives separately (keeps the 1:1
    balance). Important here because the per-species strata are tiny (C. phytofermentans n=25)."""
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    keys = ("auroc_lo", "auroc_hi", "auprc_lo", "auprc_hi")
    if len(pos) == 0 or len(neg) == 0:
        return {k: float("nan") for k in keys}
    rng = np.random.default_rng(seed)
    aurocs, auprcs = [], []
    for _ in range(nboot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        yb, pb = y[idx], prob[idx]
        if len(np.unique(yb)) < 2:
            continue
        aurocs.append(roc_auc_score(yb, pb))
        auprcs.append(average_precision_score(yb, pb))
    qq = lambda a, p: float(np.percentile(a, p)) if a else float("nan")
    return {"auroc_lo": qq(aurocs, 5), "auroc_hi": qq(aurocs, 95),
            "auprc_lo": qq(auprcs, 5), "auprc_hi": qq(auprcs, 95)}


def evaluate_all(
    data: pd.DataFrame,
    prokbert_model_path: Path | None,
    nt_adapter_path: Path | None,
    cnn_model_path: Path | None,
    cnn_v2_model_path: Path | None,
    out_dir: Path,
    nt_base_model: str,
    threshold: float,
    models: List[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    y = data["label"].astype(int).to_numpy()
    seqs = data["sequence"].astype(str).tolist()
    species = data["species"].astype(str).to_numpy()

    metrics: Dict[str, Dict[str, float]] = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probs_by_model: Dict[str, np.ndarray] = {}

    # ProkBERT
    if "prokbert" in models:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if prokbert_model_path is None:
            raise ValueError("prokbert requested but --prokbert-model-path was not provided")
        tok_pk = AutoTokenizer.from_pretrained(PROKBERT_BASE, trust_remote_code=True)
        model_pk = AutoModelForSequenceClassification.from_pretrained(str(prokbert_model_path), trust_remote_code=True).to(device).eval()
        prob_pk = predict_transformer(model_pk, tok_pk, seqs, device)
        probs_by_model["prokbert"] = prob_pk
        metrics["prokbert"] = metric_dict(y, prob_pk, threshold)

    # NT LoRA
    if "nt_lora" in models:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from peft import PeftModel
        if nt_adapter_path is None:
            raise ValueError("nt_lora requested but --nt-adapter-path was not provided")
        tok_nt = AutoTokenizer.from_pretrained(nt_base_model, trust_remote_code=True)
        base_nt = AutoModelForSequenceClassification.from_pretrained(nt_base_model, num_labels=2, trust_remote_code=True)
        model_nt = PeftModel.from_pretrained(base_nt, str(nt_adapter_path)).to(device).eval()
        prob_nt = predict_transformer(model_nt, tok_nt, seqs, device)
        probs_by_model["nt_lora"] = prob_nt
        metrics["nt_lora"] = metric_dict(y, prob_nt, threshold)

    # CNN
    if "cnn" in models:
        if cnn_model_path is None:
            raise ValueError("cnn requested but --cnn-model-path was not provided")
        model_cnn = load_cnn_model(cnn_model_path, device)
        prob_cnn = predict_cnn(model_cnn, seqs, device)
        probs_by_model["cnn"] = prob_cnn
        metrics["cnn"] = metric_dict(y, prob_cnn, threshold)

    # CNN-v2 (multi-encoding fusion; computes its own features from raw sequences)
    if "cnn_v2" in models:
        if cnn_v2_model_path is None:
            raise ValueError("cnn_v2 requested but --cnn-v2-model-path was not provided")
        v2mod, model_cnn_v2, enac_w = load_cnn_v2_model(cnn_v2_model_path, device)
        prob_cnn_v2 = v2mod.predict_proba(model_cnn_v2, seqs, str(device), enac_window=enac_w)
        probs_by_model["cnn_v2"] = prob_cnn_v2
        metrics["cnn_v2"] = metric_dict(y, prob_cnn_v2, threshold)

    # Save PER-MODEL predictions, overall metrics, and species breakdown. Per-model files mean that
    # running this script twice into the same --out (prokbert,cnn in one env, then nt_lora in the
    # nt-tf4x env) ACCUMULATES results instead of the second run clobbering the first.
    for model_name, probs in probs_by_model.items():
        pd.DataFrame({
            "species": species,
            "label": y,
            "prob": probs,
            "pred": (probs >= threshold).astype(int),
        }).to_csv(pred_dir / f"{model_name}_external_predictions.csv", index=False)

        (out_dir / f"{model_name}_metrics.json").write_text(json.dumps(metrics[model_name], indent=2) + "\n")

        srows = []
        sdf = pd.DataFrame({"species": species, "label": y, "prob": probs})
        for sp, g in sdf.groupby("species"):
            yy = g["label"].to_numpy()
            pp = g["prob"].to_numpy()
            md = metric_dict(yy, pp, threshold)
            md.update(bootstrap_ci(yy, pp))          # auroc_lo/hi, auprc_lo/hi (90% CI)
            md["model"] = model_name
            md["species"] = str(sp)
            srows.append(md)
        pd.DataFrame(srows).to_csv(out_dir / f"{model_name}_species_metrics.csv", index=False)

    # Rebuild the combined summary + species table from ALL per-model files present (so a later
    # NT-only run merges with an earlier prokbert/cnn run rather than overwriting it).
    combined = {json.loads(f.read_text()).get("model", f.stem.replace("_metrics", "")): json.loads(f.read_text())
                for f in sorted(out_dir.glob("*_metrics.json"))}
    (out_dir / "metrics_summary.json").write_text(json.dumps(combined, indent=2) + "\n")

    sp_files = [pd.read_csv(f) for f in sorted(out_dir.glob("*_species_metrics.csv"))]
    if sp_files:
        pd.concat(sp_files, ignore_index=True).to_csv(out_dir / "species_metrics.csv", index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="T24 external holdout evaluation (new script).")
    p.add_argument("--t9-holdout", default="inputs/literature_promoters_holdout.tsv",
                   help="Holdout promoter file from T9-split (tsv/parquet).")
    p.add_argument("--fallback-t9", default="data-acquisition-and-curation/synthetic-promoters/literature_promoters.tsv",
                   help="Fallback T9 file if --t9-holdout is missing.")
    p.add_argument("--out-of-genus-only", action="store_true",
                   help="Evaluate only rows with in_genus != True.")
    p.add_argument("--include-in-genus", action="store_true",
                   help="Include in-genus holdout rows as well (overrides --out-of-genus-only).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)

    p.add_argument("--models", default="prokbert,nt_lora,cnn",
                   help="Comma-separated subset to evaluate: prokbert, nt_lora, cnn, cnn_v2")

    p.add_argument("--prokbert-model-path",
                   help="Path to T18/T21 ProkBERT model dir (contains config + weights).")
    p.add_argument("--nt-adapter-path",
                   help="Path to T19/T21 NT LoRA adapter dir.")
    p.add_argument("--cnn-model-path",
                   help="Path to T20/T21 CNN best_model.pt.")
    p.add_argument("--cnn-v2-model-path",
                   help="Path to T21_final/cnn_v2 best_model.pt (config.json read from same dir).")
    p.add_argument("--nt-base-model", default=NT_ID)

    p.add_argument("--out", default="results/T24")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    valid = {"prokbert", "nt_lora", "cnn", "cnn_v2"}
    bad = [m for m in models if m not in valid]
    if bad:
        raise ValueError(f"Unknown model(s): {bad}. Choose from {sorted(valid)}")

    out_of_genus_only = True
    if args.include_in_genus:
        out_of_genus_only = False
    elif args.out_of_genus_only:
        out_of_genus_only = True

    holdout_path = Path(args.t9_holdout)
    if not holdout_path.exists():
        holdout_path = Path(args.fallback_t9)

    holdout_df = load_t9_holdout(holdout_path)
    ext = build_external_dataset(
        holdout_df,
        out_of_genus_only=out_of_genus_only,
        seed=args.seed,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext.df.to_csv(out_dir / "external_eval_dataset.csv", index=False)

    manifest = {
        "models": models,
        "t9_source": str(holdout_path),
        "rows": int(len(ext.df)),
        "positives": int(ext.positives),
        "negatives": int(ext.negatives),
        "out_of_genus_only": bool(out_of_genus_only),
        "negative_generation": "dinucleotide_shuffle_1_to_1",
        "threshold": float(args.threshold),
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    evaluate_all(
        data=ext.df,
        prokbert_model_path=(Path(args.prokbert_model_path) if args.prokbert_model_path else None),
        nt_adapter_path=(Path(args.nt_adapter_path) if args.nt_adapter_path else None),
        cnn_model_path=(Path(args.cnn_model_path) if args.cnn_model_path else None),
        cnn_v2_model_path=(Path(args.cnn_v2_model_path) if args.cnn_v2_model_path else None),
        out_dir=out_dir,
        nt_base_model=args.nt_base_model,
        threshold=args.threshold,
        models=models,
    )

    print(f"T24 external evaluation complete: {out_dir}")
    print(f"Dataset rows: {len(ext.df)} (pos={ext.positives}, neg={ext.negatives})")


if __name__ == "__main__":
    main()
