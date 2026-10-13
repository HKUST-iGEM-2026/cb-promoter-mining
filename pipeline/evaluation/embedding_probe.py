#!/usr/bin/env python3
"""
embedding_probe.py — "Engineering Cycle 0" diagnostic: does a FROZEN foundation model's embeddings
already contain promoter signal, BEFORE any fine-tuning?  (implements Ishi's "map the embeddings" idea)

What it does, for any HF model id (ProkBERT, Nucleotide Transformer, ...):
  1. Freezes the model, embeds each sequence = MEAN-POOL of the last hidden layer (masked) -> one vector.
  2. Fits a logistic probe and reports AUPRC/AUROC against the supplied label prevalence.
     With enough cluster IDs, CV keeps clusters disjoint; that reduces measured sequence leakage,
     not every related-sequence, source or genome overlap.
  3. Re-probes positives against SRS-only negatives. With verified composition matching, this tests
     sequence-order information, not a unique -10/-35 recognition mechanism. Shuffle signatures
     and residual composition differences can also separate labels.
  4. UMAP (or PCA fallback) 2-D plot, coloured by label — for EXPLORATION only (a 2-D gap can be an artifact).

Above-baseline AP establishes separability of these sampled labels. It does not guarantee a useful
fine-tune; its gap from another classifier isolates a fine-tuning gain only in a matched experiment.

Run (no fine-tuning needed; on offline compute nodes pre-stage the model first, see docs/SETUP.md):
  python embedding_probe.py --model-id neuralbioinfo/prokbert-mini --data results/T15/splits.parquet
  python embedding_probe.py --model-id InstaDeepAI/nucleotide-transformer-v2-250m-multi-species --data results/T15/splits.parquet

--data = any parquet with columns `sequence` + `label` (1=promoter, 0=negative); optional `cluster_id`
(for cluster-disjoint CV when sufficient groups exist) and a negative-subclass column naming SRS rows.
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import average_precision_score, roc_auc_score


@torch.no_grad()
def embed(seqs, tok, model, device, batch=32, max_len=128):
    """Mean-pool (masked) the last hidden layer -> (N, hidden) frozen embeddings."""
    out = []
    for i in range(0, len(seqs), batch):
        chunk = [str(s).upper() for s in seqs[i:i + batch]]          # both tokenizers expect UPPERCASE
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        o = model(**enc)
        h = o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]   # (B, L, H)
        m = enc["attention_mask"].unsqueeze(-1).float()                        # (B, L, 1)
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        out.append(mean.float().cpu().numpy())
    return np.vstack(out)


def probe(X, y, groups=None, seed=42):
    """Logistic-regression linear probe; cluster-disjoint CV if groups given. Returns (AUPRC, AUROC, baseline)."""
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(class_weight="balanced", max_iter=2000))
    if groups is not None and len(set(groups)) >= 5:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        groups = None
    proba = cross_val_predict(clf, X, y, cv=cv, groups=groups,
                              method="predict_proba", n_jobs=-1)[:, 1]
    return (float(average_precision_score(y, proba)),
            float(roc_auc_score(y, proba)),
            float(np.mean(y)))


def find_srs_mask(df, srs_col, srs_val):
    """Locate SRS (shuffled) negatives. Explicit --srs-col, else auto-scan object columns for the value."""
    if srs_col and srs_col in df.columns:
        return df[srs_col].astype(str).str.upper().eq(srs_val.upper()).to_numpy()
    for c in df.columns:
        if df[c].dtype == object:
            hit = df[c].astype(str).str.upper().eq(srs_val.upper())
            if hit.any():
                print(f"[srs] auto-detected SRS rows in column '{c}' ({int(hit.sum())} rows)")
                return hit.to_numpy()
    return None


def plot_2d(X, y, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import umap
        emb2 = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X)
        tag = "UMAP"
    except Exception as e:
        from sklearn.decomposition import PCA
        emb2 = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(X))
        tag = f"PCA (UMAP unavailable: {type(e).__name__})"
    plt.figure(figsize=(6, 5))
    for lab, name, color in [(0, "negative", "#888888"), (1, "promoter", "#d62728")]:
        s = y == lab
        plt.scatter(emb2[s, 0], emb2[s, 1], s=4, alpha=0.5, c=color, label=name)
    plt.legend(); plt.title(f"Frozen embeddings ({tag}) — exploration only, not a metric")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    print(f"[plot] wrote {path}  ({tag})")


def main():
    ap = argparse.ArgumentParser(description="Frozen-embedding linear probe for promoter signal")
    ap.add_argument("--model-id", default="neuralbioinfo/prokbert-mini")
    ap.add_argument("--data", default="results/T15/splits.parquet",
                    help="parquet with columns sequence,label (+optional cluster_id, SRS column)")
    ap.add_argument("--out", default="results/embedding_probe")
    ap.add_argument("--max-n", type=int, default=4000, help="subsample for speed (0 = use all)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--srs-col", default=None, help="column naming the negative subclass (else auto-detect)")
    ap.add_argument("--srs-val", default="SRS")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_parquet(args.data)
    for c in ("sequence", "label"):
        if c not in df.columns:
            raise SystemExit(f"{args.data}: missing required column '{c}' (need sequence + label)")
    df = df.dropna(subset=["sequence", "label"]).reset_index(drop=True)
    if args.max_n and len(df) > args.max_n:
        df = df.sample(args.max_n, random_state=args.seed).reset_index(drop=True)
    y = df["label"].astype(int).to_numpy()
    groups = df["cluster_id"].astype(str).to_numpy() if "cluster_id" in df.columns else None
    srs = find_srs_mask(df, args.srs_col, args.srs_val)
    print(f"[data] {len(df)} rows | positives {int(y.sum())} ({y.mean():.1%}) | "
          f"clusters {'yes' if groups is not None else 'no'} | SRS {'yes' if srs is not None else 'none found'}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[model] {args.model_id} on {device} (frozen)")
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True).to(device).eval()

    cache = os.path.join(args.out, "embeddings.npy")
    X = embed(df["sequence"].tolist(), tok, model, device, args.batch, args.max_len)
    np.save(cache, X)
    print(f"[embed] {X.shape} -> {cache}")

    results = {"model": args.model_id, "n": len(df), "prevalence": float(y.mean())}

    ap_all, auc_all, base = probe(X, y, groups, args.seed)
    results["all_negatives"] = {"auprc": ap_all, "auroc": auc_all, "baseline_auprc": base}
    print(f"\n[probe] positives vs ALL negatives:  AUPRC {ap_all:.3f}  (baseline {base:.3f})  AUROC {auc_all:.3f}")

    if srs is not None and srs.sum() >= 10:
        pos = y == 1
        # Positives vs SRS tests order sensitivity only with verified composition matching.
        sel = pos | srs
        ap_s, auc_s, base_s = probe(X[sel], y[sel], groups[sel] if groups is not None else None, args.seed)
        results["vs_SRS_only"] = {"auprc": ap_s, "auroc": auc_s, "baseline_auprc": base_s}
        print(f"[probe] positives vs SRS-only (order control): AUPRC {ap_s:.3f} (baseline {base_s:.3f}) AUROC {auc_s:.3f}")
        # (b) contrast: positives vs non-SRS negatives (composition differs -> can ride AT-content)
        sel2 = pos | (~pos & ~srs)
        ap_n, auc_n, base_n = probe(X[sel2], y[sel2], groups[sel2] if groups is not None else None, args.seed)
        results["vs_nonSRS_negatives"] = {"auprc": ap_n, "auroc": auc_n, "baseline_auprc": base_n}
        print(f"[probe] positives vs non-SRS negatives:     AUPRC {ap_n:.3f} (baseline {base_n:.3f}) AUROC {auc_n:.3f}")
        print("        Different SRS/non-SRS scores can reflect composition, source mix or shuffle "
              "signatures; this comparison does not isolate a motif mechanism.")
    else:
        print("[probe] no SRS negatives found -> skipping the supplied shuffle control "
              "(pass --srs-col/--srs-val, or include SRS rows).")

    plot_2d(X, y, os.path.join(args.out, "embedding_umap.png"))
    with open(os.path.join(args.out, "probe_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] {args.out}/probe_results.json")
    print("Interpretation: above-baseline AP shows separability of the supplied labels. "
          "SRS discrimination is not proof of a unique motif mechanism or a fine-tuning gain.")


if __name__ == "__main__":
    main()
