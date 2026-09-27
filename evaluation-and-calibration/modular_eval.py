#!/usr/bin/env python3
"""
Modular evaluation of the held-out test predictions — two breakdowns T22 and T25 don't give.

  1. NEGATIVES BY CLASS.  For each of the four negative types (SRS, CDS, Intergenic,
     Downstream), how well the model separates real positives from THAT class alone.
        * rejection@0.5   = fraction of the class correctly scored below 0.5 (its true-negative rate)
        * AUROC(pos vs class) = how separable positives are from this negative type only.
     A HIGH AUROC / rejection => the class is "easy" and the model may be keying on its
     signature rather than the promoter motif (Umarov's critique). A LOW one => that class is
     genuinely hard. SRS (composition-matched shuffle) is expected to be the hardest; if CDS /
     Intergenic / Downstream are near-perfect while SRS is much lower, that is direct evidence
     the structured negatives are "too easy" and the four classes are separable by type.

  2. BY SPECIES.  Per-species recall on that species' positives and rejection on its negatives,
     to expose any species that is systematically harder (missed promoters, or a genomic
     background that produces false positives). NB the 166 Clostridium NEGATIVE genomes are
     labelled at genome granularity in `species`, so per-genome negative n is small (flagged
     with ⚠ and pooled into one summary row); the POSITIVE species (E. coli / B. subtilis /
     C. acetobutylicum-beijerinckii) are well powered.

Inputs (identical to T25):
  --t15-dir results/T15   test.parquet: label, source, species (+ sequence, ...), row-aligned with T22
  --t22-dir results/T22   {model}_test_predictions.csv (columns: label, prob, pred)
Outputs:
  <out>/modular_eval.md    readable tables
  <out>/modular_eval.csv   tidy rows

sklearn + pandas; runs in the main $ENV_MAIN env. From the scaffold dir:
  python evaluation-and-calibration/modular_eval.py --t15-dir results/T15 --t22-dir results/T22 --out results/T25b
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MODELS = [("prokbert", "ProkBERT"), ("nt", "NT-LoRA"), ("cnn", "CNN"), ("cnn_v2", "CNN-v2")]
THR = 0.5           # operating point for rejection / recall (uncalibrated score)
NBOOT = 200         # bootstrap resamples for the AUROC CI
MIN_N = 20          # strata below this are flagged / pooled
MISSING = {"", "?", "nan", "NaN", "None", "<NA>", "na", "null", "Clostridium_sp"}

NEG_DISPLAY = {
    "SRS": "SRS (composition-matched shuffle)",
    "T14_SRS": "SRS (composition-matched shuffle)",
    "T14_CDS": "CDS (coding interior)",
    "T14_Intergenic_NonPromoter": "Intergenic (convergent)",
    "T14_Downstream_3Prime": "Downstream / 3'",
}


def auroc_ci(pos_p, neg_p, rng, nboot=NBOOT):
    """AUROC of the (fixed) positives vs a negative subset; 90% CI by resampling the negatives."""
    if len(pos_p) == 0 or len(neg_p) == 0:
        return (np.nan, np.nan, np.nan)
    y = np.concatenate([np.ones(len(pos_p)), np.zeros(len(neg_p))])
    point = roc_auc_score(y, np.concatenate([pos_p, neg_p]))
    if len(neg_p) < 3:
        return (point, np.nan, np.nan)
    bs = np.empty(nboot)
    for i in range(nboot):
        bn = rng.choice(neg_p, size=len(neg_p), replace=True)
        y2 = np.concatenate([np.ones(len(pos_p)), np.zeros(len(bn))])
        bs[i] = roc_auc_score(y2, np.concatenate([pos_p, bn]))
    return (point, float(np.percentile(bs, 5)), float(np.percentile(bs, 95)))


def auroc_ci_pos(pos_p, neg_p, rng, nboot=NBOOT):
    """AUROC of a (small) positive subset vs the shared negatives; 90% CI by resampling the positives."""
    if len(pos_p) == 0 or len(neg_p) == 0:
        return (np.nan, np.nan, np.nan)
    y = np.concatenate([np.ones(len(pos_p)), np.zeros(len(neg_p))])
    point = roc_auc_score(y, np.concatenate([pos_p, neg_p]))
    if len(pos_p) < 3:
        return (point, np.nan, np.nan)
    bs = np.empty(nboot)
    for i in range(nboot):
        bp = rng.choice(pos_p, size=len(pos_p), replace=True)
        y2 = np.concatenate([np.ones(len(bp)), np.zeros(len(neg_p))])
        bs[i] = roc_auc_score(y2, np.concatenate([bp, neg_p]))
    return (point, float(np.percentile(bs, 5)), float(np.percentile(bs, 95)))


def eff_species(spc, src):
    """Species column, falling back to source where species is missing (so positives that
    carry no species label still group by E. coli / B. subtilis / Hocq rather than '?')."""
    out = spc.copy().astype(object)
    for i, v in enumerate(spc):
        if str(v).strip() in MISSING:
            out[i] = str(src[i])
    return out.astype(str)


def load_preds(t22_dir, key, y):
    pp = os.path.join(t22_dir, f"{key}_test_predictions.csv")
    if not os.path.exists(pp):
        return None
    pred = pd.read_csv(pp)
    if len(pred) != len(y):
        raise SystemExit(f"{key}: {len(pred)} preds != {len(y)} test rows — not aligned.")
    if "label" in pred.columns and not np.array_equal(pred["label"].astype(int).to_numpy(), y):
        raise SystemExit(f"{key}: prediction labels disagree with T15 test labels.")
    if "prob" not in pred.columns:
        raise SystemExit(f"{key}: prediction file {pp} has no 'prob' column; has {list(pred.columns)}")
    prob = pred["prob"].to_numpy(float)
    if not np.all(np.isfinite(prob)):
        raise SystemExit(f"{key}: non-finite probabilities in {pp}.")
    return prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t15-dir", default="results/T15")
    ap.add_argument("--t22-dir", default="results/T22")
    ap.add_argument("--out", default="results/T25b")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    meta = pd.read_parquet(os.path.join(a.t15_dir, "test.parquet"))
    if "label" not in meta.columns:
        raise SystemExit("test.parquet has no `label` column.")
    y = meta["label"].astype(int).to_numpy()
    src = meta["source"].astype(str).to_numpy() if "source" in meta.columns else np.array(["?"] * len(y))
    spc = meta["species"].astype(str).to_numpy() if "species" in meta.columns else np.array(["?"] * len(y))
    spc = eff_species(spc, src)
    npos = int((y == 1).sum())

    md = ["# Modular evaluation — negatives by class, and by species\n",
          f"Test set: {len(y)} windows, {npos} positive, {int((y == 0).sum())} negative. "
          f"rejection / recall at threshold {THR} on the uncalibrated score; AUROC 90% CI by "
          f"bootstrap. `⚠` = n < {MIN_N} (do not over-read).\n"]
    rows = []

    for key, name in MODELS:
        prob = load_preds(a.t22_dir, key, y)
        if prob is None:
            md.append(f"\n## {name}\n_No predictions found — skipped._")
            continue
        pos_p = prob[y == 1]
        md.append(f"\n## {name}\n")

        # ---------- 1. negatives by class ----------
        md.append("\n**Negatives by class** — separability of real positives from each negative "
                  "type (higher = the model rejects that class more easily). Hardest first.\n")
        md.append(f"| Negative class | n | median score | rejection@{THR} | AUROC(pos vs class) [90% CI] |")
        md.append("|---|---|---|---|---|")
        block = []
        for s in sorted(set(src[y == 0])):
            m = (y == 0) & (src == s)
            n = int(m.sum())
            if n == 0:
                continue
            cp = prob[m]
            pt, lo, hi = auroc_ci(pos_p, cp, rng)
            block.append((pt, s, n, float(np.median(cp)), float((cp < THR).mean()), lo, hi))
        block.sort(key=lambda r: (np.inf if not np.isfinite(r[0]) else r[0]))   # lowest AUROC (hardest) first
        for pt, s, n, ms, rej, lo, hi in block:
            disp = NEG_DISPLAY.get(s, s)
            ci = f"[{lo:.2f}–{hi:.2f}]" if np.isfinite(lo) else "[n<3]"
            flag = "" if n >= MIN_N else " ⚠"
            md.append(f"| {disp}{flag} | {n} | {ms:.3f} | {rej:.3f} | {pt:.3f} {ci} |")
            rows.append({"model": name, "axis": "negative_class", "stratum": disp, "n": n,
                         "median_score": round(ms, 4), "rejection": round(rej, 4),
                         "auroc_pos_vs_class": round(pt, 4) if np.isfinite(pt) else "",
                         "ci_lo": round(lo, 4) if np.isfinite(lo) else "",
                         "ci_hi": round(hi, 4) if np.isfinite(hi) else ""})

        # ---------- 2a. positives by species (Hocq split into C. acetobutylicum + C. beijerinckii) ----------
        neg_all = prob[y == 0]
        md.append("\n**Positives by species** — recognition of each species' promoters. Hocq 2022 is "
                  "split into *C. acetobutylicum* and *C. beijerinckii* (they carry distinct `species` "
                  "labels; only the `source` was 'Hocq2022').\n")
        md.append(f"| Species | n_pos | recall@{THR} | AUROC (vs all neg) [90% CI] |")
        md.append("|---|---|---|---|")
        psp = []
        for s in sorted(set(spc[y == 1])):
            m = (spc == s) & (y == 1)
            n = int(m.sum())
            if n == 0:
                continue
            pp = prob[m]
            pt, lo, hi = auroc_ci_pos(pp, neg_all, rng)
            psp.append((n, s, float((pp >= THR).mean()), pt, lo, hi))
        psp.sort(key=lambda r: -r[0])
        for n, s, rec, pt, lo, hi in psp:
            ci = f"[{lo:.2f}–{hi:.2f}]" if np.isfinite(lo) else "[n<3]"
            flag = "" if n >= MIN_N else " ⚠"
            md.append(f"| {s}{flag} | {n} | {rec:.3f} | {pt:.3f} {ci} |")
            rows.append({"model": name, "axis": "positive_species", "stratum": s, "n": n,
                         "recall": round(rec, 4), "auroc_vs_allneg": round(pt, 4) if np.isfinite(pt) else "",
                         "ci_lo": round(lo, 4) if np.isfinite(lo) else "",
                         "ci_hi": round(hi, 4) if np.isfinite(hi) else ""})

        # ---------- 2b. negative background by genome ----------
        gstats = []
        for s in sorted(set(spc[y == 0])):
            m = (spc == s) & (y == 0)
            n = int(m.sum())
            if n == 0:
                continue
            gstats.append((float((prob[m] < THR).mean()), s, n, float(np.median(prob[m]))))
        overall_rej = float((neg_all < THR).mean()) if len(neg_all) else np.nan
        md.append(f"\n**Negative background by genome** — per-source rejection@{THR}; lowest = most "
                  f"false-positive-prone. The 166 Clostridium negatives are genome-level so mostly small "
                  f"(⚠). Overall rejection **{overall_rej:.3f}** over {len(neg_all)} negatives / {len(gstats)} groups.\n")
        md.append("| Genome / source | n_neg | rejection | median score |")
        md.append("|---|---|---|---|")
        gstats.sort(key=lambda r: r[0])   # worst (lowest rejection) first
        for rej, s, n, ms in gstats[:12]:
            flag = "" if n >= MIN_N else " ⚠"
            md.append(f"| {s}{flag} | {n} | {rej:.3f} | {ms:.3f} |")
            rows.append({"model": name, "axis": "negative_genome", "stratum": s, "n": n,
                         "rejection": round(rej, 4), "median_score": round(ms, 4)})
        if len(gstats) > 12:
            md.append(f"| _(+{len(gstats) - 12} more source groups)_ |  |  |  |")

    pd.DataFrame(rows).to_csv(os.path.join(a.out, "modular_eval.csv"), index=False)
    text = "\n".join(md) + "\n"
    with open(os.path.join(a.out, "modular_eval.md"), "w") as f:
        f.write(text)
    print(text)
    print("Wrote", os.path.join(a.out, "modular_eval.{md,csv}"))


if __name__ == "__main__":
    main()
