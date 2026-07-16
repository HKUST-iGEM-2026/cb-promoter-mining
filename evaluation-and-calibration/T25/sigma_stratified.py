#!/usr/bin/env python3
"""
T25 — stratified evaluation of the held-out test predictions.

The brief is "sigma-factor-stratified eval," but the sigma labels are too sparse and cross-species to
support strong conclusions on their own (see the run-time header). So this reports THREE complementary
stratifications of each model's TEST performance — all from data already on disk (T15 test metadata +
T22 predictions) — strongest first:

  1. by SOURCE / species  — the cross-species vs in-genus generalization gap (dense, decision-relevant)
  2. by GC-content quartile — a DIRECT, label-free readout of the AT-composition confound (every window)
  3. by SIGMA factor       — the named task, honestly caveated; a corroboration of (2), not a standalone

WHY sigma alone is weak (council-reviewed):
  * The only sigma labels come from T8: RegulonDB E. coli = real per-promoter sigmas; DBTBS B. subtilis =
    a BLANKET hardcoded 'SigmaA' (run_t8_pipeline.py L89). T7b (in-genus) and T13 (silver) have NO sigma.
  * So sigma is CROSS-SPECIES ONLY (no in-genus sigma exists) and is CONFOUNDED WITH SPECIES ('SigmaA' is
    exactly the B. subtilis set) — sigma-vs-sigma claims are only valid WITHIN RegulonDB.
  * The GC-motif class sigma54 is ~2.5% of RegulonDB, i.e. n≈10 in this test split — barely powered.
  ==> Treat the sigma table as a consistency check on the composition story, never as an in-genus result
      or a headline. The GC-quartile cut (axis 2) is the dense, honest version of the same question.

Metrics per stratum (all calibration-free + comparable across strata):
  n_pos · median raw score · median RANK-PERCENTILE (position among ALL test scores) · TWO AUROCs — vs the
  full negative pool AND vs the SRS-only (composition-matched dinucleotide-shuffle) negatives; the all->SRS
  drop is the composition contribution (small drop = genuine motif discrimination). Positives-bootstrap 90%
  CIs. Per-stratum AUPRC and recall@0.5 are deliberately omitted — AUPRC's baseline = n_pos/(n_pos+n_neg)
  varies by stratum (not comparable), and 0.5 is meaningless on an uncalibrated, balanced-trained score.

Inputs (NOT the Snakefile's T10 TSS calls — those carry no sigma labels and are out-of-genus):
  --t15-dir results/T15   test.parquet: sequence (for GC), label, source, sigma_factor — row-aligned w/ T22
  --t22-dir results/T22   {model}_test_predictions.csv from predict_and_score.py (columns: label, prob, pred)
Outputs:
  results/T25/stratified_eval.csv   (tidy: model, axis, stratum, n_pos, median_score, median_pctl,
                                     auroc_allneg, ci_lo, ci_hi, auroc_srs, srs_ci_lo, srs_ci_hi)
  results/T25/stratified_eval.md    (readable tables + the coverage/limits header)

sklearn + pandas; runs in the main promoter-mining-hpc4 env. Run from the scaffold dir:
  python evaluation-and-calibration/T25/sigma_stratified.py
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MODELS = [("prokbert", "ProkBERT"), ("nt", "NT-LoRA"), ("cnn", "CNN")]
MIN_N = 10       # strata with fewer positives are shown but flagged low-confidence
NBOOT = 200      # positives-bootstrap resamples for the AUROC CI


def norm_sigma(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    return None if s in ("", "nan", "NaN", "None", "Unknown", "unknown", "<NA>", "na", "null") else s


def gc_frac(seq):
    s = str(seq).upper()
    return (s.count("G") + s.count("C")) / len(s) if len(s) else np.nan


def _au(pos_p, neg_p):
    if len(pos_p) == 0 or len(neg_p) == 0:
        return np.nan
    return roc_auc_score(np.concatenate([np.ones(len(pos_p)), np.zeros(len(neg_p))]),
                         np.concatenate([pos_p, neg_p]))


def _ci(arr):
    a = arr[np.isfinite(arr)]
    return (float(np.percentile(a, 5)), float(np.percentile(a, 95))) if len(a) else (np.nan, np.nan)


def auroc_vs(pos_p, neg_all, neg_srs, rng, nboot=NBOOT):
    """AUROC of this stratum's positives vs (a) the full negative pool and (b) the SRS-only
    (composition-matched dinucleotide-shuffle) pool, each with a 90% CI from the SAME positive
    bootstrap so the two columns are directly comparable. The all→SRS drop = composition's share."""
    p_all, p_srs = _au(pos_p, neg_all), _au(pos_p, neg_srs)
    if len(pos_p) < 3:
        return (p_all, np.nan, np.nan, p_srs, np.nan, np.nan)
    b_all, b_srs = np.empty(nboot), np.empty(nboot)
    for i in range(nboot):
        bp = rng.choice(pos_p, size=len(pos_p), replace=True)
        b_all[i], b_srs[i] = _au(bp, neg_all), _au(bp, neg_srs)
    (la, ha), (ls, hs) = _ci(b_all), _ci(b_srs)
    return (p_all, la, ha, p_srs, ls, hs)


def strata_source(src, y):
    for v in sorted(set(src[y == 1])):
        yield (f"source = {v}", (src == v) & (y == 1))


def strata_gc(gcv, y):
    idx = np.where((y == 1) & np.isfinite(gcv))[0]
    if len(idx) < 8:
        return
    binned = pd.qcut(pd.Series(gcv[idx], index=idx), 4, duplicates="drop")
    for interval, grp in binned.groupby(binned, observed=True):
        mask = np.zeros(len(y), dtype=bool)
        mask[grp.index.to_numpy()] = True
        yield (f"GC {interval.left * 100:.0f}–{interval.right * 100:.0f}%", mask)


def strata_sigma(sig, y):
    labeled = sorted({s for i, s in enumerate(sig) if s is not None and y[i] == 1},
                     key=lambda s: (-int(((sig == s) & (y == 1)).sum()), s))
    for s in labeled:
        disp = "B. subtilis (DBTBS, sigma unassigned)" if s == "SigmaA" else s
        yield (f"sigma = {disp}", (sig == s) & (y == 1))
    unl = np.array([x is None for x in sig]) & (y == 1)
    if unl.sum():
        yield ("sigma = unlabeled (mostly in-genus T7b)", unl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t15-dir", default="results/T15")
    ap.add_argument("--t22-dir", default="results/T22")
    ap.add_argument("--out", default="results/T25")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    mpath = os.path.join(a.t15_dir, "test.parquet")
    if not os.path.exists(mpath):
        raise SystemExit(f"{mpath} not found — run T15 first.")
    meta = pd.read_parquet(mpath)
    if "label" not in meta.columns:
        raise SystemExit("T15 test.parquet has no `label` column.")
    y = meta["label"].astype(int).to_numpy()
    src = meta["source"].astype(str).to_numpy() if "source" in meta.columns else np.array(["?"] * len(y))
    sig = (meta["sigma_factor"].map(norm_sigma).to_numpy()
           if "sigma_factor" in meta.columns else np.array([None] * len(y), dtype=object))
    gcv = (meta["sequence"].map(gc_frac).to_numpy()
           if "sequence" in meta.columns else np.full(len(y), np.nan))
    srs_mask = (y == 0) & pd.Series(src).str.upper().str.contains("SRS", na=False).to_numpy()

    n_pos = int((y == 1).sum())
    n_lab = int(sum(1 for i in range(len(y)) if y[i] == 1 and sig[i] is not None))
    md = [
        "# T25 — stratified evaluation (source / GC / sigma)\n",
        f"Test set: {len(y)} windows, {n_pos} positive, {int((y == 0).sum())} negative (shared pool). "
        f"**Only {n_lab}/{n_pos} positives carry a sigma label** (cross-species; DBTBS = blanket 'SigmaA'); "
        "there are NO in-genus sigma labels. Metrics: median rank-percentile (position among all test "
        "scores) and two AUROCs (calibration-free, comparable across strata; 90% CI by resampling "
        f"positives). `⚠` = n < {MIN_N} (do not narrate as a finding).\n",
        f"**AUROC vs all-neg** = positives vs the full genomic negative mix. **AUROC vs SRS** = positives "
        f"vs the {int(srs_mask.sum())} composition-matched dinucleotide-shuffle negatives (the "
        "in-distribution motif test) — a large all→SRS drop means the model was leaning on base "
        "composition; a small drop means genuine motif discrimination.\n",
        "Sigma is a weak, cross-species proxy for the composition confound — the **GC quartile** cut is "
        "the dense, in-genus-inclusive version. Sigma-vs-sigma is only valid within RegulonDB (E. coli); "
        "the 'SigmaA' row is really the B. subtilis set (species-confounded).\n",
    ]

    rows = []
    for key, name in MODELS:
        pp = os.path.join(a.t22_dir, f"{key}_test_predictions.csv")
        if not os.path.exists(pp):
            md.append(f"\n## {name}\n_No predictions at {pp} — skipped._")
            continue
        pred = pd.read_csv(pp)
        if len(pred) != len(y):
            raise SystemExit(f"{key}: {len(pred)} predictions != {len(y)} test rows — not aligned.")
        if "label" in pred.columns and not np.array_equal(pred["label"].astype(int).to_numpy(), y):
            raise SystemExit(f"{key}: prediction labels disagree with T15 test labels — row misalignment.")
        prob = pred["prob"].to_numpy(dtype=float)
        if not np.all(np.isfinite(prob)):
            raise SystemExit(f"{key}: non-finite probabilities in {pp}.")
        pctl = pd.Series(prob).rank(pct=True).to_numpy()
        neg_all = prob[y == 0]
        neg_srs = prob[srs_mask]

        md.append(f"\n## {name}\n")
        axes = [("Overall", iter([("— all positives —", (y == 1))])),
                ("By source / species", strata_source(src, y)),
                ("By GC-content quartile (direct composition axis)", strata_gc(gcv, y)),
                ("By sigma factor (cross-species only — see header)", strata_sigma(sig, y))]
        for axis_name, gen in axes:
            block = list(gen)
            if not block:
                continue
            md.append(f"\n**{axis_name}**\n")
            md.append("| Stratum | n_pos | median score | median rank-pctl | AUROC vs all-neg [CI] | AUROC vs SRS [CI] |")
            md.append("|---|---|---|---|---|---|")
            for label, mask in block:
                n = int(mask.sum())
                if n == 0:
                    continue
                ms = float(np.median(prob[mask]))
                mp = float(np.median(pctl[mask]))
                pa, la, ha, ps, ls, hs = auroc_vs(prob[mask], neg_all, neg_srs, rng)
                ca = f"[{la:.2f}–{ha:.2f}]" if np.isfinite(la) else "[n<3]"
                cs = f"[{ls:.2f}–{hs:.2f}]" if np.isfinite(ls) else ("[n<3]" if np.isfinite(ps) else "")
                sa = f"{ps:.3f} {cs}".strip() if np.isfinite(ps) else "—"
                flag = "" if n >= MIN_N else " ⚠"
                md.append(f"| {label}{flag} | {n} | {ms:.3f} | {mp:.3f} | {pa:.3f} {ca} | {sa} |")
                rows.append({"model": name, "axis": axis_name, "stratum": label, "n_pos": n,
                             "median_score": round(ms, 4), "median_pctl": round(mp, 4),
                             "auroc_allneg": round(pa, 4) if np.isfinite(pa) else "",
                             "ci_lo": round(la, 4) if np.isfinite(la) else "",
                             "ci_hi": round(ha, 4) if np.isfinite(ha) else "",
                             "auroc_srs": round(ps, 4) if np.isfinite(ps) else "",
                             "srs_ci_lo": round(ls, 4) if np.isfinite(ls) else "",
                             "srs_ci_hi": round(hs, 4) if np.isfinite(hs) else ""})

    pd.DataFrame(rows).to_csv(os.path.join(a.out, "stratified_eval.csv"), index=False)
    text = "\n".join(md) + "\n"
    with open(os.path.join(a.out, "stratified_eval.md"), "w") as f:
        f.write(text)
    print(text)
    print(f"Wrote {a.out}/stratified_eval.csv and stratified_eval.md")
    print("\nNOTE: the Snakefile t25 rule input is `TSS_CALLS` (T10) — wrong for this (no sigma labels, "
          "out-of-genus). Change it to:\n"
          "  input: 'results/T15.done', 'results/T22.done'\n"
          "  shell: 'python .../T25/sigma_stratified.py --t15-dir results/T15 --t22-dir results/T22 "
          "--out results/T25 && touch {output}'\n"
          "Recommended future extensions (council): add an SRS-only-negative AUROC column (motif vs "
          "composition), and correlate score with FIMO -10/-35 vs -24/-12 motif strength (ties to T31).")


if __name__ == "__main__":
    main()
