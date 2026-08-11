#!/usr/bin/env python3
"""
T26 — Consolidated analysis of the per-model calibration outputs.

Reads every `<model>/<model>_calibration_metrics.json` + `<model>_calibration_predictions.csv`
under a calibration directory (default: this dir / calib_2) and writes ONE report as markdown
and/or LaTeX (`--format md|tex|both`, default both; the .tex is rendered from the same content
so the two never drift, and is a self-contained compilable document unless `--tex-fragment`):

  1. Provenance  — which checkpoint each calibrator was fit on (surfaces the sweep-trial vs
                   T21_final_06 mismatch at a glance).
  2. Metrics     — raw vs OOF-calibrated (ECE / Brier / log-loss / AUROC / AUPRC) for every model,
                   plus the chosen calibrator (unregularized vs L2).
  3. AT ablation — score-only vs score+AT, recomputed from the CSV (raw_logit, at_fraction, label)
                   with a pure-python repeated-stratified-KFold logistic. This is the evidence for
                   "does the AT covariate earn its place for this model" — OOF log-loss for each,
                   the fitted AT coefficient sign, and its effect on the positives.
  4. Autopsy     — what the calibrator got most wrong: per-source error (raw vs calib), rows the
                   calibrator made worse than raw, right->wrong flips, and the worst-N rows.

Stdlib only (no numpy / sklearn / torch), so it runs anywhere — locally or on HPC4 — without the
conda env and without tripping the ~/.local numpy shadowing gotcha (CLAUDE.md S9).

Usage:
  python analyze_calibration.py                          # ./calib_2 -> calibration_analysis.{md,tex}
  python analyze_calibration.py --calib-dir calib_06 --format tex    # LaTeX only
  python analyze_calibration.py --calib-dir calib_06 --out report --top 10   # report.md + report.tex
"""
import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from md2tex import md_to_tex  # noqa: E402  (shared sibling renderer, stdlib-only)


# ------------------------------------------------------------------ pure-python logistic (for AT ablation)
def _fit_logistic(X, y, ridge=1e-6, iters=100):
    """Newton/IRLS logistic regression. X rows already include the bias term as X[i][0]."""
    n, p = len(X), len(X[0])
    w = [0.0] * p
    for _ in range(iters):
        g = [0.0] * p
        H = [[0.0] * p for _ in range(p)]
        for i in range(n):
            z = sum(w[j] * X[i][j] for j in range(p))
            pr = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            wt = pr * (1.0 - pr)
            r = pr - y[i]
            for a in range(p):
                g[a] += r * X[i][a]
                for b in range(p):
                    H[a][b] += wt * X[i][a] * X[i][b]
        for a in range(p):
            g[a] += ridge * w[a]
            H[a][a] += ridge
        M = [H[k][:] + [g[k]] for k in range(p)]
        for c in range(p):                                   # Gaussian elimination
            piv = max(range(c, p), key=lambda r: abs(M[r][c]))
            M[c], M[piv] = M[piv], M[c]
            if abs(M[c][c]) < 1e-12:
                continue
            for r in range(p):
                if r != c:
                    f = M[r][c] / M[c][c]
                    for k in range(c, p + 1):
                        M[r][k] -= f * M[c][k]
        d = [M[k][p] / M[k][k] if abs(M[k][k]) > 1e-12 else 0.0 for k in range(p)]
        for a in range(p):
            w[a] -= d[a]
        if sum(abs(x) for x in d) < 1e-8:
            break
    return w


def _zstats(cols):
    out = []
    for c in cols:
        mu = sum(c) / len(c)
        sd = math.sqrt(sum((x - mu) ** 2 for x in c) / len(c)) or 1.0
        out.append((mu, sd))
    return out


def _make_X(rows, idx, stats):
    return [[1.0] + [(r[i] - stats[k][0]) / stats[k][1] for k, i in enumerate(idx)] for r in rows]


def _predict(w, X):
    return [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, sum(w[j] * r[j] for j in range(len(w)))))))
            for r in X]


def _folds(y, n_splits=5, n_repeats=5, seed=42):
    rng = random.Random(seed)
    pos = [i for i, v in enumerate(y) if v == 1]
    neg = [i for i, v in enumerate(y) if v == 0]
    for _ in range(n_repeats):
        rng.shuffle(pos)
        rng.shuffle(neg)
        pf = [pos[k::n_splits] for k in range(n_splits)]
        nf = [neg[k::n_splits] for k in range(n_splits)]
        for k in range(n_splits):
            te = set(pf[k] + nf[k])
            yield [i for i in range(len(y)) if i not in te], sorted(te)


def _log_loss(y, p):
    return -sum(a * math.log(max(1e-12, b)) + (1 - a) * math.log(max(1e-12, 1 - b))
               for a, b in zip(y, p)) / len(y)


def at_ablation(rows, seed=42):
    """OOF score-only vs score+AT logistic on (raw_logit, at_fraction, label). rows = list of dicts.

    Returns dict with OOF log-loss for each feature set, the full-fit standardized AT coefficient,
    and mean OOF prob on positives (score-only vs +AT) so the effect of AT on the positives is visible.
    """
    logit = [r["raw_logit"] for r in rows]
    at = [r["at_fraction"] for r in rows]
    y = [r["label"] for r in rows]
    n = len(rows)
    L = [(logit[i], at[i]) for i in range(n)]

    def oof(idx):
        acc = [0.0] * n
        cnt = [0] * n
        for tr, te in _folds(y, seed=seed):
            stats = _zstats([[L[i][k] for i in tr] for k in idx])
            w = _fit_logistic(_make_X([L[i] for i in tr], idx, stats), [y[i] for i in tr])
            for i, p in zip(te, _predict(w, _make_X([L[i] for i in te], idx, stats))):
                acc[i] += p
                cnt[i] += 1
        return [acc[i] / cnt[i] for i in range(n)]

    ps, psa = oof([0]), oof([0, 1])
    stats = _zstats([[L[i][k] for i in range(n)] for k in (0, 1)])
    w = _fit_logistic(_make_X(L, [0, 1], stats), y)
    pos = [i for i in range(n) if y[i] == 1]
    return {
        "ll_score": _log_loss(y, ps),
        "ll_score_at": _log_loss(y, psa),
        "at_coef": w[2],
        "meanP_pos_score": sum(ps[i] for i in pos) / len(pos) if pos else float("nan"),
        "meanP_pos_score_at": sum(psa[i] for i in pos) / len(pos) if pos else float("nan"),
        "n_pos": len(pos),
    }


# ------------------------------------------------------------------ IO / discovery
def discover(calib_dir: Path):
    """Find (model_name, metrics_json_path, predictions_csv_path) for each model subdir."""
    found = []
    for sub in sorted(p for p in calib_dir.iterdir() if p.is_dir()):
        mj = next(iter(sub.glob("*_calibration_metrics.json")), None)
        pc = next(iter(sub.glob("*_calibration_predictions.csv")), None)
        if mj and pc:
            found.append((sub.name, mj, pc))
    return found


def load_predictions(path: Path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "sequence": r["sequence"], "label": int(r["label"]), "source": r["source"],
                "at_fraction": float(r["at_fraction"]), "raw_logit": float(r["raw_logit"]),
                "raw_prob": float(r["raw_prob"]), "calib_prob_at": float(r["calib_prob_at"]),
            })
    return rows


# ------------------------------------------------------------------ analysis
def prediction_summary(rows):
    n = len(rows)
    mce = sum(abs(r["calib_prob_at"] - r["label"]) for r in rows) / n
    mre = sum(abs(r["raw_prob"] - r["label"]) for r in rows) / n
    worse = sum(1 for r in rows if abs(r["calib_prob_at"] - r["label"]) > abs(r["raw_prob"] - r["label"]) + 1e-12)
    flips = sum(1 for r in rows
                if (r["raw_prob"] > .5) == (r["label"] == 1) and (r["calib_prob_at"] > .5) != (r["label"] == 1))
    by = {}
    for r in rows:
        by.setdefault(r["source"], []).append(r)
    src = {}
    for s, xs in by.items():
        src[s] = (len(xs),
                  sum(abs(x["raw_prob"] - x["label"]) for x in xs) / len(xs),
                  sum(abs(x["calib_prob_at"] - x["label"]) for x in xs) / len(xs))
    return dict(n=n, mce=mce, mre=mre, worse=worse, flips=flips, src=src)


def f(x, d=3):
    return "nan" if x != x else f"{x:.{d}f}"


METRIC_ORDER = [("ECE", "ece", True), ("Brier", "brier", True), ("Log-loss", "log_loss", True),
                ("AUROC", "auroc", False), ("AUPRC", "auprc", False)]


def build_report(calib_dir: Path, models, top_n: int, seed: int) -> str:
    M = {name: json.loads(mj.read_text()) for name, mj, _ in models}
    P = {name: load_predictions(pc) for name, _, pc in models}
    names = [n for n, _, _ in models]

    out = []
    w = out.append
    w("# T26 — Calibration Analysis (all models)\n")
    w(f"Consolidated from `{calib_dir}` — {len(names)} model(s): {', '.join(names)}. "
      "Each model's raw promoter score is recalibrated with an AT-content-aware Platt regressor "
      "(`StandardScaler -> LogisticRegression([logit, AT])`), selected by 5x5 stratified CV.\n")
    meta = M[names[0]]
    w(f"**Eval set:** {meta.get('n_total','?')} sequences — {meta.get('n_positive','?')} positives, "
      f"{meta.get('n_negative','?')} negatives "
      f"({meta.get('n_convergent','?')} convergent-intergenic + {meta.get('n_cds','?')} CDS-interior). "
      f"Window {meta.get('window','?')} bp.\n")

    # ---- 1. provenance ----
    w("---\n## 1. Provenance — which checkpoint each calibrator was fit on\n")
    w("| Model | chosen calibrator | features | checkpoint / adapter | n_conv / n_cds |")
    w("|---|---|---|---|---|")
    for nm in names:
        pth = M[nm].get("paths", {})
        ckpt = pth.get("checkpoint") or pth.get("model_dir") or pth.get("adapter_dir") or pth.get("config_json") or "?"
        feats = ", ".join(M[nm].get("chosen_features", [])) or "?"
        w(f"| {nm} | {M[nm].get('chosen','?')} | {feats} | `{ckpt}` | "
          f"{M[nm].get('n_convergent','?')} / {M[nm].get('n_cds','?')} |")
    w("\n> If any checkpoint path is under `results/T21/<model>/trial_*` (sweep trials) while the "
      "deployed/reported model is `T21_final_06/<model>`, that calibrator is fit on the wrong "
      "checkpoint and should be re-run with the matching `--model-dir`/`--trial-dir`/`--adapter-dir`.\n")

    # ---- 2. metrics ----
    w("---\n## 2. Metrics — raw vs OOF-calibrated\n")
    w("Calibrated = `calibrated_logit_at_oof` (the out-of-fold deployed behaviour). "
      "Lower is better for ECE / Brier / log-loss; higher for AUROC / AUPRC. Arrow = calibration helped.\n")
    header = "| Metric | " + " | ".join(f"{nm} raw" for nm in names) + " | " + \
             " | ".join(f"{nm} calib" for nm in names) + " |"
    w(header)
    w("|---|" + "---|" * (2 * len(names)))
    for lab, key, low in METRIC_ORDER:
        raw = [M[nm]["metrics"]["raw"][key] for nm in names]
        cal = [(M[nm]["metrics"].get("calibrated_oof") or M[nm]["metrics"]["calibrated_logit_at_oof"])[key]
               for nm in names]
        cells = [f(v) for v in raw]
        for i in range(len(names)):
            improved = (cal[i] < raw[i]) if low else (cal[i] > raw[i])
            cells.append(f"**{f(cal[i])}** {'▲' if improved else '▼'}")
        w(f"| {lab} | " + " | ".join(cells) + " |")
    w("")

    # ---- 3. AT ablation ----
    w("---\n## 3. Does the AT covariate earn its place? (score-only vs score+AT)\n")
    w("Recomputed from each CSV with a repeated-stratified-KFold logistic. `+AT` is worth keeping only "
      "if it lowers OOF log-loss; a **positive AT coef** means high-AT sequences are pushed *up* "
      "(AT-rich promoters rescued), negative means pushed *down*.\n")
    w("| Model | OOF log-loss score-only | OOF log-loss +AT | AT helps? | AT coef | mean P(positives) score-only -> +AT |")
    w("|---|---|---|---|---|---|")
    abl = {}
    for nm in names:
        a = at_ablation(P[nm], seed=seed)
        abl[nm] = a
        helps = "yes" if a["ll_score_at"] < a["ll_score"] else "no"
        w(f"| {nm} | {f(a['ll_score'],4)} | {f(a['ll_score_at'],4)} | **{helps}** | "
          f"{a['at_coef']:+.3f} | {f(a['meanP_pos_score'])} -> {f(a['meanP_pos_score_at'])} |")
    w("\n*Decision rule for the calibrator: keep +AT only where it lowers OOF log-loss; otherwise "
      "score-only (simpler) is the honest choice.*\n")

    # ---- 4. autopsy ----
    w("---\n## 4. Where the calibrator messed up\n")
    w("| Model | mean \\|calib-label\\| | mean \\|raw-label\\| | rows made worse | right->wrong flips |")
    w("|---|---|---|---|---|")
    S = {nm: prediction_summary(P[nm]) for nm in names}
    for nm in names:
        s = S[nm]
        flag = " ⚠️" if s["worse"] > s["n"] / 2 else ""
        w(f"| {nm} | {f(s['mce'])} | {f(s['mre'])} | {s['worse']}/{s['n']}{flag} | {s['flips']} |")
    w("")
    for nm in names:
        s = S[nm]
        w(f"### {nm}\n")
        w("**Calibration error by source class:**\n")
        w("| Source | n | raw err | calib err | Δ |")
        w("|---|---|---|---|---|")
        for src in sorted(s["src"]):
            n, mre, mce = s["src"][src]
            d = mce - mre
            w(f"| {src} | {n} | {f(mre)} | {f(mce)} | {'+' if d>0 else ''}{f(d)} |")
        w("")
        w(f"**Worst {top_n} rows** (highest `|calib - label|`):\n")
        w("| label | source | AT | raw | calib | err | Δ vs raw |")
        w("|---|---|---|---|---|---|---|")
        worst = sorted(P[nm], key=lambda r: abs(r["calib_prob_at"] - r["label"]), reverse=True)[:top_n]
        for r in worst:
            ce = abs(r["calib_prob_at"] - r["label"])
            d = ce - abs(r["raw_prob"] - r["label"])
            w(f"| {'POS' if r['label']==1 else 'NEG'} | {r['source']} | {f(r['at_fraction'],2)} | "
              f"{f(r['raw_prob'])} | {f(r['calib_prob_at'])} | **{f(ce)}** | {'+' if d>0 else ''}{f(d)} |")
        w("")

    # ---- takeaways ----
    w("---\n## 5. Takeaways\n")
    at_no = [nm for nm in names if abl[nm]["ll_score_at"] >= abl[nm]["ll_score"]]
    if at_no:
        w(f"- AT does **not** lower OOF log-loss for: {', '.join(at_no)} — use score-only there.")
    at_yes = [nm for nm in names if abl[nm]["ll_score_at"] < abl[nm]["ll_score"]]
    if at_yes:
        w(f"- AT **helps** (lower OOF log-loss, positive coef = rescues AT-rich positives) for: "
          f"{', '.join(at_yes)}.")
    worst_models = [nm for nm in names if S[nm]["worse"] > S[nm]["n"] / 2]
    if worst_models:
        w(f"- The calibrator made the majority of rows worse than raw for: {', '.join(worst_models)} "
          "— that is the base-rate/intercept level-shift, not necessarily the AT term (see §3).")
    w("- Ensemble aggregation (T29) is rank-percentile and calibration-invariant, so these distortions "
      "matter only for per-model probabilities / fixed thresholds, not the ensemble ranking.")
    w("\n---")
    w("<sub>`err = |probability - label|`; `Δ vs raw = calib_err - raw_err` (positive => calibration "
      "made the row worse). §3 ablation is recomputed from raw_logit + at_fraction + label with a "
      "stdlib logistic (5x5 stratified OOF).</sub>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="T26 consolidated calibration analysis -> markdown / LaTeX")
    ap.add_argument("--calib-dir", type=Path, default=Path(__file__).resolve().parent / "calib_2",
                    help="dir containing <model>/ subdirs with *_calibration_metrics.json + _predictions.csv")
    ap.add_argument("--out", type=Path, default=None,
                    help="output path (default: <calib-dir>/calibration_analysis.{md,tex}); the "
                         "suffix is set per --format, so any extension here is fine")
    ap.add_argument("--format", choices=["md", "tex", "both"], default="both",
                    help="which report(s) to write (default: both)")
    ap.add_argument("--tex-fragment", action="store_true",
                    help="emit the .tex WITHOUT the standalone document preamble (for \\input into a "
                         "larger LaTeX doc); default is a self-contained, compilable document")
    ap.add_argument("--top", type=int, default=8, help="worst-N rows to list per model")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    calib_dir = args.calib_dir.resolve()
    if not calib_dir.is_dir():
        raise SystemExit(f"ERROR: calib dir not found: {calib_dir}")
    models = discover(calib_dir)
    if not models:
        raise SystemExit(f"ERROR: no model subdirs with *_calibration_metrics.json + "
                         f"*_calibration_predictions.csv under {calib_dir}")
    base = (args.out or calib_dir / "calibration_analysis.md").resolve()
    report = build_report(calib_dir, models, args.top, args.seed)
    print(f"analysed {len(models)} model(s): {', '.join(n for n,_,_ in models)}")
    if args.format in ("md", "both"):
        md_path = base.with_suffix(".md")
        md_path.write_text(report)
        print(f"wrote {md_path}")
    if args.format in ("tex", "both"):
        tex_path = base.with_suffix(".tex")
        tex_path.write_text(md_to_tex(report, standalone=not args.tex_fragment))
        print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
