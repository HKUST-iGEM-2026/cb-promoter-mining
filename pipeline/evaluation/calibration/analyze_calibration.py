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
                   plus the chosen score-only/score+AT and unregularized/L2 calibrator.
  3. AT ablation — score-only vs score+AT, recomputed from the CSV (raw_logit, at_fraction, label)
                   with a pure-python repeated-stratified-KFold logistic. This report-only diagnostic
                   compares OOF log-loss, the fitted AT coefficient sign and effects on reused positives;
                   it is not a new independent calibration validation or the original selection procedure.
  4. Autopsy     — what the calibrator got most wrong: per-source error (raw vs calib), rows the
                   calibrator made worse than raw, right->wrong flips, and the worst-N rows.

Stdlib only (no numpy / sklearn / torch), so it runs anywhere — locally or on a cluster — without the
conda env and without tripping the ~/.local numpy shadowing gotcha (see docs/SETUP.md).

Usage:
  python analyze_calibration.py                          # ./calib_2 -> calibration_analysis.{md,tex}
  python analyze_calibration.py --calib-dir calib_06 --format tex    # LaTeX only
  python analyze_calibration.py --calib-dir calib_06 --out report --top 10   # report.md + report.tex
"""
import argparse
import csv
import json
import math
import random
import re
from pathlib import Path


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
    w("# T26 — calibration benchmark\n")
    w(f"Source: `{calib_dir}`; models: {', '.join(names)}. Checkpoint-specific "
      "Platt maps compare score-only/+AT and unregularized/L2 fits by repeated "
      "stratified CV. Each metrics JSON records the selected recipe and tolerance.\n")
    meta = M[names[0]]
    w(f"**Eval set:** {meta.get('n_total','?')} sequences — {meta.get('n_positive','?')} positives, "
      f"{meta.get('n_negative','?')} negatives "
      f"({meta.get('n_convergent','?')} convergent-intergenic + {meta.get('n_cds','?')} CDS-interior).\n")
    w("**Biological input lengths (from stored prediction CSVs, before scorer padding/cropping):**")
    for nm in names:
        lengths = {}
        for row in P[nm]:
            lengths.setdefault(row["source"], set()).add(len(row["sequence"]))
        detail = "; ".join(f"{source}: {', '.join(map(str, sorted(values)))} bp"
                           for source, values in sorted(lengths.items()))
        w(f"- {nm}: {detail}")
    w("\nA model-window setting or padding does not add missing genomic context or measured TSS framing. "
      "Genomic label=0 controls are putative negatives. These OOF results and calibrator selection reuse "
      "the sampled calibration mixture, not an independent final test or genome-wide activity probabilities.\n")

    # ---- 1. provenance ----
    w("## Checkpoint provenance\n")
    for nm in names:
        pth = M[nm].get("paths", {})
        ckpt = pth.get("checkpoint") or pth.get("model_dir") or pth.get("adapter_dir") or pth.get("config_json")
        feats = ", ".join(M[nm].get("chosen_features", [])) or "not recorded"
        w(f"### {nm}\n")
        w(f"- Recorded selected calibrator: `{M[nm].get('chosen', 'not recorded')}`")
        w(f"- Recorded features: {feats}")
        w(f"- Sampled convergent-intergenic / CDS-interior rows: "
          f"{M[nm].get('n_convergent', 'not recorded')} / {M[nm].get('n_cds', 'not recorded')}\n")
        if ckpt:
            w(f"**Recorded checkpoint / adapter:**\n\n```\n{ckpt}\n```\n")
        else:
            w("**Recorded checkpoint / adapter:** not recorded.\n")
    w("\n> Match each calibrator to its recorded checkpoint and feature recipe. Historical sweep-trial "
      "and final checkpoints are not interchangeable, and this report does not change current deployment.\n")

    # ---- 2. metrics ----
    w("## Raw and out-of-fold metrics\n")
    w("OOF predictions describe this sampled label mixture, not target-genome probabilities. "
      "ECE/Brier/log-loss: lower is better; AUROC/AP: higher. Arrow marks improvement.\n")
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
    w("## AT ablation\n")
    w("Score-only/+AT OOF log-loss is recomputed from the stored CSVs. A positive "
      "standardized AT coefficient raises output at fixed logit; this does not prove activity "
      "or transfer to a genome scan.\n")
    w("| Model | OOF log-loss score-only | OOF log-loss +AT | AT helps? | AT coef | mean P(positives) score-only -> +AT |")
    w("|---|---|---|---|---|---|")
    abl = {}
    for nm in names:
        a = at_ablation(P[nm], seed=seed)
        abl[nm] = a
        helps = "yes" if a["ll_score_at"] < a["ll_score"] else "no"
        w(f"| {nm} | {f(a['ll_score'],4)} | {f(a['ll_score_at'],4)} | **{helps}** | "
          f"{a['at_coef']:+.3f} | {f(a['meanP_pos_score'])} -> {f(a['meanP_pos_score_at'])} |")
    w("\n*The ablation compares feature sets on reused data; it is not an independent final test. "
      "The original selected calibrator and tolerance remain recorded in its metrics JSON.*\n")

    # ---- 4. autopsy ----
    w("## Per-row error\n")
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
    w("## Interpretation\n")
    at_no = [nm for nm in names if abl[nm]["ll_score_at"] >= abl[nm]["ll_score"]]
    if at_no:
        w(f"- AT does **not** lower the diagnostic OOF log-loss for: {', '.join(at_no)} on this mixture. "
          "This report-only comparison does not replace the fitted calibrator's four-map selection procedure.")
    at_yes = [nm for nm in names if abl[nm]["ll_score_at"] < abl[nm]["ll_score"]]
    if at_yes:
        w(f"- Adding AT lowers the recomputed OOF log-loss on this mixture for: {', '.join(at_yes)}. "
          "The fitted coefficient describes score dependence, not proven promoter rescue.")
    worst_models = [nm for nm in names if S[nm]["worse"] > S[nm]["n"] / 2]
    if worst_models:
        w(f"- The calibrator made the majority of rows worse than raw for: {', '.join(worst_models)} "
          "in absolute error while aggregate proper scores can improve. Intercept, score coefficient "
          "and AT coefficient change jointly; these summaries do not isolate a base-rate/intercept mechanism.")
    w("- T29 combines raw per-model ranks. Strictly increasing score-only transforms preserve ranks, "
      "but AT-dependent maps, isotonic ties and numerical saturation can alter them. "
      "Calibration on this label mixture does not establish target-genome promoter probabilities.")
    w("\n---")
    w("<sub>`err = |probability - label|`; `Δ vs raw = calib_err - raw_err` (positive => calibration "
      "made the row worse). §3 ablation is recomputed from raw_logit + at_fraction + label with a "
      "stdlib logistic (5x5 stratified OOF).</sub>")
    return "\n".join(out)


# ------------------------------------------------------------------ markdown -> LaTeX (self-contained)
# NOTE: this renderer is intentionally duplicated in analyze_regimeC.py so each script is self-contained
# (droppable anywhere, stdlib-only). If you fix the escaping/tables here, apply the same fix there.
_TEX_SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}
_TEX_UNICODE = [("⚠️", r"\textbf{[!]}"), ("⚠", r"\textbf{[!]}"), ("▲", r"$\uparrow$"),
                ("▼", r"$\downarrow$"), ("Δ", r"$\Delta$"), ("≈", r"$\approx$"),
                ("×", r"$\times$"), ("§", r"\S{}"), ("·", r"\textperiodcentered{}"),
                ("->", r"$\to$"), ("→", r"$\to$"),
                ("—", "---"), ("–", "--"), ("’", "'"), ("‘", "`"), ("“", "``"), ("”", "''")]

_TEX_PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{array}
\usepackage{fvextra}
\RecustomVerbatimEnvironment{verbatim}{Verbatim}{breaklines=true,breakanywhere=true}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5em}
\begin{document}
"""


def _tex_esc(t: str) -> str:
    """Escape LaTeX specials (per char, so unicode passes through) then map the few glyphs used."""
    t = "".join(_TEX_SPECIAL.get(c, c) for c in t)
    for u, r in _TEX_UNICODE:
        t = t.replace(u, r)
    return t


def _tex_inline(s: str) -> str:
    """Convert inline markdown (**bold**, `code`, \\| escaped pipe) to LaTeX, escaping the rest."""
    s = s.replace(r"\|", "\x00P\x00")                       # protect escaped pipe BEFORE escaping '\'
    codes, bolds = [], []
    s = re.sub(r"`([^`]*)`", lambda m: codes.append(m.group(1)) or f"\x00C{len(codes) - 1}\x00", s)
    s = re.sub(r"\*\*(.+?)\*\*", lambda m: bolds.append(m.group(1)) or f"\x00B{len(bolds) - 1}\x00", s)
    s = _tex_esc(s)
    for i, c in enumerate(codes):
        # Preserve escaped code while allowing long paths/API names to wrap at separators.
        code = re.sub(r"(\\_|[/.-])", r"\1\\allowbreak{}", _tex_esc(c))
        s = s.replace(f"\x00C{i}\x00", r"\texttt{" + code + "}")
    for i, b in enumerate(bolds):
        s = s.replace(f"\x00B{i}\x00", r"\textbf{" + _tex_esc(b) + "}")
    return s.replace("\x00P\x00", r"\textbar{}")


def _tex_table(block) -> str:
    rows = []
    for ln in block:
        cells = re.split(r"(?<!\\)\|", ln.strip())          # split on unescaped pipes
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        cells = [c.strip() for c in cells]
        if cells and all(re.fullmatch(r":?-+:?", c or "-") for c in cells):
            continue                                        # markdown separator row
        rows.append(cells)
    if not rows:
        return ""
    ncol = max(len(r) for r in rows)
    out = [r"\begin{center}",
           r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
           r"\begin{tabular}{" + "l" + "c" * (ncol - 1) + "}", r"\hline"]
    for ri, r in enumerate(rows):
        r = r + [""] * (ncol - len(r))
        out.append(" & ".join(_tex_inline(c) for c in r) + r" \\")
        if ri == 0:
            out.append(r"\hline")
    out += [r"\hline", r"\end{tabular}%", "}", r"\end{center}"]
    return "\n".join(out)


def md_to_tex(md: str, standalone: bool = True) -> str:
    """Render the markdown report to LaTeX. standalone=True wraps it in a compilable document."""
    lines = md.split("\n")
    body, i, in_item = [], 0, False

    def close_item():
        nonlocal in_item
        if in_item:
            body.append(r"\end{itemize}")
            in_item = False

    while i < len(lines):
        st = lines[i].strip()
        if st.startswith("```"):                            # fenced code -> verbatim
            close_item()
            i += 1
            body.append(r"\begin{verbatim}")
            while i < len(lines) and not lines[i].strip().startswith("```"):
                body.append(lines[i])                       # verbatim: no escaping
                i += 1
            body.append(r"\end{verbatim}")
            i += 1                                           # skip closing fence
            continue
        if st.startswith("|") and st.endswith("|"):         # table block
            close_item()
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            body.append(_tex_table(block))
            continue
        if st == "" or st == "---":
            close_item()
            if st == "":
                body.append("")
        elif st.startswith("### "):
            close_item(); body.append(r"\subsubsection*{" + _tex_inline(st[4:]) + "}")
        elif st.startswith("## "):
            close_item(); body.append(r"\subsection*{" + _tex_inline(st[3:]) + "}")
        elif st.startswith("# "):
            close_item(); body.append(r"\section*{" + _tex_inline(st[2:]) + "}")
        elif st.startswith("> "):
            close_item(); body.append(r"{\itshape " + _tex_inline(st[2:]) + r"\par}")
        elif st.startswith("- "):
            if not in_item:
                body.append(r"\begin{itemize}"); in_item = True
            body.append(r"\item " + _tex_inline(st[2:]))
        elif st.startswith("<sub"):
            close_item(); body.append(r"{\footnotesize " + _tex_inline(re.sub(r"</?sub>", "", st)) + "}")
        elif re.fullmatch(r"_[^_].*[^_]_", st) or re.fullmatch(r"_[^_]_", st):
            close_item(); body.append(r"{\itshape " + _tex_inline(st[1:-1]) + "}")   # full-line _italic_
        else:
            close_item(); body.append(_tex_inline(st))
        i += 1
    close_item()
    doc = "\n".join(body) + "\n"
    return (_TEX_PREAMBLE + doc + r"\end{document}" + "\n") if standalone else doc


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
