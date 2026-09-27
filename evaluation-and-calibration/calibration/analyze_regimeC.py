#!/usr/bin/env python3
"""
analyze_regimeC.py — collate the Regime C genome-scale FP scan outputs into ONE report (md + tex).

This is the analysis step formerly inlined as step 4 of C_ljung_scan_*.sh. Pulling it out means the
report can be regenerated / retuned from the (already written) scan outputs at any time WITHOUT
re-running the multi-hour GPU scan, and it emits LaTeX as well as markdown via its built-in md->tex
renderer. The scan script now just calls this as its last step.

It reads, per model, from the regime dir:
  * apply_<model>.log                                 (calibrator type + logit/AT source)
  * fp_<model>_{raw,calib}_{full,restrict}/genome_fp_benchmark.md   (the four benchmark tables)
and assembles: header + method + caveats + per-model {calibrator log, four benchmark tables}.

Stdlib only (no numpy/sklearn/torch), so it runs locally or on a cluster without the conda env.

Usage:
  python analyze_regimeC.py --regime-dir results/regimeC --truth ../ljungdahlii_holdout_tss.tsv
  python analyze_regimeC.py --regime-dir results/regimeC --truth ... --label "xs retrain" --format tex
"""
import argparse
import datetime
import re
from pathlib import Path

VARIANTS = ["raw_full", "raw_restrict", "calib_full", "calib_restrict"]
CALIB_LOG_KEYS = ("calibrator", "n_features_in_", "logit source", "AT source",
                  "WARNING", "raw  prob", "calib prob", "AT fraction")

CAVEATS = """## Caveats (read before trusting any number)

1. **Genome-scale FP is the metric that matters; balanced AUPRC hides it.** At genome scale promoters
   are ~0.2% of windows, so a strong balanced classifier still emits many FP peaks per scan. These
   tables are the deployment-relevant view; the T22/T25 balanced numbers are not.
2. **Full vs restricted is one scan, two benchmarks.** The scan is always full-genome; `restricted`
   only adds `--regions clj_candidate_regions.bed` (upstream 300 / into-gene 50 bp windows around
   every gene 5'), which shrinks the FP denominator to plausible promoter neighbourhoods. The BED is
   **model-independent** and identical across all models/regimes, which is what keeps the comparison fair.
3. **Score-only calibrators are a NO-OP on FP counts.** If a model's calibrator chose `score` only
   (`n_features_in_==1`, see its `apply_*.log` below), the transform is monotonic in the score, so
   ranking — and therefore every FP-peak count — is IDENTICAL to raw. A raw-vs-calibrated difference
   for such a model would only be numerical noise. Only score+AT calibrators can move the numbers.
4. **AT length/domain shift.** The calibrators fit `at_fraction` (and its StandardScaler mean/scale)
   on the ~52 bp calibration sequences (93 T9 positives + 186 curated negatives); here AT is over the
   81 bp scored window. AT is a fraction (scale-free) so this is the right deployment choice, but the
   absolute calibrated probabilities carry that shift. Because the benchmark is rank-based, only the
   AT-driven RE-RANKING matters, which is valid regardless of the offset — so FP counts are affected
   far less than absolute probabilities.
5. **Logit provenance is exact here.** Tracks were scored with `--emit-logit`, so apply_calibrator
   reads the exact `logit` (= logit1-logit0) the calibrator was fit on — no prob->logit inversion,
   so the float32-saturation caveat (that only bit the inversion path) does not apply to this run.
6. **Calibration is prevalence-dependent.** Fit at the 93:186 (1:2) calibration operating point, NOT
   genome-wide promoter prevalence (~0.2%). Read calibrated values as recalibrated *scores*, not as
   true genome-wide promoter probabilities. The rank-percentile ensemble (T29) is calibration-invariant.
7. **Truth is incomplete.** The {n_truth} TSS are mostly primary / intergenic-sense; deep-CDS and
   antisense promoters are under-catalogued. So a counted "false positive" peak may be a real,
   unannotated promoter — treat FP counts as an upper bound on genuine errors.
8. **genome_fp_benchmark settings:** strand-aware, tol 40 bp (a window within 40 bp of a TSS covers
   it), FP windows merged into peaks within 81 bp. Contig-aware (chromosome-only here).
9. **Minor, model-code:** genome_inference's transformer scorer currently computes softmax after a
   `.float()` cast (the byte-identical prob-ordering refinement was lost in a revert). This can shift
   NT's promoter_score by ~1e-7 vs the published pre-calibration run; negligible for ranking, but note
   it if NT raw numbers differ trivially from the earlier report.
10. **Validation genome.** C. ljungdahlii is the in-genus validation genome. Its published TSS were
    never training positives, but the strain is one of the 166 genomes that supplied silver positives
    and negative windows. Do not mine candidates from it (T33's circularity rule)."""


def count_truth(truth_path: Path) -> int:
    """Number of TSS (data rows) in the truth table — skips a header row if the pos col isn't an int."""
    n = 0
    with open(truth_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            try:
                int(parts[1])
            except ValueError:
                continue                                # header or non-data row
            n += 1
    return n


def read_or_note(path: Path, what: str) -> str:
    if not path.is_file():
        return f"_(missing: {what} — `{path}` not found; benchmark not run or failed.)_"
    return path.read_text().rstrip("\n")


def calibrator_block(regime_dir: Path, model: str) -> str:
    log = regime_dir / f"apply_{model}.log"
    if not log.is_file():
        return f"_(missing: `apply_{model}.log` — calibrate step not run.)_"
    lines = log.read_text().splitlines()
    kept = [ln for ln in lines if any(k in ln for k in CALIB_LOG_KEYS)]
    body = "\n".join(kept if kept else lines)           # mirror the shell `grep ... || cat`
    return "```\n" + body + "\n```"


def build_report(regime_dir: Path, models, truth_path: Path, label: str, recalls: str,
                 genome: str) -> str:
    n_truth = count_truth(truth_path)
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out = []
    w = out.append
    w(f"# Regime C — calibrated genome-scale FP analysis ({label})\n")
    w(f"In-genus validation genome: **{genome}**, truth = {n_truth:,} TSS from "
      f"`{truth_path.name}`.")
    w(f"Generated by `analyze_regimeC.py` on {now}.")
    w("For each model: full-genome scan (`--emit-logit`) -> per-model Platt(+AT) calibrator -> "
      f"`genome_fp_benchmark` at recall {recalls}, in all four {{raw|calibrated}} x {{full|restricted}} cells.\n")
    w("**Headline question:** at matched recall, does AT-aware calibration lower the false-positive "
      "PEAK count vs the raw track? Compare the RAW vs CALIBRATED tables per model below.\n")
    w("---\n")
    w(CAVEATS.format(n_truth=f"{n_truth:,}") + "\n")
    w("---")
    for m in models:
        w(f"\n## {m}\n")
        w(f"### calibrator (from apply_{m}.log)")
        w(calibrator_block(regime_dir, m))
        for v in VARIANTS:
            w(f"\n### {v.replace('_', ' · ', 1)}")
            md = regime_dir / f"fp_{m}_{v}" / "genome_fp_benchmark.md"
            w(read_or_note(md, f"fp_{m}_{v}"))
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ markdown -> LaTeX (self-contained)
# NOTE: this renderer is intentionally duplicated in analyze_calibration.py so each script is
# self-contained (droppable anywhere, stdlib-only). If you fix the escaping/tables here, apply the
# same fix there.
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
        s = s.replace(f"\x00C{i}\x00", r"\texttt{" + _tex_esc(c) + "}")
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
    ap = argparse.ArgumentParser(description="Collate Regime C genome-scale FP outputs -> markdown / LaTeX")
    ap.add_argument("--regime-dir", type=Path, required=True,
                    help="dir holding apply_<model>.log + fp_<model>_<variant>/genome_fp_benchmark.md")
    ap.add_argument("--truth", type=Path, required=True, help="truth TSS table (to count TSS)")
    ap.add_argument("--models", default="prokbert,cnn,cnn_v2,nt", help="comma-separated model order")
    ap.add_argument("--label", default="0.6 retrain", help="round label for the title, e.g. 'xs retrain'")
    ap.add_argument("--recalls", default="0.1,0.5,0.7,0.9", help="recall targets (for the header text only)")
    ap.add_argument("--genome", default="C. ljungdahlii (NC_014328.1)", help="held-out genome label")
    ap.add_argument("--out", type=Path, default=None,
                    help="output path (default: <regime-dir>/regimeC_calibrated_analysis.{md,tex}); "
                         "the suffix is set per --format")
    ap.add_argument("--format", choices=["md", "tex", "both"], default="both",
                    help="which report(s) to write (default: both)")
    ap.add_argument("--tex-fragment", action="store_true",
                    help="emit the .tex WITHOUT the standalone preamble (for \\input into a larger doc)")
    a = ap.parse_args()

    regime_dir = a.regime_dir.resolve()
    if not regime_dir.is_dir():
        raise SystemExit(f"ERROR: regime dir not found: {regime_dir}")
    if not a.truth.is_file():
        raise SystemExit(f"ERROR: truth table not found: {a.truth}")
    models = [m for m in a.models.split(",") if m]
    base = (a.out or regime_dir / "regimeC_calibrated_analysis.md").resolve()
    report = build_report(regime_dir, models, a.truth.resolve(), a.label, a.recalls, a.genome)
    if a.format in ("md", "both"):
        p = base.with_suffix(".md")
        p.write_text(report)
        print(f"wrote {p}")
    if a.format in ("tex", "both"):
        p = base.with_suffix(".tex")
        p.write_text(md_to_tex(report, standalone=not a.tex_fragment))
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
