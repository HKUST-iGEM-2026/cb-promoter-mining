#!/usr/bin/env python3
"""
md2tex.py — shared markdown -> LaTeX renderer for the T26 analysis scripts
(analyze_calibration.py, analyze_regimeC.py). Stdlib only, so it runs anywhere without the conda env.

Handles exactly the markdown those reports emit — nothing more:
  #/##/### headers, pipe tables (with \\| escaped pipes), ```fenced code``` -> verbatim,
  **bold**, `code`, a full-line _italic_, `- ` bullets, `> ` notes, <sub>, and the few unicode
  glyphs used (arrows, Delta, warning, section sign, typographic dashes/quotes).

Public API:
  md_to_tex(md: str, standalone: bool = True) -> str
"""
import re

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
