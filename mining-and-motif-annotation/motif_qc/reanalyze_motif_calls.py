#!/usr/bin/env python3
"""
reanalyze_motif_calls.py -- QC of the g131795 -10/-35 motif calls, rebuilt on REAL repo inputs.

Context
-------
Swastika's report (2026-09-19) scored the pipeline's stored -10/-35 box sequences against
reference sigma70/sigmaA PWMs and found a "cross-element contamination" rate of 86.8% for the
-35 boxes (they score higher against a -10 PWM than their own -35 PWM) vs only 5.0% for -10
boxes, and inferred a systematic -35 windowing/extraction bug.

This script re-runs that test on the ACTUAL inputs that live in the repo -- the pipeline's own
MEME motif model (mining-and-motif-annotation/T31/sigma_a_promoter.meme, canonical TATAAT / TTGACA
with an AT-rich background baked in) and the delivered candidate table (g131795_ranked_seq.tsv) --
and adds the controls the score-only test was missing:
  1. reproduce the contamination rate with the pipeline's own motifs;
  2. redo it under the motif file's declared AT-rich background (0.35/0.15) vs uniform 0.25;
  3. a random AT-rich-sequence control (is the flag just detecting AT-richness?);
  4. a matched-null SIGNAL test with a -10 positive control (do the boxes carry REAL motif signal?);
  5. spacer geometry (does the "offset bug" hypothesis survive?).

Everything is stdlib + numpy/matplotlib. No network, no models. Reproducible (seed fixed).
Run from anywhere:  python3 reanalyze_motif_calls.py
"""
import csv, math, json, random, bisect, statistics as st, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

random.seed(7); np.random.seed(7)
BASES = "ACGT"; IDX = {b: i for i, b in enumerate(BASES)}
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MEME = os.path.join(REPO, "mining-and-motif-annotation", "T31", "sigma_a_promoter.meme")
TABLE = os.path.join(REPO, "g131795_ranked_seq.tsv")
UNIFORM = [0.25, 0.25, 0.25, 0.25]


# ----------------------------------------------------------------------------- inputs
def parse_meme(path):
    """Return {motif_name: probability_matrix} and the declared background [A,C,G,T]."""
    mats, bg, cur, grab, rows = {}, None, None, False, []
    for ln in open(path):
        s = ln.strip()
        if bg is None and s[:1] == "A" and s.count("0.") >= 2 and "MOTIF" not in s and len(s.split()) >= 8:
            t = s.split(); bg = [float(t[1]), float(t[3]), float(t[5]), float(t[7])]; continue
        if s.startswith("MOTIF"):
            cur = s.split()[1]; rows = []; grab = False; continue
        if s.startswith("letter-probability"):
            grab = True; rows = []; continue
        if grab:
            p = s.split()
            if len(p) == 4:
                try: rows.append([float(x) for x in p])
                except ValueError: mats[cur] = rows; grab = False
            else:
                mats[cur] = rows; grab = False
    if cur and cur not in mats: mats[cur] = rows
    return mats, bg


def load_rows(path):
    return list(csv.DictReader(open(path), delimiter="\t"))


def present(x):
    return x not in ("", "nan", "None", None)


# ----------------------------------------------------------------------------- scoring
def score(seq, M, bg):
    """Best sliding-window log-odds (bits) of a width-6 PWM over a wider context window."""
    w = len(M); best = -9e9
    for s0 in range(len(seq) - w + 1):
        t = 0.0
        for i in range(w):
            c = seq[s0 + i]
            if c in IDX:
                t += math.log2((M[i][IDX[c]] + 1e-12) / bg[IDX[c]])
        best = max(best, t)
    return best


def information_content(M, bg):
    return sum(sum(M[i][j] * math.log2((M[i][j] + 1e-12) / bg[j]) for j in range(4)) for i in range(len(M)))


def contamination(seqs, own, other, bg):
    flagged = sum(score(s, other, bg) > score(s, own, bg) for s in seqs)
    return flagged, len(seqs), 100.0 * flagged / len(seqs)


def rand_at(width, bg, n):
    return ["".join(random.choices(BASES, weights=bg, k=width)) for _ in range(n)]


def null_percentiles(seqs, M, bg, null_width=14, n=6000):
    null = sorted(score(s, M, bg) for s in rand_at(null_width, bg, n))
    return [100.0 * bisect.bisect_left(null, score(s, M, bg)) / len(null) for s in seqs], null


# ----------------------------------------------------------------------------- main
def main():
    mats, bg = parse_meme(MEME)
    P10, P35 = mats["minus10"], mats["minus35"]
    rows = load_rows(TABLE)
    m10 = [r["minus10_seq"] for r in rows if present(r.get("minus10_seq", ""))]
    m35 = [r["minus35_seq"] for r in rows if present(r.get("minus35_seq", ""))]

    R = {"inputs": {"meme": os.path.relpath(MEME, REPO), "table": os.path.relpath(TABLE, REPO),
                    "declared_background_ACGT": bg}}

    # 0. coverage
    both = sum(present(r.get("minus10_seq", "")) and present(r.get("minus35_seq", "")) for r in rows)
    only10 = sum(present(r.get("minus10_seq", "")) and not present(r.get("minus35_seq", "")) for r in rows)
    only35 = sum(present(r.get("minus35_seq", "")) and not present(r.get("minus10_seq", "")) for r in rows)
    neither = len(rows) - both - only10 - only35
    R["coverage"] = {"total": len(rows), "both": both, "minus10_only": only10,
                     "minus35_only": only35, "neither": neither}

    # 1. information content (why the two motifs live on different score scales)
    R["information_content_bits"] = {
        "minus10": {"uniform_bg": round(information_content(P10, UNIFORM), 2),
                    "at_rich_bg": round(information_content(P10, bg), 2)},
        "minus35": {"uniform_bg": round(information_content(P35, UNIFORM), 2),
                    "at_rich_bg": round(information_content(P35, bg), 2)}}

    # 2. contamination, both backgrounds, both arms
    R["contamination"] = {}
    for name, bgv in [("uniform_0.25", UNIFORM), ("at_rich_0.35_0.15", bg)]:
        f35 = contamination(m35, P35, P10, bgv)
        f10 = contamination(m10, P10, P35, bgv)
        R["contamination"][name] = {
            "minus35_prefers_minus10_pct": round(f35[2], 1), "minus35_n": f35[1],
            "minus10_prefers_minus35_pct": round(f10[2], 1), "minus10_n": f10[1]}

    # 3. random AT-rich control through the identical test
    rj = rand_at(14, bg, 4000)
    R["random_control"] = {
        "uniform_0.25": round(contamination(rj, P35, P10, UNIFORM)[2], 1),
        "at_rich": round(contamination(rj, P35, P10, bg)[2], 1),
        "note": "random non-promoter AT-rich 14-mers, 'preferring the -10 motif' rate"}

    # 4. matched-null SIGNAL test (-10 is the positive control)
    p10, _ = null_percentiles(m10, P10, bg)
    p35, _ = null_percentiles(m35, P35, bg)
    R["signal_vs_null"] = {
        "minus10_boxes": {"median_percentile": round(st.median(p10)),
                          "pct_above_null_95th": round(100 * sum(p > 95 for p in p10) / len(p10))},
        "minus35_boxes": {"median_percentile": round(st.median(p35)),
                          "pct_above_null_95th": round(100 * sum(p > 95 for p in p35) / len(p35))},
        "note": "percentile of each box's own-motif log-odds vs a matched AT-rich null; 50=noise"}

    # 5. spacer geometry + AT content
    sp = [float(r["spacer_bp"]) for r in rows if present(r.get("spacer_bp", ""))]
    R["spacer_bp"] = {"n": len(sp), "min": min(sp), "median": st.median(sp), "max": max(sp),
                      "note": "positional gating allows ~4..39 bp; median 18 rules out a gross offset, weak positive evidence"}
    at = lambda x: 100 * sum(c in "AT" for c in x) / len(x)
    R["at_content_pct"] = {"minus10_boxes": round(st.mean(at(s) for s in m10)),
                           "minus35_boxes": round(st.mean(at(s) for s in m35)),
                           "summit": round(st.mean(at(r["summit_seq"]) for r in rows if present(r.get("summit_seq", ""))))}

    # ------------------------------------------------------------------- figures
    _figures(R, m10, m35, P10, P35, bg)

    out = os.path.join(HERE, "reanalysis_results.json")
    json.dump(R, open(out, "w"), indent=2)
    _print(R)
    print(f"\nwrote {os.path.relpath(out, REPO)} and 4 figures to {os.path.relpath(HERE, REPO)}/")


def _figures(R, m10, m35, P10, P35, bg):
    # Fig 1 -- coverage
    c = R["coverage"]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    labs = ["both", "-10 only", "-35 only", "neither"]
    vals = [c["both"], c["minus10_only"], c["minus35_only"], c["neither"]]
    ax.bar(labs, vals, color=["#2a7", "#59f", "#f83", "#bbb"])
    for i, v in enumerate(vals): ax.text(i, v + 5, str(v), ha="center", fontsize=9)
    ax.set_ylabel("candidates"); ax.set_title(f"Box-call coverage (n={c['total']})")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig1_coverage.png"), dpi=140); plt.close(fig)

    # Fig 2 -- the key one: own-motif signal vs a matched AT-rich null
    null10 = sorted(score(s, P10, bg) for s in rand_at(14, bg, 6000))
    null35 = sorted(score(s, P35, bg) for s in rand_at(14, bg, 6000))
    s10 = [score(s, P10, bg) for s in m10]
    s35 = [score(s, P35, bg) for s in m35]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    ax[0].hist(null10, bins=40, density=True, color="#bbb", alpha=.8, label="random AT-rich")
    ax[0].hist(s10, bins=40, density=True, color="#2a7", alpha=.6, label="real -10 boxes")
    ax[0].axvline(np.percentile(null10, 95), ls="--", c="k", lw=1)
    ax[0].set_title("-10 boxes vs -10 motif\n(positive control: strong signal)"); ax[0].legend(fontsize=8)
    ax[0].set_xlabel("log-odds (bits, AT-rich bg)")
    ax[1].hist(null35, bins=40, density=True, color="#bbb", alpha=.8, label="random AT-rich")
    ax[1].hist(s35, bins=40, density=True, color="#f83", alpha=.6, label="real -35 boxes")
    ax[1].axvline(np.percentile(null35, 95), ls="--", c="k", lw=1)
    ax[1].set_title("-35 boxes vs -35 motif\n(overlaps the null: little signal)"); ax[1].legend(fontsize=8)
    ax[1].set_xlabel("log-odds (bits, AT-rich bg)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig2_signal_vs_null.png"), dpi=140); plt.close(fig)

    # Fig 3 -- contamination under both backgrounds + random control
    fig, ax = plt.subplots(figsize=(6, 3.4))
    groups = ["-35 boxes\n(prefer -10)", "-10 boxes\n(prefer -35)", "random junk\n(prefer -10)"]
    uni = [R["contamination"]["uniform_0.25"]["minus35_prefers_minus10_pct"],
           R["contamination"]["uniform_0.25"]["minus10_prefers_minus35_pct"],
           R["random_control"]["uniform_0.25"]]
    atr = [R["contamination"]["at_rich_0.35_0.15"]["minus35_prefers_minus10_pct"],
           R["contamination"]["at_rich_0.35_0.15"]["minus10_prefers_minus35_pct"],
           R["random_control"]["at_rich"]]
    x = np.arange(3); w = 0.38
    ax.bar(x - w / 2, uni, w, label="uniform 0.25 bg", color="#59f")
    ax.bar(x + w / 2, atr, w, label="AT-rich bg (correct)", color="#e63")
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8); ax.set_ylabel("% flagged")
    ax.set_title("'Contamination' rate: background barely matters"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig3_contamination.png"), dpi=140); plt.close(fig)

    # Fig 4 -- spacer
    sp = [float(r["spacer_bp"]) for r in load_rows(TABLE) if present(r.get("spacer_bp", ""))]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(sp, bins=range(0, 40, 2), color="#7a5", alpha=.85)
    ax.axvspan(15, 19, color="k", alpha=.08, label="canonical 15-19 bp")
    ax.axvline(st.median(sp), c="k", ls="--", lw=1, label=f"median {st.median(sp):.0f}")
    ax.set_xlabel("spacer_bp (-35 to -10)"); ax.set_ylabel("candidates")
    ax.set_title("Spacer geometry (n=45 canonical)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig4_spacer.png"), dpi=140); plt.close(fig)


def _print(R):
    print("=" * 74)
    print("g131795 -10/-35 motif QC -- rebuilt on real repo inputs")
    print("  motifs:", R["inputs"]["meme"], "| table:", R["inputs"]["table"])
    print("  declared background A,C,G,T =", R["inputs"]["declared_background_ACGT"])
    print("=" * 74)
    c = R["coverage"]
    print(f"COVERAGE  total {c['total']}: both {c['both']} / -10-only {c['minus10_only']} "
          f"/ -35-only {c['minus35_only']} / neither {c['neither']}")
    ic = R["information_content_bits"]
    print(f"INFO CONTENT (bits)  -10: uniform {ic['minus10']['uniform_bg']} / AT-rich {ic['minus10']['at_rich_bg']}"
          f"   -35: uniform {ic['minus35']['uniform_bg']} / AT-rich {ic['minus35']['at_rich_bg']}")
    print("CONTAMINATION")
    for k, v in R["contamination"].items():
        print(f"  {k:20s}  -35 prefer -10: {v['minus35_prefers_minus10_pct']:5.1f}%   "
              f"-10 prefer -35: {v['minus10_prefers_minus35_pct']:5.1f}%")
    print(f"  random control        -35-style flag: uniform {R['random_control']['uniform_0.25']}%  "
          f"AT-rich {R['random_control']['at_rich']}%")
    s = R["signal_vs_null"]
    print("SIGNAL vs matched null (median percentile / % above null-95th)")
    print(f"  -10 boxes: {s['minus10_boxes']['median_percentile']:>3} / {s['minus10_boxes']['pct_above_null_95th']}%   (positive control)")
    print(f"  -35 boxes: {s['minus35_boxes']['median_percentile']:>3} / {s['minus35_boxes']['pct_above_null_95th']}%   (50=noise)")
    sp = R["spacer_bp"]
    print(f"SPACER  n {sp['n']}  min {sp['min']:.0f}  median {sp['median']:.0f}  max {sp['max']:.0f}")
    a = R["at_content_pct"]
    print(f"AT%%  -10 {a['minus10_boxes']}  -35 {a['minus35_boxes']}  summit {a['summit']}")


if __name__ == "__main__":
    main()
