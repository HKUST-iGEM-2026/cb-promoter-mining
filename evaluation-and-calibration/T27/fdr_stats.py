#!/usr/bin/env python3
"""
fdr_stats.py — T27 empirical-FDR STATISTICS (pure Python; NO scanning, NO subprocess, NO env-switching).

This is the statistics half of the T27 promoter mine on the target C. butyricum. It was split out of the
old monolithic fdr_mine.py so that the expensive, multi-env GPU work (scanning) and the cheap,
single-env, reproducible-on-CPU work (the FDR maths) live in separate files, wired by run_fdr_mine.sbatch
(bash). The bash side reuses the repo's standalone scanners (genome_inference.py / evo2_scan_restricted.py),
align_tracks.py (T29) for grid reconciliation, and annotate_motifs.py (T31) for the motif pass. This file
does ONLY the parts unique to T27:

  * pooled rank-percentile ensemble (T29-style) over the RESTRICTED candidates + the convergent-intergenic
    null, pooled per model so the two sets share ONE common scale (calibration-free)
  * empirical p = (#null >= score + 1)/(Nnull + 1)
  * p-gated peak-calling (T30-style) then Benjamini-Hochberg q-values over the PEAKS (the calling unit)
  * top-N promoters + their summit sequence in reading orientation

Two subcommands (bash calls `prep` before the scans, `fdr` after align_tracks):
  prep : validate the target genome<->GFF contig ids (the 'contig_1' vs 'NC_...' trap), and build the
         packed null FASTA + its whole-contig regions BED from the convergent-intergenic negative TSV(s).
  fdr  : read the ALIGNED per-model candidate + null tracks, ensemble + p/q, write the FDR tables.

WHY the packed null (see fdr_mine history): each null sequence is scored INDEPENDENTLY — every model reads
one isolated 81 bp window and scores it from only its own bases (genome_inference/evo2 both slice
seq[s:s+81] and score each in isolation). Packing all null 81-mers end-to-end into one contig scanned
step=81 is purely a batching container: window i == block i, byte-for-byte identical to scoring block i
standalone, with zero cross-window bleed. It exists so the whole null pool batches on the GPU in one pass.

WHY p-aware peak-calling is INLINE here and not T30 call_peaks.py: T30 thresholds on a raw SCORE and does
not carry a p-value, but BH needs each peak's p (= min p over its windows). So this is a p-gated variant;
T30 stays the score-threshold caller for the plain (non-FDR) mine.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW = 81
KEYS = ["contig", "start", "strand"]                       # a scored window's identity across tracks
COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s):
    return s.translate(COMPLEMENT)[::-1]


# ------------------------------------------------------------------- IO helpers
def read_fasta(path):
    """{record_id: sequence(upper)}; record_id = first header token (matches genome_inference/T31)."""
    seqs, cid, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if cid is not None:
                    seqs[cid] = "".join(buf).upper()
                cid = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if cid is not None:
        seqs[cid] = "".join(buf).upper()
    return seqs


def gff_seqids(path):
    ids = set()
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split("\t")
            if len(p) >= 8 and p[2] in ("gene", "CDS"):
                ids.add(p[0])
    return ids


def load_track(path, sort_keys):
    df = pd.read_csv(path, usecols=lambda c: c in ("contig", "start", "end", "strand", "promoter_score"))
    for c in ("contig", "start", "strand", "promoter_score"):
        if c not in df.columns:
            raise SystemExit(f"{path}: missing column '{c}'; has {list(df.columns)}")
    df["contig"] = df["contig"].astype(str)
    df["strand"] = df["strand"].astype(str)
    return df.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)


# ------------------------------------------------------------------- statistics
def pooled_ensemble(cand_tracks, null_tracks, models):
    """Per model, rank-percentile the POOLED (candidate + null) raw scores so the two sets are on ONE
    common scale, then average across models. Returns (cand_ens, null_ens) aligned to model[0]'s row
    order. Fails loud if the models are not on an identical grid (a silent misalignment would scramble
    the ensemble — align_tracks.py upstream guarantees they match)."""
    ref = models[0]
    ncand, nnull = len(cand_tracks[ref]), len(null_tracks[ref])
    cand_ref, null_ref = cand_tracks[ref][KEYS], null_tracks[ref][KEYS]
    cand_pcts, null_pcts = [], []
    for m in models:
        ct, nt = cand_tracks[m], null_tracks[m]
        if len(ct) != ncand or not ct[KEYS].equals(cand_ref):
            raise SystemExit(f"candidate track '{m}' is not on the same window grid as '{ref}' "
                             f"({len(ct):,} vs {ncand:,} rows or key mismatch). Re-align the tracks.")
        if len(nt) != nnull or not nt[KEYS].equals(null_ref):
            raise SystemExit(f"null track '{m}' is not aligned with '{ref}' "
                             f"({len(nt):,} vs {nnull:,} rows or key mismatch).")
        pooled = np.concatenate([ct["promoter_score"].to_numpy(), nt["promoter_score"].to_numpy()])
        pct = pd.Series(pooled).rank(pct=True, method="average").to_numpy()
        cand_pcts.append(pct[:ncand])
        null_pcts.append(pct[ncand:])
    cand_ens = np.mean(np.column_stack(cand_pcts), axis=1)
    null_ens = np.mean(np.column_stack(null_pcts), axis=1)
    return cand_ens, null_ens


def empirical_p(cand_ens, null_ens):
    """p = (#{null >= score} + 1) / (Nnull + 1). The +1 avoids p=0 and bounds the smallest achievable
    p at 1/(Nnull+1) (why a big null pool matters)."""
    ns = np.sort(null_ens)
    ge = len(ns) - np.searchsorted(ns, cand_ens, side="left")
    return (ge + 1.0) / (len(ns) + 1.0)


def bh_qvalues(p):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def call_peaks(df, merge):
    """Collapse p-gated windows into loci: adjacent same-(contig,strand) windows within `merge` bp ->
    one peak; summit = highest-ensemble window; p_value = min over the peak (for BH). Mirrors T30
    call_peaks.py but is p-AWARE (T30 thresholds on score and does not carry a p-value)."""
    df = df.sort_values(["contig", "strand", "start"], kind="mergesort").reset_index(drop=True)
    contig = df["contig"].to_numpy()
    strand = df["strand"].to_numpy()
    start = df["start"].to_numpy(int)
    end = df["end"].to_numpy(int)
    ens = df["ensemble"].to_numpy(float)
    pval = df["p_value"].to_numpy(float)
    breaks = (contig[1:] != contig[:-1]) | (strand[1:] != strand[:-1]) | ((start[1:] - end[:-1]) > merge)
    pid = np.concatenate([[0], np.cumsum(breaks)])
    g = pd.DataFrame({"pid": pid, "contig": contig, "strand": strand, "start": start,
                      "end": end, "ensemble": ens, "p_value": pval})
    agg = g.groupby("pid", sort=False).agg(
        contig=("contig", "first"), strand=("strand", "first"),
        peak_start=("start", "min"), peak_end=("end", "max"),
        ensemble_score=("ensemble", "max"), p_value=("p_value", "min"),
        n_windows=("ensemble", "size"))
    summit_idx = g.groupby("pid", sort=False)["ensemble"].idxmax()
    agg["summit"] = g.loc[summit_idx.to_numpy(), "start"].to_numpy()
    return agg.reset_index(drop=True)


# ------------------------------------------------------------------- subcommand: prep
def cmd_prep(a):
    # ---- contig-id preflight (the 'contig_1' vs 'NC_...' trap) ----
    genome = read_fasta(a.genome)
    fasta_ids = set(genome)
    gff_ids = gff_seqids(a.gff)
    missing = gff_ids - fasta_ids
    if missing:
        raise SystemExit(
            "CONTIG-ID MISMATCH: GFF3 seqids are not all present in the genome FASTA — the candidate "
            "scan would join on nothing and silently find 0 promoters. Fix the annotation/FASTA pair so "
            "their contig ids agree.\n"
            f"  FASTA ids (n={len(fasta_ids)}): {sorted(fasta_ids)[:5]}{' ...' if len(fasta_ids) > 5 else ''}\n"
            f"  GFF3  ids (n={len(gff_ids)}): {sorted(gff_ids)[:5]}{' ...' if len(gff_ids) > 5 else ''}\n"
            f"  in GFF3 but NOT FASTA: {sorted(missing)[:5]}")
    print(f"[preflight] contig ids OK: {len(gff_ids)} GFF3 seqids all present among {len(fasta_ids)} FASTA records.")

    # ---- build the packed null FASTA from the convergent-intergenic TSV(s) ----
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fwd, per_file = [], []
    for tsv in a.null_tsv:                                  # POOL one or more convergent-intergenic TSVs
        ndf = pd.read_csv(tsv, sep="\t")
        if "sequence" not in ndf.columns:
            raise SystemExit(f"{tsv}: no 'sequence' column; has {list(ndf.columns)}")
        s = ndf["sequence"].astype(str).str.upper().tolist()
        per_file.append((tsv, len(s)))
        fwd.extend(s)
    if not fwd:
        raise SystemExit("null TSV(s) contributed 0 rows — an empty null pool cannot define an FDR.")
    n_pooled = len(fwd)
    fwd = list(dict.fromkeys(fwd))                         # drop any identical 81-mer shared across files
    n_dup = n_pooled - len(fwd)
    seqs = pd.Series(fwd, dtype=str)
    bad_len = int((seqs.str.len() != WINDOW).sum())
    bad_chr = int((~seqs.str.fullmatch("[ACGT]+")).sum())
    if bad_len or bad_chr:
        raise SystemExit(f"null sequences failed validation: {bad_len} not {WINDOW} bp, {bad_chr} non-ACGT. "
                         f"The null must be clean {WINDOW} bp to be scored by the {WINDOW} bp models "
                         f"(NB extract_negatives.py defaults to --window 52 — regenerate the null at 81).")
    # Pack every 81 bp block AND its reverse complement (so the null spans both orientations exactly as the
    # candidate scan does) end-to-end into ONE contig. Scored later step=81 strands=+ so window i == block i
    # and the whole pool batches. NEVER scan this contig on '-' (that RCs the WHOLE concatenation and
    # scrambles the blocks; the reverse orientation is already present as explicit rc blocks).
    blocks = fwd + [revcomp(s) for s in fwd]
    null_fasta = out / "null.fasta"
    with open(null_fasta, "w") as fh:
        fh.write(">nullcat\n" + "".join(blocks) + "\n")
    # whole-contig regions BED — ONLY used by evo2_scan_restricted.py (its scanner requires --regions;
    # genome_inference scans nullcat without --regions). Tab-separated with a header, like every repo BED.
    null_len = len(blocks) * WINDOW
    null_regions = out / "null_regions.bed"
    with open(null_regions, "w") as fh:
        fh.write("contig\tstart\tend\tstrand\n")
        fh.write(f"nullcat\t0\t{null_len}\t+\n")

    for tsv, n in per_file:
        print(f"[null]   {n:>8,} rows <- {tsv}")
    print(f"[null] {n_pooled:,} pooled -> {len(fwd):,} unique ({n_dup:,} cross-file duplicates dropped) "
          f"x2 orientations = {len(blocks):,} null blocks ({WINDOW} bp each) packed -> {null_fasta}")
    print(f"[null] whole-contig BED (evo2 only) -> {null_regions}  (nullcat 0..{null_len} +)")


# ------------------------------------------------------------------- subcommand: fdr
def cmd_fdr(a):
    genome = read_fasta(a.genome)
    models = a.models
    outdir = Path(a.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cand_dir, null_dir = Path(a.cand_dir), Path(a.null_dir)

    # --- load ALIGNED per-model tracks (align_tracks.py has already put them on one grid) ---
    cand_tracks = {m: load_track(str(cand_dir / f"{m}_aligned.csv"), KEYS) for m in models}
    null_tracks = {m: load_track(str(null_dir / f"{m}_aligned.csv"), ["contig", "strand", "start"]) for m in models}

    # Candidates are ALREADY restricted to the upstream-of-gene regions (the scans ran with --regions, and
    # align_tracks intersected to Evo2's restricted candidate set), so no re-masking is needed here — the
    # aligned candidate rows ARE the restricted candidate space. Rank over that space + the null.
    rc = cand_tracks[models[0]][["contig", "start", "end", "strand"]].copy()
    cand_ens, null_ens = pooled_ensemble(cand_tracks, null_tracks, models)
    rc["ensemble"] = cand_ens
    rc["p_value"] = empirical_p(cand_ens, null_ens)
    print(f"  candidates: {len(rc):,} windows  ·  null samples: {len(null_ens):,}  ·  models: {models}")

    # --- peak-call the p-gated windows, then BH q-values over PEAKS (the calling unit) ---
    gated = rc[rc["p_value"] <= a.peak_alpha]
    if gated.empty:
        raise SystemExit(f"no candidate window passed empirical p <= {a.peak_alpha}. Nothing to call.")
    peaks = call_peaks(gated, a.merge)
    peaks["q_value"] = bh_qvalues(peaks["p_value"].to_numpy())
    peaks = peaks.sort_values("ensemble_score", ascending=False, kind="mergesort").reset_index(drop=True)

    # --- top-N promoters + their summit sequence (reading orientation) ---
    top = peaks.head(a.top_n).copy()
    seqs = []
    for _, r in top.iterrows():
        s = genome[str(r["contig"])][int(r["summit"]):int(r["summit"]) + WINDOW]
        s = revcomp(s) if r["strand"] == "-" else s
        seqs.append(s if len(s) == WINDOW else "")         # blank = truncated at a contig end
    top["sequence"] = seqs
    dropped = int((top["sequence"].str.len() != WINDOW).sum())
    top.insert(0, "candidate_id", [f"cbut_{i + 1:04d}" for i in range(len(top))])

    peaks.to_csv(outdir / "peaks_fdr.csv", index=False)
    top_csv = outdir / f"top{a.top_n}_promoters.csv"
    top.to_csv(top_csv, index=False)
    write_report(a, models, rc, null_ens, peaks, top, dropped, outdir)
    print(f"  -> {outdir}/  (top{a.top_n}_promoters.csv, peaks_fdr.csv, analysis.md)")


def _tex_escape(s):
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_report(a, models, rc, null_ens, peaks, top, dropped, outdir):
    """Emit the same FDR analysis as BOTH analysis.md and a standalone-compilable analysis.tex — each with
    the null distribution, the discovery counts, and a p/q table of the top hits (peaks_fdr.csv is the
    full per-locus p/q record)."""
    q = np.quantile(null_ens, [0.5, 0.95, 0.99, 1.0])
    smallest_p = 1.0 / (len(null_ens) + 1)
    n_q05 = int((peaks["q_value"] <= 0.05).sum())
    n_q10 = int((peaks["q_value"] <= 0.10).sum())
    ens_lo, ens_hi = top["ensemble_score"].min(), top["ensemble_score"].max()
    q_lo, q_hi = top["q_value"].min(), top["q_value"].max()
    n_show = min(len(top), 20)                                    # preview rows in the doc; full set in the CSV
    tv = top.head(n_show)
    cols = ["candidate_id", "contig", "summit", "strand", "ensemble_score", "p_value", "q_value", "n_windows"]

    # ----- shared prose -----
    intro = (f"Candidates are the {len(rc):,} upstream-of-gene windows; the null is the {len(null_ens):,} "
             f"convergent-intergenic samples (both strands of the 81 bp negatives). Both are intergenic + "
             f"AT-rich, so the empirical p is a composition-matched comparison. Ensemble = mean "
             f"rank-percentile pooled over the RESTRICTED candidates + null (calibration-free).")
    ensemble_str = " + ".join(models)

    # ----- Markdown -----
    md_rows = "\n".join(
        f"| {r.candidate_id} | {r.contig} | {int(r.summit)} | {r.strand} | {r.ensemble_score:.4f} | "
        f"{r.p_value:.3e} | {r.q_value:.3e} | {int(r.n_windows)} |" for r in tv.itertuples())
    md = f"""# T27 empirical-FDR promoter mine — C. butyricum — ensemble `{ensemble_str}`

**Restricted scan + tailored null.** {intro}

## Null score distribution (ensemble rank-percentile)
median {q[0]:.4f} · p95 {q[1]:.4f} · p99 {q[2]:.4f} · max {q[3]:.4f}
Smallest achievable p-value = 1/(Nnull+1) = {smallest_p:.2e}.

## Discoveries
- candidate windows tested: {len(rc):,}
- windows passing empirical p <= {a.peak_alpha}: peak-called into **{len(peaks):,} promoter loci**
- loci at q <= 0.05: **{n_q05:,}**   ·   q <= 0.10: {n_q10:,}
- reported: top {len(top)} by ensemble score ({dropped} summit windows truncated at a contig end -> blank sequence)
- top-{len(top)} ensemble range [{ens_lo:.4f}, {ens_hi:.4f}], q range [{q_lo:.2e}, {q_hi:.2e}]

## Top {n_show} promoters (full set in `top{a.top_n}_promoters.csv`)
| candidate_id | contig | summit | strand | ensemble | p_value | q_value | n_windows |
|---|---|---|---|---|---|---|---|
{md_rows}

## Files
- `top{a.top_n}_promoters.csv` — {', '.join(cols)}, sequence
- `top{a.top_n}_promoters_motif.csv` — the above + FIMO -10/-35 architecture (written by the sbatch motif step)
- `peaks_fdr.csv` — every called locus with p/q (the full FDR table)

_Reproducible: ensembling + p/q are deterministic (this file is CPU-only, no GPU, no RNG)._
"""
    (outdir / "analysis.md").write_text(md)

    # ----- LaTeX (standalone; booktabs) -----
    tex_rows = " \\\\\n".join(
        f"{_tex_escape(r.candidate_id)} & {_tex_escape(r.contig)} & {int(r.summit)} & {r.strand} & "
        f"{r.ensemble_score:.4f} & {r.p_value:.3e} & {r.q_value:.3e} & {int(r.n_windows)}"
        for r in tv.itertuples())
    tex = f"""\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{siunitx}}
\\title{{T27 empirical-FDR promoter mine --- \\textit{{C. butyricum}}}}
\\date{{ensemble: {_tex_escape(ensemble_str)}}}
\\begin{{document}}
\\maketitle

\\section*{{Method}}
Restricted scan + tailored null. {_tex_escape(intro)}

\\section*{{Null score distribution (ensemble rank-percentile)}}
median {q[0]:.4f}, p95 {q[1]:.4f}, p99 {q[2]:.4f}, max {q[3]:.4f}.
Smallest achievable p-value $= 1/(N_{{\\mathrm{{null}}}}+1) = \\num{{{smallest_p:.2e}}}$
($N_{{\\mathrm{{null}}}} = {len(null_ens):,}$).

\\section*{{Discoveries}}
\\begin{{itemize}}
  \\item candidate windows tested: {len(rc):,}
  \\item windows passing empirical $p \\le {a.peak_alpha}$: peak-called into \\textbf{{{len(peaks):,} promoter loci}}
  \\item loci at $q \\le 0.05$: \\textbf{{{n_q05:,}}}; at $q \\le 0.10$: {n_q10:,}
  \\item reported: top {len(top)} by ensemble score ({dropped} summit windows truncated at a contig end)
  \\item top-{len(top)} ensemble range $[{ens_lo:.4f}, {ens_hi:.4f}]$, $q$ range $[\\num{{{q_lo:.2e}}}, \\num{{{q_hi:.2e}}}]$
\\end{{itemize}}

\\section*{{Top {n_show} promoters}}
(full set in \\texttt{{top{a.top_n}\\_promoters.csv}})
\\begin{{center}}
\\begin{{tabular}}{{llrrrrrr}}
\\toprule
candidate\\_id & contig & summit & strand & ensemble & $p$ & $q$ & n\\_windows \\\\
\\midrule
{tex_rows} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{center}}

\\end{{document}}
"""
    (outdir / "analysis.tex").write_text(tex)
    print("\n" + md)
    print(f"[report] wrote analysis.md + analysis.tex ({n_show}-row p/q preview; full table in the CSV)")


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prep", help="contig preflight + build packed null FASTA + null regions BED")
    p.add_argument("--genome", required=True, help="target FASTA (C_butyricum.fna)")
    p.add_argument("--gff", required=True, help="target GFF3 (C_butyricum.gff3)")
    p.add_argument("--null-tsv", required=True, nargs="+", metavar="TSV",
                   help="one or more convergent_intergenic_negatives.tsv (81 bp 'sequence' col); pooled")
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=cmd_prep)

    f = sub.add_parser("fdr", help="ensemble + empirical p/q over aligned candidate + null tracks")
    f.add_argument("--models", required=True, nargs="+", help="model names (must match the *_aligned.csv basenames)")
    f.add_argument("--cand-dir", required=True, help="dir of {model}_aligned.csv candidate tracks (align_tracks.py)")
    f.add_argument("--null-dir", required=True, help="dir of {model}_aligned.csv null tracks (align_tracks.py)")
    f.add_argument("--genome", required=True, help="target FASTA (for summit-sequence extraction)")
    f.add_argument("--out-dir", required=True)
    f.add_argument("--peak-alpha", type=float, default=0.05, help="empirical-p gate for a window to seed a peak")
    f.add_argument("--merge", type=int, default=WINDOW, help="bp; merge windows within this gap into one locus")
    f.add_argument("--top-n", type=int, default=80, help="how many promoters to report")
    f.set_defaults(func=cmd_fdr)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
