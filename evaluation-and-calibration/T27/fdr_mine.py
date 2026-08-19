#!/usr/bin/env python3
"""
fdr_mine.py — T27 empirical-FDR promoter mine on the TARGET, C. butyricum DSM 10702.

WHAT / WHY (read the RESTRICTED-SCAN rationale first — it is the load-bearing design choice)
============================================================================================
The four fine-tuned models score every 81 bp window, but a raw score is meaningless on its own: an
AT-rich Clostridium window scores high whether or not it holds a real -10/-35 (the project's core
false-positive trap). To turn scores into DISCOVERIES we compare each candidate against an EMPIRICAL
NULL — sequence we believe carries no promoter but shares the candidates' composition — and report
p-/q-values, not bare scores.

  *** WHY A RESTRICTED SCAN (candidate regions only), NOT the whole genome ***
  Promoters live in a small slice of the genome: the region just upstream of a gene, on the gene's
  strand. Two reasons this pipeline confines BOTH the candidates and the null to that intergenic
  slice rather than scanning genome-wide:
    1. A BETTER-TAILORED NULL. The null here is the CONVERGENT-INTERGENIC set (3'-to-3' gene
       boundaries that, BY ORIENTATION, carry no promoter traffic). It is intergenic, AT-rich,
       promoter-free — i.e. composition-matched to the upstream candidate regions. If we instead
       scanned the whole genome, most candidate windows would be deep coding sequence whose
       composition the intergenic null does NOT match, so the p-values would be comparing apples to
       oranges and the FDR would be uninterpretable. Restricting the candidates to the same
       intergenic space the null is drawn from is what makes the empirical p-value valid.
    2. FEWER, MORE HONEST TESTS. Deep-CDS windows can only add false positives (no promoter lives
       there), so scoring them just inflates the multiple-testing burden with windows we would never
       call anyway. Restricting to candidate regions is the repo's established "restrict-scan"
       pattern (genome_fp_benchmark.py --regions, the deployed Regime-C mine).
  Engineering note: the scan reuses the audited genome_inference.py (per-contig, per-strand
  coordinates handled by the engine — the safe path) with its opt-in --regions filter, so ONLY the
  candidate-region windows are scored — CDS windows are skipped UP FRONT (not scored-then-dropped),
  ~halving the scan so cnn_v2 fits a preempt window. The filter is additive + default-off, so every
  other genome_inference consumer is unaffected; the Python side still re-applies the same region
  test (idempotent, defense-in-depth). No calibration: the T29 ensemble is mean RANK-PERCENTILE,
  which is calibration-invariant.

PIPELINE (one reproducible entrypoint; the only reason it shells out is the conda-env split — NT
needs transformers 4.57 in nt-tf4x, the rest need 5.3 in the main env, and one Python process cannot
host both. So genome_inference.py is run per model as a subprocess under the right env; all the
statistics/aggregation happen in THIS process):
  1. candidate regions  = make_candidate_regions.py(gff)                       -> BED (upstream of genes)
  2. candidate scan      = genome_inference.py(genome fna, 81bp, +/-, --regions BED) x4 models ->
                           per-model window tracks RESTRICTED to the candidate regions (only upstream
                           windows scored; ~half the genome, cnn_v2-friendly). --regions is opt-in +
                           default-off, so other genome_inference consumers are byte-for-byte unaffected.
  3. null scan           = genome_inference.py(ALL convergent-intergenic 81bp seqs, both orientations,
                           packed into one contig, --step 81 --strands +)        -> per-model null scores (batched)
  4. ensemble (T29)      = POOL candidate+null per model, rank-percentile over the pool, mean across
                           models. Pooling is ESSENTIAL: rank-percentile is only comparable when the
                           candidate and null are ranked on ONE common scale.
  5. restrict candidates to the regions BED; empirical p per window = (#null >= score + 1)/(Nnull + 1)
  6. peak-call (T30)     = merge adjacent windows with p <= --peak-alpha into one locus (summit = best)
  7. FDR                 = Benjamini-Hochberg q-values over the PEAKS (the calling unit)
  8. top-N promoters     = top --top-n peaks by ensemble score; extract the summit 81bp (reading strand)
  9. motif (T31)         = FIMO the top-N for the sigma-A -10/-35 boxes (annotate_motifs.py, bio env)

Runs BOTH model sets by default (--model-set both): the deployed xs (T21_xs) and the leakage-safe
0.6 retrain (T21_final_06), into separate output subdirs, so they can be compared.

CONTIG SAFETY (the past bug: 'contig_1' vs 'NC_...'): the candidate scan joins the genome FASTA, the
GFF3, and the score track by contig id. This script PREFLIGHTS that the GFF3 seqids are a subset of
the FASTA record ids and FAILS LOUD with both id sets printed if not — no silent 0-overlap run. The
contig id is always carried in an explicit `contig` column, never positionally.

  # on HPC4, inside a GPU job (see run_fdr_mine.sbatch):
  python fdr_mine.py \
      --genome   /project/igem2026/cb-promoter-mining/evaluation-and-calibration/T26/c_butyricum_annotation/C_butyricum.fna \
      --gff      /project/igem2026/cb-promoter-mining/evaluation-and-calibration/T26/c_butyricum_annotation/C_butyricum.gff3 \
      --null-tsv /project/igem2026/cb-promoter-mining/evaluation-and-calibration/T26/c_butyricum_annotation/convergent_intergenic_negatives.tsv \
      --model-root /project/igem2026/cb-promoter-mining/infrastructure/T3/snakemake_scaffold/results \
      --model-set both --out-dir results/T27
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------- constants / model layout
WINDOW = 81
KEYS = ["contig", "start", "strand"]                       # a scored window's identity across tracks
MODELS = ["prokbert", "nt", "cnn", "cnn_v2"]               # the four ensemble members
SET_SUBDIR = {"06": "T21_final_06", "xs": "T21_xs"}        # --model-set -> results/ subdir
COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s):
    return s.translate(COMPLEMENT)[::-1]


def model_scan_args(model, model_root, subdir):
    """genome_inference.py args for one model + which conda env it must run in ('nt' or 'main')."""
    base = f"{model_root}/{subdir}"
    if model == "prokbert":
        return (f"--model-path {base}/prokbert/best_model "
                f"--model-id neuralbioinfo/prokbert-mini --batch-size 512"), "main"
    if model == "nt":
        return (f"--model-path {base}/nt_lora/best_adapter "
                f"--model-id InstaDeepAI/nucleotide-transformer-v2-250m-multi-species --batch-size 256"), "nt"
    if model == "cnn":
        return (f"--model-path {base}/cnn/best_model.pt --model-type cnn --batch-size 512"), "main"
    if model == "cnn_v2":
        return (f"--model-path {base}/cnn_v2/best_model.pt --model-type cnn_v2 --batch-size 512"), "main"
    raise ValueError(model)


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


def run(cmd, conda_base, env_prefix, hf_home, extra_exports=""):
    """Run `cmd` (a string) inside a conda env, the way the repo's sbatch scripts do."""
    exports = (f"export HF_HOME={hf_home} HF_HUB_OFFLINE=1 HF_TOKEN_PATH=/tmp/hf_notoken_$USER; "
               f"{extra_exports}")
    full = (f"source {conda_base}/etc/profile.d/conda.sh && conda activate {env_prefix} && "
            f"{exports} {cmd}")
    print(f"[run] ({env_prefix})\n      {cmd}", file=sys.stderr, flush=True)
    subprocess.run(["bash", "-c", full], check=True)


def scan(genome_inference, in_fasta, out_csv, model, model_root, subdir, envs, hf_home, conda_base,
         step, strands, regions_bed=None):
    """One genome_inference.py pass. Resume-safe: keep a completed track (skip on a preempt requeue).
    Candidates use step=1, both strands (a genomic slide) RESTRICTED to the candidate-regions BED so
    only upstream windows are scored; the null uses step=81, '+' only over the contig of
    pre-concatenated 81 bp blocks (no --regions) so every window is exactly one null sequence AND the
    whole pool is scored in batches. Both use --window 81 (the model window)."""
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        print(f"[skip] track exists: {out_csv}", file=sys.stderr)
        return
    margs, which = model_scan_args(model, model_root, subdir)
    env_prefix = envs["nt"] if which == "nt" else envs["main"]
    tmp = out_csv + ".tmp"
    cmd = (f"python {genome_inference} --genome {in_fasta} {margs} "
           f"--window {WINDOW} --step {step} --strands {strands} --out {tmp}")
    if regions_bed:
        cmd += f" --regions {regions_bed}"
    run(cmd, conda_base, env_prefix, hf_home)
    os.replace(tmp, out_csv)


def load_track(path, sort_keys):
    df = pd.read_csv(path, usecols=lambda c: c in ("contig", "start", "end", "strand", "promoter_score"))
    for c in ("contig", "start", "strand", "promoter_score"):
        if c not in df.columns:
            raise SystemExit(f"{path}: missing column '{c}'; has {list(df.columns)}")
    df["contig"] = df["contig"].astype(str)
    df["strand"] = df["strand"].astype(str)
    return df.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)


# ------------------------------------------------------------------- statistics
def pooled_ensemble(cand_tracks, null_tracks):
    """Per model, rank-percentile the POOLED (candidate + null) raw scores so the two sets are on ONE
    common scale, then average across models. Returns (cand_ens, null_ens) aligned to the input row
    order of model[0]. Fails loud if the models are not on an identical grid (a silent misalignment
    would scramble the ensemble)."""
    ncand = len(cand_tracks[MODELS[0]])
    nnull = len(null_tracks[MODELS[0]])
    cand_ref = cand_tracks[MODELS[0]][KEYS]
    null_ref = null_tracks[MODELS[0]][KEYS]
    cand_pcts, null_pcts = [], []
    for m in MODELS:
        ct, nt = cand_tracks[m], null_tracks[m]
        if len(ct) != ncand or not ct[KEYS].equals(cand_ref):
            raise SystemExit(f"candidate track '{m}' is not on the same window grid as '{MODELS[0]}' "
                             f"({len(ct):,} vs {ncand:,} rows or key mismatch). Re-scan the odd one out.")
        if len(nt) != nnull or not nt[KEYS].equals(null_ref):
            raise SystemExit(f"null track '{m}' is not aligned with '{MODELS[0]}' "
                             f"({len(nt):,} vs {nnull:,} rows or key mismatch).")
        pooled = np.concatenate([ct["promoter_score"].to_numpy(), nt["promoter_score"].to_numpy()])
        pct = pd.Series(pooled).rank(pct=True, method="average").to_numpy()
        cand_pcts.append(pct[:ncand])
        null_pcts.append(pct[ncand:])
    cand_ens = np.mean(np.column_stack(cand_pcts), axis=1)
    null_ens = np.mean(np.column_stack(null_pcts), axis=1)
    return cand_ens, null_ens


def region_mask(contig, start, strand, regions):
    """Windows [start, start+WINDOW) overlapping a candidate region on their (contig, strand).
    Mirrors genome_fp_benchmark.in_region_mask / T29 / T30."""
    keep = np.zeros(len(start), dtype=bool)
    we = start + WINDOW
    for (c, s), g in regions.groupby(["contig", "strand"]):
        gi = np.where((contig == str(c)) & (strand == str(s)))[0]
        if len(gi) == 0:
            continue
        g = g.sort_values("start")
        rs, re = g["start"].to_numpy(int), g["end"].to_numpy(int)
        idx = np.searchsorted(rs, we[gi], side="right") - 1        # region with largest start < window end
        ok = idx >= 0
        ov = np.zeros(len(gi), dtype=bool)
        ov[ok] = re[idx[ok]] > start[gi][ok]                        # ...and it ends after the window starts
        keep[gi[ov]] = True
    return keep


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
    """Collapse windows (already p-gated) into loci: adjacent same-(contig,strand) windows within
    `merge` bp -> one peak; summit = highest-ensemble window. Mirrors T30 call_peaks.py."""
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


# ------------------------------------------------------------------- one model-set run
def run_one_set(setname, a, envs, genome):
    subdir = SET_SUBDIR[setname]
    outdir = Path(a.out_dir) / f"cbut_{setname}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\n========================= MODEL SET {setname} ({subdir}) =========================")

    # --- candidate + null scans, all four models ---
    cand_tracks, null_tracks = {}, {}
    for m in MODELS:
        cpath = str(outdir / f"cand_{m}.csv")
        npath = str(outdir / f"null_{m}.csv")
        scan(a.genome_inference, a.genome, cpath, m, a.model_root, subdir, envs, a.hf_home,
             a.conda_base, step=a.step, strands="+,-", regions_bed=a.regions_bed)
        scan(a.genome_inference, a.null_fasta, npath, m, a.model_root, subdir,
             envs, a.hf_home, a.conda_base, step=WINDOW, strands="+")
        cand_tracks[m] = load_track(cpath, KEYS)
        null_tracks[m] = load_track(npath, ["contig", "strand", "start"])

    # --- RESTRICT candidates to upstream regions FIRST, then ensemble over (restricted cands + null) ---
    # Ranking over the restricted space (not the whole genome) keeps CDS score density OUT of the
    # rank-percentile normalization, consistent with the tailored intergenic null. The full-genome
    # CDS windows were scored (genome_inference can't region-limit) but are dropped here — they never
    # enter the ranking or the FDR.
    n_full = len(cand_tracks[MODELS[0]])
    base = cand_tracks[MODELS[0]]
    for m in MODELS:                                   # grids must match before we positionally mask
        ct = cand_tracks[m]
        if len(ct) != n_full or not ct[KEYS].equals(base[KEYS]):
            raise SystemExit(f"candidate track '{m}' is not on the same window grid as '{MODELS[0]}' "
                             f"({len(ct):,} vs {n_full:,} rows or key mismatch). Re-scan the odd one out.")
    regions = pd.read_csv(a.regions_bed, sep="\t")
    regions["contig"] = regions["contig"].astype(str)
    regions["strand"] = regions["strand"].astype(str)
    keep = region_mask(base["contig"].to_numpy().astype(str), base["start"].to_numpy(int),
                       base["strand"].to_numpy().astype(str), regions)
    if not keep.any():
        raise SystemExit("no candidate window overlaps a candidate region — check the regions BED / "
                         "contig ids (region contigs must match the scan-track contigs).")
    rcand_tracks = {m: cand_tracks[m].loc[keep].reset_index(drop=True) for m in MODELS}

    # --- pooled rank-percentile ensemble (T29) over RESTRICTED candidates + null; empirical p ---
    rc = rcand_tracks[MODELS[0]][["contig", "start", "end", "strand"]].copy()
    cand_ens, null_ens = pooled_ensemble(rcand_tracks, null_tracks)
    rc["ensemble"] = cand_ens
    rc["p_value"] = empirical_p(cand_ens, null_ens)
    print(f"  restricted candidates: {len(rc):,}/{n_full:,} windows ({keep.mean():.1%}); "
          f"null samples: {len(null_ens):,}")

    # --- peak-call the p-gated windows, then BH q-values over PEAKS (the calling unit) ---
    gated = rc[rc["p_value"] <= a.peak_alpha]
    if gated.empty:
        raise SystemExit(f"no restricted window passed empirical p <= {a.peak_alpha}. Nothing to call.")
    peaks = call_peaks(gated, a.merge)
    peaks["q_value"] = bh_qvalues(peaks["p_value"].to_numpy())
    peaks = peaks.sort_values("ensemble_score", ascending=False, kind="mergesort").reset_index(drop=True)

    # --- top-N promoters + their summit sequence (reading orientation) ---
    top = peaks.head(a.top_n).copy()
    seqs, drop = [], 0
    for _, r in top.iterrows():
        s = genome[str(r["contig"])][int(r["summit"]):int(r["summit"]) + WINDOW]
        s = revcomp(s) if r["strand"] == "-" else s
        seqs.append(s if len(s) == WINDOW else "")     # blank = truncated at a contig end
    top["sequence"] = seqs
    drop = int((top["sequence"].str.len() != WINDOW).sum())
    top.insert(0, "candidate_id", [f"cbut_{setname}_{i + 1:04d}" for i in range(len(top))])

    peaks.to_csv(outdir / "peaks_fdr.csv", index=False)
    top_csv = outdir / f"top{a.top_n}_promoters.csv"
    top.to_csv(top_csv, index=False)

    # --- motif detection on the top-N (T31 FIMO, bio env) ---
    motif_ok = run_motif(a, top_csv, outdir, envs)
    write_report(setname, subdir, a, n_full, rc, null_ens, peaks, top, drop, motif_ok, outdir)
    print(f"  -> {outdir}/  (top{a.top_n}_promoters.csv, peaks_fdr.csv, analysis.md)")


def run_motif(a, top_csv, outdir, envs):
    if a.skip_motif:
        print("  [motif] --skip-motif set; leaving motif annotation to a later T31 pass.", file=sys.stderr)
        return False
    out = outdir / f"top{a.top_n}_promoters_motif.csv"
    cmd = (f"python {a.annotate_motifs} --peaks {top_csv} --genome {a.genome} "
           f"--motif {a.motif} --window {WINDOW} --fimo-thresh {a.fimo_thresh} --out {out}")
    try:
        run(cmd, a.conda_base, envs["bio"], a.hf_home)
        return True
    except subprocess.CalledProcessError:
        print("  [motif] FIMO step failed (is the bio env / meme available?). The top-N table without "
              "motif columns is still written; re-run annotate_motifs.py later.", file=sys.stderr)
        return False


def write_report(setname, subdir, a, n_full, rc, null_ens, peaks, top, dropped, motif_ok, outdir):
    q = np.quantile(null_ens, [0.5, 0.95, 0.99, 1.0])
    arch = ""
    motif_file = outdir / f"top{a.top_n}_promoters_motif.csv"
    if motif_ok and motif_file.exists():
        mt = pd.read_csv(motif_file)
        if "architecture" in mt.columns:
            vc = mt["architecture"].value_counts()
            arch = "\n".join(f"  - {k}: {int(vc.get(k, 0))}"
                             for k in ("canonical", "both_bad_spacing", "minus10_only", "minus35_only", "none"))
    md = f"""# T27 empirical-FDR promoter mine — C. butyricum — model set `{setname}` ({subdir})

**Restricted scan + tailored null.** Candidates are the {len(rc):,} upstream-of-gene windows (of
{n_full:,} scanned); the null is the {len(null_ens):,} convergent-intergenic samples (both
strands of the 81 bp CB negatives). Both are intergenic + AT-rich, so the empirical p-value is a
composition-matched comparison (see the script header). Ensemble = 4-model mean rank-percentile,
pooled over the RESTRICTED candidates + null (NOT the whole genome — CDS score density is kept out of
the normalization). Calibration-free; no calibration.

## Null score distribution (ensemble rank-percentile)
median {q[0]:.4f} · p95 {q[1]:.4f} · p99 {q[2]:.4f} · max {q[3]:.4f}
Smallest achievable p-value = 1/(Nnull+1) = {1.0 / (len(null_ens) + 1):.2e}.

## Discoveries
- candidate windows tested: {len(rc):,}
- windows passing empirical p <= {a.peak_alpha}: peak-called into **{len(peaks):,} promoter loci**
- loci at q <= 0.05: **{int((peaks['q_value'] <= 0.05).sum()):,}**   ·   q <= 0.10: {int((peaks['q_value'] <= 0.10).sum()):,}
- reported: top {len(top)} by ensemble score ({dropped} summit windows truncated at a contig end -> blank sequence)
- top-{len(top)} ensemble range [{top['ensemble_score'].min():.4f}, {top['ensemble_score'].max():.4f}], q range [{top['q_value'].min():.2e}, {top['q_value'].max():.2e}]

## Top-{len(top)} promoter architecture (FIMO -10/-35)
{arch if arch else '  (motif step not run — see --skip-motif / bio env; run annotate_motifs.py on the top-N CSV)'}

## Files
- `top{a.top_n}_promoters.csv` — candidate_id, contig, summit, strand, ensemble_score, p_value, q_value, n_windows, sequence
- `top{a.top_n}_promoters_motif.csv` — the above + has_minus10/minus35, spacer, architecture {'(written)' if motif_ok else '(NOT written — motif step skipped/failed)'}
- `peaks_fdr.csv` — every called locus with p/q (the full FDR table)

_Reproducible: scoring, ensembling, p/q are deterministic; FIMO is deterministic. Both model sets
(xs, 0.6) run under --model-set both._
"""
    (outdir / "analysis.md").write_text(md)
    print("\n" + md)


# ------------------------------------------------------------------- main
def main():
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genome", required=True, help="target FASTA (C_butyricum.fna)")
    ap.add_argument("--gff", required=True, help="target GFF3 (C_butyricum.gff3) — candidate regions")
    ap.add_argument("--null-tsv", required=True, help="convergent_intergenic_negatives.tsv (81 bp 'sequence' col)")
    ap.add_argument("--model-root", required=True, help="dir holding T21_final_06/ and T21_xs/")
    ap.add_argument("--model-set", choices=["06", "xs", "both"], default="both")
    ap.add_argument("--out-dir", default="results/T27")
    # candidate-region + FDR knobs
    ap.add_argument("--up", type=int, default=300, help="bp upstream of each gene start (candidate regions)")
    ap.add_argument("--down", type=int, default=50, help="bp into the gene past its start (candidate regions)")
    ap.add_argument("--step", type=int, default=1, help="scan step (1 = every position; bump to speed cnn_v2)")
    ap.add_argument("--peak-alpha", type=float, default=0.05, help="empirical-p gate for a window to seed a peak")
    ap.add_argument("--merge", type=int, default=WINDOW, help="bp; merge windows within this gap into one locus")
    ap.add_argument("--top-n", type=int, default=80, help="how many promoters to report + motif-annotate")
    ap.add_argument("--fimo-thresh", default="1e-3", help="FIMO p-value threshold (T31)")
    ap.add_argument("--skip-motif", action="store_true", help="skip the FIMO step (no bio env available)")
    # tool + env locations (defaults track the repo layout / the sbatch scripts)
    ap.add_argument("--genome-inference", default=str(repo / "evaluation-and-calibration" / "genome_inference.py"))
    ap.add_argument("--make-regions", default=str(repo / "evaluation-and-calibration" / "make_candidate_regions.py"))
    ap.add_argument("--annotate-motifs", default=str(repo / "mining-and-motif-annotation" / "T31" / "annotate_motifs.py"))
    ap.add_argument("--motif", default=str(repo / "mining-and-motif-annotation" / "T31" / "sigma_a_promoter.meme"))
    ap.add_argument("--conda-base", default="/cm/shared/apps/Anaconda3/2023.09-0")
    ap.add_argument("--main-env", default="/project/igem2026/envs/promoter-mining-hpc4")
    ap.add_argument("--nt-env", default="/project/igem2026/envs/nt-tf4x")
    ap.add_argument("--bio-env", default="/project/igem2026/envs/meme",
                    help="conda prefix with the MEME suite (fimo) for the motif step")
    ap.add_argument("--hf-home", default="/project/igem2026/hf_shared")
    a = ap.parse_args()
    envs = {"main": a.main_env, "nt": a.nt_env, "bio": a.bio_env}
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------- preflight: contig-id consistency (the 'contig_1' vs 'NC_...' trap) ----------
    genome = read_fasta(a.genome)
    fasta_ids = set(genome)
    gff_ids = gff_seqids(a.gff)
    missing = gff_ids - fasta_ids
    if missing:
        raise SystemExit(
            "CONTIG-ID MISMATCH: GFF3 seqids are not all present in the genome FASTA — the candidate "
            "scan would join on nothing and silently find 0 promoters. Fix the annotation/FASTA pair "
            "so their contig ids agree (this multi-contig genome cannot use --contig-name).\n"
            f"  FASTA ids (n={len(fasta_ids)}): {sorted(fasta_ids)[:5]}{' ...' if len(fasta_ids) > 5 else ''}\n"
            f"  GFF3  ids (n={len(gff_ids)}): {sorted(gff_ids)[:5]}{' ...' if len(gff_ids) > 5 else ''}\n"
            f"  in GFF3 but NOT FASTA: {sorted(missing)[:5]}")
    print(f"[preflight] contig ids OK: {len(gff_ids)} GFF3 seqids all present among {len(fasta_ids)} FASTA records.")

    # ---------- candidate regions BED (once; contig ids come straight from the GFF) ----------
    a.regions_bed = str(out / "cbut_candidate_regions.bed")
    if not os.path.exists(a.regions_bed):
        run(f"python {a.make_regions} --gff {a.gff} --up {a.up} --down {a.down} --out {a.regions_bed}",
            a.conda_base, envs["main"], a.hf_home)

    # ---------- null FASTA from the TSV (validate 81 bp, clean ACGT) — written once, reused per set ----------
    null_fasta_master = out / "null.fasta"
    if not null_fasta_master.exists():
        ndf = pd.read_csv(a.null_tsv, sep="\t")
        if "sequence" not in ndf.columns:
            raise SystemExit(f"{a.null_tsv}: no 'sequence' column; has {list(ndf.columns)}")
        seqs = ndf["sequence"].astype(str).str.upper()
        if len(seqs) == 0:
            raise SystemExit(f"{a.null_tsv}: 0 rows — an empty null pool cannot define an FDR.")
        bad_len = int((seqs.str.len() != WINDOW).sum())
        bad_chr = int((~seqs.str.fullmatch("[ACGT]+")).sum())
        if bad_len or bad_chr:
            raise SystemExit(f"null sequences failed validation: {bad_len} not {WINDOW} bp, {bad_chr} non-ACGT. "
                             f"The null must be clean {WINDOW} bp to be scored by the {WINDOW} bp models.")
        # BATCHABLE null over the FULL convergent-intergenic set. Pack every 81 bp block — each
        # sequence AND its reverse complement, so the null spans both orientations exactly as the
        # candidate scan does — end-to-end into ONE contig. Scored later with --window 81 --step 81
        # --strands + so window i == block i and the whole pool goes through the model in batches.
        # NEVER scan this contig on '-' (that reverse-complements the WHOLE concatenation and
        # scrambles the blocks); the reverse orientation is already present as explicit rc blocks.
        fwd = list(seqs)
        blocks = fwd + [revcomp(s) for s in fwd]
        with open(null_fasta_master, "w") as fh:
            fh.write(">nullcat\n" + "".join(blocks) + "\n")
        print(f"[null] {len(fwd):,} convergent-intergenic seqs x2 orientations = {len(blocks):,} null "
              f"blocks ({WINDOW} bp each) packed into one batched contig -> {null_fasta_master}")
    a.null_fasta = str(null_fasta_master)      # ONE validated null FASTA, reused by every model set

    sets = ["06", "xs"] if a.model_set == "both" else [a.model_set]
    for s in sets:
        run_one_set(s, a, envs, genome)

    print(f"\n=== DONE === outputs under {a.out_dir}/cbut_{{{','.join(sets)}}}/")


if __name__ == "__main__":
    main()
