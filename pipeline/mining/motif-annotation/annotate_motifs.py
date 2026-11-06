#!/usr/bin/env python3
"""
annotate_motifs.py — T31 -10/-35 promoter-architecture annotation with FIMO.

T30 gives one call per locus, but an ensemble rank does not establish promoter activity.
This extracts each call's oriented summit window, scans it with six-base reference
PWMs for -35-like (TTGACA) and -10-like (TATAAT) patterns, and labels their presence
and edge-to-edge spacing. A "canonical" label is a sequence-based hypothesis, not
independent validation of promoter activity, sigma usage or constitutive expression.
Reference-PWM backgrounds affect scoring; AT-rich DNA can still contain strong hits.
The ordinary catalogue's original de-novo-motif scan is a separate procedure.

FIMO ships with the MEME suite (the CPU bio env, environment-bio.yml — NOT the main torch env):
    conda activate <bio env with meme>   # provides `fimo`

Input:  T30 peaks (contig, summit, strand, ...) + the genome FASTA.
Output: the peaks with added columns has_minus10 / minus10_start / minus10_pval /
        has_minus35 / minus35_start / minus35_pval / spacer / architecture
        (architecture in {canonical, both_bad_spacing, minus10_only, minus35_only, none}).

  conda activate <bio env>
  python annotate_motifs.py --peaks g409755_peaks.csv --genome g409755.fna --out g409755_peaks_motif.csv
"""
import argparse
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd

COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s):
    return s.translate(COMP)[::-1]


def load_fasta(path):
    seqs, cid, buf = {}, None, []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if cid is not None:
                    seqs[cid] = "".join(buf)
                cid = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if cid is not None:
        seqs[cid] = "".join(buf)
    return seqs


def extract(peaks, genome, window, flank):
    """summit window (+/- flank), reading-strand (revcomp on '-'). name = peak row index."""
    out = {}
    for i, r in peaks.iterrows():
        c = str(r["contig"])
        seq = genome.get(c, "")
        s = int(r["summit"])
        a = max(0, s - flank)
        b = min(len(seq), s + window + flank)
        sub = seq[a:b].upper()
        if str(r["strand"]) == "-":
            sub = revcomp(sub)
        out[f"peak{i:06d}"] = sub
    return out


def run_fimo(motif, fasta, outdir, thresh):
    if not shutil.which("fimo"):
        raise SystemExit("`fimo` is not on PATH. Activate the MEME-suite (bio) env first, e.g. "
                         "`conda activate $ENV_MEME` (docs/SETUP.md).")
    # --norc: the extracted sequences are ALREADY on the reading strand, so only scan the + strand.
    subprocess.run(["fimo", "--norc", "--thresh", str(thresh), "--oc", outdir, motif, fasta], check=True)
    return os.path.join(outdir, "fimo.tsv")


def parse_fimo(tsv):
    """best (lowest p-value) hit per (sequence, motif)."""
    df = pd.read_csv(tsv, sep="\t", comment="#")
    df = df.dropna(subset=["sequence_name", "motif_id"])
    if df.empty:
        return df
    return df.sort_values("p-value").groupby(["sequence_name", "motif_id"], sort=False).first().reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", required=True, help="T30 call list (needs contig, summit, strand)")
    ap.add_argument("--genome", required=True, help="genome FASTA (contig ids must match the peaks)")
    ap.add_argument("--motif", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sigma_a_promoter.meme"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=81, help="summit window length (match the model's window)")
    ap.add_argument("--flank", type=int, default=0, help="extend the extracted region by this many bp each side")
    ap.add_argument("--fimo-thresh", default="1e-3", help="FIMO p-value threshold")
    ap.add_argument("--spacer-min", type=int, default=14, help="min bp between -35 and -10 for 'canonical'")
    ap.add_argument("--spacer-max", type=int, default=21, help="max bp between -35 and -10 for 'canonical'")
    ap.add_argument("--fimo-tsv", default=None, help="reuse an existing fimo.tsv instead of running FIMO")
    ap.add_argument("--keep-fasta", default=None, help="also write the extracted summit sequences here")
    a = ap.parse_args()

    peaks = pd.read_csv(a.peaks)
    for c in ("contig", "summit", "strand"):
        if c not in peaks.columns:
            raise SystemExit(f"{a.peaks}: missing column '{c}'; has {list(peaks.columns)}")
    genome = load_fasta(a.genome)
    seqs = extract(peaks, genome, a.window, a.flank)

    workdir = tempfile.mkdtemp(prefix="t31_")
    fa = a.keep_fasta or os.path.join(workdir, "seqs.fasta")
    with open(fa, "w") as f:
        for name, sub in seqs.items():
            f.write(f">{name}\n{sub}\n")

    tsv = a.fimo_tsv or run_fimo(a.motif, fa, os.path.join(workdir, "fimo"), a.fimo_thresh)
    best = parse_fimo(tsv)
    m10 = best[best.motif_id == "minus10"].set_index("sequence_name") if len(best) else pd.DataFrame()
    m35 = best[best.motif_id == "minus35"].set_index("sequence_name") if len(best) else pd.DataFrame()

    recs = []
    for i in range(len(peaks)):
        name = f"peak{i:06d}"
        h10 = m10.loc[name] if len(m10) and name in m10.index else None
        h35 = m35.loc[name] if len(m35) and name in m35.index else None
        has10, has35 = h10 is not None, h35 is not None
        s10 = int(h10["start"]) if has10 else -1
        s35 = int(h35["start"]) if has35 else -1
        e35 = int(h35["stop"]) if has35 else -1
        p10 = float(h10["p-value"]) if has10 else np.nan
        p35 = float(h35["p-value"]) if has35 else np.nan
        spacer = (s10 - e35 - 1) if (has10 and has35 and s10 > e35) else -1
        if has10 and has35 and s10 > e35 and a.spacer_min <= spacer <= a.spacer_max:
            arch = "canonical"
        elif has10 and has35:
            arch = "both_bad_spacing"
        elif has10:
            arch = "minus10_only"
        elif has35:
            arch = "minus35_only"
        else:
            arch = "none"
        recs.append((has10, s10, p10, has35, s35, p35, spacer, arch))

    ann = pd.DataFrame(recs, columns=["has_minus10", "minus10_start", "minus10_pval",
                                      "has_minus35", "minus35_start", "minus35_pval", "spacer", "architecture"])
    out = pd.concat([peaks.reset_index(drop=True), ann], axis=1)
    out.to_csv(a.out, index=False)

    vc = out["architecture"].value_counts()
    print(f"{len(out):,} peaks annotated -> {a.out}")
    for k in ("canonical", "both_bad_spacing", "minus10_only", "minus35_only", "none"):
        print(f"  {k:17s}: {int(vc.get(k, 0)):,}")
    if "n_windows" in out.columns:
        strong = (out["architecture"] == "canonical") & (out["n_windows"] >= 5)
        print(f"  canonical AND n_windows>=5 (highest-confidence promoters): {int(strong.sum()):,}")


if __name__ == "__main__":
    main()
