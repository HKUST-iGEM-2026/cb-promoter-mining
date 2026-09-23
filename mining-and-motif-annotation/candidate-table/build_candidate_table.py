#!/usr/bin/env python3
"""
build_candidate_table.py — T32. The final promoter candidate table, in the PT6-locked schema and
enriched with the T31 motif annotation. Merges the T30 calls (via candidate_id) with the FIMO -10/-35
hits, converts summit coordinates to a TSS position using the model's training convention, and flags
which candidates have real promoter architecture (a correctly-placed -10 box).

PT6 coordinate convention (prototypes/pipeline-validation/validate_promoters.py): the model frames the +1 TSS at 0-based
offset 60 inside the 81 bp window, so for a call with summit S:
    forward TSS (1-based) = S + 61          reverse TSS (1-based) = S + 81 - 60 = S + 21
The -10 box sits ~5-15 bp upstream of the TSS, the -35 ~25-40 bp upstream; a FIMO hit's position in the
(reading-orientation) window is compared to the TSS at 1-based position 61 to place it.

Output columns = the PT6 schema (candidate_id, chromosome, strand, tss_position, summit_tss,
region_start, region_end, peak_score, mean_score, n_windows, nearest_known_id, distance_to_known,
is_validated) + motif columns (has_promoter_architecture, has_minus10/-35, offsets, spacer, p-values).
The three `*_known*` / `is_validated` columns stay blank in a target mine (no known TSS); pass --truth to
fill them for a validation genome.

  python build_candidate_table.py --calls calls_with_id.tsv --fimo t31_out/fimo/fimo.tsv \
      --out g409755_candidates.tsv --jsonl g409755_candidates.jsonl
"""
import argparse
import json
import numpy as np
import pandas as pd

WINDOW = 81
TSS_OFFSET = 60          # 0-based offset of the +1 TSS inside the scored window (1-based = 61)
TSS_1B = TSS_OFFSET + 1  # 61

# where the boxes sit relative to the TSS (bp upstream of the +1), a little slack around the textbook
# -10 (~7 bp up) and -35 (~35 bp up, i.e. -35 box start ~33-38 up)
M10_UP = (3, 18)
M35_UP = (22, 42)


def load_fimo(path, pthr):
    df = pd.read_csv(path, sep="\t", comment="#")
    ren = {"sequence_name": "cid", "p-value": "pvalue"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    for c in ("cid", "start", "stop", "pvalue"):
        if c not in df.columns:
            raise SystemExit(f"{path}: FIMO output missing '{c}'; has {list(df.columns)}")
    return df[df["pvalue"] <= pthr].copy()


def annotate(hits):
    """One candidate's FIMO hits -> motif summary dict. Positions are 1-based within the 81 bp window."""
    if hits.empty:
        return dict(n_motif_hits=0, has_minus10=False, has_minus35=False, minus10_offset=None,
                    minus10_pvalue=None, minus35_offset=None, spacer_bp=None,
                    has_promoter_architecture=False, best_motif_pvalue=None)
    h = hits.copy()
    h["center"] = (h["start"] + h["stop"]) / 2.0
    h["upstream"] = TSS_1B - h["center"]                       # +ve = upstream of the TSS
    m10 = h[(h["upstream"] >= M10_UP[0]) & (h["upstream"] <= M10_UP[1])]
    m35 = h[(h["upstream"] >= M35_UP[0]) & (h["upstream"] <= M35_UP[1])]
    b10 = m10.loc[m10["pvalue"].idxmin()] if not m10.empty else None
    b35 = m35.loc[m35["pvalue"].idxmin()] if not m35.empty else None
    spacer = int(round(b35["upstream"] - b10["upstream"])) if (b10 is not None and b35 is not None) else None
    return dict(
        n_motif_hits=int(len(h)),
        has_minus10=b10 is not None,
        has_minus35=b35 is not None,
        minus10_offset=(int(round(b10["upstream"])) if b10 is not None else None),
        minus10_pvalue=(float(b10["pvalue"]) if b10 is not None else None),
        minus35_offset=(int(round(b35["upstream"])) if b35 is not None else None),
        spacer_bp=spacer,
        has_promoter_architecture=(b10 is not None),           # a correctly-placed -10 is the core signal
        best_motif_pvalue=float(h["pvalue"].min()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", required=True, help="calls_with_id.tsv from extract_call_sequences.py")
    ap.add_argument("--fimo", required=True, help="fimo.tsv from annotate_motifs.sh")
    ap.add_argument("--out", required=True, help="final candidate table (TSV)")
    ap.add_argument("--jsonl", default=None, help="also write one JSON object per candidate")
    ap.add_argument("--pvalue", type=float, default=1e-3, help="FIMO p-value cutoff to count a motif hit")
    a = ap.parse_args()

    calls = pd.read_csv(a.calls, sep="\t")
    calls["strand"] = calls["strand"].astype(str)
    fwd = calls["strand"] == "+"
    calls["summit_tss"] = np.where(fwd, calls["summit"] + TSS_1B, calls["summit"] + WINDOW - TSS_OFFSET)
    calls["tss_position"] = calls["summit_tss"]                # summit is the point estimate (no per-window mean carried from T30)
    calls["region_start"] = calls["peak_start"] + 1            # 1-based inclusive
    calls["region_end"] = calls["peak_end"]                    # peak_end is 0-based exclusive == 1-based inclusive
    calls["chromosome"] = calls["contig"].astype(str)

    fimo = load_fimo(a.fimo, a.pvalue)
    groups = {k: g for k, g in fimo.groupby("cid")} if len(fimo) else {}
    empty = fimo.iloc[0:0]
    ann = pd.DataFrame([annotate(groups.get(cid, empty)) for cid in calls["candidate_id"]])
    out = pd.concat([calls.reset_index(drop=True), ann], axis=1)

    for c in ("nearest_known_id", "distance_to_known", "is_validated"):
        out[c] = None                                          # mine mode: no known TSS

    cols = ["candidate_id", "chromosome", "strand", "tss_position", "summit_tss",
            "region_start", "region_end", "peak_score", "mean_score", "n_windows",
            "nearest_known_id", "distance_to_known", "is_validated",
            "has_promoter_architecture", "has_minus10", "minus10_offset", "minus10_pvalue",
            "has_minus35", "minus35_offset", "spacer_bp", "n_motif_hits", "best_motif_pvalue"]
    out = out.reindex(columns=cols)
    out.to_csv(a.out, sep="\t", index=False)
    if a.jsonl:
        with open(a.jsonl, "w") as f:
            for _, r in out.iterrows():
                f.write(json.dumps({k: (None if (isinstance(v, float) and pd.isna(v)) else v)
                                    for k, v in r.items()}) + "\n")

    n = len(out)
    arch = int(out["has_promoter_architecture"].sum())
    both = int((out["has_minus10"] & out["has_minus35"]).sum())
    print(f"{n:,} candidates -> {a.out}")
    print(f"  placed -10 (promoter architecture): {arch:,} ({arch / n:.1%} of calls)")
    print(f"  -10 AND -35 placed:                 {both:,} ({both / n:.1%})")
    if a.jsonl:
        print(f"  JSONL -> {a.jsonl}")


if __name__ == "__main__":
    main()
