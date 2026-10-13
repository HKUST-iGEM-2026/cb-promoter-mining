#!/usr/bin/env python3
"""
join_fdr_qvalues.py — attach T27 empirical-FDR q-values to the T32 candidate table.

Now that the mine (T28->T32) and the FDR (T27) run on the SAME assembly (GCF_014131795.1 = DSM 10702
complete), their coordinates are comparable, so each candidate promoter can be annotated with the FDR
peak that covers it — the join that was blocked when the mine was g409755 and the FDR
ran on T26's 4-contig assembly.

Matches by genomic-interval overlap on (chromosome/contig, strand):
  candidate peak region [region_start, region_end]  (1-based inclusive, build_candidate_table.py)
  vs FDR peak          [peak_start,  peak_end]       (0-based half-open, fdr_stats.py peaks_fdr.csv)
When several FDR peaks overlap one candidate, the lowest q is attached. This is
an overlapping peak's selected-peak screening statistic, not a recalculated
candidate-specific q or experimental validation. A blank q means no peak matched
under this procedure; it is not proof that the candidate is a non-promoter.

  python join_fdr_qvalues.py --candidates g131795_candidates.tsv --fdr-peaks peaks_fdr.csv \
      --out g131795_candidates_fdr.tsv
"""
import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="T32 table (chromosome,strand,region_start,region_end,...)")
    ap.add_argument("--fdr-peaks", required=True, help="T27 peaks_fdr.csv (contig,strand,peak_start,peak_end,p_value,q_value[,ensemble_score])")
    ap.add_argument("--out", required=True, help="candidate table + fdr_* columns (TSV)")
    a = ap.parse_args()

    cand = pd.read_csv(a.candidates, sep="\t")
    peaks = pd.read_csv(a.fdr_peaks)
    for c in ("chromosome", "strand", "region_start", "region_end"):
        if c not in cand.columns:
            raise SystemExit(f"{a.candidates}: missing '{c}'; has {list(cand.columns)}")
    for c in ("contig", "strand", "peak_start", "peak_end", "p_value", "q_value"):
        if c not in peaks.columns:
            raise SystemExit(f"{a.fdr_peaks}: missing '{c}'; has {list(peaks.columns)}")

    cand["chromosome"] = cand["chromosome"].astype(str); cand["strand"] = cand["strand"].astype(str)
    peaks["contig"] = peaks["contig"].astype(str); peaks["strand"] = peaks["strand"].astype(str)

    # candidate region -> 0-based half-open [region_start-1, region_end) to compare with FDR peaks
    cand_s = cand["region_start"].to_numpy(int) - 1
    cand_e = cand["region_end"].to_numpy(int)

    qv = np.full(len(cand), np.nan); pv = np.full(len(cand), np.nan)
    ens = np.full(len(cand), np.nan); nmatch = np.zeros(len(cand), int)
    has_ens = "ensemble_score" in peaks.columns
    by = {k: g for k, g in peaks.groupby(["contig", "strand"])}

    for i in range(len(cand)):
        g = by.get((cand["chromosome"].iat[i], cand["strand"].iat[i]))
        if g is None:
            continue
        ov = (cand_s[i] < g["peak_end"].to_numpy(int)) & (g["peak_start"].to_numpy(int) < cand_e[i])
        if not ov.any():
            continue
        gg = g[ov]
        nmatch[i] = len(gg)
        j = gg["q_value"].astype(float).idxmin()            # most significant overlapping FDR peak
        qv[i] = float(gg["q_value"].loc[j]); pv[i] = float(gg["p_value"].loc[j])
        if has_ens:
            ens[i] = float(gg["ensemble_score"].loc[j])

    out = cand.copy()
    out["fdr_q_value"] = qv
    out["fdr_p_value"] = pv
    out["fdr_ensemble_score"] = ens
    out["fdr_n_overlapping_peaks"] = nmatch
    out["fdr_significant_q05"] = out["fdr_q_value"] <= 0.05   # NaN (unmatched) -> False
    out.to_csv(a.out, sep="\t", index=False)

    n = len(out); matched = int(np.isfinite(qv).sum())
    print(f"{n} candidates -> {a.out}")
    print(f"  matched to an FDR peak : {matched} ({matched / n:.1%})   unmatched (no FDR call): {n - matched}")
    print(f"  q <= 0.05: {int((out['fdr_q_value'] <= 0.05).sum())}   q <= 0.10: {int((out['fdr_q_value'] <= 0.10).sum())}")


if __name__ == "__main__":
    main()
