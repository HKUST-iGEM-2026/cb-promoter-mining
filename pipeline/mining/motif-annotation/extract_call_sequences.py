#!/usr/bin/env python3
"""
extract_call_sequences.py — T31 step 1. Pull the summit-window DNA for each T30 promoter call, in
reading orientation, so MEME/FIMO can look for the -10/-35 boxes.

Each call's `summit` is the start of its top-scoring 81 bp window; the model frames the TSS at offset 60
inside that window, so the -10 (~10 bp upstream of the TSS -> ~offset 50) and -35 (~35 bp upstream ->
~offset 25) boxes sit inside it. We emit exactly the window the model scored (reverse-complemented for
the - strand), named by a stable `candidate_id`, so a FIMO hit's position in the sequence is directly an
offset from the window 5' end — and thus a fixed distance from the TSS (1-based position 61).

Writes: an all-calls FASTA (FIMO scans this), a top-N FASTA (MEME learns the motif from the cleanest
calls), and calls_with_id.tsv (the T30 calls + candidate_id, for T32).

  python extract_call_sequences.py --calls g409755_peaks.csv --genome g409755.fna \
      --min-windows 5 --top-n 500 \
      --out-fasta candidates.fasta --out-top candidates_top.fasta --out-calls calls_with_id.tsv
"""
import argparse
import pandas as pd

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s):
    return s.translate(COMPLEMENT)[::-1]


def read_fasta(path):
    seqs, cid, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", required=True, help="T30 call_peaks.py output (contig,summit,strand,peak_score,n_windows,...)")
    ap.add_argument("--genome", required=True, help="genome FASTA the scan was run on (record ids must match the calls' contig)")
    ap.add_argument("--out-fasta", required=True, help="all kept calls -> FASTA (FIMO scans this)")
    ap.add_argument("--out-top", required=True, help="top-N by peak_score -> FASTA (MEME learns the motif from this)")
    ap.add_argument("--out-calls", required=True, help="kept calls + candidate_id -> TSV (input to T32)")
    ap.add_argument("--window", type=int, default=81)
    ap.add_argument("--min-windows", type=int, default=5,
                    help="drop calls with fewer supporting windows (single-window peaks are usually noise)")
    ap.add_argument("--top-n", type=int, default=500, help="how many top calls MEME learns the motif from")
    a = ap.parse_args()

    calls = pd.read_csv(a.calls)
    for c in ("contig", "summit", "strand", "peak_score", "n_windows"):
        if c not in calls.columns:
            raise SystemExit(f"{a.calls}: missing column '{c}'; has {list(calls.columns)}")
    calls["contig"] = calls["contig"].astype(str); calls["strand"] = calls["strand"].astype(str)
    calls = calls[calls["n_windows"] >= a.min_windows].copy()
    calls = calls.sort_values("peak_score", ascending=False, kind="mergesort").reset_index(drop=True)
    calls.insert(0, "candidate_id", [f"cand_{i + 1:05d}" for i in range(len(calls))])

    genome = read_fasta(a.genome)
    missing = set(calls["contig"]) - set(genome)
    if missing:
        raise SystemExit(f"{len(missing)} call contigs are not in the genome FASTA, e.g. {sorted(missing)[:3]}. "
                         f"Wrong FASTA, or contig-id mismatch.")

    def win(row):
        s = genome[row["contig"]][int(row["summit"]):int(row["summit"]) + a.window]
        return revcomp(s) if row["strand"] == "-" else s
    calls["seq"] = calls.apply(win, axis=1)
    short = (calls["seq"].str.len() != a.window).sum()
    calls = calls[calls["seq"].str.len() == a.window].reset_index(drop=True)   # drop windows truncated at a contig end

    with open(a.out_fasta, "w") as f:
        for _, r in calls.iterrows():
            f.write(f">{r['candidate_id']}\n{r['seq']}\n")
    top = calls.head(a.top_n)
    with open(a.out_top, "w") as f:
        for _, r in top.iterrows():
            f.write(f">{r['candidate_id']}\n{r['seq']}\n")
    calls.drop(columns=["seq"]).to_csv(a.out_calls, sep="\t", index=False)

    print(f"{len(calls):,} calls kept (n_windows>={a.min_windows}"
          f"{f', {short} dropped at contig ends' if short else ''}) -> {a.out_fasta}")
    print(f"  MEME set: top {len(top)} -> {a.out_top}   |   T32 input: {a.out_calls}")


if __name__ == "__main__":
    main()
