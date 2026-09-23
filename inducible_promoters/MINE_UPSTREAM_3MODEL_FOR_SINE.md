# Mining the oxidative-stress gene upstreams with the 3-model ensemble — run-book for Sine

HKUST iGEM 2026 Dry Lab. Oak asked me to check your `CB_oxidative_stress_genes.csv` and write you a
run-book for mining the **upstream regions of these genes** with our deployed 3-model ensemble
(ProkBERT + cnn_v2 + Evo2). This is the *same* machinery as the `g409755` target mine, just **restricted
to these genes' upstreams** instead of the whole genome. It reuses `mine_3model.sbatch` unchanged — you
only build a regions BED and point the job at the right genome.

**Verdict: the gene list is usable.** Coordinates are internally clean (all 30 genes' spans equal
`aa×3+3`, 0 mismatches), strand is given, and both scanners in the ensemble are multi-replicon safe. But
there is **one caveat you must act on before running** (assembly identity) and a couple of conceptual
guardrails. Read §1 first — the rest is mechanical.

---

## 1. Read first — three things about this data

**(a) These genes are on a DIFFERENT assembly than our usual target.** The CSV is annotated on
`NZ_CP040626.1` (24 genes) + `NZ_CP040627.1` (7 genes) — a **complete, two-replicon** *C. butyricum*
assembly (locus prefix `FF104`). Our whole Stage-1 pipeline mines **DSM 10702 = `g409755` =
`GCF_000409755.1`**, a *draft* assembly with `NZ_AQQF01000…` contigs. **They are not the same genome.**
Consequences:

- You must download and scan **the FASTA these coordinates index** (the `NZ_CP040626/627` assembly), *not*
  `g409755.fna`. If you point the mine at `g409755`, the coordinates are meaningless and the contig-id
  preflight will abort. §3 covers the download.
- The models themselves are genome-agnostic (they score any 81 bp of DNA and the Evo2 probe is reused
  across genomes), so scanning a different assembly is fine — it's only the **coordinates + FASTA** that
  must be self-consistent.
- **Flag for Oak:** the promoters you find will be *this strain's* promoters. If the wet-lab chassis is
  DSM 10702, someone should later map these back to DSM 10702 by orthology (or just re-run the same BED
  logic on the DSM 10702 annotation). Not your problem to solve now — just don't let it get silently
  forgotten. I could not confirm the exact strain name (NCBI's API was unreachable from here); the
  accessions are what matter and they're unambiguous.

**(b) These are CDS coordinates, not TSS.** So "upstream" is taken from the **start codon**: window
`[−300, +50]` around the gene's 5′ end (the convention in `INDUCIBLE_PROMOTER_WORKFLOW.md` Phase C and in
`make_candidate_regions.py`). Two of these are adjacent co-oriented pairs that are almost certainly
**operons** (e.g. the two glutathione peroxidases `RS21250`/`RS21255`; the `trxB`/glutaredoxin cluster
`RS17285`/`RS17290`/`RS17295`; the two EFR1 ferredoxins `RS09545`/`RS09560`) — an operon-internal gene has
no promoter of its own, so its "−300" window sits inside the operon. For a first pass, mining each gene's
own upstream is exactly what Oak asked for; if you want to be rigorous, collapse operons to their leader
first (reuse **T12**) per `INDUCIBLE_PROMOTER_WORKFLOW.md` Phase C. I'd run the simple version first and
revisit only the operon members that score low.

**(c) The ensemble finds "is this a promoter," not "is this oxidative-stress–inducible."** ProkBERT/
cnn_v2/Evo2 were trained on **constitutive** promoters. This mine **locates and scores the candidate
promoter in each upstream** — it does *not* prove the promoter switches on under O₂ stress. Inducibility
comes from the DE / PerR-operator work in `INDUCIBLE_PROMOTER_WORKFLOW.md` (Phases B/D). So the honest
framing of your output is: *"here is the predicted promoter for each oxidative-defense gene, and how
promoter-like it looks."* A **low** ensemble score is a flag for manual inspection, **not** an auto-reject
— inducible promoters can carry an operator over the −10/−35 that the constitutive models never saw.

**Data hygiene (already handled, just so you know):** the CSV has **30 unique genes**, not 31 —
`FF104_RS08110` is duplicated. Two rows (`RS19340`, the duplicate `RS08110`) have a stray `1` +
non-breaking-space in the *Proteins* column. Neither touches the coordinates/strand, and the BED builder
in §2 dedups and ignores that column, so you don't need to fix the CSV.

---

## 2. Build the upstream regions BED

Save this as `build_upstream_regions.py` (I verified it against the CSV — 30 regions, both replicons, no
negative coordinates, coordinates identical to `make_candidate_regions.py`). It emits the exact 4-column
BED `mine_3model.sbatch` expects, plus a `gene_map.tsv` so you can assign promoters back to genes at the
end.

```python
#!/usr/bin/env python3
"""CSV of genes -> upstream REGIONS BED (drop-in for mine_3model.sbatch) + a gene<->region map.
Strand-aware [-UP, +DOWN] around each gene's 5' end, same convention as make_candidate_regions.py."""
import argparse, csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--up", type=int, default=300)
    ap.add_argument("--down", type=int, default=50)
    ap.add_argument("--bed", default="cb_oxstress_upstream.bed")
    ap.add_argument("--map", default="cb_oxstress_gene_map.tsv")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8-sig")))
    seen, genes = set(), []
    for r in rows:
        contig, rng = r["Genomic location"].strip().split(":")
        s1, e1 = (int(x) for x in rng.split("-"))          # 1-based inclusive, s1 < e1 in genome order
        strand = "+" if r["Orientation"].strip().lower().startswith("plus") else "-"
        key = (r["Locus tag"].strip(), contig, s1, e1, strand)
        if key in seen:                                    # drop the duplicate row (RS08110)
            continue
        seen.add(key)
        if strand == "+":                                  # 5' end = s1
            start0, end0 = (s1 - 1) - a.up, (s1 - 1) + a.down
        else:                                              # 5' end = e1
            start0, end0 = e1 - a.down, e1 + a.up
        start0 = max(0, start0)
        genes.append(dict(locus=r["Locus tag"].strip(), contig=contig, start=start0, end=end0,
                          strand=strand, tier=r["Tier"].strip(),
                          symbol=r["Gene symbol"].strip(), func=r["Function"].strip()))

    with open(a.bed, "w") as f:                            # exact header the scanners' load_regions expects
        f.write("contig\tstart\tend\tstrand\n")
        for g in sorted(genes, key=lambda g: (g["contig"], g["start"], g["strand"])):
            f.write(f"{g['contig']}\t{g['start']}\t{g['end']}\t{g['strand']}\n")

    with open(a.map, "w") as f:
        f.write("locus_tag\tcontig\tup_start\tup_end\tstrand\ttier\tsymbol\tfunction\n")
        for g in sorted(genes, key=lambda g: (g["tier"], g["contig"], g["start"])):
            f.write(f"{g['locus']}\t{g['contig']}\t{g['start']}\t{g['end']}\t{g['strand']}"
                    f"\t{g['tier']}\t{g['symbol']}\t{g['func']}\n")

    print(f"{len(rows)} rows -> {len(genes)} unique genes -> {a.bed}  ({a.map})")

if __name__ == "__main__":
    main()
```

Run it from the `inducible_promoters/` folder:

```bash
python build_upstream_regions.py CB_oxidative_stress_genes.csv \
    --up 300 --down 50 \
    --bed cb_oxstress_upstream.bed --map cb_oxstress_gene_map.tsv
# expect: 31 rows -> 30 unique genes -> cb_oxstress_upstream.bed  (cb_oxstress_gene_map.tsv)
```

`--up 300 --down 50` matches the constitutive restricted-scan convention; leave it unless you have a
reason. Total candidate space is ~10.5 kb over 30 regions — tiny, so the mine will be fast.

---

## 3. Get the genome FASTA (the RIGHT one)

Download the two replicons **by accession** on the HPC4 login node (it has internet, same as the T7b
genome fetches):

```bash
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NZ_CP040626.1,NZ_CP040627.1&rettype=fasta&retmode=text" \
    -o cb_2repl.fna
grep '^>' cb_2repl.fna          # MUST show exactly two records: NZ_CP040626.1 and NZ_CP040627.1
```

**Contig-id parity check** (this is what the mine's preflight enforces, and the #1 way a scan silently
mis-scores): the FASTA headers' first tokens must equal the BED's contigs.

```bash
grep '^>' cb_2repl.fna | sed 's/^>//; s/ .*//' | sort         # NZ_CP040626.1 / NZ_CP040627.1
tail -n +2 cb_oxstress_upstream.bed | cut -f1 | sort -u        # same two — they must match
```

If NCBI hands back bare `CP040626.1` (no `NZ_` prefix) in the header, either re-fetch with the `NZ_`
accessions or `sed -i 's/^>CP0406/>NZ_CP0406/' cb_2repl.fna` so ids line up with the CSV/BED. (The
`ncbi-datasets-cli` in the env is an alternative if you'd rather pull the whole assembly + GFF — but it
needs the `GCF_` accession, which efetch-by-nucleotide-accession saves you from looking up.)

Stage both files somewhere on `/project` and note their **absolute** paths, e.g.:

```bash
RC=/project/igem2026/cb-promoter-mining/infrastructure/T3/snakemake_scaffold/results/regimeC
mkdir -p $RC/inducible
cp cb_2repl.fna cb_oxstress_upstream.bed cb_oxstress_gene_map.tsv $RC/inducible/
```

---

## 4. Run the 3-model ensemble mine

Reuse `mine_3model.sbatch` in **target-mine mode** (`TRUTH=""` → no benchmark, just the score-sorted
shortlist). Everything is passed on the submit line; **don't edit the script**:

```bash
cd /project/igem2026/cb-promoter-mining
RC=infrastructure/T3/snakemake_scaffold/results/regimeC

TAG=cb_oxstress TRUTH="" MODELS="prokbert cnn_v2 evo2" STEP=10 \
  FASTA=$PWD/$RC/inducible/cb_2repl.fna \
  REGIONS=$PWD/$RC/inducible/cb_oxstress_upstream.bed \
  sbatch mining-and-motif-annotation/genome-scan/mine_3model.sbatch
```

Notes:
- `TRUTH=""` (empty, not unset) is what puts it in mine mode — the script uses `${TRUTH-default}` (single
  `-`) so an explicit empty string disables the benchmark. This is the exact fix that made the `g409755`
  mine work; keep it.
- Keep `STEP=10` — it's the deployment grid Evo2's `--align-grid` and `align_tracks.py` expect. Don't
  change it or the tracks won't reconcile.
- The job auto-switches to the `evo2` env for the Evo2 member and back to the main env for T29; you don't
  manage envs. The Evo2 probe (`evo2_probe_xs.pkl`) is genome-agnostic and already staged.
- Run **one job per `TAG`** at a time (per-model tracks are cached and shared by TAG).
- Expect it to finish in well under the `g409755` mine's ~17 min — the candidate space here is ~10 kb, not
  a whole genome.

**Output** lands in `$RC/mine3_cb_oxstress/`:
- `cb_oxstress_shortlist_3m_restricted.csv` ← **the mine deliverable**: every scored window in the 30
  upstreams, ranked by the 3-model mean-rank-percentile ensemble (`promoter_score`).
- per-model tracks + `cb_oxstress_ensemble_3m.csv` cached alongside.

---

## 5. Turn the shortlist into a per-gene promoter call

Because this is a *targeted* mine (30 known genes, not a genome scan), the most useful summary is **the
best promoter window per gene**, using the `gene_map.tsv` from §2. A hard genome-wide 0.9 cutoff (what
`g409755` used) is optional here — you care about the top call in *each* gene's upstream even if some sit
below 0.9.

```python
import pandas as pd
sl  = pd.read_csv("cb_oxstress_shortlist_3m_restricted.csv")     # contig,start,end,strand,promoter_score
gm  = pd.read_csv("cb_oxstress_gene_map.tsv", sep="\t")
# assign each scored window to the gene whose upstream interval contains it (same contig+strand)
sl["contig"] = sl["contig"].astype(str)
rows = []
for _, g in gm.iterrows():
    w = sl[(sl.contig == g.contig) & (sl.strand == g.strand) &
           (sl.start >= g.up_start) & (sl.end <= g.up_end)]
    if len(w):
        best = w.loc[w.promoter_score.idxmax()]
        rows.append({**g.to_dict(), "best_score": best.promoter_score,
                     "best_start": int(best.start), "n_windows": len(w)})
    else:
        rows.append({**g.to_dict(), "best_score": None, "best_start": None, "n_windows": 0})
out = pd.DataFrame(rows).sort_values(["tier", "best_score"], ascending=[True, False])
out.to_csv("cb_oxstress_per_gene_calls.tsv", sep="\t", index=False)
print(out[["locus_tag","symbol","func","tier","best_score","n_windows"]].to_string(index=False))
```

That gives you, per gene: its predicted promoter location (`best_start`) and how promoter-like the
ensemble thinks it is (`best_score`). Sort within Tier 1 first — those are the high-priority defenses
(SOD, peroxiredoxin, AhpC, rubrerythrins, thioredoxin system).

**Optional — motif annotation** for the strong calls, identical to the `g409755` tail: run
`mining-and-motif-annotation/peak-calling/call_peaks.py --track cb_oxstress_shortlist_3m_restricted.csv
--min-score 0.9 --merge 81 --out cb_oxstress_peaks.csv`, then T31 FIMO (`extract_call_sequences.py` →
`fimo` in the `/project/igem2026/envs/meme` env → `build_candidate_table.py`) to place −10/−35 boxes. See
`mining-and-motif-annotation/T28_T29_RUN_CLAUDE.md` for that flow — but note the T31 motif is the
**constitutive σA** −10/−35; for these genes the *inducible* signal (the PerR "Per box") is a separate
scan described in `INDUCIBLE_PROMOTER_WORKFLOW.md` Phase D.

---

## 6. What "done" looks like

`cb_oxstress_per_gene_calls.tsv`: 30 genes, each with an ensemble promoter score and predicted promoter
position in its upstream, sorted by tier then score. That's the mine result. Interpreting it — which of
these are genuinely *inducible* switches — is the DE + Per-box work, not this scan.

If anything mismatches (contig ids, empty regions on a replicon, a gene with `n_windows == 0` because its
upstream ran off a contig end), check the parity step in §3 first — that's where 9/10 problems are.

---

### Quick reference — the whole thing

```bash
# 1. BED + gene map from the CSV
python build_upstream_regions.py CB_oxidative_stress_genes.csv \
    --bed cb_oxstress_upstream.bed --map cb_oxstress_gene_map.tsv

# 2. the RIGHT genome (two replicons, by accession)
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NZ_CP040626.1,NZ_CP040627.1&rettype=fasta&retmode=text" -o cb_2repl.fna
grep '^>' cb_2repl.fna    # verify 2 records, ids match the BED

# 3. mine (target-mine mode, 3-model ensemble)
TAG=cb_oxstress TRUTH="" MODELS="prokbert cnn_v2 evo2" STEP=10 \
  FASTA=$PWD/cb_2repl.fna REGIONS=$PWD/cb_oxstress_upstream.bed \
  sbatch mining-and-motif-annotation/genome-scan/mine_3model.sbatch

# 4. per-gene best call  ->  cb_oxstress_per_gene_calls.tsv   (snippet in §5)
```

Ping me/Oak if the strain identity needs settling before wet-lab hand-off — that's the one open decision
this mine leaves on the table.
