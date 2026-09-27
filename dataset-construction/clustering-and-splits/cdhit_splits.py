#!/usr/bin/env python3
"""
T15 — cluster-aware train/val/test splits for the promoter classifier.

  >>> FIRST DRAFT for Sine to review — runnable, but three choices need a human sign-off.
      Search "DECISION:" for them.  Don't treat the defaults as settled science. <<<

What it does
------------
Combines the labelled positive + negative DNA windows, removes near-duplicate *leakage* by
clustering with CD-HIT-EST, and writes cluster-disjoint 80/10/10 train/val/test splits so that
no near-identical sequence ever straddles two splits (otherwise CV scores inflate).

**Eval is VERIFIED-only:** T13 silver (inferred) positives are pinned to TRAIN, so val/test never hold
inferred labels — they measure performance on verified promoters, not the "AT-rich upstream = promoter"
assumption that would reward the AT-trap during model selection.

Wired in the T3 Snakefile (rule t15_cdhit_splits) as:

  python cdhit_splits.py --positives <T8> <T7b> <T9train> <T13> \
                         --negatives <T14> --identity 0.9 --seed 42 --out results/T15/

Sources are auto-detected by their columns and harmonised to one schema:
  T8  reference_promoters.parquet      cross-species (E. coli + B. subtilis), 81 bp, OUT-of-genus, VERIFIED
  T7b published_tss_promoters.parquet  C. acetobutylicum (Hocq), 81 bp, in-genus, VERIFIED
  T9  literature_promoters_train.pq    C. acetobutylicum curated, ~51 bp, in-genus, VERIFIED
  T13 upstream_regions.tsv             housekeeping-leader upstreams, 300 bp, in-genus, INFERRED (silver)
  T14 negative_set.tsv                 SRS/CDS/Downstream/Intergenic, 300 bp, NEGATIVE

The three DECISIONS (all exposed as CLI flags so you can sweep them):
  1. --window 81        target window length (verified positives are 81 bp). RE-WINDOWING is implemented:
                        T13 300->81 = the gene-PROXIMAL 81 bp (last 81; verified against T13's 5'->3'
                        extraction — both strands put the promoter at the 3' end); T14 CDS/Downstream/
                        Intergenic = centre 81; T14 SRS is REGENERATED as an 81 bp dinucleotide shuffle of
                        the re-windowed T13 positives (cropping the 300 bp SRS would NOT be the matched
                        null). T9 (~51 bp) is N-padded — see the T9 note below.
  2. --silver-weight 0.3  T13 is INFERRED (not experimentally confirmed). Down-weight it so the model
                        learns the -10/-35 motif from VERIFIED data, not "AT-rich upstream = promoter"
                        (the AT-rich false-positive trap). 1.0 = trust it fully; 0.0 = drop it.
                        NB down-weighting silver alone skews TRAIN negative-heavy, so step 6b rescales
                        TRAIN negative weights to a ~50/50 effective prior. The `weight` column =
                        label-confidence x class-balance; trainers MUST apply it as a per-sample loss
                        weight (else it is inert and T13 trains at full weight — the AT-trap).
  3. --neg-per-pos 1.0  negatives per positive (~50/50 by convention). 0 = keep all and balance via class
                        weights at train time instead.

  T9 (~49-52 bp) — RECOMMENDATION: hold it fully out for eval (config t9_train_fraction=0). N-padding
  ~37% of an 81 bp window is a poor substitute, and now that T7b supplies ~1,434 verified in-genus
  positives the "split T9 into training" workaround (option b, made when T7b was empty) is no longer
  needed. The code still pads T9 if it is passed, so nothing breaks — but the splits are cleaner without it.
"""
import argparse, os, sys, subprocess, shutil, random, collections
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

GOLD = 1.0


def dinucleotide_shuffle(seq, rng):
    """Dinucleotide-preserving (Eulerian) shuffle — same method as T14, so the SRS null stays
    consistent project-wide: destroys the -10/-35 motif while preserving base + dinucleotide
    composition. `rng` is a seeded random.Random for reproducibility."""
    seq = seq.replace("N", "")
    if len(seq) < 2:
        return seq
    edges = collections.defaultdict(list)
    for a, b in zip(seq, seq[1:]):
        edges[a].append(b)
    for node, e in edges.items():
        if len(e) > 1:
            rest = e[:-1]; rng.shuffle(rest); edges[node] = rest + [e[-1]]
    out, cur = [seq[0]], seq[0]
    while edges[cur]:
        nxt = edges[cur].pop(0); out.append(nxt); cur = nxt
    if len(out) < len(seq):                       # Eulerian path trapped -> plain shuffle fallback
        lst = list(seq); rng.shuffle(lst); return "".join(lst)
    return "".join(out)
OUT_COLS = ["sequence", "label", "cluster_id", "split", "in_genus", "strand",
            "source", "species", "sigma_factor", "weight", "promoter_id"]


# --------------------------------------------------------------------------- load + harmonise
def load_source(path, is_positive, silver_weight):
    """Detect a source by its columns and map it onto the common schema."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"  [warn] {path} missing/0-byte (placeholder) — skipping", file=sys.stderr)
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t") if path.endswith((".tsv", ".txt")) else pd.read_parquet(path)
    if df.empty:
        print(f"  [warn] {os.path.basename(path)} has 0 rows — skipping", file=sys.stderr)
        return pd.DataFrame()
    cols = set(df.columns)

    if "upstream_sequence" in cols:                          # ---- T13 inferred upstreams (300 bp)
        out = pd.DataFrame({
            "sequence":     df["upstream_sequence"].astype(str),
            "strand":       df.get("strand", "."),
            "species":      df.get("genome", "Clostridium_sp"),
            "sigma_factor": np.nan,
            "source":       "T13_inferred",
            "in_genus":     True,
            "weight":       silver_weight,        # DECISION: T13 is inferred -> silver/down-weighted
            "promoter_id":  df.get("locus_tag", pd.Series(df.index.astype(str))),
        })
        anchor = "end"   # VERIFIED vs T13 code: upstream_sequence runs 5'->3' toward the gene (both
                         # strands), so the gene-proximal promoter sits at the 3' end -> take the last `window` bp
    elif "negative_class" in cols:                           # ---- T14 negatives (300 bp)
        out = pd.DataFrame({
            "sequence":     df["sequence"].astype(str),
            "strand":       df.get("strand", "."),
            "species":      df.get("genome", "Clostridium_sp"),
            "sigma_factor": np.nan,
            "source":       "T14_" + df["negative_class"].astype(str),
            "in_genus":     True,
            "weight":       GOLD,
            "promoter_id":  pd.Series(df.index.astype(str)),
        })
        anchor = "center"
    elif "confidence_level" in cols:                         # ---- T8 cross-species (81 bp, out-of-genus)
        out = pd.DataFrame({
            "sequence":     df["sequence"].astype(str),
            "strand":       ".",
            "species":      df.get("species", "?"),
            "sigma_factor": df.get("sigma_factor", np.nan),
            "source":       "T8_" + df.get("source_db", "RegulonDB").astype(str),
            "in_genus":     False,                # E. coli / B. subtilis are NOT genus Clostridium
            "weight":       GOLD,
            "promoter_id":  df.get("promoter_id", pd.Series(df.index.astype(str))),
        })
        anchor = "center"
    elif "in_genus" in cols and "sequence" in cols:          # ---- T7b / T9 verified in-genus
        out = pd.DataFrame({
            "sequence":     df["sequence"].astype(str),
            "strand":       df.get("strand", "."),
            "species":      df.get("species", "?"),
            "sigma_factor": df.get("sigma_factor", np.nan),
            "source":       df.get("source_db", "curated").astype(str),  # source_db separates Hocq vs T9
            "in_genus":     df["in_genus"],
            "weight":       GOLD,
            "promoter_id":  df.get("promoter_id", pd.Series(df.index.astype(str))),
        })
        anchor = "center"
    else:
        raise ValueError(f"unrecognised schema for {path}: {sorted(cols)}")

    # Carry the long-context column + its core offset through UNCHANGED if the
    # source provides them (build_context_windows.py output). Re-windowing below
    # only touches `sequence` (the 81 bp core); sequence_context rides through as-is.
    for extra in ("sequence_context", "core_offset"):
        if extra in cols:
            out[extra] = df[extra].values

    out["label"] = 1 if is_positive else 0     # the --positives/--negatives flag is authoritative
    out["_anchor"] = anchor
    print(f"  loaded {len(out):>7} rows from {os.path.basename(path)}  "
          f"(label={out['label'].iloc[0]}, anchor={anchor})", file=sys.stderr)
    return out


# --------------------------------------------------------------------------- re-window
def rewindow(seq, target, anchor):
    """Crop or pad a sequence to `target` bp.
    DECISION/TODO (Sine): the choices here are biological assumptions, not settled —
      * T13 (300->81) uses anchor='end' assuming the promoter sits at the gene-proximal end.
      * T9 (~51->81) is centre-padded with N; ideally re-extract 81 bp of real genomic context.
    """
    seq = str(seq).upper()
    L = len(seq)
    if L == target:
        return seq
    if L > target:
        off = {"center": (L - target) // 2, "end": L - target, "start": 0}[anchor]
        return seq[off:off + target]
    pad = target - L
    return "N" * (pad // 2) + seq + "N" * (pad - pad // 2)


# --------------------------------------------------------------------------- CD-HIT-EST
def cdhit_clusters(seqs, identity, word, threads, workdir):
    """Cluster with cd-hit-est (the *nucleotide* tool) and return a per-sequence cluster id."""
    if not shutil.which("cd-hit-est"):
        sys.exit("ERROR: cd-hit-est not on PATH. It ships with the `cd-hit` conda package "
                 "(in environment.yml); `conda install -c bioconda cd-hit` if missing.")
    fa, out = os.path.join(workdir, "all.fasta"), os.path.join(workdir, "cdhit")
    with open(fa, "w") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">{i}\n{s}\n")
    subprocess.run(["cd-hit-est", "-i", fa, "-o", out, "-c", str(identity),
                    "-n", str(word), "-d", "0", "-M", "0", "-T", str(threads)],
                   check=True, stdout=subprocess.DEVNULL)
    cluster_of, cid = {}, -1
    with open(out + ".clstr") as fh:
        for line in fh:
            if line.startswith(">Cluster"):
                cid = int(line.split()[1])
            elif ">" in line:
                cluster_of[int(line.split(">")[1].split("...")[0])] = cid
    return [cluster_of[i] for i in range(len(seqs))]


def pick_word(c):   # cd-hit-est recommended word size (-n) for a given identity (-c)
    return 10 if c >= 0.95 else 8 if c >= 0.90 else 7 if c >= 0.88 else 6 if c >= 0.85 else 5 if c >= 0.80 else 4


# --------------------------------------------------------------------------- MMseqs2 (low identity)
def mmseqs_clusters(seqs, identity, coverage, cov_mode, threads, workdir):
    """Cluster nucleotide sequences with MMseqs2 easy-cluster and return per-sequence cluster ids.

    Needed because cd-hit-est cannot cluster DNA below ~0.75-0.80 identity (its short-word filter
    has no valid word size for nucleotides below that). MMseqs2 is alignment-based and handles low
    identity, but at low identity you MUST also set a coverage threshold (`-c` / `--cov-mode`), or
    sequences matching over only a short stretch get merged.
    """
    if not shutil.which("mmseqs"):
        sys.exit("ERROR: `mmseqs` not on PATH. Needed for identity < 0.8 (cd-hit-est's DNA floor). "
                 "Install with `conda install -c bioconda mmseqs2` (add it to environment.yml).")
    cdir = os.path.join(workdir, "_cluster_mmseqs")     # keep mmseqs artifacts out of the split dir
    os.makedirs(cdir, exist_ok=True)
    fa = os.path.join(cdir, "all.fasta")
    with open(fa, "w") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">{i}\n{s}\n")
    prefix, tmp = os.path.join(cdir, "mmseqs"), os.path.join(cdir, "tmp")
    subprocess.run(["mmseqs", "easy-cluster", fa, prefix, tmp,
                    "--min-seq-id", str(identity), "-c", str(coverage),
                    "--cov-mode", str(cov_mode), "--threads", str(threads)],
                   check=True, stdout=subprocess.DEVNULL)
    rep_of = {}                                    # member index -> representative index
    with open(prefix + "_cluster.tsv") as fh:
        for line in fh:
            rep, mem = line.rstrip("\n").split("\t")
            rep_of[int(mem)] = int(rep)
    missing = [i for i in range(len(seqs)) if i not in rep_of]
    if missing:
        sys.exit(f"ERROR: MMseqs2 did not return {len(missing)} of {len(seqs)} sequences "
                 f"(first few: {missing[:5]}). Check the run / inputs.")
    reps = {r: c for c, r in enumerate(sorted(set(rep_of.values())))}   # rep index -> compact cluster id
    return [reps[rep_of[i]] for i in range(len(seqs))]


def cluster_sequences(seqs, identity, tool, coverage, cov_mode, word, threads, workdir):
    """Dispatch to cd-hit-est (high identity) or MMseqs2 (low identity). tool='auto' picks by
    identity (cd-hit-est's DNA floor is ~0.8). Returns (cluster_ids, tool_name)."""
    if tool == "auto":
        tool = "cdhit" if identity >= 0.8 else "mmseqs2"
    if tool == "cdhit":
        if identity < 0.8:
            sys.exit(f"ERROR: cd-hit-est cannot cluster DNA at identity {identity} (< ~0.8). "
                     f"Use --cluster-tool mmseqs2 (or --cluster-tool auto).")
        w = word or pick_word(identity)
        print(f"\ncd-hit-est: {len(seqs)} seqs @ identity={identity} (-n {w}) ...", file=sys.stderr)
        return cdhit_clusters(seqs, identity, w, threads, workdir), "cd-hit-est"
    else:
        print(f"\nmmseqs2 easy-cluster: {len(seqs)} seqs @ min-seq-id={identity}, "
              f"cov={coverage} (mode {cov_mode}) ...", file=sys.stderr)
        return mmseqs_clusters(seqs, identity, coverage, cov_mode, threads, workdir), "mmseqs2"


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="T15 cluster-aware 80/10/10 splits")
    ap.add_argument("--positives", nargs="+", required=True)
    ap.add_argument("--negatives", nargs="+", required=True)
    ap.add_argument("--identity", type=float, default=0.6,
                    help="clustering identity (config cdhit_identity). Default 0.6 per reviewer "
                    "guidance (stricter anti-leakage than the old 0.9). NOTE cd-hit-est cannot "
                    "cluster DNA below ~0.8, so identity <0.8 auto-routes to MMseqs2.")
    ap.add_argument("--cluster-tool", choices=["auto", "cdhit", "mmseqs2"], default="auto",
                    help="clustering backend. 'auto' = cd-hit-est for identity>=0.8, MMseqs2 below.")
    ap.add_argument("--cluster-coverage", type=float, default=0.8,
                    help="MMseqs2 coverage (-c): fraction of the sequence the alignment must span. "
                    "Only used by the mmseqs2 backend; important at low identity.")
    ap.add_argument("--cov-mode", type=int, default=1, help="MMseqs2 --cov-mode (default 1).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=81, help="DECISION: target window bp (verified positives are 81)")
    ap.add_argument("--neg-per-pos", type=float, default=1.0, help="DECISION: negatives per positive (1.0=~50/50; 0=keep all)")
    ap.add_argument("--silver-weight", type=float, default=0.3, help="DECISION: weight for inferred T13 positives (0=drop)")
    ap.add_argument("--word-size", type=int, default=0, help="cd-hit-est -n (0 = auto from identity)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 1. load + harmonise --------------------------------------------------------------
    frames = [load_source(p, True, args.silver_weight) for p in args.positives] \
           + [load_source(n, False, args.silver_weight) for n in args.negatives]
    frames = [f for f in frames if not f.empty]
    if not frames:
        sys.exit("ERROR: no usable input rows — are the inputs/ files staged (run local_prep + Phase B + T7b)?")
    df = pd.concat(frames, ignore_index=True)

    # Context columns are ALL-OR-NOTHING. pd.concat unions columns and fills gaps with NaN, so if
    # some sources carry sequence_context and others don't, has_ctx would be True but a subset of
    # rows are NaN — that NaN reaches the output parquet and later breaks NTv3 tokenisation with a
    # misleading "length 3 bp" error. Fail loudly here instead.
    if "sequence_context" in df.columns and df["sequence_context"].isna().any():
        bad = sorted(df.loc[df["sequence_context"].isna(), "source"].astype(str).unique())
        sys.exit(f"ERROR: sequence_context is present but MISSING for sources {bad}. Run "
                 f"build_context_windows.py on ALL sources (every positive AND every T14 negative "
                 f"class) or none — partial context is not allowed.")

    # 2. re-window everything to one length -------------------------------------------
    short = int((df["sequence"].str.len() < args.window).sum())
    if short:
        print(f"  [note] {short} sequences are shorter than {args.window}bp and will be N-padded "
              f"(these are the ~51bp T9 promoters). Prefer holding T9 out (config t9_train_fraction=0) "
              f"now that T7b provides verified in-genus positives — see header.", file=sys.stderr)
    df["sequence"] = [rewindow(s, args.window, a) for s, a in zip(df["sequence"], df["_anchor"])]
    df = df[df["sequence"].str.fullmatch(r"[ACGTN]+")].drop(columns="_anchor").reset_index(drop=True)

    # 2b. regenerate the SRS null AT the target window ---------------------------------
    #     T14's committed SRS is a dinucleotide shuffle of the full 300bp T13 upstream; the matched null
    #     for an 81bp positive is a shuffle of THAT 81bp window. So drop T14's SRS and re-shuffle the
    #     re-windowed T13 positives 1:1 (preserves base/dinucleotide composition, destroys the motif).
    df = df[df["source"] != "T14_SRS"].reset_index(drop=True)
    has_ctx = "sequence_context" in df.columns
    t13 = df[(df.label == 1) & (df.source == "T13_inferred")]
    if len(t13):
        _rng = random.Random(args.seed)
        shuffled = [dinucleotide_shuffle(s, _rng) for s in t13["sequence"]]
        srs_cols = {
            "sequence":     shuffled,
            "label":        0,
            "in_genus":     True,
            "strand":       ".",
            "source":       "SRS",
            "species":      t13["species"].values,
            "sigma_factor": np.nan,
            "weight":       GOLD,
            "promoter_id":  "SRS_" + t13["promoter_id"].astype(str).values,
        }
        if has_ctx:
            # Matched null WITH context: keep the REAL flanks and shuffle ONLY the
            # core region, so the model cannot separate positives from negatives on
            # "real vs shuffled flanks" (the failure a naive whole-window shuffle
            # would create). Needs core_offset to locate the core in the context.
            if "core_offset" not in df.columns or t13["core_offset"].isna().any():
                sys.exit("ERROR: sequence_context present but core_offset missing/NaN for T13 — "
                         "regenerate windows with build_context_windows.py so the SRS null can keep "
                         "real flanks and shuffle only the core.")
            offs = t13["core_offset"].astype(int).values
            srs_cols["sequence_context"] = [
                str(c)[:o] + sc + str(c)[o + len(sc):]
                for c, o, sc in zip(t13["sequence_context"].values, offs, shuffled)]
            srs_cols["core_offset"] = offs
        srs = pd.DataFrame(srs_cols)
        srs = srs[srs["sequence"].str.len() == args.window]      # guard: shuffle keeps length (no N in T13)
        df = pd.concat([df, srs], ignore_index=True)
        print(f"  regenerated {len(srs)} SRS nulls @ {args.window}bp"
              f"{' (+context: real flanks kept, core shuffled)' if has_ctx else ''} "
              f"from re-windowed T13 positives", file=sys.stderr)

    # 3. dedup exact sequences (prefer verified positive > silver positive > negative) -
    df["_rank"] = df["label"] * 2 + (df["weight"] >= GOLD).astype(int)
    df = (df.sort_values("_rank", ascending=False)
            .drop_duplicates("sequence").drop(columns="_rank").reset_index(drop=True))

    # 4. DECISION: balance to ~neg_per_pos by subsampling negatives --------------------
    if args.neg_per_pos > 0:
        n_pos = int((df.label == 1).sum())
        neg = df[df.label == 0]
        target = int(n_pos * args.neg_per_pos)
        if len(neg) > target:
            neg = neg.sample(target, random_state=args.seed)
        df = pd.concat([df[df.label == 1], neg], ignore_index=True)
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # 5. cluster near-duplicates ON THE 81 bp CORE (`sequence`) — the anti-leakage step -----
    #    Clustering on the core (not the long sequence_context) keeps the guarantee tight: two
    #    promoters with an identical -10/-35 core but different flanks cluster together (over a
    #    long window they would fall below the identity threshold and leak across splits). Backend
    #    auto-selects by identity (cd-hit-est >=0.8, MMseqs2 below — cd-hit-est cannot do DNA <~0.8).
    df["cluster_id"], tool_used = cluster_sequences(
        df["sequence"].tolist(), args.identity, args.cluster_tool,
        args.cluster_coverage, args.cov_mode, args.word_size, args.threads, args.out)

    # 6. cluster-disjoint, label-stratified 80/10/10 — EVAL (val/test) is VERIFIED-ONLY -----
    #    T13 silver positives carry INFERRED labels (the "AT-rich upstream = promoter" assumption);
    #    selecting / early-stopping on them would reward the AT-trap. So any CLUSTER holding a
    #    silver positive is pinned to TRAIN, and only verified-positive + negative clusters are eligible
    #    for val/test. Splitting at the cluster level keeps the leakage guarantee intact.
    #    ALSO pin CROSS-SPECIES (in_genus=False) positives to TRAIN: the target C. butyricum is
    #    in-genus, so val/test must measure IN-GENUS performance. Leaving E. coli/B. subtilis/Bacillus/
    #    Ruminiclostridium promoters in val/test distorts the deployment metric (and did so silently for
    #    T8 before this). So any cluster holding a silver OR cross-species positive is train-pinned;
    #    val/test hold only in-genus verified positives + negatives.
    train_pin = set(df.loc[(df.label == 1) &
                           ((df.source == "T13_inferred") | (~df["in_genus"].astype(bool))),
                           "cluster_id"])
    df["split"] = "train"
    elig = df[~df["cluster_id"].isin(train_pin)]
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=args.seed)
    fold = np.full(len(elig), -1, dtype=int)
    for k, (_, te) in enumerate(sgkf.split(elig, elig["label"], groups=elig["cluster_id"])):
        fold[te] = k
    df.loc[elig.index[fold == 8], "split"] = "val"      # eligible folds 0-7 stay train; 8->val, 9->test
    df.loc[elig.index[fold == 9], "split"] = "test"

    # 6b. CLASS-BALANCE the TRAIN weights so training is EFFECTIVELY 50/50 ---------------
    #     Down-weighting silver positives to 0.3 cuts positive *mass* below the negative count, skewing
    #     the effective TRAIN prior toward negatives (raw 50/50 -> effective ~36% positive). Fix: scale
    #     TRAIN's negative weights so sum(neg weights) == sum(pos weights). The `weight` column now carries
    #     BOTH label confidence (silver 0.3 vs verified 1.0) AND the class-balance factor, so a trainer
    #     applying it as a per-sample loss weight gets a ~50/50 prior for free. val/test weights are left
    #     at natural prevalence (eval honesty) — apply them only if you want a balanced eval metric.
    #     (Reweighting must preserve class balance: Constrained Reweighting, Google Research 2022.)
    tr = df["split"] == "train"
    pos_mass = float(df.loc[tr & (df.label == 1), "weight"].sum())
    trneg = tr & (df.label == 0)
    if int(trneg.sum()) and pos_mass > 0:
        df.loc[trneg, "weight"] = pos_mass / int(trneg.sum())

    # 7. leakage guard ----------------------------------------------------------------
    spanning = df.groupby("cluster_id")["split"].nunique()
    assert (spanning == 1).all(), f"LEAKAGE: {(spanning > 1).sum()} clusters span >1 split"

    # 8. write ------------------------------------------------------------------------
    #    Carry the long-context columns to the output too, if present, so the NTv3
    #    tokeniser (encode.py --sequence-col sequence_context) can read them.
    out_cols = OUT_COLS + [c for c in ("sequence_context", "core_offset") if c in df.columns]
    for sp in ("train", "val", "test"):
        df[df.split == sp][out_cols].to_parquet(os.path.join(args.out, f"{sp}.parquet"), index=False)

    # summary -------------------------------------------------------------------------
    print("\n=== T15 splits ===")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
    # positive composition per split: silver(T13) vs verified-in-genus vs verified-cross-species
    pos = df[df.label == 1].copy()
    pos["kind"] = np.where(pos.source == "T13_inferred", "silver(T13)",
                  np.where(pos.in_genus, "verified_in_genus", "verified_cross_species"))
    print("\npositives by split x kind:\n", pos.groupby(["split", "kind"]).size().unstack(fill_value=0))
    print("\nrows by source/label:\n", df.groupby(["source", "label"]).size())
    print(f"\nclusters={df.cluster_id.nunique()}  rows={len(df)}  window={args.window}bp  "
          f"identity={args.identity} (tool={tool_used})")
    print(f"positives total: in-genus={int(pos.in_genus.sum())}  out-of-genus={int((~pos.in_genus).sum())}  "
          f"silver(T13)={int((pos.source == 'T13_inferred').sum())}")
    # EVAL INVARIANT: val/test are verified-only — report the verified-in-genus count that actually matters
    for sp in ("val", "test"):
        m = df["split"] == sp
        nsilver = int((m & (df.label == 1) & (df.source == "T13_inferred")).sum())
        nverif_ig = int((m & (df.label == 1) & df.in_genus & (df.source != "T13_inferred")).sum())
        nverif_xs = int((m & (df.label == 1) & (~df.in_genus)).sum())
        npos = int((m & (df.label == 1)).sum()); nneg = int((m & (df.label == 0)).sum())
        print(f"{sp}: verified_in_genus={nverif_ig}  verified_cross_species={nverif_xs}  "
              f"silver={nsilver} (must be 0)  |  pos={npos} neg={nneg} ({100*npos/max(npos+nneg,1):.0f}% pos)")
        assert nsilver == 0, f"EVAL LEAK: {sp} has {nsilver} silver positives — val/test must be verified-only"
    # effective class balance (weights used AS sample weights) — TRAIN is rescaled to ~50/50
    print("\neffective class balance (sum of sample weights):")
    for sp in ("train", "val", "test"):
        m = df["split"] == sp
        pm = float(df.loc[m & (df.label == 1), "weight"].sum())
        nm = float(df.loc[m & (df.label == 0), "weight"].sum())
        print(f"  {sp:5}: pos_mass={pm:8.0f}  neg_mass={nm:8.0f}  -> {100*pm/max(pm+nm,1e-9):.0f}% positive")
    print("  (TRAIN balanced to ~50/50 via negative weights; val/test left at natural prevalence)")
    ctx_note = ("\n  NOTE: leakage is enforced on the 81bp CORE (`sequence`). The long `sequence_context` "
                "windows of genome-adjacent rows CAN overlap across splits, and each context-SRS shares "
                "real flanks with its train-pinned parent — acceptable for a first pass, but tighten "
                "(cluster/pin on context proximity) before trusting NTv3 val/test as leak-free."
                if "sequence_context" in df.columns else "")
    print("leakage check: PASS — no CORE cluster spans splits; val/test are verified-only" + ctx_note)


if __name__ == "__main__":
    main()
