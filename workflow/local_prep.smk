# =====================================================================
# Phase A — light input staging plus optional T5/T6 regeneration rules.
#
# Produces the "starting state" under inputs/, uploaded to the cluster. The main
# Snakefile (Phase B onwards) consumes these; it does not build them.
#
#     snakemake -s local_prep.smk --cores 4          # light staging: T7, T7b, T8, T9
#
# Implemented commands below are not a record of a successful run on every machine.
#   T5  extract genomes      — optional (unzip committed curated zip -> results/genomes/; not in `all`)
#   T6  Bakta annotation     — REAL bakta loop; REGENERATE path (needs config[bakta_db], ~30GB).
#                              NOT in `rule all` — the committed all_bakta_output.zip is canonical
#                              and Phase B reads it via config[bakta_dir]. Run on demand:
#                                snakemake -s local_prep.smk results/bakta/all_bakta_output.zip
#   T7  dRNA-seq SRR list    — REAL (copies the curated accession list)
#   T8  RegulonDB->parquet   — run_t8_pipeline.py; reported 3,963 RegulonDB + 373 DBTBS = 4,336.
#                              Source evidence varies; not every database label is experimentally verified.
#   T9  literature promoters — build_literature_promoters.py; reported 93 short promoters.
#   T7b published in-genus TSS — build_published_tss.sh on committed Hocq workbook; reported 5,759.
#                                Needs ncbi-datasets-cli/internet; caught failures can yield partial/empty output.
#
# Optional T10 (TSSpredator) is separate -> tss_prep.smk; its output is not a deployed training positive.
# =====================================================================

import os
configfile: "config/config.yaml"   # load base config so t5/t6 see genomes_zip + bakta_db
REPO = os.path.abspath(os.path.join(workflow.basedir, ".."))

rule all:
    input:
        "inputs/SRR_dRNAseq_for_TSSpredator.txt",  # T7  -> input to T10 (tss_prep.smk)
        "inputs/reference_promoters.parquet",      # T8  -> REFDB
        "inputs/literature_promoters.tsv",         # T9  -> LIT_PROM (full curated set)
        "inputs/literature_promoters_train.parquet",  # T9-split (option b) -> in-genus training positives (T15)
        "inputs/published_tss_promoters.parquet",  # T7b -> in-genus positives -> training (T15)

# --- T5 / T6 : genome + annotation regenerate path (real; not in `all`) ---------
rule t5_extract_genomes:
    input: os.path.join(REPO, config["genomes_zip"])
    output: directory("results/genomes")
    shell:
        r"""
        rm -rf {output} results/_genraw && mkdir -p {output} results/_genraw
        unzip -q -o {input} 'ncbi_dataset/data/*/*_genomic.fna' -d results/_genraw
        find results/_genraw/ncbi_dataset/data -name '*_genomic.fna' -exec mv -t {output} {{}} +
        rm -rf results/_genraw
        n=$(ls {output}/*_genomic.fna | wc -l); echo "[T5] extracted $n genomes"; [ "$n" -eq 166 ]
        """
    # real (from-scratch provenance): datasets download genome taxon 1485 --reference
    #   --assembly-source RefSeq --include genome  (then MANUALLY drop DSM 10702; the committed
    #   curated zip already excludes it + is deterministic, so we unzip it instead). Needs `git lfs pull`.

rule t6_bakta_annotation:
    input: "results/genomes"
    output: "results/bakta/all_bakta_output.zip"
    threads: 8
    shell:
        r"""
        set -euo pipefail
        OUTROOT=results/bakta/all_bakta_output
        rm -rf "$OUTROOT" && mkdir -p "$OUTROOT"
        for FNA in {input}/*_genomic.fna; do
          PREFIX=$(basename "$FNA" .fna)                       # <ACCESSION>_<assembly>_genomic
          bakta --db {config[bakta_db]} --output "$OUTROOT/${{PREFIX}}_output" \
                --prefix "$PREFIX" --threads {threads} --force "$FNA"
        done
        cd results/bakta && zip -r -q all_bakta_output.zip all_bakta_output \
            -i 'all_bakta_output/*/*_genomic.gff3' 'all_bakta_output/*/*_genomic.fna' 'all_bakta_output/*/*_genomic.tsv'
        echo "[T6] re-annotated; wrote all_bakta_output.zip"
        """
    # Bakta 1.11.4 / DB v6.0 light. REGENERATE only — needs (a) the `bakta` binary (NOT in the
    # T4 envs; install a dedicated env: `conda create -n bakta -c conda-forge
    # -c bioconda bakta`) and (b) config[bakta_db] (~30GB, not in repo; `bakta_db download --type light`).
    # The committed pipeline/acquisition/bakta-annotations/all_bakta_output.zip is canonical (Phase B reads it
    # via config[bakta_dir]); point config[bakta_dir] at this output only if you re-annotate.

# --- T7 / T8 / T9 : the light inputs/ staging (real) ----------------------------
rule t7_drnaseq_lit_search:
    output: "inputs/SRR_dRNAseq_for_TSSpredator.txt"
    shell:
        "mkdir -p inputs && "
        "cp {REPO}/{config[srr_list]} {output}"
    # 20 dRNA-seq runs for TSSpredator (the 27 Capp-Switch/5'-RNA-seq runs use published tables, not TSSpredator).

rule t8_normalise_refdbs:
    output: "inputs/reference_promoters.parquet"
    shell:
        # run_t8_pipeline.py takes no args and uses a RELATIVE OUTPUT_DIR -> must run from repo root;
        # capture the absolute output path first, then cd. Writes 4,336 rows (3,963 RegulonDB from
        # PromoterS_Set.txt + 373 DBTBS from DBTBS_prom.fa, both committed).
        "OUT=$(pwd)/{output} && "
        "cd {REPO} && python pipeline/acquisition/curated-promoters/run_t8_pipeline.py && "
        "mkdir -p $(dirname $OUT) && "
        "cp pipeline/acquisition/curated-promoters/curated_positive_promoters.parquet $OUT"

rule t9_curate_synthetic_promoters:
    output: "inputs/literature_promoters.tsv"
    shell:
        # build_literature_promoters.py is __file__-relative + takes no args; it writes
        # pipeline/acquisition/synthetic-promoters/literature_promoters.tsv, which we stage into inputs/.
        "python {REPO}/pipeline/acquisition/synthetic-promoters/build_literature_promoters.py && "
        "mkdir -p inputs && "
        "cp {REPO}/pipeline/acquisition/synthetic-promoters/literature_promoters.tsv {output}"

rule t9_split_train:
    # Option b: split T9's VERIFIED in-genus C. acetobutylicum promoters into a TRAINING subset
    # (-> T15) + an eval HOLDOUT (-> T24/T26). Family-aware + seeded (split_t9_for_training.py).
    # Set config[t9_train_fraction]=0 to disable (T9 then stays 100% held out, the original behaviour).
    # NB: when enabled, T24/T26 must eval on literature_promoters_holdout.tsv, NOT the full T9.
    input: "inputs/literature_promoters.tsv"
    output:
        train="inputs/literature_promoters_train.parquet",
        holdout="inputs/literature_promoters_holdout.tsv",
    shell:
        "python {REPO}/pipeline/acquisition/synthetic-promoters/split_t9_for_training.py "
        "--in {input} --train-frac {config[t9_train_fraction]} --seed {config[random_seed]} "
        "--train-out {output.train} --holdout-out {output.holdout}"

rule t7b_published_tss:
    # Published-TSS-labelled in-genus positives from Hocq 2022 C. acetobutylicum/C. beijerinckii.
    # The wrapper also recognises an Al-Bassam C. ljungdahlii filename if supplied: do not stage it,
    # because those published TSS are benchmark truth. It stages the unused C. ljungdahlii
    # reference from the genome archive but reads the committed Hocq workbook for deployed sources.
    # C. acetobutylicum 1,434 + C. beijerinckii 4,325 = 5,759. Its genomes are fetched with
    # ncbi-datasets-cli; without it (or internet) those strains are skipped and the parquet can be
    # EMPTY while the pipeline still runs, so check the printed count. Do NOT stage the Al-Bassam
    # C. ljungdahlii table there: it would be ingested, and C. ljungdahlii is the validation genome.
    output: "inputs/published_tss_promoters.parquet"
    shell:
        "OUT=$(pwd)/{output} REPO={REPO} "
        "bash {REPO}/pipeline/acquisition/tss-literature/build_published_tss.sh"
