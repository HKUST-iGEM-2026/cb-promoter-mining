# =====================================================================
# Phase A (LOCAL part) — T5-T9 one-shot prep (MOCK).  Run on your own machine.
#
# Produces the "starting state" under inputs/, which you then upload to HPC4.
# The main Snakefile (Phase B onwards) consumes these files; it does not build them.
#
# NOTE: T10 (TSSpredator) is NOT here. It is CPU-heavy and runs on HPC4 (pulls
# ~36 GB of reads onto the cluster + aligns), so it lives in tss_prep.smk and is
# run with the SLURM profile. See context §3.
# (T5/T6 are already done for real — this mirrors the data flow for the mock.)
#
#     snakemake -s local_prep.smk --cores 4
#
# Each `shell:` is a mock; the `# real:` comment shows the command to use.
# T7/T8/T9 are largely manual curation tasks.
# =====================================================================

rule all:
    input:
        "inputs/annotations.gff3",             # T6  -> ANNOTATIONS (also input to T10 on HPC4)
        "inputs/SRR_dRNAseq_for_TSSpredator.txt",  # T7  -> input to T10 on HPC4 (tss_prep.smk)
        "inputs/reference_promoters.parquet",  # T8  -> REFDB
        "inputs/literature_promoters.tsv",     # T9  -> LIT_PROM (held out -> eval, T24/T26)
        "inputs/published_tss_promoters.parquet",  # T7b -> in-genus published TSS positives -> training (T15)

rule t5_download_genomes:
    output: "inputs/genomes.fna"
    shell: "echo '[T5] download Clostridium genomes (mock, done locally)' && touch {output}"
    # real: datasets download genome taxon 1485 --reference --assembly-source RefSeq --include genome
    #       (then unzip + concatenate the .fna into {output}; verify with md5sum -c)

rule t6_bakta_annotation:
    input: "inputs/genomes.fna"
    output: "inputs/annotations.gff3"
    shell: "echo '[T6] re-annotate with Bakta (mock, done locally)' && touch {output}"
    # real: bakta --db <bakta_db> --output inputs/bakta {input}
    #       (collect the per-genome .gff3 into {output})

rule t7_drnaseq_lit_search:
    output: "inputs/SRR_dRNAseq_for_TSSpredator.txt"
    shell: "echo '[T7] dRNA-seq literature search (mock, manual)' && touch {output}"
    # real: (manual) compile published Clostridium dRNA-seq SRA/ENA accessions + citations into {output}

rule t8_normalise_refdbs:
    output: "inputs/reference_promoters.parquet"
    shell: "echo '[T8] normalise RegulonDB+DBTBS to Parquet (mock)' && touch {output}"
    # real: python scripts/normalise_refdbs.py --regulondb <regulondb_dir> --dbtbs <dbtbs_dir> -o {output}

rule t9_curate_synthetic_promoters:
    output: "inputs/literature_promoters.tsv"
    shell: "echo '[T9] curate synthetic promoters (mock, manual)' && touch {output}"
    # real: (manual) curate known Clostridium synthetic promoters from the literature into {output}

rule t7b_published_tss:
    # In-genus VERIFIED promoter positives from published TSS tables (Hocq 2022 C. acetobutylicum,
    # Al-Bassam 2018 C. ljungdahlii) — the §7 fix for "no in-genus positive in training".
    output: "inputs/published_tss_promoters.parquet"
    shell: "echo '[T7b] in-genus published TSS -> 81bp promoter positives (mock)' && touch {output}"
    # real: python <repo>/data-acquisition-and-curation/T7/extract_published_tss_template.py \
    #         --tss-table <hocq2022_cac_tss.tsv> --genome <GCF_000008765.1.fna> \
    #         --species "Clostridium acetobutylicum" --source Hocq2022 \
    #         --pos-col <col> --strand-col <col> --out {output}
    #       (repeat for Al-Bassam 2018 C. ljungdahlii on GCF_000143685.1; concat the two parquets)
