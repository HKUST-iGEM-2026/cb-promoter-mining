# =====================================================================
# Phase A — LOCAL one-shot prep (MOCK).  NOT an HPC4 job.
#
# Run on your own machine. Produces the "starting state" under inputs/,
# which you then upload to HPC4. The main Snakefile (Phase B onwards)
# consumes these files; it does not build them. See context §3.
# (T5 and T6 are already done for real — this mirrors the data flow so
# the mock pipeline has its inputs.)
#
#     snakemake -s local_prep.smk --cores 4
#
# Each `shell:` is a mock; the `# real:` comment shows the command to use.
# T7/T8/T9 are largely manual curation tasks.
# =====================================================================

rule all:
    input:
        "inputs/annotations.gff3",          # T6  -> main pipeline ANNOTATIONS
        "inputs/reference_promoters.parquet",  # T8  -> REFDB
        "inputs/literature_promoters.tsv",  # T9  -> LIT_PROM
        "inputs/tss_sites.gff3",            # T10 -> TSS_CALLS

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
    output: "inputs/drnaseq_accessions.csv"
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

rule t10_tsspredator:
    input: "inputs/annotations.gff3", "inputs/drnaseq_accessions.csv"
    output: "inputs/tss_sites.gff3"
    shell: "echo '[T10] re-process dRNA-seq with TSSpredator (mock)' && touch {output}"
    # real: TSSpredator config.ini   (needs genome FASTA + {input[0]} GFF + wiggle files from
    #       aligning the {input[1]} reads against the Bakta .fna; then export TSS to {output})
