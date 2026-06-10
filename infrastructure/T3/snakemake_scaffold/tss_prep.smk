# =====================================================================
# Phase A (HPC4 part) — T10 TSSpredator (MOCK).  Runs ON HPC4, not locally.
#
# T10 is CPU-heavy: it pulls ~36 GB of dRNA-seq reads straight onto the cluster,
# aligns them, then runs TSSpredator. Only the small TSS table flows downstream,
# so its output is pre-staged like a Phase A input. Its inputs come from
# local_prep.smk (T6 annotations, T7 accessions), uploaded to HPC4 first.
#
#     # on HPC4, after uploading the local_prep outputs into inputs/:
#     snakemake -s tss_prep.smk --workflow-profile profiles/slurm
#
# CPU job on the preempt partition (no GPU). The main Snakefile then consumes
# inputs/tss_sites.gff3 as its TSS_CALLS.
# =====================================================================

rule all:
    input:
        "inputs/tss_sites.gff3",            # T10 -> main pipeline TSS_CALLS

rule t10_tsspredator:
    input:
        annotations="inputs/annotations.gff3",       # from T6 (local_prep, uploaded)
        accessions="inputs/drnaseq_accessions.csv",  # from T7 (local_prep, uploaded)
    output: "inputs/tss_sites.gff3"
    resources: cpus_per_task=8, runtime=240
    shell: "echo '[T10] re-process dRNA-seq with TSSpredator on HPC4 (mock)' && touch {output}"
    # real (HPC4): prefetch --option-file <repo>/data-acquisition-and-curation/T7/SRR_dRNAseq_for_TSSpredator.txt -O reads/
    #              fasterq-dump --split-3 reads/*/*.sra -O reads/
    #              <align reads/ to the Bakta .fna -> TEX+/TEX- wiggle tracks; aligner TBC with Sine> (to be changed later)
    #              TSSpredator <config>    # -> {output}
