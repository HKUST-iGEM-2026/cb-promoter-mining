# =====================================================================
# Phase A (cluster part) — T10 TSSpredator.  Runs on the cluster, not locally.
#
# REAL pipeline now: the rule calls data-acquisition-and-curation/cdifficile-drnaseq/run_tss_pipeline.sh
# (6 stages: reference -> SRA download -> segemehl align -> 5' wiggle -> TSSpredator x2
# -> merge to inputs/tss_sites.gff3). See that script's header for the tool list and the
# 8 open decisions (D1-D8) the team must lock before a real run.
#
#     # after staging the local_prep outputs into inputs/:
#     #  1) on a node with internet (often the login node): stages 1-2 + fetch the TSSpredator JAR
#     bash <repo>/data-acquisition-and-curation/cdifficile-drnaseq/run_tss_pipeline.sh 1   # reference
#     bash <repo>/data-acquisition-and-curation/cdifficile-drnaseq/run_tss_pipeline.sh 2   # download reads
#     #  2) then submit the compute stages (3-6) via the profile:
#     snakemake -s tss_prep.smk --workflow-profile profiles/slurm
#
# CPU job (no GPU). The main Snakefile then consumes
# inputs/tss_sites.gff3 as its TSS_CALLS.
# =====================================================================

import os
configfile: "config/config.yaml"   # consistency with the main Snakefile (safe under --config overrides)
REPO = os.path.abspath(os.path.join(workflow.basedir, "..", "..", ".."))

rule all:
    input:
        "inputs/tss_sites.gff3",            # T10 -> main pipeline TSS_CALLS

rule t10_tsspredator:
    input:
        accessions="inputs/SRR_dRNAseq_for_TSSpredator.txt",   # from T7 (local_prep, uploaded)
    output: "inputs/tss_sites.gff3"
    resources: cpus_per_task=28, runtime=720
    shell:
        # The script writes {output} (OUT_GFF). REPO lets it find the .conf files + SRR list.
        # NB: stages 1-2 need login-node internet — run them first (see header); --all assumes
        # a node with internet + CPUs, which on many clusters is NOT the case, so prefer the
        # split invocation. C. difficile = Clostridioides (out of genus 1485): cross-species-adjacent.
        "OUT_GFF=$(pwd)/{output} REPO={REPO} "
        "bash {REPO}/data-acquisition-and-curation/cdifficile-drnaseq/run_tss_pipeline.sh --all"
    # real (regenerate the cdiff genome + wig tracks the .conf files reference, then run both
    #       fixed configs fuchs_config_hpc.conf + soutourina_tap_config.conf — see docs/PIPELINE.md)
