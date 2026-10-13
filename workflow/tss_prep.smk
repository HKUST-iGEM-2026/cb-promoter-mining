# =====================================================================
# Optional raw dRNA-seq track — T10 TSSpredator (CPU, promoter-mining-bio environment).
#
# This rule calls run_tss_pipeline.sh --all: reference, SRA download, alignment,
# 5' wiggle, two TSSpredator configs and merge. The script records D1-D8 choices,
# but adapter/assay/reference assumptions still need confirmation for a new run.
#
# Activate promoter-mining-bio before invoking the shell stages or this workflow.
# The current rule does not activate it and --all unconditionally repeats stages 1-2.
# Full-rule submission therefore needs BOTH internet access and suitable CPU resources:
#     snakemake -s tss_prep.smk --workflow-profile profiles/slurm
#
# For offline compute: on a connected host, run positional stages 1 and 2 and stage the
# TSSpredator JAR; then run positional stages 3, 4, 5, 6 explicitly inside a CPU allocation.
# Preserve the same shared WORK, SRR_LIST, reference/reads, JAR and OUT_GFF across shells.
#     bash <repo>/pipeline/acquisition/cdifficile-drnaseq/run_tss_pipeline.sh 3
# Repeat with 4, 5, 6 after each preceding stage completes; this rule is NOT that split-mode launcher.
#
# The main legacy Snakefile requires inputs/tss_sites.gff3 as TSS_CALLS, but does not pass
# it to the deployed T15_xs positive set. C. difficile is Clostridioides, outside genus1485.
# =====================================================================

import os
configfile: "config/config.yaml"   # consistency with the main Snakefile (safe under --config overrides)
REPO = os.path.abspath(os.path.join(workflow.basedir, ".."))

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
        # --all requires internet for stages1-2 even when reads/reference were staged earlier.
        # For offline compute, use explicit positional stages3-6 in a suitable CPU allocation;
        # the header describes that separate invocation, not this full-rule shell command.
        "OUT_GFF=$(pwd)/{output} REPO={REPO} "
        "bash {REPO}/pipeline/acquisition/cdifficile-drnaseq/run_tss_pipeline.sh --all"
    # real (regenerate the cdiff genome + wig tracks the .conf files reference, then run both
    #       fixed configs fuchs_config_hpc.conf + soutourina_tap_config.conf — see docs/PIPELINE.md)
