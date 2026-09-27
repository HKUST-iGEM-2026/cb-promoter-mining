#!/bin/bash
# prep_g131795.sh — login-node staging for the C. butyricum DSM 10702 COMPLETE-genome re-mine.
#
# GCF_014131795.1 (ASM1413179v1) is the SAME target strain as g409755 (DSM 10702) — but a COMPLETE,
# reference-grade assembly (Flye 2.4; PacBio + Nanopore + Illumina MiSeq, 361x; "assembly from type
# material", RefSeq Reference Genome). Four replicons:
#     NZ_CP040626.1   chromosome 1     3,921,278 bp
#     NZ_CP040627.1   chromosome 2       770,199 bp
#     NZ_CP040628.1   plasmid pCB_1        6,059 bp
#     NZ_CP040629.1   plasmid pCB_2        8,060 bp    (total ~4,705,596 bp)
# vs g409755 = GCF_000409755.1 = the 96-contig NZ_AQQF01000... draft WGS. This re-mine puts the Stage-1
# deployment mine on the complete assembly, and is the natural CANONICAL target assembly for T26/T27 too
# (the draft/4-contig mismatch is why the T27 q-values didn't join the candidate tiers; see
# docs/PIPELINE.md).
#
# TARGET MINE => NO truth (the target has no known TSS). This fetches genome+gff and builds the
# restrict-scan candidate regions. Run on the LOGIN node (needs internet for `datasets`); fast (<2 min).
# Then submit mine_3model.sbatch — see docs/PIPELINE.md.
#
#   source ${PROMOTER_REPO}/env.sh
#   conda activate ${ENV_MAIN}     # has ncbi-datasets-cli
#   bash mining-and-motif-annotation/genome-scan/prep_g131795.sh
set -eo pipefail

REPO="${PROMOTER_REPO:?set it or source env.sh}"
EVAL=$REPO/evaluation-and-calibration
RES=$REPO/infrastructure/T3/snakemake_scaffold/results
OUT=$RES/regimeC/mine3_g131795       # == mine_3model.sbatch's OUT for TAG=g131795 (tracks land here too)
ACC=GCF_014131795.1
mkdir -p "$OUT" && cd "$OUT"

# 1) fetch genome + gff (RefSeq complete assembly). record ids = the NZ_CP0406xx.1 accessions.
if [[ ! -f g131795.fna ]]; then
    datasets download genome accession "$ACC" --include genome,gff3 --filename "g131795_${ACC}.zip"
    unzip -o "g131795_${ACC}.zip" -d g131795_dl
    cp "$(find g131795_dl -name '*.fna' | head -1)" g131795.fna
    cp "$(find g131795_dl \( -name '*.gff' -o -name '*.gff3' \) | head -1)" g131795.gff3
fi

# 2) sanity: expect 4 RefSeq records (2 chromosomes + 2 plasmids), ~4,705,596 bp total.
echo "=== FASTA records (id  length) ==="
awk '/^>/{if(n)print id"\t"n; id=substr($1,2); n=0; next}{n+=length($0)} END{if(n)print id"\t"n}' g131795.fna
NREC=$(grep -c '^>' g131795.fna)
LEN=$(awk '!/^>/{n+=length($0)} END{print n}' g131795.fna)
echo "records: $NREC   total: $LEN bp   (expect 4 records, ~4,705,596 bp)"
[[ "$NREC" -ge 2 ]] || { echo "ERROR: expected >=2 records, got $NREC — bad download?"; exit 1; }

# 3) candidate regions (restrict-scan search space): 300 bp upstream / 50 bp into gene, per strand.
#    MULTI-REPLICON => NO --contig: make_candidate_regions.py keys on the GFF seqids, which already match
#    the RefSeq FASTA headers (same download). (--contig is only for single-replicon Bakta renames.)
python "$EVAL/make_candidate_regions.py" --gff g131795.gff3 \
    --up 300 --down 50 --out g131795_candidate_regions.bed

# 4) contig-id parity — the #1 silent-miss failure. Every regions contig MUST be a FASTA record id.
FASTA_IDS=$(grep '^>' g131795.fna | sed 's/^>//; s/[[:space:]].*//' | sort -u)
REG_IDS=$(tail -n +2 g131795_candidate_regions.bed | cut -f1 | sort -u)
MISSING=$(comm -13 <(echo "$FASTA_IDS") <(echo "$REG_IDS") || true)
if [[ -n "$MISSING" ]]; then
    echo "ERROR: regions reference contig(s) absent from the FASTA (contig-id mismatch):"
    echo "$MISSING"; exit 1
fi
echo "contig-id parity OK: $(echo "$REG_IDS" | grep -c .) region contigs, all present in the FASTA."

echo ""
echo "DONE."
echo "  FASTA   = $OUT/g131795.fna"
echo "  regions = $OUT/g131795_candidate_regions.bed"
echo ""
echo "Next: submit the 3-model mine in TARGET-MINE mode (TRUTH=\"\" -> shortlist, no benchmark):"
echo "  TAG=g131795 TRUTH=\"\" MODELS=\"prokbert cnn_v2 evo2\" STEP=10 \\"
echo "    FASTA=$OUT/g131795.fna \\"
echo "    REGIONS=$OUT/g131795_candidate_regions.bed \\"
echo "    sbatch $REPO/mining-and-motif-annotation/genome-scan/mine_3model.sbatch"
