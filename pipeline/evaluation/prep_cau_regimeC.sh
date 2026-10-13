#!/bin/bash
# prep_cau_regimeC.sh — login-node prep for the C. autoethanogenum 2nd-validation-genome Regime C scan.
#
# C. autoethanogenum DSM 10061 (GCF_000484505.1, chromosome CP006763 / RefSeq NC_022592.1, 4,352,205 bp)
# is in-genus (taxon 1485) but ~99% ANI to the held-out C. ljungdahlii — so it was deliberately kept OUT
# of training and is used here as a SECOND held-out validation genome (a de-noised consistency check on
# the ljungdahlii genome-scale result; its 2,885 TSS give ~289 detected at recall 0.1 vs ljungdahlii's ~125).
#
# This fetches the RefSeq assembly, stamps the FASTA record id into the TSS truth (the tss file is only
# pos+strand), and builds the restrict-scan candidate regions. Run on the LOGIN node (needs internet for
# `datasets`); fast (<2 min). Then submit C_cau_scan.sbatch.
#
#   source ${PROMOTER_REPO}/env.sh
#   conda activate ${ENV_MAIN}    # has ncbi-datasets-cli
#   bash pipeline/evaluation/prep_cau_regimeC.sh
set -eo pipefail

REPO="${PROMOTER_REPO:?set it or source env.sh}"
EVAL=$REPO/pipeline/evaluation
RES=$REPO/workflow/results
OUT=$RES/regimeC
TSS=$REPO/pipeline/acquisition/additional-published-tss/autoethanogenum_tss.tsv   # pos,strand (2,885 primary TSS, Class==P)
ACC=GCF_000484505.1
mkdir -p "$OUT" && cd "$OUT"

# 1) fetch genome + gff (RefSeq). record id = the RefSeq chromosome accession (expect NC_022592.1).
if [[ ! -f cau.fna ]]; then
    datasets download genome accession "$ACC" --include genome,gff3 --filename "cau_${ACC}.zip"
    unzip -o "cau_${ACC}.zip" -d cau_dl
    cp "$(find cau_dl -name '*.fna' | head -1)" cau.fna
    cp "$(find cau_dl \( -name '*.gff' -o -name '*.gff3' \) | head -1)" cau.gff3
fi
REC=$(grep '^>' cau.fna | head -1 | sed 's/^>//; s/ .*//')
LEN=$(awk '!/^>/{n+=length($0)} END{print n}' cau.fna)
echo "record id : $REC"
echo "genome len: $LEN bp   (expect ~4,352,205)"

# 2) stamp contig into the truth:  pos,strand  ->  contig,pos,strand
#    sanity: max TSS pos must fall inside the genome, else the coords/assembly disagree.
MAXP=$(awk -F'\t' 'NR>1{if($1>m)m=$1} END{print m}' "$TSS")
[[ "$MAXP" -lt "$LEN" ]] || { echo "ERROR: max TSS pos $MAXP >= genome len $LEN — wrong assembly?"; exit 1; }
awk -v c="$REC" 'BEGIN{OFS="\t"} NR==1{print "contig","pos","strand"; next} {print c,$1,$2}' "$TSS" > cau_truth.tsv
echo "truth     : $(($(wc -l < cau_truth.tsv)-1)) TSS -> $OUT/cau_truth.tsv  (contig=$REC, max pos $MAXP)"

# 3) candidate regions (restrict-scan search space): 300 bp upstream / 50 bp into gene, per strand.
python "$EVAL/make_candidate_regions.py" --gff cau.gff3 --contig "$REC" \
    --up 300 --down 50 --out cau_candidate_regions.bed

echo ""
echo "DONE."
echo "  FASTA   = $OUT/cau.fna"
echo "  truth   = $OUT/cau_truth.tsv   (record $REC)"
echo "  regions = $OUT/cau_candidate_regions.bed"
echo "Next: sbatch $EVAL/C_cau_scan.sbatch"
