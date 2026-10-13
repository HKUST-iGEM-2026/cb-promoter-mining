#!/usr/bin/env bash
# =====================================================================
# T10 — real dRNA-seq -> TSSpredator pipeline (Clostridioides difficile 630)
#
# Reproduces TSS from the 20 dRNA-seq runs (Fuchs 2021 TEX x18 + Soutourina 2020
# TAP x2) and writes the pipeline target $OUT_GFF (inputs/tss_sites.gff3).
#
# Runs on a cluster. It is split into stages with different needs:
#   stages 1,2 + the JAR fetch  -> LOGIN node (need internet)
#   stages 3,4,5                -> COMPUTE CPU job (no internet needed)
#   stage 6                     -> trivial, anywhere
# Because compute nodes have no internet, run stages 1-2 (and the one-time JAR
# download) on the login node first, then submit stages 3-6 as a CPU job.
# Use --stage to run a single stage, or run with no args to do 1..6 in order
# (only do that on a node that has BOTH internet and enough CPUs — usually you won't).
#
# Tools (conda env: promoter-mining-bio): sra-tools, segemehl, samtools, bedtools,
# seqkit, openjdk. TSSpredator is a JAR (not conda) — fetched below. ncbi-datasets-cli
# is optional (see D1); a wget fallback is provided.
#
# -------------------- DECISIONS (resolved 2026-06) --------------------
#  D1  RESOLVED: ncbi-datasets-cli added to environment-bio.yml (stage 1 uses it; wget fallback kept).
#  D2  RESOLVED: 5' wiggle via bedtools genomecov -5 (in-env, positive-value wigs matching the configs).
#      READemption is the canonical alternative (documented in stage 4) if a TNOAR-normalized pipeline is preferred.
#  D3  RESOLVED: these are 5'-end dRNA-seq (TEX) + TAP libraries = SINGLE-END by protocol; fasterq-dump
#      --split-spot emits one <SRR>.fastq. (If a future run is paired, use read 1 as the sense read.)
#  D4  RESOLVED: cutadapt trims the 3' Illumina adapter + low-quality 3' bases BEFORE alignment (stage 3).
#      5' ends are left untouched (TSS precision preserved). CONFIRM the actual adapter for these SRA
#      submissions; AGATCGGAAGAGC (TruSeq) is the default below.
#  D5  RESOLVED: the .conf thresholds are the project's TSSpredator preset (preserved from the original
#      config: step height minCliffHeight=0.3, step factor minCliffFactor=2.0, enrichment min5primeToNormalFactor=2.0,
#      processing-site maxNormalTo5primeFactor=1.5, base height minNormalHeight=0.2, normPercentile=0.9).
#      The 170k OVER-CALL was NOT a threshold problem — it was numReplicates=1/minNumRepMatches=1; fixed to
#      numReplicates=3 + minNumRepMatches=2 (3-replicate reproducibility), which is the real over-call control.
#      Expect LOW THOUSANDS of TSS. If still high, confirm thresholds against the Fuchs 2021 Methods.
#  D6  RESOLVED: reverse strand written as POSITIVE magnitudes in *_reverse.wig (matches the configs' *Minus* slots).
#  D7  RESOLVED: feed RAW 5' counts; TSSpredator normalizes internally via normPercentile=0.9 (confirmed in its
#      Methods doc). No pre-normalization here -> no double-normalization.
#  D8  RESOLVED: stage-6 merge keeps enriched PRIMARY TSS reproducible in >=2 reps (TAP set >=1).
#      NB: C. difficile is OUT OF GENUS (taxon 272563, not 1485) -> cross-species-adjacent support; the in-genus
#      empirical positives come from the published-table path (T7b). Revisit the filter if T10 is ever promoted to a training source.
#
# !! SMOKE TEST FIRST: run one SRR (e.g. SRR11573852 TAP+) through stages 2-5 and eyeball that
#    <SRR>_forward.wig has a 5' spike just upstream of a known + strand gene before the full 20-run launch.
# =====================================================================
set -euo pipefail

# ---- config (override via env) ----
# WORK holds the reference, reads, BAMs and wiggles (tens of GB). The .conf files write it as
# <TSS_WORK>; stage 5 fills that in, so the configs never need editing.
REPO="${REPO:-${PROMOTER_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}}"
WORK="${WORK:-$REPO/workflow/results/T10_work}"
SRR_LIST="${SRR_LIST:-$REPO/pipeline/acquisition/tss-literature/SRR_dRNAseq_for_TSSpredator.txt}"
FUCHS_CONF="${FUCHS_CONF:-$REPO/pipeline/acquisition/cdifficile-drnaseq/fuchs_config_hpc.conf}"
TAP_CONF="${TAP_CONF:-$REPO/pipeline/acquisition/cdifficile-drnaseq/soutourina_tap_config.conf}"
OUT_GFF="${OUT_GFF:-$REPO/workflow/inputs/tss_sites.gff3}"
THREADS="${THREADS:-28}"
CHROM="${CHROM:-NC_009089.1}"          # C. difficile 630 chromosome (drop plasmid NC_008226.1)
ACC="${ACC:-GCF_000009205.2}"

REF_FNA="$WORK/cdiff_chrom_only.fna"   # config genome_N
REF_GFF="$WORK/cdiff_chrom_only.gff"   # config annotation_N
WIGDIR="$WORK/wig_chrom_only"          # config fivePrime*/normal* live here
IDX="$WORK/cdiff_chrom_only.idx"
BAMDIR="$WORK/bam"; FASTQDIR="$WORK/fastq"
JAR="${TSSPREDATOR_JAR:-$WORK/TSSpredator-1.1beta.jar}"
mkdir -p "$WORK" "$WIGDIR" "$BAMDIR" "$FASTQDIR"

# ---------------------------------------------------------------------
stage1_reference() {   # LOGIN node (internet)
  cd "$WORK"
  if command -v datasets >/dev/null 2>&1; then
    datasets download genome accession "$ACC" --include genome,gff3 --filename cdiff_630.zip
    unzip -o cdiff_630.zip -d cdiff_630
    RAW_FNA=$(ls cdiff_630/ncbi_dataset/data/"$ACC"/*_genomic.fna)
    RAW_GFF=cdiff_630/ncbi_dataset/data/"$ACC"/genomic.gff
  else   # D1 fallback: RefSeq FTP
    wget -O cdiff.fna.gz "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/009/205/${ACC}_ASM920v2/${ACC}_ASM920v2_genomic.fna.gz"
    wget -O cdiff.gff.gz "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/009/205/${ACC}_ASM920v2/${ACC}_ASM920v2_genomic.gff.gz"
    gunzip -f cdiff.fna.gz cdiff.gff.gz; RAW_FNA=cdiff.fna; RAW_GFF=cdiff.gff
  fi
  seqkit grep -p "$CHROM" "$RAW_FNA" > "$REF_FNA"                       # chromosome only
  awk -v c="$CHROM" '/^#/{print;next} $1==c{print}' "$RAW_GFF" > "$REF_GFF"
  [ "$(grep -c '^>' "$REF_FNA")" -eq 1 ] || { echo "ERR: REF_FNA not single-contig"; exit 1; }
  echo "[T10.1] reference ready: $REF_FNA ($CHROM)"
}

stage2_download() {    # LOGIN node (internet)
  cd "$FASTQDIR"
  while read -r SRR; do [ -z "$SRR" ] && continue
    prefetch "$SRR" --max-size 50g -O "$FASTQDIR"
    fasterq-dump "$SRR" --threads "$THREADS" --split-spot --skip-technical --outdir "$FASTQDIR"
    gzip -f "$FASTQDIR/${SRR}.fastq"
    rm -rf "$FASTQDIR/$SRR"   # drop the .sra prefetch cache
  done < "$SRR_LIST"
  echo "[T10.2] reads downloaded"
}

stage3_align() {       # COMPUTE (no internet) — D4 trim, then align
  [ -f "$IDX" ] || segemehl.x -x "$IDX" -d "$REF_FNA"
  mkdir -p "$FASTQDIR/trimmed"
  while read -r SRR; do [ -z "$SRR" ] && continue
    # D4: trim the 3' Illumina adapter + low-quality 3' bases. 5' ends (the TSS) are NOT trimmed.
    cutadapt -a "${ADAPTER:-AGATCGGAAGAGC}" -q 20 -m 12 -j "$THREADS" \
      -o "$FASTQDIR/trimmed/${SRR}.trimmed.fastq.gz" "$FASTQDIR/${SRR}.fastq.gz"
    segemehl.x --silent -i "$IDX" -d "$REF_FNA" -q "$FASTQDIR/trimmed/${SRR}.trimmed.fastq.gz" -t "$THREADS" -A 90 \
      | samtools sort -@ 4 -o "$BAMDIR/${SRR}.sorted.bam" -
    samtools index "$BAMDIR/${SRR}.sorted.bam"
  done < "$SRR_LIST"
  echo "[T10.3] trimmed (cutadapt) + aligned (segemehl)"
}

stage4_wiggle() {      # COMPUTE (no internet) — 5'-END coverage, the TSS signal
  while read -r SRR; do [ -z "$SRR" ] && continue
    BAM="$BAMDIR/${SRR}.sorted.bam"
    bedtools genomecov -ibam "$BAM" -5 -strand + -bg \
      | awk -v c="$CHROM" 'BEGIN{print "track type=wiggle_0";print "variableStep chrom="c} $1==c{for(p=$2+1;p<=$3;p++)print p"\t"$4}' \
      > "$WIGDIR/${SRR}_forward.wig"
    bedtools genomecov -ibam "$BAM" -5 -strand - -bg \
      | awk -v c="$CHROM" 'BEGIN{print "track type=wiggle_0";print "variableStep chrom="c} $1==c{for(p=$2+1;p<=$3;p++)print p"\t"$4}' \
      > "$WIGDIR/${SRR}_reverse.wig"
  done < "$SRR_LIST"
  echo "[T10.4] 5' wiggle tracks written to $WIGDIR"
}

stage5_tsspredator() { # COMPUTE (no internet)
  [ -f "$JAR" ] || { echo "ERR: TSSpredator JAR missing at $JAR — fetch on the login node:"; \
    echo "  wget -O $JAR https://uni-tuebingen.de/fileadmin/Uni_Tuebingen/Fakultaeten/InfoKogni/WSI/IntegTranskript/Softwareprojekte/TSSPredator/TSSpredator-1.1beta.jar"; exit 1; }
  cd "$WORK"
  for CONF in "$FUCHS_CONF" "$TAP_CONF"; do   # -> result_fuchs/, result_tap/
    sed "s#<TSS_WORK>#$WORK#g" "$CONF" > "$WORK/run_$(basename "$CONF")"
    java -Xmx8g -jar "$JAR" "$WORK/run_$(basename "$CONF")"
  done
  echo "[T10.5] TSSpredator done (expect LOW THOUSANDS of TSS — see D5)"
}

stage6_merge() {       # trivial
  mkdir -p "$(dirname "$OUT_GFF")"
  { echo "##gff-version 3"
    for RES in "$WORK/result_fuchs" "$WORK/result_tap"; do
      MT="$RES/MasterTable.tsv"; [ -f "$MT" ] || { echo "WARN: missing $MT" >&2; continue; }
      awk -F'\t' -v src="$(basename "$RES")" -v chrom="$CHROM" '
        NR==1{for(i=1;i<=NF;i++)h[$i]=i; next}
        { prim=$(h["Primary"]); det=$(h["detCount"]); enr=$(h["enriched"]); pos=$(h["Pos"]); str=$(h["Strand"]);
          if(prim==1 && enr==1 && (det>=2 || src=="result_tap")){ key=pos"\t"str;
            if(!(key in seen)){ seen[key]=1;
              print chrom"\tTSSpredator\tTSS\t"pos"\t"pos"\t"$(h["enrichmentFactor"])"\t"str"\t.\tID=TSS_"src"_"pos"_"str";detCount="det";source="src } } }' "$MT"
    done
  } | sort -k1,1 -k4,4n > "$OUT_GFF"
  echo "[T10.6] wrote $OUT_GFF ($(grep -vc '^#' "$OUT_GFF") TSS)"
}

case "${1:---all}" in
  --all)   stage1_reference; stage2_download; stage3_align; stage4_wiggle; stage5_tsspredator; stage6_merge ;;
  1|--ref)         stage1_reference ;;
  2|--download)    stage2_download ;;
  3|--align)       stage3_align ;;
  4|--wiggle)      stage4_wiggle ;;
  5|--tsspredator) stage5_tsspredator ;;
  6|--merge)       stage6_merge ;;
  *) echo "usage: $0 [--all | 1=ref 2=download 3=align 4=wiggle 5=tsspredator 6=merge]"; exit 2 ;;
esac
