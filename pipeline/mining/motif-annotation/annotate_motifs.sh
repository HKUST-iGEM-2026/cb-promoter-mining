#!/bin/bash
# annotate_motifs.sh — T31 step 2. De-novo motif discovery (MEME) on the cleanest calls, then scan every
# call for those motifs (FIMO). Recovers the -10/-35 boxes and tells T32 which calls actually carry them.
#
# Needs the MEME suite -> run in the BIO env (environments/environment-bio.yml, meme>=5.5):
#   source "$(conda info --base)/etc/profile.d/conda.sh"
#   conda activate "$ENV_MEME"     # the env that has `meme` + `fimo` on PATH (docs/SETUP.md)
#   bash annotate_motifs.sh candidates_top.fasta candidates.fasta t31_out
#
# Sequences are already in reading orientation (extract_call_sequences.py revcomp'd the - strand), so
# MEME/FIMO search the GIVEN strand only — do NOT pass -revcomp, or a -10 could be matched on the wrong
# strand and land at a meaningless offset.
set -eo pipefail
TOP=${1:?arg1 = top-N FASTA (MEME learns from this)}
ALL=${2:?arg2 = all-calls FASTA (FIMO scans this)}
OUT=${3:?arg3 = output dir}
mkdir -p "$OUT"

command -v meme >/dev/null || { echo "meme not on PATH — activate the bio env (environment-bio.yml)"; exit 1; }
command -v fimo >/dev/null || { echo "fimo not on PATH — activate the bio env"; exit 1; }

# de novo: DNA, zero-or-one-occurrence-per-sequence, up to 5 motifs, -10/-35 box widths (6-18 bp).
echo "=== MEME (de novo motif on $(grep -c '^>' "$TOP") top calls) ==="
meme "$TOP" -oc "$OUT/meme" -dna -mod zoops -nmotifs 5 -minw 6 -maxw 18

# scan every call for the discovered motifs. --norc: sequences are ALREADY reading-strand
# (extract_call_sequences.py revcomp'd the - strand), so scan the + strand only — else a -10 can match
# on the antisense strand at a meaningless offset and inflate the architecture flags downstream.
echo "=== FIMO (scan all $(grep -c '^>' "$ALL") calls) ==="
fimo --norc --oc "$OUT/fimo" "$OUT/meme/meme.txt" "$ALL"

echo ""
echo "DONE."
echo "  motifs (inspect the logos): $OUT/meme/meme.html"
echo "  per-call hits (-> T32):      $OUT/fimo/fimo.tsv"
echo "Next: T32 build_candidate_table.py --calls calls_with_id.tsv --fimo $OUT/fimo/fimo.tsv ..."
