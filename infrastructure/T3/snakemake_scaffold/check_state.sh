#!/bin/bash
# check_state.sh — report the pipeline data-state so we can see what's run and what's missing.
# results/ and inputs/ are BOTH git-ignored, so they do NOT transfer between clones — each machine
# regenerates them. This tells you exactly what THIS machine has.
#
# Run:  bash infrastructure/T3/snakemake_scaffold/check_state.sh
set -u
SC="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$SC/../../.." && pwd)"; cd "$SC"
sz() { [ -f "$1" ] && wc -c < "$1" | tr -d ' ' || echo MISSING; }

echo "REPO = $REPO"
echo
echo "=== git ==="; git -C "$REPO" log --oneline -1 2>/dev/null; git -C "$REPO" status --short 2>/dev/null | head
echo
echo "=== tools (need cd-hit-est + snakemake for T15) ==="
for t in cd-hit-est snakemake python; do printf "  %-12s " "$t:"; command -v "$t" 2>/dev/null || echo MISSING; done
echo
echo "=== Bakta zip (must be a real ~450MB file, NOT a ~130-byte LFS pointer) ==="
z="$REPO/data-acquisition-and-curation/bakta-annotations/all_bakta_output.zip"; printf "  all_bakta_output.zip: %s bytes\n" "$(sz "$z")"
echo
echo "=== Phase A inputs/  (0 bytes = EMPTY STUB -> T15 has no positives!) ==="
for f in reference_promoters.parquet published_tss_promoters.parquet literature_promoters_train.parquet tss_sites.gff3 literature_promoters.tsv; do
  printf "  %-38s %s bytes\n" "$f" "$(sz "inputs/$f")"
done
echo
echo "=== Phase B results/  (T11..T16) ==="
for t in T11 T12 T13 T14 T15 T16; do
  if [ -d "results/$t" ]; then printf "  results/%-4s %s file(s)\n" "$t" "$(ls "results/$t" 2>/dev/null | wc -l | tr -d ' ')";
  elif [ -f "results/$t.done" ]; then echo "  results/$t.done present";
  else echo "  results/$t  MISSING"; fi
done
echo
echo "=== VERDICT / next step ================================================="
have_pos=0; [ -s inputs/published_tss_promoters.parquet ] && [ -s inputs/reference_promoters.parquet ] && have_pos=1
have_b=0;   [ -s results/T13/upstream_regions.tsv ] && [ -s results/T14/negative_set.tsv ] && have_b=1
if [ "$have_pos" = 0 ]; then
  echo "  ✗ Phase A positives (T7b/T8) are EMPTY -> stage them first (see STEP 2A in Claude's message)."
elif [ "$have_b" = 0 ]; then
  echo "  ✗ Phase B (T13/T14) MISSING -> run:  snakemake --cores 4 results/T14/negative_set.tsv"
  echo "    then:                               snakemake --cores 4 results/T15.done"
else
  echo "  ✓ Positives + T13/T14 present -> run:  snakemake --cores 4 results/T15.done"
fi
echo "========================================================================="
