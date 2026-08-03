# T26 negative extraction — notes, concerns, and comparison

`extract_negatives.py` builds two label=0 sets from the C. acetobutylicum and
"C." phytofermentans FASTA+GFF3 pairs:

- **convergent_intergenic** (`*_CIG_*`): 52-bp windows from `CDS(+) → intergenic ← CDS(-)`
  gaps, trimmed 50 bp at each CDS-facing boundary, with any base overlapping *any* annotated
  feature removed before windowing.
- **CDS_interior** (`*_CDS_*`): 52-bp windows inside CDS after trimming 50 bp off both ends,
  oriented in the CDS direction (revcomp on `-`).

Coordinates in the TSVs are **1-based inclusive** (GFF3 convention). Windows are the maximum
strictly non-overlapping tiling of each usable interval with a single seeded random phase;
no cap by default. Duplicates dropped per file. `region_counts.tsv` records
#convergent-regions and #CDS-regions per genome.

## Validation done (synthetic genome pair)
Masking fragments a gap around an interior tRNA; a planted N-run correctly drops the window
that overlaps it (`convergent_unsupported_char`); a 5-column GFF line is skipped
(`gff_malformed`); a feature on a replicon absent from the FASTA is warned + skipped; a
plasmid replicon with no features is noted. Re-runs are byte-identical (determinism);
`-` CDS sequences equal `revcomp(reference slice)`; zero intra-interval overlaps; all windows
inside trimmed bounds; every sequence exactly 52 bp / ACGT-only.

## Concerns
1. **52 bp vs the models' 81 bp.** T15/T16 windowed everything to **81 bp** and T18–T20 were
   trained/evaluated at 81. Scoring 52-bp windows for calibration feeds the models a shorter
   context than they saw in training, which can shift the score distribution you're trying to
   calibrate. **Strongly recommend `--window 81`** unless T26 has a specific reason for 52.
   The script takes `--window`, so this is a one-flag change. (Left the default at your
   requested 52; flag the mismatch before trusting calibrated q-values.)
2. **"phytofermentans" is out-of-genus** (Lachnoclostridium). Its AT bias differs from the
   in-genus target, so in T26 keep it as a *separate* covariate stratum rather than pooling —
   otherwise the AT-vs-score fit is an average over two regimes. `source_strain` lets you split.
3. **Convergent gaps are scarce in compact AT-rich genomes.** Expect far fewer CIG windows
   than CDS windows. If CIG count is too low to fit an FDR null, consider adding tandem/
   divergent intergenic *with a promoter-side buffer* (see advice), or lean on SRS (T14) as the
   primary null and use CIG as a real-genomic sanity check.
4. **Putative, not confirmed, promoter-free.** Convergent orientation makes a promoter
   unlikely but antisense/internal TSS exist. Documented in the file and in `negative_type`;
   don't over-claim in the writeup.
5. **Dedup is per-file, not cross-file.** A sequence could in principle appear in both CIG and
   CDS (extremely unlikely at 52 bp). If the FDR null pools both, dedup across them first.

## Comparison to prior negative generation (T14 `negative_set_generation.py`)
| | T14 | this script |
|--|--|--|
| window | 300 bp (later 81 in T15) | 52 (`--window`) |
| convergent def. | same `+ → −` idea | same, **plus** removes ALL overlapping features from the gap |
| boundary buffer | none | **50 bp trim** off CDS-facing ends |
| windows/interval | one random window | **all** non-overlapping (quantify available data) |
| CDS negatives | one window, start+100..stop-100 | all non-overlapping in trimmed interior, **strand-oriented** |
| dedup | none | yes (per file) |
| seed | added late | seeded from the start |
| dependencies | Biopython/pandas | **stdlib only** (avoids the `~/.local` numpy/pandas gotcha, CLAUDE.md §9) |

Net: stricter and more reproducible than T14, at a different (shorter) window. The masking +
trim make the convergent set cleaner than T14's raw gap sampling.

## Better advice (beyond the brief)
- **Use 81 bp** (concern 1) — the single most impactful change for calibration validity.
- **Stratify by species** in the T26 regressor (concern 2).
- If more negatives are needed, add divergent/tandem intergenic **but buffer the gene-start
  side by ≥50–100 bp** (that side carries the real promoter); the convergent set needs no such
  asymmetric buffer, which is exactly why it's the clean default.
- Keep C. acetobutylicum's plasmid (pSOL1) — `replicon_type` captures it; useful for checking
  chromosome-vs-plasmid AT effects.
