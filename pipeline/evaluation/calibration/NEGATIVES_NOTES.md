# Calibration negatives

extract_negatives.py samples putative negatives from C. acetobutylicum and Lachnoclostridium phytofermentans FASTA/GFF3 pairs. Default windows are **52 bp**; coordinates are 1-based inclusive.

- **Convergent intergenic:** +→gap←− CDS boundaries, 50-bp trims and removal of every overlapping annotated feature.
- **CDS interior:** 50-bp trims at both ends, oriented in the CDS direction.

Each usable interval gets a seeded phase and maximum non-overlapping tiling, with no default cap. Deduplication is per file, not across the pooled cohort. region_counts.tsv records region/window counts and replicon_type retains chromosome/plasmid context.

## Comparison with T14

| | T14 | this script |
|--|--|--|
| window | 300 bp (later 81 in T15) | 52 (`--window`) |
| convergent def. | same `+ → −` idea | same, **plus** removes ALL overlapping features from the gap |
| boundary buffer | none | **50 bp trim** off CDS-facing ends |
| windows/interval | one random window | **all** non-overlapping (quantify available data) |
| CDS negatives | one window, start+100..stop-100 | all non-overlapping in trimmed interior, **strand-oriented** |
| dedup | none | yes (per file) |
| seed | added late | seeded from the start |
| dependencies | Biopython/pandas | **stdlib only** (avoids the `~/.local` numpy/pandas gotcha) |


## Recorded checks

Synthetic-input checks covered tRNA masking, N-run exclusion, malformed GFF rows, missing-FASTA replicons, unannotated plasmids, reverse-complement parity, trimmed bounds, non-overlap, clean 52-bp DNA and deterministic reruns. These are historical checks, not evidence that the genomic labels lack promoters.

## Context limits

The models train on 81 bp, whereas stored calibration positives are 49–52 bp and negatives 52 bp. Changing `--window` to 81 fixes negative length only; padding cannot restore positive context or locate a TSS. An 81-bp framed cohort would be a new experiment, not repaired historical calibration.

Convergent orientation does not exclude antisense/internal promoters. Phytofermentans is out-of-genus and should retain its species stratum; plasmid/chromosome strata also aid interpretation. Deduplicate across files when pooling. [Calibration notes](CALIBRATION_NOTES.md) give the sampled cohort and maps.
