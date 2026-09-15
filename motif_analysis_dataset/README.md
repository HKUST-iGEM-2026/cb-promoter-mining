# Reference promoter motif dataset (clean build)

Sigma-factor reference PWMs for scanning *Clostridium butyricum* DSM 10702
upstream regions with FIMO / TOMTOM.

## Contents

| File | What it is |
|---|---|
| `reference_pwms.meme` | 4 motifs, MEME minimal format. Pass this to FIMO/TOMTOM. |
| `reference_motifs_summary.tsv` | Position-by-position frequency comparison, E. coli vs B. subtilis. |
| `ecoli_m10_strong.txt` | 629 raw -10 hexamers the E. coli -10 matrix was counted from. |
| `ecoli_m35_strong.txt` | 606 raw -35 hexamers the E. coli -35 matrix was counted from. |

## The four motifs

| Motif | nsites | Consensus | Source |
|---|---|---|---|
| `Ecoli_sigma70_m10_strong` | 629 | TATAAT | RegulonDB, strong-confidence sigma70 promoters |
| `Ecoli_sigma70_m35_strong` | 606 | TTGACA | RegulonDB, strong-confidence sigma70 promoters |
| `Bsubtilis_SigA_m10_DBTBS` | 20 (placeholder) | TATAAT | DBTBS SigA PSSM, log-odds converted back to probabilities |
| `Bsubtilis_SigA_m35_DBTBS` | 20 (placeholder) | TTGACA | DBTBS SigA PSSM, log-odds converted back to probabilities |

## What changed from the previous build

- **Dropped** `Ecoli_sigma70_m10_all` and `Ecoli_sigma70_m35_all` (the
  all-confidence-tier matrices). They largely duplicate the strong set, and
  keeping both meant every hit got reported twice by FIMO under two motif
  names with different p-values.
- **Fixed a format bug.** The previous `combined_reference_pwms.meme` had two
  `MEME version 4` / `ALPHABET=` header blocks concatenated mid-file, which is
  not valid MEME minimal format and would likely have caused a parse failure or
  silently dropped motifs in FIMO. This build has a single header.
- Summary TSV gained `consensus_match` and `max_freq_delta` columns.

## Caveats

- **B. subtilis `nsites=20` is a placeholder.** DBTBS publishes the PSSM scores
  but not the number of sites behind it. The base *frequencies* are correct; the
  site count only affects how FIMO weights this motif's pseudocounts and
  significance relative to the E. coli matrices. If the true SigA promoter count
  turns up on DBTBS, correct it in `reference_pwms.meme`.
- **B. subtilis is a proxy, not a Clostridium PWM.** No Clostridium-specific PWM
  database exists; B. subtilis is used as the nearest well-characterised
  Firmicute. Frame it that way in the report.
- Background frequencies are set uniform (0.25 each). For scanning a genome as
  AT-rich as *C. butyricum*, consider passing FIMO a genome-derived background
  file (`fasta-get-markov`) instead — uniform background will inflate scores for
  AT-rich hexamers across the board.

## Validation performed

Both E. coli matrices were re-counted from the raw hexamer files and match the
stored frequencies to within 0.002 at every position. All four matrices sum to
1.0 per position. Consensus sequences reconstruct to TATAAT / TTGACA as
expected.
