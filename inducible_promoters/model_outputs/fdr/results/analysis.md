# T27 empirical-FDR promoter mine — C. butyricum — ensemble `prokbert + cnnv2 + evo2`

**Restricted scan + tailored null.** Candidates are the 1,026 upstream-of-gene windows; the null is the 420 convergent-intergenic samples (both strands of the 81 bp negatives). Both are intergenic + AT-rich, so the empirical p is a composition-matched comparison. Ensemble = mean rank-percentile pooled over the RESTRICTED candidates + null (calibration-free).

## Null score distribution (ensemble rank-percentile)
median 0.5189 · p95 0.8690 · p99 0.9604 · max 0.9945
Smallest achievable p-value = 1/(Nnull+1) = 2.38e-03.

## Discoveries
- candidate windows tested: 1,026
- windows passing empirical p <= 0.05: peak-called into **24 promoter loci**
- loci at q <= 0.05: **24**   ·   q <= 0.10: 24
- reported: top 24 by ensemble score (0 summit windows truncated at a contig end -> blank sequence)
- top-24 ensemble range [0.8797, 0.9963], q range [1.90e-02, 4.51e-02]

## Top 20 promoters (full set in `top80_promoters.csv`)
| candidate_id | contig | summit | strand | ensemble | p_value | q_value | n_windows |
|---|---|---|---|---|---|---|---|
| cbut_0001 | contig_1 | 3740720 | + | 0.9963 | 2.375e-03 | 1.900e-02 | 9 |
| cbut_0002 | contig_1 | 2363897 | - | 0.9949 | 2.375e-03 | 1.900e-02 | 5 |
| cbut_0003 | contig_1 | 1572280 | + | 0.9855 | 7.126e-03 | 1.900e-02 | 3 |
| cbut_0004 | contig_2 | 130280 | + | 0.9846 | 7.126e-03 | 1.900e-02 | 7 |
| cbut_0005 | contig_1 | 3054087 | - | 0.9802 | 9.501e-03 | 1.900e-02 | 7 |
| cbut_0006 | contig_1 | 1730730 | + | 0.9769 | 9.501e-03 | 1.900e-02 | 3 |
| cbut_0007 | contig_1 | 2354237 | - | 0.9730 | 9.501e-03 | 1.900e-02 | 3 |
| cbut_0008 | contig_2 | 282538 | - | 0.9719 | 1.188e-02 | 1.900e-02 | 3 |
| cbut_0009 | contig_2 | 583558 | - | 0.9714 | 1.188e-02 | 1.900e-02 | 4 |
| cbut_0010 | contig_1 | 1177980 | + | 0.9693 | 1.188e-02 | 1.900e-02 | 4 |
| cbut_0011 | contig_1 | 2140837 | - | 0.9693 | 1.188e-02 | 1.900e-02 | 5 |
| cbut_0012 | contig_1 | 3215247 | - | 0.9693 | 1.188e-02 | 1.900e-02 | 4 |
| cbut_0013 | contig_1 | 3059960 | + | 0.9689 | 1.188e-02 | 1.900e-02 | 4 |
| cbut_0014 | contig_1 | 2592437 | - | 0.9684 | 1.188e-02 | 1.900e-02 | 1 |
| cbut_0015 | contig_1 | 861180 | + | 0.9659 | 1.188e-02 | 1.900e-02 | 6 |
| cbut_0016 | contig_2 | 699238 | - | 0.9525 | 1.663e-02 | 2.217e-02 | 3 |
| cbut_0017 | contig_1 | 2139187 | - | 0.9521 | 1.663e-02 | 2.217e-02 | 5 |
| cbut_0018 | contig_1 | 2946167 | - | 0.9504 | 1.663e-02 | 2.217e-02 | 8 |
| cbut_0019 | contig_1 | 2014217 | - | 0.9458 | 1.900e-02 | 2.400e-02 | 6 |
| cbut_0020 | contig_1 | 1140037 | - | 0.9424 | 2.138e-02 | 2.565e-02 | 2 |

## Files
- `top80_promoters.csv` — candidate_id, contig, summit, strand, ensemble_score, p_value, q_value, n_windows, sequence
- `top80_promoters_motif.csv` — the above + FIMO -10/-35 architecture (written by the sbatch motif step)
- `peaks_fdr.csv` — every called locus with p/q (the full FDR table)

_Reproducible: ensembling + p/q are deterministic (this file is CPU-only, no GPU, no RNG)._
