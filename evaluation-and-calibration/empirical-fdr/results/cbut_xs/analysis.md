# T27 empirical-FDR promoter mine — C. butyricum — ensemble `prokbert + cnn_v2 + evo2`

**Restricted scan + tailored null.** Candidates are the 175,911 upstream-of-gene windows; the null is the 1,726 convergent-intergenic samples (both strands of the 81 bp negatives). Both are intergenic + AT-rich, so the empirical p is a composition-matched comparison. Ensemble = mean rank-percentile pooled over the RESTRICTED candidates + null (calibration-free).

## Null score distribution (ensemble rank-percentile)
median 0.6135 · p95 0.8911 · p99 0.9526 · max 0.9944
Smallest achievable p-value = 1/(Nnull+1) = 5.79e-04.

## Discoveries
- candidate windows tested: 175,911
- windows passing empirical p <= 0.05: peak-called into **3,358 promoter loci**
- loci at q <= 0.05: **3,358**   ·   q <= 0.10: 3,358
- reported: top 80 by ensemble score (0 summit windows truncated at a contig end -> blank sequence)
- top-80 ensemble range [0.9964, 0.9994], q range [1.01e-02, 1.01e-02]

## Top 20 promoters (full set in `top80_promoters.csv`)
| candidate_id | contig | summit | strand | ensemble | p_value | q_value | n_windows |
|---|---|---|---|---|---|---|---|
| cbut_0001 | contig_1 | 1487777 | - | 0.9994 | 5.790e-04 | 1.010e-02 | 8 |
| cbut_0002 | contig_1 | 2076077 | - | 0.9993 | 5.790e-04 | 1.010e-02 | 6 |
| cbut_0003 | contig_1 | 2796497 | - | 0.9993 | 5.790e-04 | 1.010e-02 | 8 |
| cbut_0004 | contig_1 | 1070010 | + | 0.9992 | 5.790e-04 | 1.010e-02 | 10 |
| cbut_0005 | contig_2 | 148130 | + | 0.9991 | 5.790e-04 | 1.010e-02 | 7 |
| cbut_0006 | contig_1 | 54270 | + | 0.9989 | 5.790e-04 | 1.010e-02 | 5 |
| cbut_0007 | contig_1 | 3549247 | - | 0.9989 | 5.790e-04 | 1.010e-02 | 5 |
| cbut_0008 | contig_1 | 1755450 | + | 0.9989 | 5.790e-04 | 1.010e-02 | 6 |
| cbut_0009 | contig_1 | 134840 | + | 0.9989 | 5.790e-04 | 1.010e-02 | 8 |
| cbut_0010 | contig_1 | 3132927 | - | 0.9988 | 5.790e-04 | 1.010e-02 | 10 |
| cbut_0011 | contig_1 | 1497717 | - | 0.9988 | 5.790e-04 | 1.010e-02 | 7 |
| cbut_0012 | contig_1 | 1396040 | + | 0.9987 | 5.790e-04 | 1.010e-02 | 9 |
| cbut_0013 | contig_1 | 2238110 | + | 0.9987 | 5.790e-04 | 1.010e-02 | 4 |
| cbut_0014 | contig_1 | 3384540 | + | 0.9987 | 5.790e-04 | 1.010e-02 | 7 |
| cbut_0015 | contig_1 | 1935050 | + | 0.9986 | 5.790e-04 | 1.010e-02 | 7 |
| cbut_0016 | contig_2 | 397728 | - | 0.9986 | 5.790e-04 | 1.010e-02 | 7 |
| cbut_0017 | contig_1 | 2581997 | - | 0.9986 | 5.790e-04 | 1.010e-02 | 8 |
| cbut_0018 | contig_1 | 3887740 | + | 0.9986 | 5.790e-04 | 1.010e-02 | 5 |
| cbut_0019 | contig_1 | 725580 | + | 0.9986 | 5.790e-04 | 1.010e-02 | 5 |
| cbut_0020 | contig_1 | 1764160 | + | 0.9986 | 5.790e-04 | 1.010e-02 | 6 |

## Files
- `top80_promoters.csv` — candidate_id, contig, summit, strand, ensemble_score, p_value, q_value, n_windows, sequence
- `top80_promoters_motif.csv` — the above + FIMO -10/-35 architecture (written by the sbatch motif step)
- `peaks_fdr.csv` — every called locus with p/q (the full FDR table)

_Reproducible: ensembling + p/q are deterministic (this file is CPU-only, no GPU, no RNG)._
