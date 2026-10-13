# Target FDR screening

Historical T27 restricted target run, separate from the ordinary 929/180/45 catalogue. Candidate/null counts are 175,911/1,726. Mean member percentiles are computed over each combined candidate+null pool, not ordinary target ranks. The intergenic null is putative; exchangeability, exact composition matching and promoter-free status are not established.

## Screening rule

Window p = (count(null score ≥ candidate score) + 1)/(Nnull + 1). Select p≤0.05, merge same-strand windows, assign each peak its minimum selected p, then apply BH over selected peaks. Selection/minimum p-values do not establish calibrated peak-level or biological FDR≤5%. join_fdr_qvalues.py attaches the lowest overlapping same-strand peak q; it does not retest ordinary candidates.

## Recorded result

Null median/p95/p99/max is 0.6135/0.8911/0.9526/0.9944, minimum attainable p 5.79e-04. There are 3,358 screening loci at q≤0.05 and the same count at q≤0.10. Reported top 80 scores span 0.9964–0.9994, q 1.01e-02–1.01e-02; no summit sequence was truncated.

## Top 20

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


## Files and coordinates

peaks_fdr.csv stores all peaks; top80_promoters.csv stores the reported sequence subset. The optional motif CSV is not committed here. summit is the 0-based forward-genome start of an 81-bp window, including on the reverse strand; sequence is in reading orientation, not a mapped TSS.

Reproduction needs aligned candidate **and** null raw member tracks on the same grid/settings. Stored peak tables alone cannot reconstruct pooled ranks/window p-values; a mean of raw null scores is not the pooled-rank null. Motifs and downstream gene associations need biological testing. [Interpretation](../../../../../docs/RESULTS.md#interpreting-candidates-and-downstream-units) defines the separate claims.
