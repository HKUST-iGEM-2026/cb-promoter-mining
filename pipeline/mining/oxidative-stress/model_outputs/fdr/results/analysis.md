# Stress-neighborhood FDR screening

Historical T27 stress-associated run, separate from the ordinary 929/180/45 catalogue. Candidate/null counts are 1,026/420. Mean member percentiles are computed over each combined candidate+null pool, not ordinary target ranks. The intergenic null is putative; exchangeability, exact composition matching and promoter-free status are not established.

## Screening rule

Window p = (count(null score ≥ candidate score) + 1)/(Nnull + 1). Select p≤0.05, merge same-strand windows, assign each peak its minimum selected p, then apply BH over selected peaks. Selection/minimum p-values do not establish calibrated peak-level or biological FDR≤5%. An oxidative-stress neighborhood does not prove inducibility.

## Recorded result

Null median/p95/p99/max is 0.5189/0.8690/0.9604/0.9945, minimum attainable p 2.38e-03. There are 24 screening loci at q≤0.05 and the same count at q≤0.10. Reported top 24 scores span 0.8797–0.9963, q 1.90e-02–4.51e-02; no summit sequence was truncated.

## Top 20

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


## Files and coordinates

peaks_fdr.csv stores all peaks; top80_promoters.csv stores the reported sequence subset. top80_promoters_motif.csv adds motif annotations. summit is the 0-based forward-genome start of an 81-bp window, including on the reverse strand; sequence is in reading orientation, not a mapped TSS.

Reproduction needs aligned candidate **and** null raw member tracks on the same grid/settings. Stored peak tables alone cannot reconstruct pooled ranks/window p-values; a mean of raw null scores is not the pooled-rank null. Motifs and downstream gene associations need biological testing. [Interpretation](../../../../../../docs/RESULTS.md#interpreting-candidates-and-downstream-units) defines the separate claims.
