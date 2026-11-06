# Motif QC on the complete-reference catalogue

The quality-control (QC) reanalysis uses the committed six-base reference position weight matrices (PWMs) in the MEME file and the 929-row ranked catalogue, rather than reconstructed matrices. Each matrix encodes base preferences at six DNA positions. These QC motifs are not the five de-novo motifs used in the recovered original mining scan. reanalyze_motif_calls.py (fixed seed, NumPy/Matplotlib) writes reanalysis_results.json and four panels. These retained summaries are not a new model-scoring run.

The cross-motif and null comparisons below use 180 selected −10 contexts and 333 selected −35 contexts across the catalogue. The separate logos use only the 45 both-hit candidates: their best-scoring six-base cores have observed consensuses **TATAAT/TTGATA**, whereas the supplied reference matrices have consensuses **TATAAT/TTGACA**. These are different objects; the width-six QC did not discover the original five MEME motifs.

## Inputs

- [sigma_a_promoter.meme](../motif-annotation/sigma_a_promoter.meme): width-six reference motifs TATAAT/TTGACA, declared A/T 0.35 and C/G 0.15 background; scientific origin unrecorded.
- [Catalogue](../../../results/candidates/g131795_ranked_seq.tsv): summit DNA, selected motif contexts and rounded center separations.

## Result

| Check | Result | Reading |
|---|---|---|
| Coverage (929) | both 45 · −10-only 135 · −35-only 288 · neither 461 | matches the report exactly |
| −35 "contamination" (prefers −10 motif) | **86.5 %** uniform bg · **86.2 %** AT-rich bg | cross-PWM preference, not a measured fraction of false −35 sites |
| −10 "contamination" (prefers −35 motif) | 5.0 % · 5.6 % | cross-PWM preference in selected contexts |
| Random AT-rich control (simulated, not proven promoter-free) | 61 % uniform · 53 % AT-rich "prefer −10" | a majority of *random* AT-rich DNA also flags |
| **Signal vs AT-rich null — −10 boxes** | **median pct 98 · 79 % above null-95th** | selected boxes show strong −10-like enrichment |
| **Signal vs AT-rich null — −35 boxes** | **median pct 52 · 11 % above null-95th** | weak population-level `TTGACA` signal; not proof of equivalence to null |
| Center separation (`spacer_bp`, n=45) | min 5 · median **18** · max 34 | rounded center-to-center distance, not the textbook edge-to-edge spacer |
| AT content | −10 88 % · −35 85 % · summit 79 % | boxes are very AT-rich |


## Interpretation

Changing uniform to AT-rich background changes −35 cross-motif preference only 0.3 points, 86.5→86.2%. Random AT-rich controls also prefer −10 in 53–61% of cases. Comparing raw scores from different PWMs alone is therefore a weak diagnostic.

The AT-rich null uses 6,000 synthetic 14-bp sequences with A/T=0.35 and C/G=0.15, not per-context dinucleotide shuffles. Best sliding six-base PWM log-odds scores give −10 median percentile 98 with 79% above null-95th, versus −35 median 52 with 11%. The [signal-versus-null panel](fig2_signal_vs_null.png) shows probability density on a shared, labelled y-axis; dashed lines mark the null 95th percentiles. Six-base core selection uses the same reference PWMs used to assess those cores, so this is descriptive QC, not an independent activity test or proof that −35 biology is absent. Divergence, model choice and coordinate/selection limits remain possible explanations.

The −35 center is upstream of −10 in **45/45** cases: all stored center separations are positive, and reconstructed best-scoring six-base cores preserve that order under both QC backgrounds. Median center separation 18 is not a textbook 17–19-bp edge gap; exact hit boundaries and motif widths are needed to compute the intervening bases. The historical extended-−10 TG proxy, 3/45, is not implemented/recorded in this reanalysis.

## Use

Do not re-extract solely to reduce the cross-motif flag rate; inspect exact coordinates when diagnosing an error. Promoters with both −10 and −35 motifs have two positional hit labels, not experimental confirmation. The ordinary 929-region score selection does not depend on −35, but motif-tier priority does. A Clostridium-specific PWM/extended-−10 model would be a new analysis. [Candidate interpretation](../../../docs/RESULTS.md#downstream-genes-and-putative-units) separates activity, sigma use and regulation.

The separate [legacy spacing diagnostic](../../../Updated%20Motif%20analysis/motif_analysis_script_g131795.py) incorrectly compares center separation directly with a 15–19-bp edge-gap rule. For example, `cand_00029` has center separation 28 and motif widths 8/14, giving a 17-bp edge gap despite its out-of-range flag. Do not treat that flag as corrected motif geometry or evidence against the ordinary 929-region selection.
