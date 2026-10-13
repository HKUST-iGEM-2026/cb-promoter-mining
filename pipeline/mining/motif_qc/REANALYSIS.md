# Motif QC on the complete-reference catalogue

The reanalysis uses the committed width-six QC reference MEME file and 929-row ranked catalogue, rather than reconstructed PWMs. These QC motifs are not the five de-novo motifs used in the recovered original mining scan. reanalyze_motif_calls.py (fixed seed, NumPy/Matplotlib) writes reanalysis_results.json and four panels. These retained summaries are not a new model-scoring run.

## Inputs

- [sigma_a_promoter.meme](../motif-annotation/sigma_a_promoter.meme): width-six reference motifs TATAAT/TTGACA, declared A/T 0.35 and C/G 0.15 background; scientific origin unrecorded.
- [Catalogue](../../../results/candidates/g131795_ranked_seq.tsv): summit DNA, selected motif contexts and rounded center separations.

## Result

| Check | Result | Reading |
|---|---|---|
| Coverage (929) | both 45 · −10-only 135 · −35-only 288 · neither 461 | matches the report exactly |
| −35 "contamination" (prefers −10 motif) | **86.5 %** uniform bg · **86.2 %** AT-rich bg | reproduces the report's 86.8 % |
| −10 "contamination" (prefers −35 motif) | 5.0 % · 5.6 % | reproduces the report's 5.0 % |
| Random AT-rich control (simulated, not proven promoter-free) | 61 % uniform · 53 % AT-rich "prefer −10" | a majority of *random* AT-rich DNA also flags |
| **Signal vs matched null — −10 boxes** | **median pct 98 · 79 % above null-95th** | selected boxes show strong −10-like enrichment |
| **Signal vs matched null — −35 boxes** | **median pct 52 · 11 % above null-95th** | weak population-level `TTGACA` signal; not proof of equivalence to null |
| Center separation (`spacer_bp`, n=45) | min 5 · median **18** · max 34 | rounded center-to-center distance, not the textbook edge-to-edge spacer |
| AT content | −10 88 % · −35 85 % · summit 79 % | boxes are very AT-rich |


## Interpretation

Changing uniform to AT-rich background changes −35 cross-motif preference only 0.3 points, 86.5→86.2%. Random AT-rich controls also prefer −10 in 53–61% of cases. Comparing raw scores from different PWMs alone is therefore a weak diagnostic.

Matched-null percentiles separate the selected arms: −10 median 98 with 79% above null-95th, versus −35 median 52 with 11%. Selection uses the same motif model, so this is descriptive QC, not an independent activity test or proof that −35 biology is absent. Divergence, model choice and coordinate/selection limits remain possible explanations.

The −35 center is upstream of −10 in 44/45 cases. Median center separation 18 is not a textbook 17–19-bp edge gap; even subtracting six for two width-six cores is only a provisional comparison without exact positions. The historical extended-−10 TG proxy, 3/45, is not implemented/recorded in this reanalysis.

## Use

Do not re-extract solely to reduce the cross-motif flag rate; inspect exact coordinates when diagnosing an error. Promoters with both −10 and −35 motifs have two positional hit labels, not experimental confirmation. The ordinary 929-region score selection does not depend on −35, but motif-tier priority does. A Clostridium-specific PWM/extended-−10 model would be a new analysis. [Candidate interpretation](../../../docs/RESULTS.md#interpreting-candidates-and-downstream-units) separates activity, sigma use and regulation.
