# Inducible-Promoter Mining Workflow (Stage 2) — oxygen / oxidative-stress-inducible promoters in *C. butyricum*

HKUST iGEM 2026 Dry Lab. This is the **dry-lab (computational) plan** for finding **oxygen/oxidative-stress-inducible promoters** in *Clostridium butyricum* DSM 10702. It expands the team's 6-step draft, corrects the one thing that draft gets structurally wrong, and reuses the constitutive pipeline (T10–T31) wherever possible. Wet-lab steps (reporter library, induction assay, tuning) are unchanged — this doc feeds them a ranked candidate list.

---

## 0. The one reframe that governs everything

**Constitutive mining (Stage 1) is a *sequence* problem** — "does this DNA look like a promoter?" — which is why models that score sequences work.

**Inducible mining is a *differential-expression* problem** — "does this promoter switch ON specifically under oxygen stress and stay OFF otherwise?" The switch behaviour is **not written in the promoter sequence alone**; it lives in the *expression data* (ON vs OFF conditions) plus the *regulatory logic* (which repressor/activator controls it). So the core signal is **RNA-seq under oxidative stress vs an anaerobic control**, and the sequence tools (TSS, motif, the trained promoter models) are used *afterward* to locate and characterise the promoter — not to find the inducibility.

Concretely: **induction = a fold-change between conditions (log2FC), not a high absolute expression.** A housekeeping gene sits at very high abundance but log2FC ≈ 0 — it is *not* inducible. Ranking candidates by "how strongly expressed" returns exactly the wrong set. Every threshold below is built on the ratio, and a highly-expressed-but-flat gene must be actively excluded.

## 1. The biology you're mining: PerR de-repression

In *Clostridium*, the oxygen/oxidative-stress response is run mainly by **PerR**, a Fur-family peroxide-sensing **repressor**. Under anaerobic conditions PerR (with Fe/Mn) sits on a **"Per box"** operator in its target promoters and keeps them OFF. Under oxidative stress, metal-catalysed oxidation of PerR changes its shape, it **falls off the DNA, and the target genes de-repress → ON** ([Dubbs & Mongkolsuk 2012, PerR review](https://pmc.ncbi.nlm.nih.gov/articles/PMC3458676/); [Hillmann 2009, PerR in *C. acetobutylicum*](https://pubmed.ncbi.nlm.nih.gov/19648241/)).

Three consequences that shape the whole workflow:

- **The regulatory signature to look for is a Per box upstream, near/overlapping the −10/−35.** The *B. subtilis* Per-box consensus is a 15-bp palindrome **`TTATAATnATTATAA`** ([Fuangthong & Helmann 2002](https://pmc.ncbi.nlm.nih.gov/articles/PMC135084/)). Finding it (or a Clostridial variant) separates a genuinely PerR-regulated switch from a promoter that is merely busy during stress.
- **Mechanism = de-repression, not activation.** The promoter is intrinsically ON; PerR *holds it off*. So to use one of these as an oxygen switch in a chassis, **the native PerR must be present** — which it is in a *C. butyricum* host, so this is fine for your target, but must be flagged for any heterologous use.
- **The target genes are the oxidative-defense set**, induced within minutes of O₂ exposure in Clostridia: rubrerythrin (*rbr*), **reverse rubrerythrin (*revRbr*)**, rubredoxin / desulfoferrodoxin (*rbo/dfx*), NADH:rubredoxin oxidoreductase, **A-type flavoprotein / flavodiiron (*fprA*)**, NAD(P)H peroxide reductase / **AhpCF**, superoxide dismutase (*sod*), Dps, thioredoxin system; catalase where present (many Clostridia lack it) ([Kawasaki 2005, *C. acetobutylicum*/*aminovalericum*](https://pmc.ncbi.nlm.nih.gov/articles/PMC1317462/); [Riebe 2009 / Hillmann 2009](https://pubmed.ncbi.nlm.nih.gov/19648241/)). A secondary regulator, **Rex** (NADH/redox sensing), is worth scanning for too; note many Clostridia **lack the *Bacillus* σB general-stress sigma**, so do *not* transfer the σB regulon.

## 2. The data landscape (and the honest bottleneck)

The switch signal requires a transcriptome under **oxidative stress vs anaerobic control**. The critical fact:

> ***C. butyricum* itself has no purpose-built oxidative-stress-vs-anaerobic RNA-seq.** The one recent *C. butyricum* transcriptome (CB1002, [AEM 2025](https://journals.asm.org/doi/10.1128/aem.01106-25)) is a strain comparison at mid-exponential growth, not an O₂-challenge design.

So — exactly like the constitutive work — this is a **cross-species problem**: mine the induced-gene set in a data-rich in-genus relative, then transfer to *C. butyricum* by orthology. Best sources, in order:

| Species | Condition | Use | Ref |
|---|---|---|---|
| *C. acetobutylicum* ATCC 824 | microoxic / O₂ exposure, PerR mutant | **primary** — best-characterised in-genus O₂ response | [Kawasaki 2005](https://pmc.ncbi.nlm.nih.gov/articles/PMC1317462/), [Hillmann 2009](https://pubmed.ncbi.nlm.nih.gov/19648241/) |
| *Clostridioides difficile* 630Δerm | air exposure, *perR* mutant, RNA-seq | strong PerR-regulon evidence (note: out-of-genus, *Clostridioides*) | [PerR RNA-seq 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11841134/) |
| *C. tyrobutyricum* | *perR* CRISPR deletion, oxidative defense | PerR regulon in a close relative | [Yang 2023](https://iadns.onlinelibrary.wiley.com/doi/full/10.1002/fft2.205) |
| *C. ljungdahlii* | O₂ exposure metabolic/transcriptional | secondary in-genus support | [AEM 2016](https://journals.asm.org/doi/full/10.1128/aem.02491-15) |

Pull raw reads from SRA where a proper ON-vs-OFF design exists (re-analyse yourself), and mine published DE/Northern tables where only processed data exists. **A *perR*-deletion vs wild-type contrast is gold** — the PerR regulon *is* the de-repressed set, so a Δ*perR* transcriptome directly names your candidate promoters even without an O₂ challenge.

## 3. The workflow

Your draft's 6 steps map on cleanly; the additions are the **DE core (Phase B)**, the **PerR-logic layer (Phase D)**, the **cross-species transfer to *C. butyricum* (Phase E)**, and **scoring (Phase F)**. Phases A–F are dry lab; your steps 4–6 (clone/test/tune) are the wet-lab hand-off. **Where each phase runs:** B–D operate on the **data-rich relative** (*C. acetobutylicum* etc.), where the induction signal actually exists; Phase E re-applies C–D to the ***C. butyricum* target** by orthology; Phase F ranks the target candidates.

### Phase A — Define the switch (your step 1)
**Chosen switch (2026-07): brief air exposure.** OFF = an anaerobic culture (mid-exponential); ON = the same culture **briefly exposed to air and sampled at an early timepoint** — oxidative-defense genes rise within minutes of air/O₂ exposure, so target ~15–60 min, before growth arrest and secondary cascades dominate. Fix the **regulator hypothesis: PerR de-repression** (Rex secondary) and the expected-gene shortlist (*revRbr, fprA, rbr, rbo/dfx, ahpCF, sod, Dps*). Deliverable: a one-paragraph spec (condition, control, regulator, target-gene shortlist). *Why brief + early:* air exposure is transient and toxic to an anaerobe, so the clean direct PerR signal is strongest early; long exposure muddies it with growth-arrest/death signatures. This choice also steers **Phase B** toward air-exposure/aeration datasets (*C. difficile* 630Δerm air-exposure RNA-seq; *C. acetobutylicum* microoxic aeration) and toward the **earliest post-exposure timepoint** in any time-course.

### Phase B — Differential expression → the induced-gene set (NEW; this is the core)
For each cross-species dataset with raw reads: `fastp` (gentle trim, keep a length floor) → **mask rRNA before normalising** → align with **Bowtie2/BWA-MEM** (unspliced; *not* STAR/kallisto — bacteria have no isoforms and polycistronic genes break pseudo-aligners) → **`featureCounts -t CDS -g locus_tag` with the correct strandedness (`-s 2` for typical dUTP libraries — verify empirically; wrong setting silently drops half the reads)** → **DESeq2 or edgeR**.

Call a gene **induced** with `padj < 0.05` **and** shrunken `log2FC ≥ 1` (up), tested *against* the threshold (`results(dds, lfcThreshold=1, altHypothesis="greater")`) rather than filtered post-hoc. Report the two **orthogonal** columns side by side — `log2FoldChange` (induction) and `baseMean/TPM` (abundance) — and **select on the ratio, never the abundance**. Replicate reality: ≥6/condition is ideal, ≥3 a floor; at n=2 treat it as a screen leaning on the fold-change threshold; with a Δ*perR*-vs-WT table you may be handed the regulon directly. Deliverable: a ranked induced-gene table per dataset, plus the intersection/union across datasets.

### Phase C — Induced gene → its promoter (reuse T10/T12/T13)
An internal operon gene has no promoter of its own, so: **assign each induced gene to its operon and take the operon *leader*** (reuse **T12** operon detection; rule of thumb: same strand + gap <50 bp = same operon, >100–150 bp = new transcription unit). Anchor the **TSS** (reuse **T10 / TSSpredator** on any dRNA-seq; else the constitutive models / a predictor). **Extract the upstream** (reuse **T13**): `[−60, +20]` around the TSS for model scoring, `[−100, +50]` for motif work, or `[−300, +50]` from the start codon when no TSS exists (clip at the upstream ORF and contig ends — the same contig-end guard as Stage 1). Watch **leaderless transcripts** (TSS ≈ start codon), common in Firmicutes.

### Phase D — Confirm the PerR logic (reuse T31 FIMO/MEME) — the step that makes it *inducible*, not just *active*
Take the upstreams of the co-induced leaders and either **(i) scan for the known Per box** (`TTATAATnATTATAA`) with **FIMO**, or **(ii) discover the operator de-novo** with **MEME/STREME** using a **matched control set** (upstreams of genes *expressed but not induced*) + a **higher-order Markov background** (order 2–3) — this is essential in an AT-rich genome or you rediscover the −10 box and poly-A tracts instead of the operator. Name the regulator with **Tomtom against RegPrecise/CollecTF/PRODORIC** (RegPrecise has *Clostridium* PerR/Rex regulogs — most relevant), and **discard any motif that Tomtom matches to the σ/−10/−35 consensus** (that's the "you found the promoter, not the regulator" trap). A real candidate has **both** a Per box **and** a plausible regulator context. Keep every hit's **FIMO q-value** (AT-aware; use the target-genome background) — that's the leakiness-independent confidence you'll rank on.

### Phase E — Map to *C. butyricum* (NEW; the cross-species transfer)
For each validated induced leader in the relative: find its ***C. butyricum* ortholog** by **reciprocal-best-hit (DIAMOND) confirmed by microsynteny** (RBH alone over-calls; require co-oriented flanking genes) or a Clostridium pangenome (Panaroo). Then **re-find the promoter in *C. butyricum*:** confirm the ortholog is still the operon leader there, extract its upstream (Phase C on the *C. butyricum* genome you already have from T5/T6), and **scan for the Per box in the *C. butyricum* promoter with a *C. butyricum* background + dinucleotide-shuffled empirical FDR.** Strengthen with **phylogenetic footprinting** across the operator in several *Clostridium* orthologs. **Flag the failure mode loudly:** regulatory rewiring is real — an ortholog induced in *C. acetobutylicum* may be constitutive or differently regulated in *C. butyricum*; conservation of the *operator* (not just the gene) is the evidence that transfers.

### Phase F — Score, rank, output (mirror T29/T30/T32)
**Hard gates:** `padj < 0.05`; operon-leader with a single clean primary TSS; **Per box present in the *C. butyricum* promoter**. **Then rank-percentile composite** of:

| Feature | Why | direction |
|---|---|---|
| fold-induction (shrunken log2FC) | the switch magnitude | higher better |
| dynamic range (ON − OFF) | usable output swing | higher better |
| **basal / OFF-state expression** | **leakiness — the dominant limiter of native switches** | **lower better** |
| Per-box FIMO q-value | regulator-logic confidence | lower better |
| TSS/promoter confidence | dRNA-seq-backed > predicted | higher better |
| constitutive-model promoter score (**reuse T18–T20**) | confirms it's a real promoter + its strength | higher better |
| cross-*Clostridium* operator conservation | transfer confidence | higher better |

Run a **weight sensitivity check** and report only rank-stable candidates as high-confidence. **Two ranking cautions:** (i) fold-induction and dynamic range are correlated — down-weight one so the switch magnitude isn't double-counted; (ii) treat the **model promoter score as confirmation, not a filter** — the Stage-1 models were trained on *constitutive* promoters, and inducible/repressed promoters can carry an operator overlapping the −10/−35 or otherwise non-canonical architecture the models never saw, so a *low* model score should flag a candidate for manual inspection, never auto-reject it. **Output a candidate table** (extend the T32/PT6 schema) with, per row: gene/ortholog, operon, TSS, promoter sequence, log2FC, **basal level**, Per-box coordinates + q-value, **mechanism = de-repression**, "**requires native PerR**" flag, and the model promoter score. That table is what the wet lab clones (your steps 4–6).

## 4. What you reuse vs what's new

| Constitutive asset | Reused for | New work needed |
|---|---|---|
| T5/T6 genomes + Bakta | the *C. butyricum* target + relatives | add the cross-species relatives with O₂ data |
| **T10 / TSSpredator** | TSS anchoring (Phase C) | run on the relatives' dRNA-seq if available |
| **T12 operon detection** | operon-leader selection (Phase C) | — (already strand-aware) |
| **T13 upstream extraction** | promoter windows (Phase C/E) | — |
| **T31 FIMO / MEME** | Per-box discovery + scan (Phase D) | build the Per-box PWM; matched controls |
| **T18–T20 models** | promoter-ness + strength score (Phase F) | *not* for inducibility — that's the DE data |
| T29/T30/T32 aggregation + schema | ranking + candidate table (Phase F) | add the induction/leakiness/Per-box fields |
| **— (genuinely new) —** | **Phase B DE pipeline** + **Phase E ortholog transfer** | the two pieces Stage 1 didn't need |

## 5. Pitfalls (read before running)

- **The models don't find inducibility.** T18–T20 score "is this a promoter," not "does it switch." Never let a high model score alone promote a candidate — inducibility must come from the DE data.
- **Induced ≠ highly expressed.** Rank on log2FC, not TPM. Add an ON-state expression floor only as a secondary gate.
- **Leakiness (OFF-state level) is usually the deal-breaker**, not ON strength — that's why basal expression is a first-class ranking feature, not an afterthought.
- **De-repression needs the repressor.** These promoters are ON without PerR; they only "sense oxygen" with native PerR present. Fine in *C. butyricum*; annotate it anyway.
- **Cross-species rewiring.** Gene orthology ≠ regulatory orthology. Require the *operator* to be conserved in the *C. butyricum* promoter, not just the gene.
- **AT-rich false positives** (same trap as Stage 1): the Per box and the −10 are both AT-rich, so motif scans over-fire — use matched controls, higher-order backgrounds, and empirical/dinucleotide-shuffle FDR.
- **Confounders in public data:** batch effects (model batch in the design; drop batch-confounded contrasts), growth-phase artifacts (match OD; many Clostridia have no σB so don't transfer that regulon), and indirect/secondary induction (a Δ*perR* contrast or motif/footprint support is what pins *direct* regulation).

## 6. Suggested first concrete step

Pull the ***C. acetobutylicum* Δ*perR*-vs-WT and O₂-exposure** data (Hillmann/Kawasaki + any SRA), run Phase B, and see whether the induced set recovers the known oxidative-defense genes (*revRbr*, *fprA*, *rbr*, *ahpCF*) with a Per box upstream. If that positive control lights up, the pipeline is trustworthy and you extend it to the full transfer into *C. butyricum*. That mirrors exactly how Stage 1 was de-risked (validate on a known set before mining the target).

---

### Sources
- PerR mechanism / peroxide-sensing regulators: [Dubbs & Mongkolsuk 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3458676/) · [Fuangthong & Helmann 2002 (Per box)](https://pmc.ncbi.nlm.nih.gov/articles/PMC135084/)
- Clostridial O₂/oxidative response: [Kawasaki 2005 (*C. acetobutylicum*)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1317462/) · [Hillmann/Riebe 2009 (PerR in *C. acetobutylicum*)](https://pubmed.ncbi.nlm.nih.gov/19648241/) · [*C. difficile* PerR RNA-seq 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11841134/) · [*C. tyrobutyricum* PerR 2023](https://iadns.onlinelibrary.wiley.com/doi/full/10.1002/fft2.205) · [*C. ljungdahlii* O₂ 2016](https://journals.asm.org/doi/full/10.1128/aem.02491-15) · [*C. butyricum* CB1002 transcriptome 2025](https://journals.asm.org/doi/10.1128/aem.01106-25)
- Team's own references: [Feng 2024, *C. tyrobutyricum* inducible promoters](https://doi.org/10.1002/bit.28701) · [Carrillo Rincón & Farny 2023, inducible-promoter engineering](https://pmc.ncbi.nlm.nih.gov/articles/PMC10128130/) · [Kim 2011, anaerobic *nar* promoter](https://sciencedirect.com/science/article/pii/S0168165610020432)
- Methodology (DE / motif / transfer) tools are cited inline in the team's methods appendix; key ones: DESeq2, edgeR, featureCounts, Bowtie2, TSSpredator, MEME/STREME/FIMO/Tomtom, RegPrecise, DIAMOND/OrthoFinder/Panaroo.
