# T11 — Housekeeping gene set: definition

**Purpose.** Define a set of *reliably, constitutively expressed* genes across the
166 *Clostridium* training genomes (T5), re-annotated uniformly with Bakta (T6).
Their upstream regions are used downstream as **dependable constitutive-promoter
examples**: T12 collapses them to operon leaders, T13 extracts the 100–300 bp
upstream window, and those windows feed the positive promoter set and the mining
material (T28).

Because these upstream regions become *positive* examples, this set is built for
**high precision over recall** — a mis-called "housekeeping" gene would inject a
region with no real constitutive promoter into the positives.

---

## Selection principle

Selection is **by product description, not by gene symbol.** The Bakta v6.0
*light* database leaves the `Gene` column sparse (~27% of CDS are `hypothetical
protein` and most real genes carry no symbol), but the conserved housekeeping
families are consistently named in the `Product` column. Each CDS `Product` is
matched against a curated catalog of canonical constitutive families; a match is
kept only if it does **not** also hit a global deny list that strips modifying
enzymes, fragments, and uncertain assignments.

Only `cds` features are considered. (rRNA/tRNA genes are also constitutive, but
their promoters are atypical — very strong, UP-element/Fis-dependent rrn
promoters — and the operon/upstream logic in T12–T13 is CDS-based. They can be
added later as a separate high-strength class if desired; see §"Optional
extensions".)

---

## Included families (the housekeeping set)

Each gene is tagged with a `tier`: **tier 1** = core, universally conserved,
unambiguously constitutive; **tier 2** = strong, well-established housekeeping
(constitutive in fermentative *Clostridium*). Downstream tasks may use tier 1
alone for a maximally clean positive set, or tier 1+2 for breadth.

| Category | Tier | What it is | Why housekeeping |
|---|---|---|---|
| `ribosomal_protein` | 1 | Structural 30S/50S subunit proteins (modern uS/bL + legacy naming) | The backbone of constitutive expression; ~50 per genome, universal |
| `translation_factor` | 1 | EF-Tu, EF-G, EF-Ts, EF-P, IF-1/2/3 | Core translation machinery, always on |
| `rnap_core` | 1 | RNA polymerase core subunits α, β, β′, ω (rpoA/B/C/Z) | Core transcription machinery (sigma factors excluded — see below) |
| `aminoacyl_trna_synthetase` | 1 | "<aa>--tRNA ligase" (all 20 + subunits/paralogs) | Charge tRNAs for translation; constitutive |
| `atp_synthase` | 1 | F-type and V/A-type ATP synthase subunits | Core energy metabolism |
| `glycolysis` | 2 | GAPDH, enolase, PGK, TPI, PGM, pyruvate kinase, PFK, G6P isomerase, FBP aldolase | Central fermentative energy pathway in *Clostridium* |
| `dna_replication` | 2 | DnaA, gyrase A/B, Pol III subunits, SSB, primase, DNA ligase (NAD), topoisomerases, replicative helicase | Constitutive replication/segregation machinery |
| `transcription_factor` | 2 | NusA, NusB, NusG, Rho, GreA | Constitutive transcription elongation/termination factors |

---

## Deliberate exclusions (and why)

These are intentionally **not** in the set, because their promoters are
**regulated, not constitutive**, or because the annotation is ambiguous:

- **Sigma factors** (`RNA polymerase sigma factor …`, SigA/W/V/I/S, sigma-54,
  ECF sigmas). The housekeeping sigma (RpoD/SigA) is constitutive, but the light
  DB names alternative/ECF sigmas with near-identical strings, and alternative
  sigmas are *conditionally* expressed. Including the family risks contaminating
  the positive set, so the whole family is dropped.
- **Heat-shock chaperones** (GroEL/GroES, DnaK/DnaJ/GrpE, ClpB, HtpG, Hsp33).
  Universally conserved and expressed at baseline, but in Gram-positives their
  promoters are HrcA/CIRCE heat-regulated — i.e. *not* purely constitutive.
  Excluded to honour the "constitutive" criterion.
- **RecA** — SOS/LexA-inducible; its promoter is repressed under normal growth.
- **Modifying-enzyme look-alikes** caught by the deny list: ribosomal-protein
  methyltransferases/acetylases (e.g. `[ribosomal protein uS5]-alanine
  N-acetyltransferase`, `Ribosomal protein L11 methyltransferase`, RimO/RimI),
  rRNA methyltransferases, ribosome biogenesis/assembly/maturation factors,
  `…-binding protein` partners, `… family protein` (uncertain assignment),
  `domain-containing protein`, `putative …`, and the flagellar export ATPase
  (`Flagellum-specific ATP synthase`, FliI — flagellar regulon, not energy ATP
  synthase). The single-stranded DNA-binding protein (SSB) is explicitly
  whitelisted so the `-binding protein` rule does not remove it.

---

## How the catalog is applied

Patterns and the deny list live in `define_housekeeping.py` (`CATALOG`,
`GLOBAL_DENY`). All matching is case-insensitive. First matching catalog entry
wins; ties never occur because categories are disjoint by construction. See the
script for the exact regexes — they are deliberately anchored (e.g.
`^Ribosomal protein [SL]\d+$`) so enzyme variants with trailing words are
excluded by construction.

---

## Result (166 genomes, Bakta v1.11.4 + DB v6.0 light)

- **22,760 housekeeping loci** selected from 628,150 CDS (3.6%).
- tier 1: 16,798 loci · tier 2: 5,962 loci.
- Every category is present in **all 166/166 genomes** (the conservation check),
  and every genome yields 110–156 housekeeping loci (mean 137) — no empty or
  outlier genomes.

Per-category counts and genome prevalence are in `housekeeping_summary.tsv`;
per-genome totals in `per_genome_counts.tsv`.

---

## Optional extensions (not enabled by default)

- **rRNA/tRNA operon promoters** — add `rRNA`/`tRNA` feature types as a separate
  high-strength constitutive class if T13 wants very strong promoters.
- **Tier filtering** — T12/T13 can subset to `tier == 1` for a maximally clean
  positive set.
- **Housekeeping sigma only** — if RpoD/SigA promoters are wanted, add a tightly
  anchored `^RNA polymerase factor sigma-70 RpoD$` / `^… SigA$` rule rather than
  the generic sigma family.
