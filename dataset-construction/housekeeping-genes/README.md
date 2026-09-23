# T11 — Housekeeping gene set

Defines the constitutively-expressed gene set whose upstream regions become
dependable promoter examples. **Depends on:** T6 (Bakta annotations). **Feeds:**
T12 (operon detection), T13 (upstream-region extraction).

See `housekeeping_definition.md` for the biological definition, the family
catalog, tiers, and the rationale for every inclusion/exclusion.

## Files

```
define_housekeeping.py        # selection logic (catalog + deny list)
housekeeping_definition.md    # the definition: families, tiers, exclusions
out/
  housekeeping_genes.tsv      # MAIN deliverable: one row per housekeeping locus
  housekeeping_summary.tsv    # per-category counts + genome prevalence (sanity)
  per_genome_counts.tsv       # housekeeping count per genome (sanity)
```

## Regenerate

Input is the per-genome Bakta `*.tsv` feature tables from T6 (only the `.tsv` is
needed here). Point `--tsv-dir` at a directory containing them:

```bash
python define_housekeeping.py \
    --tsv-dir /path/to/bakta_outputs_tsv \
    --out-dir out
```

The script prints a report and runs sanity checks (no genome should have zero
housekeeping genes; the core families must be present in ~all genomes).

## Output schema — `housekeeping_genes.tsv`

Tab-separated, Unix line endings, one row per selected CDS locus:

| column | meaning |
|---|---|
| `genome` | assembly accession (from the TSV filename, e.g. `GCF_000143685.1_ASM14368v1`) |
| `contig` | Bakta `Sequence Id` (the contig the gene sits on) |
| `locus_tag` | Bakta locus tag (stable per-gene key) |
| `gene` | Bakta gene symbol (often empty — do not rely on it) |
| `product` | Bakta product description (what selection was based on) |
| `start`, `stop` | 1-based inclusive coordinates on `contig` |
| `strand` | `+` / `-` |
| `hk_category` | one of the 8 families (see definition doc) |
| `tier` | `1` = core/highest-confidence, `2` = strong housekeeping |

## How T12 / T13 consume this

- **T12 (operon detection):** group these loci by `genome` + `contig` + `strand`,
  order by `start`, and keep the **operon leader** (first gene of a co-directional
  run). Apply the contig-end exclusion filter (drop loci whose upstream region is
  truncated by a contig boundary). Multi-copy families (ribosomal proteins, aaRS)
  are flagged here in full; operon detection collapses each cluster to its leader,
  so the ~137 loci/genome reduce to a much smaller set of promoter-bearing leaders.
- **T13 (upstream extraction):** for each retained leader, take 100–300 bp 5′ of
  `start` (for `+`) or 3′ of `stop` (for `-`), again excluding contig-truncated
  windows.

## Key decisions / assumptions

- **Selection is by `product`, not `gene`** — required by the light DB (sparse
  symbols, populated products). [Recorded in CLAUDE.md §8.]
- **High precision over recall** — these regions become positives, so regulated
  or ambiguous genes are excluded even at the cost of fewer hits. Sigma factors,
  heat-shock chaperones, and RecA are deliberately out (regulated promoters); see
  the definition doc.
- **CDS only** — rRNA/tRNA promoters are atypical and left out of the default set.
- The set is built on the **166 training genomes**; the mining target
  *C. butyricum* DSM 10702 is excluded from training and is not in this input.
- Coordinates are passed straight from Bakta (1-based inclusive). T13 must apply
  the strand convention above when extracting upstream windows.
