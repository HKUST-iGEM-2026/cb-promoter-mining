# Attributions

> **Standard URL:** `2026.igem.wiki/hkust/attributions`
> **Medal relevance:** Bronze (required — you must state what your team did and what others did)

> **Note.** iGEM requires that this content also be submitted via the official **Attributions Form**, not
> only on the wiki. Use this page as the working draft, then transfer it. Every claim below should be
> verified by the person named.

## Work done by our team

### Dry lab — promoter mining
The complete computational pipeline was designed, built, and run by our team: data curation, dataset
construction, training of all three models, the three-regime evaluation, calibration and false-discovery
control, and the genome-wide mine of *C. butyricum* that produced the candidate lists. See [Team](team.md)
for individual ownership and [Model](model.md) for the method.

### Wet lab
> **Placeholder — wet-lab team.** Describe the experimental work done by team members.

## Work done by others (and properly credited)

Our project builds on public data, tools, and models created by others. We used them under their
licenses and cite them here and on the relevant pages:

**Pre-trained models / foundation models**
- **ProkBERT** (Neural Bioinformatics Research Group) — base language model we fine-tuned (CC-BY-NC-4.0).
  Our fine-tune inherits those non-commercial terms.
- **Evo2** (Arc Institute) — frozen DNA foundation model we built a probe on (Apache-2.0). We publish
  only our own small probe, never a copy of the Evo2 weights.
- **Nucleotide Transformer** (InstaDeep) — evaluated and ultimately not deployed.

Our own released models are at [huggingface.co/CloBiT](https://huggingface.co/CloBiT); each model card
states its license and credits its base model and training data.

**Promoter / TSS datasets**
- **Hocq R, Jagtap S, Boutard M, Tolonen AC, Duval L, Pirayre A, Lopes Ferreira N, Wasels F.** Genome-Wide
  TSS Distribution in Three Related Clostridia with Normalized Capp-Switch Sequencing. *Microbiol Spectr.*
  2022;10(2):e0228821. doi:10.1128/spectrum.02288-21 — our in-genus training positives
  (*C. acetobutylicum*, *C. beijerinckii*).
- **Al-Bassam MM, Kim JN, Zaramela LS, Kellman BP, Zuniga C, Wozniak JM, Gonzalez DJ, Zengler K.**
  Optimization of carbon and energy utilization through differential translational efficiency.
  *Nat Commun.* 2018;9(1):4474. doi:10.1038/s41467-018-06993-6 — the *C. ljungdahlii* benchmark truth.
- **Xu C, Huang R, Teng L, et al.** Cellulosome stoichiometry in *Clostridium cellulolyticum* is regulated
  by selective RNA processing and stabilization. *Nat Commun.* 2015;6:6900. doi:10.1038/ncomms7900 — the
  *R. cellulolyticum* H10 TSS table (BioProject PRJNA247408).
- **RegulonDB** v13.5 (*E. coli* K-12 promoters; Salgado H, Gama-Castro S, et al., *Nucleic Acids Research*
  2023, doi:10.1093/nar/gkad1072) — free for academic / non-commercial use.
- **DBTBS**, the *B. subtilis* transcriptional regulation database (dbtbs.hgc.jp).
- **Wiegand S, Dietrich S, Hertel R, Bongaerts J, et al.** RNA-Seq of *Bacillus licheniformis*: active
  regulatory RNA features expressed within a productive fermentation. *BMC Genomics.* 2013;14:667.
  doi:10.1186/1471-2164-14-667 — the *B. licheniformis* DSM 13 TSS table (BioProject PRJNA188930).
- **de Souza Pinto Lemgruber R, Valgepea K, Gonzalez Garcia RA, de Bakker C, et al.** A TetR-Family
  Protein (CAETHG_0459) Activates Transcription From a New Promoter Motif Associated With Essential
  Genes for Autotrophic Growth in Acetogens. *Front Microbiol.* 2019;10:2549.
  doi:10.3389/fmicb.2019.02549 — the *C. autoethanogenum* benchmark truth (BioProject PRJNA428323).
- Genome assemblies from **NCBI RefSeq** (166 *Clostridium* reference genomes, taxon 1485).

**Software / tools**
- Genome annotation, sequence-clustering, motif-search, and workflow tools; the deep-learning and
  scientific-Python ecosystems. *(Full tool list and versions to be enumerated here.)*

**Compute**
- High-performance computing provided by **HKUST** (SuperPOD/HPC facility).

## Guidance and support

> **Placeholder.** Advice from PIs, instructors, advisors, other labs, or other iGEM teams — state what
> guidance was given and by whom, keeping it distinct from the team's own work.

## Verification checklist (before submission)

- [ ] Every named contribution confirmed by that person
- [ ] Wet-lab attributions completed
- [ ] Full software list with versions
- [ ] All external datasets/models cited with licenses
- [ ] Content transferred to the official iGEM Attributions Form
