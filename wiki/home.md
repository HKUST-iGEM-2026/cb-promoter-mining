# HKUST iGEM 2026 — Home

> **Standard URL:** `2026.igem.wiki/hkust/`
> **One line:** We built a computational pipeline that reads the genome of the industrially and
> medically important bacterium *Clostridium butyricum* and hands wet-lab biologists a ranked
> shortlist of **promoters** — the genetic "on-switches" needed to engineer it.

---

## The problem in one paragraph

To engineer any organism, you need **characterized genetic parts** — and the most fundamental part is a
**promoter**, the short stretch of DNA that tells the cell how strongly to switch a gene on. Model
organisms like *E. coli* have thousands of well-characterized promoters. Our chassis of interest,
*Clostridium butyricum*, has almost none. That missing toolkit is the bottleneck that stops teams from
engineering it. Finding promoters by hand in the lab is slow and expensive, so we asked: **can we mine
them straight out of the genome with machine learning, and give the wet lab a validated shortlist to
test?**


![Pipeline overview, phases A to E](figures/F15_pipeline_overview.png)

***Figure 1.** The five phases of the pipeline. The target genome enters only at the mining stage — it is excluded from every training step, which is what makes mining it a genuine prediction rather than recall.*

## What we did

We trained an **ensemble of three DNA machine-learning models** to recognize promoters, combined their
votes, and slid them across the *C. butyricum* genome. The pipeline narrows well over a hundred thousand
candidate windows down to **929 high-confidence promoter candidates**, of which **45 carry the textbook
two-box promoter signature** — the shortlist we hand to the wet lab.

The hard part, and the thing that makes this a real research contribution: *Clostridium* genomes are
extremely **AT-rich** (only ~28–32% G+C), and the promoter signal itself is AT-rich, so naive methods
drown in false positives. Much of our design exists to beat this **"AT-trap."**

## Explore the wiki

| Page | What's there |
|---|---|
| [Description](description.md) | The problem, why *C. butyricum*, our goals, and the project plan |
| [Design](design.md) | How we designed the solution — the ensemble approach and the biology behind it |
| [Engineering](engineering.md) | Our Design–Build–Test–Learn cycles: what failed, what we changed |
| [Model](model.md) | The full computational pipeline and the three models (our core dry-lab work) |
| [Results](results.md) | Model performance and the *C. butyricum* candidate promoters |
| [Proof of Concept](proof-of-concept.md) | Evidence the pipeline works end-to-end |
| [Contribution](contribution.md) | What we give future teams: open models, code, and datasets |
| [Human Practices](human-practices.md) · [Education](education.md) · [Safety](safety.md) | Values, outreach, and responsibility |
| [Notebook](notebook.md) · [Team](team.md) · [Attributions](attributions.md) | Timeline, people, and credit |

## Project at a glance

- **Chassis / target:** *Clostridium butyricum* DSM 10702 (a probiotic, butyrate-producing anaerobe)
- **Goal (Stage 1):** a genome-wide, ranked catalogue of **constitutive promoters**
- **Method:** 3-model ML ensemble (a fine-tuned genomic language model + a custom neural network + a
  foundation-model probe), combined by rank and filtered with statistical false-discovery control and
  motif confirmation
- **Headline result:** 929 high-confidence candidates → **45 canonical promoter candidates** for
  wet-lab characterization
- **Open contribution:** three models published openly on Hugging Face at
  [huggingface.co/CloBiT](https://huggingface.co/CloBiT), plus a fully documented, reusable pipeline
