# HKUST iGEM 2026 — Wiki draft

Draft content for the iGEM public wiki, one Markdown file per standard page. Paste each into the matching
page (standard URL pattern `2026.igem.wiki/hkust/<slug>`), or import into a GitHub/Notion wiki as-is.

## Pages

| File | Wiki slug | Status | Medal relevance |
|---|---|---|---|
| `home.md` | `/` | ✅ filled | — |
| `description.md` | `/description` | ✅ filled | Bronze (Project Description) |
| `design.md` | `/design` | ✅ filled (wet-lab section = placeholder) | — |
| `engineering.md` | `/engineering` | ✅ filled | Silver (Engineering Success) |
| `model.md` | `/model` | ✅ filled | — |
| `results.md` | `/results` | ✅ filled | — |
| `proof-of-concept.md` | `/proof-of-concept` | ✅ filled (wet-lab layer = placeholder) | — |
| `contribution.md` | `/contribution` | ✅ filled | Bronze/Silver (Contribution) |
| `human-practices.md` | `/human-practices` | 🟡 scaffold | Silver (required) |
| `education.md` | `/education` | 🟡 scaffold | — |
| `safety.md` | `/safety` | 🟡 scaffold + seed | Safety Form also required |
| `notebook.md` | `/notebook` | 🟡 dry-lab filled, wet-lab placeholder | — |
| `team.md` | `/team` | 🟡 dry-lab filled, rest placeholder | — |
| `attributions.md` | `/attributions` | 🟡 dry-lab filled, rest placeholder | Bronze (required) |

**Filled (✅)** = written from our actual dry-lab work. **Scaffold/placeholder (🟡)** = structure + prompts
for the team (mainly wet-lab, human practices, education, and personal details) to complete.

## What still needs the team

- **Wet-lab everything:** validation experiments, protocols, notebook, and the wet-lab halves of Design,
  Proof of Concept, Team, and Attributions.
- **Human Practices** and **Education** — these are engagement activities, so only the team can fill them
  (Human Practices is a Silver requirement).
- **Safety** — complete with your biosafety committee; note the strain-provenance flag in `safety.md`.
- **People** — full names, bios, PIs/advisors, and acknowledgements in `team.md`.
- **Attributions** — verify each contribution, add the software/version list, and also submit via the
  official iGEM **Attributions Form** (the wiki page alone is not enough).

## Notes on accuracy

- Dry-lab numbers (model metrics, the 929 / 180 / 45 candidate tiers, genome-scale precision) come
  straight from our project records and code — but **re-check them against your latest run** before
  publishing, as some evaluation is still being finalized. The tiers are the re-mine on the complete
  reference assembly; an earlier draft-assembly mine reported 912 / 306 / 100 and is superseded.
- Medal criteria change year to year. Confirm the exact 2026 required pages and criteria on the official
  [iGEM Medals page](https://competition.igem.org/judging/medals) before you rely on the mapping above.
- Tone is written for a **public/judging** audience (biology explained, HPC/ops detail kept light). The
  deeper technical reference lives in the repository's documentation.

## Suggested build order

1. Fill `team.md` and `attributions.md` (Bronze basics).
2. Complete `human-practices.md` (Silver).
3. Add wet-lab content across `notebook.md`, `design.md`, `proof-of-concept.md`.
4. Polish `home.md` / `description.md` for the public narrative, add figures.
5. Complete `safety.md` and submit the Safety Form.
