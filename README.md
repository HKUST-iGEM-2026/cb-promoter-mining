# Genome Mining Project for Clostridium Butyricum DSM 10702 Strain

This project uses 2 fine-tuned genomic language models ([ProkBERT-mini](https://github.com/nbrg-ppcu/prokbert) and [Nucleotide Transformer](https://github.com/instadeepai/nucleotide-transformer)), in addition to a custom trained CNN for genomic minimg of Clostridium Butyricum to discover new and novel parts (only promoter for now) as well as their strengths.

## How the Project Works

```mermaid
flowchart TB
  subgraph IN[" Step 1 · Gather the raw data "]
    direction LR
    A["Clostridium DNA<br/>the genome we want to search"]
    B["Known promoters<br/>from other, well-studied bacteria"]
    C["Lab experiment data<br/>showing where genes switch on"]
  end

  D["Step 2 · Build a training set<br/>turn the raw data into clear examples:<br/>real promoters vs. look-alike decoys"]
  E["Step 3 · Train the AI<br/>teach three models to recognise<br/>what a promoter looks like"]
  F["Step 4 · Check the models work<br/>test their accuracy and make sure<br/>their confidence scores are trustworthy"]
  G["Step 5 · Scan the target genome<br/>slide across the DNA and score<br/>every position for 'is this a promoter?'"]
  H["Step 6 · Final result<br/>a ranked, quality-checked list<br/>of candidate promoters"]

  A --> D
  B --> D
  C --> D
  D --> E --> F --> G --> H

  classDef input fill:#eaf2ff,stroke:#3b6fb5,color:#111;
  classDef goal fill:#d9f2d9,stroke:#27ae60,color:#111;
  class A,B,C input;
  class H goal;
```

### Why three models?
Three different AI models each score the DNA independently, and we combine their votes — if they agree, we're far more confident a spot is a real promoter. We also run experimental and statistical checks so the final list is reliable and not just a guess.

