from ete3 import Tree
import pandas as pd
from pathlib import Path

clostridium_genomes = pd.read_csv("/Users/ishikulshrestha/Documents/GitHub/cb-promoter-mining/data-acquisition-and-curation/T5/genome_list.tsv", sep="\t")[]

tree = Tree("bac120.tree", format=1, quoted_node_names=True)
for leaf in tree.get_leaves()[:20]:
    print(leaf.name)

# tree.prune = (["g__Clostridium", "s__Bacillus subtillus", "s__Bacillus licheniformis", "s__Ruminiclostridium cellulolyticum","s__Lachnoclostridium phytofermentans","s__Escherichia coli",])
