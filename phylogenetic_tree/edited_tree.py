from ete3 import Tree, TreeStyle, TextFace
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
GENOME_LIST = "/Users/ishikulshrestha/Documents/GitHub/cb-promoter-mining/data-acquisition-and-curation/T5/genome_list.tsv"

# accession -> species name (e.g. "GCF_000007625.1" -> "Clostridium tetani E88")
df = pd.read_csv(GENOME_LIST, sep="\t", header=None, names=["accession", "species"])
acc2species = dict(zip(df["accession"], df["species"]))

tree = Tree(str(HERE / "bac120.tree"), format=1, quoted_node_names=True)


def leaf_accession(leaf_name):
    """GTDB leaves are prefixed with 'RS_' (RefSeq) or 'GB_' (GenBank);
    strip that so 'RS_GCF_000007625.1' -> 'GCF_000007625.1'."""
    for prefix in ("RS_", "GB_"):
        if leaf_name.startswith(prefix):
            return leaf_name[len(prefix):]
    return leaf_name


# keep only the leaves whose accession is in our Clostridium list
keep = [leaf for leaf in tree.get_leaves() if leaf_accession(leaf.name) in acc2species]
missing = set(acc2species) - {leaf_accession(l.name) for l in keep}
print(f"Keeping {len(keep)} / {len(acc2species)} genomes ({len(missing)} not in bac120.tree)")
for acc in sorted(missing):
    print(f"  not in tree: {acc}\t{acc2species[acc]}")

tree.prune(keep, preserve_branch_length=True)

# leaf.name stays the accession (the identifier); species is attached for display
for leaf in tree.get_leaves():
    leaf.add_feature("species", acc2species[leaf_accession(leaf.name)])

# --- the "real" species-labelled tree ---
# 1) a newick with species names as the leaf labels
species_tree = tree.copy()
for leaf in species_tree.get_leaves():
    leaf.name = acc2species[leaf_accession(leaf.name)]
species_tree.write(outfile=str(HERE / "results" / "clostridium_species.nwk"),
                   format=1, quoted_node_names=True)


# 2) a rendered PDF showing species names instead of accessions
def layout(node):
    if node.is_leaf():
        node.add_face(TextFace(node.species, fsize=9), column=0, position="branch-right")


ts = TreeStyle()
ts.show_leaf_name = False   # hide the accession; show the species face instead
ts.layout_fn = layout
tree.render(str(HERE / "results" / "clostridium_species.pdf"), tree_style=ts)

print("Wrote results/clostridium_species.nwk and results/clostridium_species.pdf")
