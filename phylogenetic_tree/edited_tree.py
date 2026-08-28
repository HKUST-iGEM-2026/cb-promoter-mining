from ete3 import Tree, TreeStyle, TextFace, NodeStyle
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
GENOME_LIST = "/project/igem2026/cb-promoter-mining/data-acquisition-and-curation/T5/genome_list.tsv"
REMAP = HERE / "gtdb_remap.tsv"   # our_accession -> GTDB tree_leaf (built once from bac120_taxonomy_r226)

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


tree_accs = {leaf_accession(l.name) for l in tree.get_leaves()}

# --- decide which tree leaf each entry maps to, and how to label/colour it ---
# Only ~121 of our 166 assemblies are leaves in bac120.tree. gtdb_remap.tsv maps the other 45
# (+ the cross-species training donors + the target) onto the GTDB representative genome of the
# SAME species — a real, already-placed leaf, relabelled with our name. First writer of a leaf
# wins; a later entry landing on a claimed leaf is a GTDB "same-species" merge and is only noted.
CATCOLOR = {"exact": "black", "remap45": "black",
            "donor": "#1f6fb4", "target": "#d1332e"}   # donors blue, target red
leaf2label, leaf2color = {}, {}
merged, unplaced = [], []          # for the notes block

# 1) genomes whose exact assembly IS a leaf in bac120.tree (~121 of the 166)
for acc, sp in acc2species.items():
    if acc in tree_accs:
        leaf2label[acc] = sp
        leaf2color[acc] = CATCOLOR["exact"]

# 2) remapped Clostridium + cross-species donors + the target
#    (process remap45, then donor, then target, so an exact tip always wins a shared leaf)
if REMAP.exists():
    rmp = pd.read_csv(REMAP, sep="\t", dtype=str).fillna("")
    order = {"remap45": 0, "donor": 1, "target": 2}
    rmp = rmp.sort_values(by="category", key=lambda c: c.map(lambda x: order.get(x, 9)),
                          kind="mergesort")
    for r in rmp.itertuples(index=False):
        leaf, label, cat, gsp = r.tree_leaf, r.our_name, r.category, r.gtdb_species
        if not leaf:
            unplaced.append((label, r.note or "no GTDB leaf"))
            continue
        if leaf in leaf2label:
            merged.append((label, leaf2label[leaf], gsp))   # collapses onto an existing tip
            continue
        leaf2label[leaf] = label
        leaf2color[leaf] = CATCOLOR.get(cat, "black")

# --- prune the big tree down to just our leaves ---
keep = [l for l in tree.get_leaves() if leaf_accession(l.name) in leaf2label]
tree.prune(keep, preserve_branch_length=True)

# tip background: in-genus Clostridium light green, target light red, donors none
BG = {"black": "#e9f5e9", CATCOLOR["target"]: "#fbe9e7", CATCOLOR["donor"]: "#ffffff"}
for leaf in tree.get_leaves():
    acc = leaf_accession(leaf.name)
    leaf.add_feature("label", leaf2label[acc])
    leaf.add_feature("color", leaf2color[acc])
    ns = NodeStyle()
    ns["bgcolor"] = BG.get(leaf2color[acc], "#ffffff")
    ns["size"] = 0
    leaf.set_style(ns)

# --- the "real" species-labelled tree ---
(HERE / "results").mkdir(exist_ok=True)   # recreate if the results/ folder was deleted
# 1) a newick with species names as the leaf labels
species_tree = tree.copy()
for leaf in species_tree.get_leaves():
    leaf.name = leaf2label[leaf_accession(leaf.name)]
species_tree.write(outfile=str(HERE / "results" / "clostridium_species.nwk"),
                   format=1, quoted_node_names=True)


# 2) a rendered PDF showing species names (target red, cross-species donors blue)
def layout(node):
    if node.is_leaf():
        node.add_face(TextFace(node.label, fsize=9, fgcolor=node.color),
                      column=0, position="branch-right")


ts = TreeStyle()
ts.mode = "c"
ts.arc_start = -180
ts.arc_span = 180
ts.show_leaf_name = False   # hide the accession; show the species face instead
ts.layout_fn = layout
tree.render(str(HERE / "results" / "clostridium_species.pdf"), tree_style=ts)

# --- report: tips shown + the issues to drop into the figure notes ---
n_donor = sum(1 for c in leaf2color.values() if c == CATCOLOR["donor"])
print(f"Tips shown: {len(keep)}  (exact + remapped Clostridium + {n_donor} donors + target)")
print(f"\nMERGED — GTDB treats as the SAME species as an already-shown tip, so NOT shown separately: {len(merged)}")
for name, onto, gsp in merged:
    print(f"    {name}  ==  {onto}   [GTDB s__{gsp}]")
print(f"\nTOO NEW — not in this GTDB tree at all, so NOT shown: {len(unplaced)}")
for name, why in unplaced:
    print(f"    {name}   [{why}]")

print("\nWrote results/clostridium_species.nwk and results/clostridium_species.pdf")
