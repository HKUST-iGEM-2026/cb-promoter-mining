import tempfile
from pathlib import Path
import unittest

from export_gene_context import CodingLocus, downstream_unit, load_coding_loci


def locus(tag, start, end, strand):
    return CodingLocus("contig", strand, start, end, tag, "", "", (), ((start, end),))


class GeneContextTests(unittest.TestCase):
    def test_intervening_gap_boundary_on_both_strands(self):
        for strand in ("+", "-"):
            for gap in (100, 101):
                with self.subTest(strand=strand, gap=gap):
                    left = locus("left", 200, 300, strand)
                    right = locus("right", 301 + gap, 400 + gap, strand)
                    members = downstream_unit([left, right], 100 if strand == "+" else 1000, strand, 100)
                    expected = ["left", "right"] if strand == "+" else ["right", "left"]
                    self.assertEqual([member.locus_tag for member, _ in members], expected if gap == 100 else expected[:1])
                    if gap == 100:
                        self.assertEqual(members[1][1], 100)

    def test_opposite_strand_interrupts_unit(self):
        loci = [locus("first", 200, 300, "+"), locus("opposite", 310, 400, "-"), locus("last", 410, 500, "+")]
        self.assertEqual([member.locus_tag for member, _ in downstream_unit(loci, 100, "+", 100)], ["first"])

    def test_split_cds_is_one_gene_and_preserves_segments(self):
        text = (
            "contig\tRefSeq\tCDS\t900\t1000\t.\t-\t0\tID=cds-protein;Parent=gene-split;locus_tag=split;gene=release;product=release%20factor;protein_id=protein\n"
            "contig\tRefSeq\tCDS\t700\t898\t.\t-\t0\tID=cds-protein;Parent=gene-split;locus_tag=split;gene=release;product=release%20factor;protein_id=protein\n"
            "contig\tRefSeq\tCDS\t500\t599\t.\t-\t0\tID=cds-other;Parent=gene-other;locus_tag=other;product=other;protein_id=other\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotation.gff"
            path.write_text(text)
            by_contig, _ = load_coding_loci(path)
        members = downstream_unit(by_contig["contig"], 1100, "-", 100)
        self.assertEqual([member.locus_tag for member, _ in members], ["split", "other"])
        self.assertEqual(members[0][0].cds_segments, ((700, 898), (900, 1000)))
        self.assertEqual(members[0][0].protein_ids, ("protein",))
        self.assertEqual(members[0][0].product, "release factor")
        self.assertEqual(members[1][1], 100)


if __name__ == "__main__":
    unittest.main()
