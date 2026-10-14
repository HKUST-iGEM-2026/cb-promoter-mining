import unittest

import numpy as np

from analyze_oxidative_motifs import BASES, bh_qvalues, empirical_p, shuffled_sequences, window_scores


class MotifAnalysisTests(unittest.TestCase):
    def test_maximum_includes_last_possible_window(self):
        encoded = np.array([BASES.index(base) for base in "CCTATAAT"], dtype=np.uint8)
        weights = np.zeros((6, 4))
        for position, base in enumerate("TATAAT"):
            weights[position, BASES.index(base)] = 1
        scores = window_scores(encoded, weights)[0]
        self.assertEqual(int(np.argmax(scores)), 2)
        self.assertEqual(float(scores[2]), 6.0)

    def test_each_shuffle_preserves_all_base_counts(self):
        encoded = np.array([BASES.index(base) for base in "AAATTCGG"], dtype=np.uint8)
        shuffled = shuffled_sequences(encoded, 50, np.random.default_rng(7))
        expected = np.bincount(encoded, minlength=4)
        for sequence in shuffled:
            np.testing.assert_array_equal(np.bincount(sequence, minlength=4), expected)

    def test_null_ties_are_counted_conservatively(self):
        self.assertEqual(empirical_p(5.0, np.array([5.0, 5.0, 5.0])), 1.0)
        self.assertEqual(empirical_p(5.0, np.array([4.0, 5.0, 6.0])), 0.75)

    def test_adjustment_preserves_original_comparison_order(self):
        np.testing.assert_allclose(bh_qvalues([0.02, 0.001, 0.05]), [0.03, 0.003, 0.05])


if __name__ == "__main__":
    unittest.main()
