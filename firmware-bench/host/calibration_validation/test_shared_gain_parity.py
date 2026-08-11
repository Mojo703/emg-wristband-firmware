import unittest

import numpy as np

from experiment_19_shared_gain_parity import gain_delta_summary


class SharedGainParityTests(unittest.TestCase):
    def test_gain_delta_summary_covers_every_original_vector(self):
        shared = np.array([1.0, 2.0])
        originals = np.array([[1.0, 2.0], [3.0, -2.0]])

        rms, maximum = gain_delta_summary(shared, originals)

        self.assertAlmostEqual(rms, np.sqrt(5.0))
        self.assertEqual(maximum, 4.0)

    def test_gain_delta_summary_rejects_mismatched_vectors(self):
        with self.assertRaises(ValueError):
            gain_delta_summary(np.ones(2), np.ones((3, 4)))


if __name__ == "__main__":
    unittest.main()
