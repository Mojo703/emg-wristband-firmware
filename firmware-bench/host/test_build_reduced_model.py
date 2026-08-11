import unittest

import numpy as np

from build_reduced_model import (
    DEFAULT_ACTIVE, DEFAULT_TARGET_LABELS, Source, filter_by_source, read_sources, scale_bits,
    set_quantization_constant, validate_active,
)


class ReducedModelTests(unittest.TestCase):
    def test_filters_inside_each_source_and_preserves_breakdown(self):
        rows = np.arange(8 * 2, dtype=np.float32).reshape(8, 2)
        labels = np.array([2, 3, 7, 8, 1, 6, 10, 11], np.int32)
        sources = [Source("live", "command", 0, 4),
                   Source("prior", "rest", 4, 4)]
        kept, mapped, breakdown = filter_by_source(
            rows, labels, sources, DEFAULT_ACTIVE
        )
        self.assertEqual(mapped.tolist(), list(DEFAULT_TARGET_LABELS))
        self.assertEqual([entry["rows"] for entry in breakdown], [4, 2])
        np.testing.assert_array_equal(kept, rows[[0, 1, 2, 3, 6, 7]])

    def test_source_totals_are_checked(self):
        with self.assertRaisesRegex(ValueError, "describe 4 rows"):
            read_sources({"training_sources": [
                {"session": "a", "role": "x", "rows": 4}
            ]}, 5)

    def test_active_layout_requires_commands_and_final_rest(self):
        validate_active(DEFAULT_ACTIVE)
        with self.assertRaisesRegex(ValueError, "retain static and moving rest"):
            validate_active((2, 3, 7, 8, 9, 10))

    def test_emitted_quantization_bits_are_the_fit_scale(self):
        constants = {"row_quantization": {
            "scale": 99.0, "scale_bits": "0xdeadbeef",
        }}
        fit_scale = np.float32(10.0 / 127.0)
        set_quantization_constant(constants, fit_scale)
        self.assertEqual(constants["row_quantization"]["scale_bits"],
                         scale_bits(fit_scale))
        emitted = np.float32(constants["row_quantization"]["scale"])
        self.assertEqual(emitted.view(np.uint32), fit_scale.view(np.uint32))


if __name__ == "__main__":
    unittest.main()
