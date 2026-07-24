from __future__ import annotations

import unittest

from crystalflow.canonical import (
    canonical_json,
    canonical_json_bytes,
    canonical_loads,
    canonical_sha256,
)
from crystalflow.errors import CanonicalizationError


class CanonicalJSONTests(unittest.TestCase):
    def test_objects_are_sorted_and_compact(self) -> None:
        value = {"z": [True, None], "a": {"b": 2, "a": 1}}
        self.assertEqual(
            canonical_json(value),
            '{"a":{"a":1,"b":2},"z":[true,null]}',
        )

    def test_numbers_are_normalized(self) -> None:
        self.assertEqual(canonical_json(-0.0), "0")
        self.assertEqual(canonical_json(1.0), "1")
        self.assertEqual(canonical_json(1e-7), "1e-7")
        self.assertEqual(canonical_json(1e-6), "0.000001")

    def test_utf8_bytes_and_hash_are_stable(self) -> None:
        self.assertEqual(canonical_json_bytes({"é": "雪"}), '{"é":"雪"}'.encode())
        self.assertEqual(
            canonical_sha256({"b": 2, "a": 1}),
            "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777",
        )

    def test_loader_rejects_duplicate_keys(self) -> None:
        with self.assertRaises(CanonicalizationError) as caught:
            canonical_loads('{"x":1,"x":2}')
        self.assertEqual(caught.exception.code, "DUPLICATE_KEY")

    def test_non_finite_and_non_json_values_are_rejected(self) -> None:
        with self.assertRaises(CanonicalizationError) as caught:
            canonical_json(float("nan"))
        self.assertEqual(caught.exception.code, "NON_FINITE_NUMBER")
        with self.assertRaises(CanonicalizationError):
            canonical_json({1, 2})

    def test_cycles_are_rejected(self) -> None:
        value: list[object] = []
        value.append(value)
        with self.assertRaises(CanonicalizationError) as caught:
            canonical_json(value)
        self.assertEqual(caught.exception.code, "CYCLIC_VALUE")


if __name__ == "__main__":
    unittest.main()
