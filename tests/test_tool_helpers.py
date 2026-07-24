from __future__ import annotations

import unittest

from tools._shared import ToolInputError, as_non_negative_int, parse_strict_json


class ToolHelperTests(unittest.TestCase):
    def test_strict_json_rejects_duplicates_and_non_finite_numbers(self) -> None:
        for raw in ('{"x":1,"x":2}', "NaN", "Infinity", "1e9999"):
            with self.subTest(raw=raw), self.assertRaises(ToolInputError):
                parse_strict_json(raw, "value")

    def test_integer_coercion_is_bounded_and_finite(self) -> None:
        self.assertEqual(as_non_negative_int(3.0, "value"), 3)
        self.assertEqual(as_non_negative_int(3, "value"), 3)
        for value in (True, -1, 1.5, float("inf"), 10**5000):
            with self.subTest(value=type(value).__name__), self.assertRaises(ToolInputError):
                as_non_negative_int(value, "value", maximum=10)


if __name__ == "__main__":
    unittest.main()
