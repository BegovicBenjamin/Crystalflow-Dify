from __future__ import annotations

import unittest
from pathlib import Path

from crystalflow.canonical import canonical_loads
from crystalflow.engine import run, validate_program

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class ExampleTests(unittest.TestCase):
    def test_documented_examples_pass_their_vectors(self) -> None:
        for stem in ("invoice_total", "sla_classification"):
            with self.subTest(stem=stem):
                program = canonical_loads(
                    (EXAMPLES / f"{stem}.program.json").read_text(encoding="utf-8")
                )
                tests = canonical_loads(
                    (EXAMPLES / f"{stem}.tests.json").read_text(encoding="utf-8")
                )
                validate_program(program)
                for case in tests:
                    self.assertEqual(
                        run(program, case["input"]),
                        case["expected"],
                        case["name"],
                    )


if __name__ == "__main__":
    unittest.main()
