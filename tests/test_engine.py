from __future__ import annotations

import unittest
from unittest.mock import patch

from crystalflow.engine import (
    CrystalFlowEngine,
    ResourceLimits,
    evaluate,
    run,
    validate_expression,
    validate_program,
)
from crystalflow.errors import (
    DSLValidationError,
    EvaluationError,
    InputValidationError,
    OutputValidationError,
    ResourceLimitError,
)


def literal(value: object) -> dict[str, object]:
    return {"op": "literal", "value": value}


def args(op: str, *values: object) -> dict[str, object]:
    return {"op": op, "args": list(values)}


class AllocationBombString(str):
    """A string that proves expanding operations were rejected before calling C."""

    def replace(self, old: str, new: str, count: int = -1) -> str:
        raise AssertionError("replace allocated before the output preflight")

    def join(self, iterable: object) -> str:
        raise AssertionError("join allocated before the output preflight")

    def split(self, separator: str | None = None, maxsplit: int = -1) -> list[str]:
        raise AssertionError("split allocated before the output preflight")


class CountingUpperString(str):
    calls = 0

    def upper(self) -> str:
        type(self).calls += 1
        if type(self).calls > 3:
            raise AssertionError("map accumulated generated strings past its output budget")
        return super().upper()


class ValidationTests(unittest.TestCase):
    def test_unknown_operations_and_keys_are_rejected(self) -> None:
        with self.assertRaises(DSLValidationError) as caught:
            validate_expression({"op": "shell", "args": []})
        self.assertEqual(caught.exception.code, "UNKNOWN_OPERATION")
        with self.assertRaises(DSLValidationError) as caught:
            validate_expression({"op": "literal", "value": 1, "typo": True})
        self.assertEqual(caught.exception.code, "UNKNOWN_KEY")

    def test_arity_and_scopes_are_static(self) -> None:
        with self.assertRaises(DSLValidationError) as caught:
            validate_expression(args("add", literal(1)))
        self.assertEqual(caught.exception.code, "INVALID_ARITY")
        with self.assertRaises(DSLValidationError) as caught:
            validate_expression({"op": "var", "name": "item"})
        self.assertEqual(caught.exception.code, "UNBOUND_VARIABLE")

    def test_program_envelope_is_strict_and_versioned(self) -> None:
        validate_program({"version": 1, "expression": literal(1)})
        with self.assertRaises(DSLValidationError) as caught:
            validate_program({"version": "1", "expression": literal(1)})
        self.assertEqual(caught.exception.code, "UNSUPPORTED_VERSION")
        with self.assertRaises(DSLValidationError) as caught:
            validate_program({"version": 1, "expression": literal(1), "unexpected": 2})
        self.assertEqual(caught.exception.code, "UNKNOWN_KEY")


class EvaluationTests(unittest.TestCase):
    def test_value_construction_input_get_and_default(self) -> None:
        expression = {
            "op": "object",
            "fields": {
                "name": {"op": "input", "name": "name"},
                "first": {
                    "op": "get",
                    "target": {"op": "input", "name": "values"},
                    "key": literal(0),
                },
                "fallback": {
                    "op": "get",
                    "target": {"op": "input"},
                    "key": literal("missing"),
                    "default": literal("n/a"),
                },
            },
        }
        self.assertEqual(
            evaluate(expression, {"name": "Ada", "values": [3, 4]}),
            {"name": "Ada", "first": 3, "fallback": "n/a"},
        )

    def test_arithmetic(self) -> None:
        cases = [
            (args("add", literal(2), literal(3), literal(4)), 9),
            (args("sub", literal(7), literal(2)), 5),
            (args("mul", literal(2), literal(3), literal(4)), 24),
            (args("div", literal(7), literal(2)), 3.5),
            (args("mod", literal(7), literal(3)), 1),
            (args("pow", literal(2), literal(8)), 256),
            (args("round", literal(2.675), literal(2)), 2.67),
            (args("min", literal(7), literal(2), literal(4)), 2),
            (args("max", literal(7), literal(2), literal(4)), 7),
            (args("abs", literal(-3)), 3),
        ]
        for expression, expected in cases:
            with self.subTest(expression=expression):
                self.assertEqual(evaluate(expression), expected)

    def test_strings_and_slices(self) -> None:
        cases = [
            (args("concat", literal("a"), literal("b")), "ab"),
            (args("lower", literal("AÉ")), "aé"),
            (args("upper", literal("aé")), "AÉ"),
            (args("strip", literal(" x \n")), "x"),
            (args("replace", literal("a-b-a"), literal("a"), literal("x")), "x-b-x"),
            (args("split", literal("a,b"), literal(",")), ["a", "b"]),
            (args("join", literal("-"), literal(["a", "b"])), "a-b"),
            (args("length", literal({"a": 1, "b": 2})), 2),
            (args("slice", literal("abcd"), literal(1), literal(3)), "bc"),
            (args("slice", literal([0, 1, 2]), literal(1)), [1, 2]),
        ]
        for expression, expected in cases:
            with self.subTest(expression=expression):
                self.assertEqual(evaluate(expression), expected)

    def test_logic_comparison_if_and_coalesce_are_deterministic(self) -> None:
        self.assertTrue(evaluate(args("lt", literal(1), literal(2))))
        self.assertTrue(evaluate(args("eq", literal({"b": 2, "a": 1}), literal({"a": 1, "b": 2}))))
        self.assertFalse(evaluate(args("eq", literal(True), literal(1))))
        self.assertEqual(
            evaluate(
                args(
                    "if",
                    literal(False),
                    args("div", literal(1), literal(0)),
                    literal("safe"),
                )
            ),
            "safe",
        )
        self.assertEqual(
            evaluate(args("coalesce", literal(None), literal(0), literal(2))),
            0,
        )
        self.assertFalse(
            evaluate(
                args(
                    "and",
                    literal(False),
                    args("eq", args("div", literal(1), literal(0)), literal(0)),
                )
            )
        )

    def test_collection_scope_map_filter_sort_and_sum(self) -> None:
        mapped = {
            "op": "map",
            "collection": literal([1, 2, 3, 4]),
            "var": "n",
            "index": "i",
            "body": args(
                "add",
                args("mul", {"op": "var", "name": "n"}, literal(2)),
                {"op": "var", "name": "i"},
            ),
        }
        self.assertEqual(evaluate(mapped), [2, 5, 8, 11])

        filtered = {
            "op": "filter",
            "collection": literal([1, 2, 3, 4]),
            "var": "n",
            "body": args(
                "eq",
                args("mod", {"op": "var", "name": "n"}, literal(2)),
                literal(0),
            ),
        }
        self.assertEqual(evaluate(filtered), [2, 4])

        sorted_expression = {
            "op": "sort",
            "collection": literal([{"name": "b", "rank": 2}, {"name": "a", "rank": 1}]),
            "var": "row",
            "body": {
                "op": "get",
                "target": {"op": "var", "name": "row"},
                "key": literal("rank"),
            },
        }
        self.assertEqual(
            evaluate(sorted_expression),
            [{"name": "a", "rank": 1}, {"name": "b", "rank": 2}],
        )
        self.assertEqual(
            evaluate({"op": "sum", "collection": literal([1, 2.5, 3])}),
            6.5,
        )

    def test_runtime_errors_are_stable(self) -> None:
        with self.assertRaises(EvaluationError) as caught:
            evaluate(args("div", literal(1), literal(0)))
        self.assertEqual(caught.exception.code, "DIVISION_BY_ZERO")
        self.assertIn("$.args[1]", str(caught.exception))

        with self.assertRaises(EvaluationError) as caught:
            evaluate({"op": "input", "name": "missing"}, {})
        self.assertEqual(caught.exception.code, "MISSING_INPUT")


class ProgramAndBudgetTests(unittest.TestCase):
    def test_run_validates_input_and_output(self) -> None:
        program = {
            "version": 1,
            "input_schema": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
                "additionalProperties": False,
            },
            "expression": args(
                "mul",
                {"op": "input", "name": "x"},
                literal(2),
            ),
            "output_schema": {"type": "integer", "minimum": 0},
        }
        self.assertEqual(run(program, {"x": 3}), 6)
        with self.assertRaises(InputValidationError):
            run(program, {"x": "3"})
        with self.assertRaises(OutputValidationError):
            run(program, {"x": -1})

    def test_json_text_program_rejects_duplicate_keys(self) -> None:
        with self.assertRaises(Exception) as caught:
            run('{"version":1,"version":1,"expression":{"op":"literal","value":1}}')
        self.assertEqual(getattr(caught.exception, "code", None), "DUPLICATE_KEY")

    def test_node_depth_collection_output_and_power_limits(self) -> None:
        with self.assertRaises(ResourceLimitError) as caught:
            evaluate(
                args("add", literal(1), literal(2)),
                limits=ResourceLimits(max_nodes=2),
            )
        self.assertEqual(caught.exception.code, "NODE_LIMIT")

        nested = literal(1)
        for _ in range(4):
            nested = args("abs", nested)
        with self.assertRaises(ResourceLimitError) as caught:
            evaluate(nested, limits=ResourceLimits(max_depth=3))
        self.assertEqual(caught.exception.code, "DEPTH_LIMIT")

        with self.assertRaises(ResourceLimitError) as caught:
            evaluate(
                literal([1, 2, 3]),
                limits=ResourceLimits(max_collection_items=2),
            )
        self.assertEqual(caught.exception.code, "COLLECTION_LIMIT")

        with self.assertRaises(ResourceLimitError) as caught:
            evaluate(
                literal("long"),
                limits=ResourceLimits(max_output_bytes=3),
            )
        self.assertEqual(caught.exception.code, "OUTPUT_LIMIT")

        with self.assertRaises(ResourceLimitError) as caught:
            evaluate(
                args("pow", literal(2), literal(11)),
                limits=ResourceLimits(max_power_exponent=10),
            )
        self.assertEqual(caught.exception.code, "POWER_LIMIT")

        with self.assertRaises(ValueError):
            ResourceLimits(max_depth=129)

        with self.assertRaises(ResourceLimitError) as caught:
            evaluate(
                literal(1 << 15_000),
                limits=ResourceLimits(
                    max_integer_bits=20_000,
                    max_output_bytes=100_000,
                ),
            )
        self.assertEqual(caught.exception.code, "INTEGER_LIMIT")

    def test_input_and_dynamic_evaluation_limits(self) -> None:
        with self.assertRaises(ResourceLimitError) as caught:
            evaluate(
                {"op": "input"},
                {"value": "larger than ten bytes"},
                limits=ResourceLimits(max_input_bytes=10),
            )
        self.assertEqual(caught.exception.code, "INPUT_LIMIT")

        repeated_body = {
            "op": "map",
            "collection": literal([1, 2, 3]),
            "var": "item",
            "body": {"op": "var", "name": "item"},
        }
        with self.assertRaises(ResourceLimitError) as caught:
            evaluate(
                repeated_body,
                limits=ResourceLimits(max_evaluations=4),
            )
        self.assertEqual(caught.exception.code, "EVALUATION_LIMIT")

    def test_expanding_string_operations_are_rejected_before_allocation(self) -> None:
        def field(name: str) -> dict[str, str]:
            return {"op": "input", "name": name}

        limits = ResourceLimits(max_input_bytes=1_000, max_output_bytes=20)

        replace_expression = args(
            "replace",
            field("source"),
            literal("a"),
            field("replacement"),
        )
        with self.assertRaises(ResourceLimitError) as caught:
            evaluate(
                replace_expression,
                {
                    "source": AllocationBombString("aaaa"),
                    "replacement": "0123456789",
                },
                limits=limits,
            )
        self.assertEqual(caught.exception.code, "OUTPUT_LIMIT")

        join_expression = args(
            "join",
            field("separator"),
            field("parts"),
        )
        with self.assertRaises(ResourceLimitError) as caught:
            evaluate(
                join_expression,
                {
                    "separator": AllocationBombString("----------"),
                    "parts": ["aaaa", "bbbb", "cccc"],
                },
                limits=limits,
            )
        self.assertEqual(caught.exception.code, "OUTPUT_LIMIT")

        split_expression = args(
            "split",
            field("source"),
            literal(","),
        )
        with self.assertRaises(ResourceLimitError) as caught:
            evaluate(
                split_expression,
                {"source": AllocationBombString("a,a,a,a,a,a,a,a,a,a")},
                limits=ResourceLimits(max_input_bytes=1_000, max_output_bytes=30),
            )
        self.assertEqual(caught.exception.code, "OUTPUT_LIMIT")

    def test_oversized_aggregate_is_sized_without_serializing_it(self) -> None:
        expression = {
            "op": "map",
            "collection": literal([0] * 100),
            "var": "item",
            "body": {"op": "input"},
        }
        limits = ResourceLimits(max_input_bytes=100, max_output_bytes=100)
        with (
            patch(
                "crystalflow.engine.canonical_json_bytes",
                side_effect=AssertionError("oversized aggregate was serialized"),
            ),
            self.assertRaises(ResourceLimitError) as caught,
        ):
            evaluate(expression, "abcdefghij", limits=limits)
        self.assertEqual(caught.exception.code, "OUTPUT_LIMIT")

    def test_generated_map_values_are_bounded_while_accumulating(self) -> None:
        CountingUpperString.calls = 0
        expression = {
            "op": "map",
            "collection": {"op": "input", "name": "rows"},
            "var": "row",
            "body": args("upper", {"op": "input", "name": "value"}),
        }
        with self.assertRaises(ResourceLimitError) as caught:
            evaluate(
                expression,
                {
                    "rows": list(range(100)),
                    "value": CountingUpperString("abcdefghij"),
                },
                limits=ResourceLimits(max_input_bytes=1_000, max_output_bytes=30),
            )
        self.assertEqual(caught.exception.code, "OUTPUT_LIMIT")
        self.assertLessEqual(CountingUpperString.calls, 3)

    def test_reusable_engine_facade(self) -> None:
        engine = CrystalFlowEngine()
        self.assertEqual(
            engine.execute({"version": 1, "expression": literal("ok")}),
            "ok",
        )


if __name__ == "__main__":
    unittest.main()
