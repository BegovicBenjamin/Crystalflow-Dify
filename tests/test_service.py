from __future__ import annotations

import unittest

from crystalflow.service import CrystalService, CrystalTestError


class MemoryKV:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.fail_telemetry = False

    def get(self, key: str) -> bytes | None:
        return self.data.get(key)

    def set(self, key: str, value: bytes) -> None:
        if self.fail_telemetry and ":telemetry:" in key:
            raise OSError("simulated telemetry failure")
        self.data[key] = bytes(value)

    def delete(self, key: str) -> None:
        self.data.pop(key, None)


def total_program() -> dict:
    return {
        "version": 1,
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
            },
            "required": ["x", "y"],
            "additionalProperties": False,
        },
        "output_schema": {
            "type": "object",
            "properties": {"total": {"type": "integer"}},
            "required": ["total"],
            "additionalProperties": False,
        },
        "expression": {
            "op": "object",
            "fields": {
                "total": {
                    "op": "add",
                    "args": [
                        {"op": "input", "name": "x"},
                        {"op": "input", "name": "y"},
                    ],
                }
            },
        },
    }


def total_tests() -> list[dict]:
    return [
        {"name": "positive", "input": {"x": 2, "y": 3}, "expected": {"total": 5}},
        {"name": "negative", "input": {"x": -2, "y": 1}, "expected": {"total": -1}},
    ]


class CrystalServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.storage = MemoryKV()

    def service(self) -> CrystalService:
        return CrystalService(self.storage, "orders")

    def test_draft_activate_execute_telemetry_and_retire(self) -> None:
        proposed = self.service().crystallize(
            name="add_values",
            description="Add two integer values.",
            program=total_program(),
            tests=total_tests(),
            activation_policy="draft",
            estimated_tokens_per_run=125,
        )
        self.assertEqual(proposed["status"], "draft")
        self.assertTrue(proposed["created"])
        self.assertFalse(proposed["active"])

        cold = self.service().execute(
            name="add_values",
            inputs={"x": 4, "y": 7},
        )
        self.assertEqual(cold["status"], "miss")
        self.assertTrue(cold["fallback_required"])

        promoted = self.service().activate(
            name="add_values",
            version=proposed["version"],
            program_hash=proposed["program_hash"],
            confirmation="ACTIVATE add_values",
        )
        self.assertEqual(promoted["status"], "active")

        first = self.service().execute(
            name="add_values",
            inputs={"x": 4, "y": 7},
        )
        second = self.service().execute(
            name="add_values",
            inputs={"y": 7, "x": 4},
        )
        self.assertEqual(first["status"], "hit")
        self.assertEqual(first["result"], {"total": 11})
        self.assertEqual(first["result_json"], '{"total":11}')
        self.assertEqual(first["receipt"], second["receipt"])
        self.assertEqual(first["estimated_tokens_avoided"], 125)

        status = self.service().status(name="add_values")
        self.assertEqual(status["total_runs"], 2)
        self.assertEqual(status["estimated_tokens_avoided"], 250)
        self.assertNotIn("latest_version_record", status["details"])

        retired = self.service().retire(
            name="add_values",
            version=0,
            confirmation="add_values",
        )
        self.assertEqual(retired["status"], "retired")
        disabled = self.service().execute(
            name="add_values",
            inputs={"x": 1, "y": 1},
        )
        self.assertEqual(disabled["status"], "disabled")
        self.assertTrue(disabled["fallback_required"])

    def test_test_mismatch_is_not_stored(self) -> None:
        bad_tests = [{"name": "wrong", "input": {"x": 2, "y": 3}, "expected": {"total": 99}}]
        with self.assertRaises(CrystalTestError) as caught:
            self.service().crystallize(
                name="bad",
                description="A broken proposal.",
                program=total_program(),
                tests=bad_tests,
            )
        self.assertEqual(caught.exception.code, "TEST_MISMATCH")
        self.assertEqual(self.service().status()["crystal_count"], 0)

    def test_auto_activation_and_invalid_input(self) -> None:
        proposed = self.service().crystallize(
            name="auto",
            description="An explicitly auto-activated test fixture.",
            program=total_program(),
            tests=total_tests(),
            activation_policy="activate_after_tests",
        )
        self.assertEqual(proposed["status"], "active")
        invalid = self.service().execute(name="auto", inputs={"x": 1})
        self.assertEqual(invalid["status"], "invalid_input")
        self.assertTrue(invalid["fallback_required"])
        self.assertEqual(invalid["reason_code"], "MISSING_REQUIRED")

    def test_explicit_version_supports_controlled_draft_replay(self) -> None:
        proposed = self.service().crystallize(
            name="replay",
            description="Draft replay.",
            program=total_program(),
            tests=total_tests(),
        )
        replay = self.service().execute(
            name="replay",
            version=proposed["version"],
            inputs={"x": 8, "y": 9},
        )
        self.assertEqual(replay["status"], "hit")
        self.assertEqual(replay["result"], {"total": 17})

    def test_telemetry_failure_does_not_discard_valid_result(self) -> None:
        self.service().crystallize(
            name="metrics",
            description="Telemetry failure behavior.",
            program=total_program(),
            tests=total_tests(),
            activation_policy="activate_after_tests",
            estimated_tokens_per_run=10,
        )
        self.storage.fail_telemetry = True
        result = self.service().execute(
            name="metrics",
            inputs={"x": 1, "y": 2},
        )
        self.assertEqual(result["status"], "hit")
        self.assertEqual(result["result"], {"total": 3})
        self.assertFalse(result["telemetry_recorded"])

    def test_program_and_tests_are_hidden_by_default(self) -> None:
        self.service().crystallize(
            name="hidden",
            description="Visibility.",
            program=total_program(),
            tests=total_tests(),
        )
        compact = self.service().status(name="hidden", include_program=False)
        expanded = self.service().status(name="hidden", include_program=True)
        self.assertNotIn("latest_version_record", compact["details"])
        self.assertIn("latest_version_record", expanded["details"])
        self.assertEqual(
            expanded["details"]["latest_version_record"]["tests"],
            total_tests(),
        )


if __name__ == "__main__":
    unittest.main()
