from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from dify_plugin.entities.tool import ToolRuntime

from tools.crystal_status import CrystalStatusTool
from tools.crystallize import CrystallizeTool
from tools.execute_crystal import ExecuteCrystalTool

ROOT = Path(__file__).resolve().parents[1]


class MemoryKV:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    def exist(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str) -> bytes:
        if key not in self.data:
            raise LookupError("key does not exist")
        return self.data[key]

    def set(self, key: str, value: bytes) -> None:
        self.data[key] = bytes(value)

    def delete(self, key: str) -> None:
        self.data.pop(key, None)


def json_payload(messages: list) -> dict:
    for message in reversed(messages):
        value = getattr(message.message, "json_object", None)
        if isinstance(value, dict):
            return value
    raise AssertionError("tool did not emit a JSON message")


class DifyToolAdapterTests(unittest.TestCase):
    def test_status_accepts_sdk_numeric_false_for_boolean_form_field(self) -> None:
        storage = MemoryKV()
        session = SimpleNamespace(storage=storage)
        runtime = ToolRuntime(credentials={}, user_id="test", session_id="test")

        status = CrystalStatusTool(runtime=runtime, session=session)
        payload = json_payload(
            list(
                status.invoke(
                    {
                        "namespace": "adapter-test",
                        "include_program": 0,
                    }
                )
            )
        )
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["crystal_count"], 0)

    def test_crystallize_then_execute_through_sdk_messages(self) -> None:
        storage = MemoryKV()
        session = SimpleNamespace(storage=storage)
        runtime = ToolRuntime(credentials={}, user_id="test", session_id="test")
        program = (ROOT / "examples/sla_classification.program.json").read_text(encoding="utf-8")
        tests = (ROOT / "examples/sla_classification.tests.json").read_text(encoding="utf-8")

        crystallize = CrystallizeTool(runtime=runtime, session=session)
        created_messages = list(
            crystallize.invoke(
                {
                    "namespace": "adapter-test",
                    "crystal_name": "sla",
                    "description": "Classify a structured SLA request.",
                    "program_json": program,
                    "tests_json": tests,
                    "activation_policy": "activate_after_tests",
                    "estimated_tokens_per_run": 75,
                }
            )
        )
        created = json_payload(created_messages)
        self.assertEqual(created["status"], "active")
        self.assertTrue(created["tests_passed"])
        self.assertEqual(
            {message.message.variable_name for message in created_messages[:-1]},
            {
                "active",
                "created",
                "crystal_name",
                "message",
                "program_hash",
                "status",
                "test_count",
                "tests_passed",
                "version",
            },
        )

        execute = ExecuteCrystalTool(runtime=runtime, session=session)
        execution_messages = list(
            execute.invoke(
                {
                    "namespace": "adapter-test",
                    "crystal_name": "sla",
                    "input_json": json.dumps(
                        {
                            "priority": "critical",
                            "customer_tier": "standard",
                            "age_minutes": 16,
                        }
                    ),
                    "version": 0,
                }
            )
        )
        executed = json_payload(execution_messages)
        self.assertEqual(executed["status"], "hit")
        self.assertFalse(executed["fallback_required"])
        self.assertEqual(
            json.loads(executed["result_json"]),
            {"queue": "incident", "sla_breached": True},
        )
        self.assertEqual(executed["estimated_tokens_avoided"], 75)


if __name__ == "__main__":
    unittest.main()
