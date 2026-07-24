from __future__ import annotations

import json
import threading
import unittest
from typing import Any

from crystalflow.progressive import (
    ProgressiveService,
    ProgressiveValidationError,
)
from crystalflow.service import CrystalService


class MemoryKV:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.set_keys: list[str] = []

    def get(self, key: str) -> bytes | None:
        return self.data.get(key)

    def set(self, key: str, value: bytes) -> None:
        self.set_keys.append(key)
        self.data[key] = bytes(value)

    def delete(self, key: str) -> None:
        self.data.pop(key, None)


class DifyStyleMemoryKV(MemoryKV):
    def exist(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str) -> bytes:
        if key not in self.data:
            raise LookupError("missing")
        return self.data[key]


def add_program() -> dict[str, Any]:
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


class CountingFallback:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, task_key: str, inputs: Any) -> dict[str, int]:
        self.calls += 1
        return {"total": inputs["x"] + inputs["y"]}


class CountingBuilder:
    def __init__(self, result: Any = None) -> None:
        self.calls = 0
        self.examples: list[dict[str, Any]] = []
        self.result = add_program() if result is None else result

    def __call__(self, task_key: str, examples: list[dict[str, Any]]) -> Any:
        self.calls += 1
        self.examples = examples
        return self.result


class ProgressiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.storage = MemoryKV()
        self.fallback = CountingFallback()

    def service(
        self,
        *,
        builder: Any = None,
        min_examples: int = 2,
        auto_activate: bool = False,
        **kwargs: Any,
    ) -> ProgressiveService:
        return ProgressiveService(
            self.storage,
            "progressive_tests",
            fallback=self.fallback,
            builder=builder,
            min_examples=min_examples,
            auto_activate=auto_activate,
            **kwargs,
        )

    def test_warm_hit_returns_before_fallback_builder_or_learning_storage(self) -> None:
        crystal = CrystalService(self.storage, "progressive_tests")
        created = crystal.crystallize(
            name="add",
            description="Add values.",
            program=add_program(),
            tests=[{"name": "case", "input": {"x": 1, "y": 2}, "expected": {"total": 3}}],
            activation_policy="activate_after_tests",
        )
        builder = CountingBuilder()
        progressive = self.service(builder=builder, min_examples=1)

        result = progressive.run("add", {"y": 5, "x": 4}, learn=True)

        self.assertEqual(result["path"], "warm_hit")
        self.assertEqual(result["result"], {"total": 9})
        self.assertEqual(result["version"], created["version"])
        self.assertEqual(result["learn_status"], "not_applicable")
        self.assertEqual(self.fallback.calls, 0)
        self.assertEqual(builder.calls, 0)
        self.assertFalse(any(":progressive:" in key for key in self.storage.set_keys))

    def test_learning_is_opt_in_and_cold_result_is_canonical(self) -> None:
        builder = CountingBuilder()
        result = self.service(builder=builder, min_examples=1).run(
            "add",
            {"y": 2, "x": 1},
            learn=False,
        )

        self.assertEqual(result["status"], "fallback")
        self.assertEqual(result["path"], "cold_fallback")
        self.assertFalse(result["fallback_required"])
        self.assertEqual(result["result_json"], '{"total":3}')
        self.assertEqual(result["learn_status"], "disabled")
        self.assertEqual(builder.calls, 0)
        self.assertFalse(any(":progressive:" in key for key in self.storage.data))

    def test_examples_are_deduplicated_and_dify_missing_reads_are_supported(self) -> None:
        storage = DifyStyleMemoryKV()
        fallback = CountingFallback()
        progressive = ProgressiveService(
            storage,
            "dify_style",
            fallback=fallback,
            min_examples=2,
        )

        first = progressive.run("add", {"x": 1, "y": 2}, learn=True)
        second = progressive.run("add", {"y": 2, "x": 1}, learn=True)

        self.assertEqual(first["learn_status"], "observed")
        self.assertEqual(first["example_count"], 1)
        self.assertEqual(second["learn_status"], "duplicate")
        self.assertEqual(second["example_count"], 1)
        state = json.loads(storage.data["dify_style:progressive:v1:examples:add"])
        self.assertEqual(len(state["examples"]), 1)
        self.assertEqual(state["examples"][0]["input"], {"x": 1, "y": 2})

    def test_conflicting_output_quarantines_without_storing_second_output(self) -> None:
        outputs = iter([{"value": "first"}, {"value": "second"}, {"value": "first"}])
        builder = CountingBuilder()

        def changing_fallback(task_key: str, inputs: Any) -> dict[str, str]:
            return next(outputs)

        progressive = ProgressiveService(
            self.storage,
            "conflicts",
            fallback=changing_fallback,
            builder=builder,
            min_examples=2,
        )
        first = progressive.run("unstable", {"id": 1}, learn=True)
        conflict = progressive.run("unstable", {"id": 1}, learn=True)
        quarantined = progressive.run("unstable", {"id": 2}, learn=True)

        self.assertEqual(first["learn_status"], "observed")
        self.assertEqual(conflict["learn_status"], "quarantined")
        self.assertEqual(conflict["learn_reason_code"], "OUTPUT_CONFLICT")
        self.assertEqual(quarantined["learn_status"], "quarantined")
        self.assertEqual(builder.calls, 0)
        state = json.loads(self.storage.data["conflicts:progressive:v1:examples:unstable"])
        self.assertTrue(state["quarantined"])
        self.assertEqual(state["conflict_count"], 1)
        self.assertEqual(len(state["examples"]), 1)
        self.assertEqual(state["examples"][0]["output"], {"value": "first"})

    def test_threshold_builds_draft_from_observed_consistency_cases_once(self) -> None:
        builder = CountingBuilder(
            {
                "program": add_program(),
                # Suggested builder tests are deliberately wrong. Retained
                # observations remain the activation authority.
                "tests": [{"input": {"x": 0, "y": 0}, "expected": {"total": 999}}],
            }
        )
        progressive = self.service(
            builder=builder,
            min_examples=2,
            estimated_tokens_per_run=77,
        )

        first = progressive.run("add", {"x": 1, "y": 2}, learn=True)
        second = progressive.run("add", {"x": 5, "y": 8}, learn=True)
        third = progressive.run("add", {"x": 9, "y": 4}, learn=True)

        self.assertEqual(first["learn_status"], "observed")
        self.assertEqual(second["learn_status"], "candidate_created")
        self.assertEqual(second["example_count"], 2)
        self.assertEqual(third["learn_status"], "candidate_exists")
        self.assertEqual(builder.calls, 1)
        self.assertEqual(
            builder.examples,
            [
                {"name": "observed_001", "input": {"x": 1, "y": 2}, "expected": {"total": 3}},
                {"name": "observed_002", "input": {"x": 5, "y": 8}, "expected": {"total": 13}},
            ],
        )
        candidate = CrystalService(self.storage, "progressive_tests").registry.latest("add")
        self.assertEqual(candidate.state.value, "draft")
        self.assertEqual(candidate.tests, builder.examples)
        self.assertEqual(candidate.metadata["estimated_tokens_per_run"], 77)
        state = json.loads(self.storage.data["progressive_tests:progressive:v1:examples:add"])
        self.assertEqual(len(state["examples"]), 2)

    def test_conflict_while_builder_runs_prevents_candidate_activation(self) -> None:
        builder_started = threading.Event()
        release_builder = threading.Event()
        outputs = iter([{"value": "first"}, {"value": "second"}])

        def changing_fallback(task_key: str, inputs: Any) -> dict[str, str]:
            return next(outputs)

        def blocking_builder(task_key: str, examples: list[dict[str, Any]]) -> dict[str, Any]:
            builder_started.set()
            if not release_builder.wait(timeout=5):
                raise RuntimeError("test builder timed out")
            return {
                "version": 1,
                "expression": {
                    "op": "literal",
                    "value": {"value": "first"},
                },
            }

        progressive = ProgressiveService(
            self.storage,
            "concurrent_conflict",
            fallback=changing_fallback,
            builder=blocking_builder,
            min_examples=1,
            auto_activate=True,
        )
        first_result: dict[str, Any] = {}

        def first_run() -> None:
            first_result.update(progressive.run("unstable", {"id": 1}, learn=True))

        worker = threading.Thread(target=first_run)
        worker.start()
        self.assertTrue(builder_started.wait(timeout=5))
        try:
            conflict = progressive.run("unstable", {"id": 1}, learn=True)
        finally:
            release_builder.set()
            worker.join(timeout=5)

        self.assertFalse(worker.is_alive())
        self.assertEqual(conflict["learn_status"], "quarantined")
        self.assertEqual(first_result["learn_status"], "quarantined")
        self.assertEqual(
            CrystalService(self.storage, "concurrent_conflict").execute(
                name="unstable",
                inputs={"id": 1},
            )["status"],
            "miss",
        )

    def test_explicit_auto_activation_makes_the_next_call_warm(self) -> None:
        builder = CountingBuilder()
        progressive = self.service(
            builder=builder,
            min_examples=1,
            auto_activate=True,
        )

        cold = progressive.run("add", {"x": 2, "y": 3}, learn=True)
        warm = progressive.run("add", {"x": 20, "y": 3}, learn=True)

        self.assertEqual(cold["path"], "cold_fallback")
        self.assertEqual(cold["learn_status"], "candidate_active")
        self.assertEqual(warm["path"], "warm_hit")
        self.assertEqual(warm["result"], {"total": 23})
        self.assertEqual(self.fallback.calls, 1)
        self.assertEqual(builder.calls, 1)

    def test_a_retired_version_does_not_block_new_learning(self) -> None:
        crystal = CrystalService(self.storage, "progressive_tests")
        created = crystal.crystallize(
            name="add",
            description="Old learned version.",
            program=add_program(),
            tests=[
                {
                    "name": "old",
                    "input": {"x": 1, "y": 2},
                    "expected": {"total": 3},
                }
            ],
        )
        crystal.retire(
            name="add",
            version=created["version"],
            confirmation="add",
        )

        observed = self.service(min_examples=2).run(
            "add",
            {"x": 10, "y": 20},
            learn=True,
        )

        self.assertEqual(observed["path"], "cold_fallback")
        self.assertEqual(observed["learn_status"], "observed")
        self.assertEqual(observed["example_count"], 1)

    def test_auto_learning_never_reactivates_a_retired_version(self) -> None:
        crystal = CrystalService(self.storage, "progressive_tests")
        old = crystal.crystallize(
            name="add",
            description="Old learned version.",
            program=add_program(),
            tests=[
                {
                    "name": "old",
                    "input": {"x": 1, "y": 2},
                    "expected": {"total": 3},
                }
            ],
            activation_policy="activate_after_tests",
        )
        crystal.retire(
            name="add",
            version=old["version"],
            confirmation="add",
        )

        learned = self.service(
            builder=CountingBuilder(add_program()),
            min_examples=1,
            auto_activate=True,
        ).run("add", {"x": 10, "y": 20}, learn=True)

        registry = crystal.registry
        self.assertEqual(learned["learn_status"], "candidate_active")
        self.assertEqual(learned["version"], 2)
        self.assertEqual(registry.get("add", 1).state.value, "retired")
        active = registry.get("add")
        self.assertEqual(active.version, 2)
        self.assertEqual(
            active.tests,
            [
                {
                    "name": "observed_001",
                    "input": {"x": 10, "y": 20},
                    "expected": {"total": 30},
                }
            ],
        )

    def test_builder_and_storage_failures_preserve_cold_result_with_safe_codes(self) -> None:
        invalid_builder = CountingBuilder({"version": 1, "expression": {"op": "shell"}})
        progressive = self.service(builder=invalid_builder, min_examples=1)

        failed = progressive.run("add", {"x": 1, "y": 2}, learn=True)
        duplicate = progressive.run("add", {"x": 1, "y": 2}, learn=True)

        self.assertEqual(failed["result"], {"total": 3})
        self.assertEqual(failed["learn_status"], "build_failed")
        self.assertEqual(failed["learn_reason_code"], "CANDIDATE_VALIDATION_FAILED")
        self.assertNotIn("shell", failed["message"])
        self.assertEqual(duplicate["learn_status"], "duplicate")
        self.assertEqual(invalid_builder.calls, 1)

        key = "corrupt:progressive:v1:examples:add"
        corrupt_storage = MemoryKV()
        corrupt_storage.data[key] = b'{"secret":"must not appear"}'
        corrupt = ProgressiveService(
            corrupt_storage,
            "corrupt",
            fallback=CountingFallback(),
            builder=CountingBuilder(),
            min_examples=1,
        ).run("add", {"x": 1, "y": 2}, learn=True)
        self.assertEqual(corrupt["result"], {"total": 3})
        self.assertEqual(corrupt["learn_status"], "storage_error")
        self.assertNotIn("secret", corrupt["message"])

    def test_size_count_and_fallback_failures_are_bounded(self) -> None:
        with self.assertRaises(ProgressiveValidationError) as caught:
            self.service(max_input_bytes=5).run("add", {"long": "value"})
        self.assertEqual(caught.exception.code, "JSON_SIZE_LIMIT")
        self.assertEqual(self.fallback.calls, 0)

        oversized_example = self.service(
            min_examples=2,
            max_example_bytes=10,
        ).run("add", {"x": 1, "y": 2}, learn=True)
        self.assertEqual(oversized_example["result"], {"total": 3})
        self.assertEqual(oversized_example["learn_status"], "example_too_large")

        capacity_storage = MemoryKV()

        def echo(task_key: str, inputs: Any) -> Any:
            return inputs

        capacity = ProgressiveService(
            capacity_storage,
            "capacity",
            fallback=echo,
            min_examples=2,
            max_examples=2,
        )
        capacity.run("echo", {"id": 1}, learn=True)
        capacity.run("echo", {"id": 2}, learn=True)
        full = capacity.run("echo", {"id": 3}, learn=True)
        self.assertEqual(full["learn_status"], "capacity_reached")
        self.assertEqual(full["example_count"], 2)

        def broken_fallback(task_key: str, inputs: Any) -> Any:
            raise RuntimeError(f"do not leak {inputs!r}")

        error = ProgressiveService(
            MemoryKV(),
            "broken",
            fallback=broken_fallback,
        ).run("secret_task", {"password": "sensitive"}, learn=True)
        self.assertEqual(error["status"], "error")
        self.assertEqual(error["reason_code"], "FALLBACK_FAILED")
        self.assertNotIn("sensitive", error["message"])

    def test_invalid_configuration_and_task_keys_fail_before_callables(self) -> None:
        with self.assertRaises(ProgressiveValidationError):
            self.service(min_examples=3, max_examples=2)
        with self.assertRaises(ProgressiveValidationError) as caught:
            self.service().run("../not-a-task", {"x": 1})
        self.assertEqual(caught.exception.code, "INVALID_TASK_KEY")
        self.assertEqual(self.fallback.calls, 0)


if __name__ == "__main__":
    unittest.main()
