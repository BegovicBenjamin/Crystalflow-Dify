from __future__ import annotations

import json
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from crystalflow.canonical import content_hash
from crystalflow.models import Lifecycle
from crystalflow.registry import (
    ActivationError,
    ConflictError,
    CorruptionError,
    NotFoundError,
    Registry,
    StorageError,
    ValidationError,
    json_from_bytes,
    stable_json_bytes,
)


class MemoryKV:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.fail_next_index_set = False

    def get(self, key: str) -> bytes | None:
        return self.values.get(key)

    def set(self, key: str, value: bytes) -> None:
        if self.fail_next_index_set and key.endswith(":index"):
            self.fail_next_index_set = False
            raise OSError("simulated catalog outage")
        if not isinstance(value, bytes):
            raise TypeError("MemoryKV only accepts bytes")
        self.values[key] = value

    def delete(self, key: str) -> None:
        self.deleted.append(key)
        self.values.pop(key, None)


class DifyStyleMemoryKV(MemoryKV):
    """Match Dify SDK 0.9.x: missing reads raise and ``exist`` is authoritative."""

    def exist(self, key: str) -> bool:
        return key in self.values

    def get(self, key: str) -> bytes:
        if key not in self.values:
            raise LookupError("key does not exist")
        return self.values[key]


class SlowAliasReadKV(MemoryKV):
    """Capture an alias read before yielding to expose cross-instance races."""

    def get(self, key: str) -> bytes | None:
        value = super().get(key)
        if ":alias:" in key:
            time.sleep(0.03)
        return value


def program(value: str) -> dict[str, Any]:
    return {
        "version": 1,
        "expression": {"op": "literal", "value": value},
    }


class RegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kv = MemoryKV()
        self.registry = Registry(self.kv, namespace="tenant_a")

    def register(
        self,
        name: str = "invoice.total.v1",
        value: str = "one",
        *,
        tests_passed: bool = True,
        description: str | None = "Invoice total",
    ):
        crystal_program = program(value)
        return self.registry.register(
            name,
            crystal_program,
            content_hash(crystal_program),
            tests=[{"input": {}, "output": value}],
            metadata={
                "tests_passed": tests_passed,
                "engine_version": "1",
                "dsl_version": 1,
                "test_report": {"passed": tests_passed, "cases": 1},
            },
            description=description,
        )

    def test_dify_style_missing_keys_use_exist_probe(self) -> None:
        registry = Registry(DifyStyleMemoryKV(), namespace="fresh_workspace")

        self.assertEqual(registry.list(), [])
        with self.assertRaises(NotFoundError):
            registry.get("new.crystal")

        created = registry.register(
            "new.crystal",
            program("ready"),
            metadata={"tests_passed": True},
        )
        self.assertEqual(created.version, 1)
        self.assertEqual(registry.latest("new.crystal").program, program("ready"))

    def test_registry_instances_serialize_threaded_invocations(self) -> None:
        storage = SlowAliasReadKV()
        first = Registry(storage, namespace="threaded")
        second = Registry(storage, namespace="threaded")
        self.assertIs(first._lock, second._lock)
        start = threading.Barrier(3)

        def register(registry: Registry, value: str):
            start.wait()
            return registry.register(
                "shared.crystal",
                program(value),
                metadata={"tests_passed": True},
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            first_result = executor.submit(register, first, "one")
            second_result = executor.submit(register, second, "two")
            start.wait()
            created = [first_result.result(), second_result.result()]

        self.assertEqual(sorted(item.version for item in created), [1, 2])
        self.assertEqual(first.latest("shared.crystal").version, 2)
        self.assertEqual(
            {
                first.get("shared.crystal", version).program["expression"]["value"]
                for version in (1, 2)
            },
            {"one", "two"},
        )

    def test_namespace_name_hash_and_json_validation(self) -> None:
        for namespace in ("", "bad:namespace", "../bad", "a..b", "a" * 65):
            with (
                self.subTest(namespace=namespace),
                self.assertRaises(ValidationError),
            ):
                Registry(self.kv, namespace)

        with self.assertRaises(ValidationError):
            self.registry.register("bad/name", program("x"))
        with self.assertRaises(ValidationError):
            self.registry.register("bad..name", program("x"))
        with self.assertRaises(ValidationError):
            self.registry.register("ok", program("x"), "not-a-hash")
        with self.assertRaises(ValidationError):
            self.registry.register("ok", {"bad": object()})

        canonical = stable_json_bytes({"z": 1, "a": [True, None]})
        self.assertEqual(canonical, b'{"a":[true,null],"z":1}')
        self.assertEqual(json_from_bytes(canonical), {"a": [True, None], "z": 1})
        with self.assertRaises(CorruptionError):
            json_from_bytes(b'{"a":1,"a":2}')
        with self.assertRaises(CorruptionError):
            json_from_bytes(b'{"n":NaN}')

    def test_registration_is_content_addressed_snapshotted_and_idempotent(self) -> None:
        original_program = program("one")
        original_tests = [{"expected": "one"}]
        original_metadata = {"tests_passed": True, "engine_version": "1"}

        first = self.registry.register(
            "invoice.total.v1",
            original_program,
            content_hash(original_program),
            original_tests,
            original_metadata,
            description="Invoice total",
        )
        self.assertTrue(first.created)
        self.assertEqual(first.version, 1)
        self.assertEqual(first.state, Lifecycle.DRAFT)
        self.assertEqual(len(first.content_hash), 64)

        # Mutating caller-owned values cannot alter the stored artifact.
        original_program["expression"]["value"] = "mutated"
        original_tests.append({"expected": "mutated"})
        original_metadata["tests_passed"] = False
        stored = self.registry.get("invoice.total.v1", 1)
        self.assertEqual(stored.program["expression"]["value"], "one")
        self.assertEqual(stored.tests, [{"expected": "one"}])
        self.assertTrue(stored.metadata["tests_passed"])

        # Same name + canonical program hash returns the prior version, even if
        # report metadata differs.
        same_program = program("one")
        duplicate = self.registry.register(
            "invoice.total.v1",
            same_program,
            content_hash(same_program),
            tests=[],
            metadata={"tests_passed": False},
        )
        self.assertFalse(duplicate.created)
        self.assertEqual(duplicate.version, 1)
        self.assertEqual(self.registry.latest("invoice.total.v1").version, 1)

        second = self.register(value="two", description=None)
        self.assertTrue(second.created)
        self.assertEqual(second.version, 2)
        self.assertEqual(second.revision, 2)
        self.assertEqual(second.description, "Invoice total")

        record_keys = [key for key in self.kv.values if ":record:" in key]
        self.assertEqual(len(record_keys), 2)
        self.assertTrue(all(isinstance(self.kv.values[key], bytes) for key in record_keys))

    def test_registering_a_retired_program_creates_a_new_draft_version(self) -> None:
        first = self.register()
        self.registry.retire("invoice.total.v1", first.version)

        replacement = self.register()

        self.assertTrue(replacement.created)
        self.assertEqual(replacement.version, 2)
        self.assertEqual(replacement.state, Lifecycle.DRAFT)
        self.assertEqual(
            self.registry.get("invoice.total.v1", first.version).state,
            Lifecycle.RETIRED,
        )

    def test_activation_is_test_gated_and_only_alias_mutates(self) -> None:
        first = self.register(value="one")
        record_key = f"tenant_a:v1:record:{first.content_hash}"
        immutable_before = self.kv.values[record_key]

        with self.assertRaises(NotFoundError):
            self.registry.get("invoice.total.v1")

        active = self.registry.activate(
            "invoice.total.v1",
            1,
            expected_program_hash=first.program_hash,
            expected_active_version=None,
        )
        self.assertEqual(active.state, Lifecycle.ACTIVE)
        self.assertEqual(self.registry.get("invoice.total.v1").version, 1)
        self.assertEqual(self.kv.values[record_key], immutable_before)

        failing = self.register(value="two", tests_passed=False)
        with self.assertRaises(ActivationError):
            self.registry.activate("invoice.total.v1", failing.version)
        with self.assertRaises(ConflictError):
            self.registry.activate(
                "invoice.total.v1",
                failing.version,
                expected_active_version=None,
                unsafe=True,
            )

        second = self.registry.activate(
            "invoice.total.v1",
            failing.version,
            expected_active_version=1,
            unsafe=True,
        )
        self.assertEqual(second.state, Lifecycle.ACTIVE)
        self.assertEqual(self.registry.get("invoice.total.v1", 1).state, Lifecycle.RETIRED)

        # Retiring and later reactivating an immutable version supports rollback.
        retired = self.registry.retire("invoice.total.v1")
        self.assertEqual(retired.state, Lifecycle.RETIRED)
        with self.assertRaises(NotFoundError):
            self.registry.get("invoice.total.v1")
        rolled_back = self.registry.activate("invoice.total.v1", 1)
        self.assertEqual(rolled_back.state, Lifecycle.ACTIVE)
        self.assertEqual(self.kv.deleted, [])

    def test_list_and_telemetry_are_deterministic_and_versioned(self) -> None:
        zeta = self.register(name="zeta.rule", value="z")
        self.registry.activate("zeta.rule", zeta.version)
        alpha = self.register(name="alpha.rule", value="a", description="Alpha")
        self.registry.activate("alpha.rule", alpha.version)

        first = self.registry.record_run("zeta.rule", estimated_tokens_avoided=125)
        second = self.registry.record_run("zeta.rule", estimated_tokens_avoided=75, count=2)
        self.assertEqual(first.run_count, 1)
        self.assertEqual(second.run_count, 3)
        self.assertEqual(second.estimated_tokens_avoided, 200)
        self.assertEqual(second.revision, 2)

        aggregate = self.registry.telemetry("zeta.rule")
        self.assertEqual(aggregate.version, None)
        self.assertEqual(aggregate.runs, 3)
        self.assertEqual(aggregate.estimated_tokens_avoided, 200)

        summaries = self.registry.list()
        self.assertEqual([item.name for item in summaries], ["alpha.rule", "zeta.rule"])
        zeta_summary = summaries[1]
        self.assertEqual(zeta_summary.active_version, 1)
        self.assertEqual(zeta_summary.active_program_hash, zeta.program_hash)
        self.assertEqual(zeta_summary.run_count, 3)
        self.assertEqual(zeta_summary.to_dict()["estimated_tokens_avoided"], 200)

    def test_catalog_write_is_best_effort(self) -> None:
        self.kv.fail_next_index_set = True
        created = self.register()
        self.assertTrue(created.created)
        self.assertIsInstance(self.registry.last_index_error, StorageError)

        # The alias and content record remain directly reachable.
        fetched = self.registry.get("invoice.total.v1", 1)
        self.assertEqual(fetched.program_hash, created.program_hash)
        self.assertEqual(self.registry.list(), [])

        self.registry.ensure_indexed("invoice.total.v1")
        self.assertIsNone(self.registry.last_index_error)
        self.assertEqual([item.name for item in self.registry.list()], ["invoice.total.v1"])

    def test_corrupt_alias_record_and_noncanonical_storage_fail_loudly(self) -> None:
        created = self.register()
        alias_key = "tenant_a:v1:alias:invoice.total.v1"
        decoded = json.loads(self.kv.values[alias_key])
        decoded["latest_version"] = 99
        self.kv.values[alias_key] = stable_json_bytes(decoded)
        with self.assertRaises(CorruptionError):
            self.registry.get("invoice.total.v1", created.version)

        # Valid JSON with noncanonical whitespace is also a corrupt registry record.
        self.kv.values[alias_key] = json.dumps(decoded).encode("utf-8")
        with self.assertRaises(CorruptionError):
            self.registry.get("invoice.total.v1", created.version)

    def test_content_hash_detects_record_tampering(self) -> None:
        created = self.register()
        record_key = f"tenant_a:v1:record:{created.content_hash}"
        decoded = json.loads(self.kv.values[record_key])
        decoded["metadata"]["engine_version"] = "tampered"
        self.kv.values[record_key] = stable_json_bytes(decoded)
        with self.assertRaises(CorruptionError):
            self.registry.get("invoice.total.v1", 1)


if __name__ == "__main__":
    unittest.main()
