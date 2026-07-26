from __future__ import annotations

import unittest
from typing import Any

from crystalflow.adaptive_routes import (
    AdaptiveRouteStore,
    RouteValidationError,
    ToolAction,
    ToolContract,
)


class MemoryKV:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    def exist(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str) -> bytes:
        if key not in self.data:
            raise LookupError("missing")
        return self.data[key]

    def set(self, key: str, value: bytes) -> None:
        self.data[key] = bytes(value)

    def delete(self, key: str) -> None:
        self.data.pop(key, None)


def contract(
    *,
    version: str = "1",
    read_only: bool = True,
    require_format: bool = False,
) -> ToolContract:
    properties: dict[str, Any] = {
        "sop_id": {
            "type": "string",
            "description": "Stable SOP identifier.",
        }
    }
    required = ["sop_id"]
    if require_format:
        properties["format"] = {"type": "string", "default": "text"}
        required.append("format")
    return ToolContract(
        provider_type="workflow",
        provider="knowledge",
        tool_name="get_sop",
        arguments_schema={
            "type": "object",
            "title": "Get SOP",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        contract_version=version,
        read_only=read_only,
    )


def action(sop_id: str = "SOP-42", **arguments: Any) -> ToolAction:
    return ToolAction(
        provider_type="workflow",
        provider="knowledge",
        tool_name="get_sop",
        arguments={"sop_id": sop_id, **arguments},
    )


class AdaptiveRouteStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.storage = MemoryKV()
        self.tools = [contract()]
        self.store = AdaptiveRouteStore(
            self.storage,
            "route_tests",
            scope={"app_id": "app-1", "instruction_hash": "instructions-1"},
            threshold=3,
        )

    def test_repeated_success_activates_and_persists_an_exact_tool_plan(self) -> None:
        unseen = self.store.lookup(
            "What is in SOP-42?",
            context={"selected_sop": "SOP-42"},
            tools=self.tools,
        )
        self.assertEqual(unseen.status, "miss")
        self.assertEqual(unseen.reason_code, "UNSEEN_ROUTE")

        first = self.store.observe_success(
            "What is in SOP-42?",
            action(),
            context={"selected_sop": "SOP-42"},
            tools=self.tools,
        )
        second = self.store.observe_success(
            "WHAT   IS IN SOP-42？",
            action(),
            context={"selected_sop": "SOP-42"},
            tools=self.tools,
        )
        activated = self.store.observe_success(
            "what is in sop-42?",
            action(),
            context={"selected_sop": "SOP-42"},
            tools=self.tools,
        )

        self.assertEqual(first.status, "learning")
        self.assertEqual(second.status, "learning")
        self.assertTrue(activated.hit)
        self.assertEqual(activated.observations, 3)
        assert activated.plan is not None
        self.assertEqual(activated.plan.tool_name, "get_sop")
        self.assertEqual(activated.plan.arguments, {"sop_id": "SOP-42"})

        restored = AdaptiveRouteStore(
            self.storage,
            "route_tests",
            scope={"instruction_hash": "instructions-1", "app_id": "app-1"},
            threshold=3,
        )
        hit = restored.lookup(
            "  WHAT IS IN SOP-42? ",
            context={"selected_sop": "SOP-42"},
            tools=self.tools,
        )
        self.assertTrue(hit.hit)
        self.assertNotIn(b"What is in SOP-42?", b"".join(self.storage.data.values()))

        other_context = restored.lookup(
            "What is in SOP-42?",
            context={"selected_sop": "SOP-99"},
            tools=self.tools,
        )
        self.assertEqual(other_context.status, "miss")

    def test_conflicting_successful_action_quarantines_without_replaying_it(self) -> None:
        self.store.observe_success("Get the SOP", action(), tools=self.tools)
        conflict = self.store.observe_success(
            "get the sop",
            action("SOP-99"),
            tools=self.tools,
        )

        self.assertEqual(conflict.status, "quarantined")
        self.assertEqual(conflict.reason_code, "ACTION_CONFLICT")
        self.assertIsNone(conflict.plan)
        lookup = self.store.lookup("GET THE SOP", tools=self.tools)
        self.assertEqual(lookup.status, "quarantined")
        self.assertFalse(lookup.hit)
        self.assertNotIn(b"SOP-99", b"".join(self.storage.data.values()))

    def test_contract_change_invalidates_and_requires_fresh_evidence(self) -> None:
        short = AdaptiveRouteStore(
            self.storage,
            "contracts",
            scope="app-1",
            threshold=2,
        )
        short.observe_success("Get SOP-42", action(), tools=self.tools)
        self.assertTrue(short.observe_success("Get SOP-42", action(), tools=self.tools).hit)

        changed = [contract(version="2")]
        invalidated = short.lookup("Get SOP-42", tools=changed)
        self.assertEqual(invalidated.status, "invalidated")
        self.assertEqual(invalidated.reason_code, "TOOL_CONTRACT_CHANGED")
        self.assertIsNone(invalidated.plan)

        relearning = short.observe_success("Get SOP-42", action(), tools=changed)
        self.assertEqual(relearning.status, "learning")
        self.assertEqual(relearning.observations, 1)
        self.assertEqual(relearning.invalidation_count, 1)
        self.assertTrue(short.observe_success("Get SOP-42", action(), tools=changed).hit)

    def test_only_allowlisted_read_only_tools_with_valid_arguments_are_learned(self) -> None:
        missing = self.store.observe_success("Get SOP", action(), tools=[])
        self.assertEqual(missing.reason_code, "TOOL_NOT_ALLOWLISTED")

        writable = self.store.observe_success(
            "Get SOP",
            action(),
            tools=[contract(read_only=False)],
        )
        self.assertEqual(writable.reason_code, "TOOL_NOT_READ_ONLY")

        invalid = self.store.observe_success(
            "Get SOP",
            action("SOP-42", unexpected=True),
            tools=self.tools,
        )
        self.assertEqual(invalid.reason_code, "INVALID_TOOL_ARGUMENTS")
        self.assertFalse(self.storage.data)

    def test_higher_threshold_demotes_active_route_until_more_observations(self) -> None:
        low = AdaptiveRouteStore(
            self.storage,
            "thresholds",
            threshold=2,
        )
        low.observe_success("Get SOP", action(), tools=self.tools)
        self.assertTrue(low.observe_success("Get SOP", action(), tools=self.tools).hit)

        high = AdaptiveRouteStore(
            self.storage,
            "thresholds",
            threshold=3,
        )
        demoted = high.lookup("Get SOP", tools=self.tools)
        self.assertEqual(demoted.status, "learning")
        self.assertFalse(demoted.hit)
        self.assertTrue(high.observe_success("Get SOP", action(), tools=self.tools).hit)

    def test_record_hit_tracks_only_confirmed_active_route_savings(self) -> None:
        store = AdaptiveRouteStore(self.storage, "telemetry", threshold=1)
        active = store.observe_success("Get SOP", action(), tools=self.tools)

        store.record_hit(active.route_key, estimated_tokens_avoided=120)
        second = store.record_hit(active.route_key, estimated_tokens_avoided=80)
        self.assertEqual(second.hit_count, 2)
        self.assertEqual(second.estimated_tokens_avoided, 200)

        decision = store.lookup("Get SOP", tools=self.tools)
        self.assertEqual(decision.hit_count, 2)
        self.assertEqual(decision.estimated_tokens_avoided, 200)
        with self.assertRaises(RouteValidationError):
            store.record_hit("not-a-hash")


if __name__ == "__main__":
    unittest.main()
