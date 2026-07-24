from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from dify_plugin.entities.model.llm import LLMModelConfig
from dify_plugin.entities.tool import ToolConfiguration, ToolRuntime

from tools.progressive_run import ProgressiveRunTool, _bound_crystal_name

ROOT = Path(__file__).resolve().parents[1]


class MemoryKV:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    def exist(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str) -> bytes:
        return self.data[key]

    def set(self, key: str, value: bytes) -> None:
        self.data[key] = bytes(value)

    def delete(self, key: str) -> None:
        self.data.pop(key, None)


class FakeLLM:
    def __init__(self, responses: list[SimpleNamespace] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict] = []

    def invoke(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("the active crystal must not invoke the model")
        return self.responses.pop(0)


def model_response(content: str, *, prompt: int, completion: int) -> SimpleNamespace:
    return SimpleNamespace(
        message=SimpleNamespace(content=content),
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        ),
    )


def json_payload(messages: list) -> dict:
    for message in reversed(messages):
        value = getattr(message.message, "json_object", None)
        if isinstance(value, dict):
            return value
    raise AssertionError("tool did not emit a JSON message")


def parameters(**updates) -> dict:
    value = {
        "namespace": "progressive-adapter",
        "task_key": "invoice_total",
        "task_description": "Add the integer line totals and return an object with total.",
        "input_json": '{"lines":[2,3]}',
        "model": {
            "provider": "test/provider",
            "model": "test-model",
            "model_type": "llm",
            "mode": "chat",
            "completion_params": {},
        },
        "learning_enabled": False,
        "min_examples": 3,
        "learning_policy": "draft",
        "estimated_tokens_per_run": 80,
    }
    value.update(updates)
    return value


class ProgressiveToolTests(unittest.TestCase):
    def tool(self, llm: FakeLLM) -> ProgressiveRunTool:
        session = SimpleNamespace(
            storage=MemoryKV(),
            model=SimpleNamespace(llm=llm),
        )
        runtime = ToolRuntime(credentials={}, user_id="test", session_id="test")
        return ProgressiveRunTool(runtime=runtime, session=session)

    def test_warm_hit_uses_no_model_tokens(self) -> None:
        llm = FakeLLM()
        captured: dict = {}

        class Runner:
            def run(self, task_key, inputs, *, learn=False):
                captured["run"] = (task_key, inputs, learn)
                return {
                    "path": "warm_hit",
                    "result": {"total": 5},
                    "learn_status": "active_crystal",
                    "example_count": 3,
                    "crystal_name": task_key,
                    "version": 1,
                    "program_hash": "a" * 64,
                    "receipt": "b" * 64,
                    "estimated_tokens_avoided": 80,
                    "telemetry_recorded": True,
                }

        def factory(**kwargs):
            captured["factory"] = kwargs
            return Runner()

        with patch("tools.progressive_run._create_runner", side_effect=factory):
            messages = list(self.tool(llm).invoke(parameters()))

        payload = json_payload(messages)
        self.assertEqual(payload["status"], "hit")
        self.assertFalse(payload["fallback_required"])
        self.assertEqual(payload["result_json"], '{"total":5}')
        self.assertEqual(payload["model_tokens_used"], 0)
        self.assertEqual(payload["llm_calls"], 0)
        self.assertEqual(
            captured["run"],
            (
                _bound_crystal_name(
                    "invoice_total",
                    "Add the integer line totals and return an object with total.",
                ),
                {"lines": [2, 3]},
                False,
            ),
        )
        self.assertEqual(captured["factory"]["min_examples"], 3)
        self.assertEqual(captured["factory"]["estimated_tokens_per_run"], 80)
        self.assertFalse(captured["factory"]["auto_activate"])
        self.assertEqual(
            {message.message.variable_name for message in messages[:-1]},
            {
                "completion_tokens",
                "crystal_name",
                "estimated_tokens_avoided",
                "example_count",
                "fallback_required",
                "learning_status",
                "llm_calls",
                "message",
                "model_tokens_used",
                "program_hash",
                "prompt_tokens",
                "receipt",
                "result_json",
                "result_text",
                "status",
                "telemetry_recorded",
                "version",
            },
        )

    def test_cold_learning_uses_selected_model_for_fallback_and_builder(self) -> None:
        program = {
            "version": 1,
            "expression": {
                "op": "object",
                "fields": {"total": {"op": "sum", "collection": {"op": "input", "name": "lines"}}},
            },
        }
        llm = FakeLLM(
            [
                model_response('{"total":5}', prompt=11, completion=4),
                model_response(f"```json\n{json.dumps(program)}\n```", prompt=20, completion=10),
            ]
        )

        class Runner:
            def __init__(self, fallback, builder) -> None:
                self.fallback = fallback
                self.builder = builder

            def run(self, task_key, inputs, *, learn=False):
                self.assert_learning(learn)
                result = self.fallback(task_key, inputs)
                proposed = self.builder(task_key, [{"input": inputs, "expected": result}])
                if proposed != program:
                    raise AssertionError("builder did not return the parsed program")
                return {
                    "path": "cold_fallback",
                    "result": result,
                    "learn_status": "draft_created",
                    "example_count": 1,
                    "message": "Fallback completed and a draft was tested.",
                }

            @staticmethod
            def assert_learning(learn):
                if learn is not True:
                    raise AssertionError("learning flag was not forwarded")

        def factory(**kwargs):
            if kwargs["auto_activate"] is not True:
                raise AssertionError("auto-activation policy was not forwarded")
            return Runner(kwargs["fallback"], kwargs["builder"])

        configured = parameters(
            learning_enabled=True,
            learning_policy="auto_activate",
            min_examples=1,
        )
        with patch("tools.progressive_run._create_runner", side_effect=factory):
            payload = json_payload(list(self.tool(llm).invoke(configured)))

        self.assertEqual(payload["status"], "fallback")
        self.assertFalse(payload["fallback_required"])
        self.assertEqual(payload["result_json"], '{"total":5}')
        self.assertEqual(payload["learning_status"], "draft_created")
        self.assertEqual(payload["llm_calls"], 2)
        self.assertEqual(payload["prompt_tokens"], 31)
        self.assertEqual(payload["completion_tokens"], 14)
        self.assertEqual(payload["model_tokens_used"], 45)
        self.assertEqual(len(llm.calls), 2)
        self.assertTrue(all(isinstance(call["model_config"], LLMModelConfig) for call in llm.calls))
        self.assertTrue(all(call["stream"] is False for call in llm.calls))

    def test_real_progressive_core_activates_then_uses_the_zero_token_path(self) -> None:
        program = {
            "version": 1,
            "expression": {
                "op": "object",
                "fields": {
                    "total": {
                        "op": "sum",
                        "collection": {"op": "input", "name": "lines"},
                    }
                },
            },
        }
        llm = FakeLLM(
            [
                model_response('{"total":5}', prompt=11, completion=4),
                model_response(json.dumps(program), prompt=20, completion=10),
            ]
        )
        session = SimpleNamespace(
            storage=MemoryKV(),
            model=SimpleNamespace(llm=llm),
        )
        runtime = ToolRuntime(credentials={}, user_id="test", session_id="test")
        tool = ProgressiveRunTool(runtime=runtime, session=session)
        configured = parameters(
            learning_enabled=True,
            learning_policy="auto_activate",
            min_examples=1,
        )

        cold = json_payload(list(tool.invoke(configured)))
        configured["input_json"] = '{"lines":[4,6]}'
        warm = json_payload(list(tool.invoke(configured)))

        self.assertEqual(cold["status"], "fallback")
        self.assertEqual(cold["learning_status"], "candidate_active")
        self.assertEqual(cold["model_tokens_used"], 45)
        self.assertEqual(warm["status"], "hit")
        self.assertEqual(warm["result_json"], '{"total":10}')
        self.assertEqual(warm["model_tokens_used"], 0)
        self.assertEqual(len(llm.calls), 2)

    def test_changing_task_description_cannot_reuse_the_old_crystal(self) -> None:
        program = {
            "version": 1,
            "expression": {
                "op": "object",
                "fields": {
                    "total": {
                        "op": "sum",
                        "collection": {"op": "input", "name": "lines"},
                    }
                },
            },
        }
        llm = FakeLLM(
            [
                model_response('{"total":5}', prompt=11, completion=4),
                model_response(json.dumps(program), prompt=20, completion=10),
                model_response('{"total":999}', prompt=8, completion=3),
            ]
        )
        session = SimpleNamespace(
            storage=MemoryKV(),
            model=SimpleNamespace(llm=llm),
        )
        runtime = ToolRuntime(credentials={}, user_id="test", session_id="test")
        tool = ProgressiveRunTool(runtime=runtime, session=session)
        configured = parameters(
            learning_enabled=True,
            learning_policy="auto_activate",
            min_examples=1,
        )

        first = json_payload(list(tool.invoke(configured)))
        changed = parameters(
            task_description="Ignore line values and return exactly an object with total 999.",
            learning_enabled=False,
            learning_policy="auto_activate",
            min_examples=1,
        )
        second = json_payload(list(tool.invoke(changed)))

        self.assertEqual(first["learning_status"], "candidate_active")
        self.assertEqual(second["status"], "fallback")
        self.assertEqual(second["result_json"], '{"total":999}')
        self.assertEqual(second["llm_calls"], 1)
        self.assertNotEqual(first["crystal_name"], second["crystal_name"])
        self.assertEqual(len(llm.calls), 3)

    def test_invalid_model_selector_is_a_safe_error(self) -> None:
        payload = json_payload(list(self.tool(FakeLLM()).invoke(parameters(model={}))))

        self.assertEqual(payload["status"], "error")
        self.assertTrue(payload["fallback_required"])
        self.assertEqual(payload["model_tokens_used"], 0)
        self.assertEqual(payload["message"], "model must be a valid selected LLM")

    def test_failed_fallback_counts_the_model_invocation_attempt(self) -> None:
        payload = json_payload(list(self.tool(FakeLLM()).invoke(parameters())))

        self.assertEqual(payload["status"], "error")
        self.assertTrue(payload["fallback_required"])
        self.assertEqual(payload["llm_calls"], 1)
        self.assertEqual(payload["model_tokens_used"], 0)

    def test_invalid_task_key_surfaces_the_safe_progressive_error(self) -> None:
        payload = json_payload(
            list(self.tool(FakeLLM()).invoke(parameters(task_key="../not-a-task")))
        )

        self.assertEqual(payload["status"], "error")
        self.assertTrue(payload["fallback_required"])
        self.assertEqual(payload["model_tokens_used"], 0)
        self.assertEqual(payload["message"], "task_key is not a valid stable crystal name")

    def test_model_json_fence_rejects_surrounding_prose(self) -> None:
        llm = FakeLLM(
            [
                model_response(
                    'Here is the result:\n```json\n{"total":5}\n```',
                    prompt=7,
                    completion=5,
                )
            ]
        )

        class Runner:
            def __init__(self, fallback) -> None:
                self.fallback = fallback

            def run(self, task_key, inputs, *, learn=False):
                return {
                    "path": "cold_fallback",
                    "result": self.fallback(task_key, inputs),
                }

        def factory(**kwargs):
            return Runner(kwargs["fallback"])

        with patch("tools.progressive_run._create_runner", side_effect=factory):
            payload = json_payload(list(self.tool(llm).invoke(parameters())))

        self.assertEqual(payload["status"], "error")
        self.assertTrue(payload["fallback_required"])
        self.assertEqual(payload["model_tokens_used"], 12)
        self.assertEqual(payload["message"], "model fallback response is not valid strict JSON")

    def test_yaml_declares_llm_selector_and_opt_in_learning(self) -> None:
        with (ROOT / "tools/progressive_run.yaml").open(encoding="utf-8") as handle:
            configuration = ToolConfiguration.model_validate(yaml.safe_load(handle))

        model = next(item for item in configuration.parameters if item.name == "model")
        learning = next(
            item for item in configuration.parameters if item.name == "learning_enabled"
        )
        self.assertEqual(model.type.value, "model-selector")
        self.assertEqual(model.scope, "llm")
        self.assertTrue(model.required)
        self.assertEqual(learning.type.value, "boolean")
        self.assertFalse(learning.default)


if __name__ == "__main__":
    unittest.main()
