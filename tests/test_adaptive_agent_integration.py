from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import yaml
from dify_plugin import DifyPluginEnv
from dify_plugin.core.plugin_registration import PluginRegistration
from dify_plugin.entities.agent import AgentInvokeMessage, AgentRuntime
from dify_plugin.entities.model.llm import (
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
    LLMUsage,
)
from dify_plugin.entities.model.message import AssistantPromptMessage
from dify_plugin.entities.tool import (
    ToolDescription,
    ToolInvokeMessage,
    ToolParameter,
    ToolProviderType,
)
from dify_plugin.interfaces.agent import AgentModelConfig, AgentToolIdentity, ToolEntity

from strategies.progressive_function_calling import (
    ProgressiveFunctionCallingAgentStrategy,
    ProgressiveFunctionCallingParams,
)

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


class QueuedLLM:
    def __init__(self, responses: list[LLMResult]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def invoke(self, **kwargs) -> LLMResult:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("a warm crystal must not invoke the model")
        return self.responses.pop(0)


class RecordingToolInvoker:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def invoke(self, **kwargs):
        self.calls.append(kwargs)
        sop_id = kwargs["parameters"].get("sop_id", "unknown")
        tenant_id = kwargs["parameters"].get("tenant_id", "unknown")
        yield text_response(f"{sop_id} for {tenant_id}")


def llm_parameter(name: str, *, required: bool = True) -> ToolParameter:
    return ToolParameter(
        name=name,
        label={"en_US": name},
        human_description={"en_US": name},
        llm_description=name,
        type=ToolParameter.ToolParameterType.STRING,
        form=ToolParameter.ToolParameterForm.LLM,
        required=required,
    )


def make_tool(
    *,
    name: str = "get_sop",
    provider: str = "knowledge-workflow",
    runtime_parameters: dict | None = None,
    parameter_names: tuple[str, ...] = ("sop_id",),
) -> ToolEntity:
    return ToolEntity(
        identity=AgentToolIdentity(
            author="test",
            name=name,
            provider=provider,
            label={"en_US": name},
        ),
        parameters=[llm_parameter(parameter) for parameter in parameter_names],
        description=ToolDescription(
            human={"en_US": name},
            llm=f"Return answer-ready content from {name}.",
        ),
        provider_type=ToolProviderType.WORKFLOW,
        runtime_parameters=runtime_parameters or {},
    )


def make_model() -> AgentModelConfig:
    return AgentModelConfig(
        provider="test-provider",
        model="test-model",
        model_type="llm",
        mode="chat",
    )


def tool_call_result(*calls: tuple[str, dict]) -> LLMResult:
    return LLMResult(
        model="test-model",
        message=AssistantPromptMessage(
            content="",
            tool_calls=[
                AssistantPromptMessage.ToolCall(
                    id=f"call-{index}",
                    type="function",
                    function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                        name=name,
                        arguments=json.dumps(arguments),
                    ),
                )
                for index, (name, arguments) in enumerate(calls, start=1)
            ],
        ),
        usage=LLMUsage.empty_usage(),
    )


def final_result(text: str) -> LLMResult:
    return LLMResult(
        model="test-model",
        message=AssistantPromptMessage(content=text),
        usage=LLMUsage.empty_usage(),
    )


def streamed_tool_call_chunk(
    *,
    call_id: str,
    name: str,
    arguments: str,
    index: int,
) -> LLMResultChunk:
    return LLMResultChunk(
        model="test-model",
        delta=LLMResultChunkDelta(
            index=index,
            message=AssistantPromptMessage(
                content="",
                tool_calls=[
                    AssistantPromptMessage.ToolCall(
                        id=call_id,
                        type="function",
                        function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                            name=name,
                            arguments=arguments,
                        ),
                    )
                ],
            ),
        ),
    )


def text_response(text: str) -> ToolInvokeMessage:
    return ToolInvokeMessage(
        type=ToolInvokeMessage.MessageType.TEXT,
        message=ToolInvokeMessage.TextMessage(text=text),
    )


def strategy_parameters(
    tools: list[ToolEntity],
    *,
    crystallizable_tools: list[ToolEntity] | None = None,
    query: str = "What is in SOP-42?",
    threshold: int = 2,
    routing_context: object | None = None,
) -> dict:
    return {
        "query": query,
        "instruction": "Use the current company's read-only knowledge tools.",
        "model": make_model(),
        "tools": tools,
        "crystallizable_tools": (tools if crystallizable_tools is None else crystallizable_tools),
        "threshold": threshold,
        "routing_context": routing_context,
    }


def create_strategy(
    *,
    storage: MemoryKV | object,
    llm: QueuedLLM,
    tool_invoker: RecordingToolInvoker | MagicMock,
) -> ProgressiveFunctionCallingAgentStrategy:
    session = SimpleNamespace(
        app_id="company-chatbot",
        storage=storage,
        model=SimpleNamespace(llm=llm),
        tool=tool_invoker,
    )
    return ProgressiveFunctionCallingAgentStrategy(
        runtime=AgentRuntime(user_id="employee-1"),
        session=session,
    )


class AdaptiveAgentIntegrationTests(unittest.TestCase):
    def assert_terminal_contract(self, messages: list[AgentInvokeMessage]) -> dict:
        text_messages = [
            message.message.text
            for message in messages
            if message.type == AgentInvokeMessage.MessageType.TEXT and message.message.text
        ]
        self.assertTrue(text_messages, "strategy did not yield final text")

        json_messages = [
            message.message.json_object
            for message in messages
            if message.type == AgentInvokeMessage.MessageType.JSON
        ]
        self.assertTrue(json_messages, "strategy did not yield execution metadata")
        payload = json_messages[-1]
        self.assertIn("execution_metadata", payload)
        self.assertIn("crystalflow", payload)
        return payload

    def test_manifest_registers_strategy_and_required_permissions(self) -> None:
        manifest = yaml.safe_load((ROOT / "manifest.yaml").read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["plugins"]["agent_strategies"],
            ["provider/crystalflow_agent.yaml"],
        )
        self.assertEqual(set(manifest["plugins"]), {"agent_strategies"})
        permission = manifest["resource"]["permission"]
        self.assertTrue(permission["model"]["enabled"])
        self.assertTrue(permission["model"]["llm"])
        self.assertTrue(permission["tool"]["enabled"])
        self.assertTrue(permission["storage"]["enabled"])

        registration = PluginRegistration(DifyPluginEnv(_env_file=None))
        self.assertIn("crystalflow_agent", registration.agent_strategies_mapping)
        self.assertIn(
            "progressive_function_calling",
            registration.agent_strategies_mapping["crystalflow_agent"][1],
        )

    def test_current_dify_plugin_tool_type_normalizes_for_both_tool_lists(
        self,
    ) -> None:
        raw_tool = {
            "identity": {
                "author": "Acme",
                "name": "get_sop",
                "provider": "acme/sop/sop",
                "label": {"en_US": "Get SOP"},
            },
            "parameters": [
                {
                    "name": "sop_id",
                    "label": {"en_US": "SOP ID"},
                    "human_description": {"en_US": "The SOP identifier."},
                    "llm_description": "The SOP identifier.",
                    "type": "string",
                    "form": "llm",
                    "required": True,
                }
            ],
            "description": {
                "human": {"en_US": "Retrieve an SOP."},
                "llm": "Retrieve an SOP.",
            },
            "runtime_parameters": {},
            "provider_type": "plugin",
        }
        raw_parameters = {
            "query": "What is in SOP-42?",
            "model": make_model().model_dump(mode="json"),
            "tools": [raw_tool],
            "crystallizable_tools": [raw_tool],
        }

        params = ProgressiveFunctionCallingParams.model_validate(raw_parameters)

        self.assertEqual(params.tools[0].provider_type, ToolProviderType.BUILT_IN)
        self.assertEqual(
            params.crystallizable_tools[0].provider_type,
            ToolProviderType.BUILT_IN,
        )
        self.assertEqual(params.tools[0].identity.provider, "acme/sop/sop")
        self.assertEqual(params.tools[0].identity.name, "get_sop")
        self.assertEqual(
            params.crystallizable_tools[0].identity.provider,
            "acme/sop/sop",
        )
        self.assertEqual(params.crystallizable_tools[0].identity.name, "get_sop")

    def test_cold_observations_activate_then_warm_hit_skips_llm(self) -> None:
        storage = MemoryKV()
        llm = QueuedLLM(
            [
                tool_call_result(("get_sop", {"sop_id": "SOP-42"})),
                tool_call_result(("get_sop", {"sop_id": "SOP-42"})),
            ]
        )
        tool_invoker = RecordingToolInvoker()
        strategy = create_strategy(storage=storage, llm=llm, tool_invoker=tool_invoker)
        first_tool = make_tool(
            runtime_parameters={
                "tenant_id": "tenant-v1",
                "api_key": "secret-v1",
            }
        )

        first = list(strategy.invoke(strategy_parameters([first_tool])))
        second = list(strategy.invoke(strategy_parameters([first_tool])))
        first_payload = self.assert_terminal_contract(first)
        second_payload = self.assert_terminal_contract(second)
        self.assertEqual(first_payload["crystalflow"]["path"], "cold")
        self.assertEqual(second_payload["crystalflow"]["path"], "cold")
        self.assertEqual(first_payload["crystalflow"]["llm_calls"], 1)
        self.assertEqual(second_payload["crystalflow"]["llm_calls"], 1)

        current_tool = make_tool(
            runtime_parameters={
                "tenant_id": "tenant-v2",
                "api_key": "secret-v2",
            }
        )
        warm = list(strategy.invoke(strategy_parameters([current_tool])))
        warm_payload = self.assert_terminal_contract(warm)

        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(len(tool_invoker.calls), 3)
        self.assertEqual(
            tool_invoker.calls[-1]["parameters"],
            {
                "sop_id": "SOP-42",
                "tenant_id": "tenant-v2",
                "api_key": "secret-v2",
            },
        )
        self.assertEqual(warm_payload["crystalflow"]["path"], "warm")
        self.assertEqual(warm_payload["crystalflow"]["llm_calls"], 0)
        self.assertEqual(warm_payload["execution_metadata"]["total_tokens"], 0)

        persisted = b"\n".join(storage.data.values())
        for runtime_value in (
            b"tenant-v1",
            b"secret-v1",
            b"tenant-v2",
            b"secret-v2",
        ):
            self.assertNotIn(runtime_value, persisted)

    def test_conflicting_tool_arguments_never_take_the_warm_path(self) -> None:
        storage = MemoryKV()
        llm = QueuedLLM(
            [
                tool_call_result(("get_sop", {"sop_id": "SOP-42"})),
                tool_call_result(("get_sop", {"sop_id": "SOP-99"})),
                tool_call_result(("get_sop", {"sop_id": "SOP-42"})),
            ]
        )
        tool_invoker = RecordingToolInvoker()
        strategy = create_strategy(storage=storage, llm=llm, tool_invoker=tool_invoker)
        tool = make_tool(runtime_parameters={"tenant_id": "current"})
        configured = strategy_parameters(
            [tool],
            query="What is in that SOP?",
            threshold=2,
            routing_context={"selected_sop": "current-selection"},
        )

        payloads = []
        for _ in range(3):
            messages = list(strategy.invoke(configured))
            payloads.append(self.assert_terminal_contract(messages))

        self.assertEqual(len(llm.calls), 3)
        self.assertEqual(len(tool_invoker.calls), 3)
        self.assertTrue(all(payload["crystalflow"]["path"] == "cold" for payload in payloads))
        self.assertNotEqual(payloads[1]["crystalflow"]["status"], "active")

    def test_schema_or_tool_identity_change_forces_cold_fallback(self) -> None:
        changes = {
            "schema": make_tool(parameter_names=("sop_id", "language")),
            "tool": make_tool(name="retrieve_sop"),
        }
        for change_name, changed_tool in changes.items():
            with self.subTest(change=change_name):
                storage = MemoryKV()
                llm = QueuedLLM(
                    [
                        tool_call_result(("get_sop", {"sop_id": "SOP-42"})),
                        tool_call_result(("get_sop", {"sop_id": "SOP-42"})),
                        final_result("I used the normal model route."),
                    ]
                )
                tool_invoker = RecordingToolInvoker()
                strategy = create_strategy(
                    storage=storage,
                    llm=llm,
                    tool_invoker=tool_invoker,
                )
                original = make_tool()

                list(strategy.invoke(strategy_parameters([original])))
                list(strategy.invoke(strategy_parameters([original])))
                fallback = list(strategy.invoke(strategy_parameters([changed_tool])))
                payload = self.assert_terminal_contract(fallback)

                self.assertEqual(len(llm.calls), 3)
                self.assertEqual(len(tool_invoker.calls), 2)
                self.assertEqual(payload["crystalflow"]["path"], "cold")
                self.assertEqual(payload["crystalflow"]["llm_calls"], 1)

    def test_noneligible_and_multi_tool_routes_are_not_observed(self) -> None:
        eligible = make_tool()
        second = make_tool(name="get_policy")
        cases = {
            "not explicitly eligible": {
                "tools": [eligible],
                "eligible": [],
                "first_result": tool_call_result(("get_sop", {"sop_id": "SOP-42"})),
                "expected_tool_calls": 1,
            },
            "multiple tool calls": {
                "tools": [eligible, second],
                "eligible": [eligible, second],
                "first_result": tool_call_result(
                    ("get_sop", {"sop_id": "SOP-42"}),
                    ("get_policy", {"sop_id": "POLICY-7"}),
                ),
                "expected_tool_calls": 2,
            },
        }

        for case_name, case in cases.items():
            with self.subTest(case=case_name):
                route_store = MagicMock()
                route_store.lookup.return_value = SimpleNamespace(
                    hit=False,
                    plan=None,
                    status="miss",
                    reason_code="not_found",
                    route_key="route-1",
                    observations=0,
                    threshold=2,
                    hit_count=0,
                    estimated_tokens_avoided=0,
                )
                llm = QueuedLLM(
                    [
                        case["first_result"],
                        final_result("The normal agent completed the request."),
                    ]
                )
                tool_invoker = RecordingToolInvoker()
                strategy = create_strategy(
                    storage=object(),
                    llm=llm,
                    tool_invoker=tool_invoker,
                )

                with patch(
                    "strategies.progressive_function_calling.AdaptiveRouteStore",
                    return_value=route_store,
                ):
                    messages = list(
                        strategy.invoke(
                            strategy_parameters(
                                case["tools"],
                                crystallizable_tools=case["eligible"],
                            )
                        )
                    )

                payload = self.assert_terminal_contract(messages)
                route_store.observe_success.assert_not_called()
                self.assertEqual(len(llm.calls), 2)
                self.assertEqual(
                    len(tool_invoker.calls),
                    case["expected_tool_calls"],
                )
                self.assertEqual(payload["crystalflow"]["path"], "cold")

    def test_streamed_tool_call_continuation_without_id_or_name_is_assembled(self) -> None:
        llm = MagicMock()
        llm.invoke.return_value = iter(
            [
                streamed_tool_call_chunk(
                    call_id="call-1",
                    name="get_sop",
                    arguments='{"sop_id":"SOP',
                    index=0,
                ),
                streamed_tool_call_chunk(
                    call_id="",
                    name="",
                    arguments='-42"}',
                    index=0,
                ),
            ]
        )
        strategy = create_strategy(
            storage=MemoryKV(),
            llm=llm,
            tool_invoker=RecordingToolInvoker(),
        )

        turn = strategy._invoke_model(
            model=make_model(),
            messages=[],
            stop=[],
            stream=True,
            prompt_tools=[],
        )

        self.assertEqual(len(turn.calls), 1)
        self.assertEqual(turn.calls[0].call_id, "call-1")
        self.assertEqual(turn.calls[0].tool_name, "get_sop")
        self.assertEqual(turn.calls[0].arguments, {"sop_id": "SOP-42"})
        self.assertTrue(llm.invoke.call_args.kwargs["stream"])

    def test_strategy_error_still_yields_text_and_execution_metadata(self) -> None:
        strategy = create_strategy(
            storage=MemoryKV(),
            llm=QueuedLLM([]),
            tool_invoker=RecordingToolInvoker(),
        )
        sensitive_error = "provider rejected workspace-secret-123"

        with patch.object(
            strategy,
            "_invoke_progressive",
            side_effect=RuntimeError(sensitive_error),
        ):
            messages = list(strategy.invoke({}))
        payload = self.assert_terminal_contract(messages)
        diagnostic = payload["crystalflow"]

        self.assertEqual(diagnostic["path"], "error")
        self.assertEqual(diagnostic["status"], "error")
        self.assertEqual(diagnostic["stage"], "strategy_execution")
        self.assertEqual(diagnostic["error_type"], "RuntimeError")
        self.assertRegex(diagnostic["diagnostic_id"], r"^[0-9a-f]{16}$")
        self.assertEqual(payload["execution_metadata"]["total_tokens"], 0)
        serialized_messages = "\n".join(message.model_dump_json() for message in messages)
        self.assertNotIn(sensitive_error, serialized_messages)
        self.assertNotIn("workspace-secret-123", serialized_messages)


if __name__ == "__main__":
    unittest.main()
