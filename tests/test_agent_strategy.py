from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from dify_plugin.entities.agent import AgentInvokeMessage, AgentRuntime
from dify_plugin.entities.model.llm import LLMResult, LLMUsage
from dify_plugin.entities.model.message import AssistantPromptMessage
from dify_plugin.entities.tool import ToolDescription, ToolInvokeMessage, ToolProviderType
from dify_plugin.interfaces.agent import (
    AgentModelConfig,
    AgentToolIdentity,
    ToolEntity,
)

from strategies.progressive_function_calling import (
    ProgressiveFunctionCallingAgentStrategy,
)


def make_tool() -> ToolEntity:
    return ToolEntity(
        identity=AgentToolIdentity(
            author="test",
            name="get_sop",
            provider="knowledge-workflow",
            label={"en_US": "Get SOP"},
        ),
        parameters=[],
        description=ToolDescription(
            human={"en_US": "Get SOP"},
            llm="Return an answer-ready SOP by its stable ID.",
        ),
        provider_type=ToolProviderType.WORKFLOW,
        runtime_parameters={"tenant_id": "current-tenant"},
    )


def make_model() -> AgentModelConfig:
    return AgentModelConfig(
        provider="test-provider",
        model="test-model",
        model_type="llm",
        mode="chat",
    )


def text_response(text: str) -> ToolInvokeMessage:
    return ToolInvokeMessage(
        type=ToolInvokeMessage.MessageType.TEXT,
        message=ToolInvokeMessage.TextMessage(text=text),
    )


def decision(*, hit: bool, plan: object | None = None, status: str = "miss") -> object:
    return SimpleNamespace(
        hit=hit,
        plan=plan,
        status=status,
        reason_code=status,
        route_key="route-1",
        observations=5 if hit else 0,
        threshold=5,
        hit_count=0,
        estimated_tokens_avoided=0,
    )


class ProgressiveFunctionCallingAgentStrategyTests(unittest.TestCase):
    def test_tool_description_change_invalidates_contract_fingerprint(self) -> None:
        original = make_tool()
        changed = original.model_copy(
            update={
                "description": ToolDescription(
                    human={"en_US": "Get SOP"},
                    llm="Return a different document contract.",
                )
            },
            deep=True,
        )
        strategy = ProgressiveFunctionCallingAgentStrategy(
            runtime=AgentRuntime(user_id="user-1"),
            session=SimpleNamespace(),
        )

        original_contract = strategy._fast_path_contracts([original], [original])[0]
        changed_contract = strategy._fast_path_contracts([changed], [changed])[0]

        self.assertNotEqual(
            original_contract.contract_hash,
            changed_contract.contract_hash,
        )

    def test_warm_hit_invokes_tool_without_model_and_runtime_parameters_win(self) -> None:
        tool = make_tool()
        plan = SimpleNamespace(
            provider_type="workflow",
            provider="knowledge-workflow",
            tool_name="get_sop",
            arguments={"sop_id": "SOP-42", "tenant_id": "stale-tenant"},
            contract_hash="contract",
        )
        route_store = MagicMock()
        route_store.lookup.return_value = decision(hit=True, plan=plan, status="hit")
        route_store.record_hit.return_value = SimpleNamespace(
            hit_count=1,
            estimated_tokens_avoided=100,
        )
        invoke_tool = MagicMock(return_value=iter([text_response("Current SOP text")]))
        session = SimpleNamespace(
            app_id="app-1",
            storage=object(),
            tool=SimpleNamespace(invoke=invoke_tool),
        )
        strategy = ProgressiveFunctionCallingAgentStrategy(
            runtime=AgentRuntime(user_id="user-1"),
            session=session,
        )

        with patch(
            "strategies.progressive_function_calling.AdaptiveRouteStore",
            return_value=route_store,
        ):
            messages = list(
                strategy.invoke(
                    {
                        "query": "What is in SOP-42?",
                        "instruction": "Use company tools.",
                        "model": make_model(),
                        "tools": [tool],
                        "crystallizable_tools": [tool],
                        "threshold": 5,
                    }
                )
            )

        invoke_tool.assert_called_once()
        self.assertEqual(
            invoke_tool.call_args.kwargs["parameters"],
            {"sop_id": "SOP-42", "tenant_id": "current-tenant"},
        )
        route_store.record_hit.assert_called_once()
        text = [
            message.message.text
            for message in messages
            if message.type == AgentInvokeMessage.MessageType.TEXT
        ]
        self.assertIn("Current SOP text", text)
        metadata = next(
            message.message.json_object
            for message in messages
            if message.type == AgentInvokeMessage.MessageType.JSON
        )
        self.assertEqual(metadata["execution_metadata"]["total_tokens"], 0)
        self.assertEqual(metadata["crystalflow"]["llm_calls"], 0)
        self.assertEqual(metadata["crystalflow"]["path"], "warm")

    def test_cold_selected_tool_is_observed_and_returned_without_second_model_call(
        self,
    ) -> None:
        tool = make_tool()
        route_store = MagicMock()
        route_store.lookup.return_value = decision(hit=False)
        route_store.observe_success.return_value = decision(
            hit=False,
            status="learning",
        )
        model_result = LLMResult(
            model="test-model",
            message=AssistantPromptMessage(
                content="",
                tool_calls=[
                    AssistantPromptMessage.ToolCall(
                        id="call-1",
                        type="function",
                        function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                            name="get_sop",
                            arguments=json.dumps({"sop_id": "SOP-42"}),
                        ),
                    )
                ],
            ),
            usage=LLMUsage.empty_usage(),
        )
        invoke_model = MagicMock(return_value=model_result)
        invoke_tool = MagicMock(return_value=iter([text_response("Current SOP text")]))
        session = SimpleNamespace(
            app_id="app-1",
            storage=object(),
            model=SimpleNamespace(llm=SimpleNamespace(invoke=invoke_model)),
            tool=SimpleNamespace(invoke=invoke_tool),
        )
        strategy = ProgressiveFunctionCallingAgentStrategy(
            runtime=AgentRuntime(user_id="user-1"),
            session=session,
        )

        with patch(
            "strategies.progressive_function_calling.AdaptiveRouteStore",
            return_value=route_store,
        ):
            messages = list(
                strategy.invoke(
                    {
                        "query": "What is in SOP-42?",
                        "instruction": "Use company tools.",
                        "model": make_model(),
                        "tools": [tool],
                        "crystallizable_tools": [tool],
                        "threshold": 5,
                    }
                )
            )

        invoke_model.assert_called_once()
        invoke_tool.assert_called_once()
        route_store.observe_success.assert_called_once()
        action = route_store.observe_success.call_args.args[1]
        self.assertEqual(action.arguments, {"sop_id": "SOP-42"})
        self.assertNotIn("tenant_id", action.arguments)
        text = [
            message.message.text
            for message in messages
            if message.type == AgentInvokeMessage.MessageType.TEXT
        ]
        self.assertIn("Current SOP text", text)
        metadata = next(
            message.message.json_object
            for message in messages
            if message.type == AgentInvokeMessage.MessageType.JSON
        )
        self.assertEqual(metadata["crystalflow"]["llm_calls"], 1)
        self.assertEqual(metadata["crystalflow"]["path"], "cold")

    def test_deictic_query_without_routing_context_is_not_learned(self) -> None:
        self.assertFalse(
            ProgressiveFunctionCallingAgentStrategy._route_is_grounded(
                "What is in that SOP?",
                {"sop_id": "SOP-42"},
                None,
            )
        )
        self.assertTrue(
            ProgressiveFunctionCallingAgentStrategy._route_is_grounded(
                "What is in that SOP?",
                {"sop_id": "SOP-42"},
                "selected-sop-context-hash",
            )
        )
        self.assertFalse(
            ProgressiveFunctionCallingAgentStrategy._route_is_grounded(
                "What is in that SOP?",
                {"query": "What is in that SOP?"},
                None,
            )
        )
        self.assertFalse(
            ProgressiveFunctionCallingAgentStrategy._route_is_grounded(
                "Show my PTO balance",
                {"employee_id": "employee-123"},
                None,
            )
        )


if __name__ == "__main__":
    unittest.main()
