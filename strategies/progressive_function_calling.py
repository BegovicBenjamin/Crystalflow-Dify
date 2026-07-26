import hashlib
import json
import re
import time
from collections.abc import Generator, Iterable, Mapping
from contextlib import suppress
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

from dify_plugin.entities.agent import AgentInvokeMessage
from dify_plugin.entities.model import ModelFeature
from dify_plugin.entities.model.llm import (
    LLMModelConfig,
    LLMResult,
    LLMResultChunk,
    LLMUsage,
)
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageContentType,
    SystemPromptMessage,
    ToolPromptMessage,
    UserPromptMessage,
)
from dify_plugin.entities.tool import ToolInvokeMessage, ToolProviderType
from dify_plugin.interfaces.agent import AgentModelConfig, AgentStrategy, ToolEntity
from pydantic import BaseModel, Field, field_validator

from crystalflow.adaptive_routes import (
    AdaptiveRouteStore,
    RouteDecision,
    ToolAction,
    ToolContract,
)

ROUTE_NAMESPACE = "crystalflow.agent-routes.v1"
MAXIMUM_ITERATIONS = 3
DIAGNOSTIC_FINGERPRINT_LENGTH = 16
SUPPORTED_CONTRACT_SCHEMA_KEYS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minProperties",
        "maxProperties",
    }
)
IGNORED_SCHEMA_ANNOTATIONS = frozenset(
    {
        "$comment",
        "default",
        "deprecated",
        "description",
        "examples",
        "readOnly",
        "title",
        "writeOnly",
    }
)


class ContextItem(BaseModel):
    content: str = ""
    title: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProgressiveFunctionCallingParams(BaseModel):
    query: str
    instruction: str | None = None
    model: AgentModelConfig
    tools: list[ToolEntity] = Field(default_factory=list)
    crystallizable_tools: list[ToolEntity] = Field(default_factory=list)
    context: list[ContextItem] | None = None
    routing_context: Any = None
    threshold: int = Field(default=5, ge=2, le=20)

    @field_validator("tools", "crystallizable_tools", mode="before")
    @classmethod
    def normalize_tool_lists(cls, value: object) -> object:
        if value is None:
            return []
        if not isinstance(value, list):
            return value

        normalized: list[object] = []
        for item in value:
            if isinstance(item, Mapping) and item.get("provider_type") == "plugin":
                # Current Dify distinguishes plugin tools from legacy built-ins,
                # while SDK 0.9.x still invokes both through the built-in channel.
                item = {**item, "provider_type": ToolProviderType.BUILT_IN.value}
            normalized.append(item)
        return normalized


class ToolCall:
    def __init__(
        self,
        call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        self.call_id = call_id
        self.tool_name = tool_name
        self.arguments = arguments


class ToolRun:
    def __init__(
        self,
        *,
        call: ToolCall,
        tool: ToolEntity,
        prompt_output: str,
        direct_messages: list[AgentInvokeMessage],
        succeeded: bool,
    ) -> None:
        self.call = call
        self.tool = tool
        self.prompt_output = prompt_output
        self.direct_messages = direct_messages
        self.succeeded = succeeded


class ModelTurn:
    def __init__(
        self,
        *,
        text: str,
        calls: list[ToolCall],
        usage: LLMUsage | None,
    ) -> None:
        self.text = text
        self.calls = calls
        self.usage = usage


class StrategyExecutionError(Exception):
    """Attach a safe execution stage without exposing exception details."""

    def __init__(self, stage: str, cause: Exception) -> None:
        self.stage = stage
        self.cause = cause
        super().__init__(stage)


def _enum_value(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _tool_key(tool: ToolEntity) -> tuple[str, str, str]:
    return (
        _enum_value(tool.provider_type),
        tool.identity.provider,
        tool.identity.name,
    )


def _decision_value(decision: RouteDecision | None, name: str, default: object) -> object:
    if decision is None:
        return default
    return getattr(decision, name, default)


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(getattr(item, "data", "")) for item in content)
    return ""


def _error_diagnostic(exc: Exception) -> dict[str, str]:
    if isinstance(exc, StrategyExecutionError):
        stage = exc.stage
        cause = exc.cause
    else:
        stage = "strategy_execution"
        cause = exc
    error_type = type(cause).__name__
    raw_error_code = getattr(cause, "code", None)
    error_code = (
        raw_error_code
        if isinstance(raw_error_code, str) and re.fullmatch(r"[a-z][a-z0-9-]{0,63}", raw_error_code)
        else "UNCLASSIFIED"
    )
    fingerprint = _sha256(f"{stage}:{error_type}:{error_code}")[:DIAGNOSTIC_FINGERPRINT_LENGTH]
    return {
        "stage": stage,
        "error_type": error_type,
        "error_code": error_code,
        "diagnostic_id": fingerprint,
    }


class ProgressiveFunctionCallingAgentStrategy(AgentStrategy):
    """Function calling with an exact, deterministic fast path for proven routes."""

    def _invoke(
        self,
        parameters: dict[str, Any],
    ) -> Generator[AgentInvokeMessage, None, None]:
        try:
            yield from self._invoke_progressive(parameters)
        except Exception as exc:
            diagnostic = _error_diagnostic(exc)
            yield self.create_log_message(
                label=f"CRYSTALFLOW ERROR · {diagnostic['stage']}",
                data=diagnostic,
                status=ToolInvokeMessage.LogMessage.LogStatus.ERROR,
            )
            yield self.create_text_message(
                "CrystalFlow could not complete this request "
                f"({diagnostic['stage']}, {diagnostic['error_type']}, "
                f"{diagnostic['diagnostic_id']})."
            )
            yield self.create_json_message(
                {
                    "execution_metadata": self._execution_metadata(None),
                    "crystalflow": {
                        "path": "error",
                        "llm_calls": None,
                        "status": "error",
                        "reason_code": "strategy_error",
                        **diagnostic,
                    },
                }
            )

    def _invoke_progressive(
        self,
        parameters: dict[str, Any],
    ) -> Generator[AgentInvokeMessage, None, None]:
        try:
            params = ProgressiveFunctionCallingParams(**parameters)
        except Exception as exc:
            raise StrategyExecutionError("parameter_validation", exc) from exc
        try:
            tools_by_name = self._tools_by_name(params.tools)
            contracts = self._fast_path_contracts(
                params.tools,
                params.crystallizable_tools,
            )
            scope = self._route_scope(params)
            context = self._routing_context_binding(params.routing_context)
        except Exception as exc:
            raise StrategyExecutionError("strategy_setup", exc) from exc
        routes: AdaptiveRouteStore | None = None
        try:
            routes = AdaptiveRouteStore(
                self.session.storage,
                ROUTE_NAMESPACE,
                scope=scope,
                threshold=params.threshold,
            )
            decision = routes.lookup(params.query, context=context, tools=contracts)
        except Exception:
            decision = self._unavailable_route_decision(params.threshold)

        if decision.hit and decision.plan is not None:
            try:
                warm_run = self._run_crystal(decision, tools_by_name, contracts)
            except Exception:
                warm_run = None
            if warm_run is not None:
                telemetry = None
                if routes is not None:
                    with suppress(Exception):
                        telemetry = routes.record_hit(
                            decision.route_key,
                            estimated_tokens_avoided=self._estimate_tokens_avoided(params),
                        )
                yield from self._yield_warm_run(warm_run, decision, telemetry)
                return

        yield from self._run_cold_path(
            params=params,
            tools_by_name=tools_by_name,
            contracts=contracts,
            routes=routes,
            context=context,
            initial_decision=decision,
        )

    def _run_cold_path(
        self,
        *,
        params: ProgressiveFunctionCallingParams,
        tools_by_name: dict[str, ToolEntity],
        contracts: list[ToolContract],
        routes: AdaptiveRouteStore | None,
        context: str | None,
        initial_decision: RouteDecision,
    ) -> Generator[AgentInvokeMessage, None, None]:
        model = params.model
        prompt_tools = self._init_prompt_tools(params.tools)
        messages = list(deepcopy(model.history_prompt_messages))
        if params.instruction:
            messages.insert(0, SystemPromptMessage(content=params.instruction))
        messages.append(UserPromptMessage(content=params.query))

        stream = bool(
            model.entity
            and model.entity.features
            and ModelFeature.STREAM_TOOL_CALL in model.entity.features
        )
        stop = model.completion_params.get("stop", []) if model.completion_params else []
        total_usage: dict[str, LLMUsage | None] = {"usage": None}
        successful_runs: list[ToolRun] = []
        all_runs: list[ToolRun] = []
        final_text_emitted = False
        llm_calls = 0

        for iteration in range(1, MAXIMUM_ITERATIONS + 1):
            round_started = time.perf_counter()
            round_log = self.create_log_message(
                label=f"ROUND {iteration}",
                data={},
                metadata={"started_at": round_started},
                status=ToolInvokeMessage.LogMessage.LogStatus.START,
            )
            yield round_log

            prompt_messages = self._organize_prompt_messages(messages, model, iteration)
            if model.entity and model.completion_params:
                self.recalc_llm_max_tokens(
                    model.entity,
                    prompt_messages,
                    model.completion_params,
                )

            model_started = time.perf_counter()
            model_log = self.create_log_message(
                label=f"{model.model} Thought",
                data={},
                metadata={"started_at": model_started, "provider": model.provider},
                parent=round_log,
                status=ToolInvokeMessage.LogMessage.LogStatus.START,
            )
            yield model_log
            turn = self._invoke_model(
                model=model,
                messages=prompt_messages,
                stop=stop,
                stream=stream,
                prompt_tools=prompt_tools,
            )
            llm_calls += 1
            if turn.usage is not None:
                self.increase_usage(total_usage, turn.usage)

            yield self.finish_log_message(
                log=model_log,
                data={
                    "output": turn.text,
                    "tool_input": [
                        {"name": call.tool_name, "args": call.arguments} for call in turn.calls
                    ],
                },
                metadata={
                    "started_at": model_started,
                    "finished_at": time.perf_counter(),
                    "elapsed_time": time.perf_counter() - model_started,
                    "provider": model.provider,
                    "total_tokens": turn.usage.total_tokens if turn.usage else 0,
                    "total_price": turn.usage.total_price if turn.usage else 0,
                    "currency": turn.usage.currency if turn.usage else "",
                },
            )

            if not turn.calls:
                if turn.text:
                    yield self.create_text_message(turn.text)
                    final_text_emitted = True
                yield self.finish_log_message(
                    log=round_log,
                    data={"output": {"llm_response": turn.text, "tool_responses": []}},
                    metadata={
                        "started_at": round_started,
                        "finished_at": time.perf_counter(),
                        "elapsed_time": time.perf_counter() - round_started,
                    },
                )
                break

            messages.append(self._assistant_tool_call_message(turn))

            if len(turn.calls) == 1:
                direct_call = turn.calls[0]
                direct_tool = tools_by_name.get(direct_call.tool_name)
                if direct_tool is not None and self._is_contract_tool(
                    direct_tool,
                    contracts,
                ):
                    run, invoke_messages = self._run_tool(
                        direct_call,
                        tools_by_name,
                        round_log,
                    )
                    for invoke_message in invoke_messages:
                        yield invoke_message
                    all_runs.append(run)
                    if run.succeeded:
                        successful_runs.append(run)

                    yield self.finish_log_message(
                        log=round_log,
                        data={
                            "output": {
                                "llm_response": turn.text,
                                "tool_responses": [
                                    {
                                        "tool_call_name": direct_call.tool_name,
                                        "tool_call_input": direct_call.arguments,
                                        "tool_response": run.prompt_output,
                                        "succeeded": run.succeeded,
                                    }
                                ],
                            }
                        },
                        metadata={
                            "started_at": round_started,
                            "finished_at": time.perf_counter(),
                            "elapsed_time": time.perf_counter() - round_started,
                        },
                    )

                    if run.succeeded:
                        observation = initial_decision
                        if (
                            self._route_is_grounded(
                                params.query,
                                direct_call.arguments,
                                context,
                            )
                            and routes is not None
                        ):
                            with suppress(Exception):
                                observation = routes.observe_success(
                                    params.query,
                                    ToolAction(
                                        provider_type=_enum_value(direct_tool.provider_type),
                                        provider=direct_tool.identity.provider,
                                        tool_name=direct_tool.identity.name,
                                        arguments=direct_call.arguments,
                                    ),
                                    context=context,
                                    tools=contracts,
                                )
                        yield from self._yield_direct_tool_output(run)
                        if params.context:
                            yield self._retriever_resources(params.context)
                        yield self.create_json_message(
                            {
                                "execution_metadata": self._execution_metadata(
                                    total_usage["usage"]
                                ),
                                "crystalflow": self._route_metadata(
                                    "cold",
                                    observation,
                                    llm_calls=llm_calls,
                                ),
                            }
                        )
                        return

                    messages.append(
                        ToolPromptMessage(
                            content=run.prompt_output,
                            tool_call_id=direct_call.call_id,
                            name=direct_call.tool_name,
                        )
                    )
                    continue

            round_outputs: list[dict[str, Any]] = []
            for call in turn.calls:
                run, invoke_messages = self._run_tool(call, tools_by_name, round_log)
                for invoke_message in invoke_messages:
                    yield invoke_message
                all_runs.append(run)
                if run.succeeded:
                    successful_runs.append(run)
                round_outputs.append(
                    {
                        "tool_call_name": call.tool_name,
                        "tool_call_input": call.arguments,
                        "tool_response": run.prompt_output,
                        "succeeded": run.succeeded,
                    }
                )
                messages.append(
                    ToolPromptMessage(
                        content=run.prompt_output,
                        tool_call_id=call.call_id,
                        name=call.tool_name,
                    )
                )

            yield self.finish_log_message(
                log=round_log,
                data={
                    "output": {
                        "llm_response": turn.text,
                        "tool_responses": round_outputs,
                    }
                },
                metadata={
                    "started_at": round_started,
                    "finished_at": time.perf_counter(),
                    "elapsed_time": time.perf_counter() - round_started,
                },
            )

            for prompt_tool in prompt_tools:
                tool = tools_by_name.get(prompt_tool.name)
                if tool is not None:
                    self.update_prompt_message_tool(tool, prompt_tool)

            if iteration == MAXIMUM_ITERATIONS:
                for run in successful_runs[-1:]:
                    yield from self._yield_direct_tool_output(run)
                final_text_emitted = bool(successful_runs)

        if not final_text_emitted and not successful_runs:
            yield self.create_text_message("I could not complete the requested tool route.")

        if params.context:
            yield self._retriever_resources(params.context)
        yield self.create_json_message(
            {
                "execution_metadata": self._execution_metadata(total_usage["usage"]),
                "crystalflow": self._route_metadata(
                    "cold",
                    initial_decision,
                    llm_calls=llm_calls,
                ),
            }
        )

    def _run_crystal(
        self,
        decision: RouteDecision,
        tools_by_name: dict[str, ToolEntity],
        contracts: list[ToolContract],
    ) -> ToolRun | None:
        plan = decision.plan
        if plan is None:
            return None
        tool = tools_by_name.get(plan.tool_name)
        if tool is None or not self._plan_matches_tool(plan, tool):
            return None
        if not self._is_contract_tool(tool, contracts):
            return None

        call = ToolCall(
            call_id=f"crystal:{decision.route_key}",
            tool_name=plan.tool_name,
            arguments=dict(plan.arguments),
        )
        run, _logs = self._run_tool(call, tools_by_name, parent=None, emit_logs=False)
        return run if run.succeeded else None

    def _yield_warm_run(
        self,
        run: ToolRun,
        decision: RouteDecision,
        telemetry: object | None,
    ) -> Generator[AgentInvokeMessage, None, None]:
        started = time.perf_counter()
        log = self.create_log_message(
            label=f"CRYSTAL HIT · {run.tool.identity.name}",
            data={},
            metadata={"started_at": started, "provider": run.tool.identity.provider},
            status=ToolInvokeMessage.LogMessage.LogStatus.START,
        )
        yield log
        yield from self._yield_direct_tool_output(run)
        yield self.finish_log_message(
            log=log,
            data={
                "output": run.prompt_output,
                "route_key": decision.route_key,
                "llm_calls": 0,
            },
            metadata={
                "started_at": started,
                "finished_at": time.perf_counter(),
                "elapsed_time": time.perf_counter() - started,
                "provider": run.tool.identity.provider,
                "total_tokens": 0,
                "total_price": 0,
            },
        )
        yield self.create_json_message(
            {
                "execution_metadata": self._execution_metadata(None),
                "crystalflow": self._route_metadata(
                    "warm",
                    decision,
                    llm_calls=0,
                    telemetry=telemetry,
                ),
            }
        )

    def _yield_direct_tool_output(
        self,
        run: ToolRun,
    ) -> Generator[AgentInvokeMessage, None, None]:
        has_text = False
        for message in run.direct_messages:
            if message.type == AgentInvokeMessage.MessageType.TEXT and getattr(
                message.message, "text", ""
            ):
                has_text = True
            yield message
        if not has_text:
            yield self.create_text_message(run.prompt_output or "The tool completed successfully.")

    def _run_tool(
        self,
        call: ToolCall,
        tools_by_name: dict[str, ToolEntity],
        parent: AgentInvokeMessage | None,
        *,
        emit_logs: bool = True,
    ) -> tuple[ToolRun, list[AgentInvokeMessage]]:
        tool = tools_by_name.get(call.tool_name)
        if tool is None:
            missing = ToolEntity.model_construct()
            return (
                ToolRun(
                    call=call,
                    tool=missing,
                    prompt_output=f"No configured tool named {call.tool_name!r}.",
                    direct_messages=[],
                    succeeded=False,
                ),
                [],
            )

        started = time.perf_counter()
        emitted: list[AgentInvokeMessage] = []
        log: AgentInvokeMessage | None = None
        if emit_logs:
            log = self.create_log_message(
                label=f"CALL {tool.identity.name}",
                data={},
                metadata={"started_at": started, "provider": tool.identity.provider},
                parent=parent,
                status=ToolInvokeMessage.LogMessage.LogStatus.START,
            )
            emitted.append(log)

        try:
            # Persisted crystals contain only LLM-supplied arguments. Current runtime
            # parameters win so credentials and app configuration are never replayed.
            invoke_parameters = {
                **call.arguments,
                **dict(tool.runtime_parameters),
            }
            responses = list(
                self.session.tool.invoke(
                    provider_type=ToolProviderType(tool.provider_type),
                    provider=tool.identity.provider,
                    tool_name=tool.identity.name,
                    parameters=invoke_parameters,
                )
            )
            direct_messages = [
                AgentInvokeMessage.model_validate(response.model_dump()) for response in responses
            ]
            prompt_output = self._tool_prompt_output(responses)
            succeeded = any(
                response.type
                in {
                    ToolInvokeMessage.MessageType.TEXT,
                    ToolInvokeMessage.MessageType.JSON,
                    ToolInvokeMessage.MessageType.LINK,
                    ToolInvokeMessage.MessageType.IMAGE,
                    ToolInvokeMessage.MessageType.IMAGE_LINK,
                    ToolInvokeMessage.MessageType.BLOB,
                    ToolInvokeMessage.MessageType.BLOB_CHUNK,
                    ToolInvokeMessage.MessageType.FILE,
                    ToolInvokeMessage.MessageType.VARIABLE,
                }
                for response in responses
            )
            error: str | None = None
        except Exception as exc:  # Dify surfaces provider errors through the stream.
            direct_messages = []
            prompt_output = f"Tool invocation failed: {exc}"
            succeeded = False
            error = str(exc)

        if emit_logs and log is not None:
            emitted.append(
                self.finish_log_message(
                    log=log,
                    status=(
                        ToolInvokeMessage.LogMessage.LogStatus.SUCCESS
                        if succeeded
                        else ToolInvokeMessage.LogMessage.LogStatus.ERROR
                    ),
                    error=error,
                    data={
                        "output": prompt_output,
                        "tool_input": call.arguments,
                    },
                    metadata={
                        "started_at": started,
                        "finished_at": time.perf_counter(),
                        "elapsed_time": time.perf_counter() - started,
                        "provider": tool.identity.provider,
                    },
                )
            )

        return (
            ToolRun(
                call=call,
                tool=tool,
                prompt_output=prompt_output,
                direct_messages=direct_messages,
                succeeded=succeeded,
            ),
            emitted,
        )

    def _invoke_model(
        self,
        *,
        model: AgentModelConfig,
        messages: list[PromptMessage],
        stop: list[str],
        stream: bool,
        prompt_tools: list,
    ) -> ModelTurn:
        try:
            result = self.session.model.llm.invoke(
                model_config=LLMModelConfig(**model.model_dump(mode="json")),
                prompt_messages=messages,
                stop=stop,
                stream=stream,
                tools=prompt_tools,
            )
        except Exception as exc:
            raise StrategyExecutionError("model_invoke", exc) from exc

        if isinstance(result, LLMResult):
            try:
                calls = [
                    ToolCall(
                        call_id=call.id,
                        tool_name=call.function.name,
                        arguments=self._parse_arguments(call.function.arguments),
                    )
                    for call in result.message.tool_calls
                ]
                return ModelTurn(
                    text=_content_text(result.message.content),
                    calls=calls,
                    usage=result.usage,
                )
            except Exception as exc:
                raise StrategyExecutionError("model_response_parse", exc) from exc

        text_parts: list[str] = []
        raw_calls: list[AssistantPromptMessage.ToolCall] = []
        position_slots: dict[int, int] = {}
        id_slots: dict[str, int] = {}
        usage: LLMUsage | None = None
        iterator = iter(cast(Iterable[LLMResultChunk], result))
        while True:
            try:
                chunk = next(iterator)
            except StopIteration:
                break
            except Exception as exc:
                raise StrategyExecutionError("model_invoke", exc) from exc

            try:
                message = chunk.delta.message
                text_parts.append(_content_text(message.content))
                for position, call in enumerate(message.tool_calls):
                    slot = self._streamed_call_slot(
                        call,
                        position=position,
                        raw_calls=raw_calls,
                        position_slots=position_slots,
                        id_slots=id_slots,
                    )
                    previous = raw_calls[slot] if slot < len(raw_calls) else None
                    merged = self._merge_streamed_call(previous, call)
                    if previous is None:
                        raw_calls.append(merged)
                    else:
                        raw_calls[slot] = merged
                    position_slots[position] = slot
                    if merged.id:
                        id_slots[merged.id] = slot
                if chunk.delta.usage is not None:
                    usage = chunk.delta.usage
            except Exception as exc:
                raise StrategyExecutionError("model_response_parse", exc) from exc

        try:
            calls = [
                ToolCall(
                    call_id=call.id or f"index:{index}",
                    tool_name=call.function.name,
                    arguments=self._parse_arguments(call.function.arguments),
                )
                for index, call in enumerate(raw_calls)
                if call.id or call.function.name or call.function.arguments
            ]
        except Exception as exc:
            raise StrategyExecutionError("model_response_parse", exc) from exc
        return ModelTurn(text="".join(text_parts), calls=calls, usage=usage)

    @staticmethod
    def _streamed_call_slot(
        call: AssistantPromptMessage.ToolCall,
        *,
        position: int,
        raw_calls: list[AssistantPromptMessage.ToolCall],
        position_slots: dict[int, int],
        id_slots: dict[str, int],
    ) -> int:
        if call.id and call.id in id_slots:
            return id_slots[call.id]

        positioned = position_slots.get(position)
        if positioned is not None:
            previous = raw_calls[positioned]
            if not call.id or not previous.id or previous.id == call.id:
                return positioned

        if not call.id and len(raw_calls) == 1:
            return 0

        return len(raw_calls)

    @staticmethod
    def _merge_streamed_call(
        previous: AssistantPromptMessage.ToolCall | None,
        current: AssistantPromptMessage.ToolCall,
    ) -> AssistantPromptMessage.ToolCall:
        if previous is None:
            return current.model_copy(deep=True)

        old_name = previous.function.name
        new_name = current.function.name
        name = new_name if new_name.startswith(old_name) else old_name + new_name
        old_arguments = previous.function.arguments
        new_arguments = current.function.arguments
        arguments = (
            new_arguments
            if new_arguments.startswith(old_arguments)
            else old_arguments + new_arguments
        )
        return AssistantPromptMessage.ToolCall(
            id=current.id or previous.id,
            type=current.type or previous.type,
            function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                name=name,
                arguments=arguments,
            ),
        )

    @staticmethod
    def _parse_arguments(arguments: str) -> dict[str, Any]:
        if not arguments:
            return {}
        value = json.loads(arguments)
        if not isinstance(value, dict):
            raise ValueError("Tool-call arguments must be a JSON object")
        return value

    @staticmethod
    def _assistant_tool_call_message(turn: ModelTurn) -> AssistantPromptMessage:
        return AssistantPromptMessage(
            content=turn.text,
            tool_calls=[
                AssistantPromptMessage.ToolCall(
                    id=call.call_id,
                    type="function",
                    function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                        name=call.tool_name,
                        arguments=_canonical_json(call.arguments),
                    ),
                )
                for call in turn.calls
            ],
        )

    def _fast_path_contracts(
        self,
        tools: list[ToolEntity],
        crystallizable_tools: list[ToolEntity],
    ) -> list[ToolContract]:
        allowed = {_tool_key(tool) for tool in crystallizable_tools}
        contracts: list[ToolContract] = []
        for tool in tools:
            if _tool_key(tool) not in allowed:
                continue
            try:
                prompt_schema = self._convert_tool_to_prompt_message_tool(tool).parameters
                schema = self._contract_schema(prompt_schema)
                contract_version = _sha256(
                    _canonical_json(
                        {
                            "tool_description": (
                                tool.description.llm if tool.description is not None else ""
                            ),
                            "prompt_schema": prompt_schema,
                            "output_schema": tool.output_schema,
                        }
                    )
                )
                contracts.append(
                    ToolContract(
                        provider_type=_enum_value(tool.provider_type),
                        provider=tool.identity.provider,
                        tool_name=tool.identity.name,
                        arguments_schema=schema,
                        contract_version=contract_version,
                        read_only=True,
                    )
                )
            except Exception:
                # An unsupported parameter schema makes this tool ineligible for
                # crystallization, but must never stop the normal Agent path.
                continue
        return contracts

    @classmethod
    def _contract_schema(cls, schema: object) -> dict[str, Any]:
        if not isinstance(schema, Mapping):
            raise ValueError("Tool argument schema must be an object")

        result: dict[str, Any] = {}
        for key, value in schema.items():
            if key in IGNORED_SCHEMA_ANNOTATIONS:
                continue
            if key not in SUPPORTED_CONTRACT_SCHEMA_KEYS:
                raise ValueError(f"Unsupported tool schema keyword {key!r}")
            if key == "type":
                if isinstance(value, list):
                    result[key] = [_enum_value(item) for item in value]
                else:
                    result[key] = _enum_value(value)
            elif key == "properties":
                if not isinstance(value, Mapping):
                    raise ValueError("Tool schema properties must be an object")
                result[key] = {
                    str(name): cls._contract_schema(child) for name, child in value.items()
                }
            elif key == "items" or key == "additionalProperties" and isinstance(value, Mapping):
                result[key] = cls._contract_schema(value)
            else:
                result[key] = value

        if result.get("type") == "object":
            result.setdefault("additionalProperties", False)
        return result

    @staticmethod
    def _tools_by_name(tools: list[ToolEntity]) -> dict[str, ToolEntity]:
        result: dict[str, ToolEntity] = {}
        for tool in tools:
            if tool.identity.name in result:
                raise ValueError(f"Tool names must be unique; duplicate {tool.identity.name!r}")
            result[tool.identity.name] = tool
        return result

    @staticmethod
    def _is_contract_tool(tool: ToolEntity, contracts: list[ToolContract]) -> bool:
        key = _tool_key(tool)
        return any(
            key
            == (
                contract.provider_type,
                contract.provider,
                contract.tool_name,
            )
            for contract in contracts
        )

    @staticmethod
    def _plan_matches_tool(plan: object, tool: ToolEntity) -> bool:
        return (
            getattr(plan, "provider_type", None) == _enum_value(tool.provider_type)
            and getattr(plan, "provider", None) == tool.identity.provider
            and getattr(plan, "tool_name", None) == tool.identity.name
        )

    def _route_scope(self, params: ProgressiveFunctionCallingParams) -> str:
        toolset = sorted("|".join(_tool_key(tool)) for tool in params.tools)
        return _canonical_json(
            {
                "app_id": self.session.app_id or "unknown-app",
                "instruction_sha256": _sha256(params.instruction or ""),
                "toolset": toolset,
            }
        )

    @staticmethod
    def _unavailable_route_decision(threshold: int) -> RouteDecision:
        return cast(
            RouteDecision,
            SimpleNamespace(
                hit=False,
                plan=None,
                status="miss",
                reason_code="route_store_unavailable",
                route_key="",
                observations=0,
                threshold=threshold,
                hit_count=0,
                estimated_tokens_avoided=0,
            ),
        )

    @staticmethod
    def _routing_context_binding(routing_context: object) -> str | None:
        if routing_context is None:
            return None
        return _sha256(_canonical_json(routing_context))

    @staticmethod
    def _route_is_grounded(
        query: str,
        arguments: dict[str, Any],
        routing_context: str | None,
    ) -> bool:
        if routing_context is not None:
            return True
        normalized_query = query.casefold()
        if not re.search(
            (
                r"\b(this|that|these|those|it|its|same|above|previous|former|latter"
                r"|my|mine|me|our|ours|your|yours|their|theirs|his|her|hers)\b"
                r"|\bthe\s+(sop|document|file|policy|procedure)\b"
            ),
            normalized_query,
        ):
            return True
        normalized_query = " ".join(normalized_query.split())
        stable_values = [
            " ".join(str(value).casefold().split())
            for value in arguments.values()
            if isinstance(value, (str, int)) and len(str(value).strip()) >= 2
        ]
        return any(
            value != normalized_query
            and value not in {"this", "that", "it", "this sop", "that sop"}
            and value in normalized_query
            for value in stable_values
        )

    def _estimate_tokens_avoided(
        self,
        params: ProgressiveFunctionCallingParams,
    ) -> int:
        messages = list(params.model.history_prompt_messages)
        if params.instruction:
            messages.insert(0, SystemPromptMessage(content=params.instruction))
        messages.append(UserPromptMessage(content=params.query))
        try:
            message_tokens = self._get_num_tokens_by_gpt2(messages)
        except Exception:
            message_tokens = max(
                1,
                sum(len(message.get_text_content()) for message in messages) // 4,
            )
        prompt_tools = self._init_prompt_tools(params.tools)
        tool_characters = sum(
            len(prompt_tool.name)
            + len(prompt_tool.description)
            + len(_canonical_json(prompt_tool.parameters))
            for prompt_tool in prompt_tools
        )
        return message_tokens + tool_characters // 4

    @staticmethod
    def _tool_prompt_output(responses: list[ToolInvokeMessage]) -> str:
        parts: list[str] = []
        for response in responses:
            message = response.message
            if response.type in {
                ToolInvokeMessage.MessageType.TEXT,
                ToolInvokeMessage.MessageType.LINK,
                ToolInvokeMessage.MessageType.IMAGE,
                ToolInvokeMessage.MessageType.IMAGE_LINK,
            }:
                parts.append(str(getattr(message, "text", "")))
            elif response.type == ToolInvokeMessage.MessageType.JSON:
                parts.append(_canonical_json(getattr(message, "json_object", {})))
            elif response.type == ToolInvokeMessage.MessageType.BLOB:
                parts.append("Generated file")
            elif message is not None:
                parts.append(str(message))
        return "\n".join(part for part in parts if part)

    @staticmethod
    def _organize_prompt_messages(
        messages: list[PromptMessage],
        model: AgentModelConfig,
        iteration: int,
    ) -> list[PromptMessage]:
        supports_vision = bool(
            model.entity and model.entity.features and ModelFeature.VISION in model.entity.features
        )
        if supports_vision and iteration == 1:
            return list(messages)

        result = deepcopy(messages)
        for message in result:
            if isinstance(message, UserPromptMessage) and isinstance(message.content, list):
                message.content = "\n".join(
                    item.data
                    if item.type == PromptMessageContentType.TEXT
                    else f"[{item.type.value}]"
                    for item in message.content
                )
        return result

    def _retriever_resources(self, context: list[ContextItem]) -> AgentInvokeMessage:
        return self.create_retriever_resource_message(
            retriever_resources=[
                ToolInvokeMessage.RetrieverResourceMessage.RetrieverResource(
                    content=item.content,
                    position=item.metadata.get("position"),
                    dataset_id=item.metadata.get("dataset_id"),
                    dataset_name=item.metadata.get("dataset_name"),
                    document_id=item.metadata.get("document_id"),
                    document_name=item.metadata.get("document_name"),
                    data_source_type=item.metadata.get("document_data_source_type"),
                    segment_id=item.metadata.get("segment_id"),
                    retriever_from=item.metadata.get("retriever_from"),
                    score=item.metadata.get("score"),
                )
                for item in context
            ],
            context="",
        )

    @staticmethod
    def _execution_metadata(usage: LLMUsage | None) -> dict[str, Any]:
        if usage is None:
            return {
                "total_price": 0.0,
                "currency": "",
                "total_tokens": 0,
                "prompt_tokens": 0,
                "prompt_unit_price": 0.0,
                "prompt_price_unit": 0.0,
                "prompt_price": 0.0,
                "completion_tokens": 0,
                "completion_unit_price": 0.0,
                "completion_price_unit": 0.0,
                "completion_price": 0.0,
                "latency": 0.0,
            }
        return {
            "total_price": float(usage.total_price),
            "currency": usage.currency,
            "total_tokens": usage.total_tokens,
            "prompt_tokens": usage.prompt_tokens,
            "prompt_unit_price": float(usage.prompt_unit_price),
            "prompt_price_unit": float(usage.prompt_price_unit),
            "prompt_price": float(usage.prompt_price),
            "completion_tokens": usage.completion_tokens,
            "completion_unit_price": float(usage.completion_unit_price),
            "completion_price_unit": float(usage.completion_price_unit),
            "completion_price": float(usage.completion_price),
            "latency": usage.latency,
        }

    @staticmethod
    def _route_metadata(
        path: str,
        decision: RouteDecision | None,
        *,
        llm_calls: int,
        telemetry: object | None = None,
    ) -> dict[str, Any]:
        status = _decision_value(decision, "status", "miss")
        reason = _decision_value(decision, "reason_code", "")
        return {
            "path": path,
            "llm_calls": llm_calls,
            "status": _enum_value(status),
            "reason_code": _enum_value(reason),
            "route_key": _decision_value(decision, "route_key", ""),
            "observations": _decision_value(decision, "observations", 0),
            "threshold": _decision_value(decision, "threshold", 0),
            "hit_count": getattr(
                telemetry,
                "hit_count",
                _decision_value(decision, "hit_count", 0),
            ),
            "estimated_tokens_avoided": getattr(
                telemetry,
                "estimated_tokens_avoided",
                _decision_value(decision, "estimated_tokens_avoided", 0),
            ),
        }
