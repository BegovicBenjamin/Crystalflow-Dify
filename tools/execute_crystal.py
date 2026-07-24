from __future__ import annotations

from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from crystalflow.errors import CrystalFlowError
from crystalflow.registry import RegistryError
from crystalflow.service import CrystalService, ServiceValidationError
from tools._shared import (
    ToolInputError,
    as_non_negative_int,
    parse_strict_json,
    result_messages,
)

_VARIABLES = (
    "status",
    "fallback_required",
    "crystal_name",
    "version",
    "program_hash",
    "result_json",
    "receipt",
    "reason_code",
    "estimated_tokens_avoided",
    "telemetry_recorded",
)


class ExecuteCrystalTool(Tool):
    def _invoke(
        self,
        tool_parameters: dict[str, Any],
    ) -> Generator[ToolInvokeMessage]:
        name = str(tool_parameters.get("crystal_name") or "")
        payload: dict[str, Any] = {
            "status": "error",
            "fallback_required": True,
            "crystal_name": name,
            "version": 0,
            "program_hash": "",
            "result": None,
            "result_json": "",
            "receipt": "",
            "reason_code": "INVALID_REQUEST",
            "estimated_tokens_avoided": 0,
            "telemetry_recorded": False,
        }
        try:
            namespace = str(tool_parameters.get("namespace") or "default")
            inputs = parse_strict_json(tool_parameters.get("input_json"), "input_json")
            version = as_non_negative_int(
                tool_parameters.get("version", 0),
                "version",
                maximum=1_000,
            )
            service = CrystalService(self.session.storage, namespace)
            payload = service.execute(name=name, inputs=inputs, version=version)
        except (ToolInputError, ServiceValidationError) as exc:
            payload["status"] = "invalid_input"
            payload["reason_code"] = getattr(exc, "code", "INVALID_REQUEST")
        except CrystalFlowError as exc:
            payload["reason_code"] = exc.code
        except RegistryError:
            payload["reason_code"] = "REGISTRY_ERROR"
        except Exception:
            payload["reason_code"] = "INTERNAL_ERROR"

        yield from result_messages(self, payload, _VARIABLES)
