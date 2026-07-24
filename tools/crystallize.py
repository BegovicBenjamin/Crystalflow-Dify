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
    require_object,
    require_test_cases,
    result_messages,
)

_VARIABLES = (
    "status",
    "crystal_name",
    "version",
    "program_hash",
    "tests_passed",
    "test_count",
    "active",
    "created",
    "message",
)


class CrystallizeTool(Tool):
    def _invoke(
        self,
        tool_parameters: dict[str, Any],
    ) -> Generator[ToolInvokeMessage]:
        name = str(tool_parameters.get("crystal_name") or "")
        payload: dict[str, Any] = {
            "status": "invalid",
            "crystal_name": name,
            "version": 0,
            "program_hash": "",
            "tests_passed": False,
            "test_count": 0,
            "active": False,
            "created": False,
            "message": "",
        }
        try:
            namespace = str(tool_parameters.get("namespace") or "default")
            description = tool_parameters.get("description")
            if not isinstance(description, str):
                raise ToolInputError("description must be a string")
            program = require_object(
                parse_strict_json(tool_parameters.get("program_json"), "program_json"),
                "program_json",
            )
            tests = require_test_cases(
                parse_strict_json(tool_parameters.get("tests_json"), "tests_json")
            )
            activation_policy = str(tool_parameters.get("activation_policy") or "draft")
            estimate = as_non_negative_int(
                tool_parameters.get("estimated_tokens_per_run", 0),
                "estimated_tokens_per_run",
            )
            service = CrystalService(self.session.storage, namespace)
            payload = service.crystallize(
                name=name,
                description=description,
                program=program,
                tests=tests,
                activation_policy=activation_policy,
                estimated_tokens_per_run=estimate,
            )
        except (ToolInputError, ServiceValidationError) as exc:
            payload["message"] = str(exc)
        except CrystalFlowError as exc:
            payload["message"] = f"{exc.code}: {exc.message} at {exc.path_string}"
        except RegistryError as exc:
            payload["status"] = "error"
            payload["message"] = str(exc)
        except Exception:
            payload["status"] = "error"
            payload["message"] = "CrystalFlow encountered an unexpected internal error."

        yield from result_messages(self, payload, _VARIABLES)
