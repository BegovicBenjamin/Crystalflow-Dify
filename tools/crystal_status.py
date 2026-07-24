from __future__ import annotations

from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from crystalflow.errors import CrystalFlowError
from crystalflow.registry import RegistryError
from crystalflow.service import CrystalService, ServiceValidationError
from tools._shared import ToolInputError, result_messages

_VARIABLES = (
    "status",
    "crystal_name",
    "active_version",
    "crystal_count",
    "details_json",
    "total_runs",
    "estimated_tokens_avoided",
    "message",
)


class CrystalStatusTool(Tool):
    def _invoke(
        self,
        tool_parameters: dict[str, Any],
    ) -> Generator[ToolInvokeMessage]:
        name_value = tool_parameters.get("crystal_name")
        name = str(name_value) if name_value else None
        payload: dict[str, Any] = {
            "status": "error",
            "crystal_name": name or "",
            "active_version": 0,
            "crystal_count": 0,
            "details": {},
            "details_json": "{}",
            "total_runs": 0,
            "estimated_tokens_avoided": 0,
            "message": "",
        }
        try:
            namespace = str(tool_parameters.get("namespace") or "default")
            include_program_value = tool_parameters.get("include_program")
            if include_program_value is None:
                include_program = False
            elif isinstance(include_program_value, bool):
                include_program = include_program_value
            elif isinstance(include_program_value, int) and include_program_value in (0, 1):
                # Dify SDK 0.9.x models parameter defaults as numeric/string
                # scalars, so a Boolean form field can arrive as 0 or 1.
                include_program = bool(include_program_value)
            else:
                raise ToolInputError("include_program must be a boolean")
            service = CrystalService(self.session.storage, namespace)
            payload = service.status(
                name=name,
                include_program=include_program,
            )
        except (ToolInputError, ServiceValidationError) as exc:
            payload["status"] = "invalid"
            payload["message"] = str(exc)
        except (CrystalFlowError, RegistryError) as exc:
            payload["message"] = str(exc)
        except Exception:
            payload["message"] = "CrystalFlow encountered an unexpected internal error."

        yield from result_messages(self, payload, _VARIABLES)
