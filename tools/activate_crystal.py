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
    result_messages,
)

_VARIABLES = (
    "status",
    "crystal_name",
    "version",
    "program_hash",
    "active_version",
    "message",
)


class ActivateCrystalTool(Tool):
    def _invoke(
        self,
        tool_parameters: dict[str, Any],
    ) -> Generator[ToolInvokeMessage]:
        name = str(tool_parameters.get("crystal_name") or "")
        program_hash = str(tool_parameters.get("program_hash") or "")
        payload: dict[str, Any] = {
            "status": "invalid",
            "crystal_name": name,
            "version": 0,
            "program_hash": program_hash,
            "active_version": 0,
            "message": "",
        }
        try:
            namespace = str(tool_parameters.get("namespace") or "default")
            version = as_non_negative_int(
                tool_parameters.get("version"),
                "version",
                maximum=1_000,
            )
            confirmation = tool_parameters.get("confirmation")
            if not isinstance(confirmation, str):
                raise ToolInputError("confirmation must be a string")
            service = CrystalService(self.session.storage, namespace)
            payload = service.activate(
                name=name,
                version=version,
                program_hash=program_hash,
                confirmation=confirmation,
            )
        except (ToolInputError, ServiceValidationError) as exc:
            payload["message"] = str(exc)
        except (CrystalFlowError, RegistryError) as exc:
            payload["status"] = "error"
            payload["message"] = str(exc)
        except Exception:
            payload["status"] = "error"
            payload["message"] = "CrystalFlow encountered an unexpected internal error."

        yield from result_messages(self, payload, _VARIABLES)
