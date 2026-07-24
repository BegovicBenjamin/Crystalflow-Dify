"""Stable, machine-readable errors raised by CrystalFlow.

The public exceptions deliberately carry a short ``code`` in addition to a
human-readable message.  Plugin callers should branch on the code, not on the
rendered exception text.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

PathPart = str | int
Path = Sequence[PathPart]

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def format_path(path: Path = ()) -> str:
    """Render a tuple-like path as deterministic JSONPath."""

    rendered = "$"
    for part in path:
        if isinstance(part, int):
            rendered += f"[{part}]"
        elif _IDENTIFIER.fullmatch(part):
            rendered += f".{part}"
        else:
            rendered += f"[{json.dumps(part, ensure_ascii=True)}]"
    return rendered


class CrystalFlowError(Exception):
    """Base class for all expected CrystalFlow failures."""

    default_code = "CRYSTALFLOW_ERROR"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        path: Path = (),
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = code or self.default_code
        self.message = message
        self.path = tuple(path)
        self.details = dict(details or {})
        super().__init__(self.__str__())

    @property
    def path_string(self) -> str:
        return format_path(self.path)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "path": self.path_string,
        }
        if self.details:
            result["details"] = self.details
        return result

    def __str__(self) -> str:
        return f"{self.code} at {self.path_string}: {self.message}"


class DSLValidationError(CrystalFlowError):
    """The program or expression tree is not valid CrystalFlow DSL."""

    default_code = "INVALID_DSL"


class SchemaDefinitionError(CrystalFlowError):
    """An input or output schema is malformed or unsupported."""

    default_code = "INVALID_SCHEMA"


class InputValidationError(CrystalFlowError):
    """Input data did not satisfy the declared input schema."""

    default_code = "INVALID_INPUT"


class OutputValidationError(CrystalFlowError):
    """The expression result did not satisfy the output schema."""

    default_code = "INVALID_OUTPUT"


class EvaluationError(CrystalFlowError):
    """A valid expression could not be evaluated for the supplied values."""

    default_code = "EVALUATION_ERROR"


class ResourceLimitError(EvaluationError):
    """A static or runtime resource budget was exceeded."""

    default_code = "RESOURCE_LIMIT"


class CanonicalizationError(CrystalFlowError):
    """A value is not canonicalizable JSON."""

    default_code = "INVALID_JSON"
