from __future__ import annotations

import json
import math
from collections.abc import Generator, Iterable
from dataclasses import asdict, is_dataclass
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

MAX_JSON_INPUT_BYTES = 65_536


class ToolInputError(ValueError):
    """A safe, user-correctable tool input error."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ToolInputError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_non_finite(token: str) -> None:
    raise ToolInputError(f"non-finite JSON number is not allowed: {token}")


def _parse_finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ToolInputError("JSON numbers must be finite")
    return value


def parse_strict_json(raw: Any, label: str) -> Any:
    if not isinstance(raw, str):
        raise ToolInputError(f"{label} must be a JSON string")
    try:
        raw_size = len(raw.encode("utf-8"))
    except UnicodeError as exc:
        raise ToolInputError(f"{label} is not valid UTF-8 JSON text") from exc
    if raw_size > MAX_JSON_INPUT_BYTES:
        raise ToolInputError(f"{label} exceeds {MAX_JSON_INPUT_BYTES} bytes")
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
            parse_float=_parse_finite_float,
        )
    except ToolInputError:
        raise
    except (RecursionError, json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise ToolInputError(f"{label} is not valid strict JSON") from exc


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ToolInputError(f"{label} must decode to a JSON object")
    return value


def require_test_cases(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ToolInputError("tests_json must be a non-empty JSON array")
    if len(value) > 100:
        raise ToolInputError("tests_json may contain at most 100 cases")

    tests: list[dict[str, Any]] = []
    for index, case in enumerate(value):
        if not isinstance(case, dict):
            raise ToolInputError(f"test case {index} must be an object")
        unknown = set(case) - {"name", "input", "expected"}
        if unknown:
            raise ToolInputError(f"test case {index} contains unsupported fields")
        if "input" not in case or "expected" not in case:
            raise ToolInputError(f"test case {index} needs input and expected")
        name = case.get("name", f"case_{index + 1}")
        if not isinstance(name, str) or not name or len(name) > 80:
            raise ToolInputError(f"test case {index} has an invalid name")
        tests.append({"name": name, "input": case["input"], "expected": case["expected"]})
    return tests


def as_non_negative_int(value: Any, label: str, *, maximum: int = 1_000_000) -> int:
    if value in (None, ""):
        return 0
    if isinstance(value, bool):
        raise ToolInputError(f"{label} must be a non-negative integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        result = int(value)
    else:
        raise ToolInputError(f"{label} must be a non-negative integer")
    if result < 0 or result > maximum:
        raise ToolInputError(f"{label} must be between 0 and {maximum}")
    return result


def plain_data(value: Any) -> Any:
    if is_dataclass(value):
        return {key: plain_data(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_data(item) for item in value]
    return value


def result_messages(
    tool: Tool,
    payload: dict[str, Any],
    variables: Iterable[str],
) -> Generator[ToolInvokeMessage]:
    for variable in variables:
        yield tool.create_variable_message(variable, payload.get(variable))
    yield tool.create_json_message(json=payload)
