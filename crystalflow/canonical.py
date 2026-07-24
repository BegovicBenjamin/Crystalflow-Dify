"""Canonical JSON serialization and content hashing.

Only JSON-native values are accepted.  Object keys are sorted, insignificant
whitespace is removed, finite numbers are normalized, and duplicate keys can
be rejected at parse time with :func:`canonical_loads`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

from .errors import CanonicalizationError, PathPart

_EXPONENT = re.compile(r"^(?P<mantissa>.+)[eE](?P<sign>[+-]?)(?P<digits>\d+)$")


def _fail(message: str, path: tuple[PathPart, ...], code: str) -> None:
    raise CanonicalizationError(message, code=code, path=path)


def _string(value: str, path: tuple[PathPart, ...]) -> str:
    # JSON can spell lone surrogates, but such strings cannot be encoded as
    # canonical UTF-8 and are a frequent cross-runtime source of mismatches.
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        _fail("strings must not contain lone UTF-16 surrogates", path, "INVALID_STRING")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _float(value: float, path: tuple[PathPart, ...]) -> str:
    if not math.isfinite(value):
        _fail("numbers must be finite", path, "NON_FINITE_NUMBER")
    if value == 0.0:
        return "0"

    absolute = abs(value)
    spelling = repr(value).lower()

    # Match the useful JSON/ECMAScript fixed-point range. Decimal is only used
    # to change notation; it receives the already-short, deterministic repr.
    if 1e-6 <= absolute < 1e21:
        if "e" in spelling:
            spelling = format(Decimal(spelling), "f")
        if "." in spelling:
            spelling = spelling.rstrip("0").rstrip(".")
        return spelling

    match = _EXPONENT.fullmatch(spelling)
    if match is None:
        return spelling
    mantissa = match.group("mantissa")
    if mantissa.endswith(".0"):
        mantissa = mantissa[:-2]
    exponent = match.group("digits").lstrip("0") or "0"
    sign = "-" if match.group("sign") == "-" else "+"
    return f"{mantissa}e{sign}{exponent}"


def _render(
    value: Any,
    path: tuple[PathPart, ...],
    active: set[int],
) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _string(value, path)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _float(value, path)

    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            _fail("cyclic values are not JSON", path, "CYCLIC_VALUE")
        active.add(identity)
        try:
            items: list[tuple[str, Any]] = []
            for key, child in value.items():
                if not isinstance(key, str):
                    _fail("object keys must be strings", path, "INVALID_OBJECT_KEY")
                items.append((key, child))
            items.sort(key=lambda pair: pair[0])
            return (
                "{"
                + ",".join(
                    f"{_string(key, path + (key,))}:{_render(child, path + (key,), active)}"
                    for key, child in items
                )
                + "}"
            )
        finally:
            active.remove(identity)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        identity = id(value)
        if identity in active:
            _fail("cyclic values are not JSON", path, "CYCLIC_VALUE")
        active.add(identity)
        try:
            return (
                "["
                + ",".join(
                    _render(child, path + (index,), active) for index, child in enumerate(value)
                )
                + "]"
            )
        finally:
            active.remove(identity)

    _fail(
        f"unsupported value type {type(value).__name__!r}",
        path,
        "UNSUPPORTED_JSON_TYPE",
    )
    raise AssertionError("unreachable")


def canonical_json(value: Any) -> str:
    """Return the canonical JSON text for *value*."""

    try:
        return _render(value, (), set())
    except CanonicalizationError:
        raise
    except RecursionError:
        raise CanonicalizationError(
            "JSON nesting is too deep",
            code="NESTING_LIMIT",
        ) from None
    except (OverflowError, ValueError):
        raise CanonicalizationError(
            "number cannot be represented as canonical JSON",
            code="INVALID_NUMBER",
        ) from None


def canonical_json_bytes(value: Any) -> bytes:
    """Return canonical JSON encoded as UTF-8."""

    return canonical_json(value).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return the lowercase SHA-256 hex digest of canonical JSON."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def content_hash(value: Any) -> str:
    """Alias used by the registry for content-addressed values."""

    return canonical_sha256(value)


def canonical_hash(value: Any) -> str:
    """Backward-friendly alias for :func:`canonical_sha256`."""

    return canonical_sha256(value)


def canonical_loads(text: str | bytes | bytearray) -> Any:
    """Parse JSON while rejecting duplicate keys and non-finite constants."""

    def reject_constant(constant: str) -> Any:
        raise CanonicalizationError(
            f"non-finite JSON number {constant!r} is not allowed",
            code="NON_FINITE_NUMBER",
        )

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CanonicalizationError(
                    f"duplicate object key {key!r}",
                    code="DUPLICATE_KEY",
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except CanonicalizationError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError, ValueError) as exc:
        raise CanonicalizationError(
            "invalid JSON text",
            code="INVALID_JSON_TEXT",
            details={"line": getattr(exc, "lineno", 0), "column": getattr(exc, "colno", 0)},
        ) from None
    # Reuse the canonical renderer as a complete JSON-native/finite check.
    canonical_json(value)
    return value


def canonicalize_json(text: str | bytes | bytearray) -> str:
    """Parse and return canonical text in one operation."""

    return canonical_json(canonical_loads(text))


__all__ = [
    "canonical_hash",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_loads",
    "canonical_sha256",
    "canonicalize_json",
    "content_hash",
]
