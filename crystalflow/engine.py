"""Deterministic CrystalFlow expression engine (DSL version 1).

CrystalFlow programs are JSON objects::

    {
      "version": 1,
      "input_schema": {"type": "object", ...},       # optional
      "output_schema": {"type": "object", ...},      # optional
      "expression": {"op": "object", "fields": {...}}
    }

Every expression is an object with an ``op`` and strictly checked operands.
Ordinary operators use an ``args`` array.  Data constructors use ``fields`` or
``items``.  Collection operators have this form::

    {"op": "map", "collection": EXPR, "var": "item", "body": EXPR}

``filter`` has the same shape; ``sort`` additionally permits the boolean
``descending`` key.  All three permit an optional ``index`` variable.  The
variables are lexical and exist only in the body.  ``sum`` takes a
``collection`` expression.

The operation set is:

* values: ``literal``, ``input``, ``var``, ``object``, ``array``, ``get``
* numbers: ``add``, ``sub``, ``mul``, ``div``, ``mod``, ``pow``, ``round``,
  ``min``, ``max``, ``abs``
* strings/collections: ``concat``, ``lower``, ``upper``, ``strip``,
  ``replace``, ``split``, ``join``, ``length``, ``slice``
* logic: ``eq``, ``ne``, ``lt``, ``lte``, ``gt``, ``gte``, ``and``, ``or``,
  ``not``, ``if``, ``coalesce``
* collections: ``map``, ``filter``, ``sort``, ``sum``

There is intentionally no source-code execution, ambient I/O, clock, random
source, or network access.  Given the same program, input, and limits, the
result or structured error is the same.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

from .canonical import canonical_json_bytes, canonical_loads
from .errors import (
    CrystalFlowError,
    DSLValidationError,
    EvaluationError,
    PathPart,
    ResourceLimitError,
    SchemaDefinitionError,
)
from .schema import validate_instance, validate_output, validate_schema

Expression = Mapping[str, Any]
Program = Mapping[str, Any]

ENGINE_VERSION = "1.0.0"
IR_VERSION = 1
MAX_CONFIGURED_DEPTH = 128

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MISSING = object()

_ARITIES: dict[str, tuple[int, int | None]] = {
    "add": (2, None),
    "sub": (2, 2),
    "mul": (2, None),
    "div": (2, 2),
    "mod": (2, 2),
    "pow": (2, 2),
    "round": (1, 2),
    "min": (1, None),
    "max": (1, None),
    "abs": (1, 1),
    "concat": (0, None),
    "lower": (1, 1),
    "upper": (1, 1),
    "strip": (1, 1),
    "replace": (3, 3),
    "split": (2, 2),
    "join": (2, 2),
    "length": (1, 1),
    "slice": (2, 3),
    "eq": (2, 2),
    "ne": (2, 2),
    "lt": (2, 2),
    "lte": (2, 2),
    "gt": (2, 2),
    "gte": (2, 2),
    "and": (1, None),
    "or": (1, None),
    "not": (1, 1),
    "if": (3, 3),
    "coalesce": (1, None),
}
_COLLECTION_OPS = frozenset({"map", "filter", "sort"})
_KNOWN_OPS = frozenset(
    {
        "literal",
        "input",
        "var",
        "object",
        "array",
        "get",
        "sum",
        *_ARITIES,
        *_COLLECTION_OPS,
    }
)


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Static and runtime limits for one deterministic execution."""

    max_nodes: int = 1_000
    max_depth: int = 32
    max_collection_items: int = 1_000
    max_evaluations: int = 100_000
    max_input_bytes: int = 65_536
    max_output_bytes: int = 262_144
    max_power_exponent: int = 1_000
    max_integer_bits: int = 4_096
    max_round_digits: int = 100

    def __post_init__(self) -> None:
        for descriptor in fields(self):
            value = getattr(self, descriptor.name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{descriptor.name} must be a positive integer")
        if self.max_depth > MAX_CONFIGURED_DEPTH:
            raise ValueError(f"max_depth must not exceed {MAX_CONFIGURED_DEPTH}")


DEFAULT_LIMITS = ResourceLimits()
Budget = ResourceLimits


def _coerce_limits(
    limits: ResourceLimits | Mapping[str, int] | None,
) -> ResourceLimits:
    if limits is None:
        return DEFAULT_LIMITS
    if isinstance(limits, ResourceLimits):
        return limits
    if not isinstance(limits, Mapping):
        raise TypeError("limits must be ResourceLimits, a mapping, or None")
    known = {descriptor.name for descriptor in fields(ResourceLimits)}
    unknown = sorted(set(limits) - known)
    if unknown:
        raise ValueError(f"unknown resource limit {unknown[0]!r}")
    return ResourceLimits(**dict(limits))


def _dsl_error(
    message: str,
    path: tuple[PathPart, ...],
    code: str,
) -> None:
    raise DSLValidationError(message, code=code, path=path)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _json_string_content_bytes(
    value: str,
    path: tuple[PathPart, ...],
    *,
    maximum: int | None = None,
) -> int:
    """Return the exact UTF-8 byte size inside canonical JSON string quotes.

    Counting is deliberately allocation-free.  In particular, callers can use
    it to reject an expanding string operation before Python constructs the
    result.
    """

    size = 0
    for character in value:
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            raise EvaluationError(
                "strings must not contain lone UTF-16 surrogates",
                code="INVALID_STRING",
                path=path,
            )
        if character in {'"', "\\", "\b", "\f", "\n", "\r", "\t"}:
            size += 2
        elif codepoint < 0x20:
            size += 6
        elif codepoint < 0x80:
            size += 1
        elif codepoint < 0x800:
            size += 2
        elif codepoint < 0x10000:
            size += 3
        else:
            size += 4
        if maximum is not None and size > maximum:
            return maximum + 1
    return size


def _bounded_json_size(
    value: Any,
    limits: ResourceLimits,
    max_bytes: int,
    *,
    size_code: str,
    size_message: str,
    path: tuple[PathPart, ...] = (),
    depth: int = 1,
    active: set[int] | None = None,
) -> int:
    """Validate and size canonical JSON, stopping as soon as *max_bytes* wins.

    This is the allocation-safe counterpart to ``canonical_json_bytes``.  It
    counts the exact bytes that serializer would emit, but never constructs the
    serialized document.
    """

    if depth > limits.max_depth:
        raise ResourceLimitError(
            f"value depth exceeds {limits.max_depth}",
            code="VALUE_DEPTH_LIMIT",
            path=path,
        )
    if active is None:
        active = set()

    def ensure(size: int, current_path: tuple[PathPart, ...]) -> int:
        if size > max_bytes:
            raise ResourceLimitError(
                size_message,
                code=size_code,
                path=current_path,
            )
        return size

    if value is None:
        return ensure(4, path)
    if value is True:
        return ensure(4, path)
    if value is False:
        return ensure(5, path)
    if isinstance(value, str):
        content_size = _json_string_content_bytes(
            value,
            path,
            maximum=max_bytes - 2 if max_bytes >= 2 else 0,
        )
        return ensure(content_size + 2, path)
    if isinstance(value, int):
        if value.bit_length() > limits.max_integer_bits:
            raise ResourceLimitError(
                f"integer exceeds {limits.max_integer_bits} bits",
                code="INTEGER_LIMIT",
                path=path,
            )
        try:
            digits = len(str(value))
        except ValueError:
            raise ResourceLimitError(
                "integer exceeds the runtime's deterministic serialization limit",
                code="INTEGER_LIMIT",
                path=path,
            ) from None
        return ensure(digits, path)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvaluationError(
                "numbers must be finite",
                code="NON_FINITE_NUMBER",
                path=path,
            )
        # A finite binary64 spelling is tiny, so using the canonical scalar
        # formatter cannot amplify attacker-controlled memory.
        return ensure(len(canonical_json_bytes(value)), path)

    if isinstance(value, Mapping):
        if len(value) > limits.max_collection_items:
            raise ResourceLimitError(
                f"collection has {len(value)} items; limit is {limits.max_collection_items}",
                code="COLLECTION_LIMIT",
                path=path,
            )
        identity = id(value)
        if identity in active:
            raise EvaluationError(
                "cyclic values are not JSON",
                code="CYCLIC_VALUE",
                path=path,
            )
        active.add(identity)
        try:
            total = 2
            ensure(total, path)
            for index, (key, child) in enumerate(value.items()):
                if not isinstance(key, str):
                    raise EvaluationError(
                        "object keys must be strings",
                        code="INVALID_OBJECT_KEY",
                        path=path,
                    )
                if index:
                    total = ensure(total + 1, path)
                remaining = max_bytes - total
                key_size = (
                    _json_string_content_bytes(
                        key,
                        path + (key,),
                        maximum=remaining - 2 if remaining >= 2 else 0,
                    )
                    + 2
                )
                total = ensure(total + key_size + 1, path + (key,))
                child_size = _bounded_json_size(
                    child,
                    limits,
                    max_bytes - total,
                    size_code=size_code,
                    size_message=size_message,
                    path=path + (key,),
                    depth=depth + 1,
                    active=active,
                )
                total = ensure(total + child_size, path + (key,))
            return total
        finally:
            active.remove(identity)

    if _is_sequence(value):
        if len(value) > limits.max_collection_items:
            raise ResourceLimitError(
                f"collection has {len(value)} items; limit is {limits.max_collection_items}",
                code="COLLECTION_LIMIT",
                path=path,
            )
        identity = id(value)
        if identity in active:
            raise EvaluationError(
                "cyclic values are not JSON",
                code="CYCLIC_VALUE",
                path=path,
            )
        active.add(identity)
        try:
            total = 2
            ensure(total, path)
            for index, child in enumerate(value):
                if index:
                    total = ensure(total + 1, path)
                child_size = _bounded_json_size(
                    child,
                    limits,
                    max_bytes - total,
                    size_code=size_code,
                    size_message=size_message,
                    path=path + (index,),
                    depth=depth + 1,
                    active=active,
                )
                total = ensure(total + child_size, path + (index,))
            return total
        finally:
            active.remove(identity)

    raise EvaluationError(
        f"value type {type(value).__name__!r} is not JSON",
        code="UNSUPPORTED_JSON_TYPE",
        path=path,
    )


def _copy_error_with_prefix(
    error: CrystalFlowError,
    prefix: tuple[PathPart, ...],
) -> CrystalFlowError:
    return type(error)(
        error.message,
        code=error.code,
        path=prefix + error.path,
        details=error.details,
    )


class _Validator:
    def __init__(
        self,
        limits: ResourceLimits,
        allowed_variables: frozenset[str],
    ) -> None:
        self.limits = limits
        self.allowed_variables = allowed_variables
        self.nodes = 0
        self.active: set[int] = set()

    def validate(self, expression: Any) -> None:
        self._node(expression, (), 1, self.allowed_variables)

    def _keys(
        self,
        node: Mapping[str, Any],
        required: set[str],
        optional: set[str],
        path: tuple[PathPart, ...],
    ) -> None:
        for key in node:
            if not isinstance(key, str):
                _dsl_error("expression keys must be strings", path, "INVALID_KEY")
        missing = sorted(required - set(node))
        if missing:
            _dsl_error(
                f"missing required key {missing[0]!r}",
                path + (missing[0],),
                "MISSING_KEY",
            )
        unknown = sorted(set(node) - required - optional)
        if unknown:
            _dsl_error(
                f"unknown key {unknown[0]!r}",
                path + (unknown[0],),
                "UNKNOWN_KEY",
            )

    def _name(self, value: Any, path: tuple[PathPart, ...]) -> str:
        if not isinstance(value, str) or _NAME.fullmatch(value) is None:
            _dsl_error(
                "variable names must match [A-Za-z_][A-Za-z0-9_]*",
                path,
                "INVALID_VARIABLE_NAME",
            )
        return value

    def _args(
        self,
        node: Mapping[str, Any],
        path: tuple[PathPart, ...],
        depth: int,
        scope: frozenset[str],
        minimum: int,
        maximum: int | None,
    ) -> None:
        self._keys(node, {"op", "args"}, set(), path)
        args = node["args"]
        if not _is_sequence(args):
            _dsl_error("args must be an array", path + ("args",), "INVALID_ARGS")
        count = len(args)
        if count < minimum or (maximum is not None and count > maximum):
            expected = (
                str(minimum)
                if maximum == minimum
                else f"{minimum}..{maximum if maximum is not None else 'many'}"
            )
            _dsl_error(
                f"operation expects {expected} arguments, got {count}",
                path + ("args",),
                "INVALID_ARITY",
            )
        if count > self.limits.max_collection_items:
            raise ResourceLimitError(
                "argument array exceeds max_collection_items",
                code="COLLECTION_LIMIT",
                path=path + ("args",),
            )
        for index, child in enumerate(args):
            self._node(child, path + ("args", index), depth + 1, scope)

    def _node(
        self,
        node: Any,
        path: tuple[PathPart, ...],
        depth: int,
        scope: frozenset[str],
    ) -> None:
        if depth > self.limits.max_depth:
            raise ResourceLimitError(
                f"expression depth exceeds {self.limits.max_depth}",
                code="DEPTH_LIMIT",
                path=path,
            )
        self.nodes += 1
        if self.nodes > self.limits.max_nodes:
            raise ResourceLimitError(
                f"expression nodes exceed {self.limits.max_nodes}",
                code="NODE_LIMIT",
                path=path,
            )
        if not isinstance(node, Mapping):
            _dsl_error("expression must be an object", path, "EXPRESSION_NOT_OBJECT")

        identity = id(node)
        if identity in self.active:
            _dsl_error("cyclic expression trees are not allowed", path, "CYCLIC_EXPRESSION")
        self.active.add(identity)
        try:
            if "op" not in node:
                _dsl_error("missing required key 'op'", path + ("op",), "MISSING_KEY")
            operation = node["op"]
            if not isinstance(operation, str):
                _dsl_error("op must be a string", path + ("op",), "INVALID_OPERATION")
            if operation not in _KNOWN_OPS:
                _dsl_error(
                    f"unknown operation {operation!r}",
                    path + ("op",),
                    "UNKNOWN_OPERATION",
                )

            if operation == "literal":
                self._keys(node, {"op", "value"}, set(), path)
                try:
                    _bounded_json_size(
                        node["value"],
                        self.limits,
                        self.limits.max_output_bytes,
                        size_code="OUTPUT_LIMIT",
                        size_message="literal exceeds max_output_bytes",
                        path=path + ("value",),
                    )
                except EvaluationError as exc:
                    if isinstance(exc, ResourceLimitError):
                        raise
                    raise DSLValidationError(
                        f"literal is not JSON: {exc.message}",
                        code=exc.code,
                        path=exc.path,
                    ) from None
                return

            if operation == "input":
                self._keys(node, {"op"}, {"name"}, path)
                if "name" in node and not isinstance(node["name"], str):
                    _dsl_error(
                        "input name must be a string",
                        path + ("name",),
                        "INVALID_INPUT_NAME",
                    )
                return

            if operation == "var":
                self._keys(node, {"op", "name"}, set(), path)
                name = self._name(node["name"], path + ("name",))
                if name not in scope:
                    _dsl_error(
                        f"variable {name!r} is not in scope",
                        path + ("name",),
                        "UNBOUND_VARIABLE",
                    )
                return

            if operation == "object":
                self._keys(node, {"op", "fields"}, set(), path)
                object_fields = node["fields"]
                if not isinstance(object_fields, Mapping):
                    _dsl_error(
                        "object fields must be an object",
                        path + ("fields",),
                        "INVALID_FIELDS",
                    )
                if len(object_fields) > self.limits.max_collection_items:
                    raise ResourceLimitError(
                        "object fields exceed max_collection_items",
                        code="COLLECTION_LIMIT",
                        path=path + ("fields",),
                    )
                for name, child in object_fields.items():
                    if not isinstance(name, str):
                        _dsl_error(
                            "object field names must be strings",
                            path + ("fields",),
                            "INVALID_FIELD_NAME",
                        )
                    self._node(child, path + ("fields", name), depth + 1, scope)
                return

            if operation == "array":
                self._keys(node, {"op", "items"}, set(), path)
                items = node["items"]
                if not _is_sequence(items):
                    _dsl_error(
                        "array items must be an array",
                        path + ("items",),
                        "INVALID_ITEMS",
                    )
                if len(items) > self.limits.max_collection_items:
                    raise ResourceLimitError(
                        "array items exceed max_collection_items",
                        code="COLLECTION_LIMIT",
                        path=path + ("items",),
                    )
                for index, child in enumerate(items):
                    self._node(child, path + ("items", index), depth + 1, scope)
                return

            if operation == "get":
                self._keys(node, {"op", "target", "key"}, {"default"}, path)
                self._node(node["target"], path + ("target",), depth + 1, scope)
                self._node(node["key"], path + ("key",), depth + 1, scope)
                if "default" in node:
                    self._node(node["default"], path + ("default",), depth + 1, scope)
                return

            if operation in _ARITIES:
                minimum, maximum = _ARITIES[operation]
                self._args(node, path, depth, scope, minimum, maximum)
                return

            if operation in _COLLECTION_OPS:
                optional = {"index"}
                if operation == "sort":
                    optional.add("descending")
                self._keys(
                    node,
                    {"op", "collection", "var", "body"},
                    optional,
                    path,
                )
                variable = self._name(node["var"], path + ("var",))
                nested_scope = scope | {variable}
                if "index" in node:
                    index_name = self._name(node["index"], path + ("index",))
                    if index_name == variable:
                        _dsl_error(
                            "index and item variables must have different names",
                            path + ("index",),
                            "DUPLICATE_VARIABLE",
                        )
                    nested_scope |= {index_name}
                if (
                    operation == "sort"
                    and "descending" in node
                    and not isinstance(node["descending"], bool)
                ):
                    _dsl_error(
                        "descending must be a boolean",
                        path + ("descending",),
                        "INVALID_DESCENDING",
                    )
                self._node(
                    node["collection"],
                    path + ("collection",),
                    depth + 1,
                    scope,
                )
                self._node(
                    node["body"],
                    path + ("body",),
                    depth + 1,
                    frozenset(nested_scope),
                )
                return

            if operation == "sum":
                self._keys(node, {"op", "collection"}, set(), path)
                self._node(
                    node["collection"],
                    path + ("collection",),
                    depth + 1,
                    scope,
                )
                return

            raise AssertionError(f"unhandled operation {operation}")
        finally:
            self.active.remove(identity)

    def _literal_collections(
        self,
        value: Any,
        path: tuple[PathPart, ...],
        active: set[int],
    ) -> None:
        if isinstance(value, Mapping):
            if len(value) > self.limits.max_collection_items:
                raise ResourceLimitError(
                    "literal object exceeds max_collection_items",
                    code="COLLECTION_LIMIT",
                    path=path,
                )
            identity = id(value)
            if identity in active:
                return
            active.add(identity)
            try:
                for key, child in value.items():
                    self._literal_collections(child, path + (key,), active)
            finally:
                active.remove(identity)
        elif _is_sequence(value):
            if len(value) > self.limits.max_collection_items:
                raise ResourceLimitError(
                    "literal array exceeds max_collection_items",
                    code="COLLECTION_LIMIT",
                    path=path,
                )
            identity = id(value)
            if identity in active:
                return
            active.add(identity)
            try:
                for index, child in enumerate(value):
                    self._literal_collections(child, path + (index,), active)
            finally:
                active.remove(identity)


def validate_expression(
    expression: Any,
    *,
    limits: ResourceLimits | Mapping[str, int] | None = None,
    allowed_variables: Sequence[str] = (),
) -> None:
    """Statically validate a version-1 expression tree."""

    names: set[str] = set()
    for index, name in enumerate(allowed_variables):
        if not isinstance(name, str) or _NAME.fullmatch(name) is None:
            _dsl_error(
                "allowed variable names must be identifiers",
                ("allowed_variables", index),
                "INVALID_VARIABLE_NAME",
            )
        names.add(name)
    _Validator(_coerce_limits(limits), frozenset(names)).validate(expression)


def _coerce_program(program: Any) -> Mapping[str, Any]:
    if isinstance(program, (str, bytes, bytearray)):
        program = canonical_loads(program)
    if not isinstance(program, Mapping):
        _dsl_error("program must be an object", (), "PROGRAM_NOT_OBJECT")
    return program


def validate_program(
    program: Any,
    *,
    limits: ResourceLimits | Mapping[str, int] | None = None,
) -> None:
    """Validate a complete program envelope and its optional schemas."""

    parsed = _coerce_program(program)
    required = {"version", "expression"}
    optional = {"input_schema", "output_schema"}
    for key in parsed:
        if not isinstance(key, str):
            _dsl_error("program keys must be strings", (), "INVALID_KEY")
    missing = sorted(required - set(parsed))
    if missing:
        _dsl_error(
            f"missing required program key {missing[0]!r}",
            (missing[0],),
            "MISSING_KEY",
        )
    unknown = sorted(set(parsed) - required - optional)
    if unknown:
        _dsl_error(
            f"unknown program key {unknown[0]!r}",
            (unknown[0],),
            "UNKNOWN_KEY",
        )
    if type(parsed["version"]) is not int or parsed["version"] != 1:
        _dsl_error(
            "version must be the integer 1",
            ("version",),
            "UNSUPPORTED_VERSION",
        )
    for key in ("input_schema", "output_schema"):
        if key in parsed:
            try:
                validate_schema(parsed[key])
            except SchemaDefinitionError as exc:
                raise _copy_error_with_prefix(exc, (key,)) from None
    try:
        validate_expression(parsed["expression"], limits=limits)
    except (DSLValidationError, ResourceLimitError) as exc:
        raise _copy_error_with_prefix(exc, ("expression",)) from None


class _Runtime:
    def __init__(
        self,
        limits: ResourceLimits,
        inputs: Any,
        variables: Mapping[str, Any],
    ) -> None:
        self.limits = limits
        self.inputs = inputs
        self.variables = dict(variables)
        self.evaluations = 0

    def _limit_collection(
        self,
        value: Mapping[Any, Any] | Sequence[Any],
        path: tuple[PathPart, ...],
    ) -> None:
        if len(value) > self.limits.max_collection_items:
            raise ResourceLimitError(
                f"collection has {len(value)} items; limit is {self.limits.max_collection_items}",
                code="COLLECTION_LIMIT",
                path=path,
            )

    def _number(self, value: Any, path: tuple[PathPart, ...]) -> int | float:
        if not _is_number(value):
            raise EvaluationError(
                f"expected number, got {type(value).__name__}",
                code="TYPE_MISMATCH",
                path=path,
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise EvaluationError(
                "numbers must be finite",
                code="NON_FINITE_NUMBER",
                path=path,
            )
        return value

    def _integer(self, value: Any, path: tuple[PathPart, ...]) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise EvaluationError(
                f"expected integer, got {type(value).__name__}",
                code="TYPE_MISMATCH",
                path=path,
            )
        return value

    def _boolean(self, value: Any, path: tuple[PathPart, ...]) -> bool:
        if not isinstance(value, bool):
            raise EvaluationError(
                f"expected boolean, got {type(value).__name__}",
                code="TYPE_MISMATCH",
                path=path,
            )
        return value

    def _string(self, value: Any, path: tuple[PathPart, ...]) -> str:
        if not isinstance(value, str):
            raise EvaluationError(
                f"expected string, got {type(value).__name__}",
                code="TYPE_MISMATCH",
                path=path,
            )
        return value

    def _array(self, value: Any, path: tuple[PathPart, ...]) -> Sequence[Any]:
        if not _is_sequence(value):
            raise EvaluationError(
                f"expected array, got {type(value).__name__}",
                code="TYPE_MISMATCH",
                path=path,
            )
        self._limit_collection(value, path)
        return value

    def _number_result(
        self,
        value: Any,
        path: tuple[PathPart, ...],
    ) -> int | float:
        if isinstance(value, complex) or not _is_number(value):
            raise EvaluationError(
                "numeric operation produced a non-real result",
                code="INVALID_NUMBER",
                path=path,
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise EvaluationError(
                "numeric operation produced a non-finite result",
                code="NON_FINITE_NUMBER",
                path=path,
            )
        if isinstance(value, int) and value.bit_length() > self.limits.max_integer_bits:
            raise ResourceLimitError(
                f"integer exceeds {self.limits.max_integer_bits} bits",
                code="INTEGER_LIMIT",
                path=path,
            )
        return value

    def _bounded_string(
        self,
        value: str,
        path: tuple[PathPart, ...],
    ) -> str:
        self._ensure_string_content_size(
            _json_string_content_bytes(
                value,
                path,
                maximum=max(self.limits.max_output_bytes - 2, 0),
            ),
            path,
        )
        return value

    def _ensure_string_content_size(
        self,
        content_bytes: int,
        path: tuple[PathPart, ...],
    ) -> None:
        # Canonical JSON adds two quote bytes around string content.
        if content_bytes + 2 > self.limits.max_output_bytes:
            raise ResourceLimitError(
                "string exceeds max_output_bytes",
                code="OUTPUT_LIMIT",
                path=path,
            )

    def _string_content_size(
        self,
        value: str,
        path: tuple[PathPart, ...],
    ) -> int:
        return _json_string_content_bytes(value, path)

    def _preflight_string_parts(
        self,
        parts: Sequence[tuple[str, tuple[PathPart, ...]]],
        path: tuple[PathPart, ...],
        *,
        separator: tuple[str, tuple[PathPart, ...]] | None = None,
    ) -> None:
        total = 0
        for value, value_path in parts:
            total += self._string_content_size(value, value_path)
            self._ensure_string_content_size(total, path)
        if separator is not None and len(parts) > 1:
            separator_value, separator_path = separator
            separator_size = self._string_content_size(
                separator_value,
                separator_path,
            )
            total += separator_size * (len(parts) - 1)
            self._ensure_string_content_size(total, path)

    def _preflight_inputs(self) -> None:
        _bounded_json_size(
            self.inputs,
            self.limits,
            self.limits.max_input_bytes,
            size_code="INPUT_LIMIT",
            size_message=f"input exceeds {self.limits.max_input_bytes} bytes",
            path=("inputs",),
        )

    def _intermediate_size(
        self,
        value: Any,
        maximum: int,
        path: tuple[PathPart, ...],
    ) -> int:
        return _bounded_json_size(
            value,
            self.limits,
            maximum,
            size_code="OUTPUT_LIMIT",
            size_message=(f"intermediate values exceed {self.limits.max_output_bytes} bytes"),
            path=path,
        )

    def _intermediate_total(
        self,
        total: int,
        path: tuple[PathPart, ...],
    ) -> int:
        if total > self.limits.max_output_bytes:
            raise ResourceLimitError(
                f"intermediate values exceed {self.limits.max_output_bytes} bytes",
                code="OUTPUT_LIMIT",
                path=path,
            )
        return total

    def evaluate(
        self,
        expression: Expression,
        *,
        inputs_preflighted: bool = False,
    ) -> Any:
        if not inputs_preflighted:
            self._preflight_inputs()
        _bounded_json_size(
            self.variables,
            self.limits,
            self.limits.max_input_bytes,
            size_code="INPUT_LIMIT",
            size_message=f"variables exceed {self.limits.max_input_bytes} bytes",
            path=("variables",),
        )
        result = self._eval(expression, self.variables, ())
        _bounded_json_size(
            result,
            self.limits,
            self.limits.max_output_bytes,
            size_code="OUTPUT_LIMIT",
            size_message=f"output exceeds {self.limits.max_output_bytes} bytes",
        )
        return result

    def _values(
        self,
        node: Expression,
        environment: Mapping[str, Any],
        path: tuple[PathPart, ...],
    ) -> list[Any]:
        values: list[Any] = []
        total = 2
        for index, child in enumerate(node["args"]):
            value_path = path + ("args", index)
            if index:
                total = self._intermediate_total(total + 1, value_path)
            value = self._eval(child, environment, value_path)
            value_size = self._intermediate_size(
                value,
                self.limits.max_output_bytes - total,
                value_path,
            )
            total = self._intermediate_total(total + value_size, value_path)
            values.append(value)
        return values

    def _eval(
        self,
        node: Expression,
        environment: Mapping[str, Any],
        path: tuple[PathPart, ...],
    ) -> Any:
        self.evaluations += 1
        if self.evaluations > self.limits.max_evaluations:
            raise ResourceLimitError(
                f"expression evaluations exceed {self.limits.max_evaluations}",
                code="EVALUATION_LIMIT",
                path=path,
            )
        operation = node["op"]

        if operation == "literal":
            return node["value"]

        if operation == "input":
            if "name" not in node:
                return self.inputs
            name = node["name"]
            if not isinstance(self.inputs, Mapping):
                raise EvaluationError(
                    "named input lookup requires an input object",
                    code="TYPE_MISMATCH",
                    path=path,
                )
            if name not in self.inputs:
                raise EvaluationError(
                    f"input {name!r} is missing",
                    code="MISSING_INPUT",
                    path=path + ("name",),
                )
            return self.inputs[name]

        if operation == "var":
            name = node["name"]
            if name not in environment:
                raise EvaluationError(
                    f"variable {name!r} is not bound",
                    code="UNBOUND_VARIABLE",
                    path=path + ("name",),
                )
            return environment[name]

        if operation == "object":
            result: dict[str, Any] = {}
            total = 2
            for index, (name, child) in enumerate(node["fields"].items()):
                value_path = path + ("fields", name)
                if index:
                    total = self._intermediate_total(total + 1, value_path)
                # Two key quotes plus the key/value colon.
                key_size = self._string_content_size(name, value_path) + 3
                total = self._intermediate_total(total + key_size, value_path)
                value = self._eval(child, environment, value_path)
                value_size = self._intermediate_size(
                    value,
                    self.limits.max_output_bytes - total,
                    value_path,
                )
                total = self._intermediate_total(total + value_size, value_path)
                result[name] = value
            return result

        if operation == "array":
            result: list[Any] = []
            total = 2
            for index, child in enumerate(node["items"]):
                value_path = path + ("items", index)
                if index:
                    total = self._intermediate_total(total + 1, value_path)
                value = self._eval(child, environment, value_path)
                value_size = self._intermediate_size(
                    value,
                    self.limits.max_output_bytes - total,
                    value_path,
                )
                total = self._intermediate_total(total + value_size, value_path)
                result.append(value)
            return result

        if operation == "get":
            target = self._eval(node["target"], environment, path + ("target",))
            key = self._eval(node["key"], environment, path + ("key",))
            value: Any = _MISSING
            if isinstance(target, Mapping):
                if not isinstance(key, str):
                    raise EvaluationError(
                        "object lookup key must be a string",
                        code="TYPE_MISMATCH",
                        path=path + ("key",),
                    )
                value = target.get(key, _MISSING)
            elif _is_sequence(target) or isinstance(target, str):
                index = self._integer(key, path + ("key",))
                if -len(target) <= index < len(target):
                    value = target[index]
            else:
                raise EvaluationError(
                    "get target must be an object, array, or string",
                    code="TYPE_MISMATCH",
                    path=path + ("target",),
                )
            if value is not _MISSING:
                return value
            if "default" in node:
                return self._eval(node["default"], environment, path + ("default",))
            raise EvaluationError(
                f"key or index {key!r} was not found",
                code="MISSING_KEY",
                path=path + ("key",),
            )

        # Lazy logical/control operators must be handled before eager args.
        if operation == "and":
            for index, child in enumerate(node["args"]):
                value = self._eval(child, environment, path + ("args", index))
                if not self._boolean(value, path + ("args", index)):
                    return False
            return True

        if operation == "or":
            for index, child in enumerate(node["args"]):
                value = self._eval(child, environment, path + ("args", index))
                if self._boolean(value, path + ("args", index)):
                    return True
            return False

        if operation == "if":
            condition = self._eval(node["args"][0], environment, path + ("args", 0))
            branch = 1 if self._boolean(condition, path + ("args", 0)) else 2
            return self._eval(
                node["args"][branch],
                environment,
                path + ("args", branch),
            )

        if operation == "coalesce":
            for index, child in enumerate(node["args"]):
                value = self._eval(child, environment, path + ("args", index))
                if value is not None:
                    return value
            return None

        if operation in _COLLECTION_OPS:
            collection_value = self._eval(
                node["collection"],
                environment,
                path + ("collection",),
            )
            collection = self._array(collection_value, path + ("collection",))
            variable = node["var"]
            index_variable = node.get("index")

            def nested(item: Any, index: int) -> dict[str, Any]:
                child_environment = dict(environment)
                child_environment[variable] = item
                if index_variable is not None:
                    child_environment[index_variable] = index
                return child_environment

            if operation == "map":
                mapped: list[Any] = []
                total = 2
                for index, item in enumerate(collection):
                    value_path = path + ("body", index)
                    if index:
                        total = self._intermediate_total(total + 1, value_path)
                    value = self._eval(
                        node["body"],
                        nested(item, index),
                        value_path,
                    )
                    value_size = self._intermediate_size(
                        value,
                        self.limits.max_output_bytes - total,
                        value_path,
                    )
                    total = self._intermediate_total(total + value_size, value_path)
                    mapped.append(value)
                return mapped

            if operation == "filter":
                filtered: list[Any] = []
                for index, item in enumerate(collection):
                    keep = self._eval(
                        node["body"],
                        nested(item, index),
                        path + ("body", index),
                    )
                    if self._boolean(keep, path + ("body", index)):
                        filtered.append(item)
                self._limit_collection(filtered, path)
                return filtered

            decorated: list[tuple[tuple[int, Any], int, Any]] = []
            key_total = 2
            for index, item in enumerate(collection):
                key_path = path + ("body", index)
                if index:
                    key_total = self._intermediate_total(key_total + 1, key_path)
                key = self._eval(
                    node["body"],
                    nested(item, index),
                    key_path,
                )
                key_size = self._intermediate_size(
                    key,
                    self.limits.max_output_bytes - key_total,
                    key_path,
                )
                key_total = self._intermediate_total(key_total + key_size, key_path)
                decorated.append((self._sort_key(key, key_path), index, item))
            decorated.sort(
                key=lambda entry: entry[0],
                reverse=node.get("descending", False),
            )
            result = [entry[2] for entry in decorated]
            self._limit_collection(result, path)
            return result

        if operation == "sum":
            value = self._eval(
                node["collection"],
                environment,
                path + ("collection",),
            )
            collection = self._array(value, path + ("collection",))
            total: int | float = 0
            for index, item in enumerate(collection):
                total = self._number_result(
                    total + self._number(item, path + ("collection", index)),
                    path,
                )
            return total

        values = self._values(node, environment, path)

        if operation in {"add", "sub", "mul", "div", "mod", "pow", "round", "min", "max", "abs"}:
            return self._numeric(operation, values, path)

        if operation == "concat":
            parts = [
                (
                    self._string(value, path + ("args", index)),
                    path + ("args", index),
                )
                for index, value in enumerate(values)
            ]
            self._preflight_string_parts(parts, path)
            return "".join(value for value, _ in parts)
        if operation in {"lower", "upper", "strip"}:
            string = self._string(values[0], path + ("args", 0))
            if operation == "lower":
                return self._bounded_string(string.lower(), path)
            if operation == "upper":
                return self._bounded_string(string.upper(), path)
            return self._bounded_string(string.strip(), path)
        if operation == "replace":
            source = self._string(values[0], path + ("args", 0))
            old = self._string(values[1], path + ("args", 1))
            new = self._string(values[2], path + ("args", 2))
            source_size = self._string_content_size(source, path + ("args", 0))
            new_size = self._string_content_size(new, path + ("args", 2))
            if old:
                old_size = self._string_content_size(old, path + ("args", 1))
                replacements = source.count(old)
                result_size = source_size + replacements * (new_size - old_size)
            else:
                result_size = source_size + (len(source) + 1) * new_size
            self._ensure_string_content_size(result_size, path)
            return source.replace(old, new)
        if operation == "split":
            source = self._string(values[0], path + ("args", 0))
            separator = self._string(values[1], path + ("args", 1))
            if separator == "":
                raise EvaluationError(
                    "split separator must not be empty",
                    code="INVALID_SEPARATOR",
                    path=path + ("args", 1),
                )
            item_count = source.count(separator) + 1
            if item_count > self.limits.max_collection_items:
                raise ResourceLimitError(
                    f"collection has {item_count} items; limit is "
                    f"{self.limits.max_collection_items}",
                    code="COLLECTION_LIMIT",
                    path=path,
                )
            source_size = self._string_content_size(source, path + ("args", 0))
            separator_size = self._string_content_size(separator, path + ("args", 1))
            separator_count = item_count - 1
            # Exact canonical JSON size of the future string array: source
            # content minus separators, plus quotes, commas, and brackets.
            result_size = source_size - separator_count * separator_size + 3 * item_count + 1
            self._intermediate_total(result_size, path)
            result = source.split(separator)
            return result
        if operation == "join":
            separator = self._string(values[0], path + ("args", 0))
            collection = self._array(values[1], path + ("args", 1))
            parts = [
                self._string(item, path + ("args", 1, index))
                for index, item in enumerate(collection)
            ]
            parts_with_paths = [
                (item, path + ("args", 1, index)) for index, item in enumerate(parts)
            ]
            self._preflight_string_parts(
                parts_with_paths,
                path,
                separator=(separator, path + ("args", 0)),
            )
            return separator.join(parts)
        if operation == "length":
            target = values[0]
            if isinstance(target, (str, Mapping)) or _is_sequence(target):
                self._limit_collection(target, path) if not isinstance(target, str) else None
                return len(target)
            raise EvaluationError(
                "length target must be a string, array, or object",
                code="TYPE_MISMATCH",
                path=path + ("args", 0),
            )
        if operation == "slice":
            target = values[0]
            if not (isinstance(target, str) or _is_sequence(target)):
                raise EvaluationError(
                    "slice target must be a string or array",
                    code="TYPE_MISMATCH",
                    path=path + ("args", 0),
                )
            start = self._integer(values[1], path + ("args", 1))
            end = self._integer(values[2], path + ("args", 2)) if len(values) == 3 else None
            result = target[start:end]
            if isinstance(result, str):
                return self._bounded_string(result, path)
            materialized = list(result)
            self._limit_collection(materialized, path)
            return materialized

        if operation in {"eq", "ne"}:
            equal = self._equal(values[0], values[1])
            return equal if operation == "eq" else not equal
        if operation in {"lt", "lte", "gt", "gte"}:
            return self._compare(operation, values[0], values[1], path)
        if operation == "not":
            return not self._boolean(values[0], path + ("args", 0))

        raise AssertionError(f"unhandled operation {operation}")

    def _numeric(
        self,
        operation: str,
        values: list[Any],
        path: tuple[PathPart, ...],
    ) -> int | float:
        numbers = [
            self._number(value, path + ("args", index)) for index, value in enumerate(values)
        ]
        try:
            if operation == "add":
                result: int | float = numbers[0]
                for number in numbers[1:]:
                    result = self._number_result(result + number, path)
                return result
            if operation == "sub":
                return self._number_result(numbers[0] - numbers[1], path)
            if operation == "mul":
                result = numbers[0]
                for number in numbers[1:]:
                    result = self._number_result(result * number, path)
                return result
            if operation == "div":
                if numbers[1] == 0:
                    raise EvaluationError(
                        "division by zero",
                        code="DIVISION_BY_ZERO",
                        path=path + ("args", 1),
                    )
                return self._number_result(numbers[0] / numbers[1], path)
            if operation == "mod":
                if numbers[1] == 0:
                    raise EvaluationError(
                        "modulo by zero",
                        code="DIVISION_BY_ZERO",
                        path=path + ("args", 1),
                    )
                return self._number_result(numbers[0] % numbers[1], path)
            if operation == "pow":
                exponent = numbers[1]
                if abs(exponent) > self.limits.max_power_exponent:
                    raise ResourceLimitError(
                        f"power exponent exceeds {self.limits.max_power_exponent}",
                        code="POWER_LIMIT",
                        path=path + ("args", 1),
                    )
                base = numbers[0]
                if (
                    isinstance(base, int)
                    and isinstance(exponent, int)
                    and exponent > 0
                    and base
                    and base.bit_length() * exponent > self.limits.max_integer_bits
                ):
                    raise ResourceLimitError(
                        f"power result may exceed {self.limits.max_integer_bits} bits",
                        code="INTEGER_LIMIT",
                        path=path,
                    )
                return self._number_result(base**exponent, path)
            if operation == "round":
                if len(numbers) == 1:
                    return self._number_result(round(numbers[0]), path)
                digits = self._integer(values[1], path + ("args", 1))
                if abs(digits) > self.limits.max_round_digits:
                    raise ResourceLimitError(
                        f"round digits exceed {self.limits.max_round_digits}",
                        code="ROUND_LIMIT",
                        path=path + ("args", 1),
                    )
                return self._number_result(round(numbers[0], digits), path)
            if operation == "min":
                return self._number_result(min(numbers), path)
            if operation == "max":
                return self._number_result(max(numbers), path)
            if operation == "abs":
                return self._number_result(abs(numbers[0]), path)
        except OverflowError:
            raise EvaluationError(
                "numeric operation overflowed",
                code="NUMBER_OVERFLOW",
                path=path,
            ) from None
        except ZeroDivisionError:
            raise EvaluationError(
                "division by zero",
                code="DIVISION_BY_ZERO",
                path=path,
            ) from None
        except ValueError:
            raise EvaluationError(
                "numeric operation has no real result",
                code="INVALID_NUMBER",
                path=path,
            ) from None
        raise AssertionError(operation)

    def _equal(self, left: Any, right: Any) -> bool:
        if isinstance(left, bool) or isinstance(right, bool):
            return type(left) is type(right) and left == right
        if _is_number(left) and _is_number(right):
            return left == right
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            if len(left) != len(right):
                return False
            for key, value in left.items():
                if key not in right or not self._equal(value, right[key]):
                    return False
            return True
        if _is_sequence(left) and _is_sequence(right):
            return len(left) == len(right) and all(
                self._equal(left_item, right_item)
                for left_item, right_item in zip(left, right, strict=True)
            )
        return type(left) is type(right) and left == right

    def _compare(
        self,
        operation: str,
        left: Any,
        right: Any,
        path: tuple[PathPart, ...],
    ) -> bool:
        comparable = (_is_number(left) and _is_number(right)) or (
            isinstance(left, str) and isinstance(right, str)
        )
        if comparable:
            pass
        else:
            raise EvaluationError(
                "ordered comparison requires two numbers or two strings",
                code="TYPE_MISMATCH",
                path=path,
            )
        if operation == "lt":
            return left < right
        if operation == "lte":
            return left <= right
        if operation == "gt":
            return left > right
        return left >= right

    def _sort_key(
        self,
        value: Any,
        path: tuple[PathPart, ...],
    ) -> tuple[int, Any]:
        if value is None:
            return (0, 0)
        if isinstance(value, bool):
            return (1, int(value))
        if _is_number(value):
            self._number(value, path)
            return (2, value)
        if isinstance(value, str):
            return (3, value)
        raise EvaluationError(
            "sort keys must be null, boolean, number, or string",
            code="INVALID_SORT_KEY",
            path=path,
        )


def evaluate(
    expression: Any,
    inputs: Any = None,
    *,
    variables: Mapping[str, Any] | None = None,
    limits: ResourceLimits | Mapping[str, int] | None = None,
) -> Any:
    """Validate and evaluate a bare expression, returning JSON-native data."""

    resolved_limits = _coerce_limits(limits)
    resolved_inputs = {} if inputs is None else inputs
    resolved_variables = {} if variables is None else variables
    if not isinstance(resolved_variables, Mapping):
        raise TypeError("variables must be a mapping or None")
    validate_expression(
        expression,
        limits=resolved_limits,
        allowed_variables=tuple(resolved_variables),
    )
    return _Runtime(
        resolved_limits,
        resolved_inputs,
        resolved_variables,
    ).evaluate(expression)


def run(
    program: Any,
    inputs: Any = None,
    *,
    limits: ResourceLimits | Mapping[str, int] | None = None,
) -> Any:
    """Validate and execute a complete version-1 program."""

    parsed = _coerce_program(program)
    resolved_limits = _coerce_limits(limits)
    validate_program(parsed, limits=resolved_limits)
    resolved_inputs = {} if inputs is None else inputs
    runtime = _Runtime(resolved_limits, resolved_inputs, {})
    runtime._preflight_inputs()
    if "input_schema" in parsed:
        validate_instance(resolved_inputs, parsed["input_schema"])
    result = runtime.evaluate(parsed["expression"], inputs_preflighted=True)
    if "output_schema" in parsed:
        validate_output(result, parsed["output_schema"])
    return result


execute = run


class CrystalFlowEngine:
    """Reusable facade carrying an immutable set of resource limits."""

    def __init__(
        self,
        limits: ResourceLimits | Mapping[str, int] | None = None,
    ) -> None:
        self.limits = _coerce_limits(limits)

    def validate(self, program: Any) -> None:
        validate_program(program, limits=self.limits)

    def evaluate(
        self,
        expression: Any,
        inputs: Any = None,
        *,
        variables: Mapping[str, Any] | None = None,
    ) -> Any:
        return evaluate(
            expression,
            inputs,
            variables=variables,
            limits=self.limits,
        )

    def run(self, program: Any, inputs: Any = None) -> Any:
        return run(program, inputs, limits=self.limits)

    def execute(self, program: Any, inputs: Any = None) -> Any:
        return self.run(program, inputs)


Engine = CrystalFlowEngine

__all__ = [
    "Budget",
    "CrystalFlowEngine",
    "DEFAULT_LIMITS",
    "ENGINE_VERSION",
    "Engine",
    "Expression",
    "IR_VERSION",
    "Program",
    "ResourceLimits",
    "evaluate",
    "execute",
    "run",
    "validate_expression",
    "validate_program",
]
