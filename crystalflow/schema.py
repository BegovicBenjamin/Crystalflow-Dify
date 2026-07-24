"""A small, dependency-free JSON Schema subset for CrystalFlow boundaries.

Supported keywords are ``type``, ``properties``, ``required``,
``additionalProperties``, ``items``, ``enum``, ``minimum``, ``maximum``,
``minLength``, ``maxLength``, ``minItems``, ``maxItems``, ``minProperties``,
and ``maxProperties``.  Unknown keywords are rejected so a misspelled
constraint can never silently weaken validation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .canonical import canonical_json
from .errors import (
    CanonicalizationError,
    InputValidationError,
    OutputValidationError,
    PathPart,
    SchemaDefinitionError,
)

JSONSchema = Mapping[str, Any]

_TYPES = frozenset({"null", "boolean", "integer", "number", "string", "array", "object"})
_KEYWORDS = frozenset(
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
_LENGTH_KEYWORDS = (
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "minProperties",
    "maxProperties",
)
_MAX_SCHEMA_PATH_PARTS = 128


def _schema_error(
    message: str,
    path: tuple[PathPart, ...],
    code: str = "INVALID_SCHEMA",
) -> None:
    raise SchemaDefinitionError(message, code=code, path=path)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_type_keyword(
    declared: Any,
    path: tuple[PathPart, ...],
) -> frozenset[str]:
    if isinstance(declared, str):
        types = [declared]
    elif isinstance(declared, Sequence) and not isinstance(declared, (str, bytes, bytearray)):
        types = list(declared)
        if not types:
            _schema_error("type list must not be empty", path, "EMPTY_TYPE_LIST")
    else:
        _schema_error("type must be a string or non-empty array", path, "INVALID_TYPE")

    seen: set[str] = set()
    for index, item in enumerate(types):
        if not isinstance(item, str) or item not in _TYPES:
            _schema_error(
                f"unsupported JSON type {item!r}",
                path + (index,),
                "UNKNOWN_TYPE",
            )
        if item in seen:
            _schema_error(
                f"duplicate JSON type {item!r}",
                path + (index,),
                "DUPLICATE_TYPE",
            )
        seen.add(item)
    return frozenset(seen)


def validate_schema(
    schema: JSONSchema,
    *,
    _path: tuple[PathPart, ...] = (),
) -> None:
    """Validate a schema definition, raising :class:`SchemaDefinitionError`."""

    if len(_path) > _MAX_SCHEMA_PATH_PARTS:
        _schema_error(
            "schema nesting is too deep",
            _path,
            "SCHEMA_DEPTH_LIMIT",
        )
    if not isinstance(schema, Mapping):
        _schema_error("schema must be an object", _path, "SCHEMA_NOT_OBJECT")
    for key in schema:
        if not isinstance(key, str):
            _schema_error("schema keys must be strings", _path, "INVALID_SCHEMA_KEY")
        if key not in _KEYWORDS:
            _schema_error(
                f"unknown schema keyword {key!r}",
                _path + (key,),
                "UNKNOWN_SCHEMA_KEY",
            )

    declared_types: frozenset[str] | None = None
    if "type" in schema:
        declared_types = _validate_type_keyword(schema["type"], _path + ("type",))

    def require_compatible(keyword: str, expected: str) -> None:
        if keyword in schema and declared_types is not None and expected not in declared_types:
            _schema_error(
                f"{keyword} requires type {expected!r}",
                _path + (keyword,),
                "INCOMPATIBLE_SCHEMA_KEY",
            )

    for keyword in (
        "properties",
        "required",
        "additionalProperties",
        "minProperties",
        "maxProperties",
    ):
        require_compatible(keyword, "object")
    for keyword in ("items", "minItems", "maxItems"):
        require_compatible(keyword, "array")
    for keyword in ("minLength", "maxLength"):
        require_compatible(keyword, "string")
    for keyword in ("minimum", "maximum"):
        if (
            keyword in schema
            and declared_types is not None
            and not ({"number", "integer"} & declared_types)
        ):
            _schema_error(
                f"{keyword} requires a numeric type",
                _path + (keyword,),
                "INCOMPATIBLE_SCHEMA_KEY",
            )

    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, Mapping):
            _schema_error(
                "properties must be an object",
                _path + ("properties",),
                "INVALID_PROPERTIES",
            )
        for name, child_schema in properties.items():
            if not isinstance(name, str):
                _schema_error(
                    "property names must be strings",
                    _path + ("properties",),
                    "INVALID_PROPERTY_NAME",
                )
            validate_schema(child_schema, _path=_path + ("properties", name))

    required = schema.get("required")
    if required is not None:
        if not isinstance(required, Sequence) or isinstance(required, (str, bytes, bytearray)):
            _schema_error(
                "required must be an array of property names",
                _path + ("required",),
                "INVALID_REQUIRED",
            )
        seen_required: set[str] = set()
        for index, name in enumerate(required):
            if not isinstance(name, str):
                _schema_error(
                    "required entries must be strings",
                    _path + ("required", index),
                    "INVALID_REQUIRED",
                )
            if name in seen_required:
                _schema_error(
                    f"duplicate required property {name!r}",
                    _path + ("required", index),
                    "DUPLICATE_REQUIRED",
                )
            seen_required.add(name)

    additional = schema.get("additionalProperties")
    if additional is not None and not isinstance(additional, bool):
        if not isinstance(additional, Mapping):
            _schema_error(
                "additionalProperties must be a boolean or schema",
                _path + ("additionalProperties",),
                "INVALID_ADDITIONAL_PROPERTIES",
            )
        validate_schema(additional, _path=_path + ("additionalProperties",))

    if "items" in schema:
        validate_schema(schema["items"], _path=_path + ("items",))

    if "enum" in schema:
        enum = schema["enum"]
        if not isinstance(enum, Sequence) or isinstance(enum, (str, bytes, bytearray)):
            _schema_error("enum must be an array", _path + ("enum",), "INVALID_ENUM")
        if not enum:
            _schema_error("enum must not be empty", _path + ("enum",), "EMPTY_ENUM")
        spellings: set[str] = set()
        for index, item in enumerate(enum):
            try:
                spelling = canonical_json(item)
            except CanonicalizationError as exc:
                _schema_error(
                    f"enum entry is not JSON: {exc.message}",
                    _path + ("enum", index),
                    "INVALID_ENUM_VALUE",
                )
            if spelling in spellings:
                _schema_error(
                    "enum entries must be unique",
                    _path + ("enum", index),
                    "DUPLICATE_ENUM_VALUE",
                )
            spellings.add(spelling)

    for keyword in ("minimum", "maximum"):
        if keyword in schema:
            value = schema[keyword]
            if not _is_number(value) or (isinstance(value, float) and not math.isfinite(value)):
                _schema_error(
                    f"{keyword} must be a finite number",
                    _path + (keyword,),
                    "INVALID_BOUND",
                )
    if "minimum" in schema and "maximum" in schema and schema["minimum"] > schema["maximum"]:
        _schema_error(
            "minimum must not exceed maximum",
            _path + ("minimum",),
            "INVERTED_BOUNDS",
        )

    for keyword in _LENGTH_KEYWORDS:
        if keyword in schema:
            value = schema[keyword]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                _schema_error(
                    f"{keyword} must be a non-negative integer",
                    _path + (keyword,),
                    "INVALID_LENGTH_BOUND",
                )
    for lower, upper in (
        ("minLength", "maxLength"),
        ("minItems", "maxItems"),
        ("minProperties", "maxProperties"),
    ):
        if lower in schema and upper in schema and schema[lower] > schema[upper]:
            _schema_error(
                f"{lower} must not exceed {upper}",
                _path + (lower,),
                "INVERTED_BOUNDS",
            )


def _matches_type(instance: Any, declared: str) -> bool:
    if declared == "null":
        return instance is None
    if declared == "boolean":
        return isinstance(instance, bool)
    if declared == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if declared == "number":
        return _is_number(instance) and not (
            isinstance(instance, float) and not math.isfinite(instance)
        )
    if declared == "string":
        return isinstance(instance, str)
    if declared == "array":
        return isinstance(instance, Sequence) and not isinstance(
            instance, (str, bytes, bytearray, memoryview)
        )
    if declared == "object":
        return isinstance(instance, Mapping)
    raise AssertionError(f"unvalidated type {declared}")


def _json_equal(left: Any, right: Any) -> bool:
    # Canonical equality gives JSON's intended numeric equality while keeping
    # booleans distinct from the integers 0 and 1.
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if _is_number(left) and _is_number(right):
        return left == right
    return canonical_json(left) == canonical_json(right)


def _validate_value(
    instance: Any,
    schema: JSONSchema,
    path: tuple[PathPart, ...],
    error_type: type[InputValidationError] | type[OutputValidationError],
) -> None:
    def fail(message: str, code: str) -> None:
        raise error_type(message, code=code, path=path)

    declared = schema.get("type")
    if declared is not None:
        types = [declared] if isinstance(declared, str) else list(declared)
        if not any(_matches_type(instance, item) for item in types):
            fail(
                f"expected type {' or '.join(types)}, got {type(instance).__name__}",
                "TYPE_MISMATCH",
            )

    if "enum" in schema and not any(
        _json_equal(instance, candidate) for candidate in schema["enum"]
    ):
        fail("value is not one of the allowed enum entries", "ENUM_MISMATCH")

    if _is_number(instance):
        if isinstance(instance, float) and not math.isfinite(instance):
            fail("numbers must be finite", "NON_FINITE_NUMBER")
        if "minimum" in schema and instance < schema["minimum"]:
            fail(f"value is less than minimum {schema['minimum']}", "BELOW_MINIMUM")
        if "maximum" in schema and instance > schema["maximum"]:
            fail(f"value is greater than maximum {schema['maximum']}", "ABOVE_MAXIMUM")

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            fail(
                f"string is shorter than minLength {schema['minLength']}",
                "TOO_SHORT",
            )
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            fail(
                f"string is longer than maxLength {schema['maxLength']}",
                "TOO_LONG",
            )

    if isinstance(instance, Sequence) and not isinstance(
        instance, (str, bytes, bytearray, memoryview)
    ):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            fail(f"array has fewer than {schema['minItems']} items", "TOO_FEW_ITEMS")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            fail(f"array has more than {schema['maxItems']} items", "TOO_MANY_ITEMS")
        if "items" in schema:
            for index, child in enumerate(instance):
                _validate_value(child, schema["items"], path + (index,), error_type)

    if isinstance(instance, Mapping):
        for key in instance:
            if not isinstance(key, str):
                fail("object keys must be strings", "INVALID_OBJECT_KEY")
        if "minProperties" in schema and len(instance) < schema["minProperties"]:
            fail(
                f"object has fewer than {schema['minProperties']} properties",
                "TOO_FEW_PROPERTIES",
            )
        if "maxProperties" in schema and len(instance) > schema["maxProperties"]:
            fail(
                f"object has more than {schema['maxProperties']} properties",
                "TOO_MANY_PROPERTIES",
            )
        properties = schema.get("properties", {})
        for name in schema.get("required", ()):
            if name not in instance:
                raise error_type(
                    f"required property {name!r} is missing",
                    code="MISSING_REQUIRED",
                    path=path + (name,),
                )
        additional = schema.get("additionalProperties", True)
        for name, child in instance.items():
            if name in properties:
                _validate_value(child, properties[name], path + (name,), error_type)
            elif additional is False:
                raise error_type(
                    f"additional property {name!r} is not allowed",
                    code="ADDITIONAL_PROPERTY",
                    path=path + (name,),
                )
            elif isinstance(additional, Mapping):
                _validate_value(child, additional, path + (name,), error_type)


def _validate_instance(
    instance: Any,
    schema: JSONSchema,
    error_type: type[InputValidationError] | type[OutputValidationError],
) -> None:
    validate_schema(schema)
    try:
        canonical_json(instance)
    except CanonicalizationError as exc:
        raise error_type(
            exc.message,
            code=exc.code,
            path=exc.path,
            details=exc.details,
        ) from None
    _validate_value(instance, schema, (), error_type)


def validate_instance(instance: Any, schema: JSONSchema) -> None:
    """Validate input data against *schema*."""

    _validate_instance(instance, schema, InputValidationError)


def validate_output(instance: Any, schema: JSONSchema) -> None:
    """Validate a result against an output schema."""

    _validate_instance(instance, schema, OutputValidationError)


def is_valid(instance: Any, schema: JSONSchema) -> bool:
    """Return ``True`` exactly when :func:`validate_instance` succeeds."""

    try:
        validate_instance(instance, schema)
    except (InputValidationError, SchemaDefinitionError):
        return False
    return True


validate = validate_instance

__all__ = [
    "JSONSchema",
    "is_valid",
    "validate",
    "validate_instance",
    "validate_output",
    "validate_schema",
]
