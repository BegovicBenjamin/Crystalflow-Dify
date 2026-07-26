"""Persistent exact-match learning for deterministic, read-only tool routes.

This module deliberately does not invoke tools or extend Crystal IR.  It only
returns a validated :class:`ToolPlan`; the host remains responsible for tool
execution, authorization, runtime-secret injection, and result handling.
"""

from __future__ import annotations

import threading
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_json_bytes, canonical_loads, content_hash
from .registry import (
    KVStore,
    Registry,
    RegistryError,
    json_from_bytes,
    stable_json_bytes,
)
from .schema import validate_instance, validate_schema

STATE_SCHEMA = "crystalflow.adaptive-route.v1"
DEFAULT_THRESHOLD = 5
MAX_THRESHOLD = 100
MAX_MATCH_VALUE_BYTES = 64 * 1024
MAX_ARGUMENT_BYTES = 64 * 1024
MAX_SCHEMA_BYTES = 64 * 1024
MAX_STATE_BYTES = 128 * 1024
MAX_TOKEN_ESTIMATE = 1_000_000
MAX_COUNTER = (1 << 63) - 1

_HASH_LENGTH = 64
_PROCESS_LOCK = threading.RLock()
_STATE_FIELDS = {
    "schema",
    "route_key",
    "revision",
    "status",
    "reason_code",
    "observations",
    "conflict_count",
    "invalidation_count",
    "hit_count",
    "estimated_tokens_avoided",
    "action_hash",
    "contract_hash",
    "action",
}
_ACTION_FIELDS = {
    "provider_type",
    "provider",
    "tool_name",
    "arguments",
    "contract_hash",
}
_STATUSES = {"learning", "active", "quarantined", "invalidated"}
_INVALIDATION_REASONS = {
    "TOOL_NOT_ALLOWLISTED",
    "TOOL_NOT_READ_ONLY",
    "TOOL_CONTRACT_CHANGED",
    "ARGUMENTS_NO_LONGER_VALID",
    "MANUAL_INVALIDATION",
}
_SCHEMA_ANNOTATIONS = {"default", "description", "examples", "title"}

JSONValue = Any
ToolIdentity = tuple[str, str, str]


class AdaptiveRouteError(Exception):
    """Base class for safe adaptive-route failures."""

    default_code = "ADAPTIVE_ROUTE_ERROR"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        self.code = code or self.default_code
        self.message = message
        super().__init__(message)


class RouteValidationError(AdaptiveRouteError, ValueError):
    """A route request, tool contract, or observation is invalid."""

    default_code = "INVALID_ADAPTIVE_ROUTE"


class RouteStorageError(AdaptiveRouteError):
    """Persistent route evidence is unavailable or corrupt."""

    default_code = "ADAPTIVE_ROUTE_STORAGE_ERROR"


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _HASH_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _identity_part(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise RouteValidationError(f"{label} must be a string")
    normalized = unicodedata.normalize("NFC", value)
    if (
        not normalized
        or normalized != normalized.strip()
        or len(normalized.encode("utf-8")) > 256
        or any(ord(character) < 0x20 for character in normalized)
    ):
        raise RouteValidationError(
            f"{label} must be a non-empty identifier of at most 256 UTF-8 bytes"
        )
    return normalized


def _snapshot(value: JSONValue, *, label: str, max_bytes: int) -> JSONValue:
    try:
        raw = canonical_json_bytes(value)
    except Exception as exc:
        raise RouteValidationError(
            f"{label} must be finite canonical JSON",
            code="INVALID_JSON_VALUE",
        ) from exc
    if len(raw) > max_bytes:
        raise RouteValidationError(
            f"{label} exceeds the {max_bytes}-byte limit",
            code="JSON_SIZE_LIMIT",
        )
    try:
        return canonical_loads(raw)
    except Exception as exc:  # pragma: no cover - canonical output must parse
        raise RouteValidationError(f"{label} could not be normalized") from exc


def _object_snapshot(value: object, *, label: str, max_bytes: int) -> dict[str, JSONValue]:
    if not isinstance(value, Mapping):
        raise RouteValidationError(f"{label} must be a JSON object")
    snapshot = _snapshot(value, label=label, max_bytes=max_bytes)
    assert isinstance(snapshot, dict)
    return snapshot


def _strip_schema_annotations(schema: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    """Remove standard documentation annotations unsupported by the IR validator."""

    result: dict[str, JSONValue] = {}
    for key, value in schema.items():
        if key in _SCHEMA_ANNOTATIONS:
            continue
        if key == "properties" and isinstance(value, Mapping):
            result[key] = {
                name: (_strip_schema_annotations(child) if isinstance(child, Mapping) else child)
                for name, child in value.items()
            }
        elif key in {"items", "additionalProperties"} and isinstance(value, Mapping):
            result[key] = _strip_schema_annotations(value)
        else:
            result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class ToolContract:
    """One explicitly allowlisted tool contract.

    ``contract_version`` must change when behavior changes without a schema
    change.  Schema changes are detected automatically by ``contract_hash``.
    """

    provider_type: str
    provider: str
    tool_name: str
    arguments_schema: Mapping[str, JSONValue]
    contract_version: str = "1"
    read_only: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "provider_type", _identity_part(self.provider_type, "provider_type")
        )
        object.__setattr__(self, "provider", _identity_part(self.provider, "provider"))
        object.__setattr__(self, "tool_name", _identity_part(self.tool_name, "tool_name"))
        object.__setattr__(
            self,
            "contract_version",
            _identity_part(self.contract_version, "contract_version"),
        )
        if not isinstance(self.read_only, bool):
            raise RouteValidationError("read_only must be a boolean")
        schema = _object_snapshot(
            self.arguments_schema,
            label="arguments_schema",
            max_bytes=MAX_SCHEMA_BYTES,
        )
        schema = _strip_schema_annotations(schema)
        try:
            validate_schema(schema)
        except Exception as exc:
            raise RouteValidationError(
                "arguments_schema is not a supported JSON Schema",
                code="INVALID_ARGUMENT_SCHEMA",
            ) from exc
        declared_type = schema.get("type")
        object_declared = declared_type == "object" or (
            isinstance(declared_type, list) and "object" in declared_type
        )
        if not object_declared:
            raise RouteValidationError("arguments_schema must declare type 'object'")
        object.__setattr__(self, "arguments_schema", schema)

    @property
    def identity(self) -> ToolIdentity:
        return (self.provider_type, self.provider, self.tool_name)

    @property
    def contract_hash(self) -> str:
        return content_hash(
            {
                "schema": "crystalflow.tool-contract.v1",
                "provider_type": self.provider_type,
                "provider": self.provider,
                "tool_name": self.tool_name,
                "arguments_schema": self.arguments_schema,
                "contract_version": self.contract_version,
                "read_only": self.read_only,
            }
        )


@dataclass(frozen=True, slots=True)
class ToolAction:
    """A successful cold-path tool selection offered as route evidence."""

    provider_type: str
    provider: str
    tool_name: str
    arguments: Mapping[str, JSONValue]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "provider_type", _identity_part(self.provider_type, "provider_type")
        )
        object.__setattr__(self, "provider", _identity_part(self.provider, "provider"))
        object.__setattr__(self, "tool_name", _identity_part(self.tool_name, "tool_name"))
        object.__setattr__(
            self,
            "arguments",
            _object_snapshot(
                self.arguments,
                label="tool arguments",
                max_bytes=MAX_ARGUMENT_BYTES,
            ),
        )

    @property
    def identity(self) -> ToolIdentity:
        return (self.provider_type, self.provider, self.tool_name)


@dataclass(frozen=True, slots=True)
class ToolPlan:
    """A validated deterministic plan for the host to execute."""

    provider_type: str
    provider: str
    tool_name: str
    arguments: Mapping[str, JSONValue]
    contract_hash: str

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "provider_type": self.provider_type,
            "provider": self.provider,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "contract_hash": self.contract_hash,
        }


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Result of a lookup or successful-route observation."""

    status: str
    reason_code: str
    route_key: str
    observations: int
    threshold: int
    plan: ToolPlan | None
    revision: int
    conflict_count: int
    invalidation_count: int
    hit_count: int
    estimated_tokens_avoided: int

    @property
    def hit(self) -> bool:
        return self.status == "hit" and self.plan is not None

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "status": self.status,
            "reason_code": self.reason_code,
            "route_key": self.route_key,
            "observations": self.observations,
            "threshold": self.threshold,
            "plan": None if self.plan is None else self.plan.to_dict(),
            "revision": self.revision,
            "conflict_count": self.conflict_count,
            "invalidation_count": self.invalidation_count,
            "hit_count": self.hit_count,
            "estimated_tokens_avoided": self.estimated_tokens_avoided,
        }


@dataclass(frozen=True, slots=True)
class RouteTelemetry:
    """Cumulative confirmed warm-path savings for one exact route."""

    route_key: str
    hit_count: int
    estimated_tokens_avoided: int
    revision: int


class AdaptiveRouteStore:
    """Learn exact request/context-to-tool mappings in bounded persistent KV."""

    def __init__(
        self,
        storage: KVStore,
        namespace: str,
        *,
        scope: JSONValue = None,
        threshold: int = DEFAULT_THRESHOLD,
        max_state_bytes: int = MAX_STATE_BYTES,
    ) -> None:
        # Reuse Registry's namespace and KV structural validation.
        registry = Registry(storage, namespace)
        if not _is_int(threshold) or not 1 <= threshold <= MAX_THRESHOLD:
            raise RouteValidationError(f"threshold must be an integer from 1 to {MAX_THRESHOLD}")
        if not _is_int(max_state_bytes) or not 1 <= max_state_bytes <= MAX_STATE_BYTES:
            raise RouteValidationError(
                f"max_state_bytes must be an integer from 1 to {MAX_STATE_BYTES}"
            )
        scope_snapshot = _snapshot(
            scope,
            label="route scope",
            max_bytes=MAX_MATCH_VALUE_BYTES,
        )
        self._storage = storage
        self.namespace = registry.namespace
        self.threshold = threshold
        self.max_state_bytes = max_state_bytes
        self._scope_hash = content_hash(scope_snapshot)

    def lookup(
        self,
        query: JSONValue,
        *,
        context: JSONValue = None,
        tools: Iterable[ToolContract],
    ) -> RouteDecision:
        """Return a hit only when an active plan matches the current contract."""

        route_key = self._route_key(query, context)
        contracts = self._contracts(tools)
        with _PROCESS_LOCK:
            state = self._load_state(route_key)
            if state is None:
                return self._empty_decision(route_key, "UNSEEN_ROUTE")
            if state["status"] == "quarantined":
                return self._decision(state, "quarantined", state["reason_code"])
            if state["status"] == "invalidated":
                return self._decision(state, "invalidated", state["reason_code"])

            action = state["action"]
            assert isinstance(action, dict)
            identity = self._action_identity(action)
            contract = contracts.get(identity)
            if contract is None:
                return self._invalidate_state(state, "TOOL_NOT_ALLOWLISTED")
            if not contract.read_only:
                return self._invalidate_state(state, "TOOL_NOT_READ_ONLY")
            if contract.contract_hash != state["contract_hash"]:
                return self._invalidate_state(state, "TOOL_CONTRACT_CHANGED")
            try:
                validate_instance(action["arguments"], contract.arguments_schema)
            except Exception:
                return self._invalidate_state(state, "ARGUMENTS_NO_LONGER_VALID")

            if state["observations"] < self.threshold:
                if state["status"] == "active":
                    state["status"] = "learning"
                    state["revision"] += 1
                    self._write_state(state)
                return self._decision(state, "learning", "THRESHOLD_NOT_MET")
            if state["status"] == "learning":
                state["status"] = "active"
                state["revision"] += 1
                self._write_state(state)
            return self._decision(state, "hit", "ROUTE_HIT", include_plan=True)

    def observe_success(
        self,
        query: JSONValue,
        action: ToolAction,
        *,
        context: JSONValue = None,
        tools: Iterable[ToolContract],
    ) -> RouteDecision:
        """Record one successful cold-path action and activate at the threshold."""

        if not isinstance(action, ToolAction):
            raise RouteValidationError("action must be a ToolAction")
        route_key = self._route_key(query, context)
        contracts = self._contracts(tools)
        contract = contracts.get(action.identity)
        if contract is None:
            return self._empty_decision(route_key, "TOOL_NOT_ALLOWLISTED")
        if not contract.read_only:
            return self._empty_decision(route_key, "TOOL_NOT_READ_ONLY")
        try:
            validate_instance(action.arguments, contract.arguments_schema)
        except Exception:
            return self._empty_decision(route_key, "INVALID_TOOL_ARGUMENTS")

        action_record = self._action_record(action, contract)
        action_hash = content_hash(action_record)
        with _PROCESS_LOCK:
            state = self._load_state(route_key)
            if state is None:
                state = self._new_state(route_key, action_record, action_hash)
            elif state["status"] == "quarantined":
                return self._decision(state, "quarantined", state["reason_code"])
            elif state["status"] == "invalidated":
                self._restart_state(state, action_record, action_hash)
            else:
                current_action = state["action"]
                assert isinstance(current_action, dict)
                if self._action_identity(current_action) != action.identity:
                    return self._quarantine_state(state)
                if state["contract_hash"] != contract.contract_hash:
                    state["invalidation_count"] += 1
                    self._restart_state(state, action_record, action_hash)
                elif state["action_hash"] != action_hash:
                    return self._quarantine_state(state)
                else:
                    if state["observations"] == MAX_COUNTER:
                        raise RouteValidationError("observation counter would overflow")
                    state["observations"] += 1
                    state["revision"] += 1

            state["status"] = "active" if state["observations"] >= self.threshold else "learning"
            state["reason_code"] = ""
            self._write_state(state)
            if state["status"] == "active":
                return self._decision(
                    state,
                    "hit",
                    "ROUTE_ACTIVATED",
                    include_plan=True,
                )
            return self._decision(state, "learning", "THRESHOLD_NOT_MET")

    def record_hit(
        self,
        route_key: str,
        *,
        estimated_tokens_avoided: int = 0,
    ) -> RouteTelemetry:
        """Record a warm hit after the host successfully executes its tool plan."""

        self._validate_route_key(route_key)
        if (
            not _is_int(estimated_tokens_avoided)
            or not 0 <= estimated_tokens_avoided <= MAX_TOKEN_ESTIMATE
        ):
            raise RouteValidationError(
                f"estimated_tokens_avoided must be an integer from 0 to {MAX_TOKEN_ESTIMATE}"
            )
        with _PROCESS_LOCK:
            state = self._load_state(route_key)
            if state is None or state["status"] != "active":
                raise RouteValidationError(
                    "only an active route can record a hit",
                    code="ROUTE_NOT_ACTIVE",
                )
            if state["hit_count"] == MAX_COUNTER:
                raise RouteValidationError("hit counter would overflow")
            if state["estimated_tokens_avoided"] > MAX_COUNTER - estimated_tokens_avoided:
                raise RouteValidationError("token counter would overflow")
            state["hit_count"] += 1
            state["estimated_tokens_avoided"] += estimated_tokens_avoided
            state["revision"] += 1
            self._write_state(state)
            return RouteTelemetry(
                route_key=route_key,
                hit_count=state["hit_count"],
                estimated_tokens_avoided=state["estimated_tokens_avoided"],
                revision=state["revision"],
            )

    def invalidate(
        self,
        query: JSONValue,
        *,
        context: JSONValue = None,
    ) -> RouteDecision:
        """Explicitly invalidate a known exact route."""

        route_key = self._route_key(query, context)
        with _PROCESS_LOCK:
            state = self._load_state(route_key)
            if state is None:
                return self._empty_decision(route_key, "UNSEEN_ROUTE")
            return self._invalidate_state(state, "MANUAL_INVALIDATION")

    def _route_key(self, query: JSONValue, context: JSONValue) -> str:
        if isinstance(query, str):
            # This remains exact matching after a deterministic text normalizer:
            # no semantic similarity, embeddings, or model call is involved.
            query = " ".join(unicodedata.normalize("NFKC", query).casefold().split())
        query_snapshot = _snapshot(
            query,
            label="query",
            max_bytes=MAX_MATCH_VALUE_BYTES,
        )
        context_snapshot = _snapshot(
            context,
            label="routing context",
            max_bytes=MAX_MATCH_VALUE_BYTES,
        )
        return content_hash(
            {
                "schema": "crystalflow.adaptive-route-key.v1",
                "scope_hash": self._scope_hash,
                "query_hash": content_hash(query_snapshot),
                "context_hash": content_hash(context_snapshot),
            }
        )

    @staticmethod
    def _contracts(tools: Iterable[ToolContract]) -> dict[ToolIdentity, ToolContract]:
        try:
            candidates = list(tools)
        except TypeError as exc:
            raise RouteValidationError("tools must be an iterable of ToolContract") from exc
        result: dict[ToolIdentity, ToolContract] = {}
        for contract in candidates:
            if not isinstance(contract, ToolContract):
                raise RouteValidationError("tools must contain only ToolContract values")
            if contract.identity in result:
                raise RouteValidationError("tool contracts contain a duplicate identity")
            result[contract.identity] = contract
        return result

    @staticmethod
    def _action_record(
        action: ToolAction,
        contract: ToolContract,
    ) -> dict[str, JSONValue]:
        return {
            "provider_type": action.provider_type,
            "provider": action.provider,
            "tool_name": action.tool_name,
            "arguments": dict(action.arguments),
            "contract_hash": contract.contract_hash,
        }

    @staticmethod
    def _action_identity(action: Mapping[str, JSONValue]) -> ToolIdentity:
        return (
            action["provider_type"],
            action["provider"],
            action["tool_name"],
        )

    @staticmethod
    def _new_state(
        route_key: str,
        action: dict[str, JSONValue],
        action_hash: str,
    ) -> dict[str, JSONValue]:
        return {
            "schema": STATE_SCHEMA,
            "route_key": route_key,
            "revision": 1,
            "status": "learning",
            "reason_code": "",
            "observations": 1,
            "conflict_count": 0,
            "invalidation_count": 0,
            "hit_count": 0,
            "estimated_tokens_avoided": 0,
            "action_hash": action_hash,
            "contract_hash": action["contract_hash"],
            "action": action,
        }

    @staticmethod
    def _restart_state(
        state: dict[str, JSONValue],
        action: dict[str, JSONValue],
        action_hash: str,
    ) -> None:
        state["revision"] += 1
        state["status"] = "learning"
        state["reason_code"] = ""
        state["observations"] = 1
        state["action_hash"] = action_hash
        state["contract_hash"] = action["contract_hash"]
        state["action"] = action

    def _quarantine_state(self, state: dict[str, JSONValue]) -> RouteDecision:
        if state["conflict_count"] == MAX_COUNTER:
            raise RouteValidationError("conflict counter would overflow")
        state["status"] = "quarantined"
        state["reason_code"] = "ACTION_CONFLICT"
        state["conflict_count"] += 1
        state["revision"] += 1
        self._write_state(state)
        return self._decision(state, "quarantined", "ACTION_CONFLICT")

    def _invalidate_state(
        self,
        state: dict[str, JSONValue],
        reason_code: str,
    ) -> RouteDecision:
        if state["status"] == "invalidated" and state["reason_code"] == reason_code:
            return self._decision(state, "invalidated", reason_code)
        if state["invalidation_count"] == MAX_COUNTER:
            raise RouteValidationError("invalidation counter would overflow")
        state["status"] = "invalidated"
        state["reason_code"] = reason_code
        state["observations"] = 0
        state["action_hash"] = ""
        state["contract_hash"] = ""
        state["action"] = None
        state["invalidation_count"] += 1
        state["revision"] += 1
        self._write_state(state)
        return self._decision(state, "invalidated", reason_code)

    def _state_key(self, route_key: str) -> str:
        return f"{self.namespace}:adaptive-routes:v1:state:{route_key}"

    def _read_raw(self, key: str) -> bytes | None:
        exists = getattr(self._storage, "exist", None)
        if callable(exists):
            try:
                if not exists(key):
                    return None
            except Exception as exc:
                raise RouteStorageError("route storage existence check failed") from exc
        try:
            value = self._storage.get(key)
        except Exception as exc:
            raise RouteStorageError("route storage read failed") from exc
        if value is None:
            return None
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise RouteStorageError("route storage returned non-bytes")
        raw = bytes(value)
        if len(raw) > self.max_state_bytes:
            raise RouteStorageError("route record exceeds its size limit")
        return raw

    def _load_state(self, route_key: str) -> dict[str, JSONValue] | None:
        raw = self._read_raw(self._state_key(route_key))
        if raw is None:
            return None
        try:
            value = json_from_bytes(
                raw,
                source="adaptive route record",
                require_canonical=True,
                max_bytes=self.max_state_bytes,
            )
        except RegistryError as exc:
            raise RouteStorageError("adaptive route record is corrupt") from exc
        self._validate_state(value, route_key)
        assert isinstance(value, dict)
        return value

    def _write_state(self, state: dict[str, JSONValue]) -> None:
        route_key = state.get("route_key")
        self._validate_state(state, route_key)
        try:
            raw = stable_json_bytes(state, max_bytes=self.max_state_bytes)
            self._storage.set(self._state_key(route_key), raw)
        except RouteStorageError:
            raise
        except Exception as exc:
            raise RouteStorageError("adaptive route record write failed") from exc

    @classmethod
    def _validate_state(cls, value: JSONValue, route_key: object) -> None:
        if not isinstance(value, dict) or set(value) != _STATE_FIELDS:
            raise RouteStorageError("adaptive route record has an invalid shape")
        cls._validate_route_key(route_key)
        if value["schema"] != STATE_SCHEMA or value["route_key"] != route_key:
            raise RouteStorageError("adaptive route record has an invalid identity")
        if value["status"] not in _STATUSES:
            raise RouteStorageError("adaptive route record has an invalid status")
        for field in (
            "revision",
            "observations",
            "conflict_count",
            "invalidation_count",
            "hit_count",
            "estimated_tokens_avoided",
        ):
            if not _is_int(value[field]) or not 0 <= value[field] <= MAX_COUNTER:
                raise RouteStorageError("adaptive route record has an invalid counter")
        status = value["status"]
        if status == "invalidated":
            if (
                value["reason_code"] not in _INVALIDATION_REASONS
                or value["observations"] != 0
                or value["action"] is not None
                or value["action_hash"] != ""
                or value["contract_hash"] != ""
            ):
                raise RouteStorageError("adaptive route record has invalid invalidation state")
            return
        action = value["action"]
        if not isinstance(action, dict) or set(action) != _ACTION_FIELDS:
            raise RouteStorageError("adaptive route record has an invalid action")
        if value["observations"] < 1:
            raise RouteStorageError("adaptive route record has no observations")
        try:
            for field in ("provider_type", "provider", "tool_name"):
                _identity_part(action[field], field)
            _object_snapshot(
                action["arguments"],
                label="stored tool arguments",
                max_bytes=MAX_ARGUMENT_BYTES,
            )
        except RouteValidationError as exc:
            raise RouteStorageError("adaptive route record has invalid action data") from exc
        if (
            not _is_hash(value["action_hash"])
            or not _is_hash(value["contract_hash"])
            or action["contract_hash"] != value["contract_hash"]
            or content_hash(action) != value["action_hash"]
        ):
            raise RouteStorageError("adaptive route record failed its content hash")
        expected_reason = "ACTION_CONFLICT" if status == "quarantined" else ""
        if value["reason_code"] != expected_reason:
            raise RouteStorageError("adaptive route record has an invalid reason")

    @staticmethod
    def _validate_route_key(route_key: object) -> None:
        if not _is_hash(route_key):
            raise RouteValidationError("route_key must be a lowercase SHA-256 digest")

    def _empty_decision(self, route_key: str, reason_code: str) -> RouteDecision:
        return RouteDecision(
            status="miss",
            reason_code=reason_code,
            route_key=route_key,
            observations=0,
            threshold=self.threshold,
            plan=None,
            revision=0,
            conflict_count=0,
            invalidation_count=0,
            hit_count=0,
            estimated_tokens_avoided=0,
        )

    def _decision(
        self,
        state: Mapping[str, JSONValue],
        status: str,
        reason_code: str,
        *,
        include_plan: bool = False,
    ) -> RouteDecision:
        plan = None
        if include_plan:
            action = state["action"]
            assert isinstance(action, Mapping)
            plan = ToolPlan(
                provider_type=action["provider_type"],
                provider=action["provider"],
                tool_name=action["tool_name"],
                arguments=dict(action["arguments"]),
                contract_hash=action["contract_hash"],
            )
        return RouteDecision(
            status=status,
            reason_code=reason_code,
            route_key=state["route_key"],
            observations=state["observations"],
            threshold=self.threshold,
            plan=plan,
            revision=state["revision"],
            conflict_count=state["conflict_count"],
            invalidation_count=state["invalidation_count"],
            hit_count=state["hit_count"],
            estimated_tokens_avoided=state["estimated_tokens_avoided"],
        )


__all__ = [
    "AdaptiveRouteError",
    "AdaptiveRouteStore",
    "DEFAULT_THRESHOLD",
    "RouteDecision",
    "RouteStorageError",
    "RouteTelemetry",
    "RouteValidationError",
    "STATE_SCHEMA",
    "ToolAction",
    "ToolContract",
    "ToolPlan",
]
