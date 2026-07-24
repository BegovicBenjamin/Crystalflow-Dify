"""Framework-neutral progressive crystallization.

The progressive layer deliberately owns no model integration.  Callers inject a
``fallback`` callable and, optionally, a ``builder`` callable.  This keeps the
warm path deterministic: an active crystal is tried first and a successful hit
returns before either callable is invoked.

Learning is opt-in on every :meth:`ProgressiveService.run` call.  When enabled,
canonical input/output examples are stored in a bounded per-task record:

.. code-block:: json

    {
      "schema": "crystalflow.progressive.examples.v1",
      "task_key": "invoice_total",
      "revision": 3,
      "quarantined": false,
      "quarantine_reason": "",
      "conflict_count": 0,
      "build_attempt_count": 1,
      "last_build_example_count": 3,
      "examples": [
        {
          "input_hash": "<sha256>",
          "output_hash": "<sha256>",
          "input": {"subtotal": 10},
          "output": {"total": 12}
        }
      ]
    }

The record contains no clock, invocation ID, conversation text, or implicit
context.  Inputs and outputs are retained only when the caller explicitly sets
``learn=True``.  Identical inputs are deduplicated.  A different output for an
already observed input permanently quarantines that task record, because
examples no longer describe a deterministic function.

Once enough distinct examples exist, ``builder(task_key, tests)`` may return a
bare Crystal IR program or an envelope containing ``{"program": ...}``.
Observed examples, rather than model-authored tests, are the mechanical
consistency suite passed to :class:`~crystalflow.service.CrystalService`.
Passing them proves only that the candidate reproduces retained fallback
outputs; it does not independently verify that those outputs are correct.
Candidates are drafts unless ``auto_activate=True`` was explicitly configured.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .canonical import canonical_json, canonical_json_bytes, canonical_loads, content_hash
from .models import Lifecycle
from .registry import (
    MAX_PROGRAM_BYTES,
    MAX_STORED_BYTES,
    MAX_TESTS_BYTES,
    KVStore,
    Registry,
    RegistryError,
    RegistryNotFoundError,
    json_from_bytes,
    stable_json_bytes,
)
from .service import CrystalService

STATE_SCHEMA = "crystalflow.progressive.examples.v1"

DEFAULT_MIN_EXAMPLES = 5
DEFAULT_MAX_EXAMPLES = 50
DEFAULT_MAX_EXAMPLE_BYTES = 64 * 1024
DEFAULT_MAX_STATE_BYTES = 256 * 1024
DEFAULT_MAX_INPUT_BYTES = 64 * 1024
DEFAULT_MAX_OUTPUT_BYTES = 256 * 1024

MAX_EXAMPLES = 100
MAX_EXAMPLE_BYTES = 256 * 1024
MAX_STATE_BYTES = 1024 * 1024
MAX_BUILDER_RESPONSE_BYTES = 512 * 1024
MAX_TOKEN_ESTIMATE = 1_000_000
MAX_COUNTER = (1 << 63) - 1

_STATE_FIELDS = {
    "schema",
    "task_key",
    "revision",
    "quarantined",
    "quarantine_reason",
    "conflict_count",
    "build_attempt_count",
    "last_build_example_count",
    "examples",
}
_EXAMPLE_FIELDS = {"input_hash", "output_hash", "input", "output"}
_HASH_LENGTH = 64
_PROCESS_LOCK = threading.RLock()
_BUILDING: set[tuple[str, str]] = set()

JSONValue = Any


class FallbackCallable(Protocol):
    """Cold-path callable supplied by a Dify adapter or another host."""

    def __call__(self, task_key: str, inputs: JSONValue) -> JSONValue: ...


class BuilderCallable(Protocol):
    """Candidate generator supplied by a Dify adapter or another host."""

    def __call__(
        self,
        task_key: str,
        examples: list[dict[str, JSONValue]],
    ) -> JSONValue: ...


class ProgressiveError(Exception):
    """Base class for safe progressive-layer failures."""

    default_code = "PROGRESSIVE_ERROR"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        self.code = code or self.default_code
        self.message = message
        super().__init__(message)


class ProgressiveValidationError(ProgressiveError, ValueError):
    """The progressive configuration or request is invalid."""

    default_code = "INVALID_PROGRESSIVE_REQUEST"


class ProgressiveStorageError(ProgressiveError):
    """The learning record could not be safely read or written."""

    default_code = "PROGRESSIVE_STORAGE_ERROR"


@dataclass(frozen=True, slots=True)
class _LearnOutcome:
    status: str
    example_count: int
    reason_code: str = ""
    version: int = 0
    program_hash: str = ""
    message: str = ""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _bounded_int(
    value: Any,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if not _is_int(value) or not minimum <= value <= maximum:
        raise ProgressiveValidationError(
            f"{label} must be an integer from {minimum} to {maximum}",
            code="INVALID_CONFIGURATION",
        )
    return value


def _snapshot(
    value: JSONValue,
    *,
    label: str,
    max_bytes: int,
) -> tuple[JSONValue, bytes]:
    """Return a detached canonical JSON snapshot and its bytes."""

    try:
        raw = canonical_json_bytes(value)
    except Exception as exc:
        raise ProgressiveValidationError(
            f"{label} must be finite canonical JSON",
            code="INVALID_JSON_VALUE",
        ) from exc
    if len(raw) > max_bytes:
        raise ProgressiveValidationError(
            f"{label} exceeds the {max_bytes}-byte limit",
            code="JSON_SIZE_LIMIT",
        )
    try:
        return canonical_loads(raw), raw
    except Exception as exc:  # pragma: no cover - canonical output should always parse
        raise ProgressiveValidationError(
            f"{label} could not be normalized",
            code="INVALID_JSON_VALUE",
        ) from exc


class ProgressiveService:
    """Try a crystal first, then safely learn from an injected cold fallback.

    ``learn`` defaults to ``False``.  Consequently constructing this service
    never authorizes retention of runtime input or output values.
    """

    def __init__(
        self,
        storage: KVStore,
        namespace: str,
        *,
        fallback: FallbackCallable | Callable[[str, JSONValue], JSONValue],
        builder: BuilderCallable
        | Callable[[str, list[dict[str, JSONValue]]], JSONValue]
        | None = None,
        min_examples: int = DEFAULT_MIN_EXAMPLES,
        max_examples: int = DEFAULT_MAX_EXAMPLES,
        max_example_bytes: int = DEFAULT_MAX_EXAMPLE_BYTES,
        max_state_bytes: int = DEFAULT_MAX_STATE_BYTES,
        max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        auto_activate: bool = False,
        estimated_tokens_per_run: int = 0,
    ) -> None:
        if not callable(fallback):
            raise ProgressiveValidationError(
                "fallback must be callable",
                code="INVALID_CONFIGURATION",
            )
        if builder is not None and not callable(builder):
            raise ProgressiveValidationError(
                "builder must be callable or None",
                code="INVALID_CONFIGURATION",
            )
        if not isinstance(auto_activate, bool):
            raise ProgressiveValidationError(
                "auto_activate must be a boolean",
                code="INVALID_CONFIGURATION",
            )

        self.min_examples = _bounded_int(
            min_examples,
            "min_examples",
            minimum=1,
            maximum=MAX_EXAMPLES,
        )
        self.max_examples = _bounded_int(
            max_examples,
            "max_examples",
            minimum=1,
            maximum=MAX_EXAMPLES,
        )
        if self.min_examples > self.max_examples:
            raise ProgressiveValidationError(
                "min_examples must not exceed max_examples",
                code="INVALID_CONFIGURATION",
            )
        self.max_example_bytes = _bounded_int(
            max_example_bytes,
            "max_example_bytes",
            minimum=1,
            maximum=MAX_EXAMPLE_BYTES,
        )
        self.max_state_bytes = _bounded_int(
            max_state_bytes,
            "max_state_bytes",
            minimum=1,
            maximum=MAX_STATE_BYTES,
        )
        self.max_input_bytes = _bounded_int(
            max_input_bytes,
            "max_input_bytes",
            minimum=1,
            maximum=MAX_STORED_BYTES,
        )
        self.max_output_bytes = _bounded_int(
            max_output_bytes,
            "max_output_bytes",
            minimum=1,
            maximum=MAX_STORED_BYTES,
        )
        self.estimated_tokens_per_run = _bounded_int(
            estimated_tokens_per_run,
            "estimated_tokens_per_run",
            minimum=0,
            maximum=MAX_TOKEN_ESTIMATE,
        )

        # CrystalService validates the namespace and storage interface.
        self.crystals = CrystalService(storage, namespace)
        self._storage = storage
        self.namespace = namespace
        self.fallback = fallback
        self.builder = builder
        self.auto_activate = auto_activate

    def run(
        self,
        task_key: str,
        inputs: JSONValue = None,
        *,
        learn: bool = False,
    ) -> dict[str, JSONValue]:
        """Execute a warm crystal or invoke the cold fallback exactly once.

        A hit returns immediately, without reading learning state and without
        calling either injected callable.  Cold fallback and builder failures
        are represented by safe status codes; exception text and data payloads
        are never copied into an error message.
        """

        if not isinstance(learn, bool):
            raise ProgressiveValidationError(
                "learn must be a boolean",
                code="INVALID_LEARNING_FLAG",
            )
        try:
            Registry.validate_name(task_key)
        except RegistryError as exc:
            raise ProgressiveValidationError(
                "task_key is not a valid stable crystal name",
                code="INVALID_TASK_KEY",
            ) from exc

        resolved_inputs = {} if inputs is None else inputs
        input_snapshot, _ = _snapshot(
            resolved_inputs,
            label="inputs",
            max_bytes=self.max_input_bytes,
        )

        execution_error = False
        try:
            attempted = self.crystals.execute(name=task_key, inputs=input_snapshot)
        except Exception:
            # Availability favors the user's explicit fallback, but learning is
            # disabled for this call because registry integrity is uncertain.
            execution_error = True
            attempted = self._empty_attempt(
                task_key,
                reason_code="CRYSTAL_EXECUTION_ERROR",
            )

        if attempted.get("status") == "hit":
            result = dict(attempted)
            result.update(
                {
                    "path": "warm_hit",
                    "task_key": task_key,
                    "learn_status": "not_applicable",
                    "learn_reason_code": "",
                    "example_count": 0,
                    "message": "Active crystal returned a deterministic warm-path result.",
                }
            )
            return result

        try:
            fallback_value = self.fallback(task_key, input_snapshot)
        except Exception:
            return self._cold_error(
                task_key,
                attempted,
                reason_code="FALLBACK_FAILED",
                message="Cold fallback failed without returning a value.",
            )

        try:
            output_snapshot, output_raw = _snapshot(
                fallback_value,
                label="fallback output",
                max_bytes=self.max_output_bytes,
            )
        except ProgressiveValidationError as exc:
            return self._cold_error(
                task_key,
                attempted,
                reason_code=exc.code,
                message="Cold fallback returned an invalid or oversized JSON value.",
            )

        response = self._cold_success(
            task_key,
            attempted,
            output_snapshot,
            output_raw.decode("utf-8"),
        )
        if not learn:
            response["learn_status"] = "disabled"
            response["message"] = "Cold fallback returned a result; example retention was disabled."
            return response
        if execution_error:
            response.update(
                {
                    "learn_status": "crystal_error",
                    "learn_reason_code": "CRYSTAL_EXECUTION_ERROR",
                    "message": (
                        "Cold fallback returned a result; learning was skipped because "
                        "the crystal registry was unavailable."
                    ),
                }
            )
            return response

        try:
            outcome = self._learn(task_key, input_snapshot, output_snapshot)
        except Exception:
            # Learning is strictly subordinate to the already successful cold
            # fallback.  No storage, builder, or validation failure may discard
            # that usable answer or expose an exception payload.
            outcome = _LearnOutcome(
                "learning_error",
                0,
                reason_code="PROGRESSIVE_LEARNING_ERROR",
                message=("Cold fallback returned a result; progressive learning failed safely."),
            )
        response.update(
            {
                "learn_status": outcome.status,
                "learn_reason_code": outcome.reason_code,
                "example_count": outcome.example_count,
                "message": outcome.message,
            }
        )
        if outcome.version:
            response["version"] = outcome.version
        if outcome.program_hash:
            response["program_hash"] = outcome.program_hash
        return response

    def _learn(
        self,
        task_key: str,
        inputs: JSONValue,
        output: JSONValue,
    ) -> _LearnOutcome:
        """Record one example and reserve at most one in-process build."""

        build_key = (self.namespace, task_key)
        tests: list[dict[str, JSONValue]] | None = None

        with _PROCESS_LOCK:
            try:
                if self._candidate_exists(task_key):
                    return _LearnOutcome(
                        "candidate_exists",
                        self._safe_example_count(task_key),
                        message=(
                            "Cold fallback returned a result; an immutable candidate "
                            "already exists."
                        ),
                    )
                state = self._load_state(task_key)
            except ProgressiveStorageError:
                return _LearnOutcome(
                    "storage_error",
                    0,
                    reason_code="PROGRESSIVE_STORAGE_ERROR",
                    message="Cold fallback returned a result; the learning record was unavailable.",
                )

            if state["quarantined"]:
                return _LearnOutcome(
                    "quarantined",
                    len(state["examples"]),
                    reason_code=state["quarantine_reason"],
                    message="Cold fallback returned a result; this task is quarantined.",
                )

            try:
                example = self._make_example(inputs, output)
            except ProgressiveValidationError:
                return _LearnOutcome(
                    "example_too_large",
                    len(state["examples"]),
                    reason_code="EXAMPLE_SIZE_LIMIT",
                    message="Cold fallback returned a result; the example was too large to retain.",
                )

            existing = next(
                (item for item in state["examples"] if item["input_hash"] == example["input_hash"]),
                None,
            )
            if existing is not None:
                if canonical_json(existing["input"]) != canonical_json(example["input"]):
                    return self._quarantine(
                        task_key,
                        state,
                        "INPUT_HASH_COLLISION",
                    )
                if existing["output_hash"] != example["output_hash"] or canonical_json(
                    existing["output"]
                ) != canonical_json(example["output"]):
                    return self._quarantine(
                        task_key,
                        state,
                        "OUTPUT_CONFLICT",
                    )
                return _LearnOutcome(
                    "duplicate",
                    len(state["examples"]),
                    message="Cold fallback returned a result; the example was already retained.",
                )

            if len(state["examples"]) >= self.max_examples:
                return _LearnOutcome(
                    "capacity_reached",
                    len(state["examples"]),
                    reason_code="EXAMPLE_COUNT_LIMIT",
                    message="Cold fallback returned a result; the example buffer is full.",
                )

            state["examples"].append(example)
            state["revision"] += 1
            count = len(state["examples"])

            if count < self.min_examples:
                try:
                    self._write_state(task_key, state)
                except ProgressiveStorageError:
                    return _LearnOutcome(
                        "storage_error",
                        count - 1,
                        reason_code="PROGRESSIVE_STORAGE_ERROR",
                        message=(
                            "Cold fallback returned a result; the learning example "
                            "could not be persisted."
                        ),
                    )
                return _LearnOutcome(
                    "observed",
                    count,
                    message="Cold fallback returned a result and retained an example.",
                )

            if self.builder is None:
                try:
                    self._write_state(task_key, state)
                except ProgressiveStorageError:
                    return _LearnOutcome(
                        "storage_error",
                        count - 1,
                        reason_code="PROGRESSIVE_STORAGE_ERROR",
                        message=(
                            "Cold fallback returned a result; the learning example "
                            "could not be persisted."
                        ),
                    )
                return _LearnOutcome(
                    "builder_unavailable",
                    count,
                    reason_code="BUILDER_NOT_CONFIGURED",
                    message=(
                        "Cold fallback returned a result and retained an example; "
                        "no candidate builder is configured."
                    ),
                )

            if build_key in _BUILDING:
                try:
                    self._write_state(task_key, state)
                except ProgressiveStorageError:
                    return _LearnOutcome(
                        "storage_error",
                        count - 1,
                        reason_code="PROGRESSIVE_STORAGE_ERROR",
                        message=(
                            "Cold fallback returned a result; the learning example "
                            "could not be persisted."
                        ),
                    )
                return _LearnOutcome(
                    "build_in_progress",
                    count,
                    message=(
                        "Cold fallback returned a result and retained an example; "
                        "candidate generation is already in progress."
                    ),
                )

            if state["last_build_example_count"] >= count:
                # This can occur after a failed build followed by a storage
                # replay.  Do not spend model tokens twice on unchanged data.
                try:
                    self._write_state(task_key, state)
                except ProgressiveStorageError:
                    return _LearnOutcome(
                        "storage_error",
                        count - 1,
                        reason_code="PROGRESSIVE_STORAGE_ERROR",
                        message=(
                            "Cold fallback returned a result; the learning example "
                            "could not be persisted."
                        ),
                    )
                return _LearnOutcome(
                    "build_deferred",
                    count,
                    reason_code="UNCHANGED_EXAMPLES",
                    message="Cold fallback returned a result; unchanged examples were not rebuilt.",
                )

            state["last_build_example_count"] = count
            state["build_attempt_count"] += 1
            try:
                self._write_state(task_key, state)
            except ProgressiveStorageError:
                return _LearnOutcome(
                    "storage_error",
                    count - 1,
                    reason_code="PROGRESSIVE_STORAGE_ERROR",
                    message=(
                        "Cold fallback returned a result; the learning example "
                        "could not be persisted."
                    ),
                )

            _BUILDING.add(build_key)
            tests = self._tests_from_examples(state["examples"])

        assert tests is not None
        try:
            return self._build_candidate(task_key, tests)
        finally:
            with _PROCESS_LOCK:
                _BUILDING.discard(build_key)

    def _build_candidate(
        self,
        task_key: str,
        tests: list[dict[str, JSONValue]],
    ) -> _LearnOutcome:
        count = len(tests)
        assert self.builder is not None
        builder_tests, _ = _snapshot(
            tests,
            label="builder examples",
            max_bytes=MAX_TESTS_BYTES,
        )
        try:
            proposed = self.builder(task_key, builder_tests)
            program = self._program_from_builder(proposed)
        except Exception:
            return _LearnOutcome(
                "build_failed",
                count,
                reason_code="BUILDER_FAILED",
                message="Cold fallback returned a result; candidate generation was rejected.",
            )

        # Include any examples that arrived while the builder was running.  The
        # builder need not have seen a case for it to remain part of the
        # mechanical consistency suite. Holding the process lock through
        # registration keeps same-worker observers from slipping an unchecked
        # example between the refresh and the immutable candidate write.
        with _PROCESS_LOCK:
            try:
                if self._candidate_exists(task_key):
                    return _LearnOutcome(
                        "candidate_exists",
                        self._safe_example_count(task_key),
                        message=("Cold fallback returned a result; a candidate already exists."),
                    )
                current_state = self._load_state(task_key)
                if current_state["quarantined"]:
                    return _LearnOutcome(
                        "quarantined",
                        len(current_state["examples"]),
                        reason_code=current_state["quarantine_reason"],
                        message=(
                            "Cold fallback returned a result; candidate generation stopped "
                            "because this task was quarantined."
                        ),
                    )
                tests = self._tests_from_examples(current_state["examples"])
                count = len(tests)
            except ProgressiveStorageError:
                return _LearnOutcome(
                    "build_failed",
                    count,
                    reason_code="CRYSTAL_REGISTRY_ERROR",
                    message=("Cold fallback returned a result; candidate storage was unavailable."),
                )

            try:
                created = self.crystals.crystallize(
                    name=task_key,
                    description=f"Progressively learned deterministic task {task_key}.",
                    program=program,
                    tests=tests,
                    activation_policy=("activate_after_tests" if self.auto_activate else "draft"),
                    estimated_tokens_per_run=self.estimated_tokens_per_run,
                )
            except Exception:
                return _LearnOutcome(
                    "build_failed",
                    count,
                    reason_code="CANDIDATE_VALIDATION_FAILED",
                    message=(
                        "Cold fallback returned a result; the proposed crystal failed validation."
                    ),
                )

        active = created.get("active") is True
        return _LearnOutcome(
            "candidate_active" if active else "candidate_created",
            count,
            version=created["version"],
            program_hash=created["program_hash"],
            message=(
                "Cold fallback returned a result; a consistency-checked candidate is active."
                if active
                else (
                    "Cold fallback returned a result; a consistency-checked "
                    "candidate draft was created."
                )
            ),
        )

    def _program_from_builder(self, proposed: JSONValue) -> dict[str, JSONValue]:
        """Normalize a bare program or a ``{"program": ...}`` envelope.

        A builder may include suggested tests in its envelope, but only retained
        observations are used for the mechanical consistency gate.  This gate
        is not an independent correctness oracle.
        """

        if isinstance(proposed, (str, bytes, bytearray)):
            raw = proposed.encode("utf-8") if isinstance(proposed, str) else bytes(proposed)
            if len(raw) > MAX_BUILDER_RESPONSE_BYTES:
                raise ProgressiveValidationError(
                    "builder response exceeds its size limit",
                    code="BUILDER_SIZE_LIMIT",
                )
            try:
                proposed = canonical_loads(raw)
            except Exception as exc:
                raise ProgressiveValidationError(
                    "builder response must be strict JSON",
                    code="INVALID_BUILDER_RESPONSE",
                ) from exc

        normalized, _ = _snapshot(
            proposed,
            label="builder response",
            max_bytes=MAX_BUILDER_RESPONSE_BYTES,
        )
        if not isinstance(normalized, Mapping):
            raise ProgressiveValidationError(
                "builder response must be an object",
                code="INVALID_BUILDER_RESPONSE",
            )
        if "version" in normalized and "expression" in normalized:
            program = normalized
        else:
            unknown = set(normalized) - {"program", "tests", "description"}
            if unknown or "program" not in normalized:
                raise ProgressiveValidationError(
                    "builder envelope must contain program",
                    code="INVALID_BUILDER_RESPONSE",
                )
            program = normalized["program"]
        program_snapshot, _ = _snapshot(
            program,
            label="builder program",
            max_bytes=MAX_PROGRAM_BYTES,
        )
        if not isinstance(program_snapshot, dict):
            raise ProgressiveValidationError(
                "builder program must be an object",
                code="INVALID_BUILDER_RESPONSE",
            )
        return program_snapshot

    def _make_example(
        self,
        inputs: JSONValue,
        output: JSONValue,
    ) -> dict[str, JSONValue]:
        example = {
            "input_hash": content_hash(inputs),
            "output_hash": content_hash(output),
            "input": inputs,
            "output": output,
        }
        # This check happens after fallback and therefore never discards a valid
        # cold result merely because retention policy is narrower.
        snapshot, _ = _snapshot(
            example,
            label="learning example",
            max_bytes=self.max_example_bytes,
        )
        assert isinstance(snapshot, dict)
        return snapshot

    @staticmethod
    def _tests_from_examples(
        examples: Sequence[Mapping[str, JSONValue]],
    ) -> list[dict[str, JSONValue]]:
        return [
            {
                "name": f"observed_{index + 1:03d}",
                "input": example["input"],
                "expected": example["output"],
            }
            for index, example in enumerate(examples)
        ]

    def _candidate_exists(self, task_key: str) -> bool:
        try:
            latest = self.crystals.registry.latest(task_key)
        except RegistryNotFoundError:
            return False
        except RegistryError as exc:
            raise ProgressiveStorageError(
                "crystal registry could not be inspected",
                code="CRYSTAL_REGISTRY_ERROR",
            ) from exc
        try:
            for version in range(latest.version, 0, -1):
                candidate = (
                    latest
                    if version == latest.version
                    else self.crystals.registry.get(task_key, version)
                )
                if candidate.state is not Lifecycle.RETIRED:
                    return True
        except RegistryError as exc:
            raise ProgressiveStorageError(
                "crystal registry could not be inspected",
                code="CRYSTAL_REGISTRY_ERROR",
            ) from exc
        return False

    def _safe_example_count(self, task_key: str) -> int:
        try:
            return len(self._load_state(task_key)["examples"])
        except ProgressiveStorageError:
            return 0

    def _state_key(self, task_key: str) -> str:
        return f"{self.namespace}:progressive:v1:examples:{task_key}"

    def _read_raw(self, key: str) -> bytes | None:
        exists = getattr(self._storage, "exist", None)
        if callable(exists):
            try:
                if not exists(key):
                    return None
            except Exception as exc:
                raise ProgressiveStorageError("learning storage existence check failed") from exc
        try:
            value = self._storage.get(key)
        except Exception as exc:
            raise ProgressiveStorageError("learning storage read failed") from exc
        if value is None:
            return None
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise ProgressiveStorageError("learning storage returned non-bytes")
        raw = bytes(value)
        if len(raw) > self.max_state_bytes:
            raise ProgressiveStorageError("learning record exceeds its size limit")
        return raw

    def _load_state(self, task_key: str) -> dict[str, JSONValue]:
        raw = self._read_raw(self._state_key(task_key))
        if raw is None:
            return self._empty_state(task_key)
        try:
            state = json_from_bytes(
                raw,
                source="progressive learning record",
                require_canonical=True,
                max_bytes=self.max_state_bytes,
            )
        except RegistryError as exc:
            raise ProgressiveStorageError("learning record is corrupt") from exc
        self._validate_state(state, task_key)
        assert isinstance(state, dict)
        return state

    def _write_state(
        self,
        task_key: str,
        state: dict[str, JSONValue],
    ) -> None:
        self._validate_state(state, task_key)
        try:
            raw = stable_json_bytes(state, max_bytes=self.max_state_bytes)
            self._storage.set(self._state_key(task_key), raw)
        except Exception as exc:
            raise ProgressiveStorageError("learning record write failed") from exc

    @staticmethod
    def _empty_state(task_key: str) -> dict[str, JSONValue]:
        return {
            "schema": STATE_SCHEMA,
            "task_key": task_key,
            "revision": 0,
            "quarantined": False,
            "quarantine_reason": "",
            "conflict_count": 0,
            "build_attempt_count": 0,
            "last_build_example_count": 0,
            "examples": [],
        }

    def _validate_state(self, value: JSONValue, task_key: str) -> None:
        if not isinstance(value, dict) or set(value) != _STATE_FIELDS:
            raise ProgressiveStorageError("learning record has an invalid shape")
        if value["schema"] != STATE_SCHEMA or value["task_key"] != task_key:
            raise ProgressiveStorageError("learning record has an invalid identity")
        for field in (
            "revision",
            "conflict_count",
            "build_attempt_count",
            "last_build_example_count",
        ):
            number = value[field]
            if not _is_int(number) or not 0 <= number <= MAX_COUNTER:
                raise ProgressiveStorageError("learning record has an invalid counter")
        if not isinstance(value["quarantined"], bool):
            raise ProgressiveStorageError("learning record has an invalid quarantine flag")
        if value["quarantine_reason"] not in {
            "",
            "INPUT_HASH_COLLISION",
            "OUTPUT_CONFLICT",
        }:
            raise ProgressiveStorageError("learning record has an invalid quarantine reason")
        if value["quarantined"] != bool(value["quarantine_reason"]):
            raise ProgressiveStorageError("learning record has inconsistent quarantine state")

        examples = value["examples"]
        if not isinstance(examples, list) or len(examples) > MAX_EXAMPLES:
            raise ProgressiveStorageError("learning record has invalid examples")
        if value["last_build_example_count"] > len(examples):
            raise ProgressiveStorageError("learning record has an invalid build cursor")

        seen_inputs: set[str] = set()
        for example in examples:
            if not isinstance(example, dict) or set(example) != _EXAMPLE_FIELDS:
                raise ProgressiveStorageError("learning record contains an invalid example")
            input_hash = example["input_hash"]
            output_hash = example["output_hash"]
            if (
                not isinstance(input_hash, str)
                or len(input_hash) != _HASH_LENGTH
                or any(character not in "0123456789abcdef" for character in input_hash)
                or not isinstance(output_hash, str)
                or len(output_hash) != _HASH_LENGTH
                or any(character not in "0123456789abcdef" for character in output_hash)
            ):
                raise ProgressiveStorageError("learning record contains an invalid hash")
            try:
                actual_input_hash = content_hash(example["input"])
                actual_output_hash = content_hash(example["output"])
            except Exception as exc:
                raise ProgressiveStorageError("learning record contains invalid JSON") from exc
            if input_hash != actual_input_hash or output_hash != actual_output_hash:
                raise ProgressiveStorageError("learning record failed its content hash")
            if input_hash in seen_inputs:
                raise ProgressiveStorageError("learning record contains duplicate inputs")
            seen_inputs.add(input_hash)

    def _quarantine(
        self,
        task_key: str,
        state: dict[str, JSONValue],
        reason_code: str,
    ) -> _LearnOutcome:
        state["quarantined"] = True
        state["quarantine_reason"] = reason_code
        state["conflict_count"] += 1
        state["revision"] += 1
        try:
            self._write_state(task_key, state)
        except ProgressiveStorageError:
            return _LearnOutcome(
                "storage_error",
                len(state["examples"]),
                reason_code="PROGRESSIVE_STORAGE_ERROR",
                message=(
                    "Cold fallback returned a result; the deterministic conflict "
                    "could not be persisted."
                ),
            )
        return _LearnOutcome(
            "quarantined",
            len(state["examples"]),
            reason_code=reason_code,
            message=("Cold fallback returned a result; conflicting outputs quarantined this task."),
        )

    @staticmethod
    def _empty_attempt(
        task_key: str,
        *,
        reason_code: str,
    ) -> dict[str, JSONValue]:
        return {
            "status": "error",
            "fallback_required": True,
            "crystal_name": task_key,
            "version": 0,
            "program_hash": "",
            "result": None,
            "result_json": "",
            "receipt": "",
            "reason_code": reason_code,
            "estimated_tokens_avoided": 0,
            "telemetry_recorded": False,
        }

    @staticmethod
    def _cold_success(
        task_key: str,
        attempted: Mapping[str, JSONValue],
        result: JSONValue,
        result_json: str,
    ) -> dict[str, JSONValue]:
        return {
            "status": "fallback",
            "path": "cold_fallback",
            "fallback_required": False,
            "crystal_name": task_key,
            "task_key": task_key,
            "version": attempted.get("version", 0),
            "program_hash": attempted.get("program_hash", ""),
            "result": result,
            "result_json": result_json,
            "receipt": "",
            "reason_code": attempted.get("reason_code", "CRYSTAL_MISS"),
            "learn_status": "disabled",
            "learn_reason_code": "",
            "example_count": 0,
            "estimated_tokens_avoided": 0,
            "telemetry_recorded": False,
            "message": "Cold fallback returned a result.",
        }

    @staticmethod
    def _cold_error(
        task_key: str,
        attempted: Mapping[str, JSONValue],
        *,
        reason_code: str,
        message: str,
    ) -> dict[str, JSONValue]:
        return {
            "status": "error",
            "path": "cold_error",
            "fallback_required": True,
            "crystal_name": task_key,
            "task_key": task_key,
            "version": attempted.get("version", 0),
            "program_hash": attempted.get("program_hash", ""),
            "result": None,
            "result_json": "",
            "receipt": "",
            "reason_code": reason_code,
            "learn_status": "not_recorded",
            "learn_reason_code": reason_code,
            "example_count": 0,
            "estimated_tokens_avoided": 0,
            "telemetry_recorded": False,
            "message": message,
        }


__all__ = [
    "BuilderCallable",
    "DEFAULT_MAX_EXAMPLES",
    "DEFAULT_MAX_EXAMPLE_BYTES",
    "DEFAULT_MIN_EXAMPLES",
    "FallbackCallable",
    "ProgressiveError",
    "ProgressiveService",
    "ProgressiveStorageError",
    "ProgressiveValidationError",
    "STATE_SCHEMA",
]
