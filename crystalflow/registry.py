"""Deterministic, dependency-free persistence for CrystalFlow.

The storage object is deliberately limited to Dify's workspace KV contract:
``exist(key)``, ``get(key)``, ``set(key, bytes)`` and ``delete(key)``.  The
``exist`` method is probed when available because Dify raises on a missing
``get``; simpler test and embedded stores may instead return ``None``.  Since
that contract has neither enumeration nor compare-and-swap, the catalog and
telemetry are best-effort under multi-process concurrency. A process-wide lock
serializes all ``Registry`` instances inside one plugin worker; deployments
with multiple workers still need a single administrative writer.
"""

from __future__ import annotations

import json
import re
import threading
import unicodedata
from dataclasses import replace
from typing import Any, Protocol, runtime_checkable

from .canonical import canonical_json_bytes, content_hash
from .errors import CanonicalizationError
from .models import CrystalSummary, CrystalVersion, Lifecycle, Telemetry

FORMAT_VERSION = 1
DEFAULT_NAMESPACE = "crystalflow"

MAX_NAMESPACE_LENGTH = 64
MAX_NAME_LENGTH = 128
MAX_DESCRIPTION_BYTES = 4 * 1024
MAX_PROGRAM_BYTES = 256 * 1024
MAX_TESTS_BYTES = 256 * 1024
MAX_METADATA_BYTES = 64 * 1024
MAX_RECORD_BYTES = 768 * 1024
MAX_STORED_BYTES = 2 * 1024 * 1024
MAX_VERSIONS_PER_CRYSTAL = 1_000
MAX_COUNTER = (1 << 63) - 1

_NAMESPACE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_EXPECTED_UNSET = object()
_PROCESS_LOCK = threading.RLock()


class RegistryError(Exception):
    """Base class for registry failures safe to present without payloads."""


class RegistryValidationError(RegistryError, ValueError):
    """A caller supplied an invalid name, hash, value, or transition."""


class RegistryNotFoundError(RegistryError, LookupError):
    """A crystal or version does not exist (or no active version exists)."""


class RegistryConflictError(RegistryError):
    """An optimistic expectation or lifecycle gate was not satisfied."""


class RegistryActivationError(RegistryConflictError):
    """A version cannot pass the activation gate."""


class RegistryCorruptionError(RegistryError):
    """Stored bytes violate the registry format or their content hash."""


class RegistryStorageError(RegistryError):
    """The underlying KV implementation failed."""


# Short aliases are convenient for consumers while the prefixed names remain
# unambiguous when imported alongside engine validation errors.
ValidationError = RegistryValidationError
NotFoundError = RegistryNotFoundError
ConflictError = RegistryConflictError
ActivationError = RegistryActivationError
CorruptionError = RegistryCorruptionError
StorageError = RegistryStorageError


@runtime_checkable
class KVStore(Protocol):
    """Minimum structural form accepted from KV implementations.

    Dify's concrete storage additionally exposes ``exist(key)``. It is detected
    dynamically so lightweight stores whose ``get`` returns ``None`` remain
    compatible.
    """

    def get(self, key: str) -> bytes | None: ...

    def set(self, key: str, value: bytes) -> None: ...

    def delete(self, key: str) -> None: ...


def stable_json_bytes(value: Any, *, max_bytes: int = MAX_STORED_BYTES) -> bytes:
    """Serialize a JSON value to the project's stable canonical byte format."""

    try:
        encoded = canonical_json_bytes(value)
    except (
        CanonicalizationError,
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise RegistryValidationError("value is not canonical JSON data") from exc
    if not isinstance(encoded, bytes):
        raise RegistryValidationError("canonical serializer did not return bytes")
    if len(encoded) > max_bytes:
        raise RegistryValidationError(f"canonical JSON exceeds the {max_bytes}-byte storage limit")
    return encoded


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def json_from_bytes(
    payload: bytes | bytearray | memoryview,
    *,
    source: str = "stored value",
    require_canonical: bool = False,
    max_bytes: int = MAX_STORED_BYTES,
) -> Any:
    """Decode strict UTF-8 JSON, rejecting duplicates and non-finite numbers.

    Registry internals set ``require_canonical`` so hand-edited or partially
    written records fail loudly instead of being silently normalized.
    """

    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise RegistryCorruptionError(f"{source} is not bytes")
    raw = bytes(payload)
    if len(raw) > max_bytes:
        raise RegistryCorruptionError(f"{source} exceeds the storage size limit")
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise RegistryCorruptionError(f"{source} is not valid strict JSON") from exc
    if require_canonical:
        try:
            normalized = canonical_json_bytes(value)
        except (
            CanonicalizationError,
            TypeError,
            ValueError,
            OverflowError,
            RecursionError,
        ) as exc:
            raise RegistryCorruptionError(f"{source} is not canonicalizable JSON") from exc
        if normalized != raw:
            raise RegistryCorruptionError(f"{source} is not in canonical JSON form")
    return value


# Friendly aliases for integrations that prefer serialize/deserialize naming.
serialize_json = stable_json_bytes
deserialize_json = json_from_bytes


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _bounded_nonnegative_int(value: Any, label: str) -> int:
    if not _is_int(value) or value < 0 or value > MAX_COUNTER:
        raise RegistryValidationError(f"{label} must be an integer from 0 to {MAX_COUNTER}")
    return value


class Registry:
    """Content-addressed crystal versions with mutable name aliases."""

    def __init__(self, storage: KVStore, namespace: str = DEFAULT_NAMESPACE) -> None:
        self._validate_namespace(namespace)
        for method in ("get", "set", "delete"):
            if not callable(getattr(storage, method, None)):
                raise RegistryValidationError(
                    "storage must provide callable get, set, and delete methods"
                )
        self._storage = storage
        self.namespace = namespace
        self._prefix = f"{namespace}:v{FORMAT_VERSION}"
        # Dify constructs fresh tool/registry objects per request and invokes
        # them concurrently in a thread pool. Instance-local locks therefore
        # cannot protect KV read-modify-write sequences.
        self._lock = _PROCESS_LOCK
        self._last_index_error: RegistryStorageError | None = None

    @staticmethod
    def _validate_namespace(namespace: str) -> None:
        if (
            not isinstance(namespace, str)
            or not _NAMESPACE_RE.fullmatch(namespace)
            or ".." in namespace
        ):
            raise RegistryValidationError(
                "namespace must be 1-64 ASCII letters, digits, dots, underscores, "
                "or hyphens, start with a letter or digit, and not contain '..'"
            )

    @staticmethod
    def validate_name(name: str) -> str:
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name) or ".." in name:
            raise RegistryValidationError(
                "crystal name must be 1-128 ASCII letters, digits, dots, underscores, "
                "or hyphens, start with a letter or digit, and not contain '..'"
            )
        return name

    @property
    def last_index_error(self) -> RegistryStorageError | None:
        """Most recent swallowed catalog error, if the best-effort update failed."""

        return self._last_index_error

    def _key(self, *parts: object) -> str:
        return ":".join((self._prefix, *(str(part) for part in parts)))

    @property
    def _index_key(self) -> str:
        return self._key("index")

    def _alias_key(self, name: str) -> str:
        return self._key("alias", name)

    def _record_key(self, record_hash: str) -> str:
        return self._key("record", record_hash)

    def _telemetry_key(self, name: str, version: int) -> str:
        return self._key("telemetry", name, version)

    def _read_raw(self, key: str) -> bytes | None:
        exists = getattr(self._storage, "exist", None)
        if callable(exists):
            try:
                if not exists(key):
                    return None
            except Exception as exc:
                raise RegistryStorageError("KV existence check failed") from exc
        try:
            value = self._storage.get(key)
        except Exception as exc:
            raise RegistryStorageError("KV get failed") from exc
        if value is None:
            return None
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise RegistryCorruptionError("KV value is not bytes")
        raw = bytes(value)
        if len(raw) > MAX_STORED_BYTES:
            raise RegistryCorruptionError("KV value exceeds the storage size limit")
        return raw

    def _write_raw(self, key: str, value: bytes) -> None:
        if not isinstance(value, bytes):
            raise RegistryValidationError("KV writes must be bytes")
        if len(value) > MAX_STORED_BYTES:
            raise RegistryValidationError("KV value exceeds the storage size limit")
        try:
            self._storage.set(key, value)
        except Exception as exc:
            raise RegistryStorageError("KV set failed") from exc

    def _read_object(self, key: str, source: str) -> dict[str, Any] | None:
        raw = self._read_raw(key)
        if raw is None:
            return None
        value = json_from_bytes(raw, source=source, require_canonical=True)
        if not isinstance(value, dict):
            raise RegistryCorruptionError(f"{source} must be a JSON object")
        return value

    def _write_object(self, key: str, value: dict[str, Any], max_bytes: int) -> bytes:
        raw = stable_json_bytes(value, max_bytes=max_bytes)
        self._write_raw(key, raw)
        return raw

    def _snapshot(self, value: Any, label: str, max_bytes: int) -> Any:
        try:
            raw = stable_json_bytes(value, max_bytes=max_bytes)
        except RegistryValidationError as exc:
            raise RegistryValidationError(f"{label}: {exc}") from exc
        # Decoding the bytes ensures storage never retains caller-owned mutable
        # containers and returns the same normalized value that will be hashed.
        return json_from_bytes(raw, source=label)

    def _normalize_description(self, description: str | None) -> str | None:
        if description is None:
            return None
        if not isinstance(description, str):
            raise RegistryValidationError("description must be a string")
        normalized = unicodedata.normalize("NFC", description)
        if len(normalized.encode("utf-8")) > MAX_DESCRIPTION_BYTES:
            raise RegistryValidationError(
                f"description exceeds {MAX_DESCRIPTION_BYTES} UTF-8 bytes"
            )
        return normalized

    def _empty_alias(self, name: str, description: str) -> dict[str, Any]:
        return {
            "schema": "crystalflow.alias.v1",
            "name": name,
            "description": description,
            "revision": 0,
            "latest_version": 0,
            "active_version": None,
            "versions": [],
        }

    def _load_alias(self, name: str, *, required: bool) -> dict[str, Any] | None:
        alias = self._read_object(self._alias_key(name), f"alias for {name!r}")
        if alias is None:
            if required:
                raise RegistryNotFoundError(f"crystal {name!r} was not found")
            return None
        self._validate_alias(alias, expected_name=name)
        return alias

    def _validate_alias(self, alias: dict[str, Any], *, expected_name: str) -> None:
        if alias.get("schema") != "crystalflow.alias.v1":
            raise RegistryCorruptionError(f"alias for {expected_name!r} has an unknown schema")
        if alias.get("name") != expected_name:
            raise RegistryCorruptionError(f"alias for {expected_name!r} has the wrong name")
        description = alias.get("description")
        if not isinstance(description, str):
            raise RegistryCorruptionError(f"alias for {expected_name!r} has no description")
        if len(description.encode("utf-8")) > MAX_DESCRIPTION_BYTES:
            raise RegistryCorruptionError(
                f"alias for {expected_name!r} has an oversized description"
            )
        revision = alias.get("revision")
        latest = alias.get("latest_version")
        active = alias.get("active_version")
        versions = alias.get("versions")
        if not _is_int(revision) or revision < 1:
            raise RegistryCorruptionError(f"alias for {expected_name!r} has an invalid revision")
        if not _is_int(latest) or latest < 1:
            raise RegistryCorruptionError(
                f"alias for {expected_name!r} has an invalid latest version"
            )
        if active is not None and (not _is_int(active) or active < 1):
            raise RegistryCorruptionError(
                f"alias for {expected_name!r} has an invalid active version"
            )
        if not isinstance(versions, list) or not versions:
            raise RegistryCorruptionError(f"alias for {expected_name!r} has no versions")
        if len(versions) > MAX_VERSIONS_PER_CRYSTAL:
            raise RegistryCorruptionError(f"alias for {expected_name!r} has too many versions")

        seen: set[int] = set()
        active_entries = 0
        previous = 0
        for entry in versions:
            if not isinstance(entry, dict):
                raise RegistryCorruptionError(
                    f"alias for {expected_name!r} contains an invalid version entry"
                )
            version = entry.get("version")
            record_hash = entry.get("content_hash")
            program_hash = entry.get("program_hash")
            state = entry.get("state")
            created_revision = entry.get("revision")
            if not _is_int(version) or version < 1 or version in seen or version <= previous:
                raise RegistryCorruptionError(
                    f"alias for {expected_name!r} has invalid version ordering"
                )
            if not isinstance(record_hash, str) or not _HASH_RE.fullmatch(record_hash):
                raise RegistryCorruptionError(
                    f"alias for {expected_name!r} has an invalid content hash"
                )
            if not isinstance(program_hash, str) or not _HASH_RE.fullmatch(program_hash):
                raise RegistryCorruptionError(
                    f"alias for {expected_name!r} has an invalid program hash"
                )
            if state not in {item.value for item in Lifecycle}:
                raise RegistryCorruptionError(
                    f"alias for {expected_name!r} has an invalid lifecycle state"
                )
            if not _is_int(created_revision) or not 1 <= created_revision <= revision:
                raise RegistryCorruptionError(
                    f"alias for {expected_name!r} has an invalid version revision"
                )
            if state == Lifecycle.ACTIVE.value:
                active_entries += 1
                if active != version:
                    raise RegistryCorruptionError(
                        f"alias for {expected_name!r} has an inconsistent active pointer"
                    )
            seen.add(version)
            previous = version
        if latest != previous:
            raise RegistryCorruptionError(
                f"alias for {expected_name!r} has an inconsistent latest version"
            )
        if active is None and active_entries != 0:
            raise RegistryCorruptionError(
                f"alias for {expected_name!r} has an active state without a pointer"
            )
        if active is not None and (active not in seen or active_entries != 1):
            raise RegistryCorruptionError(
                f"alias for {expected_name!r} has an inconsistent active version"
            )

    def _entry(
        self, alias: dict[str, Any], version: int | None, *, require_active: bool
    ) -> dict[str, Any]:
        if version is None:
            version = alias["active_version"]
            if version is None:
                if require_active:
                    raise RegistryNotFoundError(f"crystal {alias['name']!r} has no active version")
                version = alias["latest_version"]
        if not _is_int(version) or version < 1:
            raise RegistryValidationError("version must be a positive integer")
        for entry in alias["versions"]:
            if entry["version"] == version:
                return entry
        raise RegistryNotFoundError(f"version {version} of crystal {alias['name']!r} was not found")

    def _load_record(self, name: str, entry: dict[str, Any]) -> dict[str, Any]:
        record_hash = entry["content_hash"]
        record = self._read_object(
            self._record_key(record_hash),
            f"version {entry['version']} of crystal {name!r}",
        )
        if record is None:
            raise RegistryCorruptionError(
                f"version {entry['version']} of crystal {name!r} has no record"
            )
        if record.get("schema") != "crystalflow.version.v1":
            raise RegistryCorruptionError(
                f"version {entry['version']} of crystal {name!r} has an unknown schema"
            )
        try:
            actual_hash = content_hash(record)
        except (
            CanonicalizationError,
            TypeError,
            ValueError,
            OverflowError,
            RecursionError,
        ) as exc:
            raise RegistryCorruptionError(
                f"version {entry['version']} of crystal {name!r} cannot be hashed"
            ) from exc
        if actual_hash != record_hash:
            raise RegistryCorruptionError(
                f"version {entry['version']} of crystal {name!r} failed its content hash"
            )
        if record.get("name") != name or record.get("version") != entry["version"]:
            raise RegistryCorruptionError(
                f"version {entry['version']} of crystal {name!r} has the wrong identity"
            )
        if record.get("revision") != entry["revision"]:
            raise RegistryCorruptionError(
                f"version {entry['version']} of crystal {name!r} has the wrong revision"
            )
        if record.get("program_hash") != entry["program_hash"]:
            raise RegistryCorruptionError(
                f"version {entry['version']} of crystal {name!r} has the wrong program hash"
            )
        program = record.get("program")
        metadata = record.get("metadata")
        if not isinstance(program, dict) or not isinstance(metadata, dict):
            raise RegistryCorruptionError(
                f"version {entry['version']} of crystal {name!r} has invalid data"
            )
        description = record.get("description")
        if not isinstance(description, str):
            raise RegistryCorruptionError(
                f"version {entry['version']} of crystal {name!r} has no description"
            )
        try:
            actual_program_hash = content_hash(program)
        except (
            CanonicalizationError,
            TypeError,
            ValueError,
            OverflowError,
            RecursionError,
        ) as exc:
            raise RegistryCorruptionError(
                f"version {entry['version']} of crystal {name!r} has an invalid program"
            ) from exc
        if actual_program_hash != entry["program_hash"]:
            raise RegistryCorruptionError(
                f"version {entry['version']} of crystal {name!r} failed its program hash"
            )
        return record

    def _write_immutable_record(self, record_hash: str, record: dict[str, Any]) -> None:
        raw = stable_json_bytes(record, max_bytes=MAX_RECORD_BYTES)
        key = self._record_key(record_hash)
        existing = self._read_raw(key)
        if existing is None:
            self._write_raw(key, raw)
            return
        if existing != raw:
            raise RegistryCorruptionError("content-addressed record collision or corruption")

    def _load_index(self) -> dict[str, Any]:
        index = self._read_object(self._index_key, "registry index")
        if index is None:
            return {
                "schema": "crystalflow.index.v1",
                "revision": 0,
                "names": [],
            }
        if index.get("schema") != "crystalflow.index.v1":
            raise RegistryCorruptionError("registry index has an unknown schema")
        revision = index.get("revision")
        names = index.get("names")
        if not _is_int(revision) or revision < 0:
            raise RegistryCorruptionError("registry index has an invalid revision")
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise RegistryCorruptionError("registry index has invalid names")
        if names != sorted(set(names)):
            raise RegistryCorruptionError("registry index names are not unique and sorted")
        for name in names:
            try:
                self.validate_name(name)
            except RegistryValidationError as exc:
                raise RegistryCorruptionError("registry index contains an invalid name") from exc
        return index

    def _ensure_indexed(self, name: str) -> None:
        """Best-effort catalog update after the durable alias is written."""

        self._last_index_error = None
        try:
            index = self._load_index()
            if name in index["names"]:
                return
            index["names"].append(name)
            index["names"].sort()
            index["revision"] += 1
            self._write_object(self._index_key, index, MAX_STORED_BYTES)
        except RegistryStorageError as exc:
            # The name remains directly addressable.  KV has no enumeration, so
            # this cannot be repaired automatically without being given a name.
            self._last_index_error = exc

    def register(
        self,
        name: str,
        program: dict[str, Any],
        program_hash: str | None = None,
        tests: Any = None,
        metadata: dict[str, Any] | None = None,
        *,
        description: str | None = None,
    ) -> CrystalVersion:
        """Register a draft, reusing an existing non-retired same-hash version."""

        self.validate_name(name)
        normalized_description = self._normalize_description(description)
        program_snapshot = self._snapshot(program, "program", MAX_PROGRAM_BYTES)
        if not isinstance(program_snapshot, dict):
            raise RegistryValidationError("program must be a JSON object")
        tests_snapshot = self._snapshot([] if tests is None else tests, "tests", MAX_TESTS_BYTES)
        metadata_snapshot = self._snapshot(
            {} if metadata is None else metadata,
            "metadata",
            MAX_METADATA_BYTES,
        )
        if not isinstance(metadata_snapshot, dict):
            raise RegistryValidationError("metadata must be a JSON object")
        if "tests_passed" in metadata_snapshot and not isinstance(
            metadata_snapshot["tests_passed"], bool
        ):
            raise RegistryValidationError("metadata.tests_passed must be a boolean")

        actual_program_hash = content_hash(program_snapshot)
        if program_hash is None:
            program_hash = actual_program_hash
        if not isinstance(program_hash, str) or not _HASH_RE.fullmatch(program_hash):
            raise RegistryValidationError(
                "program_hash must be a 64-character lowercase SHA-256 hex digest"
            )
        if program_hash != actual_program_hash:
            raise RegistryValidationError("program_hash does not match the canonical program")

        with self._lock:
            alias = self._load_alias(name, required=False)
            if alias is not None:
                for existing_entry in alias["versions"]:
                    if (
                        existing_entry["program_hash"] == program_hash
                        and existing_entry["state"] != Lifecycle.RETIRED.value
                    ):
                        self._ensure_indexed(name)
                        result = self._materialize(alias, existing_entry)
                        return replace(result, created=False)
                if len(alias["versions"]) >= MAX_VERSIONS_PER_CRYSTAL:
                    raise RegistryValidationError(
                        f"crystal {name!r} reached the {MAX_VERSIONS_PER_CRYSTAL}-version limit"
                    )
                inherited_description = alias["description"]
            else:
                inherited_description = ""
                alias = self._empty_alias(name, inherited_description)

            version_description = (
                inherited_description if normalized_description is None else normalized_description
            )
            version = alias["latest_version"] + 1
            revision = alias["revision"] + 1
            record = {
                "schema": "crystalflow.version.v1",
                "name": name,
                "version": version,
                "revision": revision,
                "description": version_description,
                "program_hash": program_hash,
                "program": program_snapshot,
                "tests": tests_snapshot,
                "metadata": metadata_snapshot,
            }
            record_hash = content_hash(record)
            if not isinstance(record_hash, str) or not _HASH_RE.fullmatch(record_hash):
                raise RegistryValidationError("content hash function returned an invalid digest")
            self._write_immutable_record(record_hash, record)

            alias["revision"] = revision
            alias["latest_version"] = version
            alias["description"] = version_description
            alias["versions"].append(
                {
                    "version": version,
                    "content_hash": record_hash,
                    "program_hash": program_hash,
                    "state": Lifecycle.DRAFT.value,
                    "revision": revision,
                }
            )
            self._write_object(self._alias_key(name), alias, MAX_STORED_BYTES)
            self._ensure_indexed(name)
            result = self._materialize(alias, alias["versions"][-1])
            return replace(result, created=True)

    def _materialize(self, alias: dict[str, Any], entry: dict[str, Any]) -> CrystalVersion:
        record = self._load_record(alias["name"], entry)
        telemetry = self._load_telemetry(alias["name"], entry)
        return CrystalVersion(
            name=alias["name"],
            version=entry["version"],
            content_hash=entry["content_hash"],
            program_hash=entry["program_hash"],
            program=record["program"],
            tests=record["tests"],
            metadata=record["metadata"],
            description=record["description"],
            state=Lifecycle(entry["state"]),
            revision=entry["revision"],
            alias_revision=alias["revision"],
            run_count=telemetry.run_count,
            estimated_tokens_avoided=telemetry.estimated_tokens_avoided,
        )

    def get(self, name: str, version: int | None = None) -> CrystalVersion:
        """Get an explicit version, or the active version when omitted."""

        self.validate_name(name)
        with self._lock:
            alias = self._load_alias(name, required=True)
            assert alias is not None
            entry = self._entry(alias, version, require_active=version is None)
            return self._materialize(alias, entry)

    get_version = get

    def latest(self, name: str) -> CrystalVersion:
        """Return the latest registered version regardless of lifecycle state."""

        self.validate_name(name)
        with self._lock:
            alias = self._load_alias(name, required=True)
            assert alias is not None
            entry = self._entry(alias, alias["latest_version"], require_active=False)
            return self._materialize(alias, entry)

    def activate(
        self,
        name: str,
        version: int,
        *,
        expected_program_hash: str | None = None,
        expected_active_version: int | None | object = _EXPECTED_UNSET,
        unsafe: bool = False,
    ) -> CrystalVersion:
        """Point the alias at a tested version without rewriting its artifact.

        ``unsafe=True`` is an explicit escape hatch for local recovery and test
        fixtures.  Normal plugin paths must leave it disabled.
        """

        self.validate_name(name)
        if not _is_int(version) or version < 1:
            raise RegistryValidationError("version must be a positive integer")
        if expected_program_hash is not None and (
            not isinstance(expected_program_hash, str)
            or not _HASH_RE.fullmatch(expected_program_hash)
        ):
            raise RegistryValidationError("expected_program_hash is not a SHA-256 digest")
        if (
            expected_active_version is not _EXPECTED_UNSET
            and expected_active_version is not None
            and (not _is_int(expected_active_version) or expected_active_version < 1)
        ):
            raise RegistryValidationError(
                "expected_active_version must be null or a positive integer"
            )
        if not isinstance(unsafe, bool):
            raise RegistryValidationError("unsafe must be a boolean")

        with self._lock:
            alias = self._load_alias(name, required=True)
            assert alias is not None
            if (
                expected_active_version is not _EXPECTED_UNSET
                and alias["active_version"] != expected_active_version
            ):
                raise RegistryConflictError(
                    f"active version changed from {expected_active_version!r} "
                    f"to {alias['active_version']!r}"
                )
            target = self._entry(alias, version, require_active=False)
            if (
                expected_program_hash is not None
                and target["program_hash"] != expected_program_hash
            ):
                raise RegistryConflictError("candidate program hash does not match")
            record = self._load_record(name, target)
            if not unsafe and record["metadata"].get("tests_passed") is not True:
                raise RegistryActivationError(
                    "activation requires metadata.tests_passed to be true"
                )
            if alias["active_version"] == version and target["state"] == Lifecycle.ACTIVE.value:
                return self._materialize(alias, target)

            for entry in alias["versions"]:
                if entry["state"] == Lifecycle.ACTIVE.value:
                    entry["state"] = Lifecycle.RETIRED.value
            target["state"] = Lifecycle.ACTIVE.value
            alias["active_version"] = version
            alias["revision"] += 1
            self._write_object(self._alias_key(name), alias, MAX_STORED_BYTES)
            return self._materialize(alias, target)

    promote = activate

    def retire(self, name: str, version: int | None = None) -> CrystalVersion:
        """Retire a version; records and telemetry are never physically deleted."""

        self.validate_name(name)
        if version is not None and (not _is_int(version) or version < 1):
            raise RegistryValidationError("version must be a positive integer")
        with self._lock:
            alias = self._load_alias(name, required=True)
            assert alias is not None
            target = self._entry(alias, version, require_active=False)
            if target["state"] == Lifecycle.RETIRED.value:
                return self._materialize(alias, target)
            target["state"] = Lifecycle.RETIRED.value
            if alias["active_version"] == target["version"]:
                alias["active_version"] = None
            alias["revision"] += 1
            self._write_object(self._alias_key(name), alias, MAX_STORED_BYTES)
            return self._materialize(alias, target)

    def _load_telemetry(self, name: str, entry: dict[str, Any]) -> Telemetry:
        value = self._read_object(
            self._telemetry_key(name, entry["version"]),
            f"telemetry for version {entry['version']} of crystal {name!r}",
        )
        if value is None:
            return Telemetry(
                name=name,
                version=entry["version"],
                run_count=0,
                estimated_tokens_avoided=0,
                revision=0,
            )
        if value.get("schema") != "crystalflow.telemetry.v1":
            raise RegistryCorruptionError(
                f"telemetry for version {entry['version']} of crystal {name!r} "
                "has an unknown schema"
            )
        if (
            value.get("name") != name
            or value.get("version") != entry["version"]
            or value.get("content_hash") != entry["content_hash"]
        ):
            raise RegistryCorruptionError(
                f"telemetry for version {entry['version']} of crystal {name!r} "
                "has the wrong identity"
            )
        for field in ("run_count", "estimated_tokens_avoided", "revision"):
            item = value.get(field)
            if not _is_int(item) or item < 0 or item > MAX_COUNTER:
                raise RegistryCorruptionError(
                    f"telemetry for version {entry['version']} of crystal {name!r} "
                    f"has an invalid {field}"
                )
        return Telemetry(
            name=name,
            version=entry["version"],
            run_count=value["run_count"],
            estimated_tokens_avoided=value["estimated_tokens_avoided"],
            revision=value["revision"],
        )

    def record_run(
        self,
        name: str,
        version: int | None = None,
        estimated_tokens_avoided: int = 0,
        *,
        count: int = 1,
    ) -> Telemetry:
        """Increment best-effort counters for an explicit or active version."""

        self.validate_name(name)
        avoided = _bounded_nonnegative_int(estimated_tokens_avoided, "estimated_tokens_avoided")
        count = _bounded_nonnegative_int(count, "count")
        if count == 0:
            raise RegistryValidationError("count must be at least 1")
        if version is not None and (not _is_int(version) or version < 1):
            raise RegistryValidationError("version must be a positive integer")
        with self._lock:
            alias = self._load_alias(name, required=True)
            assert alias is not None
            entry = self._entry(alias, version, require_active=version is None)
            previous = self._load_telemetry(name, entry)
            if previous.run_count > MAX_COUNTER - count:
                raise RegistryValidationError("run_count would overflow")
            if previous.estimated_tokens_avoided > MAX_COUNTER - avoided:
                raise RegistryValidationError("estimated_tokens_avoided would overflow")
            if previous.revision == MAX_COUNTER:
                raise RegistryValidationError("telemetry revision would overflow")
            value = {
                "schema": "crystalflow.telemetry.v1",
                "name": name,
                "version": entry["version"],
                "content_hash": entry["content_hash"],
                "run_count": previous.run_count + count,
                "estimated_tokens_avoided": (previous.estimated_tokens_avoided + avoided),
                "revision": previous.revision + 1,
            }
            self._write_object(
                self._telemetry_key(name, entry["version"]),
                value,
                MAX_STORED_BYTES,
            )
            return Telemetry(
                name=name,
                version=entry["version"],
                run_count=value["run_count"],
                estimated_tokens_avoided=value["estimated_tokens_avoided"],
                revision=value["revision"],
            )

    increment_telemetry = record_run

    def telemetry(self, name: str, version: int | None = None) -> Telemetry:
        """Return version counters, or aggregate all versions when omitted."""

        self.validate_name(name)
        if version is not None and (not _is_int(version) or version < 1):
            raise RegistryValidationError("version must be a positive integer")
        with self._lock:
            alias = self._load_alias(name, required=True)
            assert alias is not None
            if version is not None:
                entry = self._entry(alias, version, require_active=False)
                return self._load_telemetry(name, entry)
            return self._aggregate_telemetry(alias)

    get_telemetry = telemetry

    def _aggregate_telemetry(self, alias: dict[str, Any]) -> Telemetry:
        runs = 0
        avoided = 0
        revision = 0
        for entry in alias["versions"]:
            item = self._load_telemetry(alias["name"], entry)
            runs += item.run_count
            avoided += item.estimated_tokens_avoided
            revision += item.revision
            if runs > MAX_COUNTER or avoided > MAX_COUNTER or revision > MAX_COUNTER:
                raise RegistryCorruptionError("aggregate telemetry counters overflow")
        return Telemetry(
            name=alias["name"],
            version=None,
            run_count=runs,
            estimated_tokens_avoided=avoided,
            revision=revision,
        )

    def list(self) -> list[CrystalSummary]:
        """List indexed crystals in deterministic name order.

        A stale index entry whose alias is absent is skipped because KV lacks a
        transaction spanning alias and catalog writes.  Malformed present data
        still raises ``RegistryCorruptionError``.
        """

        with self._lock:
            index = self._load_index()
            summaries: list[CrystalSummary] = []
            for name in index["names"]:
                alias = self._load_alias(name, required=False)
                if alias is None:
                    continue
                aggregate = self._aggregate_telemetry(alias)
                active_entry = None
                if alias["active_version"] is not None:
                    active_entry = self._entry(alias, alias["active_version"], require_active=False)
                summaries.append(
                    CrystalSummary(
                        name=name,
                        description=alias["description"],
                        revision=alias["revision"],
                        version_count=len(alias["versions"]),
                        latest_version=alias["latest_version"],
                        active_version=alias["active_version"],
                        active_program_hash=(
                            None if active_entry is None else active_entry["program_hash"]
                        ),
                        draft_versions=tuple(
                            entry["version"]
                            for entry in alias["versions"]
                            if entry["state"] == Lifecycle.DRAFT.value
                        ),
                        retired_versions=tuple(
                            entry["version"]
                            for entry in alias["versions"]
                            if entry["state"] == Lifecycle.RETIRED.value
                        ),
                        run_count=aggregate.run_count,
                        estimated_tokens_avoided=aggregate.estimated_tokens_avoided,
                    )
                )
            return summaries

    list_summaries = list

    def ensure_indexed(self, name: str) -> None:
        """Given a known crystal name, retry its best-effort catalog update."""

        self.validate_name(name)
        with self._lock:
            self._load_alias(name, required=True)
            self._ensure_indexed(name)
            if self._last_index_error is not None:
                raise self._last_index_error
