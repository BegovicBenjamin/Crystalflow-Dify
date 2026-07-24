"""Public data models for the CrystalFlow registry.

The registry stores JSON snapshots, so these models intentionally contain only
JSON-compatible data and deterministic counters.  There are no timestamps or
process-specific identifiers in the storage format.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Lifecycle(StrEnum):
    """Lifecycle state of one immutable crystal version."""

    DRAFT = "draft"
    ACTIVE = "active"
    RETIRED = "retired"


# A precise recursive JSON alias is awkward for callers that construct programs
# incrementally.  ``Any`` is used at the API edge; Registry validates and
# snapshots every value through the canonical JSON serializer before storage.
JSONValue = Any


@dataclass(frozen=True, slots=True)
class CrystalVersion:
    """A materialized crystal version.

    ``program``, ``tests`` and ``metadata`` are fresh decoded snapshots.  Their
    stored representation is immutable even though nested Python containers in
    this returned view can still be mutated by a caller.
    """

    name: str
    version: int
    content_hash: str
    program_hash: str
    program: JSONValue
    tests: JSONValue
    metadata: Mapping[str, JSONValue]
    description: str
    state: Lifecycle
    revision: int
    alias_revision: int
    run_count: int = 0
    estimated_tokens_avoided: int = 0
    created: bool = False

    @property
    def record_hash(self) -> str:
        """Compatibility name for the content-addressed record hash."""

        return self.content_hash

    @property
    def version_hash(self) -> str:
        """Compatibility name for the content-addressed record hash."""

        return self.content_hash

    @property
    def status(self) -> str:
        """String lifecycle state, convenient for plugin responses."""

        return self.state.value

    @property
    def runs(self) -> int:
        """Compatibility alias for ``run_count``."""

        return self.run_count

    @property
    def tests_passed(self) -> bool:
        """Whether this version carries the explicit activation gate."""

        return self.metadata.get("tests_passed") is True

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a JSON-compatible public representation."""

        return {
            "name": self.name,
            "version": self.version,
            "content_hash": self.content_hash,
            "program_hash": self.program_hash,
            "program": self.program,
            "tests": self.tests,
            "metadata": dict(self.metadata),
            "description": self.description,
            "state": self.state.value,
            "revision": self.revision,
            "alias_revision": self.alias_revision,
            "run_count": self.run_count,
            "estimated_tokens_avoided": self.estimated_tokens_avoided,
            "created": self.created,
        }


@dataclass(frozen=True, slots=True)
class Telemetry:
    """Best-effort execution counters for one version or an aggregate."""

    name: str
    version: int | None
    run_count: int
    estimated_tokens_avoided: int
    revision: int

    @property
    def runs(self) -> int:
        return self.run_count

    @property
    def count(self) -> int:
        return self.run_count

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "name": self.name,
            "version": self.version,
            "run_count": self.run_count,
            "estimated_tokens_avoided": self.estimated_tokens_avoided,
            "revision": self.revision,
        }


@dataclass(frozen=True, slots=True)
class CrystalSummary:
    """Compact deterministic catalog entry returned by ``Registry.list``."""

    name: str
    description: str
    revision: int
    version_count: int
    latest_version: int
    active_version: int | None
    active_program_hash: str | None
    draft_versions: tuple[int, ...]
    retired_versions: tuple[int, ...]
    run_count: int
    estimated_tokens_avoided: int

    @property
    def runs(self) -> int:
        return self.run_count

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "name": self.name,
            "description": self.description,
            "revision": self.revision,
            "version_count": self.version_count,
            "latest_version": self.latest_version,
            "active_version": self.active_version,
            "active_program_hash": self.active_program_hash,
            "draft_versions": list(self.draft_versions),
            "retired_versions": list(self.retired_versions),
            "run_count": self.run_count,
            "estimated_tokens_avoided": self.estimated_tokens_avoided,
        }
