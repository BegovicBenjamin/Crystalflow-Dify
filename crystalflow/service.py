"""Application service shared by the Dify adapters and end-to-end tests."""

from __future__ import annotations

import hmac
from collections.abc import Mapping, Sequence
from typing import Any

from .canonical import canonical_json, content_hash
from .engine import ENGINE_VERSION, IR_VERSION, CrystalFlowEngine
from .errors import CrystalFlowError, InputValidationError
from .models import CrystalSummary, Lifecycle
from .registry import (
    KVStore,
    Registry,
    RegistryError,
    RegistryNotFoundError,
)

MAX_TEST_CASES = 100
MAX_TOKEN_ESTIMATE = 1_000_000


class ServiceValidationError(ValueError):
    """A safe validation error at the crystallization lifecycle boundary."""

    def __init__(self, message: str, *, code: str = "INVALID_REQUEST") -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class CrystalTestError(ServiceValidationError):
    """A proposed program failed one of its exact acceptance tests."""


def _non_negative_int(value: Any, label: str, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= maximum:
        raise ServiceValidationError(
            f"{label} must be an integer from 0 to {maximum}",
            code="INVALID_INTEGER",
        )
    return value


def _validated_tests(tests: Any) -> list[dict[str, Any]]:
    if not isinstance(tests, Sequence) or isinstance(tests, (str, bytes, bytearray, memoryview)):
        raise ServiceValidationError("tests must be a non-empty array", code="INVALID_TESTS")
    if not tests:
        raise ServiceValidationError(
            "at least one exact test is required",
            code="TESTS_REQUIRED",
        )
    if len(tests) > MAX_TEST_CASES:
        raise ServiceValidationError(
            f"at most {MAX_TEST_CASES} tests are allowed",
            code="TOO_MANY_TESTS",
        )

    normalized: list[dict[str, Any]] = []
    for index, test in enumerate(tests):
        if not isinstance(test, Mapping):
            raise ServiceValidationError(
                f"test {index + 1} must be an object",
                code="INVALID_TEST",
            )
        unknown = set(test) - {"name", "input", "expected"}
        if unknown:
            raise ServiceValidationError(
                f"test {index + 1} contains unsupported fields",
                code="INVALID_TEST",
            )
        if "input" not in test or "expected" not in test:
            raise ServiceValidationError(
                f"test {index + 1} needs input and expected",
                code="INVALID_TEST",
            )
        name = test.get("name", f"case_{index + 1}")
        if not isinstance(name, str) or not name or len(name) > 80:
            raise ServiceValidationError(
                f"test {index + 1} has an invalid name",
                code="INVALID_TEST",
            )
        normalized.append(
            {
                "name": name,
                "input": test["input"],
                "expected": test["expected"],
            }
        )
    return normalized


def _summary_for(registry: Registry, name: str) -> CrystalSummary:
    for summary in registry.list():
        if summary.name == name:
            return summary
    # A best-effort catalog write can fail while the directly addressed alias
    # remains durable. Retrying with a known name repairs that narrow case.
    registry.ensure_indexed(name)
    for summary in registry.list():
        if summary.name == name:
            return summary
    raise RegistryNotFoundError(f"crystal {name!r} was not found in the catalog")


class CrystalService:
    """Validated lifecycle operations over a deterministic engine and registry."""

    def __init__(
        self,
        storage: KVStore,
        namespace: str,
        *,
        engine: CrystalFlowEngine | None = None,
    ) -> None:
        self.registry = Registry(storage, namespace=namespace)
        self.engine = engine or CrystalFlowEngine()

    def crystallize(
        self,
        *,
        name: str,
        description: str,
        program: dict[str, Any],
        tests: Any,
        activation_policy: str = "draft",
        estimated_tokens_per_run: int = 0,
    ) -> dict[str, Any]:
        if activation_policy not in {"draft", "activate_after_tests"}:
            raise ServiceValidationError(
                "activation_policy must be draft or activate_after_tests",
                code="INVALID_ACTIVATION_POLICY",
            )
        if not isinstance(description, str) or not description.strip():
            raise ServiceValidationError(
                "description must not be empty",
                code="INVALID_DESCRIPTION",
            )
        estimate = _non_negative_int(
            estimated_tokens_per_run,
            "estimated_tokens_per_run",
            MAX_TOKEN_ESTIMATE,
        )
        cases = _validated_tests(tests)

        self.engine.validate(program)
        for test in cases:
            try:
                actual = self.engine.run(program, test["input"])
                actual_json = canonical_json(actual)
                expected_json = canonical_json(test["expected"])
            except CrystalFlowError as exc:
                raise CrystalTestError(
                    f"test {test['name']!r} could not execute ({exc.code})",
                    code="TEST_EXECUTION_FAILED",
                ) from exc
            if actual_json != expected_json:
                raise CrystalTestError(
                    f"test {test['name']!r} did not match its expected JSON",
                    code="TEST_MISMATCH",
                )

        program_hash = content_hash(program)
        metadata = {
            "engine_version": ENGINE_VERSION,
            "ir_version": IR_VERSION,
            "tests_passed": True,
            "test_count": len(cases),
            "estimated_tokens_per_run": estimate,
        }
        version = self.registry.register(
            name=name,
            program=program,
            program_hash=program_hash,
            tests=cases,
            metadata=metadata,
            description=description.strip(),
        )

        if activation_policy == "activate_after_tests":
            version = self.registry.activate(
                name,
                version.version,
                expected_program_hash=program_hash,
            )

        is_active = version.state is Lifecycle.ACTIVE
        if is_active:
            message = (
                f"Crystal {name!r} version {version.version} is active after "
                f"{len(cases)} passing tests."
            )
        elif version.created:
            message = (
                f"Crystal {name!r} version {version.version} is a tested draft; "
                "review its hash before activation."
            )
        else:
            message = (
                f"Crystal {name!r} already has this program as version "
                f"{version.version} ({version.state.value})."
            )
        return {
            "status": "active" if is_active else "draft",
            "crystal_name": name,
            "version": version.version,
            "program_hash": version.program_hash,
            "tests_passed": True,
            "test_count": len(cases),
            "active": is_active,
            "created": version.created,
            "message": message,
        }

    def execute(
        self,
        *,
        name: str,
        inputs: Any,
        version: int = 0,
    ) -> dict[str, Any]:
        selected_version = _non_negative_int(version, "version", 1_000)
        try:
            crystal = self.registry.get(
                name,
                version=None if selected_version == 0 else selected_version,
            )
        except RegistryNotFoundError:
            return self._missing_execution(name)

        if crystal.state is Lifecycle.RETIRED:
            return self._execution_failure(
                name,
                status="disabled",
                reason_code="VERSION_RETIRED",
                version=crystal.version,
                program_hash=crystal.program_hash,
            )
        if crystal.metadata.get("tests_passed") is not True:
            return self._execution_failure(
                name,
                status="error",
                reason_code="UNTESTED_VERSION",
                version=crystal.version,
                program_hash=crystal.program_hash,
            )
        if crystal.metadata.get("engine_version") != ENGINE_VERSION:
            return self._execution_failure(
                name,
                status="error",
                reason_code="ENGINE_VERSION_MISMATCH",
                version=crystal.version,
                program_hash=crystal.program_hash,
            )

        try:
            result = self.engine.run(crystal.program, inputs)
        except InputValidationError as exc:
            return self._execution_failure(
                name,
                status="invalid_input",
                reason_code=exc.code,
                version=crystal.version,
                program_hash=crystal.program_hash,
            )
        except CrystalFlowError as exc:
            return self._execution_failure(
                name,
                status="error",
                reason_code=exc.code,
                version=crystal.version,
                program_hash=crystal.program_hash,
            )

        result_json = canonical_json(result)
        receipt = content_hash(
            {
                "engine_version": ENGINE_VERSION,
                "program_hash": crystal.program_hash,
                "crystal_name": name,
                "version": crystal.version,
                "input": inputs,
                "output": result,
            }
        )
        estimate = crystal.metadata.get("estimated_tokens_per_run", 0)
        if not isinstance(estimate, int) or isinstance(estimate, bool) or estimate < 0:
            return self._execution_failure(
                name,
                status="error",
                reason_code="INVALID_STORED_METADATA",
                version=crystal.version,
                program_hash=crystal.program_hash,
            )

        telemetry_recorded = True
        try:
            self.registry.record_run(
                name,
                version=crystal.version,
                estimated_tokens_avoided=estimate,
            )
        except RegistryError:
            # Aggregate counters are explicitly best effort. A valid pure
            # result must not be discarded merely because a metric write raced.
            telemetry_recorded = False

        return {
            "status": "hit",
            "fallback_required": False,
            "crystal_name": name,
            "version": crystal.version,
            "program_hash": crystal.program_hash,
            "result": result,
            "result_json": result_json,
            "receipt": receipt,
            "reason_code": "",
            "estimated_tokens_avoided": estimate,
            "telemetry_recorded": telemetry_recorded,
        }

    def _missing_execution(self, name: str) -> dict[str, Any]:
        try:
            latest = self.registry.latest(name)
        except RegistryNotFoundError:
            return self._execution_failure(
                name,
                status="miss",
                reason_code="CRYSTAL_NOT_FOUND",
            )
        status = "disabled" if latest.state is Lifecycle.RETIRED else "miss"
        reason = "NO_ACTIVE_VERSION"
        if latest.state is Lifecycle.RETIRED:
            reason = "NO_ACTIVE_VERSION"
        return self._execution_failure(
            name,
            status=status,
            reason_code=reason,
            version=latest.version,
            program_hash=latest.program_hash,
        )

    @staticmethod
    def _execution_failure(
        name: str,
        *,
        status: str,
        reason_code: str,
        version: int = 0,
        program_hash: str = "",
    ) -> dict[str, Any]:
        return {
            "status": status,
            "fallback_required": True,
            "crystal_name": name,
            "version": version,
            "program_hash": program_hash,
            "result": None,
            "result_json": "",
            "receipt": "",
            "reason_code": reason_code,
            "estimated_tokens_avoided": 0,
            "telemetry_recorded": False,
        }

    def status(
        self,
        *,
        name: str | None = None,
        include_program: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(include_program, bool):
            raise ServiceValidationError(
                "include_program must be a boolean",
                code="INVALID_REQUEST",
            )

        if not name:
            summaries = [summary.to_dict() for summary in self.registry.list()]
            total_runs = sum(item["run_count"] for item in summaries)
            total_avoided = sum(item["estimated_tokens_avoided"] for item in summaries)
            return {
                "status": "ok",
                "crystal_name": "",
                "active_version": 0,
                "crystal_count": len(summaries),
                "details": summaries,
                "details_json": canonical_json(summaries),
                "total_runs": total_runs,
                "estimated_tokens_avoided": total_avoided,
                "message": f"{len(summaries)} crystals are indexed in this namespace.",
            }

        try:
            latest = self.registry.latest(name)
            summary = _summary_for(self.registry, name)
        except RegistryNotFoundError:
            return {
                "status": "not_found",
                "crystal_name": name,
                "active_version": 0,
                "crystal_count": 0,
                "details": {},
                "details_json": "{}",
                "total_runs": 0,
                "estimated_tokens_avoided": 0,
                "message": f"Crystal {name!r} was not found.",
            }

        details = summary.to_dict()
        details["latest_state"] = latest.state.value
        details["latest_program_hash"] = latest.program_hash
        if include_program:
            details["latest_version_record"] = latest.to_dict()
        return {
            "status": "ok",
            "crystal_name": name,
            "active_version": summary.active_version or 0,
            "crystal_count": 1,
            "details": details,
            "details_json": canonical_json(details),
            "total_runs": summary.run_count,
            "estimated_tokens_avoided": summary.estimated_tokens_avoided,
            "message": (
                f"Crystal {name!r} has {summary.version_count} version(s); "
                f"active version is {summary.active_version or 'none'}."
            ),
        }

    def activate(
        self,
        *,
        name: str,
        version: int,
        program_hash: str,
        confirmation: str,
    ) -> dict[str, Any]:
        selected_version = _non_negative_int(version, "version", 1_000)
        if selected_version == 0:
            raise ServiceValidationError(
                "version must be positive",
                code="INVALID_VERSION",
            )
        if confirmation != f"ACTIVATE {name}":
            raise ServiceValidationError(
                "confirmation must exactly match ACTIVATE <crystal_name>",
                code="CONFIRMATION_MISMATCH",
            )
        candidate = self.registry.get(name, selected_version)
        if not isinstance(program_hash, str) or not hmac.compare_digest(
            candidate.program_hash,
            program_hash,
        ):
            raise ServiceValidationError(
                "program hash does not match the selected version",
                code="HASH_MISMATCH",
            )
        self.engine.validate(candidate.program)
        if candidate.metadata.get("engine_version") != ENGINE_VERSION:
            raise ServiceValidationError(
                "candidate was built for a different engine version",
                code="ENGINE_VERSION_MISMATCH",
            )
        active = self.registry.activate(
            name,
            selected_version,
            expected_program_hash=program_hash,
        )
        return {
            "status": "active",
            "crystal_name": name,
            "version": active.version,
            "program_hash": active.program_hash,
            "active_version": active.version,
            "message": (f"Crystal {name!r} version {active.version} is now active."),
        }

    def retire(
        self,
        *,
        name: str,
        confirmation: str,
        version: int = 0,
    ) -> dict[str, Any]:
        selected_version = _non_negative_int(version, "version", 1_000)
        if confirmation != name:
            raise ServiceValidationError(
                "confirmation must exactly match crystal_name",
                code="CONFIRMATION_MISMATCH",
            )
        if selected_version == 0:
            target = self.registry.get(name)
            selected_version = target.version
        retired = self.registry.retire(name, selected_version)
        summary = _summary_for(self.registry, name)
        return {
            "status": "retired",
            "crystal_name": name,
            "version": retired.version,
            "active_version": summary.active_version or 0,
            "message": (
                f"Crystal {name!r} version {retired.version} is retired; "
                "its immutable record is retained."
            ),
        }


__all__ = [
    "CrystalService",
    "CrystalTestError",
    "ENGINE_VERSION",
    "IR_VERSION",
    "ServiceValidationError",
]
