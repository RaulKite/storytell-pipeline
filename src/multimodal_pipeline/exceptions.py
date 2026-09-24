"""Pipeline exception hierarchy (leaf module: importable from anywhere).

``StageError`` marks a stage that failed; ``ValidationError`` marks output that
exists but is semantically wrong. The orchestrator records both as ``failed``,
while the reuse test treats ``ValidationError`` as "not reusable".
"""

from __future__ import annotations

from typing import Any, Sequence


class ConfigError(RuntimeError):
    """The configuration itself is unusable (bad YAML, bad key, bad ``.env`` line).

    Distinct from :class:`StageError`: nothing ran yet, so there is no stage to mark
    failed and nothing to resume from. The CLI turns it into one readable line and a
    usage exit code rather than a traceback.
    """

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class StageError(RuntimeError):
    """A stage failed in a way that should be recorded and reported."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class ValidationError(StageError):
    """Output was produced but did not pass semantic validation.

    Usable in both styles used across the codebase::

        ValidationError("segment 3 ends before it starts")
        ValidationError("whisperx", ["missing column x", "unordered rows"])
    """

    def __init__(self, message_or_stage: str, issues: Sequence[str] | None = None,
                 *, details: dict[str, Any] | None = None) -> None:
        merged: dict[str, Any] = dict(details or {})
        if issues is None:
            message = str(message_or_stage)
            merged.setdefault("stage", "")
            merged.setdefault("issues", [])
        else:
            problems = [str(issue) for issue in issues]
            stage = str(message_or_stage)
            message = f"{stage}: " + "; ".join(problems[:12]) or stage
            merged["stage"] = stage
            merged["issues"] = problems
        super().__init__(message, details=merged)
        self.stage = merged["stage"]
        self.issues = merged["issues"]


# Historical name used by the validation helpers.
ValidationIssue = ValidationError


class WorkerError(RuntimeError):
    """An isolated uv worker reported failure or produced no usable result."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}
