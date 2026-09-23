"""Per-video processing state: resumable, atomic, hash-aware.

``status.json`` is the single source of truth for resume. A stage may be reused
only when its status is ``completed``, its recorded outputs still validate, its
own configuration hash is unchanged and every upstream dependency still points
at the same dependency hash. That is what makes ``--force-stage`` and
"changed the Whisper model" behave differently and correctly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .artifacts import VideoPaths, atomic_write_json, read_json

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

VALID_STATUSES = {STATUS_PENDING, STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED, STATUS_SKIPPED}

STATE_SCHEMA_VERSION = "1.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _duration_seconds(started: str | None, completed: str | None) -> float | None:
    if not started or not completed:
        return None
    try:
        start = datetime.fromisoformat(started.replace("Z", "+00:00"))
        end = datetime.fromisoformat(completed.replace("Z", "+00:00"))
    except ValueError:  # pragma: no cover - defensive against hand-edited state
        return None
    return round((end - start).total_seconds(), 3)


@dataclass
class StageRecord:
    name: str
    status: str = STATUS_PENDING
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    config_hash: str | None = None
    dependency_hash: str | None = None
    input_artifacts: list[str] | None = None
    output_artifacts: list[str] | None = None
    executable: str | None = None
    command: list[str] | None = None
    tool_version: str | None = None
    model_version: str | None = None
    exit_code: int | None = None
    validation_result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    reuse_count: int = 0
    #: Row count of each Parquet output as recorded when the stage completed.
    #: Lets ``validate`` notice an artifact someone truncated or replaced by hand,
    #: which "the file exists" and "the file parses" both miss.
    output_row_counts: dict[str, int] | None = None
    #: Position in this video's monotonic stage-execution sequence. A dependant is
    #: stale when a dependency's sequence moved past the one it was built from,
    #: which catches a rerun whose configuration did not change (--force-stage, a
    #: crash mid-write) — the case configuration hashes cannot see.
    run_sequence: int | None = None

    @property
    def last_config_hash(self) -> str | None:
        """Most recent *executed* config hash; survives a forced rerun."""
        return self.config_hash

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "config_hash": self.config_hash,
            "dependency_hash": self.dependency_hash,
            "input_artifacts": self.input_artifacts,
            "output_artifacts": self.output_artifacts,
            "executable": self.executable,
            "command": self.command,
            "tool_version": self.tool_version,
            "model_version": self.model_version,
            "exit_code": self.exit_code,
            "validation_result": self.validation_result,
            "error": self.error,
            "reuse_count": self.reuse_count,
            "run_sequence": self.run_sequence,
            "output_row_counts": self.output_row_counts,
        }
        return payload

    @classmethod
    def from_dict(cls, name: str, payload: dict[str, Any]) -> "StageRecord":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(name=name, **{k: v for k, v in payload.items() if k in known and k != "name"})


class VideoState:
    """Mutable view over one video's ``status.json`` with atomic persistence."""

    def __init__(self, paths: VideoPaths, video_id: str, source_path: str | None = None) -> None:
        self.paths = paths
        self.video_id = video_id
        self.source_path = source_path
        self.stages: dict[str, StageRecord] = {}
        self.stage_order: list[str] = []
        self.created_at: str | None = None
        self.updated_at: str | None = None
        #: Highest stage-execution sequence recorded so far for this video.
        self.sequence: int = 0

    # ---------------------------------------------------------------- loading

    @classmethod
    def load(cls, paths: VideoPaths, video_id: str, source_path: str | None = None) -> "VideoState":
        state = cls(paths, video_id, source_path)
        if paths.status.is_file():
            payload = read_json(paths.status)
            state.created_at = payload.get("created_at")
            state.updated_at = payload.get("updated_at")
            state.sequence = int(payload.get("sequence") or 0)
            state.source_path = payload.get("source", {}).get("path", source_path)
            for name, record in (payload.get("stages") or {}).items():
                state.stages[name] = StageRecord.from_dict(name, record or {})
            state.stage_order = list(payload.get("stage_order") or state.stages.keys())
        # Resume-safe: never restart the counter below a sequence already recorded.
        state.sequence = max([state.sequence] + [record.run_sequence or 0 for record in state.stages.values()])
        return state

    def next_sequence(self) -> int:
        """Allocate the next execution sequence number for this video."""
        self.sequence += 1
        return self.sequence

    def bind_stages(self, order: Iterable[str]) -> None:
        """Declare the canonical stage list, preserving any recorded history."""
        order = list(order)
        self.stage_order = order
        for name in order:
            self.stages.setdefault(name, StageRecord(name=name))
        # Keep history for stages that vanished from the pipeline definition.
        for name in self.stages:
            if name not in order:
                self.stage_order.append(name)

    # ------------------------------------------------------------------ access

    def stage(self, name: str) -> StageRecord:
        if name not in self.stages:
            self.stages[name] = StageRecord(name=name)
        return self.stages[name]

    def status_of(self, name: str) -> str:
        return self.stage(name).status

    def is_completed(self, name: str) -> bool:
        return self.status_of(name) == STATUS_COMPLETED

    @property
    def overall_status(self) -> str:
        """completed | partial | failed | running | pending for the batch report.

        ``failed`` means *nothing usable was produced*: a video whose first stage
        failed and whose dependants were therefore blocked is not a partial result
        someone can consume, it is a failure. Reporting it as partial would hide a
        broken file in a folder full of ``partial`` entries.
        """
        statuses = [self.stage(name).status for name in self.stage_order if name in self.stages]
        if not statuses:
            return STATUS_PENDING
        if STATUS_RUNNING in statuses:
            return STATUS_RUNNING
        usable = STATUS_COMPLETED in statuses
        if STATUS_FAILED in statuses:
            return "partial" if usable else STATUS_FAILED
        settled = [status for status in statuses if status in {STATUS_COMPLETED, STATUS_SKIPPED}]
        if len(settled) == len(statuses):
            return STATUS_COMPLETED
        return "partial" if usable else STATUS_PENDING

    # ---------------------------------------------------------------- mutation

    def mark_running(self, name: str) -> StageRecord:
        record = self.stage(name)
        record.status = STATUS_RUNNING
        record.started_at = utc_now()
        record.completed_at = None
        record.error = None
        record.exit_code = None
        self.save()
        return record

    def mark_completed(
        self,
        name: str,
        *,
        config_hash: str | None = None,
        dependency_hash: str | None = None,
        input_artifacts: Iterable[str] | None = None,
        output_artifacts: Iterable[str] | None = None,
        executable: str | None = None,
        command: list[str] | None = None,
        tool_version: str | None = None,
        model_version: str | None = None,
        exit_code: int | None = None,
        validation_result: dict[str, Any] | None = None,
        run_sequence: int | None = None,
        output_row_counts: dict[str, int] | None = None,
    ) -> StageRecord:
        record = self.stage(name)
        record.status = STATUS_COMPLETED
        record.started_at = record.started_at or utc_now()
        record.completed_at = utc_now()
        record.duration_seconds = _duration_seconds(record.started_at, record.completed_at)
        record.config_hash = config_hash
        record.dependency_hash = dependency_hash
        record.input_artifacts = list(input_artifacts or [])
        record.output_artifacts = list(output_artifacts or [])
        record.executable = executable
        record.command = command
        record.tool_version = tool_version
        record.model_version = model_version
        record.exit_code = exit_code
        record.validation_result = validation_result
        record.run_sequence = run_sequence
        record.error = None
        self.save()
        return record

    def mark_failed(self, name: str, error: dict[str, Any], *, exit_code: int | None = None,
                    command: list[str] | None = None, executable: str | None = None) -> StageRecord:
        record = self.stage(name)
        record.status = STATUS_FAILED
        record.completed_at = utc_now()
        record.duration_seconds = _duration_seconds(record.started_at, record.completed_at)
        record.error = error
        record.exit_code = exit_code
        if command is not None:
            record.command = command
        if executable is not None:
            record.executable = executable
        self.save()
        return record

    def mark_skipped(self, name: str, reason: str) -> StageRecord:
        record = self.stage(name)
        record.status = STATUS_SKIPPED
        record.started_at = None
        record.completed_at = utc_now()
        record.duration_seconds = None
        record.error = None
        record.validation_result = {"skipped": True, "reason": reason}
        self.save()
        return record

    def mark_reused(self, name: str) -> StageRecord:
        record = self.stage(name)
        record.reuse_count += 1
        self.save()
        return record

    def reset_stage(self, name: str) -> StageRecord:
        record = self.stage(name)
        record.status = STATUS_PENDING
        record.started_at = None
        record.completed_at = None
        record.duration_seconds = None
        record.error = None
        self.save()
        return record

    # --------------------------------------------------------------- summary

    def summary(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "status": self.overall_status,
            "stages": {name: self.stage(name).status for name in self.stage_order},
            "failed_stages": [
                name for name in self.stage_order if self.stage(name).status == STATUS_FAILED
            ],
            "duration_seconds": round(
                sum(self.stage(n).duration_seconds or 0.0 for n in self.stage_order), 3
            ),
        }

    # ---------------------------------------------------------------- persist

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "video_id": self.video_id,
            "source": {"path": self.source_path, "filename": Path(self.source_path).name if self.source_path else None},
            "created_at": self.created_at or utc_now(),
            "updated_at": utc_now(),
            "overall_status": self.overall_status,
            "sequence": self.sequence,
            "stage_order": self.stage_order,
            "stages": {name: self.stages[name].to_dict() for name in self.stage_order if name in self.stages},
        }

    def save(self) -> Path:
        self.updated_at = utc_now()
        if self.created_at is None:
            self.created_at = self.updated_at
        return atomic_write_json(self.paths.status, self.to_dict())
