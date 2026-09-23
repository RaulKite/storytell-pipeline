"""Batch report: the machine-readable answer to "what happened to my folder?".

Written to ``<output>/batch_report.json`` after every batch. Statuses are
per-video aggregates of stage statuses — ``completed`` when every stage settled,
``partial`` when something succeeded but a stage failed or was skipped mid-way,
``failed`` when nothing usable was produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .artifacts import atomic_write_json
from .exceptions import ValidationError

BATCH_SCHEMA_VERSION = "1.0"


@dataclass
class BatchReport:
    videos_discovered: int
    completed: int
    partial: int
    failed: int
    elapsed_seconds: float
    entries: list[dict[str, Any]]
    output_directory: str
    path: str = ""
    schema_version: str = BATCH_SCHEMA_VERSION
    pipeline_version: str = __version__
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pipeline_version": self.pipeline_version,
            "generated_at": self.generated_at,
            "output_directory": self.output_directory,
            "summary": {
                "videos_discovered": self.videos_discovered,
                "completed": self.completed,
                "partial": self.partial,
                "failed": self.failed,
                "processed_in_this_run": len(self.entries),
                "elapsed_seconds": round(self.elapsed_seconds, 2),
            },
            "videos": self.entries,
        }


def collect_report(results: Sequence[Any], *, discovered: int, elapsed_seconds: float,
                   output_directory: Path | str) -> BatchReport:
    entries = [result.to_dict() for result in results]
    completed = sum(1 for entry in entries if entry["status"] == "completed")
    failed = sum(1 for entry in entries if entry["status"] == "failed")
    partial = sum(1 for entry in entries if entry["status"] not in {"completed", "failed"})
    return BatchReport(
        videos_discovered=discovered,
        completed=completed,
        partial=partial,
        failed=failed,
        elapsed_seconds=elapsed_seconds,
        entries=entries,
        output_directory=str(output_directory),
    )


def write_batch_report(report: BatchReport) -> Path:
    destination = Path(report.output_directory) / "batch_report.json"
    path = atomic_write_json(destination, report.to_dict())
    report.path = str(path)
    return path


def validate_batch_report(path: Path) -> dict[str, Any]:
    """Used by integration tests: the report must be self-consistent."""
    import json

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("batch_report", [f"unreadable: {exc}"]) from exc
    summary = payload.get("summary") or {}
    videos = payload.get("videos") or []
    issues: list[str] = []
    for key in ("videos_discovered", "completed", "partial", "failed"):
        if key not in summary:
            issues.append(f"summary.{key} missing")
    if summary.get("completed", 0) + summary.get("partial", 0) + summary.get("failed", 0) != len(videos):
        issues.append("summary counts do not match the videos list")
    for entry in videos:
        for key in ("video_id", "source_path", "dataset_dir", "status", "stages"):
            if key not in entry:
                issues.append(f"{entry.get('video_id', '?')}: {key} missing")
        if entry.get("status") not in {"completed", "partial", "failed"}:
            issues.append(f"{entry.get('video_id', '?')}: impossible status {entry.get('status')}")
    if issues:
        raise ValidationError("batch_report", issues[:20])
    return {"videos": len(videos), **summary}
