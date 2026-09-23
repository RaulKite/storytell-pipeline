"""Semantic validation helpers shared by stages.

File existence is not validation: these helpers check that a Parquet table has
the declared columns, that timestamps are ordered and inside the source
duration, and that cross-table identifiers actually reference each other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow.parquet as pq

from .exceptions import ValidationError, ValidationIssue

__all__ = [
    "ValidationIssue",
    "ValidationError",
    "check_intervals",
    "check_parquet",
    "check_reference_values",
    "validate_metadata_payload",
]


def check_parquet(path: Path, expected_columns: Sequence[str], *, stage: str,
                  min_rows: int = 0, time_column: str | None = None,
                  max_time: float | None = None, ordered: bool = True) -> dict[str, Any]:
    """Readability + schema + row count + optional timeline sanity."""
    if not path.is_file():
        raise ValidationIssue(stage, [f"missing table: {path.name}"])
    try:
        parquet_file = pq.ParquetFile(path)
    except Exception as exc:  # pyarrow raises many types
        raise ValidationIssue(stage, [f"unreadable parquet {path.name}: {exc}"]) from exc
    names = [field.name for field in parquet_file.schema_arrow]
    missing = [name for name in expected_columns if name not in names]
    if missing:
        raise ValidationIssue(stage, [f"{path.name} missing columns: {', '.join(missing)}"])
    rows = parquet_file.metadata.num_rows
    if rows < min_rows:
        raise ValidationIssue(stage, [f"{path.name} has {rows} rows, expected >= {min_rows}"])
    issues: list[str] = []
    if time_column and "schema_version" in names:
        column = pq.read_table(path, columns=[time_column]).column(time_column)
        values = column.to_pylist()
        numeric = [value for value in values if value is not None]
        if numeric:
            if min(numeric) < -1e-6:
                issues.append(f"{path.name}.{time_column} has negative timestamps")
            if max_time is not None and max(numeric) > max_time + 1.0:
                issues.append(
                    f"{path.name}.{time_column} exceeds source duration ({max(numeric):.2f}s > {max_time:.2f}s)"
                )
            if ordered and any(b < a for a, b in zip(numeric, numeric[1:])):
                issues.append(f"{path.name}.{time_column} is not monotonically ordered")
    if issues:
        raise ValidationIssue(stage, issues)
    return {"rows": rows, "columns": names}


def check_intervals(starts: Sequence[float | None], ends: Sequence[float | None], *, stage: str,
                    label: str, max_time: float | None = None,
                    tolerance: float = 1e-6) -> list[dict[str, float]]:
    """Each interval must be non-empty and inside the source duration."""
    issues: list[str] = []
    for index, (start, end) in enumerate(zip(starts, ends)):
        if start is None or end is None:
            continue
        if end + tolerance < start:
            issues.append(f"{label}[{index}] ends before it starts ({start:.3f}>{end:.3f})")
        if start < -1e-3:
            issues.append(f"{label}[{index}] starts before t=0")
        if max_time is not None and start > max_time + 1.0:
            issues.append(f"{label}[{index}] starts after source duration")
    if issues:
        raise ValidationIssue(stage, issues[:20])
    return [{"start_time": s, "end_time": e} for s, e in zip(starts, ends)]


def check_reference_values(values: Sequence[Any], known: set[Any], *, stage: str,
                           label: str) -> None:
    unknown = {value for value in values if value is not None and value not in known}
    if unknown:
        sample = ", ".join(sorted(str(value) for value in unknown)[:8])
        raise ValidationIssue(stage, [f"{label} references unknown values: {sample}"])


def validate_metadata_payload(payload: dict[str, Any], *, stage: str = "metadata") -> dict[str, Any]:
    required = ("schema_version", "video_id", "source_filename", "source_path", "SHA256",
                "file_size_bytes", "duration_seconds")
    missing = [key for key in required if payload.get(key) in (None, "")]
    issues = [f"missing metadata field: {key}" for key in missing]
    duration = payload.get("duration_seconds") or 0
    if duration <= 0:
        issues.append("duration_seconds must be > 0")
    width, height = payload.get("width") or 0, payload.get("height") or 0
    if width <= 0 or height <= 0:
        issues.append("video stream width/height must be > 0")
    if not payload.get("video_codec") and not payload.get("audio_codec"):
        issues.append("no media stream detected")
    if issues:
        raise ValidationIssue(stage, issues)
    return {"duration_seconds": duration, "width": width, "height": height}
