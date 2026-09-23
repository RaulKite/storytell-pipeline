"""Per-video manifest: the single programmatic entry point to a dataset.

Generated from the validated artifact registry rather than a hardcoded list, so
it can only ever describe files that actually exist. Absent optional artifacts
are reported as absent instead of being silently promised.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import MANIFEST_ARTIFACTS, ArtifactRegistry, VideoPaths, atomic_write_json, read_json
from .config import mask_secrets
from .exceptions import ValidationError

MANIFEST_SCHEMA_VERSION = "1.0"


def build_manifest(*, video_id: str, metadata: dict[str, Any], overall_status: str,
                   registry: ArtifactRegistry, stage_statuses: dict[str, str],
                   detected_language: str | None = None,
                   schema_version: str = MANIFEST_SCHEMA_VERSION) -> dict[str, Any]:
    present = registry.describe()
    artifacts: dict[str, str] = {}
    optional_missing: dict[str, str] = {}
    for name in MANIFEST_ARTIFACTS:
        info = present.get(name)
        if info:
            artifacts[name] = info["path"]
        else:
            optional_missing[name] = "not_generated"
    return {
        "schema_version": schema_version,
        "video_id": video_id,
        "source": {
            "filename": metadata.get("source_filename"),
            "path": metadata.get("source_path"),
            "sha256": metadata.get("SHA256"),
            "file_size_bytes": metadata.get("file_size_bytes"),
            "container": metadata.get("container"),
            "duration_seconds": metadata.get("duration_seconds"),
            "fps": metadata.get("average_frame_rate_float"),
            "fps_rational": metadata.get("average_frame_rate_rational"),
            "frame_rate_rational": metadata.get("frame_rate_rational"),
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "pixel_format": metadata.get("pixel_format"),
            "video_codec": metadata.get("video_codec"),
            "audio_codec": metadata.get("audio_codec"),
            "audio_sample_rate": metadata.get("audio_sample_rate"),
            "audio_channels": metadata.get("audio_channels"),
            "frame_count": metadata.get("frame_count"),
            "detected_language": detected_language,
        },
        "processing": {
            "status": overall_status,
            "stages": dict(stage_statuses),
        },
        "temporal_model": {
            "unit": "seconds_from_video_start",
            "interval_columns": ["start_time", "end_time"],
            "instant_columns": ["timestamp"],
            "frame_columns": ["frame_number"],
        },
        "artifacts": artifacts,
        "artifacts_not_generated": optional_missing,
        "artifact_details": {
            name: {key: value for key, value in info.items() if key != "path"}
            for name, info in sorted(present.items())
        },
    }


def write_manifest(paths: VideoPaths, manifest: dict[str, Any]) -> Path:
    return atomic_write_json(paths.manifest, manifest)


def validate_manifest(paths: VideoPaths, manifest: dict[str, Any]) -> dict[str, Any]:
    """Every promised artifact must exist, and paths must stay inside the dataset."""
    issues: list[str] = []
    if not manifest.get("schema_version"):
        issues.append("manifest has no schema_version")
    if not manifest.get("video_id"):
        issues.append("manifest has no video_id")
    artifacts = manifest.get("artifacts") or {}
    if not artifacts:
        issues.append("manifest lists no artifacts")
    for name, relative in artifacts.items():
        candidate = Path(relative)
        # Absolute paths or traversal out of the dataset directory are hard errors:
        # a manifest is a data contract consumed by other software.
        if candidate.is_absolute() or ".." in candidate.parts:
            issues.append(f"artifact {name} has an unsafe relative path: {relative}")
            continue
        resolved = (paths.dataset_dir / candidate).resolve()
        try:
            inside = resolved.is_relative_to(paths.dataset_dir.resolve())
        except AttributeError:  # pragma: no cover - Python < 3.9
            inside = str(resolved).startswith(str(paths.dataset_dir.resolve()))
        if not inside:
            issues.append(f"artifact {name} escapes the dataset directory: {relative}")
            continue
        if not resolved.exists():
            issues.append(f"artifact {name} is listed but missing: {relative}")
    if issues:
        raise ValidationError("manifest", issues)
    return {"artifacts": len(artifacts), "all_present": True}


def read_manifest(paths: VideoPaths) -> dict[str, Any]:
    return read_json(paths.manifest)


def manifest_for_status(paths: VideoPaths) -> dict[str, Any]:
    """Fallback manifest used when finalization must report a broken run."""
    if not paths.manifest.is_file():
        return {}
    try:
        return mask_secrets(read_manifest(paths))
    except (OSError, ValueError):
        return {}
