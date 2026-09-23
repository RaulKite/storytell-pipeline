"""Finalization: cross-modal consistency, manifest, provenance.

Every other stage validates its own output; this stage is the only place that
can check the dataset *as a whole* — that tables agree with each other, that the
single timeline really is single, and that the manifest describes what exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..artifacts import STAGE_LOG_NAMES, ArtifactRegistry, read_json
from ..exceptions import ValidationError
from ..manifest import build_manifest, validate_manifest, write_manifest
from ..provenance import git_commit, processing_report, tools_report, write_provenance
from ..schemas import read_table
from .base import STAGE_ORDER, Stage, StageContext

#: Tables whose timestamps must live inside the media duration and start at t>=0.
TIMED_TABLES = (
    ("speech_segments", "start_time"),
    ("speech_words", "start_time"),
    ("speaker_turns", "start_time"),
    ("translation_segments", "start_time"),
    ("acoustic_frames", "timestamp"),
    ("acoustic_segments", "start_time"),
    ("pose_body", "timestamp"),
    ("pose_hands", "timestamp"),
    ("pose_face", "timestamp"),
)


class FinalizationStage(Stage):
    name = "finalization"
    inputs = ("metadata",)
    outputs = ("manifest", "provenance_config", "provenance_tools", "provenance_processing")
    config_keys = ()

    def config_fingerprint(self, ctx: StageContext) -> dict[str, Any]:
        from ..config import configuration_hash

        # The fingerprint is the *content* of every stage result: if any upstream
        # artifact changes, the manifest and provenance must be regenerated.
        registry = ArtifactRegistry(ctx.paths).refresh()
        return {
            "stage": self.name,
            "configuration_hash": configuration_hash(ctx.config),
            "pipeline_version": git_commit(ctx.config.project_root),
            "artifacts": {name: info.get("size_bytes") for name, info in sorted(registry.present.items())},
            "stage_hashes": {name: ctx.state.stage(name).config_hash
                             for name in STAGE_ORDER if name != self.name},
        }

    def prepare(self, ctx: StageContext) -> None:
        ctx.paths.ensure_dirs()

    def execute(self, ctx: StageContext) -> dict[str, Any]:
        summary = write_dataset_summary(ctx)
        ctx.log(f"manifest written with {summary['artifacts']} artifacts "
                f"(status={summary['status']})")
        return {"tool_version": None, "model_version": None,
                "extra": {"artifacts": summary["artifacts"],
                          "detected_language": summary["detected_language"]}}


    # ---------------------------------------------------------------- validation

    def validate(self, ctx: StageContext) -> dict[str, Any]:
        issues: list[str] = []
        metadata_path = ctx.artifact("metadata")
        if not metadata_path.is_file():
            raise ValidationError(self.name, ["source/metadata.json missing"])
        metadata = read_json(metadata_path)
        duration = metadata.get("duration_seconds")

        manifest = read_json(ctx.paths.manifest) if ctx.paths.manifest.is_file() else {}
        if not manifest:
            raise ValidationError(self.name, ["manifest.json missing"])
        try:
            validate_manifest(ctx.paths, manifest)
        except ValidationError as exc:
            issues.extend(exc.issues)

        # Provenance files must be readable JSON: they are the reproducibility record.
        for name in ("provenance_config", "provenance_tools", "provenance_processing"):
            path = ctx.artifact(name)
            if not path.is_file():
                issues.append(f"provenance file missing: {path.name}")
                continue
            try:
                read_json(path)
            except (OSError, ValueError) as exc:
                issues.append(f"{path.name} unreadable: {exc}")

        # Cross-modal timeline consistency: every timestamp within the media.
        checked = 0
        for artifact_name, column in TIMED_TABLES:
            path = ctx.artifact(artifact_name)
            if not path.is_file():
                continue
            try:
                values = [row[column] for row in read_table(path, columns=[column]).to_pylist()]
            except Exception as exc:  # noqa: BLE001 - reported as a validation issue
                issues.append(f"{path.name} unreadable: {exc}")
                continue
            numeric = [value for value in values if isinstance(value, (int, float))]
            checked += 1
            if numeric and duration:
                if min(numeric) < -1e-3:
                    issues.append(f"{path.name} has timestamps before t=0")
                if max(numeric) > float(duration) + 1.0:
                    issues.append(f"{path.name} exceeds the media duration "
                                  f"({max(numeric):.2f}s > {float(duration):.2f}s)")

        issues.extend(self._cross_references(ctx))
        if issues:
            raise ValidationError(self.name, issues)
        return {"artifacts": len(manifest.get("artifacts", {})), "timed_tables_checked": checked,
                "status": ctx.state.overall_status}

    @staticmethod
    def _cross_references(ctx: StageContext) -> list[str]:
        """Identifiers must resolve across modalities, not just inside a table."""
        issues: list[str] = []
        if not (ctx.artifact("speech_segments").is_file() and ctx.artifact("translation_segments").is_file()):
            return issues
        segments = {row["segment_id"] for row in read_table(ctx.artifact("speech_segments"),
                                                          columns=["segment_id"]).to_pylist()}
        try:
            translated = {row["segment_id"] for row in read_table(ctx.artifact("translation_segments"),
                                                                 columns=["segment_id"]).to_pylist()}
        except Exception as exc:  # noqa: BLE001
            return [f"segments_en.parquet unreadable: {exc}"]
        missing = translated - segments
        if missing:
            issues.append(f"translation rows with no source segment: {sorted(missing)[:5]}")
        speakers: set[str] = set()
        if ctx.artifact("speaker_turns").is_file():
            speakers = {row["speaker_id"] for row in read_table(ctx.artifact("speaker_turns"),
                                                               columns=["speaker_id"]).to_pylist()}
            speakers.discard(None)
        if speakers:
            segment_speakers = {row["speaker_id"] for row in read_table(ctx.artifact("speech_segments")).to_pylist()
                                if row.get("speaker_id")}
            unknown = segment_speakers - speakers
            if unknown:
                issues.append(f"transcript speakers absent from diarization: {sorted(unknown)[:5]}")
        return issues

# ---------------------------------------------------------------------- summary


def write_dataset_summary(ctx: StageContext) -> dict[str, Any]:
    """Write ``manifest.json`` and the provenance files from current state.

    Called twice by design: once from the finalization stage, and once when the
    runner finishes the video. A manifest can only be written while a stage is
    running, so on its own it would record finalization as ``running`` and the
    overall status as whatever preceded it — a dataset that permanently misreports
    itself. The closing rewrite is what makes the on-disk summary true.
    """
    metadata_path = ctx.artifact("metadata")
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    registry = ArtifactRegistry(ctx.paths).refresh()
    stage_statuses = {name: ctx.state.stage(name).status for name in STAGE_ORDER}
    detected_language = _detected_language(ctx, registry)
    manifest = build_manifest(
        video_id=ctx.video_id,
        metadata=metadata,
        overall_status=ctx.state.overall_status,
        registry=registry,
        stage_statuses=stage_statuses,
        detected_language=detected_language,
    )
    validate_manifest(ctx.paths, manifest)
    write_manifest(ctx.paths, manifest)

    stage_records = {name: ctx.state.stage(name).to_dict() for name in STAGE_ORDER}
    processing = processing_report(
        video_id=ctx.video_id,
        config=ctx.config,
        state_summary=ctx.state.summary(),
        stage_records=stage_records,
        artifacts=registry.describe(),
        started_at=ctx.tools.get("run_started_at"),
    )
    write_provenance(ctx.paths, ctx.config, tools_report(ctx.config), processing)
    return {
        "artifacts": len(manifest["artifacts"]),
        "status": ctx.state.overall_status,
        "detected_language": detected_language,
    }


def _detected_language(ctx: StageContext, registry: ArtifactRegistry) -> str | None:
    from ..schemas import iter_rows

    if registry.has("speech_segments"):
        for row in iter_rows(ctx.artifact("speech_segments"), ["language"]):
            if row.get("language"):
                return str(row["language"])
    return None
