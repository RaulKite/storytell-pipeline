"""Finalization: cross-modal consistency, manifest, provenance.

Every other stage validates its own output; this stage is the only place that
can check the dataset *as a whole* — that tables agree with each other, that the
single timeline really is single, and that the manifest describes what exists.
"""

from __future__ import annotations

from typing import Any

from ..artifacts import ArtifactRegistry, read_json
from ..exceptions import ValidationError
from ..manifest import build_manifest, validate_manifest, write_manifest
from ..provenance import git_commit, processing_report, tools_report, write_provenance
from ..schemas import read_table
from .base import STAGE_ORDER, Stage, StageContext

#: Tables whose timestamps must live inside the media duration and start at t>=0.
TIMED_TABLES = (
    ("speech_segments", "start_time"),
    ("speech_words", "start_time"),
    ("speaker_turns_nemotron", "start_time"),
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

        # Only things finalization does not itself write. An earlier version folded
        # in each artifact's size_bytes, but finalization writes the manifest and
        # the provenance files, so writing them changed the fingerprint it was
        # about to be compared against: every subsequent run called finalization
        # stale and rewrote the whole summary, forever.
        #
        # Content changes are already covered by the dependency hash, which tracks
        # every upstream stage's configuration *and* execution sequence. A hand-edited
        # artifact makes that stage fail its own integrity check, the stage reruns,
        # its sequence moves, and finalization follows.
        return {
            "stage": self.name,
            "configuration_hash": configuration_hash(ctx.config),
            "pipeline_version": git_commit(ctx.config.project_root),
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
        stale = stale_artifacts(ctx)
        if stale:
            # Reported, never raised: the dataset is usable, one table in it is not current.
            # "Validate exits 0 while two tables disagree" was the defect; "validate fails a
            # corpus because the translation endpoint was unset" would be a worse one.
            for name, reason in stale.items():
                ctx.log(f"stale artifact {name}: {reason}", level=30)
        return {"artifacts": len(manifest.get("artifacts", {})), "timed_tables_checked": checked,
                "status": ctx.state.overall_status, "stale_artifacts": stale}

    @staticmethod
    def _cross_references(ctx: StageContext) -> list[str]:
        """Identifiers must resolve across modalities, not just inside a table.

        Each check runs independently of the others. An early version returned as
        soon as the translation table was absent, which silently disabled the
        speaker cross-check for every dataset without translation — the common case
        whenever the translation endpoint is unconfigured.
        """
        issues: list[str] = []
        if ctx.artifact("speech_segments").is_file():
            issues.extend(_translation_references(ctx))
            issues.extend(_speaker_references(ctx))
        return issues


def _translation_references(ctx: StageContext) -> list[str]:
    if not ctx.artifact("translation_segments").is_file():
        return []
    segments = {row["segment_id"] for row in read_table(ctx.artifact("speech_segments"),
                                                        columns=["segment_id"]).to_pylist()}
    try:
        translated = {row["segment_id"] for row in read_table(ctx.artifact("translation_segments"),
                                                             columns=["segment_id"]).to_pylist()}
    except Exception as exc:  # noqa: BLE001 - reported as a validation issue
        return [f"segments_en.parquet unreadable: {exc}"]
    missing = translated - segments
    if missing:
        return [f"translation rows with no source segment: {sorted(missing)[:5]}"]
    return []


def stale_artifacts(ctx: StageContext) -> dict[str, str]:
    """Artifacts on disk that no longer agree with the tables they were derived from.

    The check that matters today is the translation table. Its rows carry the speaker label
    of the source segment they translated, so when ``speaker_assignment`` labels the
    transcript afterwards — or when a translation was written against a pyannote-labelled
    transcript and the diarization has since changed — the English table keeps describing a
    speaker the dataset no longer has. The joined tables then disagree segment by segment
    while every individual table validates, and the manifest promises both.

    Returning a reason rather than raising keeps this a statement about one artifact: the
    translation is excluded from the manifest and reported by ``validate``, its file is
    preserved, and re-running the translation stage is the fix.
    """
    reason = _translation_speaker_mismatch(ctx)
    return {"translation_segments": reason} if reason else {}


def _translation_speaker_mismatch(ctx: StageContext) -> str | None:
    """Why the translated table's speaker labels contradict the transcript, if they do."""
    if not (ctx.artifact("translation_segments").is_file()
            and ctx.artifact("speech_segments").is_file()):
        return None
    try:
        source = {row["segment_id"]: row.get("speaker_id")
                  for row in read_table(ctx.artifact("speech_segments")).to_pylist()}
        translated = [(row["segment_id"], row.get("speaker_id"))
                      for row in read_table(ctx.artifact("translation_segments")).to_pylist()]
    except Exception:  # noqa: BLE001 - an unreadable table is reported by its own stage
        return None
    mismatched = [segment_id for segment_id, speaker in translated
                  if segment_id in source and source[segment_id] != speaker]
    if not mismatched:
        return None
    return (f"speaker labels disagree with the transcript for {len(mismatched)} segment(s), "
            f"e.g. {sorted(mismatched)[:3]}; re-run the translation stage")


def _speaker_references(ctx: StageContext) -> list[str]:
    """Every speaker the transcript cites must exist in the diarization output.

    With no diarization table the check used to return quietly, which is how a transcript
    still carrying pyannote labels passed finalization after the Nemotron stage was
    skipped: there was nothing to compare against and therefore no complaint. Absence of a
    timeline is now itself the finding — ``speaker_assignment`` clears those labels, so a
    label with no table behind it means that stage did not run over this transcript.
    """
    if not ctx.artifact("speaker_turns_nemotron").is_file():
        try:
            cited = {row["speaker_id"] for row in read_table(ctx.artifact("speech_segments")).to_pylist()
                     if row.get("speaker_id")}
        except Exception as exc:  # noqa: BLE001
            return [f"segments.parquet unreadable: {exc}"]
        if cited:
            return [f"transcript cites {sorted(cited)[:5]} with no diarization timeline "
                    f"(speaker_turns_nemotron.parquet is absent): re-run speaker_assignment"]
        return []
    try:
        speakers = {row["speaker_id"] for row in read_table(ctx.artifact("speaker_turns_nemotron"),
                                                            columns=["speaker_id"]).to_pylist()}
    except Exception as exc:  # noqa: BLE001
        return [f"speaker_turns_nemotron.parquet unreadable: {exc}"]
    speakers.discard(None)
    if not speakers:
        return []
    try:
        cited = {row["speaker_id"] for row in read_table(ctx.artifact("speech_segments")).to_pylist()
                 if row.get("speaker_id")}
    except Exception as exc:  # noqa: BLE001
        return [f"segments.parquet unreadable: {exc}"]
    unknown = cited - speakers
    if unknown:
        return [f"transcript speakers absent from diarization: {sorted(unknown)[:5]}"]
    return []

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
    stale = stale_artifacts(ctx)
    manifest = build_manifest(
        video_id=ctx.video_id,
        metadata=metadata,
        overall_status=ctx.state.overall_status,
        registry=registry,
        stage_statuses=stage_statuses,
        detected_language=detected_language,
        stale=stale,
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
        "stale_artifacts": stale,
    }


def _detected_language(ctx: StageContext, registry: ArtifactRegistry) -> str | None:
    from ..schemas import iter_rows

    if registry.has("speech_segments"):
        for row in iter_rows(ctx.artifact("speech_segments"), ["language"]):
            if row.get("language"):
                return str(row["language"])
    return None
