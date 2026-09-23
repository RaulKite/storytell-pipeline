"""Speaker-assignment stage: normalise transcript rows onto diarization turns.

Deliberately a distinct stage: raw WhisperX and raw Pyannote outputs stay
untouched, so this normalisation can be recomputed (with different rules or a
better timeline) without re-running either model.
"""

from __future__ import annotations

import json
from typing import Any

import pyarrow as pa

from ..exceptions import StageError
from ..schemas import SEGMENTS_SCHEMA, SPEAKER_TURNS_SCHEMA, WORDS_SCHEMA, read_table, write_table
from ..speaker_assignment import assign_speaker, coverage_report, normalise_turns
from ..state import STATUS_COMPLETED, STATUS_FAILED, STATUS_SKIPPED  # noqa: F401
from ..validation import check_reference_values
from .base import Stage, StageContext, ValidationError


class SpeakerAssignmentStage(Stage):
    name = "speaker_assignment"
    inputs = ("speech_segments", "speech_words", "speaker_turns")
    outputs = ("speech_segments", "speech_words")
    config_keys = ("diarization",)

    def config_fingerprint(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.config.diarization
        return {
            "stage": self.name,
            "assignment_method": "max_overlap",
            "prefer_exclusive": cfg.use_exclusive_diarization_for_alignment,
            "pipeline": cfg.pipeline,
            "turns_digest": self._turns_digest(ctx),
        }

    @staticmethod
    def _turns_digest(ctx: StageContext) -> str | None:
        """Bind the fingerprint to the exact diarization table in use."""
        from ..stages.metadata import sha256_of

        try:
            path = ctx.input("speaker_turns")
        except StageError:
            return None
        return sha256_of(path)

    def enabled(self, ctx: StageContext) -> tuple[bool, str]:
        if not ctx.config.diarization.enabled:
            return False, "diarization.enabled = false (no speaker timeline to assign)"
        if ctx.state.stage("whisperx").status != "completed" and not ctx.artifact("speech_words").is_file():
            return False, "transcript unavailable"
        # ``diarization.enabled`` is only an intention: the stage can still be skipped
        # (no HF token) or fail. Assigning speakers over a timeline that was never
        # produced would fail this stage and take the transcript consumers with it.
        diarization_status = ctx.state.stage("diarization").status
        if diarization_status == STATUS_SKIPPED:
            prior = ctx.state.stage("diarization")
            why = (prior.validation_result or {}).get("reason")
            return False, f"diarization produced no speaker turns (skipped: {why or 'unknown'})"
        if diarization_status == STATUS_FAILED:
            return False, "diarization failed; no speaker timeline available"
        if diarization_status != STATUS_COMPLETED and not ctx.artifact("speaker_turns").is_file():
            return False, "speaker turns unavailable (run the diarization stage first)"
        return True, ""

    def prepare(self, ctx: StageContext) -> None:
        ctx.input("speech_segments")
        ctx.input("speech_words")
        ctx.input("speaker_turns")

    def execute(self, ctx: StageContext) -> dict[str, Any]:
        turns = normalise_turns(
            (row["start_time"], row["end_time"], row["speaker_id"])
            for row in read_table(ctx.input("speaker_turns")).to_pylist()
        )
        diarization_type = self._diarization_type(ctx)
        duration = self._duration(ctx)
        stats: dict[str, Any] = {}

        segments = read_table(ctx.input("speech_segments")).to_pylist()
        assigned_segments = [self._assign_row(row, turns) for row in segments]
        write_table(
            ctx.artifact("speech_segments"),
            pa.Table.from_pylist(assigned_segments, schema=SEGMENTS_SCHEMA),
            SEGMENTS_SCHEMA,
            extra_metadata={"speaker_assignment": "max_overlap", "diarization_type": diarization_type,
                            "video_id": ctx.video_id},
        )

        words = read_table(ctx.input("speech_words")).to_pylist()
        assigned_words = [self._assign_row(row, turns) for row in words]
        write_table(
            ctx.artifact("speech_words"),
            pa.Table.from_pylist(assigned_words, schema=WORDS_SCHEMA),
            WORDS_SCHEMA,
            extra_metadata={"speaker_assignment": "max_overlap", "diarization_type": diarization_type,
                            "video_id": ctx.video_id},
        )

        stats["segments"] = len(assigned_segments)
        stats["words"] = len(assigned_words)
        stats["segments_unassigned"] = sum(1 for row in assigned_segments if not row["speaker_id"])
        stats["words_unassigned"] = sum(1 for row in assigned_words if not row["speaker_id"])
        stats["speakers"] = sorted({row["speaker_id"] for row in assigned_segments if row["speaker_id"]})
        stats["coverage"] = coverage_report(turns, duration=duration)
        ctx.scratch["speaker_assignment"] = stats
        ctx.log(
            f"assigned {len(assigned_segments) - stats['segments_unassigned']}/{len(segments)} segments and "
            f"{len(assigned_words) - stats['words_unassigned']}/{len(words)} words to "
            f"{len(stats['speakers'])} speaker(s); diarized {stats['coverage']['speaker_seconds']:.1f}s"
        )
        return {"tool_version": None, "model_version": diarization_type,
                "extra": {"turns": len(turns), **stats}}

    @staticmethod
    def _assign_row(row: dict[str, Any], turns) -> dict[str, Any]:
        assignment = assign_speaker(row.get("start_time"), row.get("end_time"), turns)
        updated = dict(row)
        updated.update(assignment.as_fields())
        return updated

    @staticmethod
    def _diarization_type(ctx: StageContext) -> str | None:
        try:
            path = ctx.input("speaker_turns")
        except StageError:
            return None
        rows = read_table(path).to_pylist()
        return rows[0]["diarization_type"] if rows else None

    @staticmethod
    def _duration(ctx: StageContext) -> float | None:
        path = ctx.artifact("metadata")
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8")).get("duration_seconds")
            except (OSError, json.JSONDecodeError):
                return None
        return None

    # ---------------------------------------------------------------- validation

    def validate(self, ctx: StageContext) -> dict[str, Any]:
        turns_path = ctx.artifact("speaker_turns")
        if not turns_path.is_file():
            raise ValidationError(self.name, ["speaker_turns.parquet missing"])
        known_speakers = {row["speaker_id"] for row in read_table(turns_path).to_pylist() if row["speaker_id"]}
        segments = read_table(ctx.artifact("speech_segments")).to_pylist()
        words = read_table(ctx.artifact("speech_words")).to_pylist()
        if not segments:
            raise ValidationError(self.name, ["segments table is empty after assignment"])
        check_reference_values([row["speaker_id"] for row in segments], known_speakers,
                               stage=self.name, label="segment speaker_id")
        check_reference_values([row["speaker_id"] for row in words], known_speakers,
                               stage=self.name, label="word speaker_id")
        for index, row in enumerate(segments):
            ratio = row.get("speaker_overlap_ratio")
            if ratio is not None and not (0.0 <= ratio <= 1.0 + 1e-9):
                raise ValidationError(self.name, [f"segment[{index}] overlap ratio {ratio} outside [0, 1]"])
            seconds = row.get("speaker_overlap_seconds")
            duration_value = row.get("duration")
            if seconds is not None and duration_value and seconds > duration_value + 1e-6:
                raise ValidationError(
                    self.name, [f"segment[{index}] overlap {seconds:.3f}s exceeds its duration {duration_value:.3f}s"]
                )
        assigned = sum(1 for row in segments if row["speaker_id"])
        return {
            "segments": len(segments),
            "words": len(words),
            "assigned_segments": assigned,
            "assignment_rate": round(assigned / len(segments), 4),
            "speakers": len(known_speakers),
        }
