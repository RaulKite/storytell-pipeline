"""Speaker-assignment stage: normalise transcript rows onto Nemotron diarization turns.

Nemotron is the only diarizer, so ``speech/speaker_turns_nemotron.parquet`` is the one
timeline this stage reads. Its turns overlap across speaker channels, and the assignment
rule is unchanged by that: every transcript interval takes the speaker whose turns cover
most of it, with ties broken deterministically. There is no exclusive variant to prefer —
an exclusive timeline is what a different engine produced, and it is not here any more.

This stage is also the *only* place that decides what ``speaker_id`` means, and that makes
it the one that has to clear stale labels. Datasets transcribed while pyannote was the
diarizer carry the labels it wrote, both in ``segments``/``words`` and in the preserved
raw WhisperX document. When the diarizer produced no timeline this time — its environment
is not installed, its checkpoint is absent, or the operator switched it off — skipping
this stage would publish another engine's speaker labels under a config that names a
different one, and a dataset whose ``speaker_turns_nemotron`` table is absent would have
finalization and the ELAN export succeed while the transcript cites ``SPEAKER_00``. So
this stage runs whenever there is a transcript: it either labels from the cached turn
table or rewrites the derived tables with ``speaker_id = null`` and an explicit
``no_diarization`` method. Raw WhisperX output is never touched, so the evidence survives.

Deliberately a distinct stage: raw WhisperX and raw Nemotron outputs stay untouched, so
this normalisation can be recomputed (with different rules or a better timeline) without
re-running either model.
"""

from __future__ import annotations

import json
from typing import Any

import pyarrow as pa

from ..exceptions import StageError
from ..schemas import SEGMENTS_SCHEMA, WORDS_SCHEMA, read_table, write_table
from ..speaker_assignment import (
    METHOD_NO_DIARIZATION,
    assign_speaker,
    coverage_report,
    normalise_turns,
)
from ..state import STATUS_COMPLETED, STATUS_FAILED, STATUS_SKIPPED  # noqa: F401
from ..validation import check_reference_values
from .base import Stage, StageContext, ValidationError, transcript_skip_reason


class SpeakerAssignmentStage(Stage):
    name = "speaker_assignment"
    inputs = ("speech_segments", "speech_words", "speaker_turns_nemotron")
    outputs = ("speech_segments", "speech_words")
    config_keys = ("diarization_nemotron",)

    def config_fingerprint(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.config.diarization_nemotron
        return {
            "stage": self.name,
            "assignment_method": "max_overlap",
            "model": cfg.model,
            "turns_digest": self._turns_digest(ctx),
        }

    @staticmethod
    def _turns_digest(ctx: StageContext) -> str | None:
        """Bind the fingerprint to the exact diarization table in use."""
        from ..stages.metadata import sha256_of

        try:
            path = ctx.input("speaker_turns_nemotron")
        except StageError:
            return None
        return sha256_of(path)

    def enabled(self, ctx: StageContext) -> tuple[bool, str]:
        """Run whenever there is a transcript, with or without a speaker timeline.

        The two conditions this gate used to require — the diarizer's flag, and a
        *completed* diarizer stage — made a skipped diarizer skip this stage too, which is
        how pyannote labels survived a run that had no diarizer at all. A cached turn table
        is usable input whether or not the stage that wrote it was re-run today, and with no
        turn table this stage still has work to do: clearing labels it did not produce.
        """
        unavailable = transcript_skip_reason(ctx)
        if unavailable:
            return False, unavailable
        return True, ""

    def prepare(self, ctx: StageContext) -> None:
        ctx.input("speech_segments")
        ctx.input("speech_words")
        # Deliberately not ``ctx.input("speaker_turns_nemotron")``: a missing timeline is a
        # supported state (see ``turns``), and this hook ran before every execution.

    def turns(self, ctx: StageContext) -> list[tuple[float, float, str]] | None:
        """The cached timeline, or ``None`` when this dataset has none.

        Three states, the same discipline as the OpenPose and checkpoint gates: absent means
        "no diarization happened here" and the caller clears; present and readable means the
        rows are used; present and unreadable is a defect raised loudly, because silently
        clearing over a corrupt table would delete speaker labels and call it a result.
        """
        path = ctx.artifact("speaker_turns_nemotron")
        if not path.is_file():
            return None
        try:
            rows = read_table(path).to_pylist()
        except Exception as exc:  # noqa: BLE001 - reported as a stage error naming the file
            raise StageError(f"speaker_turns_nemotron.parquet is unreadable: {exc}") from exc
        return normalise_turns((row["start_time"], row["end_time"], row["speaker_id"])
                               for row in rows)

    def execute(self, ctx: StageContext) -> dict[str, Any]:
        turns = self.turns(ctx)
        if turns is None:
            return self._clear(ctx)
        return self._assign(ctx, turns)

    def _assign(self, ctx: StageContext, turns) -> dict[str, Any]:
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

    def _clear(self, ctx: StageContext) -> dict[str, Any]:
        """Remove every speaker label this dataset did not get from a Nemotron timeline.

        ``speaker_id = null`` plus an explicit ``no_diarization`` method is a recorded fact —
        "there is no speaker here because nothing diarized this video" — which is the same
        reading :data:`~multimodal_pipeline.speaker_assignment.METHOD_NO_OVERLAP` gives a gap.
        The overlap columns are cleared with it: they are the *previous* engine's audit of a
        decision that is no longer in this file, and leaving them beside a null label would
        describe a 100%-confident assignment to nobody.
        """
        cleared = 0
        counts: dict[str, int] = {}
        for artifact, schema in (("speech_segments", SEGMENTS_SCHEMA),
                                 ("speech_words", WORDS_SCHEMA)):
            rows = read_table(ctx.input(artifact)).to_pylist()
            counts[artifact] = len(rows)
            cleared += sum(1 for row in rows if row.get("speaker_id"))
            cleaned = [{**row,
                        "speaker_id": None,
                        "speaker_overlap_seconds": None,
                        "speaker_overlap_ratio": None,
                        "speaker_assignment_method": METHOD_NO_DIARIZATION}
                       for row in rows]
            write_table(
                ctx.artifact(artifact),
                pa.Table.from_pylist(cleaned, schema=schema),
                schema,
                extra_metadata={"speaker_assignment": METHOD_NO_DIARIZATION,
                                "diarization_type": None, "video_id": ctx.video_id},
            )
        stats = {"segments": counts["speech_segments"], "words": counts["speech_words"],
                 "cleared_labels": cleared,
                 "reason": "no speaker_turns_nemotron table in this dataset"}
        ctx.scratch["speaker_assignment"] = stats
        ctx.log(
            f"no diarization timeline: cleared {cleared} speaker label(s) from the transcript "
            f"(segments={stats['segments']})"
        )
        return {"tool_version": None, "model_version": None,
                "extra": {**stats, "turns": 0, "speakers": []}}

    @staticmethod
    def _assign_row(row: dict[str, Any], turns) -> dict[str, Any]:
        assignment = assign_speaker(row.get("start_time"), row.get("end_time"), turns)
        updated = dict(row)
        updated.update(assignment.as_fields())
        return updated

    @staticmethod
    def _diarization_type(ctx: StageContext) -> str | None:
        try:
            path = ctx.input("speaker_turns_nemotron")
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
        turns_path = ctx.artifact("speaker_turns_nemotron")
        segments = read_table(ctx.artifact("speech_segments")).to_pylist()
        words = read_table(ctx.artifact("speech_words")).to_pylist()
        if not turns_path.is_file():
            # The cleared case, checked rather than trusted: a label with no timeline to
            # justify it is exactly the defect this stage exists to prevent, and it is the
            # state a dataset reaches by skipping this stage on an old transcript.
            cited = sorted({row["speaker_id"] for row in segments + words if row.get("speaker_id")})
            if cited:
                raise ValidationError(self.name, [
                    f"transcript cites {cited[:5]} with no diarization timeline "
                    f"(speaker_turns_nemotron.parquet is absent); re-run "
                    f"speaker_assignment to clear labels no current diarizer produced"])
            return {"segments": len(segments), "words": len(words), "assigned_segments": 0,
                    "assignment_rate": None, "speakers": 0, "cleared": True}
        known_speakers = {row["speaker_id"] for row in read_table(turns_path).to_pylist() if row["speaker_id"]}
        # An empty transcript is a legitimate outcome, not a broken assignment:
        # whisperx completes happily on a silent video, and the diarizer may still find
        # a turn in background noise. This path was unreachable while diarization
        # skipped for a missing token, so the check rejected correct output as soon as
        # a real speaker timeline existed.
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
            # No segments means no rate. Averaging over an empty transcript is
            # meaningless, so report the absence rather than divide by zero.
            "assignment_rate": round(assigned / len(segments), 4) if segments else None,
            "speakers": len(known_speakers),
        }
