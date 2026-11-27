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
import math
import os
from pathlib import Path
from typing import Any

import pyarrow as pa

from ..artifacts import read_json
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
from .base import (
    TRANSCRIPT_ARTIFACTS,
    Stage,
    StageContext,
    ValidationError,
    raw_sidecar_path,
    transcript_skip_reason,
)

#: The two tables this stage labels, in the order it reports them.
TRANSCRIPT_TABLES: tuple[str, ...] = TRANSCRIPT_ARTIFACTS

#: Each transcript table's schema, keyed by artifact name so the label/clear passes can iterate
#: over "whichever tables exist" instead of over a literal tuple that one of them has to miss.
_SCHEMAS = {"speech_segments": SEGMENTS_SCHEMA, "speech_words": WORDS_SCHEMA}

#: Columns a turn table must have before its rows can label anything.
TURN_COLUMNS: tuple[str, ...] = ("start_time", "end_time", "speaker_id")

#: Slack allowed on a turn's end against the container's measured duration.
#:
#: Not zero, and the corpus is the reason: on ``La-1`` the transcript container measures
#: 8.008008 s while Nemotron's last turn ends at 8.01 s — the diarizer works on its own frame
#: grid and its last frame closes after the container's last sample rounded down. A real defect
#: here is not two milliseconds; it is a timeline from another recording, which is off by
#: seconds or by minutes. Any slack would let that through, so the bound is only as wide as the
#: measurement discrepancy it has to absorb.
TURN_TIME_TOLERANCE_SECONDS = 0.05


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
        """Run whenever *a* transcript table exists, with or without a speaker timeline.

        The two conditions this gate used to require — the diarizer's flag, and a
        *completed* diarizer stage — made a skipped diarizer skip this stage too, which is
        how pyannote labels survived a run that had no diarizer at all. A cached turn table
        is usable input whether or not the stage that wrote it was re-run today, and with no
        turn table this stage still has work to do: clearing labels it did not produce.

        **Either transcript table is enough**, and that is a defect fix rather than a
        generalisation for its own sake. Requiring both made a dataset whose
        ``speech/segments.parquet`` was missing skip this stage entirely, and the words table it
        did have kept its pyannote ``SPEAKER_00``: the ELAN words tier published that label, and
        finalization — which read only the segments table for its own speaker cross-check — had
        nothing to say about it. A table that exists is a table this stage has to label or clear.
        """
        if self.transcript_tables(ctx):
            return True, ""
        # Both absent: quote the same sentence every other transcript consumer uses, so the
        # batch report reads as one statement about the missing transcript.
        return False, transcript_skip_reason(ctx) or (
            "upstream skipped: transcript unavailable: neither speech/segments.parquet nor "
            "speech/words.parquet exists in this dataset — run the whisperx stage first")

    @staticmethod
    def transcript_tables(ctx: StageContext) -> list[str]:
        """The transcript tables this dataset actually has, in canonical order.

        One reader for "which tables do I label", because ``prepare``, the assign pass, the clear
        pass and ``validate`` all have to answer it the same way. Disagreement between two of
        them is how a words-only dataset got validated as a segments dataset.
        """
        return [name for name in TRANSCRIPT_TABLES if ctx.artifact(name).is_file()]

    def prepare(self, ctx: StageContext) -> None:
        for name in self.transcript_tables(ctx):
            ctx.input(name)
        # Deliberately not ``ctx.input("speaker_turns_nemotron")``: a missing timeline is a
        # supported state (see ``turns``), and this hook ran before every execution.

    def turns(self, ctx: StageContext) -> list[tuple[float, float, str]] | None:
        """The cached timeline, or ``None`` when this dataset has none.

        Four states now, and the fourth is the one that used to be silent: absent means "no
        diarization happened here" and the caller clears; present and readable means the rows are
        used; present and unreadable is a defect raised loudly, because silently clearing over a
        corrupt table would delete speaker labels and call it a result; and present, readable and
        **disagreeing with this video** is a defect too — see :meth:`_check_timeline`.

        ``None`` is reserved for genuine absence. That distinction is the whole finding: a
        damaged timeline used to pass ``normalise_turns`` (which drops what it cannot parse, by
        design, being a pure function), label nothing, and produce a transcript of null labels
        with the run reported as a success.
        """
        path = ctx.artifact("speaker_turns_nemotron")
        if not path.is_file():
            return None
        try:
            rows = read_table(path).to_pylist()
        except Exception as exc:  # noqa: BLE001 - reported as a stage error naming the file
            raise StageError(f"speaker_turns_nemotron.parquet is unreadable: {exc}") from exc
        self._check_timeline(ctx, path, rows)
        return normalise_turns((row["start_time"], row["end_time"], row["speaker_id"])
                               for row in rows)

    def _check_timeline(self, ctx: StageContext, path: Path, rows: list[dict[str, Any]]) -> None:
        """Refuse a turn table that cannot honestly label this transcript.

        Four independent questions, each its own message because each is its own way for the file
        to be wrong, and each answered *before* a single label is written:

        * **schema** — the three columns a row needs to be usable at all;
        * **shape** — every interval finite with ``start < end`` inside the audio. A reversed or
          out-of-range interval is not a turn to skip: it is proof that this file was not written
          by the stage that writes it, and the label pass would have quietly produced nulls;
        * **identity** — the ``video_id`` column must name *this* video. A turn table copied from
          another recording (a hand copy, or a dataset assembled by moving directories) otherwise
          labels every segment against a timeline of the wrong conversation, and the overlap
          looks plausible because every recording has speech at t=0;
        * **provenance** — Nemotron's own sidecar records the ``audio_sha256`` its timeline was
          produced from. A different digest means the audio was re-extracted or replaced under a
          cached timeline. Absent provenance is not a mismatch: an older dataset has no sidecar,
          and "cannot prove" must not read as "proved wrong".

        Raised as :class:`StageError`, so the stage fails the video and its record names the file
        and the defect. The alternative — clearing labels because the timeline was unusable —
        would delete a measurement and report the dataset as complete.
        """
        if not rows:
            # An empty table is the silent video's own outcome (`pipeline_silent` on this corpus
            # has zero turns and zero transcript), so it is a supported state, not a defect.
            return
        missing = [column for column in TURN_COLUMNS if column not in rows[0]]
        if missing:
            raise StageError(
                f"speaker_turns_nemotron.parquet has no {', '.join(missing)} column: the table "
                f"cannot label a transcript (columns present: {', '.join(sorted(rows[0]))})")
        duration = self._duration(ctx)
        for index, row in enumerate(rows):
            start, end = row.get("start_time"), row.get("end_time")
            if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                       and math.isfinite(float(value)) for value in (start, end)):
                raise StageError(
                    f"speaker_turns_nemotron.parquet row {index} has a non-numeric interval "
                    f"(start_time={start!r}, end_time={end!r}): the timeline is damaged, and "
                    f"labelling from it would produce null speakers rather than an error")
            if float(start) >= float(end):
                raise StageError(
                    f"speaker_turns_nemotron.parquet row {index} is not a forward interval "
                    f"(start_time={start}, end_time={end}): the timeline is reversed or empty, "
                    f"so it cannot be the output of the diarization stage that writes it")
            if float(start) < -TURN_TIME_TOLERANCE_SECONDS:
                raise StageError(
                    f"speaker_turns_nemotron.parquet row {index} starts before the audio "
                    f"(start_time={start}): the timeline does not describe this clip")
            if duration and float(end) > float(duration) + TURN_TIME_TOLERANCE_SECONDS:
                raise StageError(
                    f"speaker_turns_nemotron.parquet row {index} ends at {end}s, past this "
                    f"video's {float(duration):.3f}s of audio (tolerance "
                    f"{TURN_TIME_TOLERANCE_SECONDS}s): the timeline was produced from a "
                    f"different recording or a different audio extraction")
        others = sorted({str(row["video_id"]) for row in rows
                         if row.get("video_id") and str(row["video_id"]) != ctx.video_id})
        if others:
            raise StageError(
                f"speaker_turns_nemotron.parquet was written for {others[:3]}, not for "
                f"{ctx.video_id}: a timeline from another recording would label this transcript "
                f"against the wrong conversation")
        self._check_audio_provenance(ctx)

    def _check_audio_provenance(self, ctx: StageContext) -> None:
        """The diarizer's own record of which audio produced this timeline, when it exists.

        Read from the preserved raw document's sidecar (``request.audio_sha256``), never from the
        stage's configuration: the question is what the worker was *given*, and that is what the
        sidecar recorded at the time. Anything that cannot be read, or a dataset written before
        the digest was recorded, yields no claim rather than a failure.
        """
        sidecar = raw_sidecar_path(ctx.artifact("nemotron_diarization_raw"))
        audio = ctx.artifact("audio")
        if not (sidecar.is_file() and audio.is_file()):
            return
        try:
            recorded = ((read_json(sidecar) or {}).get("request") or {}).get("audio_sha256")
        except (OSError, ValueError):
            return
        if not recorded:
            return
        from .metadata import sha256_of

        current = sha256_of(audio)
        if current and current != recorded:
            raise StageError(
                f"speaker_turns_nemotron.parquet was diarized from different audio "
                f"(audio_sha256 {recorded[:12]}… in the Nemotron sidecar, "
                f"{current[:12]}… for audio/audio.wav): re-run the diarization_nemotron stage "
                f"for this audio, or the labels would describe a timeline of another signal")

    def execute(self, ctx: StageContext) -> dict[str, Any]:
        turns = self.turns(ctx)
        if turns is None:
            return self._clear(ctx)
        return self._assign(ctx, turns)

    def _assign(self, ctx: StageContext, turns) -> dict[str, Any]:
        """Label every transcript table this dataset has, from one timeline.

        Both tables are read and both results are built **before either file is replaced**. That
        ordering is the fix for a real failure: with the words table corrupt, the old code wrote
        the labelled segments first and then raised on the words read — leaving a dataset whose
        two transcript tables described different runs, mid-stage, with the state file saying the
        stage failed and no way back to the previous transcript.

        A dataset with one of the two tables is labelled on the one it has (see
        :meth:`transcript_tables`); the statistics name only the tables that were written, so a
        report never claims to have labelled a file that was not there.
        """
        diarization_type = self._diarization_type(ctx)
        duration = self._duration(ctx)
        labels = {"speaker_assignment": "max_overlap", "diarization_type": diarization_type,
                  "video_id": ctx.video_id}
        stats: dict[str, Any] = {}

        # Read everything first: a table that cannot be read must cost nothing.
        assigned: dict[str, list[dict[str, Any]]] = {}
        for artifact in self.transcript_tables(ctx):
            rows = self._read_transcript(ctx, artifact)
            assigned[artifact] = [self._assign_row(row, turns) for row in rows]

        self._publish(ctx, {artifact: (rows, _SCHEMAS[artifact])
                            for artifact, rows in assigned.items()}, labels=labels)

        if "speech_segments" in assigned:
            segments = assigned["speech_segments"]
            stats["segments"] = len(segments)
            stats["segments_unassigned"] = sum(1 for row in segments if not row["speaker_id"])
            stats["speakers"] = sorted({row["speaker_id"] for row in segments if row["speaker_id"]})
        if "speech_words" in assigned:
            words = assigned["speech_words"]
            stats["words"] = len(words)
            stats["words_unassigned"] = sum(1 for row in words if not row["speaker_id"])
        stats["coverage"] = coverage_report(turns, duration=duration)
        total = (stats.get("segments", 0) - stats.get("segments_unassigned", 0),
                 stats.get("words", 0) - stats.get("words_unassigned", 0))
        ctx.scratch["speaker_assignment"] = stats
        ctx.log(
            f"assigned {total[0]}/{stats.get('segments', 0)} segments and "
            f"{total[1]}/{stats.get('words', 0)} words to "
            f"{len(stats.get('speakers', []))} speaker(s); "
            f"diarized {stats['coverage']['speaker_seconds']:.1f}s"
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
        cleaned_tables: dict[str, tuple[list[dict[str, Any]], pa.schema]] = {}
        for artifact in self.transcript_tables(ctx):
            rows = self._read_transcript(ctx, artifact)
            counts[artifact] = len(rows)
            cleared += sum(1 for row in rows if row.get("speaker_id"))
            cleaned_tables[artifact] = ([{**row,
                                          "speaker_id": None,
                                          "speaker_overlap_seconds": None,
                                          "speaker_overlap_ratio": None,
                                          "speaker_assignment_method": METHOD_NO_DIARIZATION}
                                         for row in rows], _SCHEMAS[artifact])
        # Same ordering rule as `_assign`: every table is built before any is replaced.
        self._publish(ctx, cleaned_tables,
                      labels={"speaker_assignment": METHOD_NO_DIARIZATION,
                              "diarization_type": None, "video_id": ctx.video_id})
        stats: dict[str, Any] = {"cleared_labels": cleared,
                                 "reason": "no speaker_turns_nemotron table in this dataset"}
        if "speech_segments" in counts:
            stats["segments"] = counts["speech_segments"]
        if "speech_words" in counts:
            stats["words"] = counts["speech_words"]
        ctx.scratch["speaker_assignment"] = stats
        counted = ", ".join(f"{artifact}={count}" for artifact, count in sorted(counts.items()))
        ctx.log(f"no diarization timeline: cleared {cleared} speaker label(s) from the "
                f"transcript ({counted})")
        return {"tool_version": None, "model_version": None,
                "extra": {**stats, "turns": 0, "speakers": []}}

    @staticmethod
    def _publish(ctx: StageContext, tables: dict[str, tuple[list[dict[str, Any]], pa.schema]],
                 *, labels: dict[str, Any]) -> None:
        """Write every staged table, then replace them all.

        ``write_table`` writes straight to its destination, which is fine for a stage that owns
        one file and wrong for this one: it owns the pair that has to agree. Each table is
        therefore written to a ``.staging`` sibling first, and only once *all* of them exist is
        any destination replaced. A crash or a full disk before that point leaves the previous
        transcript intact; the two renames that follow cannot themselves fail on a healthy
        filesystem, and cannot be made one atomic step across two paths — so the guarantee is
        stated as it is: no read or write failure can leave a half-labelled pair.
        """
        staged: list[tuple[Path, Path]] = []
        try:
            for artifact, (rows, schema) in tables.items():
                target = ctx.artifact(artifact)
                temporary = target.with_name(target.name + ".staging")
                write_table(temporary, pa.Table.from_pylist(rows, schema=schema), schema,
                            extra_metadata=labels)
                staged.append((temporary, target))
        except BaseException:
            for temporary, _ in staged:
                temporary.unlink(missing_ok=True)
            raise
        for temporary, target in staged:
            os.replace(temporary, target)

    @staticmethod
    def _read_transcript(ctx: StageContext, artifact: str) -> list[dict[str, Any]]:
        """One transcript table, or a stage error naming it.

        An unreadable Parquet file raises ``pyarrow.lib.ArrowInvalid`` out of ``read_table``, and
        "Parquet magic bytes not found in footer" does not tell an operator which of the two
        transcript files broke, nor that the fix is a rerun of the stage that writes it. The
        message is wrapped rather than swallowed: this is a defect, not a reason to skip.
        """
        try:
            return read_table(ctx.input(artifact)).to_pylist()
        except StageError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported as a stage error naming the file
            raise StageError(f"{artifact} is unreadable: {exc}") from exc

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
        """The labels it published are the ones this dataset's timeline justifies.

        Reads whichever transcript tables exist, for the same reason every other pass does: a
        dataset with only ``words.parquet`` has labels to check, and requiring the segments table
        here made its stale ``SPEAKER_00`` invisible to the one stage that owns ``speaker_id``.
        """
        turns_path = ctx.artifact("speaker_turns_nemotron")
        present = self.transcript_tables(ctx)
        tables = {artifact: self._read_transcript(ctx, artifact) for artifact in present}
        segments = tables.get("speech_segments", [])
        words = tables.get("speech_words", [])
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
        result: dict[str, Any] = {
            "assigned_segments": assigned,
            # No segments means no rate. Averaging over an empty transcript is
            # meaningless, so report the absence rather than divide by zero.
            "assignment_rate": round(assigned / len(segments), 4) if segments else None,
            "speakers": len(known_speakers),
        }
        # Only the tables this dataset has: "segments: 0" for a table that does not exist reads
        # as an empty transcript rather than as a missing one.
        for artifact, rows in tables.items():
            result["segments" if artifact == "speech_segments" else "words"] = len(rows)
        return result

    def outputs_present(self, ctx: StageContext) -> bool:
        """Reusable when the transcript tables *this dataset has* are all on disk.

        ``Stage.outputs_present`` requires every declared output, which for a words-only dataset
        can never be true: it would rerun this stage on every ``run`` forever, and the rerun is
        the only thing standing between that dataset and a permanently stale label. The tables
        that do exist are what this stage can be asked about, and ``validate`` — which reuse also
        requires — is what checks the labels inside them.
        """
        present = self.transcript_tables(ctx)
        return bool(present)
