"""Speaker assignment: max overlap, deterministic ties, gaps, overlapping speech."""

from __future__ import annotations

import math

import pytest

from multimodal_pipeline.speaker_assignment import (
    METHOD_MAX_OVERLAP,
    METHOD_NO_OVERLAP,
    METHOD_POINT_CONTAINMENT,
    assign_intervals,
    assign_speaker,
    coverage_report,
    normalise_turns,
    overlap_seconds,
)

TURNS = normalise_turns([
    (0.0, 5.0, "SPEAKER_00"),
    (5.0, 10.0, "SPEAKER_01"),
    (12.0, 15.0, "SPEAKER_00"),
])


class TestOverlap:
    @pytest.mark.parametrize("a,b,expected", [
        ((0, 10), (0, 10), 10.0),
        ((0, 10), (5, 15), 5.0),
        ((0, 10), (20, 30), 0.0),
        ((0, 10), (-5, 3), 3.0),
        ((0, 10), (2, 4), 2.0),
        ((5, 5), (0, 10), 0.0),
    ])
    def test_overlap_length(self, a, b, expected) -> None:
        assert overlap_seconds(*a, *b) == pytest.approx(expected)

    def test_never_negative(self) -> None:
        assert overlap_seconds(10, 0, 0, 5) >= 0.0


class TestNormaliseTurns:
    def test_sorts_by_start(self) -> None:
        turns = normalise_turns([(9.0, 10.0, "B"), (1.0, 2.0, "A")])
        assert [t[0] for t in turns] == [1.0, 9.0]

    @pytest.mark.parametrize("bad", [
        (None, 5.0, "A"), (0.0, None, "A"), (0.0, 5.0, None),
        (5.0, 1.0, "A"), ("x", 5.0, "A"),
    ])
    def test_drops_malformed_turns(self, bad) -> None:
        assert normalise_turns([bad, (0.0, 1.0, "OK")]) == [(0.0, 1.0, "OK")]

    def test_zero_length_turn_survives(self) -> None:
        assert normalise_turns([(5.0, 5.0, "A")]) == [(5.0, 5.0, "A")]

    def test_numeric_strings_are_coerced(self) -> None:
        assert normalise_turns([("1.5", "2.5", "A")]) == [(1.5, 2.5, "A")]


class TestMaxOverlap:
    def test_clear_winner(self) -> None:
        result = assign_speaker(0.0, 4.0, TURNS)
        assert result.speaker_id == "SPEAKER_00"
        assert result.method == METHOD_MAX_OVERLAP
        assert result.overlap_seconds == pytest.approx(4.0)
        assert result.overlap_ratio == pytest.approx(1.0)

    def test_minority_overlap_loses(self) -> None:
        result = assign_speaker(4.0, 9.0, TURNS)  # 1s of 00, 4s of 01
        assert result.speaker_id == "SPEAKER_01"
        assert result.overlap_seconds == pytest.approx(4.0)
        assert result.runner_up_speaker_id == "SPEAKER_00"
        assert result.runner_up_overlap_seconds == pytest.approx(1.0)

    def test_gap_yields_no_speaker_instead_of_a_guess(self) -> None:
        result = assign_speaker(10.5, 11.5, TURNS)
        assert result.speaker_id is None
        assert result.method == METHOD_NO_OVERLAP
        assert result.overlap_seconds == 0.0
        assert result.contributions == ()

    def test_ratio_is_a_fraction_of_the_intervals_own_duration(self) -> None:
        # 4..6 straddles the 5.0 turn boundary: 1s of each speaker, so the winner
        # covers half the interval and the other half is genuinely ambiguous.
        result = assign_speaker(4.0, 6.0, TURNS)
        assert result.overlap_ratio == pytest.approx(0.5)
        assert result.overlap_seconds == pytest.approx(1.0)
        assert result.runner_up_overlap_seconds == pytest.approx(1.0)
        # A long interval touching only part of the diarization stays < 1.
        assert assign_speaker(4.0, 20.0, TURNS).overlap_ratio < 1.0

    def test_disjoint_turns_of_one_speaker_are_aggregated(self) -> None:
        # B owns 2..8 = 6s; A owns two turns totalling 4s. Summing per speaker
        # (instead of per turn) is what lets a speaker with several short turns win.
        turns = normalise_turns([(0.0, 2.0, "A"), (8.0, 10.0, "A"), (2.0, 8.0, "B")])
        result = assign_speaker(0.0, 10.0, turns)
        assert result.speaker_id == "B"
        assert result.overlap_seconds == pytest.approx(6.0)
        assert dict(result.contributions)["A"] == pytest.approx(4.0)

    def test_two_speakers_each_covering_everything_cannot_exceed_ratio_one(self) -> None:
        turns = normalise_turns([(0.0, 10.0, "A"), (0.0, 10.0, "B")])
        result = assign_speaker(0.0, 10.0, turns)
        assert result.overlap_ratio == pytest.approx(1.0)

    def test_empty_turn_list(self) -> None:
        result = assign_speaker(0.0, 5.0, [])
        assert result.speaker_id is None and result.method == METHOD_NO_OVERLAP


class TestDeterminism:
    def test_tie_breaks_on_earliest_turn_then_label(self) -> None:
        turns = normalise_turns([(5.0, 7.0, "SPEAKER_09"), (0.0, 2.0, "SPEAKER_01")])
        # Equal overlap (2s each) for an interval spanning both.
        result = assign_speaker(0.0, 7.0, turns)
        assert result.overlap_seconds == pytest.approx(result.runner_up_overlap_seconds)
        assert result.speaker_id == "SPEAKER_01"

    def test_input_turn_order_does_not_change_the_result(self) -> None:
        shuffled = [(12.0, 15.0, "SPEAKER_00"), (5.0, 10.0, "SPEAKER_01"), (0.0, 5.0, "SPEAKER_00")]
        assert assign_speaker(4.0, 9.0, normalise_turns(shuffled)).as_fields() == \
            assign_speaker(4.0, 9.0, TURNS).as_fields()

    def test_same_label_tie_prefers_lexicographic(self) -> None:
        turns = normalise_turns([(0.0, 1.0, "Z"), (0.0, 1.0, "A")])
        assert assign_speaker(0.0, 1.0, turns).speaker_id == "A"


class TestDegenerateIntervals:
    def test_missing_end_is_a_probe_point(self) -> None:
        result = assign_speaker(6.0, None, TURNS)
        assert result.speaker_id == "SPEAKER_01"
        assert result.method == METHOD_POINT_CONTAINMENT

    def test_zero_length_interval_inside_a_turn(self) -> None:
        result = assign_speaker(6.0, 6.0, TURNS)
        assert result.speaker_id == "SPEAKER_01"
        assert result.method == METHOD_POINT_CONTAINMENT

    def test_point_in_a_gap(self) -> None:
        assert assign_speaker(11.0, 11.0, TURNS).speaker_id is None

    def test_point_on_a_boundary_is_deterministic(self) -> None:
        # t=5.0 belongs to both turns; the earlier turn's owner wins.
        assert assign_speaker(5.0, 5.0, TURNS).speaker_id == "SPEAKER_00"

    def test_missing_start_is_unassignable(self) -> None:
        assert assign_speaker(None, 5.0, TURNS).speaker_id is None


class TestFieldsAndRows:
    def test_as_fields_rounds_and_names_every_column(self) -> None:
        fields = assign_speaker(4.0, 9.0, TURNS).as_fields()
        assert set(fields) == {"speaker_id", "speaker_overlap_seconds",
                               "speaker_overlap_ratio", "speaker_assignment_method"}
        assert all(not isinstance(v, float) or math.isfinite(v) for v in fields.values())

    def test_assign_intervals_preserves_order(self) -> None:
        rows = [{"start_time": 6.0, "end_time": 9.0}, {"start_time": 1.0, "end_time": 2.0},
                {"start_time": 11.0, "end_time": 11.5}]
        speakers = [a.speaker_id for a in assign_intervals(rows, TURNS)]
        assert speakers == ["SPEAKER_01", "SPEAKER_00", None]

    def test_no_speaker_is_explicit_not_absent(self) -> None:
        fields = assign_speaker(11.0, 11.5, TURNS).as_fields()
        assert "speaker_id" in fields and fields["speaker_id"] is None


class TestCoverage:
    def test_disjoint_turns_sum(self) -> None:
        report = coverage_report(TURNS, duration=20.0)
        assert report["speaker_seconds"] == pytest.approx(13.0)  # 5 + 5 + 3
        assert report["speaker_time"] == pytest.approx(0.65)
        assert report["overlap_seconds"] == pytest.approx(0.0)

    def test_overlapping_speech_is_counted_once_as_coverage(self) -> None:
        turns = normalise_turns([(0.0, 10.0, "A"), (5.0, 15.0, "B")])
        report = coverage_report(turns, duration=15.0)
        assert report["speaker_seconds"] == pytest.approx(15.0)
        assert report["overlap_seconds"] == pytest.approx(5.0)

    def test_nested_turns(self) -> None:
        turns = normalise_turns([(0.0, 10.0, "A"), (4.0, 6.0, "B")])
        report = coverage_report(turns, duration=10.0)
        assert report["speaker_seconds"] == pytest.approx(10.0)
        assert report["overlap_seconds"] == pytest.approx(2.0)

    def test_triple_stack_counts_time_with_two_or_more_speakers(self) -> None:
        # Active speakers per second: 1,2,3,2,1 -> 6s have two or more speakers.
        # Overlap is wall-clock time with concurrent speech, not speaker-seconds above
        # the single-speaker baseline.
        turns = normalise_turns([(0.0, 6.0, "A"), (2.0, 8.0, "B"), (4.0, 10.0, "C")])
        report = coverage_report(turns, duration=10.0)
        assert report["speaker_seconds"] == pytest.approx(10.0)
        assert report["overlap_seconds"] == pytest.approx(6.0)

    def test_empty_and_missing_duration(self) -> None:
        assert coverage_report([])["speaker_seconds"] == 0.0
        assert coverage_report(TURNS)["speaker_time"] == 0.0


class TestSilentVideoIsValid:
    """A silent video has no segments; that is a correct result, not a failure.

    The empty-transcript path was originally unreachable: the stage skipped whenever the
    diarizer's credential was absent, so it was never exercised with a speaker timeline
    present. A silent video produces turns and zero segments, and the stage used to
    reject its own correct output.
    """

    def test_empty_transcript_validates(self, context):
        import pyarrow as pa
        from multimodal_pipeline.schemas import (
            SEGMENTS_SCHEMA, SPEAKER_TURNS_NEMOTRON_SCHEMA, WORDS_SCHEMA, write_table,
        )
        from multimodal_pipeline.stages.speaker_assignment import SpeakerAssignmentStage

        write_table(context.artifact("speech_segments"),
                    pa.Table.from_pylist([], schema=SEGMENTS_SCHEMA), SEGMENTS_SCHEMA)
        write_table(context.artifact("speech_words"),
                    pa.Table.from_pylist([], schema=WORDS_SCHEMA), WORDS_SCHEMA)
        write_table(context.artifact("speaker_turns_nemotron"),
                    pa.Table.from_pylist(
                        [{"schema_version": "1.0", "video_id": context.video_id,
                          "turn_id": "turn000001", "start_time": 0.0, "end_time": 1.0,
                          "duration": 1.0, "speaker_id": "speaker_0",
                          "diarization_type": "overlapping", "overlap_s": 0.0}],
                        schema=SPEAKER_TURNS_NEMOTRON_SCHEMA),
                    SPEAKER_TURNS_NEMOTRON_SCHEMA)
        stage = SpeakerAssignmentStage()
        result = stage.validate(context)
        assert result["segments"] == 0
        # No segments means no rate; the key must not be a ZeroDivisionError.
        assert result["assigned_segments"] == 0
        assert result["speakers"] == 1

    def test_execute_writes_metadata_without_turn_type(self, context):
        """The table metadata records the assignment method and the diarizer's own type."""
        import pyarrow as pa
        from multimodal_pipeline.schemas import (
            SEGMENTS_SCHEMA, SPEAKER_TURNS_NEMOTRON_SCHEMA, WORDS_SCHEMA, read_table,
            write_table,
        )
        from multimodal_pipeline.stages.speaker_assignment import SpeakerAssignmentStage

        write_table(context.artifact("speech_segments"),
                    pa.Table.from_pylist([], schema=SEGMENTS_SCHEMA), SEGMENTS_SCHEMA)
        write_table(context.artifact("speech_words"),
                    pa.Table.from_pylist([], schema=WORDS_SCHEMA), WORDS_SCHEMA)
        write_table(context.artifact("speaker_turns_nemotron"),
                    pa.Table.from_pylist(
                        [{"schema_version": "1.0", "video_id": context.video_id,
                          "turn_id": "turn000001", "start_time": 0.0, "end_time": 1.0,
                          "duration": 1.0, "speaker_id": "speaker_0",
                          "diarization_type": "overlapping", "overlap_s": 0.0}],
                        schema=SPEAKER_TURNS_NEMOTRON_SCHEMA),
                    SPEAKER_TURNS_NEMOTRON_SCHEMA)
        SpeakerAssignmentStage().execute(context)
        meta = read_table(context.artifact("speech_segments")).schema.metadata
        assert meta is not None and meta[b"speaker_assignment"] == b"max_overlap"


#: A transcript as an old pyannote run left it: labels *and* the overlap columns its
#: assignment wrote. `whisperx_legacy_raw` is the preserved native document beside it.
LEGACY_SEGMENTS = [
    {"schema_version": "1.0", "video_id": "conversation_001", "segment_id": "seg0001",
     "start_time": 0.0, "end_time": 2.0, "duration": 2.0, "language": "es",
     "speaker_id": "SPEAKER_00", "text": "hola", "confidence": 0.9,
     "speaker_overlap_seconds": 2.0, "speaker_overlap_ratio": 1.0,
     "speaker_assignment_method": "max_overlap"},
    {"schema_version": "1.0", "video_id": "conversation_001", "segment_id": "seg0002",
     "start_time": 2.0, "end_time": 4.0, "duration": 2.0, "language": "es",
     "speaker_id": "SPEAKER_01", "text": "adios", "confidence": 0.8,
     "speaker_overlap_seconds": 2.0, "speaker_overlap_ratio": 1.0,
     "speaker_assignment_method": "max_overlap"},
]

LEGACY_WORDS = [
    {"schema_version": "1.0", "video_id": "conversation_001", "segment_id": "seg0001",
     "word_id": "w0001", "start_time": 0.0, "end_time": 0.5, "duration": 0.5,
     "speaker_id": "SPEAKER_00", "word": "hola", "confidence": 1.0,
     "alignment_status": "aligned", "character_start": 0, "character_end": 4,
     "speaker_overlap_seconds": 0.5, "speaker_overlap_ratio": 1.0,
     "speaker_assignment_method": "max_overlap"},
]

NEMOTRON_TURNS = [
    {"schema_version": "1.0", "video_id": "conversation_001", "turn_id": "turn000001",
     "start_time": 0.0, "end_time": 2.5, "duration": 2.5, "speaker_id": "speaker_0",
     "diarization_type": "overlapping", "overlap_s": 0.0},
    {"schema_version": "1.0", "video_id": "conversation_001", "turn_id": "turn000002",
     "start_time": 2.5, "end_time": 4.0, "duration": 1.5, "speaker_id": "speaker_1",
     "diarization_type": "overlapping", "overlap_s": 0.0},
]


def _write_transcript(ctx, *, turns: bool) -> None:
    """A transcript carrying pyannote labels, optionally beside a cached Nemotron table."""
    import pyarrow as pa
    from multimodal_pipeline.schemas import (
        SEGMENTS_SCHEMA, SPEAKER_TURNS_NEMOTRON_SCHEMA, WORDS_SCHEMA, write_table,
    )

    write_table(ctx.artifact("speech_segments"),
                pa.Table.from_pylist(LEGACY_SEGMENTS, schema=SEGMENTS_SCHEMA), SEGMENTS_SCHEMA)
    write_table(ctx.artifact("speech_words"),
                pa.Table.from_pylist(LEGACY_WORDS, schema=WORDS_SCHEMA), WORDS_SCHEMA)
    if turns:
        write_table(ctx.artifact("speaker_turns_nemotron"),
                    pa.Table.from_pylist(NEMOTRON_TURNS, schema=SPEAKER_TURNS_NEMOTRON_SCHEMA),
                    SPEAKER_TURNS_NEMOTRON_SCHEMA)


class TestDiarizerWithoutItsEnvironment:
    """A skipped Nemotron stage may not leave pyannote labels in the transcript.

    Reproduced on corpus copies with the Nemotron environment missing: the diarizer and
    ``speaker_assignment`` both skipped, and depending on whether a turn table was on disk
    either finalization failed on ``SPEAKER_00`` or — with no turn table — finalization and
    the ELAN export succeeded while ``segments``/``words`` still cited a speaker that no
    table in the dataset contains. Both outcomes are silent lies about the data.
    """

    def test_cached_turns_are_used_even_though_the_diarizer_stage_skipped(self, context):
        """The turn table is the input; the stage that wrote it need not have run now."""
        from multimodal_pipeline.schemas import read_table
        from multimodal_pipeline.stages.speaker_assignment import SpeakerAssignmentStage

        _write_transcript(context, turns=True)
        context.state.mark_skipped("diarization_nemotron",
                                   "diarization_nemotron environment not installed at /nowhere")
        context.state.mark_completed("whisperx")
        stage = SpeakerAssignmentStage()
        enabled, reason = stage.enabled(context)
        assert enabled is True, reason
        stage.execute(context)
        speakers = [row["speaker_id"] for row in read_table(context.artifact("speech_segments")).to_pylist()]
        assert speakers == ["speaker_0", "speaker_1"]
        words = read_table(context.artifact("speech_words")).to_pylist()
        assert words[0]["speaker_id"] == "speaker_0"
        assert "SPEAKER_00" not in speakers

    def test_no_turns_clears_every_legacy_label(self, context):
        """No timeline means no labels: null them rather than publish another engine's."""
        from multimodal_pipeline.schemas import read_table
        from multimodal_pipeline.stages.speaker_assignment import SpeakerAssignmentStage

        _write_transcript(context, turns=False)
        context.state.mark_skipped("diarization_nemotron", "environment not installed")
        context.state.mark_completed("whisperx")
        stage = SpeakerAssignmentStage()
        enabled, reason = stage.enabled(context)
        assert enabled is True, reason
        stage.execute(context)

        segments = read_table(context.artifact("speech_segments")).to_pylist()
        words = read_table(context.artifact("speech_words")).to_pylist()
        assert [row["speaker_id"] for row in segments] == [None, None]
        assert [row["speaker_id"] for row in words] == [None]
        # The pyannote assignment's own audit columns go with the label.
        assert {row["speaker_assignment_method"] for row in segments} == {"no_diarization"}
        assert {row["speaker_overlap_ratio"] for row in segments} == {None}
        # Text and timing are untouched: this stage labels, it does not transcribe.
        assert [row["text"] for row in segments] == ["hola", "adios"]
        assert [row["end_time"] for row in segments] == [2.0, 4.0]
        assert stage.validate(context)["assigned_segments"] == 0

    def test_the_preserved_raw_transcript_stays_byte_identical(self, context):
        """The clear pass rewrites derived tables only — the native document is evidence."""
        from multimodal_pipeline.stages.speaker_assignment import SpeakerAssignmentStage

        _write_transcript(context, turns=False)
        raw = context.artifact("whisperx_raw")
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(b'{"language": "es", "segments": [{"speaker": "SPEAKER_00"}]}')
        before = raw.read_bytes()
        SpeakerAssignmentStage().execute(context)
        assert raw.read_bytes() == before

    def test_validate_rejects_a_label_left_without_a_timeline(self, context):
        """The invariant is checked, not assumed: a leftover label is a defect."""
        from multimodal_pipeline.exceptions import ValidationError
        from multimodal_pipeline.stages.speaker_assignment import SpeakerAssignmentStage

        _write_transcript(context, turns=False)
        with pytest.raises(ValidationError, match="no diarization timeline"):
            SpeakerAssignmentStage().validate(context)

    def test_a_disabled_diarizer_still_uses_a_cached_timeline(self, context):
        """The rule is "a compatible table on disk", not "the stage ran in this invocation".

        ``diarization_nemotron.enabled: false`` says "do not load the model", and with a
        timeline already in the dataset the labels it produced are still this pipeline's own.
        Clearing them because a flag was switched off would delete a measurement the config
        never asked to forget, so the cached table wins. Reversible alternative, if an
        operator ever wants a label-free transcript while keeping the table: add a
        ``diarization_nemotron.keep_labels: false`` switch rather than overloading ``enabled``.
        """
        from multimodal_pipeline.schemas import read_table
        from multimodal_pipeline.stages.speaker_assignment import SpeakerAssignmentStage

        _write_transcript(context, turns=True)
        context.config.diarization_nemotron.enabled = False
        stage = SpeakerAssignmentStage()
        enabled, reason = stage.enabled(context)
        assert enabled is True, reason
        stage.execute(context)
        assert {row["speaker_id"] for row in read_table(context.artifact("speech_segments")).to_pylist()} == {
            "speaker_0", "speaker_1"}

    def test_an_unreadable_turn_table_fails_loudly(self, context):
        """A present-but-broken timeline is a defect, never a reason to quietly clear."""
        from multimodal_pipeline.exceptions import StageError
        from multimodal_pipeline.stages.speaker_assignment import SpeakerAssignmentStage

        _write_transcript(context, turns=False)
        context.artifact("speaker_turns_nemotron").write_bytes(b"not parquet")
        with pytest.raises(StageError, match="speaker_turns_nemotron"):
            SpeakerAssignmentStage().execute(context)
