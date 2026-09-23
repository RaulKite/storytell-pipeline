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
        # Overlap is wall-clock time with concurrent speech (pyannote semantics),
        # not speaker-seconds above the single-speaker baseline.
        turns = normalise_turns([(0.0, 6.0, "A"), (2.0, 8.0, "B"), (4.0, 10.0, "C")])
        report = coverage_report(turns, duration=10.0)
        assert report["speaker_seconds"] == pytest.approx(10.0)
        assert report["overlap_seconds"] == pytest.approx(6.0)

    def test_empty_and_missing_duration(self) -> None:
        assert coverage_report([])["speaker_seconds"] == 0.0
        assert coverage_report(TURNS)["speaker_time"] == 0.0
