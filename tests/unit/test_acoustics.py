"""Acoustic aggregation maths: voiced-only statistics, pauses, null policy."""

from __future__ import annotations

import math

import pytest

from multimodal_pipeline.acoustics import (
    aggregate_segment,
    clip_silences,
    detect_silences,
    frame_in_segment,
    frame_is_quiet,
    interval_intersections,
    is_number,
    normalise_frame_row,
    summarise,
    voiced_ratio,
)


def frame(t: float, *, f0: float | None = None, db: float | None = None, voiced: bool | None = None,
          f1: float | None = None, f2: float | None = None, f3: float | None = None) -> dict:
    return {"timestamp": t, "f0_hz": f0, "intensity_db": db, "voiced": voiced,
            "f1_hz": f1, "f2_hz": f2, "f3_hz": f3}


class TestNumberGuard:
    @pytest.mark.parametrize("value,expected", [
        (1, True), (1.5, True), (0, True), (-3.2, True),
        (None, False), (float("nan"), False), (float("inf"), False),
        ("1.5", False), (True, False), ({}, False),
    ])
    def test_is_number(self, value, expected: bool) -> None:
        assert is_number(value) is expected


class TestSummarise:
    def test_basic_statistics(self) -> None:
        result = summarise([10.0, 20.0, 30.0])
        assert result == {"mean": 20.0, "median": 20.0, "min": 10.0, "max": 30.0,
                          "std": round(math.sqrt(200 / 3), 6)}

    def test_even_count_median_is_the_middle_pair(self) -> None:
        assert summarise([1.0, 2.0, 4.0, 8.0])["median"] == 3.0

    def test_single_value_has_zero_std(self) -> None:
        result = summarise([440.0])
        assert result["std"] == 0.0 and result["mean"] == 440.0

    def test_empty_is_all_null_not_zero(self) -> None:
        assert summarise([]) == {"mean": None, "median": None, "min": None, "max": None, "std": None}

    def test_nulls_and_nan_are_skipped_not_counted_as_zero(self) -> None:
        result = summarise([100.0, None, float("nan"), 300.0])
        assert result["mean"] == 200.0 and result["min"] == 100.0

    def test_all_null_input_is_no_measurement(self) -> None:
        assert summarise([None, None])["mean"] is None

    def test_population_not_sample_stdev(self) -> None:
        # Sample stdev of [2, 4] is sqrt(2)≈1.414; population stdev is 1.0.
        assert summarise([2.0, 4.0])["std"] == pytest.approx(1.0)

    def test_negative_decibels_survive(self) -> None:
        assert summarise([-60.0, -30.0])["mean"] == -45.0


class TestVoicedRatio:
    def test_fraction_of_voiced_frames(self) -> None:
        frames = [frame(0.0, voiced=True), frame(0.01, voiced=False),
                  frame(0.02, voiced=True), frame(0.03, voiced=False)]
        assert voiced_ratio(frames) == pytest.approx(0.5)

    def test_all_unvoiced_is_zero_not_null(self) -> None:
        assert voiced_ratio([frame(0.0, voiced=False)]) == 0.0

    def test_no_frames_is_unknown(self) -> None:
        assert voiced_ratio([]) is None

    def test_null_voicing_counts_as_unvoiced(self) -> None:
        assert voiced_ratio([frame(0.0, voiced=None), frame(0.01, voiced=True)]) == pytest.approx(0.5)


class TestSilenceDetection:
    def test_contiguous_unvoiced_run_becomes_one_pause(self) -> None:
        frames = [frame(0.0, voiced=True), frame(0.1, voiced=False), frame(0.2, voiced=False),
                  frame(0.3, voiced=True)]
        assert detect_silences(frames, minimum_duration=0.1, frame_step=0.1) == [(0.1, 0.3)]

    def test_short_blips_are_dropped(self) -> None:
        frames = [frame(0.0, voiced=True), frame(0.1, voiced=False), frame(0.2, voiced=True)]
        assert detect_silences(frames, minimum_duration=0.5, frame_step=0.1) == []

    def test_two_separate_pauses(self) -> None:
        frames = [frame(0.0, voiced=False), frame(0.1, voiced=False), frame(0.2, voiced=True),
                  frame(0.3, voiced=True), frame(0.4, voiced=False), frame(0.5, voiced=False)]
        pauses = detect_silences(frames, minimum_duration=0.1, frame_step=0.1)
        assert pauses == [(0.0, 0.2), (0.4, 0.6)]

    def test_edge_pauses_are_kept(self) -> None:
        frames = [frame(0.0, voiced=False), frame(0.1, voiced=False), frame(0.2, voiced=True)]
        assert detect_silences(frames, minimum_duration=0.1, frame_step=0.1) == [(0.0, 0.2)]

    def test_trailing_pause_is_kept(self) -> None:
        frames = [frame(0.0, voiced=True), frame(0.1, voiced=False), frame(0.2, voiced=False)]
        assert detect_silences(frames, minimum_duration=0.1, frame_step=0.1) == [(0.1, 0.3)]

    def test_intensity_threshold_overrides_voicing(self) -> None:
        # Loud but unvoiced frames (fricatives) are NOT silence when a dB floor is set.
        frames = [frame(0.0, db=-20.0, voiced=False), frame(0.1, db=-25.0, voiced=False)]
        assert detect_silences(frames, minimum_duration=0.1, silence_threshold_db=-60.0,
                              frame_step=0.1) == []

    def test_quiet_but_voiced_is_silence_when_a_threshold_is_set(self) -> None:
        frames = [frame(0.0, db=-80.0, voiced=True), frame(0.1, db=-85.0, voiced=True)]
        assert detect_silences(frames, minimum_duration=0.1, silence_threshold_db=-60.0,
                              frame_step=0.1) == [(0.0, 0.2)]

    def test_frame_without_intensity_falls_back_to_voicing(self) -> None:
        frames = [frame(0.0, db=None, voiced=False), frame(0.1, db=None, voiced=False)]
        assert detect_silences(frames, minimum_duration=0.1, silence_threshold_db=-60.0,
                              frame_step=0.1) == [(0.0, 0.2)]

    def test_step_is_inferred_when_not_given(self) -> None:
        frames = [frame(0.0, voiced=True), frame(0.05, voiced=False), frame(0.10, voiced=False),
                  frame(0.15, voiced=True)]
        assert detect_silences(frames, minimum_duration=0.08) == [(0.05, 0.15)]

    def test_no_frames(self) -> None:
        assert detect_silences([], minimum_duration=0.1) == []

    def test_frame_is_quiet_semantics(self) -> None:
        assert frame_is_quiet(frame(0.0, voiced=False), silence_threshold_db=None) is True
        assert frame_is_quiet(frame(0.0, voiced=True), silence_threshold_db=None) is False
        assert frame_is_quiet(frame(0.0, db=-70.0, voiced=True), silence_threshold_db=-60.0) is True
        assert frame_is_quiet(frame(0.0, db=-10.0, voiced=False), silence_threshold_db=-60.0) is False


class TestIntervalMaths:
    def test_intersections_per_interval(self) -> None:
        result = interval_intersections([(0.0, 10.0), (20.0, 30.0)], [(5.0, 15.0), (25.0, 26.0)])
        assert result == [5.0, 1.0]

    def test_no_overlap_is_zero_not_null(self) -> None:
        assert interval_intersections([(0.0, 1.0)], [(5.0, 6.0)]) == [0.0]

    def test_clip_silences_bounds_the_ratio(self) -> None:
        clipped = clip_silences([(0.0, 100.0)], 10.0, 12.0)
        assert clipped == [(10.0, 12.0)]

    def test_clip_drops_outside_silences(self) -> None:
        assert clip_silences([(0.0, 5.0)], 10.0, 20.0) == []

    def test_clip_with_open_bounds(self) -> None:
        assert clip_silences([(1.0, 2.0)], None, None) == [(1.0, 2.0)]

    def test_frame_in_segment_boundaries_are_inclusive(self) -> None:
        assert frame_in_segment(frame(5.0), 5.0, 10.0) is True
        assert frame_in_segment(frame(10.0), 5.0, 10.0) is True
        assert frame_in_segment(frame(4.99), 5.0, 10.0) is False
        assert frame_in_segment(frame(10.01), 5.0, 10.0) is False

    def test_frame_in_segment_with_missing_bounds(self) -> None:
        assert frame_in_segment(frame(5.0), None, None) is True

    def test_frame_without_timestamp_is_never_inside(self) -> None:
        assert frame_in_segment({"timestamp": None}, 0.0, 10.0) is False


class TestAggregateSegment:
    SEGMENT = {"segment_id": "seg000001", "start_time": 1.0, "end_time": 3.0, "speaker_id": "S1"}

    def frames(self) -> list[dict]:
        return [
            frame(0.5, f0=100.0, db=-30.0, voiced=True, f1=400.0, f2=1500.0, f3=2500.0),   # before
            frame(1.0, f0=200.0, db=-25.0, voiced=True, f1=500.0, f2=1600.0, f3=2600.0),
            frame(1.5, f0=300.0, db=-20.0, voiced=True, f1=600.0, f2=1700.0, f3=2700.0),
            frame(2.0, f0=None, db=-40.0, voiced=False, f1=None, f2=None, f3=None),
            frame(3.0, f0=400.0, db=-15.0, voiced=True, f1=700.0, f2=1800.0, f3=2800.0),
            frame(3.5, f0=999.0, db=-10.0, voiced=True, f1=9999.0, f2=9999.0, f3=9999.0),  # after
        ]

    def test_only_frames_inside_the_segment_are_used(self) -> None:
        result = aggregate_segment(self.SEGMENT, self.frames(), silences=[])
        assert result.values["f0_mean"] == pytest.approx(300.0)  # 200+300+400, not 100 or 999
        assert result.values["f0_min"] == 200.0 and result.values["f0_max"] == 400.0
        assert result.values["f1_mean"] == pytest.approx(600.0)

    def test_unvoiced_frames_do_not_enter_pitch_statistics(self) -> None:
        result = aggregate_segment(self.SEGMENT, self.frames(), silences=[])
        # The unvoiced frame at t=2.0 must not drag the mean toward zero.
        assert result.values["f0_mean"] == pytest.approx(300.0)
        assert result.values["voiced_ratio"] == pytest.approx(0.75)  # 3 of 4 inside frames

    def test_intensity_includes_unvoiced_frames(self) -> None:
        result = aggregate_segment(self.SEGMENT, self.frames(), silences=[])
        assert result.values["intensity_mean"] == pytest.approx(-25.0)  # -25,-20,-40,-15

    def test_segment_without_voiced_frames_has_null_pitch(self) -> None:
        frames = [frame(1.0, f0=None, db=-70.0, voiced=False), frame(2.0, f0=None, db=-80.0, voiced=False)]
        result = aggregate_segment(self.SEGMENT, frames, silences=[])
        assert result.values["f0_mean"] is None
        assert result.values["f0_std"] is None
        assert result.values["intensity_mean"] == pytest.approx(-75.0)
        assert result.values["voiced_ratio"] == 0.0

    def test_pause_metrics_are_clipped_to_the_segment(self) -> None:
        result = aggregate_segment(self.SEGMENT, self.frames(), silences=[(2.0, 2.5), (50.0, 60.0)])
        assert result.values["pause_count"] == 1
        assert result.values["pause_duration"] == pytest.approx(0.5)
        assert result.values["pause_ratio"] == pytest.approx(0.25)

    def test_pause_ratio_cannot_exceed_one(self) -> None:
        result = aggregate_segment(self.SEGMENT, self.frames(), silences=[(-100.0, 100.0)])
        assert result.values["pause_ratio"] == pytest.approx(1.0)

    def test_no_silences(self) -> None:
        result = aggregate_segment(self.SEGMENT, self.frames(), silences=[])
        assert result.values["pause_count"] == 0
        assert result.values["pause_duration"] == 0.0
        assert result.values["pause_ratio"] == pytest.approx(0.0)

    def test_duration_is_derived_from_the_interval(self) -> None:
        assert aggregate_segment(self.SEGMENT, [], silences=[]).values["duration"] == pytest.approx(2.0)

    def test_explicit_duration_wins(self) -> None:
        result = aggregate_segment(self.SEGMENT, [], silences=[], duration=10.0)
        assert result.values["duration"] == 10.0

    def test_identity_fields_pass_through(self) -> None:
        row = aggregate_segment(self.SEGMENT, [], silences=[]).as_row()
        assert row["segment_id"] == "seg000001"
        assert row["speaker_id"] == "S1"
        assert row["start_time"] == 1.0 and row["end_time"] == 3.0

    def test_overlapping_segments_are_aggregated_independently(self) -> None:
        left = {"segment_id": "a", "start_time": 0.0, "end_time": 2.0}
        right = {"segment_id": "b", "start_time": 1.0, "end_time": 3.0}
        frames = [frame(0.5, f0=100.0, db=-30.0, voiced=True), frame(1.5, f0=300.0, db=-30.0, voiced=True)]
        assert aggregate_segment(left, frames, silences=[]).values["f0_mean"] == pytest.approx(200.0)
        assert aggregate_segment(right, frames, silences=[]).values["f0_mean"] == pytest.approx(300.0)

    def test_missing_bounds_yield_nulls_not_crashes(self) -> None:
        row = aggregate_segment({"segment_id": "x"}, self.frames(), silences=[]).as_row()
        assert row["start_time"] is None and row["end_time"] is None
        assert row["pause_ratio"] is None

    def test_zero_duration_segment_has_null_pause_ratio(self) -> None:
        segment = {"segment_id": "x", "start_time": 5.0, "end_time": 5.0}
        assert aggregate_segment(segment, [], silences=[]).values["pause_ratio"] is None


class TestFrameRowNormalisation:
    def test_full_record(self) -> None:
        row = normalise_frame_row([1.5, 200.0, -25.0, True, 500.0, 1500.0, 2500.0], video_id="v")
        assert row["timestamp"] == 1.5 and row["f0_hz"] == 200.0
        assert row["voiced"] is True and row["video_id"] == "v"
        assert row["schema_version"] == "1.0"

    def test_nulls_stay_null(self) -> None:
        row = normalise_frame_row([1.5, None, None, False, None, None, None], video_id="v")
        assert row["f0_hz"] is None and row["intensity_db"] is None
        assert row["voiced"] is False

    def test_nan_from_praat_becomes_null(self) -> None:
        row = normalise_frame_row([1.5, float("nan"), float("nan"), False, float("nan"),
                                   float("nan"), float("nan")], video_id="v")
        assert row["f0_hz"] is None and row["f1_hz"] is None

    def test_infinity_becomes_null(self) -> None:
        assert normalise_frame_row([1.5, float("inf"), None, False, None, None, None],
                                   video_id="v")["f0_hz"] is None

    def test_short_record_is_padded_with_nulls(self) -> None:
        row = normalise_frame_row([1.5, 200.0], video_id="v")
        assert row["f0_hz"] == 200.0 and row["f3_hz"] is None and row["voiced"] is None

    def test_every_column_is_present(self) -> None:
        row = normalise_frame_row([0.0], video_id="v")
        assert set(row) == {"schema_version", "video_id", "timestamp", "f0_hz", "intensity_db",
                            "voiced", "f1_hz", "f2_hz", "f3_hz"}

    def test_schema_version_is_overridable(self) -> None:
        assert normalise_frame_row([0.0], video_id="v", schema_version="2.0")["schema_version"] == "2.0"

    def test_a_bad_timestamp_yields_null_rather_than_raising(self) -> None:
        assert normalise_frame_row(["nope"], video_id="v")["timestamp"] is None
