"""Shared validation helpers and the Parquet write layer.

These are the checks every stage leans on, so a subtle bug here means every
dataset is quietly less trustworthy than it looks. They are pure logic over real
Parquet files — no mocks, because a mocked pyarrow would test the mock.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from multimodal_pipeline.exceptions import ValidationError, ValidationIssue
from multimodal_pipeline.schemas import (
    ChunkedParquetWriter,
    FRAME_INDEX_SCHEMA,
    read_table,
    table_columns,
    table_rows,
    write_table,
)
from multimodal_pipeline.validation import (
    check_intervals,
    check_parquet,
    check_reference_values,
    validate_metadata_payload,
)


def make_table(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


class TestFiniteNumbers:
    """``acoustics.is_number`` guards every numeric column that reaches Parquet."""

    @pytest.mark.parametrize("value", [1, 0, -3, 1.5, 0.0, float("1e3")])
    def test_real_numbers_pass(self, value: Any) -> None:
        from multimodal_pipeline.acoustics import is_number

        assert is_number(value)

    @pytest.mark.parametrize("value", [
        None, True, False, float("nan"), float("inf"), float("-inf"),
        "120", "", [], {}, object(),
    ])
    def test_non_measurements_are_rejected(self, value: Any) -> None:
        """A ``voiced`` bool must never be averaged into a pitch statistic as 0/1."""
        from multimodal_pipeline.acoustics import is_number

        assert not is_number(value)


class TestCheckParquet:
    def test_a_good_table_reports_rows_and_columns(self, tmp_path: Path) -> None:
        path = make_table(tmp_path / "w.parquet",
                          [{"schema_version": "1.0", "start_time": 0.0},
                           {"schema_version": "1.0", "start_time": 1.0}])
        result = check_parquet(path, ["start_time"], stage="s", min_rows=2)
        assert result["rows"] == 2
        assert "schema_version" in result["columns"]

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationIssue, match="missing table"):
            check_parquet(tmp_path / "nope.parquet", ["a"], stage="s")

    def test_corrupt_file_is_reported_not_raised_raw(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.parquet"
        path.write_bytes(b"definitely not parquet")
        with pytest.raises(ValidationIssue, match="unreadable parquet"):
            check_parquet(path, ["a"], stage="s")

    def test_missing_column_names_the_table(self, tmp_path: Path) -> None:
        path = make_table(tmp_path / "w.parquet", [{"a": 1}])
        with pytest.raises(ValidationIssue, match="w.parquet missing columns: start_time"):
            check_parquet(path, ["start_time"], stage="s")

    def test_row_count_floor(self, tmp_path: Path) -> None:
        path = make_table(tmp_path / "w.parquet", [{"a": 1}])
        with pytest.raises(ValidationIssue, match="has 1 rows, expected >= 2"):
            check_parquet(path, ["a"], stage="s", min_rows=2)

    def test_an_empty_table_passes_when_allowed(self, tmp_path: Path) -> None:
        path = make_table(tmp_path / "w.parquet", [])
        assert check_parquet(path, [], stage="s")["rows"] == 0

    def test_negative_timestamps(self, tmp_path: Path) -> None:
        path = make_table(tmp_path / "w.parquet",
                          [{"schema_version": "1.0", "start_time": -0.5}])
        with pytest.raises(ValidationIssue, match="negative timestamps"):
            check_parquet(path, ["start_time"], stage="s", time_column="start_time")

    def test_a_timestamp_at_exactly_zero_is_fine(self, tmp_path: Path) -> None:
        path = make_table(tmp_path / "w.parquet",
                          [{"schema_version": "1.0", "start_time": 0.0}])
        check_parquet(path, ["start_time"], stage="s", time_column="start_time")

    def test_beyond_the_source_duration(self, tmp_path: Path) -> None:
        path = make_table(tmp_path / "w.parquet",
                          [{"schema_version": "1.0", "start_time": 99.0}])
        with pytest.raises(ValidationIssue, match="exceeds source duration"):
            check_parquet(path, ["start_time"], stage="s",
                          time_column="start_time", max_time=10.0)

    def test_duration_tolerance_absorbs_frame_rounding(self, tmp_path: Path) -> None:
        """A last-frame timestamp can land a hair past the reported duration."""
        path = make_table(tmp_path / "w.parquet",
                          [{"schema_version": "1.0", "start_time": 10.4}])
        check_parquet(path, ["start_time"], stage="s",
                      time_column="start_time", max_time=10.0)

    def test_out_of_order_timestamps(self, tmp_path: Path) -> None:
        path = make_table(tmp_path / "w.parquet",
                          [{"schema_version": "1.0", "start_time": t} for t in (0.0, 2.0, 1.0)])
        with pytest.raises(ValidationIssue, match="not monotonically ordered"):
            check_parquet(path, ["start_time"], stage="s",
                          time_column="start_time", ordered=True)

    def test_unordered_is_allowed_when_declared(self, tmp_path: Path) -> None:
        path = make_table(tmp_path / "w.parquet",
                          [{"schema_version": "1.0", "start_time": t} for t in (2.0, 0.0)])
        check_parquet(path, ["start_time"], stage="s",
                      time_column="start_time", ordered=False)

    def test_time_checks_need_a_versioned_schema(self, tmp_path: Path) -> None:
        """Tables without ``schema_version`` are foreign; do not police their times."""
        path = make_table(tmp_path / "w.parquet", [{"start_time": -5.0}])
        check_parquet(path, ["start_time"], stage="s", time_column="start_time")

    def test_null_timestamps_are_ignored(self, tmp_path: Path) -> None:
        """WhisperX emits null timings for unaligned words; that is not corruption."""
        path = make_table(tmp_path / "w.parquet",
                          [{"schema_version": "1.0", "start_time": None},
                           {"schema_version": "1.0", "start_time": 1.0}])
        check_parquet(path, ["start_time"], stage="s", time_column="start_time",
                      max_time=5.0)


class TestCheckIntervals:
    def test_valid_intervals_are_returned(self) -> None:
        rows = check_intervals([0.0, 1.0], [0.5, 2.0], stage="s", label="seg")
        assert rows == [{"start_time": 0.0, "end_time": 0.5},
                        {"start_time": 1.0, "end_time": 2.0}]

    def test_inverted_interval(self) -> None:
        with pytest.raises(ValidationIssue, match=r"seg\[1\] ends before it starts"):
            check_intervals([0.0, 5.0], [1.0, 4.0], stage="s", label="seg")

    def test_zero_length_interval_is_valid(self) -> None:
        """A single-frame word legitimately has start == end."""
        check_intervals([1.0], [1.0], stage="s", label="seg")

    def test_start_before_zero(self) -> None:
        with pytest.raises(ValidationIssue, match=r"seg\[0\] starts before t=0"):
            check_intervals([-0.2], [1.0], stage="s", label="seg")

    def test_start_beyond_the_duration(self) -> None:
        with pytest.raises(ValidationIssue, match="starts after source duration"):
            check_intervals([50.0], [51.0], stage="s", label="seg", max_time=10.0)

    def test_missing_times_are_skipped_not_flagged(self) -> None:
        """A diarization row without timing is incomplete, not invalid."""
        check_intervals([None, 0.0], [None, 1.0], stage="s", label="seg")

    def test_issue_list_is_capped(self) -> None:
        """A pathological table reports a bounded list, not 50k strings."""
        with pytest.raises(ValidationIssue) as excinfo:
            check_intervals([float(i) for i in range(100)], [0.0] * 100,
                            stage="s", label="seg")
        assert len(excinfo.value.issues) == 20


class TestCheckReferences:
    def test_known_values_pass(self) -> None:
        check_reference_values(["a", "b", None], {"a", "b"}, stage="s", label="id")

    def test_unknown_values_are_sampled(self) -> None:
        with pytest.raises(ValidationIssue, match="id references unknown values: z"):
            check_reference_values(["a", "z"], {"a"}, stage="s", label="id")

    def test_sample_is_bounded_and_sorted(self) -> None:
        values = [f"id{i:03d}" for i in range(50)]
        with pytest.raises(ValidationIssue) as excinfo:
            check_reference_values(values, set(), stage="s", label="id")
        listed = excinfo.value.issues[0].split(": ", 1)[1].split(", ")
        assert len(listed) == 8
        assert listed == sorted(listed)

    def test_empty_reference_set_catches_everything(self) -> None:
        with pytest.raises(ValidationIssue):
            check_reference_values(["x"], set(), stage="s", label="id")


class TestMetadataPayload:
    GOOD = {
        "schema_version": "1.0", "video_id": "v", "source_filename": "v.mp4",
        "source_path": "/in/v.mp4", "SHA256": "a" * 64, "file_size_bytes": 10,
        "duration_seconds": 5.0, "width": 640, "height": 480, "video_codec": "h264",
    }

    def test_good_payload_returns_a_summary(self) -> None:
        assert validate_metadata_payload(dict(self.GOOD)) == {
            "duration_seconds": 5.0, "width": 640, "height": 480}

    @pytest.mark.parametrize("field", ["schema_version", "video_id", "source_filename",
                                       "source_path", "SHA256", "file_size_bytes",
                                       "duration_seconds"])
    def test_every_required_field_is_required(self, field: str) -> None:
        payload = dict(self.GOOD)
        payload[field] = None
        with pytest.raises(ValidationIssue, match=field):
            validate_metadata_payload(payload)

    def test_zero_duration_is_rejected(self) -> None:
        with pytest.raises(ValidationIssue, match="duration_seconds must be > 0"):
            validate_metadata_payload({**self.GOOD, "duration_seconds": 0})

    @pytest.mark.parametrize("wh", [{"width": 0}, {"height": 0}, {"width": -1}])
    def test_bad_geometry(self, wh: dict) -> None:
        with pytest.raises(ValidationIssue, match="width/height"):
            validate_metadata_payload({**self.GOOD, **wh})

    def test_an_audio_only_file_is_accepted(self) -> None:
        """A podcast-style upload has no video stream but is still processable."""
        payload = {k: v for k, v in self.GOOD.items() if k != "video_codec"}
        payload["audio_codec"] = "aac"
        validate_metadata_payload(payload)

    def test_a_file_with_no_streams_at_all_is_rejected(self) -> None:
        payload = {k: v for k, v in self.GOOD.items() if k != "video_codec"}
        with pytest.raises(ValidationIssue, match="no media stream"):
            validate_metadata_payload(payload)

    def test_all_problems_are_reported_together(self) -> None:
        """One edit should fix the whole file, not one error per run."""
        payload = {**self.GOOD, "duration_seconds": -1, "width": 0, "video_codec": None}
        payload.pop("SHA256")
        with pytest.raises(ValidationIssue) as excinfo:
            validate_metadata_payload(payload)
        assert len(excinfo.value.issues) == 4

    def test_issue_carries_the_stage_name(self) -> None:
        with pytest.raises(ValidationIssue) as excinfo:
            validate_metadata_payload({"duration_seconds": 1}, stage="finalization")
        assert excinfo.value.stage == "finalization"


class TestValidationErrorShape:
    def test_message_includes_stage_and_issues(self) -> None:
        error = ValidationError("whisperx", ["bad", "worse"])
        assert "whisperx" in str(error)
        assert "bad" in str(error) and "worse" in str(error)

    def test_issues_are_readable(self) -> None:
        assert ValidationIssue("s", ["x"]).issues == ["x"]


class TestWriteTable:
    def test_round_trip_through_the_declared_schema(self, tmp_path: Path) -> None:
        rows = [{"schema_version": "1.0", "video_id": "v", "frame_number": i,
                 "pts_seconds": i / 25.0} for i in range(3)]
        path = tmp_path / "frame_index.parquet"
        write_table(path, pa.Table.from_pylist(rows), FRAME_INDEX_SCHEMA)
        assert read_table(path).to_pylist() == rows

    def test_absent_columns_become_nulls(self, tmp_path: Path) -> None:
        """A producer that omits an optional column still satisfies the schema."""
        path = tmp_path / "t.parquet"
        write_table(path, pa.Table.from_pylist([{"frame_number": 0, "pts_seconds": 0.0}]),
                    FRAME_INDEX_SCHEMA)
        row = read_table(path).to_pylist()[0]
        assert row["video_id"] is None and row["frame_number"] == 0

    def test_columns_are_reordered_to_the_schema(self, tmp_path: Path) -> None:
        path = tmp_path / "t.parquet"
        table = pa.Table.from_pylist([{"pts_seconds": 1.0, "frame_number": 1,
                                       "video_id": "v", "schema_version": "1.0"}])
        write_table(path, table, FRAME_INDEX_SCHEMA)
        assert list(table_columns(path)) == list(FRAME_INDEX_SCHEMA.names)

    def test_parent_directories_are_created(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "t.parquet"
        write_table(path, pa.Table.from_pylist([]).slice(0, 0), FRAME_INDEX_SCHEMA)
        assert path.is_file()

    def test_extra_metadata_reaches_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "t.parquet"
        write_table(path, pa.Table.from_pylist([]).slice(0, 0), FRAME_INDEX_SCHEMA,
                    extra_metadata={"origin": "test"})
        assert pq.read_schema(path).metadata[b"origin"] == b"test"

    def test_row_count_helper(self, tmp_path: Path) -> None:
        path = tmp_path / "t.parquet"
        write_table(path, pa.Table.from_pylist([{"frame_number": i, "pts_seconds": 0.0,
                                                 "video_id": "v", "schema_version": "1.0"}
                                                for i in range(4)]), FRAME_INDEX_SCHEMA)
        assert table_rows(path) == 4

    def test_column_projection(self, tmp_path: Path) -> None:
        path = tmp_path / "t.parquet"
        write_table(path, pa.Table.from_pylist([{"frame_number": 0, "pts_seconds": 0.0,
                                                 "video_id": "v", "schema_version": "1.0"}]),
                    FRAME_INDEX_SCHEMA)
        assert read_table(path, columns=["pts_seconds"]).column_names == ["pts_seconds"]


class TestChunkedWriter:
    """The writer exists to keep a million-row pose table inside RAM."""

    def schema(self):
        return FRAME_INDEX_SCHEMA

    def test_rows_are_flushed_in_groups(self, tmp_path: Path) -> None:
        path = tmp_path / "t.parquet"
        writer = ChunkedParquetWriter(path, self.schema(), rows_per_group=2)
        writer.extend({"frame_number": i, "pts_seconds": float(i), "video_id": "v",
                       "schema_version": "1.0"} for i in range(5))
        assert writer.close() == 5
        assert pq.ParquetFile(path).num_row_groups == 3  # 2 + 2 + 1
        assert table_rows(path) == 5

    def test_memory_stays_bounded_between_flushes(self, tmp_path: Path) -> None:
        writer = ChunkedParquetWriter(tmp_path / "t.parquet", self.schema(), rows_per_group=3)
        for index in range(10):
            writer.add({"frame_number": index, "pts_seconds": 0.0, "video_id": "v",
                        "schema_version": "1.0"})
            assert len(writer.rows) <= 3
        writer.close()

    def test_an_unwritten_writer_still_produces_a_file(self, tmp_path: Path) -> None:
        """Zero detections is a real result; the artifact must exist and be empty."""
        path = tmp_path / "empty.parquet"
        writer = ChunkedParquetWriter(path, self.schema())
        assert writer.close() == 0
        assert path.is_file()
        assert table_rows(path) == 0
        assert list(table_columns(path)) == list(self.schema().names)

    def test_flushing_an_empty_buffer_is_a_no_op(self, tmp_path: Path) -> None:
        writer = ChunkedParquetWriter(tmp_path / "t.parquet", self.schema())
        writer.flush()
        assert not (tmp_path / "t.parquet").exists()

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        writer = ChunkedParquetWriter(tmp_path / "t.parquet", self.schema())
        writer.add({"frame_number": 0, "pts_seconds": 0.0, "video_id": "v",
                    "schema_version": "1.0"})
        assert writer.close() == 1
        assert writer.close() == 1

    def test_missing_optional_columns_become_nulls(self, tmp_path: Path) -> None:
        path = tmp_path / "t.parquet"
        writer = ChunkedParquetWriter(path, self.schema())
        writer.add({"frame_number": 1, "pts_seconds": 0.0})
        writer.close()
        assert read_table(path).to_pylist()[0]["video_id"] is None

    def test_metadata_survives_the_first_flush(self, tmp_path: Path) -> None:
        path = tmp_path / "t.parquet"
        writer = ChunkedParquetWriter(path, self.schema(), rows_per_group=1,
                                      extra_metadata={"frames": "2"})
        writer.add({"frame_number": 0, "pts_seconds": 0.0, "video_id": "v",
                    "schema_version": "1.0"})
        writer.add({"frame_number": 1, "pts_seconds": 0.0, "video_id": "v",
                    "schema_version": "1.0"})
        writer.close()
        assert pq.read_schema(path).metadata[b"frames"] == b"2"

    def test_values_survive_a_round_trip_exactly(self, tmp_path: Path) -> None:
        """Timestamps must not drift through buffering — they are the timeline."""
        path = tmp_path / "t.parquet"
        times = [1 / 3, 2 / 3, 1.0 / 25.0, 9999.9999]
        writer = ChunkedParquetWriter(path, self.schema(), rows_per_group=1)
        writer.extend({"frame_number": i, "pts_seconds": t, "video_id": "v",
                       "schema_version": "1.0"} for i, t in enumerate(times))
        writer.close()
        assert [row["pts_seconds"] for row in read_table(path).to_pylist()] == times
