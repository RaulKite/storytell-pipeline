"""Explicit Parquet schemas for every normalised table.

Column order and dtype are pinned here so downstream consumers (PyArrow,
Polars, DuckDB, pandas) always see the same types, and so ``validate`` can
check a table's structure instead of trusting the writer.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

# --- speech ---------------------------------------------------------------

SEGMENTS_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("segment_id", pa.string()),
        ("start_time", pa.float64()),
        ("end_time", pa.float64()),
        ("duration", pa.float64()),
        ("language", pa.string()),
        ("speaker_id", pa.string()),
        ("text", pa.string()),
        ("confidence", pa.float64()),
        ("speaker_overlap_seconds", pa.float64()),
        ("speaker_overlap_ratio", pa.float64()),
        ("speaker_assignment_method", pa.string()),
    ]
)

WORDS_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("segment_id", pa.string()),
        ("word_id", pa.string()),
        ("start_time", pa.float64()),
        ("end_time", pa.float64()),
        ("duration", pa.float64()),
        ("speaker_id", pa.string()),
        ("word", pa.string()),
        ("confidence", pa.float64()),
        ("alignment_status", pa.string()),
        ("character_start", pa.int64()),
        ("character_end", pa.int64()),
        ("speaker_overlap_seconds", pa.float64()),
        ("speaker_overlap_ratio", pa.float64()),
        ("speaker_assignment_method", pa.string()),
    ]
)

SPEAKER_TURNS_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("turn_id", pa.string()),
        ("speaker_id", pa.string()),
        ("start_time", pa.float64()),
        ("end_time", pa.float64()),
        ("duration", pa.float64()),
        ("diarization_type", pa.string()),
    ]
)

# --- translation ----------------------------------------------------------

TRANSLATION_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("segment_id", pa.string()),
        ("speaker_id", pa.string()),
        ("start_time", pa.float64()),
        ("end_time", pa.float64()),
        ("source_language", pa.string()),
        ("source_text", pa.string()),
        ("english_text", pa.string()),
        ("translation_model", pa.string()),
        ("translation_prompt_version", pa.string()),
    ]
)

# --- linguistic -----------------------------------------------------------

TOKENS_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("variant", pa.string()),  # source | english
        ("segment_id", pa.string()),
        ("sentence_id", pa.string()),
        ("token_id", pa.string()),
        ("token_index", pa.int64()),
        ("speaker_id", pa.string()),
        ("text", pa.string()),
        ("lower", pa.string()),
        ("lemma", pa.string()),
        ("pos", pa.string()),
        ("tag", pa.string()),
        ("morph", pa.string()),
        ("dep", pa.string()),
        ("head_token_id", pa.string()),
        ("head_text", pa.string()),
        ("head_pos", pa.string()),
        ("ent_type", pa.string()),
        ("is_alpha", pa.bool_()),
        ("is_stop", pa.bool_()),
        ("is_digit", pa.bool_()),
        ("like_num", pa.bool_()),
        ("shape", pa.string()),
        ("char_start", pa.int64()),
        ("char_end", pa.int64()),
        ("segment_start_time", pa.float64()),
        ("segment_end_time", pa.float64()),
        ("token_start_time", pa.float64()),
        ("token_end_time", pa.float64()),
        ("timestamp_alignment_status", pa.string()),
        ("timestamp_alignment_confidence", pa.float64()),
    ]
)

SENTENCES_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("variant", pa.string()),
        ("segment_id", pa.string()),
        ("sentence_id", pa.string()),
        ("sentence_index", pa.int64()),
        ("speaker_id", pa.string()),
        ("text", pa.string()),
        ("token_count", pa.int64()),
        ("char_start", pa.int64()),
        ("char_end", pa.int64()),
        ("segment_start_time", pa.float64()),
        ("segment_end_time", pa.float64()),
    ]
)

# --- acoustic -------------------------------------------------------------

ACOUSTIC_FRAMES_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("timestamp", pa.float64()),
        ("f0_hz", pa.float64()),
        ("intensity_db", pa.float64()),
        ("voiced", pa.bool_()),
        ("f1_hz", pa.float64()),
        ("f2_hz", pa.float64()),
        ("f3_hz", pa.float64()),
    ]
)

ACOUSTIC_SEGMENTS_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("segment_id", pa.string()),
        ("speaker_id", pa.string()),
        ("start_time", pa.float64()),
        ("end_time", pa.float64()),
        ("duration", pa.float64()),
        ("voiced_ratio", pa.float64()),
        ("f0_mean", pa.float64()),
        ("f0_median", pa.float64()),
        ("f0_min", pa.float64()),
        ("f0_max", pa.float64()),
        ("f0_std", pa.float64()),
        ("intensity_mean", pa.float64()),
        ("intensity_median", pa.float64()),
        ("intensity_min", pa.float64()),
        ("intensity_max", pa.float64()),
        ("intensity_std", pa.float64()),
        ("f1_mean", pa.float64()),
        ("f2_mean", pa.float64()),
        ("f3_mean", pa.float64()),
        ("pause_count", pa.int64()),
        ("pause_duration", pa.float64()),
        ("pause_ratio", pa.float64()),
    ]
)

# --- pose -----------------------------------------------------------------

BODY_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("frame_number", pa.int64()),
        ("timestamp", pa.float64()),
        ("detection_index", pa.int64()),
        ("keypoint_id", pa.int64()),
        ("keypoint_name", pa.string()),
        ("x", pa.float64()),
        ("y", pa.float64()),
        ("confidence", pa.float64()),
    ]
)

HANDS_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("frame_number", pa.int64()),
        ("timestamp", pa.float64()),
        ("detection_index", pa.int64()),
        ("hand", pa.string()),
        ("keypoint_id", pa.int64()),
        ("keypoint_name", pa.string()),
        ("x", pa.float64()),
        ("y", pa.float64()),
        ("confidence", pa.float64()),
    ]
)

FACE_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("frame_number", pa.int64()),
        ("timestamp", pa.float64()),
        ("detection_index", pa.int64()),
        ("landmark_id", pa.int64()),
        ("x", pa.float64()),
        ("y", pa.float64()),
        ("confidence", pa.float64()),
    ]
)

# --- media ----------------------------------------------------------------

FRAME_INDEX_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("frame_number", pa.int64()),
        ("pts_seconds", pa.float64()),
    ]
)

TABLE_SCHEMAS: dict[str, pa.schema] = {
    "speech_segments": SEGMENTS_SCHEMA,
    "speech_words": WORDS_SCHEMA,
    "speaker_turns": SPEAKER_TURNS_SCHEMA,
    "translation_segments": TRANSLATION_SCHEMA,
    "linguistic_source_tokens": TOKENS_SCHEMA,
    "linguistic_source_sentences": SENTENCES_SCHEMA,
    "linguistic_english_tokens": TOKENS_SCHEMA,
    "linguistic_english_sentences": SENTENCES_SCHEMA,
    "acoustic_frames": ACOUSTIC_FRAMES_SCHEMA,
    "acoustic_segments": ACOUSTIC_SEGMENTS_SCHEMA,
    "pose_body": BODY_SCHEMA,
    "pose_hands": HANDS_SCHEMA,
    "pose_face": FACE_SCHEMA,
    "frame_index": FRAME_INDEX_SCHEMA,
}

# Standard OpenPose BODY_25 landmark names (verified against
# /opt/openpose/src/openpose/pose/poseParameters.cpp POSE_BODY_25_BODY_PARTS).
BODY_25_KEYPOINT_NAMES: tuple[str, ...] = (
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow",
    "LWrist", "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel",
    "RBigToe", "RSmallToe", "RHeel", "Background",
)

# OpenPose outputs 21 hand keypoints; names follow the official hand model.
HAND_KEYPOINT_NAMES: tuple[str, ...] = (
    "Wrist", "TH1", "TH2", "TH3", "TH4", "Index1", "Index2", "Index3", "Index4",
    "Middle1", "Middle2", "Middle3", "Middle4", "Ring1", "Ring2", "Ring3",
    "Ring4", "Pinky1", "Pinky2", "Pinky3", "Pinky4",
)

FACE_KEYPOINT_COUNT = 70  # OpenPose face model: 70 landmarks


def write_table(path, table: pa.Table, schema: pa.schema, *, extra_metadata: dict[str, str] | None = None) -> None:
    """Validate against the declared schema, then write with compression."""
    if extra_metadata:
        merged = {**(schema.metadata or {}), **extra_metadata}
        schema = schema.with_metadata(merged)
    table = _coerce(table, schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def _coerce(table: pa.Table, schema: pa.schema) -> pa.Table:
    missing = [field.name for field in schema if field.name not in table.column_names]
    for name in missing:
        # pyarrow has append_column (singular); there is no append_columns, so the
        # plural call raised AttributeError and every "optional column absent"
        # write died instead of writing nulls.
        table = table.append_column(schema.field(name), pa.nulls(len(table), type=schema.field(name).type))
    table = table.select([field.name for field in schema])
    return table.cast(schema, safe=False)


class ChunkedParquetWriter:
    """Bounded-memory writer: buffers rows, flushes row groups to one file."""

    def __init__(self, path, schema: pa.schema, *, rows_per_group: int = 200_000,
                 extra_metadata: dict[str, str] | None = None) -> None:
        self.path = path
        self.schema = schema
        self.rows_per_group = rows_per_group
        self.rows: list[dict[str, object]] = []
        self.rows_written = 0
        self._writer: pq.ParquetWriter | None = None
        self._extra_metadata = extra_metadata or {}

    def add(self, row: dict[str, object]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.rows_per_group:
            self.flush()

    def extend(self, rows: Iterable[dict[str, object]]) -> None:
        for row in rows:
            self.add(row)

    def flush(self) -> None:
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        table = _coerce(table, self.schema)
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            metadata = {**(self.schema.metadata or {}), **self._extra_metadata}
            self._writer = pq.ParquetWriter(
                self.path, self.schema.with_metadata(metadata), compression="zstd"
            )
        self._writer.write_table(table)
        self.rows_written += len(self.rows)
        self.rows.clear()

    def close(self) -> int:
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        elif not self.path.exists():
            # Never leave a missing file: an empty table is a valid, honest result.
            write_table(self.path, pa.table({f.name: [] for f in self.schema}, schema=self.schema),
                        self.schema, extra_metadata=self._extra_metadata)
        return self.rows_written


def read_table(path, columns: Sequence[str] | None = None) -> pa.Table:
    """Read a table, optionally projecting columns (cheap on wide tables)."""
    return pq.read_table(path, columns=list(columns) if columns else None)


def table_rows(path) -> int:
    return pq.read_metadata(path).num_rows


def table_columns(path) -> Sequence[str]:
    return [field.name for field in pq.read_schema(path)]


def iter_rows(path, columns: Sequence[str] | None = None) -> Iterator[dict[str, object]]:
    """Stream rows in row-group batches without materialising the whole table."""
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=20_000, columns=list(columns) if columns else None):
        yield from batch.to_pylist()
