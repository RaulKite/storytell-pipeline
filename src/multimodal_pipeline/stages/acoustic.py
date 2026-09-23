"""Acoustic analysis stage (Praat through Parselmouth).

Raw measurements are preserved as JSONL under ``acoustic/raw/`` so the frame
timeline can be re-aggregated after a schema or rule change without asking Praat
to recompute anything: the expensive part (signal analysis) happens once.

The worker streams, and so does normalisation — frames are read in batches and
written as Parquet row groups, keeping memory flat for hour-long recordings.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Deque, Iterator

import pyarrow as pa

from ..acoustics import aggregate_segment, detect_silences, normalise_frame_row
from ..artifacts import read_json
from ..exceptions import StageError, ValidationError
from ..schemas import (
    ACOUSTIC_FRAMES_SCHEMA,
    ACOUSTIC_SEGMENTS_SCHEMA,
    ChunkedParquetWriter,
    read_table,
)
from .base import StageContext, WorkerStage

FRAMES_PER_GROUP = 250_000


class AcousticStage(WorkerStage):
    name = "acoustic"
    raw_artifact = "acoustic_raw"
    inputs = ("audio", "speech_segments")
    outputs = ("acoustic_raw", "acoustic_frames", "acoustic_segments")
    config_keys = ("acoustic",)

    # ------------------------------------------------------------------ request

    def request(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.config.acoustic
        return {
            "stage": self.name,
            "backend": cfg.backend,
            "time_step": cfg.time_step,
            "pitch_floor": cfg.pitch_floor,
            "pitch_ceiling": cfg.pitch_ceiling,
            "number_of_formants": cfg.number_of_formants,
            "formant_ceiling": cfg.formant_ceiling,
            "silence_threshold_db": cfg.silence_threshold_db,
            "minimum_pause_duration": cfg.minimum_pause_duration,
            "chunk_seconds": cfg.chunk_seconds,
            "uv_project": str(ctx.config.resolve(cfg.uv_project)),
            "worker": str(ctx.config.resolve(cfg.worker)),
            "audio_sha256": self._audio_digest(ctx),
        }

    @staticmethod
    def _audio_digest(ctx: StageContext) -> str | None:
        from ..stages.metadata import sha256_of

        try:
            path = ctx.input("audio")
        except StageError:
            return None
        cached = ctx.scratch.get("audio_sha256")
        if cached:
            return cached
        digest = sha256_of(path)
        ctx.scratch["audio_sha256"] = digest
        return digest

    # ------------------------------------------------------------------ worker

    def enabled(self, ctx: StageContext) -> tuple[bool, str]:
        if not ctx.config.acoustic.enabled:
            return False, "acoustic.enabled = false"
        return True, ""

    def uv_project(self, ctx: StageContext) -> Path:
        return ctx.config.resolve(ctx.config.acoustic.uv_project)

    def worker_script(self, ctx: StageContext) -> Path:
        return ctx.config.resolve(ctx.config.acoustic.worker)

    def python_version(self, ctx: StageContext) -> str | None:
        return ctx.config.acoustic.python_version

    def worker_argv(self, ctx: StageContext, raw_path: Path, request_digest: str) -> list[str]:
        cfg = ctx.config.acoustic
        args = [
            "--audio", str(ctx.input("audio")),
            "--raw-output", str(raw_path),
            "--video-id", ctx.video_id,
            "--time-step", str(cfg.time_step),
            "--pitch-floor", str(cfg.pitch_floor),
            "--pitch-ceiling", str(cfg.pitch_ceiling),
            "--number-of-formants", str(cfg.number_of_formants),
            "--formant-ceiling", str(cfg.formant_ceiling),
            "--minimum-pause-duration", str(cfg.minimum_pause_duration),
            "--chunk-seconds", str(cfg.chunk_seconds),
            "--request-hash", request_digest,
        ]
        if cfg.silence_threshold_db is not None:
            args += ["--silence-threshold-db", str(cfg.silence_threshold_db)]
        return args

    # ------------------------------------------------------------ normalisation

    def normalize(self, ctx: StageContext) -> dict[str, Any]:
        header = self.read_header(ctx)
        step = float(header.get("frame_step_seconds") or ctx.config.acoustic.time_step)
        writer = ChunkedParquetWriter(
            ctx.artifact("acoustic_frames"), ACOUSTIC_FRAMES_SCHEMA,
            rows_per_group=FRAMES_PER_GROUP,
            extra_metadata={"video_id": ctx.video_id,
                            "backend": str(header.get("parameters", {}).get("backend")),
                            "parselmouth_version": str(header.get("parameters", {}).get("parselmouth_version"))},
        )
        frames = 0
        voiced = 0
        # Silence detection needs every frame, but only (timestamp, voiced,
        # intensity) — so the running set stays small even for long recordings.
        silence_inputs: list[dict[str, Any]] = []
        for row in self.iter_frames(ctx):
            normalised = normalise_frame_row(row["frame"], video_id=ctx.video_id)
            if normalised["timestamp"] is None:
                continue
            writer.add(normalised)
            frames += 1
            if normalised["voiced"]:
                voiced += 1
            silence_inputs.append({"timestamp": normalised["timestamp"],
                                   "voiced": normalised["voiced"],
                                   "intensity_db": normalised["intensity_db"]})
        frame_rows = writer.close()

        cfg = ctx.config.acoustic
        silences = detect_silences(silence_inputs, minimum_duration=cfg.minimum_pause_duration,
                                   silence_threshold_db=cfg.silence_threshold_db, frame_step=step)
        segments = read_table(ctx.artifact("speech_segments")).to_pylist()
        duration = source_duration(ctx)
        aggregate_writer = ChunkedParquetWriter(
            ctx.artifact("acoustic_segments"), ACOUSTIC_SEGMENTS_SCHEMA,
            rows_per_group=10_000, extra_metadata={"video_id": ctx.video_id},
        )
        # Frames are re-read streaming rather than kept in RAM: the segment
        # aggregate is the only place that needs them grouped.
        with FrameIndex(self.iter_frames(ctx), ctx.video_id) as index:
            for segment in segments:
                rows = index.between(segment.get("start_time"), segment.get("end_time"))
                aggregate_writer.add(aggregate_segment(segment, rows, silences=silences,
                                                      duration=duration).as_row())
        aggregate_rows = aggregate_writer.close()
        ctx.scratch["acoustic"] = {"frames": frames, "voiced_frames": voiced,
                                   "silences": len(silences), "segments": aggregate_rows}
        ctx.log(f"acoustic: {frames} frames ({voiced} voiced), {len(silences)} pauses, "
                f"{aggregate_rows} segment aggregates")
        return {"frames": frames, "voiced_frames": voiced, "pause_count": len(silences),
                "segment_aggregates": aggregate_rows}

    def read_header(self, ctx: StageContext) -> dict[str, Any]:
        path = ctx.artifact(self.raw_artifact)
        if not path.is_file():
            raise ValidationError(self.name, [f"raw acoustic output missing: {path.name}"])
        first = first_json_line(path)
        if not isinstance(first, dict):
            raise ValidationError(self.name, [f"{path.name} first line is not a header object"])
        return first

    def iter_frames(self, ctx: StageContext) -> Iterator[dict[str, Any]]:
        """Stream the raw JSONL frame records (header line skipped)."""
        path = ctx.artifact(self.raw_artifact)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValidationError(self.name, [f"{path.name}:{line_number + 1} is not valid JSON: {exc}"]) from exc
                if line_number == 0 and isinstance(record, dict) and "frame" not in record:
                    continue
                if isinstance(record, dict) and "frame" in record:
                    yield record
                elif isinstance(record, list):
                    yield {"frame": record}

    # ---------------------------------------------------------------- validation

    def validate(self, ctx: StageContext) -> dict[str, Any]:
        frames_path = ctx.artifact("acoustic_frames")
        segments_path = ctx.artifact("acoustic_segments")
        duration = source_duration(ctx)
        from ..validation import check_parquet

        frames_info = check_parquet(frames_path, [field.name for field in ACOUSTIC_FRAMES_SCHEMA],
                                    stage=self.name, time_column="timestamp", max_time=duration)
        aggregate_info = check_parquet(segments_path, [field.name for field in ACOUSTIC_SEGMENTS_SCHEMA],
                                       stage=self.name)
        transcript = {row["segment_id"]: (row["start_time"], row["end_time"])
                      for row in read_table(ctx.artifact("speech_segments")).to_pylist()}
        aggregate_rows = read_table(segments_path).to_pylist()
        unknown = {row["segment_id"] for row in aggregate_rows} - set(transcript)
        if unknown:
            raise ValidationError(self.name, [f"segment aggregates reference unknown segments: {sorted(unknown)[:5]}"])
        drifted = [
            row["segment_id"] for row in aggregate_rows
            if transcript.get(row["segment_id"]) != (row["start_time"], row["end_time"])
        ]
        if drifted:
            raise ValidationError(
                self.name, [f"segment timing drifted from the transcript: {sorted(drifted)[:5]}"])
        ratio = next((row["pause_ratio"] for row in aggregate_rows
                      if row["pause_ratio"] is not None and row["pause_ratio"] > 1.0 + 1e-6), None)
        if ratio is not None:
            raise ValidationError(self.name, [f"pause_ratio {ratio} exceeds 1.0"])
        return {"frame_rows": frames_info["rows"], "segment_rows": aggregate_info["rows"]}


class FrameIndex:
    """One streaming pass kept alive so segment aggregation never loads it all.

    Segments are traversed in time order, so a single cursor over the raw frame
    stream answers every ``between()`` query with O(1) memory.
    """

    def __init__(self, records: Iterator[dict[str, Any]], video_id: str) -> None:
        self._records = records
        self._video_id = video_id
        self._pending: Deque[dict[str, Any]] = deque()
        self._exhausted = False
        self._watermark: float | None = None

    def __enter__(self) -> "FrameIndex":
        return self

    def __exit__(self, *exc: object) -> None:
        self._pending.clear()

    def between(self, start: Any, end: Any) -> list[dict[str, Any]]:
        from ..acoustics import is_number

        if not (is_number(start) and is_number(end)):
            return []
        lower, upper = float(start), float(end)
        if self._watermark is not None and lower < self._watermark - 1e-9:
            # Evicting below the watermark would silently lose frames for the
            # earlier segment, so out-of-order queries are a hard error instead.
            raise StageError(
                f"acoustic aggregation asked for segment starting at {lower:.3f}s after one starting at "
                f"{self._watermark:.3f}s; transcript segments must be in time order"
            )
        self._watermark = lower if self._watermark is None else max(self._watermark, lower)
        # Frames before this segment can never be needed again (segments advance).
        while self._pending and self._pending[0]["timestamp"] < lower - 1e-9:
            self._pending.popleft()
        # Read up to the segment end, but keep later frames buffered: they belong
        # to the next segment, and dropping them would lose data.
        while not self._exhausted:
            latest = self._pending[-1]["timestamp"] if self._pending else None
            if latest is not None and latest > upper + 1e-9:
                break
            frame = self._next_frame()
            if frame is None:
                break
        return [frame for frame in self._pending if frame["timestamp"] <= upper + 1e-9]

    def _next_frame(self) -> dict[str, Any] | None:
        while not self._exhausted:
            try:
                record = next(self._records)
            except StopIteration:
                self._exhausted = True
                return None
            row = normalise_frame_row(record["frame"], video_id=self._video_id)
            if row["timestamp"] is not None:
                self._pending.append(row)
                return row
        return None


def first_json_line(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    return None


def source_duration(ctx: StageContext) -> float | None:
    path = ctx.artifact("metadata")
    if not path.is_file():
        return None
    try:
        return read_json(path).get("duration_seconds")
    except (OSError, ValueError):
        return None
