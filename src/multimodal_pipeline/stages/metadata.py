"""Media metadata via ffprobe (authoritative) plus source hashing.

Frame rates are kept in *both* representations: ffprobe's rational
(``30000/1001``) is exact and lossless, the float is for consumers. Frame
timings themselves come from per-frame PTS when a precise mapping is needed
(``source/frame_index.parquet``).
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import pyarrow as pa

from ..artifacts import atomic_write_json
from ..config import stable_hash
from ..schemas import FRAME_INDEX_SCHEMA, read_table, write_table
from ..subprocess_utils import CommandError, require_executable, run_command
from ..validation import ValidationIssue, validate_metadata_payload
from .base import Stage, StageContext, StageError

SHA256_CHUNK = 1024 * 1024


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(SHA256_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_rational(value: Any) -> Fraction | None:
    """Parse ffprobe rationals like ``30000/1001``; ``0/0`` and junk become None."""
    if value in (None, "", "0/0", "N/A"):
        return None
    try:
        fraction = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return None
    return fraction if fraction > 0 else None


def rational_to_float(value: Fraction | None) -> float | None:
    return float(value) if value else None


def ffprobe_streams(video: Path, *, ffprobe: str = "ffprobe", timeout: float = 300.0) -> dict[str, Any]:
    argv = [
        ffprobe, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", "-show_chapters", str(video),
    ]
    result = run_command(argv, timeout=timeout)
    try:
        return json.loads(result.output or "{}")
    except json.JSONDecodeError as exc:
        raise StageError(f"ffprobe produced unparseable JSON for {video.name}: {exc}") from exc


def frame_pts(video: Path, *, ffprobe: str = "ffprobe", timeout: float = 1800.0) -> list[tuple[int, float]]:
    """``(frame_number, pts_seconds)`` for every video frame.

    Uses **packet** timing: no decoding happens, so it stays fast on long
    recordings, and packets are sorted into presentation order because that is
    the order a video producer (and therefore OpenPose) yields frames in.
    """
    argv = [
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
        "packet=pts_time,duration_time,flags", "-of", "json", str(video),
    ]
    result = run_command(argv, timeout=timeout)
    payload = json.loads(result.output or "{}")
    timestamps: list[float] = []
    for packet in payload.get("packets", []):
        raw = packet.get("pts_time")
        try:
            timestamps.append(max(float(raw), 0.0))
        except (TypeError, ValueError):
            continue
    if not timestamps:
        return []
    timestamps.sort()
    return list(enumerate(timestamps))


def select_stream(payload: dict[str, Any], kind: str) -> dict[str, Any] | None:
    streams = [s for s in payload.get("streams", []) if s.get("codec_type") == kind]
    if not streams:
        return None
    # Prefer the stream with explicit dimensions / most metadata.
    streams.sort(key=lambda s: (bool(s.get("width")), bool(s.get("height")), s.get("index", 0)), reverse=True)
    return streams[0]


def build_metadata(video_id: str, source_path: Path, payload: dict[str, Any],
                   *, sha256: str, size_bytes: int) -> dict[str, Any]:
    video_stream = select_stream(payload, "video") or {}
    audio_stream = select_stream(payload, "audio") or {}
    fmt = payload.get("format", {}) or {}
    duration = _first_float(fmt.get("duration"), video_stream.get("duration"), audio_stream.get("duration"))
    frame_rate = parse_rational(video_stream.get("r_frame_rate"))
    average_rate = parse_rational(video_stream.get("avg_frame_rate")) or frame_rate
    nb_frames = _first_int(video_stream.get("nb_frames"))
    if nb_frames is None and duration and average_rate:
        nb_frames = int(round(duration * float(average_rate)))
    creation = {
        key: value for key, value in (fmt.get("tags") or {}).items()
        if key.lower() in {"creation_time", "encoder", "major_brand", "minor_version",
                           "compatible_brands", "title", "comment", "date", "time"}
    }
    return {
        "schema_version": "1.0",
        "video_id": video_id,
        "source_filename": source_path.name,
        "source_path": str(source_path),
        "SHA256": sha256,
        "file_size_bytes": size_bytes,
        "container": fmt.get("format_name"),
        "container_long_name": fmt.get("format_long_name"),
        "duration_seconds": duration,
        "video_codec": video_stream.get("codec_name"),
        "video_codec_profile": (video_stream.get("profile") or ""),
        "pixel_format": video_stream.get("pix_fmt"),
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "display_aspect_ratio": video_stream.get("display_aspect_ratio"),
        "sample_aspect_ratio": video_stream.get("sample_aspect_ratio"),
        "frame_rate_rational": _rational_string(frame_rate),
        "frame_rate_float": rational_to_float(frame_rate),
        "average_frame_rate_rational": _rational_string(average_rate),
        "average_frame_rate_float": rational_to_float(average_rate),
        "frame_count": nb_frames,
        "time_base": video_stream.get("time_base"),
        "audio_codec": audio_stream.get("codec_name"),
        "audio_sample_rate": _first_int(audio_stream.get("sample_rate")),
        "audio_channels": _first_int(audio_stream.get("channels")),
        "audio_channel_layout": audio_stream.get("channel_layout"),
        "audio_time_base": audio_stream.get("time_base"),
        "creation_metadata": creation,
        "ffprobe_format_name": fmt.get("format_name"),
        "streams_present": sorted({s.get("codec_type") for s in payload.get("streams", []) if s.get("codec_type")}),
    }


def _rational_string(value: Fraction | None) -> str | None:
    return f"{value.numerator}/{value.denominator}" if value else None


def _first_float(*values: Any) -> float | None:
    for value in values:
        try:
            if value not in (None, "", "N/A"):
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        try:
            if value not in (None, "", "N/A"):
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


class MetadataStage(Stage):
    name = "metadata"
    inputs: tuple[str, ...] = ()
    outputs = ("metadata", "frame_index")
    config_keys = ("ffmpeg",)

    def config_fingerprint(self, ctx: StageContext) -> dict[str, Any]:
        from ..stages.base import fingerprint_config

        return {
            "ffmpeg": fingerprint_config(ctx, ("ffmpeg",)),
            "source_sha256": ctx.scratch.get("source_sha256"),
            "source_size": ctx.scratch.get("source_size"),
            "frame_index_mode": "ffprobe_pts",
        }

    def prepare(self, ctx: StageContext) -> None:
        require_executable(ctx.config.ffmpeg.ffprobe, hint="install ffmpeg or set ffmpeg.ffprobe")

    def execute(self, ctx: StageContext) -> dict[str, Any]:
        source = ctx.source.path
        ctx.log(f"hashing {source.name}")
        digest = sha256_of(source)
        ctx.scratch["source_sha256"] = digest
        payload = ffprobe_streams(source, ffprobe=ctx.config.ffmpeg.ffprobe)
        metadata = build_metadata(
            ctx.video_id, source, payload, sha256=digest, size_bytes=source.stat().st_size
        )
        atomic_write_json(ctx.artifact("metadata"), metadata)
        ctx.log(
            f"metadata: {metadata['width']}x{metadata['height']} @ "
            f"{metadata['frame_rate_rational']} fps, {metadata['duration_seconds']:.2f}s"
        )
        frames = frame_pts(source, ffprobe=ctx.config.ffmpeg.ffprobe)
        if not frames:
            # Fall back to a uniform grid so downstream timing still works.
            rate = metadata.get("average_frame_rate_float") or 25.0
            count = metadata.get("frame_count") or int((metadata.get("duration_seconds") or 0) * rate)
            frames = [(index, index / rate) for index in range(count)]
            ctx.log(f"frame PTS unavailable; using uniform {rate:g} fps grid", level=30)
        write_table(
            ctx.artifact("frame_index"),
            pa.Table.from_pylist(
                [{"schema_version": "1.0", "video_id": ctx.video_id, "frame_number": n, "pts_seconds": t}
                 for n, t in frames],
                schema=FRAME_INDEX_SCHEMA,
            ),
            FRAME_INDEX_SCHEMA,
        )
        ctx.scratch["metadata"] = metadata
        return {
            "tool_version": ffprobe_version(ctx),
            "model_version": None,
            "extra": {"frame_count": len(frames), "sha256": digest},
        }

    def validate(self, ctx: StageContext) -> dict[str, Any]:
        path = ctx.artifact("metadata")
        if not path.is_file():
            raise ValidationIssue(self.name, ["metadata.json missing"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = validate_metadata_payload(payload, stage=self.name)
        table = ctx.artifact("frame_index")
        if not table.is_file():
            raise ValidationIssue(self.name, ["frame_index.parquet missing"])
        rows = read_table(table).num_rows
        if rows == 0:
            raise ValidationIssue(self.name, ["frame_index.parquet is empty"])
        return {"duration_seconds": summary["duration_seconds"], "frame_index_rows": rows, **summary}


def ffprobe_version(ctx: StageContext) -> str | None:
    from ..subprocess_utils import probe_version

    return probe_version([ctx.config.ffmpeg.ffprobe, "-version"])
