"""Audio extraction for the speech/acoustic stages.

One canonical WAV is produced: 16 kHz mono PCM s16le. That single format is
what WhisperX, Pyannote Community-1 and Parselmouth all expect, so timeline
synchronisation is guaranteed by construction instead of by three conversions.

The native rate/channel layout is preserved (``audio_source_info``) for
acoustics: downmixing can hide channel-specific information, and recording
which decision was taken is part of reproducibility.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

from ..artifacts import atomic_write_json
from ..config import stable_hash
from ..subprocess_utils import require_executable, run_command
from ..validation import ValidationIssue
from .base import Stage, StageContext, StageError

TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1


class AudioStage(Stage):
    name = "audio"
    inputs = ("metadata",)
    outputs = ("audio",)
    config_keys = ("ffmpeg",)

    def config_fingerprint(self, ctx: StageContext) -> dict[str, Any]:
        metadata = ctx.scratch.get("metadata") or self._metadata(ctx)
        return {
            "ffmpeg": {
                "executable": ctx.config.ffmpeg.executable,
                "ffprobe": ctx.config.ffmpeg.ffprobe,
            },
            "source_sha256": metadata.get("SHA256"),
            "sample_rate": TARGET_SAMPLE_RATE,
            "channels": TARGET_CHANNELS,
            "sample_format": "s16le",
        }

    def _metadata(self, ctx: StageContext) -> dict[str, Any]:
        path = ctx.artifact("metadata")
        if not path.is_file():
            raise StageError("audio stage requires metadata.json (run the metadata stage first)")
        return json.loads(path.read_text(encoding="utf-8"))

    def prepare(self, ctx: StageContext) -> None:
        require_executable(ctx.config.ffmpeg.executable, hint="install ffmpeg or set ffmpeg.executable")
        metadata = ctx.scratch.get("metadata") or self._metadata(ctx)
        if not metadata.get("audio_codec"):
            raise StageError(f"source video has no audio stream: {ctx.source.path.name}")

    def build_command(self, ctx: StageContext, destination: Path) -> list[str]:
        return [
            ctx.config.ffmpeg.executable,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "info",
            "-y",
            "-i",
            str(ctx.source.path),
            "-vn",                    # drop video entirely
            "-map",
            "0:a:0",                  # first audio stream only
            "-ac",
            str(TARGET_CHANNELS),
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-sample_fmt",
            "s16",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]

    def execute(self, ctx: StageContext) -> dict[str, Any]:
        destination = ctx.artifact("audio")
        destination.parent.mkdir(parents=True, exist_ok=True)
        argv = self.build_command(ctx, destination)
        ctx.log(f"extracting audio -> {destination.relative_to(ctx.paths.dataset_dir)}")
        result = run_command(argv, log_path=ctx.paths.log(self.name), timeout=None)
        info = read_wav_info(destination)
        payload = {
            "schema_version": "1.0",
            "video_id": ctx.video_id,
            "path": str(destination.relative_to(ctx.paths.dataset_dir)),
            "command": result.argv_masked,
            "ffmpeg_version": ffmpeg_version(ctx),
            "target": {
                "sample_rate": TARGET_SAMPLE_RATE,
                "channels": TARGET_CHANNELS,
                "sample_format": "s16le",
            },
            "output": info,
        }
        atomic_write_json(ctx.paths.artifact("audio").with_name("audio_info.json"), payload)
        ctx.scratch["audio_info"] = info
        return {
            "tool_version": payload["ffmpeg_version"],
            "model_version": None,
            "extra": {"duration_seconds": info.get("duration_seconds"), **payload["target"]},
        }

    def validate(self, ctx: StageContext) -> dict[str, Any]:
        path = ctx.artifact("audio")
        if not path.is_file():
            raise ValidationIssue(self.name, ["audio/audio.wav missing"])
        try:
            info = read_wav_info(path)
        except Exception as exc:
            raise ValidationIssue(self.name, [f"unreadable WAV: {exc}"]) from exc
        if info["sample_rate"] != TARGET_SAMPLE_RATE:
            raise ValidationIssue(self.name, [f"sample rate {info['sample_rate']} != {TARGET_SAMPLE_RATE}"])
        if info["channels"] != TARGET_CHANNELS:
            raise ValidationIssue(self.name, [f"channels {info['channels']} != {TARGET_CHANNELS}"])
        if info["duration_seconds"] <= 0:
            raise ValidationIssue(self.name, ["extracted audio duration is zero"])
        metadata = self._metadata(ctx)
        expected = metadata.get("duration_seconds")
        if expected:
            drift = abs(info["duration_seconds"] - expected)
            # Containers round durations differently; >2 s means real desync.
            if drift > max(2.0, 0.02 * expected):
                raise ValidationIssue(
                    self.name,
                    [f"audio duration {info['duration_seconds']:.2f}s diverges from video "
                     f"{expected:.2f}s by {drift:.2f}s"],
                )
        return {**info, "duration_drift_seconds": round(abs(info["duration_seconds"] - (expected or 0)), 3)}


def read_wav_info(path: Path) -> dict[str, Any]:
    """Header-only WAV inspection (no audio decoding, works on huge files)."""
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
    return {
        "sample_rate": rate,
        "channels": channels,
        "sample_width_bytes": width,
        "sample_format": {1: "s8", 2: "s16le", 4: "s32le"}.get(width, f"width={width}"),
        "frame_count": frames,
        "duration_seconds": round(frames / rate, 6) if rate else 0.0,
        "size_bytes": Path(path).stat().st_size,
    }


def ffmpeg_version(ctx: StageContext) -> str | None:
    from ..subprocess_utils import probe_version

    return probe_version([ctx.config.ffmpeg.executable, "-version"])
