"""Metadata + audio stages against real ffmpeg/ffprobe output.

These use the machine's real ffmpeg 7.x: rational frame rates, packet timing and
WAV headers are exactly the things a mock would hide.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import wave
from fractions import Fraction
from pathlib import Path

import pytest

from multimodal_pipeline.artifacts import read_json
from multimodal_pipeline.schemas import read_table
from multimodal_pipeline.stages.audio import AudioStage, read_wav_info
from multimodal_pipeline.stages.metadata import (
    MetadataStage,
    build_metadata,
    ffprobe_streams,
    frame_pts,
    parse_rational,
    sha256_of,
)
from multimodal_pipeline.exceptions import StageError, ValidationError

from tests.conftest import make_test_video

def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not installed")


requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


@pytest.fixture(scope="module")
def ntsc_video(tmp_path_factory) -> Path:
    """29.97 fps (30000/1001) video: the rational-rate case that breaks naively."""
    _require_ffmpeg()
    return make_test_video(tmp_path_factory.mktemp("media") / "ntsc.mp4",
                           seconds=2.0, rate="30000/1001", size="160x120")


@pytest.fixture(scope="module")
def silent_video(tmp_path_factory) -> Path:
    _require_ffmpeg()
    return make_test_video(tmp_path_factory.mktemp("media") / "silent.mp4",
                           seconds=1.5, fps=25, with_audio=False)


class TestRationalFrameRates:
    @pytest.mark.parametrize("raw,expected", [
        ("30000/1001", Fraction(30000, 1001)),
        ("24000/1001", Fraction(24000, 1001)),
        ("60000/1001", Fraction(60000, 1001)),
        ("25/1", Fraction(25)),
        ("25", Fraction(25)),
    ])
    def test_parses_media_ratios_exactly(self, raw: str, expected: Fraction) -> None:
        assert parse_rational(raw) == expected

    @pytest.mark.parametrize("raw", ["0/0", "N/A", "", None, "abc", "0/1", "-1/1"])
    def test_rejects_non_positive_or_junk(self, raw) -> None:
        assert parse_rational(raw) is None

    def test_1001_denominator_is_not_rounded_to_30(self) -> None:
        rate = parse_rational("30000/1001")
        assert float(rate) != 30.0
        assert abs(float(rate) - 29.970029970) < 1e-8


@requires_ffmpeg
class TestFfprobe:
    def test_probe_reports_streams(self, ntsc_video: Path) -> None:
        payload = ffprobe_streams(ntsc_video)
        kinds = {stream["codec_type"] for stream in payload["streams"]}
        assert {"video", "audio"} <= kinds

    def test_metadata_preserves_rational_and_float_rates(self, ntsc_video: Path) -> None:
        payload = ffprobe_streams(ntsc_video)
        metadata = build_metadata("vid", ntsc_video, payload, sha256="ab", size_bytes=10)
        assert metadata["frame_rate_rational"] in {"30000/1001", "24000/1001", "60000/1001"}
        assert abs(metadata["frame_rate_float"] - 29.970029970) < 1e-4

    def test_metadata_shape(self, ntsc_video: Path) -> None:
        payload = ffprobe_streams(ntsc_video)
        metadata = build_metadata("vid", ntsc_video, payload, sha256="abc123", size_bytes=4242)
        for key in ("schema_version", "video_id", "source_filename", "source_path", "SHA256",
                    "file_size_bytes", "container", "duration_seconds", "video_codec",
                    "pixel_format", "width", "height", "frame_rate_rational", "frame_rate_float",
                    "average_frame_rate_rational", "average_frame_rate_float", "frame_count", "audio_codec", "audio_sample_rate",
                    "audio_channels", "creation_metadata"):
            assert key in metadata, key
        assert metadata["width"] == 160 and metadata["height"] == 120
        assert metadata["file_size_bytes"] == 4242

    def test_frame_count_matches_duration_times_rate(self, ntsc_video: Path) -> None:
        payload = ffprobe_streams(ntsc_video)
        metadata = build_metadata("vid", ntsc_video, payload, sha256="x", size_bytes=1)
        expected = metadata["duration_seconds"] * metadata["average_frame_rate_float"]
        assert abs(metadata["frame_count"] - expected) <= 2

    def test_sha256_is_stable_and_correct(self, ntsc_video: Path) -> None:
        import hashlib

        digest = sha256_of(ntsc_video)
        assert digest == hashlib.sha256(ntsc_video.read_bytes()).hexdigest()

    def test_frame_pts_is_monotonic_and_matches_frame_count(self, ntsc_video: Path) -> None:
        frames = frame_pts(ntsc_video)
        assert len(frames) > 0
        stamps = [stamp for _index, stamp in frames]
        assert stamps == sorted(stamps)
        assert frames[0][0] == 0
        assert frames[0][1] == pytest.approx(0.0, abs=1e-3)
        metadata = build_metadata("v", ntsc_video, ffprobe_streams(ntsc_video), sha256="x", size_bytes=1)
        assert abs(len(frames) - metadata["frame_count"]) <= 2

    def test_frame_pts_step_matches_frame_rate(self, ntsc_video: Path) -> None:
        stamps = [stamp for _index, stamp in frame_pts(ntsc_video)]
        steps = [b - a for a, b in zip(stamps, stamps[1:])]
        expected = 1001 / 30000
        assert all(abs(step - expected) < 2e-3 for step in steps)


@requires_ffmpeg
class TestMetadataStage:
    def test_execute_and_validate(self, context, ntsc_video) -> None:
        from multimodal_pipeline.discovery import VideoSource

        context.source = VideoSource(path=ntsc_video, relative_path=Path(ntsc_video.name),
                                     video_id="ntsc")
        context.scratch.clear()
        stage = MetadataStage()
        stage.prepare(context)
        extras = stage.execute(context)
        validation = stage.validate(context)
        payload = read_json(context.artifact("metadata"))
        assert payload["video_id"] == "ntsc"
        assert payload["duration_seconds"] > 0
        assert validation["frame_index_rows"] > 0
        assert extras["tool_version"].startswith("ffprobe")

    def test_frame_index_parquet_is_ordered(self, context, ntsc_video) -> None:
        from multimodal_pipeline.discovery import VideoSource

        context.source = VideoSource(path=ntsc_video, relative_path=Path(ntsc_video.name), video_id="ntsc")
        stage = MetadataStage()
        stage.execute(context)
        rows = read_table(context.artifact("frame_index")).to_pylist()
        assert [row["frame_number"] for row in rows] == list(range(len(rows)))


@requires_ffmpeg
class TestAudioStage:
    def test_command_targets_pcm_16k_mono(self, context) -> None:
        argv = AudioStage().build_command(context, Path("/tmp/out.wav"))
        joined = " ".join(argv)
        assert "-ar 16000" in joined
        assert "-ac 1" in joined
        assert "pcm_s16le" in joined
        assert argv[0] == context.config.ffmpeg.executable

    def test_command_never_uses_a_shell_string(self, context) -> None:
        argv = AudioStage().build_command(context, Path("/tmp/out.wav"))
        # Argument arrays only: no shell metacharacters anywhere.
        assert not any(token in item for item in argv for token in ("&&", ";", "|", "$("))
        assert isinstance(argv, list) and all(isinstance(item, str) for item in argv)

    def test_execute_produces_readable_wav(self, context, ntsc_video) -> None:
        from multimodal_pipeline.discovery import VideoSource

        context.source = VideoSource(path=ntsc_video, relative_path=Path(ntsc_video.name), video_id="ntsc")
        MetadataStage().execute(context)
        stage = AudioStage()
        stage.prepare(context)
        stage.execute(context)
        info = read_wav_info(context.artifact("audio"))
        assert info["sample_rate"] == 16000
        assert info["channels"] == 1
        assert info["sample_format"] == "s16le"
        assert stage.validate(context)["duration_drift_seconds"] < 0.05

    def test_duration_matches_source_video(self, context, ntsc_video) -> None:
        from multimodal_pipeline.discovery import VideoSource

        context.source = VideoSource(path=ntsc_video, relative_path=Path(ntsc_video.name), video_id="ntsc")
        metadata = MetadataStage().execute(context)
        AudioStage().execute(context)
        with wave.open(str(context.artifact("audio")), "rb") as handle:
            wav_duration = handle.getnframes() / handle.getframerate()
        expected = read_json(context.artifact("metadata"))["duration_seconds"]
        assert abs(wav_duration - expected) < 0.05

    def test_audio_without_audio_stream_is_reported(self, context, silent_video) -> None:
        from multimodal_pipeline.discovery import VideoSource

        context.source = VideoSource(path=silent_video, relative_path=Path(silent_video.name),
                                     video_id="silent")
        context.scratch.clear()
        MetadataStage().execute(context)
        stage = AudioStage()
        # No output exists yet, so this is a stage failure, not a validation one.
        with pytest.raises(StageError, match="audio"):
            stage.prepare(context)

    def test_validate_rejects_wrong_sample_rate(self, context, ntsc_video, tmp_path) -> None:
        from multimodal_pipeline.discovery import VideoSource

        context.source = VideoSource(path=ntsc_video, relative_path=Path(ntsc_video.name), video_id="ntsc")
        MetadataStage().execute(context)
        AudioStage().execute(context)
        # Replace the WAV with a 44.1 kHz one: validation must notice.
        destination = context.artifact("audio")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=2",
                        "-ac", "1", "-ar", "44100", str(destination)], check=True)
        with pytest.raises(ValidationError, match="sample rate"):
            AudioStage().validate(context)
