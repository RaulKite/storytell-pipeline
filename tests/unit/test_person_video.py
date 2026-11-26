"""The rendered person-track preview's validation, against a real encoder's output.

The encoder itself lives in ``workers/persons_worker.py`` — it needs cv2 and the frames
ultralytics decoded, and the orchestrator may import neither (AGENTS.md). What lives in
``person_video.py`` is the *check* on the published file, and the properties worth trusting
are the ones only a real container can answer: does ffprobe count the frames the worker wrote,
and does it refuse a file that lost one.

So most of this file runs the real ``ffmpeg`` and ``ffprobe`` when they are installed, and the
strictness cases use a fake ffprobe that answers with the exact stream metadata a truncated
file would carry.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from multimodal_pipeline.exceptions import StageError
from multimodal_pipeline.person_video import check_track_video

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "src" / "multimodal_pipeline" / "person_video.py"

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe unavailable")


def encode(path: Path, *, frames: int = 5, with_audio: bool = False) -> Path:
    """A real, decodable mp4 of exactly ``frames`` frames at 25 fps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    argv = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
            "-i", f"color=c=black:s=64x48:r=25:d={frames / 25:.4f}"]
    if with_audio:
        argv += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={frames / 25:.4f}",
                 "-c:a", "aac", "-shortest"]
    argv += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(path)]
    subprocess.run(argv, check=True, capture_output=True)
    return path


def fake_ffprobe(tmp_path, payload: dict, *, name: str = "ffprobe"):
    """A stand-in ffprobe that answers with exactly the stream metadata a file would carry.

    The payload lives in a side file rather than in the script's source, so the test states
    the metadata as data and the stub cannot mangle it by quoting.
    """
    answer = tmp_path / f"{name}-answer.json"
    answer.write_text(json.dumps(payload), encoding="utf-8")
    script = tmp_path / name
    script.write_text(
        f"#!{sys.executable}\nimport sys\nfrom pathlib import Path\n"
        f"sys.stdout.write(Path({json.dumps(str(answer))}).read_text())\n",
        encoding="utf-8")
    script.chmod(0o755)
    return script


def streams(count: int, *, audio: bool = False) -> dict:
    payload = {"streams": [{"codec_type": "video", "nb_frames": str(count),
                            "avg_frame_rate": "25/1", "duration": str(count / 25)}]}
    if audio:
        payload["streams"].append({"codec_type": "audio"})
    return payload


class TestAgainstRealMedia:
    @requires_ffmpeg
    def test_a_real_preview_passes_when_its_frame_count_matches(self, tmp_path):
        video = encode(tmp_path / "tracks.mp4", frames=5)
        checked = check_track_video(video, expected_frames=5)
        assert checked["frames"] == 5
        assert checked["has_audio"] is False

    @requires_ffmpeg
    def test_audio_is_reported_and_required_only_when_the_source_had_it(self, tmp_path):
        silent = encode(tmp_path / "silent.mp4")
        assert check_track_video(silent)["has_audio"] is False
        loud = encode(tmp_path / "loud.mp4", with_audio=True)
        assert check_track_video(loud, require_audio=True)["has_audio"] is True

    @requires_ffmpeg
    def test_a_preview_that_lost_frames_is_refused(self, tmp_path):
        """One missing frame shifts every later overlay against ``persons/frames.parquet``.

        The OpenPose preview is allowed three trailing frames because upstream's renderer is a
        black box; this encoder is ours, so there is nothing to be tolerant about.
        """
        video = encode(tmp_path / "tracks.mp4", frames=5)
        with pytest.raises(StageError, match="has 5 frame\\(s\\), expected 6"):
            check_track_video(video, expected_frames=6)
        with pytest.raises(StageError, match="has 5 frame\\(s\\), expected 4"):
            check_track_video(video, expected_frames=4)

    @requires_ffmpeg
    def test_a_silent_preview_is_refused_when_audio_was_promised(self, tmp_path):
        video = encode(tmp_path / "silent.mp4")
        with pytest.raises(StageError, match="audio"):
            check_track_video(video, require_audio=True)


class TestFileStates:
    def test_a_missing_preview_is_named(self, tmp_path):
        with pytest.raises(StageError, match="missing or empty"):
            check_track_video(tmp_path / "nope.mp4")

    def test_an_empty_preview_is_named(self, tmp_path):
        empty = tmp_path / "empty.mp4"
        empty.write_bytes(b"")
        with pytest.raises(StageError, match="missing or empty"):
            check_track_video(empty)

    def test_a_file_with_no_video_stream_is_refused(self, tmp_path):
        clip = tmp_path / "tracks.mp4"
        clip.write_bytes(b"x")
        probe = fake_ffprobe(tmp_path, {"streams": [{"codec_type": "audio"}]})
        with pytest.raises(StageError, match="no video stream"):
            check_track_video(clip, ffprobe=str(probe))


class TestFrameCountIsStrict:
    """The count comes from the container, and any difference is a broken preview."""

    def _check(self, tmp_path, payload, expected):
        probe = fake_ffprobe(tmp_path, payload)
        clip = tmp_path / "tracks.mp4"
        clip.write_bytes(b"x")
        return check_track_video(clip, ffprobe=str(probe), expected_frames=expected)

    def test_an_exact_match_passes(self, tmp_path):
        assert self._check(tmp_path, streams(120), 120)["frames"] == 120

    @pytest.mark.parametrize("expected", [119, 121, 123])
    def test_a_difference_of_any_size_fails_with_both_numbers(self, tmp_path, expected):
        with pytest.raises(StageError, match=f"has 120 frame\\(s\\), expected {expected}"):
            self._check(tmp_path, streams(120), expected)

    def test_a_container_that_will_not_state_its_frame_count_is_refused(self, tmp_path):
        """Guessing from duration would let a truncated file through under a rounding.

        ``nb_frames`` is always written by the encoder this stage uses, so its absence means
        the file did not come from here.
        """
        payload = {"streams": [{"codec_type": "video", "avg_frame_rate": "25/1",
                                "duration": "4.8"}]}
        with pytest.raises(StageError, match="cannot be verified"):
            self._check(tmp_path, payload, 120)

    def test_a_container_that_reports_no_frames_is_refused(self, tmp_path):
        """A 0-frame container against a 120-frame measurement is the worst case: the preview
        plays and shows nothing, while the tables describe 120 frames of people."""
        with pytest.raises(StageError, match="has 0 frame"):
            self._check(tmp_path, streams(0), 120)


class TestOrchestratorStaysLight:
    """AGENTS.md: the orchestrator never imports torch/ultralytics/spacy/parselmouth.

    cv2 joins that list for this module by the same argument — it is only available inside
    ``environments/persons`` — so the check on the rendered file has to be pure python plus
    ffprobe, and the encoder has to stay in the worker.
    """

    def test_the_module_imports_no_heavy_visual_dependency(self):
        forbidden = {"cv2", "torch", "ultralytics", "numpy", "pyannote", "spacy",
                     "parselmouth"}
        imported: set[str] = set()
        for node in ast.parse(MODULE.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])
        assert not imported & forbidden, imported & forbidden

    def test_the_worker_owns_the_encoder_and_imports_cv2_lazily(self):
        """A module-scope ``import cv2`` would break the suite that imports the worker here."""
        source = (ROOT / "workers" / "persons_worker.py").read_text(encoding="utf-8")
        assert "class TrackVideoWriter" in source
        tree = ast.parse(source)
        top_level = {alias.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import)
                     for alias in node.names}
        assert "cv2" not in top_level
        assert "import cv2" in source
