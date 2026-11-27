"""The rendered person-track preview's validation, against a real encoder's output.

The encoder itself lives in ``workers/persons_worker.py`` — it needs cv2 and the frames
ultralytics decoded, and the orchestrator may import neither (AGENTS.md). What lives in
``person_video.py`` is the *check* on the published file, and the properties worth trusting
are the ones only a real container can answer: does ffprobe count the frames the worker wrote,
and does it refuse a file that lost one.

So most of this file runs the real ``ffmpeg`` and ``ffprobe`` when they are installed, and the
strictness cases use a fake ffprobe that answers with the exact stream metadata a truncated or
substituted file would carry.

The subtlety that decides the shape of these tests: a truncated mp4 still *declares* its
original frame count in the container header. The reviewer measured it — a file whose
``nb_frames`` said 125 decoded 61 frames — so every strictness case here gives the fake probe a
different ``nb_frames`` and ``nb_read_frames`` and asserts the refusal quotes the **counted**
one.
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
from multimodal_pipeline.person_video import FPS_TOLERANCE, check_track_video

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "src" / "multimodal_pipeline" / "person_video.py"

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe unavailable")


def encode(path: Path, *, frames: int = 5, with_audio: bool = False, fps: int = 25) -> Path:
    """A real, decodable mp4 of exactly ``frames`` frames at ``fps``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = frames / fps
    argv = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
            "-i", f"color=c=black:s=64x48:r={fps}:d={duration:.4f}"]
    if with_audio:
        argv += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration:.4f}",
                 "-c:a", "aac", "-shortest"]
    argv += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(path)]
    subprocess.run(argv, check=True, capture_output=True)
    return path


def fake_ffprobe(tmp_path, *, frames: int | str = 5, declared: int | str | None = None,
                 fps: str = "25/1", size: tuple[int, int] = (64, 48), audio: bool = False,
                 video: bool = True, name: str = "ffprobe"):
    """A stand-in ffprobe that answers both probes this module makes.

    ``frames`` is what ``nb_read_frames`` reports (the decoded count), ``declared`` what the
    container header claims (``nb_frames``). They are separate parameters because the whole
    point of the check is that they can disagree.

    The payloads live in side files rather than in the script's source, so the test states the
    metadata as data and the stub cannot mangle it by quoting.
    """
    counted: dict = {"streams": []}
    if video:
        stream: dict = {"codec_type": "video", "nb_read_frames": str(frames),
                        "avg_frame_rate": fps, "width": size[0], "height": size[1]}
        if declared is not None:
            stream["nb_frames"] = str(declared)
        counted["streams"] = [stream]
    kinds = {"streams": ([{"codec_type": "video"}] if video else [])
             + ([{"codec_type": "audio"}] if audio else [])}
    counted_path = tmp_path / f"{name}-counted.json"
    kinds_path = tmp_path / f"{name}-kinds.json"
    counted_path.write_text(json.dumps(counted), encoding="utf-8")
    kinds_path.write_text(json.dumps(kinds), encoding="utf-8")
    script = tmp_path / name
    script.write_text(
        f"#!{sys.executable}\nimport sys\nfrom pathlib import Path\n"
        f"counted = Path({json.dumps(str(counted_path))})\n"
        f"kinds = Path({json.dumps(str(kinds_path))})\n"
        f"sys.stdout.write((counted if '-count_frames' in sys.argv else kinds).read_text())\n",
        encoding="utf-8")
    script.chmod(0o755)
    return script


class TestAgainstRealMedia:
    @requires_ffmpeg
    def test_a_real_preview_passes_when_its_frame_count_matches(self, tmp_path):
        video = encode(tmp_path / "tracks.mp4", frames=5)
        checked = check_track_video(video, expected_frames=5, expected_fps=25.0,
                                    expected_size=(64, 48))
        assert checked["frames"] == 5
        assert checked["fps"] == pytest.approx(25.0)
        assert (checked["width"], checked["height"]) == (64, 48)
        assert checked["has_audio"] is False

    @requires_ffmpeg
    def test_the_count_is_what_decodes_not_what_the_header_declares(self, tmp_path):
        """The measured defect: a truncated file's header keeps its original count.

        A real 5-frame file, and a fake probe that reports the 5 decoded frames but a header
        count of 125 — the shape the reviewer's truncated render had. The refusal must quote
        the counted number, because that is what a player shows.
        """
        video = encode(tmp_path / "tracks.mp4", frames=5)
        probe = fake_ffprobe(tmp_path, frames=5, declared=125)
        with pytest.raises(StageError, match="decodes to 5 frame"):
            check_track_video(video, ffprobe=str(probe), expected_frames=125)
        # And a header that under-reports does not fail a good file either.
        assert check_track_video(video, ffprobe=str(probe), expected_frames=5)["frames"] == 5

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
        with pytest.raises(StageError, match="decodes to 5 frame\\(s\\), expected 6"):
            check_track_video(video, expected_frames=6)
        with pytest.raises(StageError, match="decodes to 5 frame\\(s\\), expected 4"):
            check_track_video(video, expected_frames=4)

    @requires_ffmpeg
    def test_a_real_file_at_another_clock_is_refused(self, tmp_path):
        """Same frame count, wrong duration: every overlay lands at the wrong instant."""
        video = encode(tmp_path / "tracks.mp4", frames=5, fps=25)
        with pytest.raises(StageError, match="fps"):
            check_track_video(video, expected_fps=30000 / 1001)

    @requires_ffmpeg
    def test_a_really_truncated_file_is_refused(self, tmp_path):
        """The reviewer's measurement, reproduced with a real half-written file.

        Half an mp4 keeps the moov header (faststart puts it first) and therefore keeps
        ``nb_frames``, while ffprobe either decodes fewer frames or cannot finish the decode at
        all. Either way the preview must be refused -- what must *not* happen is a pass based on
        the declared count.
        """
        video = encode(tmp_path / "full.mp4", frames=100)
        cut = tmp_path / "cut.mp4"
        data = video.read_bytes()
        cut.write_bytes(data[: len(data) // 2])
        with pytest.raises(StageError):
            check_track_video(cut, expected_frames=100)

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
        probe = fake_ffprobe(tmp_path, video=False, audio=True)
        with pytest.raises(StageError, match="no video stream"):
            check_track_video(clip, ffprobe=str(probe))


class TestFrameCountIsStrict:
    """The count is decoded, and any difference is a broken preview."""

    def _check(self, tmp_path, *, expected, **probe):
        path = fake_ffprobe(tmp_path, **probe)
        clip = tmp_path / "tracks.mp4"
        clip.write_bytes(b"x")
        return check_track_video(clip, ffprobe=str(path), expected_frames=expected)

    def test_an_exact_match_passes(self, tmp_path):
        assert self._check(tmp_path, expected=120, frames=120)["frames"] == 120

    @pytest.mark.parametrize("expected", [119, 121, 123])
    def test_a_difference_of_any_size_fails_with_both_numbers(self, tmp_path, expected):
        with pytest.raises(StageError, match=f"decodes to 120 frame\\(s\\), expected {expected}"):
            self._check(tmp_path, expected=expected, frames=120)

    def test_a_header_count_that_lies_does_not_save_the_file(self, tmp_path):
        """The reviewer's measurement, as a test: declared 125, decoded 61."""
        with pytest.raises(StageError, match="decodes to 61 frame\\(s\\), expected 125"):
            self._check(tmp_path, expected=125, frames=61, declared=125)

    def test_a_container_that_cannot_be_counted_is_refused(self, tmp_path):
        """Guessing from the duration would let a truncated file through under a rounding."""
        with pytest.raises(StageError, match="decoded frame count could not be read"):
            self._check(tmp_path, expected=120, frames="", declared="125")

    def test_a_container_that_reports_no_frames_is_refused(self, tmp_path):
        """A 0-frame container against a 120-frame measurement is the worst case: the preview
        plays and shows nothing, while the tables describe 120 frames of people."""
        with pytest.raises(StageError, match="decodes to 0 frame"):
            self._check(tmp_path, expected=120, frames=0)


class TestRateAndSizeAreChecked:
    def _check(self, tmp_path, **kwargs):
        probe_kwargs = {key: kwargs.pop(key) for key in list(kwargs)
                        if key in ("frames", "fps", "size")}
        path = fake_ffprobe(tmp_path, **probe_kwargs)
        clip = tmp_path / "tracks.mp4"
        clip.write_bytes(b"x")
        return check_track_video(clip, ffprobe=str(path), **kwargs)

    def test_a_same_count_replacement_at_another_rate_is_refused(self, tmp_path):
        with pytest.raises(StageError, match=r"runs at 25\.000000 fps, not the 29\.970030"):
            self._check(tmp_path, frames=120, fps="25/1", expected_fps=30000 / 1001)

    def test_rounding_of_the_recorded_rate_is_not_a_mismatch(self, tmp_path):
        """ffmpeg writes the nearest rational, so 30000/1001 against 29.970030 must pass."""
        checked = self._check(tmp_path, frames=120, fps="30000/1001",
                              expected_fps=30000 / 1001)
        assert checked["fps"] == pytest.approx(30000 / 1001)
        assert FPS_TOLERANCE < 0.01

    def test_a_preview_with_no_rate_is_refused_when_a_rate_was_recorded(self, tmp_path):
        with pytest.raises(StageError, match="states no frame rate"):
            self._check(tmp_path, frames=120, fps="0/0", expected_fps=25.0)

    def test_a_preview_at_another_size_is_refused(self, tmp_path):
        with pytest.raises(StageError, match=r"is 64x48, not the 66x50"):
            self._check(tmp_path, frames=120, size=(64, 48), expected_size=(66, 50))

    def test_a_preview_at_the_recorded_size_passes(self, tmp_path):
        checked = self._check(tmp_path, frames=120, size=(66, 50), expected_size=(66, 50))
        assert (checked["width"], checked["height"]) == (66, 50)


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

    def test_the_frame_count_is_asked_for_by_counting(self):
        """``-count_frames`` is the whole guarantee; a plain probe reads the header."""
        source = MODULE.read_text(encoding="utf-8")
        assert "\"-count_frames\"" in source
        assert "nb_read_frames" in source
