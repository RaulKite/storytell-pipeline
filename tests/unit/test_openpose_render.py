"""Unit tests for opt-in rendered OpenPose skeletons (``openpose.write_images``).

The stage had no tests at all before this feature, so the contract that matters most
is the argv one: this build of OpenPose renders per module (``--render_pose``,
``--face_render`` and ``--hand_render`` each accept ``-1`` to inherit), and a flag
group that is silently wrong produces either a dataset with no images or a run that
costs gigabytes of disk for nothing. Two failure modes are treated as hard failures
here:

* the command changes when rendering is **off** — the default must keep producing the
  exact argv the pipeline has always produced, or every existing pose dataset reruns
  the slowest stage for no reason;
* rendering is asked for and nothing is written — a run that "succeeds" with zero
  images is worse than a failed one, because the dataset looks complete.

Nothing here mocks the argument construction: ``openpose.executable`` and
``openpose.model_folder`` point at a throwaway executable/directory in ``tmp_path``,
so the real resolution code runs and only the binary itself is absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from multimodal_pipeline.artifacts import atomic_write_json
from multimodal_pipeline.exceptions import StageError, ValidationError
from multimodal_pipeline.schemas import (
    BODY_SCHEMA,
    FACE_SCHEMA,
    FRAME_INDEX_SCHEMA,
    HANDS_SCHEMA,
    write_table,
)
from multimodal_pipeline.stages import openpose as openpose_module
from multimodal_pipeline.stages.openpose import OpenPoseStage

IMAGE_SUFFIXES = (".jpg", ".png", ".jpeg")


class Recorder:
    """Stand-in for ``StageLogger`` that keeps (message, level) for assertions."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, int]] = []

    def __call__(self, message: str, level: int = 20) -> None:
        self.lines.append((str(message), int(level)))

    def text(self) -> str:
        return "\n".join(message for message, _ in self.lines)

    def at(self, level: int) -> list[str]:
        return [message for message, recorded in self.lines if recorded == level]


@pytest.fixture
def fake_openpose(tmp_path: Path) -> dict[str, str]:
    """A real executable file and a real model folder, so resolution is not faked.

    ``build_command`` resolves both through the production code paths
    (``require_executable`` and the ``model_folder`` directory check), which is the
    only way to reach argv without the 4 GB binary. The argument logic itself is
    never stubbed.
    """
    binary = tmp_path / "bin" / "openpose.bin"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    models = tmp_path / "models"
    models.mkdir()
    return {"executable": str(binary), "model_folder": str(models)}


@pytest.fixture
def stage_ctx(context, fake_openpose: dict[str, str]):
    """The shared context with a real metadata artifact and fake binary paths."""
    context.config.openpose.executable = fake_openpose["executable"]
    context.config.openpose.model_folder = fake_openpose["model_folder"]
    write_metadata(context)
    context.log = Recorder()
    return context


def write_metadata(ctx, **overrides: Any) -> Path:
    """A metadata artifact shaped like the real one (it carries width/height)."""
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "video_id": ctx.video_id,
        "source_filename": "conversation_001.mp4",
        "source_path": str(ctx.source.path),
        "SHA256": "0" * 64,
        "file_size_bytes": 1234,
        "duration_seconds": 4.0,
        "width": 1280,
        "height": 720,
        "frame_count": 100,
    }
    payload.update(overrides)
    path = ctx.artifact("metadata")
    atomic_write_json(path, payload)
    return path


def render_config(ctx, **overrides: Any) -> None:
    for key, value in overrides.items():
        setattr(ctx.config.openpose, key, value)


def value_of(argv: list[str], flag: str) -> str | None:
    """Value of a ``--flag value`` pair, or None when the flag is absent."""
    return argv[argv.index(flag) + 1] if flag in argv else None


def seed_raw(ctx, frames: int) -> int:
    raw = ctx.artifact("pose_raw")
    raw.mkdir(parents=True, exist_ok=True)
    for index in range(frames):
        document = {"version": 1.3, "people": []}
        (raw / f"{ctx.video_id}_{index:012d}_keypoints.json").write_text(
            json.dumps(document), encoding="utf-8"
        )
    return frames


def seed_images(ctx, count: int, *, suffix: str = ".jpg") -> int:
    images = ctx.artifact("pose_images_raw")
    images.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (images / f"{ctx.video_id}_keypoints_{index:012d}{suffix}").write_bytes(b"\xff" * 64)
    return count


def seed_frame_index(ctx, frames: int) -> None:
    """The ffprobe packet index execute() needs to time the keypoints."""
    write_table(ctx.artifact("frame_index"), pa.Table.from_pylist([
        {"schema_version": "1.0", "video_id": ctx.video_id, "frame_number": index,
         "pts_seconds": round(index * 0.04, 6)}
        for index in range(frames)
    ]), FRAME_INDEX_SCHEMA)


def seed_tables(ctx, frames: int) -> None:
    """Minimal but schema-valid body/hands/face tables for ``validate``."""
    timestamps = [round(index * 0.04, 6) for index in range(frames)]
    write_table(ctx.artifact("pose_body"), pa.Table.from_pylist([
        {"schema_version": "1.0", "video_id": ctx.video_id, "frame_number": index,
         "timestamp": timestamps[index], "detection_index": 0, "keypoint_id": 1,
         "keypoint_name": "Neck", "x": 10.0, "y": 20.0, "confidence": 0.9}
        for index in range(frames)
    ]), BODY_SCHEMA)
    write_table(ctx.artifact("pose_hands"), pa.Table.from_pylist([
        {"schema_version": "1.0", "video_id": ctx.video_id, "frame_number": index,
         "timestamp": timestamps[index], "detection_index": 0, "hand": "right",
         "keypoint_id": 0, "keypoint_name": "Wrist", "x": 11.0, "y": 21.0, "confidence": 0.8}
        for index in range(frames)
    ]), HANDS_SCHEMA)
    write_table(ctx.artifact("pose_face"), pa.Table.from_pylist([
        {"schema_version": "1.0", "video_id": ctx.video_id, "frame_number": index,
         "timestamp": timestamps[index], "detection_index": 0, "landmark_id": 0,
         "x": 12.0, "y": 22.0, "confidence": 0.7}
        for index in range(frames)
    ]), FACE_SCHEMA)


class FakeRun:
    """Stands in for ``run_command`` only: the binary is the absent dependency here."""

    def __init__(self, argv: list[str]) -> None:
        self.argv = list(argv)
        self.returncode = 0

    @property
    def argv_masked(self) -> list[str]:
        return list(self.argv)


def install_fake_run(monkeypatch: pytest.MonkeyPatch, ctx, writes: int) -> list[list[str]]:
    """Replace only the subprocess call.

    ``writes`` is how many rendered files the pretend binary leaves behind, which is
    the whole variable these tests exercise: OpenPose can exit 0 having written no
    images at all, and the stage has to notice.
    """
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs: Any) -> FakeRun:
        calls.append(list(argv))
        if writes:
            seed_images(ctx, writes)
        return FakeRun(argv)

    monkeypatch.setattr(openpose_module, "run_command", fake_run)
    return calls


class TestArtifactRegistry:
    def test_rendered_images_have_a_layout_entry(self) -> None:
        from multimodal_pipeline.artifacts import ARTIFACT_LAYOUT

        assert ARTIFACT_LAYOUT["pose_images_raw"] == "pose/raw_images"

    def test_stage_declares_it_as_an_output(self) -> None:
        assert "pose_images_raw" in OpenPoseStage.outputs

    def test_the_slot_is_pre_created_like_pose_raw(self, stage_ctx) -> None:
        """Otherwise ``outputs_present`` fails and reuse never happens when off.

        ``pose/raw_images`` is an optional output: it stays declared so a dataset
        rendered with the flag on is reported completely, and the directory exists
        but empty when the flag is off -- exactly how ``pose/raw`` already behaves.
        """
        assert stage_ctx.artifact("pose_images_raw").is_dir()
        assert not stage_ctx.registry.refresh().has("pose_images_raw")

    def test_a_rendered_directory_is_reported_as_an_artifact(self, stage_ctx) -> None:
        seed_images(stage_ctx, 2)
        info = stage_ctx.registry.refresh().present["pose_images_raw"]
        assert info["kind"] == "directory"
        assert info["file_count"] == 2


class TestDefaultCommandUnchanged:
    """Default-off must mean byte-identical argv, not 'equivalent' argv."""

    def test_rendering_off_keeps_the_pre_change_command(self, stage_ctx) -> None:
        raw = stage_ctx.artifact("pose_raw")
        argv = OpenPoseStage().build_command(stage_ctx, raw)
        assert argv == [
            stage_ctx.config.openpose.executable,
            "--video", str(stage_ctx.source.path),
            "--model_folder", stage_ctx.config.openpose.model_folder,
            "--write_json", str(raw),
            "--num_gpu", "1",
            "--num_gpu_start", "0",
            "--display", "0",
            "--render_pose", "0",
            "--disable_multi_thread",
            "--model_pose", "BODY_25",
            "--hand",
            "--face",
            "--logtostderr",
            "--alsologtostderr",
        ]

    def test_rendering_off_writes_no_images_and_no_output_resolution(self, stage_ctx) -> None:
        argv = OpenPoseStage().build_command(stage_ctx, stage_ctx.artifact("pose_raw"))
        assert "--write_images" not in argv
        assert "--write_images_format" not in argv
        assert "--output_resolution" not in argv

    def test_image_max_side_alone_does_not_touch_the_command(self, stage_ctx) -> None:
        """Rendering off means off: a stray resolution must not change the argv."""
        render_config(stage_ctx, image_max_side=640)
        argv = OpenPoseStage().build_command(stage_ctx, stage_ctx.artifact("pose_raw"))
        assert "--output_resolution" not in argv
        assert "--render_pose" in argv and value_of(argv, "--render_pose") == "0"


class TestRenderFlags:
    """(b) the flag group, and the per-module switches honouring the enabled config."""

    @pytest.mark.parametrize(
        "body_enabled,hands_enabled,face_enabled,expected",
        [
            (True, True, True, {"--render_pose": "-1", "--face_render": "1", "--hand_render": "1"}),
            (True, False, True, {"--render_pose": "-1", "--face_render": "1", "--hand_render": "0"}),
            (True, True, False, {"--render_pose": "-1", "--face_render": "0", "--hand_render": "1"}),
            # Body off: the CLI cannot disable body *tracking*, so the module runs, but
            # its skeleton must not be drawn while pose/body.parquet is unpublished.
            (False, True, True, {"--render_pose": "0", "--face_render": "1", "--hand_render": "1"}),
        ],
    )
    def test_switches_follow_the_enabled_modules(
        self, stage_ctx, body_enabled: bool, hands_enabled: bool, face_enabled: bool,
        expected: dict[str, str],
    ) -> None:
        stage_ctx.config.openpose.body.enabled = body_enabled
        stage_ctx.config.openpose.hands = {"enabled": hands_enabled}
        stage_ctx.config.openpose.face = {"enabled": face_enabled}
        render_config(stage_ctx, write_images=True)

        argv = OpenPoseStage().build_command(stage_ctx, stage_ctx.artifact("pose_raw"))

        for flag, value in expected.items():
            assert flag in argv, f"{flag} missing from {argv}"
            assert value_of(argv, flag) == value
        # The detector switches and the render switches must agree.
        assert ("--hand=0" in argv) is not hands_enabled
        assert ("--face=0" in argv) is not face_enabled

    def test_write_images_points_at_the_layout_directory_with_jpg(self, stage_ctx) -> None:
        render_config(stage_ctx, write_images=True)
        argv = OpenPoseStage().build_command(stage_ctx, stage_ctx.artifact("pose_raw"))

        images = stage_ctx.artifact("pose_images_raw")
        assert value_of(argv, "--write_images") == str(images)
        assert value_of(argv, "--write_images_format") == "jpg"
        assert images.parts[-2:] == ("pose", "raw_images")

    def test_extra_args_still_come_last(self, stage_ctx) -> None:
        stage_ctx.config.openpose.extra_args = ["--hand_size", "1.0"]
        render_config(stage_ctx, write_images=True)
        argv = OpenPoseStage().build_command(stage_ctx, stage_ctx.artifact("pose_raw"))
        assert argv[-2:] == ["--hand_size", "1.0"]


class TestOutputResolution:
    """(c) --output_resolution only with image_max_side, scaled to fit the source."""

    def test_absent_without_image_max_side(self, stage_ctx) -> None:
        render_config(stage_ctx, write_images=True)
        argv = OpenPoseStage().build_command(stage_ctx, stage_ctx.artifact("pose_raw"))
        assert "--output_resolution" not in argv

    def test_present_and_aspect_preserving_when_set(self, stage_ctx) -> None:
        write_metadata(stage_ctx, width=1280, height=720)
        render_config(stage_ctx, write_images=True, image_max_side=640)
        argv = OpenPoseStage().build_command(stage_ctx, stage_ctx.artifact("pose_raw"))
        # 1280x720 scaled to fit 640 on the long side is 640x360.
        assert value_of(argv, "--output_resolution") == "640x360"

    def test_uses_the_short_side_when_the_video_is_portrait(self, stage_ctx) -> None:
        write_metadata(stage_ctx, width=720, height=1280)
        render_config(stage_ctx, write_images=True, image_max_side=640)
        argv = OpenPoseStage().build_command(stage_ctx, stage_ctx.artifact("pose_raw"))
        assert value_of(argv, "--output_resolution") == "360x640"

    def test_never_upscales(self, stage_ctx) -> None:
        write_metadata(stage_ctx, width=320, height=240)
        render_config(stage_ctx, write_images=True, image_max_side=640)
        argv = OpenPoseStage().build_command(stage_ctx, stage_ctx.artifact("pose_raw"))
        assert value_of(argv, "--output_resolution") == "320x240"

    @pytest.mark.parametrize("payload", [
        {"width": None, "height": None},
        {"width": 640, "height": None},
        {"width": None, "height": 480},
    ])
    def test_skips_the_downscale_and_says_so_when_metadata_lacks_dimensions(
        self, stage_ctx, monkeypatch: pytest.MonkeyPatch, payload: dict
    ) -> None:
        """Rendering still happens at full resolution, so the run must not be quiet."""
        write_metadata(stage_ctx, **payload)
        render_config(stage_ctx, write_images=True, image_max_side=640)
        argv = OpenPoseStage().build_command(stage_ctx, stage_ctx.artifact("pose_raw"))
        assert "--output_resolution" not in argv
        assert value_of(argv, "--write_images") == str(stage_ctx.artifact("pose_images_raw"))

        install_fake_run(monkeypatch, stage_ctx, writes=2)
        seed_raw(stage_ctx, 2)
        seed_frame_index(stage_ctx, 2)
        seed_tables(stage_ctx, 2)
        OpenPoseStage().execute(stage_ctx)

        assert any("image_max_side" in line and "resolution" in line.lower()
                   for line in stage_ctx.log.at(30)), stage_ctx.log.lines


class TestExecuteRenderReport:
    """(d) the rendered count is reported, and an empty render fails loudly."""

    def test_zero_rendered_images_fails_the_run(self, stage_ctx, monkeypatch: pytest.MonkeyPatch) -> None:
        render_config(stage_ctx, write_images=True)
        install_fake_run(monkeypatch, stage_ctx, writes=0)
        seed_raw(stage_ctx, 3)
        seed_frame_index(stage_ctx, 3)
        seed_tables(stage_ctx, 3)

        with pytest.raises(StageError) as excinfo:
            OpenPoseStage().execute(stage_ctx)

        message = str(excinfo.value)
        for flag in ("--write_images", "--render_pose", "--face_render", "--hand_render"):
            assert flag in message, f"{flag} not named in the failure: {message}"
        assert stage_ctx.log.at(40), "the empty render must also be logged as an error"

    def test_rendered_count_reaches_the_run_summary(
        self, stage_ctx, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        render_config(stage_ctx, write_images=True)
        install_fake_run(monkeypatch, stage_ctx, writes=5)
        seed_raw(stage_ctx, 5)
        seed_frame_index(stage_ctx, 5)
        seed_tables(stage_ctx, 5)

        extras = OpenPoseStage().execute(stage_ctx)

        assert extras["extra"]["rendered_images"] == 5
        assert extras["extra"]["raw_frames"] == 5
        assert extras["extra"]["render"]["render_pose"] == "-1"
        assert "5 rendered images" in stage_ctx.log.text()

    def test_rendering_off_reports_zero_without_failing(
        self, stage_ctx, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_run(monkeypatch, stage_ctx, writes=0)
        seed_raw(stage_ctx, 4)
        seed_frame_index(stage_ctx, 4)
        seed_tables(stage_ctx, 4)

        extras = OpenPoseStage().execute(stage_ctx)

        assert extras["extra"]["rendered_images"] == 0
        assert not stage_ctx.log.at(40)

    def test_warns_once_about_full_resolution(
        self, stage_ctx, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        render_config(stage_ctx, write_images=True, image_max_side=None)
        install_fake_run(monkeypatch, stage_ctx, writes=2)
        seed_raw(stage_ctx, 2)
        seed_frame_index(stage_ctx, 2)
        seed_tables(stage_ctx, 2)

        OpenPoseStage().execute(stage_ctx)

        warnings = [line for line in stage_ctx.log.at(30) if "image_max_side" in line]
        assert len(warnings) == 1, f"expected exactly one disk warning, got {warnings}"

    def test_stays_silent_about_disk_when_downscaling(
        self, stage_ctx, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        render_config(stage_ctx, write_images=True, image_max_side=640)
        install_fake_run(monkeypatch, stage_ctx, writes=2)
        seed_raw(stage_ctx, 2)
        seed_frame_index(stage_ctx, 2)
        seed_tables(stage_ctx, 2)

        OpenPoseStage().execute(stage_ctx)

        assert not [line for line in stage_ctx.log.at(30) if "image_max_side" in line]


class TestValidationCountRule:
    """(e) one image per processed frame, with one frame of slack."""

    @pytest.mark.parametrize("images,ok", [(3, True), (2, True), (1, False), (0, False)])
    def test_count_must_match_the_raw_frames(self, stage_ctx, images: int, ok: bool) -> None:
        render_config(stage_ctx, write_images=True)
        seed_raw(stage_ctx, 3)
        seed_tables(stage_ctx, 3)
        if images:
            seed_images(stage_ctx, images)

        if ok:
            report = OpenPoseStage().validate(stage_ctx)
            assert report["rendered_images"] == images
        else:
            with pytest.raises(ValidationError) as excinfo:
                OpenPoseStage().validate(stage_ctx)
            assert "pose/raw_images" in str(excinfo.value)

    def test_png_renders_count_too(self, stage_ctx) -> None:
        """``extra_args`` can override the format; counting must follow the files."""
        render_config(stage_ctx, write_images=True)
        seed_raw(stage_ctx, 2)
        seed_frame_index(stage_ctx, 2)
        seed_tables(stage_ctx, 2)
        seed_images(stage_ctx, 2, suffix=".png")
        assert OpenPoseStage().validate(stage_ctx)["rendered_images"] == 2

    def test_rendering_off_does_not_require_images(self, stage_ctx) -> None:
        seed_raw(stage_ctx, 3)
        seed_tables(stage_ctx, 3)
        images = stage_ctx.artifact("pose_images_raw")
        # The slot is pre-created like pose/raw; prove both shapes are accepted.
        assert images.is_dir()
        assert OpenPoseStage().validate(stage_ctx)["rendered_images"] == 0
        import shutil
        shutil.rmtree(images)
        assert OpenPoseStage().validate(stage_ctx)["rendered_images"] == 0

    def test_a_stale_directory_is_ignored_when_rendering_is_off(self, stage_ctx) -> None:
        """Turning the flag back off must not make an existing dataset invalid.

        The leftover count is still reported -- the report says what is on disk --
        it is simply not required. The fingerprint, not validation, is what decides
        that a dataset rendered under a different configuration needs a rerun.
        """
        seed_raw(stage_ctx, 3)
        seed_tables(stage_ctx, 3)
        seed_images(stage_ctx, 1)  # leftover from an earlier opt-in run
        assert OpenPoseStage().validate(stage_ctx)["rendered_images"] == 1

    def test_validate_still_catches_the_pre_existing_problems(self, stage_ctx) -> None:
        """The new check must not soften anything that was already checked."""
        render_config(stage_ctx, write_images=True)
        seed_raw(stage_ctx, 3)
        seed_images(stage_ctx, 3)
        # No Parquet tables at all.
        with pytest.raises(ValidationError):
            OpenPoseStage().validate(stage_ctx)


class TestFingerprint:
    """(f) both new keys participate in invalidation."""

    def test_keys_are_present(self, stage_ctx) -> None:
        fingerprint = OpenPoseStage().config_fingerprint(stage_ctx)
        assert fingerprint["write_images"] is False
        assert fingerprint["image_max_side"] is None

    def test_flipping_write_images_changes_the_hash(self, stage_ctx) -> None:
        from multimodal_pipeline.config import stable_hash

        stage = OpenPoseStage()
        before = stage.config_fingerprint(stage_ctx)
        render_config(stage_ctx, write_images=True)
        after = stage.config_fingerprint(stage_ctx)
        assert before != after
        assert stable_hash(before) != stable_hash(after)

    def test_changing_image_max_side_changes_the_hash(self, stage_ctx) -> None:
        render_config(stage_ctx, write_images=True)
        stage = OpenPoseStage()
        before = stage.config_fingerprint(stage_ctx)
        render_config(stage_ctx, image_max_side=640)
        assert stage.config_fingerprint(stage_ctx) != before

    def test_rendering_off_still_hashes_image_max_side(self, stage_ctx) -> None:
        """A pending resolution must not become live the moment rendering is enabled."""
        stage = OpenPoseStage()
        before = stage.config_fingerprint(stage_ctx)
        render_config(stage_ctx, image_max_side=640)
        assert stage.config_fingerprint(stage_ctx) != before
