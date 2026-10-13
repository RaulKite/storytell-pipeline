"""Generated ``explanation.md`` guides inside one clip's output directory.

Two public seams are exercised, and both matter:

* :func:`write_explanations` on a real tree built by ``VideoPaths.ensure_dirs`` — the
  directory set comes from the production artifact registry, not a parallel list;
* :class:`VideoRunner`'s ``run``, the only caller allowed to write. A fully reused run
  and a run whose OpenPose stage empties ``pose/raw_images`` are the two cases that
  break a naive "write it in ``ensure_dirs``" implementation.

Files are read from disk as text: whether a person's notes survived is a fact about a
file, not about an internal data structure.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest

from multimodal_pipeline import output_explanations as guides
from multimodal_pipeline.artifacts import ARTIFACT_LAYOUT, ArtifactRegistry, VideoPaths
from multimodal_pipeline.discovery import VideoSource
from multimodal_pipeline.orchestrator import VideoRunner
from multimodal_pipeline.stages.base import Stage, StageContext
from multimodal_pipeline.stages.openpose import OpenPoseStage

#: Directories whose guide an analyst actually needs. Naming them here (rather than
#: iterating the module's own list) makes this test die when the set shrinks.
EXPECTED_DIRS = (
    ".", "source", "audio", "speech", "speech/raw", "translation", "translation/raw",
    "linguistic", "linguistic/source", "linguistic/source/raw", "linguistic/english",
    "linguistic/english/raw", "acoustic", "acoustic/raw", "pose", "pose/raw",
    "pose/raw_images", "persons", "persons/raw", "speaker", "speaker/raw",
    "stories", "stories/raw", "elan", "logs", "provenance",
)


@pytest.fixture
def dataset(tmp_path: Path) -> VideoPaths:
    """A real output tree: exactly what ``ensure_dirs`` promises, with no artifacts."""
    paths = VideoPaths(tmp_path / "conversation_001")
    paths.ensure_dirs()
    return paths


def guide_text(paths: VideoPaths, relative_dir: str) -> str:
    return (paths.dataset_dir / relative_dir / guides.EXPLANATION_FILENAME).read_text(encoding="utf-8")


def all_guides(paths: VideoPaths) -> list[Path]:
    return sorted(paths.dataset_dir.rglob(guides.EXPLANATION_FILENAME))


# --------------------------------------------------------------------- the directory set

class TestDirectorySet:
    def test_every_existing_output_directory_gets_a_guide(self, dataset: VideoPaths) -> None:
        guides.write_explanations(dataset)
        missing = [d for d in EXPECTED_DIRS
                   if not (dataset.dataset_dir / d / guides.EXPLANATION_FILENAME).is_file()]
        assert missing == []
        assert set(EXPECTED_DIRS) <= set(guides.documented_dirs())

    def test_writing_is_idempotent_and_reports_no_changes(self, dataset: VideoPaths) -> None:
        first = guides.write_explanations(dataset)
        assert first["written"]
        stamp = {p: p.stat().st_mtime_ns for p in all_guides(dataset)}
        second = guides.write_explanations(dataset)
        assert second["written"] == ()
        assert second["unchanged"] == first["written"]
        assert {p: p.stat().st_mtime_ns for p in all_guides(dataset)} == stamp

    def test_a_directory_that_was_never_produced_is_not_created_for_a_guide(
            self, dataset: VideoPaths) -> None:
        shutil.rmtree(dataset.dataset_dir / "persons")
        guides.write_explanations(dataset)
        assert not (dataset.dataset_dir / "persons").exists()
        assert not (dataset.dataset_dir / "persons" / "explanation.md").exists()

    def test_a_symlinked_directory_is_not_written_through(self, dataset: VideoPaths,
                                                         tmp_path: Path) -> None:
        """A guide must stay inside the clip directory, never follow a link out of it."""
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        shutil.move(str(dataset.dataset_dir / "speaker"), str(outside / "speaker"))
        dataset.dataset_dir.joinpath("speaker").symlink_to(outside / "speaker", target_is_directory=True)

        guides.write_explanations(dataset)

        assert not (outside / "speaker" / "explanation.md").exists()
        assert "speaker/explanation.md" not in guides.write_explanations(dataset)["unchanged"]

    def test_every_guide_is_static_markdown(self, dataset: VideoPaths) -> None:
        """No configuration values, no machine paths, no clock in a guide."""
        guides.write_explanations(dataset)
        for path in all_guides(dataset):
            text = path.read_text(encoding="utf-8")
            assert str(dataset.dataset_dir) not in text
            assert str(dataset.dataset_dir.parent) not in text
            assert "sk-" not in text
            assert "/opt/" not in text

    def test_the_generator_never_reads_pipeline_configuration(self) -> None:
        source = Path(guides.__file__).read_text(encoding="utf-8")
        assert "from .config" not in source
        assert "PipelineConfig" not in source


# ---------------------------------------------------------------------- file ownership

class TestOwnership:
    def test_the_marker_is_the_first_line_of_a_generated_guide(self, dataset: VideoPaths) -> None:
        guides.write_explanations(dataset)
        lines = guide_text(dataset, "speech").splitlines()
        assert lines[0] == guides.GENERATOR_MARKER
        assert guides.MARKER_TOKEN in lines[0]
        assert guides.is_generated("\n".join(lines))

    def test_human_note_mentioning_the_marker_token_is_not_generator_owned(self, dataset):
        path = dataset.dataset_dir / 'speech' / guides.EXPLANATION_FILENAME
        text = 'Notes about multimodal-pipeline:output-explanations ownership\nMy notes\n'
        path.write_text(text)
        report = guides.write_explanations(dataset)
        assert path.read_text() == text
        assert 'speech/explanation.md' in report['preserved']

    def test_an_unmarked_guide_belongs_to_a_person_and_is_never_touched(
            self, dataset: VideoPaths) -> None:
        mine = dataset.dataset_dir / "speech" / guides.EXPLANATION_FILENAME
        mine.write_text("my notes: the speaker in shot 3 is not the voice\n", encoding="utf-8")
        report = guides.write_explanations(dataset)
        assert mine.read_text(encoding="utf-8").startswith("my notes:")
        assert "speech/explanation.md" in report["preserved"]
        # The rest of the dataset is still documented.
        assert (dataset.dataset_dir / "pose" / guides.EXPLANATION_FILENAME).is_file()

    def test_a_generated_guide_is_replaced_but_a_persons_is_not(self, dataset: VideoPaths) -> None:
        guides.write_explanations(dataset)
        stale = guide_text(dataset, "pose").replace("body.parquet", "old_name.parquet")
        (dataset.dataset_dir / "pose" / guides.EXPLANATION_FILENAME).write_text(stale, encoding="utf-8")
        report = guides.write_explanations(dataset)
        assert "pose/explanation.md" in report["written"]
        assert "body.parquet" in guide_text(dataset, "pose")

    def test_an_unreadable_existing_file_is_reported_not_raised(self, dataset: VideoPaths,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
        speech = dataset.dataset_dir / "speech"
        (speech / guides.EXPLANATION_FILENAME).write_text("mine", encoding="utf-8")
        real_read_text = Path.read_text

        def boom(self: Path, *args: Any, **kwargs: Any) -> str:
            if self.parent == speech:
                raise OSError("permission denied")
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", boom)
        report = guides.write_explanations(dataset)
        assert "speech/explanation.md" in report["skipped"]
        assert (dataset.dataset_dir / "pose" / guides.EXPLANATION_FILENAME).is_file()

    def test_a_hand_written_guide_that_is_not_utf8_is_left_alone(self, dataset: VideoPaths) -> None:
        """An un-marked guide in someone else's encoding cannot abort the pipeline.

        It cannot be decoded, so it cannot be recognised as ours: the safe answer is
        ``preserved``, and the bytes stay exactly as they were.
        """
        foreign = dataset.dataset_dir / "speech" / guides.EXPLANATION_FILENAME
        foreign.write_bytes(b"\xff\xferemarki: to\x92 samo \xa7")
        before = foreign.read_bytes()
        assert not guides.is_generated(before.decode("utf-8", errors="replace"))

        report = guides.write_explanations(dataset)

        assert foreign.read_bytes() == before
        assert "speech/explanation.md" in report["preserved"]
        assert (dataset.dataset_dir / "pose" / guides.EXPLANATION_FILENAME).is_file()

    def test_a_guide_that_is_a_symlink_is_never_written_through(self, dataset: VideoPaths,
                                                               tmp_path: Path) -> None:
        """Even a marked target stays out of reach: the link is the operator's, not ours."""
        outside = tmp_path / "someone-elses-guide.md"
        outside.write_text(guides.GENERATOR_MARKER + "\nkeep me\n", encoding="utf-8")
        (dataset.dataset_dir / "speech" / guides.EXPLANATION_FILENAME).symlink_to(outside)

        report = guides.write_explanations(dataset)

        assert outside.read_text(encoding="utf-8") == guides.GENERATOR_MARKER + "\nkeep me\n"
        assert "speech/explanation.md" in report["skipped"]


# ------------------------------------------------------------- directories on disk only

class TestExistingDirectories:
    """A real corpus holds more than the layout's fixed slots.

    ``translation/raw`` accumulates request and cache subdirectories, and a stage leaves
    temporary directories behind. Every directory that exists gets a guide; a hidden or
    linked one does not.
    """

    def test_nested_directories_are_documented_with_their_parents_purpose(
            self, dataset: VideoPaths) -> None:
        nested = dataset.dataset_dir / "translation/raw/cache/a1b2c3d4e5f6a7b8"
        nested.mkdir(parents=True)
        (dataset.dataset_dir / "pose/raw/subset").mkdir()

        guides.write_explanations(dataset)

        for directory in ("translation/raw/cache", "translation/raw/cache/a1b2c3d4e5f6a7b8",
                          "pose/raw/subset"):
            path = dataset.dataset_dir / directory / guides.EXPLANATION_FILENAME
            assert path.is_file(), directory
        cached = guide_text(dataset, "translation/raw/cache/a1b2c3d4e5f6a7b8")
        # Inherited semantics, not a generic stub: it says which stage this belongs to.
        assert "translation/raw" in cached
        assert "translation" in cached.lower()
        assert "recursively counted parent directory artifact" in cached

    def test_hidden_and_temporary_directories_are_left_alone(self, dataset: VideoPaths) -> None:
        (dataset.dataset_dir / ".skeleton-mux-1x2y").mkdir()
        (dataset.dataset_dir / "pose/raw_images/.tmp").mkdir()

        guides.write_explanations(dataset)

        assert not (dataset.dataset_dir / ".skeleton-mux-1x2y" / guides.EXPLANATION_FILENAME).exists()
        assert not (dataset.dataset_dir / "pose/raw_images/.tmp" / guides.EXPLANATION_FILENAME).exists()
        written = [str(path.relative_to(dataset.dataset_dir)) for path in all_guides(dataset)]
        assert [item for item in written
                if any(part.startswith(".") for part in item.split("/")[:-1])] == []

    def test_a_linked_subdirectory_is_not_descended(self, dataset: VideoPaths,
                                                   tmp_path: Path) -> None:
        outside = tmp_path / "other-corpus"
        (outside / "inner").mkdir(parents=True)
        (dataset.dataset_dir / "speech" / "other").symlink_to(outside, target_is_directory=True)

        guides.write_explanations(dataset)

        assert not (outside / guides.EXPLANATION_FILENAME).exists()
        assert not (outside / "inner" / guides.EXPLANATION_FILENAME).exists()


# ---------------------------------------------------------------------------- content

class TestContent:
    def test_each_directory_lists_the_files_registered_for_it(self, dataset: VideoPaths) -> None:
        guides.write_explanations(dataset)
        speech = guide_text(dataset, "speech")
        for name in ("segments.parquet", "words.parquet", "speaker_turns.parquet"):
            assert name in speech
        assert "video_out.avi" in guide_text(dataset, "speaker/raw")

    def test_a_file_registered_after_this_module_was_written_is_still_listed(
            self, dataset: VideoPaths, monkeypatch: pytest.MonkeyPatch) -> None:
        """The inventory follows the registry, so it cannot go stale silently."""
        monkeypatch.setitem(ARTIFACT_LAYOUT, "pose_debug_overlay", "pose/debug_overlay.mp4")
        guides.write_explanations(dataset)
        assert "debug_overlay.mp4" in guide_text(dataset, "pose")

    def test_the_registered_skeleton_preview_is_listed_where_it_lives(self, dataset: VideoPaths) -> None:
        """Current registry facts: two independent opt-in previews, one per key."""
        guides.write_explanations(dataset)
        pose = guide_text(dataset, "pose")
        assert "skeletons.mp4" in pose
        assert "openpose.write_video" in pose
        # A native render is the renderer's constant rate, not the measured PTS clock.
        assert "constant rate" in pose
        assert "no audio stream" in pose
        assert "skeletons.mp4" not in guide_text(dataset, "pose/raw_images")

    def test_the_root_guide_explains_the_clocks(self, dataset: VideoPaths) -> None:
        guides.write_explanations(dataset)
        root = guide_text(dataset, ".")
        for phrase in ("pts_seconds", "25 FPS", "source_timestamp", "frame_number",
                       "seconds from video start"):
            assert phrase in root

    def test_the_root_guide_explains_the_id_namespaces(self, dataset: VideoPaths) -> None:
        guides.write_explanations(dataset)
        root = guide_text(dataset, ".")
        for identifier in ("speaker_id", "person_id", "track_id", "segment_id"):
            assert identifier in root
        assert "namespace" in root

    def test_raw_is_explained_as_preserved_tool_output(self, dataset: VideoPaths) -> None:
        guides.write_explanations(dataset)
        raw = guide_text(dataset, "speech/raw")
        assert "byte-identical" in raw
        assert ".provenance.json" in raw
        assert "normalized" in guide_text(dataset, "speech")

    def test_prose_notes_render_as_paragraphs_not_one_run_on_sentence_block(
            self, dataset: VideoPaths) -> None:
        guides.write_explanations(dataset)
        lines = guide_text(dataset, "speech/raw").splitlines()
        index = next(i for i, line in enumerate(lines) if "byte-identical" in line)
        assert lines[index + 1] == ""

    def test_an_absent_optional_output_is_explained(self, dataset: VideoPaths) -> None:
        guides.write_explanations(dataset)
        root = guide_text(dataset, ".")
        assert "artifacts_not_generated" in root
        assert "status.json" in root

    def test_the_native_asd_video_is_not_the_selected_face_table(self, dataset: VideoPaths) -> None:
        guides.write_explanations(dataset)
        raw = guide_text(dataset, "speaker/raw")
        assert "video_out.avi" in raw
        assert "TalkNet" in raw
        assert "selected" in raw
        assert "is_active_speaker" in raw

    def test_the_openpose_skeleton_preview_is_explained(self, dataset: VideoPaths) -> None:
        guides.write_explanations(dataset)
        images = guide_text(dataset, "pose/raw_images")
        assert "openpose.write_images" in images
        assert "preview" in images
        # Render regeneration removes images but preserves folder documentation.
        assert "clears previous render files" in images
        assert "`explanation.md` is preserved" in images
        pose = guide_text(dataset, "pose")
        assert "source pixels" in pose
        assert "normalized.parquet" in pose

    def test_an_unlisted_directory_is_explained_and_otherwise_left_alone(
            self, dataset: VideoPaths) -> None:
        """A folder the layout does not know still gets a guide, and nothing else.

        The guide says what it is not: not registered, not the pipeline's data, not
        going to be written into or deleted from.
        """
        (dataset.dataset_dir / "figures").mkdir()
        (dataset.dataset_dir / "figures" / "pose.png").write_bytes(b"png-bytes")

        guides.write_explanations(dataset)

        assert "figures" in guide_text(dataset, ".")
        guide = guide_text(dataset, "figures")
        assert guides.is_generated(guide)
        assert "no fixed name" in guide
        assert "pose.png" not in guide
        assert (dataset.dataset_dir / "figures" / "pose.png").is_file()


    def test_the_raw_cache_folders_are_named_without_being_enumerated(
            self, dataset: VideoPaths) -> None:
        """A cache holds one file per request: name the folder, never its contents."""
        for folder in ("translation/raw/cache", "stories/raw/cache"):
            (dataset.dataset_dir / folder).mkdir(parents=True)
        for index in range(3):
            (dataset.dataset_dir / "translation/raw/cache" / f"req_{index}.json").write_text(
                "{}", encoding="utf-8")

        guides.write_explanations(dataset)

        translation = guide_text(dataset, "translation/raw")
        assert "`cache/`" in translation
        for index in range(3):
            assert f"req_{index}.json" not in translation
        assert "`cache/`" in guide_text(dataset, "stories/raw")


# ---------------------------------------------------------------- the batch directory

class TestBatchRoot:
    """``output.directory`` itself is an output directory, and gets its own guide."""

    def test_the_batch_directory_is_documented_without_naming_each_clip(self,
                                                                       tmp_path: Path) -> None:
        out = tmp_path / "out"
        (out / "conversation_001" / "speech").mkdir(parents=True)
        (out / "batch_report.json").write_text("{}", encoding="utf-8")

        report = guides.write_batch_explanation(out)

        assert report["written"] == (guides.EXPLANATION_FILENAME,)
        text = (out / guides.EXPLANATION_FILENAME).read_text(encoding="utf-8")
        assert guides.is_generated(text)
        assert "batch_report.json" in text
        assert "explanation.md" in text
        # Static: the clip folders are described, not enumerated, so the text does not
        # change every time a video is added to the corpus.
        assert "conversation_001" not in text

    def test_the_batch_guide_is_idempotent_and_respects_a_persons_file(
            self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        assert guides.write_batch_explanation(out)["written"]
        second = guides.write_batch_explanation(out)
        assert second["unchanged"] == (guides.EXPLANATION_FILENAME,)
        (out / guides.EXPLANATION_FILENAME).write_text("my corpus notes", encoding="utf-8")
        assert guides.write_batch_explanation(out)["preserved"] == (guides.EXPLANATION_FILENAME,)
        assert (out / guides.EXPLANATION_FILENAME).read_text(encoding="utf-8") == "my corpus notes"

    def test_a_symlinked_batch_root_is_not_written_through(self, tmp_path: Path) -> None:
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real, target_is_directory=True)
        assert guides.write_batch_explanation(link)["written"] == ()
        assert not (real / guides.EXPLANATION_FILENAME).exists()


# ---------------------------------------------------------------- the artifact registry

class TestRegistryUnaffected:
    """A guide is documentation, so it must never look like data to the registry.

    ``ArtifactRegistry`` counts the files inside a directory artifact. If a guide were
    counted, an empty ``pose/raw_images`` would be promised in the manifest as a render
    that never happened, and a rerun's byte totals would move with the prose.
    """

    def test_a_generated_guide_changes_no_count_and_adds_no_artifact(self, dataset: VideoPaths) -> None:
        raw = dataset.artifact("pose_raw")
        (raw / "frame_000000000042_keypoints.json").write_text("{}", encoding="utf-8")
        registry = ArtifactRegistry(dataset).refresh()
        before = dict(registry.present["pose_raw"])

        guides.write_explanations(dataset)
        registry.refresh()

        assert registry.present["pose_raw"] == before

    def test_a_directory_holding_only_a_guide_is_still_not_an_artifact(self, dataset: VideoPaths) -> None:
        guides.write_explanations(dataset)
        registry = ArtifactRegistry(dataset).refresh()
        for name in ("pose_raw", "pose_images_raw", "translation_raw", "stories_raw"):
            assert not registry.has(name), f"{name} would be promised to the manifest"

    def test_a_handwritten_guide_is_excluded_by_the_same_rule(self, dataset: VideoPaths) -> None:
        raw = dataset.artifact("pose_raw")
        (raw / "explanation.md").write_text("my notes", encoding="utf-8")
        (raw / "frame_0.json").write_text("{}", encoding="utf-8")
        registry = ArtifactRegistry(dataset).refresh()
        assert registry.present["pose_raw"]["file_count"] == 1


# ------------------------------------------------------------------- the runner seam

class FakeStage(Stage):
    """Writes a real artifact into a real registry slot, like the resume tests' fake."""

    executions: dict[str, int] = {}
    hooks: dict[str, Callable[[StageContext], None]] = {}

    def __init__(self, name: str, artifact: str, *, depends_on: tuple[str, ...] = ()) -> None:
        self.name = name
        self.outputs = (artifact,)
        self.inputs = tuple(dep_artifact[dep] for dep in depends_on)

    def config_fingerprint(self, ctx: StageContext) -> dict[str, Any]:
        return {"stage": self.name}

    def execute(self, ctx: StageContext) -> dict[str, Any]:
        hook = type(self).hooks.get(self.name)
        if hook:
            hook(ctx)
        type(self).executions[self.name] = type(self).executions.get(self.name, 0) + 1
        path = ctx.artifact(self.outputs[0])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"stage": self.name}), encoding="utf-8")
        return {}

    def validate(self, ctx: StageContext) -> dict[str, Any]:
        path = ctx.artifact(self.outputs[0])
        if not path.is_file():
            from multimodal_pipeline.exceptions import ValidationError

            raise ValidationError(self.name, [f"missing {path.name}"])
        return {"ok": True}


FAKE_DAG = (("metadata", "metadata", ()), ("audio", "audio", ("metadata",)),
            ("pose", "pose_body", ("metadata",)))
dep_artifact = {"metadata": "metadata", "audio": "audio", "pose": "pose_body"}


@pytest.fixture
def runner(config, monkeypatch: pytest.MonkeyPatch) -> VideoRunner:
    FakeStage.executions = {}
    FakeStage.hooks = {}
    stages = [FakeStage(name, artifact, depends_on=deps) for name, artifact, deps in FAKE_DAG]
    names = tuple(stage.name for stage in stages)
    dependencies = {name: deps for name, _artifact, deps in FAKE_DAG}
    monkeypatch.setattr("multimodal_pipeline.stages.base.STAGE_ORDER", names)
    monkeypatch.setattr("multimodal_pipeline.stages.base.STAGE_DEPENDENCIES", dependencies)
    monkeypatch.setattr("multimodal_pipeline.orchestrator.STAGE_ORDER", names)
    monkeypatch.setattr("multimodal_pipeline.orchestrator.build_stages", lambda: list(stages))
    source = VideoSource(path=config.input.directory / "clip.mp4",
                         relative_path=Path("clip.mp4"), video_id="clip")
    source.path.write_bytes(b"video-bytes")
    return VideoRunner(config, source, tools={"schema_version": "1.0"})


class TestRunnerWritesGuides:
    def test_a_run_leaves_a_guide_in_every_existing_directory(self, runner: VideoRunner) -> None:
        runner.run()
        missing = [d for d in EXPECTED_DIRS
                   if not (runner.paths.dataset_dir / d / guides.EXPLANATION_FILENAME).is_file()]
        assert missing == []

    def test_a_fully_reused_run_backfills_the_guides(self, runner: VideoRunner) -> None:
        """Reuse returns before any stage runs; the guides must not depend on a stage."""
        runner.run()
        for path in all_guides(runner.paths):
            path.unlink()
        FakeStage.executions.clear()

        resumed = VideoRunner(runner.config, runner.source, tools={"schema_version": "1.0"})
        resumed.run()

        assert FakeStage.executions == {}
        assert all(path.is_file() for path in all_guides(runner.paths))
        assert len(all_guides(runner.paths)) >= len(EXPECTED_DIRS)

    def test_a_guide_removed_with_the_previous_render_comes_back(self, runner: VideoRunner) -> None:
        """OpenPose empties ``pose/raw_images`` mid-run, so docs are rewritten after it."""
        images = runner.paths.artifact("pose_images_raw")
        images.mkdir(parents=True, exist_ok=True)
        (images / "frame_000000.jpg").write_bytes(b"jpeg-bytes")
        FakeStage.hooks["pose"] = lambda ctx: OpenPoseStage().clear_render_dir(ctx)

        runner.run()

        assert not (images / "frame_000000.jpg").exists()
        assert (images / guides.EXPLANATION_FILENAME).is_file()
        assert guides.is_generated((images / guides.EXPLANATION_FILENAME).read_text(encoding="utf-8"))

    def test_planning_and_status_never_write_a_guide(self, runner: VideoRunner) -> None:
        runner.plan()
        runner.plan(force_stages=["pose"])
        runner.stage_context()
        assert all_guides(runner.paths) == []

    def test_a_broken_guide_writer_cannot_break_a_run(
            self, runner: VideoRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        def explode(paths: VideoPaths) -> dict[str, tuple[str, ...]]:
            raise RuntimeError("disk is having a day")

        monkeypatch.setattr("multimodal_pipeline.orchestrator.write_explanations", explode)
        result = runner.run()
        assert result.status == "completed"
        assert FakeStage.executions == {"metadata": 1, "audio": 1, "pose": 1}

    def test_the_run_still_records_its_stages(self, runner: VideoRunner) -> None:
        """PRESERVE: adding docs to ``run`` must not disturb what it already reported."""
        result = runner.run()
        assert result.status == "completed"
        assert result.stage_outcomes == {"metadata": "completed", "audio": "completed",
                                         "pose": "completed"}
        payload = json.loads(runner.paths.status.read_text(encoding="utf-8"))
        assert payload["stages"]["pose"]["status"] == "completed"
        assert FakeStage.executions == {"metadata": 1, "audio": 1, "pose": 1}

    def test_a_normal_run_also_documents_the_batch_directory(self, runner: VideoRunner) -> None:
        """``output.directory/<video_id>`` is the normal shape, and its parent is an output dir."""
        assert runner.paths.dataset_dir.parent == runner.config.output.directory
        runner.run()
        batch = runner.config.output.directory / guides.EXPLANATION_FILENAME
        assert batch.is_file()
        assert guides.is_generated(batch.read_text(encoding="utf-8"))

    def test_a_scratch_dataset_parent_is_not_documented_as_the_corpus(
            self, runner: VideoRunner, tmp_path: Path) -> None:
        """A ``dataset_root`` override must not get a guide that claims to be the batch root."""
        scratch = tmp_path / "scratch" / "clip"
        second = VideoRunner(runner.config, runner.source, dataset_root=scratch,
                             tools={"schema_version": "1.0"})
        second.run()
        assert (scratch / guides.EXPLANATION_FILENAME).is_file()
        assert not (scratch.parent / guides.EXPLANATION_FILENAME).exists()
        assert not (runner.config.output.directory / guides.EXPLANATION_FILENAME).exists()
