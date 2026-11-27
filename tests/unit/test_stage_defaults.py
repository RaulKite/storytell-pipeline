"""Every stage runs by default, and a missing prerequisite skips instead of failing.

Two operator decisions are pinned here, and they only work together:

1. **Every tool/analysis stage is enabled by default**, in the code defaults and in
   ``config/config.example.yaml``. A config that names nothing but its input and output
   directories still measures everything this pipeline can measure. Preview rendering
   (``openpose.write_images``, ``openpose.write_video``, ``activespeaker.write_video``) is
   the deliberate exception: it produces no analysis and each one reruns its stage.
2. **A missing prerequisite is a skip, not a failure.** Environments, binaries and
   checkpoints are install state, not defects in the video, so the stage reports a reason
   naming what to install and the rest of the corpus completes. A *present but broken*
   install still fails loudly — that is a real defect and this file does not excuse it.

The second half is the half that costs something to get wrong: with every stage on, no
machine has all of ``environments/*``, an OpenPose checkout, a TalkNet checkout and an LLM
endpoint, so failing instead of skipping would turn the default run on a fresh clone into a
corpus of failed videos.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from multimodal_pipeline.config import PipelineConfig, load_config
from multimodal_pipeline.exceptions import StageError, WorkerError
from multimodal_pipeline.orchestrator import build_stages
from multimodal_pipeline.stages.base import STAGE_ORDER

ROOT = Path(__file__).resolve().parents[2]

#: Stage -> the config section that owns its enable flag. Stages absent from this map have no
#: flag at all (``metadata``, ``audio``, ``speaker_assignment``, ``finalization``): they are
#: unconditional, or they run when the input they read exists.
STAGE_FLAGS: dict[str, str] = {
    "whisperx": "whisperx",
    "diarization_nemotron": "diarization_nemotron",
    "translation": "translation",
    "spacy_source": "spacy",
    "spacy_english": "spacy",
    "acoustic": "acoustic",
    "openpose": "openpose",
    "pose_normalized": "pose_normalized",
    "activespeaker": "activespeaker",
    "persons": "persons",
    "stories": "stories",
    "elan": "elan",
}

#: The switches that stay off by default: they render a preview of a measurement instead of
#: producing one, so each costs a rerun and disk without adding a column to the dataset.
PREVIEW_SWITCHES = (
    ("openpose", "write_images"),
    ("openpose", "write_video"),
    ("activespeaker", "write_video"),
)

#: False flags that are not stage switches: discovery/execution policy, not a tool or an
#: analysis. Listed so the "nothing else is off" assertion below stays honest.
NON_STAGE_FALSE_FLAGS = {
    ("input", "recursive"),
    ("execution", "stop_on_video_error"),
    ("execution", "keep_temp_files"),
}


def stage_named(name: str):
    return {candidate.name: candidate for candidate in build_stages()}[name]


def write_config(root: Path, payload: dict) -> Path:
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "videos").mkdir(parents=True, exist_ok=True)
    document = {"project_root": str(root),
                "input": {"directory": str(root / "videos")},
                "output": {"directory": str(root / "out")},
                **payload}
    path = root / "config" / "config.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


class TestEveryStageIsOnByDefault:
    def test_a_config_with_no_stage_sections_enables_every_stage(self, tmp_path: Path) -> None:
        """The acceptance check: input and output are the only required keys."""
        config = load_config(write_config(tmp_path, {}))
        off = [stage for stage in STAGE_ORDER
               if stage in STAGE_FLAGS
               and not getattr(getattr(config, STAGE_FLAGS[stage]), "enabled", True)]
        assert off == [], f"stages off in a config that named none: {off}"
        # The two that were historically off, asserted by name rather than by inference.
        assert config.persons.enabled is True
        assert config.activespeaker.enabled is True

    def test_the_code_defaults_match(self) -> None:
        config = PipelineConfig.model_validate({"input": {"directory": "/tmp/in"},
                                               "output": {"directory": "/tmp/out"}})
        for stage, section in STAGE_FLAGS.items():
            assert getattr(getattr(config, section), "enabled", True) is True, stage

    def test_the_shipped_example_enables_every_stage(self) -> None:
        """The example is the file an operator copies; it cannot disagree with the defaults."""
        raw = yaml.safe_load((ROOT / "config" / "config.example.yaml").read_text())
        for stage, section in STAGE_FLAGS.items():
            assert section in raw, f"{section} lost from config.example.yaml ({stage})"
            assert raw[section].get("enabled", True) is True, (
                f"{stage}: config.example.yaml ships {section}.enabled off")

    def test_only_preview_rendering_stays_off(self) -> None:
        """The exception is a decision, not a leftover: assert it for what it is."""
        raw = yaml.safe_load((ROOT / "config" / "config.example.yaml").read_text())
        for section, key in PREVIEW_SWITCHES:
            assert raw[section].get(key, False) is False, f"{section}.{key} must stay opt-in"
        # No other false flag exists in the example, so a new "off by default" has to be
        # added here deliberately instead of drifting in.
        false_flags = {(section, key) for section, body in raw.items()
                       if isinstance(body, dict)
                       for key, value in body.items() if value is False}
        unexpected = false_flags - set(PREVIEW_SWITCHES) - NON_STAGE_FALSE_FLAGS
        assert unexpected == set(), unexpected


class TestMissingPrerequisitesSkip:
    """One filesystem fact per stage, and none of them may fail a video."""

    @pytest.mark.parametrize(("stage", "section"), [
        ("whisperx", "whisperx"),
        ("diarization_nemotron", "diarization_nemotron"),
        ("acoustic", "acoustic"),
        ("spacy_source", "spacy"),
        ("persons", "persons"),
        ("activespeaker", "activespeaker"),
    ])
    def test_a_missing_environment_skips_with_the_remedy(self, context, tmp_path: Path,
                                                        stage: str, section: str) -> None:
        """Enabled but never synced: skip, name the directory, name ``uv sync``.

        The shared fixture's project root has no ``environments/`` at all, which is exactly
        the fresh-clone state. ``activespeaker`` gets a valid TalkNet checkout first so that
        the environment is the gate under test rather than the checkout.
        """
        if stage == "activespeaker":
            checkout = tmp_path / "TalkNet-ASD"
            checkout.mkdir()
            (checkout / "run_talknet.py").write_text("# stub", encoding="utf-8")
            context.config.activespeaker.talknet_root = checkout
        absent = tmp_path / "environments" / section
        setattr(getattr(context.config, section), "uv_project", absent)
        enabled, reason = stage_named(stage).enabled(context)
        assert enabled is False
        assert str(absent) in reason
        assert "uv sync" in reason, "the skip reason must carry the remedy"

    def test_a_missing_openpose_install_skips_instead_of_failing_the_video(
            self, context, tmp_path: Path) -> None:
        """OpenPose is a third-party binary this repository never fetches.

        Before this gate ``prepare()`` raised ``StageError`` straight from discovery, so a
        default run on a machine without OpenPose reported the video failed while everything
        else had actually worked. ``inspect-environment`` still announces the same finding
        before the batch starts.
        """
        context.config.openpose.root = tmp_path / "definitely-not-openpose"
        enabled, reason = stage_named("openpose").enabled(context)
        assert enabled is False
        assert "definitely-not-openpose" in reason

    def test_the_skip_is_what_a_run_records(self, context, tmp_path: Path) -> None:
        """Not just ``enabled()``: the body never runs and nothing is written.

        Checked through ``run()`` because that is the call the orchestrator makes. A stage
        that answered ``(False, reason)`` and still executed would pass a test of
        ``enabled()`` alone.
        """
        context.config.whisperx.uv_project = tmp_path / "environments" / "whisperx"
        outcome = stage_named("whisperx").run(context)
        assert outcome.status == "skipped"
        assert "uv sync" in (outcome.message or "")
        assert not context.artifact("whisperx_raw").exists()


class TestMissingTranscriptSkips:
    """No transcript is a skip for every reader of it, not a failure.

    Reproduced from a fresh project with the WhisperX environment missing: whisperx
    skipped correctly, and then ``translation`` and ``spacy_source`` failed on the absent
    ``speech/segments.parquet``. A failed stage poisons only its dependants, so
    ``finalization`` still ran, ``validate`` reported the two failures and the whole batch
    exited 2 — over an install state the pipeline had already reported as a skip.
    """

    #: Stages that read the transcript tables and had no gate for their absence.
    CONSUMERS = ("translation", "spacy_source", "acoustic", "stories", "speaker_assignment")

    @staticmethod
    def _whisperx_skipped(context) -> None:
        context.state.mark_skipped("whisperx", "whisperx environment not installed at /nowhere")

    @pytest.mark.parametrize("stage", CONSUMERS)
    def test_a_consumer_skips_when_the_transcript_is_absent(self, context, stage: str) -> None:
        # `stories` shares translation's endpoint rule and the shared fixture configures only
        # translation: a mock endpoint keeps the transcript the reason under test.
        context.config.stories.provider = "mock"
        # A missing environment is reported before a missing input (install first), so the
        # stages that have one need it present for the transcript to be the gate under test.
        section = {"spacy_source": "spacy", "acoustic": "acoustic"}.get(stage)
        if section:
            cfg = getattr(context.config, section)
            project = context.config.resolve(cfg.uv_project)
            project.mkdir(parents=True, exist_ok=True)
            worker = context.config.resolve(cfg.worker)
            worker.parent.mkdir(parents=True, exist_ok=True)
            worker.write_text("# stub", encoding="utf-8")
        self._whisperx_skipped(context)
        assert not context.artifact("speech_segments").is_file()
        enabled, reason = stage_named(stage).enabled(context)
        assert enabled is False, f"{stage} ran with no transcript to read"
        assert reason.startswith("upstream skipped: transcript unavailable"), reason
        # The reason carries the upstream cause, so a skip is diagnosable from the record.
        assert "whisperx" in reason

    def test_english_spacy_skips_because_its_input_never_arrived(self, context) -> None:
        """`spacy_english` reads the translation table, so the transcript gate is upstream."""
        self._whisperx_skipped(context)
        enabled, reason = stage_named("spacy_english").enabled(context)
        assert enabled is False
        assert "English translation" in reason

    def test_a_present_transcript_keeps_the_stage_enabled(self, context, write_parquet) -> None:
        """The gate is about a missing table, never about the flag that would produce it."""
        from multimodal_pipeline.schemas import SEGMENTS_SCHEMA, WORDS_SCHEMA

        self._whisperx_skipped(context)
        write_parquet(context.artifact("speech_segments"), [], SEGMENTS_SCHEMA)
        write_parquet(context.artifact("speech_words"), [], WORDS_SCHEMA)
        enabled, reason = stage_named("translation").enabled(context)
        assert enabled is True, reason


class TestPresentButBrokenStillFails:
    """The other half of the rule: a synced environment that breaks is a defect."""

    def test_a_failing_worker_is_reported_as_a_failure_not_a_skip(
            self, context, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The guard reads the filesystem; it never runs the tool to find out.

        Both files exist here and every input is present, so ``enabled()`` says yes and the
        worker's own error surfaces as a stage failure — the repo's rule that a present but
        broken install must never be silenced into a skip.
        """
        environment = tmp_path / "environments" / "whisperx"
        environment.mkdir(parents=True)
        context.config.whisperx.uv_project = environment
        context.config.whisperx.worker = ROOT / "workers" / "whisperx_worker.py"
        context.artifact("audio").parent.mkdir(parents=True, exist_ok=True)
        context.artifact("audio").write_bytes(b"RIFF....WAVE")

        def explode(**_kwargs):
            raise WorkerError("worker reported status=error: CUDA out of memory")

        monkeypatch.setattr("multimodal_pipeline.uv_worker.run_worker", explode)
        stage = stage_named("whisperx")
        assert stage.enabled(context) == (True, "")
        with pytest.raises(StageError, match="CUDA out of memory"):
            stage.run(context)


def test_no_source_document_still_says_stages_ship_disabled() -> None:
    """A stale "off by default" sentence is how a default regresses later.

    Cheap, and it names file and line: the alternative is an operator reading a docstring
    that contradicts what a run does.
    """
    offenders = []
    sources = [*sorted((ROOT / "src").rglob("*.py")),
               ROOT / "config" / "config.example.yaml",
               ROOT / "AGENTS.md"]
    for path in sources:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            lowered = line.lower()
            if "off by default" not in lowered and "ships disabled" not in lowered:
                continue
            # The preview switches and the render paths are the deliberate exception.
            if any(word in lowered for word in ("write_video", "write_images", "render",
                                                "image_max_side", "skeleton", "raw_images")):
                continue
            offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    assert offenders == [], offenders


class TestPresentButUnusableFails:
    """Three states, three answers: absent skips, broken fails, usable runs.

    A path that *exists* is not an absent prerequisite. ``openpose.executable`` pointing at
    a file without the execute bit, or at a directory where a binary should be, is a broken
    install: skipping it reported a working-looking batch whose pose tables silently were not
    produced, and nothing in the run said why.
    """

    def test_an_executable_without_the_execute_bit_fails_loudly(self, context, tmp_path):
        candidate = tmp_path / "openpose.bin"
        candidate.write_bytes(b"not a binary")
        candidate.chmod(0o644)
        context.config.openpose.executable = str(candidate)
        with pytest.raises(StageError, match="not executable"):
            stage_named("openpose").enabled(context)

    def test_a_configured_executable_path_that_is_a_directory_fails_loudly(
            self, context, tmp_path):
        candidate = tmp_path / "openpose.bin"
        candidate.mkdir()
        context.config.openpose.executable = str(candidate)
        with pytest.raises(StageError, match="not a file"):
            stage_named("openpose").enabled(context)

    def test_a_missing_executable_still_skips(self, context, tmp_path):
        context.config.openpose.executable = str(tmp_path / "definitely-absent.bin")
        enabled, reason = stage_named("openpose").enabled(context)
        assert enabled is False and "definitely-absent.bin" in reason

    def test_a_model_folder_that_is_a_file_fails_loudly(self, context, tmp_path):
        executable = tmp_path / "openpose.bin"
        executable.write_bytes(b"stub")
        executable.chmod(0o755)
        model_file = tmp_path / "models"
        model_file.write_bytes(b"not a directory")
        context.config.openpose.executable = str(executable)
        context.config.openpose.model_folder = str(model_file)
        with pytest.raises(StageError, match="not a directory"):
            stage_named("openpose").enabled(context)

    def test_an_auto_discovered_binary_without_the_execute_bit_fails_loudly(
            self, context, tmp_path, monkeypatch):
        """Discovery answers "is it there"; the gate has to ask "can it run"."""
        root = tmp_path / "openpose"
        binary = root / "build/examples/openpose/openpose.bin"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"not a binary")
        binary.chmod(0o644)
        models = root / "models"
        models.mkdir()
        context.config.openpose.root = str(root)
        context.config.openpose.executable = "auto"
        context.config.openpose.model_folder = "auto"
        with pytest.raises(StageError, match="not executable"):
            stage_named("openpose").enabled(context)
