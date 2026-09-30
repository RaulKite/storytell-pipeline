"""The stage around the builder: its path, its skips, its reuse, its registration (T22).

`test_elan.py` proves the mapping from tables to a `.eaf` is right. This file proves the
things only the pipeline can get wrong: the file lands at its registered artifact path, a
dataset with no transcript is refused rather than exported empty, the stage record's counts
match the bytes, reuse notices an edit to the python that builds the tiers, and the
registration that makes the stage run at all — class map, dependency order, the manifest
deadlock, log names — is present.

The synthetic clip is imported from `test_elan.py` rather than rebuilt: `make_dataset` is
five tables whose shapes each fire one export rule, and a second copy would be a second
fixture to keep true. `test_persons_config.py` imports its sibling's builders for the same
reason.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest
from multimodal_pipeline import elan as elan_core
from multimodal_pipeline.artifacts import (
    ARTIFACT_LAYOUT,
    MANIFEST_ARTIFACTS,
    VideoPaths,
)
from multimodal_pipeline.config import PipelineConfig
from multimodal_pipeline.exceptions import ValidationError
from multimodal_pipeline.schemas import SEGMENTS_SCHEMA, WORDS_SCHEMA
from multimodal_pipeline.stages.elan import ElanStage
from tests.unit.test_elan import _word, _write, make_dataset, real_tier_names

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def dataset(tmp_path: Path) -> dict[str, Path]:
    """The sibling module's synthetic clip, as a fixture of this file."""
    return make_dataset(tmp_path)

# ------------------------------------------------------------------ the stage itself


class FakeSource:
    def __init__(self, path: Path, video_id: str = "clip") -> None:
        self.path = path
        self.video_id = video_id


class FakeLog:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, message: Any, level: int = logging.INFO) -> None:
        self.lines.append(str(message))


def stage_context(config: PipelineConfig, dataset_dir: Path, video: Path):
    from multimodal_pipeline.artifacts import ArtifactRegistry
    from multimodal_pipeline.state import VideoState
    from multimodal_pipeline.stages.base import STAGE_ORDER, StageContext

    paths = VideoPaths(dataset_dir)
    paths.ensure_dirs()
    state = VideoState.load(paths, "clip", str(video))
    state.bind_stages(STAGE_ORDER)
    return StageContext(config=config, source=FakeSource(video), paths=paths, state=state,
                        registry=ArtifactRegistry(paths).refresh(), log=FakeLog(),
                        tools={"schema_version": "1.0"})


@pytest.fixture
def stage_config(tmp_path: Path) -> PipelineConfig:
    (tmp_path / "videos").mkdir()
    return PipelineConfig.model_validate({
        "project_root": str(tmp_path),
        "input": {"directory": str(tmp_path / "videos")},
        "output": {"directory": str(tmp_path / "out")},
    })


class TestStageWiring:
    def test_execute_writes_the_file_at_its_registered_path(self, stage_config, dataset):
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        extras = stage.run(ctx)
        assert extras.status == "completed", extras.message
        assert ctx.artifact("elan_annotations").is_file()
        assert ctx.artifact("elan_annotations") == (dataset["dir"]
                                                    / "elan" / "annotations.eaf")

    def test_the_recorded_counts_match_the_file(self, stage_config, dataset):
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        summary = stage.run(ctx).detail["provenance"]["extra"]
        eaf_text = ctx.artifact("elan_annotations").read_text(encoding="utf-8")
        root = ET.fromstring(eaf_text)
        assert summary["tiers"] == len(real_tier_names(root))
        assert summary["tier_counts"] == {"words": 3, "turns_pyannote": 1, "asd_speaking": 3,
                                          "face_tracks": 1, "pose_presence": 2}
        assert summary["annotations"] == 10
        assert summary["skipped_tiers"] == ["segments_src", "gloss_en", "turns_nemotron",
                                            "fusion_pyannote", "fusion_nemotron",
                                            "person_tracks", "voiced_blocks"]
        assert summary["media_url"] == dataset["video"].resolve().as_uri()
        assert summary["mimetype"] == "video/mp4"
        assert summary["bytes"] == ctx.artifact("elan_annotations").stat().st_size

    def test_no_temporary_file_is_left_behind(self, stage_config, dataset):
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        strays = [p.name for p in ctx.artifact("elan_annotations").parent.iterdir()
                  if p.name != "annotations.eaf"]
        assert not strays, f"the atomic write left {strays} behind"

    def test_a_dataset_with_no_transcript_is_refused(self, stage_config, dataset):
        """Both transcript tables absent is a different case from a partial export.

        The .eaf would open to an empty grid and read as a broken export, so this is the
        ValidationError style the other derived stages use, not a skip.
        """
        (dataset["dir"] / "speech" / "words.parquet").unlink()
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        assert ctx.artifact("speech_segments").is_file() is False
        enabled, reason = stage.enabled(ctx)
        assert not enabled and "whisperx" in reason
        with pytest.raises(ValidationError) as raised:
            stage.execute(ctx)
        assert "no transcript" in str(raised.value)

    def test_one_transcript_table_alone_is_enough(self, stage_config, dataset):
        """Words-only and segments-only datasets are both normal, so neither is refused."""
        (dataset["dir"] / "speech" / "words.parquet").unlink()
        _write(SEGMENTS_SCHEMA, dataset["dir"] / "speech" / "segments.parquet", [{
            "schema_version": "1.0", "video_id": "clip", "segment_id": "seg-0",
            "start_time": 0.0, "end_time": 1.0, "duration": 1.0, "language": "en",
            "speaker_id": "SPEAKER_00", "text": "hello there", "confidence": 0.9,
        }])
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        enabled, reason = stage.enabled(ctx)
        assert enabled, reason
        summary = stage.run(ctx).detail["provenance"]["extra"]
        # `words` is absent from the census entirely rather than reported as 0: a tier whose
        # file was never produced is skipped, which is the difference between "this dataset has
        # no words" and "the words table exists and is empty".
        assert summary["tier_counts"] == {"segments_src": 1, "turns_pyannote": 1,
                                         "asd_speaking": 3, "face_tracks": 1,
                                         "pose_presence": 2}
        assert "words" in summary["skipped_tiers"]

    def test_disabled_by_config(self, stage_config, dataset):
        stage_config.elan.enabled = False
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        outcome = stage.run(ctx)
        assert outcome.status == "skipped"
        assert outcome.message == "elan.enabled = false"
        assert not ctx.artifact("elan_annotations").exists()

    def test_a_missing_source_video_is_refused(self, stage_config, dataset):
        """The media descriptor cannot be recovered later, so this cannot be a skip."""
        dataset["video"].unlink()
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        with pytest.raises(ValidationError, match="source video not found"):
            stage.execute(ctx)

    def test_validate_accepts_what_execute_wrote(self, stage_config, dataset):
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        summary = stage.validate(ctx)
        assert summary["tiers"] == 5 and summary["media_descriptors"] == 1

    def test_validate_reports_a_missing_file(self, stage_config, dataset):
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        with pytest.raises(ValidationError, match="ELAN export missing"):
            stage.validate(ctx)

    def test_validate_reports_a_truncated_file(self, stage_config, dataset):
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        path = ctx.artifact("elan_annotations")
        path.write_text(path.read_text(encoding="utf-8")[:400], encoding="utf-8")
        with pytest.raises(ValidationError, match="not well-formed XML"):
            stage.validate(ctx)

    def test_validate_reports_media_that_is_not_reachable(self, stage_config, dataset):
        """The .eaf is the file a user opens in ELAN; a dead link is the failure they see."""
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        dataset["video"].unlink()
        with pytest.raises(ValidationError, match="not reachable from the dataset directory"):
            stage.validate(ctx)

    def test_validate_reports_a_census_that_lost_a_tier(self, stage_config, dataset):
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        path = ctx.artifact("elan_annotations")
        text = path.read_text(encoding="utf-8")
        # Drop one tier's element wholesale, leaving parseable XML behind — the shape a
        # hand edit or a partial truncation produces.
        start = text.index('<TIER TIER_ID="asd_speaking"')
        end = text.index("</TIER>", start) + len("</TIER>")
        path.write_text(text[:start] + text[end:], encoding="utf-8")
        with pytest.raises(ValidationError, match="census are absent"):
            stage.validate(ctx)

    def test_rerunning_replaces_the_file_rather_than_appending(self, stage_config, dataset):
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        first = ctx.artifact("elan_annotations").read_text(encoding="utf-8")
        stage.run(ctx)
        second = ctx.artifact("elan_annotations").read_text(encoding="utf-8")
        assert len(real_tier_names(ET.fromstring(second))) == 5
        # Only the writer's timestamp differs; the annotations must not grow.
        assert first.count("<ANNOTATION ") == second.count("<ANNOTATION ")

    def test_the_fingerprint_notices_a_changed_input_table(self, stage_config, dataset):
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        before = stage.config_fingerprint(ctx)
        assert before["words_digest"] is not None
        assert before["person_tracks_digest"] is None, "absent inputs hash as None, not absent"
        ctx.scratch.clear()
        _write(WORDS_SCHEMA, dataset["dir"] / "speech" / "words.parquet",
               [_word("changed", 0.0, 0.4)])
        after = stage.config_fingerprint(ctx)
        assert after["words_digest"] != before["words_digest"]
        assert after["tiers"] == before["tiers"]

    def test_the_fingerprint_carries_the_python_source_digest(self, stage_config, dataset):
        """The hole §31 closed for pose_normalized, which has no worker either.

        Without it a change to the millisecond rounding in ``elan.py`` leaves config,
        thresholds and every input digest identical, and the fix does nothing until someone
        deletes the .eaf by hand.
        """
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        digest = stage.config_fingerprint(ctx)["_python_code_sha256"]
        assert digest and len(digest) == 64
        from multimodal_pipeline.stages.base import python_source_digest
        from multimodal_pipeline.stages import elan as stage_module

        assert digest == python_source_digest(elan_core, stage_module)

    def test_declared_inputs_are_registered_artifacts(self) -> None:
        stage = ElanStage()
        assert set(stage.inputs) <= set(ARTIFACT_LAYOUT)
        assert set(stage.outputs) <= set(ARTIFACT_LAYOUT)
        assert stage.outputs == ("elan_annotations",)

    def test_outputs_present_only_asks_for_the_eaf(self, stage_config, dataset):
        """Declared inputs are read softly; they must not gate reuse."""
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        assert not stage.outputs_present(ctx)
        stage.run(ctx)
        assert stage.outputs_present(ctx)


# ------------------------------------------------------------------ wiring elsewhere


class TestConfigAndRegistry:
    def test_elan_is_enabled_by_default(self) -> None:
        """The default is a decision: no GPU, no download, seconds per video. See ElanConfig."""
        from multimodal_pipeline.config import ElanConfig

        assert ElanConfig().enabled is True

    def test_the_section_reaches_the_stage_configs_map(self, stage_config) -> None:
        assert stage_config.stage_configs.get("elan") is stage_config.elan

    def test_the_section_is_in_the_configured_field_list(self) -> None:
        assert "elan" in PipelineConfig.model_fields

    def test_an_unknown_elan_key_is_still_an_error(self, tmp_path) -> None:
        """The section is in the same `_Model` family, so config drift still fails loudly."""
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            PipelineConfig.model_validate({
                "project_root": str(tmp_path),
                "input": {"directory": str(tmp_path)},
                "output": {"directory": str(tmp_path)},
                "elan": {"enabledl": True},
            })

    def test_the_export_is_a_manifest_artifact(self) -> None:
        assert "elan_annotations" in MANIFEST_ARTIFACTS
        assert len(MANIFEST_ARTIFACTS) == 44

    def test_the_registered_path_lives_in_its_own_directory(self) -> None:
        assert ARTIFACT_LAYOUT["elan_annotations"] == "elan/annotations.eaf"

    def test_ensure_dirs_creates_the_elan_directory(self, tmp_path) -> None:
        """The promise every stage relies on: your output directory already exists."""
        VideoPaths(tmp_path / "dataset").ensure_dirs()
        assert (tmp_path / "dataset" / "elan").is_dir()

    def test_the_example_config_ships_the_section(self) -> None:
        import yaml

        text = (ROOT / "config" / "config.example.yaml").read_text(encoding="utf-8")
        payload = yaml.safe_load(text)
        assert payload["elan"] == {"enabled": True}

    def test_the_library_is_a_root_dependency_and_not_a_new_environment(self) -> None:
        """The orchestrator builds the .eaf itself, so there is no uv project for it.

        Asserted rather than trusted because the *absence* of an `environments/*` project is
        the design: a reviewer cannot tell a deliberate pure-python stage from a forgotten
        environment by reading the diff.
        """
        import tomllib

        payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        deps = " ".join(payload["project"]["dependencies"])
        assert "pympi-ling" in deps
        assert not (ROOT / "environments" / "elan").exists()

    def test_the_dependency_resolves_and_imports(self) -> None:
        """The pyproject edit is what persists; this says the pin is installable."""
        from importlib import metadata

        import pympi.Elan

        assert metadata.version("pympi-ling")
        assert pympi.Elan.Eaf is not None


class TestStageOrder:
    def test_elan_is_the_last_stage(self) -> None:
        from multimodal_pipeline.stages.base import STAGE_ORDER

        assert STAGE_ORDER[-1] == "elan"
        assert STAGE_ORDER.index("finalization") == len(STAGE_ORDER) - 2

    def test_it_depends_on_finalization_alone(self) -> None:
        from multimodal_pipeline.stages.base import STAGE_DEPENDENCIES

        assert STAGE_DEPENDENCIES["elan"] == ("finalization",)

    def test_finalization_does_not_depend_on_elan(self) -> None:
        """The deadlock guard.

        `finalization` writes the manifest that has to declare `elan`, so making it depend on
        `elan` would need `elan` to have already run — and `elan` depends on `finalization`.
        Asserted on both sides so the exclusion cannot be undone by a tidy-up of the tuple.
        """
        from multimodal_pipeline.stages.base import (
            STAGE_DEPENDENCIES,
            STAGE_ORDER,
            dependency_chain,
        )

        assert "elan" not in STAGE_DEPENDENCIES["finalization"]
        assert "finalization" not in STAGE_DEPENDENCIES["finalization"]
        assert len(STAGE_DEPENDENCIES["finalization"]) == len(STAGE_ORDER) - 2
        assert dependency_chain("finalization") == list(STAGE_ORDER[:-2])
        # And elan's transitive chain is everything, in canonical order.
        assert dependency_chain("elan") == list(STAGE_ORDER[:-1])

    def test_the_stage_has_its_own_log_name(self) -> None:
        from multimodal_pipeline.artifacts import STAGE_LOG_NAMES
        from multimodal_pipeline.stages.base import STAGE_ORDER

        assert "elan" in STAGE_LOG_NAMES
        assert set(STAGE_LOG_NAMES) == set(STAGE_ORDER)

    def test_dependants_of_finalization_is_only_elan(self) -> None:
        from multimodal_pipeline.stages.base import dependants_of

        assert dependants_of("finalization") == ["elan"]

class TestOrchestratorRegistration:
    """The wiring this stage needs in `orchestrator.py`, asserted from here.

    `orchestrator.py` is outside this change's edit surface, so these three tests are the
    specification the parent's one-line registration has to satisfy — not a description of what
    is on disk today. Two of them (`STAGE_CLASSES`, `enabled_stage_names`) fail until `elan` is
    added there, and `build_stages()` raises `KeyError: 'elan'` for every CLI command until the
    first lands. They are kept rather than dropped because a stage that is in `STAGE_ORDER` but
    absent from the class map is a pipeline that cannot start, and that has to be a red test
    rather than a paragraph in a handoff.
    """

    def test_elan_is_in_the_stage_class_map(self) -> None:
        """Without this `build_stages()` raises KeyError and every CLI command dies."""
        from multimodal_pipeline.orchestrator import STAGE_CLASSES
        from multimodal_pipeline.stages.base import STAGE_ORDER

        assert set(STAGE_CLASSES) == set(STAGE_ORDER)
        assert STAGE_CLASSES["elan"] is ElanStage

    def test_the_enabled_stage_names_still_cover_the_whole_order(self, stage_config) -> None:
        """The status table's column layout is built on this equality.

        `elan` defaults to true, so it belongs in `enabled_stage_names`; a default-true stage
        missing from the map keeps a column the header numbers but never fills.
        """
        from multimodal_pipeline.orchestrator import enabled_stage_names
        from multimodal_pipeline.stages.base import STAGE_ORDER

        assert len(enabled_stage_names(stage_config)) == len(STAGE_ORDER)

    def test_disabling_it_is_reported_by_the_name_map(self, tmp_path) -> None:
        """The invariant `test_persons_config` asserts for `persons`, inverted for a
        default-true stage: the map must report the flag, or `status` calls a stage it will
        skip enabled."""
        from multimodal_pipeline.orchestrator import enabled_stage_names
        from multimodal_pipeline.stages.base import STAGE_ORDER

        config = PipelineConfig.model_validate({
            "project_root": str(tmp_path),
            "input": {"directory": str(tmp_path)},
            "output": {"directory": str(tmp_path)},
            "elan": {"enabled": False},
        })
        names = enabled_stage_names(config)
        assert "elan" not in names
        assert len(names) == len(STAGE_ORDER) - 1
