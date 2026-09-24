"""``inspect-environment`` must warn about the degradations that are silent today.

The command's own docstring and the README promise it names "things that will make a
stage skip" before a long run. What it actually reported (measured against the shipped
local config, which produced zero warnings) was HF_TOKEN, the translation endpoint, a
missing input directory, the OpenPose binary and absent uv-project directories.

Two real degradations were invisible:

* ``activespeaker.enabled: true`` with no ``talknet_root`` — the stage skips at runtime
  with a precise reason, but nothing said so up front, so an operator only learned it
  from a status table after a batch.
* a configured spaCy model that is not installed — the resolver falls through to a
  family cousin or to ``blank``, and ``blank`` means tokenization plus a sentencizer
  only: the tables come back with empty lemmas/POS/dep and look like results. That cost
  this project a wrong-language annotation incident (see
  ``odd/tasks/multimodal-video-pipeline.md`` §16) and it produces *no warning at all*.

A uv project that exists but has never been synced is deliberately **not** warned
about: verified against a throwaway project on this machine, ``uv run --project``
creates and syncs the environment itself on first use, so "present but unsynced" is
not a problem the operator has to fix. Only a missing project directory is.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from multimodal_pipeline.cli import _environment_warnings
from multimodal_pipeline.config import PipelineConfig


@pytest.fixture
def base_config(tmp_path: Path) -> PipelineConfig:
    """A config that warns about nothing.

    Every uv project directory is created, because the current warning loop reports a
    missing project for *disabled* stages too; tests that assert on uv warnings delete
    the one they care about. That pre-existing noise is exactly why each test here
    filters by needle instead of comparing whole lists.
    """
    (tmp_path / "videos").mkdir()
    for name in ("whisperx", "diarization", "spacy", "acoustic", "activespeaker"):
        (tmp_path / "environments" / name).mkdir(parents=True)
    return PipelineConfig.model_validate(
        {
            "project_root": str(tmp_path),
            "input": {"directory": str(tmp_path / "videos")},
            "output": {"directory": str(tmp_path / "out")},
            "diarization": {"enabled": False},
            "translation": {"enabled": False},
            "openpose": {"enabled": False},
            "activespeaker": {"enabled": False},
            "spacy": {"enabled": False},
            "whisperx": {"enabled": False},
            "acoustic": {"enabled": False},
        }
    )


def warnings_containing(warnings: list[str], needle: str) -> list[str]:
    return [w for w in warnings if needle in w]


class TestActivespeakerGateIsAnnounced:
    def test_missing_talknet_root_is_warned_when_enabled(self, base_config: PipelineConfig) -> None:
        config = base_config.model_copy(deep=True)
        object.__setattr__(config.activespeaker, "enabled", True)
        object.__setattr__(config.activespeaker, "talknet_root", None)
        warnings = _environment_warnings(config)
        hits = warnings_containing(warnings, "activespeaker.talknet_root")
        assert hits, f"expected a talknet_root warning, got {warnings}"
        assert "not set" in hits[0]

    def test_nonexistent_talknet_root_is_warned(self, base_config: PipelineConfig, tmp_path: Path) -> None:
        config = base_config.model_copy(deep=True)
        object.__setattr__(config.activespeaker, "enabled", True)
        object.__setattr__(config.activespeaker, "talknet_root", tmp_path / "nope")
        hits = warnings_containing(_environment_warnings(config), "activespeaker.talknet_root")
        assert hits and "does not exist" in hits[0]

    def test_root_without_the_entrypoint_is_warned(self, base_config: PipelineConfig, tmp_path: Path) -> None:
        root = tmp_path / "not-talknet"
        root.mkdir()
        config = base_config.model_copy(deep=True)
        object.__setattr__(config.activespeaker, "enabled", True)
        object.__setattr__(config.activespeaker, "talknet_root", root)
        hits = warnings_containing(_environment_warnings(config), "run_talknet.py")
        assert hits, f"expected the missing-entrypoint warning, got {_environment_warnings(config)}"

    def test_a_valid_checkout_produces_no_warning(self, base_config: PipelineConfig, tmp_path: Path) -> None:
        root = tmp_path / "TalkNet-ASD"
        root.mkdir()
        (root / "run_talknet.py").write_text("# stub\n", encoding="utf-8")
        config = base_config.model_copy(deep=True)
        object.__setattr__(config.activespeaker, "enabled", True)
        object.__setattr__(config.activespeaker, "talknet_root", root)
        assert not warnings_containing(_environment_warnings(config), "activespeaker")

    def test_disabled_activespeaker_is_not_warned_about(self, base_config: PipelineConfig) -> None:
        # An explicitly disabled stage is a decision, not a defect.
        assert not warnings_containing(_environment_warnings(base_config), "activespeaker")


class TestSpacyDegradationIsAnnounced:
    """Only the English model is warnable before a run — see ``_spacy_warnings``.

    Its language is always ``en`` and it runs exactly when translation produces segments,
    so a missing ``english_model`` predicts a real empty-lemmas outcome. Source models are
    not inventoried: the configured map covers eight languages the corpus may never
    contain, and the language is only known after transcription.
    """

    def configured(self, base_config: PipelineConfig) -> PipelineConfig:
        config = base_config.model_copy(deep=True)
        object.__setattr__(config.spacy, "enabled", True)
        object.__setattr__(config.spacy, "source_models", {"es": "es_core_news_lg"})
        object.__setattr__(config.spacy, "english_model", "en_core_web_lg")
        # English linguistics only run when translation is configured; make that true so
        # these tests exercise the model check rather than the translation gate.
        translation = config.translation.model_copy(
            update={
                "enabled": True,
                "base_url": "http://127.0.0.1:1/v1",
                "model": "test-chat",
                "api_key": "k",
            }
        )
        object.__setattr__(config, "translation", translation)
        assert config.translation.endpoint_configured
        return config

    def install(self, tmp_path: Path, *names: str) -> None:
        site = tmp_path / "environments" / "spacy" / ".venv" / "lib" / "python3.12" / "site-packages"
        for name in names:
            model_dir = site / name
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "meta.json").write_text(
                json.dumps({"name": name, "version": "3.8.0", "spacy_version": ">=3.8.0"}),
                encoding="utf-8",
            )

    def test_missing_english_model_is_warned(self, base_config: PipelineConfig, tmp_path: Path) -> None:
        config = self.configured(base_config)
        hits = warnings_containing(_environment_warnings(config), "spaCy model")
        assert hits, f"expected an english_model warning, got {_environment_warnings(config)}"
        assert "en_core_web_lg" in hits[0]
        # It must say what the fallback costs, not only that one happens.
        assert "blank" in hits[0] and "lemma" in hits[0].lower()

    def test_installed_english_model_produces_no_warning(self, base_config: PipelineConfig, tmp_path: Path) -> None:
        config = self.configured(base_config)
        self.install(tmp_path, "en_core_web_lg")
        assert not warnings_containing(_environment_warnings(config), "spaCy model")

    def test_no_warning_when_translation_is_not_configured(
        self, base_config: PipelineConfig, tmp_path: Path
    ) -> None:
        # No English text will ever exist, so warning about the English model is noise.
        config = base_config.model_copy(deep=True)
        object.__setattr__(config.spacy, "enabled", True)
        object.__setattr__(config.spacy, "english_model", "en_core_web_lg")
        assert not config.translation.endpoint_configured
        assert not warnings_containing(_environment_warnings(config), "spaCy model")

    def test_no_warning_when_english_processing_is_off(
        self, base_config: PipelineConfig, tmp_path: Path
    ) -> None:
        config = self.configured(base_config)
        object.__setattr__(config.spacy, "process_english", False)
        assert not warnings_containing(_environment_warnings(config), "spaCy model")

    def test_source_languages_are_not_speculatively_inventory_warned(
        self, base_config: PipelineConfig, tmp_path: Path
    ) -> None:
        """The defect this test was written for.

        The first implementation warned for every configured source model that was not
        installed. Against this machine's real config that emitted five warnings about
        German, French, Italian, Dutch and Portuguese for a corpus of English and Spanish
        clips — noise that buries the warnings that matter.
        """
        config = self.configured(base_config)
        object.__setattr__(
            config.spacy, "source_models", {"de": "de_dep_news_trf", "fr": "fr_dep_news_trf"}
        )
        self.install(tmp_path, "en_core_web_lg")
        names = " ".join(_environment_warnings(config))
        assert "de_dep_news_trf" not in names
        assert "fr_dep_news_trf" not in names

    def test_a_same_family_substitute_is_not_warned_about(
        self, base_config: PipelineConfig, tmp_path: Path
    ) -> None:
        config = self.configured(base_config)
        object.__setattr__(config.spacy, "english_model", "en_core_web_trf")
        self.install(tmp_path, "en_core_web_lg")
        assert not warnings_containing(_environment_warnings(config), "spaCy model")


class TestUvProjectsWarnOnlyWhenAbsent:
    def test_absent_uv_project_is_warned(self, base_config: PipelineConfig, tmp_path: Path) -> None:
        config = base_config.model_copy(deep=True)
        object.__setattr__(config.whisperx, "enabled", True)
        shutil.rmtree(tmp_path / "environments" / "whisperx")
        hits = warnings_containing(_environment_warnings(config), "uv project missing")
        assert hits, f"expected the missing-project warning, got {_environment_warnings(config)}"
        assert "environments/whisperx" in hits[0]

    def test_present_but_unsynced_project_is_not_warned(
        self, base_config: PipelineConfig, tmp_path: Path
    ) -> None:
        """uv syncs on first `uv run --project`, verified on this machine."""
        project = tmp_path / "environments" / "whisperx"
        (project / "pyproject.toml").write_text(
            '[project]\nname="w"\nversion="0"\nrequires-python=">=3.10"\ndependencies=[]\n',
            encoding="utf-8",
        )
        config = base_config.model_copy(deep=True)
        object.__setattr__(config.whisperx, "enabled", True)
        assert not warnings_containing(_environment_warnings(config), "uv project missing")

    def test_disabled_stage_with_no_project_is_not_warned(
        self, base_config: PipelineConfig, tmp_path: Path
    ) -> None:
        """A stage the operator switched off does not need its environment.

        The warning loop today walks every stage config regardless of `enabled`, so a
        config with diarization disabled still reports its missing uv project. That is
        noise that buries the warnings which matter.
        """
        shutil.rmtree(tmp_path / "environments" / "diarization")
        config = base_config.model_copy(deep=True)  # diarization already disabled
        assert not warnings_containing(_environment_warnings(config), "diarization")


class TestWarningOrder:
    def test_credential_warning_stays_first(self, base_config: PipelineConfig) -> None:
        """tests/e2e/test_cli_smoke.py asserts environment_warnings[0] names HF_TOKEN."""
        config = base_config.model_copy(deep=True)
        object.__setattr__(config.diarization, "enabled", True)
        object.__setattr__(config.activespeaker, "enabled", True)
        object.__setattr__(config.spacy, "enabled", True)
        warnings = _environment_warnings(config)
        assert "HF_TOKEN" in warnings[0], warnings


class TestReportStaysMachineReadable:
    def test_every_warning_is_a_plain_string(self, base_config: PipelineConfig) -> None:
        config = base_config.model_copy(deep=True)
        object.__setattr__(config.activespeaker, "enabled", True)
        object.__setattr__(config.spacy, "enabled", True)
        warnings: list[Any] = _environment_warnings(config)
        assert warnings
        assert all(isinstance(w, str) and w for w in warnings)
        # The payload is printed with json.dumps; nothing here may break that.
        json.dumps({"environment_warnings": warnings})


class TestSourceModelDecisionIsRecorded:
    """Where source-language degradation *is* visible: at the moment it happens.

    The pre-flight cannot know the language, so the contract is that the run records the
    decision. These lock the two places a reader can find it.
    """

    def test_select_model_records_the_fallback_and_its_cost(self) -> None:
        from multimodal_pipeline.stages.spacy_source import select_model

        decision = select_model("es", {"es": "es_core_news_lg"}, {"en_core_web_lg"})
        assert decision["status"] == "fallback_missing_model"
        assert decision["model"] == "blank"
        assert decision["capabilities"] == "tokenization,sentencizer"

    def test_the_stage_logs_the_model_it_chose(self) -> None:
        import inspect

        from multimodal_pipeline.stages.spacy_source import SpacySourceStage

        source = inspect.getsource(SpacySourceStage)
        assert "selected_model" in source, "the stage stopped surfacing the chosen model"
        assert "ctx.log" in source, "the stage stopped logging the model per video"
