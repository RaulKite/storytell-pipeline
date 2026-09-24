"""spaCy worker helpers: token/timing alignment, model selection, variants.

The worker lives outside the package (it runs inside its own uv project), so it
is imported by path here. These are the parts that are pure logic and therefore
testable without spaCy, torch or a GPU.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from multimodal_pipeline.stages.spacy_source import SpacySourceStage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER = PROJECT_ROOT / "workers" / "spacy_worker.py"


def load_worker():
    spec = importlib.util.spec_from_file_location("spacy_worker_under_test", WORKER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = pytest.fixture(scope="module")(lambda: load_worker())


def word(text: str, start: float | None, end: float | None, **extra: Any) -> dict[str, Any]:
    return {"word": text, "start_time": start, "end_time": end,
            "alignment_status": extra.pop("alignment_status", "aligned"), **extra}


class TestCoreText:
    def test_punctuation_is_stripped_from_the_edges(self, worker) -> None:
        assert worker.core_text("world,") == "world"
        assert worker.core_text('"hello"') == "hello"
        assert worker.core_text("—wait—") == "wait"

    def test_case_folds(self, worker) -> None:
        assert worker.core_text("Hello") == worker.core_text("hello")

    def test_inner_apostrophes_survive(self, worker) -> None:
        # "don't" must not become "don": the apostrophe is part of the word.
        assert worker.core_text("don't") == "don't"
        assert worker.core_text("L'homme") == "l'homme"

    def test_pure_punctuation_has_no_core(self, worker) -> None:
        for token in [",", ".", "!", "…", "—", '"', "  ", ""]:
            assert worker.core_text(token) == ""

    def test_unicode_forms_compare_equal(self, worker) -> None:
        assert worker.core_text("café") == worker.core_text("café")
        assert worker.core_text("ＥＮＤ") == "end"  # fullwidth folds under NFKC

    def test_has_core(self, worker) -> None:
        assert worker.has_core("word") is True
        assert worker.has_core(",") is False


class TestAlignTokenTexts:
    def test_identical_sequences_are_all_aligned(self, worker) -> None:
        tokens = ["hola", "mundo"]
        words = [word("hola", 0.0, 0.5), word("mundo", 0.5, 1.0)]
        result = worker.align_token_texts(tokens, words)
        assert [r["timestamp_alignment_status"] for r in result] == ["aligned", "aligned"]
        assert result[1]["token_start_time"] == 0.5

    def test_punctuation_attached_by_the_asr_still_aligns(self, worker) -> None:
        """The bug this exists for: WhisperX says "world," spaCy says "world" + ","."""
        tokens = ["Hello", "world", ",", "this", "is"]
        words = [word("Hello", 0.2, 0.5), word("world,", 0.6, 1.0),
                 word("this", 1.1, 1.3), word("is", 1.4, 1.5)]
        result = worker.align_token_texts(tokens, words)
        assert [r["timestamp_alignment_status"] for r in result] == \
            ["aligned", "aligned", "approximate", "aligned", "aligned"]
        assert result[1]["token_start_time"] == 0.6
        # The comma borrows the span of the word it belongs to, at low confidence.
        assert result[2]["token_start_time"] == 0.6 and result[2]["timestamp_alignment_confidence"] == 0.2

    def test_a_stray_comma_does_not_shift_every_later_word(self, worker) -> None:
        """Positional matching drifted everything after the first comma."""
        tokens = ["a", "b", ",", "c", "d", "e"]
        words = [word("a", 0, 1), word("b,", 1, 2), word("c", 2, 3),
                 word("d", 3, 4), word("e", 4, 5)]
        result = worker.align_token_texts(tokens, words)
        aligned = [r for r in result if r["timestamp_alignment_status"] == "aligned"]
        assert len(aligned) == 5
        by_token = dict(zip(tokens, result))
        assert by_token["c"]["token_start_time"] == 2 and by_token["e"]["token_start_time"] == 4

    def test_punctuation_token_borrows_the_preceding_word(self, worker) -> None:
        result = worker.align_token_texts(["hi", "."], [word("hi", 3.0, 4.0)])
        assert result[1]["token_start_time"] == 3.0
        assert result[1]["timestamp_alignment_status"] == "approximate"

    def test_leading_punctuation_borrows_the_word_that_follows(self, worker) -> None:
        # "(" opens "hi", so "hi"'s span is where it was spoken; there is no
        # preceding word to borrow, and null would be a worse answer.
        result = worker.align_token_texts(["(", "hi"], [word("hi", 0.0, 1.0)])
        assert result[0]["token_start_time"] == 0.0
        assert result[0]["timestamp_alignment_status"] == "approximate"
        assert result[0]["timestamp_alignment_confidence"] == 0.2

    def test_punctuation_the_asr_timed_as_its_own_word_is_matched_exactly(self, worker) -> None:
        result = worker.align_token_texts(["hi", "."], [word("hi", 0.0, 0.5), word(".", 0.5, 0.6)])
        assert result[1]["timestamp_alignment_status"] == "aligned"
        assert result[1]["token_start_time"] == 0.5

    def test_spacy_splitting_one_word_does_not_consume_it_twice(self, worker) -> None:
        # spaCy splits "l'homme" into "l'" + "homme"; WhisperX kept one token.
        tokens = ["l'", "homme", "parle"]
        words = [word("l'homme", 0.0, 1.0), word("parle", 1.0, 2.0)]
        result = worker.align_token_texts(tokens, words)
        assert result[0]["token_start_time"] == 0.0
        assert result[1]["token_start_time"] == 0.0  # borrowed, not consumed
        # Neither fragment consumed the word, so "parle" still finds its own.
        assert result[2]["token_start_time"] == 1.0
        assert result[2]["timestamp_alignment_status"] in {"aligned", "approximate"}

    def test_forward_search_reports_approximate_with_lower_confidence(self, worker) -> None:
        # ASR dropped a word the transcript has, so the match sits two ahead.
        tokens = ["one", "three"]
        words = [word("one", 0.0, 1.0), word("two", 1.0, 2.0), word("three", 2.0, 3.0)]
        result = worker.align_token_texts(tokens, words)
        assert result[1]["timestamp_alignment_status"] == "approximate"
        assert result[1]["token_start_time"] == 2.0
        assert 0.3 <= result[1]["timestamp_alignment_confidence"] < 1.0

    def test_confidence_never_reaches_one_for_an_approximate_match(self, worker) -> None:
        tokens = ["a", "b"]
        words = [word("x", 0.0, 1.0), word("a", 1.0, 2.0), word("b", 2.0, 3.0)]
        for row in worker.align_token_texts(tokens, words):
            if row["timestamp_alignment_status"] != "aligned":
                assert row["timestamp_alignment_confidence"] < 1.0

    def test_far_away_match_is_reported_as_unmatched_not_guessed(self, worker) -> None:
        tokens = ["zebra"]
        words = [word(f"w{i}", float(i), float(i) + 1) for i in range(10)]
        result = worker.align_token_texts(tokens, words)
        assert result[0]["timestamp_alignment_status"] == "unmatched"
        assert result[0]["token_start_time"] is None

    def test_no_words_at_all_means_no_timing(self, worker) -> None:
        result = worker.align_token_texts(["hola", "mundo"], [])
        assert [r["timestamp_alignment_status"] for r in result] == ["no_timing", "no_timing"]

    def test_word_without_timestamp_propagates_its_own_status(self, worker) -> None:
        words = [word("hola", None, None, alignment_status="missing_timestamp")]
        result = worker.align_token_texts(["hola"], words)
        assert result[0]["timestamp_alignment_status"] == "missing_timestamp"
        assert result[0]["token_start_time"] is None

    def test_more_tokens_than_words_leaves_the_extra_ones_unmatched(self, worker) -> None:
        result = worker.align_token_texts(["a", "b", "c"], [word("a", 0.0, 1.0)])
        assert result[0]["timestamp_alignment_status"] == "aligned"
        assert result[2]["token_start_time"] is None

    def test_alignment_is_monotonic_in_time(self, worker) -> None:
        tokens = ["one", "two", "three", "four"]
        words = [word(t, float(i), float(i) + 0.5) for i, t in enumerate(["one", "two", "three", "four"])]
        times = [r["token_start_time"] for r in worker.align_token_texts(tokens, words)]
        assert times == sorted(times)

    def test_output_length_always_matches_input(self, worker) -> None:
        tokens = ["a", ",", "b", "!", "c"]
        words = [word("a,", 0.0, 1.0), word("b!", 1.0, 2.0), word("c", 2.0, 3.0)]
        assert len(worker.align_token_texts(tokens, words)) == len(tokens)

    def test_every_row_has_the_full_timing_contract(self, worker) -> None:
        rows = worker.align_token_texts(["a", ","], [word("a,", 0.0, 1.0)])
        for row in rows:
            assert set(row) == {"token_start_time", "token_end_time",
                                "timestamp_alignment_status", "timestamp_alignment_confidence"}
            assert row["timestamp_alignment_status"] in {
                "aligned", "approximate", "unmatched", "no_timing", "missing_timestamp", "segment_only"}

    def test_empty_token_list(self, worker) -> None:
        assert worker.align_token_texts([], [word("a", 0.0, 1.0)]) == []

    def test_realistic_two_sentence_segment(self, worker) -> None:
        tokens = ["Hello", "world", ",", "this", "is", "a", "test", "."]
        words = [word("Hello", 0.233, 0.554), word("world,", 0.594, 0.974),
                 word("this", 1.155, 1.295), word("is", 1.415, 1.515), word("a", 1.555, 1.575),
                 word("test", 1.615, 1.9), word(".", 1.9, 1.95)]
        rows = worker.align_token_texts(tokens, words)
        aligned = sum(1 for r in rows if r["timestamp_alignment_status"] == "aligned")
        # Only the trailing period is a borrowed span; nothing is lost.
        assert aligned == 7
        assert all(r["token_start_time"] is not None for r in rows)


class TestWorkerCodeDigest:
    def test_digest_is_stable(self) -> None:
        from multimodal_pipeline.stages.base import worker_code_digest

        assert worker_code_digest(WORKER) == worker_code_digest(WORKER)

    def test_digest_changes_when_the_worker_changes(self, tmp_path: Path) -> None:
        from multimodal_pipeline.stages.base import worker_code_digest

        script = tmp_path / "w.py"
        script.write_text("print('v1')")
        before = worker_code_digest(script)
        script.write_text("print('v2')")
        assert worker_code_digest(script) != before

    def test_missing_worker_has_no_digest_instead_of_crashing(self, tmp_path: Path) -> None:
        from multimodal_pipeline.stages.base import worker_code_digest

        assert worker_code_digest(tmp_path / "absent.py") is None


class TestInstalledModelInventory:
    """Installing a language model must invalidate a stage that settled for blank.

    Model resolution is configured -> same family -> discovered -> blank, so which
    models exist changes the output as much as the configured names. Leaving them out
    of the fingerprint made an installed `es_core_news_lg` serve a stale `blank`
    result forever: lemmas and POS tags came back empty and nothing said so.
    """

    @staticmethod
    def _model(root, name, version="3.8.0", spacy_version=">=3.8.0,<3.9.0"):
        directory = root / ".venv" / "lib" / "python3.12" / "site-packages" / name
        directory.mkdir(parents=True, exist_ok=True)
        parts = name.split("_", 1)
        meta = {"name": parts[1] if len(parts) > 1 else name, "version": version}
        if spacy_version:
            meta["spacy_version"] = spacy_version
        (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    def test_reads_names_and_versions_from_the_environment(self, tmp_path):
        from multimodal_pipeline.stages.spacy_source import installed_model_inventory

        self._model(tmp_path, "en_core_web_lg", "3.8.0")
        self._model(tmp_path, "es_core_news_lg", "3.8.0")
        assert installed_model_inventory(tmp_path) == {
            "en_core_web_lg": "3.8.0", "es_core_news_lg": "3.8.0"}

    def test_ignores_packages_that_are_not_spacy_models(self, tmp_path):
        """tokenizers and friends also ship a meta.json; they are not pipelines."""
        from multimodal_pipeline.stages.spacy_source import installed_model_inventory

        self._model(tmp_path, "en_core_web_lg")
        self._model(tmp_path, "tokenizers", "0.20.0", spacy_version=None)
        assert installed_model_inventory(tmp_path) == {"en_core_web_lg": "3.8.0"}

    def test_unreadable_environment_is_empty_not_fatal(self, tmp_path):
        """A missing venv means 'no models', which is what the stage would use anyway."""
        from multimodal_pipeline.stages.spacy_source import installed_model_inventory

        assert installed_model_inventory(tmp_path / "absent") == {}

    def test_unreadable_meta_json_is_skipped(self, tmp_path):
        from multimodal_pipeline.stages.spacy_source import installed_model_inventory

        self._model(tmp_path, "en_core_web_lg")
        broken = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages" / "xx_broken"
        broken.mkdir(parents=True)
        (broken / "meta.json").write_text("{not json", encoding="utf-8")
        assert installed_model_inventory(tmp_path) == {"en_core_web_lg": "3.8.0"}

    def test_a_newly_installed_model_invalidates_the_stage(self, context, tmp_path):
        stage = SpacySourceStage()
        context.config.spacy.uv_project = tmp_path
        before = stage.request_digest(context)
        self._model(tmp_path, "es_core_news_lg")
        assert stage.request_digest(context) != before

    def test_reinstalling_the_same_model_does_not_invalidate(self, context, tmp_path):
        """The inventory is a fingerprint, not a timestamp: re-syncing must not rerun
        every linguistics stage."""
        stage = SpacySourceStage()
        context.config.spacy.uv_project = tmp_path
        self._model(tmp_path, "es_core_news_lg")
        before = stage.request_digest(context)
        self._model(tmp_path, "es_core_news_lg")
        assert stage.request_digest(context) == before

    def test_a_model_version_bump_invalidates(self, context, tmp_path):
        stage = SpacySourceStage()
        context.config.spacy.uv_project = tmp_path
        self._model(tmp_path, "es_core_news_lg", "3.8.0")
        before = stage.request_digest(context)
        self._model(tmp_path, "es_core_news_lg", "3.8.1")
        assert stage.request_digest(context) != before

    def test_english_stage_inherits_the_binding(self, context, tmp_path):
        from multimodal_pipeline.stages.spacy_english import SpacyEnglishStage

        stage = SpacyEnglishStage()
        context.config.spacy.uv_project = tmp_path
        before = stage.request_digest(context)
        self._model(tmp_path, "en_core_web_lg")
        assert stage.request_digest(context) != before
