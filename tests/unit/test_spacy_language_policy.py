"""What the spaCy stages do with WhisperX's opinion about their input language.

WhisperX has graded its own detection for a while: ``language_detection`` in
``speech/raw/whisperx.json`` says ``configured`` (the operator pinned the language),
``ok``, or ``low`` (short audio, or a probability under 0.5). Nothing in ``src/`` or
``workers/`` ever read it — ``grep -rn language_detection --include='*.py' src/ workers/``
returned only the whisperx worker. So the spaCy stage received the detected *string* and
nothing about how much it was worth.

Measured on this corpus, that grade is pessimistic: every clip is shorter than WhisperX's
30 s detection window, so **all seven** auto-detected clips are ``low`` — including
``en`` at 0.997 — and the model each one produced is measurably correct (La 1 reached a
real Spanish pipeline with verified Spanish lemmas, the English clips reached
``en_core_web_lg``). A policy that demoted ``low`` unconditionally would therefore have
stripped the full pipeline from the only Spanish clip in the corpus. Hence:

* the default keeps today's model choice and *says* the grade was low;
* ``spacy.trust_low_language_detection = false`` refuses to build a full linguistic
  layer on a sub-window guess and takes the honest-empty path instead;
* the English variant is excluded, because it forces ``en`` and has no detection to
  trust or distrust.

The worker runs inside ``environments/spacy`` with real spaCy, so it is exercised here
against a stand-in module — the same approach ``tests/unit/test_whisperx_worker.py``
takes for whisperx. What is under test is the *decision and its record*: which language
reaches the resolver, which selection falls out, what gets logged, and what the raw
document says about it. Model loading itself is owned by the real runs.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import types
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from multimodal_pipeline.config import PipelineConfig
from multimodal_pipeline.stages.base import StageContext
from multimodal_pipeline.stages.spacy_english import SpacyEnglishStage
from multimodal_pipeline.stages.spacy_source import SpacySourceStage, select_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER = PROJECT_ROOT / "workers" / "spacy_worker.py"

CONFIGURED = {"en": "en_core_web_lg", "es": "es_core_news_lg"}

#: The La 1 clip's actual grade, measured off the corpus: Spanish, below the 30 s
#: window, and correct.
LA_UNO_GRADE = {"status": "low", "probability": 0.883,
                "reasons": ["audio is 8.0s, below the 30s detection window"]}


def load_worker():
    spec = importlib.util.spec_from_file_location("spacy_policy_worker", WORKER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------- fake spaCy

class FakeToken:
    """Just enough of a spaCy Token for ``build_document`` to read it."""

    def __init__(self, text: str, index: int, head_index: int, char_start: int):
        self.text = text
        self.i = index
        self.idx = char_start
        self.head_index = head_index
        self.lower_ = text.lower()
        self.lemma_ = text.lower()
        self.pos_ = "NOUN"
        self.tag_ = "N"
        self.morph = ""
        self.dep_ = "ROOT"
        self.ent_type_ = ""
        self.is_alpha = text.isalpha()
        self.is_stop = False
        self.is_digit = text.isdigit()
        self.like_num = text.isdigit()

    #: build_document resolves the head through the sentence's token list, the way
    #: spaCy resolves it through the parent Doc.
    _sentence: list["FakeToken"] = []

    @property
    def head(self) -> "FakeToken":
        return self._sentence[self.head_index]


class FakeSentence:
    def __init__(self, tokens: list[FakeToken], text: str, start_char: int):
        self.tokens = tokens
        self.text = text
        self.start = tokens[0].i
        self.end = tokens[-1].i + 1
        self.start_char = start_char
        self.end_char = start_char + len(text)

    def __iter__(self):
        return iter(self.tokens)


class FakeDoc:
    def __init__(self, tokens: list[FakeToken], sentences: list[FakeSentence]):
        self._tokens = tokens
        self.sents = sentences

    def __iter__(self):
        return iter(self._tokens)

    def __len__(self):
        return len(self._tokens)


def fake_doc(text: str) -> FakeDoc:
    """One sentence, one token per whitespace word, head = itself.

    The tokens are wired through a shared list so ``token.head`` resolves the way
    spaCy's does; ``build_document`` only ever reads head.i / .text / .pos_.
    """
    words = [word for word in text.split() if word]
    tokens: list[FakeToken] = []
    char = 0
    for index, word in enumerate(words):
        token = FakeToken(word, index, index, char)
        tokens.append(token)
        char += len(word) + 1
    for token in tokens:
        token._sentence = tokens
    return FakeDoc(tokens, [FakeSentence(tokens, text, 0)])


class FakeNlp:
    def __init__(self, name: str, *, full: bool):
        self.name = name
        self.max_length = 1_000_000
        self.pipe_names = ("tokenizer", "tagger", "attribute_ruler", "parser", "lemmatizer") \
            if full else ("tokenizer", "sentencizer")

    def __call__(self, text: str) -> FakeDoc:
        return fake_doc(text)


def install_fake_spacy(monkeypatch: pytest.MonkeyPatch, installed: set[str]) -> None:
    """spaCy stand-in: loads only what is "installed", blanks anything else.

    ``spacy.util`` is registered as its own module because the worker reaches it with
    ``from spacy.util import get_installed_models``; an attribute alone is not enough for
    that import form, and swallowing the ImportError would report "no models installed"
    and quietly turn every test into a blank-pipeline test.
    """
    module = types.ModuleType("spacy")
    module.__version__ = "3.8.16"

    def load(name: str):
        if name not in installed:
            raise OSError(f"[E048] Can't find model '{name}'")
        return FakeNlp(name, full=True)

    def blank(language: str):
        return FakeNlp(f"blank[{language}]", full=False)

    util = types.ModuleType("spacy.util")
    util.get_installed_models = lambda: sorted(installed)  # type: ignore[attr-defined]

    module.load = load  # type: ignore[attr-defined]
    module.blank = blank  # type: ignore[attr-defined]
    module.util = util  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spacy", module)
    monkeypatch.setitem(sys.modules, "spacy.util", util)


# ------------------------------------------------------------------ runner

def write_segments(path: Path, language: str | None, text: str = "Muy buena entrada.") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"segment_id": "seg000001", "start_time": 0.0, "end_time": 2.5,
             "language": language, "text": text, "speaker_id": "SPEAKER_00"}]
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def run_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture,
               *, language: str | None,
               installed: set[str], source_models: dict[str, str],
               language_detection: str | None, trust_low: bool,
               variant: str = "source") -> dict[str, Any]:
    """Run the real worker ``main`` with spaCy stood in; return its outputs.

    Returns ``{"document": ..., "result": ..., "warnings": [...]}`` — the raw document
    the worker wrote, the result summary it wrote, and every warning the worker logged.

    Warnings are read with ``caplog``: the worker logs through ``logging``, which is what
    the orchestrator's per-stage log file captures (``run_command`` merges the worker's
    stdout and stderr). Asserting on a returned string instead would stay green while the
    operator saw nothing.
    """
    worker = load_worker()
    install_fake_spacy(monkeypatch, installed)
    segments = write_segments(tmp_path / "segments.parquet", language)
    raw_output = tmp_path / "raw" / "spacy_source.json"
    result_json = tmp_path / "raw" / "worker_result.json"

    argv = ["--variant", variant, "--video-id", "clip", "--segments", str(segments),
            "--raw-output", str(raw_output), "--result-json", str(result_json),
            "--source-models", json.dumps(source_models), "--english-model", "en_core_web_lg",
            "--fallback-model", "blank"]
    if variant == "english":
        translations = tmp_path / "translations.parquet"
        translations.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(
            [{"segment_id": "seg000001", "english_text": "A very good entry."}]), translations)
        argv += ["--translations", str(translations)]
    if language_detection is not None:
        argv += ["--language-detection", language_detection]
    if trust_low:
        argv += ["--trust-low-language-detection"]

    caplog.set_level(logging.WARNING, logger="spacy_worker")
    exit_code = worker.main(argv)
    report = json.loads(result_json.read_text(encoding="utf-8")) if result_json.is_file() else {}
    assert exit_code == 0, f"worker failed: {report.get('error')}\n{report.get('traceback', '')}"
    return {
        "document": json.loads(raw_output.read_text(encoding="utf-8")),
        "result": report,
        "warnings": [
            record.getMessage()
            for record in caplog.records
            if record.name == "spacy_worker" and record.levelno >= logging.WARNING
        ],
    }


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
         caplog: pytest.LogCaptureFixture):
    """``run(...)`` with the per-test plumbing already bound."""
    def _run(**kwargs: Any) -> dict[str, Any]:
        return run_worker(monkeypatch, tmp_path, caplog, **kwargs)

    return _run


# ------------------------------------------------------------------ config

class TestConfigDefault:
    def test_trust_low_language_detection_defaults_to_true(self) -> None:
        """The default must keep today's behaviour: on this corpus a false default would
        demote every auto-detected clip, including the correct ones."""
        from multimodal_pipeline.config import SpacyConfig

        assert SpacyConfig().trust_low_language_detection is True

    def test_the_key_is_loadable_from_yaml(self, config: PipelineConfig) -> None:
        assert config.spacy.trust_low_language_detection is True


# ------------------------------------------------------- stage request/argv

class TestStageRequestCarriesTheGrade:
    def test_source_request_includes_the_whisperx_grade(self, context: StageContext) -> None:
        write_whisperx_raw(context, LA_UNO_GRADE)
        request = SpacySourceStage().request(context)
        assert request["language_detection"] == LA_UNO_GRADE
        assert request["trust_low_language_detection"] is True

    def test_absent_raw_file_keeps_the_key_with_none(self, context: StageContext) -> None:
        """Old datasets must keep working, and the absence must be visible rather than
        silently equal to some other value."""
        assert context.artifact("whisperx_raw").is_file() is False
        request = SpacySourceStage().request(context)
        assert "language_detection" in request
        assert request["language_detection"] is None

    def test_raw_document_without_the_key_reads_as_absent(self, context: StageContext) -> None:
        path = context.artifact("whisperx_raw")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"segments": [], "language": "es"}), encoding="utf-8")
        assert SpacySourceStage().request(context)["language_detection"] is None

    def test_an_unreadable_raw_document_is_absent_not_fatal(self, context: StageContext) -> None:
        path = context.artifact("whisperx_raw")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        assert SpacySourceStage().request(context)["language_detection"] is None

    def test_a_regrade_invalidates_the_source_stage(self, context: StageContext) -> None:
        """WhisperX re-grading changes how much the linguistics layer should trust its
        input, so it must not be served from a cache produced under another grade."""
        stage = SpacySourceStage()
        before = stage.request_digest(context)
        write_whisperx_raw(context, LA_UNO_GRADE)
        assert stage.request_digest(context) != before

    def test_the_flag_is_part_of_the_fingerprint(self, context: StageContext) -> None:
        stage = SpacySourceStage()
        before = stage.request_digest(context)
        context.config.spacy.trust_low_language_detection = False
        assert stage.request_digest(context) != before

    def test_english_request_does_not_carry_the_key(self, context: StageContext) -> None:
        """The English variant forces ``en``; making its cache depend on a source-language
        detection it never consults would be a false dependency."""
        write_whisperx_raw(context, LA_UNO_GRADE)
        request = SpacyEnglishStage().request(context)
        assert "language_detection" not in request
        assert "trust_low_language_detection" not in request
        # The source variant still sees it, from the same context.
        assert SpacySourceStage().request(context)["language_detection"] == LA_UNO_GRADE

    def test_english_digest_is_unchanged_by_a_regrade(self, context: StageContext) -> None:
        stage = SpacyEnglishStage()
        before = stage.request_digest(context)
        write_whisperx_raw(context, LA_UNO_GRADE)
        assert stage.request_digest(context) == before


class TestWorkerArgv:
    """argv is the only channel the worker gets, so the grade has to be visible in it."""

    @pytest.fixture(autouse=True)
    def transcript(self, context: StageContext) -> StageContext:
        """``worker_argv`` resolves its input paths eagerly, so they must exist."""
        for name in ("speech_segments", "speech_words", "translation_segments"):
            path = context.artifact(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.is_file():
                pq.write_table(pa.Table.from_pylist([{"segment_id": "seg000001"}]), path)
        return context
    def test_source_argv_passes_the_grade_as_compact_json(self, context: StageContext) -> None:
        write_whisperx_raw(context, LA_UNO_GRADE)
        argv = SpacySourceStage().worker_argv(
            context, context.artifact("spacy_source_raw"), "digest")
        value = argv[argv.index("--language-detection") + 1]
        assert json.loads(value) == LA_UNO_GRADE
        assert value == json.dumps(LA_UNO_GRADE, separators=(",", ":"), ensure_ascii=False), \
            "the grade is passed as compact JSON"
        assert "--trust-low-language-detection" in argv

    def test_source_argv_passes_none_when_no_grade_exists(self, context: StageContext) -> None:
        argv = SpacySourceStage().worker_argv(
            context, context.artifact("spacy_source_raw"), "digest")
        assert argv[argv.index("--language-detection") + 1] == "none"

    def test_the_flag_follows_the_config(self, context: StageContext) -> None:
        context.config.spacy.trust_low_language_detection = False
        argv = SpacySourceStage().worker_argv(
            context, context.artifact("spacy_source_raw"), "digest")
        assert "--trust-low-language-detection" not in argv

    def test_english_argv_carries_neither(self, context: StageContext) -> None:
        write_whisperx_raw(context, LA_UNO_GRADE)
        argv = SpacyEnglishStage().worker_argv(
            context, context.artifact("spacy_english_raw"), "digest")
        assert "--language-detection" not in argv
        assert "--trust-low-language-detection" not in argv


def write_whisperx_raw(context: StageContext, grade: dict[str, Any] | None) -> Path:
    path = context.artifact("whisperx_raw")
    path.parent.mkdir(parents=True, exist_ok=True)
    document: dict[str, Any] = {"segments": [], "language": "es"}
    if grade is not None:
        document["language_detection"] = grade
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


# ------------------------------------------------------------ worker policy

class TestWorkerTrustPolicy:
    """The three outcomes, measured on the worker's own selection + provenance."""

    def test_low_and_trusted_picks_the_same_model_as_today(
            self, run) -> None:
        """The corpus case: La 1 detected `es` at 0.88 on 8 s of audio, graded low, and
        its Spanish model is the right one. Trusting it must not change the choice."""
        out = run(language="es",
                         installed={"es_core_news_lg", "en_core_web_lg"},
                         source_models=CONFIGURED,
                         language_detection=json.dumps(LA_UNO_GRADE), trust_low=True)
        expected = select_model("es", CONFIGURED, {"es_core_news_lg", "en_core_web_lg"},
                                fallback="blank")
        assert out["document"]["selected_model"] == expected["model"]
        assert out["document"]["model_selection_status"] == expected["status"]
        assert out["document"]["language_reliability"] == LA_UNO_GRADE
        assert out["document"]["language_reliability_trusted"] is True

    def test_low_and_untrusted_lands_on_the_honest_path(
            self, run) -> None:
        """No language reaches the resolver, so it reports what it can do without one
        instead of confidently annotating text in a language nobody proved."""
        out = run(language="es",
                         installed={"es_core_news_lg", "en_core_web_lg"},
                         source_models=CONFIGURED,
                         language_detection=json.dumps(LA_UNO_GRADE), trust_low=False)
        assert out["document"]["model_selection_status"] == "fallback_no_model"
        assert out["document"]["selected_model"] == "blank"
        assert out["document"]["language_reliability_trusted"] is False
        assert "full" not in out["document"]["capabilities"]
        # The language the transcript carries is still recorded: the demotion is about
        # the model, not about rewriting what WhisperX reported.
        assert out["document"]["language"] == "es"
        expected = select_model(None, CONFIGURED, {"es_core_news_lg", "en_core_web_lg"},
                                fallback="blank")
        assert out["document"]["model_selection_status"] == expected["status"]

    @pytest.mark.parametrize("grade", [
        {"status": "configured", "probability": None, "reasons": []},
        {"status": "ok", "probability": 0.99, "reasons": []},
    ])
    def test_configured_or_ok_never_warns_and_keeps_the_choice(
            self, run, grade: dict) -> None:
        out = run(language="es",
                         installed={"es_core_news_lg"}, source_models=CONFIGURED,
                         language_detection=json.dumps(grade), trust_low=False)
        assert out["warnings"] == []
        assert out["document"]["selected_model"] == "es_core_news_lg"
        assert out["document"]["language_reliability"] == grade
        assert out["document"]["language_reliability_trusted"] is True

    def test_a_missing_grade_keeps_the_default_and_warns_once(
            self, run) -> None:
        """A dataset produced before whisperx graded anything must keep working exactly
        as it does today — but the check silently not running is not the same as the
        check passing, so it says so."""
        out = run(language="es",
                         installed={"es_core_news_lg"}, source_models=CONFIGURED,
                         language_detection=None, trust_low=False)
        assert out["document"]["selected_model"] == "es_core_news_lg"
        assert out["document"]["language_reliability"] == {"status": "absent"}
        assert out["document"]["language_reliability_trusted"] is True
        assert len(out["warnings"]) == 1
        assert "no WhisperX language_detection grade was available" in out["warnings"][0]
        assert "could not be checked" in out["warnings"][0]

    def test_the_none_literal_behaves_like_a_missing_grade(
            self, run) -> None:
        out = run(language="es",
                         installed={"es_core_news_lg"}, source_models=CONFIGURED,
                         language_detection="none", trust_low=True)
        assert out["document"]["selected_model"] == "es_core_news_lg"
        assert out["document"]["language_reliability"] == {"status": "absent"}
        assert any("no WhisperX language_detection grade" in line for line in out["warnings"])

    @pytest.mark.parametrize("payload", ["{not json", "[1,2]", "\"low\"", ""])
    def test_a_malformed_grade_keeps_the_default_and_warns(
            self, run, payload: str) -> None:
        """Never crash on a grade: the transcript is worth more than the confidence
        number, which is the same rule the whisperx worker follows."""
        out = run(language="es",
                         installed={"es_core_news_lg"}, source_models=CONFIGURED,
                         language_detection=payload, trust_low=True)
        assert out["document"]["selected_model"] == "es_core_news_lg"
        assert out["document"]["language_reliability"] == {"status": "absent"}
        assert any("no WhisperX language_detection grade" in line for line in out["warnings"])

    def test_english_never_applies_the_policy(
            self, run) -> None:
        """Even an explicitly hostile grade cannot demote the English pass — it forces
        ``en`` and the key is not even in its argv."""
        out = run(language="es",
                         installed={"en_core_web_lg"}, source_models=CONFIGURED,
                         language_detection=json.dumps(LA_UNO_GRADE), trust_low=False,
                         variant="english")
        assert out["document"]["model_selection_status"] == "english_default"
        assert out["document"]["selected_model"] == "en_core_web_lg"
        assert out["document"]["language_reliability"] == {"status": "not_applicable"}
        assert out["document"]["language_reliability_trusted"] is True
        assert out["warnings"] == []


class TestWorkerWarnings:
    """The warnings are the whole point of the default: a choice that is kept silently
    is indistinguishable from one that was never questioned."""

    def test_trusted_low_names_the_language_probability_model_and_config_key(
            self, run) -> None:
        out = run(language="es",
                         installed={"es_core_news_lg"}, source_models=CONFIGURED,
                         language_detection=json.dumps(LA_UNO_GRADE), trust_low=True)
        assert len(out["warnings"]) == 1
        message = out["warnings"][0]
        assert "low reliability" in message
        assert "es" in message
        assert "0.88" in message, "the probability the grade reported"
        assert "below the 30s detection window" in message, "the grade's own reason"
        assert "es_core_news_lg" in message, "the model that was still selected"
        assert "spacy.trust_low_language_detection = false" in message, "how to reverse it"

    def test_untrusted_low_names_the_demotion_and_the_key_that_reverses_it(
            self, run) -> None:
        out = run(language="es",
                         installed={"es_core_news_lg"}, source_models=CONFIGURED,
                         language_detection=json.dumps(LA_UNO_GRADE), trust_low=False)
        assert len(out["warnings"]) == 1
        message = out["warnings"][0]
        assert "low reliability" in message
        assert "es" in message and "0.88" in message
        assert "demoted" in message
        assert "blank" in message
        assert "spacy.trust_low_language_detection = true" in message

    def test_a_grade_without_reasons_still_produces_a_readable_warning(
            self, run) -> None:
        out = run(language="es",
                         installed={"es_core_news_lg"}, source_models=CONFIGURED,
                         language_detection=json.dumps({"status": "low"}), trust_low=True)
        assert len(out["warnings"]) == 1
        assert "low reliability" in out["warnings"][0]
        assert "unknown" in out["warnings"][0], "a missing probability is stated, not hidden"


class TestWorkerResultSummary:
    """The result JSON is what the orchestrator reads back; the grade has to be in it or
    the provenance stops at a file nobody summarises."""

    def test_result_summary_carries_the_grade_for_every_source_branch(
            self, run) -> None:
        cases = [
            (json.dumps(LA_UNO_GRADE), True, LA_UNO_GRADE, True),
            (json.dumps(LA_UNO_GRADE), False, LA_UNO_GRADE, False),
            (None, True, {"status": "absent"}, True),
        ]
        for payload, trust_low, expected_grade, expected_trusted in cases:
            out = run(language="es",
                             installed={"es_core_news_lg"}, source_models=CONFIGURED,
                             language_detection=payload, trust_low=trust_low)
            assert out["result"]["language_reliability"] == expected_grade
            assert out["result"]["language_reliability_trusted"] is expected_trusted
            assert out["result"]["model_selection_status"]

    def test_the_tokens_and_sentences_counts_survive_the_new_keys(
            self, run) -> None:
        out = run(language="es",
                         installed={"es_core_news_lg"}, source_models=CONFIGURED,
                         language_detection=json.dumps(LA_UNO_GRADE), trust_low=True)
        assert out["result"]["status"] == "ok"
        assert out["result"]["tokens"] == 3
        assert out["result"]["selected_model"] == "es_core_news_lg"
        assert list(out["result"]).count("language_reliability") == 1


class TestLanguageIsStillRecorded:
    def test_an_untrusted_detection_does_not_blank_the_language(
            self, run) -> None:
        """The demotion is about which model annotates the text. WhisperX's reported
        language is a fact about the transcript and stays in the document."""
        out = run(language="es",
                         installed={"es_core_news_lg"}, source_models=CONFIGURED,
                         language_detection=json.dumps(LA_UNO_GRADE), trust_low=False)
        assert out["document"]["language"] == "es"
        assert out["result"]["status"] == "ok"
