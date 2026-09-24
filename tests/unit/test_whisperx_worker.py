"""WhisperX worker helpers that are pure logic and therefore testable without a GPU.

The one that matters is language confidence. whisperx detects the language from the
first 30 seconds only, warns in its own log that shorter audio may be inaccurate,
computes a probability -- and returns the code alone. Losing that number is what let a
4.2 s Spanish news clip enter the dataset as Catalan with no warning anywhere: the
downstream spaCy stage then did the correct thing with a wrong answer, loaded the
Catalan model, and tagged Spanish text as PROPN with invented dependencies. An honest
blank pipeline would have been the better result and nothing could have chosen it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER = PROJECT_ROOT / "workers" / "whisperx_worker.py"


def load_worker():
    """Import the worker by path: it runs inside its own uv project and must not be
    importable as a package module (it imports whisperx at call time, not here)."""
    spec = importlib.util.spec_from_file_location("whisperx_worker_under_test", WORKER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _InnerModel:
    """faster_whisper.WhisperModel stand-in: returns (code, probability, all_probs).

    ``whisperx.asr.WhisperModel`` does not override ``detect_language``, so on the real
    object this is the implementation that runs -- and it is the only one that still knows
    the probability.
    """

    result = ("es", 0.93, [("es", 0.93), ("ca", 0.04)])
    raises = False

    def __init__(self):
        self.calls = []
        self.kwargs = None

    def detect_language(self, audio=None, **kwargs):
        self.calls.append(audio)
        # Record the bound parameter too: with `audio` named in the signature, **kwargs
        # alone is empty when it is passed as a keyword, which is the thing asserted.
        self.kwargs = {"audio": audio, **kwargs}
        if type(self).raises:
            raise RuntimeError("shape changed upstream")
        return type(self).result


class _Pipeline:
    """whisperx.asr.FasterWhisperPipeline stand-in.

    ``load_model()`` returns this, not a WhisperModel. Its own ``detect_language`` is
    whisperx's override: it computes the probability, logs it, and returns the bare code.
    """

    def __init__(self, inner=None):
        self.model = inner if inner is not None else _InnerModel()
        self.override_calls = []

    def detect_language(self, audio=None, **kwargs):
        self.override_calls.append(audio)
        return "es"


def make_pipeline(result=None, raises=False):
    inner = _InnerModel()
    if result is not None:
        type(inner).result = result
    type(inner).raises = raises
    return _Pipeline(inner), inner


class TestDetectLanguage:
    def test_returns_the_probability_whisperx_discards(self):
        """The reason this helper exists."""
        worker = load_worker()
        pipeline, _ = make_pipeline(result=("es", 0.93, [("es", 0.93)]))
        assert worker.detect_language(pipeline, "AUDIO") == ("es", 0.93)

    def test_asks_the_inner_model_rather_than_the_pipeline_override(self):
        """whisperx's override returns the code alone, so going through it loses the
        number. An earlier version of this helper did that and reported
        ``probability: null`` on a real run while every unit test still passed."""
        worker = load_worker()
        pipeline, inner = make_pipeline()
        worker.detect_language(pipeline, "AUDIO")
        assert inner.calls == ["AUDIO"]
        assert pipeline.override_calls == []

    def test_passes_audio_as_the_keyword_the_real_signature_uses(self):
        """faster_whisper's signature is ``detect_language(self, audio=None, features=...)``.
        A version of this helper called it positionally through the wrong object and the
        failure was swallowed, so the calling convention is asserted, not trusted."""
        worker = load_worker()
        pipeline, inner = make_pipeline()
        worker.detect_language(pipeline, "AUDIO")
        assert inner.kwargs.get("audio") == "AUDIO"

    def test_missing_probability_is_reported_as_absent_not_zero(self):
        worker = load_worker()
        pipeline, _ = make_pipeline(result=("es",))
        assert worker.detect_language(pipeline, "AUDIO") == ("es", None)

    def test_a_non_numeric_probability_is_not_invented(self):
        worker = load_worker()
        pipeline, _ = make_pipeline(result=("es", "0.9", []))
        assert worker.detect_language(pipeline, "AUDIO") == ("es", None)

    def test_a_non_string_code_does_not_become_a_language(self):
        worker = load_worker()
        pipeline, _ = make_pipeline(result=(None, 0.4, []))
        assert worker.detect_language(pipeline, "AUDIO") == (None, None)

    def test_a_plain_string_return_is_still_a_language(self):
        worker = load_worker()
        pipeline, _ = make_pipeline(result="ca")
        assert worker.detect_language(pipeline, "AUDIO") == ("ca", None)

    def test_a_detection_failure_costs_the_number_not_the_transcript(self):
        """``(None, None)`` is not a failure here: with no language the worker lets
        transcribe() detect it as it always did, and the report records that the
        confidence is unavailable. Losing a transcript over a confidence value would be."""
        worker = load_worker()
        pipeline, _ = make_pipeline(raises=True)
        assert worker.detect_language(pipeline, "AUDIO") == (None, None)

    def test_a_pipeline_without_an_inner_model_degrades_quietly_but_not_loudly(self):
        worker = load_worker()
        pipeline = _Pipeline(inner=None)
        pipeline.model = None
        assert worker.detect_language(pipeline, "AUDIO") == (None, None)
        assert pipeline.override_calls == []


class TestLanguageReliability:
    def test_explicit_language_is_not_graded(self):
        """An operator-set whisperx.language is a decision, not a guess to score."""
        worker = load_worker()
        result = worker.language_reliability(4.2, None, "es")
        assert result["status"] == "configured"
        assert result["reasons"] == []

    def test_short_audio_is_low_confidence(self):
        worker = load_worker()
        result = worker.language_reliability(4.2, 0.71, None)
        assert result["status"] == "low"
        assert any("4.2s" in reason for reason in result["reasons"])

    def test_low_probability_is_low_confidence(self):
        worker = load_worker()
        result = worker.language_reliability(120.0, 0.31, None)
        assert result["status"] == "low"
        assert any("0.31" in reason for reason in result["reasons"])

    def test_missing_probability_is_itself_a_reason(self):
        """'We do not know how sure we were' must not read as 'we were sure'."""
        worker = load_worker()
        result = worker.language_reliability(120.0, None, None)
        assert result["status"] == "low"
        assert any("unavailable" in reason for reason in result["reasons"])

    def test_long_audio_and_high_probability_is_ok(self):
        worker = load_worker()
        result = worker.language_reliability(120.0, 0.98, None)
        assert result == {"status": "ok", "probability": 0.98, "reasons": []}

    def test_the_detection_window_is_the_documented_thirty_seconds(self):
        """whisperx hardcodes a 30 s detection window; the threshold must match it, not
        a number chosen to make a test pass."""
        worker = load_worker()
        assert worker.LANGUAGE_DETECTION_WINDOW_SECONDS == 30.0
        at_least = worker.language_reliability(30.0, 0.9, None)
        below = worker.language_reliability(29.9, 0.9, None)
        assert at_least["status"] == "ok"
        assert below["status"] == "low"

    def test_both_problems_are_reported_not_just_the_first(self):
        worker = load_worker()
        result = worker.language_reliability(3.0, 0.2, None)
        assert len(result["reasons"]) == 2


class TestWorkerContract:
    def test_language_detection_reaches_the_raw_artifact(self):
        """The whole point is that downstream stages can read this, so it must be in the
        document the spaCy stage consumes, not only in the result summary."""
        source = WORKER.read_text(encoding="utf-8")
        assert 'native["language_detection"]' in source

    def test_detection_only_runs_when_no_language_was_configured(self):
        """Otherwise an explicit language gets graded as if it were a guess, and the
        pipeline would warn about a decision the operator made on purpose."""
        source = WORKER.read_text(encoding="utf-8")
        guard = source.split("language_detection = language_reliability")[0]
        assert "if language is None:" in guard

    def test_the_reported_state_comes_from_the_real_path_not_the_handler(self):
        """The exact shape of the La 1 clip: 8 s of audio, so the detection is a guess
        whatever the number says. A previous version of this helper produced
        ``probability: null`` on a real run while passing every unit test, because its own
        try/except swallowed the call failure -- so the two functions are driven together
        here and the number has to arrive."""
        worker = load_worker()
        pipeline, _ = make_pipeline(result=("ca", 0.53, [("ca", 0.53), ("es", 0.44)]))
        code, probability = worker.detect_language(pipeline, "AUDIO")
        report = worker.language_reliability(8.0, probability, None)
        assert code == "ca" and probability == 0.53
        assert report["status"] == "low"
        assert report["probability"] == 0.53
        assert any("8.0s" in reason for reason in report["reasons"])
