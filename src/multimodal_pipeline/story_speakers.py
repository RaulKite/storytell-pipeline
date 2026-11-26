"""Speaker attribution for the transcript the `stories` prompt is shown.

The stories stage needs *a* speaker label beside each word it sends, and the operator's
decision is Nemotron diarization for it. Nemotron's turns **overlap**: two speakers can
hold the floor at the same moment, so "the turn that contains this word" is not a
function — it can have two answers. The rule here is deliberately conservative:

* the dominant speaker must cover at least ``min_coverage`` of the word's own duration,
  and
* no other speaker may cover at least ``max_competing`` of it,

or the word is labelled :data:`UNKNOWN_SPEAKER`. A boundary built on a coin-flip
attribution is worse than "unknown", because the prompt tells the model that a label is
automatic, arbitrary and possibly wrong; an honest gap is something the model can mark as
uncertain, while a wrong label reads as evidence.

The thresholds are fractions of the *word's* duration, not seconds, so the same rule
behaves the same way for a 60 ms "a" and for a 900 ms word. A word with no timing at all
cannot be attributed and is ``unknown`` rather than given its neighbouring label.

Nothing here reads a file, imports a heavy tool or knows what a story is: it is interval
arithmetic over turns and word timings, and it is unit-tested on its own by
``tests/unit/test_story_speakers.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

#: The label the prompt is told to expect, and the only value that means "not decidable".
UNKNOWN_SPEAKER = "unknown"

#: Defaults, mirrored by ``StoriesConfig``. Duplicated as literals so this module has no
#: import from ``config`` and stays importable without pydantic in a unit test.
DEFAULT_MIN_COVERAGE = 0.6
DEFAULT_MAX_COMPETING = 0.25


def _number(value: Any) -> float | None:
    """A finite-looking number as float, or None.

    ``True`` is excluded on purpose: ``isinstance(True, int)`` is true, and a JSON
    ``true`` where a time belongs is not a 1-second interval.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def coverage(turns: Sequence[Mapping[str, Any]], start: float, end: float) -> dict[str, float]:
    """Seconds of ``[start, end)`` each speaker's turns cover.

    Summed per speaker, so two turns of the same speaker covering the same instant count
    twice. That is intentional: it can only make a competing speaker look *stronger*, and
    the rule's job in the overlap case is to refuse a label, not to award one.
    """
    totals: dict[str, float] = {}
    for turn in turns:
        speaker = str(turn.get("speaker_id") or "").strip()
        if not speaker:
            continue
        turn_start = _number(turn.get("start_time"))
        turn_end = _number(turn.get("end_time"))
        if turn_start is None or turn_end is None or turn_end <= turn_start:
            continue
        overlap = max(0.0, min(end, turn_end) - max(start, turn_start))
        if overlap:
            totals[speaker] = totals.get(speaker, 0.0) + overlap
    return totals


def attribute_interval(turns: Sequence[Mapping[str, Any]], start: Any, end: Any, *,
                       min_coverage: float = DEFAULT_MIN_COVERAGE,
                       max_competing: float = DEFAULT_MAX_COMPETING) -> str:
    """The one speaker that may be claimed for ``[start, end)``, or ``unknown``.

    The interval is attributed when exactly one speaker has enough evidence and nobody
    else has a meaningful share. Everything else — no turns, no timing, a zero-length
    interval, a split interval, a genuinely simultaneous stretch — is ``unknown``.
    """
    start_s, end_s = _number(start), _number(end)
    if start_s is None or end_s is None or end_s <= start_s:
        return UNKNOWN_SPEAKER
    totals = coverage(turns, start_s, end_s)
    if not totals:
        return UNKNOWN_SPEAKER
    duration = end_s - start_s
    dominant, covered = max(totals.items(), key=lambda item: (item[1], item[0]))
    if covered < min_coverage * duration:
        return UNKNOWN_SPEAKER
    for speaker, amount in totals.items():
        if speaker != dominant and amount >= max_competing * duration:
            return UNKNOWN_SPEAKER
    return dominant


def attribute_words(words: Sequence[Mapping[str, Any]],
                    turns: Sequence[Mapping[str, Any]], *,
                    min_coverage: float = DEFAULT_MIN_COVERAGE,
                    max_competing: float = DEFAULT_MAX_COMPETING) -> list[str]:
    """One label per word, in order. Untimed words are ``unknown``, never inherited."""
    labels: list[str] = []
    for word in words:
        start, end = word.get("start_time"), word.get("end_time")
        if _number(start) is None or _number(end) is None:
            labels.append(UNKNOWN_SPEAKER)
            continue
        labels.append(attribute_interval(turns, start, end, min_coverage=min_coverage,
                                         max_competing=max_competing))
    return labels


def attribute_segments(segments: Sequence[Mapping[str, Any]],
                       turns: Sequence[Mapping[str, Any]], *,
                       min_coverage: float = DEFAULT_MIN_COVERAGE,
                       max_competing: float = DEFAULT_MAX_COMPETING) -> list[str]:
    """The same rule over whole segment intervals.

    Used when ``speech/words.parquet`` is missing, or when one segment has no word rows:
    the segment's own interval is the only timing evidence left, and applying the same
    thresholds keeps the two paths from quietly disagreeing about what "attributed" means.
    """
    return [attribute_interval(turns, segment.get("start_time"), segment.get("end_time"),
                               min_coverage=min_coverage, max_competing=max_competing)
            for segment in segments]


def render_speaker_runs(text: str, spans: Sequence[tuple[int, int] | None],
                        labels: Sequence[str]) -> str:
    """``[speaker_0] words… [unknown] more words`` for one segment's text.

    ``spans`` are the word character ranges inside ``text`` (``None`` for a word whose
    position could not be resolved) and ``labels`` the label of each word. Every piece is
    a **substring of the segment text**, so a quote the model copies out of this rendering
    is still findable in the transcript; only whitespace at a run boundary is collapsed.
    A word without a span stays inside the surrounding run rather than being dropped or
    moved.

    With no words at all the whole text becomes one ``[unknown]`` run: the prompt must
    never show text with no label, because the model is asked to say where a boundary
    sits in *attributed* speech.
    """
    if not text.strip():
        return f"[{UNKNOWN_SPEAKER}]"
    pieces: list[tuple[str, int, int]] = []
    current: str | None = None
    run_start = 0
    last_end = 0
    for span, label in zip(spans, labels):
        if span is None:
            continue
        start, end = span
        if current is None:
            current, run_start, last_end = label, 0, end
            continue
        if label == current:
            last_end = end
            continue
        pieces.append((current, run_start, last_end))
        # The next run starts where the previous word ended, so the characters between
        # two words land in exactly one piece and no character is invented or lost.
        current, run_start, last_end = label, last_end, end
    if current is None:
        return f"[{UNKNOWN_SPEAKER}] {text.strip()}"
    pieces.append((current, run_start, len(text)))
    rendered = " ".join(f"[{label}] {text[start:stop].strip()}"
                        for label, start, stop in pieces if text[start:stop].strip())
    return rendered or f"[{UNKNOWN_SPEAKER}] {text.strip()}"
