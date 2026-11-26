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
    """Seconds of ``[start, end)`` each speaker's turns cover, as a **union**.

    A speaker's turns are merged before they are measured, so two of their own turns
    covering the same instant cover it once. Summing them would let a speaker reach
    ``min_coverage`` on 40 % of a word by being diarized twice over it, which is exactly
    the arithmetic the rule exists to refuse: the quantity being compared against a share
    of the word's duration is *time held*, and no instant is held twice.
    """
    intervals: dict[str, list[tuple[float, float]]] = {}
    for turn in turns:
        speaker = str(turn.get("speaker_id") or "").strip()
        if not speaker:
            continue
        turn_start = _number(turn.get("start_time"))
        turn_end = _number(turn.get("end_time"))
        if turn_start is None or turn_end is None or turn_end <= turn_start:
            continue
        low, high = max(start, turn_start), min(end, turn_end)
        if high > low:
            intervals.setdefault(speaker, []).append((low, high))
    totals: dict[str, float] = {}
    for speaker, spans in intervals.items():
        total = 0.0
        current_start, current_end = spans[0]
        for low, high in sorted(spans)[1:]:
            if low > current_end:
                total += current_end - current_start
                current_start, current_end = low, high
            else:
                current_end = max(current_end, high)
        total += current_end - current_start
        if total:
            totals[speaker] = total
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


def speaker_runs(text: str, spans: Sequence[tuple[int, int] | None],
                 labels: Sequence[str]) -> list[tuple[str, str]]:
    """``[(label, piece), …]``: the runs one segment's text is split into.

    ``spans`` are the word character ranges inside ``text`` (``None`` for a word whose
    position could not be resolved) and ``labels`` the label of each word. Every piece is
    a **substring of the segment text**, so a quote the model copies out of the rendering
    is still findable in the transcript; only whitespace at a run boundary is collapsed.
    A word without a span stays inside the surrounding run rather than being dropped or
    moved — which also means its own label never reaches the prompt, and so cannot be
    named by the answer.

    With no words at all the whole text becomes one ``[unknown]`` run: the prompt must
    never show text with no label, because the model is asked to say where a boundary
    sits in *attributed* speech.
    """
    if not text.strip():
        return [(UNKNOWN_SPEAKER, "")]
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
        return [(UNKNOWN_SPEAKER, text.strip())]
    pieces.append((current, run_start, len(text)))
    runs = [(label, text[start:stop].strip()) for label, start, stop in pieces
            if text[start:stop].strip()]
    return runs or [(UNKNOWN_SPEAKER, text.strip())]


def render_speaker_runs(text: str, spans: Sequence[tuple[int, int] | None],
                        labels: Sequence[str]) -> str:
    """``[speaker_0] words… [unknown] more words`` for one segment's text."""
    return render_runs(speaker_runs(text, spans, labels))


def render_runs(runs: Sequence[tuple[str, str]]) -> str:
    """The one-line form of ``speaker_runs``: ``[label] piece`` joined by a space."""
    return " ".join(f"[{label}] {piece}".strip() for label, piece in runs)


def emitted_labels(runs: Sequence[tuple[str, str]]) -> list[str]:
    """The labels a rendering actually shows, in order of first appearance.

    This, and not the per-word label list, is what the answer may name: a word whose
    position could not be resolved is folded into a neighbouring run, so its own label is
    absent from the prompt and naming it would be a claim about words the model never saw.
    """
    shown: list[str] = []
    for label, _piece in runs:
        if label not in shown:
            shown.append(label)
    return shown
