"""Interval-overlap speaker assignment (pure logic, no I/O).

Assigns a speaker to every transcript segment and word by measuring how much of
each diarization turn overlaps it. The winner is the turn with the largest
overlap; ties break deterministically (earlier ``start_time``, then
``speaker_id``) so reruns are bit-identical.

Everything needed to audit a decision is returned: the winning overlap in
seconds, that overlap as a ratio of the interval's own duration, and an
explicit method string — so "no speaker here because the diarization had a gap"
is a recorded fact rather than a silent null.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

#: A diarization turn. ``(start_time, end_time, speaker_id)``.
Turn = tuple[float, float, str]

METHOD_MAX_OVERLAP = "max_overlap"
METHOD_POINT_CONTAINMENT = "point_containment"
METHOD_NO_OVERLAP = "no_overlap"
EPSILON = 1e-9


@dataclass(frozen=True)
class Assignment:
    speaker_id: str | None
    overlap_seconds: float
    overlap_ratio: float
    method: str
    #: Second-best speaker, kept because overlapping speech is real and a
    #: consumer may want to know the interval was ambiguous.
    runner_up_speaker_id: str | None = None
    runner_up_overlap_seconds: float = 0.0
    #: Every speaker that overlapped, ordered by descending overlap.
    contributions: tuple[tuple[str, float], ...] = ()

    def as_fields(self) -> dict[str, object]:
        return {
            "speaker_id": self.speaker_id,
            "speaker_overlap_seconds": round(self.overlap_seconds, 6),
            "speaker_overlap_ratio": round(self.overlap_ratio, 6),
            "speaker_assignment_method": self.method,
        }


def overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    """Length of ``[start_a, end_a) ∩ [start_b, end_b)``; never negative."""
    left = max(start_a, start_b)
    right = min(end_a, end_b)
    return max(0.0, right - left)


def normalise_turns(turns: Iterable[Sequence[object]]) -> list[Turn]:
    """Coerce loosely-typed turn rows into sorted, validated tuples."""
    cleaned: list[Turn] = []
    for turn in turns:
        start, end, speaker = turn[0], turn[1], turn[2]
        if start is None or end is None or speaker is None:
            continue
        try:
            start_f, end_f = float(start), float(end)
        except (TypeError, ValueError):
            continue
        if end_f + EPSILON < start_f:
            continue  # malformed turn: skip rather than poison the assignment
        cleaned.append((start_f, end_f, str(speaker)))
    cleaned.sort(key=lambda item: (item[0], item[1], item[2]))
    return cleaned


def assign_speaker(
    start: float | None,
    end: float | None,
    turns: Sequence[Turn],
    *,
    duration_fallback: float | None = None,
) -> Assignment:
    """Pick the speaker whose turn overlaps ``[start, end)`` the most.

    Degenerate (zero-length) intervals fall back to point containment, and an
    interval with no overlapping turn at all is reported as ``no_overlap`` with
    a null speaker instead of being guessed.
    """
    if start is None:
        return Assignment(None, 0.0, 0.0, METHOD_NO_OVERLAP)

    # A missing end is treated as "we only know where it starts": a probe point.
    point_like = end is None or abs(float(end) - float(start)) <= EPSILON
    end_value = float(end) if end is not None else float(start)

    if point_like:
        containing = [(speaker, 0.0) for (t_start, t_end, speaker) in turns
                      if t_start - EPSILON <= start <= t_end + EPSILON]
        if containing:
            speaker = _pick_deterministic({speaker for speaker, _ in containing}, turns)
            return Assignment(speaker, 0.0, 1.0, METHOD_POINT_CONTAINMENT)
        return Assignment(None, 0.0, 0.0, METHOD_NO_OVERLAP)

    interval_length = max(end_value - float(start), 0.0)
    if interval_length <= EPSILON:
        return Assignment(None, 0.0, 0.0, METHOD_NO_OVERLAP)

    contributions: list[tuple[str, float]] = []
    for t_start, t_end, speaker in turns:
        overlap = overlap_seconds(float(start), end_value, t_start, t_end)
        if overlap > EPSILON:
            contributions.append((speaker, overlap))

    if not contributions:
        return Assignment(None, 0.0, 0.0, METHOD_NO_OVERLAP)

    # Aggregate per speaker: one speaker may own several disjoint turns here.
    merged: dict[str, float] = {}
    for speaker, overlap in contributions:
        merged[speaker] = merged.get(speaker, 0.0) + overlap
    ordered = sorted(merged.items(), key=lambda item: (-item[1], _first_start(item[0], turns), item[0]))
    winner_speaker, winner_overlap = ordered[0]
    runner_up = ordered[1] if len(ordered) > 1 else (None, 0.0)
    # Ratio is capped at 1.0: with inclusive diarization two speakers can each
    # cover the whole interval, and a ratio > 1 would break consumer assumptions.
    ratio = min(1.0, winner_overlap / interval_length)
    return Assignment(
        speaker_id=winner_speaker,
        overlap_seconds=winner_overlap,
        overlap_ratio=ratio,
        method=METHOD_MAX_OVERLAP,
        runner_up_speaker_id=runner_up[0],
        runner_up_overlap_seconds=runner_up[1],
        contributions=tuple(ordered),
    )


def _first_start(speaker: str, turns: Sequence[Turn]) -> float:
    for t_start, _, t_speaker in turns:
        if t_speaker == speaker:
            return t_start
    return float("inf")


def _pick_deterministic(speakers: set[str], turns: Sequence[Turn]) -> str:
    """Stable tie-break: earliest owning turn, then speaker label."""
    return sorted(speakers, key=lambda speaker: (_first_start(speaker, turns), speaker))[0]


def assign_intervals(
    intervals: Iterable[dict[str, object]],
    turns: Sequence[Turn],
    *,
    start_key: str = "start_time",
    end_key: str = "end_time",
) -> list[Assignment]:
    """Assign every transcript row, preserving input order."""
    return [assign_speaker(row.get(start_key), row.get(end_key), turns) for row in intervals]  # type: ignore[arg-type]


def coverage_report(turns: Sequence[Turn], *, duration: float | None = None) -> dict[str, float]:
    """How much of the timeline any speaker covers — a cheap quality signal."""
    if not turns:
        return {"speaker_seconds": 0.0, "speaker_time": 0.0, "overlap_seconds": 0.0}
    events: list[tuple[float, int]] = []
    for start, end, _ in turns:
        events.append((start, 1))
        events.append((end, -1))
    events.sort(key=lambda item: (item[0], -item[1]))
    covered = 0.0
    stacked = 0.0
    active = 0
    previous = events[0][0]
    for at, delta in events:
        if active >= 1:
            covered += at - previous
        if active >= 2:
            stacked += at - previous
        active += delta
        previous = at
    return {
        "speaker_seconds": round(covered, 6),
        "speaker_time": round(covered / duration, 6) if duration else 0.0,
        "overlap_seconds": round(stacked, 6),
    }
