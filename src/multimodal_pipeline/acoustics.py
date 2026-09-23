"""Acoustic measurement maths, independent of Praat.

Kept apart from the worker so the numerical behaviour (voiced-only statistics,
pause detection, segment aggregation) is unit-testable without Parselmouth
installed and without any audio file. The same rules apply whether measurements
came from Praat or from a hand-written test fixture.

Null policy: an unvoiced frame has ``f0_hz = None``, and a segment with no
voiced frames has ``f0_mean = None`` — never 0. A fabricated zero would be
averaged into group statistics and silently corrupt pitch comparisons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Any, Iterable, Sequence

Frame = dict[str, Any]


def is_number(value: Any) -> bool:
    """True for a usable real measurement.

    ``bool`` is excluded even though it subclasses ``int``: a ``voiced`` flag must
    never be averaged into a pitch or intensity statistic as a 0/1 value.
    Non-finite values are excluded so no NaN/inf survives into Parquet.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def summarise(values: Iterable[Any]) -> dict[str, float | None]:
    """mean/median/min/max/std over the *numeric* values only.

    Returns all-nulls for an empty or non-numeric input so a caller can tell
    "no measurement" apart from "measured as zero".
    """
    numbers = [float(value) for value in values if is_number(value)]
    if not numbers:
        return {"mean": None, "median": None, "min": None, "max": None, "std": None}
    ordered = sorted(numbers)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[middle]
    else:
        median = (ordered[middle - 1] + ordered[middle]) / 2.0
    return {
        "mean": round(fmean(numbers), 6),
        "median": round(median, 6),
        "min": round(ordered[0], 6),
        "max": round(ordered[-1], 6),
        # Population stdev: these are complete segment measurements, not samples.
        "std": round(pstdev(numbers), 6) if len(numbers) > 1 else 0.0,
    }


def voiced_ratio(frames: Sequence[Frame]) -> float | None:
    """Share of frames the tracker considered voiced; null when nothing measured."""
    if not frames:
        return None
    voiced = sum(1 for frame in frames if frame.get("voiced") is True)
    return round(voiced / len(frames), 6)


def interval_intersections(intervals: Sequence[tuple[float, float]],
                           ranges: Sequence[tuple[float, float]]) -> list[float]:
    """Per-interval total overlap with a set of ranges (pauses against segments)."""
    totals: list[float] = []
    for start, end in intervals:
        total = 0.0
        for other_start, other_end in ranges:
            left, right = max(start, other_start), min(end, other_end)
            if right > left:
                total += right - left
        totals.append(round(total, 6))
    return totals


def detect_silences(frames: Sequence[Frame], *, minimum_duration: float,
                    silence_threshold_db: float | None = None,
                    frame_step: float | None = None) -> list[tuple[float, float]]:
    """Contiguous quiet runs at least ``minimum_duration`` long.

    Quiet means "below ``silence_threshold_db`` when an intensity threshold is
    configured, otherwise unvoiced". Both definitions matter: a whispered
    sentence is unvoiced but not silent, while background hum is voiced but
    quiet — the threshold, when supplied, is the more faithful signal.

    Runs touching the recording edges are kept: they are real pauses for
    prosody, and dropping them would bias ``pause_ratio`` low.
    """
    if not frames:
        return []
    step = frame_step or _median_step(frames)
    runs: list[tuple[float, float]] = []
    run_start: float | None = None
    previous_time: float | None = None
    for frame in frames:
        quiet = frame_is_quiet(frame, silence_threshold_db=silence_threshold_db)
        timestamp = frame.get("timestamp")
        if not is_number(timestamp):
            continue
        current = float(timestamp)
        if quiet:
            if run_start is None:
                run_start = current
        elif run_start is not None:
            runs.append((run_start, (previous_time + step) if previous_time is not None else current))
            run_start = None
        previous_time = current
    if run_start is not None and previous_time is not None:
        runs.append((run_start, previous_time + step))
    return [(round(start, 6), round(end, 6)) for start, end in runs
            if (end - start) + 1e-9 >= minimum_duration]


def frame_is_quiet(frame: Frame, *, silence_threshold_db: float | None) -> bool:
    if silence_threshold_db is not None:
        intensity = frame.get("intensity_db")
        if is_number(intensity):
            return float(intensity) < silence_threshold_db
        # No intensity reading for this frame: fall back to voicing.
    return frame.get("voiced") is not True


def _median_step(frames: Sequence[Frame]) -> float:
    times = [float(frame["timestamp"]) for frame in frames if is_number(frame.get("timestamp"))]
    if len(times) < 2:
        return 0.0
    deltas = sorted(b - a for a, b in zip(times, times[1:]) if b > a)
    return deltas[len(deltas) // 2] if deltas else 0.0


@dataclass
class SegmentAcoustics:
    """Aggregated acoustic description of one transcript segment."""

    segment_id: str
    values: dict[str, float | None]

    def as_row(self) -> dict[str, Any]:
        return {"segment_id": self.segment_id, **self.values}


def aggregate_segment(segment: dict[str, Any], frames: Sequence[Frame], *,
                      silences: Sequence[tuple[float, float]],
                      duration: float | None = None) -> SegmentAcoustics:
    """Aggregate only the frames that fall inside the segment's own interval.

    Filtering by interval (rather than by index arithmetic) keeps the result
    correct when segments overlap or when diarization left gaps.
    """
    start, end = segment.get("start_time"), segment.get("end_time")
    inside = [frame for frame in frames if frame_in_segment(frame, start, end)]
    span = duration
    if not is_number(span) and is_number(start) and is_number(end):
        span = max(float(end) - float(start), 0.0)
    pause_ranges = clip_silences(silences, start, end)
    pause_duration = round(sum(end_ - start_ for start_, end_ in pause_ranges), 6)
    pause_ratio = round(pause_duration / float(span), 6) if is_number(span) and span > 0 else None
    pitch = [frame.get("f0_hz") for frame in inside if frame.get("voiced") is True]
    f0 = summarise(pitch)
    intensity = summarise(frame.get("intensity_db") for frame in inside)
    return SegmentAcoustics(
        segment_id=str(segment.get("segment_id")),
        values={
            "speaker_id": segment.get("speaker_id"),
            "start_time": float(start) if is_number(start) else None,
            "end_time": float(end) if is_number(end) else None,
            "duration": round(float(span), 6) if is_number(span) else None,
            "voiced_ratio": voiced_ratio(inside),
            "f0_mean": f0["mean"],
            "f0_median": f0["median"],
            "f0_min": f0["min"],
            "f0_max": f0["max"],
            "f0_std": f0["std"],
            "intensity_mean": intensity["mean"],
            "intensity_median": intensity["median"],
            "intensity_min": intensity["min"],
            "intensity_max": intensity["max"],
            "intensity_std": intensity["std"],
            "f1_mean": summarise(frame.get("f1_hz") for frame in inside)["mean"],
            "f2_mean": summarise(frame.get("f2_hz") for frame in inside)["mean"],
            "f3_mean": summarise(frame.get("f3_hz") for frame in inside)["mean"],
            "pause_count": len(pause_ranges),
            "pause_duration": pause_duration,
            "pause_ratio": pause_ratio,
        },
    )


def frame_in_segment(frame: Frame, start: Any, end: Any) -> bool:
    timestamp = frame.get("timestamp")
    if not is_number(timestamp):
        return False
    if is_number(start) and float(timestamp) < float(start) - 1e-9:
        return False
    if is_number(end) and float(timestamp) > float(end) + 1e-9:
        return False
    return True


def clip_silences(silences: Sequence[tuple[float, float]], start: Any, end: Any) -> list[tuple[float, float]]:
    """Silence intervals intersected with a segment, so ratios cannot exceed 1."""
    clipped: list[tuple[float, float]] = []
    for silence_start, silence_end in silences:
        left = max(silence_start, float(start)) if is_number(start) else silence_start
        right = min(silence_end, float(end)) if is_number(end) else silence_end
        if right > left:
            clipped.append((left, right))
    return clipped


def normalise_frame_row(raw: Sequence[Any], *, video_id: str, schema_version: str = "1.0") -> dict[str, Any]:
    """Compact raw record ``[t, f0, intensity, voiced, f1, f2, f3]`` → named row.

    Non-finite values are coerced to ``None`` so a stray ``NaN`` from Praat
    cannot survive into Parquet and break a later ``WHERE f0_hz IS NOT NULL``.
    """
    timestamp, f0, intensity, voiced, f1, f2, f3 = (list(raw) + [None] * 7)[:7]
    return {
        "schema_version": schema_version,
        "video_id": video_id,
        "timestamp": float(timestamp) if is_number(timestamp) else None,
        "f0_hz": float(f0) if is_number(f0) else None,
        "intensity_db": float(intensity) if is_number(intensity) else None,
        "voiced": bool(voiced) if voiced is not None else None,
        "f1_hz": float(f1) if is_number(f1) else None,
        "f2_hz": float(f2) if is_number(f2) else None,
        "f3_hz": float(f3) if is_number(f3) else None,
    }
