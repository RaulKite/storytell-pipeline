#!/usr/bin/env python3
"""Parselmouth (Praat) acoustic worker — runs inside ``environments/acoustic`` only.

Emits a high-resolution frame timeline (F0, intensity, voicing, F1-F3) plus the
silence structure as newline-delimited JSON, which the orchestrator normalises
into Parquet.

Three properties are deliberate:

* **Bounded memory.** Praat objects are built per ``chunk_seconds`` window and
  frames are written incrementally, so an hour-long recording costs the same RAM
  as a ten-second one.
* **Honest nulls.** Praat reports unvoiced F0 as ``None``/NaN; those become JSON
  ``null`` rather than 0 or -1, because a fabricated zero pitch would be averaged
  into downstream statistics.
* **No re-analysis for re-aggregation.** Everything Praat computed is preserved,
  so changing an aggregation rule later never re-runs signal analysis.

Raw output schema (``acoustic/raw/acoustic_features.jsonl``)::

    line 1   {"schema_version": ..., "video_id": ..., "parameters": {...},
              "frame_step_seconds": 0.01, "audio": {...}}
    line n   {"frame": [t, f0_hz, intensity_db, voiced, f1_hz, f2_hz, f3_hz]}
    last     {"silences": [[start, end], ...]}
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

# Bound once here: every Praat call below depends on it, and a lazy import per
# frame would be pure overhead.
import parselmouth  # noqa: E402 - worker runs only inside environments/acoustic


def praat_version() -> str | None:
    """The bundled Praat version, not just the Python binding's."""
    return getattr(parselmouth, "PRAAT_VERSION", None) or None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Praat/Parselmouth acoustic feature worker")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--raw-output", required=True, type=Path)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--time-step", type=float, default=0.01)
    parser.add_argument("--pitch-floor", type=float, default=75.0)
    parser.add_argument("--pitch-ceiling", type=float, default=500.0)
    parser.add_argument("--number-of-formants", type=int, default=5)
    parser.add_argument("--formant-ceiling", type=float, default=5500.0)
    parser.add_argument("--silence-threshold-db", type=float, default=None)
    parser.add_argument("--minimum-pause-duration", type=float, default=0.2)
    parser.add_argument("--chunk-seconds", type=float, default=120.0)
    parser.add_argument("--request-hash", default=None)
    parser.add_argument("--result-json", type=Path, default=None)
    return parser.parse_args(argv)


def clean(value: Any) -> float | None:
    """Praat sends ``None``/NaN for unmeasured frames; both become JSON null."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, 6)


def extract_window(sound, start: float, end: float, params: dict[str, Any]) -> Iterator[list[Any]]:
    """Yield ``[t, f0, intensity, voiced, f1, f2, f3]`` for one audio window.

    Pitch, intensity and formants are built per window so an hour of audio costs
    the same RAM as ten seconds; ``preserve_times`` keeps the window's own clock
    at zero, so each frame time is offset by ``start`` to land on the global
    video timeline.
    """
    step = params["time_step"]
    pitch = sound.to_pitch(time_step=step, pitch_floor=params["pitch_floor"],
                           pitch_ceiling=params["pitch_ceiling"])
    # Praat needs a jitter floor to place the intensity grid; the same floor used
    # for pitch keeps all three analyses on the same 10 ms frame clock.
    intensity = sound.to_intensity(time_step=step, minimum_pitch=params["pitch_floor"])
    # Parselmouth 0.4.x separates the *count* of tracked formants from their
    # frequency ceiling; conflating them silently drops formants.
    formants = sound.to_formant_burg(time_step=step,
                                     max_number_of_formants=params["number_of_formants"],
                                     maximum_formant=params["formant_ceiling"])
    formant_ceiling = params["formant_ceiling"]
    number_of_formants = min(params["number_of_formants"], 3)
    for timestamp in pitch.ts():
        global_time = start + timestamp
        f0 = clean(pitch.get_value_at_time(timestamp))
        voiced = f0 is not None
        intensity_value = clean(intensity.get_value(timestamp, parselmouth.ValueInterpolation.LINEAR))
        formant_values: list[float | None] = []
        for index in range(1, 4):
            if index > number_of_formants:
                formant_values.append(None)
                continue
            # Formant.get_value_at_time takes (formant_number, time), unlike the
            # time-first accessors on Pitch and Intensity.
            value = clean(formants.get_value_at_time(index, timestamp))
            # The Burg tracker is unstable on whispered/noisy frames; an
            # out-of-range formant is worse than no formant.
            formant_values.append(value if value is not None and 0.0 < value <= formant_ceiling else None)
        yield [round(global_time, 6), f0, intensity_value, voiced, *formant_values]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    raw_output = Path(args.raw_output)
    result_path = args.result_json or (raw_output.parent / "acoustic_worker_result.json")
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"stage": "acoustic", "status": "running",
                                  "request_hash": args.request_hash}
    try:
        if not args.audio.is_file():
            raise FileNotFoundError(f"audio file not found: {args.audio}")

        params = {
            "time_step": args.time_step,
            "pitch_floor": args.pitch_floor,
            "pitch_ceiling": args.pitch_ceiling,
            "number_of_formants": args.number_of_formants,
            "formant_ceiling": args.formant_ceiling,
        }
        sound = parselmouth.Sound(str(args.audio))
        duration = float(sound.duration)
        header = {
            "schema_version": "1.0",
            "video_id": args.video_id,
            "frame_step_seconds": args.time_step,
            "parameters": {
                **params,
                "backend": "parselmouth",
                "silence_threshold_db": args.silence_threshold_db,
                "minimum_pause_duration": args.minimum_pause_duration,
                "chunk_seconds": args.chunk_seconds,
                "parselmouth_version": getattr(parselmouth, "VERSION", None),
                "praat_version": praat_version(),
            },
            "audio": {
                "sample_rate": int(sound.sampling_frequency),
                "channels": int(sound.n_channels),
                "duration_seconds": round(duration, 6),
            },
        }
        frames = 0
        voiced_frames = 0
        quiet_spans: list[tuple[float, float]] = []
        with raw_output.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(header, ensure_ascii=False) + "\n")
            position = 0.0
            while position < duration:
                stop = min(duration, position + max(args.chunk_seconds, args.time_step * 10))
                # A fresh Praat object per window is what keeps memory bounded;
                # ``preserve_times`` leaves its clock at zero so ``position`` can be
                # added back to place frames on the global video timeline.
                window = sound.extract_part(from_time=position, to_time=stop, preserve_times=True)
                for frame in extract_window(window, position, stop, params):
                    handle.write(json.dumps({"frame": frame}) + "\n")
                    frames += 1
                    if frame[3]:
                        voiced_frames += 1
                    quiet = is_quiet(frame, threshold=args.silence_threshold_db)
                    if quiet:
                        # Merge with the previous frame when contiguous.
                        if quiet_spans and abs(quiet_spans[-1][1] - (frame[0] - args.time_step)) < args.time_step * 1.5:
                            quiet_spans[-1] = (quiet_spans[-1][0], frame[0])
                        else:
                            quiet_spans.append((frame[0], frame[0]))
                position = stop
            silences = [(round(start_, 6), round(end_ + args.time_step, 6))
                        for start_, end_ in quiet_spans
                        if (end_ + args.time_step - start_) + 1e-9 >= args.minimum_pause_duration]
            handle.write(json.dumps({"silences": [[s, e] for s, e in silences]}) + "\n")

        payload.update({
            "status": "ok",
            "tool_version": getattr(parselmouth, "VERSION", None),
            "model_version": f"parselmouth/{praat_version() or 'unknown'}",
            "frames": frames,
            "voiced_frames": voiced_frames,
            "silences": len(silences),
            "audio_seconds": round(duration, 3),
            "duration_seconds": round(time.time() - started, 3),
        })
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 0
    except Exception as exc:  # noqa: BLE001 - the orchestrator reads this file, not stderr
        payload.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"[:2000],
                        "traceback": traceback.format_exc()[-6000:],
                        "duration_seconds": round(time.time() - started, 3)})
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"acoustic worker failed: {exc}", file=sys.stderr)
        return 1


def is_quiet(frame: list[Any], *, threshold: float | None) -> bool:
    """Threshold-based silence when configured, otherwise unvoiced = quiet."""
    if threshold is not None and frame[2] is not None:
        return float(frame[2]) < threshold
    return not frame[3]


if __name__ == "__main__":
    sys.exit(main())
