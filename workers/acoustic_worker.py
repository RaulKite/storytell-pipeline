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

    Formants and intensity come from objects built with the *whole* window so
    their internal edge effects land on window boundaries rather than inside
    speech, and timestamps stay on the global timeline.
    """
    import parselmouth  # noqa: PLC0415 - imported by the caller's contract

    window = sound.extract_part(from_time=start, to_time=end, preserve_times=True)
    step = params["time_step"]
    pitch = window.to_pitch(time_step=step, pitch_floor=params["pitch_floor"],
                            pitch_ceiling=params["pitch_ceiling"])
    intensity = window.to_intensity(time_step=step, minimum_pitch=params["pitch_floor"])
    formants = window.to_formant_burg(time_step=step, maximum_formant=params["number_of_formants"],
                                      bandwidth=50.0)
    formant_ceiling = params["formant_ceiling"]
    number_of_formants = params["number_of_formants"]
    for timestamp in pitch.ts():
        global_time = start + timestamp
        f0 = clean(pitch.get_value_at_time(timestamp))
        voiced = f0 is not None
        intensity_value = clean(intensity.get_value_at_time(timestamp))
        formant_values: list[float | None] = []
        for index in range(1, 4):
            if index > number_of_formants:
                formant_values.append(None)
                continue
            value = clean(formants.get_value_for_formant_number(timestamp, index))
            # Praat's Burg formant tracker is unstable above the configured
            # ceiling on whispered/noisy frames; an out-of-range formant is
            # worse than no formant.
            formant_values.append(value if value is not None and value <= formant_ceiling else None)
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
        import parselmouth  # noqa: PLC0415 - import failures must land in the result file

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
                "praat_version": getattr(parselmouth, "praat_version", None),
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
                for frame in extract_window(sound, position, stop, params):
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
            "model_version": f"parselmouth/{getattr(parselmouth, 'praat_version', 'unknown')}",
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
