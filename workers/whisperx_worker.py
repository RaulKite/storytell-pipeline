#!/usr/bin/env python3
"""WhisperX worker — runs inside ``environments/whisperx`` only.

Invoked by the orchestrator as::

    uv run --project environments/whisperx python workers/whisperx_worker.py ...

Writes the native WhisperX result verbatim to ``--output`` plus a
machine-readable status file the orchestrator reads as proof of work.

API contract checked against the **installed** release (whisperx 3.8.6,
inspected 2026-09-23 in ``environments/whisperx/.venv``), not the README,
because they differ:

* ``load_model(whisper_arch, device, device_index, compute_type, asr_options,
  language, vad_method, vad_options, download_root, threads, use_auth_token)``
  — ``beam_size`` lives in ``asr_options``, not in ``transcribe``;
* ``pipeline.transcribe(audio, batch_size, language, chunk_size, ...)`` returns
  ``{"segments": [...], "language": ...}`` and has **no** ``vad=`` parameter —
  VAD is selected by ``vad_method`` at load time and is always applied;
* ``load_align_model(language_code, device, model_name=None)`` and
  ``align(transcript, model, metadata, audio, device, return_char_alignments=)``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WhisperX transcription + alignment worker")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--language", default="auto", help="'auto' lets Whisper detect the language")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--align-model", default=None)
    parser.add_argument("--vad-method", default="pyannote", choices=("pyannote", "silero"))
    parser.add_argument("--vad-merge-chunk-seconds", type=int, default=30)
    parser.add_argument("--asr-options", default="{}", help="JSON dict forwarded to load_model(asr_options=...)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--download-root", default=None)
    parser.add_argument("--request-hash", default=None)
    parser.add_argument("--result-json", type=Path, default=None,
                        help="Override the status file path (default: alongside --output)")
    return parser.parse_args(argv)


#: whisperx detects the language from this much audio and no more. Its own code warns
#: "Audio is shorter than 30s, language detection may be inaccurate", which is exactly
#: the case where the guess is wrong most often -- and short clips are this pipeline's
#: normal input.
LANGUAGE_DETECTION_WINDOW_SECONDS = 30.0


def detect_language(pipeline, audio):
    """Return ``(code, probability)`` for the audio; ``None`` if it cannot be determined.

    Measured against the installed whisperx 3.8.6, not inferred from documentation:
    ``load_model()`` returns a ``whisperx.asr.FasterWhisperPipeline``
    (a ``transformers.Pipeline``), whose ``.model`` is a ``whisperx.asr.WhisperModel``
    subclassing ``faster_whisper.WhisperModel``.

    * ``pipeline.detect_language(audio)`` is whisperx's own override: it computes
      ``language_probability``, writes it to a log line, and returns the bare code. Going
      through it is what made an early version of this helper report
      ``probability: null`` while every unit test passed.
    * ``pipeline.model`` does **not** override ``detect_language``; it inherits
      faster_whisper's, which returns ``(language, probability, all_language_probs)`` and
      takes audio. That is the one to call.

    Both answers were observed on the La 1 clip, and they differ: the pipeline's own
    detection said ``ca`` at 0.53 while this returns ``es`` at 0.88. Which is right is a
    question about 8 seconds of audio, not something to resolve silently here -- the
    worker records its own detection with its confidence and lets ``language_reliability``
    say how much to trust it.

    Every failure mode returns a missing probability rather than raising: a transcript is
    never worth losing over a confidence value. It must not be *silent* either, so
    ``language_reliability`` treats an absent probability as its own reason to downgrade.
    """
    try:
        raw = pipeline.model.detect_language(audio=audio)
    except Exception:  # noqa: BLE001 - never lose a transcript over a confidence value
        return None, None
    if isinstance(raw, tuple) and raw and isinstance(raw[0], str):
        probability = (raw[1] if len(raw) > 1
                       and isinstance(raw[1], (int, float)) else None)
        return raw[0], float(probability) if probability is not None else None
    return (raw if isinstance(raw, str) else None), None


def language_reliability(audio_seconds, probability, explicit_language):
    """Grade the auto-detection in the artifact, where downstream stages can read it.

    Only auto-detection is graded: an explicit ``whisperx.language`` is an operator
    decision, not a guess to score.
    """
    if explicit_language:
        return {"status": "configured", "probability": probability, "reasons": []}
    reasons = []
    if audio_seconds is not None and audio_seconds < LANGUAGE_DETECTION_WINDOW_SECONDS:
        reasons.append(
            f"audio is {audio_seconds:.1f}s, below the "
            f"{LANGUAGE_DETECTION_WINDOW_SECONDS:.0f}s detection window")
    if probability is None:
        reasons.append("detection probability unavailable")
    elif probability < 0.5:
        reasons.append(f"detection probability {probability:.2f}")
    return {"status": "low" if reasons else "ok", "probability": probability,
            "reasons": reasons}


def resolve_batch_size(value: str | int) -> int:
    """``auto`` scales with available VRAM; explicit values pass through."""
    if isinstance(value, int):
        return max(1, value)
    if str(value).strip().lower() != "auto":
        return max(1, int(value))
    try:
        import torch

        if torch.cuda.is_available():
            free_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
            if free_mb >= 20_000:
                return 16
            if free_mb >= 12_000:
                return 8
            if free_mb >= 8_000:
                return 4
    except Exception:  # noqa: BLE001 - fall back to the safest value
        pass
    return 4


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    result_path = args.result_json or (args.output.parent / "whisperx_worker_result.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"stage": "whisperx", "status": "running",
                                  "request_hash": args.request_hash}
    try:
        if not args.audio.is_file():
            raise FileNotFoundError(f"audio file not found: {args.audio}")
        import whisperx

        version = package_version("whisperx")
        device = args.device
        if device == "cuda":
            import torch

            if not torch.cuda.is_available():
                device = "cpu"
                payload["device_fallback_reason"] = "torch.cuda.is_available() was False"

        language = None if str(args.language).lower() in {"auto", "", "none"} else args.language
        batch_size = resolve_batch_size(args.batch_size)
        asr_options = json.loads(args.asr_options or "{}")
        asr_options.setdefault("beam_size", args.beam_size)

        pipeline = whisperx.load_model(
            args.model,
            device,
            device_index=args.device_index,
            compute_type=args.compute_type,
            asr_options=asr_options,
            language=language,
            vad_method=args.vad_method,
            vad_options={"chunk_size": args.vad_merge_chunk_seconds},
            download_root=args.download_root,
            threads=args.threads,
            use_auth_token=os.environ.get("HF_TOKEN") or None,
        )
        audio = whisperx.load_audio(str(args.audio))
        # Detect the language ourselves when it was not configured, so its confidence
        # survives; whisperx would recompute it internally and discard the number.
        language_detection = {"status": "configured", "probability": None, "reasons": []}
        if language is None:
            detected_code, probability = detect_language(pipeline, audio)
            audio_seconds = _wav_duration(args.audio)
            language = detected_code
            language_detection = language_reliability(audio_seconds, probability, None)
        transcription = pipeline.transcribe(audio, batch_size=batch_size, language=language)
        detected_language = transcription.get("language") or language

        alignment_meta: dict[str, object] = {"status": "not_attempted"}
        aligned = transcription
        if detected_language:
            try:
                align_model, align_metadata = whisperx.load_align_model(
                    language_code=detected_language, device=device, model_name=args.align_model
                )
                aligned = whisperx.align(
                    transcription["segments"], align_model, align_metadata, audio, device,
                    return_char_alignments=False,
                )
                aligned["language"] = detected_language
                alignment_meta = {
                    "status": "aligned",
                    "align_model": _align_model_id(align_metadata) or args.align_model,
                    "language": detected_language,
                }
            except Exception as exc:  # noqa: BLE001 - alignment is best-effort by design
                # An unsupported alignment language must never cost us the transcript.
                alignment_meta = {"status": "failed", "language": detected_language,
                                  "error": f"{type(exc).__name__}: {exc}"[:500]}
                aligned = transcription

        native = dict(aligned)
        native["language"] = detected_language
        native["language_detection"] = language_detection
        native["_worker"] = {
            "whisperx_version": version,
            "model": args.model,
            "model_revision": model_revision(pipeline, args.download_root),
            "device": device,
            "requested_device": args.device,
            "device_index": args.device_index,
            "compute_type": args.compute_type,
            "batch_size": batch_size,
            "asr_options": asr_options,
            "vad_method": args.vad_method,
            "vad_merge_chunk_seconds": args.vad_merge_chunk_seconds,
            "alignment": alignment_meta,
            "audio_path": str(args.audio),
            "audio_seconds": _wav_duration(args.audio),
            "elapsed_seconds": round(time.time() - started, 3),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(native, indent=2, ensure_ascii=False), encoding="utf-8")

        segments = native.get("segments") or []
        words = sum(len(segment.get("words") or []) for segment in segments)
        payload.update({
            "status": "ok",
            "tool_version": version,
            "model_version": args.model,
            "detected_language": detected_language,
            "language_detection": language_detection,
            "segments": len(segments),
            "words": words,
            "alignment": alignment_meta,
            "duration_seconds": round(time.time() - started, 3),
        })
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 0
    except Exception as exc:  # noqa: BLE001 - the orchestrator reads this file, not stderr
        payload.update({
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}"[:2000],
            "traceback": traceback.format_exc()[-6000:],
            "duration_seconds": round(time.time() - started, 3),
        })
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"whisperx worker failed: {exc}", file=sys.stderr)
        return 1


def _align_model_id(metadata) -> str | None:
    """Best-effort alignment model identity for provenance."""
    if isinstance(metadata, dict):
        return metadata.get("model_id") or metadata.get("model_name")
    return getattr(metadata, "model_id", None)


def model_revision(pipeline, download_root: str | None) -> str | None:
    """CTranslate2 snapshots are not git checkouts, so a revision is often absent.

    Returning ``None`` (recorded as "not pinned by upstream") is deliberate: an
    invented revision would be worse than an honest gap in provenance.
    """
    model_dir = getattr(pipeline, "model_dir", None) or getattr(
        getattr(pipeline, "model", None), "model_dir", None)
    for candidate in filter(None, [model_dir]):
        for name in ("revision.txt", "VERSION"):
            path = Path(candidate) / name
            if path.is_file():
                try:
                    return path.read_text(encoding="utf-8").strip()[:128]
                except OSError:
                    return None
        git_info = Path(candidate) / ".git" / "HEAD"
        if git_info.is_file():
            try:
                return git_info.read_text(encoding="utf-8").strip()[:128]
            except OSError:
                return None
    return None


def package_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _wav_duration(path: Path) -> float | None:
    import wave

    try:
        with wave.open(str(path), "rb") as handle:
            return round(handle.getnframes() / handle.getframerate(), 3)
    except Exception:  # noqa: BLE001 - informational only
        return None


if __name__ == "__main__":
    sys.exit(main())
