#!/usr/bin/env python3
"""Pyannote diarization worker — runs inside ``environments/diarization`` only.

Uses the Community pipeline (default
``pyannote/speaker-diarization-community-1``), which returns an inclusive
timeline *and* an exclusive one. Both are emitted: the exclusive timeline is
what the orchestrator prefers for labelling transcript intervals, because it
assigns exactly one speaker per instant.

The Hugging Face token arrives via the ``HF_TOKEN`` environment variable — never
via argv — so it cannot leak into ``status.json``, logs or provenance.

API contract verified against the official model card (huggingface.co,
2026-09): ``Pipeline.from_pretrained(..., token=...)`` → ``pipeline.to("cuda")``
→ ``pipeline(audio)``, with ``output.speaker_diarization`` and
``output.exclusive_speaker_diarization``.
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
    parser = argparse.ArgumentParser(description="Pyannote Community diarization worker")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclusive-output", type=Path, default=None)
    parser.add_argument("--rttm-output", type=Path, default=None)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--pipeline", default="pyannote/speaker-diarization-community-1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--request-hash", default=None)
    parser.add_argument("--result-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    result_path = args.result_json or (args.output.parent / "diarization_worker_result.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"stage": "diarization", "status": "running",
                                  "request_hash": args.request_hash}
    try:
        if not args.audio.is_file():
            raise FileNotFoundError(f"audio file not found: {args.audio}")
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) is not set in the worker environment")

        import torch
        from pyannote.audio import Pipeline

        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            payload["device_fallback_reason"] = "torch.cuda.is_available() was False"

        pipeline = Pipeline.from_pretrained(args.pipeline, token=token)
        if pipeline is None:
            raise RuntimeError(
                f"Pipeline.from_pretrained returned None for {args.pipeline}. "
                "Accept the model user conditions on huggingface.co and check the token."
            )
        if device == "cuda":
            pipeline.to(torch.device("cuda"))

        call_kwargs: dict[str, int] = {}
        if args.num_speakers is not None:
            call_kwargs["num_speakers"] = args.num_speakers
        if args.min_speakers is not None:
            call_kwargs["min_speakers"] = args.min_speakers
        if args.max_speakers is not None:
            call_kwargs["max_speakers"] = args.max_speakers

        output = pipeline(str(args.audio), **call_kwargs)
        inclusive = turns_of(output.speaker_diarization)
        exclusive = turns_of(getattr(output, "exclusive_speaker_diarization", None))

        document = {
            "schema_version": "1.0",
            "video_id": args.video_id,
            "pipeline_id": args.pipeline,
            "pipeline_revision": os.environ.get("PYANNOTE_PIPELINE_REVISION") or None,
            "device": device,
            "requested_device": args.device,
            "inference_options": call_kwargs,
            "turns": inclusive,
            "exclusive_turns": exclusive,
            "speakers": sorted({turn[2] for turn in inclusive}),
            "rttm": rttm_of(output.speaker_diarization, file_id=args.video_id or "unknown"),
            "duration_seconds": None,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")

        if args.exclusive_output is not None:
            args.exclusive_output.parent.mkdir(parents=True, exist_ok=True)
            args.exclusive_output.write_text(
                json.dumps({**document, "turns": exclusive or [], "diarization_type": "exclusive"},
                           indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        if args.rttm_output is not None and document["rttm"] is not None:
            # A diarization that found no speech still has an RTTM: zero lines. It is
            # written even when empty because the stage declares it as an output, and
            # `validate` is right to expect every declared output to exist.
            args.rttm_output.write_text(str(document["rttm"]), encoding="utf-8")

        payload.update({
            "status": "ok",
            "tool_version": package_version("pyannote.audio"),
            "model_version": args.pipeline,
            "turns": len(inclusive),
            "exclusive_turns": len(exclusive or []),
            "speakers": len(document["speakers"]),
            "duration_seconds": round(time.time() - started, 3),
        })
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 0
    except Exception as exc:  # noqa: BLE001 - orchestrator reads the result file
        payload.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"[:2000],
                        "traceback": traceback.format_exc()[-6000:],
                        "duration_seconds": round(time.time() - started, 3)})
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"diarization worker failed: {exc}", file=sys.stderr)
        return 1


def turns_of(diarization) -> list[list[object]] | None:
    """``Annotation``/``Timeline`` → ``[[start, end, speaker], ...]``."""
    if diarization is None:
        return None
    turns: list[list[object]] = []
    for segment, _track, speaker in diarization.itertracks(yield_label=True):
        turns.append([round(float(segment.start), 6), round(float(segment.end), 6), str(speaker)])
    turns.sort(key=lambda item: (item[0], item[1], item[2]))
    return turns


def rttm_of(diarization, *, file_id: str = "unknown") -> str | None:
    """Standard RTTM so the raw result stays interoperable with other tooling."""
    if diarization is None:
        return None
    lines = []
    for segment, _track, speaker in diarization.itertracks(yield_label=True):
        lines.append(
            f"SPEAKER {file_id} 1 {segment.start:.3f} {segment.duration:.3f} "
            f"<NA> <NA> {speaker} <NA> <NA>"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def package_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


if __name__ == "__main__":
    sys.exit(main())
