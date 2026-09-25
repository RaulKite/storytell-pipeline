#!/usr/bin/env python3
"""Nemotron 3 Diarization worker — runs inside ``environments/diarization_nemotron`` only.

A *second, independent* diarizer that runs next to pyannote so the operator can compare
two engines on the same corpus. Nothing here reads or rewrites a pyannote artifact.

Route chosen by measurement, not preference: ``nemo-toolkit`` could not load this
checkpoint at all (NeMo 3.0.0's encoder rejects ``self_attention_model='rope'``), so this
uses the HuggingFace implementation NVIDIA also ships. See
``environments/diarization_nemotron/pyproject.toml`` for the full probe table.

The Hugging Face token arrives through ``HF_TOKEN`` — never argv — so it cannot leak into
``status.json``, logs or provenance. The model is not gated, so the token is only a
rate-limit convenience here; it is still read the same way as every other stage.

API contract verified against this checkpoint on 2026-09-25:
``AutoProcessor.from_pretrained`` + ``AutoModelForAudioFrameClassification.from_pretrained``
→ ``processor(audio, sampling_rate)`` → ``model(**inputs).logits`` →
``processor.extract_speaker_dict(logits, attention_mask, threshold=...)`` giving
``{"Start": float, "End": float, "Speaker": int}`` per segment, sorted by start, speakers
numbered by first arrival. Segments from different channels MAY overlap — that is the
model's headline feature — so the overlap is preserved rather than flattened.

NO OPERATING-POINT FLAGS ON PURPOSE
The card and the NVIDIA blog expose five operating points (spkcache_len, fifo_len,
chunk_len, chunk_right_context, spkcache_update_period) and show them being assigned on
``model.sortformer_modules``. That object belongs to the NeMo path and does not exist here.
The HuggingFace path needs none of them for batch work: reading
``transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py`` shows
``forward()`` enters offline mode when neither ``speaker_cache`` nor
``num_lookahead_frames`` is passed, and splits the input by ``config.chunk_length`` with
``config.chunk_right_context`` look-ahead. The checkpoint's own resolved values —
``chunk_length=340``, ``chunk_right_context=40``, ``fifo_length=40``,
``speaker_cache_update_period=300``, ``streaming_config.speaker_cache_length=264`` — are
already the card's "Offline Style" row, the accuracy point the card recommends for
offline inference. So calling the model once with the whole clip *is* the offline operating
point, and a ``--operating-point`` flag here would be a control that controls nothing.
Streaming is deliberately not wired: batch diarization of a whole clip wants the accuracy
point, and a streaming session would need per-chunk cache plumbing whose benefit here is
nil. They would produce different segments, so if streaming is ever added it must be a
separate artifact, never a silent mode switch on this one.
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
    parser = argparse.ArgumentParser(description="Nemotron 3 Diarization worker")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--model", default="nvidia/Nemotron-3-Diarization")
    parser.add_argument("--device", default="cuda")
    #: Mirrors ``diarization_nemotron.fallback_to_cpu``. Without it the worker behaves like
    #: the pyannote worker (silently degrades cuda -> cpu); with ``--no-cpu-fallback`` a
    #: machine that cannot see a GPU fails loudly instead of producing a result that took
    #: ten times longer and nobody noticed.
    parser.add_argument("--cpu-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-speakers", type=int, default=8)
    # Two thresholds exist and they are not the same knob. ``--threshold`` is the frame
    # probability used to turn logits into segments, and defaults to the value documented
    # on ``extract_speaker_dict`` (0.5). The checkpoint's own
    # ``streaming_config.prediction_score_threshold`` is 0.25 and governs the streaming
    # cache, which this worker does not use. The resolved value is recorded in the raw
    # output so a reviewer can see both rather than guess which one produced a segment.
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--request-hash", default=None)
    parser.add_argument("--result-json", type=Path, default=None)
    return parser.parse_args(argv)


def resolve_device(*, requested: str, cpu_fallback: bool, cuda_available: bool) -> tuple[str, str | None]:
    """Decide the device to run on, or refuse to decide.

    A pure function so the policy is testable without torch, without a GPU and without
    mocking ``torch.cuda`` — a mock of the thing under test would make the fallback branch
    green whatever it did. The pyannote worker only ever degrades cuda→cpu, which is right
    there because diarization is mandatory; here the second engine is optional, so an
    operator may prefer a fast failure to an hour-long batch that quietly crawled on CPU.

    Returns ``(device, fallback_reason)``; reason is None unless a degradation happened.
    """
    if requested != "cuda" or cuda_available:
        return requested, None
    if not cpu_fallback:
        raise RuntimeError(
            "--device cuda was requested and torch.cuda.is_available() is False, and "
            "--no-cpu-fallback forbids degrading. Run on a GPU host, or set "
            "diarization_nemotron.fallback_to_cpu = true to accept the slowdown, or set "
            "diarization_nemotron.device = 'cpu' to say so explicitly."
        )
    return "cpu", "torch.cuda.is_available() was False"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    # The harness looks for ``{stage_name}_worker_result.json`` next to the raw artifact
    # (see uv_worker.worker_result_path and WorkerStage.run_model). The name is therefore
    # load-bearing, not cosmetic: a worker that names its own default differently fails with
    # "worker produced no result JSON" even though it diarized successfully. That mismatch is
    # what the end-to-end test caught here before this line was right.
    result_path = args.result_json or (args.output.parent / "diarization_nemotron_worker_result.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"stage": "diarization_nemotron", "status": "running",
                                  "request_hash": args.request_hash}
    try:
        if not args.audio.is_file():
            raise FileNotFoundError(f"audio file not found: {args.audio}")
        if not 0.0 < args.threshold < 1.0:
            raise ValueError(f"--threshold must be a probability in (0, 1), got {args.threshold}")

        # torch is imported inside the try because this module is also exercised from the
        # orchestrator's own environment in unit tests, which has no torch at all; a bad
        # argument must be reported as a bad argument rather than as a missing dependency.
        # torch is imported inside the try because this module is also exercised from the
        # orchestrator's own environment in unit tests, which has no torch at all; a bad
        # argument must be reported as a bad argument rather than as a missing dependency.
        import torch

        # Device gate BEFORE the heavy imports. transformers is an unreleased checkout and
        # the checkpoint is a hub download; discovering an impossible device after paying
        # for both wastes the run and buries the reason under a download error.
        device, fallback_reason = resolve_device(
            requested=args.device, cpu_fallback=args.cpu_fallback,
            cuda_available=bool(torch.cuda.is_available()),
        )
        if fallback_reason:
            payload["device_fallback_reason"] = fallback_reason

        from transformers import (
            AutoConfig,
            AutoModelForAudioFrameClassification,
            AutoProcessor,
        )
        from transformers.audio_utils import load_audio

        load_started = time.time()
        processor = AutoProcessor.from_pretrained(args.model)
        config = AutoConfig.from_pretrained(args.model)
        model = AutoModelForAudioFrameClassification.from_pretrained(
            args.model, device_map=device if device == "cuda" else None,
        )
        if device != "cuda":
            model = model.to(device)
        model.eval()
        load_seconds = round(time.time() - load_started, 3)

        sample_rate = int(processor.feature_extractor.sampling_rate)
        audio = load_audio(str(args.audio), sampling_rate=sample_rate)
        duration_seconds = round(float(audio.shape[0]) / sample_rate, 6)

        infer_started = time.time()
        inputs = processor(audio, sampling_rate=sample_rate)
        inputs = inputs.to(model.device, dtype=model.dtype)
        with torch.inference_mode():
            logits = model(**inputs).logits
        inference_seconds = round(time.time() - infer_started, 3)

        extracted = processor.extract_speaker_dict(
            logits, inputs.attention_mask, threshold=args.threshold
        )[0]
        channels = int(logits.shape[-1])
        dropped_channels: dict[str, int] = {}
        segments: list[dict[str, object]] = []
        for item in extracted:
            speaker = int(item["Speaker"])
            if speaker >= args.max_speakers:
                # A truncated speaker space must be visible in the raw output, not silently
                # shortened: a diarization that quietly lost its 9th speaker looks clean.
                key = f"speaker_{speaker}"
                dropped_channels[key] = dropped_channels.get(key, 0) + 1
                continue
            segments.append({
                "start": round(float(item["Start"]), 6),
                "end": round(float(item["End"]), 6),
                "speaker_index": speaker,
                "speaker_id": f"speaker_{speaker}",
            })
        segments.sort(key=lambda row: (row["start"], row["end"], row["speaker_id"]))

        stream_cfg = getattr(config, "streaming_config", None)
        document = {
            "schema_version": "1.0",
            "video_id": args.video_id,
            "model_id": args.model,
            "runtime": "transformers",
            "runtime_version": package_version("transformers"),
            "torch_version": torch.__version__,
            "device": device,
            "requested_device": args.device,
            "max_speakers": args.max_speakers,
            "threshold": args.threshold,
            "sample_rate": sample_rate,
            "duration_seconds": duration_seconds,
            "logits_shape": [int(dim) for dim in logits.shape],
            # Resolved offline geometry, recorded rather than assumed. These are what
            # forward() actually used; if a future checkpoint changes them, the diarization
            # changes with them and the raw output says so.
            "offline_geometry": {
                "chunk_length": int(getattr(config, "chunk_length", -1)),
                "chunk_right_context": int(getattr(config, "chunk_right_context", -1)),
                "fifo_length": int(getattr(config, "fifo_length", -1)),
                "speaker_cache_update_period": int(getattr(config, "speaker_cache_update_period", -1)),
                "speaker_cache_length": int(getattr(stream_cfg, "speaker_cache_length", -1)),
                "prediction_score_threshold": float(
                    getattr(stream_cfg, "prediction_score_threshold", -1.0)),
                "frame_stride_ms": float(processor.feature_extractor.hop_length) / sample_rate * 1000.0,
                "channels": channels,
            },
            "segments": segments,
            "speakers": sorted({str(row["speaker_id"]) for row in segments}),
            "dropped_over_max_speakers": dropped_channels,
            "load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")

        payload.update({
            "status": "ok",
            "tool_version": package_version("transformers"),
            "model_version": args.model,
            "segments": len(segments),
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
        print(f"nemotron diarization worker failed: {exc}", file=sys.stderr)
        return 1


def package_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


if __name__ == "__main__":
    sys.exit(main())
