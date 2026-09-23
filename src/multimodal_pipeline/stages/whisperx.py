"""WhisperX stage: transcription, language detection and word alignment.

The heavy work runs in the isolated uv environment's worker. This stage owns:

* building the worker *request* (everything that can change the output) and
  hashing it;
* preserving the native result verbatim under ``speech/raw/whisperx.json``,
  stamped with the request that produced it;
* normalising that raw file into ``speech/segments.parquet`` / ``words.parquet``.

Because the raw file carries its request hash, changing a normalisation-only
detail re-uses the expensive model output and redoes just the Parquet tables.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa

from ..exceptions import StageError, ValidationError
from ..normalization import whisperx_segment_rows, whisperx_word_rows
from ..schemas import SEGMENTS_SCHEMA, WORDS_SCHEMA, read_table, write_table
from .base import StageContext, WorkerStage, raw_request_matches


class WhisperXStage(WorkerStage):
    name = "whisperx"
    raw_artifact = "whisperx_raw"
    inputs = ("audio",)
    outputs = ("whisperx_raw", "speech_segments", "speech_words")
    config_keys = ("whisperx",)

    # ------------------------------------------------------------------ request

    def request(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.config.whisperx
        return {
            "stage": self.name,
            "model": cfg.model,
            "language": cfg.language,
            "device": cfg.device,
            "device_index": cfg.device_index,
            "compute_type": cfg.compute_type,
            "batch_size": cfg.batch_size,
            "beam_size": cfg.beam_size,
            "align_model": cfg.align_model,
            "vad_method": cfg.vad_method,
            "vad_merge_chunk_seconds": cfg.vad_merge_chunk_seconds,
            "threads": cfg.threads,
            "asr_options": cfg.asr_options,
            "extra_args": cfg.extra_args,
            "uv_project": str(ctx.config.resolve(cfg.uv_project)),
            "worker": str(ctx.config.resolve(cfg.worker)),
            "audio_sha256": self._audio_digest(ctx),
        }

    def _audio_digest(self, ctx: StageContext) -> str | None:
        from ..stages.metadata import sha256_of

        try:
            path = ctx.input("audio")
        except StageError:
            return None
        cached = ctx.scratch.get("audio_sha256")
        if cached:
            return cached
        digest = sha256_of(path)
        ctx.scratch["audio_sha256"] = digest
        return digest

    # ------------------------------------------------------------------ worker

    def enabled(self, ctx: StageContext) -> tuple[bool, str]:
        if not ctx.config.whisperx.enabled:
            return False, "whisperx.enabled = false"
        return True, ""

    def uv_project(self, ctx: StageContext) -> Path:
        return ctx.config.resolve(ctx.config.whisperx.uv_project)

    def worker_script(self, ctx: StageContext) -> Path:
        return ctx.config.resolve(ctx.config.whisperx.worker)

    def python_version(self, ctx: StageContext) -> str | None:
        return ctx.config.whisperx.python_version

    def worker_environment(self, ctx: StageContext) -> dict[str, str]:
        cfg = ctx.config.whisperx
        return {"CUDA_VISIBLE_DEVICES": str(cfg.device_index)} if cfg.device == "cuda" else {}

    def worker_argv(self, ctx: StageContext, raw_path: Path, request_digest: str) -> list[str]:
        cfg = ctx.config.whisperx
        args = [
            "--audio", str(ctx.input("audio")),
            "--output", str(raw_path),
            "--video-id", ctx.video_id,
            "--model", cfg.model,
            "--language", cfg.language,
            "--device", cfg.device,
            "--device-index", str(cfg.device_index),
            "--compute-type", cfg.compute_type,
            "--beam-size", str(cfg.beam_size),
            "--vad-method", cfg.vad_method,
            "--vad-merge-chunk-seconds", str(cfg.vad_merge_chunk_seconds),
            "--threads", str(cfg.threads),
            "--asr-options", json.dumps(cfg.asr_options),
            "--batch-size", str(cfg.batch_size),
            "--request-hash", request_digest,
        ]
        if cfg.download_root:
            args += ["--download-root", str(cfg.download_root)]
        if cfg.align_model:
            args += ["--align-model", cfg.align_model]
        return args + list(cfg.extra_args)

    # ------------------------------------------------------------ normalisation

    def normalize(self, ctx: StageContext) -> dict[str, Any]:
        raw_path = ctx.artifact(self.raw_artifact)
        payload = self.validate_raw(ctx)
        segments = whisperx_segment_rows(payload, ctx.video_id)
        words = whisperx_word_rows(payload, ctx.video_id)
        language = payload.get("language")
        write_table(
            ctx.artifact("speech_segments"),
            pa.Table.from_pylist(segments, schema=SEGMENTS_SCHEMA),
            SEGMENTS_SCHEMA,
            extra_metadata={"language": str(language), "source": "whisperx", "video_id": ctx.video_id},
        )
        write_table(
            ctx.artifact("speech_words"),
            pa.Table.from_pylist(words, schema=WORDS_SCHEMA),
            WORDS_SCHEMA,
            extra_metadata={"language": str(language), "source": "whisperx", "video_id": ctx.video_id},
        )
        aligned = sum(1 for word in words if word["alignment_status"] == "aligned")
        coverage = round(aligned / len(words), 4) if words else 0.0
        ctx.scratch["whisperx"] = {"language": language, "segments": len(segments),
                                   "words": len(words), "alignment_coverage": coverage}
        ctx.log(f"normalised {len(segments)} segments / {len(words)} words (word alignment {coverage:.1%})")
        return {"detected_language": language, "segments": len(segments),
                "words": len(words), "alignment_coverage": coverage}

    # ---------------------------------------------------------------- validation

    def validate(self, ctx: StageContext) -> dict[str, Any]:
        payload = self.validate_raw(ctx)
        if not payload.get("language"):
            raise ValidationError(self.name, ["raw result has no detected language"])
        if not raw_request_matches(ctx.artifact(self.raw_artifact), self.request_digest(ctx)):
            raise ValidationError(self.name, ["raw result was produced by a different configuration"])

        duration = source_duration(ctx)
        for name, schema in (("speech_segments", SEGMENTS_SCHEMA), ("speech_words", WORDS_SCHEMA)):
            path = ctx.artifact(name)
            if not path.is_file():
                raise ValidationError(self.name, [f"missing table: {path.name}"])
            from ..schemas import table_columns

            columns = set(table_columns(path))
            missing = [field.name for field in schema if field.name not in columns]
            if missing:
                raise ValidationError(self.name, [f"{path.name} missing columns: {', '.join(missing)}"])

        segments = read_table(ctx.artifact("speech_segments")).to_pylist()
        words = read_table(ctx.artifact("speech_words")).to_pylist()
        check = ctx.tools.get("check_intervals")
        from ..validation import check_intervals

        check_intervals([row["start_time"] for row in segments], [row["end_time"] for row in segments],
                        stage=self.name, label="segment", max_time=duration)
        starts = [row["start_time"] for row in segments]
        for index in range(1, len(starts)):
            if starts[index] is not None and starts[index - 1] is not None and starts[index] < starts[index - 1]:
                raise ValidationError(self.name, [f"segments out of order at row {index}"])
        check_intervals([row["start_time"] for row in words], [row["end_time"] for row in words],
                        stage=self.name, label="word", max_time=duration)
        known_segments = {row["segment_id"] for row in segments}
        orphans = {row["segment_id"] for row in words} - known_segments
        if orphans:
            raise ValidationError(self.name, [f"words reference unknown segments: {sorted(orphans)[:5]}"])
        aligned = sum(1 for row in words if row["alignment_status"] == "aligned")
        return {
            "language": payload.get("language"),
            "segments": len(segments),
            "words": len(words),
            "aligned_words": aligned,
            "alignment_coverage": round(aligned / len(words), 4) if words else None,
        }


def source_duration(ctx: StageContext) -> float | None:
    path = ctx.artifact("metadata")
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("duration_seconds")
    except (OSError, json.JSONDecodeError):
        return None
