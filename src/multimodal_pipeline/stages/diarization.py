"""Pyannote Community diarization stage.

Default pipeline is ``pyannote/speaker-diarization-community-1``, which returns
both an inclusive (overlapping speech preserved) and an *exclusive* timeline.
Both are preserved; the exclusive one is what speaker assignment prefers for
reconciling with ASR timestamps.

Authentication comes from the environment variable named by
``diarization.hf_token_env`` (``HF_TOKEN`` by default) and is forwarded to the
worker through the process environment only — never through argv, so it cannot
leak into ``status.json`` or logs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa

from ..artifacts import atomic_write_json
from ..exceptions import StageError, ValidationError
from ..normalization import diarization_turn_rows
from ..schemas import SPEAKER_TURNS_SCHEMA, read_table, write_table
from .base import StageContext, WorkerStage


class DiarizationStage(WorkerStage):
    name = "diarization"
    raw_artifact = "diarization_raw"
    inputs = ("audio",)
    outputs = ("diarization_raw", "exclusive_diarization_raw", "diarization_rttm", "speaker_turns")
    config_keys = ("diarization",)

    # ------------------------------------------------------------------ request

    def request(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.config.diarization
        return {
            "stage": self.name,
            "provider": cfg.provider,
            "pipeline": cfg.pipeline,
            "device": cfg.device,
            "device_index": cfg.device_index,
            "min_speakers": cfg.min_speakers,
            "max_speakers": cfg.max_speakers,
            "num_speakers": cfg.num_speakers,
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
        cfg = ctx.config.diarization
        if not cfg.enabled:
            return False, "diarization.enabled = false"
        if not os.environ.get(cfg.hf_token_env):
            return False, f"missing credential {cfg.hf_token_env} (export it to enable diarization)"
        return True, ""

    def uv_project(self, ctx: StageContext) -> Path:
        return ctx.config.resolve(ctx.config.diarization.uv_project)

    def worker_script(self, ctx: StageContext) -> Path:
        return ctx.config.resolve(ctx.config.diarization.worker)

    def python_version(self, ctx: StageContext) -> str | None:
        return ctx.config.diarization.python_version

    def worker_environment(self, ctx: StageContext) -> dict[str, str]:
        cfg = ctx.config.diarization
        env: dict[str, str] = {}
        token = os.environ.get(cfg.hf_token_env)
        if token:
            env["HF_TOKEN"] = token
            env["HUGGING_FACE_HUB_TOKEN"] = token
        if cfg.device == "cuda":
            env["CUDA_VISIBLE_DEVICES"] = str(cfg.device_index)
        return env

    def worker_argv(self, ctx: StageContext, raw_path: Path, request_digest: str) -> list[str]:
        cfg = ctx.config.diarization
        exclusive = ctx.paths.artifact("exclusive_diarization_raw")
        rttm = ctx.paths.artifact("diarization_rttm")
        args = [
            "--audio", str(ctx.input("audio")),
            "--output", str(raw_path),
            "--exclusive-output", str(exclusive),
            "--rttm-output", str(rttm),
            "--video-id", ctx.video_id,
            "--pipeline", cfg.pipeline,
            "--device", cfg.device,
            "--device-index", str(cfg.device_index),
            "--request-hash", request_digest,
        ]
        if cfg.min_speakers is not None:
            args += ["--min-speakers", str(cfg.min_speakers)]
        if cfg.max_speakers is not None:
            args += ["--max-speakers", str(cfg.max_speakers)]
        if cfg.num_speakers is not None:
            args += ["--num-speakers", str(cfg.num_speakers)]
        return args + list(cfg.extra_args)

    def run_model(self, ctx, request, raw_path, digest) -> None:  # noqa: ANN001
        """Preserve the exclusive timeline too before stamping provenance."""
        super().run_model(ctx, request, raw_path, digest)
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        exclusive = payload.get("exclusive_turns")
        if exclusive is not None:
            exclusive_path = ctx.paths.artifact("exclusive_diarization_raw")
            atomic_write_json(exclusive_path, {
                "schema_version": "1.0",
                "video_id": ctx.video_id,
                "pipeline": request["pipeline"],
                "turns": exclusive,
                "_pipeline_request": (payload.get("_pipeline_request") or {}),
            })
        rttm = payload.get("rttm")
        if rttm:
            ctx.paths.artifact("diarization_rttm").write_text(str(rttm), encoding="utf-8")

    # ------------------------------------------------------------ normalisation

    def normalize(self, ctx: StageContext) -> dict[str, Any]:
        payload = self.validate_raw(ctx)
        cfg = ctx.config.diarization
        exclusive = payload.get("exclusive_turns") or []
        inclusive = payload.get("turns") or []
        # Exclusive diarization gives each instant exactly one speaker, which is
        # the behaviour we want when labelling transcript intervals.
        if cfg.use_exclusive_diarization_for_alignment and exclusive:
            rows = diarization_turn_rows({"turns": exclusive}, ctx.video_id, "exclusive")
        else:
            rows = diarization_turn_rows({"turns": inclusive}, ctx.video_id, "inclusive")
        write_table(
            ctx.artifact("speaker_turns"),
            pa.Table.from_pylist(rows, schema=SPEAKER_TURNS_SCHEMA),
            SPEAKER_TURNS_SCHEMA,
            extra_metadata={"pipeline": str(cfg.pipeline), "video_id": ctx.video_id},
        )
        speakers = sorted({row["speaker_id"] for row in rows})
        ctx.scratch["diarization"] = {"turns": len(rows), "speakers": speakers}
        ctx.log(f"normalised {len(rows)} speaker turns across {len(speakers)} speaker(s)")
        return {"turns": len(rows), "speakers": len(speakers),
                "diarization_type": rows[0]["diarization_type"] if rows else None}

    # ---------------------------------------------------------------- validation

    def validate(self, ctx: StageContext) -> dict[str, Any]:
        payload = self.validate_raw(ctx)
        duration = self._duration(ctx)
        path = ctx.artifact("speaker_turns")
        if not path.is_file():
            raise ValidationError(self.name, ["speaker_turns.parquet missing"])
        turns = read_table(path).to_pylist()
        from ..validation import check_intervals

        check_intervals([row["start_time"] for row in turns], [row["end_time"] for row in turns],
                        stage=self.name, label="turn", max_time=duration)
        speakers = {row["speaker_id"] for row in turns if row["speaker_id"]}
        if turns and not speakers:
            raise ValidationError(self.name, ["speaker turns carry no speaker identifiers"])
        if payload.get("pipeline_id") and payload["pipeline_id"] != ctx.config.diarization.pipeline:
            raise ValidationError(
                self.name,
                [f"raw result came from {payload['pipeline_id']}, config asks for "
                 f"{ctx.config.diarization.pipeline}"],
            )
        request = payload.get("_pipeline_request") or {}
        from ..config import stable_hash

        if request.get("request_hash") != stable_hash(self.request(ctx), length=16):
            raise ValidationError(self.name, ["raw result was produced by a different configuration"])
        return {"turns": len(turns), "speakers": len(speakers)}

    def _duration(self, ctx: StageContext) -> float | None:
        path = ctx.artifact("metadata")
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("duration_seconds")
        except (OSError, json.JSONDecodeError):
            return None
