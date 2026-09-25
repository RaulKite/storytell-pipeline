"""NVIDIA Nemotron 3 Diarization stage — a second engine, next to pyannote.

This stage exists so two diarizers can be compared on the same corpus. It is a parallel
opinion, not a replacement:

* it reads ``audio/audio.wav`` and depends on the ``audio`` stage only — never on
  ``diarization``, so either engine can be switched off and the other still runs;
* it writes its own raw JSON and its own Parquet table;
* ``speaker_assignment`` does not read it. Choosing an engine is a decision the operator
  makes after comparing the two tables, and wiring that choice into the DAG now would
  delete the alternative this stage exists to preserve.

Why the HuggingFace route and not the ``nemo-toolkit`` route the NVIDIA blog shows: NeMo
3.0.0 cannot load this checkpoint at all (its encoder raises
``self_attention_model='rope' is not supported``), and with its default dependency resolution
it also selects a torch build the local CUDA driver rejects. The probe table is recorded in
``environments/diarization_nemotron/pyproject.toml``.

The Hugging Face token is forwarded through the process environment only, never argv. The
model is not gated, so the token is a rate-limit convenience here rather than a
prerequisite — which is why a missing token disables no stage.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa

from ..exceptions import StageError, ValidationError
from ..normalization import nemotron_turn_rows
from ..schemas import SPEAKER_TURNS_NEMOTRON_SCHEMA, read_table, write_table
from .base import StageContext, WorkerStage, raw_request_matches


class DiarizationNemotronStage(WorkerStage):
    name = "diarization_nemotron"
    raw_artifact = "nemotron_diarization_raw"
    inputs = ("audio",)
    outputs = ("nemotron_diarization_raw", "speaker_turns_nemotron")
    config_keys = ("diarization_nemotron",)

    # ------------------------------------------------------------------ request

    def request(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.config.diarization_nemotron
        return {
            "stage": self.name,
            "model": cfg.model,
            "device": cfg.device,
            "device_index": cfg.device_index,
            "max_speakers": cfg.max_speakers,
            "threshold": cfg.threshold,
            "fallback_to_cpu": cfg.fallback_to_cpu,
            "extra_args": cfg.extra_args,
            "uv_project": str(ctx.config.resolve(cfg.uv_project)),
            "worker": str(ctx.config.resolve(cfg.worker)),
            # The environment, not just its path. This stage's transformers pin is an
            # unreleased git checkout, so two machines with the same path and different
            # content must not claim each other's cached raw output. Resolving the actual
            # commit is left to the install; recording the project directory plus the
            # worker source digest (added by digest_payload) is what the fingerprint can
            # honestly assert today.
            "uv_project_present": self._project_present(ctx),
            "audio_sha256": self._audio_digest(ctx),
        }

    def _project_present(self, ctx: StageContext) -> bool:
        return self.uv_project(ctx).is_dir()

    @staticmethod
    def _audio_digest(ctx: StageContext) -> str | None:
        from ..stages.metadata import sha256_of

        try:
            path = ctx.input("audio")
        except StageError:
            return None
        # Distinct scratch key from DiarizationStage: both stages read the same file, but
        # sharing one cache entry would make the second stage's digest depend on which
        # stage happened to run first in this process.
        cached = ctx.scratch.get("audio_sha256_nemotron")
        if cached:
            return cached
        digest = sha256_of(path)
        ctx.scratch["audio_sha256_nemotron"] = digest
        return digest

    # ------------------------------------------------------------------ enablement

    def enabled(self, ctx: StageContext) -> tuple[bool, str]:
        """Skip, never fail, when the optional engine is not available.

        Three states are distinguished because they need different answers from an
        operator: switched off, environment not installed, and environment installed but
        broken. The last one still runs and fails loudly, because at that point silence
        would hide a real defect.
        """
        cfg = ctx.config.diarization_nemotron
        if not cfg.enabled:
            return False, "diarization_nemotron.enabled = false"
        project = self.uv_project(ctx)
        if not project.is_dir():
            return False, (
                f"nemotron environment not installed at {project} — create it and run "
                "`uv sync --python 3.12` there to enable the second diarizer"
            )
        worker = self.worker_script(ctx)
        if not worker.is_file():
            return False, f"nemotron worker script missing at {worker}"
        return True, ""

    # ------------------------------------------------------------------ worker

    def uv_project(self, ctx: StageContext) -> Path:
        return ctx.config.resolve(ctx.config.diarization_nemotron.uv_project)

    def worker_script(self, ctx: StageContext) -> Path:
        return ctx.config.resolve(ctx.config.diarization_nemotron.worker)

    def python_version(self, ctx: StageContext) -> str | None:
        return ctx.config.diarization_nemotron.python_version

    def worker_timeout(self, ctx: StageContext) -> float | None:
        return ctx.config.diarization_nemotron.timeout_seconds

    def worker_environment(self, ctx: StageContext) -> dict[str, str]:
        cfg = ctx.config.diarization_nemotron
        env: dict[str, str] = {}
        # Optional: the model is not gated, so a missing token must not disable the stage.
        token = os.environ.get(cfg.hf_token_env)
        if token:
            env["HF_TOKEN"] = token
            env["HUGGING_FACE_HUB_TOKEN"] = token
        if cfg.device == "cuda":
            env["CUDA_VISIBLE_DEVICES"] = str(cfg.device_index)
        return env

    def worker_argv(self, ctx: StageContext, raw_path: Path, request_digest: str) -> list[str]:
        cfg = ctx.config.diarization_nemotron
        args = [
            "--audio", str(ctx.input("audio")),
            "--output", str(raw_path),
            "--video-id", ctx.video_id,
            "--model", cfg.model,
            "--device", cfg.device,
            "--max-speakers", str(cfg.max_speakers),
            "--threshold", str(cfg.threshold),
            "--cpu-fallback" if cfg.fallback_to_cpu else "--no-cpu-fallback",
            "--request-hash", request_digest,
        ]
        return args + list(cfg.extra_args)

    # ------------------------------------------------------------ normalisation

    def normalize(self, ctx: StageContext) -> dict[str, Any]:
        payload = self.validate_raw(ctx)
        rows = nemotron_turn_rows(payload, ctx.video_id)
        write_table(
            ctx.artifact("speaker_turns_nemotron"),
            pa.Table.from_pylist(rows, schema=SPEAKER_TURNS_NEMOTRON_SCHEMA),
            SPEAKER_TURNS_NEMOTRON_SCHEMA,
            extra_metadata={"model": str(ctx.config.diarization_nemotron.model),
                            "video_id": ctx.video_id},
        )
        speakers = sorted({row["speaker_id"] for row in rows})
        overlapping = sum(1 for row in rows if row["overlap_s"] > 0)
        ctx.scratch["diarization_nemotron"] = {"turns": len(rows), "speakers": speakers}
        ctx.log(f"normalised {len(rows)} nemotron turns across {len(speakers)} speaker(s), "
                f"{overlapping} overlapping")
        return {"turns": len(rows), "speakers": len(speakers), "overlapping_turns": overlapping,
                "diarization_type": rows[0]["diarization_type"] if rows else None}

    # ---------------------------------------------------------------- validation

    def validate(self, ctx: StageContext) -> dict[str, Any]:
        payload = self.validate_raw(ctx)
        duration = self._duration(ctx)
        path = ctx.artifact("speaker_turns_nemotron")
        if not path.is_file():
            raise ValidationError(self.name, ["speaker_turns_nemotron.parquet missing"])
        turns = read_table(path).to_pylist()
        from ..validation import check_intervals

        check_intervals([row["start_time"] for row in turns], [row["end_time"] for row in turns],
                        stage=self.name, label="turn", max_time=duration)
        speakers = {row["speaker_id"] for row in turns if row["speaker_id"]}
        if turns and not speakers:
            raise ValidationError(self.name, ["nemotron turns carry no speaker identifiers"])
        # The engine must not be able to quietly become the other one. A raw file from a
        # different model id means this table is not what the config asked for.
        raw_model = payload.get("model_id")
        if raw_model and raw_model != ctx.config.diarization_nemotron.model:
            raise ValidationError(
                self.name,
                [f"raw result came from {raw_model}, config asks for "
                 f"{ctx.config.diarization_nemotron.model}"],
            )
        # Overlap is the reason this table exists. A normalisation bug that flattened it
        # would leave a table that still validates as intervals and still has speakers, so
        # the invariant is asserted here rather than trusted to the writer.
        for row in turns:
            if float(row["duration"]) <= 0:
                raise ValidationError(self.name, [f"{row['turn_id']} has non-positive duration"])
            if float(row["overlap_s"]) < 0:
                raise ValidationError(self.name, [f"{row['turn_id']} has negative overlap"])
        max_speakers = ctx.config.diarization_nemotron.max_speakers
        if len(speakers) > max_speakers:
            raise ValidationError(
                self.name,
                [f"{len(speakers)} speakers in the table, config allows at most {max_speakers}"],
            )
        if not raw_request_matches(ctx.artifact(self.raw_artifact), self.request_digest(ctx)):
            raise ValidationError(self.name, ["raw result was produced by a different configuration"])
        return {"turns": len(turns), "speakers": len(speakers),
                "overlapping_turns": sum(1 for row in turns if float(row["overlap_s"]) > 0)}

    def _duration(self, ctx: StageContext) -> float | None:
        path = ctx.artifact("metadata")
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("duration_seconds")
        except (OSError, json.JSONDecodeError):
            return None
