"""TalkNet-ASD active-speaker detection stage.

Complements the two audio-only speaker signals: pyannote says *when* somebody is
speaking and OpenPose says *where bodies are*; neither says *which visible face is
producing the audio*. TalkNet answers that, scoring every face track per frame.

Facts verified on this machine (2026-09-24) rather than assumed:

* TalkNet's entire frame axis is **constant-rate 25 FPS**, so its frame numbers are
  only meaningful against the converted video; every output row therefore carries
  both the 25 FPS timestamp and the nearest original-video timestamp;
* the score array is **shorter than the visual track** (a 105-frame track yielded
  104 scores, MFCC windowing trimming the tail), which the worker imputes and
  flags per row rather than dropping silently;
* the upstream checkpoints are fetched with gdown into paths resolved against
  ``os.getcwd()``, which is why the worker runs with ``cwd=talknet_root`` and why a
  read-only checkout needs ``activespeaker.weights_dir``;
* ``torch`` must stay at 2.5.x: ``talkNet.py`` and the S3FD face detector call
  ``torch.load()`` without ``weights_only=``, whose default became ``True`` in
  torch 2.6 and rejects the 2021 checkpoints.

Deliberate design choice: the frames table is **dense** — one row per output frame,
including frames with no face. This stage exists to remove exactly that ambiguity,
and a sparse table would force every consumer to reconstruct the timeline.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pyarrow as pa

from ..exceptions import ValidationError
from ..schemas import (
    ACTIVE_SPEAKER_FRAMES_SCHEMA,
    ACTIVE_SPEAKER_TRACKS_SCHEMA,
    read_table,
    write_table,
)
from ..validation import check_intervals
from .base import StageContext, WorkerStage, raw_request_matches

OUTPUT_FPS = 25


class ActiveSpeakerStage(WorkerStage):
    """Run TalkNet in its own uv environment and normalise it onto the timeline."""

    name = "activespeaker"
    inputs = ("metadata", "audio")
    outputs = (
        "activespeaker_raw",
        "activespeaker_tracks_pkl",
        "activespeaker_scores_pkl",
        "activespeaker_scenes_csv",
        "active_speaker_frames",
        "active_speaker_tracks",
    )
    config_keys = ("activespeaker",)
    raw_artifact = "activespeaker_raw"

    # ------------------------------------------------------------- fingerprint

    def request(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.config.activespeaker
        return {
            "stage": self.name,
            "provider": "talknet",
            "talknet_root": str(ctx.config.resolve(cfg.talknet_root)) if cfg.talknet_root else None,
            "weights_dir": str(ctx.config.resolve(cfg.weights_dir)) if cfg.weights_dir else None,
            "device": cfg.device,
            "device_index": cfg.device_index,
            "speaker_threshold": cfg.speaker_threshold,
            "score_window": cfg.score_window,
            "switch_margin": cfg.switch_margin,
            "switch_frames": cfg.switch_frames,
            "extra_args": cfg.extra_args,
            "uv_project": str(ctx.config.resolve(cfg.uv_project)),
            "worker": str(ctx.config.resolve(cfg.worker)),
            "source_sha256": self._source_digest(ctx),
        }

    @staticmethod
    def _source_digest(ctx: StageContext) -> str | None:
        from ..stages.metadata import sha256_of

        try:
            path = ctx.input("audio")
        except Exception:  # noqa: BLE001 - a missing upstream artifact is not fatal here
            return None
        cached = ctx.scratch.get("activespeaker_audio_sha256")
        if cached:
            return cached
        digest = sha256_of(path)
        ctx.scratch["activespeaker_audio_sha256"] = digest
        return digest

    # ---------------------------------------------------------------- gating

    def enabled(self, ctx: StageContext) -> tuple[bool, str]:
        cfg = ctx.config.activespeaker
        if not cfg.enabled:
            return False, "activespeaker.enabled = false"
        if cfg.talknet_root is None:
            return False, ("activespeaker.talknet_root is not set (point it at a "
                           "TalkNet-ASD checkout to enable active speaker detection)")
        if not Path(cfg.talknet_root).is_dir():
            return False, f"activespeaker.talknet_root does not exist: {cfg.talknet_root}"
        if not (Path(cfg.talknet_root) / "run_talknet.py").is_file():
            return False, (f"activespeaker.talknet_root has no run_talknet.py: "
                           f"{cfg.talknet_root} (is it a TalkNet-ASD checkout?)")
        return True, ""

    # --------------------------------------------------------------- worker

    def uv_project(self, ctx: StageContext) -> Path:
        return ctx.config.resolve(ctx.config.activespeaker.uv_project)

    def worker_script(self, ctx: StageContext) -> Path:
        return ctx.config.resolve(ctx.config.activespeaker.worker)

    def python_version(self, ctx: StageContext) -> str | None:
        return ctx.config.activespeaker.python_version

    def worker_timeout(self, ctx: StageContext) -> float | None:
        # TalkNet is fast (seconds on a 4090) but loads two checkpoints and can fall
        # back to CPU; the default None means "no limit", and the operator can cap it.
        return ctx.config.activespeaker.timeout_seconds

    def worker_environment(self, ctx: StageContext) -> dict[str, str]:
        cfg = ctx.config.activespeaker
        env: dict[str, str] = {}
        if cfg.device == "cuda":
            # Same convention as the diarization worker: one GPU per stage run.
            env["CUDA_VISIBLE_DEVICES"] = str(cfg.device_index)
        return env

    def prepare(self, ctx: StageContext) -> None:
        super().prepare(ctx)
        ctx.artifact("activespeaker_raw").parent.mkdir(parents=True, exist_ok=True)

    def worker_argv(self, ctx: StageContext, raw_path: Path, request_digest: str) -> list[str]:
        cfg = ctx.config.activespeaker
        from ..uv_worker import worker_result_path

        raw_dir = raw_path.parent
        args = [
            "--video", str(ctx.source.path),
            "--audio", str(ctx.input("audio")),
            "--talknet-root", str(ctx.config.resolve(cfg.talknet_root)),
            "--raw-dir", str(raw_dir),
            "--output-json", str(raw_path),
            "--result-path", str(worker_result_path(raw_dir, f"{self.name}_worker_result.json")),
            "--video-id", ctx.video_id,
            "--device", cfg.device,
            "--speaker-threshold", str(cfg.speaker_threshold),
            "--score-window", str(cfg.score_window),
            "--switch-margin", str(cfg.switch_margin),
            "--switch-frames", str(cfg.switch_frames),
        ]
        if cfg.weights_dir is not None:
            args += ["--weights-dir", str(ctx.config.resolve(cfg.weights_dir))]
        return args + list(cfg.extra_args)

    # ------------------------------------------------------------ normalisation

    def normalize(self, ctx: StageContext) -> dict[str, Any]:
        document = self.validate_raw(ctx)
        frames = document.get("frames")
        if not isinstance(frames, list):
            raise ValidationError(self.name, ["raw document has no frames list"])

        frame_rows = [self._frame_row(ctx.video_id, row) for row in frames]
        write_table(
            ctx.artifact("active_speaker_frames"),
            pa.Table.from_pylist(frame_rows, schema=ACTIVE_SPEAKER_FRAMES_SCHEMA),
            ACTIVE_SPEAKER_FRAMES_SCHEMA,
            extra_metadata={
                "video_id": ctx.video_id,
                "output_fps": document.get("output_fps"),
                "source_fps": document.get("source_fps"),
                "model": "TalkNet-ASD",
                "device": document.get("device"),
                "speaker_threshold": document.get("parameters", {}).get("speaker_threshold"),
            },
        )
        track_rows = track_summary_rows(ctx.video_id, frame_rows)
        write_table(
            ctx.artifact("active_speaker_tracks"),
            pa.Table.from_pylist(track_rows, schema=ACTIVE_SPEAKER_TRACKS_SCHEMA),
            ACTIVE_SPEAKER_TRACKS_SCHEMA,
            extra_metadata={"video_id": ctx.video_id, "model": "TalkNet-ASD"},
        )
        summary = {
            "frames": len(frame_rows),
            "frames_with_face": sum(1 for row in frame_rows if row["track_id"] is not None),
            "active_frames": sum(1 for row in frame_rows if row["is_active_speaker"]),
            "tracks": len(track_rows),
            "scenes": document.get("scene_count"),
        }
        ctx.scratch["activespeaker"] = summary
        ctx.log(f"normalised {len(frame_rows)} frames, {summary['frames_with_face']} with a "
                f"face, {summary['active_frames']} speaking, {len(track_rows)} track(s)")
        return summary

    @staticmethod
    def _frame_row(video_id: str, row: dict[str, Any]) -> dict[str, Any]:
        """One raw frame to a schema-shaped row, keeping absence explicit."""
        bbox = [row.get(key) for key in ("x1", "y1", "x2", "y2")]
        present = row.get("track_id") is not None and all(v is not None for v in bbox)
        return {
            "schema_version": "1.0",
            "video_id": video_id,
            "frame_number": row.get("frame_25fps"),
            "timestamp": row.get("timestamp_sec"),
            "source_timestamp": row.get("source_timestamp_sec"),
            "scene_id": row.get("scene_id"),
            "track_id": row.get("track_id"),
            "x1": bbox[0] if present else None,
            "y1": bbox[1] if present else None,
            "x2": bbox[2] if present else None,
            "y2": bbox[3] if present else None,
            "talknet_score_raw": row.get("talknet_score_raw"),
            "talknet_score": row.get("talknet_score"),
            "score_imputed": bool(row.get("score_imputed", False)),
            "is_active_speaker": bool(row.get("is_active_speaker", False)),
        }

    # ---------------------------------------------------------------- validation

    @staticmethod
    def _duration(ctx: StageContext) -> float | None:
        path = ctx.artifact("metadata")
        if not path.is_file():
            return None
        try:
            import json

            return json.loads(path.read_text(encoding="utf-8")).get("duration_seconds")
        except (OSError, ValueError):
            return None

    def validate(self, ctx: StageContext) -> dict[str, Any]:
        document = self.validate_raw(ctx)
        if not raw_request_matches(ctx.artifact(self.raw_artifact), self.request_digest(ctx)):
            raise ValidationError(
                self.name, ["raw result was produced by a different configuration"])

        path = ctx.artifact("active_speaker_frames")
        if not path.is_file():
            raise ValidationError(self.name, ["active_speaker_frames.parquet missing"])
        rows = read_table(path).to_pylist()
        if not rows:
            raise ValidationError(self.name, ["active_speaker_frames.parquet is empty"])

        problems: list[str] = []
        # The dense-timeline guarantee: exactly one row per frame, in order. A gap
        # would silently shift every downstream timestamp mapping.
        indices = [row["frame_number"] for row in rows]
        if indices != list(range(len(indices))):
            problems.append("frame_number is not a dense 0..N-1 sequence")

        timestamps = [row["timestamp"] for row in rows]
        # Each row is an instant rather than a span, so start == end is the honest
        # representation; check_intervals only rejects an interval that ends before
        # it starts, which for a point means a negative duration it cannot have.
        check_intervals(timestamps, timestamps, stage=self.name, label="frame timestamp",
                        max_time=self._duration(ctx))

        for index, row in enumerate(rows):
            if row["track_id"] is None:
                if row["is_active_speaker"]:
                    problems.append(f"frame {index} has no face but is marked active")
                continue
            if row["talknet_score"] is None or not math.isfinite(row["talknet_score"]):
                problems.append(f"frame {index} has a face with a non-finite score")
            coordinates = (row["x1"], row["y1"], row["x2"], row["y2"])
            if any(value is None or not math.isfinite(value) for value in coordinates):
                # Checked before any comparison: subtracting or comparing a missing
                # coordinate raises TypeError, which the orchestrator would record as
                # a stage crash instead of naming the offending frame.
                problems.append(f"frame {index} has a face with missing bbox coordinates")
                if len(problems) >= 8:
                    break
                continue
            if not (row["x1"] <= row["x2"] and row["y1"] <= row["y2"]):
                problems.append(f"frame {index} has an inverted bbox")
            if len(problems) >= 8:
                break

        if document.get("frame_count") is not None and document["frame_count"] != len(rows):
            problems.append(f"raw document declares {document['frame_count']} frames, "
                            f"table has {len(rows)}")
        if problems:
            raise ValidationError(self.name, problems)

        return {
            "frames": len(rows),
            "frames_with_face": sum(1 for row in rows if row["track_id"] is not None),
            "active_frames": sum(1 for row in rows if row["is_active_speaker"]),
            "tracks": len(read_table(ctx.artifact("active_speaker_tracks")).to_pylist()),
        }


def track_summary_rows(video_id: str, frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse the dense frames table into one row per TalkNet track.

    Deriving this from the frames table rather than from the pickles keeps the two
    tables consistent by construction: whatever survived stabilization is what gets
    summarised.
    """
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in frame_rows:
        if row["track_id"] is not None:
            grouped.setdefault(row["track_id"], []).append(row)

    rows: list[dict[str, Any]] = []
    for track_id in sorted(grouped):
        frames = grouped[track_id]
        scores = [row["talknet_score"] for row in frames if row["talknet_score"] is not None]
        active = [row for row in frames if row["is_active_speaker"]]
        areas = [
            (row["x2"] - row["x1"]) * (row["y2"] - row["y1"])
            for row in frames
            if all(row[key] is not None for key in ("x1", "y1", "x2", "y2"))
        ]
        rows.append({
            "schema_version": "1.0",
            "video_id": video_id,
            "track_id": track_id,
            "first_timestamp": min(row["timestamp"] for row in frames),
            "last_timestamp": max(row["timestamp"] for row in frames),
            "frame_count": len(frames),
            "active_frame_count": len(active),
            "active_ratio": round(len(active) / len(frames), 4),
            "mean_score": round(sum(scores) / len(scores), 4) if scores else None,
            "max_score": round(max(scores), 4) if scores else None,
            "scenes": sorted({row["scene_id"] for row in frames if row["scene_id"] is not None}),
            "mean_bbox_area": round(sum(areas) / len(areas), 2) if areas else None,
        })
    return rows
