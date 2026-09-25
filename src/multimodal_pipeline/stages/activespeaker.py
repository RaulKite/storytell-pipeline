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

import logging
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
        # Two things the worker knows and the tables cannot show. Both are recorded in
        # the raw JSON, but nobody reads that during a batch, and each one changes how
        # the output should be interpreted.
        fallback_reason = document.get("device_fallback_reason")
        if fallback_reason:
            ctx.log(
                f"TalkNet did not use the requested device "
                f"'{document.get('requested_device')}': {fallback_reason}",
                logging.WARNING,
            )
        unscored = sum(1 for row in frame_rows
                       if row["face_status"] == "tracked_unscored")
        if unscored:
            # The bounded imputation rule lives in the worker; the stage only reports
            # what its own table shows, so this wording never duplicates their budget.
            ctx.log(
                f"{unscored} frame(s) have a tracked face with no usable TalkNet score "
                f"(past the imputable tail, or a non-finite score); their scores are "
                f"null and they are never marked active",
                logging.WARNING,
            )
        ctx.log(f"normalised {len(frame_rows)} frames, {summary['frames_with_face']} with a "
                f"face, {summary['active_frames']} speaking, {len(track_rows)} track(s)")
        return summary

    @staticmethod
    def _frame_row(video_id: str, row: dict[str, Any]) -> dict[str, Any]:
        """One raw frame to a schema-shaped row, keeping absence explicit.

        A raw artifact written before `face_status` existed carries no key. Deriving it
        from what is present (a track id with a score is tracked; a track id without one
        is tracked_unscored) keeps old datasets normalising instead of failing, which is
        what a schema addition in a resumable pipeline owes them.
        """
        bbox = [row.get(key) for key in ("x1", "y1", "x2", "y2")]
        present = row.get("track_id") is not None and all(v is not None for v in bbox)
        status = row.get("face_status")
        if row.get("track_id") is None:
            status = "no_face"
        elif status not in ("no_face", "tracked", "tracked_unscored"):
            # A raw artifact written before face_status existed carries no key; derive
            # it so old datasets normalise instead of failing, which is what a schema
            # addition in a resumable pipeline owes them.
            status = "tracked" if row.get("talknet_score") is not None else "tracked_unscored"
        return {
            "schema_version": "1.1",
            "video_id": video_id,
            "frame_number": row.get("frame_25fps"),
            "timestamp": row.get("timestamp_sec"),
            "source_timestamp": row.get("source_timestamp_sec"),
            "scene_id": row.get("scene_id"),
            # track_id is passed through even when the bbox is incomplete: a corrupt
            # raw row must still name its track in validation instead of being
            # normalised into "no face" and hiding the corruption.
            "track_id": row.get("track_id"),
            "face_status": status,
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
        break_kind = dense_sequence_break(indices)
        if break_kind is not None:
            # Three different faults used to share one string. They have different causes
            # and different fixes, so the message names the fault and the row, and the
            # same line goes to the stage log where an operator revalidating a stale
            # dataset will actually look for it (review finding R3-001).
            message = (f"frame_number is not a dense 0..N-1 sequence: {break_kind}")
            ctx.log(f"{ctx.artifact('active_speaker_frames').name}: {message}",
                    logging.WARNING)
            problems.append(message)

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
                if row["face_status"] != "no_face":
                    problems.append(f"frame {index} is {row['face_status']} with no track")
                continue
            if row["face_status"] == "tracked_unscored":
                # The honest-empty state: a face was located and could not be scored.
                # Anything else in that row means the worker and the schema disagree
                # about what "no evidence" looks like.
                if row["talknet_score"] is not None or row["talknet_score_raw"] is not None:
                    problems.append(f"frame {index} is tracked_unscored but carries a score")
                if row["score_imputed"] or row["is_active_speaker"]:
                    problems.append(f"frame {index} is tracked_unscored but is imputed "
                                    f"or marked active")
            elif row["talknet_score"] is None or not math.isfinite(row["talknet_score"]):
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


def dense_sequence_break(indices: list[Any]) -> str | None:
    """Describe why ``indices`` is not ``0..N-1``, or None when it is.

    The stage's contract is exactly one row per output frame in order, so a violation is
    always fatal. It used to be fatal *and* silent about its shape: a reordered table, a
    gap, and a duplicated frame all produced "not a dense 0..N-1 sequence". They have
    different causes -- a reordering says the writer sorted by something else, a gap says
    rows were dropped, a duplicate says a frame was emitted twice -- and the operator
    revalidating a stale dataset needs one of those three, not the umbrella.
    """
    if indices == list(range(len(indices))):
        return None
    for position, value in enumerate(indices):
        if not isinstance(value, int) or isinstance(value, bool):
            return f"row {position} has no frame_number"
    duplicates = sorted({value for value in indices if indices.count(value) > 1})
    if duplicates:
        shown = ", ".join(str(value) for value in duplicates[:5])
        return (f"frame_number {shown} is a duplicate, appearing more than once "
                f"({len(indices)} rows, {len(set(indices))} distinct)")
    first_bad = next(i for i, value in enumerate(indices) if value != i)
    if sorted(indices) == list(range(len(indices))):
        return (f"row {first_bad} holds frame_number {indices[first_bad]} instead, "
                f"the rows are out of order")
    expected = set(range(len(indices)))
    missing = sorted(expected - set(indices))
    extra = sorted(set(indices) - expected)
    parts = []
    if missing:
        parts.append(f"{len(missing)} frame number(s) missing (first: {missing[0]}"
                     + (f", last: {missing[-1]})" if len(missing) > 1 else ")"))
    if extra:
        parts.append(f"frame number(s) outside 0..{len(indices) - 1} "
                     f"(first: {extra[0]})")
    return ", ".join(parts) or f"row {first_bad} holds {indices[first_bad]}"
