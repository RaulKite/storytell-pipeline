"""OpenPose BODY_25 + hands + face, and chunked normalisation of its output.

Facts verified on this machine (2026-09-23) rather than assumed from docs:

* binary ``/opt/openpose/build/examples/openpose/openpose.bin``, linked against
  ``libcudart.so.11.0`` and ``libcudnn.so.9`` (CUDA 11.8 toolkit at
  ``/usr/local/cuda``) — it is a prebuilt binary, never rebuilt here;
* ``--disable_display`` **does not exist** in this build (it aborts with
  ``unknown command line flag``); headless operation is
  ``--display 0 --render_pose 0``;
* ``--write_json`` emits one ``<base>_000000000042_keypoints.json`` per frame of
  version ``1.3`` with flat ``[x, y, score, ...]`` triples;
* BODY_25 has 26 named parts including ``Background``
  (``/opt/openpose/src/openpose/pose/poseParameters.cpp``).

A 4-hour 50 fps recording is ~720k JSON files, so raw parsing is streamed and
Parquet row groups are written incrementally: memory stays flat regardless of
video length.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from ..artifacts import read_json
from ..config import stable_hash
from ..exceptions import StageError, ValidationError
from ..normalization import openpose_frame_number, openpose_frame_rows
from ..provenance import openpose_report
from ..schemas import BODY_SCHEMA, FACE_SCHEMA, HANDS_SCHEMA, ChunkedParquetWriter, read_table
from ..subprocess_utils import probe_version, require_executable, run_command
from ..validation import check_intervals
from .base import Stage, StageContext

#: Keypoint rows per frame group: bounds the row buffer, not the run length.
ROWS_PER_GROUP = 150_000


class OpenPoseStage(Stage):
    name = "openpose"
    inputs = ("metadata", "frame_index")
    outputs = ("pose_raw", "pose_body", "pose_hands", "pose_face")
    config_keys = ("openpose",)

    # ---------------------------------------------------------------- fingerprint

    def config_fingerprint(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.config.openpose
        discovery = openpose_report(ctx.config)
        return {
            "stage": self.name,
            "executable": discovery.get("executable"),
            "model_folder": discovery.get("model_folder"),
            "model_pose": cfg.body.model if cfg.body.enabled else None,
            "body_enabled": cfg.body.enabled,
            "hands_enabled": cfg.hands_enabled,
            "face_enabled": cfg.face_enabled,
            "gpu": cfg.gpu,
            "disable_multi_thread": cfg.disable_multi_thread,
            "extra_args": cfg.extra_args,
            # Model file identity: swapping a caffemodel must re-run pose.
            "model_files": discovery.get("models"),
            "source_sha256": self._source_sha(ctx),
        }

    @staticmethod
    def _source_sha(ctx: StageContext) -> str | None:
        path = ctx.artifact("metadata")
        if path.is_file():
            try:
                return read_json(path).get("SHA256")
            except (OSError, ValueError):
                return None
        return None

    # -------------------------------------------------------------------- run

    def enabled(self, ctx: StageContext) -> tuple[bool, str]:
        if not ctx.config.openpose.enabled:
            return False, "openpose.enabled = false"
        return True, ""

    def resolve_executable(self, ctx: StageContext) -> Path:
        cfg = ctx.config.openpose
        if cfg.executable != "auto":
            return require_executable(cfg.executable, hint="set openpose.executable")
        discovery = openpose_report(ctx.config)
        candidate = discovery.get("executable")
        if not candidate:
            raise StageError(
                f"OpenPose binary not found under {cfg.root}. Expected e.g. "
                f"{cfg.root}/build/examples/openpose/openpose.bin. Set openpose.executable explicitly."
            )
        return Path(candidate)

    def resolve_model_folder(self, ctx: StageContext) -> Path:
        cfg = ctx.config.openpose
        if cfg.model_folder != "auto":
            folder = Path(cfg.model_folder)
            if not folder.is_dir():
                raise StageError(f"openpose.model_folder is not a directory: {folder}")
            return folder
        discovery = openpose_report(ctx.config)
        candidate = discovery.get("model_folder")
        if not candidate:
            raise StageError(f"OpenPose model folder not found under {cfg.root}/models")
        return Path(candidate)

    def build_command(self, ctx: StageContext, output_dir: Path) -> list[str]:
        cfg = ctx.config.openpose
        argv: list[str] = [
            str(self.resolve_executable(ctx)),
            "--video", str(ctx.source.path),
            "--model_folder", str(self.resolve_model_folder(ctx)),
            "--write_json", str(output_dir),
            "--num_gpu", "1",
            "--num_gpu_start", str(cfg.gpu),
            # Headless: this build has no --disable_display flag at all.
            "--display", "0",
            "--render_pose", "0",
            # Keeping every frame in RAM is the main memory risk on long videos.
            "--disable_multi_thread" if cfg.disable_multi_thread else "--disable_multi_thread=false",
        ]
        # A model is always required by this build, and there is no CLI switch to
        # turn body tracking off (that exists only in the C++ API). So a disabled
        # body means "run the tracker but do not publish the body table", which is
        # recorded in the run log rather than pretended away with a bogus flag.
        argv += ["--model_pose", cfg.body.model or "BODY_25"]
        argv += ["--hand"] if cfg.hands_enabled else ["--hand=0"]
        argv += ["--face"] if cfg.face_enabled else ["--face=0"]
        # glog output must go to stderr so it lands in the stage log file.
        argv += ["--logtostderr", "--alsologtostderr"]
        return argv + list(cfg.extra_args)

    def publish_body(self, ctx: StageContext) -> bool:
        return bool(ctx.config.openpose.body.enabled)

    def prepare(self, ctx: StageContext) -> None:
        ctx.input("metadata")
        self.resolve_executable(ctx)
        self.resolve_model_folder(ctx)

    def execute(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.config.openpose
        raw_dir = ctx.artifact("pose_raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        argv = self.build_command(ctx, raw_dir)
        ctx.log(f"running OpenPose (model={cfg.body.model}, hands={cfg.hands_enabled}, "
                f"face={cfg.face_enabled}, gpu={cfg.gpu})")
        if not cfg.body.enabled:
            ctx.log("openpose.body.enabled=false: the CLI cannot disable body tracking, so the "
                    "tracker runs but pose/body.parquet is not published", level=30)
        result = run_command(argv, log_path=ctx.paths.log(self.name), timeout=cfg.timeout_seconds,
                             cwd=str(raw_dir.parent.parent))
        summary = self.normalize(ctx)
        return {
            "tool_version": openpose_version(ctx),
            "model_version": cfg.body.model if cfg.body.enabled else None,
            "command": result.argv_masked,
            "executable": argv[0],
            "exit_code": result.returncode,
            "extra": {"raw_frames": summary.get("raw_frames"), **summary},
        }

    # ------------------------------------------------------------ normalisation

    def normalize(self, ctx: StageContext) -> dict[str, Any]:
        """Stream raw per-frame JSON into three Parquet tables with row groups."""
        raw_dir = ctx.artifact("pose_raw")
        timings = self.frame_timings(ctx)
        writers = {
            "body": ChunkedParquetWriter(ctx.artifact("pose_body"), BODY_SCHEMA,
                                         rows_per_group=ROWS_PER_GROUP,
                                         extra_metadata={"model": self.model_name(ctx),
                                                         "video_id": ctx.video_id,
                                                         "coordinate_space": "source pixels (x,y)"}),
            "hands": ChunkedParquetWriter(ctx.artifact("pose_hands"), HANDS_SCHEMA,
                                          rows_per_group=ROWS_PER_GROUP,
                                          extra_metadata={"model": "OpenPose hand (21 keypoints)",
                                                          "video_id": ctx.video_id}),
            "face": ChunkedParquetWriter(ctx.artifact("pose_face"), FACE_SCHEMA,
                                         rows_per_group=ROWS_PER_GROUP,
                                         extra_metadata={"model": "OpenPose face (70 landmarks)",
                                                         "video_id": ctx.video_id}),
        }
        frames = 0
        for path, frame_number, document in self.iter_raw_frames(ctx):
            timestamp = timings.get(frame_number)
            if timestamp is None:
                # A frame beyond the index: extrapolate rather than drop data.
                timestamp = self.extrapolate(timings, frame_number)
            rows = openpose_frame_rows(document, ctx.video_id, frame_number, round(timestamp, 6))
            publish_body = self.publish_body(ctx)
            for key, rows_for_key in rows.items():
                if key == "body" and not publish_body:
                    continue
                writers[key].extend(rows_for_key)
            frames += 1
            if frames % 5_000 == 0:
                ctx.log(f"pose normalisation: {frames} frames")
        counts = {key: writer.close() for key, writer in writers.items()}
        ctx.scratch["openpose"] = {"raw_frames": frames, **counts}
        ctx.log(f"OpenPose normalisation: {frames} frames -> body={counts['body']} rows, "
                f"hands={counts['hands']} rows, face={counts['face']} rows")
        return {"raw_frames": frames, "body_rows": counts["body"], "hands_rows": counts["hands"],
                "face_rows": counts["face"]}

    def iter_raw_frames(self, ctx: StageContext) -> Iterator[tuple[Path, int, dict[str, Any]]]:
        """Yield frames in file order, one document in memory at a time."""
        raw_dir = ctx.artifact("pose_raw")
        if not raw_dir.is_dir():
            raise StageError(f"OpenPose raw output directory missing: {raw_dir}")
        for path in sorted(raw_dir.glob("*.json")):
            frame_number = openpose_frame_number(path.name)
            if frame_number is None:
                continue
            try:
                document = read_json(path)
            except (OSError, ValueError):
                continue
            if isinstance(document, dict):
                yield path, frame_number, document

    def frame_timings(self, ctx: StageContext) -> dict[int, float]:
        """frame_number → seconds, from the authoritative ffprobe packet index.

        OpenPose enumerates frames in presentation order, so its frame index
        matches the ffprobe packet index and the pose timeline inherits the
        media timeline exactly (rational frame rates included).
        """
        from ..schemas import iter_rows

        path = ctx.artifact("frame_index")
        if not path.is_file():
            raise StageError("frame_index.parquet is required to time pose keypoints")
        return {int(row["frame_number"]): float(row["pts_seconds"])
                for row in iter_rows(path, ["frame_number", "pts_seconds"])}

    @staticmethod
    def extrapolate(timings: dict[int, float], frame_number: int) -> float:
        if not timings:
            return float(frame_number) / 25.0
        last_index = max(timings)
        if frame_number <= last_index:
            return timings.get(frame_number, float(frame_number) / 25.0)
        ordered = sorted(timings)
        step = (timings[ordered[-1]] - timings[ordered[0]]) / max(len(ordered) - 1, 1) or 0.04
        return timings[last_index] + step * (frame_number - last_index)

    def model_name(self, ctx: StageContext) -> str:
        return ctx.config.openpose.body.model

    # ---------------------------------------------------------------- validation

    def validate(self, ctx: StageContext) -> dict[str, Any]:
        raw_dir = ctx.artifact("pose_raw")
        if not raw_dir.is_dir():
            raise ValidationError(self.name, ["pose/raw missing"])
        raw_files = list(raw_dir.glob("*.json"))
        if not raw_files:
            raise ValidationError(self.name, ["pose/raw contains no OpenPose JSON output"])
        duration = self._duration(ctx)
        report: dict[str, Any] = {"raw_files": len(raw_files)}
        for name, schema, time_col in (
            ("pose_body", BODY_SCHEMA, "timestamp"),
            ("pose_hands", HANDS_SCHEMA, "timestamp"),
            ("pose_face", FACE_SCHEMA, "timestamp"),
        ):
            path = ctx.artifact(name)
            if not path.is_file():
                raise ValidationError(self.name, [f"missing table: {path.name}"])
            from ..schemas import table_columns

            columns = set(table_columns(path))
            missing = [field.name for field in schema if field.name not in columns]
            if missing:
                raise ValidationError(self.name, [f"{path.name} missing columns: {', '.join(missing)}"])
            rows = read_table(path).to_pylist()
            report[f"{name}_rows"] = len(rows)
            if not rows:
                continue
            check_intervals([row["timestamp"] for row in rows], [row["timestamp"] for row in rows],
                            stage=self.name, label=f"{name}.timestamp", max_time=duration)
            frame_numbers = {row["frame_number"] for row in rows}
            if min(frame_numbers) < 0:
                raise ValidationError(self.name, [f"{name}.frame_number is negative"])
            if any(row["confidence"] is None or row["confidence"] <= 0 for row in rows):
                raise ValidationError(self.name, [f"{name} contains non-positive confidence rows"])
            if any(row["x"] is None or row["y"] is None for row in rows):
                raise ValidationError(self.name, [f"{name} contains null coordinates"])
        body_rows = report.get("pose_body_rows", 0)
        if self._config(ctx).body.enabled and self._config(ctx).body.model == "BODY_25" and body_rows:
            from ..schemas import BODY_25_KEYPOINT_NAMES

            names = {row["keypoint_name"] for row in read_table(ctx.artifact("pose_body")).to_pylist()}
            unknown = names - set(BODY_25_KEYPOINT_NAMES)
            if unknown:
                raise ValidationError(self.name, [f"pose_body has keypoints outside BODY_25: {sorted(unknown)[:6]}"])
        return report

    @staticmethod
    def _config(ctx: StageContext):
        return ctx.config.openpose

    @staticmethod
    def _duration(ctx: StageContext) -> float | None:
        path = ctx.artifact("metadata")
        if not path.is_file():
            return None
        try:
            return read_json(path).get("duration_seconds")
        except (OSError, ValueError):
            return None


def openpose_version(ctx: StageContext) -> str | None:
    """OpenPose has no stable --version; gflags prints a version banner at startup."""
    try:
        executable = OpenPoseStage().resolve_executable(ctx)
    except (StageError, FileNotFoundError):
        return None
    return probe_version([str(executable), "--version"], timeout=30.0)
