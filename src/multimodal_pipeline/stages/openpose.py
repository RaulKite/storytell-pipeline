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

Rendering is a separate concern and is off by default. Verified against this build:
``--write_images <dir>`` picks the output directory and ``--write_images_format`` the
container, while rendering is decided per module — ``--render_pose``, ``--face_render``
and ``--hand_render`` are independent switches, each accepting ``-1`` to inherit
``--render_pose``. So body+hands+face skeletons in one pass are one extra flag group on
the run OpenPose already does, not a second run. ``--display 0`` stays: rendering is
independent of visual display and this build has no display to turn off.

Those flags are two different axes, and reading them as one is the classic misreading here.
``--render_pose`` / ``--face_render`` / ``--hand_render`` choose the rendering *engine* and whether
a module is drawn at all (``-1`` inherits, which on this build resolves to the GPU rendering
path); they say nothing about size. Frame size is the other axis, ``--output_resolution``, whose
default ``-1x-1`` means "whatever the input is", i.e. full input resolution. So a provenance
record of ``render_pose: "-1"`` is a statement about the engine, not a missing resolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from ..artifacts import read_json
from ..exceptions import StageError, ValidationError
from ..normalization import openpose_frame_number, openpose_frame_rows
from ..provenance import openpose_report
from ..schemas import BODY_SCHEMA, FACE_SCHEMA, HANDS_SCHEMA, ChunkedParquetWriter, read_table
from ..subprocess_utils import probe_version, require_executable, run_command
from ..validation import check_intervals
from .base import Stage, StageContext

#: Keypoint rows per frame group: bounds the row buffer, not the run length.
ROWS_PER_GROUP = 150_000

#: Container for rendered frames. ``jpg`` because one PNG per frame at full
#: resolution is the disk cost this feature is most likely to blow up on.
IMAGE_FORMAT = "jpg"

#: File suffixes counted as rendered images. OpenPose chooses the encoder from
#: ``--write_images_format`` and ``extra_args`` can override it, so counting follows
#: the files on disk rather than the flag we passed.
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")

#: ``--render_pose`` is the rendering-engine axis, not a size. ``-1`` means "inherit", which on
#: this build resolves to the GPU rendering path; ``0`` draws nothing. Frame size belongs to
#: ``--output_resolution`` alone (its default ``-1x-1`` = full input resolution). Strings, because
#: that is how they reach argv.
RENDER_POSE_INHERIT = "-1"
RENDER_POSE_OFF = "0"

#: The same two axes, emitted with every run's provenance so a reader of the JSON alone cannot
#: take ``render_pose: "-1"`` for a resolution that failed to be computed.
RENDER_AXES = {
    "render_pose": "rendering engine/mode, not a size: -1 = inherit = GPU rendering path on "
                   "this build, 0 = nothing drawn",
    "output_resolution": "frame size WxH: null = OpenPose default (-1x-1) = full input "
                         "resolution",
}


def scaled_output_resolution(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Largest WxH that fits ``max_side`` on the long side, aspect preserved.

    Never upscales: a 320x240 source stays 320x240. OpenPose would happily render a
    bigger frame, and a bigger frame is only an interpolation of the same pixels.
    """
    ratio = min(1.0, float(max_side) / float(max(width, height)))
    return (max(int(round(width * ratio)), 1), max(int(round(height * ratio)), 1))


class OpenPoseStage(Stage):
    name = "openpose"
    inputs = ("metadata", "frame_index")
    outputs = ("pose_raw", "pose_images_raw", "pose_body", "pose_hands", "pose_face")
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
            # Both change what is written to pose/, so both invalidate the stage.
            # ``image_max_side`` is hashed even while rendering is off: enabling the
            # flag must not silently resurrect a resolution recorded earlier.
            "write_images": cfg.write_images,
            "image_max_side": cfg.image_max_side,
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
            # One occurrence only: gflags would let a later duplicate win, and a
            # command whose meaning depends on flag order is not reviewable. Default
            # off keeps the literal "0" this stage has always passed.
            "--render_pose", self.render_pose_value(cfg),
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
        argv += self.render_args(ctx)
        # glog output must go to stderr so it lands in the stage log file.
        argv += ["--logtostderr", "--alsologtostderr"]
        return argv + list(cfg.extra_args)

    @staticmethod
    def render_pose_value(cfg) -> str:
        """The rendering-engine switch: ``-1`` = inherit, i.e. GPU rendering on the CUDA path.

        Nothing about frame size is decided here; ``output_resolution()`` owns that axis.
        """
        return RENDER_POSE_INHERIT if (cfg.write_images and cfg.body.enabled) else RENDER_POSE_OFF

    def render_args(self, ctx: StageContext) -> list[str]:
        """The flag group that asks OpenPose for its own rendered frames.

        Returns nothing at all when ``write_images`` is off, which is what keeps the
        default command byte-identical to the one from before this feature existed.
        ``--render_pose`` is passed by ``build_command`` itself, in its original
        position; only the remaining switches are added here.
        """
        cfg = ctx.config.openpose
        if not cfg.write_images:
            return []
        # Rendering is per module in this build, so each switch follows the config
        # that decides whether that module's keypoints are even produced. When body
        # publishing is off its skeleton is not drawn either, because an image for a
        # table that was never published is a lie of the same size as the table.
        argv = [
            "--write_images", str(ctx.artifact("pose_images_raw")),
            "--write_images_format", IMAGE_FORMAT,
            "--face_render", "1" if cfg.face_enabled else "0",
            "--hand_render", "1" if cfg.hands_enabled else "0",
        ]
        # The size axis is appended here; the engine axis (--render_pose) is added by
        # build_command, so the two never arrive mixed up in one flag group.
        resolution = self.output_resolution(ctx)
        if resolution:
            argv += ["--output_resolution", resolution]
        return argv

    def output_resolution(self, ctx: StageContext) -> str | None:
        """``WxH`` for ``--output_resolution``, or None to leave OpenPose's ``-1x-1`` default.

        None is not "no resolution": the render then runs at full input resolution. This is the
        frame-size axis and has nothing to do with ``--render_pose``, which is the engine axis.
        Dimensions come from the metadata artifact (ffprobe's video stream), not from
        a second probe: the stage already depends on that file and its hash.
        """
        cfg = ctx.config.openpose
        if not cfg.write_images or cfg.image_max_side is None:
            return None
        path = ctx.artifact("metadata")
        try:
            metadata = read_json(path)
        except (OSError, ValueError):
            return None
        width, height = metadata.get("width"), metadata.get("height")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            return None
        w, h = scaled_output_resolution(width, height, cfg.image_max_side)
        return f"{w}x{h}"

    @staticmethod
    def count_rendered_images(ctx: StageContext) -> int:
        images = ctx.artifact("pose_images_raw")
        if not images.is_dir():
            return 0
        return sum(1 for path in images.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)

    def clear_render_dir(self, ctx: StageContext) -> int:
        """Empty ``pose/raw_images`` before a rendering run and report how many files went.

        ``count_rendered_images`` counts what is on disk, so without this it counts every run
        that ever wrote there -- which is what defeats the zero-render guard in ``execute``: flip
        ``write_images`` off and back on, the fingerprint re-runs the stage, and a run in which
        OpenPose draws nothing still sees the previous run's thousands of images, passes the
        guard, and is recorded complete with zero skeletons from this run.

        Only ``pose/raw_images``, and only when rendering was asked for: with the flag off the
        stage has no business erasing images it was never asked to redo, even though it still
        counts them. ``pose/raw`` (the JSON keypoints) is never touched from here.
        """
        images = ctx.artifact("pose_images_raw")
        if not images.is_dir():
            images.mkdir(parents=True, exist_ok=True)
            return 0
        removed = 0
        for path in sorted(images.rglob("*"), reverse=True):
            if not path.is_file():
                continue
            try:
                path.unlink()
            except OSError as exc:
                # A file that survives here puts the defeated guard straight back, so this is a
                # failed run rather than a warning: the render count would describe that leftover.
                raise StageError(
                    f"cannot clear rendered frame {path} before the render: "
                    "openpose.write_images=true needs an empty pose/raw_images so the rendered "
                    "count describes this run"
                ) from exc
            removed += 1
        return removed

    @staticmethod
    def draws_any_module(cfg) -> bool:
        """Whether at least one skeleton would be drawn on the frames OpenPose writes.

        The three render switches are the only thing that puts ink on a frame, and each follows
        one config key (``--render_pose`` by ``render_pose_value``, ``--face_render`` and
        ``--hand_render`` by ``face_enabled``/``hands_enabled``). So this is decidable without
        running anything: no pixel inspection, no post-run stat.
        """
        return bool(cfg.body.enabled or cfg.face_enabled or cfg.hands_enabled)

    def refuse_undrawn_render(self, ctx: StageContext) -> None:
        """Refuse ``write_images=true`` when no module is enabled to be drawn.

        Measured on this build (249-frame clip): with ``--render_pose 0 --face_render 0
        --hand_render 0`` the binary still writes one image per processed frame -- all 249 --
        and they are the *source* frames at full source resolution, because on that path
        ``--output_resolution`` is ignored. The run exits 0 with a full image directory, so the
        zero-render guard in ``execute`` can never fire for it: that guard sees ``rendered == 0``,
        which is now only ever a run that wrote no files at all. Config-time refusal is the only
        place left, and the request is unambiguous from config alone.

        Called before the render directory is cleared, so refusing costs nothing the operator
        had already paid for.
        """
        cfg = ctx.config.openpose
        if not cfg.write_images or self.draws_any_module(cfg):
            return
        message = (
            "openpose.write_images=true but openpose.body.enabled, openpose.face.enabled and "
            "openpose.hands.enabled are all false, so nothing would be drawn: in this build "
            "--write_images still writes one image per processed frame with --render_pose 0 "
            "--face_render 0 --hand_render 0, and --output_resolution does not bound those "
            "images, so the run would produce full-resolution frames containing no skeleton. "
            "Enable a module (body.enabled, face.enabled or hands.enabled) or set "
            "openpose.write_images=false."
        )
        ctx.log(message, level=40)
        raise StageError(message, details={
            "write_images": True,
            "body_enabled": bool(cfg.body.enabled),
            "face_enabled": bool(cfg.face_enabled),
            "hands_enabled": bool(cfg.hands_enabled),
        })

    def publish_body(self, ctx: StageContext) -> bool:
        return bool(ctx.config.openpose.body.enabled)

    def prepare(self, ctx: StageContext) -> None:
        ctx.input("metadata")
        self.resolve_executable(ctx)
        self.resolve_model_folder(ctx)

    def execute(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.config.openpose
        # Before the directory is emptied and before the binary: a render with nothing to draw
        # must not even cost the operator their previous frames.
        self.refuse_undrawn_render(ctx)
        raw_dir = ctx.artifact("pose_raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        argv = self.build_command(ctx, raw_dir)
        ctx.log(f"running OpenPose (model={cfg.body.model}, hands={cfg.hands_enabled}, "
                f"face={cfg.face_enabled}, gpu={cfg.gpu})")
        if not cfg.body.enabled:
            ctx.log("openpose.body.enabled=false: the CLI cannot disable body tracking, so the "
                    "tracker runs but pose/body.parquet is not published", level=30)
        if cfg.write_images:
            removed = self.clear_render_dir(ctx)
            if removed:
                # Disk churn on a rerun is the operator's to see: these are files they may have
                # believed were still the current render.
                ctx.log(f"openpose.write_images=true: removed {removed} file(s) left in "
                        f"{ctx.artifact('pose_images_raw')} by an earlier run, so the rendered "
                        "count below describes this run only", level=30)
            self.log_render_plan(ctx)
        result = run_command(argv, log_path=ctx.paths.log(self.name), timeout=cfg.timeout_seconds,
                             cwd=str(raw_dir.parent.parent))
        rendered = self.count_rendered_images(ctx)
        if cfg.write_images and rendered == 0:
            # The trap this guard exists for: OpenPose exits 0 having written no
            # images, and the dataset looks complete from here on. Fail now, before
            # normalisation, with the flags that decide rendering named. The directory was
            # emptied above, so `rendered == 0` here really means this run rendered nothing.
            ctx.log(
                f"openpose.write_images=true but {ctx.artifact('pose_images_raw')} contains no "
                f"rendered images: --write_images/--render_pose/--face_render/--hand_render "
                f"produced nothing (exit {result.returncode})", level=40)
            raise StageError(
                "OpenPose rendered no images although write_images is enabled: "
                f"{ctx.artifact('pose_images_raw')} is empty. The run would otherwise be "
                "recorded as complete with zero skeletons. Flags that decide rendering: "
                "--write_images, --write_images_format, --render_pose, --face_render, "
                "--hand_render (each render switch accepts -1 to inherit --render_pose).",
                details={"rendered_images": 0, "exit_code": result.returncode,
                         "command": result.argv_masked})
        summary = self.normalize(ctx)
        if cfg.write_images:
            ctx.log(f"OpenPose render: {rendered} rendered images in "
                    f"{ctx.artifact('pose_images_raw').relative_to(ctx.paths.dataset_dir)}")
        return {
            "tool_version": openpose_version(ctx),
            "model_version": cfg.body.model if cfg.body.enabled else None,
            "command": result.argv_masked,
            "executable": argv[0],
            "exit_code": result.returncode,
            "extra": {"raw_frames": summary.get("raw_frames"), "rendered_images": rendered,
                      **summary, **self.render_provenance(ctx, rendered=rendered)},
        }

    def log_render_plan(self, ctx: StageContext) -> None:
        """One warning per run about the resolution the render will use.

        Resolution is the ``--output_resolution`` axis. ``--render_pose`` is the engine axis and
        never bounds what a frame costs.

        Only ``--output_resolution`` bounds the cost, and its default is the input
        resolution: at 25 fps a 4-hour recording is ~360k full-frame images. The
        stage says so once instead of deciding the downscale for the operator, and it
        says so again when a requested cap cannot be applied -- silently ignoring
        ``image_max_side`` would leave the operator believing the run was cheap.
        """
        cfg = ctx.config.openpose
        if cfg.image_max_side is None:
            ctx.log("openpose.write_images=true with image_max_side=null: OpenPose renders at full "
                    "input resolution (--output_resolution -1x-1), which costs hundreds of MB to GBs "
                    "per video; set openpose.image_max_side to cap the longest rendered side",
                    level=30)
        elif self.output_resolution(ctx) is None:
            ctx.log(f"openpose.image_max_side={cfg.image_max_side} ignored: the metadata artifact "
                    f"has no width/height, so no --output_resolution could be computed and the "
                    "render runs at full input resolution", level=30)

    def render_provenance(self, ctx: StageContext, *, rendered: int) -> dict[str, Any]:
        """What was asked for and what arrived, alongside the run's other extras.

        ``validate`` reports the same count, and its report is the part that is
        persisted per stage; this records the request that produced it, so a later
        reader can tell "no images" from "no images requested". When ``requested`` is
        false the count is a leftover from some earlier run: the directory is
        deliberately not cleared then.

        ``axes`` exists because the flag names read like one axis. ``render_pose`` is the
        rendering engine (``-1`` = inherit = GPU rendering on this build); ``output_resolution``
        is the frame size (``null`` = OpenPose's default = full input resolution).
        """
        cfg = ctx.config.openpose
        return {"render": {
            "requested": bool(cfg.write_images),
            "directory": str(ctx.artifact("pose_images_raw").relative_to(ctx.paths.dataset_dir)),
            "format": IMAGE_FORMAT if cfg.write_images else None,
            "render_pose": self.render_pose_value(cfg),
            "face_render": "1" if (cfg.write_images and cfg.face_enabled) else "0",
            "hand_render": "1" if (cfg.write_images and cfg.hands_enabled) else "0",
            "output_resolution": self.output_resolution(ctx),
            "image_max_side": cfg.image_max_side,
            "rendered_images": rendered,
            "axes": dict(RENDER_AXES),
        }}

    # ------------------------------------------------------------ normalisation

    def normalize(self, ctx: StageContext) -> dict[str, Any]:
        """Stream raw per-frame JSON into three Parquet tables with row groups."""
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
        # Reported as a pair on purpose: rendered_images says how many files are on disk, and
        # only render_requested says whether the run being validated asked for them.
        report["render_requested"] = bool(self._config(ctx).write_images)
        report["rendered_images"] = self.validate_renders(ctx, raw_frames=len(raw_files))
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

    def validate_renders(self, ctx: StageContext, *, raw_frames: int) -> int:
        """Check ``pose/raw_images`` against the frame count, or excuse it entirely.

        OpenPose writes one rendered frame per processed frame, so a directory that
        is materially short means a truncated or partially deleted render. One frame
        of slack is allowed: the pipeline counts JSON files while OpenPose may finish
        writing the last image after the last JSON it flushed.

        With rendering off, the directory is not checked at all. A dataset that was
        once produced with the flag on keeps its images, and asking a run that never
        requested them to account for them would turn an opt-in off-switch into a
        permanent validation failure -- the fingerprint already forces the rerun that
        actually decides what is there.

        The count is therefore about whoever wrote last, which is why ``validate`` reports it
        beside ``render_requested`` instead of on its own. A count of *this* run comes from
        ``execute``, which empties the directory first when rendering is on.
        """
        rendered = self.count_rendered_images(ctx)
        if not self._config(ctx).write_images:
            return rendered
        if not ctx.artifact("pose_images_raw").is_dir():
            raise ValidationError(self.name,
                                  [f"pose/raw_images missing although openpose.write_images=true "
                                   f"({ctx.artifact('pose_images_raw')})"])
        # ``>= raw_frames - 1``: one image per processed frame, one frame of slack.
        if rendered < raw_frames - 1:
            raise ValidationError(self.name,
                                  [f"pose/raw_images has {rendered} images but pose/raw has "
                                   f"{raw_frames} frame JSON files (expected at least "
                                   f"{max(raw_frames - 1, 0)}, one render per processed frame)"])
        return rendered

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
