#!/usr/bin/env python3
"""Ultralytics YOLO person worker: one video in, one per-frame person document out.

Runs only inside ``environments/persons``. Everything below was measured on this machine
(2026-09-26, RTX 4090, driver 555.42.06 = CUDA 12.5) rather than read off the ultralytics
docs, because each item decides something in this file:

* **one video, one process, one model object.** The tracker hazard the parent measured is
  real: reusing one ``YOLO(...)`` across sources produced **489** ``GMC failed, falling back
  to identity: ...`` warnings in one batch, and the visible symptom — fragmented person ids —
  reads like a model-quality problem and is not one. What is actually going on, read off the
  installed source rather than guessed:

  * ultralytics' default ``tracker`` is ``tracktrack.yaml``, **not** bytetrack. TrackTrack,
    BoT-Sort and DeepOCSORT each build a ``GMC`` instance, which keeps
    ``prevFrame``/``prevKeyPoints``/``prevDescriptors`` on itself;
  * ``BYTETracker`` has no ``gmc`` attribute at all — the call site is guarded by
    ``if hasattr(self, "gmc") ...`` — so with ``tracker=bytetrack.yaml`` camera-motion
    compensation never runs and the warning cannot occur by construction;
  * ``BYTETracker.reset()`` clears tracks, the id counter and the Kalman filter but not GMC
    state, and ultralytics only calls ``reset()`` when the source path changes *and*
    ``persist`` is false. So the combination that breaks is a tracker that owns GMC state
    plus a reused model object.

  This worker pins ``tracker=bytetrack.yaml`` explicitly (leaving it unset would take
  ultralytics' ``tracktrack.yaml`` default), builds its own model inside the tracking
  function, and passes ``persist=False``. The warning counter below is the belt to that
  brace: a tracker change is one config value, and if a future setting reintroduces GMC the
  count lands in the document and the stage logs it, instead of 489 lines disappearing into
  a log file nobody opens.
* **``classes=[0]`` and ``persist=False`` are not interchangeable with the defaults.**
  Without the class filter the document carries cars and chairs; with ``persist=True`` the
  tracker carries live ids from a previous video in the same process, which is the same
  bug in a quieter form.
* **frame indices are not timestamps.** Ultralytics reports a 0-based index into the frames
  it decoded. The only authority on what an index means in seconds is
  ``source/frame_index.parquet``, which ``metadata`` wrote from ffprobe packet timestamps, so
  the worker maps every index through it and refuses to guess. VFR input is the case that
  makes the difference: this corpus's news clips are 30000/1001, and 25 fps arithmetic would
  drift by ~0.1% of the clip length.
* **the weights file is part of the result.** A person count means nothing without the model
  that produced it, so the resolved path, the file size and the sha256 all go in the document
  (§20.2 asks for a weights policy and this is the half of it that survives a resume).

Throughput measured here with ``device="cuda"``, ``classes=[0]``, ``stream=True``: KABC 126
frames at ~74 fps, CNN 124 at ~72, La-1 240 at ~94, person_demo 205 at ~67.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

#: Schema of the raw document this worker writes. Checked by the stage, so a change here
#: has to be a change there.
SCHEMA_VERSION = "1.0"

#: COCO class id for `person`, and the only reason `classes` defaults to it.
PERSON_CLASS = 0

#: The one ultralytics log line that means the ids about to be written are untrustworthy.
#: Matched on the library's own wording; see the module docstring for why it matters.
GMC_FAILURE_MARKERS = ("GMC failed",)


class WorkerFailure(RuntimeError):
    """An unrecoverable worker problem, reported as ``status: "error"``."""


# --------------------------------------------------------------------- helpers


def log(message: str) -> None:
    print(f"[persons] {message}", flush=True)


def package_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def sha256_of(path: Path) -> str | None:
    """Digest of a file, or None when it cannot be read.

    None rather than an exception: an unreadable weights file is reported as a null digest
    in provenance and stays ``validate``'s problem to complain about. Failing here would
    turn "we cannot prove which model ran" into "nothing ran", which loses a result that is
    otherwise fine.
    """
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


# ----------------------------------------------------------------- frame index


def load_frame_index(path: Path) -> dict[int, float]:
    """frame_number -> pts_seconds, from the metadata stage's ffprobe packet index.

    Read with the *pipeline's* pyarrow, which lives in this environment because
    ``environments/persons`` declares pyarrow. Deliberately not pandas: the file is two
    int/float columns and a DataFrame would be a second copy of it in memory for a 4-hour
    recording's 720k rows.
    """
    import pyarrow.parquet as pq

    try:
        table = pq.read_table(path, columns=["frame_number", "pts_seconds"])
    except Exception as exc:  # pyarrow raises many types
        raise WorkerFailure(f"frame index {path.name} could not be read: {exc}") from exc
    numbers = table.column("frame_number").to_pylist()
    stamps = table.column("pts_seconds").to_pylist()
    index: dict[int, float] = {}
    for number, stamp in zip(numbers, stamps):
        if number is None or stamp is None:
            continue
        index[int(number)] = float(stamp)
    if not index:
        raise WorkerFailure(f"frame index {path.name} contains no usable frame rows")
    return index


def resolve_timestamp(index: Mapping[int, float], frame_number: int) -> float | None:
    """Seconds for one decoded-frame index, or None when the index is not in the table.

    None rather than an extrapolation. The other visual stages extrapolate because OpenPose
    can emit a frame the packet index never listed and dropping it would lose a pose; here a
    person sighting whose instant cannot be named is a sighting that cannot be placed on the
    timeline, and the row says so in ``timestamp`` while ``validate`` counts how many did.
    A wrong timestamp, by contrast, would silently misalign every join this table makes.
    """
    return index.get(frame_number)


# --------------------------------------------------------------------- weights


def resolve_weights(model: str, weights_dir: Path | None) -> tuple[str, Path | None, str | None]:
    """Decide which checkpoint the model is built from, and say what it is.

    Returns ``(argument_passed_to_ultralytics, resolved_path_or_None, note)``.

    ``weights_dir`` in ultralytics' own settings is a **relative** path (``weights``) held in
    a user-level config file, so it resolves against whoever happened to start the process
    and is not governed by this repository's YAML. That is why this resolution is explicit
    and why the stage passes a directory only when the operator set one.

    When no ``weights_dir`` is given the name is handed to ultralytics untouched, so its
    normal download-on-first-use still works — and the path it actually used is read back off
    the model object afterwards and recorded. A configured ``weights_dir`` is a promise about
    a file, so a missing checkpoint is raised here instead of becoming an unattended download
    into a directory nobody chose (the same refusal TalkNet's worker makes).
    """
    if weights_dir is None:
        return model, None, "no persons.weights_dir: ultralytics resolves and may download"
    if not weights_dir.is_dir():
        raise WorkerFailure(f"--weights-dir is not a directory: {weights_dir}")
    candidate = weights_dir / model
    if not candidate.is_file():
        raise WorkerFailure(
            f"--weights-dir is set but {candidate} is missing; refusing to fall back to an "
            f"unattended ultralytics download of {model} into a directory nobody chose"
        )
    return str(candidate), candidate, f"loaded from persons.weights_dir: {candidate}"


def model_weights_path(model: Any, fallback: Path | None) -> Path | None:
    """The checkpoint the built model is actually running.

    ``YOLO.ckpt_path`` is set by ultralytics during ``__init__`` to the local file it loaded,
    whether that came from a path argument or from a download it performed. That is what makes
    an unset ``weights_dir`` auditable: the operator does not have to know where ultralytics
    put the file, the document says so.
    """
    if fallback is not None:
        return fallback
    reported = getattr(model, "ckpt_path", None)
    if not reported:
        return None
    path = Path(str(reported))
    return path if path.is_file() else None


# ------------------------------------------------------------------- device


def resolve_device(requested: str) -> tuple[Any, str, str | None]:
    """Translate the config's device into what ultralytics accepts.

    Returns ``(device_argument, resolved_name, fallback_reason)``. ``"auto"`` becomes
    ``None``, which is ultralytics' own "select cuda if available, else cpu" — so the
    decision is made once by torch and reported here rather than guessed at twice.
    """
    import torch

    if requested == "cuda":
        if not torch.cuda.is_available():
            # Not a fallback: the operator asked for the GPU and the environment cannot see
            # one. Almost always the wheel train in the pyproject comment, and the worst
            # possible silent-degradation case, so it stops here.
            raise WorkerFailure(
                "persons.device=cuda but torch.cuda.is_available() is False in this "
                f"environment (torch {torch.__version__}). The usual cause is a torch built "
                "for a CUDA newer than the local driver; environments/persons pins the "
                "cu126 wheel train for exactly this reason."
            )
        return 0, "cuda", None
    if requested == "cpu":
        return "cpu", "cpu", None
    if torch.cuda.is_available():
        return 0, "cuda", None
    return "cpu", "cpu", "torch.cuda.is_available() was False"


# --------------------------------------------------------------- the tracking


class GmcWarningCounter:
    """Count ultralytics' own \"GMC failed\" warnings, which are otherwise invisible.

    Installed as a ``logging.Handler`` on the ``ultralytics`` logger for the duration of the
    run. With the default ``bytetrack`` tracker this counter is expected to stay at zero,
    because ``BYTETracker`` has no ``gmc`` attribute and the call site is guarded by
    ``hasattr``. It is here for the day that stops being true: ultralytics' own default
    tracker (TrackTrack) does run GMC, a tracker change is one config value, and the failure
    it watches for raises no exception — one WARNING per frame, then tracking continues with
    identity warps and the ids come out fragmented. The measured batch this exists for had
    489 such lines, invisible to anything but someone reading the log.

    A handler rather than a stderr scrape because the library logs through
    ``logging.getLogger("ultralytics")`` and a scrape breaks the moment the formatting
    changes — and because the count has to become data in the document, not prose.
    """

    def __init__(self) -> None:
        import logging

        self.count = 0
        self.first_reason: str | None = None
        self._seen: list[str] = []
        self._handler = logging.Handler()
        self._handler.emit = self._emit  # type: ignore[method-assign]

    def _emit(self, record: Any) -> bool:
        message = record.getMessage()
        if any(marker in message for marker in GMC_FAILURE_MARKERS):
            self.count += 1
            if self.first_reason is None:
                self.first_reason = message.strip()[:400]
        return True

    def __enter__(self) -> "GmcWarningCounter":
        import logging

        logger = logging.getLogger("ultralytics")
        logger.addHandler(self._handler)
        self._logger = logger
        return self

    def __exit__(self, *exc_info: Any) -> None:
        try:
            self._logger.removeHandler(self._handler)
        except Exception:  # noqa: BLE001 - teardown must not mask the run's outcome
            pass


def track_video(*, video: Path, model_name: str, weights_dir: Path | None, device: str,
                tracker: str, conf: float, classes: list[int], imgsz: int,
                frame_index: Mapping[int, float],
                extra: Sequence[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run detection+tracking over one video and return (document, frame rows).

    The model is built *inside* this function and never lifted out, because the tracker
    state that causes id fragmentation lives on the model object. One video per process is
    the primary defence; building here is the second.
    """
    from ultralytics import YOLO

    argument, resolved, note = resolve_weights(model_name, weights_dir)
    log(f"loading {model_name}: {note}")
    device_arg, device_name, fallback_reason = resolve_device(device)
    started = time.time()
    model = YOLO(argument)
    weights = model_weights_path(model, resolved)
    counter = GmcWarningCounter()

    rows: list[dict[str, Any]] = []
    # `persons_in_frame` for a frame is the number of ids Ultralytics reported on it, and it
    # is only knowable after the whole result arrives, so a frame's rows are buffered until
    # its result is complete. Bounded by people-per-frame, not by video length.
    frames_measured = 0
    frames_with_person = 0
    per_frame: dict[int, int] = {}
    ids_seen: dict[int, int] = {}
    max_in_frame = 0
    unknown_timestamps = 0

    kwargs: dict[str, Any] = {
        "stream": True,
        # False, unconditionally, and not a config knob. `persist=True` carries live track
        # ids from a previous source in the same process — the same class of defect as the
        # GMC state above, only quieter, because nothing logs it. A worker that processes
        # exactly one video has no previous source to leak from, so the only value that can
        # be correct here is False, and a knob that can only be wrong one way is a way to
        # break the output.
        "persist": False,
        "conf": conf,
        "imgsz": imgsz,
        "tracker": f"{tracker}.yaml",
        "verbose": False,
    }
    if classes:
        kwargs["classes"] = list(classes)
    if device_arg is not None:
        kwargs["device"] = device_arg

    with counter:
        for result in model.track(source=str(video), **kwargs):
            # Ultralytics decodes the source in order and emits one result per frame, so the
            # 0-based count of results so far *is* the frame index it decoded. Cross-checked
            # against the library's own counter when it has one, because if that ever stopped
            # being true every timestamp in the table would drift silently.
            frames_measured += 1
            frame_number = frames_measured - 1
            reported = getattr(result, "frame_id", None)
            if reported is not None and int(reported) != frame_number:
                raise WorkerFailure(
                    f"ultralytics reported frame_id {int(reported)} for the "
                    f"{frames_measured}-th decoded frame, so frame order can no longer be "
                    "assumed and every timestamp in this table would be wrong; the worker "
                    "refuses to guess"
                )
            boxes = result.boxes
            ids = ([int(v) for v in boxes.id.cpu().numpy()] if boxes.id is not None
                   and boxes.id.numel() else [])
            xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy.numel() else []
            scores = boxes.conf.cpu().numpy() if boxes.conf.numel() else []
            stamp = resolve_timestamp(frame_index, frame_number)
            if stamp is None:
                unknown_timestamps += 1
            if ids:
                frames_with_person += 1
                max_in_frame = max(max_in_frame, len(ids))
                per_frame[frame_number] = len(ids)
            for index, person_id in enumerate(ids):
                rows.append(_detection_row(
                    frame_number=frame_number, timestamp=stamp, person_id=person_id,
                    box=xyxy[index], score=float(scores[index]),
                    persons_in_frame=len(ids),
                ))
                ids_seen[person_id] = ids_seen.get(person_id, 0) + 1
    elapsed = time.time() - started

    if frames_measured == 0:
        raise WorkerFailure(
            f"ultralytics decoded 0 frames from {video.name}; nothing was measured, so no "
            "person count can be reported"
        )

    document = {
        "schema_version": SCHEMA_VERSION,
        "video_id": None,  # filled by the caller, which has the CLI argument
        "model": model_name,
        "weights_path": str(weights) if weights else None,
        "weights_sha256": sha256_of(weights) if weights else None,
        "weights_size_bytes": (weights.stat().st_size if weights and weights.is_file()
                               else None),
        "tracker": tracker,
        "device": device_name,
        "requested_device": device,
        "device_fallback_reason": fallback_reason,
        "parameters": {
            "conf": conf,
            "imgsz": imgsz,
            "classes": list(classes),
            "ultralytics_args": {key: value for key, value in kwargs.items()
                                 if key != "classes"},
            "extra_args": list(extra),
        },
        "ultralytics_version": package_version("ultralytics"),
        "torch_version": package_version("torch"),
        "frames_measured": frames_measured,
        "frames_with_person": frames_with_person,
        "max_persons_in_frame": max_in_frame,
        # Sorted so the document is byte-stable across runs of the same video, which is what
        # lets the raw artifact be diffed at all.
        "person_ids": sorted(ids_seen),
        "person_frame_counts": {str(key): ids_seen[key] for key in sorted(ids_seen)},
        "frames_without_timestamp": unknown_timestamps,
        "gmc_failure_count": counter.count,
        "gmc_failure_reason": counter.first_reason,
        "elapsed_seconds": round(elapsed, 3),
        "frames": rows,
    }
    return document, rows


def _detection_row(*, frame_number: int, timestamp: float | None, person_id: int,
                   box: Any, score: float, persons_in_frame: int) -> dict[str, Any]:
    """One Ultralytics detection to a raw row.

    The bbox is written as-is from the model's post-NMS output, which ultralytics has
    already mapped back to the original frame — no rescaling here, because a second
    implementation of the letterbox inverse is a second thing to get wrong. Coordinates are
    rounded rather than truncated; the Parquet layer does not re-derive them.
    """
    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    return {
        "frame_number": frame_number,
        "timestamp": round(timestamp, 6) if timestamp is not None else None,
        "person_id": person_id,
        "x1": round(x1, 3), "y1": round(y1, 3), "x2": round(x2, 3), "y2": round(y2, 3),
        "confidence": round(score, 6),
        # ByteTrack carries the detection score forward as its track score, so this tracker
        # always has one. `confidence_reason` says that explicitly rather than leaving a
        # reader to infer it from a non-null column.
        "track_confidence": round(score, 6),
        "confidence_reason": "tracked",
        "persons_in_frame": persons_in_frame,
    }


# ---------------------------------------------------------------------- output


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Never leave a half-written artifact for the next resume to trust."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.",
                                         suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


# ------------------------------------------------------------------------ main


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frame-index", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--video-id", default="unknown")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--weights-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--tracker", default="bytetrack")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--classes", default=str(PERSON_CLASS),
                        help="comma-separated COCO class ids; empty means no class filter")
    parser.add_argument("--request-hash", default=None)
    return parser.parse_args(argv)


def parse_classes(raw: str) -> list[int]:
    if not raw.strip():
        return []
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError as exc:
            raise WorkerFailure(f"--classes contains {part!r}, which is not an integer") from exc
    return values


def _same_path(one: Path, two: Path) -> bool:
    """True when two spellings reach the same file: same realpath, or same live inode.

    Both calls are guarded, not just ``samefile``: ``resolve()`` also fails, and this runs
    before the try block in ``main``, so an escaping exception here would kill the worker with
    no result document at all -- the stage would report a missing result instead of the reason.
    It fails with two different exception types, which is why both are named: ``OSError`` for a
    path that cannot be reached, and ``RuntimeError``, because ``pathlib.resolve`` converts
    ``ELOOP`` into that on a symlink loop (measured, not assumed: a two-symlink loop raised
    ``RuntimeError: Symlink loop`` straight out of ``main``). An unresolvable path cannot be
    shown to name the same file as anything, so it is reported as "not an alias" and the
    ordinary input/output checks go on to reject it themselves.
    """
    try:
        if one.resolve() == two.resolve():
            return True
        return os.path.samefile(one, two)
    except (OSError, RuntimeError):
        return False


class OutputAlias(NamedTuple):
    """One pair among this run's four paths that turns out to name the same file."""

    out_flag: str
    in_flag: str
    out_path: Path
    in_path: Path

    @property
    def other_kind(self) -> str:
        """"input" or "output": ``--output-json`` is not an input, and the refusal should not say it is."""
        return "input" if self.in_flag in ("--video", "--frame-index") else "output"


def output_aliases(args: argparse.Namespace) -> list[OutputAlias]:
    """Every pair among the inputs and outputs that resolves to one file.

    Checked at the boundary rather than in the stage because the stage assembles these paths
    itself and nothing upstream would notice a mistake: an artifact is written with
    ``os.replace()``, so an output that names an input silently swaps the operator's video or
    frame index for JSON after a full tracking run and reports success.
    """
    inputs = {"--video": args.video, "--frame-index": args.frame_index}
    outputs = {"--output-json": args.output_json, "--result-path": args.result_path}
    found: list[OutputAlias] = []
    for out_flag, out in outputs.items():
        for in_flag, source in inputs.items():
            if _same_path(out, source):
                found.append(OutputAlias(out_flag, in_flag, out, source))
    if _same_path(args.output_json, args.result_path):
        found.append(OutputAlias("--result-path", "--output-json",
                                 args.result_path, args.output_json))
    return found


def validate(args: argparse.Namespace) -> None:
    if not args.video.is_file():
        raise WorkerFailure(f"input video not found: {args.video}")
    if not args.frame_index.is_file():
        raise WorkerFailure(
            f"frame index not found: {args.frame_index}. The persons stage maps frame "
            "indices to seconds through it, so the metadata stage has to run first."
        )
    if not 0.0 < args.conf < 1.0:
        raise WorkerFailure("--conf must be a confidence in (0, 1)")
    if args.imgsz <= 0:
        raise WorkerFailure("--imgsz must be a positive pixel count")
    if not args.model.strip():
        raise WorkerFailure("--model must name a checkpoint")
    aliases = output_aliases(args)
    if aliases:
        raise WorkerFailure(
            "; ".join(f"{a.out_flag} points at the {a.in_flag} "
                      f"{a.other_kind} ({a.out_path})" for a in aliases)
            + " -- refusing to write an artifact over another path this run uses"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    # The result contract is written in a ``finally`` block, so a refusal would itself be
    # delivered by overwriting whatever --result-path names. If that is an input, honouring
    # the contract *is* the data loss: say so on stderr, write nothing anywhere, exit non-zero
    # (the stage then fails on its own missing-result check rather than on a video-shaped JSON).
    if any(a.out_flag == "--result-path" and a.in_flag in ("--video", "--frame-index")
           for a in output_aliases(args)):
        log(f"refusing to run: --result-path names {args.result_path}, which this run also "
            f"reads as an input; no file was written")
        return 1
    payload: dict[str, Any] = {"status": "error", "stage": "persons"}
    try:
        validate(args)
        classes = parse_classes(args.classes)
        frame_index = load_frame_index(args.frame_index)
        document, rows = track_video(
            video=args.video, model_name=args.model, weights_dir=args.weights_dir,
            device=args.device, tracker=args.tracker, conf=args.conf, classes=classes,
            imgsz=args.imgsz, frame_index=frame_index, extra=[],
        )
        document["video_id"] = args.video_id
        write_json_atomic(args.output_json, document)
        payload.update({
            "status": "ok",
            "tool_version": document["ultralytics_version"],
            "model_version": (f"{document['model']}@"
                              f"{(document['weights_sha256'] or '')[:12]}"),
            "frames_measured": document["frames_measured"],
            "frames_with_person": document["frames_with_person"],
            "persons": len(document["person_ids"]),
            "max_persons_in_frame": document["max_persons_in_frame"],
            "device": document["device"],
            "device_fallback_reason": document["device_fallback_reason"],
            "weights_path": document["weights_path"],
            "weights_sha256": document["weights_sha256"],
            "gmc_failure_count": document["gmc_failure_count"],
        })
        log(f"tracked {len(document['person_ids'])} person id(s) over "
            f"{document['frames_measured']} frame(s) on {document['device']} "
            f"({len(rows)} detection row(s), {document['elapsed_seconds']}s)")
        if document["gmc_failure_count"]:
            log(f"WARNING: {document['gmc_failure_count']} frame(s) lost camera-motion "
                f"compensation; ids may be fragmented by the harness, not the scene")
    except Exception as exc:  # noqa: BLE001 - reported through the result contract
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
        log(f"failed: {payload['error']}")
    finally:
        args.result_path.parent.mkdir(parents=True, exist_ok=True)
        args.result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
