"""``persons``: YOLO person detection + tracking, one row per person per frame (§20.2 / T15).

The third visual signal in this pipeline, and the only one that answers two specific
questions: **how many distinct people appear in a video**, and **when is each one on
screen**. OpenPose answers *where are the bodies, joint by joint*; TalkNet answers *which
visible face is producing the audio*; neither of them keeps an identity across frames in a
way that yields a count.

Facts verified on this machine (2026-09-26) rather than assumed, because each one decides
something in the design:

* **the environment is load-bearing.** PyPI's default resolution for ultralytics 8.4.x
  installs ``torch 2.14.0+cu130``, which on this driver (555.42.06 = CUDA 12.5) reports
  ``torch.cuda.is_available() -> False`` and runs the whole batch on CPU with nothing in
  the log saying so. ``environments/persons`` pins ``torch==2.8.0`` to the cu126 wheel
  train, measured here at ``cuda available: True`` on the RTX 4090. This is the same
  failure class ``environments/whisperx`` and ``environments/diarization_nemotron`` each
  document for their own tool;
* **the GPU was not enough — a reused tracker fragmented the ids.** The parent measured 489
  ultralytics ``GMC failed`` warnings in one batch with a single ``YOLO(...)`` object reused
  across four clips, and the symptom was visibly fragmented person ids: it reads exactly like
  a model-quality problem and is not one. Ultralytics' camera-motion compensation keeps a
  ``prevFrame`` on its ``GMC`` instance; reused across sources it raises every frame, the
  tracker falls back to identity warps, and all that surfaces is a ``WARNING`` line. Note for
  whoever revisits this: ``BYTETracker`` carries no GMC instance at all (the call site is
  guarded by ``hasattr``), so the default ``bytetrack`` configuration cannot produce that
  warning — but ultralytics' own default tracker is ``tracktrack.yaml``, which can, which is
  why the worker names its tracker explicitly and counts the warning anyway. A process that
  tracks exactly one video is clean (0 warnings); see
  ``workers/persons_worker.py`` for the source-level detail;
* **one video per process, because a reused tracker fragments ids.** Measured on this
  corpus with ``yolo11n.pt`` and one fresh tracker per video: KABC **2** ids (126 and 112
  frames of 126), CNN **4** ids (124, 124, 124, 108 of 124), La-1 **6** ids (69, 10, 69, 69,
  94, 94 of 240), person_demo **8** ids, and ``pipeline_demo`` — a synthetic TTS/testsrc
  clip — **0** ids. La-1's six ids over 240 frames are the fragmentation this design accepts
  rather than hides: a *person count* from a tracker is a count of identity hypotheses, and
  the two columns that make that legible are ``frame_count`` and
  ``longest_gap_seconds``. ``pipeline_demo``'s zero is the honest empty case, the same shape
  as its ``pose_body`` table having no rows;
* **person counts are only interpretable against the weights that produced them.** The
  worker therefore records the checkpoint path *and its sha256* in provenance, and the
  fingerprint carries both the model name and its digest (§20.2 asks for a weights policy
  and this is the part of it that survives contact with a resumed dataset).

What this stage deliberately does **not** do: cross-check its own ids against
``speaker/raw/scenes.csv``. §20.2 raises scene cuts as a use of the output, not as a
computation to embed here; the stage emits facts — tracks, spans, confidences, box areas,
persons per frame — and a second scene engine inside a tracking stage would be a second
opinion nobody asked to maintain.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import pyarrow as pa

from ..exceptions import ValidationError
from ..schemas import (
    PERSON_FRAMES_SCHEMA,
    PERSON_TRACKS_SCHEMA,
    iter_rows,
    read_table,
    table_columns,
    write_table,
)
from ..validation import check_intervals
from .base import StageContext, WorkerStage, raw_request_matches

#: The closed set of reasons a person row does or does not carry a tracker confidence.
#: Same purpose as ``activespeaker.FRAME_REASONS``: the vocabularies a reader switches on
#: have to be enumerable, or a ninth value is silently unhandled.
CONFIDENCE_REASONS = (
    "tracked",
    "no_track_confidence",
    "unknown",
)

#: Every frame the worker read, whether or not it saw anybody. A document without it cannot
#: distinguish "0 people detected in 126 frames" from "nothing was measured", which is the
#: distinction the whole stage rests on.
REQUIRED_RAW_KEYS = ("frames_measured", "frames_with_person", "person_ids", "frames")


class PersonsStage(WorkerStage):
    """Track people with YOLO in its own uv environment, normalise onto the timeline."""

    name = "persons"
    raw_artifact = "persons_raw"
    # metadata only, and deliberately so. This stage reads the video file and the frame
    # timing index; it never reads a transcript, an audio table, a pose table or a face
    # track. Depending on more than metadata would let a transcription failure cost the
    # person counts, which is the same reasoning that keeps `openpose` metadata-only, and
    # §20.2 says so: it is an independent visual signal.
    inputs = ("metadata", "frame_index")
    outputs = ("persons_raw", "person_frames", "person_tracks")
    config_keys = ("persons",)

    # ------------------------------------------------------------------ request

    def request(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.config.persons
        return {
            "stage": self.name,
            "provider": "ultralytics",
            "model": cfg.model,
            # The bytes, not just the name. Two machines with "yolo11n.pt" and different
            # weights must not claim each other's cached raw output, and a retrained or
            # re-downloaded checkpoint changes every person count. None when the
            # checkpoint has not been fetched yet, which is honest: nothing was cached
            # from a model we have not read.
            "weights_sha256": self._weights_digest(ctx),
            "weights_dir": str(ctx.config.resolve(cfg.weights_dir)) if cfg.weights_dir else None,
            "device": cfg.device,
            "device_index": cfg.device_index,
            "tracker": cfg.tracker,
            "conf": cfg.conf,
            "classes": list(cfg.classes),
            "imgsz": cfg.imgsz,
            "extra_args": cfg.extra_args,
            "uv_project": str(ctx.config.resolve(cfg.uv_project)),
            "worker": str(ctx.config.resolve(cfg.worker)),
            "source_sha256": self._source_digest(ctx),
        }

    # ------------------------------------------------------------------ fingerprint

    def config_fingerprint(self, ctx: StageContext) -> dict[str, Any]:
        """The parent's payload — which is the worker's digest — plus this module's source.

        `WorkerStage` already mixes `workers/persons_worker.py` into its fingerprint, which
        covers the detection. It cannot cover the normalisation: turning that raw JSON into
        `person_frames` and `person_tracks` happens here — the span summary, the confidence
        reasons, the gap arithmetic — and a change to any of it left a completed run looking
        "reusable" while the numbers it reported had moved. Same defect class as
        `pose_normalized` and `speaker_fusion`; here the parent's payload needed extending,
        not replacing.

        It is extended at this hook rather than in `digest_payload`, and the difference is the
        cost of a fix. `digest_payload` is also what `request_digest()` hashes, and that value
        is the `request_hash` stamped into the raw sidecar — the thing `validate()` compares to
        decide whether the preserved YOLO output belongs to this configuration at all. Mixing
        the *normaliser's* bytes in there would claim that re-normalising requires a new
        detection run: editing a docstring in this file would invalidate every preserved raw
        artifact on disk and cost a full GPU pass per video, which is the same semantic mix-up
        that keeps `request()` untouched. Fingerprint here, raw request there — a fix to the
        maths reruns the seconds-long normalisation, a fix to the worker reruns the model.
        """
        from . import persons as persons_stage
        from .base import python_source_digest

        payload = dict(super().config_fingerprint(ctx))
        payload["_python_code_sha256"] = python_source_digest(persons_stage)
        return payload

    def _weights_digest(self, ctx: StageContext) -> str | None:
        """SHA256 of the checkpoint this configuration would load, or None.

        Resolved the same way the worker resolves it, so the fingerprint describes the file
        that will actually be used rather than a guess at it. ``None`` means "not on disk
        yet" — an unconfigured run downloads on first use, and the worker then records the
        digest it read into the raw document, which ``validate`` compares against.

        The scratch cache is keyed on the file's size and mtime as well as its path. Caching on
        path alone would make a checkpoint replaced in place return the digest of the bytes
        that used to be there — which is precisely the case the whole weights policy exists to
        catch, and a plan pass reads this value more than once.
        """
        from ..stages.metadata import sha256_of

        path = self.weights_path(ctx)
        if path is None or not path.is_file():
            return None
        try:
            stat = path.stat()
        except OSError:
            return None
        cache_key = (f"persons_weights_sha256:{path}:{stat.st_size}:{stat.st_mtime_ns}")
        cached = ctx.scratch.get(cache_key)
        if cached:
            return cached
        digest = sha256_of(path)
        if digest is not None:
            ctx.scratch[cache_key] = digest
        return digest

    def weights_path(self, ctx: StageContext) -> Path | None:
        """The checkpoint file ``weights_dir`` points at, or None when unset.

        Ultralytics' own ``weights_dir`` lives in a user-level settings file and is a
        *relative* path (``weights``), so it is not governed by this config file at all and
        cannot be consulted here. This is the pipeline's own, explicit resolution.
        """
        cfg = ctx.config.persons
        if cfg.weights_dir is None:
            return None
        return ctx.config.resolve(cfg.weights_dir) / cfg.model

    @staticmethod
    def _source_digest(ctx: StageContext) -> str | None:
        """SHA256 of the source video, cached in scratch like the other video stages.

        Keyed distinctly from ``activespeaker``'s: both stages hash a file derived from the
        source, and sharing one scratch entry would make one stage's digest depend on which
        ran first. This stage hashes the *source*, because it reads the source.
        """
        from ..stages.metadata import sha256_of

        path = ctx.source.path
        cached = ctx.scratch.get("persons_source_sha256")
        if cached:
            return cached
        try:
            digest = sha256_of(path)
        except OSError:
            return None
        ctx.scratch["persons_source_sha256"] = digest
        return digest

    # ------------------------------------------------------------------ enablement

    def enabled(self, ctx: StageContext) -> tuple[bool, str]:
        """Skip with a reason when the stage cannot run, never emit an empty table.

        The reasons are distinct because the fixes are distinct, and they are read from the
        filesystem rather than from a flag alone: an operator who switched the stage on and
        never installed the environment needs to be told which of the two is missing. A
        *present but broken* environment is not screened here — it runs and fails loudly,
        because at that point silence would hide a real defect. Same three-state discipline
        as ``diarization_nemotron``.
        """
        cfg = ctx.config.persons
        if not cfg.enabled:
            return False, "persons.enabled = false"
        project = self.uv_project(ctx)
        if not project.is_dir():
            return False, (
                f"persons environment not installed at {project} — create it and run "
                "`uv sync --python 3.12` there to enable person tracking"
            )
        worker = self.worker_script(ctx)
        if not worker.is_file():
            return False, f"persons worker script missing at {worker}"
        if cfg.weights_dir is not None:
            # An explicit weights_dir is a promise about a file, so breaking it is reported
            # here rather than discovered by the worker on the first frame of the first
            # video. Note this is a *skip*, not a failure: the other interpretation (fall
            # back to an unattended download) is the one TalkNet's worker refuses for the
            # same reason — a download nobody asked for, into a directory nobody chose.
            weights = self.weights_path(ctx)
            if weights is not None and not weights.is_file():
                return False, (
                    f"persons.weights_dir is set but {weights} is missing — put the "
                    f"checkpoint there, or unset persons.weights_dir to let ultralytics "
                    f"download {cfg.model} on first use"
                )
        return True, ""

    # ------------------------------------------------------------------ worker

    def uv_project(self, ctx: StageContext) -> Path:
        return ctx.config.resolve(ctx.config.persons.uv_project)

    def worker_script(self, ctx: StageContext) -> Path:
        return ctx.config.resolve(ctx.config.persons.worker)

    def python_version(self, ctx: StageContext) -> str | None:
        return ctx.config.persons.python_version

    def worker_timeout(self, ctx: StageContext) -> float | None:
        # None by default. Ultralytics runs at ~70-95 fps on this machine's clips, so a
        # four-minute 25 fps video is seconds — but a 4-hour recording is 720k frames and a
        # guessed ceiling would fail a legitimate long run. The operator sets it.
        return ctx.config.persons.timeout_seconds

    def worker_environment(self, ctx: StageContext) -> dict[str, str]:
        cfg = ctx.config.persons
        env: dict[str, str] = {}
        if cfg.device == "cuda":
            # Same convention as the diarization and activespeaker workers: one GPU per
            # stage run, selected by the pipeline rather than by the worker.
            env["CUDA_VISIBLE_DEVICES"] = str(cfg.device_index)
        return env

    def prepare(self, ctx: StageContext) -> None:
        super().prepare(ctx)
        # `frame_index` is a hard input, not a nice-to-have: ultralytics reports frame
        # *indices*, and the only authority on what an index is in seconds is the ffprobe
        # packet index metadata wrote. Without it this stage could publish an FPS-derived
        # timestamp for a VFR clip, which is the error every other visual stage avoids.
        ctx.input("frame_index")

    def worker_argv(self, ctx: StageContext, raw_path: Path, request_digest: str) -> list[str]:
        cfg = ctx.config.persons
        from ..uv_worker import worker_result_path

        args = [
            "--video", str(ctx.source.path),
            "--frame-index", str(ctx.input("frame_index")),
            "--output-json", str(raw_path),
            "--result-path", str(worker_result_path(raw_path.parent,
                                                    f"{self.name}_worker_result.json")),
            "--video-id", ctx.video_id,
            "--model", cfg.model,
            "--device", cfg.device,
            "--tracker", cfg.tracker,
            "--conf", str(cfg.conf),
            "--imgsz", str(cfg.imgsz),
            "--request-hash", request_digest,
        ]
        if cfg.classes:
            args += ["--classes", ",".join(str(value) for value in cfg.classes)]
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
            ctx.artifact("person_frames"),
            pa.Table.from_pylist(frame_rows, schema=PERSON_FRAMES_SCHEMA),
            PERSON_FRAMES_SCHEMA,
            extra_metadata={
                "video_id": ctx.video_id,
                "id_namespace": "person_id (YOLO/ByteTracker) — NOT TalkNet's track_id and "
                                "NOT a speaker id; never join these",
                "track_kind": "person (body) track, not a face track",
                "model": document.get("model"),
                "weights_sha256": document.get("weights_sha256"),
                "tracker": document.get("tracker"),
                "device": document.get("device"),
                # What the detector was allowed to call a person. Written into the file, not
                # left in the raw JSON, because `person_classes_only: false` lets a non-person
                # COCO class into a column named person_id and the table is then the only thing
                # a consumer opens.
                "coco_classes": json.dumps(document.get("parameters", {})
                                          .get("classes", [])),
                "conf": document.get("parameters", {}).get("conf"),
                "timestamp_column": "timestamp (source pts_seconds), the pipeline timeline",
            },
        )
        track_rows = person_track_rows(ctx.video_id, frame_rows,
                                      frames_measured=int(document.get("frames_measured")
                                                          or 0))
        write_table(
            ctx.artifact("person_tracks"),
            pa.Table.from_pylist(track_rows, schema=PERSON_TRACKS_SCHEMA),
            PERSON_TRACKS_SCHEMA,
            extra_metadata={
                "video_id": ctx.video_id,
                "id_namespace": "person_id (YOLO/ByteTracker) — NOT TalkNet's track_id and "
                                "NOT a speaker id; never join these",
                "track_kind": "person (body) track, not a face track",
                "model": document.get("model"),
                "weights_sha256": document.get("weights_sha256"),
                "person_count": document.get("person_ids"),
                "coco_classes": json.dumps(document.get("parameters", {})
                                          .get("classes", [])),
            },
        )

        summary = {
            "frames_measured": document.get("frames_measured"),
            "frames_with_person": document.get("frames_with_person"),
            "persons": len(track_rows),
            "max_persons_in_frame": document.get("max_persons_in_frame"),
            "device": document.get("device"),
        }
        ctx.scratch["persons"] = summary
        ctx.log(f"persons: {len(track_rows)} person id(s) over "
                f"{summary['frames_measured']} frame(s), "
                f"{summary['frames_with_person']} with at least one person")

        # Two things the worker knows and the tables cannot show, both of which change how
        # the counts should be read. Recorded in the raw JSON, but nobody reads that during
        # a batch — which is why activespeaker logs the same shape of news and why this
        # stage does too.
        fallback = document.get("device_fallback_reason")
        if fallback:
            ctx.log(f"YOLO did not use the requested device "
                    f"'{document.get('requested_device')}': {fallback}", logging.WARNING)
        gmc = document.get("gmc_failure_count")
        if gmc:
            # The symptom looks like a model-quality problem (fragmented ids) and is not
            # one: the tracker lost its camera-motion compensation and fell back to identity
            # warps. Fragmented ids are exactly what that produces, so a person count read
            # off this run is wrong in a way that has nothing to do with the weights.
            ctx.log(f"{gmc} frame(s) failed camera-motion compensation, so ids may be "
                    f"fragmented by the harness rather than by the scene "
                    f"({document.get('gmc_failure_reason') or 'cause not reported'}); "
                    f"treat the person count as suspect", logging.WARNING)
        return summary

    @staticmethod
    def _frame_row(video_id: str, row: dict[str, Any]) -> dict[str, Any]:
        """One raw detection to a schema-shaped row, keeping absence explicit.

        A raw artifact written before ``track_confidence`` existed carries no key; deriving
        it from what is present keeps old datasets normalising instead of failing, which is
        what a schema addition in a resumable pipeline owes them. A *present* but
        unrecognised reason is passed through untouched — rewriting it here would hide a
        malformed worker row behind a legal default, and ``validate`` is what has to name
        the frame that carries it.
        """
        confidence = row.get("confidence")
        track_confidence = row.get("track_confidence")
        reason = row.get("confidence_reason")
        if reason is None:
            reason = ("tracked" if track_confidence is not None
                      else "no_track_confidence")
        bbox = [row.get(key) for key in ("x1", "y1", "x2", "y2")]
        complete = all(value is not None for value in bbox)
        return {
            "schema_version": "1.0",
            "video_id": video_id,
            "frame_number": row.get("frame_number"),
            "timestamp": row.get("timestamp"),
            "person_id": row.get("person_id"),
            "x1": bbox[0],
            "y1": bbox[1],
            "x2": bbox[2],
            "y2": bbox[3],
            "confidence": confidence,
            "track_confidence": track_confidence,
            # Recomputed here rather than trusted from the worker: the row's own geometry
            # is what this column claims to be, and a normalizer that recomputes an area
            # cannot inherit a worker's arithmetic slip. Absent bbox means absent area, not
            # zero area — a zero would say "a person of no size was measured".
            "bbox_area": ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) if complete else None,
            "persons_in_frame": row.get("persons_in_frame"),
            "confidence_reason": reason,
        }

    # ---------------------------------------------------------------- validation

    @staticmethod
    def _duration(ctx: StageContext) -> float | None:
        path = ctx.artifact("metadata")
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("duration_seconds")
        except (OSError, ValueError):
            return None

    def validate(self, ctx: StageContext) -> dict[str, Any]:
        """Check the tables' shape, their vocabularies, and that they came from this config.

        Three layers, in the order that fails cheapest: the raw document must be this
        configuration's output; the columns must all be there; then the rows must be honest
        about what they measured.
        """
        document = self.validate_raw(ctx)
        if not raw_request_matches(ctx.artifact(self.raw_artifact), self.request_digest(ctx)):
            raise ValidationError(
                self.name, ["raw result was produced by a different configuration"])

        problems: list[str] = []
        missing_keys = [key for key in REQUIRED_RAW_KEYS if key not in document]
        if missing_keys:
            # A document without `frames_measured` cannot tell 0 people from no measurement,
            # which is the one distinction this stage exists to preserve.
            raise ValidationError(self.name,
                                  [f"raw document is missing {key} "
                                   f"(a pre-1.0 artifact; rerun the persons stage)"
                                   for key in missing_keys])

        for name, schema in (("person_frames", PERSON_FRAMES_SCHEMA),
                             ("person_tracks", PERSON_TRACKS_SCHEMA)):
            path = ctx.artifact(name)
            if not path.is_file():
                raise ValidationError(self.name, [f"{name} missing ({path.name})"])
            columns = set(table_columns(path))
            absent = [field.name for field in schema if field.name not in columns]
            if absent:
                # Naming the column is the whole diagnosis: it says which stage to rerun
                # instead of leaving a KeyError that the orchestrator records as a crash.
                raise ValidationError(self.name,
                                      [f"{path.name} missing columns: {', '.join(absent)}"])

        duration = self._duration(ctx)
        # The tables carry the model and checkpoint that produced them in file metadata, so a
        # dataset can be caught answering a different question than the raw document was
        # written from -- e.g. a table copied in from another video's run. Checked against the
        # raw document rather than the config, because validate's job is to describe what is
        # on disk: a config that has since moved on is the fingerprint's business.
        identity_problems = self._check_recorded_identity(ctx, document)
        problems.extend(identity_problems)
        frame_problems, frame_stats = self._check_frames(ctx, duration)
        problems.extend(frame_problems)
        track_problems = self._check_tracks(ctx, document, frame_stats)
        problems.extend(track_problems)
        if problems:
            raise ValidationError(self.name, problems)

        return {
            "frames_measured": document.get("frames_measured"),
            "frames_with_person": document.get("frames_with_person"),
            "frame_rows": frame_stats["rows"],
            "persons": frame_stats["persons"],
            "max_persons_in_frame": frame_stats["max_persons_in_frame"],
            "device": document.get("device"),
            "weights_sha256": document.get("weights_sha256"),
        }

    def _check_recorded_identity(self, ctx: StageContext,
                                document: dict[str, Any]) -> list[str]:
        """Do the Parquet files name the same model and weights as the raw document?

        Only a *stated* mismatch is reported. A table with no such metadata (written before
        the key existed) and a raw document that could not resolve a weights file (no
        ``weights_dir``, nothing on disk to hash) each prove nothing, and inventing a failure
        from an absent value would fail datasets that are simply older.
        """
        problems: list[str] = []
        for name in ("person_frames", "person_tracks"):
            path = ctx.artifact(name)
            recorded = self.recorded_identity(path)
            if recorded is None:
                continue
            for key in ("model", "weights_sha256"):
                stated = recorded.get(key)
                expected = document.get(key)
                if stated is not None and expected is not None and stated != expected:
                    problems.append(f"{path.name} states {key}={stated} while the raw "
                                    f"document it came from records {key}={expected}: the "
                                    f"table was not written from this raw output — rerun "
                                    f"the persons stage")
        return problems

    def _check_frames(self, ctx: StageContext,
                      duration: float | None) -> tuple[list[str], dict[str, Any]]:
        """Rows that are here are honest. Streams, so a 720k-frame run stays flat."""
        path = ctx.artifact("person_frames")
        columns = ("video_id", "frame_number", "timestamp", "person_id", "x1", "y1", "x2",
                   "y2", "confidence", "track_confidence", "bbox_area",
                   "persons_in_frame", "confidence_reason")
        problems: list[str] = []
        rows = 0
        ids: set[int] = set()
        video_ids: set[str] = set()
        # Recomputed for the cross-table check: persons per frame and the id set are what
        # the tracks table's counts have to agree with, and re-reading them here is what
        # makes "the two tables describe the same run" a checked property.
        per_frame: dict[int, int] = {}
        # frame -> the persons_in_frame values its own rows declare. Usually one value per
        # frame; more than one is itself the defect.
        declared_per_frame: dict[int, set[int]] = {}
        timestamps: list[float] = []
        for row in iter_rows(path, columns):
            rows += 1
            video_ids.add(str(row["video_id"]))
            timestamps.append(row["timestamp"])
            person_id = row["person_id"]
            if person_id is None:
                if len(problems) < 20:
                    problems.append(f"row {rows - 1} has no person_id: this table describes "
                                    f"people, and a row without an id describes nobody")
                continue
            ids.add(int(person_id))
            frame_number = int(row["frame_number"])
            per_frame[frame_number] = per_frame.get(frame_number, 0) + 1
            if row["persons_in_frame"] is not None:
                declared_per_frame.setdefault(frame_number, set()).add(
                    int(row["persons_in_frame"]))
            if len(problems) < 20:
                problem = self._check_frame_row(row)
                if problem is not None:
                    problems.append(f"row {rows - 1}: {problem}")
        # Interval check after the loop: one call, and check_intervals already caps what it
        # raises. Only when the rows themselves are clean, so a malformed row does not
        # produce a second complaint about the same data.
        if not problems:
            try:
                check_intervals(timestamps, timestamps, stage=self.name,
                                label="person frame timestamp", max_time=duration)
            except ValidationError as exc:
                problems.extend(exc.issues)
        if rows and video_ids != {ctx.video_id}:
            problems.append(f"rows belong to video(s) {sorted(video_ids)}, expected "
                            f"{ctx.video_id!r}")
        # `persons_in_frame` is the number a "how many people were on screen" query reads, and
        # it is copied onto every row of a frame by the worker. It is also the one per-frame
        # value a partial write can get wrong without breaking anything else: dropping the
        # second row of a two-person frame leaves a consistent table that says 2 while holding
        # 1, and the row-level check only proves the value is at least 1. So compare the
        # declared count against the rows actually here, once per frame, at the end.
        for frame_number in sorted(per_frame):
            declared = declared_per_frame.get(frame_number, set())
            if len(declared) > 1:
                if len(problems) < 20:
                    problems.append(
                        f"frame {frame_number} declares persons_in_frame={sorted(declared)} "
                        f"across its own rows but holds {per_frame[frame_number]}"
                    )
            elif declared and next(iter(declared)) != per_frame[frame_number]:
                if len(problems) < 20:
                    problems.append(
                        f"frame {frame_number} declares persons_in_frame="
                        f"{next(iter(declared))} but the table holds "
                        f"{per_frame[frame_number]} row(s) on it"
                    )
        max_in_frame = max(per_frame.values()) if per_frame else 0
        stats = {"rows": rows, "persons": len(ids), "max_persons_in_frame": max_in_frame,
                 "per_frame": per_frame, "ids": ids}
        return problems, stats

    @staticmethod
    def _check_frame_row(row: dict[str, Any]) -> str | None:
        """The first problem with one frame row, or None."""
        reason = row["confidence_reason"]
        if reason not in CONFIDENCE_REASONS:
            return (f"unrecognised confidence_reason {reason!r} (expected one of: "
                    f"{', '.join(CONFIDENCE_REASONS)})")
        has_track_confidence = row["track_confidence"] is not None
        if has_track_confidence and reason != "tracked":
            # The mirror of the check below, and the reason the vocabulary is a closed set
            # rather than a hint: "no tracker confidence" and "there is one" cannot both be
            # true of one row, and a reader branches on this string.
            return f"claims {reason!r} while carrying a track_confidence"
        if not has_track_confidence and reason == "tracked":
            return "claims tracked confidence while carrying none"
        confidence = row["confidence"]
        if confidence is None:
            return "carries no detection confidence"
        if not 0.0 <= float(confidence) <= 1.0:
            return f"detection confidence {confidence} is outside [0, 1]"
        coordinates = (row["x1"], row["y1"], row["x2"], row["y2"])
        if any(value is None or not math.isfinite(value) for value in coordinates):
            # Checked before any comparison: subtracting a missing coordinate raises
            # TypeError, which the orchestrator records as a stage crash instead of naming
            # the frame.
            return "has a person with missing bbox coordinates"
        if not (row["x1"] <= row["x2"] and row["y1"] <= row["y2"]):
            return f"has an inverted bbox ({row['x1']},{row['y1']},{row['x2']},{row['y2']})"
        area = row["bbox_area"]
        if area is None:
            return "has a complete bbox but no bbox_area"
        expected = (row["x2"] - row["x1"]) * (row["y2"] - row["y1"])
        if abs(float(area) - expected) > max(1e-6, abs(expected) * 1e-6):
            return f"bbox_area {area} does not match its own bbox ({expected})"
        if row["persons_in_frame"] is None:
            return "carries no persons_in_frame"
        if int(row["persons_in_frame"]) < 1:
            # This row's own existence is a person in this frame, so a count below 1 is
            # self-contradictory and every "people on screen at once" query goes wrong.
            return f"persons_in_frame is {row['persons_in_frame']} on a row that is a person"
        return None

    def _check_tracks(self, ctx: StageContext, document: dict[str, Any],
                      frame_stats: dict[str, Any]) -> list[str]:
        """The summary answers the same questions the frames table answers, consistently."""
        problems: list[str] = []
        rows = read_table(ctx.artifact("person_tracks")).to_pylist()
        declared = document.get("person_ids")
        if isinstance(declared, list) and set(declared) != frame_stats["ids"]:
            # The document's own claim about who appeared, against the table built from it.
            # A mismatch means the normaliser dropped or invented an identity.
            missing = sorted(set(declared) - frame_stats["ids"])
            extra = sorted(frame_stats["ids"] - set(declared))
            parts = []
            if missing:
                parts.append(f"declared but absent from the table: {missing[:8]}")
            if extra:
                parts.append(f"in the table but never declared: {extra[:8]}")
            problems.append("person ids disagree with the raw document — " + "; ".join(parts))
        if len(rows) != len(frame_stats["ids"]):
            problems.append(f"person_tracks has {len(rows)} row(s), the frames table has "
                            f"{len(frame_stats['ids'])} distinct person id(s)")
        # The per-frame count, re-derived here rather than read from the document: two
        # tables that each trust the raw JSON can both be wrong the same way, and
        # `max_persons_in_frame` is the number a reader quotes.
        document_max = document.get("max_persons_in_frame")
        if document_max is not None and document_max != frame_stats["max_persons_in_frame"]:
            problems.append(f"raw document declares max_persons_in_frame={document_max}, the "
                            f"frames table shows {frame_stats['max_persons_in_frame']}")
        measured = int(document.get("frames_measured") or 0)
        frames_with_person = int(document.get("frames_with_person") or 0)
        if frames_with_person != len(frame_stats["per_frame"]):
            problems.append(f"raw document declares {frames_with_person} frame(s) with a "
                            f"person, the frames table covers {len(frame_stats['per_frame'])}")
        if len(problems) >= 20:
            return problems

        per_id_frames: dict[int, list[dict[str, Any]]] = {}
        for row in iter_rows(ctx.artifact("person_frames"),
                             ("person_id", "timestamp", "confidence", "bbox_area",
                              "frame_number")):
            per_id_frames.setdefault(int(row["person_id"]), []).append(row)

        seen: set[int] = set()
        for index, row in enumerate(rows):
            person_id = row["person_id"]
            if person_id in seen:
                problems.append(f"track row {index} repeats person_id {person_id}: one "
                                f"identity must get exactly one summary row")
                continue
            seen.add(int(person_id))
            frames = per_id_frames.get(int(person_id), [])
            if not frames:
                problems.append(f"person_id {person_id} has a summary row and no frames")
                continue
            checks = (
                ("frame_count", len(frames)),
                ("first_timestamp", min(float(f["timestamp"]) for f in frames)),
                ("last_timestamp", max(float(f["timestamp"]) for f in frames)),
            )
            for column, expected in checks:
                actual = row[column]
                if actual is None:
                    problems.append(f"person_id {person_id} has a null {column}")
                    continue
                if abs(float(actual) - expected) > 1e-6:
                    problems.append(f"person_id {person_id} has {column}={actual}, its "
                                    f"frames give {expected}")
            duration = float(row["duration_seconds"] or 0.0)
            span = float(row["last_timestamp"] or 0.0) - float(row["first_timestamp"] or 0.0)
            if abs(duration - span) > 1e-6:
                problems.append(f"person_id {person_id} has duration_seconds={duration} "
                                f"spanning {span}")
            coverage = float(row["frame_coverage"] or 0.0)
            expected_coverage = (len(frames) / measured) if measured else 0.0
            if abs(coverage - expected_coverage) > 1e-3:
                problems.append(f"person_id {person_id} has frame_coverage={coverage}, "
                                f"{len(frames)}/{measured} is {expected_coverage:.4f}")
            if coverage > 1.0:
                # More frames credited to one person than the run measured: the denominator
                # was wrong, so every share in the table is inflated.
                problems.append(f"person_id {person_id} covers {coverage:.2%} of the "
                                f"measured frames, which is more than all of them")
            gap = float(row["longest_gap_seconds"] or 0.0)
            stamps = sorted(float(f["timestamp"]) for f in frames)
            expected_gap = max((b - a for a, b in zip(stamps, stamps[1:])), default=0.0)
            if abs(gap - expected_gap) > 1e-3:
                problems.append(f"person_id {person_id} has longest_gap_seconds={gap}, its "
                                f"frames give {expected_gap:.4f}")
            if len(problems) >= 20:
                break
        return problems

    def outputs_present(self, ctx: StageContext) -> bool:
        """Existence plus the identity of the weights that produced the tables.

        The base implementation checks that each declared output exists, which is not enough
        here. ``weights_dir`` may hold a checkpoint that was later replaced in place — same
        name, same path, different bytes — and a person count is only interpretable against
        the model that produced it (§20.2). Refusing reuse on that evidence sends the stage
        back through ``execute``, which re-runs the detector against the weights that are
        actually there now.
        """
        for name in self.outputs:
            path = ctx.paths.get(name)
            if path is None or not path.exists():
                return False
        return not self._foreign_frames(ctx)

    # ------------------------------------------------------- weights identity

    def frame_identity(self, ctx: StageContext) -> dict[str, Any]:
        """The model name plus the checkpoint digest this configuration will use.

        Both, not just the digest: switching ``model`` from ``yolo11n.pt`` to ``yolo11s.pt``
        with no ``weights_dir`` set leaves the stage unable to hash anything before the run, so
        a digest-only check would let a model change look reusable. ``None`` values mean
        "cannot say", and ``_foreign_frames`` treats those as *no evidence* rather than as
        evidence of a mismatch.
        """
        return {"model": ctx.config.persons.model,
                "weights_sha256": self._weights_digest(ctx)}

    def recorded_identity(self, path: Path) -> dict[str, Any] | None:
        """What the table on disk says produced it, or None when it cannot say.

        Reads file metadata only — no rows, no column projection, nothing to get wrong, and
        never a filename. None means "cannot tell", and every caller reads that as *keep the
        file*: an empty table is the honest result for a video with no people, a table written
        before these metadata keys existed proves nothing, and a broken file is ``validate``'s
        to report.
        """
        if not path.is_file():
            return None
        import pyarrow.parquet as pq

        try:
            metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
        except Exception:  # noqa: BLE001 - a broken file is validate()'s problem, not ours
            return None

        def key(name: str) -> str | None:
            raw = metadata.get(name.encode())
            if raw is None:
                return None
            return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)

        recorded = {"model": key("model"), "weights_sha256": key("weights_sha256")}
        if recorded["model"] is None and recorded["weights_sha256"] is None:
            return None
        return recorded

    def _foreign_frames(self, ctx: StageContext) -> dict[str, str]:
        """Declared outputs that name a different model or checkpoint than this config uses.

        Positive evidence only: a table that *states* an identity the configured checkpoint
        does not have. A table that states nothing (written before the key existed), and a
        checkpoint that has not been downloaded yet, each prove nothing and are never treated
        as foreign — the absence of evidence is not evidence of staleness, and deleting or
        recomputing on it would cost a GPU run to fix nothing.
        """
        wanted = self.frame_identity(ctx)
        found: dict[str, str] = {}
        for name in ("person_frames", "person_tracks"):
            recorded = self.recorded_identity(ctx.artifact(name))
            if recorded is None:
                continue
            differences = [f"{key}={recorded[key]}" for key in sorted(wanted)
                           if recorded.get(key) is not None and wanted.get(key) is not None
                           and recorded[key] != wanted[key]]
            if differences:
                asks = ", ".join(f"{key}={wanted[key]}"
                                 for key in ("model", "weights_sha256")
                                 if wanted.get(key) is not None)
                found[name] = ("written by " + ", ".join(differences)
                               + f"; this configuration asks for {asks}")
        return found

    def execute(self, ctx: StageContext) -> dict[str, Any]:
        extras = super().execute(ctx)
        # What the tables now on disk claim produced them, captured for `run` to report.
        # Read after the write rather than from the raw document so it is the file a consumer
        # will actually open, not the value the worker intended to write.
        ctx.scratch["persons_recorded_identity"] = (
            self.recorded_identity(ctx.artifact("person_frames")))
        return extras

    def run(self, ctx: StageContext):
        """Name, once per video, the checkpoint the person counts came from.

        Not a warning in the ordinary case: with ``weights_dir`` unset the stage could not hash
        the checkpoint before the run, so the digest in the fingerprint is None while the file
        carries the digest the worker read. That asymmetry is the point of the log line — a
        reader comparing two datasets can see they used different bytes without opening a raw
        JSON. When the stage *did* hash the checkpoint and the file it just wrote contradicts
        that, the worker resolved a different file than the stage did, which is worth a
        warning rather than a silently inconsistent dataset.
        """
        outcome = super().run(ctx)
        recorded = ctx.scratch.get("persons_recorded_identity")
        wanted = self.frame_identity(ctx)
        if recorded is None:
            return outcome
        if wanted["weights_sha256"] is not None and recorded.get("weights_sha256") not in (
                None, wanted["weights_sha256"]):
            ctx.log(f"person tables state weights_sha256="
                    f"{recorded['weights_sha256'][:12]}… while this run resolved "
                    f"{wanted['weights_sha256'][:12]}… — the worker loaded a checkpoint other "
                    f"than the one the stage hashed, so the counts and the fingerprint "
                    f"describe different models", logging.WARNING)
        elif wanted["weights_sha256"] is None and recorded.get("weights_sha256"):
            ctx.log(f"person counts come from {recorded.get('model') or wanted['model']} at "
                    f"weights_sha256={recorded['weights_sha256'][:12]}… "
                    f"(persons.weights_dir is unset, so ultralytics resolved the checkpoint; "
                    f"set it to make this reproducible without reading the table)")
        return outcome


def person_track_rows(video_id: str, frame_rows: list[dict[str, Any]], *,
                      frames_measured: int) -> list[dict[str, Any]]:
    """Collapse the frames table to one row per person id.

    Derived from the normalised frame rows rather than from the raw document so the two
    tables agree by construction: whatever survived normalisation is what gets summarised,
    which is the same reasoning as ``activespeaker.track_summary_rows``.

    ``frames_measured`` is the denominator for coverage and it is a parameter rather than
    ``len(frame_rows)`` on purpose: the frames table holds only frames where a person was
    detected, so dividing by its length would report "this person was on screen for 100% of
    the video" for a video that had one person in one frame out of two hundred.
    """
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in frame_rows:
        if row["person_id"] is not None:
            grouped.setdefault(int(row["person_id"]), []).append(row)

    rows: list[dict[str, Any]] = []
    for person_id in sorted(grouped):
        frames = sorted(grouped[person_id], key=lambda r: (r["frame_number"] is None,
                                                           r["frame_number"]))
        stamps = [float(row["timestamp"]) for row in frames
                  if row["timestamp"] is not None]
        confidences = [float(row["confidence"]) for row in frames
                       if row["confidence"] is not None]
        areas = [float(row["bbox_area"]) for row in frames if row["bbox_area"] is not None]
        first = min(stamps) if stamps else None
        last = max(stamps) if stamps else None
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        rows.append({
            "schema_version": "1.0",
            "video_id": video_id,
            "person_id": person_id,
            "first_timestamp": first,
            "last_timestamp": last,
            "duration_seconds": round(last - first, 6) if stamps else None,
            "frame_count": len(frames),
            "frame_coverage": (round(len(frames) / frames_measured, 4)
                               if frames_measured else None),
            # 0.0 for a single sighting: there is no gap in a person seen once, and null
            # there would make `MAX(longest_gap_seconds)` silently ignore the case.
            "longest_gap_seconds": round(max(gaps), 6) if gaps else 0.0,
            "mean_confidence": (round(sum(confidences) / len(confidences), 4)
                                if confidences else None),
            "max_confidence": round(max(confidences), 4) if confidences else None,
            "mean_bbox_area": round(sum(areas) / len(areas), 2) if areas else None,
            "max_bbox_area": round(max(areas), 2) if areas else None,
            # Filled after the loop, because "order of appearance" needs everybody's first
            # sighting, not this one's.
            "appearance_order": None,
        })

    # Ranked by first sighting, ties broken by id so the order is deterministic even when
    # two people first appear in the same frame.
    ordered = sorted(range(len(rows)),
                     key=lambda index: (rows[index]["first_timestamp"] is None,
                                        rows[index]["first_timestamp"],
                                        rows[index]["person_id"]))
    for order, index in enumerate(ordered):
        rows[index]["appearance_order"] = order
    return rows
