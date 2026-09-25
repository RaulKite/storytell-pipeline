#!/usr/bin/env python3
"""TalkNet-ASD active-speaker worker: one video in, one per-frame document out.

Runs only inside ``environments/activespeaker``. Facts verified on this machine
(2026-09-24) rather than assumed from the repository README:

* ``run_talknet.py`` resolves ``--videoName`` with
  ``glob(videoFolder/videoName.*)``, so the converted file has to be named
  ``temp_video_25fps.avi`` and passed as ``--videoName temp_video_25fps``;
* it expects ``<videoFolder>/{pyavi,pyframes,pywork}`` and writes
  ``pywork/tracks.pckl`` + ``pywork/scores.pckl``;
* both checkpoints self-download through ``gdown`` into paths resolved against
  ``os.getcwd()`` -- ``pretrain_TalkSet.model`` (repo root) and
  ``model/faceDetector/s3fd/sfd_face.pth`` -- which is why the runner is invoked
  with ``cwd=talknet-root`` and why a read-only checkout needs ``--weights-dir``;
* ``tracks.pckl`` is a list of ``{track: {frame: int64[N], bbox: float[N,4]},
  proc_track: {x,y,s}, embeddings: float[N,512]}`` and ``scores.pckl`` is a
  parallel list of float arrays;
* **the score array is shorter than the track**: a 105-frame track produced 104
  scores, because MFCC windowing trims the tail. Scores are indexed by position
  inside the track, never by absolute frame number.

Torch is pinned to 2.5.1 in the environment, and deliberately not to something
newer: ``talkNet.py:84`` and ``model/faceDetector/s3fd/__init__.py:22`` call
``torch.load()`` without ``weights_only=``, whose default flipped to ``True`` in
torch 2.6, which rejects these 2021 checkpoints.

The stabilization algorithm (scene-clipped smoothing plus switch hysteresis) is
the maintainer's production logic and is reproduced faithfully: the point of the
stage is one row per output frame with gaps made explicit, so downstream code
can treat the timeline as dense.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

OUTPUT_FPS = 25
MAX_IMPUTED_TAIL_FRAMES = 2
TALKNET_VIDEO_NAME = "temp_video_25fps"

#: Why a frame row does or does not carry a TalkNet score. The stage publishes these in
#: its `frame_reason` column; ``unknown`` exists for Candidate values that were never
#: routed through the branch that knows (none in this worker's own pipeline).
REASON_SCORED = "scored"
REASON_IMPUTED_TAIL = "imputed_tail"
REASON_NO_FACE = "no_face"
REASON_SCORE_NOT_FINITE = "score_not_finite"
REASON_TRACK_HAS_NO_SCORES = "track_has_no_scores"
REASON_PAST_SCORED_TAIL = "past_scored_tail"
REASON_TAIL_SCORE_NOT_FINITE = "tail_score_not_finite"
REASON_UNKNOWN = "unknown"

#: gdown ids for the two checkpoints TalkNet expects relative to its cwd.
PRETRAIN_MODEL = ("pretrain_TalkSet.model", "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea")
S3FD_MODEL = ("model/faceDetector/s3fd/sfd_face.pth", "1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt")


class WorkerFailure(RuntimeError):
    """An unrecoverable worker problem, reported as ``status: "error"``."""


@dataclass
class Candidate:
    """One face track at one 25 FPS frame, scored or not.

    ``raw_score``/``smoothed_score`` are ``None`` when S3FD located this face but TalkNet
    never produced a usable score for the frame. Carrying the face anyway is the point:
    dropping it would erase the difference between "nobody was visible" and "we could
    not score the person who was". Every consumer must treat ``None`` as *no evidence*,
    never as a zero score.

    ``score_reason`` names *which* of those situations this is, set at the branch that
    decides it. Four different causes leave ``raw_score`` None and their rows come out
    byte-identical, so the cause has to be written down here, where it is still known:
    neither the position inside the track nor the score count is ever serialised.
    """

    track_id: int
    bbox: tuple[float, float, float, float]
    raw_score: float | None
    score_imputed: bool
    smoothed_score: float | None = None
    score_reason: str = REASON_UNKNOWN


# --------------------------------------------------------------------- helpers


def log(message: str) -> None:
    print(f"[activespeaker] {message}", flush=True)


def run_command(command: Sequence[str], stage: str, *, cwd: Path | None = None,
                capture: bool = False) -> subprocess.CompletedProcess[str]:
    log(stage)
    try:
        return subprocess.run([str(c) for c in command], cwd=str(cwd) if cwd else None,
                              check=True, text=True, capture_output=capture)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise WorkerFailure(f"{stage} failed with exit code {exc.returncode}"
                            + (f": {detail[-2000:]}" if detail else ".")) from exc
    except OSError as exc:
        raise WorkerFailure(f"{stage} could not start: {exc}") from exc


def require_tool(name: str) -> str:
    """Resolve a CLI, preferring the one installed beside this interpreter.

    ``scenedetect`` lives in the uv environment, which is not on the orchestrator's
    PATH; ``sys.executable`` is the venv python under ``uv run``, so its directory is
    the only location that is guaranteed to be the environment we are executing in.
    """
    beside = Path(sys.executable).parent / name
    if beside.is_file():
        return str(beside)
    found = shutil.which(name)
    if found is None:
        raise WorkerFailure(f"required executable not found: {name} "
                            f"(looked in {beside.parent} and PATH)")
    return found


def package_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


# ------------------------------------------------------------------- weights


def place_weights(root: Path, weights_dir: Path | None) -> list[str]:
    """Ensure TalkNet's two cwd-relative checkpoints exist; report what we did.

    The runner shells out to gdown when a weight is missing, and gdown fails
    *quietly* against a read-only directory: the real failure then surfaces as a
    ``torch.load`` error several minutes later. So a missing file with an unwritable
    destination is raised here, naming the path, instead.
    """
    actions: list[str] = []
    for relative, _gdown_id in (PRETRAIN_MODEL, S3FD_MODEL):
        target = root / relative
        if target.is_file():
            continue
        if weights_dir is None:
            actions.append(f"will download {relative} via gdown")
            continue
        source = weights_dir / Path(relative).name
        if not source.is_file():
            raise WorkerFailure(
                f"--weights-dir given but {source} is missing; refusing to fall back "
                "to an unattended gdown download"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        if not os.access(target.parent, os.W_OK):
            raise WorkerFailure(f"cannot stage {relative}: {target.parent} is not writable")
        shutil.copyfile(source, target)
        actions.append(f"staged {relative} from {source}")
    return actions


# ------------------------------------------------------------------ ffmpeg I/O


def convert_video(ffmpeg: str, video: Path, converted: Path) -> None:
    """Constant-rate 25 FPS, the only timeline TalkNet understands.

    ``fps=25`` (not ``-r``) is deliberate: it resamples by duplication/drop to a
    true CFR axis, so a frame number is a well-defined instant. VFR input would
    otherwise make TalkNet's frame indices and the scene CSV disagree.
    """
    run_command([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                 "-an", "-vf", f"fps={OUTPUT_FPS}", "-c:v", "mpeg4", "-q:v", "2",
                 str(converted)], "Converting video to constant-rate 25 FPS")


def extract_frames(ffmpeg: str, converted: Path, frames_dir: Path) -> int:
    run_command([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(converted),
                 "-q:v", "2", str(frames_dir / "%07d.jpg")], "Extracting 25 FPS frames")
    count = sum(1 for path in frames_dir.glob("*.jpg") if path.is_file())
    if count == 0:
        raise WorkerFailure("no frames were extracted from the converted video")
    return count


def source_timestamps(ffprobe: str, video: Path) -> list[float]:
    """Per-frame presentation times of the ORIGINAL video, normalized to zero.

    TalkNet works on a resampled 25 FPS axis, so without this mapping the output
    could only ever claim synthetic timestamps. Reading real PTS values and taking
    the nearest source frame keeps every row anchored to the pipeline timeline.
    """
    for field in ("best_effort_timestamp_time", "pts_time", "pkt_dts_time"):
        result = run_command(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames",
             "-show_entries", f"frame={field}",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            f"Reading source frame timestamps ({field})", capture=True)
        stamps = []
        for line in result.stdout.splitlines():
            try:
                value = float(line.strip())
            except ValueError:
                continue
            if math.isfinite(value):
                stamps.append(value)
        if stamps:
            stamps.sort()
            first = stamps[0]
            return [max(0.0, value - first) for value in stamps]
    raise WorkerFailure("ffprobe returned no usable frame timestamps")


def source_frame_rate(ffprobe: str, video: Path) -> float:
    result = run_command(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate,r_frame_rate", "-of", "json", str(video)],
        "Reading source frame rate", capture=True)
    try:
        stream = json.loads(result.stdout)["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        raise WorkerFailure("ffprobe returned no video stream metadata") from exc
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            rate = float(Fraction(stream.get(key) or "0"))
        except (ValueError, ZeroDivisionError, TypeError):
            continue
        if math.isfinite(rate) and rate > 0:
            return rate
    raise WorkerFailure("could not determine a valid source frame rate")


# ------------------------------------------------------------------ scenes


def detect_scenes(scenedetect: str, converted: Path, scenes_csv: Path) -> None:
    """Scene detection on the CONVERTED video, so cuts share TalkNet's frame axis."""
    run_command([scenedetect, "-i", str(converted), "list-scenes",
                 "--filename", str(scenes_csv), "--quiet"], "Detecting scenes")
    if not scenes_csv.is_file():
        raise WorkerFailure(f"scenedetect did not write {scenes_csv}")


def parse_scene_ids(scenes_csv: Path, frame_count: int) -> list[int]:
    """Map every zero-based frame index to a scene id, filling any CSV gaps."""
    with scenes_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    header_row = next((i for i, row in enumerate(rows)
                       if any(cell.strip() == "Scene Number" for cell in row)), None)
    if header_row is None:
        raise WorkerFailure("scenedetect CSV has no 'Scene Number' header row")
    headers = [cell.strip() for cell in rows[header_row]]
    try:
        scene_col = headers.index("Scene Number")
        start_col = headers.index("Start Frame")
        end_col = headers.index("End Frame")
    except ValueError as exc:
        raise WorkerFailure("scenedetect CSV is missing Scene Number/Start Frame/End Frame") from exc

    scene_ids: list[int | None] = [None] * frame_count
    for row in rows[header_row + 1:]:
        if len(row) <= max(scene_col, start_col, end_col):
            continue
        try:
            scene_id = int(row[scene_col].strip())
            # PySceneDetect frame numbers are one-based and inclusive; -1 start with
            # an exclusive end is the [start, end) interval we need.
            start = int(row[start_col].strip()) - 1
            end = int(row[end_col].strip())
        except ValueError:
            continue
        for index in range(max(0, start), min(frame_count, end)):
            scene_ids[index] = scene_id

    known = [sid for sid in scene_ids if sid is not None]
    if not known:
        raise WorkerFailure("scenedetect CSV contained no usable scene rows")
    # Boundary rounding can leave frames outside every listed scene. Fill backwards
    # then forwards so no frame is dropped, without inventing a new scene.
    following = known[0]
    for index in range(frame_count - 1, -1, -1):
        if scene_ids[index] is None:
            scene_ids[index] = following
        else:
            following = scene_ids[index]
    previous = scene_ids[0]
    for index in range(frame_count):
        if scene_ids[index] is None:
            scene_ids[index] = previous
        else:
            previous = scene_ids[index]
    return [int(sid) for sid in scene_ids]


# --------------------------------------------------------------- pkl -> frames


def load_pickle(path: Path) -> Any:
    """Load a 2021 TalkNet pickle, falling back to the legacy string encoding.

    These were written by Python 2 tooling in some distributions of TalkNet, and
    numpy 2 / Python 3.12 refuse the implicit bytes->str coercion. Trying the plain
    load first keeps the normal case honest rather than silently masking it.
    """
    try:
        with path.open("rb") as handle:
            return pickle.load(handle), "bytes"
    except (UnicodeDecodeError, ValueError):
        with path.open("rb") as handle:
            return pickle.load(handle, encoding="latin1"), "latin1"


def build_candidates(tracks: Any, scores: Any, scene_ids: Sequence[int]
                     ) -> list[dict[int, Candidate]]:
    """One {track_id: Candidate} map per frame, from position-aligned pkl arrays."""
    if not isinstance(tracks, list) or not isinstance(scores, list):
        raise WorkerFailure("TalkNet pkl outputs are not lists")
    by_frame: list[dict[int, Candidate]] = [{} for _ in range(len(scene_ids))]
    for track_id, track_data in enumerate(tracks):
        if track_id >= len(scores):
            continue
        try:
            frames = track_data["track"]["frame"]
            bboxes = track_data["track"]["bbox"]
            track_scores = list(scores[track_id])
        except (KeyError, TypeError) as exc:
            raise WorkerFailure(f"track {track_id} has an unexpected pkl structure") from exc
        score_count = len(track_scores)
        for position in range(min(len(frames), len(bboxes))):
            frame_index = int(frames[position])
            if not 0 <= frame_index < len(scene_ids):
                continue
            raw: float | None = None
            imputed = False
            if position < score_count:
                raw = float(track_scores[position])
                if not math.isfinite(raw):
                    # A NaN/inf score is as unusable as no score, but the face is real.
                    raw = None
                    reason = REASON_SCORE_NOT_FINITE
                else:
                    reason = REASON_SCORED
            elif score_count == 0:
                # Nothing to carry: this track never produced a score at all, which is a
                # different complaint from "we ran out of scores a few frames ago".
                reason = REASON_TRACK_HAS_NO_SCORES
            elif position < score_count + MAX_IMPUTED_TAIL_FRAMES:
                # MFCC windowing leaves the score array 1-2 samples short. Carry the
                # last score over that bounded tail and disclose it downstream.
                carried = float(track_scores[score_count - 1])
                if math.isfinite(carried):
                    raw, imputed = carried, True
                    reason = REASON_IMPUTED_TAIL
                else:
                    # Inside the window, and the only value available to carry is broken.
                    # Calling this past_scored_tail would say we ran out of scores.
                    reason = REASON_TAIL_SCORE_NOT_FINITE
            else:
                reason = REASON_PAST_SCORED_TAIL
            # Anything past the tail keeps raw=None: we may not invent a third score.
            # The face still gets a row, as tracked_unscored.
            try:
                bbox = tuple(float(value) for value in bboxes[position][:4])
            except (TypeError, ValueError, IndexError):
                continue
            if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
                # A garbage box is not a location. Inventing one would put a person
                # somewhere the detector never saw them, so this face is dropped.
                continue
            by_frame[frame_index][track_id] = Candidate(track_id, bbox, raw, imputed,
                                                        score_reason=reason)
    return by_frame


def smooth_scores(by_frame: Sequence[dict[int, Candidate]], scene_ids: Sequence[int],
                  window: int) -> None:
    """Centered moving average per track, clipped to the current scene.

    Clipping matters at cuts: a talk-head from the previous shot must not damp the
    score of the same track id continuing after the cut, and vice versa.
    """
    half = window // 2
    last = len(by_frame) - 1
    for index, candidates in enumerate(by_frame):
        scene = scene_ids[index]
        first, stop = max(0, index - half), min(last, index + half)
        for track_id, candidate in candidates.items():
            if candidate.raw_score is None:
                # Unscored stays unscored: smoothing cannot create the measurement we
                # refused to invent, and joining this track's average with nothing
                # would make its neighbours look less confident than they are.
                continue
            neighbours = [
                by_frame[neighbour][track_id].raw_score
                for neighbour in range(first, stop + 1)
                if scene_ids[neighbour] == scene and track_id in by_frame[neighbour]
                and by_frame[neighbour][track_id].raw_score is not None
            ]
            candidate.smoothed_score = sum(neighbours) / len(neighbours)


def select_stable(by_frame: Sequence[dict[int, Candidate]], scene_ids: Sequence[int],
                  switch_margin: float, switch_frames: int) -> list[Candidate | None]:
    """Hysteresis: keep the incumbent unless a challenger persistently beats it.

    Without this, per-frame argmax flickers between two faces whose scores are
    within noise of each other, which is useless for labelling who is speaking.
    """
    chosen: list[Candidate | None] = []
    current: int | None = None
    challenger: int | None = None
    wins = 0
    previous_scene: int | None = None
    for index, candidates in enumerate(by_frame):
        scene = scene_ids[index]
        if scene != previous_scene:
            # A cut is a new population: nothing carried over means anything.
            current = challenger = None
            wins = 0
            previous_scene = scene
        if not candidates:
            current = challenger = None
            wins = 0
            chosen.append(None)
            continue
        scored = [c for c in candidates.values() if c.smoothed_score is not None]
        if not scored:
            # Faces are visible here but nothing can be said about who is talking.
            # Selecting one would publish a guess as a label.
            current = challenger = None
            wins = 0
            chosen.append(None)
            continue
        best = max(scored, key=lambda c: (c.smoothed_score, -c.track_id))
        if current not in {c.track_id for c in scored}:
            current = best.track_id
            challenger, wins = None, 0
        elif best.track_id == current:
            challenger, wins = None, 0
        elif best.smoothed_score >= candidates[current].smoothed_score + switch_margin:
            if challenger == best.track_id:
                wins += 1
            else:
                challenger, wins = best.track_id, 1
            if wins >= switch_frames:
                current = best.track_id
                challenger, wins = None, 0
        else:
            challenger, wins = None, 0
        chosen.append(candidates[current])
    return chosen


def nearest_timestamp(target: float, stamps: Sequence[float]) -> float:
    index = bisect.bisect_left(stamps, target)
    if index == 0:
        return stamps[0]
    if index >= len(stamps):
        return stamps[-1]
    before, after = stamps[index - 1], stamps[index]
    return before if target - before <= after - target else after


# ---------------------------------------------------------------- TalkNet run


def resolve_device(requested: str) -> tuple[str, str | None]:
    if requested != "auto":
        return requested, None
    import torch

    if torch.cuda.is_available():
        return "cuda", None
    return "cpu", "torch.cuda.is_available() was False"


def run_talknet(talknet_root: Path, temp_dir: Path, scenes_csv: Path, audio: Path,
                device: str) -> tuple[Path, Path]:
    runner = talknet_root / "run_talknet.py"
    if not runner.is_file():
        raise WorkerFailure(f"TalkNet runner not found: {runner}")
    run_command([sys.executable, str(runner), "--pathToScenes", str(scenes_csv),
                 "--videoName", TALKNET_VIDEO_NAME, "--videoFolder", str(temp_dir),
                 "--audioFilePath", str(audio), "--device", device],
                f"Running TalkNet on {device}", cwd=talknet_root)
    work = temp_dir / "pywork"
    tracks, scores = work / "tracks.pckl", work / "scores.pckl"
    if not tracks.is_file() or not scores.is_file():
        raise WorkerFailure("TalkNet finished without writing tracks.pckl and scores.pckl")
    return tracks, scores


# --------------------------------------------------------------------- output


def build_document(*, video_id: str, source_fps: float, stamps: Sequence[float],
                   scene_ids: Sequence[int], chosen: Sequence[Candidate | None],
                   visible: Sequence[dict[int, Candidate]],
                   track_count: int, pickle_encoding: str, device: str,
                   requested_device: str, fallback_reason: str | None,
                   params: dict[str, Any]) -> dict[str, Any]:
    """Assemble the frame table from two different questions per frame.

    ``chosen`` answers *who is speaking* and can only ever hold a scored candidate;
    ``visible`` answers *which faces S3FD located*. face_status comes from the second,
    because a frame where a person was visible but unscored is neither "no face" nor
    "someone is speaking" -- collapsing those is the bug this table used to have.
    The lowest unscored track id is reported when several faces share the frame: it is
    arbitrary but deterministic, and the scores stay null so nothing pretends to know
    more than it does. ``score_reason`` travels with whichever face is reported, so the
    row's stated cause describes the same face its bbox and (absent) score describe; a
    frame with no candidates at all is ``no_face``, which no track can answer for it.
    """
    frames: list[dict[str, Any]] = []
    for index, candidate in enumerate(chosen):
        stamp = index / OUTPUT_FPS
        candidates = visible[index]
        if candidate is not None:
            face_status, located = "tracked", candidate
        elif candidates:
            face_status = "tracked_unscored"
            located = min(candidates.values(), key=lambda c: c.track_id)
        else:
            face_status, located = "no_face", None
        unscored = located is None or located.smoothed_score is None
        row: dict[str, Any] = {
            "frame_25fps": index,
            "timestamp_sec": round(stamp, 6),
            "source_timestamp_sec": round(nearest_timestamp(stamp, stamps), 6),
            "scene_id": scene_ids[index],
            "track_id": located.track_id if located is not None else None,
            "face_status": face_status,
            # no_face is a fact about the frame, not about a track: build_candidates
            # never ran for it, so the reason is set here rather than on a Candidate.
            "score_reason": (REASON_NO_FACE if located is None
                             else located.score_reason or REASON_UNKNOWN),
            "x1": None, "y1": None, "x2": None, "y2": None,
            "talknet_score_raw": None,
            "talknet_score": None,
            "score_imputed": False,
            "is_active_speaker": False,
        }
        if located is not None:
            row.update({
                # A located face is a located face whether or not it was scored; the
                # scores stay null so absence is never read as a measurement of zero.
                "x1": round(located.bbox[0], 3), "y1": round(located.bbox[1], 3),
                "x2": round(located.bbox[2], 3), "y2": round(located.bbox[3], 3),
                "talknet_score_raw": None if unscored else round(located.raw_score, 4),
                "talknet_score": None if unscored else round(located.smoothed_score, 4),
                "score_imputed": located.score_imputed if not unscored else False,
                "is_active_speaker": face_status == "tracked"
                                     and located.smoothed_score >= params["speaker_threshold"],
            })
        frames.append(row)
    return {
        "schema_version": "1.0",
        "video_id": video_id,
        "output_fps": OUTPUT_FPS,
        "source_fps": round(source_fps, 6),
        "device": device,
        "requested_device": requested_device,
        "device_fallback_reason": fallback_reason,
        "pickle_encoding": pickle_encoding,
        "parameters": params,
        "scene_count": len(set(scene_ids)),
        "frame_count": len(frames),
        "track_count": track_count,
        "active_frame_count": sum(1 for row in frames if row["is_active_speaker"]),
        "frames": frames,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Never leave a half-written artifact for the next resume to trust."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
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
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--talknet-root", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--video-id", default="unknown")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--speaker-threshold", type=float, default=0.0)
    parser.add_argument("--score-window", type=int, default=5)
    parser.add_argument("--switch-margin", type=float, default=0.5)
    parser.add_argument("--switch-frames", type=int, default=3)
    return parser.parse_args(argv)


def validate(args: argparse.Namespace) -> None:
    if not args.video.is_file():
        raise WorkerFailure(f"input video not found: {args.video}")
    if not args.audio.is_file():
        raise WorkerFailure(f"input audio not found: {args.audio}")
    if not args.talknet_root.is_dir():
        raise WorkerFailure(
            f"talknet root is not a directory: {args.talknet_root}. "
            "Clone TalkNet-ASD and point activespeaker.talknet_root at it."
        )
    if args.score_window <= 0 or args.score_window % 2 == 0:
        raise WorkerFailure("--score-window must be a positive odd integer")
    if args.switch_frames <= 0:
        raise WorkerFailure("--switch-frames must be a positive integer")
    if args.switch_margin < 0:
        raise WorkerFailure("--switch-margin must not be negative")
    for name in ("speaker_threshold", "switch_margin"):
        if not math.isfinite(getattr(args, name.replace("-", "_"))):
            raise WorkerFailure(f"--{name} must be finite")


def process(args: argparse.Namespace) -> dict[str, Any]:
    ffmpeg = require_tool("ffmpeg")
    ffprobe = require_tool("ffprobe")
    scenedetect = require_tool("scenedetect")

    source_fps = source_frame_rate(ffprobe, args.video)
    stamps = source_timestamps(ffprobe, args.video)
    device, fallback = resolve_device(args.device)
    staged = place_weights(args.talknet_root, args.weights_dir)
    for action in staged:
        log(action)

    with tempfile.TemporaryDirectory(prefix="activespeaker_") as name:
        temp = Path(name)
        frames_dir = temp / "pyframes"
        for sub in (frames_dir, temp / "pyavi", temp / "pywork"):
            sub.mkdir()
        converted = temp / f"{TALKNET_VIDEO_NAME}.avi"
        convert_video(ffmpeg, args.video, converted)
        frame_count = extract_frames(ffmpeg, converted, frames_dir)
        scenes_csv = temp / "scenes.csv"
        detect_scenes(scenedetect, converted, scenes_csv)

        tracks_path, scores_path = run_talknet(args.talknet_root, temp, scenes_csv,
                                               args.audio, device)
        tracks, pickle_encoding = load_pickle(tracks_path)
        scores, _ = load_pickle(scores_path)

        # Prefer the extracted-frame count: it is what the pkl frame indices count.
        expected = max(frame_count, max((len(t["track"]["frame"]) for t in tracks),
                                        default=0))
        scene_ids = parse_scene_ids(scenes_csv, expected)
        by_frame = build_candidates(tracks, scores, scene_ids)
        smooth_scores(by_frame, scene_ids, args.score_window)
        chosen = select_stable(by_frame, scene_ids, args.switch_margin, args.switch_frames)

        args.raw_dir.mkdir(parents=True, exist_ok=True)
        for source in (tracks_path, scores_path, scenes_csv):
            shutil.copyfile(source, args.raw_dir / source.name)

    params = {
        "speaker_threshold": args.speaker_threshold,
        "score_window": args.score_window,
        "switch_margin": args.switch_margin,
        "switch_frames": args.switch_frames,
    }
    document = build_document(
        video_id=args.video_id, source_fps=source_fps, stamps=stamps, scene_ids=scene_ids,
        chosen=chosen, visible=by_frame, track_count=len(tracks),
        pickle_encoding=pickle_encoding,
        device=device, requested_device=args.device, fallback_reason=fallback,
        params=params,
    )
    write_json_atomic(args.output_json, document)
    document["_weights_staged"] = staged
    return document


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload: dict[str, Any] = {"status": "error", "stage": "activespeaker"}
    try:
        validate(args)
        document = process(args)
        payload.update({
            "status": "ok",
            "tool_version": package_version("torch"),
            "model_version": "TalkNet (pretrain_TalkSet)",
            "frames": document["frame_count"],
            "tracks": document["track_count"],
            "scenes": document["scene_count"],
            "active_frames": document["active_frame_count"],
            "device": document["device"],
            "device_fallback_reason": document["device_fallback_reason"],
            "pickle_encoding": document["pickle_encoding"],
        })
        log(f"normalised {document['frame_count']} frames across "
            f"{document['track_count']} tracks, {document['active_frame_count']} active")
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
