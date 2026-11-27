"""Check the rendered person-track preview that the ``persons`` stage publishes.

The encoder deliberately lives in ``workers/persons_worker.py``: drawing needs cv2 and the
exact frames ultralytics decoded, and the orchestrator may import neither (AGENTS.md keeps the
heavy visual dependencies inside ``environments/persons`` and reaches them through
``uv run --project``). What this module owns is the *check* on the published file, which needs
nothing but ffprobe and a rule.

The rule is harsher than the OpenPose preview's on purpose. Upstream's renderer is a black box
that is measured to lose one to three trailing frames, so ``skeleton_video`` allows three. Here
the encoder is this repository's own, driven by a frame count the worker already compares
against the frames it measured, so a preview one frame short is a defect rather than a
tolerance: every overlay after the gap would sit on the wrong frame while the file still
played. It is also a preview, not a measurement — the authoritative timing stays in
``source/frame_index.parquet``, and the ``id=`` overlay is a ByteTracker track id, which
fragments and is not an identity.

**The frame count is counted, not read off the container.** ``nb_frames`` is the muxer's own
header field, written when the file was created, and a file truncated afterwards keeps it: a
reviewer's truncated preview still declared 125 frames while 61 decoded, which is exactly the
failure this check exists to catch. ``nb_read_frames`` comes from ``ffprobe -count_frames`` and
is what a player would actually show. It costs a decode of the preview, which is the price of
the guarantee; ``persons`` also records the file's sha256 in its completion state, so an
ordinary resume can prove the bytes are the ones already validated without trusting a count
alone.

The rate and the encoded size are checked as well, because a same-length replacement is
invisible to every other check: a 126-frame 25 fps file dropped over a 126-frame 29.97 fps
preview has the right frame count, the wrong duration, and would retime every overlay against
``frames.parquet``.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

from .exceptions import StageError

#: How far the file's average rate may sit from the rate the worker recorded, as a fraction of
#: that rate. ffmpeg writes the nearest rational to what it was told, so the difference is
#: rounding; a substitution is 20 % off at 25 against 29.97. 0.1 % separates the two without
#: being a tolerance on the *content* the way a frame-count slack would be.
FPS_TOLERANCE = 1e-3

#: ``-count_frames`` is what makes the count honest. Only the entries this module reads are
#: requested, and only the video stream is decoded: asking for every stream would decode a
#: four-hour audio track to answer a question about the picture. Audio presence comes from a
#: second, demux-only probe.
_COUNT_ENTRIES = "stream=codec_type,nb_read_frames,avg_frame_rate,width,height"


def _probe(argv: list[str], path: Path) -> dict[str, Any]:
    """One ffprobe call, with stderr kept out of the JSON.

    ``run_command`` merges stderr into stdout, which is right for a log and wrong here: a
    corrupt file makes ffprobe complain *and* print, and a parser that reads the mixture fails
    with "unparseable JSON" instead of saying what the file did. Separate streams keep the two
    apart, and the stderr tail still reaches the message.
    """
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=1800.0,
                                   check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise StageError(f"person track preview could not be probed ({path}): {exc}") from exc
    if completed.returncode != 0 and not (completed.stdout or "").strip():
        detail = (completed.stderr or "").strip()[-300:]
        raise StageError(f"ffprobe could not read the person track preview ({path}, exit "
                         f"{completed.returncode}): {detail or 'no output from ffprobe'}")
    try:
        return json.loads(completed.stdout or "{}")
    except ValueError as exc:
        raise StageError(f"ffprobe produced unparseable JSON for {path}: {exc}") from exc


def _video_stream(path: Path, *, ffprobe: str) -> dict[str, Any]:
    """The video stream, with ``nb_read_frames`` measured by decoding the file."""
    payload = _probe([ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
                      "-show_entries", _COUNT_ENTRIES, "-of", "json", str(path)], path)
    streams = payload.get("streams") or []
    if not streams:
        raise StageError(f"person track preview has no video stream: {path}")
    return streams[0]


def _has_audio(path: Path, *, ffprobe: str) -> bool:
    """Whether the container carries an audio stream, without decoding anything."""
    payload = _probe([ffprobe, "-v", "error", "-show_entries", "stream=codec_type",
                      "-of", "json", str(path)], path)
    return any(stream.get("codec_type") == "audio"
               for stream in payload.get("streams") or [])


def _integer(raw: Any) -> int | None:
    if isinstance(raw, bool):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _rate(raw: Any) -> float | None:
    """ffprobe's ``avg_frame_rate`` (``"30000/1001"``, ``"0/0"``, ``"25"``) as a float."""
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            value = float(numerator) / float(denominator)
        else:
            value = float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def check_track_video(path: Path, *, ffprobe: str = "ffprobe",
                      expected_frames: int | None = None,
                      expected_fps: float | None = None,
                      expected_size: tuple[int, int] | None = None,
                      require_audio: bool = False) -> dict[str, Any]:
    """Describe the preview, refusing anything that is not a complete render.

    Returns ``{"frames", "fps", "width", "height", "has_audio"}``. Every refusal names the
    file: this runs during ``validate`` on a dataset someone is trying to resume, and "the
    video is wrong" without a path sends them to the wrong stage.
    """
    if not path.is_file() or path.stat().st_size == 0:
        raise StageError(f"person track preview is missing or empty: {path}")
    video = _video_stream(path, ffprobe=ffprobe)
    has_audio = _has_audio(path, ffprobe=ffprobe)
    if require_audio and not has_audio:
        # The worker only promises audio when the source had a stream to mux, so a missing one
        # here is the container losing it, not the clip being silent.
        raise StageError(f"person track preview carries no audio although the source had an "
                         f"audio stream: {path}")
    frames = _integer(video.get("nb_read_frames"))
    fps = _rate(video.get("avg_frame_rate"))
    width, height = _integer(video.get("width")), _integer(video.get("height"))
    if frames is None:
        # Counting is the point: the header count is what a truncated file keeps, and deriving
        # a count from the duration would round a truncated preview into passing.
        raise StageError(f"person track preview's decoded frame count could not be read, so its "
                         f"completeness cannot be verified: {path}")
    if expected_frames is not None and frames != int(expected_frames):
        raise StageError(
            f"person track preview decodes to {frames} frame(s), expected {expected_frames}: the "
            "preview would put every later overlay on the wrong frame. No slack is allowed here "
            f"because this render is produced by this repository's own encoder: {path}")
    if expected_fps is not None:
        if fps is None:
            raise StageError(f"person track preview states no frame rate, so it cannot be held "
                             f"to the {expected_fps} fps the worker recorded: {path}")
        if abs(fps - float(expected_fps)) > float(expected_fps) * FPS_TOLERANCE:
            raise StageError(
                f"person track preview runs at {fps:.6f} fps, not the {float(expected_fps):.6f} "
                "fps the tracking run recorded: the frame count can match while every overlay "
                f"sits at the wrong instant, so this is not this run's render: {path}")
    if expected_size is not None:
        if (width, height) != (int(expected_size[0]), int(expected_size[1])):
            raise StageError(
                f"person track preview is {width}x{height}, not the "
                f"{int(expected_size[0])}x{int(expected_size[1])} the worker recorded encoding: "
                f"{path}")
    return {"frames": frames, "fps": fps, "width": width, "height": height,
            "has_audio": has_audio}
