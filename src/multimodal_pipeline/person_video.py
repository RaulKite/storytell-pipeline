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
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .exceptions import StageError
from .stages.metadata import ffprobe_streams


def check_track_video(path: Path, *, ffprobe: str = "ffprobe",
                      expected_frames: int | None = None,
                      require_audio: bool = False) -> dict[str, Any]:
    """Describe the preview, refusing anything that is not a complete render.

    Returns ``{"frames", "has_audio"}``. Every refusal names the file: this runs during
    ``validate`` on a dataset someone is trying to resume, and "the video is wrong" without a
    path sends them to the wrong stage.
    """
    if not path.is_file() or path.stat().st_size == 0:
        raise StageError(f"person track preview is missing or empty: {path}")
    streams = ffprobe_streams(path, ffprobe=ffprobe).get("streams", [])
    kinds = {stream.get("codec_type") for stream in streams}
    if "video" not in kinds:
        raise StageError(f"person track preview has no video stream: {path}")
    if require_audio and "audio" not in kinds:
        # The worker only promises audio when the source had a stream to mux, so a missing one
        # here is the container losing it, not the clip being silent.
        raise StageError(f"person track preview carries no audio although the source had an "
                         f"audio stream: {path}")
    video = next(stream for stream in streams if stream.get("codec_type") == "video")
    declared = video.get("nb_frames")
    frames: int | None = None
    if declared is not None:
        try:
            frames = int(declared)
        except (TypeError, ValueError) as exc:
            raise StageError(f"person track preview states an unreadable frame count "
                             f"({declared!r}): {path}") from exc
    if expected_frames is not None:
        if frames is None:
            # The encoder this stage uses always writes nb_frames. Its absence means the file
            # did not come from here, and deriving a count from duration would round a
            # truncated preview into passing.
            raise StageError(f"person track preview states no frame count, so its completeness "
                             f"cannot be verified against {expected_frames} measured frame(s): "
                             f"{path}")
        if frames != int(expected_frames):
            raise StageError(
                f"person track preview has {frames} frame(s), expected {expected_frames}: the "
                "preview would put every later overlay on the wrong frame. No slack is allowed "
                f"here because this render is produced by this repository's own encoder: {path}")
    return {"frames": frames, "has_audio": "audio" in kinds}
