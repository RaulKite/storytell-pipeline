"""Publish a native OpenPose preview, muxing source audio without shell strings.

The native renderer's constant-rate preview is not the authoritative PTS timeline.
No normalized landmarks or native JSON are rewritten here.
"""
from __future__ import annotations

import os
import math
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from .exceptions import StageError
from .stages.metadata import ffprobe_streams
from .subprocess_utils import run_command


def check_skeleton_video(path: Path, *, ffprobe: str = 'ffprobe',
                         require_audio: bool = False,
                         expected_frames: int | None = None) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise StageError(f'OpenPose skeleton video is missing or empty: {path}')
    streams = ffprobe_streams(path, ffprobe=ffprobe).get('streams', [])
    kinds = {stream.get('codec_type') for stream in streams}
    if 'video' not in kinds or (require_audio and 'audio' not in kinds):
        raise StageError(f'OpenPose skeleton video lacks required streams: {path}')
    if expected_frames is not None:
        video = next(s for s in streams if s.get('codec_type') == 'video')
        try:
            count = int(video['nb_frames'])
            fps = float(Fraction(video['avg_frame_rate']))
            duration = float(video['duration'])
        except (KeyError, ValueError, ZeroDivisionError, TypeError) as exc:
            raise StageError(f'Cannot verify skeleton video completeness: {path}') from exc
        # Calibrated on seven real clips: native CFR/MP4 can lose 1–3 trailing
        # frames. Never use percentage slack: it could hide hours of missing data.
        if (abs(count - expected_frames) > 3 or not math.isfinite(fps) or fps <= 0
                or not math.isfinite(duration) or duration <= 0
                or abs(duration * fps - expected_frames) > 3.01):
            raise StageError(f'Incomplete skeleton video: {count} frames, '
                             f'expected {expected_frames} (at most 3-frame native slack)')


def publish_skeleton_video(native: Path, source: Path, destination: Path, *,
                           ffmpeg: str = 'ffmpeg', ffprobe: str = 'ffprobe',
                           timeout: float | None = None,
                           log_path: Path | None = None) -> Path:
    """Atomically replace the preview only after a successful mux and stream check.

    Audio is optional when the source genuinely has no audio stream. AAC makes
    source audio usable in MP4 without upstream's codec-copy/container failures.
    """
    check_skeleton_video(native, ffprobe=ffprobe)
    source_streams = ffprobe_streams(source, ffprobe=ffprobe).get('streams', [])
    has_audio = any(s.get('codec_type') == 'audio' for s in source_streams)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.skeleton-mux-', dir=destination.parent) as temp:
        candidate = Path(temp) / 'skeletons.mp4'
        argv = [ffmpeg, '-v', 'error', '-nostdin', '-y', '-i', str(native)]
        if has_audio:
            argv += ['-i', str(source), '-map', '0:v:0', '-map', '1:a:0',
                     '-c:v', 'copy', '-c:a', 'aac', '-shortest']
        else:
            argv += ['-map', '0:v:0', '-c:v', 'copy', '-an']
        argv += ['-movflags', '+faststart', str(candidate)]
        run_command(argv, timeout=timeout, log_path=log_path)
        check_skeleton_video(candidate, ffprobe=ffprobe, require_audio=has_audio)
        os.replace(candidate, destination)
    return destination
