"""Video discovery and stable video IDs.

Ordering is deterministic (sorted by relative path) so a rerun processes the
same videos in the same order, and IDs are derived from the filename stem with
a short hash suffix only when two sources would otherwise collide.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import PipelineConfig

# Anything not alphanumeric/_/- collapses to a single underscore.
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_-]+")
# Leading/trailing separators left by non-latin filenames (e.g. "sesión → ses_n").
_EDGE_RE = re.compile(r"^[_-]+|[_-]+$")


@dataclass(frozen=True)
class VideoSource:
    path: Path
    relative_path: Path
    video_id: str

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def stem(self) -> str:
        return self.path.stem


def slugify(stem: str, max_length: int = 80) -> str:
    slug = _EDGE_RE.sub("", _UNSAFE_RE.sub("_", stem))
    if not slug:
        slug = "video"
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("_-") or "video"
    return slug


def path_hash(relative_path: Path, length: int = 8) -> str:
    return hashlib.sha256(str(relative_path).encode("utf-8")).hexdigest()[:length]


def discover_videos(config: PipelineConfig, root: Path | None = None) -> list[VideoSource]:
    """Return supported videos under ``input.directory`` in deterministic order."""
    base = root or config.input.directory
    base = Path(base).expanduser()
    if not base.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {base}")
    extensions = {ext.lower() for ext in config.input.extensions}
    iterator = base.rglob("*") if config.input.recursive else base.glob("*")
    candidates: list[Path] = []
    for entry in iterator:
        try:
            if not entry.is_file():
                continue
        except OSError:  # unreadable entry: skip rather than abort the batch
            continue
        if entry.suffix.lower() in extensions:
            candidates.append(entry)
    candidates.sort(key=lambda p: str(p.relative_to(base)))

    stems = {p: slugify(p.stem) for p in candidates}
    counts: dict[str, int] = {}
    for path in candidates:
        counts[stems[path]] = counts.get(stems[path], 0) + 1

    sources: list[VideoSource] = []
    used: set[str] = set()
    for path in candidates:
        stem = stems[path]
        relative = path.relative_to(base)
        video_id = stem if counts[stem] == 1 else f"{stem}-{path_hash(relative)}"
        if video_id in used:  # two identical relative paths cannot exist, but stay defensive
            video_id = f"{video_id}-{path_hash(relative)}"
        used.add(video_id)
        sources.append(VideoSource(path=path, relative_path=relative, video_id=video_id))
    return sources


def discover_single(path: Path | str, config: PipelineConfig | None = None) -> VideoSource:
    """Build a :class:`VideoSource` for one explicit file (``process-video``)."""
    video_path = Path(path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"video file not found: {video_path}")
    base = video_path.parent
    relative = video_path.relative_to(base)
    video_id = slugify(video_path.stem)
    # A sibling sharing the stem (even with another extension) must not silently
    # share a dataset directory, so both sides of the collision get a hash.
    same_stem = [p for p in base.iterdir() if p.is_file() and p.stem == video_path.stem]
    if len(same_stem) > 1:
        video_id = f"{video_id}-{path_hash(relative)}"
    return VideoSource(path=video_path, relative_path=relative, video_id=video_id)


def find_video_by_id(config: PipelineConfig, video_id: str) -> VideoSource | None:
    for source in discover_videos(config):
        if source.video_id == video_id:
            return source
    return None
