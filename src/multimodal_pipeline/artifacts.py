"""Per-video output layout and the artifact registry.

One dataset directory per video. ``paths.py``-style knowledge lives here so a
stage never concatenates output paths by hand and the manifest is always
generated from the same registry the stages wrote into.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Logical artifact name -> path relative to the video dataset directory.
ARTIFACT_LAYOUT: dict[str, str] = {
    "metadata": "source/metadata.json",
    "audio": "audio/audio.wav",
    "frame_index": "source/frame_index.parquet",
    "whisperx_raw": "speech/raw/whisperx.json",
    "diarization_raw": "speech/raw/diarization.json",
    "diarization_rttm": "speech/raw/diarization.rttm",
    "exclusive_diarization_raw": "speech/raw/exclusive_diarization.json",
    "speech_segments": "speech/segments.parquet",
    "speech_words": "speech/words.parquet",
    "speaker_turns": "speech/speaker_turns.parquet",
    "translation_raw": "translation/raw",
    "translation_segments": "translation/segments_en.parquet",
    "spacy_source_raw": "linguistic/source/raw/spacy_source.json",
    "spacy_english_raw": "linguistic/english/raw/spacy_english.json",
    "spacy_source_tokens": "linguistic/source/tokens.parquet",
    "spacy_source_sentences": "linguistic/source/sentences.parquet",
    "spacy_english_tokens": "linguistic/english/tokens.parquet",
    "spacy_english_sentences": "linguistic/english/sentences.parquet",
    "acoustic_raw": "acoustic/raw/acoustic_features.jsonl",
    "acoustic_frames": "acoustic/frame_features.parquet",
    "acoustic_segments": "acoustic/segment_features.parquet",
    "pose_raw": "pose/raw",
    "pose_body": "pose/body.parquet",
    "pose_hands": "pose/hands.parquet",
    "pose_face": "pose/face.parquet",
    "pipeline_log": "logs/pipeline.log",
    "provenance_config": "provenance/config.json",
    "provenance_tools": "provenance/tools.json",
    "provenance_processing": "provenance/processing.json",
    "manifest": "manifest.json",
    "status": "status.json",
}

STAGE_LOG_NAMES = (
    "metadata",
    "audio",
    "whisperx",
    "diarization",
    "speaker_assignment",
    "translation",
    "spacy_source",
    "spacy_english",
    "acoustic",
    "openpose",
    "finalization",
)

# Names used by the manifest's ``artifacts`` block (everything except bookkeeping files).
MANIFEST_ARTIFACTS = tuple(
    name for name in ARTIFACT_LAYOUT if name not in {"manifest", "status", "pipeline_log"}
)


def stage_log(name: str) -> str:
    return f"logs/{name}.log"


@dataclass
class VideoPaths:
    """Every path a stage may need for one video dataset."""

    dataset_dir: Path

    def __post_init__(self) -> None:
        self.dataset_dir = Path(self.dataset_dir)

    @property
    def status(self) -> Path:
        return self.dataset_dir / "status.json"

    @property
    def manifest(self) -> Path:
        return self.dataset_dir / "manifest.json"

    def artifact(self, name: str) -> Path:
        try:
            relative = ARTIFACT_LAYOUT[name]
        except KeyError:  # pragma: no cover - programming error
            raise KeyError(f"unknown artifact name: {name}") from None
        return self.dataset_dir / relative

    def get(self, name: str) -> Path | None:
        return self.artifact(name) if name in ARTIFACT_LAYOUT else None

    def log(self, stage: str) -> Path:
        return self.dataset_dir / stage_log(stage)

    def ensure_dirs(self) -> None:
        for relative in {
            "source",
            "audio",
            "speech/raw",
            "translation/raw",
            "linguistic/source/raw",
            "linguistic/english/raw",
            "acoustic/raw",
            "pose/raw",
            "logs",
            "provenance",
        }:
            (self.dataset_dir / relative).mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: Path, text: str) -> Path:
    """Write via temp file + ``os.replace`` so a crash never leaves half a JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return path


def atomic_write_json(path: Path, payload: Any) -> Path:
    return atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class ArtifactRegistry:
    """Which logical artifacts exist on disk right now, and how big they are."""

    paths: VideoPaths
    present: dict[str, dict[str, Any]] = field(default_factory=dict)

    def refresh(self, names: Iterable[str] | None = None) -> "ArtifactRegistry":
        targets = names or ARTIFACT_LAYOUT.keys()
        self.present = {}
        for name in targets:
            path = self.paths.get(name)
            if path is None:
                continue
            if path.is_dir():
                entries = [p for p in path.rglob("*") if p.is_file()]
                self.present[name] = {
                    "path": str(path.relative_to(self.paths.dataset_dir)),
                    "kind": "directory",
                    "file_count": len(entries),
                    "size_bytes": sum(p.stat().st_size for p in entries),
                }
            elif path.is_file():
                self.present[name] = {
                    "path": str(path.relative_to(self.paths.dataset_dir)),
                    "kind": "file",
                    "size_bytes": path.stat().st_size,
                }
        return self

    def has(self, name: str) -> bool:
        return name in self.present

    def describe(self) -> dict[str, dict[str, Any]]:
        return {name: dict(info) for name, info in sorted(self.present.items())}
