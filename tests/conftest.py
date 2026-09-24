"""Shared fixtures: a synthetic project, config and stage context."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from multimodal_pipeline.artifacts import ArtifactRegistry, VideoPaths
from multimodal_pipeline.config import PipelineConfig, load_config
from multimodal_pipeline.discovery import VideoSource
from multimodal_pipeline.log import StageLogger
from multimodal_pipeline.state import VideoState
from multimodal_pipeline.stages.base import STAGE_ORDER, StageContext

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def hermetic_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the suite independent of the credentials installed on this machine.

    Two independent leaks are closed here:

    * ``load_config`` reads the project-root ``.env``, so a developer's real
      ``HF_TOKEN`` would decide whether assertions such as "diarization skipped
      because the token is missing" pass -- green on CI, red at the desk that has
      credentials. The opt-out variable stops only that implicit read.
    * A test that loads a ``.env`` mutates ``os.environ`` in place, and pytest shares
      one process, so those values would leak into every later test in the same run --
      including the e2e test that asserts an unset token is reported as unset.
    """
    monkeypatch.setenv("MULTIMODAL_PIPELINE_NO_DOTENV", "1")
    before = dict(os.environ)
    try:
        yield
    finally:
        for key in list(os.environ):
            if key not in before:
                del os.environ[key]
        os.environ.update(before)


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    (root / "output").mkdir()
    (root / "videos").mkdir()
    return root


@pytest.fixture
def config_dict(project_root: Path) -> dict:
    return {
        "project_root": str(project_root),
        "input": {"directory": str(project_root / "videos"), "recursive": False},
        "output": {"directory": str(project_root / "output")},
        "whisperx": {"uv_project": "environments/whisperx", "model": "large-v3"},
        "diarization": {"uv_project": "environments/diarization"},
        "translation": {"provider": "mock", "model": "test-model", "api_key": "sk-secret-123456"},
        "spacy": {"uv_project": "environments/spacy"},
        "acoustic": {"uv_project": "environments/acoustic"},
        "openpose": {"root": "/opt/openpose"},
    }


@pytest.fixture
def config_file(project_root: Path, config_dict: dict) -> Path:
    path = project_root / "config" / "config.local.yaml"
    path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")
    return path


@pytest.fixture
def config(config_file: Path) -> PipelineConfig:
    return load_config(config_file)


def make_source(video_dir: Path, name: str = "conversation_001.mp4") -> VideoSource:
    path = video_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"stub-video-bytes")
    return VideoSource(path=path, relative_path=Path(name), video_id=Path(name).stem)


@pytest.fixture
def context(config: PipelineConfig, tmp_path: Path) -> StageContext:
    source = make_source(config.input.directory)
    dataset = config.output.directory / source.video_id
    paths = VideoPaths(dataset)
    paths.ensure_dirs()
    state = VideoState.load(paths, source.video_id, str(source.path))
    state.bind_stages(STAGE_ORDER)
    registry = ArtifactRegistry(paths).refresh()
    logger = StageLogger(paths.dataset_dir / "logs", "test")
    ctx = StageContext(
        config=config,
        source=source,
        paths=paths,
        state=state,
        registry=registry,
        log=logger,
        tools={"schema_version": "1.0"},
    )
    yield ctx  # type: ignore[misc]
    logger.close()


@pytest.fixture
def write_parquet():
    def _write(path: Path, rows: list[dict], schema=None) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows) if schema is None else pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, path)
        return path

    return _write


def requires_ffmpeg() -> bool:
    from shutil import which

    return which("ffmpeg") is not None and which("ffprobe") is not None


def make_test_video(path: Path, *, seconds: float = 2.0, fps: int = 30, size: str = "160x120",
                    rate: str | None = None, with_audio: bool = True) -> Path:
    """Synthetic test video: colour bars + optional sine tone. Fast to encode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    video_src = f"testsrc=size={size}:rate={fps}:duration={seconds}"
    inputs = ["-f", "lavfi", "-i", video_src]
    maps = ["-map", "0:v:0"]
    if with_audio:
        tone = f"sine=frequency=440:sample_rate=48000:duration={seconds}"
        inputs += ["-f", "lavfi", "-i", tone]
        maps += ["-map", "1:a:0", "-c:a", "aac", "-b:a", "96k", "-shortest"]
    argv = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs, *maps,
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)]
    if rate:
        argv = argv[:-1] + ["-r", rate, str(path)]
    subprocess.run(argv, check=True, capture_output=True)
    return path


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> Path:
    if not requires_ffmpeg():
        pytest.skip("ffmpeg not available")
    path = tmp_path_factory.mktemp("media") / "sample.mp4"
    return make_test_video(path, seconds=2.0, fps=30)
