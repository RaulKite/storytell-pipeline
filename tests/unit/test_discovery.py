"""Video discovery: extension matching, traversal, ordering, stable IDs."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from multimodal_pipeline.config import load_config
from multimodal_pipeline.discovery import discover_single, discover_videos, slugify


@pytest.fixture
def video_dir(tmp_path: Path) -> Path:
    root = tmp_path / "videos"
    root.mkdir()
    return root


def config_for(tmp_path: Path, video_dir: Path, **input_options):
    payload = {
        "project_root": str(tmp_path),
        "input": {"directory": str(video_dir), **input_options},
        "output": {"directory": str(tmp_path / "out")},
    }
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(payload))
    return load_config(path)


def touch(path: Path, content: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


class TestExtensions:
    def test_only_configured_extensions(self, tmp_path: Path) -> None:
        video_dir = tmp_path / "videos"
        for name in ("a.mp4", "b.MOV", "c.txt", "d.mkv", "e.jpg"):
            touch(video_dir / name)
        config = config_for(tmp_path, video_dir, extensions=[".mp4", ".mov"])
        found = [source.filename for source in discover_videos(config)]
        assert found == ["a.mp4", "b.MOV"]

    def test_extensions_are_case_insensitive(self, tmp_path: Path) -> None:
        video_dir = tmp_path / "videos"
        touch(video_dir / "CLIP.MP4")
        config = config_for(tmp_path, video_dir, extensions=[".mp4"])
        assert [s.filename for s in discover_videos(config)] == ["CLIP.MP4"]

    def test_no_extension_match_yields_empty(self, tmp_path: Path) -> None:
        video_dir = tmp_path / "videos"
        touch(video_dir / "notes.txt")
        config = config_for(tmp_path, video_dir)
        assert discover_videos(config) == []

    def test_directories_are_not_videos(self, tmp_path: Path) -> None:
        video_dir = tmp_path / "videos"
        (video_dir / "clip.mp4").mkdir(parents=True)
        config = config_for(tmp_path, video_dir)
        assert discover_videos(config) == []

    def test_missing_input_directory_raises(self, tmp_path: Path) -> None:
        config = config_for(tmp_path, tmp_path / "absent")
        with pytest.raises(FileNotFoundError):
            discover_videos(config)


class TestTraversal:
    def test_non_recursive_ignores_subdirectories(self, tmp_path: Path) -> None:
        video_dir = tmp_path / "videos"
        touch(video_dir / "top.mp4")
        touch(video_dir / "nested" / "deep.mp4")
        config = config_for(tmp_path, video_dir, recursive=False)
        assert [s.relative_path.as_posix() for s in discover_videos(config)] == ["top.mp4"]

    def test_recursive_includes_subdirectories(self, tmp_path: Path) -> None:
        video_dir = tmp_path / "videos"
        touch(video_dir / "top.mp4")
        touch(video_dir / "nested" / "deep.mp4")
        touch(video_dir / "nested" / "deeper" / "deepest.mov")
        config = config_for(tmp_path, video_dir, recursive=True, extensions=[".mp4", ".mov"])
        found = [s.relative_path.as_posix() for s in discover_videos(config)]
        assert sorted(found) == found
        assert set(found) == {"top.mp4", "nested/deep.mp4", "nested/deeper/deepest.mov"}


class TestOrdering:
    def test_order_is_deterministic_across_runs(self, tmp_path: Path) -> None:
        video_dir = tmp_path / "videos"
        for name in ("zeta.mp4", "alpha.mp4", "Middle.mp4"):
            touch(video_dir / name)
        config = config_for(tmp_path, video_dir)
        first = [s.video_id for s in discover_videos(config)]
        second = [s.video_id for s in discover_videos(config)]
        assert first == second == ["Middle", "alpha", "zeta"]

    def test_sort_uses_relative_path_not_absolute(self, tmp_path: Path) -> None:
        video_dir = tmp_path / "videos"
        touch(video_dir / "b" / "a.mp4")
        touch(video_dir / "a" / "z.mp4")
        config = config_for(tmp_path, video_dir, recursive=True)
        assert [s.relative_path.as_posix() for s in discover_videos(config)] == ["a/z.mp4", "b/a.mp4"]


class TestVideoIds:
    def test_stem_is_the_id_basis(self, tmp_path: Path) -> None:
        video_dir = tmp_path / "videos"
        touch(video_dir / "conversation_001.mp4")
        config = config_for(tmp_path, video_dir)
        source = discover_videos(config)[0]
        assert source.video_id == "conversation_001"
        assert source.filename == "conversation_001.mp4"

    def test_colliding_stems_get_hash_suffixes(self, tmp_path: Path) -> None:
        video_dir = tmp_path / "videos"
        touch(video_dir / "day1" / "session.mp4")
        touch(video_dir / "day2" / "session.mp4")
        config = config_for(tmp_path, video_dir, recursive=True)
        ids = [s.video_id for s in discover_videos(config)]
        assert len(ids) == 2 and len(set(ids)) == 2
        assert all(item.startswith("session-") for item in ids)

    def test_collision_suffix_is_stable(self, tmp_path: Path) -> None:
        video_dir = tmp_path / "videos"
        touch(video_dir / "day1" / "session.mp4")
        touch(video_dir / "day2" / "session.mp4")
        config = config_for(tmp_path, video_dir, recursive=True)
        assert [s.video_id for s in discover_videos(config)] == [
            s.video_id for s in discover_videos(config)
        ]

    def test_same_stem_different_extension_both_kept(self, tmp_path: Path) -> None:
        video_dir = tmp_path / "videos"
        touch(video_dir / "clip.mp4")
        touch(video_dir / "clip.mov")
        config = config_for(tmp_path, video_dir, extensions=[".mp4", ".mov"])
        ids = [s.video_id for s in discover_videos(config)]
        assert len(ids) == 2 and len(set(ids)) == 2

    def test_non_ascii_stem_is_slugged(self, tmp_path: Path) -> None:
        video_dir = tmp_path / "videos"
        touch(video_dir / "sesión piloto A.mp4")
        config = config_for(tmp_path, video_dir)
        video_id = discover_videos(config)[0].video_id
        assert video_id.isascii()
        assert video_id.startswith("ses")

    def test_slug_never_empty(self) -> None:
        assert slugify("###") == "video"
        assert slugify("   ") == "video"

    def test_slug_length_bounded(self) -> None:
        assert len(slugify("x" * 500)) <= 80


class TestDiscoverSingle:
    def test_single_file_outside_input_directory(self, tmp_path: Path) -> None:
        fixture = touch(tmp_path / "fixtures" / "sample.mp4")
        config = config_for(tmp_path, tmp_path / "videos")
        source = discover_single(fixture, config)
        assert source.video_id == "sample"
        assert source.path == fixture

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        config = config_for(tmp_path, tmp_path / "videos")
        with pytest.raises(FileNotFoundError):
            discover_single(tmp_path / "nope.mp4", config)

    def test_sibling_stem_collision_still_unique(self, tmp_path: Path) -> None:
        touch(tmp_path / "shots" / "take.mp4")
        target = touch(tmp_path / "shots" / "take.mov")
        config = config_for(tmp_path, tmp_path / "videos")
        source = discover_single(target, config)
        assert source.video_id != "take"
