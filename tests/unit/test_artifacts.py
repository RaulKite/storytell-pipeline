"""Artifact layout, atomic writes, the registry, manifest and batch report."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from multimodal_pipeline.artifacts import (
    ARTIFACT_LAYOUT,
    MANIFEST_ARTIFACTS,
    STAGE_LOG_NAMES,
    ArtifactRegistry,
    VideoPaths,
    atomic_write_json,
    atomic_write_text,
    read_json,
)
from multimodal_pipeline.exceptions import ValidationError
from multimodal_pipeline.manifest import build_manifest, validate_manifest, write_manifest
from multimodal_pipeline.report import BatchReport, collect_report, write_batch_report


@pytest.fixture
def paths(tmp_path: Path) -> VideoPaths:
    video_paths = VideoPaths(tmp_path / "dataset")
    video_paths.ensure_dirs()
    return video_paths


class TestLayout:
    def test_every_artifact_path_is_relative_and_safe(self) -> None:
        for name, relative in ARTIFACT_LAYOUT.items():
            candidate = Path(relative)
            assert not candidate.is_absolute(), name
            assert ".." not in candidate.parts, name
            assert not candidate.name.startswith("."), name

    def test_paths_are_unique(self) -> None:
        values = list(ARTIFACT_LAYOUT.values())
        assert len(values) == len(set(values))

    def test_stage_logs_are_declared_for_every_stage(self) -> None:
        from multimodal_pipeline.stages.base import STAGE_ORDER

        assert set(STAGE_LOG_NAMES) == set(STAGE_ORDER)

    def test_manifest_excludes_only_bookkeeping(self) -> None:
        assert set(MANIFEST_ARTIFACTS) == set(ARTIFACT_LAYOUT) - {"manifest", "status", "pipeline_log"}

    def test_each_modality_has_its_own_directory(self) -> None:
        top_levels = {Path(relative).parts[0] for relative in ARTIFACT_LAYOUT.values()}
        assert {"source", "audio", "speech", "translation", "linguistic", "acoustic",
                "pose", "logs", "provenance"} <= top_levels

    def test_raw_and_normalised_outputs_never_collide(self) -> None:
        raw = {name for name in ARTIFACT_LAYOUT if name.endswith("_raw")}
        for name in raw:
            assert "raw" in Path(ARTIFACT_LAYOUT[name]).parts, name

    def test_ensure_dirs_creates_the_full_tree(self, tmp_path: Path) -> None:
        video_paths = VideoPaths(tmp_path / "fresh")
        video_paths.ensure_dirs()
        assert (tmp_path / "fresh" / "speech" / "raw").is_dir()
        assert (tmp_path / "fresh" / "logs").is_dir()

    def test_unknown_artifact_raises(self, paths: VideoPaths) -> None:
        with pytest.raises(KeyError):
            paths.artifact("not_a_real_artifact")

    def test_get_returns_none_for_unknown(self, paths: VideoPaths) -> None:
        assert paths.get("not_a_real_artifact") is None

    def test_stage_log_path(self, paths: VideoPaths) -> None:
        assert paths.log("whisperx") == paths.dataset_dir / "logs" / "whisperx.log"


class TestAtomicWrites:
    def test_json_round_trip(self, paths: VideoPaths) -> None:
        atomic_write_json(paths.artifact("metadata"), {"a": 1, "b": [1, 2]})
        assert read_json(paths.artifact("metadata")) == {"a": 1, "b": [1, 2]}

    def test_no_temp_files_are_left(self, paths: VideoPaths) -> None:
        atomic_write_json(paths.artifact("metadata"), {"a": 1})
        assert list(paths.dataset_dir.rglob("*.tmp")) == []

    def test_parent_directories_are_created(self, tmp_path: Path) -> None:
        atomic_write_text(tmp_path / "deep" / "nested" / "f.txt", "x")
        assert (tmp_path / "deep" / "nested" / "f.txt").read_text() == "x"

    def test_unicode_is_preserved_not_escaped(self, paths: VideoPaths) -> None:
        atomic_write_json(paths.artifact("metadata"), {"text": "sesión ñ"})
        assert "sesión ñ" in paths.artifact("metadata").read_text(encoding="utf-8")

    def test_a_failing_rename_keeps_the_previous_content(self, paths: VideoPaths, monkeypatch) -> None:
        import os as os_module

        atomic_write_json(paths.artifact("metadata"), {"version": 1})
        original = paths.artifact("metadata").read_text()
        real_replace = os_module.replace

        def explode(src, dst, *args, **kwargs):
            if Path(dst).name == "metadata.json":
                raise OSError("disk full")
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(os_module, "replace", explode)
        with pytest.raises(OSError):
            atomic_write_json(paths.artifact("metadata"), {"version": 2})
        monkeypatch.undo()
        assert paths.artifact("metadata").read_text() == original
        assert list(paths.dataset_dir.rglob("*.tmp")) == []

    def test_concurrent_writers_never_produce_a_torn_file(self, paths: VideoPaths) -> None:
        payload = {"blob": "x" * 20000}

        def write(index: int) -> None:
            atomic_write_json(paths.artifact("metadata"), {**payload, "writer": index})

        threads = [threading.Thread(target=write, args=(i,)) for i in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        loaded = read_json(paths.artifact("metadata"))
        assert loaded["blob"] == "x" * 20000
        assert isinstance(loaded["writer"], int)

    def test_non_serialisable_values_are_stringified(self, paths: VideoPaths) -> None:
        atomic_write_json(paths.artifact("metadata"), {"path": Path("/tmp/x")})
        assert read_json(paths.artifact("metadata"))["path"] == "/tmp/x"


class TestRegistry:
    def test_only_existing_files_are_reported(self, paths: VideoPaths) -> None:
        registry = ArtifactRegistry(paths).refresh()
        assert registry.present == {}

    def test_a_written_artifact_is_described(self, paths: VideoPaths) -> None:
        atomic_write_json(paths.artifact("metadata"), {"a": 1})
        registry = ArtifactRegistry(paths).refresh()
        assert registry.has("metadata")
        info = registry.present["metadata"]
        assert info["path"] == "source/metadata.json"
        assert info["kind"] == "file"
        assert info["size_bytes"] > 0

    def test_a_directory_is_counted_not_sized_as_one_file(self, paths: VideoPaths) -> None:
        raw = paths.artifact("pose_raw")
        for index in range(3):
            (raw / f"f{index}.json").write_text("x" * 10)
        registry = ArtifactRegistry(paths).refresh()
        info = registry.present["pose_raw"]
        assert info["kind"] == "directory"
        assert info["file_count"] == 3
        assert info["size_bytes"] == 30

    def test_a_pre_created_empty_directory_is_not_an_artifact(self, paths: VideoPaths) -> None:
        """``ensure_dirs`` makes pose/raw and translation/raw; they must not be promised."""
        assert (paths.dataset_dir / "pose" / "raw").is_dir()
        registry = ArtifactRegistry(paths).refresh()
        assert not registry.has("pose_raw")
        assert not registry.has("translation_raw")
        manifest = build_manifest(video_id="v", metadata=METADATA, overall_status="completed",
                                 registry=registry, stage_statuses={})
        assert "pose_raw" in manifest["artifacts_not_generated"]

    def test_refresh_forgets_deleted_files(self, paths: VideoPaths) -> None:
        atomic_write_json(paths.artifact("metadata"), {"a": 1})
        registry = ArtifactRegistry(paths)
        registry.refresh()
        paths.artifact("metadata").unlink()
        registry.refresh()
        assert not registry.has("metadata")

    def test_refresh_can_be_scoped(self, paths: VideoPaths) -> None:
        atomic_write_json(paths.artifact("metadata"), {"a": 1})
        atomic_write_json(paths.artifact("provenance_config"), {"b": 2})
        registry = ArtifactRegistry(paths).refresh(["metadata"])
        assert set(registry.present) == {"metadata"}

    def test_describe_is_sorted_and_detached(self, paths: VideoPaths) -> None:
        atomic_write_json(paths.artifact("metadata"), {"a": 1})
        atomic_write_json(paths.artifact("provenance_config"), {"b": 2})
        registry = ArtifactRegistry(paths).refresh()
        described = registry.describe()
        assert list(described) == sorted(described)
        described["metadata"]["size_bytes"] = 999
        assert registry.present["metadata"]["size_bytes"] != 999


METADATA = {
    "source_filename": "clip.mp4",
    "source_path": "/in/clip.mp4",
    "SHA256": "a" * 64,
    "file_size_bytes": 100,
    "container": "mov,mp4,m4a",
    "duration_seconds": 12.5,
    "frame_rate_rational": "30000/1001",
    "average_frame_rate_rational": "30000/1001",
    "average_frame_rate_float": 29.97,
    "width": 640,
    "height": 480,
    "pixel_format": "yuv420p",
    "video_codec": "h264",
    "audio_codec": "aac",
    "audio_sample_rate": 48000,
    "audio_channels": 2,
    "frame_count": 375,
}


class TestManifest:
    def build(self, paths: VideoPaths, **overrides):
        atomic_write_json(paths.artifact("metadata"), METADATA)
        registry = ArtifactRegistry(paths).refresh()
        payload = build_manifest(
            video_id="vid", metadata=METADATA, overall_status="completed", registry=registry,
            stage_statuses={"metadata": "completed", "audio": "skipped"},
            detected_language="es", **overrides)
        return payload

    def test_source_identity_is_carried(self, paths: VideoPaths) -> None:
        manifest = self.build(paths)
        assert manifest["source"]["sha256"] == "a" * 64
        assert manifest["source"]["fps_rational"] == "30000/1001"
        assert manifest["source"]["detected_language"] == "es"
        assert manifest["video_id"] == "vid"

    def test_missing_optional_artifacts_are_declared_not_promised(self, paths: VideoPaths) -> None:
        manifest = self.build(paths)
        assert "speaker_turns" in manifest["artifacts_not_generated"]
        assert "speaker_turns" not in manifest["artifacts"]

    def test_paths_are_relative_to_the_dataset(self, paths: VideoPaths) -> None:
        manifest = self.build(paths)
        assert manifest["artifacts"]["metadata"] == "source/metadata.json"

    def test_temporal_model_is_declared(self, paths: VideoPaths) -> None:
        model = self.build(paths)["temporal_model"]
        assert model["unit"] == "seconds_from_video_start"
        assert set(model) == {"unit", "interval_columns", "instant_columns", "frame_columns"}

    def test_processing_status_and_stages(self, paths: VideoPaths) -> None:
        processing = self.build(paths)["processing"]
        assert processing["status"] == "completed"
        assert processing["stages"]["audio"] == "skipped"

    def test_validation_passes_for_a_real_dataset(self, paths: VideoPaths) -> None:
        manifest = self.build(paths)
        assert validate_manifest(paths, manifest)["all_present"] is True

    def test_a_promised_but_missing_file_fails(self, paths: VideoPaths) -> None:
        manifest = self.build(paths)
        paths.artifact("metadata").unlink()
        with pytest.raises(ValidationError, match="listed but missing"):
            validate_manifest(paths, manifest)

    @pytest.mark.parametrize("bad", ["/etc/passwd", "../outside.json", "../../escape.json"])
    def test_unsafe_paths_are_rejected(self, paths: VideoPaths, bad: str) -> None:
        manifest = self.build(paths)
        manifest["artifacts"]["metadata"] = bad
        with pytest.raises(ValidationError, match="unsafe relative path"):
            validate_manifest(paths, manifest)

    def test_a_symlink_to_outside_the_dataset_is_rejected(self, paths: VideoPaths) -> None:
        outside = paths.dataset_dir.parent / "secret.json"
        outside.write_text("{}")
        manifest = self.build(paths)
        manifest["artifacts"]["audio"] = "../secret.json"
        with pytest.raises(ValidationError, match="unsafe relative path"):
            validate_manifest(paths, manifest)

    def test_empty_artifact_list_is_rejected(self, paths: VideoPaths) -> None:
        with pytest.raises(ValidationError, match="no artifacts"):
            validate_manifest(paths, {"schema_version": "1.0", "video_id": "v", "artifacts": {}})

    def test_missing_identity_is_rejected(self, paths: VideoPaths) -> None:
        manifest = self.build(paths)
        manifest.pop("video_id")
        with pytest.raises(ValidationError, match="video_id"):
            validate_manifest(paths, manifest)

    def test_write_then_read(self, paths: VideoPaths) -> None:
        manifest = self.build(paths)
        write_manifest(paths, manifest)
        assert read_json(paths.manifest)["video_id"] == "vid"

    def test_a_hidden_file_never_enters_the_manifest(self, paths: VideoPaths) -> None:
        atomic_write_json(paths.artifact("metadata"), METADATA)
        (paths.dataset_dir / "source" / ".hidden.tmp").write_text("junk")
        registry = ArtifactRegistry(paths).refresh()
        manifest = build_manifest(video_id="v", metadata=METADATA, overall_status="completed",
                                  registry=registry, stage_statuses={})
        assert all(not Path(p).name.startswith(".") for p in manifest["artifacts"].values())


class TestBatchReport:
    def build(self, *entries, discovered: int | None = None, tmp_path=None) -> BatchReport:
        return collect_report(
            [SimpleNamespace(to_dict=lambda entry=entry: entry) for entry in entries],
            discovered=discovered if discovered is not None else len(entries),
            elapsed_seconds=1.25,
            output_directory=tmp_path or Path("/out"),
        )

    def test_summary_counts(self) -> None:
        report = self.build(
            {"video_id": "a", "status": "completed", "duration_seconds": 10},
            {"video_id": "b", "status": "partial", "duration_seconds": 5},
            {"video_id": "c", "status": "failed", "duration_seconds": 1},
        )
        summary = report.to_dict()["summary"]
        assert summary["completed"] == 1
        assert summary["partial"] == 1
        assert summary["failed"] == 1

    def test_partial_covers_every_non_terminal_status(self) -> None:
        report = self.build(
            {"video_id": "a", "status": "skipped", "duration_seconds": 0},
            {"video_id": "b", "status": "running", "duration_seconds": 0},
        )
        assert report.to_dict()["summary"]["partial"] == 2

    def test_discovered_can_exceed_processed(self) -> None:
        report = self.build({"video_id": "a", "status": "completed", "duration_seconds": 0},
                           discovered=7)
        summary = report.to_dict()["summary"]
        assert summary["videos_discovered"] == 7
        assert summary["processed_in_this_run"] == 1

    def test_write_produces_valid_json(self, tmp_path: Path) -> None:
        report = self.build({"video_id": "a", "status": "completed", "duration_seconds": 2.5},
                           tmp_path=tmp_path)
        path = write_batch_report(report)
        payload = json.loads(path.read_text())
        assert payload["videos"][0]["video_id"] == "a"
        assert payload["summary"]["elapsed_seconds"] == 1.25
        assert payload["schema_version"] and payload["pipeline_version"]

    def test_report_is_written_into_the_output_directory(self, tmp_path: Path) -> None:
        report = self.build({"video_id": "a", "status": "completed", "duration_seconds": 0},
                           tmp_path=tmp_path)
        assert write_batch_report(report) == tmp_path / "batch_report.json"

    def test_videos_keep_input_order(self, tmp_path: Path) -> None:
        report = self.build(*[{"video_id": name, "status": "completed", "duration_seconds": 0}
                             for name in ("z", "a", "m")], tmp_path=tmp_path)
        payload = json.loads(write_batch_report(report).read_text())
        assert [video["video_id"] for video in payload["videos"]] == ["z", "a", "m"]

    def test_masked_credentials_survive_and_secrets_do_not_appear(self, tmp_path: Path) -> None:
        report = self.build({"video_id": "a", "status": "failed", "duration_seconds": 0,
                             "errors": {"whisperx": {
                                 "command": ["uv", "run", "--api-key", "***masked***"]}}},
                            tmp_path=tmp_path)
        text = write_batch_report(report).read_text()
        assert "***masked***" in text and "sk-secret" not in text

    def test_empty_batch_is_still_a_valid_report(self, tmp_path: Path) -> None:
        report = self.build(discovered=0, tmp_path=tmp_path)
        payload = json.loads(write_batch_report(report).read_text())
        assert payload["videos"] == []
        assert payload["summary"]["completed"] == 0

    def test_generated_at_is_utc_z(self) -> None:
        assert self.build({"video_id": "a", "status": "completed", "duration_seconds": 0} \
                          ).to_dict()["generated_at"].endswith("Z")
