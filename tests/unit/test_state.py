"""Processing state: transitions, atomic writes, resume and invalidation inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from multimodal_pipeline.artifacts import VideoPaths, atomic_write_json, read_json
from multimodal_pipeline.state import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_SKIPPED,
    VideoState,
)


@pytest.fixture
def paths(tmp_path: Path) -> VideoPaths:
    video_paths = VideoPaths(tmp_path / "dataset")
    video_paths.ensure_dirs()
    return video_paths


def load(paths: VideoPaths) -> VideoState:
    state = VideoState.load(paths, "vid", "/src/vid.mp4")
    state.bind_stages(["metadata", "audio", "whisperx"])
    return state


class TestTransitions:
    def test_new_state_is_all_pending(self, paths: VideoPaths) -> None:
        state = load(paths)
        assert all(state.status_of(name) == STATUS_PENDING for name in state.stage_order)
        assert state.overall_status == STATUS_PENDING

    def test_running_then_completed(self, paths: VideoPaths) -> None:
        state = load(paths)
        state.mark_running("metadata")
        assert state.status_of("metadata") == STATUS_RUNNING
        state.mark_completed("metadata", config_hash="abc")
        record = state.stage("metadata")
        assert record.status == STATUS_COMPLETED
        assert record.started_at and record.completed_at
        assert record.duration_seconds is not None

    def test_failure_keeps_error_context(self, paths: VideoPaths) -> None:
        state = load(paths)
        state.mark_running("audio")
        state.mark_failed("audio", {"type": "CommandError", "message": "ffmpeg died"}, exit_code=1,
                          command=["ffmpeg", "-i", "x"])
        record = state.stage("audio")
        assert record.status == STATUS_FAILED
        assert record.exit_code == 1
        assert record.error["type"] == "CommandError"

    def test_retry_clears_previous_error(self, paths: VideoPaths) -> None:
        state = load(paths)
        state.mark_failed("audio", {"message": "boom"})
        state.reset_stage("audio")
        assert state.status_of("audio") == STATUS_PENDING
        state.mark_running("audio")
        assert state.stage("audio").error is None

    def test_skipped_records_reason(self, paths: VideoPaths) -> None:
        state = load(paths)
        state.mark_skipped("whisperx", "whisperx.enabled = false")
        assert state.stage("whisperx").validation_result["reason"] == "whisperx.enabled = false"

    def test_completed_after_failure_settles(self, paths: VideoPaths) -> None:
        state = load(paths)
        state.mark_failed("audio", {"message": "x"})
        state.mark_completed("audio", config_hash="h")
        assert state.stage("audio").error is None
        assert state.overall_status == "partial"


class TestOverallStatus:
    def test_all_completed_is_completed(self, paths: VideoPaths) -> None:
        state = load(paths)
        for name in state.stage_order:
            state.mark_completed(name, config_hash="h")
        assert state.overall_status == STATUS_COMPLETED

    def test_completed_plus_skipped_is_completed(self, paths: VideoPaths) -> None:
        state = load(paths)
        state.mark_completed("metadata", config_hash="h")
        state.mark_completed("audio", config_hash="h")
        state.mark_skipped("whisperx", "disabled")
        assert state.overall_status == STATUS_COMPLETED

    def test_mixed_is_partial(self, paths: VideoPaths) -> None:
        state = load(paths)
        state.mark_completed("metadata", config_hash="h")
        state.mark_failed("audio", {"message": "x"})
        assert state.overall_status == "partial"

    def test_only_failure_is_failed(self, paths: VideoPaths) -> None:
        state = load(paths)
        state.mark_failed("metadata", {"message": "x"})
        assert state.overall_status == STATUS_FAILED

    def test_running_wins(self, paths: VideoPaths) -> None:
        state = load(paths)
        state.mark_completed("metadata", config_hash="h")
        state.mark_running("audio")
        assert state.overall_status == STATUS_RUNNING


class TestAtomicPersistence:
    def test_state_round_trips_through_disk(self, paths: VideoPaths) -> None:
        state = load(paths)
        state.mark_completed("metadata", config_hash="h", tool_version="ffprobe 7.1.1",
                             output_artifacts=["metadata"])
        reloaded = load(paths)
        assert reloaded.stage("metadata").config_hash == "h"
        assert reloaded.stage("metadata").tool_version == "ffprobe 7.1.1"
        assert reloaded.stage("metadata").output_artifacts == ["metadata"]

    def test_no_temp_files_left_behind(self, paths: VideoPaths) -> None:
        state = load(paths)
        state.mark_completed("metadata", config_hash="h")
        leftovers = list(paths.dataset_dir.glob("*.tmp"))
        assert leftovers == []

    def test_crash_between_write_and_rename_keeps_previous_file(self, paths: VideoPaths, monkeypatch) -> None:
        """The temp-file + rename protocol must survive a failure at rename time."""
        import os

        state = load(paths)
        state.mark_completed("metadata", config_hash="first")
        original_content = paths.status.read_text()
        real_replace = os.replace

        def exploding_replace(src, dst, *args, **kwargs):
            if Path(dst).name == "status.json":
                raise OSError("simulated crash during rename")
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(os, "replace", exploding_replace)
        with pytest.raises(OSError):
            state.mark_completed("audio", config_hash="second")
        monkeypatch.undo()

        assert paths.status.read_text() == original_content
        assert load(paths).stage("metadata").config_hash == "first"
        assert list(paths.dataset_dir.glob("*.tmp")) == []

    def test_reuse_count_accumulates(self, paths: VideoPaths) -> None:
        state = load(paths)
        state.mark_completed("metadata", config_hash="h")
        state.mark_reused("metadata")
        state.mark_reused("metadata")
        assert load(paths).stage("metadata").reuse_count == 2

    def test_state_file_is_human_readable(self, paths: VideoPaths) -> None:
        state = load(paths)
        state.mark_completed("metadata", config_hash="h")
        payload = json.loads(paths.status.read_text())
        assert payload["video_id"] == "vid"
        assert payload["schema_version"] == "1.0"
        assert payload["source"]["filename"] == "vid.mp4"


class TestStageBinding:
    def test_bind_preserves_existing_history(self, paths: VideoPaths) -> None:
        state = load(paths)
        state.mark_completed("metadata", config_hash="h")
        reloaded = VideoState.load(paths, "vid")
        reloaded.bind_stages(["metadata", "audio", "whisperx", "new_stage"])
        assert reloaded.stage("metadata").status == STATUS_COMPLETED
        assert reloaded.stage("new_stage").status == STATUS_PENDING

    def test_summary_lists_every_stage(self, paths: VideoPaths) -> None:
        state = load(paths)
        state.mark_completed("metadata", config_hash="h")
        summary = state.summary()
        assert summary["stages"].keys() == {"metadata", "audio", "whisperx"}
        assert summary["duration_seconds"] >= 0

    def test_unknown_stage_autocreates(self, paths: VideoPaths) -> None:
        state = load(paths)
        assert state.status_of("not_declared") == STATUS_PENDING
