"""Finalization: the dataset as a whole, and the manifest that describes it.

Every other stage validates itself; this stage is the only one that can notice
that two modalities disagree. Tables are written as real Parquet because the
checks read columns out of them — a patched reader would test the mock.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from multimodal_pipeline.artifacts import ArtifactRegistry, atomic_write_json
from multimodal_pipeline.exceptions import ValidationError
from multimodal_pipeline.manifest import read_manifest
from multimodal_pipeline.stages.base import STAGE_ORDER, StageContext
from multimodal_pipeline.stages.finalization import (
    FinalizationStage,
    write_dataset_summary,
)

METADATA = {
    "schema_version": "1.0",
    "video_id": "clip",
    "source_filename": "clip.mp4",
    "source_path": "/in/clip.mp4",
    "SHA256": "a" * 64,
    "file_size_bytes": 1000,
    "duration_seconds": 10.0,
    "width": 640,
    "height": 480,
    "video_codec": "h264",
    "audio_codec": "aac",
    "average_frame_rate_float": 25.0,
}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows) if rows else pa.table({}).slice(0, 0), path)
    return path


@pytest.fixture
def dataset(context: StageContext) -> StageContext:
    """A minimal but real dataset: metadata, a frame index, transcript and pose."""
    atomic_write_json(context.artifact("metadata"), METADATA)
    write_rows(context.artifact("frame_index"),
               [{"frame_number": i, "pts_seconds": i / 25.0} for i in range(250)])
    write_rows(context.artifact("speech_segments"),
               [{"segment_id": "seg1", "start_time": 0.5, "end_time": 2.0,
                 "speaker_id": None, "language": "en", "text": "hello"}])
    write_rows(context.artifact("speech_words"),
               [{"segment_id": "seg1", "word": "hello", "start_time": 0.5, "end_time": 0.9}])
    write_rows(context.artifact("pose_body"),
               [{"frame_number": 10, "timestamp": 0.4, "person_id": 0, "keypoint_name": "Nose"}])
    write_rows(context.artifact("acoustic_frames"),
               [{"timestamp": 0.5, "f0_hz": 120.0}])
    context.state.mark_completed("metadata")
    context.state.mark_completed("audio")
    context.state.mark_completed("whisperx")
    context.state.mark_completed("openpose")
    context.state.mark_completed("acoustic")
    context.registry.refresh()
    return context


#: One instance is enough: the stage holds no per-video state.
stage = FinalizationStage()


def finalize(context: StageContext) -> dict[str, Any]:
    write_dataset_summary(context)
    context.registry.refresh()
    return read_manifest(context.paths)


class TestSummary:
    def test_manifest_and_provenance_are_written(self, dataset: StageContext) -> None:
        manifest = finalize(dataset)
        # The summary describes the context's video, not the metadata fixture's copy.
        assert manifest["video_id"] == dataset.video_id
        assert manifest["source"]["duration_seconds"] == 10.0
        for name in ("provenance_config", "provenance_tools", "provenance_processing"):
            assert dataset.artifact(name).is_file(), name

    def test_detected_language_comes_from_the_transcript(self, dataset: StageContext) -> None:
        assert finalize(dataset)["source"]["detected_language"] == "en"

    def test_language_is_null_without_a_transcript(self, dataset: StageContext) -> None:
        dataset.artifact("speech_segments").unlink()
        assert finalize(dataset)["source"]["detected_language"] is None

    def test_stage_statuses_are_recorded(self, dataset: StageContext) -> None:
        manifest = finalize(dataset)
        assert manifest["processing"]["stages"]["whisperx"] == "completed"
        assert manifest["processing"]["stages"]["diarization"] == "pending"

    def test_overall_status_is_the_states_aggregate(self, dataset: StageContext) -> None:
        assert finalize(dataset)["processing"]["status"] == dataset.state.overall_status

    def test_no_secrets_reach_the_provenance(self, dataset: StageContext) -> None:
        finalize(dataset)
        text = dataset.artifact("provenance_config").read_text()
        assert "sk-secret-123456" not in text
        assert "***masked***" in text

    def test_a_summary_survives_a_later_state_change(self, dataset: StageContext) -> None:
        """The runner rewrites the summary after the stage loop; that is the point."""
        finalize(dataset)
        before = read_manifest(dataset.paths)["processing"]["stages"]["finalization"]
        dataset.state.mark_completed("finalization")
        write_dataset_summary(dataset)
        after = read_manifest(dataset.paths)["processing"]["stages"]["finalization"]
        assert before != after == "completed"


class TestValidation:
    def test_a_valid_dataset_passes(self, dataset: StageContext) -> None:
        finalize(dataset)
        result = stage.validate(dataset)
        assert result["status"] == dataset.state.overall_status
        assert result["timed_tables_checked"] >= 3

    def test_missing_manifest_is_an_error(self, dataset: StageContext) -> None:
        with pytest.raises(ValidationError, match="manifest.json missing"):
            stage.validate(dataset)

    def test_a_manifest_promise_that_was_broken_is_caught(self, dataset: StageContext) -> None:
        manifest = finalize(dataset)
        dataset.artifact("pose_body").unlink()
        # The manifest still promises it: that is a broken data contract.
        with pytest.raises(ValidationError, match="missing"):
            stage.validate(dataset)

    @pytest.mark.parametrize("timestamp", [-0.5, 99.0])
    def test_timestamps_outside_the_media_are_caught(self, dataset: StageContext,
                                                     timestamp: float) -> None:
        write_rows(dataset.artifact("pose_body"),
                   [{"frame_number": 1, "timestamp": timestamp, "person_id": 0,
                     "keypoint_name": "Nose"}])
        finalize(dataset)
        with pytest.raises(ValidationError) as excinfo:
            stage.validate(dataset)
        expected = "before t=0" if timestamp < 0 else "exceeds the media duration"
        assert expected in " ".join(excinfo.value.issues)

    def test_a_translation_row_with_no_source_segment_is_caught(self, dataset: StageContext) -> None:
        write_rows(dataset.artifact("translation_segments"),
                   [{"segment_id": "ghost", "start_time": 0.5, "end_time": 2.0}])
        finalize(dataset)
        with pytest.raises(ValidationError, match="no source segment"):
            stage.validate(dataset)

    def test_a_speaker_absent_from_diarization_is_caught(self, dataset: StageContext) -> None:
        write_rows(dataset.artifact("speaker_turns"),
                   [{"speaker_id": "SPEAKER_01", "start_time": 0.0, "end_time": 1.0}])
        write_rows(dataset.artifact("speech_segments"),
                   [{"segment_id": "seg1", "start_time": 0.5, "end_time": 2.0,
                     "speaker_id": "SPEAKER_07", "language": "en", "text": "hello"}])
        dataset.state.mark_completed("diarization")
        dataset.state.mark_completed("speaker_assignment")
        finalize(dataset)
        with pytest.raises(ValidationError, match="absent from diarization"):
            stage.validate(dataset)

    def test_matching_speakers_pass(self, dataset: StageContext) -> None:
        write_rows(dataset.artifact("speaker_turns"),
                   [{"speaker_id": "SPEAKER_00", "start_time": 0.0, "end_time": 9.0}])
        write_rows(dataset.artifact("speech_segments"),
                   [{"segment_id": "seg1", "start_time": 0.5, "end_time": 2.0,
                     "speaker_id": "SPEAKER_00", "language": "en", "text": "hello"}])
        finalize(dataset)
        stage.validate(dataset)

    def test_corrupt_provenance_is_reported(self, dataset: StageContext) -> None:
        finalize(dataset)
        dataset.artifact("provenance_tools").write_text("{ not json", encoding="utf-8")
        with pytest.raises(ValidationError, match="tools.json unreadable"):
            stage.validate(dataset)

    def test_an_unreadable_table_is_reported_not_raised(self, dataset: StageContext) -> None:
        finalize(dataset)
        dataset.artifact("acoustic_frames").write_bytes(b"not parquet")
        with pytest.raises(ValidationError, match="acoustic"):
            stage.validate(dataset)

    def test_a_silent_video_with_no_speech_still_validates(self, dataset: StageContext) -> None:
        for name in ("speech_segments", "speech_words", "acoustic_frames"):
            dataset.artifact(name).unlink()
        finalize(dataset)
        stage.validate(dataset)


class TestReuse:
    """Finalization must settle: writing its own outputs must not invalidate it."""

    def test_fingerprint_ignores_the_files_finalization_writes(self, dataset: StageContext) -> None:
        before = stage.config_fingerprint(dataset)
        finalize(dataset)
        dataset.registry.refresh()
        assert stage.config_fingerprint(dataset) == before

    def test_fingerprint_follows_a_configuration_change(self, dataset: StageContext) -> None:
        before = stage.config_fingerprint(dataset)
        object.__setattr__(dataset.config.whisperx, "model", "large-v2")
        assert stage.config_fingerprint(dataset) != before

    def test_a_completed_finalization_is_reusable(self, dataset: StageContext) -> None:
        """After a full run the summary must settle instead of rewriting forever.

        The fingerprint used to include the sizes of the manifest and provenance
        files finalization itself writes, so every later run called it stale.
        """
        from multimodal_pipeline.config import stable_hash
        from multimodal_pipeline.orchestrator import VideoRunner
        from multimodal_pipeline.stages.base import should_reuse

        finalize(dataset)
        dataset.state.mark_completed(stage.name)
        runner = VideoRunner(dataset.config, dataset.source, tools={"schema_version": "1.0"})
        runner.state = dataset.state
        config_hash = stable_hash(stage.config_fingerprint(dataset), length=16)
        dependency_hash = runner.dependency_hash_for(stage, dataset)
        dataset.state.stage(stage.name).config_hash = config_hash
        dataset.state.stage(stage.name).dependency_hash = dependency_hash
        dataset.state.save()
        reusable, reason = should_reuse(stage, dataset, config_hash=config_hash,
                                        dependency_hash=dependency_hash, force=False)
        assert reusable, reason
        # Writing the summary again must not change the answer.
        write_dataset_summary(dataset)
        assert should_reuse(stage, dataset, config_hash=config_hash,
                            dependency_hash=dependency_hash, force=False)[0]
