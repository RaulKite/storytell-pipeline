"""Unit tests for the TalkNet active-speaker stage.

The stage's whole value is a *dense, honest* per-frame timeline, so these tests
concentrate on the guarantees that make it trustworthy: one row per frame, absence
kept as absence, an imputed score always disclosed, and a fingerprint that notices
when the model or its tuning changed.
"""

from __future__ import annotations

import json
from typing import Any

import pyarrow as pa
import pytest

from multimodal_pipeline.exceptions import ValidationError
from multimodal_pipeline.schemas import (
    ACTIVE_SPEAKER_FRAMES_SCHEMA,
    read_table,
    write_table,
)
from multimodal_pipeline.stages.activespeaker import (
    ActiveSpeakerStage,
    track_summary_rows,
)
from multimodal_pipeline.stages.base import raw_request_matches


def make_frames_document(**overrides: Any) -> dict[str, Any]:
    """A minimal but structurally real worker document."""
    parameters = {"speaker_threshold": 0.0, "score_window": 5,
                  "switch_margin": 0.5, "switch_frames": 3}
    parameters.update(overrides.pop("parameters", {}))
    document = {
        "schema_version": "1.0",
        "video_id": "conversation_001",
        "output_fps": 25,
        "source_fps": 29.97,
        "device": "cuda",
        "requested_device": "cuda",
        "device_fallback_reason": None,
        "pickle_encoding": "bytes",
        "parameters": parameters,
        "scene_count": 1,
        "frame_count": 4,
        "track_count": 2,
        "active_frame_count": 2,
        "frames": [
            {"frame_25fps": 0, "timestamp_sec": 0.0, "source_timestamp_sec": 0.0,
             "scene_id": 1, "track_id": 0, "x1": 10.0, "y1": 20.0, "x2": 50.0, "y2": 70.0,
             "talknet_score_raw": 1.5, "talknet_score": 1.6, "score_imputed": False,
             "is_active_speaker": True},
            {"frame_25fps": 1, "timestamp_sec": 0.04, "source_timestamp_sec": 0.033,
             "scene_id": 1, "track_id": 0, "x1": 10.0, "y1": 20.0, "x2": 50.0, "y2": 70.0,
             "talknet_score_raw": 2.0, "talknet_score": 1.8, "score_imputed": True,
             "is_active_speaker": True},
            # A gap: no face at all in this frame.
            {"frame_25fps": 2, "timestamp_sec": 0.08, "source_timestamp_sec": 0.067,
             "scene_id": 1, "track_id": None, "x1": None, "y1": None, "x2": None, "y2": None,
             "talknet_score_raw": None, "talknet_score": None, "score_imputed": False,
             "is_active_speaker": False},
            {"frame_25fps": 3, "timestamp_sec": 0.12, "source_timestamp_sec": 0.1,
             "scene_id": 1, "track_id": 1, "x1": 5.0, "y1": 5.0, "x2": 35.0, "y2": 45.0,
             "talknet_score_raw": -1.0, "talknet_score": -0.5, "score_imputed": False,
             "is_active_speaker": False},
        ],
    }
    document.update(overrides)
    return document


@pytest.fixture
def seeded(context):
    """A context carrying a valid raw document plus its provenance sidecar."""
    stage = ActiveSpeakerStage()
    document = make_frames_document()
    raw = context.artifact("activespeaker_raw")
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(document), encoding="utf-8")
    from multimodal_pipeline.stages.base import stamp_raw

    stamp_raw(raw, request=stage.request(context), digest=stage.request_digest(context),
              worker=None)
    # metadata supplies the duration bound used by interval validation
    context.artifact("metadata").parent.mkdir(parents=True, exist_ok=True)
    context.artifact("metadata").write_text(json.dumps({"duration_seconds": 4.2}),
                                            encoding="utf-8")
    return context


def resample(context, stage) -> None:
    """Re-stamp the raw artifact after the caller mutated config or document."""
    from multimodal_pipeline.stages.base import stamp_raw

    raw = context.artifact("activespeaker_raw")
    stamp_raw(raw, request=stage.request(context), digest=stage.request_digest(context),
              worker=None)


class TestGating:
    def test_disabled_without_a_talknet_root(self, context):
        enabled, reason = ActiveSpeakerStage().enabled(context)
        assert enabled is False
        assert "talknet_root" in reason

    def test_disabled_when_root_is_not_a_talknet_checkout(self, context, tmp_path):
        context.config.activespeaker.talknet_root = tmp_path
        enabled, reason = ActiveSpeakerStage().enabled(context)
        assert enabled is False
        assert "run_talknet.py" in reason

    def test_missing_root_directory_says_so(self, context, tmp_path):
        context.config.activespeaker.talknet_root = tmp_path / "absent"
        enabled, reason = ActiveSpeakerStage().enabled(context)
        assert enabled is False
        assert "does not exist" in reason

    def test_enabled_with_a_real_checkout(self, context, tmp_path):
        (tmp_path / "run_talknet.py").write_text("# stub", encoding="utf-8")
        context.config.activespeaker.talknet_root = tmp_path
        assert ActiveSpeakerStage().enabled(context) == (True, "")

    def test_explicit_disable_wins_over_a_valid_root(self, context, tmp_path):
        (tmp_path / "run_talknet.py").write_text("# stub", encoding="utf-8")
        context.config.activespeaker.talknet_root = tmp_path
        context.config.activespeaker.enabled = False
        enabled, reason = ActiveSpeakerStage().enabled(context)
        assert enabled is False
        assert "enabled = false" in reason


class TestRequestBindsWhatChangesTheOutput:
    def test_every_tuning_knob_is_in_the_request(self, context):
        request = ActiveSpeakerStage().request(context)
        for key in ("speaker_threshold", "score_window", "switch_margin", "switch_frames",
                    "talknet_root", "device", "extra_args"):
            assert key in request

    @pytest.mark.parametrize("key,value", [
        ("speaker_threshold", 1.5),
        ("score_window", 9),
        ("switch_margin", 0.9),
        ("switch_frames", 5),
        ("device", "cpu"),
    ])
    def test_changing_a_knob_changes_the_digest(self, context, key, value):
        stage = ActiveSpeakerStage()
        before = stage.request_digest(context)
        setattr(context.config.activespeaker, key, value)
        assert stage.request_digest(context) != before

    def test_worker_code_change_invalidates(self, context):
        """The base contract, asserted here so a refactor cannot quietly drop it."""
        stage = ActiveSpeakerStage()
        payload = stage.digest_payload(context)
        assert "_worker_code_sha256" in payload

    def test_source_audio_identity_is_bound(self, context):
        # Without this, a re-extracted audio track would keep serving stale scores.
        assert "source_sha256" in ActiveSpeakerStage().request(context)


class TestNormalization:
    def test_frames_table_is_dense_and_ordered(self, seeded):
        stage = ActiveSpeakerStage()
        summary = stage.normalize(seeded)
        rows = read_table(seeded.artifact("active_speaker_frames")).to_pylist()
        assert [row["frame_number"] for row in rows] == [0, 1, 2, 3]
        assert summary["frames"] == 4
        assert summary["frames_with_face"] == 3
        assert summary["active_frames"] == 2

    def test_gap_frame_keeps_absence_explicit(self, seeded):
        ActiveSpeakerStage().normalize(seeded)
        rows = read_table(seeded.artifact("active_speaker_frames")).to_pylist()
        gap = rows[2]
        assert gap["track_id"] is None
        assert gap["x1"] is None and gap["talknet_score"] is None
        assert gap["is_active_speaker"] is False

    def test_imputation_survives_normalization(self, seeded):
        ActiveSpeakerStage().normalize(seeded)
        rows = read_table(seeded.artifact("active_speaker_frames")).to_pylist()
        assert rows[1]["score_imputed"] is True
        assert rows[0]["score_imputed"] is False

    def test_metadata_records_device_and_fps(self, seeded):
        ActiveSpeakerStage().normalize(seeded)
        table = read_table(seeded.artifact("active_speaker_frames"))
        metadata = table.schema.metadata
        assert metadata[b"output_fps"] == b"25"
        assert metadata[b"device"] == b"cuda"

    def test_track_summary_is_derived_from_selection(self, seeded):
        rows = track_summary_rows("v", [
            {"frame_number": i, "timestamp": i * 0.04, "scene_id": 1, "track_id": 0,
             "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0, "talknet_score": float(i),
             "is_active_speaker": i > 0}
            for i in range(4)
        ])
        assert len(rows) == 1
        track = rows[0]
        assert track["track_id"] == 0
        assert track["frame_count"] == 4
        assert track["active_frame_count"] == 3
        assert track["active_ratio"] == 0.75
        assert track["first_timestamp"] == 0.0
        assert track["last_timestamp"] == 0.12
        assert track["scenes"] == [1]
        assert track["mean_bbox_area"] == 100.0

    def test_track_summary_orders_tracks_and_skips_gaps(self, seeded):
        frames = [{"frame_number": 0, "timestamp": 0.0, "scene_id": 1, "track_id": 1,
                   "x1": 0.0, "y1": 0.0, "x2": 2.0, "y2": 2.0, "talknet_score": 1.0,
                   "is_active_speaker": True},
                  {"frame_number": 1, "timestamp": 0.04, "scene_id": 1, "track_id": None,
                   "x1": None, "y1": None, "x2": None, "y2": None, "talknet_score": None,
                   "is_active_speaker": False},
                  {"frame_number": 2, "timestamp": 0.08, "scene_id": 1, "track_id": 0,
                   "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0, "talknet_score": -1.0,
                   "is_active_speaker": False}]
        rows = track_summary_rows("v", frames)
        assert [row["track_id"] for row in rows] == [0, 1]

    def test_normalize_rejects_a_document_without_frames(self, seeded):
        raw = seeded.artifact("activespeaker_raw")
        raw.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")
        with pytest.raises(ValidationError, match="no frames"):
            ActiveSpeakerStage().normalize(seeded)


class TestValidation:
    def test_clean_dataset_validates(self, seeded):
        stage = ActiveSpeakerStage()
        stage.normalize(seeded)
        result = stage.validate(seeded)
        assert result["frames"] == 4
        assert result["frames_with_face"] == 3
        assert result["tracks"] == 2

    def test_non_dense_frame_numbers_are_rejected(self, seeded):
        """The dense timeline is the stage's contract; a gap must not pass quietly."""
        document = make_frames_document()
        document["frames"][2]["frame_25fps"] = 7
        seeded.artifact("activespeaker_raw").write_text(json.dumps(document), encoding="utf-8")
        stage = ActiveSpeakerStage()
        resample(seeded, stage)
        stage.normalize(seeded)
        with pytest.raises(ValidationError, match="dense"):
            stage.validate(seeded)

    def test_active_frame_without_a_face_is_rejected(self, seeded):
        document = make_frames_document()
        document["frames"][2]["is_active_speaker"] = True
        seeded.artifact("activespeaker_raw").write_text(json.dumps(document), encoding="utf-8")
        stage = ActiveSpeakerStage()
        resample(seeded, stage)
        stage.normalize(seeded)
        with pytest.raises(ValidationError, match="no face but is marked active"):
            stage.validate(seeded)

    def test_inverted_bbox_is_rejected(self, seeded):
        document = make_frames_document()
        document["frames"][0]["x2"] = 1.0
        seeded.artifact("activespeaker_raw").write_text(json.dumps(document), encoding="utf-8")
        stage = ActiveSpeakerStage()
        resample(seeded, stage)
        stage.normalize(seeded)
        with pytest.raises(ValidationError, match="inverted bbox"):
            stage.validate(seeded)

    def test_frame_count_disagreement_with_raw_is_rejected(self, seeded):
        document = make_frames_document()
        document["frames"] = document["frames"][:3]
        # frame_count still says 4, so raw and table disagree
        seeded.artifact("activespeaker_raw").write_text(json.dumps(document), encoding="utf-8")
        stage = ActiveSpeakerStage()
        resample(seeded, stage)
        stage.normalize(seeded)
        with pytest.raises(ValidationError, match="declares 4 frames"):
            stage.validate(seeded)

    def test_raw_from_a_different_configuration_is_rejected(self, seeded):
        stage = ActiveSpeakerStage()
        stage.normalize(seeded)
        seeded.config.activespeaker.speaker_threshold = 2.0
        with pytest.raises(ValidationError, match="different configuration"):
            stage.validate(seeded)

    def test_matching_configuration_still_validates(self, seeded):
        """Guard for the inverse: the check must not reject what it just produced."""
        stage = ActiveSpeakerStage()
        stage.normalize(seeded)
        assert raw_request_matches(seeded.artifact("activespeaker_raw"),
                                   stage.request_digest(seeded))

    def test_missing_frames_table_is_reported(self, seeded):
        with pytest.raises(ValidationError, match="active_speaker_frames.parquet missing"):
            ActiveSpeakerStage().validate(seeded)

    def test_empty_frames_table_is_reported(self, seeded):
        write_table(seeded.artifact("active_speaker_frames"),
                    pa.Table.from_pylist([], schema=ACTIVE_SPEAKER_FRAMES_SCHEMA),
                    ACTIVE_SPEAKER_FRAMES_SCHEMA)
        with pytest.raises(ValidationError, match="empty"):
            ActiveSpeakerStage().validate(seeded)

    def test_missing_raw_document_is_reported(self, context):
        with pytest.raises(ValidationError, match="raw artifact missing"):
            ActiveSpeakerStage().validate(context)


class TestWorkerInvocation:
    def test_argv_carries_every_input_the_worker_needs(self, context, tmp_path):
        (tmp_path / "run_talknet.py").write_text("# stub", encoding="utf-8")
        context.config.activespeaker.talknet_root = tmp_path
        context.artifact("audio").parent.mkdir(parents=True, exist_ok=True)
        context.artifact("audio").write_bytes(b"RIFFfake")
        stage = ActiveSpeakerStage()
        stage.prepare(context)
        argv = stage.worker_argv(context, context.artifact("activespeaker_raw"), "digest")
        text = " ".join(argv)
        for flag in ("--video", "--audio", "--talknet-root", "--raw-dir", "--output-json",
                     "--result-path", "--video-id", "--device", "--speaker-threshold",
                     "--score-window", "--switch-margin", "--switch-frames"):
            assert flag in argv, flag
        assert str(context.source.path) in text
        assert "--weights-dir" not in argv

    def test_weights_dir_is_forwarded_when_configured(self, context, tmp_path):
        (tmp_path / "run_talknet.py").write_text("# stub", encoding="utf-8")
        weights = tmp_path / "weights"
        weights.mkdir()
        context.config.activespeaker.talknet_root = tmp_path
        context.config.activespeaker.weights_dir = weights
        context.artifact("audio").parent.mkdir(parents=True, exist_ok=True)
        context.artifact("audio").write_bytes(b"RIFFfake")
        stage = ActiveSpeakerStage()
        stage.prepare(context)
        argv = stage.worker_argv(context, context.artifact("activespeaker_raw"), "digest")
        assert "--weights-dir" in argv
        assert str(weights) in argv

    def test_extra_args_are_appended(self, context, tmp_path):
        (tmp_path / "run_talknet.py").write_text("# stub", encoding="utf-8")
        context.config.activespeaker.talknet_root = tmp_path
        context.config.activespeaker.extra_args = ["--visualisation"]
        context.artifact("audio").parent.mkdir(parents=True, exist_ok=True)
        context.artifact("audio").write_bytes(b"RIFFfake")
        stage = ActiveSpeakerStage()
        stage.prepare(context)
        argv = stage.worker_argv(context, context.artifact("activespeaker_raw"), "digest")
        assert argv[-1] == "--visualisation"

    def test_cuda_pins_one_visible_device(self, context):
        context.config.activespeaker.device = "cuda"
        context.config.activespeaker.device_index = 1
        env = ActiveSpeakerStage().worker_environment(context)
        assert env["CUDA_VISIBLE_DEVICES"] == "1"

    def test_cpu_requests_no_gpu_visibility_change(self, context):
        context.config.activespeaker.device = "cpu"
        assert ActiveSpeakerStage().worker_environment(context) == {}

    def test_timeout_comes_from_config(self, context):
        assert ActiveSpeakerStage().worker_timeout(context) is None
        context.config.activespeaker.timeout_seconds = 60.0
        assert ActiveSpeakerStage().worker_timeout(context) == 60.0

    def test_raw_provenance_sidecar_is_stamped_with_the_request(self, seeded):
        """The sidecar is what validation compares against; without it the stage
        could never prove which configuration produced a cached raw document."""
        from multimodal_pipeline.stages.base import _read_sidecar

        # _read_sidecar takes the *raw* path and derives the sidecar itself.
        sidecar = _read_sidecar(seeded.artifact("activespeaker_raw"))
        assert sidecar is not None
        assert sidecar["request"]["switch_frames"] == 3


class TestMalformedRawRows:
    """The raw document comes from an external tool; odd rows must be *named*.

    validate() compared bbox coordinates before checking they exist, so a row that
    carried a track_id with null coordinates raised TypeError from inside validation
    -- which the orchestrator records as an opaque stage crash instead of the one
    line that says which frame and which field.
    """

    def test_face_with_null_coordinates_is_reported_not_crashing(self, seeded):
        document = make_frames_document()
        for key in ("x1", "y1", "x2", "y2"):
            document["frames"][0][key] = None
        seeded.artifact("activespeaker_raw").write_text(json.dumps(document), encoding="utf-8")
        stage = ActiveSpeakerStage()
        resample(seeded, stage)
        # normalize must tolerate it (absence is absence), and validate must name it.
        stage.normalize(seeded)
        with pytest.raises(ValidationError, match="frame 0.*missing bbox"):
            stage.validate(seeded)

    def test_non_finite_score_is_reported_not_crashing(self, seeded):
        document = make_frames_document()
        # json has no NaN literal, so write the raw text with a NaN token.
        text = json.dumps(document).replace('"talknet_score": 1.6', '"talknet_score": NaN')
        assert "NaN" in text
        seeded.artifact("activespeaker_raw").write_text(text, encoding="utf-8")
        stage = ActiveSpeakerStage()
        resample(seeded, stage)
        stage.normalize(seeded)
        with pytest.raises(ValidationError, match="non-finite score"):
            stage.validate(seeded)
