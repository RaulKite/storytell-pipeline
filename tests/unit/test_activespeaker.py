"""Unit tests for the TalkNet active-speaker stage.

The stage's whole value is a *dense, honest* per-frame timeline, so these tests
concentrate on the guarantees that make it trustworthy: one row per frame, absence
kept as absence, an imputed score always disclosed, and a fingerprint that notices
when the model or its tuning changed.
"""

from __future__ import annotations

import json
import sys
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


def load_asd_worker():
    """Import workers/activespeaker_worker.py by path: it runs in its own uv project
    and is not an importable module of this package (same pattern as the spaCy tests)."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "workers" / "activespeaker_worker.py"
    spec = importlib.util.spec_from_file_location("activespeaker_worker_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before executing: this worker declares dataclasses, and dataclass field
    # resolution looks the owning module up in sys.modules. A module loaded by path that
    # was never registered resolves to None and fails inside stdlib code with an
    # AttributeError about __dict__ instead of naming the worker.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_track(frames, bboxes):
    """The pkl shape TalkNet writes: one dict per track, frame/bbox aligned arrays."""
    return {"track": {"frame": list(frames), "bbox": [list(b) for b in bboxes]}}


BOX = (10.0, 20.0, 50.0, 70.0)


class TestUnscoredFaceStaysVisible:
    """A located face whose score does not exist is its own state.

    Before face_status, build_candidates dropped every frame it could not score, so
    the emitted row was identical to a frame with no face: a reader of
    track_id = null could not tell "nobody was on screen" from "S3FD had a person and
    TalkNet had no measurement". Absence of a score became absence of a person.
    """

    def test_past_the_imputable_tail_keeps_the_face_without_a_score(self):
        worker = load_asd_worker()
        # One track, four frames, one score. The budget of two carries the last score
        # over positions 1 and 2; position 3 is past it and must stay visible with no
        # score at all -- this is the frame that used to vanish and reappear as no face.
        tracks = [make_track([0, 1, 2, 3], [BOX] * 4)]
        scores = [[1.0]]
        by_frame = worker.build_candidates(tracks, scores, [1, 1, 1, 1])
        assert by_frame[0][0].raw_score == 1.0
        assert not by_frame[0][0].score_imputed
        assert by_frame[1][0].raw_score == 1.0 and by_frame[1][0].score_imputed
        assert by_frame[2][0].raw_score == 1.0 and by_frame[2][0].score_imputed
        assert by_frame[3][0].raw_score is None
        assert not by_frame[3][0].score_imputed

    def test_non_finite_score_leaves_the_face_located(self):
        worker = load_asd_worker()
        tracks = [make_track([0, 1], [BOX, BOX])]
        scores = [[1.0, float("nan")]]
        by_frame = worker.build_candidates(tracks, scores, [1, 1])
        assert by_frame[0][0].raw_score == 1.0
        assert by_frame[1][0].raw_score is None

    def test_malformed_bbox_is_still_dropped(self):
        """A garbage box is not a location: inventing one would place a person where
        the detector never saw anybody, so this face stays dropped."""
        worker = load_asd_worker()
        junk = (float("nan"), 20.0, 50.0, 70.0)
        tracks = [make_track([0, 1], [BOX, junk])]
        scores = [[1.0, 1.0]]
        by_frame = worker.build_candidates(tracks, scores, [1, 1])
        assert 0 in by_frame[0]
        assert by_frame[1] == {}

    def test_smoothing_never_averages_an_unscored_frame(self):
        worker = load_asd_worker()
        # Track 0 scored at frames 0..2; frame 3 sits inside its smoothing window but
        # is unscored. The mean at frame 2 must be of the three real scores only.
        by_frame = [
            {0: worker.Candidate(0, BOX, 1.0, False)},
            {0: worker.Candidate(0, BOX, 2.0, False)},
            {0: worker.Candidate(0, BOX, 3.0, False)},
            {0: worker.Candidate(0, BOX, None, False)},
        ]
        worker.smooth_scores(by_frame, [1, 1, 1, 1], window=3)
        assert by_frame[2][0].smoothed_score == pytest.approx((2.0 + 3.0) / 2)
        assert by_frame[3][0].smoothed_score is None

    def test_selection_never_picks_an_unscored_face(self):
        worker = load_asd_worker()
        by_frame = [
            {0: worker.Candidate(0, BOX, 1.0, False, smoothed_score=1.0)},
            {1: worker.Candidate(1, BOX, None, False)},
        ]
        worker.smooth_scores(by_frame, [1, 1], window=1)
        chosen = worker.select_stable(by_frame, [1, 1], 0.5, 3)
        assert chosen[0] is not None and chosen[0].track_id == 0
        # Frame 1 has a visible face but no evidence about speaking: no active label.
        assert chosen[1] is None

    def test_document_reports_the_three_face_states(self):
        worker = load_asd_worker()
        scored = worker.Candidate(0, BOX, 2.0, False, smoothed_score=2.0)
        unscored = worker.Candidate(1, BOX, None, False)
        by_frame = [{0: scored}, {1: unscored}, {}]
        document = worker.build_document(
            video_id="v", source_fps=30.0, stamps=[0.0, 0.04, 0.08],
            scene_ids=[1, 1, 1], chosen=[scored, None, None], visible=by_frame,
            track_count=2, pickle_encoding="bytes", device="cpu",
            requested_device="cpu", fallback_reason=None,
            params={"speaker_threshold": 0.0, "score_window": 5,
                    "switch_margin": 0.5, "switch_frames": 3},
        )
        rows = document["frames"]
        assert [r["face_status"] for r in rows] == ["tracked", "tracked_unscored", "no_face"]
        assert rows[1]["track_id"] == 1 and rows[1]["x1"] == pytest.approx(10.0)
        assert rows[1]["talknet_score"] is None
        assert rows[1]["talknet_score_raw"] is None
        assert rows[1]["is_active_speaker"] is False
        assert rows[1]["score_imputed"] is False


class TestFrameStatusInTable:
    """The table and the validator must agree about what each state may contain."""

    def _row(self, index, **overrides):
        base = {"frame_25fps": index, "timestamp_sec": index * 0.04,
                "source_timestamp_sec": index * 0.033, "scene_id": 1,
                "track_id": 0, "x1": 10.0, "y1": 20.0, "x2": 50.0, "y2": 70.0,
                "talknet_score_raw": 1.0, "talknet_score": 1.0,
                "score_imputed": False, "is_active_speaker": True}
        base.update(overrides)
        return base

    def test_normalize_derives_face_status_for_old_raw_artifacts(self, seeded):
        """A dataset processed before face_status existed must resume, not fail: the
        state is derivable from track_id plus score presence."""
        raw_path = seeded.artifact("activespeaker_raw")
        document = json.loads(raw_path.read_text(encoding="utf-8"))
        for frame in document["frames"]:
            frame.pop("face_status", None)
        raw_path.write_text(json.dumps(document), encoding="utf-8")
        stage = ActiveSpeakerStage()
        summary = stage.normalize(seeded)
        table = read_table(seeded.artifact("active_speaker_frames")).to_pylist()
        statuses = [row["face_status"] for row in table]
        assert statuses == ["tracked", "tracked", "no_face", "tracked"]
        assert summary["frames_with_face"] == 3

    def test_validate_accepts_a_tracked_unscored_row(self, seeded):
        raw_path = seeded.artifact("activespeaker_raw")
        document = json.loads(raw_path.read_text(encoding="utf-8"))
        document["frames"].append(
            self._row(4, track_id=1, face_status="tracked_unscored",
                      talknet_score_raw=None, talknet_score=None,
                      is_active_speaker=False))
        document["frame_count"] = 5
        raw_path.write_text(json.dumps(document), encoding="utf-8")
        stage = ActiveSpeakerStage()
        resample(seeded, stage)
        stage.normalize(seeded)
        stage.validate(seeded)

    def test_validate_still_rejects_a_tracked_row_without_a_score(self, seeded):
        raw_path = seeded.artifact("activespeaker_raw")
        document = json.loads(raw_path.read_text(encoding="utf-8"))
        document["frames"].append(
            self._row(4, face_status="tracked", talknet_score=None))
        document["frame_count"] = 5
        raw_path.write_text(json.dumps(document), encoding="utf-8")
        stage = ActiveSpeakerStage()
        resample(seeded, stage)
        stage.normalize(seeded)
        with pytest.raises(ValidationError, match="frame 4.*non-finite score"):
            stage.validate(seeded)

    def test_validate_rejects_an_unscored_row_carrying_evidence(self, seeded):
        """tracked_unscored with a score means the worker and the schema disagree
        about what 'no evidence' looks like; that must not pass as absence."""
        raw_path = seeded.artifact("activespeaker_raw")
        document = json.loads(raw_path.read_text(encoding="utf-8"))
        document["frames"].append(
            self._row(4, face_status="tracked_unscored", talknet_score=0.7,
                      is_active_speaker=False))
        document["frame_count"] = 5
        raw_path.write_text(json.dumps(document), encoding="utf-8")
        stage = ActiveSpeakerStage()
        resample(seeded, stage)
        stage.normalize(seeded)
        with pytest.raises(ValidationError, match="frame 4.*tracked_unscored but carries"):
            stage.validate(seeded)

    def test_track_summary_skips_unscored_scores(self, seeded):
        rows = [
            {"track_id": 0, "timestamp": 0.0, "is_active_speaker": True, "scene_id": 1,
             "x1": 0.0, "y1": 0.0, "x2": 2.0, "y2": 2.0, "talknet_score": 1.0},
            {"track_id": 0, "timestamp": 0.04, "is_active_speaker": False, "scene_id": 1,
             "x1": 0.0, "y1": 0.0, "x2": 2.0, "y2": 2.0, "talknet_score": None},
        ]
        summary = track_summary_rows("v", rows)
        assert summary[0]["frame_count"] == 2
        assert summary[0]["mean_score"] == pytest.approx(1.0)
        assert summary[0]["active_ratio"] == pytest.approx(0.5)


class TestDenseSequenceDiagnosis:
    """R3-001: the dense check rejected three different faults with one string, and
    left nothing in the stage log. A rejected table is usually a stale or hand-edited
    artifact, and "not dense" does not tell the operator which of the three happened.
    """

    @pytest.mark.parametrize("indices, word", [
        ([0, 1, 3, 2], "out of order"),        # same frames, wrong order
        ([0, 1, 2, 7], "missing"),             # 3,4,5,6 absent
        ([0, 1, 1, 2], "duplicate"),           # frame 1 twice
    ])
    def test_the_break_is_named_by_kind(self, indices, word):
        from multimodal_pipeline.stages.activespeaker import dense_sequence_break

        detail = dense_sequence_break(indices)
        assert detail is not None
        assert word in detail

    def test_a_dense_sequence_reports_no_break(self):
        from multimodal_pipeline.stages.activespeaker import dense_sequence_break

        assert dense_sequence_break([0, 1, 2, 3]) is None

    def test_the_break_names_the_offending_row(self):
        from multimodal_pipeline.stages.activespeaker import dense_sequence_break

        # [0, 1, 3, 2]: rows 0 and 1 are right, row 2 is the first that diverges.
        assert "row 2" in dense_sequence_break([0, 1, 3, 2])
        # A gap names both halves of the fault: which number never arrived, and which
        # number arrived that should not have been there.
        gap = dense_sequence_break([0, 1, 2, 7])
        assert "missing" in gap and "3" in gap and "7" in gap
        assert "frame_number 1" in dense_sequence_break([0, 1, 1, 2])  # frame 1 repeats

    def test_missing_frame_number_rows_are_reported_as_such(self):
        from multimodal_pipeline.stages.activespeaker import dense_sequence_break

        assert "row 1 has no frame_number" in dense_sequence_break([0, None, 2])

    def _reject(self, seeded, mutate):
        document = make_frames_document()
        mutate(document)
        seeded.artifact("activespeaker_raw").write_text(json.dumps(document), encoding="utf-8")
        stage = ActiveSpeakerStage()
        resample(seeded, stage)
        # normalize is what turns the raw frames into the parquet that validate reads, so
        # it runs first: the dense check lives in validate and inspects the table, not the
        # worker JSON. This is the stale-rerun shape the operator actually hits.
        stage.normalize(seeded)
        return stage

    def test_reordered_frames_are_rejected_by_name(self, seeded):
        def mutate(document):
            document["frames"][2], document["frames"][3] = (
                document["frames"][3], document["frames"][2])
        stage = self._reject(seeded, mutate)
        with pytest.raises(ValidationError, match="out of order"):
            stage.validate(seeded)

    def test_duplicate_frames_are_rejected_by_name(self, seeded):
        def mutate(document):
            document["frames"][3]["frame_25fps"] = 1   # duplicate of frame 1
            document["frames"][3]["timestamp_sec"] = 0.04
        stage = self._reject(seeded, mutate)
        with pytest.raises(ValidationError, match="duplicate"):
            stage.validate(seeded)

    def test_the_rejection_is_written_to_the_stage_log(self, seeded):
        """R3-001's actual complaint: a rejected table left no trace in the log."""
        def mutate(document):
            document["frames"][2]["frame_25fps"] = 9
        stage = self._reject(seeded, mutate)
        log_path = seeded.paths.dataset_dir / "logs" / "test.log"   # the fixture's StageLogger
        before = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
        with pytest.raises(ValidationError):
            stage.validate(seeded)
        written = log_path.read_text(encoding="utf-8")[len(before):]
        assert "WARNING" in written
        assert "missing" in written

    def test_a_healthy_table_logs_no_dense_warning(self, seeded):
        """Otherwise the new warning is just noise on every valid revalidation."""
        def mutate(document):
            pass
        stage = self._reject(seeded, mutate)
        log_path = seeded.paths.dataset_dir / "logs" / "test.log"
        before = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
        stage.validate(seeded)
        written = log_path.read_text(encoding="utf-8")[len(before):]
        assert "not dense" not in written
