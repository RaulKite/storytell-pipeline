"""Raw tool output → canonical rows. The silent-corruption surface of the pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from multimodal_pipeline.normalization import (
    BODY_25_NAME_LIST,
    diarization_turn_rows,
    iterate_keypoints,
    openpose_frame_number,
    openpose_frame_rows,
    parse_rttm,
    whisperx_segment_rows,
    whisperx_word_rows,
)
from multimodal_pipeline.schemas import BODY_25_KEYPOINT_NAMES, HAND_KEYPOINT_NAMES

PROJECT_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def triple(x: float, y: float, score: float) -> list[float]:
    return [x, y, score]


def body_array(scores: float = 0.9, points: int = 25) -> list[float]:
    """A BODY_25 array: 25 (or 26 with Background) x/y/score triples."""
    return [value for i in range(points) for value in (float(i), float(i * 2), scores)]


class TestWhisperxSegments:
    def test_minimal_payload(self) -> None:
        payload = {"language": "es", "segments": [{"start": 0.0, "end": 2.5, "text": " Hola ",
                                                   "avg_logprob": -0.3}]}
        rows = whisperx_segment_rows(payload, "vid")
        assert len(rows) == 1
        row = rows[0]
        assert row["video_id"] == "vid"
        assert row["segment_id"] == "seg000001"
        assert row["language"] == "es"
        assert row["text"] == "Hola"
        assert row["duration"] == pytest.approx(2.5)
        assert row["confidence"] == pytest.approx(-0.3)
        assert row["speaker_id"] is None

    def test_ids_come_from_the_model_when_present(self) -> None:
        payload = {"segments": [{"id": 3, "start": 0, "end": 1}, {"id": 7, "start": 1, "end": 2}]}
        assert [r["segment_id"] for r in whisperx_segment_rows(payload, "v")] == ["seg000004", "seg000008"]

    def test_string_ids_are_preserved(self) -> None:
        payload = {"segments": [{"id": "custom-1", "start": 0, "end": 1}]}
        assert whisperx_segment_rows(payload, "v")[0]["segment_id"] == "custom-1"

    def test_empty_and_missing_payloads(self) -> None:
        assert whisperx_segment_rows({}, "v") == []
        assert whisperx_segment_rows({"segments": None}, "v") == []
        assert whisperx_segment_rows({"segments": []}, "v") == []

    def test_missing_times_become_null_not_zero(self) -> None:
        row = whisperx_segment_rows({"segments": [{"text": "x"}]}, "v")[0]
        assert row["start_time"] is None and row["end_time"] is None and row["duration"] is None

    def test_inverted_times_clamp_to_zero_duration(self) -> None:
        row = whisperx_segment_rows({"segments": [{"start": 5.0, "end": 2.0}]}, "v")[0]
        assert row["duration"] == 0.0

    def test_confidence_falls_back_but_never_invents(self) -> None:
        assert whisperx_segment_rows({"segments": [{"start": 0, "end": 1}]}, "v")[0]["confidence"] is None


class TestWhisperxWords:
    def test_aligned_words(self) -> None:
        payload = {"segments": [{"start": 0.0, "end": 2.0, "text": "hola mundo", "words": [
            {"word": "hola", "start": 0.0, "end": 0.5, "score": 0.9, "start_char": 0, "end_char": 4},
            {"word": "mundo", "start": 0.6, "end": 1.2, "score": 0.8, "start_char": 5, "end_char": 10},
        ]}]}
        rows = whisperx_word_rows(payload, "v")
        assert [r["word"] for r in rows] == ["hola", "mundo"]
        assert rows[0]["word_id"] == "seg000001-w00000"
        assert rows[1]["word_id"] == "seg000001-w00001"
        assert all(r["alignment_status"] == "aligned" for r in rows)
        assert rows[0]["confidence"] == pytest.approx(0.9)
        assert rows[0]["character_start"] == 0 and rows[0]["character_end"] == 4
        assert rows[1]["duration"] == pytest.approx(0.6)

    def test_inherits_segment_speaker(self) -> None:
        payload = {"segments": [{"start": 0, "end": 1, "speaker": "SPEAKER_01",
                                 "words": [{"word": "a", "start": 0, "end": 1}]}]}
        assert whisperx_word_rows(payload, "v")[0]["speaker_id"] == "SPEAKER_01"

    def test_word_level_speaker_wins(self) -> None:
        payload = {"segments": [{"start": 0, "end": 1, "speaker": "SEG",
                                 "words": [{"word": "a", "start": 0, "end": 1, "speaker": "WORD"}]}]}
        assert whisperx_word_rows(payload, "v")[0]["speaker_id"] == "WORD"

    def test_missing_word_timestamp_is_marked_not_invented(self) -> None:
        payload = {"segments": [{"start": 0, "end": 1, "words": [{"word": "a", "start": None, "end": None}]}]}
        row = whisperx_word_rows(payload, "v")[0]
        assert row["alignment_status"] == "missing_timestamp"
        assert row["start_time"] is None and row["end_time"] is None

    def test_unaligned_segment_falls_back_to_segment_timing(self) -> None:
        payload = {"segments": [{"start": 1.0, "end": 3.0, "text": "una sola linea"}]}
        rows = whisperx_word_rows(payload, "v")
        assert [r["word"] for r in rows] == ["una", "sola", "linea"]
        assert all(r["alignment_status"] == "segment_only" for r in rows)
        assert all(r["start_time"] == 1.0 and r["end_time"] == 3.0 for r in rows)
        assert all(r["confidence"] is None for r in rows)

    def test_character_offsets_in_fallback_are_relative_to_the_text(self) -> None:
        payload = {"segments": [{"start": 0, "end": 1, "text": "hola mundo"}]}
        rows = whisperx_word_rows(payload, "v")
        assert (rows[0]["character_start"], rows[0]["character_end"]) == (0, 4)
        assert (rows[1]["character_start"], rows[1]["character_end"]) == (5, 10)

    def test_word_ids_are_unique_across_segments(self) -> None:
        payload = {"segments": [
            {"start": 0, "end": 1, "words": [{"word": "a", "start": 0, "end": 1}]},
            {"start": 1, "end": 2, "words": [{"word": "b", "start": 1, "end": 2}]},
        ]}
        ids = [r["word_id"] for r in whisperx_word_rows(payload, "v")]
        assert len(ids) == len(set(ids)) == 2


class TestDiarizationTurns:
    def test_rows_are_ordered_by_start(self) -> None:
        payload = {"turns": [[5.0, 8.0, "B"], [0.0, 3.0, "A"]]}
        rows = diarization_turn_rows(payload, "v")
        assert [r["start_time"] for r in rows] == [0.0, 5.0]
        assert rows[0]["turn_id"] == "turn000001"
        assert rows[1]["duration"] == pytest.approx(3.0)
        assert rows[0]["diarization_type"] == "inclusive"

    def test_exclusive_timeline_is_labelled(self) -> None:
        rows = diarization_turn_rows({"exclusive_turns": [[0.0, 1.0, "A"]]}, "v",
                                     diarization_type="exclusive")
        assert rows[0]["diarization_type"] == "exclusive"

    def test_short_turn_rows_are_dropped(self) -> None:
        assert diarization_turn_rows({"turns": [[0.0, 1.0]]}, "v") == []

    def test_empty(self) -> None:
        assert diarization_turn_rows({}, "v") == []


class TestRttm:
    def test_parses_speaker_lines(self) -> None:
        text = (
            "SPEAKER vid 1 0.500 3.200 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
            "SPEAKER vid 1 4.000 1.000 <NA> <NA> SPEAKER_01 <NA> <NA>\n"
            "COMMENT whatever\n"
        )
        rows = parse_rttm(text)
        assert len(rows) == 2
        assert rows[0]["speaker_id"] == "SPEAKER_00"
        assert rows[0]["start_time"] == pytest.approx(0.5)
        assert rows[0]["end_time"] == pytest.approx(3.7)
        assert rows[1]["duration"] == pytest.approx(1.0)

    def test_garbage_is_skipped(self) -> None:
        assert parse_rttm("") == []
        assert parse_rttm("SPEAKER a b not_a_number 1.0 x y S1 z w") == []


class TestKeypointIteration:
    def test_triples(self) -> None:
        assert list(iterate_keypoints([1, 2, 0.5, 3, 4, 0.25])) == [(1.0, 2.0, 0.5), (3.0, 4.0, 0.25)]

    def test_partial_trailing_values_are_dropped(self) -> None:
        assert list(iterate_keypoints([1, 2, 0.5, 9, 9])) == [(1.0, 2.0, 0.5)]

    def test_empty(self) -> None:
        assert list(iterate_keypoints([])) == []

    def test_body_25_array_length_matches_the_official_source(self) -> None:
        # Verified against /opt/openpose/src/openpose/pose/poseParameters.cpp:
        # Background is the *last* entry, not the first.
        assert len(BODY_25_KEYPOINT_NAMES) == 26
        assert BODY_25_KEYPOINT_NAMES[0] == "Nose"
        assert BODY_25_KEYPOINT_NAMES[1] == "Neck"
        assert BODY_25_KEYPOINT_NAMES[25] == "Background"
        assert len(HAND_KEYPOINT_NAMES) == 21


class TestOpenposeFrameRows:
    def frame(self, *people: dict) -> dict:
        return {"version": "1.3", "people": list(people)}

    def person(self, body_points: int = 26, hand_points: int = 21, face_points: int = 70,
               score: float = 0.8) -> dict:
        return {
            "pose_keypoints_2d": body_array(score, body_points),
            "hand_left_keypoints_2d": [v for i in range(hand_points) for v in (1.0, 2.0, score)],
            "hand_right_keypoints_2d": [v for i in range(hand_points) for v in (1.0, 2.0, score)],
            "face_keypoints_2d": [v for i in range(face_points) for v in (1.0, 2.0, score)],
            "person_id": 0,
        }

    def test_one_person_full_frame(self) -> None:
        rows = openpose_frame_rows(self.frame(self.person()), "vid", 42, 1.68)
        assert len(rows["body"]) == 25  # Background excluded by default
        assert len(rows["hands"]) == 42  # 21 left + 21 right
        assert len(rows["face"]) == 70
        assert all(row["frame_number"] == 42 and row["timestamp"] == pytest.approx(1.68)
                   for group in rows.values() for row in group)
        assert {row["keypoint_name"] for row in rows["body"]} == set(BODY_25_NAME_LIST) - {"Background"}

    def test_background_is_opt_in(self) -> None:
        rows = openpose_frame_rows(self.frame(self.person()), "v", 0, 0.0, include_background=True)
        assert len(rows["body"]) == 26
        background = [row for row in rows["body"] if row["keypoint_name"] == "Background"]
        assert len(background) == 1 and background[0]["keypoint_id"] == 25

    def test_zero_score_keypoints_are_dropped(self) -> None:
        person = self.person(score=0.0)
        rows = openpose_frame_rows(self.frame(person), "v", 0, 0.0)
        assert rows["body"] == [] and rows["hands"] == [] and rows["face"] == []

    def test_partially_detected_person_keeps_the_confident_points(self) -> None:
        person = self.person()
        # Zero out the score of keypoint id 3 (RElbow) only.
        person["pose_keypoints_2d"][3 * 3 + 2] = 0.0
        rows = openpose_frame_rows(self.frame(person), "v", 0, 0.0)
        names = {row["keypoint_name"] for row in rows["body"]}
        assert "RElbow" not in names and len(rows["body"]) == 24
        assert "Nose" in names

    def test_body_keypoint_ids_match_the_official_order(self) -> None:
        rows = openpose_frame_rows(self.frame(self.person()), "v", 0, 0.0, include_background=True)
        by_id = {row["keypoint_id"]: row["keypoint_name"] for row in rows["body"]}
        assert by_id[0] == "Nose"
        assert by_id[1] == "Neck"
        assert by_id[2] == "RShoulder"
        assert by_id[24] == "RHeel"
        assert by_id[25] == "Background"

    def test_hands_are_labelled_by_side(self) -> None:
        rows = openpose_frame_rows(self.frame(self.person()), "v", 0, 0.0)
        sides = {row["hand"] for row in rows["hands"]}
        assert sides == {"left", "right"}
        assert sum(1 for row in rows["hands"] if row["hand"] == "left") == 21

    def test_missing_hand_or_face_arrays_simply_produce_nothing(self) -> None:
        person = {"pose_keypoints_2d": body_array(0.9, 26)}
        rows = openpose_frame_rows(self.frame(person), "v", 0, 0.0)
        assert len(rows["body"]) == 25 and rows["hands"] == [] and rows["face"] == []

    def test_people_have_frame_local_detection_indices(self) -> None:
        rows = openpose_frame_rows(self.frame(self.person(), self.person()), "v", 0, 0.0)
        assert {row["detection_index"] for row in rows["body"]} == {0, 1}

    def test_extra_keypoints_beyond_body_25_are_ignored(self) -> None:
        person = {"pose_keypoints_2d": body_array(0.9, 40)}
        rows = openpose_frame_rows(self.frame(person), "v", 0, 0.0)
        assert len(rows["body"]) == 25

    def test_face_landmarks_stay_within_the_70_point_canonical_set(self) -> None:
        person = self.person(face_points=90)
        rows = openpose_frame_rows(self.frame(person), "v", 0, 0.0)
        assert len(rows["face"]) == 70
        assert {row["landmark_id"] for row in rows["face"]} == set(range(70))

    def test_empty_people(self) -> None:
        rows = openpose_frame_rows({"version": "1.3", "people": []}, "v", 0, 0.0)
        assert rows == {"body": [], "hands": [], "face": []}

    def test_no_people_key(self) -> None:
        assert openpose_frame_rows({}, "v", 0, 0.0) == {"body": [], "hands": [], "face": []}

    def test_coordinates_and_confidence_are_floats(self) -> None:
        rows = openpose_frame_rows(self.frame(self.person()), "v", 0, 0.0)
        for row in rows["body"]:
            assert isinstance(row["x"], float) and isinstance(row["y"], float)
            assert isinstance(row["confidence"], float)
            assert math.isfinite(row["x"]) and math.isfinite(row["y"])


class TestFrameNumberParsing:
    @pytest.mark.parametrize("name,expected", [
        ("video_000000000042_keypoints.json", 42),
        ("video_000000000000_keypoints.json", 0),
        ("openpose_00000123_keypoints.json", 123),
        ("clip_0000000000000099_keypoints.json", 99),
    ])
    def test_canonical_names(self, name: str, expected: int | None) -> None:
        assert openpose_frame_number(name) == expected

    def test_unparseable(self) -> None:
        assert openpose_frame_number("keypoints.json") is None

    def test_sequence_order_is_sorted_numerically_not_lexically(self) -> None:
        names = ["video_000000000009_keypoints.json", "video_000000000010_keypoints.json"]
        assert sorted(names, key=lambda n: (openpose_frame_number(n) or 0, n)) == names


class TestOpenposeRealFrame:
    """Golden test on one genuine OpenPose 1.3 frame produced on this machine.

    Synthetic people cannot catch layout surprises: this frame has 25 pose points
    (Background is *not* emitted), ``person_id`` as a **list**, and partially
    detected limbs — none of which the hand-built fixtures reproduce.
    """

    @pytest.fixture
    def frame(self) -> dict:
        path = PROJECT_FIXTURES / "openpose_frame_real.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_layout_assumptions(self, frame: dict) -> None:
        person = frame["people"][0]
        assert len(person["pose_keypoints_2d"]) == 75
        assert len(person["hand_left_keypoints_2d"]) == 63
        assert len(person["face_keypoints_2d"]) == 210
        # OpenPose emits person_id as a list, so it cannot be used as an int index.
        assert isinstance(person["person_id"], list)

    def test_normalisation_of_the_real_frame(self, frame: dict) -> None:
        rows = openpose_frame_rows(frame, "real", 4, 0.16)
        detected_pose = sum(1 for i in range(25) if person_score(frame, 0, "pose_keypoints_2d", i) > 0)
        assert len(rows["body"]) == detected_pose == 11
        left = sum(1 for i in range(21) if person_score(frame, 0, "hand_left_keypoints_2d", i) > 0)
        right = sum(1 for i in range(21) if person_score(frame, 0, "hand_right_keypoints_2d", i) > 0)
        assert len(rows["hands"]) == left + right == 21  # right hand fully undetected
        assert all(row["confidence"] > 0 for group in rows.values() for row in group)
        assert all(row["keypoint_name"] in BODY_25_NAME_LIST for row in rows["body"])
        assert {row["hand"] for row in rows["hands"]} == {"left"}

    def test_face_is_the_full_70_landmark_set(self, frame: dict) -> None:
        rows = openpose_frame_rows(frame, "real", 4, 0.16)
        assert len(rows["face"]) == 70
        assert {row["landmark_id"] for row in rows["face"]} == set(range(70))

    def test_every_row_carries_timeline_and_provenance(self, frame: dict) -> None:
        rows = openpose_frame_rows(frame, "real", 4, 0.16)
        for group in rows.values():
            for row in group:
                assert row["video_id"] == "real"
                assert row["frame_number"] == 4
                assert row["timestamp"] == pytest.approx(0.16)
                assert row["schema_version"] == "1.0"

    def test_background_never_appears_because_openpose_does_not_emit_it(self, frame: dict) -> None:
        rows = openpose_frame_rows(frame, "real", 4, 0.16, include_background=True)
        assert [row for row in rows["body"] if row["keypoint_name"] == "Background"] == []


def person_score(frame: dict, person_index: int, key: str, point: int) -> float:
    return float(frame["people"][person_index][key][point * 3 + 2])


class TestDiarizationValidationBindsToRequest:
    """`validate` must accept the raw a run produced and reject one from another config.

    It used to compare a request hash read out of the worker's own JSON, which never
    contains one, so the comparison was None != <hash> and rejected every successful
    diarization. That was invisible while the stage skipped for a missing HF token.
    """

    @staticmethod
    def _seed(context, stage):
        import json
        import pyarrow as pa
        from multimodal_pipeline.schemas import SPEAKER_TURNS_SCHEMA, write_table
        from multimodal_pipeline.stages.base import stamp_raw

        raw = context.artifact("diarization_raw")
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(json.dumps({
            "video_id": context.video_id,
            "pipeline_id": context.config.diarization.pipeline,
            "turns": [[0.0, 1.0, "SPEAKER_00"]], "exclusive_turns": [], "rttm": "",
        }), encoding="utf-8")
        stamp_raw(raw, request=stage.request(context),
                  digest=stage.request_digest(context), worker=None)
        write_table(context.artifact("speaker_turns"),
                    pa.Table.from_pylist(
                        [{"schema_version": "1.0", "video_id": context.video_id,
                          "turn_index": 0, "start_time": 0.0, "end_time": 1.0,
                          "duration": 1.0, "speaker_id": "SPEAKER_00",
                          "diarization_type": "inclusive", "confidence": None}],
                        schema=SPEAKER_TURNS_SCHEMA),
                    SPEAKER_TURNS_SCHEMA)
        return raw

    def test_matching_request_validates(self, context):
        from multimodal_pipeline.stages.diarization import DiarizationStage

        stage = DiarizationStage()
        self._seed(context, stage)
        assert stage.validate(context)["speakers"] == 1

    def test_raw_from_a_different_config_is_rejected(self, context):
        import pytest as _pt
        from multimodal_pipeline.exceptions import ValidationError
        from multimodal_pipeline.stages.diarization import DiarizationStage

        stage = DiarizationStage()
        self._seed(context, stage)
        # Same raw file, different request: the sidecar must notice.
        context.config.diarization.device = "cpu"
        with _pt.raises(ValidationError, match="different configuration"):
            stage.validate(context)
