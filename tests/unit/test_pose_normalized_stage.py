"""What the `pose_normalized` stage does as a pipeline member, and what it refuses to do.

The coordinate algebra is `test_pose_normalize_math.py`'s job; the agreement with the R
reference is `test_pose_normalized.py`'s. What is tested here is the stage contract: that
it skips for the right reason, names absence instead of encoding it, fingerprints the
basis triple so a changed frame cannot silently redefine the table, validates its own
output, and does not leave a stale table behind when the configuration stops producing
one.

Nothing here is mocked at the level the stage reads. Every case seeds a real
``pose/body.parquet`` and runs the real stage, because the failure being guarded against
(a stale fused or derived table read as current) lives in the interaction between the
stage and the files on disk, not inside a function.
"""

from __future__ import annotations

from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from multimodal_pipeline.exceptions import StageError, ValidationError
from multimodal_pipeline.pose_normalize import (
    BASIS_DEGENERATE,
    BASIS_MISSING_JOINT,
    BASIS_OK,
    VALUE_BASIS_UNUSABLE,
    VALUE_NO_COORDINATE,
    VALUE_NORMALIZED,
)
from multimodal_pipeline.schemas import (
    BODY_25_KEYPOINT_NAMES,
    BODY_SCHEMA,
    POSE_NORMALIZED_SCHEMA,
    read_table,
    write_table,
)
from multimodal_pipeline.stages.metadata import sha256_of
from multimodal_pipeline.stages.pose_normalized import PoseNormalizedStage

def run_stage(context) -> dict[str, Any]:
    """Run the real stage over whatever is on disk and return its provenance extras.

    The stage is never called method-by-method: skip, execute, validate and prune are a
    contract with the filesystem, and driving them one at a time would test the test.
    """
    outcome = PoseNormalizedStage().run(context)
    assert outcome.status == "completed", outcome.message
    return outcome.detail["provenance"]["extra"]


def body_row(frame: int, person: int, keypoint_id: int, x: float | None, y: float | None,
             *, confidence: float = 0.9, timestamp: float = 0.0,
             video_id: str = "conversation_001") -> dict[str, Any]:
    """One BODY_SCHEMA row, built through the schema so its types are the real ones."""
    values = {
        "schema_version": "1.0", "video_id": video_id, "frame_number": frame,
        "timestamp": timestamp, "detection_index": person, "keypoint_id": keypoint_id,
        "keypoint_name": BODY_25_KEYPOINT_NAMES[keypoint_id], "x": x, "y": y,
        "confidence": confidence,
    }
    return pa.Table.from_pylist([values], schema=BODY_SCHEMA).to_pylist()[0]


def seed_body(context, rows: Sequence[dict[str, Any]]) -> None:
    write_table(context.artifact("pose_body"),
                pa.Table.from_pylist(list(rows), schema=BODY_SCHEMA), BODY_SCHEMA)


def log_recorder(context) -> list[str]:
    messages: list[str] = []

    def _log(message: str, level: int = 20) -> None:
        messages.append(str(message))

    context.log = _log
    return messages


def a_person(*, origin: tuple[float, float] = (100.0, 300.0),
             neck: tuple[float, float] = (100.0, 100.0),
             extra: dict[int, tuple[float, float]] | None = None,
             without: Sequence[int] = (), frame: int = 0, person: int = 0,
             timestamp: float = 0.0) -> list[dict[str, Any]]:
    """Rows for one person-frame: MidHip, Neck and anything else asked for."""
    points: dict[int, tuple[float, float]] = {8: origin, 1: neck}
    points.update(extra or {})
    return [body_row(frame, person, keypoint_id, x, y, timestamp=timestamp)
            for keypoint_id, (x, y) in sorted(points.items()) if keypoint_id not in without]


class TestStageGating:
    def test_disabled_by_config(self, context):
        context.config.pose_normalized.enabled = False
        seed_body(context, a_person())
        enabled, reason = PoseNormalizedStage().enabled(context)
        assert enabled is False
        assert reason == "pose_normalized.enabled = false"

    def test_enabled_when_the_body_table_is_there(self, context):
        seed_body(context, a_person())
        assert PoseNormalizedStage().enabled(context) == (True, "")

    def test_no_body_table_names_the_stage_to_run(self, context):
        enabled, reason = PoseNormalizedStage().enabled(context)
        assert enabled is False
        assert "pose/body.parquet unavailable" in reason
        assert "openpose" in reason

    def test_openpose_switched_off_says_so_rather_than_blaming_a_missing_file(self, context):
        context.config.openpose.enabled = False
        enabled, reason = PoseNormalizedStage().enabled(context)
        assert enabled is False
        assert "openpose.enabled = false" in reason

    def test_a_body_table_that_was_never_published_gets_its_own_reason(self, context):
        # openpose runs but publishes no body table, which is a different fix.
        context.config.openpose.body.enabled = False
        enabled, reason = PoseNormalizedStage().enabled(context)
        assert enabled is False
        assert "openpose.body.enabled = false" in reason

    def test_nothing_to_normalise_writes_no_table(self, context):
        """Spec §20.4: skip with a reason, not an empty table a reader would misread."""
        outcome = PoseNormalizedStage().run(context)
        assert outcome.status == "skipped"
        assert not context.artifact("pose_normalized").exists()

    def test_the_audio_branch_is_never_read(self, context):
        """The stage's declared inputs are the whole of its dependency on the pipeline."""
        stage = PoseNormalizedStage()
        assert stage.inputs == ("pose_body",)
        assert not any(name.startswith(("speech", "speaker", "acoustic", "linguistic"))
                       for name in stage.inputs)
        seed_body(context, a_person())
        # No audio artifact, no transcript, no diarizer output exists, and the run completes.
        assert stage.run(context).status == "completed"


class TestStageExecution:
    def test_one_row_per_body_row_in_the_body_tables_order(self, context):
        rows = (a_person(extra={0: (90.0, 60.0), 5: (160.0, 120.0)}, frame=0)
                + a_person(extra={0: (95.0, 62.0)}, frame=1, timestamp=0.04))
        seed_body(context, rows)
        run_stage(context)
        body = read_table(context.artifact("pose_body")).to_pylist()
        ours = read_table(context.artifact("pose_normalized")).to_pylist()
        assert len(ours) == len(body)
        assert [(r["frame_number"], r["detection_index"], r["keypoint_id"]) for r in ours] \
            == [(r["frame_number"], r["detection_index"], r["keypoint_id"]) for r in body]

    def test_the_table_matches_the_declared_schema(self, context):
        seed_body(context, a_person(extra={4: (20.0, 250.0)}))
        run_stage(context)
        assert read_table(context.artifact("pose_normalized")).schema == POSE_NORMALIZED_SCHEMA

    def test_the_pixel_table_is_never_touched(self, context):
        """§20.4: pixels are the measured quantity; this stage writes beside them."""
        seed_body(context, a_person())
        before = context.artifact("pose_body").read_bytes()
        run_stage(context)
        assert context.artifact("pose_body").read_bytes() == before

    def test_the_frame_and_timestamp_come_from_the_body_row(self, context):
        seed_body(context, a_person(extra={0: (90.0, 60.0)}, frame=7)
                  + a_person(frame=8, timestamp=0.32))
        run_stage(context)
        ours = read_table(context.artifact("pose_normalized")).to_pylist()
        assert {row["frame_number"] for row in ours} == {7, 8}
        assert {row["timestamp"] for row in ours} == {0.0, 0.32}

    def test_the_provenance_metadata_names_the_frame_and_the_reference(self, context):
        seed_body(context, a_person())
        run_stage(context)
        metadata = pq.read_metadata(context.artifact("pose_normalized")).metadata
        values = {key.decode(): value.decode() for key, value in metadata.items()}
        assert values["origin_keypoint"] == "MidHip"
        assert values["basis_keypoint"] == "Neck"
        assert "multimolang" in values["reference"]
        assert "pixels" in values["coordinate_space"]

    def test_the_run_logs_the_row_count_and_the_states(self, context):
        seed_body(context, a_person(extra={0: (90.0, 60.0)}))
        messages = log_recorder(context)
        extra = run_stage(context)
        assert extra["rows"] == 3
        assert extra["basis_states"] == {"basis_ok": 3}
        assert extra["transformation"] == ["MidHip", "Neck", "perpendicular"]
        assert any("3 row(s) -> normalized.parquet" in message for message in messages)

    def test_an_empty_body_table_is_an_empty_honest_table(self, context):
        """A video with no person is a valid outcome, and the stage says it out loud."""
        seed_body(context, [])
        messages = log_recorder(context)
        assert PoseNormalizedStage().run(context).status == "completed"
        assert read_table(context.artifact("pose_normalized")).num_rows == 0
        assert any("0 row" in message for message in messages)
        summary = PoseNormalizedStage().validate(context)
        # The row count is checked against pose_body here, because pose_body is present.
        assert summary["rows"] == 0 and "row_count_checked" not in summary

    def test_a_non_contiguous_body_table_is_refused_with_the_reason(self, context):
        """Streaming one frame at a time is only sound if frames are contiguous.

        A table that revisits a frame would have its second half normalised against a
        person-frame the stage had already finished with — i.e. against a missing basis,
        quietly. Refusing is the only honest option.
        """
        rows = a_person(frame=0) + a_person(frame=1) + a_person(frame=0, person=1)
        seed_body(context, rows)
        with pytest.raises(StageError, match="not contiguous"):
            PoseNormalizedStage().run(context)

    def test_a_body_table_missing_a_read_column_names_it(self, context):
        seed_body(context, a_person())
        path = context.artifact("pose_body")
        pq.write_table(pq.read_table(path).drop(["confidence"]), path)
        # confidence is not read by the transform; x is, and dropping it must be a message.
        assert PoseNormalizedStage().run(context).status == "completed"
        seed_body(context, a_person())
        pq.write_table(pq.read_table(path).drop(["x"]), path)
        with pytest.raises(ValidationError, match=r"missing columns: x"):
            PoseNormalizedStage().run(context)


class TestAbsenceIsNamed:
    """§20.4: missing keypoints are the normal case, and each kind gets a name."""

    def rows(self, context) -> dict[int, dict[str, Any]]:
        return {row["keypoint_id"]: row
                for row in read_table(context.artifact("pose_normalized")).to_pylist()}

    def test_a_person_frame_without_its_origin_keeps_every_row_and_says_why(self, context):
        seed_body(context, a_person(without=[8]))
        run_stage(context)
        ours = self.rows(context)
        # The Neck row survives — the person did not stop existing because the hip was
        # never located. It simply has nowhere to be expressed.
        assert set(ours) == {1}
        assert ours[1]["basis_state"] == BASIS_MISSING_JOINT
        assert ours[1]["x_norm"] is None and ours[1]["y_norm"] is None
        assert ours[1]["value_status"] == VALUE_BASIS_UNUSABLE
        assert "MidHip" in ours[1]["basis_detail"]

    def test_a_person_frame_whose_joints_coincide_is_degenerate(self, context):
        seed_body(context, a_person(origin=(64.0, 128.0), neck=(64.0, 128.0),
                                    extra={0: (70.0, 100.0)}))
        run_stage(context)
        ours = self.rows(context)
        assert {row["basis_state"] for row in ours.values()} == {BASIS_DEGENERATE}
        assert all(row["x_norm"] is None for row in ours.values())
        assert all(row["value_status"] == VALUE_BASIS_UNUSABLE for row in ours.values())
        assert "determinant" in ours[0]["basis_detail"]

    def test_a_keypoint_with_a_zero_coordinate_keeps_its_row_and_loses_the_number(self,
                                                                                 context):
        """The per-coordinate rule, at stage level.

        A keypoint reported at (0, 250) has a *known* y and an unknown x. The reference
        keeps the y and NA-ises the x; either way there is no coordinate pair to place, so
        this row must exist, be null, and be labelled ``no_coordinate`` — never a zero,
        which on these axes would mean "exactly on the hip line".
        """
        seed_body(context, a_person(extra={4: (0.0, 250.0), 7: (200.0, 250.0)}))
        run_stage(context)
        ours = self.rows(context)
        assert ours[4]["value_status"] == VALUE_NO_COORDINATE
        assert ours[4]["x_norm"] is None and ours[4]["y_norm"] is None
        assert ours[4]["basis_state"] == BASIS_OK  # the frame is fine; this joint is not
        assert ours[7]["value_status"] == VALUE_NORMALIZED

    def test_a_basis_joint_with_a_zero_coordinate_makes_the_frame_unusable(self, context):
        # MidHip at x = 0 has no usable x, so there is no origin: that is a missing joint,
        # not a body whose hip is at the left edge of the frame.
        seed_body(context, a_person(origin=(0.0, 300.0), extra={0: (90.0, 60.0)}))
        run_stage(context)
        ours = self.rows(context)
        assert {row["basis_state"] for row in ours.values()} == {BASIS_MISSING_JOINT}
        assert "half measured" in ours[0]["basis_detail"]

    def test_the_origin_and_basis_joints_get_real_numbers_not_nulls(self, context):
        # Their coordinates are the definition of the frame, so they are the two rows that
        # are always (0,0) and (1,0) — and a table that nulled them would be hiding them.
        seed_body(context, a_person())
        run_stage(context)
        ours = self.rows(context)
        assert ours[8]["value_status"] == VALUE_NORMALIZED
        assert (ours[8]["x_norm"], ours[8]["y_norm"]) == (0.0, 0.0)
        assert ours[1]["value_status"] == VALUE_NORMALIZED
        assert ours[1]["x_norm"] == pytest.approx(1.0)

    def test_two_people_in_one_frame_get_their_own_frames(self, context):
        seed_body(context, a_person(person=0) + a_person(person=1, origin=(400.0, 300.0),
                                   neck=(400.0, 250.0)))
        run_stage(context)
        rows = read_table(context.artifact("pose_normalized")).to_pylist()
        by_person: dict[int, dict[int, dict[str, Any]]] = {}
        for row in rows:
            by_person.setdefault(row["detection_index"], {})[row["keypoint_id"]] = row
        # Both are basis_ok, and the same MidHip pixel in the second person's frame is still
        # (0, 0) — a per-person frame, not one shared origin.
        assert by_person[0][8]["basis_state"] == BASIS_OK
        assert by_person[1][8]["basis_state"] == BASIS_OK
        assert (by_person[1][8]["x_norm"], by_person[1][8]["y_norm"]) == (0.0, 0.0)
        assert by_person[1][1]["x_norm"] == pytest.approx(1.0)

    def test_a_missing_person_does_not_make_the_others_unframeable(self, context):
        seed_body(context, a_person(person=0, without=[8]) + a_person(person=1))
        run_stage(context)
        rows = read_table(context.artifact("pose_normalized")).to_pylist()
        states = {(row["detection_index"]): row["basis_state"] for row in rows}
        assert states[0] == BASIS_MISSING_JOINT
        assert states[1] == BASIS_OK


class TestStageFingerprint:
    def test_a_rewritten_body_table_invalidates_the_normalisation(self, context):
        """Re-extracting pose must not leave coordinates computed against the old pixels."""
        seed_body(context, a_person())
        stage = PoseNormalizedStage()
        before = stage.config_fingerprint(context)["body_digest"]
        assert before is not None
        seed_body(context, a_person(extra={0: (91.5, 61.5)}))
        context.scratch.clear()  # the digest is memoised per run, as in every other stage
        after = stage.config_fingerprint(context)["body_digest"]
        assert after != before and after is not None

    def test_an_absent_input_is_recorded_as_absent(self, context):
        payload = PoseNormalizedStage().config_fingerprint(context)
        assert payload["body_digest"] is None

    def test_identical_inputs_reproduce_the_fingerprint(self, context):
        seed_body(context, a_person())
        assert (PoseNormalizedStage().config_fingerprint(context)
                == PoseNormalizedStage().config_fingerprint(context))

    def test_the_normaliser_source_is_in_the_fingerprint(self, context):
        """The python that computes the coordinates has to be part of what reuse compares.

        This stage has no worker, so nothing in the config or the input digests can notice an
        edit to `pose_normalize.py`. The shape is asserted next to the key: a value that stopped
        being a sha256 hex would still move when the source moved, and a reviewer would still
        read the fingerprint as covering the code. Length is compared against a real
        `sha256_of` result rather than against a remembered 64.
        """
        seed_body(context, a_person())
        payload = PoseNormalizedStage().config_fingerprint(context)
        digest = payload["_python_code_sha256"]
        assert digest is not None
        assert len(digest) == len(sha256_of(context.artifact("pose_body")))
        assert all(c in "0123456789abcdef" for c in digest)

    def test_editing_the_module_that_computes_rows_changes_the_fingerprint(self, context, monkeypatch):
        """The defect this stage was missing: new numbers from the same config and bytes.

        Patched at the seam that reads the source rather than by rewriting the repository, so
        the test stays deterministic and never touches a tracked file. Everything else in the
        payload is held fixed on purpose: if the assertion passed for a reason other than the
        source digest, this test would be evidence for nothing.
        """
        seed_body(context, a_person())
        stage = PoseNormalizedStage()
        before = stage.config_fingerprint(context)

        # Patched where the stage looks it up: `config_fingerprint` imports the helper from
        # `stages.base` inside the call, so the attribute on that module is the seam.
        import multimodal_pipeline.stages.base as base_module
        monkeypatch.setattr(base_module, "python_source_digest", lambda *modules: "f" * 64)
        after = stage.config_fingerprint(context)

        assert after["_python_code_sha256"] == "f" * 64
        assert after != before
        assert {k: v for k, v in after.items() if k != "_python_code_sha256"} \
            == {k: v for k, v in before.items() if k != "_python_code_sha256"}, \
            "something besides the source digest moved, so this proves nothing about it"

    def test_the_source_digest_is_stable_between_two_calls(self, context):
        """A fingerprint that drifted run to run would invalidate the whole corpus every time."""
        import multimodal_pipeline.pose_normalize as pose_normalize_module
        import multimodal_pipeline.stages.pose_normalized as pose_normalized_stage_module
        from multimodal_pipeline.stages.base import python_source_digest

        first = PoseNormalizedStage().config_fingerprint(context)["_python_code_sha256"]
        second = PoseNormalizedStage().config_fingerprint(context)["_python_code_sha256"]
        assert first == second
        # And it is the digest of the two modules the call site claims to cover.
        assert first == python_source_digest(pose_normalize_module,
                                            pose_normalized_stage_module)

    def test_the_orchestrator_reuses_a_completed_normalisation(self, context):
        """A completed run with unchanged inputs is reused, and only then.

        The hash is built the way ``VideoRunner.config_payload_for`` builds it (stage
        fingerprint + stage name + schema version), because that is the value
        ``should_reuse`` compares against.
        """
        from multimodal_pipeline.config import stable_hash
        from multimodal_pipeline.stages.base import should_reuse

        seed_body(context, a_person(extra={0: (90.0, 60.0)}))
        stage = PoseNormalizedStage()
        stage.run(context)
        context.state.mark_completed(stage.name)

        payload = dict(stage.config_fingerprint(context))
        payload.setdefault("stage", stage.name)
        payload.setdefault("schema_version", context.tools.get("schema_version"))
        digest = stable_hash(payload, length=16)
        context.state.stage(stage.name).config_hash = digest
        context.state.stage(stage.name).dependency_hash = "y"
        assert should_reuse(stage, context, config_hash=digest,
                            dependency_hash="y", force=False)[0] is True
        assert should_reuse(stage, context, config_hash="stale",
                            dependency_hash="y", force=False) == (
            False, "configuration changed")

        context.config.pose_normalized.origin_keypoint = "RHip"
        moved = stable_hash(dict(stage.config_fingerprint(context), stage=stage.name,
                                 schema_version=context.tools["schema_version"]), length=16)
        assert moved != digest
        assert should_reuse(stage, context, config_hash=moved,
                            dependency_hash="y", force=False) == (
            False, "configuration changed")


class TestStageValidation:
    def seeded(self, context) -> PoseNormalizedStage:
        seed_body(context, a_person(extra={0: (90.0, 60.0), 5: (160.0, 120.0)}))
        stage = PoseNormalizedStage()
        assert stage.run(context).status == "completed"
        return stage

    def rewrite_column(self, context, **values: Any) -> None:
        """Overwrite one column of the normalised table on disk with a single value."""
        path = context.artifact("pose_normalized")
        table = pq.read_table(path)
        for name, value in values.items():
            index = table.schema.get_field_index(name)
            array = pa.array([value] * table.num_rows, type=table.schema.field(name).type)
            table = table.set_column(index, name, array)
        pq.write_table(table, path)

    def test_a_missing_output_is_named(self, context):
        stage = self.seeded(context)
        context.artifact("pose_normalized").unlink()
        with pytest.raises(ValidationError, match="pose_normalized missing"):
            stage.validate(context)

    def test_a_stale_table_names_the_missing_column_instead_of_raising_keyerror(self, context):
        stage = self.seeded(context)
        path = context.artifact("pose_normalized")
        pq.write_table(pq.read_table(path).drop(["value_status", "basis_detail"]), path)
        with pytest.raises(ValidationError, match=r"missing columns: .*value_status"):
            stage.validate(context)

    def test_a_dropped_row_is_reported_as_staleness_with_the_fix(self, context):
        stage = self.seeded(context)
        path = context.artifact("pose_normalized")
        pq.write_table(pq.read_table(path).slice(0, 2), path)
        with pytest.raises(ValidationError) as excinfo:
            stage.validate(context)
        message = str(excinfo.value)
        # Both counts, the input's name, the diagnosis, and the remedy.
        assert "normalized.parquet" in message and "pose_body" in message
        assert " 2 " in message and " 4 " in message
        assert "stale" in message and "rerun pose_normalized" in message

    def test_a_table_written_for_another_basis_is_refused(self, context):
        """The numbers mean something else, and only the columns say so.

        A basis change normally reruns the stage through the fingerprint; this is the case
        where the state file does not know (a copied-in table, or a reset status.json), and
        validation is the last thing standing between it and a reader.
        """
        stage = self.seeded(context)
        self.rewrite_column(context, origin_keypoint_name="RHip")
        with pytest.raises(ValidationError, match="different basis frame|basis frame"):
            stage.validate(context)
        assert PoseNormalizedStage().outputs_present(context) is False

    def test_a_basis_state_outside_the_vocabulary_is_rejected(self, context):
        stage = self.seeded(context)
        self.rewrite_column(context, basis_state="probably_fine")
        with pytest.raises(ValidationError, match="unrecognised basis_state"):
            stage.validate(context)

    def test_a_value_status_outside_the_vocabulary_is_rejected(self, context):
        stage = self.seeded(context)
        self.rewrite_column(context, value_status="missing")
        with pytest.raises(ValidationError, match="unrecognised value_status"):
            stage.validate(context)

    def test_coordinates_inside_a_frame_that_was_never_built_are_rejected(self, context):
        stage = self.seeded(context)
        self.rewrite_column(context, basis_state="basis_degenerate")
        with pytest.raises(ValidationError, match="while basis_state is"):
            stage.validate(context)

    def test_a_normalized_claim_with_null_coordinates_is_rejected(self, context):
        stage = self.seeded(context)
        self.rewrite_column(context, x_norm=None, y_norm=None)
        with pytest.raises(ValidationError, match="claims normalized with null"):
            stage.validate(context)

    def test_basis_unusable_inside_a_usable_frame_is_rejected(self, context):
        # Nulls too, so the only thing wrong with the table is the claim itself: with
        # coordinates still present the validator reports the coordinates first (a row
        # carrying numbers while claiming there is no frame is the louder defect), and this
        # invariant would never be the one named.
        stage = self.seeded(context)
        self.rewrite_column(context, x_norm=None, y_norm=None,
                            value_status=VALUE_BASIS_UNUSABLE)
        with pytest.raises(ValidationError, match="claims basis_unusable"):
            stage.validate(context)

    def test_an_empty_basis_detail_is_rejected(self, context):
        stage = self.seeded(context)
        self.rewrite_column(context, basis_detail="")
        with pytest.raises(ValidationError, match="empty basis_detail"):
            stage.validate(context)

    def test_a_keypoint_id_that_disagrees_with_its_name_is_rejected(self, context):
        # A consumer selecting by one would get the joint the other names.
        stage = self.seeded(context)
        path = context.artifact("pose_normalized")
        table = pq.read_table(path)
        index = table.schema.get_field_index("keypoint_name")
        names = [BODY_25_KEYPOINT_NAMES[int(value) + 1 if int(value) < 24 else 0]
                 for value in table.column("keypoint_id").to_pylist()]
        pq.write_table(table.set_column(index, "keypoint_name",
                                        pa.array(names, type=pa.string())), path)
        with pytest.raises(ValidationError, match="pairs keypoint_id"):
            stage.validate(context)

    def test_a_background_row_is_rejected(self, context):
        # BODY_25 index 25 is a filler channel that pose/body.parquet never carries, so a
        # normalised table containing one was not written from this input.
        stage = self.seeded(context)
        self.rewrite_column(context, keypoint_id=25, keypoint_name="Background")
        with pytest.raises(ValidationError, match="Background is a filler"):
            stage.validate(context)

    def test_rows_from_another_video_are_rejected(self, context):
        stage = self.seeded(context)
        self.rewrite_column(context, video_id="someone_elses_clip")
        with pytest.raises(ValidationError, match="video"):
            stage.validate(context)

    def test_the_run_end_to_end_validates_with_a_readable_summary(self, context):
        stage = self.seeded(context)
        summary = stage.validate(context)
        assert summary["rows"] == 4
        assert summary["basis_states"] == {"basis_ok": 4}
        assert summary["value_states"] == {"normalized": 4}
        assert summary["frame"] == {"origin_keypoint_name": "MidHip",
                                    "basis_keypoint_name": "Neck",
                                    "second_axis": "perpendicular"}


class TestAStaleFrameIsNotLeftBehind:
    """A table written for a different basis outlives the config that made it.

    ``validate`` reports the mismatch and the reuse test refuses it, but neither fixes the
    file — and a run that *skips* never reaches either. The result is a table of coordinates
    in a frame the configuration does not ask for, sitting in the dataset indefinitely and
    indistinguishable to a reader from a current one. Deleting is safe here and only here:
    the table is recomputable from ``pose/body.parquet`` in seconds, so this is not the
    TalkNet-shaped risk of destroying work the pipeline cannot regenerate.
    """

    def foreign_table(self, context) -> PoseNormalizedStage:
        stage = PoseNormalizedStage()
        seed_body(context, a_person(extra={0: (90.0, 60.0)}))
        assert stage.run(context).status == "completed"
        path = context.artifact("pose_normalized")
        table = pq.read_table(path)
        index = table.schema.get_field_index("origin_keypoint_name")
        rows = table.num_rows
        pq.write_table(table.set_column(
            index, "origin_keypoint_name",
            pa.array(["RHip"] * rows, type=pa.string())), path)
        return stage

    def test_a_skipped_run_still_removes_a_table_in_another_basis(self, context):
        self.foreign_table(context)
        context.artifact("pose_body").unlink()
        messages = log_recorder(context)
        outcome = PoseNormalizedStage().run(context)
        assert outcome.status == "skipped"
        assert not context.artifact("pose_normalized").exists()
        assert any("removed stale normalized.parquet" in message for message in messages)
        assert "RHip" in "".join(messages), "the log must name the frame it deleted"

    def test_a_skip_that_proves_nothing_deletes_nothing(self, context):
        """An absent input is not evidence that the table on disk is wrong.

        The table is in the frame the config asks for, so nothing here says it was written
        from anything else. Deleting it would turn a half-built dataset into lost work for
        no gain — the same empty-producible-set trap `speaker_fusion` records.
        """
        stage = PoseNormalizedStage()
        seed_body(context, a_person(extra={0: (90.0, 60.0)}))
        assert stage.run(context).status == "completed"
        before = context.artifact("pose_normalized").read_bytes()
        context.artifact("pose_body").unlink()
        messages = log_recorder(context)
        assert PoseNormalizedStage().run(context).status == "skipped"
        assert context.artifact("pose_normalized").read_bytes() == before
        assert not any("removed stale" in message for message in messages)
        del stage

    def test_a_disabled_stage_prunes_nothing(self, context):
        """`enabled: false` is the operator pausing this stage over a table they kept."""
        self.foreign_table(context)
        context.artifact("pose_body").unlink()
        context.config.pose_normalized.enabled = False
        messages = log_recorder(context)
        assert PoseNormalizedStage().run(context).status == "skipped"
        assert context.artifact("pose_normalized").is_file()
        assert not any("removed stale" in message for message in messages)

    def test_reuse_refuses_a_table_in_another_basis(self, context):
        self.foreign_table(context)
        assert PoseNormalizedStage().outputs_present(context) is False

    def test_pruning_never_reaches_another_stages_artifact(self, context):
        """Only this stage's declared outputs are candidates."""
        self.foreign_table(context)
        body = context.artifact("pose_body")
        body_bytes = body.read_bytes()
        body.unlink()
        PoseNormalizedStage().run(context)
        assert not context.artifact("pose_normalized").exists()
        # The input the operator deleted stays deleted; nothing recreates or touches it.
        assert not body.exists()
        assert body_bytes != b""


class TestItIsPurePython:
    def test_no_subprocess_no_uv_environment_of_its_own(self, context):
        """Two parquet in, one parquet out.

        Asserted structurally, because every other pose-adjacent stage reaches a GPU
        through `WorkerStage` and a reader of this class should be able to see that it
        does not.
        """
        from multimodal_pipeline.stages.base import Stage, WorkerStage

        assert issubclass(PoseNormalizedStage, Stage)
        assert not issubclass(PoseNormalizedStage, WorkerStage)
        assert not hasattr(PoseNormalizedStage, "raw_artifact")
        assert not hasattr(PoseNormalizedStage, "uv_project")

    def test_the_math_module_imports_nothing_heavy(self):
        """`pose_normalize.py` is functions and floats: no Stage class, no torch, no pyarrow.

        The split exists so the algebra can be driven by the reference comparison without a
        pipeline, and so nothing in the orchestrator's import path grows a native dependency.
        """
        import ast
        import inspect

        import multimodal_pipeline.pose_normalize as module

        tree = ast.parse(inspect.getsource(module))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        # Anything outside the standard library would be a new dependency for a stage whose
        # whole job is six multiplications.
        assert imported <= {"__future__", "dataclasses", "math", "typing"}, \
            f"the pure module imports {sorted(imported)}"
        for forbidden in ("torch", "pyannote", "spacy", "parselmouth", "pyarrow", "subprocess"):
            assert forbidden not in imported, f"the pure module imports {forbidden}"
        assert "class Stage" not in inspect.getsource(module)

    def test_declares_its_contract(self):
        stage = PoseNormalizedStage()
        assert stage.name == "pose_normalized"
        assert stage.inputs == ("pose_body",)
        assert stage.outputs == ("pose_normalized",)
        assert stage.config_keys == ("pose_normalized",)


class TestPythonSourceDigest:
    """The helper itself, against files on disk.

    Lives beside the stage that motivated it, because its only job is to answer one question
    about a stage: would an edit to the python that computes my rows be noticed? A test that
    imported two canned modules from the repository could not answer that, because nothing in
    it would ever be edited.
    """

    @staticmethod
    def load(tmp_path, name: str, source: str):
        """Write ``name.py`` into tmp_path and import it, so the digest has a real file behind it.

        ``sys.modules`` is cleaned up by the caller's fixture only if we leave the entry behind,
        so each test uses a name no other test will reuse rather than deleting global state
        another test may be relying on.
        """
        import importlib.util
        import sys

        path = tmp_path / f"{name}.py"
        path.write_text(source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def test_editing_the_source_changes_the_digest(self, tmp_path):
        """The one behaviour the reuse test depends on: new bytes, new digest.

        Re-imported under a second name rather than mutating the first module object, which is
        what a real edit followed by a new process does.
        """
        from multimodal_pipeline.stages.base import python_source_digest

        first = self.load(tmp_path, "pn_digest_before", "ORIGIN = 'MidHip'\n")
        before = python_source_digest(first)
        assert before is not None

        second = self.load(tmp_path, "pn_digest_before", "ORIGIN = 'RHip'\n")
        after = python_source_digest(second)
        assert after != before, (
            "editing the maths left the digest unchanged, so a fix to it would never "
            "invalidate a cached table")

    def test_two_different_modules_give_two_different_digests(self, tmp_path):
        from multimodal_pipeline.stages.base import python_source_digest

        left = self.load(tmp_path, "pn_digest_left", "A = 1\n")
        right = self.load(tmp_path, "pn_digest_right", "A = 1\n")
        assert python_source_digest(left) != python_source_digest(right), (
            "identical bytes under two names collapsed to one digest, so swapping which module "
            "is covered would be invisible")

    def test_the_order_of_the_modules_matters(self, tmp_path):
        """Naming the covered modules in a different order is a different claim."""
        from multimodal_pipeline.stages.base import python_source_digest

        left = self.load(tmp_path, "pn_digest_order_a", "A = 1\n")
        right = self.load(tmp_path, "pn_digest_order_b", "B = 2\n")
        assert python_source_digest(left, right) != python_source_digest(right, left)

    def test_an_unreadable_source_gives_none_instead_of_raising(self, tmp_path):
        """A fingerprint is computed on ``status --plan`` too; it may not be the thing that crashes.

        ``inspect.getsource`` raises ``TypeError`` for a module with no source file and
        ``OSError`` for one whose file has gone away. Both have to arrive at ``None``, which is
        the same contract `worker_code_digest` has for a script that is not on disk yet.
        """
        import math
        import types

        from multimodal_pipeline.stages.base import python_source_digest

        assert python_source_digest(math) is None  # built-in: TypeError from getsource

        ghost = types.ModuleType("pn_digest_ghost")  # no __file__ at all
        assert python_source_digest(ghost) is None

        deleted = self.load(tmp_path, "pn_digest_deleted", "A = 1\n")
        assert python_source_digest(deleted) is not None
        # The file the module was loaded from is gone; the cache `inspect` reads through has to
        # be dropped, exactly as a new process would never have had it.
        import linecache

        linecache.clearcache()
        (tmp_path / "pn_digest_deleted.py").unlink()
        assert python_source_digest(deleted) is None  # OSError from getsource

    def test_a_mixture_of_readable_and_unreadable_gives_none(self, tmp_path):
        """One unreadable module must not be quietly skipped from an otherwise real digest."""
        import math

        from multimodal_pipeline.stages.base import python_source_digest

        readable = self.load(tmp_path, "pn_digest_mixed", "A = 1\n")
        assert python_source_digest(readable, math) is None

    def test_the_repository_modules_it_is_called_on_are_readable(self):
        """Guards the helper's own use: if these ever return None, the coverage is silently gone.

        A module that became a C extension, or a source-less loader, would leave the fingerprint
        carrying ``None`` forever and every test above still green.
        """
        import multimodal_pipeline.fusion as fusion
        import multimodal_pipeline.pose_normalize as pose_normalize
        import multimodal_pipeline.stages.persons as persons
        import multimodal_pipeline.stages.pose_normalized as pose_normalized_stage
        import multimodal_pipeline.stages.speaker_fusion as speaker_fusion_stage
        from multimodal_pipeline.stages.base import python_source_digest

        assert python_source_digest(pose_normalize, pose_normalized_stage) is not None
        assert python_source_digest(fusion, speaker_fusion_stage) is not None
        assert python_source_digest(persons) is not None
