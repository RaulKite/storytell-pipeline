"""Behaviour of the ``persons`` stage: normalisation, validation, gating, fingerprint.

The tests are built around the two claims the stage makes and can be wrong about:

1. **a person count is a fact about the run, not about the model's mood.** So the fingerprint
   has to notice the checkpoint, the tracker and the detector settings, and the raw document
   has to keep ``frames_measured`` distinct from ``frames_with_person`` — otherwise "0 people"
   and "nothing measured" are the same table.
2. **the two tables describe the same run.** ``person_tracks`` is derived from
   ``person_frames``, and ``validate`` re-derives every summary number from the frames rather
   than trusting the summary, which is what makes a truncated or hand-edited table fail
   instead of validate clean.

Nothing here mocks the detector: the stage's job starts where the worker's output ends, so the
input is a structurally real raw document and the assertions are on the Parquet that comes out.
The worker's own logic and the real-model run are in ``test_persons_worker.py``.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from multimodal_pipeline.exceptions import StageError, ValidationError
from multimodal_pipeline.schemas import read_table
from multimodal_pipeline.stages.base import stamp_raw
from multimodal_pipeline.stages.persons import PersonsStage, person_track_rows

# --------------------------------------------------------------------- fixtures


def make_document(**overrides: Any) -> dict[str, Any]:
    """A structurally real worker document: 4 measured frames, 2 people, one empty frame.

    Mirrors what ``workers/persons_worker.py`` writes. The shape is chosen so every summary
    field has a distinct value and the cross-table checks have something to catch:

    * frame 0 holds one person, frame 1 holds **two at once** (so ``max_persons_in_frame``
      and ``persons_in_frame`` are exercised), frame 2 holds nobody and therefore contributes
      no row, frame 3 holds one again;
    * person 1 is seen three times with a hole between the second and third sighting, so
      ``longest_gap_seconds`` is non-zero and not the whole span;
    * the two people have different confidences and areas.

    The declared counts and the rows are derived from the same four-row story on purpose:
    ``validate`` compares them, and a fixture where they disagreed would fail every
    happy-path test for the wrong reason.
    """
    document = {
        "schema_version": "1.0",
        "video_id": "conversation_001",
        "model": "yolo11n.pt",
        "weights_path": "/weights/yolo11n.pt",
        "weights_sha256": "a" * 64,
        "weights_size_bytes": 5432100,
        "tracker": "bytetrack",
        "device": "cuda",
        "requested_device": "cuda",
        "device_fallback_reason": None,
        "parameters": {"conf": 0.25, "imgsz": 640, "classes": [0]},
        "ultralytics_version": "8.4.163",
        "torch_version": "2.8.0+cu126",
        "frames_measured": 4,
        "frames_with_person": 3,
        "max_persons_in_frame": 2,
        "person_ids": [1, 2],
        "person_frame_counts": {"1": 3, "2": 1},
        "frames_without_timestamp": 0,
        "gmc_failure_count": 0,
        "gmc_failure_reason": None,
        "elapsed_seconds": 0.5,
        "frames": [
            _row(0, 0.0, 1, 10.0, 20.0, 60.0, 220.0, 0.91, in_frame=1),
            _row(1, 0.033333, 1, 12.0, 22.0, 62.0, 222.0, 0.88, in_frame=2),
            _row(1, 0.033333, 2, 300.0, 40.0, 380.0, 200.0, 0.42, in_frame=2),
            _row(3, 0.1, 1, 11.0, 21.0, 61.0, 221.0, 0.93, in_frame=1),
        ],
    }
    document.update(overrides)
    return document


def _row(frame: int, stamp: float | None, person: int, x1: float, y1: float, x2: float,
         y2: float, conf: float, *, in_frame: int = 1) -> dict[str, Any]:
    return {
        "frame_number": frame, "timestamp": stamp, "person_id": person,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "confidence": conf, "track_confidence": conf, "confidence_reason": "tracked",
        "persons_in_frame": in_frame,
    }


def strip_recorded_identity(context) -> None:
    """Rewrite both tables without their file metadata, as a pre-feature dataset has them.

    Used wherever a test needs "the table cannot say what produced it", which is the state
    every refusal path must treat as *no evidence* rather than as a mismatch.

    ``pq.read_table`` hands back a schema that *carries* the metadata, so writing it back
    unchanged would keep the values this helper exists to remove; the schema is rebuilt from
    its fields with no metadata at all.
    """
    for name in ("person_frames", "person_tracks"):
        path = context.artifact(name)
        table = pq.read_table(path)
        bare = pa.schema([field for field in table.schema])
        pq.write_table(table.cast(bare), path)


def document_for(context, **overrides: Any) -> dict[str, Any]:
    """A document whose recorded weights digest matches the file actually configured.

    The fixture's ``"a" * 64`` is a stand-in for a real checkpoint. Tests that configure a real
    ``weights_dir`` need the two to agree, because comparing them is the behaviour under test;
    fabricating one while the other is hashed from disk would make the stage look broken when
    it is reporting the mismatch correctly.
    """
    document = make_document(**overrides)
    stage = PersonsStage()
    path = stage.weights_path(context)
    if path is not None and path.is_file():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        document["weights_sha256"] = digest
        document["weights_path"] = str(path)
    return document


def seed_raw(context, stage: PersonsStage, document: dict[str, Any]) -> Any:
    """Write a raw document plus the sidecar that makes it this configuration's output.

    ``stamp_raw`` is the pipeline's own stamping function, not a stand-in: the reuse test the
    stage depends on compares the sidecar's ``request_hash`` with ``request_digest()``, so a
    hand-written sidecar would test the test.
    """
    raw = context.artifact("persons_raw")
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(document), encoding="utf-8")
    stamp_raw(raw, request=stage.request(context), digest=stage.request_digest(context),
              worker=None)
    return raw


def restamp(context, stage: PersonsStage) -> None:
    """Re-stamp after the caller mutated config or document."""
    raw = context.artifact("persons_raw")
    stamp_raw(raw, request=stage.request(context), digest=stage.request_digest(context),
              worker=None)


@pytest.fixture
def stage() -> PersonsStage:
    return PersonsStage()


@pytest.fixture
def seeded(context, stage):
    """A context with a valid raw document, its sidecar, and metadata for the duration bound."""
    seed_raw(context, stage, make_document())
    write_metadata(context)
    return context


def write_metadata(context) -> None:
    """metadata.json with a duration, which bounds every interval check in validate()."""
    path = context.artifact("metadata")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"duration_seconds": 4.2}), encoding="utf-8")


REPO_ROOT = Path(__file__).resolve().parents[2]


def make_installed(context, tmp_path, *, weights: bytes | None = None):
    """Point the stage at an environment and worker that exist, so ``enabled()`` can say yes.

    The shared ``project_root`` fixture is a temp directory with no ``workers/`` and no
    ``environments/``, which is right for every other stage's tests and wrong for the three
    gates this stage adds. The real repository worker is used rather than a stub, so the gate
    is checked against the file the pipeline will actually invoke.
    """
    (tmp_path / "env").mkdir(exist_ok=True)
    context.config.persons.enabled = True
    context.config.persons.uv_project = tmp_path / "env"
    context.config.persons.worker = REPO_ROOT / "workers" / "persons_worker.py"
    if weights is not None:
        (tmp_path / "weights").mkdir(exist_ok=True)
        (tmp_path / "weights" / context.config.persons.model).write_bytes(weights)
        context.config.persons.weights_dir = tmp_path / "weights"
    return context


class TestGating:
    """Every unusable state skips with the fix in the message, and never emits a table."""

    def test_disabled_by_configuration(self, context, stage):
        context.config.persons.enabled = False
        enabled, reason = stage.enabled(context)
        assert enabled is False
        assert reason == "persons.enabled = false"

    def test_a_missing_environment_names_the_directory_and_the_remedy(self, context, stage,
                                                                       tmp_path):
        context.config.persons.enabled = True
        context.config.persons.uv_project = tmp_path / "absent-env"
        enabled, reason = stage.enabled(context)
        assert enabled is False
        assert "absent-env" in reason
        assert "uv sync" in reason, "the skip reason must carry the remedy"

    def test_a_missing_worker_script_says_so(self, context, stage, tmp_path):
        (tmp_path / "env").mkdir()
        context.config.persons.enabled = True
        context.config.persons.uv_project = tmp_path / "env"
        context.config.persons.worker = tmp_path / "gone.py"
        enabled, reason = stage.enabled(context)
        assert enabled is False
        assert "worker script missing" in reason

    def test_an_explicit_weights_dir_without_the_checkpoint_skips_with_both_fixes(
            self, context, stage, tmp_path):
        """Two remedies in one message because there genuinely are two, and they differ.

        Either put the file there or stop promising it is there. Naming only the first would
        send someone to download 5 MB when the right answer was to unset one key.
        """
        make_installed(context, tmp_path)  # env exists, weights_dir does not
        context.config.persons.weights_dir = tmp_path / "weights"
        enabled, reason = stage.enabled(context)
        assert enabled is False
        assert "weights_dir is set" in reason and "yolo11n.pt" in reason
        assert "unset persons.weights_dir" in reason

    def test_a_weights_dir_holding_the_checkpoint_enables_the_stage(self, context, stage,
                                                                    tmp_path):
        make_installed(context, tmp_path, weights=b"not-a-checkpoint")
        assert stage.enabled(context) == (True, "")

    def test_an_unset_weights_dir_does_not_gate_anything(self, context, stage, tmp_path):
        """The download-on-first-use path: nothing on disk to check, so nothing to skip on."""
        make_installed(context, tmp_path)
        context.config.persons.weights_dir = None
        assert stage.enabled(context) == (True, "")

    def test_the_skip_writes_no_table(self, context, stage):
        """§20.2's requirement: skip with a reason rather than emit an empty table.

        Checked on disk, not just on the returned status: the defect this guards is a stage
        that "succeeds" by publishing an empty frames table that validates clean and reads as
        "nobody appeared in this video".
        """
        context.config.persons.enabled = False
        outcome = stage.run(context)
        assert outcome.status == "skipped"
        for name in ("person_frames", "person_tracks"):
            assert not context.artifact(name).exists(), name


class TestNormalize:
    def test_the_frames_table_has_one_row_per_detection(self, seeded, stage):
        summary = stage.normalize(seeded)
        table = read_table(seeded.artifact("person_frames"))
        assert table.num_rows == 4
        assert summary["persons"] == 2
        assert summary["frames_measured"] == 4
        assert summary["frames_with_person"] == 3

    def test_bbox_area_is_computed_from_the_rows_own_box(self, seeded, stage):
        stage.normalize(seeded)
        rows = read_table(seeded.artifact("person_frames")).to_pylist()
        first = rows[0]
        assert first["bbox_area"] == pytest.approx((60.0 - 10.0) * (220.0 - 20.0))

    def test_persons_in_frame_is_carried_through(self, seeded, stage):
        stage.normalize(seeded)
        rows = read_table(seeded.artifact("person_frames")).to_pylist()
        # frame 0 has one person, frame 1 has two (so both of its rows say 2), frame 3 one.
        assert [row["persons_in_frame"] for row in rows] == [1, 2, 2, 1]

    def test_the_metadata_names_the_id_namespace_and_the_timeline(self, seeded, stage):
        stage.normalize(seeded)
        metadata = pq.ParquetFile(seeded.artifact("person_frames")).schema_arrow.metadata
        assert b"weights_sha256" in metadata
        assert metadata[b"weights_sha256"].decode() == "a" * 64
        assert metadata[b"timestamp_column"].decode().startswith("timestamp")

    def test_both_tables_record_which_coco_classes_were_counted(self, seeded, stage):
        """What a person column actually contains, stated in the file itself.

        `person_classes_only: false` allows a non-person COCO class into a column named
        ``person_id``; the guard makes that an explicit choice, and this makes it visible to a
        reader who never opens the raw JSON or the manifest. Checked on both tables because a
        consumer may query either one alone.
        """
        stage.normalize(seeded)
        for name in ("person_frames", "person_tracks"):
            metadata = pq.ParquetFile(seeded.artifact(name)).schema_arrow.metadata
            assert metadata[b"coco_classes"].decode() == "[0]", name

    def test_a_document_with_zero_people_still_writes_both_tables(self, seeded, stage):
        """The honest empty case, measured on ``pipeline_demo``: 0 ids, and a table that says so.

        Distinct from the skip path above: here the stage *ran* and measured frames. Both
        tables exist with zero rows and ``frames_measured`` says how many frames were looked
        at, so a reader can tell "nobody was detected" from "nothing was run".
        """
        document = make_document(frames=[], frames_with_person=0, person_ids=[],
                                 max_persons_in_frame=0, person_frame_counts={})
        seed_raw(seeded, stage, document)
        summary = stage.normalize(seeded)
        assert read_table(seeded.artifact("person_frames")).num_rows == 0
        assert read_table(seeded.artifact("person_tracks")).num_rows == 0
        assert summary["persons"] == 0
        assert summary["frames_measured"] == 4

    def test_a_raw_row_without_confidence_reason_still_normalises(self, seeded, stage):
        """A pre-1.0 artifact carries no reason key; deriving it beats failing the resume."""
        document = make_document()
        for row in document["frames"]:
            del row["confidence_reason"]
        seed_raw(seeded, stage, document)
        stage.normalize(seeded)
        rows = read_table(seeded.artifact("person_frames")).to_pylist()
        assert {row["confidence_reason"] for row in rows} == {"tracked"}

    def test_a_present_but_unrecognised_reason_is_passed_through_not_repaired(
            self, seeded, stage):
        """Rewriting a bad value behind a legal default would hide a malformed worker row.

        ``validate`` is what has to name the frame that carries it.
        """
        document = make_document()
        document["frames"][0]["confidence_reason"] = "guessed"
        seed_raw(seeded, stage, document)
        stage.normalize(seeded)
        with pytest.raises(ValidationError, match="unrecognised confidence_reason"):
            stage.validate(seeded)

    def test_the_worker_gmc_warning_is_logged_as_a_warning(self, seeded, stage, caplog):
        """The 489-warning failure mode has to reach the operator, not a log file.

        Asserted on the stage's own log callable rather than on `caplog`, because
        ``StageContext.log`` is the seam the orchestrator routes to both the stage log and the
        console; a test that only watched the root logger would pass if the stage printed.
        """
        seen: list[tuple[str, int]] = []
        seeded.log = lambda message, level=20: seen.append((str(message), level))
        document = make_document(gmc_failure_count=489,
                                 gmc_failure_reason="GMC failed, falling back to identity: x")
        seed_raw(seeded, stage, document)
        stage.normalize(seeded)
        warnings = [m for m, level in seen if level >= 30 and "camera-motion" in m]
        assert warnings, seen
        assert "489" in warnings[0]
        assert "fragmented" in warnings[0]

    def test_a_clean_run_logs_no_gmc_warning(self, seeded, stage):
        seen: list[tuple[str, int]] = []
        seeded.log = lambda message, level=20: seen.append((str(message), level))
        stage.normalize(seeded)
        assert not [m for m, level in seen if level >= 30 and "camera-motion" in m]

    def test_a_device_fallback_is_logged(self, seeded, stage):
        seen: list[tuple[str, int]] = []
        seeded.log = lambda message, level=20: seen.append((str(message), level))
        document = make_document(device="cpu",
                                 device_fallback_reason="torch.cuda.is_available() was False")
        seed_raw(seeded, stage, document)
        stage.normalize(seeded)
        assert [m for m, level in seen if level >= 30 and "did not use the requested" in m]


class TestTrackSummary:
    def test_one_row_per_person_with_the_measured_span(self, seeded, stage):
        stage.normalize(seeded)
        rows = {row["person_id"]: row
                for row in read_table(seeded.artifact("person_tracks")).to_pylist()}
        assert sorted(rows) == [1, 2]
        assert rows[1]["frame_count"] == 3
        assert rows[1]["first_timestamp"] == pytest.approx(0.0)
        assert rows[1]["last_timestamp"] == pytest.approx(0.1)
        assert rows[1]["duration_seconds"] == pytest.approx(0.1)
        assert rows[2]["frame_count"] == 1

    def test_coverage_uses_frames_measured_and_not_the_row_count(self, seeded, stage):
        """The denominator bug this column exists to make impossible.

        Dividing by the frames-table length would report person 2 (1 of 4 measured frames) as
        covering 100% of the video, because it is one of four rows.
        """
        stage.normalize(seeded)
        rows = {row["person_id"]: row
                for row in read_table(seeded.artifact("person_tracks")).to_pylist()}
        assert rows[1]["frame_coverage"] == pytest.approx(0.75)
        assert rows[2]["frame_coverage"] == pytest.approx(0.25)

    def test_appearance_order_ranks_by_first_sighting(self, seeded, stage):
        stage.normalize(seeded)
        rows = {row["person_id"]: row
                for row in read_table(seeded.artifact("person_tracks")).to_pylist()}
        assert rows[1]["appearance_order"] == 0
        assert rows[2]["appearance_order"] == 1

    def test_a_single_sighting_reports_a_zero_gap_not_null(self):
        """A person seen once has no gap; null would make MAX() silently skip them."""
        rows = person_track_rows("v", [
            {"person_id": 5, "frame_number": 10, "timestamp": 1.0, "confidence": 0.5,
             "bbox_area": 100.0},
        ], frames_measured=100)
        assert rows[0]["longest_gap_seconds"] == 0.0
        assert rows[0]["frame_coverage"] == pytest.approx(0.01)

    def test_the_gap_is_the_biggest_hole_between_consecutive_sightings(self):
        frames = [{"person_id": 1, "frame_number": n, "timestamp": t, "confidence": 0.5,
                   "bbox_area": 10.0}
                  for n, t in [(0, 0.0), (1, 0.1), (10, 1.0), (11, 1.1)]]
        rows = person_track_rows("v", frames, frames_measured=12)
        assert rows[0]["longest_gap_seconds"] == pytest.approx(0.9)

    def test_out_of_order_input_is_sorted_before_the_gap_is_measured(self):
        """Otherwise a shuffled raw document yields a negative or absurd gap."""
        frames = [{"person_id": 1, "frame_number": n, "timestamp": t, "confidence": 0.5,
                   "bbox_area": 10.0}
                  for n, t in [(11, 1.1), (0, 0.0), (10, 1.0), (1, 0.1)]]
        rows = person_track_rows("v", frames, frames_measured=12)
        assert rows[0]["longest_gap_seconds"] == pytest.approx(0.9)
        assert rows[0]["first_timestamp"] == pytest.approx(0.0)

    def test_coverage_is_null_when_nothing_was_measured(self):
        """Zero measured frames means the share is undefined, not 0.0.

        A 0.0 there would claim a measurement of "not on screen at all" for a run that looked
        at nothing.
        """
        rows = person_track_rows("v", [
            {"person_id": 1, "frame_number": 0, "timestamp": 0.0, "confidence": 0.5,
             "bbox_area": 1.0},
        ], frames_measured=0)
        assert rows[0]["frame_coverage"] is None


class TestValidate:
    def test_a_seeded_run_validates(self, seeded, stage):
        stage.normalize(seeded)
        summary = stage.validate(seeded)
        assert summary["persons"] == 2
        assert summary["frame_rows"] == 4
        assert summary["max_persons_in_frame"] == 2

    def test_a_raw_document_from_a_different_configuration_is_rejected(self, seeded, stage):
        """The cached raw output belongs to the settings that produced it.

        The fingerprint is what the reuse test compares against the sidecar, so a changed
        detector setting has to make the preserved artifact unreadable-as-valid; otherwise
        turning conf from 0.25 to 0.4 would keep reporting the counts the old threshold found.
        """
        stage.normalize(seeded)
        seeded.config.persons.conf = 0.4
        with pytest.raises(ValidationError, match="different configuration"):
            stage.validate(seeded)

    @pytest.mark.parametrize("column", ["person_id", "bbox_area", "persons_in_frame",
                                        "confidence_reason"])
    def test_a_missing_column_is_named_rather_than_raising_keyerror(self, seeded, stage,
                                                                    column: str):
        """A table from an older build must say which column is stale, not crash.

        A KeyError here is recorded by the orchestrator as a stage crash; the point of the
        message is that it names the stage to rerun.
        """
        stage.normalize(seeded)
        path = seeded.artifact("person_frames")
        table = pq.read_table(path).drop([column])
        pq.write_table(table, path)
        with pytest.raises(ValidationError, match=f"missing columns: .*{column}"):
            stage.validate(seeded)

    def test_a_missing_raw_key_is_reported_as_staleness(self, seeded, stage):
        """No ``frames_measured`` means 0 people and no measurement are the same table."""
        stage.normalize(seeded)
        document = make_document()
        del document["frames_measured"]
        seed_raw(seeded, stage, document)
        with pytest.raises(ValidationError, match="missing frames_measured"):
            stage.validate(seeded)

    def test_an_inverted_bbox_is_named_with_its_coordinates(self, seeded, stage):
        document = make_document()
        document["frames"][1]["x2"] = document["frames"][1]["x1"] - 1
        seed_raw(seeded, stage, document)
        stage.normalize(seeded)
        with pytest.raises(ValidationError, match="inverted bbox"):
            stage.validate(seeded)

    def test_a_bbox_area_that_disagrees_with_its_own_box_fails(self, seeded, stage):
        """The column claims to be this row's geometry, so it has to be derived from it.

        Mutated in the Parquet rather than the raw document, because that is where a hand
        edit or a partial rewrite would leave the inconsistency.
        """
        self._rewrite_frames(seeded, lambda rows: rows[0].__setitem__("bbox_area", 12345.0))
        with pytest.raises(ValidationError, match="does not match its own bbox"):
            stage.validate(seeded)

    def test_a_row_claiming_confidence_it_does_not_carry_fails(self, seeded, stage):
        self._rewrite_frames(
            seeded, lambda rows: rows[0].update({"track_confidence": None}))
        with pytest.raises(ValidationError,
                           match="claims tracked confidence while carrying none"):
            stage.validate(seeded)

    def test_a_persons_in_frame_of_zero_fails(self, seeded, stage):
        """Self-contradictory: the row *is* a person in that frame."""
        self._rewrite_frames(seeded, lambda rows: rows[0].update({"persons_in_frame": 0}))
        with pytest.raises(ValidationError, match="persons_in_frame is 0"):
            stage.validate(seeded)

    def test_a_frame_whose_rows_understate_its_own_count_fails(self, seeded, stage):
        """The count a "people on screen at once" query reads, checked against the rows.

        Both rows of the two-person frame claim there was only one person in it. Everything
        else in the table stays consistent -- two rows, two ids, correct areas and confidences
        -- so the row-level check (>= 1) passes and the defect is invisible to every other
        assertion. Found by review (R3-persons-in-frame-unverified): the validator already
        recomputed the per-frame counts and never compared them to what the rows declared.
        """
        self._rewrite_frames(seeded, lambda rows: [row.update({"persons_in_frame": 1})
                                                   for row in rows if row["frame_number"] == 1])
        with pytest.raises(ValidationError, match="frame 1 declares persons_in_frame=1"):
            stage.validate(seeded)

    def test_rows_of_one_frame_that_disagree_with_each_other_fail(self, seeded, stage):
        """Two rows of the same frame cannot disagree about how many people were in it."""
        self._rewrite_frames(seeded, lambda rows: rows[2].update({"persons_in_frame": 5}))
        with pytest.raises(ValidationError,
                           match=r"frame 1 declares persons_in_frame=\[2, 5\]"):
            stage.validate(seeded)

    @staticmethod
    def _rewrite_frames(context, mutate) -> None:
        """Normalise, then rewrite the frames table with one row changed.

        Round-trips through the file's own schema so the only difference is the mutated value
        — a test that also reordered or retyped columns would pass or fail for two reasons.
        """
        PersonsStage().normalize(context)
        path = context.artifact("person_frames")
        table = pq.read_table(path)
        rows = table.to_pylist()
        mutate(rows)
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)

    @staticmethod
    def _rewrite_tracks(context, mutate) -> None:
        PersonsStage().normalize(context)
        path = context.artifact("person_tracks")
        table = pq.read_table(path)
        rows = table.to_pylist()
        mutate(rows)
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)

    def test_a_summary_row_that_disagrees_with_the_frames_fails(self, seeded, stage):
        """The point of re-deriving every summary number instead of trusting the table."""
        self._rewrite_tracks(seeded, lambda rows: rows[0].update(
            {"frame_count": rows[0]["frame_count"] + 5}))
        with pytest.raises(ValidationError, match="frame_count"):
            stage.validate(seeded)

    def test_a_duplicate_summary_row_for_one_person_is_refused(self, seeded, stage):
        """Two rows for one id make "how many people" answerable two different ways.

        The count a reader takes from `tracks.parquet` and the count from the frames table
        would disagree, and nothing else in the table says which to believe.
        """
        self._rewrite_tracks(seeded, lambda rows: rows.append(dict(rows[0])))
        with pytest.raises(ValidationError, match="repeats person_id"):
            stage.validate(seeded)

    def test_a_summary_row_for_a_person_with_no_frames_fails(self, seeded, stage):
        import pyarrow as pa

        stage.normalize(seeded)
        path = seeded.artifact("person_tracks")
        table = pq.read_table(path)
        extra = dict(table.to_pylist()[0])
        extra.update({"person_id": 99, "frame_count": 3})
        pq.write_table(pa.Table.from_pylist(table.to_pylist() + [extra], schema=table.schema),
                       path)
        with pytest.raises(ValidationError, match="person_id 99"):
            stage.validate(seeded)

    def test_a_coverage_above_one_is_refused(self, seeded, stage):
        """More frames credited than the run measured: the denominator was wrong."""
        self._rewrite_tracks(seeded, lambda rows: rows[0].update(
            {"frame_count": 99, "frame_coverage": 24.75}))
        with pytest.raises(ValidationError, match="more than all of them"):
            stage.validate(seeded)

    def test_a_summary_not_matching_the_documents_declared_ids_fails(self, seeded, stage):
        document = make_document(person_ids=[1, 2, 7])
        seed_raw(seeded, stage, document)
        stage.normalize(seeded)
        # The frames table still has ids {1,2}; the document now claims 7 appeared too.
        with pytest.raises(ValidationError, match="disagree with the raw document"):
            stage.validate(seeded)

    def test_a_max_persons_in_frame_that_the_frames_contradict_fails(self, seeded, stage):
        document = make_document(max_persons_in_frame=5)
        seed_raw(seeded, stage, document)
        stage.normalize(seeded)
        with pytest.raises(ValidationError, match="max_persons_in_frame"):
            stage.validate(seeded)

    def test_a_frames_count_the_raw_document_denies_fails(self, seeded, stage):
        document = make_document(frames_with_person=9)
        seed_raw(seeded, stage, document)
        stage.normalize(seeded)
        with pytest.raises(ValidationError, match="with a person"):
            stage.validate(seeded)

    def test_rows_from_another_video_are_refused(self, seeded, stage):
        """One video per dataset directory: a second id means the file came from elsewhere."""
        import pyarrow as pa

        stage.normalize(seeded)
        path = seeded.artifact("person_frames")
        table = pq.read_table(path)
        rows = table.to_pylist()
        rows[0]["video_id"] = "some_other_clip"
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
        with pytest.raises(ValidationError, match="some_other_clip"):
            stage.validate(seeded)

    def test_a_table_that_names_a_different_weights_file_than_the_raw_says_fails(
            self, seeded, stage):
        """Cross-check between the Parquet's own metadata and the raw document it claims.

        Catches a table copied in from another run — the file parses, has plausible rows, and
        answers a question about different weights.
        """
        stage.normalize(seeded)
        document = make_document(weights_sha256="b" * 64)
        seed_raw(seeded, stage, document)
        with pytest.raises(ValidationError, match="was not written from this raw output"):
            stage.validate(seeded)

    def test_a_table_with_no_weights_metadata_is_not_failed_for_its_absence(self, context,
                                                                           stage, tmp_path):
        """A pre-existing dataset carries no such metadata; failing it would punish age."""
        make_installed(context, tmp_path, weights=b"first")
        seed_raw(context, stage, document_for(context))
        write_metadata(context)
        stage.normalize(context)
        strip_recorded_identity(context)
        # Now the raw document names a checkpoint the tables make no claim about.
        seed_raw(context, stage, make_document(weights_sha256="b" * 64))
        summary = stage.validate(context)  # no raise
        assert summary["persons"] == 2

    def test_a_timestamp_past_the_source_duration_is_refused(self, seeded, stage):
        document = make_document()
        for row in document["frames"]:
            row["timestamp"] = (row["timestamp"] or 0.0) + 40.0
        seed_raw(seeded, stage, document)
        stage.normalize(seeded)
        with pytest.raises(ValidationError, match="after source duration"):
            stage.validate(seeded)


class TestFingerprint:
    def test_every_setting_that_changes_the_output_moves_the_digest(self, context, stage):
        """Each mutation is a different answer to "how many people are in this video".

        Asserted one at a time so a key dropped from ``request()`` names itself.
        """
        base = stage.request_digest(context)
        mutations = {
            "model": "yolo11s.pt",
            "tracker": "botsort",
            "conf": 0.4,
            "imgsz": 1280,
            "device": "cpu",
            "device_index": 1,
        }
        for key, value in mutations.items():
            clone = context.config.model_copy(deep=True)
            setattr(clone.persons, key, value)
            stage_clone = PersonsStage()
            import copy as _copy

            other = _copy.copy(context)
            other.config = clone
            other.scratch = dict(context.scratch)
            assert stage_clone.request_digest(other) != base, f"{key} is missing from the digest"

    def test_the_class_filter_is_in_the_digest(self, context, stage):
        import copy as _copy

        base = stage.request_digest(context)
        other = _copy.copy(context)
        other.config = context.config.model_copy(deep=True)
        # The opt-in route, because `classes: [2]` and `classes: []` are now refused by default
        # (the column this stage writes is called person_id). Relaxing the guard is still a
        # different measurement of who is on screen and must still invalidate.
        other.config.persons.person_classes_only = False
        other.config.persons.classes = [2]
        other.scratch = dict(context.scratch)
        assert stage.request_digest(other) != base, (
            "dropping the class filter changes what counts as a person and must invalidate")

    def test_the_checkpoint_digest_reaches_the_fingerprint(self, context, stage, tmp_path):
        """Same model *name*, different bytes: two machines must not claim each other's cache."""
        import copy as _copy

        make_installed(context, tmp_path, weights=b"first weights")
        other = _copy.copy(context)
        other.config = context.config.model_copy(deep=True)
        other.scratch = {}
        before = PersonsStage().request_digest(other)

        (tmp_path / "weights" / "yolo11n.pt").write_bytes(b"second weights, same filename")
        after = PersonsStage().request_digest(other)
        assert before != after, (
            "a checkpoint replaced in place left the fingerprint unchanged, so every cached "
            "person table still looks reusable")

    def test_the_worker_source_digest_invalidates_a_worker_change(self, context, stage):
        """`digest_payload` mixes in the worker file's sha256; the stage must not undo that."""
        payload = stage.digest_payload(context)
        assert "_worker_code_sha256" in payload

    def test_the_stage_source_is_in_the_fingerprint(self, context, stage):
        """The python that builds the two Parquet tables is this file, and reuse has to see it.

        `WorkerStage` covers `workers/persons_worker.py` — the detection. Nothing covered the
        normalisation, so a change to the span summary or the gap arithmetic left a completed run
        reusable while the numbers it reported had moved.
        """
        from multimodal_pipeline.stages.metadata import sha256_of

        payload = stage.config_fingerprint(context)
        digest = payload["_python_code_sha256"]
        assert digest is not None
        assert len(digest) == len(sha256_of(__file__))
        assert all(c in "0123456789abcdef" for c in digest)
        # The worker's own digest is still reached through the parent's payload — this stage
        # extends what WorkerStage reports, it does not replace it.
        assert "_worker_code_sha256" in payload

    def test_editing_the_module_that_builds_the_tables_changes_the_fingerprint(self, context, stage,
                                                                               monkeypatch):
        """Same checkpoint, same worker, same raw JSON: new tables, so reuse must refuse.

        Patched at the source-reading seam rather than by rewriting the repository; every other
        key is held equal so the assertion can only pass because of the source digest.
        """
        import multimodal_pipeline.stages.base as base_module

        before = stage.config_fingerprint(context)

        monkeypatch.setattr(base_module, "python_source_digest", lambda *modules: "a" * 64)
        after = stage.config_fingerprint(context)

        assert after["_python_code_sha256"] == "a" * 64
        assert after != before
        assert {k: v for k, v in after.items() if k != "_python_code_sha256"} \
            == {k: v for k, v in before.items() if k != "_python_code_sha256"}, \
            "something besides the source digest moved, so this proves nothing about it"

    def test_the_source_digest_is_stable_between_two_calls(self, context, stage):
        """And it digests this stage's module, which is where the normalisation lives."""
        import multimodal_pipeline.stages.persons as persons_module
        from multimodal_pipeline.stages.base import python_source_digest

        first = stage.config_fingerprint(context)["_python_code_sha256"]
        second = stage.config_fingerprint(context)["_python_code_sha256"]
        assert first == second
        assert first == python_source_digest(persons_module)

    def test_the_stage_source_is_not_in_the_raw_request_digest(self, context, stage):
        """The boundary that decides what a fix here costs.

        `request_digest` is the `request_hash` written into the raw sidecar, and `validate()`
        compares it to decide whether the preserved YOLO output belongs to this configuration.
        Putting the *normaliser's* bytes in there would make any edit to this file — a docstring
        included — invalidate every preserved raw artifact on disk and cost a full detection run
        per video to re-derive tables that only needed re-normalising. The two digests mean
        different things and are asserted apart on purpose.
        """
        payload = stage.digest_payload(context)
        assert "_python_code_sha256" not in payload
        assert "_python_code_sha256" not in stage.request(context)
        assert "_python_code_sha256" in stage.config_fingerprint(context)

    def test_a_source_edit_moves_the_fingerprint_without_invalidating_the_preserved_raw(self, seeded,
                                                                                       monkeypatch):
        """The consequence the boundary above exists for, observed rather than asserted.

        `execute()` decides whether to invoke YOLO by comparing the sidecar's `request_hash`
        with `request_digest()`, and `validate()` refuses a raw file whose hash moved. So if the
        normaliser's source were mixed into that digest, editing this file — a docstring
        included — would send every existing dataset through the detector again. Here the
        fingerprint moves (the tables will be rebuilt) while the preserved raw output stays this
        configuration's, which is the whole point: a fix to the maths costs a re-normalise.
        """
        import multimodal_pipeline.stages.base as base_module
        from multimodal_pipeline.stages.base import raw_request_matches

        stage = PersonsStage()
        raw = seeded.artifact("persons_raw")
        request_hash = stage.request_digest(seeded)
        assert raw_request_matches(raw, request_hash) is True

        monkeypatch.setattr(base_module, "python_source_digest", lambda *modules: "b" * 64)

        assert stage.config_fingerprint(seeded)["_python_code_sha256"] == "b" * 64
        assert stage.request_digest(seeded) == request_hash, (
            "the raw request digest moved because the normaliser's source changed, so the next "
            "run would re-run YOLO to re-derive two parquet files")
        assert raw_request_matches(raw, request_hash) is True


class TestReuseAndWeightsIdentity:
    """§20.2's weights policy, as a reuse rule.

    A person count is only interpretable against the model that produced it, so a checkpoint
    swapped underneath an existing table has to force a rerun instead of leaving a count that
    silently answers a different question.
    """

    def test_outputs_present_is_true_for_a_matching_run(self, context, stage, tmp_path):
        make_installed(context, tmp_path, weights=b"weights bytes")
        seed_raw(context, stage, document_for(context))
        write_metadata(context)
        stage.normalize(context)
        assert stage.outputs_present(context) is True

    def test_a_missing_table_makes_the_stage_not_reusable(self, seeded, stage):
        stage.normalize(seeded)
        seeded.artifact("person_tracks").unlink()
        assert stage.outputs_present(seeded) is False

    def test_a_checkpoint_replaced_in_place_refuses_reuse(self, context, stage, tmp_path):
        make_installed(context, tmp_path, weights=b"first")
        seed_raw(context, stage, document_for(context))
        write_metadata(context)
        stage.normalize(context)
        assert stage.outputs_present(context) is True

        (tmp_path / "weights" / "yolo11n.pt").write_bytes(b"second, different bytes")
        assert PersonsStage().outputs_present(context) is False, (
            "the checkpoint changed underneath the tables and they were still reusable")

    def test_a_model_name_change_refuses_reuse_even_with_nothing_to_hash(self, seeded, stage,
                                                                         tmp_path):
        """With no ``weights_dir`` the stage cannot hash anything before the run.

        So a digest-only check would let ``yolo11n.pt`` -> ``yolo11s.pt`` look reusable. The
        model *name* is compared alongside the digest precisely to cover this case.
        """
        make_installed(context=seeded, tmp_path=tmp_path)
        seeded.config.persons.weights_dir = None
        seed_raw(seeded, stage, make_document())
        write_metadata(seeded)
        stage.normalize(seeded)
        assert PersonsStage().outputs_present(seeded) is True

        seeded.config.persons.model = "yolo11s.pt"
        assert PersonsStage().outputs_present(seeded) is False, (
            "the configured model changed and the existing tables were still reusable")

    def test_a_table_with_no_recorded_identity_is_still_reusable(self, context, stage,
                                                                 tmp_path):
        """Absence of evidence is not evidence of staleness.

        A dataset written before the metadata key existed proves nothing about its weights,
        and treating it as foreign would force a GPU re-run on every old dataset to fix
        nothing.
        """
        make_installed(context, tmp_path, weights=b"first")
        seed_raw(context, stage, document_for(context))
        write_metadata(context)
        stage.normalize(context)
        strip_recorded_identity(context)
        assert PersonsStage().outputs_present(context) is True

    def test_no_weights_dir_means_no_positive_evidence_and_no_refusal(self, seeded, stage,
                                                                      tmp_path):
        """With nothing hashed before the run there is no mismatch to detect, so none claimed."""
        make_installed(context=seeded, tmp_path=tmp_path)
        seeded.config.persons.weights_dir = None
        seed_raw(seeded, stage, make_document())
        write_metadata(seeded)
        stage.normalize(seeded)
        assert PersonsStage().outputs_present(seeded) is True


class TestPrepare:
    def test_a_run_without_the_frame_index_fails_naming_its_purpose(self, context, stage):
        """Frame *indices* are meaningless without the ffprobe packet index.

        The stage could invent timestamps from an fps, which is the error every other visual
        stage avoids on variable-frame-rate input — this corpus's news clips are 30000/1001.
        """
        context.config.persons.enabled = True
        (context.config.project_root / "env").mkdir(exist_ok=True)
        context.config.persons.uv_project = context.config.project_root / "env"
        with pytest.raises(Exception) as raised:
            stage.prepare(context)
        assert type(raised.value).__name__ == "StageError"
        assert "frame_index" in str(raised.value)


# --------------------------------------------------- rendered track preview (stage)


def render_block(**overrides: Any) -> dict[str, Any]:
    """The worker's render block for the fixture document: 4 measured frames.

    ``encoded_width``/``encoded_height`` are the padded size the encoder was actually fed; the
    fixture frames are 64x48, so they equal the source size here.
    """
    block = {"fps": 25.0, "fps_source": "frame_index", "frames_written": 4,
             "width": 64, "height": 48, "encoded_width": 64, "encoded_height": 48,
             "has_audio": False}
    block.update(overrides)
    return block


def fake_ffprobe(tmp_path, *, nb_frames: int = 4, audio: bool = False,
                 name: str = "ffprobe") -> str:
    """An ffprobe that reports the frame count a container would carry.

    The stage's frame-count check has to be exercised with a *mismatched* count, and the only
    deterministic way to make one is to answer ffprobe differently — truncating a real render
    depends on where the truncation lands in the moov atom.
    """
    answer = tmp_path / f"{name}-answer.json"
    streams = [{"codec_type": "video", "nb_frames": str(nb_frames)}]
    if audio:
        streams.append({"codec_type": "audio"})
    answer.write_text(json.dumps({"streams": streams}), encoding="utf-8")
    script = tmp_path / name
    script.write_text(
        f"#!{sys.executable}\nimport sys\nfrom pathlib import Path\n"
        f"sys.stdout.write(Path({json.dumps(str(answer))}).read_text())\n",
        encoding="utf-8")
    script.chmod(0o755)
    return str(script)


class RenderedStage(PersonsStage):
    """The stage with the worker replaced at the seam the pipeline uses to start it.

    ``run_model`` is the pipeline's own subprocess boundary (``uv run --project ... worker.py``),
    so faking it leaves the staging, stamping, normalisation and publication code under test
    identical to the code a GPU run uses. It honours the worker's contract: the argv it was
    given, a raw document, the worker result JSON beside it, and — when ``--output-video`` was
    passed — a video at the staged path. On render failure it writes nothing at all, which is
    what the real worker does when ffmpeg exits non-zero.
    """

    def __init__(self, *, render_ok: bool = True) -> None:
        self.render_ok = render_ok
        self.argvs: list[list[str]] = []

    def run_model(self, ctx, request, raw_path, digest) -> None:
        from multimodal_pipeline.uv_worker import worker_result_path

        argv = self.worker_argv(ctx, raw_path, digest)
        self.argvs.append(argv)
        destination = (Path(argv[argv.index("--output-video") + 1])
                       if "--output-video" in argv else None)
        if destination is not None and not self.render_ok:
            raise StageError("persons worker failed: ffmpeg exited 1 writing the person "
                             "track preview")
        document = make_document()
        if destination is not None:
            document["render"] = render_block()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(document), encoding="utf-8")
        worker_result_path(raw_path.parent,
                           "persons_worker_result.json").write_text(
            json.dumps({"status": "ok"}), encoding="utf-8")
        stamp_raw(raw_path, request=request, digest=digest, worker=None)
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"pretend preview bytes")


def rendering(context, tmp_path, **probe: Any):
    """Switch the render on, point ffprobe at the stub, and supply the two inputs."""
    context.config.persons.write_video = True
    context.config.ffmpeg.ffprobe = fake_ffprobe(tmp_path, **probe)
    context.artifact("frame_index").write_bytes(b"frame index fixture")
    write_metadata(context)
    return context


def published_paths(context) -> list[Path]:
    """Every file the persons stage puts on disk when it renders."""
    raw = context.artifact("persons_raw")
    return [raw, raw.with_name(raw.name + ".provenance.json"),
            raw.parent / "persons_worker_result.json",
            context.artifact("person_frames"), context.artifact("person_tracks"),
            context.artifact("person_track_video")]


class TestRenderFlagsArePipelineOwned:
    def test_the_worker_sees_no_render_flags_when_the_toggle_is_off(self, context):
        context.artifact("frame_index").write_bytes(b"frame index fixture")
        argv = PersonsStage().worker_argv(context, context.artifact("persons_raw"), "digest")
        for flag in ("--output-video", "--ffmpeg", "--ffprobe"):
            assert flag not in argv, f"{flag} was passed although nothing was asked to render"

    def test_the_worker_gets_the_destination_and_the_configured_tools(self, context, tmp_path):
        rendering(context, tmp_path)
        argv = PersonsStage().worker_argv(context, context.artifact("persons_raw"), "digest")
        assert argv[argv.index("--output-video") + 1] == str(
            context.artifact("person_track_video"))
        assert argv[argv.index("--ffmpeg") + 1] == context.config.ffmpeg.executable
        assert argv[argv.index("--ffprobe") + 1] == context.config.ffmpeg.ffprobe

    @pytest.mark.parametrize("flag", ["--output-video", "--ffmpeg", "--ffprobe"])
    @pytest.mark.parametrize("write_video", [True, False])
    def test_extra_args_cannot_override_the_pipeline_owned_flags(self, context, flag,
                                                                write_video):
        """One flag, two directions of damage.

        An ``--output-video`` from ``extra_args`` points the render at a file the stage will
        never publish, so the preview is silently lost; an ``--ffmpeg`` makes the stage validate
        a file some other binary wrote. Both hold whatever the toggle says, so the refusal
        cannot be disabled by the config that is being overridden.
        """
        context.config.persons.write_video = write_video
        context.config.persons.extra_args = [flag, "/tmp/not-yours.mp4"]
        context.artifact("frame_index").write_bytes(b"frame index fixture")
        # The frame index exists so the run reaches the refusal instead of stopping at a
        # missing input, and the pattern keeps the dot in `persons.extra_args`: pytest's
        # temporary directory is named after this test, so a looser pattern is satisfied by an
        # unrelated message that quotes the path. Measured -- the first draft passed that way.
        with pytest.raises(StageError, match=rf"persons\.extra_args.*{re.escape(flag)}"):
            PersonsStage().execute(context)

    @pytest.mark.parametrize("flag,owned", [
        ("--output-v", "--output-video"),
        ("--output-vid", "--output-video"),
        ("--ffm", "--ffmpeg"),
        ("--ffp", "--ffprobe"),
        ("--ff", "--ffprobe"),
    ])
    def test_an_abbreviation_of_an_owned_flag_is_refused_too(self, context, flag, owned):
        """argparse expands any unambiguous prefix, so ``--output-v`` reaches the same flag.

        The worker parser now sets ``allow_abbrev=False``; this is the other side of the same
        boundary, because a guard that only knows whole names is a guard one keystroke wide.
        """
        context.config.persons.extra_args = [flag, "/tmp/not-yours.mp4"]
        with pytest.raises(StageError, match=rf"persons\.extra_args.*{re.escape(owned)}"):
            PersonsStage().prepare(context)

    def test_the_refusal_arrives_before_any_worker_could_start(self, context):
        context.config.persons.write_video = True
        context.config.persons.extra_args = ["--ffprobe", "/bin/false"]
        with pytest.raises(StageError, match="persons.extra_args"):
            PersonsStage().prepare(context)


class TestRenderedExecution:
    def test_flipping_the_toggle_invalidates_a_preserved_raw_run_either_way(self, context,
                                                                           stage):
        """The video can only be made during the tracking pass, so the toggle is a request.

        And turning it *off* invalidates too: a raw document carrying a render block is not what
        a no-render configuration produced, and the reuse test compares requests.
        """
        from multimodal_pipeline.stages.base import raw_request_matches

        rendering(context, context.config.project_root)
        seed_raw(context, stage, make_document())
        raw = context.artifact("persons_raw")
        assert raw_request_matches(raw, stage.request_digest(context)) is True
        context.config.persons.write_video = False
        assert raw_request_matches(raw, stage.request_digest(context)) is False
        context.config.persons.write_video = True
        assert raw_request_matches(raw, stage.request_digest(context)) is True

    def test_a_rendered_run_publishes_the_video_and_everything_it_already_wrote(self, context,
                                                                              tmp_path):
        rendering(context, tmp_path)
        detail = RenderedStage().execute(context)
        assert detail["extra"]["track_video"] == "persons/tracks.mp4"
        for path in published_paths(context):
            assert path.is_file() and path.stat().st_size > 0, path
        assert not list(context.paths.dataset_dir.rglob(".person-tracks-*")), \
            "a staged tree survived the run"
        assert PersonsStage().validate(context)["track_video"] == "persons/tracks.mp4"
        assert PersonsStage().outputs_present(context) is True

    def test_the_render_never_targets_the_canonical_path(self, context, tmp_path):
        """The path the worker is handed has to be inside the staged tree.

        The preservation argument rests on this: ffmpeg truncates its destination before it has
        produced anything, so a worker writing straight to ``persons/tracks.mp4`` destroys the
        previous render even when its own run then fails.
        """
        rendering(context, tmp_path)
        stage = RenderedStage()
        stage.execute(context)
        assert len(stage.argvs) == 1
        destination = Path(stage.argvs[0][stage.argvs[0].index("--output-video") + 1])
        assert destination.name == "tracks.mp4"
        assert destination.parent != context.artifact("person_track_video").parent

    def test_a_failed_render_leaves_the_previous_outputs_byte_identical(self, context, tmp_path):
        rendering(context, tmp_path)
        RenderedStage().execute(context)
        before = {path: path.read_bytes() for path in published_paths(context)}
        assert before[context.artifact("person_track_video")] == b"pretend preview bytes"

        with pytest.raises(StageError, match="ffmpeg exited 1"):
            RenderedStage(render_ok=False).execute(context)
        assert {path: path.read_bytes() for path in published_paths(context)} == before

    def test_a_run_with_the_toggle_off_writes_no_video_and_keeps_the_old_one(self, context,
                                                                           tmp_path):
        """Disabling the feature deletes nothing the operator already had."""
        video = context.artifact("person_track_video")
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"from an earlier render")
        seed_raw(context, PersonsStage(), make_document())
        write_metadata(context)
        context.artifact("frame_index").write_bytes(b"frame index fixture")
        PersonsStage().normalize(context)
        assert video.read_bytes() == b"from an earlier render"
        assert PersonsStage().request(context)["write_video"] is False


class TestRenderedValidation:
    def test_validate_requires_the_preview_frame_count_to_match_exactly(self, context, tmp_path):
        """No three-frame tolerance, because this encoder is ours.

        The skeleton preview allows three trailing frames because upstream's renderer is a black
        box that measurably loses them. Here a short preview means the pipe and the tables
        disagree about which frame is which, and every overlay after the gap sits on the wrong
        picture while the file still plays.
        """
        rendering(context, tmp_path, nb_frames=4)
        RenderedStage().execute(context)
        context.config.ffmpeg.ffprobe = fake_ffprobe(tmp_path, nb_frames=3, name="short-probe")
        with pytest.raises(ValidationError, match="has 3 frame.*expected 4"):
            PersonsStage().validate(context)

    def test_a_raw_document_without_a_render_block_is_reported_when_the_toggle_is_on(self,
                                                                                   context,
                                                                                   tmp_path):
        """The video and the document have to tell the same story.

        A raw document with no ``render`` block came from a run that encoded nothing, so a video
        beside it came from somewhere else. Passing that silently would make the preview
        evidence for a tracking run that never produced one.
        """
        rendering(context, tmp_path)
        stage = PersonsStage()
        seed_raw(context, stage, make_document())
        stage.normalize(context)
        context.artifact("person_track_video").write_bytes(b"video from elsewhere")
        with pytest.raises(ValidationError, match="render block"):
            stage.validate(context)

    def test_an_old_raw_document_still_validates_when_the_toggle_is_off(self, seeded, stage):
        """No ``render`` key is the normal state of every dataset on disk today.

        The default configuration must keep reusing those, or shipping this feature costs a GPU
        pass per video for a field that is only meaningful to the runs that asked for it.
        """
        stage.normalize(seeded)
        assert "render" not in json.loads(seeded.artifact("persons_raw").read_text())
        assert stage.validate(seeded)["persons"] == 2


class TestRenderedReuse:
    def test_outputs_present_requires_the_video_only_when_rendering_is_on(self, context, stage):
        seed_raw(context, stage, make_document())
        write_metadata(context)
        stage.normalize(context)
        video = context.artifact("person_track_video")

        assert stage.outputs_present(context) is True, "off: the stage must not demand a video"
        video.write_bytes(b"frames")
        context.config.persons.write_video = True
        assert stage.outputs_present(context) is True
        video.write_bytes(b"")
        assert stage.outputs_present(context) is False, "an empty preview was reusable"
        video.unlink()
        assert stage.outputs_present(context) is False, "a deleted preview was reusable"
        context.config.persons.write_video = False
        assert stage.outputs_present(context) is True, "off: a missing video must not force a run"
