"""The ``persons`` stage's shape: config, schemas, and everywhere it is registered.

Contract tests rather than behaviour tests — they fail when a column is renamed, reordered or
retyped, when the layout stops declaring a file, or when a knob stops being validated. The
behaviour lives in ``test_persons_stage.py``; the worker's own logic in
``test_persons_worker.py``.

Two things make this file stricter than a routine registration test:

* **column order is asserted, not membership.** These tables are read positionally by
  consumers that did not write them, and ``write_table`` reorders by schema anyway, so a
  reordered schema is a silent change to every dataset produced afterwards.
* **the id-namespace separation is asserted structurally.** §20.2 says a YOLO tracker id and
  TalkNet's ``track_id`` are unrelated and "a consumer that joins on them gets nonsense".
  Prose about that is easy to delete in a refactor; ``TestIDNamespacesCannotBeConfused``
  makes the separation a property of the schemas, the layout and the file metadata at once.
"""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow as pa
import pytest

from multimodal_pipeline.artifacts import ARTIFACT_LAYOUT, MANIFEST_ARTIFACTS
from multimodal_pipeline.config import (
    PERSON_TRACKER_TYPES,
    PersonsConfig,
    PipelineConfig,
    load_config,
)
from multimodal_pipeline.schemas import (
    ACTIVE_SPEAKER_FRAMES_SCHEMA,
    ACTIVE_SPEAKER_TRACKS_SCHEMA,
    PERSON_FRAMES_SCHEMA,
    PERSON_TRACKS_SCHEMA,
    SPEAKER_TURNS_NEMOTRON_SCHEMA,
    SPEAKER_TURNS_SCHEMA,
    TABLE_SCHEMAS,
)
from multimodal_pipeline.stages.persons import CONFIDENCE_REASONS, PersonsStage

ROOT = Path(__file__).resolve().parents[2]

#: The declared order of each table, as one list so a reorder names its neighbours.
FRAMES_COLUMNS: tuple[tuple[str, pa.DataType], ...] = (
    ("schema_version", pa.string()),
    ("video_id", pa.string()),
    ("frame_number", pa.int64()),
    ("timestamp", pa.float64()),
    ("person_id", pa.int64()),
    ("x1", pa.float64()),
    ("y1", pa.float64()),
    ("x2", pa.float64()),
    ("y2", pa.float64()),
    ("confidence", pa.float64()),
    ("track_confidence", pa.float64()),
    ("confidence_reason", pa.string()),
    ("bbox_area", pa.float64()),
    ("persons_in_frame", pa.int64()),
)

TRACKS_COLUMNS: tuple[tuple[str, pa.DataType], ...] = (
    ("schema_version", pa.string()),
    ("video_id", pa.string()),
    ("person_id", pa.int64()),
    ("first_timestamp", pa.float64()),
    ("last_timestamp", pa.float64()),
    ("duration_seconds", pa.float64()),
    ("frame_count", pa.int64()),
    ("frame_coverage", pa.float64()),
    ("longest_gap_seconds", pa.float64()),
    ("mean_confidence", pa.float64()),
    ("max_confidence", pa.float64()),
    ("mean_bbox_area", pa.float64()),
    ("max_bbox_area", pa.float64()),
    ("appearance_order", pa.int64()),
)


class TestConfigDefaults:
    def test_off_by_default_with_the_measured_checkpoint(self):
        cfg = PersonsConfig()
        # Off, like the other optional GPU stages: a fresh clone must not be pushed into a
        # sixth torch environment it did not ask for, and the corpus completes without it.
        assert cfg.enabled is False
        assert cfg.model == "yolo11n.pt"
        assert cfg.tracker == "bytetrack"
        assert cfg.conf == 0.25
        assert cfg.classes == [0]
        assert cfg.imgsz == 640
        assert cfg.device == "auto"
        assert cfg.device_index == 0
        # Unset means "let ultralytics resolve and download", which is a decision, not an
        # omission — see the weights_dir comment in config.example.yaml.
        assert cfg.weights_dir is None
        assert cfg.timeout_seconds is None

    def test_the_environment_and_worker_it_defaults_to_exist_in_the_repository(self):
        """A default that points at a nonexistent path is a stage that always skips.

        ``enabled()`` refuses to run without both, so a typo here is indistinguishable from a
        missing install for every user of the example config.
        """
        cfg = PersonsConfig()
        assert (ROOT / cfg.uv_project / "pyproject.toml").is_file()
        assert (ROOT / cfg.worker).is_file()

    def test_the_pipeline_offers_the_section_without_being_told(self):
        assert isinstance(PipelineConfig.model_construct().persons, PersonsConfig)

    def test_the_section_is_part_of_the_behaviour_hash(self):
        """A knob change must invalidate the run, not silently reuse a cached table."""
        from multimodal_pipeline.config import stable_hash

        assert "persons" in PipelineConfig.model_fields
        before = {"persons": PersonsConfig().model_dump(mode="json")}
        after = {"persons": PersonsConfig(conf=0.4).model_dump(mode="json")}
        assert stable_hash(before) != stable_hash(after)

    def test_it_reaches_the_stage_fingerprint(self, context):
        stage = PersonsStage()
        before = stage.config_fingerprint(context)
        context.config.persons.tracker = "botsort"
        assert stage.config_fingerprint(context) != before


class TestConfigValidators:
    def test_a_device_outside_the_three_is_refused(self):
        with pytest.raises(ValueError, match="persons.device"):
            PersonsConfig(device="tpu")

    @pytest.mark.parametrize("value", [0.0, -0.1, 1.0, 1.5])
    def test_a_confidence_outside_the_open_interval_is_refused(self, value: float):
        # 1.0 is refused, not just values above it: "keep only detections I am certain about"
        # keeps nothing, and writes an empty persons table that validates clean.
        with pytest.raises(ValueError, match="persons.conf"):
            PersonsConfig(conf=value)

    def test_a_usable_confidence_is_accepted(self):
        assert PersonsConfig(conf=0.9).conf == 0.9

    def test_an_unknown_tracker_is_refused_naming_the_builtins(self):
        with pytest.raises(ValueError, match="not one of the built-in trackers"):
            PersonsConfig(tracker="mytracker")

    def test_a_tracker_yaml_path_is_refused_and_says_why(self):
        """The defect the refusal prevents is a fingerprint that records a filename.

        Ultralytics accepts a path here. Passing one through would mean editing the YAML in
        place leaves every existing person table looking reusable while the association
        parameters behind it had changed — a count silently redefined under a stable hash.
        """
        with pytest.raises(ValueError, match="custom tracker YAML is not accepted"):
            PersonsConfig(tracker="/opt/my/trackers/precise.yaml")

    def test_every_advertised_tracker_is_accepted(self):
        """One source of truth: the refusal and the acceptance read the same tuple."""
        for name in PERSON_TRACKER_TYPES:
            assert PersonsConfig(tracker=name).tracker == name

    def test_a_non_coco_class_is_refused(self):
        # Ultralytics raises ValueError on the first frame for id 80; failing at load beats
        # failing per video.
        with pytest.raises(ValueError, match="not a COCO class id"):
            PersonsConfig(classes=[0, 80])

    def test_the_person_class_alone_is_accepted(self):
        assert PersonsConfig(classes=[0]).classes == [0]

    def test_a_non_person_class_is_refused_because_the_column_is_called_person_id(self):
        """Native review R3-non-person-classes, and it was a real hole in the contract.

        `classes: [2]` used to load, the worker dutifully tracked cars, and the result landed in
        a column named ``person_id`` with a summary of "how many people appear". The run was
        traceable (the raw document and the fingerprint both recorded the classes) and the
        artifact was still unreadable as what it was. So the guard is about what a reader can
        tell, not what a log contains.
        """
        with pytest.raises(ValueError, match="person_id"):
            PersonsConfig(classes=[2])

    def test_an_empty_class_list_is_refused_because_it_means_all_eighty_classes(self):
        """The indirect route to the same defect: no filter means every COCO class."""
        with pytest.raises(ValueError, match="person_classes_only"):
            PersonsConfig(classes=[])

    def test_tracking_other_classes_is_still_expressible(self):
        """The refusal is a default, not a policy the config cannot state.

        Deliberately opting in keeps the column name, which is the honest limitation: such a
        dataset measures detected objects, and the table's ``coco_classes`` metadata says which.
        """
        assert PersonsConfig(classes=[2], person_classes_only=False).classes == [2]
        assert PersonsConfig(classes=[], person_classes_only=False).classes == []

    def test_the_empty_classes_message_names_both_settings_it_could_mean(self):
        """A refusal that names only one fix sends the reader to the wrong one."""
        with pytest.raises(ValueError) as excinfo:
            PersonsConfig(classes=[])
        message = str(excinfo.value)
        assert "persons.classes" in message and "person_classes_only" in message

    @pytest.mark.parametrize("value", [0, -64])
    def test_a_non_positive_input_size_is_refused(self, value: int):
        with pytest.raises(ValueError, match="persons.imgsz"):
            PersonsConfig(imgsz=value)

    def test_a_stray_key_in_the_section_is_refused(self):
        with pytest.raises(ValueError, match="iou_threshold"):
            PersonsConfig(iou_threshold=0.3)

    def test_a_typo_in_the_yaml_names_this_sections_keys(self):
        """The CLI's "unknown setting (known: ...)" helper reaches the new section."""
        from multimodal_pipeline.cli import _known_settings

        known = _known_settings("persons.tracker_kind")
        for key in ("enabled", "model", "weights_dir", "tracker", "conf", "classes"):
            assert key in known, f"the error message for a typo lost {key}: {known}"


class TestSchemaShape:
    def test_the_frames_columns_are_the_declared_ones_in_the_declared_order(self):
        assert [(f.name, f.type) for f in PERSON_FRAMES_SCHEMA] == list(FRAMES_COLUMNS)

    def test_the_tracks_columns_are_the_declared_ones_in_the_declared_order(self):
        assert [(f.name, f.type) for f in PERSON_TRACKS_SCHEMA] == list(TRACKS_COLUMNS)

    def test_the_columns_that_exist_because_of_an_honesty_problem_are_commented(self):
        """Each of these has prose above it in schemas.py, not only a name.

        Arrow carries no per-column docstring into the file, so the source comment is the only
        place the reasoning survives for someone who never opens the stage: why ``person_id``
        is not ``track_id``, why ``confidence`` and ``track_confidence`` are two columns, and
        why ``persons_in_frame`` is denormalised on purpose. Comments are the easiest thing to
        delete in a refactor, which is why this is a test and not a convention.
        """
        source = (ROOT / "src" / "multimodal_pipeline" / "schemas.py").read_text(encoding="utf-8")
        block = source.split("PERSON_FRAMES_SCHEMA = pa.schema(")[1].split("\n)\n")[0]
        commented: set[str] = set()
        pending: list[str] = []
        for line in block.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                pending.append(stripped)
                continue
            if not stripped.startswith("("):
                continue
            if pending and '"' in stripped:
                commented.add(stripped.split('"')[1])
            pending = []
        for column in ("person_id", "timestamp", "track_confidence", "confidence_reason",
                       "bbox_area", "persons_in_frame"):
            assert column in commented, (
                f"{column} is declared with no comment above it in PERSON_FRAMES_SCHEMA")

    def test_the_confidence_reason_vocabulary_is_a_closed_set(self):
        assert set(CONFIDENCE_REASONS) == {"tracked", "no_track_confidence", "unknown"}
        # `unknown` must exist: a row whose cause was never routed through the branch that
        # knows has to say "cannot diagnose" rather than borrow one of the two real reasons.
        assert "unknown" in CONFIDENCE_REASONS

    def test_absence_columns_are_nullable_and_the_counts_are_not_optional_in_intent(self):
        # Arrow float/int are nullable already; asserting the *intent* is the point: no
        # absence in this table is encoded as a zero. A 0.0 bbox_area would read as "a person
        # of no size was measured" and a 0 track_confidence as "measured, no confidence".
        for column in ("track_confidence", "bbox_area", "longest_gap_seconds",
                       "mean_confidence", "mean_bbox_area"):
            assert column in {field.name for field in PERSON_FRAMES_SCHEMA} | {
                field.name for field in PERSON_TRACKS_SCHEMA}
        assert PERSON_TRACKS_SCHEMA.field("frame_count").type == pa.int64()

    def test_both_tables_carry_a_schema_version_string(self):
        for schema in (PERSON_FRAMES_SCHEMA, PERSON_TRACKS_SCHEMA):
            assert schema.field("schema_version").type == pa.string()


class TestIDNamespacesCannotBeConfused:
    """§20.2's constraint, made structural instead of documented.

    A YOLO ByteTracker id and TalkNet's S3FD ``track_id`` are both small integers starting near
    zero. That is exactly what makes a join between them dangerous: it runs, returns rows, and
    means nothing. So the separation is enforced at four levels a refactor has to breach
    deliberately — column name, schema object, on-disk directory, and the file metadata every
    written table carries.
    """

    def test_the_person_tables_have_no_track_id_column(self):
        for schema in (PERSON_FRAMES_SCHEMA, PERSON_TRACKS_SCHEMA):
            assert "track_id" not in schema.names, (
                "a `track_id` column on a person table names TalkNet's face-track id space")

    def test_the_asd_tables_have_no_person_id_column(self):
        for schema in (ACTIVE_SPEAKER_FRAMES_SCHEMA, ACTIVE_SPEAKER_TRACKS_SCHEMA):
            assert "person_id" not in schema.names

    def test_no_id_column_is_shared_with_the_face_tables(self):
        # The precise property is about *identifiers*, not column names in general:
        # `first_timestamp` and `mean_bbox_area` are shared measurement names and sharing them
        # is correct. A second shared `*_id` column would mean an id space started being
        # joined by construction, which is what §20.2 forbids.
        person_ids = {name for name in PERSON_TRACKS_SCHEMA.names if name.endswith("_id")}
        face_ids = {name for name in ACTIVE_SPEAKER_TRACKS_SCHEMA.names if name.endswith("_id")}
        assert person_ids == {"video_id", "person_id"}
        assert face_ids == {"video_id", "track_id"}
        assert person_ids & face_ids == {"video_id"}

    def test_no_person_schema_is_the_same_object_as_an_asd_or_speaker_schema(self):
        others = (ACTIVE_SPEAKER_FRAMES_SCHEMA, ACTIVE_SPEAKER_TRACKS_SCHEMA,
                  SPEAKER_TURNS_SCHEMA, SPEAKER_TURNS_NEMOTRON_SCHEMA)
        for person_schema in (PERSON_FRAMES_SCHEMA, PERSON_TRACKS_SCHEMA):
            for other in others:
                assert person_schema is not other

    def test_the_tables_live_in_a_different_directory_from_the_face_tables(self):
        assert Path(ARTIFACT_LAYOUT["person_frames"]).parts[0] == "persons"
        assert Path(ARTIFACT_LAYOUT["person_tracks"]).parts[0] == "persons"
        assert Path(ARTIFACT_LAYOUT["active_speaker_tracks"]).parts[0] == "speaker"
        # A reader browsing the dataset cannot mistake one for the other by opening the wrong
        # folder, which is the level of confusion a directory split actually prevents.
        assert Path(ARTIFACT_LAYOUT["person_tracks"]) != Path(
            ARTIFACT_LAYOUT["active_speaker_tracks"])

    def test_the_non_joinability_is_stated_where_a_reader_will_hit_it(self):
        """The prohibition must exist in the artifacts this change owns.

        A consumer reaches it in three places without opening the stage: the schema source
        above the columns, the shipped example config, and the metadata of every file they
        load. Asserted against all three because a comment deleted in a refactor is the
        failure mode this repository has already paid for elsewhere.

        The README is the fourth place and is *not* asserted here: README.md is outside this
        change's allowed edit surfaces, so a test on it would be a red test nobody in scope
        can turn green. The sentence to add is reported to the parent instead.
        """
        schemas = (ROOT / "src" / "multimodal_pipeline" / "schemas.py").read_text(encoding="utf-8")
        # The paragraph above PERSON_FRAMES_SCHEMA must name both id spaces and contain a
        # sentence that actually forbids joining them. Direction matters: the words "join",
        # "person_id" and "track_id" are all present in a sentence that *recommends* the join
        # too, so the assertion is on a join-sentence carrying a negation.
        header = schemas.split("PERSON_FRAMES_SCHEMA = pa.schema(")[0]
        header = header[header.rindex("# --- persons"):]
        assert "track_id" in header and "person_id" in header
        prose = " ".join(line.lstrip("#").strip() for line in header.splitlines())
        joining = [sentence for sentence in re.split(r"(?<=[.!?])\s+", prose)
                   if "join" in sentence.lower()]
        assert joining, "the schema note no longer discusses joining the two id spaces"
        assert any(negation in sentence.lower() for sentence in joining
                   for negation in ("nothing", "nonsense", " never", "not ")), (
            "the schema note mentions the join without forbidding it: "
            f"{joining}")
        assert "not a face track" in prose.lower().replace("*", ""), (
            "the schema note no longer says a person track is not a face track")

        example = (ROOT / "config" / "config.example.yaml").read_text(encoding="utf-8")
        assert "person_id" in example and "track_id" in example, (
            "config.example.yaml no longer contrasts the two id names")

    def test_a_person_track_is_declared_not_to_be_a_face_track(self):
        """Saying so where a reader hits it: the schema source and every written file."""
        schemas = (ROOT / "src" / "multimodal_pipeline" / "schemas.py").read_text(encoding="utf-8")
        header = schemas.split("PERSON_FRAMES_SCHEMA = pa.schema(")[0]
        header = header[header.rindex("# --- persons"):]
        assert "face" in header.lower()
        stage_source = (ROOT / "src" / "multimodal_pipeline" / "stages" /
                        "persons.py").read_text(encoding="utf-8")
        assert "face" in stage_source.split('"""')[1].lower(), (
            "the stage docstring no longer tells a reader what a person track is not")

    def test_every_written_person_table_carries_the_warning_in_its_metadata(self, context):
        """File metadata, because that is what a consumer sees without reading the README.

        Normalised from a structurally real raw document, so the assertion is about what the
        stage writes rather than about a dict pasted into this test.
        """
        import pyarrow.parquet as pq

        from tests.unit.test_persons_stage import make_document, seed_raw

        stage = PersonsStage()
        seed_raw(context, stage, make_document())
        stage.normalize(context)

        for name in ("person_frames", "person_tracks"):
            metadata = pq.ParquetFile(context.artifact(name)).schema_arrow.metadata or {}
            namespace = metadata.get(b"id_namespace", b"").decode()
            assert "person_id" in namespace, name
            assert "track_id" in namespace, (
                f"{name} metadata does not name the id space it must not be joined to")
            assert "not" in namespace.lower(), name
            assert metadata.get(b"track_kind", b"").decode().startswith("person"), name
        assert pq.ParquetFile(context.artifact("person_frames")).metadata.num_rows == 4


class TestLayoutRegistration:
    def test_the_three_artifacts_are_registered_under_persons(self):
        assert ARTIFACT_LAYOUT["persons_raw"] == "persons/raw/yolo_track.json"
        assert ARTIFACT_LAYOUT["person_frames"] == "persons/frames.parquet"
        assert ARTIFACT_LAYOUT["person_tracks"] == "persons/tracks.parquet"

    def test_they_are_in_the_manifest_set(self):
        assert {"persons_raw", "person_frames", "person_tracks"} <= set(MANIFEST_ARTIFACTS)

    def test_the_raw_namespace_invariant_holds(self):
        """The preserved native artifact lives in a directory that says so.

        Same invariant test_artifacts.py asserts for every `*_raw` key; asserted here for the
        new one because it is the one this change could break.
        """
        parts = Path(ARTIFACT_LAYOUT["persons_raw"]).parts
        assert any(segment.startswith("raw") for segment in parts)
        # And the derived tables are not in it: a cleanup that deletes raw tool output must
        # not take the Parquet built from it.
        for name in ("person_frames", "person_tracks"):
            assert not any(segment.startswith("raw")
                           for segment in Path(ARTIFACT_LAYOUT[name]).parts)

    def test_ensure_dirs_creates_the_persons_tree(self, tmp_path):
        from multimodal_pipeline.artifacts import VideoPaths

        VideoPaths(tmp_path / "dataset").ensure_dirs()
        assert (tmp_path / "dataset" / "persons" / "raw").is_dir()

    def test_no_two_artifacts_share_a_path(self):
        values = list(ARTIFACT_LAYOUT.values())
        assert len(values) == len(set(values))

    def test_the_stage_declares_exactly_the_registered_outputs(self):
        stage = PersonsStage()
        assert set(stage.outputs) <= set(ARTIFACT_LAYOUT)
        assert {ARTIFACT_LAYOUT[name] for name in stage.outputs} == {
            "persons/raw/yolo_track.json", "persons/frames.parquet",
            "persons/tracks.parquet"}

    def test_a_stage_log_is_declared_for_the_new_stage(self):
        from multimodal_pipeline.artifacts import STAGE_LOG_NAMES
        from multimodal_pipeline.stages.base import STAGE_ORDER

        assert "persons" in STAGE_LOG_NAMES
        assert set(STAGE_LOG_NAMES) == set(STAGE_ORDER)


class TestDagRegistration:
    def test_the_stage_is_in_the_order_and_the_class_map(self):
        from multimodal_pipeline.orchestrator import STAGE_CLASSES
        from multimodal_pipeline.stages.base import STAGE_ORDER

        assert set(STAGE_CLASSES) == set(STAGE_ORDER)
        assert STAGE_CLASSES["persons"] is PersonsStage
        assert "persons" in STAGE_ORDER

    def test_it_sits_with_the_other_video_readers_and_before_finalization(self):
        from multimodal_pipeline.stages.base import STAGE_ORDER

        assert STAGE_ORDER.index("metadata") < STAGE_ORDER.index("persons")
        assert STAGE_ORDER.index("persons") < STAGE_ORDER.index("finalization")

    def test_it_depends_on_metadata_alone(self):
        """§20.2's independence, pinned.

        Depending on the audio branch would let a transcription failure cost the person
        counts; depending on `activespeaker` would let a *face* failure cost the *body* table
        that is the evidence surviving it. metadata is also the whole of what it reads, since
        metadata writes frame_index.parquet too.
        """
        from multimodal_pipeline.stages.base import STAGE_DEPENDENCIES

        assert STAGE_DEPENDENCIES["persons"] == ("metadata",)
        stage = PersonsStage()
        assert set(stage.inputs) <= {"metadata", "frame_index"}
        assert "audio" not in stage.inputs

    def test_a_transcription_failure_does_not_block_it(self):
        """The observable consequence of the dependency choice, not just the tuple.

        `dependants_of("whisperx")` is what the orchestrator poisons when transcription
        fails; the person table must not be in it.
        """
        from multimodal_pipeline.stages.base import dependants_of

        assert "persons" not in dependants_of("whisperx")
        assert "persons" not in dependants_of("diarization")
        assert "persons" not in dependants_of("activespeaker")
        assert "persons" in dependants_of("metadata")

    def test_it_is_not_in_the_enabled_stage_name_map_so_the_status_table_keeps_its_shape(self):
        """A default-False stage stays out of ``enabled_stage_names``.

        The map's default is True. Listing a stage that ships disabled makes the function
        return fewer names than ``STAGE_ORDER`` on an untouched config, which silently drops a
        column from every ``status`` table — the same reason ``activespeaker`` and
        ``diarization_nemotron`` are absent. Asserted as the invariant rather than as a
        membership list, so the next default-False stage inherits the rule instead of having
        to rediscover it.
        """
        from multimodal_pipeline.orchestrator import enabled_stage_names
        from multimodal_pipeline.stages.base import STAGE_ORDER

        for flag in ("true", "false"):
            config = config_from_fragment({"persons": {"enabled": flag == "true"}})
            names = enabled_stage_names(config)
            assert len(names) == len(STAGE_ORDER), (
                f"enabled_stage_names returned {len(names)} of {len(STAGE_ORDER)} stages when "
                f"persons.enabled = {flag}: the status header lost a column")
            assert "persons" in names


def config_from_fragment(extra: dict) -> PipelineConfig:
    """A real config from a fragment, through ``load_config`` rather than by hand.

    Building a PipelineConfig in memory would skip the code path the CLI takes, and the tests
    using this are about what the CLI sees.
    """
    import tempfile

    import yaml

    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        (root / "config").mkdir()
        (root / "videos").mkdir()
        payload: dict = {"project_root": str(root),
                         "input": {"directory": str(root / "videos")},
                         "output": {"directory": str(root / "out")}}
        payload.update(extra)
        path = root / "config" / "config.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return load_config(path)


class TestSchemaRegistry:
    def test_each_artifact_maps_to_its_own_schema(self):
        assert TABLE_SCHEMAS["person_frames"] is PERSON_FRAMES_SCHEMA
        assert TABLE_SCHEMAS["person_tracks"] is PERSON_TRACKS_SCHEMA

    def test_the_two_tables_do_not_share_a_schema_object(self):
        # Unlike the two speaker_fusion files, which differ only by a column *value*. Here the
        # grain differs — one row per (frame, person) versus one row per person — so one
        # schema object could not describe both and a shared object would be a bug.
        assert PERSON_FRAMES_SCHEMA is not PERSON_TRACKS_SCHEMA

    def test_the_new_schemas_did_not_displace_any_existing_entry(self):
        # The ASD tables are absent from this registry by pre-existing design (they are written
        # with their schema object passed explicitly); what is asserted here is that adding two
        # entries did not disturb the ones that were registered.
        for name in ("pose_body", "pose_normalized", "speaker_turns", "frame_index"):
            assert name in TABLE_SCHEMAS


class TestEnvironmentInventory:
    def test_a_fresh_clone_without_the_environment_is_warned(self, tmp_path):
        """The pre-flight requirement: a missing ``environments/persons`` is named before a
        long run, not discovered as a skip reason afterwards.

        Goes through the generic ``stage_configs`` walk, which is why the section carries a
        ``uv_project`` at all — asserted here so the mechanism cannot be broken by dropping
        the section from that mapping.
        """
        from multimodal_pipeline.cli import _environment_warnings

        (tmp_path / "videos").mkdir()
        config = PipelineConfig.model_validate({
            "project_root": str(tmp_path),
            "input": {"directory": str(tmp_path / "videos")},
            "output": {"directory": str(tmp_path / "out")},
            "diarization": {"enabled": False}, "translation": {"enabled": False},
            "openpose": {"enabled": False}, "activespeaker": {"enabled": False},
            "spacy": {"enabled": False}, "whisperx": {"enabled": False},
            "acoustic": {"enabled": False}, "persons": {"enabled": True},
        })
        hits = [w for w in _environment_warnings(config) if "persons" in w]
        assert hits, "a missing environments/persons produced no warning"
        assert "uv project missing" in hits[0]

    def test_a_disabled_stage_with_no_environment_is_not_warned(self, tmp_path):
        """A stage the operator switched off is a decision, not a defect."""
        from multimodal_pipeline.cli import _environment_warnings

        (tmp_path / "videos").mkdir()
        config = PipelineConfig.model_validate({
            "project_root": str(tmp_path),
            "input": {"directory": str(tmp_path / "videos")},
            "output": {"directory": str(tmp_path / "out")},
            "diarization": {"enabled": False}, "translation": {"enabled": False},
            "openpose": {"enabled": False}, "activespeaker": {"enabled": False},
            "spacy": {"enabled": False}, "whisperx": {"enabled": False},
            "acoustic": {"enabled": False},
        })
        assert not [w for w in _environment_warnings(config) if "persons" in w]

    def test_provenance_inventories_the_environment_alongside_the_other_five(self):
        """`tools.json` answers "which uv projects exist?" from `stage_configs`."""
        from multimodal_pipeline.provenance import tools_report

        config = config_from_fragment({"persons": {"enabled": False}})
        report = tools_report(config)
        assert "persons" in report["uv_projects"], report["uv_projects"].keys()
        # The *path* is what this fixture can assert: the temp project_root has no
        # environments/ directory, so `exists` is honestly False here and asserting True
        # would test the fixture rather than the report.
        assert report["uv_projects"]["persons"]["project"].endswith(
            "environments/persons"), report["uv_projects"]["persons"]
        assert report["uv_projects"]["persons"]["exists"] is False


class TestExampleConfig:
    def test_the_shipped_example_carries_the_section(self):
        import yaml

        raw = yaml.safe_load((ROOT / "config" / "config.example.yaml").read_text())
        assert "persons" in raw
        assert raw["persons"]["enabled"] is False
        assert raw["persons"]["model"] == "yolo11n.pt"
        assert raw["persons"]["classes"] == [0]

    def test_the_example_validates_against_the_schema(self):
        """A shipped example that does not load is worse than no example."""
        import yaml

        raw = yaml.safe_load((ROOT / "config" / "config.example.yaml").read_text())
        PipelineConfig.model_validate(dict(raw, project_root=str(ROOT)))
