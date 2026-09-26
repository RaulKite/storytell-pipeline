"""The normalised table's shape, its registration, and the config that drives it.

Split from `test_pose_normalized.py` because these are contract tests rather than behaviour
tests: they fail when a column is renamed, reordered or retyped, when the layout stops
declaring the file, or when a knob stops being validated. Nothing here normalises a
keypoint.

Column **order** is asserted, not just membership. These tables are read positionally by
consumers that did not write them (DuckDB exports, notebooks that `select` by index), and
`write_table` reorders by schema anyway — so a reordered schema is a silent change to every
dataset produced afterwards, which is exactly the kind of change that should cost an
intentional edit to this file.
"""

from __future__ import annotations

import math
from pathlib import Path

import pyarrow as pa
import pytest
from multimodal_pipeline.artifacts import ARTIFACT_LAYOUT, MANIFEST_ARTIFACTS
from multimodal_pipeline.config import PipelineConfig, PoseNormalizedConfig
from multimodal_pipeline.pose_normalize import (
    BASIS_STATES,
    SECOND_AXIS_PERPENDICULAR,
    VALUE_STATES,
)
from multimodal_pipeline.schemas import (
    BODY_25_KEYPOINT_NAMES,
    POSE_NORMALIZED_SCHEMA,
    TABLE_SCHEMAS,
)
from multimodal_pipeline.stages.pose_normalized import PoseNormalizedStage

ROOT = Path(__file__).resolve().parents[2]

#: The declared order, as one list so a reorder fails with the two neighbours named.
EXPECTED_COLUMNS: tuple[tuple[str, pa.DataType], ...] = (
    ("schema_version", pa.string()),
    ("video_id", pa.string()),
    ("frame_number", pa.int64()),
    ("timestamp", pa.float64()),
    ("detection_index", pa.int64()),
    ("keypoint_id", pa.int64()),
    ("keypoint_name", pa.string()),
    ("origin_keypoint_name", pa.string()),
    ("basis_keypoint_name", pa.string()),
    ("second_axis", pa.string()),
    ("basis_state", pa.string()),
    ("basis_detail", pa.string()),
    ("x_norm", pa.float64()),
    ("y_norm", pa.float64()),
    ("value_status", pa.string()),
)


class TestSchemaShape:
    def test_columns_are_the_declared_ones_in_the_declared_order(self):
        actual = [(field.name, field.type) for field in POSE_NORMALIZED_SCHEMA]
        assert actual == list(EXPECTED_COLUMNS)

    def test_the_columns_that_exist_because_of_an_honesty_problem_are_commented(self):
        """Each of these has prose above it in schemas.py, not only a name.

        Arrow carries no per-column docstring into the file, so the source comment is the
        only place the reasoning survives for someone who never opens the stage: why the
        frame triple travels on every row, and why two state columns rather than nulls
        alone. Comments are the easiest thing to delete in a refactor, which is why this is
        a test and not a convention.
        """
        source = (ROOT / "src" / "multimodal_pipeline" / "schemas.py").read_text(encoding="utf-8")
        block = source.split("POSE_NORMALIZED_SCHEMA = pa.schema(")[1].split("\n)\n")[0]
        commented: set[str] = set()
        pending: list[str] = []
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("#"):
                pending.append(line)
                continue
            if not line.startswith("("):
                continue
            column = line.split('"')[1] if '"' in line else ""
            if pending:
                commented.add(column)
            pending = []
        for column in ("detection_index", "origin_keypoint_name", "basis_state",
                       "basis_detail", "x_norm", "value_status"):
            assert column in commented, (
                f"{column} is declared with no comment above it in POSE_NORMALIZED_SCHEMA")

    def test_the_state_columns_are_strings_because_they_are_a_vocabulary(self):
        # An enum-ish column only survives as text if it is text on disk: an int code would
        # need a second document nobody keeps in sync.
        for column in ("basis_state", "value_status", "second_axis"):
            assert POSE_NORMALIZED_SCHEMA.field(column).type == pa.string()

    def test_the_coordinates_are_the_only_float_pair_that_can_be_null(self):
        # Nullable by intent: null means "no coordinate in this frame" and `value_status`
        # says which of the two reasons it was. A zero there would be a measurement.
        for column in ("x_norm", "y_norm"):
            field = POSE_NORMALIZED_SCHEMA.field(column)
            assert field.type == pa.float64() and field.nullable

    def test_the_vocabulary_tuples_are_closed_and_disjoint(self):
        assert list(BASIS_STATES) == ["basis_ok", "basis_missing_joint", "basis_degenerate"]
        assert list(VALUE_STATES) == ["normalized", "no_coordinate", "basis_unusable"]
        # A value that belongs to both lists cannot be switched on, and the stage's
        # cross-checks ("coordinates while basis_state != basis_ok") assume they differ.
        assert not set(BASIS_STATES) & set(VALUE_STATES)

    def test_the_origin_and_basis_names_are_real_body_25_keypoints(self):
        from multimodal_pipeline.pose_normalize import DEFAULT_TRANSFORMATION

        origin, basis, second_axis = DEFAULT_TRANSFORMATION
        assert origin in BODY_25_KEYPOINT_NAMES and basis in BODY_25_KEYPOINT_NAMES
        assert second_axis == SECOND_AXIS_PERPENDICULAR
        # The measured basis: MidHip is index 8 and Neck is index 1, which is the triple
        # the reference fixtures were produced with (`c(1, 8, 1, 1)`).
        assert (BODY_25_KEYPOINT_NAMES.index(origin),
                BODY_25_KEYPOINT_NAMES.index(basis)) == (8, 1)


class TestLayoutRegistration:
    def test_the_normalised_table_is_registered_next_to_the_pixel_table(self):
        assert ARTIFACT_LAYOUT["pose_normalized"] == "pose/normalized.parquet"
        # Same directory as pose/body.parquet: the pixel table and the re-expression of it
        # are found together, and a fourth directory would hide it from anyone reading the
        # layout.
        assert Path(ARTIFACT_LAYOUT["pose_body"]).parent == Path("pose")

    def test_it_is_in_the_manifest_set(self):
        assert "pose_normalized" in MANIFEST_ARTIFACTS

    def test_the_stage_log_exists_for_the_new_stage(self):
        from multimodal_pipeline.artifacts import STAGE_LOG_NAMES
        from multimodal_pipeline.stages.base import STAGE_ORDER

        assert "pose_normalized" in STAGE_LOG_NAMES
        assert set(STAGE_LOG_NAMES) == set(STAGE_ORDER)

    def test_no_two_artifacts_share_a_path(self):
        values = list(ARTIFACT_LAYOUT.values())
        assert len(values) == len(set(values))

    def test_the_derived_id_does_not_claim_the_raw_namespace(self):
        # A name ending in `_raw` is a preserved tool output; this is a derived table and
        # putting it there would invite a cleanup to delete it with the native artifacts.
        assert not "pose_normalized_raw" in ARTIFACT_LAYOUT
        assert not ARTIFACT_LAYOUT["pose_normalized"].split("/")[1].startswith("raw")

    def test_the_stage_declares_exactly_the_registered_output(self):
        stage = PoseNormalizedStage()
        assert set(stage.outputs) <= set(ARTIFACT_LAYOUT)
        assert {ARTIFACT_LAYOUT[name] for name in stage.outputs} == {"pose/normalized.parquet"}
        assert stage.inputs == ("pose_body",)

    def test_the_stage_is_registered_in_the_orchestrator_classes(self):
        from multimodal_pipeline.orchestrator import STAGE_CLASSES
        from multimodal_pipeline.stages.base import STAGE_ORDER

        assert set(STAGE_CLASSES) == set(STAGE_ORDER)
        assert STAGE_CLASSES["pose_normalized"] is PoseNormalizedStage

    def test_it_is_ordered_after_openpose_and_before_finalization(self):
        from multimodal_pipeline.stages.base import STAGE_ORDER

        assert STAGE_ORDER.index("openpose") < STAGE_ORDER.index("pose_normalized")
        assert STAGE_ORDER.index("pose_normalized") < STAGE_ORDER.index("finalization")

    def test_it_depends_on_openpose_alone(self):
        """§20.4: openpose output only, so the audio branch can never cost this table."""
        from multimodal_pipeline.stages.base import STAGE_DEPENDENCIES

        assert STAGE_DEPENDENCIES["pose_normalized"] == ("openpose",)
        audio_branch = {"audio", "whisperx", "diarization", "diarization_nemotron",
                        "speaker_assignment", "translation", "spacy_source", "spacy_english",
                        "acoustic", "activespeaker"}
        chain = {name for name in ("pose_normalized",)}
        from multimodal_pipeline.stages.base import dependency_chain

        chain = set(dependency_chain("pose_normalized"))
        assert chain == {"metadata", "openpose"}
        assert not chain & audio_branch
        # finalization must follow it, or the manifest will not describe the new table.
        assert "pose_normalized" in STAGE_DEPENDENCIES["finalization"]

    def test_the_status_table_knows_it_has_an_enable_check(self, context):
        from multimodal_pipeline.orchestrator import enabled_stage_names
        from multimodal_pipeline.stages.base import STAGE_ORDER

        assert "pose_normalized" in STAGE_ORDER
        # Built the way the CLI builds it: a stage with an enable check that is missing
        # from that mapping is reported as enabled while Stage.enabled gives a skip reason.
        assert "pose_normalized" in enabled_stage_names(context.config)
        context.config.pose_normalized.enabled = False
        assert "pose_normalized" not in enabled_stage_names(context.config)

    def test_the_dataset_diagram_in_the_readme_lists_the_new_table(self):
        """README's tree is a claim about the registry; the other ratchet only checks one way.

        `test_readme_claims` proves every Parquet named in the README is registered. This
        proves the reverse direction for this file, so adding the stage cannot leave the
        layout diagram quietly describing a dataset without it.
        """
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "normalized.parquet" in readme


class TestSchemaRegistry:
    def test_the_artifact_maps_to_the_normalised_schema(self):
        assert TABLE_SCHEMAS["pose_normalized"] is POSE_NORMALIZED_SCHEMA

    def test_the_pixel_table_keeps_its_own_schema_untouched(self):
        """The derived table is not a variant of BODY_SCHEMA with extra columns.

        Asserted against the imported object, because a shared schema would mean the pixel
        table gained the frame columns — i.e. the corpus redefined, which is the one thing
        §20.4 forbids.
        """
        from multimodal_pipeline.schemas import BODY_SCHEMA

        assert TABLE_SCHEMAS["pose_body"] is BODY_SCHEMA
        assert [field.name for field in BODY_SCHEMA] == [
            "schema_version", "video_id", "frame_number", "timestamp", "detection_index",
            "keypoint_id", "keypoint_name", "x", "y", "confidence"]
        assert set(BODY_SCHEMA.names) & set(POSE_NORMALIZED_SCHEMA.names) - {
            "schema_version", "video_id", "frame_number", "timestamp", "detection_index",
            "keypoint_id", "keypoint_name"} == set()

    def test_the_coordinate_columns_are_renamed_not_reused(self):
        # x/y in the body table are pixels; x_norm/y_norm are basis coordinates. Two names
        # in one file would let a positional reader pull pixels out of a normalised table.
        assert {"x", "y", "confidence"} & set(POSE_NORMALIZED_SCHEMA.names) == set()
        assert {"x_norm", "y_norm", "value_status"} <= set(POSE_NORMALIZED_SCHEMA.names)


class TestConfigDefaults:
    def test_defaults_are_the_measured_triple(self):
        cfg = PoseNormalizedConfig()
        assert cfg.enabled is True
        assert cfg.origin_keypoint == "MidHip"
        assert cfg.basis_keypoint == "Neck"
        assert cfg.second_axis == SECOND_AXIS_PERPENDICULAR

    def test_the_pipeline_offers_the_section_without_being_told(self):
        config = PipelineConfig.model_construct()
        assert isinstance(config.pose_normalized, PoseNormalizedConfig)

    def test_the_section_is_part_of_the_behaviour_hash(self):
        """A different basis must invalidate the run, not silently reuse a cached table."""
        from multimodal_pipeline.config import stable_hash

        assert "pose_normalized" in PipelineConfig.model_fields
        behaviour = {"pose_normalized": PoseNormalizedConfig().model_dump(mode="json")}
        changed = {"pose_normalized": PoseNormalizedConfig(origin_keypoint="RHip")
                   .model_dump(mode="json")}
        assert stable_hash(behaviour) != stable_hash(changed)

    def test_it_reaches_the_stage_fingerprint(self, context):
        stage = PoseNormalizedStage()
        before = stage.config_fingerprint(context)
        context.config.pose_normalized.origin_keypoint = "RHip"
        assert stage.config_fingerprint(context) != before

    def test_the_fingerprint_names_the_triple_under_a_stable_key(self, context):
        # §20.4 asks for the chosen triple in the request digest. Asserted as one key with
        # the three names in order, so renaming `transformation` costs an edit here.
        from multimodal_pipeline.config import stable_hash

        stage = PoseNormalizedStage()
        payload = stage.config_fingerprint(context)
        assert payload["transformation"] == ["MidHip", "Neck", SECOND_AXIS_PERPENDICULAR]
        # A basis change moves the hash even with the input bytes identical.
        context.config.pose_normalized.basis_keypoint = "LShoulder"
        assert stable_hash(stage.config_fingerprint(context)) != stable_hash(payload)


class TestConfigValidators:
    def test_a_keypoint_that_is_not_in_body_25_is_refused_naming_the_valid_ones(self):
        with pytest.raises(ValueError, match="not a BODY_25 keypoint name"):
            PoseNormalizedConfig(origin_keypoint="Sternum")
        with pytest.raises(ValueError, match="MidHip"):
            PoseNormalizedConfig(basis_keypoint="Sternum")

    def test_a_near_miss_spelling_is_refused_too(self):
        with pytest.raises(ValueError, match="midhip"):
            PoseNormalizedConfig(origin_keypoint="midhip")

    def test_the_origin_and_the_basis_may_not_be_the_same_joint(self):
        # The message has to say *why*: the determinant would be zero, so nothing could be
        # divided by it. A bare "must differ" invites someone to pick a third name.
        with pytest.raises(ValueError, match="must name different keypoints"):
            PoseNormalizedConfig(origin_keypoint="Neck", basis_keypoint="Neck")
        with pytest.raises(ValueError, match="determinant is 0"):
            PoseNormalizedConfig(origin_keypoint="Nose", basis_keypoint="Nose")

    def test_background_is_refused_even_though_it_is_a_body_25_name(self):
        # It is in the name list and never in pose/body.parquet, so a frame defined by it
        # could only ever be basis_missing_joint.
        assert "Background" in BODY_25_KEYPOINT_NAMES
        with pytest.raises(ValueError, match="Background"):
            PoseNormalizedConfig(origin_keypoint="Background")

    def test_the_second_axis_sentinel_is_not_accepted_as_a_keypoint(self):
        """Native review R4-001 (CRITICAL), reproduced before it was fixed.

        One field validator covered `origin_keypoint`, `basis_keypoint` *and* `second_axis`,
        and its `value != SECOND_AXIS_PERPENDICULAR` exemption therefore applied to all
        three. So `origin_keypoint: perpendicular` passed startup validation, and the stage
        died later in `execute()` on `BODY_25_KEYPOINT_NAMES.index("perpendicular")` — a
        config mistake surfacing as a mid-run failure with the output file unwritten.
        A sentinel belongs to one field, not to the type of string.
        """
        from multimodal_pipeline.pose_normalize import SECOND_AXIS_PERPENDICULAR

        assert SECOND_AXIS_PERPENDICULAR == "perpendicular"
        for field in ("origin_keypoint", "basis_keypoint"):
            with pytest.raises(ValueError, match="not a BODY_25 keypoint name"):
                PoseNormalizedConfig(**{field: SECOND_AXIS_PERPENDICULAR})

    def test_the_sentinel_still_validates_as_a_second_axis(self):
        """The fix must not break the field that legitimately uses the sentinel."""
        from multimodal_pipeline.pose_normalize import SECOND_AXIS_PERPENDICULAR

        cfg = PoseNormalizedConfig(second_axis=SECOND_AXIS_PERPENDICULAR)
        assert cfg.second_axis == SECOND_AXIS_PERPENDICULAR

    def test_a_third_joint_for_the_second_axis_is_refused_not_ignored(self):
        # Only the perpendicular branch was validated against the reference. Accepting a
        # joint name and then computing the perpendicular anyway would label every row with
        # a frame that was never built.
        with pytest.raises(ValueError, match="not implemented"):
            PoseNormalizedConfig(second_axis="LShoulder")

    def test_every_body_25_joint_except_background_is_a_valid_origin(self):
        for name in BODY_25_KEYPOINT_NAMES[:-1]:
            if name == "Neck":
                continue  # would collide with the default basis point
            assert PoseNormalizedConfig(origin_keypoint=name).origin_keypoint == name

    def test_a_stray_key_in_the_yaml_names_the_valid_keys_of_this_section(self):
        """The CLI's "unknown setting (known: ...)" helper reaches the new section.

        That help text walks `PipelineConfig.model_fields` one dotted part at a time, so a
        section whose fields are not plain sub-models silently reports "none" and leaves the
        operator guessing. Asserted through the helper rather than a subprocess because the
        helper is the thing that can be wrong here.
        """
        from multimodal_pipeline.cli import _known_settings

        known = _known_settings("pose_normalized.origin")
        for key in ("enabled", "origin_keypoint", "basis_keypoint", "second_axis"):
            assert key in known, f"the error message for a typo lost {key}: {known}"

    def test_an_unknown_key_in_the_section_is_still_refused(self):
        # pydantic's extra="forbid": a typo in the YAML fails the load rather than being
        # silently dropped, which is what makes "I set the basis and nothing changed" a
        # startup error instead of a debugging session.
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            PoseNormalizedConfig(basis="Neck")

    def test_no_nan_or_empty_name_sneaks_through(self):
        # math.nan is a float where a str is declared; pydantic's strict str rejects it,
        # and without that it would reach the fingerprint as a name no reader can decode.
        with pytest.raises(ValueError):
            PoseNormalizedConfig(origin_keypoint=math.nan)
        with pytest.raises(ValueError):
            PoseNormalizedConfig(basis_keypoint="")
