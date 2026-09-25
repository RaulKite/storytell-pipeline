"""The fused table's shape, its registration, and the config that drives it.

Split from `test_speaker_fusion.py` because these are contract tests rather than behaviour
tests: they fail when a column is renamed, reordered or retyped, when the layout stops
declaring the file, or when a knob stops being validated. Nothing here computes an
agreement verdict; they check that the artifact and the config can be trusted to describe
one.

Column **order** is asserted, not just membership. These tables are read positionally by
consumers that did not write them (DuckDB exports, notebooks that `select` by index), and
`write_table` reorders by schema anyway — so a reordered schema is a silent change to every
dataset produced afterwards, which is exactly the kind of change that should cost an
intentional edit to this file.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest
from multimodal_pipeline.artifacts import ARTIFACT_LAYOUT, MANIFEST_ARTIFACTS
from multimodal_pipeline.config import PipelineConfig, SpeakerFusionConfig
from multimodal_pipeline.fusion import AGREEMENT_STATES, FUSION_ENGINES
from multimodal_pipeline.schemas import SPEAKER_FUSION_SCHEMA, TABLE_SCHEMAS
from multimodal_pipeline.stages.speaker_fusion import SpeakerFusionStage

#: The declared order, as one list so a reorder fails with the two neighbours named.
EXPECTED_COLUMNS: tuple[tuple[str, pa.DataType], ...] = (
    ("schema_version", pa.string()),
    ("video_id", pa.string()),
    ("engine", pa.string()),
    ("turn_id", pa.string()),
    ("speaker_id", pa.string()),
    ("start_time", pa.float64()),
    ("end_time", pa.float64()),
    ("duration", pa.float64()),
    ("diarization_type", pa.string()),
    ("overlap_s", pa.float64()),
    ("face_track_id", pa.int64()),
    ("face_active_frames", pa.int64()),
    ("face_frames_in_turn", pa.int64()),
    ("frames_in_turn", pa.int64()),
    ("face_mean_score", pa.float64()),
    ("face_score_max", pa.float64()),
    ("agreement", pa.string()),
    ("agreement_detail", pa.string()),
)


class TestSchemaShape:
    def test_columns_are_the_declared_ones_in_the_declared_order(self):
        actual = [(field.name, field.type) for field in SPEAKER_FUSION_SCHEMA]
        assert actual == list(EXPECTED_COLUMNS)

    def test_the_columns_that_exist_because_of_an_honesty_problem_are_commented(self):
        """Each of these columns has prose above it in schemas.py, not only a name.

        Arrow carries no per-column docstring into the file, so the source comment is the
        only place the reasoning survives for someone who never opens the stage: why
        `engine` is not a join key, why three counts instead of one ratio, and why a null
        `overlap_s` is not a zero. Comments are the easiest thing to delete in a refactor,
        which is why this is a test and not a convention.
        """
        source = (Path(__file__).resolve().parents[2] / "src" / "multimodal_pipeline" /
                  "schemas.py").read_text(encoding="utf-8")
        block = source.split("SPEAKER_FUSION_SCHEMA = pa.schema(")[1].split("\n)\n")[0]
        lines = [line.strip() for line in block.splitlines()]
        commented: set[str] = set()
        pending: list[str] = []
        for line in lines:
            if line.startswith("#"):
                pending.append(line.lstrip("# ").strip())
                continue
            if not line.startswith("("):
                continue
            column = line.split('"')[1] if '"' in line else ""
            if pending:
                commented.add(column)
            pending = []
        for column in ("engine", "overlap_s", "face_track_id", "face_active_frames",
                       "frames_in_turn", "agreement", "agreement_detail"):
            assert column in commented, (
                f"{column} is declared with no comment above it in SPEAKER_FUSION_SCHEMA")

    def test_the_nullable_columns_are_the_ones_that_mean_absence(self):
        # float64/int64 in Arrow are nullable already; the point is that no absence is
        # encoded as a zero, so a reader can tell "not measured" from "measured as none".
        nullable = {field.name for field in SPEAKER_FUSION_SCHEMA if field.nullable}
        assert {"overlap_s", "face_track_id", "face_mean_score", "face_score_max"} <= nullable
        # The three counts are never optional: an absent count would be indistinguishable
        # from a measured zero, which is the confusion the columns exist to remove.
        for column in ("face_active_frames", "face_frames_in_turn", "frames_in_turn"):
            assert column in nullable, "Arrow types are nullable; this asserts the intent"

    def test_schema_version_is_a_string_column_like_every_other_table(self):
        assert SPEAKER_FUSION_SCHEMA.field("schema_version").type == pa.string()

    def test_the_agreement_vocabulary_and_the_engine_list_are_closed_sets(self):
        from multimodal_pipeline import fusion

        assert set(AGREEMENT_STATES) == {
            "face_matched", "face_partial", "no_face_visible",
            "face_never_active", "no_frames_measured",
        }
        # The two absence states must be distinct values, or the table cannot tell
        # "nobody on screen" from "nobody looked".
        assert "no_face_visible" != "no_frames_measured"
        assert list(FUSION_ENGINES) == ["pyannote", "nemotron"]
        assert set(fusion.TURN_TABLES) == set(FUSION_ENGINES)


class TestLayoutRegistration:
    def test_both_fused_tables_are_registered_beside_the_asd_tables(self):
        assert ARTIFACT_LAYOUT["speaker_fusion_pyannote"] == "speaker/fusion_pyannote.parquet"
        assert ARTIFACT_LAYOUT["speaker_fusion_nemotron"] == "speaker/fusion_nemotron.parquet"
        # Same directory as the ASD output: this is a speaker-table consumer, not a new
        # modality, and a fourth directory would hide it from anyone reading the layout.
        top = {Path(relative).parts[0] for relative in ARTIFACT_LAYOUT.values()}
        assert "speaker" in top

    def test_they_are_in_the_manifest_set(self):
        assert {"speaker_fusion_pyannote", "speaker_fusion_nemotron"} <= set(MANIFEST_ARTIFACTS)

    def test_the_stage_log_exists_for_the_new_stage(self):
        from multimodal_pipeline.artifacts import STAGE_LOG_NAMES
        from multimodal_pipeline.stages.base import STAGE_ORDER

        assert "speaker_fusion" in STAGE_LOG_NAMES
        assert set(STAGE_LOG_NAMES) == set(STAGE_ORDER)

    def test_no_two_artifacts_share_a_path(self):
        values = list(ARTIFACT_LAYOUT.values())
        assert len(values) == len(set(values))

    def test_the_raw_namespace_invariant_is_untouched(self):
        """The `*_raw` layout invariant, re-checked rather than assumed.

        The fused ids deliberately do *not* end in `_raw`: they are derived tables, and
        naming them `speaker_fusion_raw` would put them in the raw namespace and invite a
        cleanup to delete derived data with the native artifacts.
        """
        raw = {name for name in ARTIFACT_LAYOUT if name.endswith("_raw")}
        assert raw
        for name in raw:
            parts = Path(ARTIFACT_LAYOUT[name]).parts
            assert any(segment.startswith("raw") for segment in parts), name
        assert not any(name.endswith("_raw")
                       for name in ("speaker_fusion_pyannote", "speaker_fusion_nemotron"))

    def test_the_stage_declares_exactly_the_registered_outputs(self):
        stage = SpeakerFusionStage()
        assert set(stage.outputs) <= set(ARTIFACT_LAYOUT)
        assert {ARTIFACT_LAYOUT[name] for name in stage.outputs} == {
            "speaker/fusion_pyannote.parquet", "speaker/fusion_nemotron.parquet"}

    def test_the_stage_is_registered_in_the_orchestrator_classes(self):
        from multimodal_pipeline.orchestrator import STAGE_CLASSES
        from multimodal_pipeline.stages.base import STAGE_ORDER

        assert set(STAGE_CLASSES) == set(STAGE_ORDER)
        assert STAGE_CLASSES["speaker_fusion"] is SpeakerFusionStage

    def test_the_dataset_diagram_in_the_readme_lists_the_fused_tables(self):
        """README's tree is a claim about the registry; the ratchet only checks one way.

        `test_readme_claims` proves every Parquet named in the README is registered. This
        proves the reverse direction for these two files, so adding the stage cannot leave
        the layout diagram quietly describing a dataset without them.
        """
        readme = (Path(__file__).resolve().parents[2] / "README.md").read_text(encoding="utf-8")
        assert "fusion_pyannote.parquet" in readme
        assert "fusion_nemotron.parquet" in readme


class TestSchemaRegistry:
    def test_the_fused_artifacts_map_to_the_fused_schema(self):
        assert TABLE_SCHEMAS["speaker_fusion_pyannote"] is SPEAKER_FUSION_SCHEMA
        assert TABLE_SCHEMAS["speaker_fusion_nemotron"] is SPEAKER_FUSION_SCHEMA

    def test_the_two_engines_share_one_schema_because_the_columns_do_not_depend_on_engine(self):
        # Same schema object on purpose: two copies would drift, and the engine difference
        # is a value in `engine`, never a column.
        assert (TABLE_SCHEMAS["speaker_fusion_pyannote"]
                is TABLE_SCHEMAS["speaker_fusion_nemotron"])

    def test_the_turn_tables_keep_their_own_separate_schemas(self):
        # The fused table borrows the turn columns, but the source tables must not merge:
        # that is where the namespaces are kept apart.
        from multimodal_pipeline.schemas import (
            SPEAKER_TURNS_NEMOTRON_SCHEMA,
            SPEAKER_TURNS_SCHEMA,
        )

        assert SPEAKER_TURNS_SCHEMA is not SPEAKER_TURNS_NEMOTRON_SCHEMA
        turn_names = [field.name for field in SPEAKER_TURNS_SCHEMA]
        nemotron_names = [field.name for field in SPEAKER_TURNS_NEMOTRON_SCHEMA]
        assert nemotron_names == turn_names + ["overlap_s"]


class TestConfigDefaults:
    def test_defaults_are_enabled_with_pyannote_only(self):
        cfg = SpeakerFusionConfig()
        assert cfg.enabled is True
        assert cfg.engines == ["pyannote"]
        assert cfg.min_active_ratio == 0.5
        assert cfg.min_face_frames == 2

    def test_the_pipeline_offers_the_section_without_being_told(self):
        config = PipelineConfig.model_construct()
        assert isinstance(config.speaker_fusion, SpeakerFusionConfig)

    def test_the_section_is_part_of_the_behaviour_hash(self):
        """A threshold change must invalidate the run, not silently reuse a cached table.

        `configuration_hash` excludes only placement fields, so this asserts the knob is
        inside the payload rather than trusting the exclusion list.
        """
        from multimodal_pipeline.config import stable_hash

        assert "speaker_fusion" in PipelineConfig.model_fields
        behaviour = {"speaker_fusion": SpeakerFusionConfig().model_dump(mode="json")}
        changed = {"speaker_fusion": SpeakerFusionConfig(min_active_ratio=0.9)
                   .model_dump(mode="json")}
        assert stable_hash(behaviour) != stable_hash(changed)

    def test_it_reaches_the_stage_fingerprint(self, context):
        stage = SpeakerFusionStage()
        before = stage.config_fingerprint(context)
        context.config.speaker_fusion.min_face_frames = 7
        assert stage.config_fingerprint(context) != before


class TestConfigValidators:
    def test_an_unknown_engine_is_refused_naming_the_allowed_ones(self):
        with pytest.raises(ValueError, match="unknown entries"):
            SpeakerFusionConfig(engines=["whisperx"])

    def test_a_misspelling_of_a_real_engine_is_refused_too(self):
        with pytest.raises(ValueError, match="pyannote"):
            SpeakerFusionConfig(engines=["Pyannote"])

    def test_a_duplicate_engine_is_refused(self):
        # Writing the same table twice doubles its rows, and the table still validates.
        with pytest.raises(ValueError, match="more than once"):
            SpeakerFusionConfig(engines=["pyannote", "pyannote"])

    def test_an_empty_engine_list_is_refused(self):
        with pytest.raises(ValueError, match="at least one"):
            SpeakerFusionConfig(engines=[])

    def test_both_engines_are_accepted(self):
        assert SpeakerFusionConfig(engines=["pyannote", "nemotron"]).engines == \
            ["pyannote", "nemotron"]

    def test_nemotron_can_be_selected_on_its_own(self):
        assert SpeakerFusionConfig(engines=["nemotron"]).engines == ["nemotron"]

    @pytest.mark.parametrize("value", [0.0, -0.1, 1.5, 2.0])
    def test_a_ratio_outside_the_unit_interval_is_refused(self, value: float):
        with pytest.raises(ValueError, match="min_active_ratio"):
            SpeakerFusionConfig(min_active_ratio=value)

    def test_a_ratio_of_exactly_one_is_accepted(self):
        """A sustained speaker is a legitimate demand, so 1.0 is inside the domain."""
        assert SpeakerFusionConfig(min_active_ratio=1.0).min_active_ratio == 1.0

    @pytest.mark.parametrize("value", [0, -1])
    def test_a_min_face_frames_below_one_is_refused(self, value: int):
        with pytest.raises(ValueError, match="min_face_frames"):
            SpeakerFusionConfig(min_face_frames=value)

    def test_the_engine_list_is_validated_against_the_fusion_registry(self):
        """One source of truth for the allowed engines, not a copy in each validator."""
        from multimodal_pipeline.fusion import TURN_TABLES

        for engine in TURN_TABLES:
            SpeakerFusionConfig(engines=[engine])

    def test_an_unknown_key_in_the_section_is_still_refused(self):
        with pytest.raises(ValueError, match="min_overlap_ratio"):
            SpeakerFusionConfig(min_overlap_ratio=0.3)

    def test_a_stray_key_in_the_yaml_names_the_valid_keys_of_this_section(self):
        """The CLI's "unknown setting (known: ...)" helper reaches the new section.

        That help text walks `PipelineConfig.model_fields` one dotted part at a time, so a
        section whose fields are not plain sub-models silently reports "none" and leaves the
        operator guessing. Asserted through the helper rather than a subprocess because the
        helper is the thing that can be wrong here.
        """
        from multimodal_pipeline.cli import _known_settings

        known = _known_settings("speaker_fusion.threshold")
        for key in ("enabled", "engines", "min_active_ratio", "min_face_frames"):
            assert key in known, f"the error message for a typo lost {key}: {known}"
