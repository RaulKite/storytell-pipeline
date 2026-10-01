"""The stage around the builder: its path, its skips, its reuse, its registration (T22).

`test_elan.py` proves the mapping from tables to a `.eaf` is right. This file proves the
things only the pipeline can get wrong: the file lands at its registered artifact path, a
dataset with no transcript is refused rather than exported empty, the stage record's counts
match the bytes, reuse notices an edit to the python that builds the tiers, and the
registration that makes the stage run at all — class map, dependency order, the manifest
deadlock, log names — is present.

The synthetic clip is imported from `test_elan.py` rather than rebuilt: `make_dataset` is
five tables whose shapes each fire one export rule, and a second copy would be a second
fixture to keep true. `test_persons_config.py` imports its sibling's builders for the same
reason.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest
from multimodal_pipeline import elan as elan_core
from multimodal_pipeline.artifacts import (
    ARTIFACT_LAYOUT,
    MANIFEST_ARTIFACTS,
    VideoPaths,
)
from multimodal_pipeline.config import PipelineConfig
from multimodal_pipeline.elan import TIERS
from multimodal_pipeline.exceptions import ValidationError
from multimodal_pipeline.schemas import FACE_SCHEMA, SEGMENTS_SCHEMA, WORDS_SCHEMA
from multimodal_pipeline.stages.elan import ElanStage
from tests.unit.test_elan import (_face_row, _word, _write, make_dataset, real_tier_names)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def dataset(tmp_path: Path) -> dict[str, Path]:
    """The sibling module's synthetic clip, as a fixture of this file."""
    return make_dataset(tmp_path)

# ------------------------------------------------------------------ the stage itself


class FakeSource:
    def __init__(self, path: Path, video_id: str = "clip") -> None:
        self.path = path
        self.video_id = video_id


class FakeLog:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, message: Any, level: int = logging.INFO) -> None:
        self.lines.append(str(message))


def stage_context(config: PipelineConfig, dataset_dir: Path, video: Path):
    from multimodal_pipeline.artifacts import ArtifactRegistry
    from multimodal_pipeline.state import VideoState
    from multimodal_pipeline.stages.base import STAGE_ORDER, StageContext

    paths = VideoPaths(dataset_dir)
    paths.ensure_dirs()
    state = VideoState.load(paths, "clip", str(video))
    state.bind_stages(STAGE_ORDER)
    return StageContext(config=config, source=FakeSource(video), paths=paths, state=state,
                        registry=ArtifactRegistry(paths).refresh(), log=FakeLog(),
                        tools={"schema_version": "1.0"})


@pytest.fixture
def stage_config(tmp_path: Path) -> PipelineConfig:
    (tmp_path / "videos").mkdir()
    return PipelineConfig.model_validate({
        "project_root": str(tmp_path),
        "input": {"directory": str(tmp_path / "videos")},
        "output": {"directory": str(tmp_path / "out")},
    })


class TestStageWiring:
    def test_execute_writes_the_file_at_its_registered_path(self, stage_config, dataset):
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        extras = stage.run(ctx)
        assert extras.status == "completed", extras.message
        assert ctx.artifact("elan_annotations").is_file()
        assert ctx.artifact("elan_annotations") == (dataset["dir"]
                                                    / "elan" / "annotations.eaf")

    def test_the_recorded_counts_match_the_file(self, stage_config, dataset):
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        summary = stage.run(ctx).detail["provenance"]["extra"]
        eaf_text = ctx.artifact("elan_annotations").read_text(encoding="utf-8")
        root = ET.fromstring(eaf_text)
        assert summary["tiers"] == len(real_tier_names(root))
        assert summary["tier_counts"] == {"words": 3, "turns_pyannote": 1, "asd_speaking": 3,
                                          "face_tracks": 1, "pose_presence": 2}
        assert summary["annotations"] == 10
        assert summary["skipped_tiers"] == ["segments_src", "gloss_en", "turns_nemotron",
                                            "fusion_pyannote", "fusion_nemotron",
                                            "person_tracks", "voiced_blocks",
                                            "spacy_source_tokens", "spacy_source_sentences",
                                            "spacy_english_tokens", "spacy_english_sentences",
                                            "acoustic_segments"]
        assert summary["media_url"] == dataset["video"].resolve().as_uri()
        assert summary["mimetype"] == "video/mp4"
        assert summary["bytes"] == ctx.artifact("elan_annotations").stat().st_size

    def test_the_record_names_the_projected_tiers_with_both_counts(self, stage_config, dataset):
        """`tier_counts` is what ELAN shows; the record also has to say what it was projected from.

        A reader who compares the record against the table sees 3 annotations for 2 rows and needs
        the second number in the same record, not only inside the XML. An unprojected dataset
        reports an empty map rather than omitting the key, so "nothing was re-cut" and "this
        version never considered re-cutting" stay distinguishable.
        """
        from multimodal_pipeline.schemas import SEGMENTS_SCHEMA

        from tests.unit.test_elan import _segment_row

        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        assert stage.run(ctx).detail["provenance"]["extra"]["projected_tiers"] == {}

        _write(SEGMENTS_SCHEMA, dataset["dir"] / "speech" / "segments.parquet", [
            _segment_row("seg-0", 0.0, 2.0, "outer"),
            _segment_row("seg-1", 1.0, 3.0, "inner"),
        ])
        ctx.scratch.clear()
        summary = stage.run(ctx).detail["provenance"]["extra"]
        assert summary["projected_tiers"] == {"segments_src": {"logical_rows": 2, "emitted": 3}}
        assert summary["tier_counts"]["segments_src"] == 3

    def test_no_temporary_file_is_left_behind(self, stage_config, dataset):
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        strays = [p.name for p in ctx.artifact("elan_annotations").parent.iterdir()
                  if p.name != "annotations.eaf"]
        assert not strays, f"the atomic write left {strays} behind"

    def test_a_dataset_with_no_transcript_is_refused(self, stage_config, dataset):
        """Both transcript tables absent is a different case from a partial export.

        The .eaf would open to an empty grid and read as a broken export, so this is the
        ValidationError style the other derived stages use, not a skip.
        """
        (dataset["dir"] / "speech" / "words.parquet").unlink()
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        assert ctx.artifact("speech_segments").is_file() is False
        enabled, reason = stage.enabled(ctx)
        assert not enabled and "whisperx" in reason
        with pytest.raises(ValidationError) as raised:
            stage.execute(ctx)
        assert "no transcript" in str(raised.value)

    def test_one_transcript_table_alone_is_enough(self, stage_config, dataset):
        """Words-only and segments-only datasets are both normal, so neither is refused."""
        (dataset["dir"] / "speech" / "words.parquet").unlink()
        _write(SEGMENTS_SCHEMA, dataset["dir"] / "speech" / "segments.parquet", [{
            "schema_version": "1.0", "video_id": "clip", "segment_id": "seg-0",
            "start_time": 0.0, "end_time": 1.0, "duration": 1.0, "language": "en",
            "speaker_id": "SPEAKER_00", "text": "hello there", "confidence": 0.9,
        }])
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        enabled, reason = stage.enabled(ctx)
        assert enabled, reason
        summary = stage.run(ctx).detail["provenance"]["extra"]
        # `words` is absent from the census entirely rather than reported as 0: a tier whose
        # file was never produced is skipped, which is the difference between "this dataset has
        # no words" and "the words table exists and is empty".
        assert summary["tier_counts"] == {"segments_src": 1, "turns_pyannote": 1,
                                         "asd_speaking": 3, "face_tracks": 1,
                                         "pose_presence": 2}
        assert "words" in summary["skipped_tiers"]

    def test_disabled_by_config(self, stage_config, dataset):
        stage_config.elan.enabled = False
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        outcome = stage.run(ctx)
        assert outcome.status == "skipped"
        assert outcome.message == "elan.enabled = false"
        assert not ctx.artifact("elan_annotations").exists()

    def test_a_missing_source_video_is_refused(self, stage_config, dataset):
        """The media descriptor cannot be recovered later, so this cannot be a skip."""
        dataset["video"].unlink()
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        with pytest.raises(ValidationError, match="source video not found"):
            stage.execute(ctx)

    def test_validate_accepts_what_execute_wrote(self, stage_config, dataset):
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        summary = stage.validate(ctx)
        assert summary["tiers"] == 5 and summary["media_descriptors"] == 1

    def test_validate_reports_a_missing_file(self, stage_config, dataset):
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        with pytest.raises(ValidationError, match="ELAN export missing"):
            stage.validate(ctx)

    def test_validate_reports_a_truncated_file(self, stage_config, dataset):
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        path = ctx.artifact("elan_annotations")
        path.write_text(path.read_text(encoding="utf-8")[:400], encoding="utf-8")
        with pytest.raises(ValidationError, match="not well-formed XML"):
            stage.validate(ctx)

    def test_validate_reports_media_that_is_not_reachable(self, stage_config, dataset):
        """The .eaf is the file a user opens in ELAN; a dead link is the failure they see."""
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        dataset["video"].unlink()
        with pytest.raises(ValidationError, match="not reachable from elan/"):
            stage.validate(ctx)

    def test_validate_rejects_a_relative_url_based_on_the_dataset_directory(
            self, stage_config, dataset):
        """A link that only resolves from ``<dataset>`` is broken, and must be reported.

        This is the exact document the first version of the writer produced: the relative URL was
        one ``../`` short, so it pointed at ``<dataset>/../input_videos`` instead of the real
        ``<dataset>/../../input_videos``. The old check resolved it from the dataset directory,
        agreed with the writer, and passed. The video is untouched here — the file on disk is
        fine and the link in the document is not, which is what a validation step exists to catch.
        """
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        path = ctx.artifact("elan_annotations")
        text = path.read_text(encoding="utf-8")
        good = "../../../input_videos/"
        assert good in text, "fixture changed shape; this test would pass vacuously"
        path.write_text(text.replace(good, "../../input_videos/"), encoding="utf-8")
        assert dataset["video"].is_file()
        with pytest.raises(ValidationError, match="not reachable from elan/"):
            stage.validate(ctx)

    def test_validate_rejects_a_document_written_for_another_video(self, stage_config, dataset):
        """Close the branch no test could reach until now (review advisory R2-002).

        The check exists for a stale export copied between dataset directories: it parses, it
        links a video that exists, its census is intact, and it shows the wrong clip in ELAN.
        Nothing in this file could previously make it fire, so nothing said it still worked.

        The foreign name is chosen to *contain* this video's name — `xclip.mp4` contains
        `clip.mp4` — because that is the shape that killed the original implementation, which
        tested the two URLs for substring containment and called the foreign document a match.
        The foreign file is written to disk so reachability cannot be the reason this raises:
        the wrong name has to be.
        """
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        original = dataset["video"].name
        foreign = dataset["video"].parent / f"x{original}"
        assert original in foreign.name, "the mutant relies on containment; keep it"
        foreign.write_bytes(b"stub")
        path = ctx.artifact("elan_annotations")
        text = path.read_text(encoding="utf-8")
        assert original in text, "fixture changed shape; this test would pass vacuously"
        path.write_text(text.replace(original, foreign.name), encoding="utf-8")
        with pytest.raises(ValidationError, match="was written for another source"):
            stage.validate(ctx)

    def test_validate_rejects_a_name_that_only_appears_as_a_directory(self, stage_config, dataset):
        """Review advisory R3-001: the name must be the linked file, not a directory on the way.

        A descriptor may be written for ``<dataset>/clip.mp4/other.mp4`` — a directory named like
        this video holding some other file. The path reaches a real file, so reachability is not
        the reason to complain, and a check that scanned every path segment found ``clip.mp4``
        sitting in the middle and called it a match. Tightened to the last segment of each URL.
        """
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        original = dataset["video"].name
        # <dataset>/clip.mp4/other.mp4, reached as ../clip.mp4/other.mp4 from elan/
        decoy_dir = dataset["dir"] / original
        decoy_dir.mkdir()
        (decoy_dir / "other.mp4").write_bytes(b"stub")
        path = ctx.artifact("elan_annotations")
        text = path.read_text(encoding="utf-8")
        good = "../../../input_videos/"
        assert good in text, "fixture changed shape; this test would pass vacuously"
        path.write_text(text.replace(good + original, "../" + original + "/other.mp4"),
                        encoding="utf-8")
        assert (decoy_dir / "other.mp4").is_file()
        with pytest.raises(ValidationError, match="was written for another source"):
            stage.validate(ctx)

    def test_validate_reports_a_census_that_lost_a_tier(self, stage_config, dataset):
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        path = ctx.artifact("elan_annotations")
        text = path.read_text(encoding="utf-8")
        # Drop one tier's element wholesale, leaving parseable XML behind — the shape a
        # hand edit or a partial truncation produces.
        start = text.index('<TIER TIER_ID="asd_speaking"')
        end = text.index("</TIER>", start) + len("</TIER>")
        path.write_text(text[:start] + text[end:], encoding="utf-8")
        with pytest.raises(ValidationError, match="census are absent"):
            stage.validate(ctx)

    def test_validate_accepts_a_projected_document(self, stage_config, dataset):
        """The tiers this export now re-cuts still pass their own validation.

        Two overlapping segments in `segments_src` are the trigger: the emitted document is
        disjoint by construction, so the new check has to accept it (and accept the extra
        property) rather than complain about the shape it was written to police.
        """
        from multimodal_pipeline.schemas import SEGMENTS_SCHEMA

        from tests.unit.test_elan import _segment_row

        _write(SEGMENTS_SCHEMA, dataset["dir"] / "speech" / "segments.parquet", [
            _segment_row("seg-0", 0.0, 2.0, "outer"),
            _segment_row("seg-1", 1.0, 3.0, "inner"),
        ])
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        summary = stage.run(ctx).detail["provenance"]["extra"]
        assert summary["tier_counts"]["segments_src"] == 3
        assert stage.validate(ctx)["tiers"] == 6

    @pytest.mark.parametrize("mutation,expected", [
        # The two segments as written are (0, 1000) and (1000, 2000): legal, touching, half-open.
        # Each mutation retargets one of those four slots and nothing else.
        ("second-start-earlier", "overlapping"),   # (0, 1000) + (500, 2000) -> partial overlap
        ("nested-inside", "overlapping"),          # (0, 2000) + (500, 900) -> wholly inside
        ("zero-width", "start < end"),             # (1000, 1000) -> no width to select
        ("reversed", "start < end"),               # (2000, 900) -> negative
    ])
    def test_validate_rejects_same_tier_overlap_written_by_hand(self, stage_config, dataset,
                                                               mutation: str,
                                                               expected: str) -> None:
        """The overlap check resolves the XML itself, and does not trust the document's metadata.

        Every case edits `TIME_VALUE`s only. The tier census and `pipeline-overlap-projection` are
        left exactly as the writer produced them, so a check that consulted either would pass all
        four — and the metadata is the tempting place to look, because the writer's own account of
        the tier always says the tier was written correctly. The last two cases exercise the same
        walk's other rule: `start >= end` is refused even though both slots are fine integers on
        their own, because an interval with no width (or a negative one) cannot be selected.

        The slots are found by id from the tier's own annotations rather than hardcoded, so this
        keeps testing the mutation even when pympi's slot numbering changes.
        """
        import re

        from multimodal_pipeline.schemas import SEGMENTS_SCHEMA

        from tests.unit.test_elan import _segment_row

        _write(SEGMENTS_SCHEMA, dataset["dir"] / "speech" / "segments.parquet", [
            _segment_row("seg-0", 0.0, 1.0, "first"),
            _segment_row("seg-1", 1.0, 2.0, "second"),
        ])
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        path = ctx.artifact("elan_annotations")
        text = path.read_text(encoding="utf-8")

        root = ET.fromstring(text)
        tier = next(element for element in root.iter("TIER")
                    if element.attrib.get("TIER_ID") == "segments_src")
        refs = [(element.attrib["TIME_SLOT_REF1"], element.attrib["TIME_SLOT_REF2"])
                for element in tier.iter("ALIGNABLE_ANNOTATION")]
        assert refs and len(refs) == 2, refs
        first_start, first_end = refs[0]
        second_start, second_end = refs[1]

        def retarget(slot_id: str, value: int) -> None:
            nonlocal text
            pattern = (f'TIME_SLOT_ID="{slot_id}" TIME_VALUE="\\d+"')
            replaced, count = re.subn(pattern, f'TIME_SLOT_ID="{slot_id}" '
                                      f'TIME_VALUE="{value}"', text)
            assert count == 1, f"slot {slot_id} not found once in the document"
            text = replaced

        if mutation == "second-start-earlier":
            retarget(second_start, 500)
        elif mutation == "nested-inside":
            retarget(first_end, 2000)
            retarget(second_start, 500)
            retarget(second_end, 900)
        elif mutation == "zero-width":
            retarget(second_end, 1000)
        else:
            retarget(second_end, 900)
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValidationError, match="tier segments_src") as raised:
            stage.validate(ctx)
        # The message says which rule the file broke, so a reader is not sent looking for the
        # wrong defect: overlap and an unselectable interval are different repairs.
        assert expected in str(raised.value), str(raised.value)

    def test_validate_rejects_overlap_even_though_the_projection_property_says_otherwise(
            self, stage_config, dataset):
        """Metadata is the writer's account of itself, not evidence about the bars.

        The document here really is projected (`pipeline-overlap-projection` is present and
        internally consistent), and the edit then re-overlaps one emitted segment while leaving
        that property and the tier census untouched. A validator that trusted either would call
        this file fine; ELAN would not open it.
        """
        import re

        from multimodal_pipeline.schemas import SEGMENTS_SCHEMA

        from tests.unit.test_elan import _segment_row

        _write(SEGMENTS_SCHEMA, dataset["dir"] / "speech" / "segments.parquet", [
            _segment_row("seg-0", 0.0, 2.0, "outer"),
            _segment_row("seg-1", 1.0, 3.0, "inner"),
        ])
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        path = ctx.artifact("elan_annotations")
        text = path.read_text(encoding="utf-8")
        assert "pipeline-overlap-projection" in text, "fixture is not projected; test vacuous"
        assert "segments_src=3" in text

        root = ET.fromstring(text)
        tier = next(element for element in root.iter("TIER")
                    if element.attrib.get("TIER_ID") == "segments_src")
        refs = [(element.attrib["TIME_SLOT_REF1"], element.attrib["TIME_SLOT_REF2"])
                for element in tier.iter("ALIGNABLE_ANNOTATION")]
        assert len(refs) == 3, refs
        # Widen the first segment (0, 1000) to (0, 2500): it now covers the second and third.
        slot = refs[0][1]
        text, count = re.subn(f'TIME_SLOT_ID="{slot}" TIME_VALUE="\\d+"',
                              f'TIME_SLOT_ID="{slot}" TIME_VALUE="2500"', text)
        assert count == 1
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValidationError) as raised:
            stage.validate(ctx)
        assert "overlapping" in str(raised.value)

    def test_validate_allows_touching_annotations_in_one_tier(self, stage_config, dataset):
        """Half-open neighbours share a boundary; that is the legal case, not an overlap.

        Without this the check would fail every ordinary transcript on the corpus, and the
        projection's own "touching is not an overlap" rule would have no counterparty on the
        validation side.
        """
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        assert stage.validate(ctx)["tiers"] == 5

    def test_validate_reports_a_time_slot_that_is_not_an_integer(self, stage_config, dataset):
        """A `TIME_VALUE` that is not an integer is a different failure from a bad interval.

        It is also the one case where nothing can be resolved, so the message has to name the
        slots rather than silently treating them as zero.
        """
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        path = ctx.artifact("elan_annotations")
        text = path.read_text(encoding="utf-8")
        text = text.replace('<TIME_SLOT TIME_SLOT_ID="ts3" TIME_VALUE="400" />',
                            '<TIME_SLOT TIME_SLOT_ID="ts3" TIME_VALUE="four" />')
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValidationError, match="not integers"):
            stage.validate(ctx)

    def test_rerunning_replaces_the_file_rather_than_appending(self, stage_config, dataset):
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        first = ctx.artifact("elan_annotations").read_text(encoding="utf-8")
        stage.run(ctx)
        second = ctx.artifact("elan_annotations").read_text(encoding="utf-8")
        assert len(real_tier_names(ET.fromstring(second))) == 5
        # Only the writer's timestamp differs; the annotations must not grow.
        assert first.count("<ANNOTATION ") == second.count("<ANNOTATION ")

    def test_the_fingerprint_notices_a_changed_input_table(self, stage_config, dataset):
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        before = stage.config_fingerprint(ctx)
        # Keyed by artifact, not by tier, because the set of hashed inputs is now wider than the
        # set of tiers: every key names a file, and a reader comparing the payload against
        # `stage.inputs` finds the same names on both sides.
        assert before["speech_words_digest"] is not None
        assert before["person_tracks_digest"] is None, "absent inputs hash as None, not absent"
        ctx.scratch.clear()
        _write(WORDS_SCHEMA, dataset["dir"] / "speech" / "words.parquet",
               [_word("changed", 0.0, 0.4)])
        after = stage.config_fingerprint(ctx)
        assert after["speech_words_digest"] != before["speech_words_digest"]
        assert after["tiers"] == before["tiers"]

    def test_the_secondary_inputs_are_declared_next_to_the_primary_ones(self) -> None:
        """`person_frames` and `frame_index` are inputs, not side reads.

        ``stage.inputs`` is what `status --plan` and the state record print as the stage's
        dependencies, and a table that a tier reads without being declared is the reuse hole
        §31 describes from the other side: the export changes when that file changes, so it has
        to be in the list. One exported list (`elan.SECONDARY_INPUTS`) feeds the tier reader,
        this declaration and the fingerprint, so the three cannot drift.
        """
        from multimodal_pipeline.elan import SECONDARY_INPUTS

        stage = ElanStage()
        assert set(SECONDARY_INPUTS["person_tracks"]) == {"person_frames", "frame_index"}
        assert stage.inputs == tuple(spec.artifact for spec in TIERS) + tuple(
            name for _tier, names in SECONDARY_INPUTS.items() for name in names)
        assert "person_frames" in stage.inputs and "frame_index" in stage.inputs
        assert set(stage.inputs) <= set(ARTIFACT_LAYOUT)

    def test_the_fingerprint_records_the_secondary_dependency_itself(
            self, stage_config, dataset):
        """Adding a dependency is a change to what the export can claim, so it hashes.

        A tier that starts reading one more table can produce a different .eaf from byte-identical
        inputs and an unchanged tier list. The dependency map is in the payload for that reason:
        the fingerprint moves when the *contract* moves, not only when a file does.
        """
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        payload = stage.config_fingerprint(ctx)
        assert payload["secondary_inputs"] == {"person_tracks": ["person_frames",
                                                                "frame_index"]}

    def test_the_fingerprint_carries_a_key_per_secondary_input(self, stage_config, dataset):
        """An absent secondary input is a `None` key, exactly like an absent primary one.

        The synthetic clip has neither file, so both keys are present and null. Dropping the key
        when the file is missing would make "file absent" and "this version of the stage never
        considered the file" hash alike.
        """
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        fingerprint = stage.config_fingerprint(ctx)
        assert fingerprint["person_frames_digest"] is None
        assert fingerprint["frame_index_digest"] is None

    def test_the_fingerprint_notices_an_inventoried_table_appearing(
            self, stage_config, dataset) -> None:
        """A table no tier reads still moves the fingerprint, because it moves the .eaf's claim.

        `pose/face.parquet` is not an input: nothing is read from it, so its bytes appear in no
        digest. But the coverage inventory stats it, and its state in the document flips from
        `absent` to `present, not exported` when it shows up. Without this key the stage would look
        reusable while the file it reuses says something false about a table that now exists — the
        same class of hole as an unhashed dependency, reached from the opposite direction.

        Existence is what is hashed, not contents: the state depends on a stat() alone, so hashing
        unread bytes would rerun the export every time `openpose` reran for a file whose entry could
        not have changed.
        """
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        before = stage.config_fingerprint(ctx)
        assert before["inventory_present"]["pose_face"] is False
        assert before["inventory_present"]["speech_words"] is True

        path = dataset["dir"] / ARTIFACT_LAYOUT["pose_face"]
        path.parent.mkdir(parents=True, exist_ok=True)
        _write(FACE_SCHEMA, path, [_face_row(0, 0)])
        ctx.scratch.clear()
        added = stage.config_fingerprint(ctx)
        assert added["inventory_present"]["pose_face"] is True
        assert added != before

        # Rewriting the same table with different rows must NOT move it: contents are not the state.
        _write(FACE_SCHEMA, path, [_face_row(0, 0), _face_row(1, 0)])
        ctx.scratch.clear()
        assert stage.config_fingerprint(ctx) == added

        path.unlink()
        ctx.scratch.clear()
        assert stage.config_fingerprint(ctx)["inventory_present"]["pose_face"] is False

    def test_the_fingerprint_inventory_covers_exactly_the_inventoried_tables(
            self, stage_config, dataset) -> None:
        """The fingerprint and the property are kept answering about the same set of artifacts.

        Both are derived from the registry, so a new table lands in both; asserting the equality
        here is what stops one of the two from being narrowed to a hardcoded subset and quietly
        reopening the reuse hole.
        """
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        fingerprint = stage.config_fingerprint(ctx)
        summary = record_of(stage, ctx)
        assert set(fingerprint["inventory_present"]) == set(summary["coverage"])

    @pytest.mark.parametrize("artifact,relative", [
        ("person_frames", "persons/frames.parquet"),
        ("frame_index", "source/frame_index.parquet"),
    ])
    def test_the_fingerprint_notices_a_secondary_input_appearing_or_changing(
            self, stage_config, dataset, artifact: str, relative: str) -> None:
        """The three states a dependency passes through must all move the hash.

        Absent → present is the one that a `None`-or-missing-key mix-up hides; present →
        edited is the one every table gets; deleting it again has to come back to the first
        hash, which is what shows the key is keyed on the file and not on the attempt.
        """
        from multimodal_pipeline.schemas import FRAME_INDEX_SCHEMA, PERSON_FRAMES_SCHEMA

        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        key = f"{artifact}_digest"
        before = stage.config_fingerprint(ctx)
        assert before[key] is None
        path = dataset["dir"] / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        schema = PERSON_FRAMES_SCHEMA if artifact == "person_frames" else FRAME_INDEX_SCHEMA
        rows = ([{"schema_version": "1.0", "video_id": "clip", "frame_number": 0,
                  "timestamp": 0.0, "person_id": 1, "x1": 1.0, "y1": 1.0, "x2": 2.0,
                  "y2": 2.0, "confidence": 0.5, "track_confidence": None,
                  "confidence_reason": "no_track_confidence", "bbox_area": 1.0,
                  "persons_in_frame": 1}] if artifact == "person_frames"
                else [{"schema_version": "1.0", "video_id": "clip", "frame_number": 0,
                       "pts_seconds": 0.0}])
        _write(schema, path, rows)
        ctx.scratch.clear()
        added = stage.config_fingerprint(ctx)
        assert added[key] is not None
        _write(schema, path, rows + rows)
        ctx.scratch.clear()
        edited = stage.config_fingerprint(ctx)
        assert edited[key] != added[key]
        path.unlink()
        ctx.scratch.clear()
        assert stage.config_fingerprint(ctx)[key] is None

    def test_the_fingerprint_carries_the_python_source_digest(self, stage_config, dataset):
        """The hole §31 closed for pose_normalized, which has no worker either.

        Without it a change to the millisecond rounding in ``elan.py`` leaves config,
        thresholds and every input digest identical, and the fix does nothing until someone
        deletes the .eaf by hand.
        """
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        digest = stage.config_fingerprint(ctx)["_python_code_sha256"]
        assert digest and len(digest) == 64
        from multimodal_pipeline.stages.base import python_source_digest
        from multimodal_pipeline.stages import elan as stage_module

        assert digest == python_source_digest(elan_core, stage_module)

    def test_declared_inputs_are_registered_artifacts(self) -> None:
        stage = ElanStage()
        assert set(stage.inputs) <= set(ARTIFACT_LAYOUT)
        assert set(stage.outputs) <= set(ARTIFACT_LAYOUT)
        assert stage.outputs == ("elan_annotations",)

    def test_outputs_present_only_asks_for_the_eaf(self, stage_config, dataset):
        """Declared inputs are read softly; they must not gate reuse."""
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        assert not stage.outputs_present(ctx)
        stage.run(ctx)
        assert stage.outputs_present(ctx)


# ------------------------------------------------------------------ wiring elsewhere


class TestConfigAndRegistry:
    def test_elan_is_enabled_by_default(self) -> None:
        """The default is a decision: no GPU, no download, seconds per video. See ElanConfig."""
        from multimodal_pipeline.config import ElanConfig

        assert ElanConfig().enabled is True

    def test_the_section_reaches_the_stage_configs_map(self, stage_config) -> None:
        assert stage_config.stage_configs.get("elan") is stage_config.elan

    def test_the_section_is_in_the_configured_field_list(self) -> None:
        assert "elan" in PipelineConfig.model_fields

    def test_an_unknown_elan_key_is_still_an_error(self, tmp_path) -> None:
        """The section is in the same `_Model` family, so config drift still fails loudly."""
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            PipelineConfig.model_validate({
                "project_root": str(tmp_path),
                "input": {"directory": str(tmp_path)},
                "output": {"directory": str(tmp_path)},
                "elan": {"enabledl": True},
            })

    def test_the_export_is_a_manifest_artifact(self) -> None:
        assert "elan_annotations" in MANIFEST_ARTIFACTS
        assert len(MANIFEST_ARTIFACTS) == 44

    def test_the_registered_path_lives_in_its_own_directory(self) -> None:
        assert ARTIFACT_LAYOUT["elan_annotations"] == "elan/annotations.eaf"

    def test_ensure_dirs_creates_the_elan_directory(self, tmp_path) -> None:
        """The promise every stage relies on: your output directory already exists."""
        VideoPaths(tmp_path / "dataset").ensure_dirs()
        assert (tmp_path / "dataset" / "elan").is_dir()

    def test_the_example_config_ships_the_section(self) -> None:
        import yaml

        text = (ROOT / "config" / "config.example.yaml").read_text(encoding="utf-8")
        payload = yaml.safe_load(text)
        assert payload["elan"] == {"enabled": True}

    def test_the_library_is_a_root_dependency_and_not_a_new_environment(self) -> None:
        """The orchestrator builds the .eaf itself, so there is no uv project for it.

        Asserted rather than trusted because the *absence* of an `environments/*` project is
        the design: a reviewer cannot tell a deliberate pure-python stage from a forgotten
        environment by reading the diff.
        """
        import tomllib

        payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        deps = " ".join(payload["project"]["dependencies"])
        assert "pympi-ling" in deps
        assert not (ROOT / "environments" / "elan").exists()

    def test_the_dependency_resolves_and_imports(self) -> None:
        """The pyproject edit is what persists; this says the pin is installable."""
        from importlib import metadata

        import pympi.Elan

        assert metadata.version("pympi-ling")
        assert pympi.Elan.Eaf is not None


class TestStageOrder:
    def test_elan_is_the_last_stage(self) -> None:
        from multimodal_pipeline.stages.base import STAGE_ORDER

        assert STAGE_ORDER[-1] == "elan"
        assert STAGE_ORDER.index("finalization") == len(STAGE_ORDER) - 2

    def test_it_depends_on_finalization_alone(self) -> None:
        from multimodal_pipeline.stages.base import STAGE_DEPENDENCIES

        assert STAGE_DEPENDENCIES["elan"] == ("finalization",)

    def test_finalization_does_not_depend_on_elan(self) -> None:
        """The deadlock guard.

        `finalization` writes the manifest that has to declare `elan`, so making it depend on
        `elan` would need `elan` to have already run — and `elan` depends on `finalization`.
        Asserted on both sides so the exclusion cannot be undone by a tidy-up of the tuple.
        """
        from multimodal_pipeline.stages.base import (
            STAGE_DEPENDENCIES,
            STAGE_ORDER,
            dependency_chain,
        )

        assert "elan" not in STAGE_DEPENDENCIES["finalization"]
        assert "finalization" not in STAGE_DEPENDENCIES["finalization"]
        assert len(STAGE_DEPENDENCIES["finalization"]) == len(STAGE_ORDER) - 2
        assert dependency_chain("finalization") == list(STAGE_ORDER[:-2])
        # And elan's transitive chain is everything, in canonical order.
        assert dependency_chain("elan") == list(STAGE_ORDER[:-1])

    def test_the_stage_has_its_own_log_name(self) -> None:
        from multimodal_pipeline.artifacts import STAGE_LOG_NAMES
        from multimodal_pipeline.stages.base import STAGE_ORDER

        assert "elan" in STAGE_LOG_NAMES
        assert set(STAGE_LOG_NAMES) == set(STAGE_ORDER)

    def test_dependants_of_finalization_is_only_elan(self) -> None:
        from multimodal_pipeline.stages.base import dependants_of

        assert dependants_of("finalization") == ["elan"]


# ---------------------------------------------------- coverage and drop counts in the record (B5)


def record_of(stage: ElanStage, ctx: Any) -> dict[str, Any]:
    """The `extra` block of a run's provenance record — the thing that lands in `status.json`."""
    return stage.run(ctx).detail["provenance"]["extra"]


def coverage_envelope_of(path: Path) -> dict[str, Any]:
    """The whole coverage property read out of the XML, not through pympi's property dict.

    `validate` reads the same way (:mod:`xml.etree.ElementTree` over the file text), so a test that
    mutated the property and then read it back through the library would be testing pympi's parser
    rather than the check.
    """
    import json as _json

    root = ET.fromstring(path.read_text(encoding="utf-8"))
    raw = next((element.text or "") for element in root.iter("PROPERTY")
               if element.attrib.get("NAME") == "pipeline-coverage")
    return _json.loads(raw)


def coverage_artifacts_of(path: Path) -> dict[str, Any]:
    """The per-artifact map inside that property."""
    return coverage_envelope_of(path)["artifacts"]


def rewrite_coverage(path: Path, artifacts: dict[str, Any]) -> None:
    """Replace the coverage property's artifact map, leaving every bar and slot alone.

    The mutation every one of these tests needs: a document whose *claim* about what it represents
    changed while the annotations did not. Editing a bar instead would let the overlap check do the
    complaining, and the coverage check would never be the reason.
    """
    import json as _json
    import re

    text = path.read_text(encoding="utf-8")
    document = coverage_envelope_of(path)
    document["artifacts"] = artifacts
    payload = _json.dumps(document, ensure_ascii=False, separators=(",", ":"))
    # The property's text sits between its own open and close tag; escape is XML, not JSON.
    payload = (payload.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    pattern = r'(<PROPERTY NAME="pipeline-coverage">).*?(</PROPERTY>)'
    replaced, count = re.subn(pattern, lambda m: m.group(1) + payload + m.group(2),
                              text, flags=re.DOTALL)
    assert count == 1, "the coverage property is not in the document once; fixture changed shape"
    path.write_text(replaced, encoding="utf-8")


class TestRecordReportsWhatWasLeftOut:
    """`status.json` has to answer "is this signal missing from the clip or from the export".

    The stage record already carried `skipped_tiers`, which names a tier that got no bars. It did
    not carry *why*: a skipped tier is a skipped tier whether its table was never written or was
    written and is not something the export represents. Both halves are here, plus the record's
    drop counters — the mirror image of `projected_tiers`, and the one where rows really are lost.
    """

    def test_the_record_carries_the_coverage_map_the_document_carries(
            self, stage_config, dataset) -> None:
        """Same keys, same states, read out of the written file.

        Asserted against the bytes rather than against `coverage_inventory(...)`, because the point
        of reading the document is that the record and the file cannot then disagree.
        """
        _write(FACE_SCHEMA, dataset["dir"] / ARTIFACT_LAYOUT["pose_face"], [_face_row(0, 0)])
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        summary = record_of(stage, ctx)
        on_disk = coverage_artifacts_of(ctx.artifact("elan_annotations"))
        assert summary["coverage"] == on_disk
        assert summary["coverage"]["pose_face"]["state"] == "present, not exported"
        assert summary["coverage"]["speech_words"] == {
            "state": "exported", "tier": "words", "path": "speech/words.parquet"}

    def test_a_skipped_tier_and_an_unrepresented_table_are_different_answers(
            self, stage_config, dataset) -> None:
        """`gloss_en` is skipped because no file exists; `pose_face` is unread because no tier reads it.

        Both leave something out of the grid, and before this change `status.json` reported the
        first one (`skipped_tiers`) and was silent about the second. The pair is the distinction the
        operator's question turns on, asserted in the record rather than in the builder.
        """
        _write(FACE_SCHEMA, dataset["dir"] / ARTIFACT_LAYOUT["pose_face"], [_face_row(0, 0)])
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        summary = record_of(stage, ctx)
        assert "gloss_en" in summary["skipped_tiers"]
        assert summary["coverage"]["translation_segments"]["state"] == "absent"
        assert "reason" not in summary["coverage"]["translation_segments"]
        assert summary["coverage"]["pose_face"]["state"] == "present, not exported"
        assert "dense per-joint numeric tracks" in summary["coverage"]["pose_face"]["reason"]

    def test_the_record_reports_dropped_rows_only_for_the_tiers_that_refused_some(
            self, stage_config, dataset) -> None:
        """24 rows, 20 bars: the record names the four and which of the two rules took them.

        Two rows with no time and one NaN in one tier, and a clean tier that must not appear. The
        log line said this and nothing else did — which meant the answer lived in a terminal buffer.
        """
        from multimodal_pipeline.schemas import WORDS_SCHEMA

        from tests.unit.test_elan import _word

        _write(WORDS_SCHEMA, dataset["dir"] / "speech" / "words.parquet", [
            _word("a", 0.0, 0.4),
            _word("b", 1.0, None),
            _word("c", None, 2.0),
            _word("d", float("nan"), 3.0),
        ])
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        summary = record_of(stage, ctx)
        assert summary["dropped_rows"] == {"words": {"missing_time": 2, "non_finite": 1}}
        assert summary["tier_counts"]["words"] == 1
        # The clean tiers are absent rather than listed as zeroes, so an empty map is a real answer.
        assert "pose_presence" not in summary["dropped_rows"]

    def test_a_projected_tier_and_a_dropped_tier_are_reported_apart(
            self, stage_config, dataset) -> None:
        """More bars than rows is a layout choice; fewer is a producer defect. Never one number.

        One tier is re-cut (2 rows, 3 bars) while another refuses rows, in the same run. If the two
        mechanisms were ever merged into a single "rows adjusted" figure, a table with null
        timestamps would read as a formatting detail, so the record keeps them in two keys and this
        test keeps them in two places.
        """
        from multimodal_pipeline.schemas import SEGMENTS_SCHEMA, WORDS_SCHEMA

        from tests.unit.test_elan import _segment_row, _word

        _write(SEGMENTS_SCHEMA, dataset["dir"] / "speech" / "segments.parquet", [
            _segment_row("seg-0", 0.0, 2.0, "outer"),
            _segment_row("seg-1", 1.0, 3.0, "inner"),
        ])
        _write(WORDS_SCHEMA, dataset["dir"] / "speech" / "words.parquet", [
            _word("a", 0.0, 0.4), _word("b", 1.0, None)])
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        summary = record_of(stage, ctx)
        assert summary["projected_tiers"] == {"segments_src": {"logical_rows": 2, "emitted": 3}}
        assert summary["dropped_rows"] == {"words": {"missing_time": 1, "non_finite": 0}}

    def test_the_record_shape_is_stable_when_nothing_was_dropped_or_left_out(
            self, stage_config, dataset) -> None:
        """Both keys exist on a clean run, so a consumer never has to guess which schema it has."""
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        summary = record_of(stage, ctx)
        assert summary["dropped_rows"] == {}
        assert summary["projected_tiers"] == {}
        assert len(summary["coverage"]) == 22


class TestValidateChecksCoverageAgainstTheDocument:
    """The inventory is cross-checked against the document's own tiers, and against nothing else.

    `_check_census` already compares a property against the tier list, and the temptation is to
    read this as covered. It is not: the census maps **tier to count**, so an edit that retargets
    one artifact onto a different tier — or deletes an entry while leaving the tier — moves no
    census name. Coverage is the only property that maps an artifact to a tier, which makes it the
    only one that can be contradicted by the document.

    The rule runs one way: a tier the document *declares* has to be claimed by some entry. The
    reverse is a legitimate state and is left alone — an entry naming a tier the document omits is
    what a skipped tier looks like, and failing it would send a partial export through an endless
    rerun (see `test_validate_accepts_a_tier_skipped_because_its_table_was_unreadable`).

    The check is deliberately blind to the registry. Asking `ARTIFACT_LAYOUT` what tables exist
    would ask the tree, and the tree changes when somebody reruns a stage: a finished export would
    become unvalidateable because a producer wrote one more file afterwards. Consistency between
    the property and the `TIER` elements is the whole claim, and both are in the file.
    """

    def test_validate_accepts_the_document_the_writer_produced(self, stage_config, dataset) -> None:
        """Baseline for the three mutations below: without it they could pass on a broken check."""
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        assert stage.validate(ctx)["tiers"] == 5

    def test_validate_returns_the_coverage_states_because_it_is_the_persisted_channel(
            self, stage_config, dataset) -> None:
        """`validate`'s return value is the only stage output that survives into `status.json`.

        `execute` returns a rich provenance block (tier counts, skipped tiers, coverage, drop
        counters), but the orchestrator persists only `tool_version`, `model_version`, `command`,
        `executable`, `exit_code` and whatever `validate` returned — measured on this corpus, no
        `status.json` on disk has ever contained `tier_counts`, `coverage` or `projected_tiers`, for
        any stage. So a claim that "is this signal missing because of the video or because of the
        export" is answerable without opening the XML has to be delivered by this function, or it
        has to stop being made.

        Returning the state counts (not the whole 22-entry inventory) keeps the record small and
        keeps the detail in the file it describes.
        """
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        summary = record_of(stage, ctx)
        result = stage.validate(ctx)
        states = {name: entry["state"] for name, entry in summary["coverage"].items()}
        counts: dict[str, int] = {}
        for state in states.values():
            counts[state] = counts.get(state, 0) + 1
        assert result["coverage_states"] == counts, result
        assert result["coverage_not_exported"] == sorted(
            name for name, state in states.items() if state == "present, not exported"), result

    def test_validate_returns_the_projection_counts_read_from_the_document(
            self, stage_config, dataset) -> None:
        """"More bars than rows" has to be answerable from the record, and this is the channel.

        Two segment rows sharing an instant are re-cut into three bars. The run record says so, but
        the run record is not persisted; the validated document is, and this reads the same property
        back out of the XML rather than reusing anything the builder remembered. A tier that was not
        projected is absent rather than listed as equal counts, so an empty map is a real answer.
        """
        from multimodal_pipeline.schemas import SEGMENTS_SCHEMA

        from tests.unit.test_elan import _segment_row

        _write(SEGMENTS_SCHEMA, dataset["dir"] / "speech" / "segments.parquet", [
            _segment_row("seg-0", 0.0, 2.0, "outer"),
            _segment_row("seg-1", 1.0, 3.0, "inner"),
        ])
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        summary = record_of(stage, ctx)
        assert summary["projected_tiers"]["segments_src"] == {"logical_rows": 2, "emitted": 3}
        result = stage.validate(ctx)
        assert result["projected_tiers"] == {"segments_src": {"logical_rows": 2, "emitted": 3}}, result

    def test_the_persisted_projection_counts_follow_the_file_not_the_run(
            self, stage_config, dataset) -> None:
        """The value must come from the document, so a stale record cannot contradict the file.

        The projection property is rewritten to name counts the bars do not support, and `validate`
        reports what the document says. This is the cheap half of the promise: the record is only
        worth reading if it is a reading of the file rather than a memory of the run that wrote it.
        (The bars themselves are still checked against the `TIME_SLOT` values, not against this
        property — a metadata block cannot make an overlapping pair legal.)
        """
        from multimodal_pipeline.schemas import SEGMENTS_SCHEMA

        from tests.unit.test_elan import _segment_row

        _write(SEGMENTS_SCHEMA, dataset["dir"] / "speech" / "segments.parquet", [
            _segment_row("seg-0", 0.0, 2.0, "outer"),
            _segment_row("seg-1", 1.0, 3.0, "inner"),
        ])
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        summary = record_of(stage, ctx)
        path = ctx.artifact("elan_annotations")
        import json as _json
        import re

        text = path.read_text(encoding="utf-8")
        match = re.search(r'(<PROPERTY NAME="pipeline-overlap-projection">)(.*?)(</PROPERTY>)',
                          text, flags=re.DOTALL)
        assert match, "the projected fixture stopped writing a projection property"
        document = _json.loads(match.group(2))
        document["tiers"]["segments_src"]["final_annotation_count"] = 99
        payload = (_json.dumps(document, ensure_ascii=False, separators=(",", ":"))
                   .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        path.write_text(text[:match.start()] + match.group(1) + payload + match.group(3)
                        + text[match.end():], encoding="utf-8")
        result = stage.validate(ctx)
        assert result["projected_tiers"]["segments_src"]["emitted"] == 99, result
        # ...and the bars are still the bars: they are checked against the TIME_SLOT values, not
        # against this property, so the tier count is the one the run itself reported.
        assert result["tiers"] == summary["tiers"], result

    def test_validate_accepts_a_tier_skipped_because_its_table_was_unreadable(
            self, stage_config, dataset) -> None:
        """An export that lost one tier to a corrupt table is still a consistent document.

        The writer's own two rules collide here unless the check is pointed the right way: a table
        that exists but cannot be read leaves its tier undeclared (`words` never gets a `TIER`
        element) while its artifact stays `exported`, because a tier that reads a table and fails is
        not a table the export never represents. So the property legitimately names a tier the
        document omits. Making that an error would fail the documented "lose one tier, keep the
        other sixteen" state — and because `validate` is the reuse gate, the stage would re-export,
        skip the same tier and fail the same way forever instead of settling.

        This is the case found while writing the check, not a case imagined after it: the first
        version complained here, on a document no edit had touched.
        """
        path = dataset["dir"] / ARTIFACT_LAYOUT["speech_words"]
        path.write_bytes(b"not a parquet file")
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        summary = record_of(stage, ctx)
        assert "words" in summary["skipped_tiers"], summary["skipped_tiers"]
        assert summary["coverage"]["speech_words"]["state"] == "exported"
        assert summary["coverage"]["speech_words"]["tier"] not in summary["tier_counts"]
        # The four tiers the fixture can still build are all it declares, and that is fine.
        assert stage.validate(ctx)["tiers"] == summary["tiers"]

    def test_validate_reports_a_tier_deleted_from_the_document_as_a_census_loss(
            self, stage_config, dataset) -> None:
        """Removing a whole tier is the census check's job, and coverage does not double-report it.

        The edit deletes the `TIER` element and leaves both properties intact, so the census names
        the loss (it compares its own names against the tier list) while the coverage entry for that
        tier becomes the legitimate skipped-tier shape above. Asserting the message that comes back
        is the point: a reader has to be told *which* tier went missing, and coverage claiming a
        tier the file omits must not be the sentence they get instead.
        """
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        path = ctx.artifact("elan_annotations")
        text = path.read_text(encoding="utf-8")
        start = text.index('<TIER TIER_ID="face_tracks"')
        end = text.index("</TIER>", start) + len("</TIER>")
        path.write_text(text[:start] + text[end:], encoding="utf-8")
        with pytest.raises(ValidationError) as raised:
            stage.validate(ctx)
        message = str(raised.value)
        assert "census are absent" in message, message
        assert "named by no coverage entry" not in message, message

    def test_validate_rejects_an_entry_retargeted_onto_a_tier_that_does_not_exist(
            self, stage_config, dataset) -> None:
        """The hand edit the census cannot see, caught by the one direction that is an error.

        Pointing `speech_words` at a tier called `words_that_do_not_exist` removes no census name
        (the count for `words` is still recorded) and deletes no `TIER` element, so nothing else in
        `validate` notices — while the file now says a table is represented by a tier nobody can
        select in ELAN, and no longer says which tier its real bars came from. Retargeting is the
        edit that makes a declared tier unclaimed, which is why this direction is the one the check
        enforces.
        """
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        path = ctx.artifact("elan_annotations")
        artifacts = coverage_artifacts_of(path)
        artifacts["speech_words"]["tier"] = "words_that_do_not_exist"
        rewrite_coverage(path, artifacts)
        with pytest.raises(ValidationError, match="named by no coverage entry") as raised:
            stage.validate(ctx)
        assert "words" in str(raised.value), str(raised.value)

    def test_validate_rejects_a_tier_that_no_coverage_entry_claims(
            self, stage_config, dataset) -> None:
        """The other direction: delete one entry outright.

        The tier is still declared, its bars are still there, and nothing but this check notices
        that the document no longer says which table those bars came from. The key is the artifact
        name (`active_speaker_tracks`), because the inventory is keyed by table, not by tier.
        """
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        path = ctx.artifact("elan_annotations")
        artifacts = coverage_artifacts_of(path)
        del artifacts["active_speaker_tracks"]
        rewrite_coverage(path, artifacts)
        with pytest.raises(ValidationError, match="named by no coverage"):
            stage.validate(ctx)

    def test_validate_rejects_a_state_outside_the_four_it_knows(
            self, stage_config, dataset) -> None:
        """A made-up state is not a smaller claim, it is an unreadable one.

        `not exported` — the merged word the four states exist to prevent — is the value used here,
        because that is the edit a well-meaning human reaches for.
        """
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        path = ctx.artifact("elan_annotations")
        artifacts = coverage_artifacts_of(path)
        artifacts["pose_face"]["state"] = "not exported"
        artifacts["pose_face"].pop("reason", None)
        rewrite_coverage(path, artifacts)
        with pytest.raises(ValidationError, match="not exported"):
            stage.validate(ctx)

    def test_validate_rejects_a_coverage_property_that_is_not_json(
            self, stage_config, dataset) -> None:
        """Absence is forgiven and corruption is not, and the difference is the operator's.

        A file from before the property exists is fine; a half-written property is a file whose
        claim about what it leaves out cannot be read, which is exactly what somebody needs to know
        before trusting an empty tier to mean "the clip has nothing".
        """
        import re

        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        path = ctx.artifact("elan_annotations")
        text = path.read_text(encoding="utf-8")
        replaced, count = re.subn(r'(<PROPERTY NAME="pipeline-coverage">).*?(</PROPERTY>)',
                                  lambda m: m.group(1) + "{not json" + m.group(2),
                                  text, flags=re.DOTALL)
        assert count == 1
        path.write_text(replaced, encoding="utf-8")
        with pytest.raises(ValidationError, match="pipeline-coverage is not valid JSON"):
            stage.validate(ctx)

    def test_validate_accepts_a_document_written_before_coverage_existed(
            self, stage_config, dataset) -> None:
        """The reuse gate must not destroy a usable export over a missing property.

        Deleting the property is the shape of an older file. Its bars are fine, ELAN opens it, and
        the honest response is a document that validates with one less thing to say — the same
        tolerance `_check_census` already has, for the same reason.
        """
        import re

        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        path = ctx.artifact("elan_annotations")
        text = path.read_text(encoding="utf-8")
        replaced, count = re.subn('<PROPERTY NAME="pipeline-coverage">.*?</PROPERTY>', "",
                                  text, flags=re.DOTALL)
        assert count == 1
        path.write_text(replaced, encoding="utf-8")
        assert stage.validate(ctx)["tiers"] == 5

    def test_validate_refuses_a_coverage_version_this_code_cannot_read(
            self, stage_config, dataset) -> None:
        """A future shape must not be validated as if it were this one.

        The shared reader ignores the marker and hands back whatever `artifacts` holds, so a
        document from a later export could be pronounced consistent against meanings its keys no
        longer carry. Failing it is the reversible choice: the rerun is a two-second re-export.
        """
        import json as _json
        import re

        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        path = ctx.artifact("elan_annotations")
        document = coverage_envelope_of(path)
        document["version"] = 99
        payload = _json.dumps(document, ensure_ascii=False, separators=(",", ":"))
        text = path.read_text(encoding="utf-8")
        replaced, count = re.subn(r'(<PROPERTY NAME="pipeline-coverage">).*?(</PROPERTY>)',
                                  lambda m: m.group(1) + payload + m.group(2),
                                  text, flags=re.DOTALL)
        assert count == 1
        path.write_text(replaced, encoding="utf-8")
        with pytest.raises(ValidationError, match="version 99"):
            stage.validate(ctx)

    def test_the_coverage_check_does_not_consult_the_registry(
            self, stage_config, dataset, monkeypatch) -> None:
        """A registry that grew a table cannot make a finished export fail its own check.

        The document is written first, then the registry gains an artifact — which is what enabling
        a new stage does to the tree around an export that is already on disk. The file is untouched
        and still has to validate: its consistency is a question about the file, and the tree is
        edited by rerunning producers.
        """
        import multimodal_pipeline.artifacts as artifacts_module

        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        summary = record_of(stage, ctx)
        assert len(summary["coverage"]) == 22, "the inventory stopped covering the 22 tables"
        monkeypatch.setitem(artifacts_module.ARTIFACT_LAYOUT, "eye_gaze", "gaze/eye.parquet")
        assert len([key for key, relative in artifacts_module.ARTIFACT_LAYOUT.items()
                    if relative.endswith(".parquet")]) == 23
        assert stage.validate(ctx)["tiers"] == 5

    @pytest.mark.parametrize("bad_tier", [
        pytest.param(["words"], id="list"),
        pytest.param({"name": "words"}, id="dict"),
        pytest.param(["words", "segments_src"], id="two-element-list"),
    ])
    def test_validate_rejects_a_tier_field_that_cannot_be_a_name(
            self, stage_config, dataset, bad_tier) -> None:
        """A hand-edited `tier` must fail as a refused document, never as a crash.

        `validate` is also the reuse gate, and an exception that is not `ValidationError` escapes
        both that contract and the orchestrator's handling of it. The field is read straight out of
        JSON, so nothing in the file format stops an editor from writing a list or an object where a
        name belongs — the value then reaches `claimed.add(tier)` and raises `TypeError: unhashable
        type`, which is what an operator would otherwise see instead of "this .eaf was modified
        after elan wrote it".

        The `state` sibling already handles this shape (an unhashable state is formatted into a
        message and the entry is skipped), so the check was not blind to malformed JSON in general:
        it was blind to it on the one field it puts into a set. A scalar wrong name (`7`) was
        already reported correctly, which is why the fix is about hashability and not about type.
        """
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        path = ctx.artifact("elan_annotations")
        artifacts = coverage_artifacts_of(path)
        artifacts["speech_words"]["tier"] = bad_tier
        rewrite_coverage(path, artifacts)
        with pytest.raises(ValidationError) as raised:
            stage.validate(ctx)
        message = str(raised.value)
        # Named as an unreadable claim about this artifact, and the now-unclaimed tier is still
        # reported: the edit costs the tier its provenance, and that is the fact worth printing.
        assert "speech_words" in message, message
        assert "named by no coverage entry" in message, message

    def test_an_unhashable_tier_on_a_summarised_entry_is_refused_too(
            self, stage_config, dataset) -> None:
        """The guard has to cover both claiming states, not just `exported`.

        `summarised` entries carry a tier too (on the real corpus `persons/frames.parquet` ->
        `person_tracks`), and the unhashable value reaches the same set from the same line. Fixing
        only the `exported` branch would leave this one crashing. The stage fixture writes no
        persons tables, so nothing here is `summarised` as built — measured, the list came back
        empty — so the state is written explicitly rather than hunted for, which tests the branch
        without depending on which producers the fixture happens to run.
        """
        stage = ElanStage()
        ctx = stage_context(stage_config, dataset["dir"], dataset["video"])
        stage.run(ctx)
        path = ctx.artifact("elan_annotations")
        artifacts = coverage_artifacts_of(path)
        artifacts["speech_words"]["state"] = "summarised"
        artifacts["speech_words"]["tier"] = ["words"]
        rewrite_coverage(path, artifacts)
        with pytest.raises(ValidationError) as raised:
            stage.validate(ctx)
        message = str(raised.value)
        assert "speech_words" in message, message
        assert "named by no coverage entry" in message, message

class TestOrchestratorRegistration:
    """The wiring this stage needs in `orchestrator.py`, asserted from here.

    `orchestrator.py` is outside this change's edit surface, so these three tests are the
    specification the parent's one-line registration has to satisfy — not a description of what
    is on disk today. Two of them (`STAGE_CLASSES`, `enabled_stage_names`) fail until `elan` is
    added there, and `build_stages()` raises `KeyError: 'elan'` for every CLI command until the
    first lands. They are kept rather than dropped because a stage that is in `STAGE_ORDER` but
    absent from the class map is a pipeline that cannot start, and that has to be a red test
    rather than a paragraph in a handoff.
    """

    def test_elan_is_in_the_stage_class_map(self) -> None:
        """Without this `build_stages()` raises KeyError and every CLI command dies."""
        from multimodal_pipeline.orchestrator import STAGE_CLASSES
        from multimodal_pipeline.stages.base import STAGE_ORDER

        assert set(STAGE_CLASSES) == set(STAGE_ORDER)
        assert STAGE_CLASSES["elan"] is ElanStage

    def test_the_enabled_stage_names_still_cover_the_whole_order(self, stage_config) -> None:
        """The status table's column layout is built on this equality.

        `elan` defaults to true, so it belongs in `enabled_stage_names`; a default-true stage
        missing from the map keeps a column the header numbers but never fills.
        """
        from multimodal_pipeline.orchestrator import enabled_stage_names
        from multimodal_pipeline.stages.base import STAGE_ORDER

        assert len(enabled_stage_names(stage_config)) == len(STAGE_ORDER)

    def test_disabling_it_is_reported_by_the_name_map(self, tmp_path) -> None:
        """The invariant `test_persons_config` asserts for `persons`, inverted for a
        default-true stage: the map must report the flag, or `status` calls a stage it will
        skip enabled."""
        from multimodal_pipeline.orchestrator import enabled_stage_names
        from multimodal_pipeline.stages.base import STAGE_ORDER

        config = PipelineConfig.model_validate({
            "project_root": str(tmp_path),
            "input": {"directory": str(tmp_path)},
            "output": {"directory": str(tmp_path)},
            "elan": {"enabled": False},
        })
        names = enabled_stage_names(config)
        assert "elan" not in names
        assert len(names) == len(STAGE_ORDER) - 1
