"""Stale derived tables: the one rule that keeps every speaker label honest.

The rule is the fix for a defect reproduced on corpus copies of ``La-1``: after
``speaker_assignment`` relabelled the transcript from the Nemotron timeline, ``segments_src``
read ``speaker_0``/``speaker_1`` while ``gloss_en`` and the four spaCy tables still read
pyannote's ``SPEAKER_00``. Every table validated on its own, the manifest promised all of them,
and the .eaf published both labels for the same segment. Only the translation was checked;
these tests pin the generalisation, table by table.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from multimodal_pipeline.artifacts import ARTIFACT_LAYOUT
from multimodal_pipeline.schemas import (
    ACOUSTIC_SEGMENTS_SCHEMA,
    SEGMENTS_SCHEMA,
    SENTENCES_SCHEMA,
    SPEAKER_TURNS_NEMOTRON_SCHEMA,
    TOKENS_SCHEMA,
    TRANSLATION_SCHEMA,
    WORDS_SCHEMA,
)
from multimodal_pipeline.staleness import (
    SPEAKER_DERIVED_TABLES,
    TRANSCRIPT_ARTIFACT,
    owner_stage,
    speaker_staleness,
)

#: Every artifact in the registry whose schema carries a ``speaker_id``. Listed with its schema
#: so the guard below can derive "which artifacts have a speaker column" from the schemas rather
#: than from this module's own list.
SCHEMAS_WITH_SPEAKER = {
    "speech_segments": SEGMENTS_SCHEMA,
    "speech_words": WORDS_SCHEMA,
    "speaker_turns_nemotron": SPEAKER_TURNS_NEMOTRON_SCHEMA,
    "translation_segments": TRANSLATION_SCHEMA,
    "acoustic_segments": ACOUSTIC_SEGMENTS_SCHEMA,
    "spacy_source_tokens": TOKENS_SCHEMA,
    "spacy_source_sentences": SENTENCES_SCHEMA,
    "spacy_english_tokens": TOKENS_SCHEMA,
    "spacy_english_sentences": SENTENCES_SCHEMA,
}


def write_rows(root: Path, artifact: str, rows: list[dict], schema: pa.Schema | None = None) -> Path:
    path = root / ARTIFACT_LAYOUT[artifact]
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema) if schema else pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    return path


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """A dataset directory with only a labelled transcript in it."""
    root = tmp_path / "processed" / "clip"
    write_rows(root, TRANSCRIPT_ARTIFACT, [
        {"segment_id": "seg1", "start_time": 0.0, "end_time": 2.0, "speaker_id": "speaker_0"},
        {"segment_id": "seg2", "start_time": 2.0, "end_time": 4.0, "speaker_id": "speaker_1"},
    ], SEGMENTS_SCHEMA)
    return root


class TestTheTableList:
    def test_every_table_that_carries_a_speaker_is_either_the_source_or_checked(self) -> None:
        """A new speaker-bearing table must be declared here or explicitly excluded.

        The defect this module fixes was not a wrong comparison — it was a list with one entry.
        So "which tables carry a speaker label" is asked of the schemas, and any table that has
        one has to appear in :data:`SPEAKER_DERIVED_TABLES` or in the transcript set that is the
        reference rather than a copy.
        """
        reference = {TRANSCRIPT_ARTIFACT, "speech_words"}
        checked = {artifact for artifact, _ in SPEAKER_DERIVED_TABLES}
        unchecked = set(SCHEMAS_WITH_SPEAKER) - reference - checked - {"speaker_turns_nemotron"}
        assert unchecked == set(), (
            "these tables carry a speaker_id and are neither the reference transcript nor "
            f"checked for staleness: {sorted(unchecked)}")

    def test_the_diarization_table_is_not_treated_as_a_copy(self) -> None:
        """Nemotron's turns are where the labels *come from*; comparing them would invert the rule."""
        checked = {artifact for artifact, _ in SPEAKER_DERIVED_TABLES}
        assert "speaker_turns_nemotron" not in checked

    @pytest.mark.parametrize("artifact,join_key", SPEAKER_DERIVED_TABLES)
    def test_every_entry_joins_on_a_column_its_table_has(self, artifact, join_key) -> None:
        """The join key has to be a real column, or the check silently reports nothing."""
        schema = SCHEMAS_WITH_SPEAKER[artifact]
        assert join_key in schema.names
        assert "speaker_id" in schema.names


class TestDetection:
    def test_a_matching_table_is_not_stale(self, dataset: Path) -> None:
        write_rows(dataset, "translation_segments", [
            {"segment_id": "seg1", "speaker_id": "speaker_0", "english_text": "hello"},
            {"segment_id": "seg2", "speaker_id": "speaker_1", "english_text": "bye"},
        ])
        assert speaker_staleness(dataset) == {}

    @pytest.mark.parametrize("artifact", [artifact for artifact, _ in SPEAKER_DERIVED_TABLES])
    def test_a_relabelled_transcript_makes_every_derived_table_stale(
            self, dataset: Path, artifact: str) -> None:
        """One rule for all six, so a table cannot be forgotten again."""
        write_rows(dataset, artifact, [
            {"segment_id": "seg1", "speaker_id": "SPEAKER_00"},
            {"segment_id": "seg2", "speaker_id": "SPEAKER_01"},
        ])
        stale = speaker_staleness(dataset)
        assert artifact in stale
        assert "speaker labels disagree with the transcript" in stale[artifact]
        assert "seg1" in stale[artifact]

    def test_one_disagreeing_row_is_enough(self, dataset: Path) -> None:
        write_rows(dataset, "spacy_source_tokens", [
            {"segment_id": "seg1", "speaker_id": "speaker_0"},
            {"segment_id": "seg2", "speaker_id": "SPEAKER_00"},
        ])
        stale = speaker_staleness(dataset)
        assert list(stale) == ["spacy_source_tokens"]
        assert "1 row(s)" in stale["spacy_source_tokens"]
        # The agreeing row is not named: the reason points at what is wrong, not at the table.
        assert "seg1" not in stale["spacy_source_tokens"]

    def test_a_cleared_transcript_makes_a_labelled_table_stale(self, dataset: Path) -> None:
        """No diarization means null labels; a derived table still naming a speaker is stale."""
        write_rows(dataset, TRANSCRIPT_ARTIFACT, [
            {"segment_id": "seg1", "speaker_id": None},
            {"segment_id": "seg2", "speaker_id": None},
        ], SEGMENTS_SCHEMA)
        write_rows(dataset, "translation_segments", [
            {"segment_id": "seg1", "speaker_id": "SPEAKER_00"},
        ])
        assert "translation_segments" in speaker_staleness(dataset)

    def test_absent_tables_are_not_stale(self, dataset: Path) -> None:
        """Absence is the manifest's `not_generated`, never a disagreement."""
        assert speaker_staleness(dataset) == {}

    def test_an_unreadable_table_is_left_to_its_own_stage(self, dataset: Path) -> None:
        path = dataset / ARTIFACT_LAYOUT["translation_segments"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not parquet")
        assert speaker_staleness(dataset) == {}

    def test_a_table_without_the_speaker_column_says_nothing(self, dataset: Path) -> None:
        """A schema written before the column existed is not a disagreement."""
        write_rows(dataset, "spacy_english_tokens", [{"segment_id": "seg1", "token": "hello"}])
        assert speaker_staleness(dataset) == {}

    def test_rows_for_segments_the_transcript_no_longer_has_are_not_compared(
            self, dataset: Path) -> None:
        """A re-transcription renumbers segments; that is `finalization`'s own finding."""
        write_rows(dataset, "translation_segments", [
            {"segment_id": "old-1", "speaker_id": "SPEAKER_00"},
        ])
        assert speaker_staleness(dataset) == {}

    def test_no_transcript_means_nothing_to_compare(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        write_rows(root, "translation_segments", [{"segment_id": "seg1", "speaker_id": "SPEAKER_00"}])
        assert speaker_staleness(root) == {}


class TestTheReason:
    def test_the_reason_names_the_stage_that_can_fix_it(self, dataset: Path) -> None:
        write_rows(dataset, "translation_segments", [{"segment_id": "seg1", "speaker_id": "SPEAKER_00"}])
        write_rows(dataset, "spacy_english_sentences", [{"segment_id": "seg1", "speaker_id": "SPEAKER_00"}])
        stale = speaker_staleness(dataset)
        assert stale["translation_segments"].endswith("re-run the translation stage")
        assert stale["spacy_english_sentences"].endswith("re-run the spacy_english stage")

    def test_owner_lookup_covers_exactly_what_the_stages_declare(self) -> None:
        from multimodal_pipeline.orchestrator import STAGE_CLASSES

        for artifact, _ in SPEAKER_DERIVED_TABLES:
            owner = owner_stage(artifact)
            assert owner in STAGE_CLASSES
            assert artifact in STAGE_CLASSES[owner].outputs
