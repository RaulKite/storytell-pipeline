"""Derived tables that no longer agree with the transcript they were derived from.

``speaker_assignment`` is the only stage that decides what ``speaker_id`` means. It labels
``speech/segments.parquet`` and ``speech/words.parquet`` in place, and every table built from
those two copies the label into its own rows: the translation carries the speaker of the
segment it translated, the linguistic tables carry the speaker of the token's segment, and
``acoustic/segment_features.parquet`` carries the speaker of the interval it measured. When
the diarizer changes — a different engine, a rerun with a different speaker count — those
*tables* keep describing the previous run while the transcript describes the current one. Each
one validates on its own, the manifest promises both, and a consumer that joins them gets a
segment labelled ``speaker_0`` in one table and ``SPEAKER_00`` in the other.

The columns cannot be recomputed here: a translation's speaker is a fact about the run that
produced it, and rewriting the column would be inventing provenance for a table this module
never wrote. So the answer is to *say* the table is stale — the manifest keeps it out of the
``artifacts`` block it promises, ``validate`` reports it, and the ELAN export declines to
publish the tier it would otherwise have built from it — and leave the file where it is for
the stage that can regenerate it.

Two properties make this one rule for six tables instead of six special cases:

* every stale candidate is joined back to the transcript by a **column of its own rows** that
  points at a segment (today ``segment_id`` for all of them), so "does this table still agree"
  is one comparison of two id→speaker maps;
* the transcript is the reference, never the reverse. ``speech_segments`` is what
  ``speaker_assignment`` writes, so it is the current truth even when it disagrees with
  everything else.

The scan reads two columns per table because the tables are wide (a token table can hold
hundreds of thousands of rows) and a projected read costs one column's pages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: The artifact every derived table is compared against. Named rather than inlined because the
#: direction of the comparison is the whole claim: this one is current, the others may not be.
TRANSCRIPT_ARTIFACT = "speech_segments"

#: Derived artifacts that copy ``speaker_id`` from a transcript segment, with the column that
#: joins their rows back to it. Ordered as the pipeline writes them so a reported list is stable.
#:
#: Membership rule, so a future table has a testable home: the artifact has a ``speaker_id``
#: column *and* derives that column from a transcript row rather than from a diarization turn.
#: ``speaker_turns_nemotron`` is excluded by the second half — its speaker ids come from the
#: diarizer itself, so it is the names the transcript is labelled with, not a copy of them.
SPEAKER_DERIVED_TABLES: tuple[tuple[str, str], ...] = (
    ("translation_segments", "segment_id"),
    ("spacy_source_tokens", "segment_id"),
    ("spacy_source_sentences", "segment_id"),
    ("spacy_english_tokens", "segment_id"),
    ("spacy_english_sentences", "segment_id"),
    ("acoustic_segments", "segment_id"),
)


def speaker_staleness(dataset_dir: Path) -> dict[str, str]:
    """Stale derived artifacts, as ``{artifact: one-sentence reason}``.

    An absent file is not stale — it is absent, which the manifest already reports as
    ``not_generated`` and the ELAN export as ``absent``. A file that cannot be read is not
    stale either: its own stage reports that, and calling it stale here would add a second,
    weaker explanation of the same problem.

    A table whose ``segment_id`` no longer resolves — a re-transcription that renumbered
    segments — is not reported *here*. Nothing about its speaker labels can be judged against
    a segment map it does not join to, and ``finalization`` already refuses a translation with
    rows for segments the transcript does not have.
    """
    root = Path(dataset_dir)
    source = _speaker_map(root / _relative(TRANSCRIPT_ARTIFACT), "segment_id")
    if not source:
        # No transcript, or a transcript with no segments: no derived row joins to anything, so
        # nothing can be said to disagree with it. A *cleared* transcript is not this case — its
        # segments are present and map to ``None``, which is a comparison like any other.
        return {}
    stale: dict[str, str] = {}
    for artifact, join_key in SPEAKER_DERIVED_TABLES:
        path = root / _relative(artifact)
        if not path.is_file():
            continue
        reason = _mismatch(path, join_key, source, artifact)
        if reason:
            stale[artifact] = reason
    return stale


def _mismatch(path: Path, join_key: str, source: dict[str, Any], artifact: str) -> str | None:
    """Why ``path``'s speaker labels contradict the transcript, or ``None`` when they do not.

    Compared row by row on the join key, and only for rows whose key the transcript still has.
    After ``speaker_assignment`` cleared the transcript (no diarization timeline), ``source``
    maps every segment to ``None``; a derived table still naming ``SPEAKER_00`` for one of those
    segments is exactly as stale as one naming a changed speaker id, so both sides are compared
    including the null.
    """
    try:
        derived = _read_pairs(path, join_key)
    except Exception:  # noqa: BLE001 - an unreadable table is reported by its own stage
        return None
    if derived is None:
        return None
    disagreeing = sorted({key for key, speaker in derived
                          if key in source and source[key] != speaker})
    if not disagreeing:
        return None
    owner = owner_stage(artifact)
    return (f"speaker labels disagree with the transcript for {len(disagreeing)} row(s), "
            f"e.g. {disagreeing[:3]}; re-run the {owner} stage" if owner else
            f"speaker labels disagree with the transcript for {len(disagreeing)} row(s), "
            f"e.g. {disagreeing[:3]}")


def owner_stage(artifact: str) -> str | None:
    """The stage whose declared outputs include ``artifact``, or ``None``.

    Read from the stage classes rather than a second hand-kept list: the reason a reader acts on
    is "re-run the translation stage", and that name has to be the stage that actually writes
    the file. Imported lazily because the orchestrator imports the stages, which import this
    module's callers.
    """
    try:
        from .orchestrator import STAGE_CLASSES
    except Exception:  # noqa: BLE001 - a reason without a stage name is still a reason
        return None
    for name, cls in STAGE_CLASSES.items():
        if artifact in getattr(cls, "outputs", ()):
            return name
    return None


def _speaker_map(path: Path, join_key: str) -> dict[Any, Any]:
    """``{join key: speaker}`` from the transcript, or ``{}`` when it is absent or unreadable."""
    if not path.is_file():
        return {}
    rows = _read_pairs(path, join_key)
    if not rows:
        return {}
    return dict(rows)


def _read_pairs(path: Path, join_key: str) -> list[tuple[Any, Any]] | None:
    """``(join key, speaker)`` per row, or ``None`` when the table cannot make the claim.

    ``None`` for a table without both columns, as distinct from an empty list: a table written
    before this column existed says nothing about speakers, and reporting it would be inventing
    a disagreement out of a schema difference. The column check reads the schema, because
    projecting an absent column raises rather than answering.
    """
    import pyarrow.parquet as pq

    from .schemas import read_table

    try:
        columns = set(pq.read_schema(path).names)
        if join_key not in columns or "speaker_id" not in columns:
            return None
        table = read_table(path, columns=[join_key, "speaker_id"])
    except Exception:  # noqa: BLE001 - an unreadable table is reported by its own stage
        return None
    return [(row[join_key], row["speaker_id"]) for row in table.to_pylist()
            if row.get(join_key) is not None]


def _relative(artifact: str) -> str:
    from .artifacts import ARTIFACT_LAYOUT

    return ARTIFACT_LAYOUT[artifact]
