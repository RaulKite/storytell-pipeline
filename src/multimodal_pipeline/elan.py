"""Build one ELAN ``.eaf`` from the tables every other stage already wrote (T22).

Pure functions over files on disk: no stage imports, no config, no subprocess — so the
whole mapping from Parquet to EAF can be driven against a synthetic dataset directory in a
test, and ``stages/elan.py`` stays the reader, the writer and the reuse guarantee around it.

Why twelve fixed flat tiers instead of a hierarchy, and why twelve is not "one per module".
ELAN's tier structure is a parent/child relation between annotation tiers, and deriving it from
the data (one tier per speaker, one per detected face) would make the file's *shape* depend on
what happened in a clip: two datasets could then not be compared column-for-column, and a tier
rename would look like a new tier. Twelve tiers with fixed names are the contract every other
stage follows — the schema is known before the file is opened, and an absent producer is an
absent tier rather than a renamed one. What they are *not* is a module list: those twelve tiers
read twelve of the twenty-two normalised tables, because two diarizers and two fusions account
for four of them and the token, sentence, per-segment-acoustic, hand, face and normalised-pose
tables are simply not exported yet. A tier is a decision, so adding one is a change to this
list rather than a name being reused for something else.

Absence is a named state here, exactly as ``face_status`` makes it in the ASD table. Each
tier reads exactly one producer's file; when that file is not there the tier is skipped and
one line is logged naming what was missing. An empty tier would be ambiguous between "nobody
spoke", "no face was on screen" and "this engine never ran", which is the collapse the
pipeline has refused everywhere else (§17's ``face_status``, §20.2's person counts).

Four things are not obvious from reading the code — three about the format, one about what a
tier is *for*:

* **Time slots are integer milliseconds and ``start < end`` is a hard requirement.** See
  :func:`seconds_to_ms` and :func:`interval_ms`.
* **Every per-frame signal is collapsed into blocks.** The ASD, pose and acoustic tables
  are dense grids at three different rates — the ASD stage's 25 FPS working timeline, the
  source video's own PTS list, a 10 ms Praat step. One annotation per frame would put tens
  of thousands of rows in a tier ELAN cannot render, and contiguous equal-label runs are
  what an analyst reads anyway. Because the three grids differ, each block's end is extended
  by *that table's own* median step (`median_positive_step`), never by a shared constant. The
  block ELAN then stores is half-open — `start ≤ t < end`, with `end` already carrying that
  extension — so the last sampled frame is inside the block rather than on its edge.
* **A label is the only place a tier's meaning lives.** Every value printed here is a summary
  of a producer's row, and the parts of it that could be misread — which id space an id came
  from, whether a number was measured, whether a segment-level translation is a word gloss —
  are stated in the label and repeated in the document's `pipeline-tier-semantics` property,
  because a tier value gets quoted out of the file and the README does not travel with it.
* **The pose tier groups by PTS seconds, not by ``frame_number``.** See
  :func:`pose_presence_rows` — the ASD and pose grids number the same instant differently,
  and §20.2 already documents what happens when two unrelated integer id spaces are treated
  as one key.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import read_table

# ------------------------------------------------------------------ time conversion

#: ELAN time slots are integer milliseconds, so anything narrower lands on a single slot.
MIN_INTERVAL_MS = 1


class NonFiniteTimestamp(ValueError):
    """A producer wrote NaN or infinity where ELAN needs an integer millisecond.

    Named rather than caught as a bare ``ValueError`` for the reason §30 named
    ``basis_non_finite``: "the numbers are not numbers" is a different state from "the
    numbers are bad", and the caller has to be able to react to *this* one without also
    swallowing every other conversion bug. It carries no payload beyond the message — the
    tier and the count are known by the caller that catches it.
    """


def seconds_to_ms(value: Any, end: bool = False) -> int:
    """Convert seconds to the integer milliseconds ELAN's ``TIME_VALUE`` requires.

    Three rules, applied in this order:

    1. **Round half up, never truncate.** A pose timestamp of 4.169999 s is 4170 ms.
       Truncating shortens every interval by up to a millisecond, and a tier whose ends are
       systematically early is a tier whose blocks visibly do not line up with the video.
       ``math.floor(x + 0.5)`` rather than ``round``: Python's ``round`` is banker's rounding,
       so an exact half-millisecond tie would fall to the even value and 0.0005 s would become
       0 ms — a rule nobody reading ``round(ms)`` would predict, and one this function's whole
       purpose is to make predictable.
    2. **Never negative.** A negative ``TIME_VALUE`` is not representable: ELAN refuses the
       file. Rounding at t=0 and a producer that emits ``-1e-6`` are both real, so the value is
       clamped rather than costing the export.
    3. **A zero-width interval gets +1 ms at the end.** ELAN requires ``start < end`` for
       every annotation: an interval with no width cannot be selected or dragged in its grid,
       and two annotations sharing both slots are one interval. Praat emits zero-width pitch
       marks, a track can start and end on one frame, and a collapsed word alignment is legal
       input. Widening the *end* keeps the interval where it was measured; the start is never
       moved, because shifting a start forward deletes when something began.

    ``end`` exists only for rule 3, and applies to a null end too — a null is a missing
    measurement, and (0, 0) is not writable.

    ``None`` becomes 0 for a start: a null timestamp is a missing measurement, and putting the
    annotation at t=0 keeps it visible next to its siblings instead of dropping it from a tier
    the user already sees as complete.

    A NaN or infinity raises :class:`NonFiniteTimestamp` instead of being clamped to 0, which
    is the one case clamping would be a lie: a null is *known* to be missing and lands beside
    its siblings, while a NaN clamped to 0 would claim the annotation starts at second zero.
    It raises rather than returns a sentinel because every caller here is placing an
    annotation, and "this interval has no time" is only useful as something to skip.
    """
    if value is None:
        ms = 0
    else:
        seconds = float(value)
        if not math.isfinite(seconds):
            raise NonFiniteTimestamp(f"timestamp is {seconds}, not a measurable time")
        ms = int(math.floor(seconds * 1000.0 + 0.5))
    if ms < 0:
        ms = 0
    if end and ms == 0:
        return MIN_INTERVAL_MS
    return ms


def interval_ms(start: Any, end: Any) -> tuple[int, int]:
    """A ``(start_ms, end_ms)`` pair satisfying ELAN's ``start < end``, for one interval.

    Checked as a pair rather than by calling :func:`seconds_to_ms` twice, because rounding
    creates a zero-width interval the producer never reported: a 0.0004 s span rounds 0→0,
    and a span that ends 0.3 ms before its start rounds to end < start. Both are the same
    case as an honest start == end once the millisecond grid has had its way, and both must
    leave here with ``start < end``.

    A non-finite endpoint propagates :class:`seconds_to_ms`'s ``NonFiniteTimestamp``.
    """
    start_ms = seconds_to_ms(start)
    end_ms = seconds_to_ms(end, end=True)
    if end_ms <= start_ms:
        end_ms = start_ms + MIN_INTERVAL_MS
    return start_ms, end_ms


def median_positive_step(values: Sequence[Any]) -> float:
    """The sampling step of a timestamp grid: median of its positive consecutive gaps.

    Used for every per-frame tier. The median rather than the mean because these grids are
    rebuilt from a video's own PTS list: one dropped or duplicated frame makes a single gap
    twice the step, a mean inherits that error into *every* block in the tier, and the
    median ignores it.

    Returns 0.0 when there is no positive gap — an empty grid, or a single row — because
    there is no step to infer, and inventing one (0.04 s, "it is usually 25 FPS") would
    extend a block past the last sample that exists. Callers add the 0.0, which leaves the
    block ending exactly at its last measured sample.

    Duplicated timestamps are harmless: their gap is zero and zero gaps are excluded, which
    is what lets a caller pass a column with repeats rather than de-duplicating first.
    """
    ordered = sorted(float(value) for value in values if value is not None)
    gaps = sorted(later - earlier for earlier, later in zip(ordered, ordered[1:])
                  if later > earlier)
    if not gaps:
        return 0.0
    middle = len(gaps) // 2
    if len(gaps) % 2:
        return gaps[middle]
    return (gaps[middle - 1] + gaps[middle]) / 2.0


# --------------------------------------------------------------- run collapsing

def collapse_runs(values: Sequence[Any], labels: Sequence[Any]) -> list[tuple[int, int, Any]]:
    """Contiguous runs of an equal label, as ``(start_index, end_index_inclusive, label)``.

    Index-based on purpose: the caller owns the timestamps, so this stays a pure statement
    about "equal neighbours become one run" — testable without a Parquet file, a video or an
    ELAN library, and reusable by the three tiers that need it.

    The end index is the *last index of the run*, not the one after it, so the caller adds
    one grid step to reach the end of the interval rather than subtracting a frame.

    ``None`` and ``False`` are the same label. Every caller here maps a nullable boolean
    onto a two-state reading ("not speaking", "not voiced"), and leaving the two distinct
    would split a block for a difference no viewer can see.

    A non-contiguous or unsorted input is neither detected nor repaired: the sort key
    belongs to the caller, and re-sorting here would make a tier's blocks depend on a
    comparison this function cannot know the meaning of.
    """
    if len(values) != len(labels):
        raise ValueError(f"collapse_runs: {len(values)} values but {len(labels)} labels")
    if not values:
        return []
    runs: list[tuple[int, int, Any]] = []
    start = 0
    current = _label_key(labels[0])
    for index in range(1, len(values) + 1):
        if index < len(values) and _label_key(labels[index]) == current:
            continue
        runs.append((start, index - 1, current))
        start = index
        if index < len(values):
            current = _label_key(labels[index])
    return runs


def _label_key(label: Any) -> Any:
    return False if label is None else label


# ------------------------------------------------------------------- tier plumbing

@dataclass(frozen=True)
class TierInput:
    """What one tier builder may read: its own table, and the dataset it came from.

    ``dataset_dir`` is here because one tier needs a second producer's file: the ASD
    *tracks* table reports a track's endpoints but not the frame step those endpoints were
    sampled on, which lives in the ASD *frames* table. Passing the directory rather than a
    pre-read table keeps that dependency explicit at the call site instead of threading a
    sixth optional argument through every builder.
    """

    tier: str
    artifact: str
    path: Path
    dataset_dir: Path

    def rows(self, columns: Sequence[str]) -> list[dict[str, Any]]:
        """The table's rows, projected to the columns the tier actually reads.

        Projection is not a micro-optimisation: ``pose/body.parquet`` on this corpus is one
        row per person per keypoint, and the presence tier needs two of its ten columns.
        """
        return read_table(self.path, columns=list(columns)).to_pylist()

    def sorted_rows(self, columns: Sequence[str], *keys: str) -> list[dict[str, Any]]:
        """Rows ordered by the named timestamp columns, nulls last.

        Every tier is written in time order because that is how ELAN's grid is read, and
        because :func:`collapse_runs` is only correct on a sorted input. Sorting here rather
        than in each builder means a tier cannot forget to.
        """
        rows = self.rows(columns)
        rows.sort(key=lambda row: tuple(_seconds(row.get(key)) for key in keys))
        return rows


def _seconds(value: Any) -> float:
    """A timestamp usable as a sort key: None sorts first instead of raising."""
    return 0.0 if value is None else float(value)


def _flag(row: dict[str, Any], key: str) -> bool:
    """A nullable boolean column as the two-state reading it describes."""
    value = row.get(key)
    if value is True:
        return True
    if value is False or value is None:
        return False
    return bool(value)


def _text(value: Any) -> str:
    """An annotation's value: never empty, never multi-line.

    ELAN stores annotation values as text nodes and folds whitespace on its own re-save, so
    a value carrying a newline renders as a broken row and comes back different. Every
    string in a tier here is built from producer text — a transcript line, a model-emitted
    gloss, a diarizer's label — any of which can hold one, so they are flattened in one
    place rather than at each call site where one would be forgotten. An empty value becomes
    a marker rather than an empty annotation, because ELAN renders an empty row as an
    invisible sliver that reads as a rendering bug.
    """
    collapsed = " ".join(("" if value is None else str(value)).split())
    return collapsed if collapsed else EMPTY_TEXT_MARKER


def _num(value: Any, places: int) -> str:
    """A display number, rounded, and ``unknown`` when there is no number to display.

    Rounding is for the reader: the tables carry six-decimal PTS values and ``conf
    0.928214`` in a tier label says nothing more than ``conf 0.928``. ``places`` is per call
    because the quantities differ in useful precision — a confidence lives in [0,1], a
    TalkNet score is an unbounded logit.

    A null or non-finite value becomes the word ``unknown``, **not** ``0.000``. The first
    version of this function formatted ``None`` as a zero, which put a measurement in the file
    that was never taken: ``mean 0.000`` reads as "measured, and it came out zero", while the
    row's own column says no score exists. §17 built ``face_status`` and §20.2 built the person
    counts to refuse exactly that collapse, and this display helper was the last place still
    making it. A real finite zero still prints as ``0.000`` — the distinction is the whole
    point, so it has its own test.
    """
    if value is None:
        return UNKNOWN_DISPLAY
    try:
        number = float(value)
    except (TypeError, ValueError):
        return UNKNOWN_DISPLAY
    if not math.isfinite(number):
        return UNKNOWN_DISPLAY
    return f"{number:.{places}f}"


def _id(value: Any) -> str:
    """An identifier inside a label: flattened, never ``None``, never blank.

    Same argument as :func:`_num`. A producer that wrote no id used to make the tier print the
    string ``None``, which reads as a rendering bug rather than a missing value. The ids
    themselves come straight from the tables (`word_id`, `segment_id`, `face_track_id`) — this
    formats one, it never invents, renumbers or cross-walks one.
    """
    if value is None:
        return UNKNOWN_DISPLAY
    text = _text(value)
    return text if text != EMPTY_TEXT_MARKER else UNKNOWN_DISPLAY


# ------------------------------------------------------------------ display vocabulary

#: What "there is no value here" looks like inside a tier label.
#:
# It is a word and not a number for the reason the schemas give for every nullable column:
# ``0.000`` is a measurement, and an analyst cannot tell it apart from the missing one. Used
# by :func:`_num`, :func:`_id` and the ASD state below, and named in :data:`TIER_SEMANTICS`
# so the file explains the word it prints.
UNKNOWN_DISPLAY = "unknown"

#: An annotation value cannot be empty in ELAN (see :func:`_text`), so emptiness gets a marker.
EMPTY_TEXT_MARKER = "(empty)"

#: Column names, quoted in :data:`TIER_SEMANTICS` and in labels, so a reader can find the
#: column a label fragment came from. They are names of *columns*, not keys to be joined: the
#: point of naming them is that they belong to different producers' id spaces.
WORD_NS = "word_id"
SPEAKER_NS = "speaker_id"
SEGMENT_NS = "segment_id"
FACE_TRACK_NS = "face_track_id"
#: `SPEAKER_FUSION_SCHEMA.engine` — which diarizer wrote the row, and therefore which
#: `speaker_id` namespace that row's ids came from. Printed on the fusion label so a quoted
#: verdict says its own namespace; the tier name carries the same fact but does not travel.
ENGINE_NS = "engine"

#: What the tiers summarise and what they leave out, written into every document.
#:
# A tier label is the part of this file that travels — into a screenshot, an issue, a paper —
# without the README next to it, and three readings of it are wrong in ways that cost someone
# an afternoon: `gloss_en` is a *segment-level* translation although its name says "gloss"
# (which means a word-by-word gloss to anyone who has read a linguistics interlinear); the
# score on `face_tracks` is a TalkNet logit-like number and not a probability; and the id spaces
# that really are unrelated — each diarizer's `speaker_id` and YOLO's `person_id` — can print
# equal digits for different things (§20.2). The opposite error costs the same afternoon: the
# fusion row's `face_track_id` is *not* a fourth space, it is the same TalkNet `track_id` the
# `face_tracks` tier is built from (`fuse_turn_table` copies the winning frame's id through), so
# calling it unrelated would hide the one link that tier has. The clause therefore names the
# spaces that are separate and the one that is shared. Putting this in the document rather than
# only in the README is what makes the claim travel with the file. It is one property, kept to a
# few clauses, and it describes only tiers this module actually writes.
TIER_SEMANTICS: str = (
    "gloss_en = segment-level English translation of a whole segment, not word gloss. "
    f"Text tiers print the producer's own identity after the text: {WORD_NS} and {SEGMENT_NS} "
    "on words, " + SEGMENT_NS + " on segments_src and gloss_en, turn_id on turns_* and "
    "fusion_*, so rows link across tiers by those ids and by nothing else; fusion_* also print "
    f"their own {ENGINE_NS} column in parentheses, so a detached verdict says which diarizer "
    "namespace its turn speaker came from. "
    f"Namespaces: {SPEAKER_NS} (pyannote 'SPEAKER_00'), {SPEAKER_NS} (nemotron 'speaker_0') and "
    "YOLO 'person_id' are separate id spaces — equal digits name different things, never join "
    f"them. {FACE_TRACK_NS} on fusion_* is the same TalkNet 'track_id' as the face_tracks tier, "
    "so those two tiers link. "
    "Numbers: face_tracks 'mean' is a TalkNet logit-like score (unbounded), not a probability; "
    f"a missing or non-finite value prints '{UNKNOWN_DISPLAY}', never a measured 0.000. "
    "ASD states: 'no face' means nothing was located, 'not evaluated' means a face was located "
    "but never scored, 'not speaking' means the face track selected for that frame was measured "
    "and came out inactive — evidence about that track's mouth, not about the audio, so another "
    "or an off-screen speaker may still be talking in the same second. A score carried from the "
    "previous frame says 'imputed tail score' on either activity state. "
    "Blocks are half-open in milliseconds: a block covers [start, end), and the last sampled "
    "frame is inside it. "
    "Coverage: this document is a summary of the dataset's tables, not every number in them "
    "— dense per-frame signals are collapsed to runs and nothing here is a raw measurement."
)


# ------------------------------------------------------------------ tier builders

def words_rows(item: TierInput) -> list[dict[str, Any]]:
    """One annotation per aligned word: the word first, then the ids that place it.

    The word leads because this is the tier an analyst actually reads, and a row that opened
    with `seg000001-w00000` would make every other word in the file a page-turn away. The ids
    trail for a reason that is not decoration: `WORDS_SCHEMA` carries `word_id` and
    `segment_id`, and without them printed there is nothing in the .eaf that says which segment
    a word belongs to, so linking `words` to `segments_src` or `gloss_en` means eyeballing
    timestamps. The row is *not* rendered as a JSON dump of the record — the whole tier has to
    stay scannable, and only these two columns earn a place.
    """
    rows = item.sorted_rows(("start_time", "end_time", "word", "word_id", "segment_id",
                             "speaker_id"), "start_time", "end_time")
    return [{"start": row["start_time"], "end": row["end_time"],
             "text": _text(f"{row['word']} · {_id(row['speaker_id'])} · "
                           f"{_id(row['word_id'])} · [{_id(row['segment_id'])}]")}
            for row in rows]


def segments_rows(item: TierInput) -> list[dict[str, Any]]:
    """Source segments: speaker, text, and the `segment_id` the other tiers point at.

    The speaker goes through :func:`_id` rather than straight into the text, because
    `SEGMENTS_SCHEMA.speaker_id` is nullable (no diarizer ran, or the segment was never
    assigned) and a literal `None:` at the head of the label — the part an analyst quotes —
    reads as a rendering bug. It is the same missing-measurement state as a null `segment_id`
    on the same row, so it prints the same word.
    """
    rows = item.sorted_rows(("start_time", "end_time", "speaker_id", "text", "segment_id"),
                            "start_time", "end_time")
    return [{"start": row["start_time"], "end": row["end_time"],
             "text": _text(f"{_id(row['speaker_id'])}: {row['text']} · "
                           f"[{_id(row['segment_id'])}]")} for row in rows]


def translation_rows(item: TierInput) -> list[dict[str, Any]]:
    """The English side, labelled for what it is: a segment-level translation.

    The tier is called `gloss_en` and its artifact is `translation_segments`, whose schema is
    keyed by `segment_id` with one `english_text` per segment. "Gloss" in an ELAN file means a
    word-by-word gloss to most readers, and the name cannot be changed without breaking the
    tier contract every other stage follows (§20.4's argument against silent renames), so the
    semantics are carried instead: the label names the segment it translates and the speaker it
    belongs to, and :data:`TIER_SEMANTICS` says in the document that this is segment-level and
    not a word gloss.

    `translation_model` is deliberately *not* in every label: it is one value for the whole
    file (the stage writes the same model per run), so repeating it on every row would cost
    readability and buy nothing. It is in the table, and so is `translation_prompt_version`.

    The speaker is the same nullable column as on `segments_src` and prints the same way, so one
    unassigned segment does not read `unknown: …` on one tier and `None: …` on the other.
    """
    rows = item.sorted_rows(("start_time", "end_time", "english_text", "speaker_id",
                             "segment_id"), "start_time", "end_time")
    return [{"start": row["start_time"], "end": row["end_time"],
             "text": _text(f"{_id(row['speaker_id'])}: {row['english_text']} · "
                           f"[{_id(row['segment_id'])}]")} for row in rows]


def turn_rows_factory(engine: str) -> Callable[[TierInput], list[dict[str, Any]]]:
    """One builder for both diarizers' turn tables.

    ``engine`` is passed in rather than read off the filename because the two tables are
    produced by two stages and the tier name is the thing that says which is which; a
    builder that sniffed ``path.name`` would silently label Nemotron turns "pyannote" the
    week one of the two files was renamed.

    The label says ``speaker <id>`` rather than printing a bare id because pyannote's
    ``SPEAKER_00`` and Nemotron's arrival-ordered ``speaker_0`` are unrelated clusters over
    unrelated channels, and a tier label is the one part of the file that leaves it — into a
    screenshot or a quote — without the tier header that names the engine. The word is the
    namespace marker; the id itself is copied through untouched.
    """

    def build(item: TierInput) -> list[dict[str, Any]]:
        rows = item.sorted_rows(("start_time", "end_time", "speaker_id", "diarization_type",
                                 "turn_id"), "start_time", "end_time")
        return [{"start": row["start_time"], "end": row["end_time"],
                 "text": _text(f"speaker {_id(row['speaker_id'])} ({engine}, "
                               f"{_id(row['diarization_type'])}) · "
                               f"{_id(row['turn_id'])}")}
                for row in rows]

    return build


def fusion_rows(item: TierInput) -> list[dict[str, Any]]:
    """The A/V verdict, its turn's speaker and engine, the winning face track, the arithmetic.

    Five parts, because the row answers one question with numbers from two id spaces:

    * ``agreement`` — the verdict, unchanged (`fusion.AGREEMENT_STATES`);
    * ``turn <turn_id>`` — which diarizer turn the verdict is about, i.e. the one legitimate
      link between this tier and the `turns_*` tier built from the same table;
    * ``turn speaker <id> (<engine>)`` — the diarizer's label and the column that says which
      engine's namespace it came from;
    * ``face track <id>`` — the TalkNet track with the strongest claim on the turn, which is the
      *same* id space as `ACTIVE_SPEAKER_FRAMES_SCHEMA.track_id` and the `face_tracks` tier
      (`fuse_turn_table` copies the winning frame's `track_id`), so the two tiers link on it;
    * ``agreement_detail`` — the measured numbers in words, kept verbatim.

    The engine comes from the row's own `engine` column rather than from the tier name or the
    shape of the id, for the reason `turn_rows_factory` gives for its own parameter: a label is
    quoted out of the file without the tier header, and pyannote's ``SPEAKER_00`` and Nemotron's
    ``speaker_0`` are unrelated clusters. `speaker_fusion`'s validator already rejects a file
    whose rows name another engine, so on a well-formed table this prints what the tier name
    says; on a row that predates that check, or a null engine, the label prints the row's own
    value (`unknown` for a null) instead of asserting something the row does not claim.

    The speaker and face-track ids sit next to each other on purpose, *labelled*: they are
    different spaces — one diarizer cluster, one TalkNet track — and a label that printed only
    one of them would leave a reader to guess which the verdict was about. `agreement_detail`
    was the only part of the row a tier used to show, so the verdict could not be traced to
    either the voice or the face it compared.
    """
    rows = item.sorted_rows(("start_time", "end_time", "agreement", "agreement_detail",
                             "speaker_id", "face_track_id", "turn_id", "engine"),
                            "start_time", "end_time")
    return [{"start": row["start_time"], "end": row["end_time"],
             "text": _text(f"{row['agreement']}: turn {_id(row['turn_id'])} · turn speaker "
                           f"{_id(row['speaker_id'])} ({_id(row['engine'])}) | face track "
                           f"{_id(row['face_track_id'])} | {row['agreement_detail']}")}
            for row in rows]


#: The readings the ASD frames table supports, in the order they are decided.
ASD_NO_FACE = "no face"
ASD_NOT_SPEAKING = "not speaking"
#: A face was located and TalkNet never produced a score for it. Deliberately not
#: ``ASD_NOT_SPEAKING``: see :func:`asd_label`.
ASD_NOT_EVALUATED = "not evaluated"
#: Suffix for a score carried from the last real one rather than measured on this frame
#: (`frame_reason='imputed_tail'`, `score_imputed=True`). Appended to **either** activity state:
#: it is a statement about where the number came from, and the row is just as much an
#: extrapolation when the carried score landed below the threshold. See :func:`asd_label`.
ASD_IMPUTED_SUFFIX = " (imputed tail score)"

#: The two ``face_status`` values this module branches on, named rather than inlined because
#: ``tracked`` and ``tracked_unscored`` differ by a suffix and a typo is silent.
ASD_NO_FACE_STATUS = "no_face"
ASD_TRACKED_UNSCORED = "tracked_unscored"

#: `stages.activespeaker.UNSCORED_FRAME_REASONS`, restated here rather than imported.
#:
# `elan` is a leaf by design (its module docstring: no stage imports, no config, no
# subprocess) so the tier algebra stays drivable against a directory of Parquet files. The
# stage's validator is the authority on this set and its own test names the members; this one
# asserts the two copies agree.
UNSCORED_FRAME_REASONS: frozenset[str] = frozenset({
    "score_not_finite",      # the score itself was NaN/inf
    "track_has_no_scores",   # the track never produced a score at all
    "past_scored_tail",      # beyond the two frames TalkNet is allowed to carry a score for
    "tail_score_not_finite", # a carried tail score that was not a number
    "unknown",               # a row the producer could not diagnose
})


def asd_label(row: dict[str, Any]) -> str:
    """One frame's reading, decided in the order that keeps the strongest *honest* claim.

    Three states, and the third one is the reason this function has an order at all:

    1. **no face.** Checked first, whatever ``is_active_speaker`` says: a frame where no face
       was located has no track to be active, and the flag on such a row is a leftover. The
       frames table names the cause in two columns (``face_status`` for "was a face located",
       ``frame_reason`` for "why is there no score"); either of them saying no-face is enough,
       because they answer different questions and an old dataset carries only one of them.
    2. **not evaluated.** A face was located and never scored — ``face_status``
       ``tracked_unscored``, or one of the :data:`UNSCORED_FRAME_REASONS`. Calling that "not
       speaking" is the collapse §17 built ``face_status`` to prevent: it turns missing
       evidence into evidence of silence, and an off-screen or unfocusable face then reads as a
       person who said nothing. This state outranks a true ``is_active_speaker`` on purpose:
       `ActiveSpeakerStage.validate` rejects a ``tracked_unscored`` row that is imputed or
       marked active, so a row carrying both is a producer defect whose flag is stale — the
       verdict outlived the measurement it was derived from, and the measurement state wins.
    3. **speaking / not speaking.** The two readings of a row that *was* scored, straight off
       ``is_active_speaker``. An ``imputed_tail`` score is still a real verdict from the stage
       (it is the last measured score carried forward, which is why the stage allows two such
       frames), so the state is kept — but the label says where the number came from on **both**
       activity states, so nobody reads an extrapolation as a frame-level measurement. The
       suffix is about the score, not the verdict, and the corpus proves the case is not
       hypothetical: seven rows are imputed across four datasets, five active and **two** not
       (La-1 frame 60 at 2.40 s, carried score −1.4667, and ``person_demo`` frame 96 at 3.84 s).
       Dropping the provenance from those two reported a measured "no" where the producer wrote
       "carried over".

    Neither state is a claim about the *audio*: `is_active_speaker` is TalkNet's verdict on the
    one track this frame selected, so "not speaking" means that face's mouth was measured
    inactive while an off-screen or out-of-frame speaker may still be talking (the fusion
    table's `no_face_visible` verdict names that case). :data:`TIER_SEMANTICS` says so in the
    file.

    A missing ``frame_reason`` is tolerated the way the stage's own compat shim tolerates it:
    an old dataset carries ``face_status`` without a usable reason, and reading only the reason
    would call those rows "not speaking".
    """
    reason = row.get("frame_reason")
    status = row.get("face_status")
    if reason == "no_face" or status == ASD_NO_FACE_STATUS:
        return ASD_NO_FACE
    if status == ASD_TRACKED_UNSCORED or reason in UNSCORED_FRAME_REASONS:
        return ASD_NOT_EVALUATED
    imputed = bool(row.get("score_imputed")) or reason == "imputed_tail"
    if _flag(row, "is_active_speaker"):
        track = row.get("track_id")
        base = f"speaking track {track}" if track is not None else "speaking"
        return base + ASD_IMPUTED_SUFFIX if imputed else base
    return ASD_NOT_SPEAKING + ASD_IMPUTED_SUFFIX if imputed else ASD_NOT_SPEAKING



def asd_step_from(frames_path: Path) -> float:
    """The ASD working-timeline step, inferred from the frames table; 0.0 when unknowable.

    Inferred rather than read from config because the export runs over whatever is on disk:
    a dataset produced by a run with different ``activespeaker`` settings keeps the blocks
    its own frames imply. See :func:`median_positive_step` for why the median and why a
    single-row table honestly has no step.
    """
    if not frames_path.is_file():
        return 0.0
    try:
        values = read_table(frames_path, columns=["timestamp"]).column("timestamp").to_pylist()
    except Exception:  # noqa: BLE001 - an unreadable frames table yields a 0 step, not a crash
        return 0.0
    return median_positive_step(values)


def asd_speaking_rows(item: TierInput) -> list[dict[str, Any]]:
    """Per-frame ASD labels collapsed into blocks over the 25 FPS working timeline.

    `score_imputed` is read as well as the two state columns, because the imputed-tail state
    cannot be named without it (see :func:`asd_label`).
    """
    rows = item.sorted_rows(("timestamp", "is_active_speaker", "face_status",
                             "frame_reason", "track_id", "score_imputed"), "timestamp")
    if not rows:
        return []
    step = median_positive_step([row["timestamp"] for row in rows])
    stamps = [row["timestamp"] for row in rows]
    labels = [asd_label(row) for row in rows]
    return [{"start": stamps[first], "end": _shift(stamps[last], step), "text": text}
            for first, last, text in collapse_runs(stamps, labels)]


def face_track_rows(item: TierInput) -> list[dict[str, Any]]:
    """One annotation per TalkNet face track, widened by one ASD step.

    ``last_timestamp`` in the tracks table is the last frame *sampled*, so a block ending
    there stops half a frame short of the ``asd_speaking`` tier built from the same frames,
    and the two tiers would disagree about when a face left the screen. The step therefore
    comes from the frames table (:func:`asd_step_from`), and when that file is absent the
    step is 0.0 and the block ends at the last measured frame — the same rule every other
    block tier follows.
    """
    rows = item.sorted_rows(("track_id", "first_timestamp", "last_timestamp", "frame_count",
                             "active_frame_count", "mean_score"),
                            "first_timestamp", "track_id")
    step = asd_step_from(item.dataset_dir / artifact_path("active_speaker_frames"))
    return [{"start": row["first_timestamp"], "end": _shift(row["last_timestamp"], step),
             "text": _text(f"track {row['track_id']} · {row['active_frame_count']}/"
                           f"{row['frame_count']} act · mean {_num(row['mean_score'], 3)}")}
            for row in rows]


def person_track_rows(item: TierInput) -> list[dict[str, Any]]:
    """One annotation per person id.

    Not widened by a step: ``persons/frames.parquet`` is sampled from the source at the
    stage's own stride, so its step is a sampling decision rather than a frame boundary, and
    extending a sighting by one stride would claim the person was on screen in a frame that
    was never looked at. The span is the endpoints, exactly as measured.
    """
    rows = item.sorted_rows(("person_id", "first_timestamp", "last_timestamp", "frame_count",
                             "mean_confidence"), "first_timestamp", "person_id")
    return [{"start": row["first_timestamp"], "end": row["last_timestamp"],
             "text": _text(f"person {row['person_id']} · {row['frame_count']} fr · "
                           f"conf {_num(row['mean_confidence'], 3)}")} for row in rows]


#: A keypoint counts as a measured body only above this OpenPose confidence.
POSE_PRESENCE_MIN_CONFIDENCE = 0.3


def pose_presence_rows(item: TierInput) -> list[dict[str, Any]]:
    """Blocks in which at least one body keypoint was measured, grouped by PTS seconds.

    Grouping is by **seconds** (`timestamp`) and not by ``frame_number``, which is the point
    the design made about two grids that disagree. It is worth being precise about which
    column carries those seconds, because ``pose/body.parquet`` has no ``source_timestamp``:
    BODY_SCHEMA's PTS column is called ``timestamp`` and it is filled from
    ``source/frame_index.parquet``'s ``pts_seconds`` (see ``OpenPoseStage.frame_timings``),
    i.e. it *is* the source presentation time. The ASD table is the one with two columns —
    ``timestamp`` (its synthetic 25 FPS working axis) and ``source_timestamp`` (nearest real
    PTS) — so a builder that asked the pose table for ``source_timestamp`` would be copying
    the ASD naming onto a table that never had it, and would fail the tier on every video.

    Why seconds rather than the frame number matters regardless of the name: pose is timed to
    the source's own PTS list (29.97 fps here) while ASD's ``frame_number`` is its resampled
    25 FPS axis, so the same integer names two different instants. Joining on it is the
    mistake §20.2 names for ``person_id`` versus ``track_id`` — two unrelated small integers
    that look like a key. ``manifest.temporal_model`` declares seconds the pipeline's one
    timeline, so seconds are what this tier groups on.

    ``confidence`` is per keypoint, so a timestamp is "present" when any of its keypoints
    cleared ``POSE_PRESENCE_MIN_CONFIDENCE``; blocks are then runs over the *whole* distinct
    timestamp grid, so a timestamp where nothing cleared the floor is a hole in coverage
    rather than a gap the block quietly bridges.
    """
    rows = item.rows(("timestamp", "confidence"))
    stamps = sorted({row["timestamp"] for row in rows if row["timestamp"] is not None})
    if not stamps:
        return []
    step = median_positive_step(stamps)
    present = {row["timestamp"] for row in rows
               if row["timestamp"] is not None
               and row["confidence"] is not None
               and float(row["confidence"]) >= POSE_PRESENCE_MIN_CONFIDENCE}
    return [{"start": stamps[first], "end": _shift(stamps[last], step),
             "text": "body present"}
            for first, last, seen in collapse_runs(stamps, [s in present for s in stamps])
            if seen]


def voiced_rows(item: TierInput) -> list[dict[str, Any]]:
    """Contiguous voiced runs from the Praat frame table, on its own step.

    Voiced means ``f0_hz`` is not null, which is the column the stage writes when Parselmouth
    returned a pitch; ``voiced`` is deliberately not read, because a pitch period and the
    stage's voicing flag are two different judgements and this tier claims the first one
    (its text says "voiced (f0)", so the reading is stated in the file).
    """
    rows = item.sorted_rows(("timestamp", "f0_hz"), "timestamp")
    if not rows:
        return []
    step = median_positive_step([row["timestamp"] for row in rows])
    stamps = [row["timestamp"] for row in rows]
    flags = [row["f0_hz"] is not None for row in rows]
    return [{"start": stamps[first], "end": _shift(stamps[last], step),
             "text": "voiced (f0)"}
            for first, last, voiced in collapse_runs(stamps, flags) if voiced]


def _shift(value: Any, step: float) -> Any:
    """Extend an interval's end by one grid step, leaving None as None.

    ``None`` stays ``None`` so :func:`interval_ms` still sees "this producer gave us no end"
    and applies the zero-width rule, rather than receiving a 0.0 that would pull the end of
    a late interval back to the start of the video.
    """
    return None if value is None else float(value) + step


#: Registry artifact key -> dataset-relative path.
def artifact_path(artifact: str) -> str:
    """The dataset-relative path of a registry artifact, read from the registry each call.

    Two reasons not to snapshot ``ARTIFACT_LAYOUT`` into a module constant at import time:
    an artifact path is defined in exactly one place and copying it here is the second copy
    that eventually disagrees; and a test that repoints the registry must not find this
    module still holding the old path. ``artifacts`` is a leaf module (it imports nothing
    from the package), so importing it lazily costs nothing and keeps this file free of any
    stage or config dependency.
    """
    from .artifacts import ARTIFACT_LAYOUT

    return ARTIFACT_LAYOUT[artifact]


@dataclass(frozen=True)
class TierSpec:
    tier: str
    artifact: str
    build: Callable[[TierInput], list[dict[str, Any]]]
    #: The flat-text tiers a reviewer reads first, so a tier list is checkable by name.
    note: str = ""


#: The twelve tiers, in the order they are written. Declared here so the tier set is one
#: list a reviewer can count and a test can assert against, rather than twelve calls
#: scattered through a build function.
TIERS: tuple[TierSpec, ...] = (
    TierSpec("words", "speech_words", words_rows, "word-level transcript"),
    TierSpec("segments_src", "speech_segments", segments_rows, "source-text segments"),
    TierSpec("gloss_en", "translation_segments", translation_rows, "English gloss"),
    TierSpec("turns_pyannote", "speaker_turns", turn_rows_factory("pyannote"), "engine 1 turns"),
    TierSpec("turns_nemotron", "speaker_turns_nemotron", turn_rows_factory("nemotron"),
             "engine 2 turns"),
    TierSpec("fusion_pyannote", "speaker_fusion_pyannote", fusion_rows, "engine 1 A/V verdict"),
    TierSpec("fusion_nemotron", "speaker_fusion_nemotron", fusion_rows, "engine 2 A/V verdict"),
    TierSpec("asd_speaking", "active_speaker_frames", asd_speaking_rows, "ASD blocks"),
    TierSpec("face_tracks", "active_speaker_tracks", face_track_rows, "TalkNet tracks"),
    TierSpec("person_tracks", "person_tracks", person_track_rows, "YOLO person tracks"),
    TierSpec("pose_presence", "pose_body", pose_presence_rows, "body-present blocks"),
    TierSpec("voiced_blocks", "acoustic_frames", voiced_rows, "voiced blocks"),
)

#: Mimetype by suffix. pympi's own guess table covers wav/mpg/mpeg/xml and nothing else, so
#: leaving an .mp4 to the library raises KeyError; and ELAN will not open a linked file whose
#: MIME type it has no player for, so this is the value it expects rather than a decoration.
MIME_TYPES: dict[str, str] = {
    ".mp4": "video/mp4",
    ".m4v": "video/x-m4v",
    ".mov": "video/quicktime",
}

#: The suffix pympi would refuse (``KeyError``) and whose container we cannot know from the
#: name. The empty string is what goes into the descriptor: ``MIME_TYPE`` is an optional
#: attribute of ``MEDIA_DESCRIPTOR``, an empty one serialises and round-trips (measured with
#: pympi 1.7: ``MIME_TYPE=""`` written and read back), and ELAN keys playback off the file
#: extension regardless. What matters is that we do not **name** a container we did not
#: measure. The corpus proved this was not hypothetical: its ``person_demo.avi`` really is a
#: QuickTime container, so a catch-all of ``video/mp4`` would have described it falsely on
#: the one machine the README example walks.
UNKNOWN_MIME_TYPE = ""


def mimetype_for(video_path: Path) -> str:
    """The declared container for ``video_path``, or "" when the suffix does not say.

    Correct-but-absent beats wrong-and-present: a wrong type tells ELAN (and any human
    reading the XML) a container the file may not have, and the file plays off its
    extension either way.
    """
    return MIME_TYPES.get(Path(video_path).suffix.lower(), UNKNOWN_MIME_TYPE)


def eaf_directory() -> str:
    """The dataset-relative directory the .eaf is written into, read from the registry.

    One line, and it is the base the media descriptor's relative URL is measured from — ELAN
    resolves that URL against the directory holding the .eaf, so the writer cannot assume the
    dataset directory is the starting point.

    It is derived from ``ARTIFACT_LAYOUT`` rather than written as ``"elan"`` so the base and the
    destination move together: a hardcoded base plus a moved artifact leaves every existing file
    with a link that no longer resolves, which is the bug this function was extracted to fix.
    """
    from .artifacts import ARTIFACT_LAYOUT

    return Path(ARTIFACT_LAYOUT["elan_annotations"]).parent.as_posix()


def add_media_descriptor(eaf: Any, *, dataset_dir: Path, video_path: Path) -> dict[str, str]:
    """Link the source video so ELAN opens it in place *and* the .eaf survives a move.

    Both URLs go into the one ``MEDIA_DESCRIPTOR``, because ELAN uses both and neither alone
    is enough:

    * ``MEDIA_URL`` — an absolute ``file://`` URL. Without it ELAN cannot find the media on a
      fresh open where the .eaf sits in place, which is the case that matters on this machine.
    * ``RELATIVE_MEDIA_URL`` — the path from **the directory the .eaf itself sits in** to the
      video, with forward slashes (ELAN's convention on every platform). Without it the .eaf is
      welded to one absolute location: copy ``data/processed/`` to another disk, or move one
      dataset next to its videos, and every link breaks although the two files are beside each
      other.

    The base is the .eaf's own directory (``eaf_directory()``, i.e. ``<dataset>/elan``) and not
    the dataset directory, because that is what ELAN documents and does: the manual has it
    search "the same directory the .eaf file is in", and the format's own examples carry
    ``RELATIVE_MEDIA_URL="../../audio.wav"`` for a file two levels up from the annotation file.
    A path from ``<dataset>`` instead of ``<dataset>/elan`` is short one ``../``: it parses, it
    round-trips, and it resolves to a sibling of the dataset directory that has never existed —
    the first version of this function did exactly that, and the stage's own reachability check
    agreed with it because it resolved the same wrong way from the same wrong base.

    ``os.path.relpath`` and not ``Path.relative_to``, because the video normally lives
    *outside* the dataset directory (``data/input_videos/`` beside ``data/processed/``) and
    ``relative_to`` refuses to express a path that leaves its base. The ``ValueError`` guard
    is for Windows, where a path on another drive has no relative form at all: that costs the
    relative URL and never the export.

    ``time_origin=0`` because every tier in this file is already expressed in seconds from
    the video's start — the pipeline's one timeline — so there is no offset to declare.
    """
    video_path = Path(video_path)
    absolute = video_path.resolve().as_uri()
    eaf_dir = Path(dataset_dir).resolve() / eaf_directory()
    try:
        relpath = Path(os.path.relpath(str(video_path.resolve()),
                                       str(eaf_dir))).as_posix()
    except ValueError:  # pragma: no cover - different drive on Windows
        relpath = ""
    mimetype = mimetype_for(video_path)
    eaf.add_linked_file(file_path=absolute, relpath=relpath or None, mimetype=mimetype,
                        time_origin=0)
    return {"media_url": absolute, "relative_media_url": relpath, "mimetype": mimetype}


def build_eaf(dataset_dir: Path, video_path: Path, log: Callable[..., None] = print) -> Any:
    """Read a dataset's tables and return a populated :class:`pympi.Elan.Eaf`.

    The stage writes the file; this function only builds it, so the whole mapping runs in a
    test with no stage, no config and no output directory.

    A tier whose input file is absent is skipped with **one** logged line naming the file.
    Absent is the normal state for most of these tables — Nemotron may never have run,
    translation has no endpoint by default, ``persons`` ships disabled, OpenPose is its own GPU
    stage — and the pipeline's rule is that an absent artifact is reported rather than
    promised (``manifest.artifacts_not_generated``). The .eaf follows that rule.

    A file that exists but cannot be read is skipped too, with its exception named. That is a
    deliberate asymmetry with the stage's own ``validate``: one corrupt table should cost its
    own tier and not the eleven that were already built correctly, and a .eaf with eleven
    tiers and one logged line is worth more to a user than no .eaf at all.

    One row with a non-finite timestamp costs **that row**, not its tier. Placing an
    annotation needs an integer millisecond, and a NaN in a timestamp column is producible by
    an upstream stage that wrote a division it never checked; the tier's other rows were
    measured and belong in the file. The count is logged, so a tier that dropped half its rows
    says so in the run output — the difference between "this clip has few words" and "the
    words table is full of NaN" stays readable.

    The tier census is written as a document property, so a reader of the file alone can tell
    "this clip has no person tier because ``persons`` was off" from "the export lost it" —
    the same argument the manifest makes in JSON.
    """
    from pympi.Elan import Eaf

    dataset_dir = Path(dataset_dir)
    eaf = Eaf(author="multimodal-pipeline", suppress_version_warning=True)
    built: dict[str, int] = {}
    skipped: dict[str, str] = {}

    for spec in TIERS:
        relative = artifact_path(spec.artifact)
        path = dataset_dir / relative
        if not path.is_file():
            reason = f"{relative} not produced"
        else:
            reason = ""
            try:
                rows = spec.build(TierInput(tier=spec.tier, artifact=spec.artifact,
                                            path=path, dataset_dir=dataset_dir))
            except Exception as exc:  # noqa: BLE001 - one bad table must not lose eleven
                reason = f"{path.name} unreadable ({type(exc).__name__}: {exc})"
                rows = []
            if not reason:
                eaf.add_tier(tier_id=spec.tier)
                dropped = 0
                for row in rows:
                    try:
                        start_ms, end_ms = interval_ms(row["start"], row["end"])
                    except NonFiniteTimestamp:
                        dropped += 1
                        continue
                    eaf.add_annotation(spec.tier, start_ms, end_ms, _text(row["text"]))
                built[spec.tier] = len(rows) - dropped
                if dropped:
                    log(f"elan: tier {spec.tier} dropped {dropped} of {len(rows)} "
                        f"annotation(s) with a non-finite timestamp")
        if reason:
            skipped[spec.tier] = reason
            log(f"elan: tier {spec.tier} skipped ({reason})")

    media = add_media_descriptor(eaf, dataset_dir=dataset_dir, video_path=video_path)
    census = " ".join(f"{name}={count}" for name, count in sorted(built.items()))
    eaf.add_property("pipeline-tiers", census or "none")
    # How to *read* the tiers, in the file itself. The census says what is in the document; this
    # says what the labels mean, because a tier value is the part of an .eaf that leaves it
    # (see :data:`TIER_SEMANTICS`). One property, fixed text, no per-dataset content — so it
    # costs nothing and cannot drift from a clip.
    eaf.add_property("pipeline-tier-semantics", TIER_SEMANTICS)
    eaf.add_property("pipeline-media", f"{media['media_url']} | {media['relative_media_url']}")
    log(f"elan: {len(built)} tier(s), {sum(built.values())} annotation(s) for "
        f"{Path(video_path).name}; skipped {len(skipped)} "
        f"({', '.join(sorted(skipped)) or 'none'})")
    return eaf


def tier_counts(eaf: Any) -> dict[str, int]:
    """Annotations per tier of a built Eaf, excluding pympi's implicit ``default`` tier.

    ``Eaf.tiers[name]`` is pympi's four-part tuple — ``(annotations, ref_annotations,
    tier_dict, tier_type)`` — so the count is element 0, not ``len`` of the tuple (which is
    always 4 and would report a confident, wrong number in every provenance record).

    ``default`` is excluded because it is not one of this module's tiers: pympi creates it for
    every document and nothing writes into it. Leaving it in would report thirteen tiers for
    twelve, and a count off by one is the kind of claim a reader believes.
    """
    counts: dict[str, int] = {}
    for name, data in eaf.tiers.items():
        if name == "default":
            continue
        annotations = data[0] if isinstance(data, tuple) else data
        counts[name] = len(annotations)
    return counts
