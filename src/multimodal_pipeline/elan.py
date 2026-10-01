"""Build one ELAN ``.eaf`` from the tables every other stage already wrote (T22).

Pure functions over files on disk: no stage imports, no config, no subprocess — so the
whole mapping from Parquet to EAF can be driven against a synthetic dataset directory in a
test, and ``stages/elan.py`` stays the reader, the writer and the reuse guarantee around it.

Why seventeen fixed flat tiers instead of a hierarchy, and why seventeen is not "one per module".
ELAN's tier structure is a parent/child relation between annotation tiers, and deriving it from
the data (one tier per speaker, one per detected face) would make the file's *shape* depend on
what happened in a clip: two datasets could then not be compared column-for-column, and a tier
rename would look like a new tier. Seventeen tiers with fixed names are the contract every other
stage follows — the schema is known before the file is opened, and an absent producer is an
absent tier rather than a renamed one. What they are *not* is a module list: those seventeen tiers
read nineteen of the twenty-two normalised tables, because two diarizers and two fusions account
for four of them, ``person_tracks`` reads two more as **support** for its sightings (the per-frame
detections and the clip's frame list place them; neither has a tier of its own, and neither is an
exported analysis), and the hand, face and normalised-pose tables get no tier. The four spaCy
tables (source and english, tokens and sentences) do have tiers of their own
and they are flat peers, not a parent/child chain: see :func:`spacy_token_rows` for why the
sentence tier is not the token tier's parent and why no tier is created per token. The last tier
added, ``acoustic_segments``, is the per-segment acoustic aggregate — one flat peer, not a parent
of the frame-based ``voiced_blocks`` tier, because the two tables answer different questions and
a hierarchy would make the file's shape depend on which of the two a run produced. Which tables
still belong on a coverage list is settled with the corpus refresh, not here. A tier is a
decision, so adding one is a change to this list rather than a name being reused for something
else.

Absence is a named state here, exactly as ``face_status`` makes it in the ASD table. Each tier
reads one producer's file — one, plus whatever :data:`SECONDARY_INPUTS` declares for it — and
when a file it needs is not there the tier is skipped and one line is logged naming what was
missing. An empty tier would be ambiguous between "nobody
spoke", "no face was on screen" and "this engine never ran", which is the collapse the
pipeline has refused everywhere else (§17's ``face_status``, §20.2's person counts).

What that rule does *not* answer is the other half of the question, and the half an operator
cannot recover from an opened file: "this clip has no person data" and "this export never
represents pose" both leave a tier missing. :data:`COVERAGE_PROPERTY` answers it by naming every
normalised Parquet artifact in the registry and giving it exactly one of four states — exported,
summarised, present and not exported, absent — derived from :data:`ARTIFACT_LAYOUT`,
:data:`TIERS` and :data:`SECONDARY_INPUTS` plus one stat() per file. So the inventory is a
function of the same objects this function iterates, and a future artifact appears in it without
anyone editing a list.

Nine things are not obvious from reading the code — four about the format, one about what a
tier is *for*, one about what a person tier is entitled to claim, one about what a linguistic
tier's bar is entitled to claim, one about what a tier of numbers is entitled to claim, one
about what an inventory of a file's own contents is entitled to claim:

* **Time slots are integer milliseconds, ``start < end`` is a hard requirement, and a row with no
  usable time is not exported at all.** See :func:`seconds_to_ms` and :func:`interval_ms`. ELAN has
  no "time unknown" annotation, so a null endpoint has no honest representation: the export boundary
  drops such a row and counts it, rather than putting it at second zero and letting it read as
  something that happened when the clip started. The same applies to an endpoint materially below
  zero — only the narrow band the millisecond grid cannot tell apart from t=0 is kept, and clamping
  anything wider would fabricate that same claim (:data:`NEGATIVE_TOLERANCE_SECONDS`).
* **Every per-frame signal is collapsed into blocks.** The ASD, pose and acoustic tables
  are dense grids at three different rates — the ASD stage's 25 FPS working timeline, the
  source video's own PTS list, a 10 ms Praat step. One annotation per frame would put tens
  of thousands of rows in a tier ELAN cannot render, and contiguous equal-label runs are
  what an analyst reads anyway. Because the three grids differ, each block's end is extended
  by *that table's own* median step (`median_positive_step`), never by a shared constant. The
  block ELAN then stores is half-open — `start ≤ t < end`, with `end` already carrying that
  extension — so the last sampled frame is inside the block rather than on its edge. The person
  tier is the deliberate exception: a sighting is one frame and nothing says the next one was
  ever sampled, so its interval is never widened by a step (see :func:`person_track_rows`).
* **A label is the only place a tier's meaning lives.** Every value printed here is a summary
  of a producer's row, and the parts of it that could be misread — which id space an id came
  from, whether a number was measured, whether a segment-level translation is a word gloss —
  are stated in the label and repeated in the document's `pipeline-tier-semantics` property,
  because a tier value gets quoted out of the file and the README does not travel with it.
* **The pose tier groups by PTS seconds, not by ``frame_number``.** See
  :func:`pose_presence_rows` — the ASD and pose grids number the same instant differently,
  and §20.2 already documents what happens when two unrelated integer id spaces are treated
  as one key.
* **A person sighting run is bounded by the clip's frame list, not by the track's endpoints.**
  See :func:`person_track_rows`. ``persons/frames.parquet`` records detections only — no stride,
  no sampling grid, no record of frames looked at and found empty — so the only adjacency that
  can be checked comes from ``source/frame_index.parquet``, and everything the index cannot place
  or confirm becomes a lone mark that says so in the label.
* **A linguistic bar is placed by whatever time its row actually carries, and says which one it
  was.** See :func:`spacy_token_rows`. A token is a span of *text*, not an event: an English token
  has no word timings at all (the worker is never given a word list for that variant), a source
  token's times can be non-finite or unmatched, and a sentence row carries only its segment's
  bounds. So a row with finite token times is placed on them, a row without them is placed on its
  enclosing segment and labelled ``placement=segment context (not token aligned)``, and a row with
  neither is dropped and counted rather than exported at second zero. No tier is created per token
  and no sentence tier is a parent: the four linguistic tiers are flat peers linked by ids.
* **A tier of numbers prints its units, its denominators, and the filter behind each number.** See
  :func:`acoustic_segment_rows`. ``acoustic_segments`` is the one tier whose labels are mostly
  measurements, and a printed value is read as one whatever produced it, so every number carries
  its unit (Hz, dB, seconds), each 0-1 value names what it is a ratio of, and a missing value says
  ``unknown`` with no unit after it. The bar is the *transcript* segment's interval — the acoustic
  stage aggregated the frames inside it — so the label says the numbers describe a window and are
  not independently timed, which is the linguistic tiers' lesson about a borrowed interval in the
  tier where it is easiest to miss, because the numbers really were measured. And the producer
  filters its two families differently: the F0 statistics cover the frames its worker flagged
  ``voiced`` while ``voiced_blocks`` blocks on ``f0_hz`` being present and never reads that flag,
  so the label names which filter each family used rather than letting a reader compare the
  numbers against the bars above them and see a contradiction.
* **Rows that overlap in time inside one tier are re-cut before they are written.** ELAN tiers are
  independent: two annotations in the same tier may not overlap, and pympi neither enforces that
  nor complains — the corpus's own tables (two people in one frame, two TalkNet tracks alive at
  once, two diarizer turns a second apart) already produced files ELAN cannot render. See
  :func:`project_independent_tier`: the emitted intervals become the disjoint segments of a
  boundary sweep over the producers' *own* endpoints, each carrying every label active in it,
  while the producers' logical intervals and their segment membership go into the
  ``pipeline-overlap-projection`` property. Nothing is staggered, nothing is dropped, and no
  offset is invented.
* **The document states what it left out, and "left out" is two states, not one.** See
  :func:`coverage_inventory`. An opened ``.eaf`` proves what it contains; nothing in it proves
  what was never exported. A reader who finds no pose tier cannot tell a clip with no person in
  frame from a table with 16,799 unread rows, and the second reading is the one that costs an
  afternoon, because it ends the investigation. So every normalised Parquet artifact in the
  registry is inventoried as ``exported`` (a tier is named after it), ``summarised`` (a tier
  reads it as support), ``present, not exported`` (the file is there and nothing reads it) or
  ``absent`` (no file, so nothing could have been exported). The last two are the distinction the
  whole property exists to keep, because both look like a missing tier and only one is a decision
  made here: ``present, not exported`` carries a specific reason naming what was deferred, and
  ``absent`` deliberately carries none — a file that was never written has no export-side decision
  to explain, and inventing one would be the same fabrication as a timestamp at second zero.
"""

from __future__ import annotations

import json
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

#: How far below zero an endpoint may sit and still be treated as *this* millisecond rather than
#: as a time the export may not place.
#
# The number is the width of a half slot: `seconds_to_ms` rounds half up onto a 1 ms grid, so
# exactly the values in [-0.0005, 0] — nothing else — convert to 0 ms. A producer that subtracts an
# offset and writes `-1e-6` for a first frame is therefore inside this band, and so is every other
# value that a millisecond export cannot distinguish from t=0; anything past it would have rounded
# into a slot that does not exist, which is a different claim and not noise. The pipeline's own
# producers support the wide half of the band and nothing near its edge: the persons and pose
# stages round their timestamps to 6 decimals (so a real first-frame value is 0.0 or a few
# microseconds off it, 500 times inside this band) and the finest timestamp grid any stage writes is
# the acoustic stage's 10 ms Praat step — 20 times wider than the band, and 80 times on the ASD
# stage's 0.04 s working grid. The band is also half of the shortest interval ELAN can store, so a
# kept value cannot move an annotation by a whole slot: the rule only decides whether a row is kept
# (inside, clamped) or refused (:class:`MissingTimestamp`, outside).
NEGATIVE_TOLERANCE_SECONDS = 0.0005


class NonFiniteTimestamp(ValueError):
    """A producer wrote NaN or infinity where ELAN needs an integer millisecond.

    Named rather than caught as a bare ``ValueError`` for the reason §30 named
    ``basis_non_finite``: "the numbers are not numbers" is a different state from "the
    numbers are bad", and the caller has to be able to react to *this* one without also
    swallowing every other conversion bug. It carries no payload beyond the message — the
    tier and the count are known by the caller that catches it.
    """


class MissingTimestamp(ValueError):
    """A producer wrote no timestamp where ELAN needs an integer millisecond.

    Split from :class:`NonFiniteTimestamp` because the two states are counted and reported
    separately in the run log, and for the same reason §17 keeps ``face_status`` apart from a
    score: "there is no measurement" and "the measurement is not a number" imply different
    fixes upstream. It is raised by :func:`interval_ms` — the only time path `build_eaf` uses —
    and never by :func:`seconds_to_ms`, which stays a plain converter.

    The pre-B2 rule here was "``None`` becomes 0, so the annotation stays visible next to its
    siblings." That rule put a false fact in the file. ELAN has no "time unknown" annotation, so
    an untimed row landed at second zero and read as "this happened when the clip started": a
    person with no timestamp was exported as on screen at t=0, and a word with a null
    ``end_time`` was exported over ``[0, 1)`` ms. A reader cannot tell that annotation from one
    genuinely measured at the start, and the file is the part that leaves the repository. Dropping
    the row loses one invisible bar; inventing a time loses the reader's ability to trust the
    rest of the tier.
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
       clamped rather than costing the export. **The clamp is a converter's answer, not a
       placement decision:** :func:`interval_ms` refuses a materially negative endpoint (see
       :data:`NEGATIVE_TOLERANCE_SECONDS`) rather than letting this rule turn ``(-2.0, -1.0)``
       into a bar at the start of the clip.
    3. **A zero-width interval gets +1 ms at the end.** ELAN requires ``start < end`` for
       every annotation: an interval with no width cannot be selected or dragged in its grid,
       and two annotations sharing both slots are one interval. Praat emits zero-width pitch
       marks, a track can start and end on one frame, and a collapsed word alignment is legal
       input. Widening the *end* keeps the interval where it was measured; the start is never
       moved, because shifting a start forward deletes when something began.

    ``end`` exists only for rule 3, and applies to a null end too.

    The pair this returns is what :func:`project_independent_tier` re-cuts, so the +1 ms display
    width is already in the endpoints a sweep is built from: a segment's boundaries are always
    endpoints some row actually owns.

    **``None`` becomes 0 here, and that is a converter answer, not an export policy.**
    :func:`interval_ms` — the only time path `build_eaf` takes — refuses a missing endpoint with
    :class:`MissingTimestamp` rather than exporting it at second zero. Earlier exports used this
    function's null rule; new exports refuse that invented placement. The rule stays because it is the
    documented answer to "what should a converter do with no value", and because keeping it here
    makes the refusal legible: one function decides what an .eaf may claim about *when*, and it no
    longer calls this one for nulls. An .eaf cannot say "time unknown" — every ``TIME_VALUE`` is a
    claim about when — so the export boundary drops an untimed row and counts it instead.

    A NaN or infinity raises :class:`NonFiniteTimestamp` instead of being clamped to 0, which
    is the one case clamping would be a lie: a NaN clamped to 0 makes the same claim about second
    zero that a null used to make. It raises rather than returns a sentinel because every caller
    here is placing an annotation, and "this interval has no time" is only useful as something to
    skip.
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

    **Either endpoint missing raises :class:`MissingTimestamp`.** This is the export boundary for
    every tier, so no builder — not `words`, not `person_tracks`, not a later segment-context tier
    — can land a row at second zero because its producer wrote null. A legitimate ``start == end``
    still gets its +1 ms display width; a *missing* end does not get one, because the pair it would
    widen is invented rather than measured. A materially negative endpoint is refused the same way,
    and for the same reason: see :data:`NEGATIVE_TOLERANCE_SECONDS`.

    **A materially negative endpoint raises the same way; one inside rounding noise of zero does
    not.** :func:`seconds_to_ms` clamps a negative to 0 because a converter has to answer
    something, and that is the right cost at t=0 and the wrong answer as a placement: it turned a
    ``(-2.0, -1.0)`` row into a ``[0, 1) ms`` bar — a fact about the start of the clip that no
    producer measured, in a file where a reader cannot tell it from a real first-frame annotation.
    So the refusal lives here, the one time path `build_eaf` takes, and it is sized by
    :data:`NEGATIVE_TOLERANCE_SECONDS`: within that of zero the value is what rounding at t=0
    produces and stays clamped; beyond it the row has no time this export may place, and it is
    dropped and counted on the same line as a null. No new bar is invented and no tier is lost.

    The two states raise two exceptions rather than one, because `build_eaf` counts them on two
    separate log lines and they describe different producer defects — the same reason B1 kept a
    null score apart from a NaN one inside a label.
    """
    if start is None or end is None:
        raise MissingTimestamp("endpoint is null; ELAN has no 'time unknown' slot")
    for value in (start, end):
        if not math.isfinite(float(value)):
            raise NonFiniteTimestamp(f"timestamp is {float(value)}, not a measurable time")
    for name, value in (("start", start), ("end", end)):
        if float(value) < -NEGATIVE_TOLERANCE_SECONDS:
            raise MissingTimestamp(
                f"{name} endpoint is {float(value)} s, more than "
                f"{NEGATIVE_TOLERANCE_SECONDS} s before zero: a negative TIME_VALUE is "
                "unrepresentable and clamping it would place this row at second zero, which no "
                "producer measured")
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

class TierDependencyMissing(Exception):
    """A tier's *secondary* input is absent or unreadable, so the tier cannot be built honestly.

    Distinct from :class:`NonFiniteTimestamp` (one row is unusable) and from the bare
    "the tier's own file is not there" case `build_eaf` already handles: here the primary table
    exists, has rows, and would export *something* — but the extra file that makes the claim
    precise is gone, and the remaining choices are a span dressed up as a sighting or a tier
    that says it cannot be read. This exception is how a builder picks the second one and says
    why; `build_eaf` turns it into the same one logged line an absent primary input gets.
    """


@dataclass(frozen=True)
class TierInput:
    """What one tier builder may read: its own table, its secondary inputs, and the dataset.

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

    def secondary_rows(self, artifact: str, columns: Sequence[str]) -> list[dict[str, Any]]:
        """Another producer's table, or :class:`TierDependencyMissing`.

        Three things make this different from calling :func:`read_table` directly:

        * the artifact name is resolved through :func:`artifact_path`, so a tier cannot invent
          a filename the registry does not know;
        * an absent file and an unreadable one raise the *same* exception, because from the
          tier's side they are one state — "I cannot verify what I was about to claim" — and a
          builder that handled them separately would end up handling only one of them;
        * the message names the dataset-relative path, which is what the run log and the
          per-tier skip line have to print for a reader to act on.

        Callers list what they need in :data:`SECONDARY_INPUTS`, so this stays one mechanism
        for every tier that reads more than its own table rather than one bespoke try/except
        per builder.
        """
        relative = artifact_path(artifact)
        path = self.dataset_dir / relative
        if not path.is_file():
            raise TierDependencyMissing(f"{relative} not produced")
        try:
            return read_table(path, columns=list(columns)).to_pylist()
        except Exception as exc:  # noqa: BLE001 - unreadable is the same state as absent here
            raise TierDependencyMissing(f"{relative} unreadable "
                                        f"({type(exc).__name__}: {exc})") from exc


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


def _field(value: Any, *, null_is_answer: bool = False) -> str:
    """An analysis field inside a label, keeping "absent" and "unknown" two different words.

    :func:`_id` and :func:`_num` both fold "there is nothing here" into :data:`UNKNOWN_DISPLAY`,
    which is right for an id or a score. It is wrong for the spaCy analysis columns, where the
    *expression that wrote the column* decides what a null means. `workers/spacy_worker.py`
    writes ``morph`` as ``str(token.morph)``, so an empty string is spaCy answering "this token has
    no morphology" and a null there really is an unanswered question. It writes ``ent_type`` as
    ``token.ent_type_ or None``, so the producer's own "this token is inside no named entity"
    arrives as a **null** and can never arrive as an empty string: on that column a null is the
    answer, not the absence of one, and ``null_is_answer=True`` says so. Guessing which column
    needs it from the value alone is impossible — that is exactly the collapse this function
    exists to avoid — so the caller names it. A real finite ``0`` is neither state and prints
    ``0.000`` via :func:`_num`, because a zero confidence is a measured zero.
    """
    if value is None:
        return ABSENT_DISPLAY if null_is_answer else UNKNOWN_DISPLAY
    if isinstance(value, str) and not value.strip():
        return ABSENT_DISPLAY
    return _text(value)


#: Document property holding the per-table provenance of the linguistic tiers (model + variant).
#: Named in :data:`TIER_SEMANTICS` so a reader of the file can find it without the README.
LINGUISTIC_PROVENANCE_PROPERTY = "pipeline-linguistic-provenance"

#: Shape/version marker of that property's JSON.
LINGUISTIC_PROVENANCE_VERSION = 1


# ----------------------------------------------------- independent-tier overlap projection

#: Document property holding the logical intervals behind a tier whose rows overlapped.
#: Named in :data:`TIER_SEMANTICS` so a reader of the file can find it without the README.
OVERLAP_PROJECTION_PROPERTY = "pipeline-overlap-projection"

#: Shape/version marker of that property's JSON, so a later re-shaping is visible in the file.
OVERLAP_PROJECTION_VERSION = 1

#: A segment carrying more than one label is a JSON list, not a delimiter-joined string.
# A readable delimiter (" · ", already the label's own separator) would be ambiguous the moment
# a producer's own text contained it — transcripts and Nemotron turns do contain it — so the
# membership would depend on metadata nobody can check while reading a bar. JSON round-trips any
# text, including quotes, newlines-as-spaces and Unicode, and `json.loads` recovers the exact
# member list without a delimiter to disagree about.


def intervals_overlap(rows: Sequence[dict[str, Any]]) -> bool:
    """Does any pair of these already-converted intervals genuinely overlap?

    Half-open, so two rows that merely touch (`[0, 1000)`, `[1000, 2000)`) are not an overlap:
    ELAN renders them side by side and nothing has to be split. Coincident rows are the narrowest
    real case (two people sighting in one frame, two TalkNet tracks over one turn) and two rows
    that rounded onto one pair of time slots land here too.

    Sorts a copy of the pairs and carries the running maximum end, so it is `O(n log n)` on the
    tier's rows rather than a pass over every pair.
    """
    ordered = sorted((int(row["start_ms"]), int(row["end_ms"])) for row in rows)
    furthest_end = None
    for start, end in ordered:
        if furthest_end is not None and start < furthest_end:
            return True
        furthest_end = end if furthest_end is None else max(furthest_end, end)
    return False


def project_independent_tier(tier: str, rows: Sequence[dict[str, Any]]
                             ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Re-cut one tier's rows into the disjoint segments ELAN's independent tiers require.

    Returns ``(rows_to_emit, projection_or_None)``. Every returned dict carries ``value`` — the
    text to write — and is otherwise the caller's row. A tier with no same-tier overlap is returned
    **with its rows unchanged and in the builder's own order**, and ``None`` as its metadata:
    almost every tier in every dataset has no overlap, and rewriting its intervals into a sweep of
    itself would move bytes for no reason and hide the tiers that really were re-cut.

    When a tier *does* overlap:

    * the boundaries are the rows' own converted millisecond endpoints and nothing else — no grid
      step, no offset, no rounding pass; consecutive boundaries become half-open segments, and a
      stretch of the timeline no row covers stays empty rather than being bridged;
    * a segment's value is the single active row's text when exactly one row is active (so an
      unaffected annotation keeps the exact label this module always wrote), or a JSON list of
      every active row's text when several are — see the note on
      :data:`OVERLAP_PROJECTION_PROPERTY` for why not a delimiter. Every text reaching here has
      been through :func:`_text`, so no member can carry a newline into the list;
    * group order is ``(text, source order)``, which is the same list whichever order Parquet
      returned the rows in. Two rows carrying the same text stay two members: identities are never
      equated and text is never deduplicated, because two diarizer turns with the same words are
      two turns;
    * emitted order is time order. The builders' orders (person ids in the order the clip
      introduced them) are a property of the *rows*; once rows are re-cut, one row's segment and
      its neighbour's interleave in time, so no row-level order describes the segments. The
      logical order survives in the metadata.

    The metadata names each logical row (``<tier>:<position in the tier's own row list>``), its
    producer ids when the builder supplied them (the row's ``source`` dict — the id columns that
    tier actually read, never a cross-walk), its interval in the producer's own seconds *and* in
    milliseconds, its text, and the segments it was cut into — so the union of a row's segments
    recovers that row's emitted range and the text/ID mapping survives the split even when one
    source interval is now three bars. Rows dropped upstream for a missing or non-finite timestamp
    never reach here, so they are neither emitted nor counted.

    The work is one boundary sweep: every row is added at its own start boundary and removed at
    its own end boundary, so cost is `O(n log n + s)` for `n` rows and `s` segments, never a step
    over milliseconds.
    """
    if not intervals_overlap(rows):
        return [dict(row, value=row["text"]) for row in rows], None

    points = sorted({int(row["start_ms"]) for row in rows} | {int(row["end_ms"]) for row in rows})
    position = {point: index for index, point in enumerate(points)}
    adds: list[list[dict[str, Any]]] = [[] for _ in points]
    removes: list[list[dict[str, Any]]] = [[] for _ in points]
    for row in rows:
        adds[position[int(row["start_ms"])]].append(row)
        removes[position[int(row["end_ms"])]].append(row)

    emitted: list[dict[str, Any]] = []
    segments_of: dict[Any, list[list[int]]] = {}
    active: dict[Any, dict[str, Any]] = {}
    for index in range(len(points) - 1):
        # Remove before add: the intervals are half-open, so a row ending on this boundary is not
        # active in the segment that starts here while a row starting here is. `interval_ms` keeps
        # every row's converted end at least 1 ms past its start, so no row is added and removed at
        # the same boundary and this ordering cannot lose one.
        for row in removes[index]:
            active.pop(row["_row_id"], None)
        for row in adds[index]:
            active[row["_row_id"]] = row
        start, end = points[index], points[index + 1]
        if not active:
            continue
        members = sorted(active.values(), key=lambda row: (row["text"], row["_row_id"]))
        if len(members) == 1:
            value = members[0]["text"]
        else:
            # `default=str` covers a producer id of a type json has never seen (a Decimal score, a
            # date); it cannot change a str/int/float/bool, so the common path is unaffected.
            value = json.dumps([member["text"] for member in members], ensure_ascii=False,
                               separators=(",", ":"), default=str)
        emitted.append({"start_ms": start, "end_ms": end, "value": value})
        for member in members:
            segments_of.setdefault(member["_row_id"], []).append([start, end])

    logical = [{
        "row_id": row["_row_id"],
        **({"source": row["source"]} if row.get("source") else {}),
        "start_seconds": row["start"],
        "end_seconds": row["end"],
        "start_ms": int(row["start_ms"]),
        "end_ms": int(row["end_ms"]),
        "text": row["text"],
        "segments": segments_of.get(row["_row_id"], []),
    } for row in rows]
    projection = {
        "logical_row_count": len(rows),
        "final_annotation_count": len(emitted),
        "logical": logical,
    }
    return emitted, {"version": OVERLAP_PROJECTION_VERSION, "tiers": {tier: projection}}


#: Document property naming what this export represents and what it left out.
COVERAGE_PROPERTY = "pipeline-coverage"

#: States of :data:`COVERAGE_PROPERTY`. Named constants because the four words are the claim:
#: `present, not exported` (a file exists and nothing reads it) and `absent` (no file) are the
#: pair a reader must not be able to collapse into one "not in the file".
COVERAGE_EXPORTED = "exported"
COVERAGE_SUMMARISED = "summarised"
COVERAGE_PRESENT_NOT_EXPORTED = "present, not exported"
COVERAGE_ABSENT = "absent"

#: The four states as a set, so a reader of the property can tell "not a state we know" from
#: "a state we know this document has". Ordered as exported → summarised → unread → absent.
COVERAGE_STATES: tuple[str, ...] = (COVERAGE_EXPORTED, COVERAGE_SUMMARISED,
                                    COVERAGE_PRESENT_NOT_EXPORTED, COVERAGE_ABSENT)

#: Shape/version marker of the property's JSON, so a later re-shaping is visible in the file.
COVERAGE_VERSION = 1

#: Why one artifact has no tier, keyed by registry artifact name.
# Only the artifacts that really go unread are here, and each entry states what the export
# declines to represent rather than advertising a future tier. `pose_hands` and `pose_face` are
# the same deferral in two tables, so they share one string — written once so the two cannot drift
# apart and claim a table is deferred for a reason the other is not.
#
# Each wording names the table a tier *does* read, because that is the checkable half of the
# sentence: `pose_presence_rows` reads `pose/body.parquet` and two columns of it
# (`timestamp`, `confidence`), so no bar here can reach the hands, the face, or a normalised
# coordinate. Naming that keeps the reason falsifiable — a later tier that started reading one of
# these files would make its own sentence false, which is what the state map then catches.
POSE_DENSE_TRACK_REASON = (
    "dense per-joint numeric tracks are not represented by this export: no tier reads this table, "
    "and the pose tier (`pose_presence`) blocks over pose/body.parquet's timestamps and "
    "confidences alone, so these joints' coordinates stay in the Parquet table")

TIER_ABSENT_REASONS: dict[str, str] = {
    "pose_hands": POSE_DENSE_TRACK_REASON,
    "pose_face": POSE_DENSE_TRACK_REASON,
    "pose_normalized": (
        "dense per-joint numeric tracks are not represented by this export, and this table is a "
        "change of basis over the same BODY_25 keypoints the pose tier already blocks over — "
        "`pose_presence` reads pose/body.parquet's timestamps and confidences and never a "
        "coordinate, so neither x_norm/y_norm nor the basis columns here have a bar to reach"),
}

#: A reason the export could not name specifically says so, rather than going silent.
COVERAGE_REASON_UNKNOWN = (
    "no specific reason is recorded for this artifact being left out; the export writes no tier "
    "from it")

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

#: What "the producer wrote an empty value here" looks like, as distinct from :data:`UNKNOWN_DISPLAY`.
#: Two states, two words, for the reason the module already gives for a null score versus a NaN one.
#: Which state an *empty* value is is a property of the column: an empty ``morph`` (``str(token.morph)``)
#: means "spaCy answered: no morphology on this token", and so does the null that ``token.ent_type_ or
#: None`` produces for a token inside no named entity — see :func:`_field`, which takes that per
#: column. Printing :data:`UNKNOWN_DISPLAY` for either would turn a checked-and-absent analysis into
#: a missing measurement, the collapse §17 refuses everywhere else in the pipeline.
ABSENT_DISPLAY = "none"

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
    "Blocks are half-open in milliseconds: a block in asd_speaking, pose_presence or "
    "voiced_blocks covers [start, end), where end already carries that tier's own median "
    "grid-step extension, so the last sampled frame is inside the block rather than on its "
    "edge; person_tracks is the deliberate exception and adds no grid step of its own. "
    "Persons: person_id is a tracker trajectory, not a human — ids are recycled and lost, so "
    "counts of ids are not counts of people; 'sighting run N frames of M' is this id's own "
    "N de-duplicated sightings joined only across consecutive source frames that "
    "source/frame_index.parquet places and whose times agree, over M sightings in total; "
    "'sighting mark 1 frame' is a sighting the grouping did not join to a neighbour, and the "
    "adjacency marker on an annotation says whether that annotation's own boundary was checked "
    "against the clip's frame list, including a check that ended the run, so 'verified' never "
    "means 'known to continue', and 'source adjacency unverified (no source frame index, "
    "coverage unknown)' names its own cause; a person interval "
    "spans its own sightings' endpoints — it starts at the first sighting's time and ends at the "
    "last sighting's measured PTS, with no grid step added, because nothing says the next frame "
    "was ever sampled; 'max gap reported S s' (seconds, printed with the number so a missing "
    "value says only 'unknown') is the persons track table's own "
    "longest_gap_seconds column printed as reported and never recomputed here, and that column is "
    "the elapsed time between two consecutive sightings, which on a regularly sampled clip "
    "includes the sampling interval and is therefore not a measure of absence — it prints "
    f"'{UNKNOWN_DISPLAY}' when the id has a single sighting or no reported value, because the "
    "producer writes 0.0 where no pair exists; a mark's 1 ms width is "
    "ELAN's minimum representable interval, not a measured duration; a sighting whose row carries "
    "no timestamp is dropped from the file rather than placed at second zero, and the drop is "
    "counted in the run log. Sampling coverage is not "
    "established anywhere here even where the grouping is verified: the detection table records "
    "what was seen and never which frames were looked at, so absence of a sighting is not "
    "evidence that nobody was there. "
    "Independent tiers: annotations inside one tier may not overlap, and the producers' tables do "
    "overlap — two people sighted in one frame, two face tracks alive at once, two diarizer turns "
    "a second apart. Where they do, the emitted intervals are re-cut into the disjoint half-open "
    "segments of a sweep over those rows' own millisecond endpoints, and each segment carries the "
    "text of every row active in it: one text as a plain label, several as a JSON list. Those "
    "segments are a projection of what the tables already said, not new events and not new "
    "measurements — nothing is staggered, dropped, merged or shifted by an offset, and a tier with "
    "no overlap is written exactly as its rows were measured. The producers' own intervals, ids, "
    "texts and segment membership are in the " + OVERLAP_PROJECTION_PROPERTY + " property, which "
    "lists per affected tier the logical_row_count (rows the tables carried) beside the "
    "final_annotation_count (annotations this tier emits), because those two numbers are "
    "different facts and only the second one is what ELAN shows. "
    "Linguistic tiers: spacy_source_tokens, spacy_source_sentences, spacy_english_tokens and "
    "spacy_english_sentences are four flat peer tiers, not a hierarchy — a sentence tier is not "
    "the parent of a token tier and no tier is created per token; they link by the sentence_id "
    "and segment_id printed on every row. A token is a span of text, not an event, so each row "
    "says how it was placed, and the words follow the bar rather than the variant: a token whose "
    "own token_start_time and token_end_time are finite, usable (defined below) and ordered sits "
    "on them, "
    "and a row without usable token times sits on its enclosing segment's bounds with a placement "
    "fragment that says 'segment context (not token aligned)', keeping that row's own "
    "timestamp_alignment_status and timestamp_alignment_confidence rather than improving or "
    "discarding them. Such a bar is context, not a word boundary, and every English row — of "
    "spacy_english_tokens and of spacy_english_sentences alike — says that its text is '"
    # The exact text of ENGLISH_TRANSLATION_TEXT, inlined the way this property already inlines
    # the placement fragments defined below it (a module-level constant cannot forward-reference).
    # `test_the_semantics_property_...` checks the constant and this clause still agree.
    "translation text, not word alignment to the source'; the status no_timing "
    "is printed on the rows whose own column holds it, which is every English row this pipeline "
    "writes. A sentence is never timed by its first token — the only bounds the sentence table "
    "carries are the segment's. Usable means finite, ordered, and no further below zero than "
    # The tolerance, in the same words the code applies: quoting a number here is a claim about
    # behaviour, so it is formatted from the constant rather than typed twice. A test ratchets it.
    f"{NEGATIVE_TOLERANCE_SECONDS * 1000.0:g} ms: "
    "a value within half a millisecond of zero is rounding noise at the start of the clip and "
    "is placed at 0 ms, "
    "while a materially negative endpoint is a producer defect that no tier clamps onto second "
    "zero — the refusal lives in interval_ms, the one time path every tier takes, so a negative "
    "costs its row (dropped and counted with a missing timestamp) in words and person_tracks and "
    "the linguistic tiers alike. A row with neither usable token times nor usable segment bounds "
    "is dropped and "
    "counted rather than placed at second zero. 'none' and 'unknown' are two words, and which "
    "one a null takes is decided per column because the producer writes these columns "
    "differently: ent_type is written as token.ent_type_ or None, so the producer's own answer "
    "'this token is inside no named entity' arrives as a null and an empty value never reaches "
    "that column — there a null prints 'none'; morph is written as str(token.morph), so there an "
    "empty string prints 'none' (spaCy answered: no morphology on this token) and only a null "
    "prints 'unknown' (nothing reached the table). The four lexical flags work the same way on "
    "the whole fragment: a row that flagged none of the four says 'flags none', a row whose four "
    "flag columns are all null says 'flags unknown', and a confidence the producer measured as "
    "zero prints as a measured 0.000. A dependency head is printed as text inside its "
    "token's own label and is never an ELAN reference relation. Which spaCy model and which "
    "variant produced each linguistic table is recorded in the " + LINGUISTIC_PROVENANCE_PROPERTY
    + " property, once per table rather than on every token, read from the table's own metadata; "
    "a table that never recorded its model says 'unknown' there. "
    "Acoustic summary tier: acoustic_segments prints one transcript segment's measured numbers, "
    "read from acoustic/segment_features.parquet, and the bar spans that row's own start_time "
    "and end_time. Those are the transcript segment's times — they are the window the acoustic "
    "stage aggregated frames inside — so the numbers describe a window and the bar is not "
    "independently timed: nothing here measured when a pitch or a pause began or ended. Units "
    "travel with the values: f0_mean, f0_median, f0_min, f0_max, f0_std, f1_mean, f2_mean and "
    "f3_mean are Hz, intensity_mean, intensity_median, intensity_min, intensity_max and "
    "intensity_std are dB, and duration and pause_duration are seconds. The two dimensionless "
    "values name their denominators instead of a unit: pause_ratio is pause_duration over this "
    "row's duration, and that column is the span the producer was handed — the source media "
    "duration when the stage knew it, the segment span only when it did not — which is why the "
    "label calls it 'as reported' rather than calling it the segment's length; the producer "
    "leaves pause_ratio null whenever that span is unknown or zero, and a null there prints "
    "'unknown' rather than the 0.000 that would claim a window with no silence in it. "
    "voiced_ratio is the share of the frames sampled in the window. pause_count is a count of "
    "clipped silence runs, not a measurement. The pitch family covers the frames the producer "
    "flagged voiced, which is not the same criterion the voiced (f0) blocks are built on — that "
    "tier blocks on f0_hz being present and never reads the flag — while the intensity and "
    "formant means are summarised over every frame in the window, so the pitch numbers neither "
    "describe nor contradict the bars above them. Values are rounded to three decimals for "
    "display and a missing or non-finite one prints 'unknown' with no unit after it, because a "
    "unit on a non-measurement is the same lie as a zero; a measured zero still prints 0.000. A "
    "row whose own two times are null, non-finite, reversed or zero-width carries no time this "
    "export may place, so it is dropped and counted rather than widened by the display rule and "
    "given a full set of statistics over a bar no audio spans. "
    "Coverage: this document is a summary of the dataset's tables, not every number in them "
    "— dense per-frame signals are collapsed to runs and nothing here is a raw measurement. "
    "What represents each normalised table is named per artifact in the " + COVERAGE_PROPERTY
    + " property, because a missing tier alone cannot tell a reader whether the data is missing "
    "or the export never represents it: exported means a tier of this document is built from that "
    "table, summarised means a tier reads it as support for rows built from another table "
    "(persons/frames.parquet places the sighting bars and no bar is built from it), 'present, not "
    "exported' means the file is on disk and no tier reads it — with the reason naming what was "
    "deferred — and absent means the producer never wrote the file, which carries no reason "
    "because no export decision was involved. 'present, not exported' and absent are different "
    "states on purpose and are never merged: the first is a fact about this export, the second is "
    "a fact about this dataset. "
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

    The ids are also listed under `_id_keys` for the overlap projection's `source` map: this tier
    reads them, so they are what lets a reader trace a split word back to its row.
    """
    rows = item.sorted_rows(("start_time", "end_time", "word", "word_id", "segment_id",
                             "speaker_id"), "start_time", "end_time")
    return [{"start": row["start_time"], "end": row["end_time"],
             "text": _text(f"{row['word']} · {_id(row['speaker_id'])} · "
                           f"{_id(row['word_id'])} · [{_id(row['segment_id'])}]"),
             "_id_keys": ("word_id", SEGMENT_NS, SPEAKER_NS),
             **{key: row[key] for key in ("word_id", "segment_id", "speaker_id")}}
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
                           f"[{_id(row['segment_id'])}]"),
             "_id_keys": (SEGMENT_NS, SPEAKER_NS),
             **{key: row[key] for key in ("segment_id", "speaker_id")}} for row in rows]


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
                           f"[{_id(row['segment_id'])}]"),
             "_id_keys": (SEGMENT_NS, SPEAKER_NS),
             **{key: row[key] for key in ("segment_id", "speaker_id")}} for row in rows]


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
                               f"{_id(row['turn_id'])}"),
                 # The turn tables have no engine column — the tier's own engine *is* the row's
                 # namespace, so it travels in `source` rather than being guessed from a filename.
                 "_id_keys": ("turn_id", SPEAKER_NS, ENGINE_NS),
                 **{key: row[key] for key in ("turn_id", "speaker_id",
                                              "diarization_type")},
                 "engine": engine}
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
                           f"{_id(row['face_track_id'])} | {row['agreement_detail']}"),
             "_id_keys": ("turn_id", SPEAKER_NS, ENGINE_NS, FACE_TRACK_NS),
             **{key: row[key] for key in ("turn_id", "speaker_id", "engine",
                                          "face_track_id", "agreement")}}
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
    return [{"start": stamps[first], "end": _shift(stamps[last], step), "text": text,
             "_id_keys": ()}
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
                           f"{row['frame_count']} act · mean {_num(row['mean_score'], 3)}"),
             "_id_keys": ("track_id",), "track_id": row["track_id"]}
            for row in rows]


# What one sighting is, and what may join two of them into a run.
#
# `persons/frames.parquet` is a **detection** table: one row per (person, frame) the detector
# reported, and nothing else. It carries no stride, no sampling grid, and no record of the
# frames it looked at and found nobody — `stages.persons.person_track_rows` shows how little the
# track table knows by computing `longest_gap_seconds` from consecutive *sightings* alone. So a
# track's two endpoints are a span and not a sighting history, and on this corpus the difference
# is not small: La-1's id 10 disappears for a whole second (source frames 114 → 144), ids 1 and 2
# are seen in two frames each, and the pre-B2 tier gave every one of them exactly one annotation.
#
# The one adjacency the tables do prove is the source's. `source/frame_index.parquet` names every
# decodable frame of the clip with its PTS, and on all four corpus clips
# `persons.frames.frame_number` / `.timestamp` match `frame_index.frame_number` / `.pts_seconds`
# exactly. Consecutive *source* frames that the index places and whose times agree are therefore
# one sighting run; anything else — a frame the index does not name, a timestamp that disagrees
# with it, a missing or unreadable index, a frame the detector looked at and did not report this
# id in — ends the run.
#
# Adjacency is a claim about frames the detector may never have looked at, so each label below
# says how far the evidence goes instead of leaving a run to be read as a continuous sighting.
#: This annotation's own junctions were all checked against the clip's frame list.
ADJACENCY_VERIFIED = "source adjacency verified"
#: A neighbouring sighting could not be placed or confirmed, so at least one of this run's
#: boundaries is silence about the frames rather than evidence about them.
ADJACENCY_SPLIT = "run split at an unverifiable source frame"
#: Adjacency could not be established for this annotation at all, so it stands alone.
ADJACENCY_UNVERIFIED = "source adjacency unverified"
#: :data:`ADJACENCY_UNVERIFIED` with its cause, for the case where the whole clip's frame list is
#: unavailable: without it the *coverage* of the clip is unknown too, and saying so is what keeps
#: a lone mark from being read as a checked-and-isolated sighting.
ADJACENCY_NO_INDEX = "source adjacency unverified (no source frame index, coverage unknown)"

#: How far a sighting's own timestamp may sit from the index's PTS for the same frame number
#: before the two tables are treated as a mismatched pair.
#
# The producer copies the PTS through (`PersonsStage._frame_row` fills `timestamp` from
# `source/frame_index.parquet`), so on a coherent dataset the two agree to the last bit —
# measured on all four corpus clips, zero rows disagree. One microsecond is therefore generous
# without being wide enough to hide a real mismatch: consecutive source frames on a 29.97 fps
# clip are 33 ms apart, four orders of magnitude outside this band.
PTS_TOLERANCE_SECONDS = 1e-6


@dataclass(frozen=True)
class _Sighting:
    """One person observed at one source frame, de-duplicated and (maybe) placed."""

    frame: int | None
    timestamp: float | None


def _reported_gap(value: Any, sightings: int) -> str:
    """The track table's own ``longest_gap_seconds``, printed as reported, or ``unknown``.

    Read from the column rather than recomputed from the sightings, which is what the label's
    "reported" says and what makes the number checkable against the table: a label that
    paraphrases a column invites a later reader (or a later version of this file) to derive it
    again, and the derivation is exactly where a subtle re-interpretation would enter. The old
    helper here recomputed the same quantity and called it "elapsed"; the value agreed with the
    column on every corpus clip, so nothing caught the mismatch between the word and the source.

    ``unknown`` when this id has fewer than two sightings, whatever the column says. That is not
    distrusting the producer, it is reading its own documented placeholder: `PersonsStage` writes
    ``0.0`` for an id seen once because "there is no gap, and null there would make
    MAX(longest_gap_seconds) silently ignore the case". In a tier label 0.000 reads as "never lost
    sight of them", which is the collapse §17 refuses everywhere else, so the placeholder is
    printed as the state it is. A reported 0.000 on an id with two or more sightings still prints
    as 0.000 — that one is a measurement.

    The unit travels with the number rather than sitting in the label, so a missing value prints
    ``unknown`` and not ``unknown s``: a unit on a non-number is the same small lie as a zero in
    its place.

    Deliberately not called "absence": the column is the elapsed time between two consecutive
    *sightings*, so on a clip sampled at a fixed stride it contains the sampling interval, and
    nothing in a detection table can tell a frame that was looked at and rejected from one that
    was never sampled.
    """
    if sightings < 2:
        return UNKNOWN_DISPLAY
    rendered = _num(value, 3)
    return f"{rendered} s" if rendered != UNKNOWN_DISPLAY else UNKNOWN_DISPLAY


def _group_sightings(rows: Sequence[dict[str, Any]]) -> dict[int, list[_Sighting]]:
    """Frames-table rows into one sighting per (person id, source frame), in time order.

    De-duplicated on the pair rather than per row, because a sighting is a fact about a person
    at an instant and not a count of the rows carrying it. The corpus has no duplicate pairs
    today; the direction a duplicate pushes is *up*, which is the direction worth refusing. Two
    different ids in one frame stay two sightings — ``persons_in_frame`` is a property of the
    frame and never enters here.

    A row with no ``frame_number`` keeps its own slot per distinct timestamp: it is a real
    detection and dropping it would lose a sighting, but it cannot be placed next to anything,
    so it never joins a run.
    """
    grouped: dict[int, dict[Any, _Sighting]] = {}
    for row in rows:
        if row["person_id"] is None:
            continue
        person_id = int(row["person_id"])
        number = None if row["frame_number"] is None else int(row["frame_number"])
        stamp = None if row["timestamp"] is None else float(row["timestamp"])
        bucket = grouped.setdefault(person_id, {})
        key: Any = number if number is not None else ("no-frame", stamp)
        existing = bucket.get(key)
        if existing is None:
            bucket[key] = _Sighting(frame=number, timestamp=stamp)
            continue
        # Keep whichever row carries the measurement: a duplicate with a time beats one without.
        if existing.timestamp is None and stamp is not None:
            bucket[key] = _Sighting(frame=existing.frame, timestamp=stamp)
    return {
        person_id: sorted(
            bucket.values(),
            key=lambda s: (s.timestamp is None, s.timestamp or 0.0,
                           s.frame if s.frame is not None else -1))
        for person_id, bucket in grouped.items()
    }


def _person_sighting_rows(item: TierInput) -> list[dict[str, Any]]:
    """Sighting runs and marks, joined only over source frames the index can verify.

    Reads ``person_frames`` (required — a missing one raises
    :class:`TierDependencyMissing`, because a track's two endpoints are a span and not a
    sighting history) and ``frame_index`` (optional — without it nothing is verifiable and every
    sighting becomes an isolated :data:`ADJACENCY_NO_INDEX` mark).

    Each junction between consecutive sightings of one id is decided three ways, never two:

    * **joined** — both sightings are placed in the index with agreeing times and their frame
      numbers differ by exactly 1;
    * **broken** — both are verifiable and their frame numbers differ by more than 1. The run
      ends. Whether the frames in between were sampled and empty or never sampled is not
      knowable from a detection table, so nothing in the label claims either;
    * **unverifiable** — the index is missing, unreadable, names no such frame, or disagrees
      with the row's own time. Adjacency is unknown, so the run also ends, and the annotations
      on either side say so rather than implying the gap was measured.

    Nothing is repaired or interpolated: a sparse stride stays sparse, because nothing in a
    detection table says what the stride was.
    """
    rows = item.secondary_rows("person_frames", ("frame_number", "timestamp", "person_id"))
    try:
        index_rows = item.secondary_rows("frame_index", ("frame_number", "pts_seconds"))
    except TierDependencyMissing:
        index: dict[int, float] = {}
    else:
        index = {int(row["frame_number"]): float(row["pts_seconds"]) for row in index_rows
                 if row["frame_number"] is not None and row["pts_seconds"] is not None}

    def placed(sighting: _Sighting) -> bool:
        """Is this sighting's source frame in the index, at the time the row claims?"""
        if not index or sighting.frame is None or sighting.timestamp is None:
            return False
        return sighting.frame in index \
            and abs(index[sighting.frame] - sighting.timestamp) <= PTS_TOLERANCE_SECONDS

    annotations: list[dict[str, Any]] = []
    for person_id, sightings in _group_sightings(rows).items():
        runs: list[list[_Sighting]] = []
        # `boundary_unverified[i]` is the state of the gap between runs[i-1] and runs[i], so a run
        # can report that one of its own edges is not evidence.
        boundary_unverified: list[bool] = []
        for sighting in sightings:
            joined = False
            unverified = not placed(sighting)
            if runs:
                previous = runs[-1][-1]
                both_placed = placed(previous) and placed(sighting)
                joined = (both_placed and sighting.frame is not None
                          and previous.frame is not None
                          and sighting.frame == previous.frame + 1)
                if not joined:
                    # A boundary is *unverified* when at least one side could not be read; when
                    # both were read and were simply not neighbours, the run ends on evidence.
                    unverified = unverified or not both_placed
            if joined:
                runs[-1].append(sighting)
            else:
                if runs:
                    boundary_unverified.append(unverified)
                runs.append([sighting])

        for position, run in enumerate(runs):
            edge_unverified = (position > 0 and boundary_unverified[position - 1]) or \
                (position < len(boundary_unverified) and boundary_unverified[position])
            single = len(run) == 1
            # Only a junction that held can put more than one sighting in a run, so this says
            # "at least one boundary of this annotation was checked and joined".
            joined_here = not single
            head = "sighting mark 1 frame" if single else f"sighting run {len(run)} frames"
            if single:
                covered = (f"covers src frame {run[0].frame}" if run[0].frame is not None
                           else "covers src frame unknown")
            else:
                covered = f"covers src {run[0].frame}-{run[-1].frame}"
            # The marker describes *this* annotation, not the id's whole grouping: a singleton
            # whose only boundary was unreadable established nothing and may not borrow a
            # neighbour's verified marker, while a run that joined sightings and then met an
            # unreadable boundary reports both, because both are true of it.
            if not index:
                marker = ADJACENCY_NO_INDEX
            elif not placed(run[0]) or (edge_unverified and not joined_here):
                marker = ADJACENCY_UNVERIFIED
            elif edge_unverified:
                marker = f"{ADJACENCY_VERIFIED}; {ADJACENCY_SPLIT}"
            else:
                marker = ADJACENCY_VERIFIED
            annotations.append({
                "start": run[0].timestamp,
                # A sighting is an instant, so no grid step extends the end: the next frame may
                # never have been sampled. `interval_ms` widens a zero-width pair by 1 ms, which
                # is ELAN's representational minimum and not a measured duration — the `covers`
                # fragment says which frame the mark stands for, so the difference is in the file.
                "end": run[-1].timestamp,
                "_person_id": person_id,
                "_total": len(sightings),
                "_marker": marker,
                "_head": head,
                "_covered": covered,
            })
    return annotations


def person_track_rows(item: TierInput) -> list[dict[str, Any]]:
    """Sighting runs and marks per person id, over verified source-frame adjacency.

    Three inputs, and the tier names the one it could not use:

    * ``person_tracks`` — the ids the tracker reported and each one's mean confidence;
    * ``person_frames`` — what was seen in which source frame. Without it there is nothing to
      group, so the tier is **skipped** with that dependency named rather than falling back to
      one annotation per track span: a span says something was seen at both ends and nothing
      about the frames between, and the corpus proves the gap is not theoretical (La-1's id 10
      loses a full second between source frames 114 and 144). Skipping one tier also keeps the
      other sixteen, which is the asymmetry :func:`build_eaf` already documents;
    * ``frame_index`` — the clip's own frame list, the only thing that makes "adjacent" mean
      anything. Absent or unreadable, every sighting is a lone mark.

    An id the track table reports but the frames table never names produces no annotation: the
    sighting has to have been observed somewhere, and a row in a summary table is not an
    observation. A sighting whose own row carries no timestamp is grouped and counted (it is a real
    detection, and `of M` says it happened) but produces no annotation, because :func:`interval_ms`
    refuses a missing endpoint rather than placing it at t=0; `build_eaf` logs the drop with the
    tier's other missing-time drops.

    Ids are **tracker trajectories, not humans**: ByteTracker recycles and loses ids, so
    `person_demo` produces 75 ids over 205 sampled frames. The label says "person <id>" because
    the column is `person_id`, and :data:`TIER_SEMANTICS` carries the caveat into the file.
    """
    tracks = item.sorted_rows(("person_id", "first_timestamp", "mean_confidence",
                               "frame_coverage", "longest_gap_seconds"),
                              "first_timestamp", "person_id")
    rank = {int(row["person_id"]): position
            for position, row in enumerate(tracks) if row["person_id"] is not None}

    rows = _person_sighting_rows(item)
    # This id's own sighting count, straight from the grouping: one annotation can carry several
    # sightings, so the count that decides whether a reported 0.0 is a measurement or a
    # "no pair exists" placeholder comes from the grouping, not from the annotation width.
    sightings_of = {row["_person_id"]: row["_total"] for row in rows}
    # The three numbers are read out of the track table and printed as reported; none of them is
    # recomputed from the frames table here (see :func:`_reported_gap` for why that matters for
    # the gap in particular). An id the track table does not know prints `unknown` three times.
    reported = {int(row["person_id"]): (
        _num(row["mean_confidence"], 3), _num(row["frame_coverage"], 3),
        _reported_gap(row["longest_gap_seconds"],
                      sightings_of.get(int(row["person_id"]), 0)))
        for row in tracks if row["person_id"] is not None}

    # The producer's appearance order first, then time, so the tier reads in the order the clip
    # introduced people rather than in whatever order Parquet returned. An id the track table
    # does not know sorts last rather than disappearing.
    rows.sort(key=lambda row: (rank.get(row["_person_id"], len(rank)),
                               row["start"] is None, _seconds(row["start"]),
                               row["_person_id"]))
    return [{"start": row["start"], "end": row["end"],
             "text": _text(f"person {row['_person_id']} · {row['_head']} of {row['_total']} · "
                           f"conf {conf} · track coverage {coverage} · "
                           f"max gap reported {gap} · {row['_covered']} · "
                           f"{row['_marker']}"),
             "_id_keys": ("person_id",), "person_id": row["_person_id"]}
            for row in rows
            for conf, coverage, gap in [reported.get(row["_person_id"],
                                                     (UNKNOWN_DISPLAY, UNKNOWN_DISPLAY,
                                                      UNKNOWN_DISPLAY))]]


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
             "text": "body present", "_id_keys": ()}
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
             "text": "voiced (f0)", "_id_keys": ()}
            for first, last, voiced in collapse_runs(stamps, flags) if voiced]


def _shift(value: Any, step: float) -> Any:
    """Extend an interval's end by one grid step, leaving None as None.

    ``None`` stays ``None`` so :func:`interval_ms` still sees "this producer gave us no end"
    and applies the zero-width rule, rather than receiving a 0.0 that would pull the end of
    a late interval back to the start of the video.
    """
    return None if value is None else float(value) + step


# --------------------------------------------------------------- linguistic tiers

#: The four linguistic tiers' ``artifact`` keys. Named for the **registry keys** and not for the
#: :data:`~multimodal_pipeline.schemas.TABLE_SCHEMAS` aliases ``linguistic_source_tokens`` /
#: ``linguistic_english_sentences``: those two registries are not the same namespace, and the one
#: this module can actually resolve a path with is ``ARTIFACT_LAYOUT`` (see
#: :func:`artifact_path`), which spells them ``spacy_*``. Reusing the schema alias as an artifact
#: name would fail every one of these tiers on a well-formed dataset, silently, through the
#: per-tier skip line.
SPACY_SOURCE_TOKENS = "spacy_source_tokens"
SPACY_SOURCE_SENTENCES = "spacy_source_sentences"
SPACY_ENGLISH_TOKENS = "spacy_english_tokens"
SPACY_ENGLISH_SENTENCES = "spacy_english_sentences"

#: How a linguistic annotation was placed, printed on every row of the four linguistic tiers.
# The pipeline's other tiers are timed by the event they describe: a sighting has a PTS, a turn has
# a diarizer's endpoints. A spaCy token is not an event — it is a span of *text* inside a segment,
# and whether it has a time of its own depends on the variant, on the tokeniser, and on whether the
# alignment could name the word. One word for all four cases would put a bar on the timeline and
# leave the reader to guess which of these three claims the bar is entitled to make.
#: The token carries its own ``token_start_time``/``token_end_time`` and they are usable (finite,
#: non-negative, ``start < end``, per :func:`_valid_pair`), so the bar spans the interval the
#: producer measured for that token.
TIMING_TOKEN_ALIGNED = "token aligned"
#: The token carries usable times, but the producer's own ``timestamp_alignment_status`` says the
#: pairing is only a borrowing (``approximate``: the timestamp is right while the 1:1 pairing is not
#: provable; ``unmatched``: no timestamp, so no time could be fabricated from it either). The bar is
#: still placed on the times the table holds; the label says they are not a provable word boundary.
TIMING_TOKEN_REPORTED = "token time as reported"
#: No usable time for this token, so the bar spans its **segment's** endpoints. The label says the
#: bar is segment context and not a token-aligned span, and it keeps the row's own alignment status
#: and confidence, because those are the producer's statement about the timing and nothing here
#: improves or discards them. Printed only of a row the export actually placed there: it names
#: where the bar sits, so it cannot be added to a variant on the strength of what that variant
#: usually looks like (see :data:`ENGLISH_TRANSLATION_TEXT`).
TIMING_SEGMENT_CONTEXT = "placement=segment context (not token aligned)"

#: The exact ``timestamp_alignment_status`` values this module branches on, quoted from the
#: producer's vocabulary (`workers/spacy_worker.py`) rather than paraphrased: ``aligned`` is the one
#: state allowed to claim a token's own span, and ``no_timing`` is what the English variant carries
#: on every row because the worker is given no word list for a translation at all. ``unmatched`` is
#: deliberately absent: it has no branch here — an unmatched row may still hold finite times, and
#: this module places those and prints the producer's word unchanged.
STATUS_ALIGNED = "aligned"
STATUS_NO_TIMING = "no_timing"

#: The linguistic tier's own words for "this row's variant has no timing to place it by", built
#: from :data:`STATUS_NO_TIMING` so the label can never drift from the column value it names.
#: Named rather than inlined because :data:`TIMING_SEGMENT_CONTEXT` already contains the word
#: ``context``, so a test that looked for the bare phrase would pass on a token row. Printed on a
#: row only when its own status column holds that value **and** the row was placed on its segment —
#: which is every English row of every table this pipeline writes today, and still a measurement of
#: the row rather than a property of the tier name.
LINGUISTIC_NO_TIMING = f"variant {STATUS_NO_TIMING}"

#: The English tier's claim about its own **text**, printed on every English row whatever the bar
#: is placed on. Separate from the placement words on purpose: "this text translates a source line
#: and is not a word-level alignment to it" is true of an English row whether or not the table
#: happens to carry token times for it, while "segment context" is a statement about the interval
#: and belongs only on a row the export really placed on its segment.
ENGLISH_TRANSLATION_TEXT = "translation text, not word alignment to the source"

#: The four nullable lexical-flag columns, as ``(label printed, column name)``.
#: One list so the "all four unread" state and the "which were true" list are read off the same
#: columns and cannot drift apart.
LEXICAL_FLAGS: tuple[tuple[str, str], ...] = (
    ("alpha", "is_alpha"), ("stop", "is_stop"), ("digit", "is_digit"), ("num", "like_num"))

#: Parquet key/value metadata key the spaCy stages write per table
#: (``stages/spacy_source.py::normalize``). Read from the file's own schema metadata, never from
#: config: the export runs over whatever is on disk, and the model that produced a table is a fact
#: about that table, not about the settings of the run that happens to be configured now.
SPACY_MODEL_METADATA_KEY = "spacy_model"

#: A model name that reaches the table as the string ``"None"``
# ``normalize`` writes ``"spacy_model": str(payload.get("selected_model"))``, and
# ``write_table`` drops a metadata value only when it is ``None`` — so a document with no
# ``selected_model`` lands on disk as the four-character string "None". That is a missing value
# written through ``str()``, not a model anybody installed, so the export reports it as
# :data:`UNKNOWN_DISPLAY` rather than naming a model that does not exist. It is not hypothetical:
# nothing in either spaCy stage guarantees the key.
UNWRITE_MODEL_NAMES: frozenset[str] = frozenset({"none", "null", "", "unknown"})

#: Column sets read by the two linguistic builders, exported so a test can check them against
#: ``TOKENS_SCHEMA`` / ``SENTENCES_SCHEMA`` rather than against this file's memory of them.
SPACY_TOKEN_COLUMNS: tuple[str, ...] = (
    "segment_id", "sentence_id", "token_id", "token_index", "speaker_id", "text", "lemma",
    "pos", "tag", "morph", "dep", "head_token_id", "head_text", "head_pos", "ent_type",
    "is_alpha", "is_stop", "is_digit", "like_num", "char_start", "char_end",
    "segment_start_time", "segment_end_time", "token_start_time", "token_end_time",
    "timestamp_alignment_status", "timestamp_alignment_confidence",
)
SPACY_SENTENCE_COLUMNS: tuple[str, ...] = (
    "segment_id", "sentence_id", "sentence_index", "speaker_id", "text", "token_count",
    "char_start", "char_end", "segment_start_time", "segment_end_time",
)


def _valid_pair(start: Any, end: Any) -> bool:
    """Do these two producer values form a time this export is allowed to print?

    Finite, ``start`` no further below zero than :data:`NEGATIVE_TOLERANCE_SECONDS`, and
    ``start < end``. Deliberately **not** the `start <= end` rule :func:`interval_ms` enforces:
    that one widens a genuine zero-width measurement by a millisecond because ELAN cannot store a
    zero-width bar, which is right for a real span and wrong for a segment context fallback. A row
    whose segment bounds are equal or reversed is not a usable enclosing interval, and the honest
    outcome is the row's own drop, not a bar whose width came from the display rule.

    A **materially negative endpoint is not usable either**, which is a change from the first
    version of this check. It used to allow negatives on the grounds that :func:`seconds_to_ms`
    clamps them to 0; that is a converter's answer and it is the wrong one here, because the clamp
    turns ``(-2.0, -1.0)`` into ``[0, 1) ms`` — a bar at the start of the clip, which is precisely
    the invented placement :class:`MissingTimestamp` exists to refuse. No producer in this pipeline
    writes negative times (`workers/spacy_worker.py` forwards WhisperX segment and word times
    unmodified), so a negative is a producer defect, and the export's job is to drop the row and
    count it rather than launder the defect into second zero.

    The tolerance is shared with :func:`interval_ms` rather than restated as `>= 0` so the two
    rules answer one question the same way: a token whose first frame time is `-1e-6` keeps **its
    own** span instead of being demoted to segment context over a rounding artefact. This check can
    therefore never call usable a pair the export would then refuse.
    """
    try:
        first, second = float(start), float(end)
    except (TypeError, ValueError):
        return False
    return (math.isfinite(first) and math.isfinite(second)
            and first >= -NEGATIVE_TOLERANCE_SECONDS and second > first)


def _alignment_fragment(row: dict[str, Any]) -> str:
    """The producer's own alignment verdict, verbatim, with the confidence as a measured number.

    The status is :func:`_field`-formatted, so a null says ``unknown`` and an empty string says
    ``none`` rather than both saying ``unknown``. The confidence is :func:`_num`-formatted, so a
    real ``0.0`` prints ``0.000`` and not ``unknown``: the worker writes 0.0 as a *measured* "no
    confidence in this pairing" for every ``unmatched`` and ``no_timing`` row, and hiding it behind
    the word for "no measurement" would be the opposite collapse from the one :func:`_field` avoids.
    """
    return (f"alignment={_field(row.get('timestamp_alignment_status'))} "
            f"conf={_num(row.get('timestamp_alignment_confidence'), 3)}")


def _char_span(row: dict[str, Any]) -> str:
    """The token's own character span inside the segment text, or ``char unknown``.

    Character offsets are not a time and are never used to place a bar; they are printed because
    they are the one span a linguistic row always has, and they let a reader find the token in the
    sentence the same producer wrote. A null is not turned into ``0`` — ``char 0-0`` would claim
    the token sits at the very start of the text.
    """
    start, end = row.get("char_start"), row.get("char_end")
    if start is None or end is None:
        return "char unknown"
    return f"char {_id(start)}-{_id(end)}"


def _flags_fragment(row: dict[str, Any]) -> str:
    """The four nullable lexical flags as one fragment, with "unread" kept distinct from "all false".

    ``is_alpha``/``is_stop``/``is_digit``/``like_num`` are nullable in ``TOKENS_SCHEMA``, and the
    :func:`_flag` reading (null behaves like false) is right for the ASD boolean it was written
    for and wrong for the label: a row whose flags were never measured printed exactly what a row
    measured false on all four prints. So the fragment says ``unknown`` only when **all four**
    columns are null, and otherwise names the ones that hold — one word of cost, no per-flag
    commentary, because the pair of states worth separating is "answered none" and "not asked".
    """
    if all(row.get(key) is None for _label, key in LEXICAL_FLAGS):
        return f"flags {UNKNOWN_DISPLAY}"
    names = " ".join(label for label, key in LEXICAL_FLAGS if _flag(row, key))
    return f"flags {names if names else ABSENT_DISPLAY}"


def _time_value(value: Any) -> str:
    """A producer's seconds as the millisecond this export actually writes.

    Rendered through :func:`seconds_to_ms` rather than printed as seconds, so the fragment can be
    compared against the document's own ``TIME_VALUE``. A non-finite value says ``non-finite`` and a
    null says ``unknown``: neither reaches the file, because the row is dropped, but the fragment is
    built before that decision and must not print a 0 that no bar carries.
    """
    if value is None:
        return "unknown"
    try:
        if not math.isfinite(float(value)):
            return "non-finite"
    except (TypeError, ValueError):
        return "non-finite"
    return f"{seconds_to_ms(value)} ms"


def _variant_of(item: TierInput) -> str:
    """The variant a linguistic table holds, from the artifact key that named it.

    Read from the registry key rather than the table's own ``variant`` column: the column is
    populated by the normalizer (``stages/spacy_source.py`` stamps it), but the *tier* is chosen by
    which file is being read, and a label must not disagree with the file it came from. ``None``
    means this is not a linguistic tier, and no caller in the registry reaches it.
    """
    return {SPACY_SOURCE_TOKENS: "source", SPACY_SOURCE_SENTENCES: "source",
            SPACY_ENGLISH_TOKENS: "english", SPACY_ENGLISH_SENTENCES: "english"}.get(item.artifact)


def spacy_model_of(item: TierInput) -> str:
    """The model that produced one table, read from that table's own Parquet metadata.

    One value per **table**, written once into the document's provenance property rather than
    repeated on every annotation: the stage writes the same ``spacy_model`` for a whole file
    (``normalize`` passes one ``selected_model`` per run), so a per-word copy would cost the
    scannability the labels exist to keep and buy nothing a reader can check.

    Three states, and only the first is a name: the key present and non-empty (the corpus's own
    tables carry ``en_core_web_lg`` / ``es_core_news_lg`` / ``blank``, all of them reported exactly
    as written); the key absent (a table written before the metadata existed, or by a different
    normalizer) → :data:`UNKNOWN_DISPLAY`; the key present but holding the string ``"None"``
    (``normalize`` formats it with ``str()``, and ``write_table`` drops only a real ``None``) →
    :data:`UNKNOWN_DISPLAY`, because "None" is a missing value wearing a model name.

    Config is never read. This function takes a table, not a context, and the answer it gives is a
    fact about the bytes on disk: a dataset re-exported after somebody edited the config keeps the
    model that actually produced its tokens.

    An unreadable file yields :data:`UNKNOWN_DISPLAY` rather than raising: this is provenance about
    a tier that was read successfully (the rows are read separately, under the per-tier guard), and
    losing the model name is not a reason to lose the tier a second time.
    """
    try:
        # The footer's schema metadata, not the rows: this answers "what did the writer record about
        # this file", and a tokens table on a long clip is thousands of rows read for one key.
        import pyarrow.parquet as pq

        metadata = pq.read_schema(item.path).metadata or {}
    except Exception:  # noqa: BLE001 - provenance is not worth a second tier failure
        return UNKNOWN_DISPLAY
    raw = metadata.get(SPACY_MODEL_METADATA_KEY.encode())
    if raw is None:
        raw = metadata.get(SPACY_MODEL_METADATA_KEY)
    if raw is None:
        return UNKNOWN_DISPLAY
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    text = " ".join(str(raw).split())
    if text.lower() in UNWRITE_MODEL_NAMES:
        return UNKNOWN_DISPLAY
    return text


def spacy_token_rows(item: TierInput) -> list[dict[str, Any]]:
    """One annotation per spaCy token: the text first, then the analysis that describes it.

    The word leads for the reason :func:`words_rows` gives for its own: this is the tier an analyst
    reads, and a row that opened with ``seg000001-s001-t0003`` would put every token a page-turn
    away. Everything after it is a ``key value`` pair with a name, because the parts of a spaCy row
    that are easy to misread — which ``head`` is meant literally, whether an empty ``morph`` means
    "no morphology" or "no analysis", whether a bar over ``[1.0, 2.0)`` ms is *the token's* time or
    its segment's — have to be stated rather than inferred from column order.

    The head is labelled ``dep head`` and never drawn as an ELAN relation: ELAN's
    ``REF_ANNOTATION`` points a *child tier* at a parent's annotation, which is the hierarchy this
    module refuses on purpose (see the module docstring and :func:`spacy_sentence_rows`). A
    dependency arc inside one tier has no representation in the format that is not a hierarchy, so
    the arc travels as text.

    Timing is decided per row and printed as one of the :data:`TIMING_TOKEN_ALIGNED` /
    :data:`TIMING_TOKEN_REPORTED` / :data:`TIMING_SEGMENT_CONTEXT` states:

    * finite, non-negative, valid ``token_start_time``/``token_end_time`` → the bar spans **those**
      times. The producer's own ``timestamp_alignment_status`` still rides on the label, so an
      ``approximate`` borrowing reads as one instead of being laundered into a word boundary;
    * token times missing, non-finite, negative or not a span → the bar spans the **enclosing
      segment's** bounds, labelled :data:`TIMING_SEGMENT_CONTEXT`, with the row's original
      alignment status and confidence preserved so the label does not claim more than the token's
      own row did;
    * neither pair usable → the row carries ``start = end = None`` and :func:`interval_ms` raises
      :class:`MissingTimestamp` in `build_eaf`, which drops the row and counts it. No time is
      invented here, and no time is invented at zero.

    The fragment therefore follows **the bar**, on every variant including English. The English
    tier used to overwrite it unconditionally with the no-timing/segment-context wording, on the
    reasoning that the worker is given no word list for a translation and so every row lands on its
    segment — true of every table this pipeline writes today, and still not a licence to print a
    placement the row does not have: a row that ever carried finite token times was drawn over the
    token's bounds while its own label denied it. What the English tier states unconditionally is
    :data:`ENGLISH_TRANSLATION_TEXT`, because that is a claim about the *text*, and
    :data:`LINGUISTIC_NO_TIMING` is printed whenever the row's own status really holds
    :data:`STATUS_NO_TIMING` **and** the row was placed on its segment.

    The sort is segment-first rather than time-first, and that is not an oversight. English rows all
    share their segment's interval, so a time sort would order them by Parquet's accident; segment
    then ``token_index`` is the order the sentence was written in, and for a source token it
    disagrees with a time sort only where the producer's own alignment says the pairing is loose.
    """
    rows = item.rows(SPACY_TOKEN_COLUMNS)
    # From the artifact key the tier was registered under, so a label can never disagree with the
    # file it came from (see :func:`_variant_of`).
    variant = _variant_of(item) or UNKNOWN_DISPLAY
    rows.sort(key=lambda row: (_seconds(row.get("segment_start_time")),
                               str(row.get("segment_id") or ""),
                               -1 if row.get("token_index") is None else int(row["token_index"])))
    annotations: list[dict[str, Any]] = []
    for row in rows:
        token_ok = _valid_pair(row.get("token_start_time"), row.get("token_end_time"))
        segment_ok = _valid_pair(row.get("segment_start_time"), row.get("segment_end_time"))
        status = row.get("timestamp_alignment_status")
        if token_ok:
            start, end = row["token_start_time"], row["token_end_time"]
            timing = (TIMING_TOKEN_ALIGNED if status == STATUS_ALIGNED
                      else TIMING_TOKEN_REPORTED)
        elif segment_ok:
            start, end = row["segment_start_time"], row["segment_end_time"]
            timing = TIMING_SEGMENT_CONTEXT
            if status == STATUS_NO_TIMING:
                # The producer's own word for "this variant had no timing to place by", printed
                # from the column and only on the rows placed where that leaves them.
                timing = f"{LINGUISTIC_NO_TIMING} · {timing}"
        else:
            # No fabricated time: build_eaf drops this row and counts the reason.
            start, end = None, None
            timing = "no usable timing (token and segment bounds both unusable)"
        if item.artifact == SPACY_ENGLISH_TOKENS:
            # What an English row states about itself whatever the bar is placed on: the text is a
            # translation of a source line, not a word aligned to a word in it. Naming that on the
            # row is what stops a reader from reading two English bars over the same second as two
            # words that were spoken in that second.
            timing = f"{timing} · {ENGLISH_TRANSLATION_TEXT}"
        analysis = (f"lemma {_field(row.get('lemma'))} · pos {_field(row.get('pos'))} · "
                    f"tag {_field(row.get('tag'))} · morph {_field(row.get('morph'))} · "
                    f"dep {_field(row.get('dep'))} · "
                    f"dep head {_field(row.get('head_text'))} ({_field(row.get('head_pos'))}) "
                    f"[{_id(row.get('head_token_id'))}] · "
                    f"ent {_field(row.get('ent_type'), null_is_answer=True)}")
        flags = _flags_fragment(row)
        annotations.append({
            "start": start,
            "end": end,
            "text": _text(f"{_text(row.get('text'))} · {_id(row.get('speaker_id'))} · "
                          f"{variant} · "
                          f"token {_id(row.get('token_id'))} · sentence "
                          f"{_id(row.get('sentence_id'))} · [{_id(row.get('segment_id'))}] · "
                          f"{analysis} · {_char_span(row)} · {timing} · "
                          f"placed {_time_value(start)}-{_time_value(end)} · "
                          f"{_alignment_fragment(row)} · "
                          f"{flags}"),
            "_id_keys": ("token_id", "sentence_id", SEGMENT_NS, SPEAKER_NS),
            **{key: row[key] for key in ("token_id", "sentence_id", "segment_id", "speaker_id")},
        })
    return annotations


def spacy_sentence_rows(item: TierInput) -> list[dict[str, Any]]:
    """One annotation per spaCy sentence, always placed by its segment's bounds.

    A sentence has no timing of its own anywhere in the pipeline: ``SENTENCES_SCHEMA`` carries only
    ``segment_start_time``/``segment_end_time``, so the only interval a sentence row can honestly
    print is its enclosing segment's. The label says so, and nothing here measures a sentence's
    onset by its first token: that would turn the *tokeniser's* idea of a boundary into an event on
    the timeline, and for the English variant there is no word timing to take an onset from in the
    first place.

    That makes the segment pair the tier's **only** candidate, so it goes through the same
    :func:`_valid_pair` check the token tier uses rather than straight to :func:`interval_ms`.
    The difference is one row's worth of time: `interval_ms` exists to keep a *measured* interval
    storable, so it widens an equal or reversed pair to a 1 ms bar at whatever millisecond the
    conversion produced — a zero-width segment at 2.0 s used to export as ``[2000, 2001)`` and a
    reversed one at 3.0 s as ``[3000, 3001)``, and a negative pair was clamped onto second zero.
    All three are the row having no usable time, and the honest outcome is the drop
    :func:`interval_ms` performs for a null or a materially negative endpoint, reached here by
    checking the pair first (see :func:`spacy_token_rows`).

    The tier is a flat peer of the token tier, not its parent. Making it a parent would put ELAN's
    ``REF_ANNOTATION`` hierarchy into the file — a shape two datasets could no longer compare
    column-for-column, and one this module has refused since the first tier list. The link between
    them is the ``sentence_id`` printed on both, exactly as ``segment_id`` links `words` to
    `segments_src`.

    What an English **sentence** states about its own text is the same claim an English token
    states: :data:`ENGLISH_TRANSLATION_TEXT`. The prose and the document's semantics property both
    promised that on every English row, and this tier was the one that did not deliver it — a
    translated sentence is exactly as little a word alignment to the source as a translated token
    is, and the worker is given no word list for either. It travels with the variant prefix, so it
    is a property of the table being read (one constant, not a per-row inference) rather than of
    where the bar happens to sit.
    """
    rows = item.rows(SPACY_SENTENCE_COLUMNS)
    variant = _variant_of(item) or UNKNOWN_DISPLAY
    # The English sentence table is the one whose text is not what anybody said, so its rows carry
    # the variant marker and the claim about that text ahead of the placement fragment. Built once,
    # not per row, because the tier is chosen by the file being read and cannot vary inside it.
    english_prefix = (f"{LINGUISTIC_NO_TIMING} · {ENGLISH_TRANSLATION_TEXT} · "
                      if item.artifact == SPACY_ENGLISH_SENTENCES else "")
    rows.sort(key=lambda row: (_seconds(row.get("segment_start_time")),
                               str(row.get("segment_id") or ""),
                               -1 if row.get("sentence_index") is None
                               else int(row["sentence_index"])))
    annotations: list[dict[str, Any]] = []
    for row in rows:
        usable = _valid_pair(row.get("segment_start_time"), row.get("segment_end_time"))
        annotations.append({
            # Unusable pair → no time at all, so build_eaf drops the row and counts the reason.
            "start": row["segment_start_time"] if usable else None,
            "end": row["segment_end_time"] if usable else None,
            "text": _text(f"{_text(row.get('text'))} · {_id(row.get('speaker_id'))} · "
                          f"{variant} sentence {_id(row.get('sentence_id'))} · "
                          f"tokens {_id(row.get('token_count'))} · "
                          f"[{_id(row.get('segment_id'))}] · {_char_span(row)} · "
                          f"{english_prefix}{TIMING_SEGMENT_CONTEXT} · the only bounds this "
                          f"table carries are the segment's, never a sentence-onset "
                          f"measurement"),
            "_id_keys": ("sentence_id", SEGMENT_NS, SPEAKER_NS),
            **{key: row[key] for key in ("sentence_id", "segment_id", "speaker_id")}})
    return annotations


# --------------------------------------------------------- acoustic summary tier

#: Columns the acoustic segment builder reads, exported so a test can check them against
#: ``ACOUSTIC_SEGMENTS_SCHEMA`` rather than against this file's memory of them.
#
#: `pause_ratio` is read and never recomputed: `acoustics.aggregate_segment` divides
#: `pause_duration` by the span its *caller* supplied, and `stages/acoustic.py` supplies the
#: source media duration (falling back to end−start only when the metadata had none), so the
#: denominator is a fact about the run that no longer exists by the time the export reads the
#: table. Recomputing it from `end_time − start_time` here would print a different number from
#: the column and silently disagree with the stage's own validator, which rejects a stored
#: `pause_ratio` above 1.0.
ACOUSTIC_SEGMENT_COLUMNS: tuple[str, ...] = (
    "segment_id", "speaker_id", "start_time", "end_time", "duration",
    "voiced_ratio", "f0_mean", "f0_median", "f0_min", "f0_max", "f0_std",
    "intensity_mean", "intensity_median", "intensity_min", "intensity_max", "intensity_std",
    "f1_mean", "f2_mean", "f3_mean", "pause_count", "pause_duration", "pause_ratio",
)

#: How many decimals a summary number is shown with.
# Three, because that is what the quantities live in: the producer rounds to six decimals, and a
# formant mean differs between segments in the first decimal, so 584.424409 in a label says no
# more than 584.424. Rounding to three decimals is lossy and is not claimed to separate nearby
# values: 147.699967 and 147.70002 both print 147.700. A ratio lives in [0, 1], so three decimals
# resolve 0.001 - a tenth of a percent - which is the same precision the linguistic tiers print
# their alignment confidence with, for the same reason.
ACOUSTIC_PLACES = 3

#: The four families the label groups its numbers under, printed once each as a header.
# Named as headers rather than repeating a prefix per number because the label is 20 values long:
# a key-per-number dump is unreadable in a screenshot, and the four families are how the
# producer's own docstring groups them. Two of the headers carry a qualifier, and both are the
# producer's behaviour rather than this file's inference — see :func:`acoustic_segment_rows`.
ACOUSTIC_PITCH_FAMILY = "pitch (over the frames flagged voiced, not the voiced (f0) bars)"
ACOUSTIC_INTENSITY_FAMILY = "intensity (over every frame in the window)"
ACOUSTIC_FORMANT_FAMILY = "formants (F1, F2, F3 mean over every frame in the window)"
ACOUSTIC_PAUSE_FAMILY = "pauses (clipped silence runs inside the window)"

#: What the row's own `duration` column actually holds, printed because it is not the segment's
#: span wherever the pipeline knew the media length.
ACOUSTIC_DURATION_NOTE = "as reported: source duration when known, else segment span"


def _acoustic_value(value: Any, unit: str = "") -> str:
    """A summary number with its unit, or ``unknown`` **without** one.

    :func:`_num` already refuses to turn a null into ``0.000``; this only appends the unit, and
    appends it to the number rather than to the word for "no number". ``unknown Hz`` would put a
    unit on a non-measurement, which is the same small lie as the zero it replaces — the person
    tier's reported gap already follows this rule for the same reason.
    """
    rendered = _num(value, ACOUSTIC_PLACES)
    return f"{rendered} {unit}" if unit and rendered != UNKNOWN_DISPLAY else rendered


def _acoustic_ratio(column: str, value: Any, denominator: str) -> str:
    """One dimensionless column, named, with what it is a ratio *of*.

    A bare 0-1 number is the one kind of value in this table with no physical unit, so its
    meaning is entirely its denominator. Naming both the column and the denominator is what stops
    a reader from taking `pause_ratio 0.098` for a share of the clip when it is a share of the
    span the producer divided by, or `voiced_ratio 0.759` for a share of the audio's time. The
    column name is printed even when the value is missing, so the state is "this ratio was not
    computed" rather than a bare word no reader can attribute to a column.
    """
    return f"{column} {_acoustic_value(value)} of {denominator}"


def _acoustic_pair(row: dict[str, Any]) -> tuple[Any, Any]:
    """The row's own two times, or ``(None, None)`` for a pair this export may not place.

    Three states, and the third is deliberately left for :func:`interval_ms` to answer:

    * a null endpoint → ``(None, None)``, so `build_eaf` counts a missing timestamp;
    * a finite pair that is reversed, zero-width or materially negative → ``(None, None)``, on
      the same line. Handing it to :func:`interval_ms` instead would take its 1 ms widening —
      right for a genuine zero-width measurement, wrong for a broken row, because twenty
      measured statistics would then sit over a bar no audio spans. :func:`_valid_pair` is the
      check the linguistic tiers already use, so the two tiers cannot disagree about what a
      usable pair is;
    * a NaN or an infinity → passed through unchanged. The producer wrote a value that is not a
      number, which is the *other* counted state: rewriting it to null would move the row from
      the non-finite line onto the missing one and merge two defects the run log keeps apart on
      purpose (:class:`NonFiniteTimestamp` and :class:`MissingTimestamp` exist for that split).
    """
    start, end = row.get("start_time"), row.get("end_time")
    if start is None or end is None:
        return None, None
    try:
        first, second = float(start), float(end)
    except (TypeError, ValueError):
        return None, None
    if not (math.isfinite(first) and math.isfinite(second)):
        return start, end
    return (start, end) if _valid_pair(first, second) else (None, None)


def acoustic_segment_rows(item: TierInput) -> list[dict[str, Any]]:
    """One annotation per measured segment: the segment id, then the four families of numbers.

    The id leads for the reason :func:`words_rows` gives for the word: this is the part of the
    label that links. `segment_id` is the one key shared with `segments_src`, `gloss_en` and the
    four linguistic tiers, so a quoted fragment stays traceable to the words and the translation
    it describes. A label that opened with a pitch value could not be.

    It is then repeated in the bracket the other tiers use — the label says the id twice, which is
    a deliberate 14-character cost. Every tier in this file carries its segment link as
    ``[segment_id]``, so one grep finds every annotation of every tier belonging to one segment;
    the alternative (leading with the id and dropping the bracket) saves about 1.6 % of a label
    that is already long and breaks that single cross-tier pattern. Rephrasing it away is the
    reversible direction if a reviewer prefers the shorter label.

    The four families are printed as headers rather than 20 prefixed numbers, and the units are
    printed with the values (Hz, dB, seconds) because a tier value leaves the file — into a
    screenshot, an issue, a slide — and ``f0 147.70`` is Hz or dB or a ratio depending only on a
    column name that is no longer next to it.

    Two of the family headers carry a qualifier, and both are read off the producer rather than
    guessed:

    * `acoustics.aggregate_segment` builds the pitch list from frames where ``voiced is True`` and
      summarises that, while :func:`voiced_rows` builds the `voiced (f0)` blocks from
      ``f0_hz is not None`` and documents that it deliberately does not read the flag. Those are
      two criteria over two tables, so the pitch header names its own and says plainly that it is
      not the bars a reader sees above it in the grid. Left unstated, a reader comparing
      `f0_mean` against the block layout would see a disagreement where the tables answered
      different questions.
    * intensity and the formant means are summarised over *every* frame inside the interval, with
      no voicing filter, so their headers say so. A single "voiced only" note placed over all
      four families would have made three of them look narrower than they are.

    Placement is the row's own `start_time`/`end_time`, and those are the **transcript**
    segment's times: the acoustic stage aggregated the frames that fell inside them, so the bar
    is a window and not an event, and the label says the numbers describe a window and are not
    independently timed. That is the same lesson the linguistic tiers learned about a borrowed
    interval, in a tier where the borrowing is easier to miss because the numbers really were
    measured — it is the *when* that is someone else's.

    A pair that is not usable — null, reversed, zero-width or materially negative — is refused here
    rather than handed to :func:`interval_ms`, which widens an equal or reversed pair to a 1 ms bar
    at whatever millisecond the conversion produced and clamps a negative onto t=0. That widening
    and that clamp are right for a real measurement landing on a coarse grid and wrong for a broken
    row: they would print twenty measured statistics over a bar no audio spans, or over a bar at the
    start of a clip the row was never timed in, which is the invented placement every other tier
    already refuses. The row's ``start``/``end`` stay ``None`` and `build_eaf` drops and counts it
    on the existing missing line. :func:`_valid_pair` is the same check the linguistic tiers use, so
    the two rules cannot disagree about what a usable pair is. A non-finite endpoint is the one
    unusable pair *not* rewritten here, so that it reaches :func:`interval_ms` and is counted on the
    non-finite line — see :func:`_acoustic_pair`.

    The segment's `duration` is printed as *reported*, not as the segment's length. Measured in
    the producer: `aggregate_segment` takes the span from its caller and `stages/acoustic.py`
    passes the source media duration, falling back to end−start only when the metadata had none.
    On this corpus that column reads 4.204204 s over segment spans of 3.152 s and 0.808 s, so
    labelling it "segment duration" would put a false fact in the same label as the
    `pause_ratio` that was divided by it.
    """
    rows = item.sorted_rows(ACOUSTIC_SEGMENT_COLUMNS, "start_time", "end_time")
    annotations: list[dict[str, Any]] = []
    for row in rows:
        start, end = _acoustic_pair(row)
        pitch = " · ".join(
            f"{column} {_acoustic_value(row.get(column), 'Hz')}"
            for column in ("f0_mean", "f0_median", "f0_min", "f0_max", "f0_std"))
        loudness = " · ".join(
            f"{column} {_acoustic_value(row.get(column), 'dB')}"
            for column in ("intensity_mean", "intensity_median", "intensity_min",
                           "intensity_max", "intensity_std"))
        formants = " · ".join(
            f"{column} {_acoustic_value(row.get(column), 'Hz')}"
            for column in ("f1_mean", "f2_mean", "f3_mean"))
        # `pause_count` is a count of runs and takes no unit; `pause_duration` is seconds.
        pauses = (f"pause_count {_num(row.get('pause_count'), 0)} · "
                  f"pause_duration {_acoustic_value(row.get('pause_duration'), 's')} · "
                  + _acoustic_ratio("pause_ratio", row.get("pause_ratio"),
                                    "this row's duration"))
        annotations.append({
            "start": start,
            "end": end,
            "text": _text(f"{_id(row.get('segment_id'))} · {_id(row.get('speaker_id'))} · "
                          f"[{_id(row.get('segment_id'))}] · "
                          f"acoustic summary over this window · "
                          f"timed by the transcript segment, not independently timed · "
                          f"duration {_acoustic_value(row.get('duration'), 's')} "
                          f"({ACOUSTIC_DURATION_NOTE}) · "
                          + _acoustic_ratio("voiced_ratio", row.get("voiced_ratio"),
                                            "the frames sampled in the window") + " · "
                          f"{ACOUSTIC_PITCH_FAMILY} · {pitch} · "
                          f"{ACOUSTIC_INTENSITY_FAMILY} · {loudness} · "
                          f"{ACOUSTIC_FORMANT_FAMILY} · {formants} · "
                          f"{ACOUSTIC_PAUSE_FAMILY} · {pauses}"),
            "_id_keys": (SEGMENT_NS, SPEAKER_NS),
            **{key: row[key] for key in ("segment_id", "speaker_id")},
        })
    return annotations


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


#: The seventeen tiers, in the order they are written. Declared here so the tier set is one
#: list a reviewer can count and a test can assert against, rather than seventeen calls
#: scattered through a build function.
# The four linguistic tiers and the acoustic summary tier are appended rather than interleaved:
# the twelve names they follow and their order are the contract every other stage and every
# existing .eaf follows, and a tier that moves is indistinguishable from a tier that was renamed.
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
    TierSpec("person_tracks", "person_tracks", person_track_rows, "YOLO person sightings"),
    TierSpec("pose_presence", "pose_body", pose_presence_rows, "body-present blocks"),
    TierSpec("voiced_blocks", "acoustic_frames", voiced_rows, "voiced blocks"),
    TierSpec("spacy_source_tokens", SPACY_SOURCE_TOKENS, spacy_token_rows,
             "source-language tokens"),
    TierSpec("spacy_source_sentences", SPACY_SOURCE_SENTENCES, spacy_sentence_rows,
             "source-language sentences"),
    TierSpec("spacy_english_tokens", SPACY_ENGLISH_TOKENS, spacy_token_rows,
             "English tokens"),
    TierSpec("spacy_english_sentences", SPACY_ENGLISH_SENTENCES, spacy_sentence_rows,
             "English sentences"),
    TierSpec("acoustic_segments", "acoustic_segments", acoustic_segment_rows,
             "per-segment acoustic summaries"),
)

#: The four linguistic artifact keys, in tier order.
# Used by `build_eaf` to decide which tiers carry per-table provenance, and exported so a test can
# assert the set rather than re-listing it: a fifth linguistic tier that forgot to register here
# would be written into the file with no model recorded beside it.
LINGUISTIC_ARTIFACTS: tuple[str, ...] = (
    SPACY_SOURCE_TOKENS, SPACY_SOURCE_SENTENCES, SPACY_ENGLISH_TOKENS, SPACY_ENGLISH_SENTENCES)

#: Artifacts a tier reads besides its own, keyed by the tier that reads them.
# A tier's primary artifact is what names it and what its absence skips it for. Some tiers need
# more than that one table: `person_tracks` cannot say anything honest about sightings without
# the per-frame detections and the clip's own frame list.
#
# This is one exported list rather than a lookup inside each builder, because three things have
# to agree about the same set of files and only one of them can be checked locally:
# `ElanStage.inputs` (what `status --plan` and the state record report as the stage's
# dependencies), `ElanStage.config_fingerprint` (a file the export reads must move the hash, or a
# hand-replaced table exports its new contents under a fingerprint that still says "reusable"),
# and the builders themselves via :meth:`TierInput.secondary_rows`. Hardcoded paths at each of
# the three are how a dependency ends up read but undeclared and unhashed.
SECONDARY_INPUTS: dict[str, tuple[str, ...]] = {
    "person_tracks": ("person_frames", "frame_index"),
}

#: Every artifact the export may read: each tier's own table, then the secondary inputs.
#: Order is deliberate — tier order first, so the fingerprint's keys stay in tier order.
ALL_INPUTS: tuple[str, ...] = tuple(spec.artifact for spec in TIERS) + tuple(
    name for names in SECONDARY_INPUTS.values() for name in names)

# ------------------------------------------------------------------ coverage inventory

# The state names and their reasons live above :data:`TIER_SEMANTICS`, which quotes them; the
# functions that compute an inventory live here, next to :data:`ALL_INPUTS`, because two of the
# four states are derived from the objects in it. See :func:`coverage_inventory`.


def normalized_artifact_names() -> tuple[str, ...]:
    """Every normalised Parquet artifact in the registry, in :data:`ALL_INPUTS` order.

    The set is **derived from the registry**, not listed here: the registry is the one place a
    dataset-relative path is defined, and a hand-written list of the 22 tables would age into a
    coverage inventory that silently omits the 23rd. What is kept out is equally derived — any
    path not ending in ``.parquet`` is a raw tool output, a directory, or document/file
    (``manifest.json``, ``source/metadata.json``, ``pose/raw/``, ``elan/annotations.eaf``), and
    this inventory is about the normalised tables the export is a summary **of**.

    Order is tier order first (:data:`ALL_INPUTS`), so a reader scanning the property meets the
    exported tables in the sequence the tiers are written, and the unread ones after them.
    """
    from .artifacts import ARTIFACT_LAYOUT

    parquet = {name for name, relative in ARTIFACT_LAYOUT.items()
               if relative.endswith(".parquet")}
    ordered = [name for name in ALL_INPUTS if name in parquet]
    # Anything the registry has that no tier or secondary input names, appended in name order so
    # the property is stable across runs. A brand-new registry key lands here, and its state is
    # then decided by whether the file is on disk — `absent` or `present, not exported`.
    ordered.extend(sorted(parquet - set(ordered)))
    return tuple(ordered)


def coverage_inventory(dataset_dir: Path, names: Sequence[str] | None = None) -> dict[str, Any]:
    """One state per normalised Parquet artifact: what represents it in this document, or nothing.

    The point is the pair a reader of an opened ``.eaf`` cannot otherwise distinguish. "This clip
    has no person data" and "this export never represents pose" both leave a tier missing from
    the grid; one of them is a fact about the video and the other is a fact about this file, and
    only the first is answerable from what is on screen. So the inventory names every table in
    the registry and gives each exactly one state:

    * ``absent`` — no file on disk, so nothing could have been exported from it whatever the tier
      set said. Carries **no** reason, on purpose.
    * ``exported`` — the file is there and a tier in :data:`TIERS` is named after it; the entry
      carries that tier's name.
    * ``summarised`` — the file is there and :data:`SECONDARY_INPUTS` has a tier reading it as
      support; the entry carries the consuming tier. `persons/frames.parquet` is the example: it
      places every sighting bar and no bar is built *from* it.
    * ``present, not exported`` — the file is there and no tier reads it at all. Carries a reason
      naming what the export declines to represent.

    **The disk is consulted first, and that ordering is the claim.** `absent` outranks the other
    three: a dataset whose translation stage never ran has no `gloss_en` tier because no file was
    ever written, and calling `translation_segments` `exported` — because some tier is named after
    it in the abstract — would tell the reader the opposite of what the empty grid means. Two of
    the four states are decided from the same objects :func:`build_eaf` iterates
    (:data:`TIERS`, :data:`SECONDARY_INPUTS`), so the inventory cannot drift from the export's own
    tier list; the other two need one ``is_file()`` per artifact and nothing else.

    Whether a tier ended up **empty** because its table had no rows is deliberately not a fifth
    state — that is a fact about the clip, it is already in the tier census and in
    :func:`tier_counts`, and folding it in here would put two overlapping answers about one tier
    in a single document. A tier **skipped for a reason other than absence** (its table was
    unreadable, or a secondary dependency was not there) is still `exported`: a tier that reads a
    table and fails is not a table this export never represents, and the skip is named in
    `skipped_tiers` and in the run log rather than reported by silently relabelling the artifact.

    **Why ``absent`` carries no reason, and why that is not an omission.** A reason is a claim
    about a decision the export made, and an artifact that was never produced involved no such
    decision: the ``openpose`` stage never ran, so "this export declines to represent dense pose
    tracks" would be a true sentence about a table this dataset does not have, printed where a
    reader would take it as the cause of the absence. The state already carries the whole truth:
    there was no file to read. ``present, not exported`` is the only state with something to
    explain, and a :data:`TIER_ABSENT_REASONS` gap there prints
    :data:`COVERAGE_REASON_UNKNOWN` rather than staying silent or borrowing a neighbouring
    artifact's wording.
    """
    root = Path(dataset_dir)
    exported = {spec.artifact: spec.tier for spec in TIERS}
    summarised: dict[str, str] = {}
    for tier, artifacts in SECONDARY_INPUTS.items():
        for artifact in artifacts:
            # First consumer wins, and `SECONDARY_INPUTS` is keyed by tier, so the map is
            # deterministic. Nothing in this pipeline has a second consumer today; if one is
            # added, a single name would be a false claim and this is where it gets widened.
            summarised.setdefault(artifact, tier)

    inventory: dict[str, Any] = {}
    for name in (normalized_artifact_names() if names is None else names):
        relative = artifact_path(name)
        if not (root / relative).is_file():
            # First, and deliberately: with no file there was nothing to export, whatever the tier
            # set says about this name. See the docstring's ordering rule.
            inventory[name] = {"state": COVERAGE_ABSENT, "path": relative}
        elif name in exported:
            inventory[name] = {"state": COVERAGE_EXPORTED, "tier": exported[name],
                               "path": relative}
        elif name in summarised:
            inventory[name] = {"state": COVERAGE_SUMMARISED, "tier": summarised[name],
                               "path": relative}
        else:
            entry: dict[str, Any] = {"state": COVERAGE_PRESENT_NOT_EXPORTED, "path": relative}
            entry["reason"] = TIER_ABSENT_REASONS.get(name, COVERAGE_REASON_UNKNOWN)
            inventory[name] = entry
    return inventory


def coverage_of(eaf: Any) -> dict[str, Any]:
    """The coverage document stored in a built or reopened Eaf (``{}`` if none).

    Same shape as :func:`overlap_projection`: one reader so a consumer never parses the property
    by hand, and a document written before the property existed reads as "nothing inventoried"
    rather than raising. The ``artifacts`` map is returned rather than the whole envelope, so the
    version marker stays with the writer.
    """
    raw = dict(eaf.properties).get(COVERAGE_PROPERTY)
    if not raw:
        return {}
    try:
        return json.loads(raw).get("artifacts", {})
    except (ValueError, AttributeError):  # a hand-edited property is no inventory we can report
        return {}


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


def build_eaf(dataset_dir: Path, video_path: Path,
              log: Callable[..., None] = print) -> tuple[Any, dict[str, Any]]:
    """Read a dataset's tables and return ``(Eaf, report)``.

    The second value is what the export knows that the document does not carry as bars:
    ``{"dropped": {tier: {"missing_time": n, "non_finite": n, "rows": n}}}`` for every tier that
    was built (see :func:`drop_counts`). Returning it beside the document, rather than only
    logging the same numbers, is what lets ``stages/elan.py`` put them in the stage record — a
    reader of ``status.json`` must be able to ask "why does this tier show 20 bars for 24 rows"
    without the run log.

    The stage writes the file; this function only builds it, so the whole mapping runs in a
    test with no stage, no config and no output directory.

    A tier whose input file is absent is skipped with **one** logged line naming the file.
    Absent is the normal state for most of these tables — Nemotron may never have run,
    translation has no endpoint by default, ``persons`` ships disabled, OpenPose is its own GPU
    stage — and the pipeline's rule is that an absent artifact is reported rather than
    promised (``manifest.artifacts_not_generated``). The .eaf follows that rule.

    A file that exists but cannot be read is skipped too, with its exception named. That is a
    deliberate asymmetry with the stage's own ``validate``: one corrupt table should cost its
    own tier and not the sixteen that were already built correctly, and a .eaf with sixteen
    tiers and one logged line is worth more to a user than no .eaf at all.

    One row with an unusable timestamp costs **that row**, not its tier, and the two unusable
    states are counted and logged apart. Placing an annotation needs an integer millisecond: a
    NaN in a timestamp column is producible by an upstream stage that wrote a division it never
    checked, and a null is a measurement the producer never took. The tier's other rows were
    measured and belong in the file.

    **A missing or materially negative endpoint drops the row; it is never placed at t=0.** The rule
    used to be the opposite — "a null lands at zero so the annotation stays visible next to its
    siblings" — and that put a false fact in a file an analyst trusts: ELAN has no "time unknown"
    annotation, so an untimed row looked like something that happened when the clip started. The
    check lives here, at the one place every tier's rows pass through :func:`interval_ms`, so no
    builder (word timing, person sightings, or a future segment-context tier) can invent a time by
    omission — and the same placement covers a negative, because :func:`seconds_to_ms`'s clamp would
    otherwise hand a ``(-2.0, -1.0)`` row to the same invented bar at second zero. The counts are
    logged, so a tier that dropped half its rows says so in the run output — the difference between
    "this clip has few words" and "the words table has no times in it" stays readable.

    Overlap is resolved **here**, after the two drop rules and never inside a builder: a tier's
    rows are converted to integer milliseconds, rows with no usable time are dropped and counted,
    and only then is the tier projected (:func:`project_independent_tier`). That order matters
    twice over — a row dropped for having no time must not appear in the projection's logical
    count, and the sweep must run on the same endpoints that are about to be written, or the
    property would describe intervals the file does not contain. Per-tier isolation is unchanged:
    a projection rewrites one tier's annotations and touches no other tier, and the two
    missing/non-finite drop lines still name their own tier.

    The tier census is written as a document property, so a reader of the file alone can tell
    "this clip has no person tier because ``persons`` was off" from "the export lost it" —
    the same argument the manifest makes in JSON. The census counts what was **emitted**; where a
    tier was projected, its logical row count is a different number and lives in the projection
    property rather than here.

    The census and :data:`COVERAGE_PROPERTY` are two halves of the same answer and neither is a
    copy of the other. The census names the tiers that got bars; the inventory names every table
    in the registry and says what represents it, including the ones no tier reads. The inventory
    is computed from :data:`TIERS` and :data:`SECONDARY_INPUTS` **before** the tier loop reads a
    single table, plus one stat() per artifact: it is an answer about this export and this
    dataset's files, not a restatement of which tiers happened to end up with bars. A tier skipped
    because its producer never ran therefore reads `absent` rather than `exported`, and a tier
    skipped because its table could not be read stays `exported` — see
    :func:`coverage_inventory` for why those two are the right way round.

    Rows dropped for having no usable time are **returned as well as logged**, per tier and as
    two separate counters. The log line reaches whoever watched the run; ``status.json`` is what
    reaches the person wondering why a tier shows 20 bars for the 24 rows in the table. Nothing
    about the drop rule changes here: the same two exceptions, the same two counts, the same two
    log lines.
    """
    from pympi.Elan import Eaf

    dataset_dir = Path(dataset_dir)
    eaf = Eaf(author="multimodal-pipeline", suppress_version_warning=True)
    built: dict[str, int] = {}
    skipped: dict[str, str] = {}
    # Only the tiers that actually had to be re-cut, so the property names the exceptions instead
    # of restating seventeen unaffected tiers.
    projections: dict[str, Any] = {}
    # Per-table provenance for the linguistic tiers: which model and which variant produced the
    # bytes this tier was built from. One entry per table that was actually read, because a tier
    # that was skipped has no provenance to report — writing `unknown` for it would read as "a
    # table with no model was exported" rather than "no table was there".
    linguistic: dict[str, Any] = {}
    # Computed before any table is opened, from the tier declarations themselves — see the
    # docstring. It is the one part of the document that describes the export rather than the clip.
    coverage = coverage_inventory(dataset_dir)
    # Per-tier drop counters, returned with the document. Two states, two counters, never summed:
    # the run log prints them on two lines and the record has to keep them as far apart.
    drops: dict[str, dict[str, int]] = {}

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
            except TierDependencyMissing as exc:
                # The tier's own table was there; a file it needs in order to be truthful was
                # not. Same one logged line as an absent primary input, and the same decision:
                # lose this tier, keep the other sixteen.
                reason = f"requires {exc}"
                rows = []
            except Exception as exc:  # noqa: BLE001 - one bad table must not lose sixteen
                reason = f"{path.name} unreadable ({type(exc).__name__}: {exc})"
                rows = []
            if not reason:
                eaf.add_tier(tier_id=spec.tier)
                # Two drop counters, two log lines. A row the producer wrote no time for and a
                # row whose time is a NaN are different defects — one needs a timestamp, the
                # other needs a working division upstream — and B1's rule applies to the run log
                # as well as to a label: never merge two states into one printable number.
                missing_time = 0
                non_finite = 0
                placed: list[dict[str, Any]] = []
                for position, row in enumerate(rows):
                    try:
                        start_ms, end_ms = interval_ms(row["start"], row["end"])
                    except MissingTimestamp:
                        missing_time += 1
                        continue
                    except NonFiniteTimestamp:
                        non_finite += 1
                        continue
                    # `_row_id` keys the projection's membership lists; `source` is the row's own
                    # producer identity, built from the id columns the builder declared in
                    # `_id_keys` (never a cross-walk between spaces). Both are metadata: a tier
                    # that is not projected writes neither into the file, and no builder uses
                    # either name for its own keys.
                    placed.append({**row, "start_ms": start_ms, "end_ms": end_ms,
                                   "text": _text(row["text"]),
                                   "_row_id": f"{spec.tier}:{position}",
                                   "source": {key: row[key]
                                              for key in row.get("_id_keys", ())
                                              if row.get(key) is not None}})
                emitted, projection = project_independent_tier(spec.tier, placed)
                if projection is not None:
                    projections[spec.tier] = projection["tiers"][spec.tier]
                for annotation in emitted:
                    eaf.add_annotation(spec.tier, annotation["start_ms"],
                                       annotation["end_ms"], annotation["value"])
                built[spec.tier] = len(emitted)
                if spec.artifact in LINGUISTIC_ARTIFACTS:
                    item = TierInput(tier=spec.tier, artifact=spec.artifact, path=path,
                                     dataset_dir=dataset_dir)
                    linguistic[spec.tier] = {
                        "artifact": relative,
                        "variant": _variant_of(item),
                        "spacy_model": spacy_model_of(item),
                    }
                # Recorded whether or not either counter is non-zero, so the record's shape does
                # not depend on the data. `rows` is the count the log line prints its ratio
                # against, and it is the tier builder's logical row count *before* the drops —
                # not the projection's logical count, which is a different denominator.
                drops[spec.tier] = {"missing_time": missing_time, "non_finite": non_finite,
                                    "rows": len(rows)}
                if missing_time:
                    log(f"elan: tier {spec.tier} dropped {missing_time} of {len(rows)} "
                        f"annotation(s) with a missing timestamp (no time is exported rather "
                        f"than an invented one at t=0)")
                if non_finite:
                    log(f"elan: tier {spec.tier} dropped {non_finite} of {len(rows)} "
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
    # Always written, including when every state is `exported`: "nothing was left out" is a claim
    # worth carrying, and a property that appeared only when something was missing would make a
    # document with no property ambiguous between "nothing left out" and "written before this
    # existed" — the ambiguity `absent` vs `present, not exported` exists to remove.
    eaf.add_property(COVERAGE_PROPERTY, json.dumps(
        {"version": COVERAGE_VERSION, "artifacts": coverage},
        ensure_ascii=False, separators=(",", ":"), default=str))
    # Written only when something was re-cut. Compact by construction: one entry per affected
    # tier, and each logical row carries its segment list rather than a copy of the table.
    if projections:
        eaf.add_property(OVERLAP_PROJECTION_PROPERTY, json.dumps(
            {"version": OVERLAP_PROJECTION_VERSION, "tiers": projections},
            ensure_ascii=False, separators=(",", ":"), default=str))
    # Written only when a linguistic table was read. The model name is a property of the table, not
    # of the annotation, so it appears once per table here rather than on every token bar.
    if linguistic:
        eaf.add_property(LINGUISTIC_PROVENANCE_PROPERTY, json.dumps(
            {"version": LINGUISTIC_PROVENANCE_VERSION, "tiers": linguistic},
            ensure_ascii=False, separators=(",", ":"), default=str))
    log(f"elan: {len(built)} tier(s), {sum(built.values())} annotation(s) for "
        f"{Path(video_path).name}; skipped {len(skipped)} "
        f"({', '.join(sorted(skipped)) or 'none'})")
    return eaf, {"dropped": drops}


def drop_counts(report: Any) -> dict[str, dict[str, int]]:
    """The per-tier drop counters from a :func:`build_eaf` report, with zero rows kept out.

    One reader for the report's shape, for the same reason :func:`overlap_projection` exists: a
    consumer should not be unpacking ``report["dropped"][tier]["missing_time"]`` in three places
    and inventing three answers to "what if the key is not there".

    **Only the non-zero tiers are returned.** `build_eaf` records every built tier so the report
    is uniform in memory, and the record keeps only the tiers with something to account for, so
    ``dropped_rows: {}`` reads as "no row was refused" rather than as seventeen tiers each
    reporting zero. A tier that was **skipped** is absent from this map as well: a skipped tier
    dropped no rows, it never read any — its answer is in `skipped_tiers`, and counting its
    unread rows as drops would report a producer defect that did not happen.
    """
    dropped = (report or {}).get("dropped", {})
    return {tier: {"missing_time": int(counts["missing_time"]),
                   "non_finite": int(counts["non_finite"])}
            for tier, counts in sorted(dropped.items())
            if counts["missing_time"] or counts["non_finite"]}


def tier_counts(eaf: Any) -> dict[str, int]:
    """Annotations per tier of a built Eaf, excluding pympi's implicit ``default`` tier.

    ``Eaf.tiers[name]`` is pympi's four-part tuple — ``(annotations, ref_annotations,
    tier_dict, tier_type)`` — so the count is element 0, not ``len`` of the tuple (which is
    always 4 and would report a confident, wrong number in every provenance record).

    ``default`` is excluded because it is not one of this module's tiers: pympi creates it for
    every document and nothing writes into it. Leaving it in would report eighteen tiers for
    seventeen, and a count off by one is the kind of claim a reader believes.

    These are the **emitted** counts. Where a tier was projected (:func:`overlap_projection`),
    the number of producer rows behind those annotations is a different number and lives in that
    property, not here.
    """
    counts: dict[str, int] = {}
    for name, data in eaf.tiers.items():
        if name == "default":
            continue
        annotations = data[0] if isinstance(data, tuple) else data
        counts[name] = len(annotations)
    return counts


def overlap_projection(eaf: Any) -> dict[str, Any]:
    """The per-tier projection document stored in a built or reopened Eaf (``{}`` if none).

    One reader for the property, so a consumer never parses the JSON by hand and a document
    written before this rule existed reads as "nothing was projected" rather than a KeyError. The
    per-tier payload carries ``logical_row_count``, ``final_annotation_count`` and the ``logical``
    rows (each with its own interval, producer ids, text and segment membership).
    """
    raw = dict(eaf.properties).get(OVERLAP_PROJECTION_PROPERTY)
    if not raw:
        return {}
    try:
        return json.loads(raw).get("tiers", {})
    except (ValueError, AttributeError):  # a hand-edited property is no projection we can report
        return {}
