"""The ELAN export: millisecond maths, block collapsing, and a real ``.eaf`` on disk.

Three kinds of test, and each is here for a different reason.

``TestTimeConversion``, ``TestCollapseRuns`` and ``TestMedianStep`` drive the pure helpers
with hand-written numbers, because all three encode a rule the ELAN format imposes and a
Parquet file does not: time slots are integers, ``start < end`` is mandatory, and a dense
grid has to become blocks. Every expected value is computed in the test from the literals
written in the test, because "about a millisecond" cannot catch an off-by-one that shifts
every interval in a tier.

``TestBuildEaf`` builds a dataset directory with the pipeline's **real** ``write_table`` and
the real schemas, then reads the resulting XML. Synthetic rather than ``data/processed/``
because the shapes that matter are ones the corpus does not contain — a face→no-face→face
sequence, a zero-width interval, a word whose text carries a newline — and a test that only
runs where somebody has run five GPU stages guards nothing on a fresh clone. The XML is
parsed rather than inspected through pympi, because ELAN is what has to read the file.

``TestAgainstTheCorpus`` then runs the same builder over the real tables when they exist,
which is the only place the column *names* are not this file's invention. The stage around
this builder — its artifact path, its skips, its reuse, its registration — is
``test_elan_stage.py``, which imports this file's ``make_dataset`` rather than rebuilding it.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pytest
from multimodal_pipeline import elan as elan_core
from multimodal_pipeline.artifacts import ARTIFACT_LAYOUT
from multimodal_pipeline.elan import (
    ADJACENCY_NO_INDEX,
    ADJACENCY_SPLIT,
    ADJACENCY_UNVERIFIED,
    ADJACENCY_VERIFIED,
    ALL_INPUTS,
    ASD_IMPUTED_SUFFIX,
    ASD_NOT_EVALUATED,
    ASD_NOT_SPEAKING,
    ENGINE_NS,
    FACE_TRACK_NS,
    FORMANT_BANDS_HZ,
    INTENSITY_DB_STEP,
    PRAAT_FRAME_TIERS,
    PTS_TOLERANCE_SECONDS,
    SECONDARY_INPUTS,
    SEGMENT_NS,
    SPEAKER_NS,
    TIERS,
    TIER_SEMANTICS,
    UNKNOWN_MIME_TYPE,
    WORD_NS,
    MissingTimestamp,
    NonFiniteTimestamp,
    asd_label,
    build_eaf,
    collapse_runs,
    eaf_directory,
    interval_ms,
    median_positive_step,
    seconds_to_ms,
    tier_counts,
)
from multimodal_pipeline.schemas import (
    ACOUSTIC_FRAMES_SCHEMA,
    ACTIVE_SPEAKER_FRAMES_SCHEMA,
    ACTIVE_SPEAKER_TRACKS_SCHEMA,
    BODY_SCHEMA,
    FACE_SCHEMA,
    HANDS_SCHEMA,
    FRAME_INDEX_SCHEMA,
    PERSON_FRAMES_SCHEMA,
    POSE_NORMALIZED_SCHEMA,
    PERSON_TRACKS_SCHEMA,
    SEGMENTS_SCHEMA,
    SPEAKER_FUSION_SCHEMA,
    SPEAKER_TURNS_NEMOTRON_SCHEMA,
    SPEAKER_TURNS_SCHEMA,
    TRANSLATION_SCHEMA,
    WORDS_SCHEMA,
    read_table,
    write_table,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

#: The synthetic clip's ASD grid. 25 FPS is what the TalkNet stage works on, so the
#: expected millisecond values below are this constant and not a remembered number.
STEP = 0.04


# --------------------------------------------------------------- hand-computed maths


class TestTimeConversion:
    """Seconds to ELAN's integer milliseconds, on the cases that have a rule attached."""

    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (0.0, 0),
            (1.0, 1000),
            # Rounds rather than truncates: 4.169999 s is PTS arithmetic on a 29.97 fps
            # clip, and truncating would shorten every pose interval by a millisecond.
            (4.169999, 4170),
            (0.0004, 0),
            (0.0005, 1),
            (0.0006, 1),
            # A 25 FPS frame boundary and a 10 ms Praat step, the two grids most tiers use.
            (0.04, 40),
            (0.01, 10),
            (2.5, 2500),
            (1e-9, 0),
        ],
    )
    def test_a_start_rounds_and_is_never_negative(self, seconds: float, expected: int) -> None:
        assert seconds_to_ms(seconds) == expected

    @pytest.mark.parametrize("seconds,expected", [(-1.0, 0), (-0.0004, 0), (-2.5, 0)])
    def test_a_negative_timestamp_is_clamped_not_exported(self, seconds: float,
                                                          expected: int) -> None:
        """A negative TIME_VALUE is unrepresentable, so ELAN would refuse the file.

        Rounding at t=0 is enough to produce one: a producer that subtracts a small offset
        and emits ``-1e-6`` for its first frame. Clamping costs a fraction of a millisecond
        on one interval; the alternative is losing the whole export.
        """
        assert seconds_to_ms(seconds) == expected
        assert seconds_to_ms(seconds, end=True) >= 0

    def test_a_zero_width_end_widens_by_one_millisecond(self) -> None:
        """ELAN requires start < end, and an interval with no width cannot be selected.

        A zero-width annotation is not a rare input: Praat emits zero-width pitch marks, a
        track can start and end on one frame, and a collapsed word alignment is legal.
        """
        assert seconds_to_ms(0.0, end=True) == 1
        assert seconds_to_ms(1.0, end=True) == 1000
        # The widening is +1 ms only for a *zero* end; a nonzero end is the measurement.
        assert seconds_to_ms(0.0) == 0

    def test_a_null_timestamp_lands_at_zero_in_the_helper_alone(self) -> None:
        """The helper still answers ``None`` with 0; the **export** no longer calls it that way.

        Kept as a documented helper behaviour (:func:`seconds_to_ms` is a converter, not a
        policy), while :func:`interval_ms` — the only time path `build_eaf` uses — refuses a
        missing endpoint. The distinction matters because the helper's 0 is indistinguishable
        from a real t=0 once it reaches the file, and an .eaf cannot tell the reader which one
        it got.
        """
        assert seconds_to_ms(None) == 0
        assert seconds_to_ms(None, end=True) == 1

    @pytest.mark.parametrize("start,end", [
        (None, None),
        (None, 1.0),
        (1.0, None),
    ])
    def test_an_interval_refuses_a_missing_endpoint_rather_than_inventing_one(self, start: Any,
                                                                            end: Any) -> None:
        """A missing start or end is not a time, and ELAN has no way to say "unknown".

        ``(0, 1)`` looked like a sighting at second zero: a person with no timestamp was
        exported as being on screen at the very start of the clip, in a file an analyst will
        trust. The export drops the row and counts it instead.
        """
        with pytest.raises(MissingTimestamp):
            interval_ms(start, end)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_timestamp_raises_rather_than_landing_at_zero(
            self, bad: float) -> None:
        """Clamping a NaN to 0 would claim the annotation starts at second zero.

        A null is *known* to be missing, so t=0 keeps it beside its siblings; a NaN is not a
        time at all, and silently exporting one at t=0 puts a wrong interval in a file an
        analyst will trust. Raising here is what lets `build_eaf` drop the row and count it.
        """
        with pytest.raises(NonFiniteTimestamp):
            seconds_to_ms(bad)
        with pytest.raises(NonFiniteTimestamp):
            seconds_to_ms(bad, end=True)

    @pytest.mark.parametrize(
        "start,end,expected",
        [
            (0.0, 0.0, (0, 1)),
            (1.5, 1.5, (1500, 1501)),
            # Rounding creates zero width the producer never reported: 0.4 ms -> 0 ms/0 ms.
            (0.0, 0.0004, (0, 1)),
            # And it can invert the pair, which is the case a pair-check catches and two
            # independent calls cannot: 0.0006 -> 1 ms, 0.0009 -> 1 ms.
            (0.0006, 0.0009, (1, 2)),
            (0.0, 0.04, (0, 40)),
            # A start inside rounding noise of zero is clamped, so the pair survives.
            (-1e-06, 0.5, (0, 500)),
        ],
    )
    def test_an_interval_always_satisfies_start_less_than_end(self, start: Any, end: Any,
                                                              expected: tuple[int, int]) -> None:
        got = interval_ms(start, end)
        assert got == expected
        assert got[0] < got[1], "ELAN refuses an annotation whose start is not before its end"

    @pytest.mark.parametrize("start,end,label", [
        (-2.0, -1.0, "both-negative"),
        (-0.5, 0.5, "straddles-zero"),
        (-3.0, -3.0, "zero-width-negative"),
        (0.5, -0.5, "end-negative"),
        (-0.002, 5.0, "start-two-ms-negative"),
    ], ids=["both-negative", "straddles-zero", "zero-width-negative", "end-negative",
            "start-two-ms-negative"])
    def test_an_interval_refuses_a_materially_negative_endpoint(self, start: float, end: float,
                                                               label: str) -> None:
        """A negative time is refused like a missing one, in **every** tier.

        `(-2.0, -1.0)` used to come back as `(0, 1)`: the converter's clamp put the row over the
        first millisecond of the clip, which is a fact no producer measured and the same invented
        placement `build_eaf` already drops a null for. `(-0.5, 0.5)` → `(0, 500)` was worse in the
        other direction — a half-second bar that starts a fifth of a second before the clip
        exists. Both are refused with :class:`MissingTimestamp`, the exception `build_eaf` already
        counts, so no tier needs new plumbing and no new bar is invented.

        The tolerance is what keeps this from costing real data: the boundary case is
        :func:`test_an_interval_keeps_a_negative_within_the_tolerance_at_zero`.
        """
        with pytest.raises(MissingTimestamp):
            interval_ms(start, end)

    @pytest.mark.parametrize("start,end,expected", [
        # The producer that subtracts an offset and emits -1e-6 for its first frame.
        (-1e-06, 0.4, (0, 400)),
        # Half a millisecond of noise either side of t=0 still rounds to 0 ms.
        (-0.0004, 0.0004, (0, 1)),
        # The tolerance is one millisecond, so -0.0005 s is still noise, not a placement.
        (-0.0005, 1.0, (0, 1000)),
    ], ids=["one-microsecond", "four-hundredths-of-ms", "half-a-ms"])
    def test_an_interval_keeps_a_negative_within_the_tolerance_at_zero(
            self, start: float, end: float, expected: tuple[int, int]) -> None:
        """Rounding noise at t=0 stays a real row, clamped to zero.

        Dropping every negative would drop rows whose time is 0 as far as a millisecond grid can
        tell, and the row's content (a word, a sighting, a token) is worth more than the sign bit
        that a subtraction put there. `NEGATIVE_TOLERANCE_SECONDS` is the line.
        """
        assert interval_ms(start, end) == expected

    def test_the_negative_tolerance_is_narrower_than_one_millisecond(self) -> None:
        """The tolerance may not swallow a millisecond, or it would move a real boundary.

        A value beyond it is refused rather than rounded, so the widest change the rule can make
        to a placed interval is the sub-millisecond it started with.
        """
        tolerance = elan_core.NEGATIVE_TOLERANCE_SECONDS
        assert 0.0 < tolerance < 0.001
        assert interval_ms(-tolerance, 1.0) == (0, 1000)
        with pytest.raises(MissingTimestamp):
            interval_ms(-tolerance * 2.0, 1.0)

    def test_no_interval_ever_comes_out_negative_or_inverted(self) -> None:
        """The property, over a sweep — the pair rule has to hold for every input.

        Parametrised cases show the interesting ones; this one says the rule is total, which
        is what the format actually requires. Includes values that round to zero width and
        values whose end precedes their start. Materially negative pairs are excluded from the
        sweep because they now raise, and are covered by
        :func:`test_an_interval_refuses_a_materially_negative_endpoint`.
        """
        values = [-0.0004, 0.0, 0.0004, 0.0006, 0.0009, 0.01, 0.039999, 1.0, 4.169999]
        for start in values:
            for end in values:
                low, high = interval_ms(start, end)
                assert low >= 0, (start, end)
                assert high > low, (start, end, low, high)
        tolerance = elan_core.NEGATIVE_TOLERANCE_SECONDS
        material = [-3.0, -0.002, -0.5]
        for start in material + values:
            for end in material + values:
                if start < -tolerance or end < -tolerance:
                    with pytest.raises(MissingTimestamp):
                        interval_ms(start, end)


class TestCollapseRuns:
    """Contiguous equal labels become one run, and only contiguous ones do."""

    def test_an_empty_input_is_no_runs(self) -> None:
        assert collapse_runs([], []) == []

    def test_a_single_sample_is_one_run_spanning_itself(self) -> None:
        assert collapse_runs([0.0], [True]) == [(0, 0, True)]

    def test_the_three_state_sequence_is_three_runs(self) -> None:
        """speaking -> not speaking -> speaking, the case the ASD tier exists for."""
        stamps = [0.0, 0.04, 0.08, 0.12, 0.16, 0.2]
        labels = ["a", "a", "b", "b", "c", "c"]
        assert collapse_runs(stamps, labels) == [(0, 1, "a"), (2, 3, "b"), (4, 5, "c")]

    def test_a_label_that_leaves_and_returns_is_two_runs_not_one(self) -> None:
        """Non-adjacent equals are *not* merged, which is what makes these blocks.

        Merging them would need a label-keyed grouping, and the resulting interval would
        cover the frames in between — claiming a person spoke through the silence.
        """
        assert collapse_runs([0.0, 0.04, 0.08], ["a", "b", "a"]) == [
            (0, 0, "a"), (1, 1, "b"), (2, 2, "a")]

    def test_the_end_index_is_the_last_of_the_run(self) -> None:
        """Inclusive on both ends, so the caller adds one grid step rather than subtracting
        a frame — the convention every block tier depends on."""
        runs = collapse_runs([0.0, 0.04, 0.08], ["a", "a", "a"])
        assert runs == [(0, 2, "a")]

    def test_none_and_false_are_the_same_label(self) -> None:
        """A nullable boolean column and an explicit false read identically.

        Leaving them distinct would split a "not voiced" block wherever Parselmouth returned
        a null instead of a false, which is a difference no viewer can see.
        """
        assert collapse_runs([0.0, 0.04, 0.08], [None, False, True]) == [
            (0, 1, False), (2, 2, True)]

    def test_a_mismatched_length_is_refused(self) -> None:
        with pytest.raises(ValueError, match="2 values but 1 labels"):
            collapse_runs([0.0, 0.04], ["a"])

    def test_every_index_of_the_input_appears_in_exactly_one_run(self) -> None:
        """The property: runs partition the input, covering it once and in order."""
        labels = ["a", "a", "b", "a", "c", "c", "c", "b"]
        runs = collapse_runs([float(i) for i in range(len(labels))], labels)
        covered = [index for first, last, _ in runs for index in range(first, last + 1)]
        assert covered == list(range(len(labels)))
        assert [label for _, _, label in runs] == ["a", "b", "a", "c", "b"]


class TestMedianStep:
    """The grid step a block's end is extended by, inferred rather than assumed."""

    def test_an_empty_grid_has_no_step(self) -> None:
        assert median_positive_step([]) == 0.0

    def test_a_single_sample_has_no_step(self) -> None:
        """One row gives no gap, and inventing 0.04 would extend the block past the data."""
        assert median_positive_step([1.0]) == 0.0

    @pytest.mark.parametrize(
        "values,expected",
        [
            ([0.0, 0.04, 0.08], 0.04),
            ([0.0, 0.01, 0.02, 0.03], 0.01),
            # A dropped frame doubles one gap; the median ignores it and a mean would not.
            ([0.0, 0.04, 0.08, 0.16], 0.04),
            # Unsorted input, and duplicated timestamps (their gap is zero and excluded).
            ([0.08, 0.0, 0.04], 0.04),
            ([0.0, 0.0, 0.04, 0.08], 0.04),
            # Nulls are skipped rather than sorted to the front and counted as a gap.
            ([None, 0.0, 0.04], 0.04),
            # Descending input still reports a positive step.
            ([0.08, 0.04, 0.0], 0.04),
        ],
    )
    def test_the_step_is_the_median_positive_gap(self, values: Sequence[Any],
                                                 expected: float) -> None:
        assert median_positive_step(values) == pytest.approx(expected)

    def test_an_all_identical_grid_has_no_step(self) -> None:
        assert median_positive_step([0.5, 0.5, 0.5]) == 0.0

    def test_the_two_source_grids_are_not_the_same_step(self) -> None:
        """Why every tier infers its own step: ASD, pose and Praat disagree by design.

        The pose expectation is the *rounded* PTS step rather than 1001/30000, because that is
        what the tables hold: the normalizers store six-decimal PTS values, so a grid of
        0.0, 0.033367, 0.066733 has a 0.033367 step. Asserting the exact rational here would be
        a claim about the pipeline's storage that this function does not make.
        """
        pose_step = round(1001 / 30000, 6)
        asd = median_positive_step([round(i * 0.04, 6) for i in range(30)])
        pose = median_positive_step([round(i * pose_step, 6) for i in range(30)])
        praat = median_positive_step([round(i * 0.01, 6) for i in range(30)])
        assert asd == pytest.approx(0.04)
        assert pose == pytest.approx(pose_step)
        assert praat == pytest.approx(0.01)
        assert len({round(asd, 5), round(pose, 5), round(praat, 5)}) == 3


# -------------------------------------------------------------- synthetic dataset build


def _write(schema: Any, path: Path, rows: list[dict[str, Any]]) -> Path:
    """A table built through the real schema, so a renamed column fails here."""
    known = {field.name for field in schema}
    unknown = sorted({key for row in rows for key in row} - known)
    assert not unknown, f"{path.name}: columns not in the schema: {unknown}"
    write_table(path, pa.Table.from_pylist(rows, schema=schema), schema)
    return path


#: "caller did not supply this id" for the row builders below — distinct from a real null.
_DEFAULT = object()


def _word(word: str, start: float, end: float, *, segment_id: str = "seg-0",
          word_id: Any = _DEFAULT, **extra: Any) -> dict[str, Any]:
    """One word row.

    `segment_id` and `word_id` are parameters rather than constants because the identity the
    tier now prints is the thing under test: a fixture in which every word shares one id could
    not tell a linked label from a hardcoded prefix. `_DEFAULT` (rather than `None`) means
    "caller did not say", so a test can ask for a row whose id really is null. `duration` is
    left null when either endpoint is, because a row with no time has no duration to report and
    the fixture must not invent one the producer could not have written.
    """
    duration = (end - start) if (start is not None and end is not None) else None
    return {"schema_version": "1.0", "video_id": "clip", "segment_id": segment_id,
            "word_id": f"w-{start}" if word_id is _DEFAULT else word_id,
            "start_time": start, "end_time": end,
            "duration": duration, "speaker_id": "SPEAKER_00", "word": word,
            "confidence": 0.95, "alignment_status": "aligned", **extra}


def _asd_frame(index: int, label: str, track_id: int | None, **overrides: Any
               ) -> dict[str, Any]:
    """One ASD frame: `label` is speaking | not_speaking | no_face.

    `overrides` let a test build the states the corpus does not happen to contain —
    ``tracked_unscored`` with a stale active flag, an imputed tail, a NaN score — which is
    the whole point of a synthetic fixture: those rows exist in the schema and in the stage's
    validator, and the tier has to read them correctly whether or not this machine has one.
    """
    stamp = round(index * STEP, 6)
    face_status = "no_face" if label == "no_face" else "tracked"
    row = {
        "schema_version": "1.2", "video_id": "clip", "frame_number": index,
        "timestamp": stamp, "source_timestamp": stamp, "scene_id": 1,
        "track_id": track_id, "face_status": face_status,
        "frame_reason": "no_face" if label == "no_face" else "scored",
        "x1": None if label == "no_face" else 1.0,
        "y1": None if label == "no_face" else 2.0,
        "x2": None if label == "no_face" else 3.0,
        "y2": None if label == "no_face" else 4.0,
        "talknet_score_raw": None if label == "no_face" else 1.0,
        "talknet_score": None if label == "no_face" else 1.2,
        "score_imputed": False,
        "is_active_speaker": label == "speaking",
    }
    row.update(overrides)
    return row


def _pose_row(index: int, confidence: float) -> dict[str, Any]:
    """A pose row on the *source* grid (29.97 fps), deliberately not on the ASD grid.

    Two different grids in one dataset is the point: it is what stops a presence block from
    being built by joining a pose frame number to an ASD frame number.
    """
    return {
        "schema_version": "1.0", "video_id": "clip", "frame_number": index,
        "timestamp": round(index * (1001 / 30000), 6), "detection_index": 0,
        "keypoint_id": 0, "keypoint_name": "Nose", "x": 10.0, "y": 20.0,
        "confidence": confidence,
    }


def _face_row(index: int, detection_index: int) -> dict[str, Any]:
    """One row of `pose/face.parquet` — the table no tier reads.

    The coverage tests need that file to *exist* with a row in it, which is the state a reader has
    to be able to tell from "the openpose stage never ran". Built through the real schema like every
    other fixture row, so a column rename in `FACE_SCHEMA` lands here rather than in a fixture that
    quietly stops matching the producer.
    """
    return {"schema_version": "1.0", "video_id": "clip", "frame_number": index,
            "timestamp": round(index * (1001 / 30000), 6),
            "detection_index": detection_index, "landmark_id": 0, "x": 1.0, "y": 2.0,
            "confidence": 0.9}


def _write_unread_pose_tables(root: Path) -> None:
    """Put the three tables no tier reads on disk, with a row in each.

    `make_dataset` writes only the five producers its rules need, so hands/face/normalized are
    absent there. The corpus is the other way round — every dataset on this disk has all three, and
    `pose/face.parquet` alone holds 16,799 rows — so a test that wants `present, not exported` has
    to create them. Rows are built through the real schemas, because a fixture that wrote a stub
    would blur the state this file exists to keep apart: an unreadable file and an unread one are
    different answers.
    """
    _write(HANDS_SCHEMA, root / ARTIFACT_LAYOUT["pose_hands"], [{
        "schema_version": "1.0", "video_id": "clip", "frame_number": 0,
        "timestamp": 0.0, "detection_index": 0, "hand": "left", "keypoint_id": 0,
        "keypoint_name": "wrist", "x": 1.0, "y": 2.0, "confidence": 0.9}])
    _write(FACE_SCHEMA, root / ARTIFACT_LAYOUT["pose_face"], [_face_row(0, 0)])
    _write(POSE_NORMALIZED_SCHEMA, root / ARTIFACT_LAYOUT["pose_normalized"], [{
        "schema_version": "1.0", "video_id": "clip", "frame_number": 0, "timestamp": 0.0,
        "detection_index": 0, "keypoint_id": 0, "keypoint_name": "Nose",
        "origin_keypoint_name": "MidHip", "basis_keypoint_name": "Neck",
        "second_axis": "perpendicular", "basis_state": "basis_ok", "basis_detail": "|vi| 1.0",
        "x_norm": 0.1, "y_norm": 0.2, "value_status": "normalized"}])


def make_dataset(tmp_path: Path) -> dict[str, Path]:
    """A dataset directory with five producers' tables and a video outside it.

    Chosen so each interesting rule fires exactly once:

    * ``words`` — three words, one carrying a newline and one with zero width;
    * ``speaker_turns`` — one turn;
    * ``active_speaker_frames`` — face→no-face→face, i.e. three collapsed blocks;
    * ``active_speaker_tracks`` — one track whose endpoints need the ASD step;
    * ``pose_body`` — a low-confidence frame in the middle, so presence is two blocks;
    * the video sits in a sibling directory, because that is where the real videos are and
      it is what makes the relative media URL a ``../`` path rather than a name.
    """
    root = tmp_path / "processed" / "clip"
    video_dir = tmp_path / "input_videos"
    video_dir.mkdir(parents=True)
    video = video_dir / "clip.mp4"
    video.write_bytes(b"stub")

    _write(WORDS_SCHEMA, root / "speech" / "words.parquet", [
        _word("hello", 0.0, 0.4),
        _word("multi\nline", 0.5, 0.9),
        _word("zero", 1.0, 1.0),
    ])
    _write(SPEAKER_TURNS_SCHEMA, root / "speech" / "speaker_turns.parquet", [{
        "schema_version": "1.0", "video_id": "clip", "turn_id": "t-0",
        "speaker_id": "SPEAKER_00", "start_time": 0.0, "end_time": 1.0, "duration": 1.0,
        "diarization_type": "exclusive",
    }])
    _write(ACTIVE_SPEAKER_FRAMES_SCHEMA, root / "speaker" / "active_speaker_frames.parquet", [
        _asd_frame(0, "speaking", 0), _asd_frame(1, "speaking", 0),
        _asd_frame(2, "no_face", None), _asd_frame(3, "no_face", None),
        _asd_frame(4, "speaking", 1),
    ])
    _write(ACTIVE_SPEAKER_TRACKS_SCHEMA, root / "speaker" / "active_speaker_tracks.parquet", [{
        "schema_version": "1.0", "video_id": "clip", "track_id": 0, "first_timestamp": 0.0,
        "last_timestamp": 0.04, "frame_count": 2, "active_frame_count": 2,
        "active_ratio": 1.0, "mean_score": 1.2345, "max_score": 1.4, "scenes": [1],
        "mean_bbox_area": 100.0,
    }])
    _write(BODY_SCHEMA, root / "pose" / "body.parquet", [
        _pose_row(0, 0.9), _pose_row(1, 0.9), _pose_row(2, 0.1), _pose_row(3, 0.8),
    ])
    return {"dir": root, "video": video}


def _segment(segment_id: str, start: float, end: float, text: str,
             speaker_id: str = "SPEAKER_00") -> dict[str, Any]:
    return {"schema_version": "1.0", "video_id": "clip", "segment_id": segment_id,
            "start_time": start, "end_time": end, "duration": end - start,
            "language": "en", "speaker_id": speaker_id, "text": text, "confidence": -0.15}


def _translation(segment_id: str, start: float, end: float, english_text: str,
                 speaker_id: str = "SPEAKER_00",
                 source_text: str = "texto fuente") -> dict[str, Any]:
    return {"schema_version": "1.0", "video_id": "clip", "segment_id": segment_id,
            "speaker_id": speaker_id, "start_time": start, "end_time": end,
            "source_language": "es", "source_text": source_text,
            "english_text": english_text, "translation_model": "chat",
            "translation_prompt_version": "v1"}


def _fusion_row(turn_id: str, speaker_id: str, start: float, end: float,
                agreement: str = "face_matched", detail: str = "track 0 active on 4/5 frames",
                face_track_id: int | None = 0, engine: str = "pyannote",
                **extra: Any) -> dict[str, Any]:
    row = {"schema_version": "1.0", "video_id": "clip", "engine": engine,
           "turn_id": turn_id, "speaker_id": speaker_id, "start_time": start,
           "end_time": end, "duration": end - start, "diarization_type": "exclusive",
           "overlap_s": None, "face_track_id": face_track_id, "face_active_frames": 4,
           "face_frames_in_turn": 5, "frames_in_turn": 5, "face_mean_score": 2.506753,
           "face_score_max": 3.86, "agreement": agreement,
           "agreement_detail": detail}
    row.update(extra)
    return row


@pytest.fixture
def dataset(tmp_path: Path) -> dict[str, Path]:
    """The pytest view of :func:`make_dataset`.

    A fixture over a plain function would be enough for this file alone, but
    `test_elan_stage.py` builds the same clip on purpose — one copy of the shapes, two
    files that fail for different reasons — and a sibling module cannot ask for a fixture.
    """
    return make_dataset(tmp_path)


def tiers_of(eaf: Any) -> list[str]:
    return [name for name in eaf.tiers if name != "default"]


def annotations(eaf: Any, tier: str) -> list[tuple[int, int, str]]:
    """A tier's annotations as (start_ms, end_ms, value), resolved through the time slots.

    Read through pympi rather than the XML so the pair is what a consumer sees; the XML
    itself is asserted separately, because ELAN reads the file and not this test.
    """
    timeslots = eaf.timeslots
    return [(int(timeslots[start]), int(timeslots[end]), value)
            for start, end, value, _svg in eaf.tiers[tier][0].values()]


def logical_rows(eaf: Any, tier: str) -> list[dict[str, Any]]:
    """A tier's **logical** rows: one per producer row, with its own interval and text.

    For a tier with no same-tier overlap this is the emitted annotation list, in the same order.
    For a projected tier it comes from `pipeline-overlap-projection`, which is the only place the
    producers' uncut intervals still exist. A test that wants "one bar per sighting" has to ask
    the metadata once a tier was re-cut, and asking the emitted annotations instead would make the
    assertion depend on whether some other producer happened to overlap.
    """
    emitted = annotations(eaf, tier)
    document = projection_of(eaf).get(tier)
    if document is None:
        return [{"start_ms": start, "end_ms": end, "text": text}
                for start, end, text in emitted]
    return list(document["logical"])


def logical_texts(eaf: Any, tier: str) -> list[str]:
    """Every logical row's text, including the rows a projection merged into a shared segment."""
    return [row["text"] for row in logical_rows(eaf, tier)]


class TestBuildEaf:
    """The fifteen tiers, built from real Parquet and read back out of real XML."""

    def test_only_the_tiers_with_input_files_are_present(self, dataset: dict[str, Path]) -> None:
        eaf, _report = build_eaf(dataset["dir"], dataset["video"], log=lambda *a, **k: None)
        assert tiers_of(eaf) == ["words", "turns_pyannote", "asd_speaking", "face_tracks",
                                 "pose_presence"]
        assert set(tier_counts(eaf)) == set(tiers_of(eaf))

    def test_a_tier_without_input_is_skipped_with_one_logged_reason(self,
                                                                    dataset: dict[str, Path]
                                                                    ) -> None:
        """Absence is reported, never promised — the manifest's rule, applied to the .eaf.

        One line per missing tier, naming the file: translation, persons and the acoustics
        are off or absent for most datasets, so a warning per tier per video would be noise
        while silence would be a lost signal.
        """
        lines: list[str] = []
        build_eaf(dataset["dir"], dataset["video"],
                  log=lambda *a, **k: lines.append(str(a[0])))[0]
        # "skipped (" and not bare "skipped": the closing census line also reports the count,
        # and counting it would make this assertion pass at seven tiers or at seventy.
        # Ten: fifteen tiers minus the five producers this clip writes. The four Praat tiers are
        # four lines naming the *same* missing file, which is what four tiers over one table
        # looks like when it is absent — and the linguistic tiers are gone from this count
        # because no tier reads their tables any more (they are on disk here and unread).
        skipped = [line for line in lines if "skipped (" in line]
        assert len(skipped) == 10, skipped
        assert len([line for line in skipped
                    if "acoustic/frame_features.parquet not produced" in line]) == 4, skipped
        assert not [line for line in skipped if "linguistic/" in line], skipped
        assert sum(1 for line in skipped if "gloss_en" in line) == 1
        assert any("translation/segments_en.parquet not produced" in line for line in skipped)
        # The census line agrees with the per-tier lines rather than restating a constant.
        assert any(line.startswith("elan: 5 tier(s)") and "skipped 10" in line for line in lines)

    def test_a_known_word_lands_on_the_expected_millisecond_pair(self,
                                                                dataset: dict[str, Path]
                                                                ) -> None:
        """The millisecond pairs are the point; the label is asserted as a prefix.

        The ids the label now carries come from `_word`'s fixture defaults, so they are pinned
        in `TestSegmentIdentityLinksTiers` rather than duplicated here — this test exists to
        catch a rounding or zero-width change, and a pair-only assertion keeps it that way.
        """
        got = annotations(eaf_of(dataset), "words")
        assert [(start, end) for start, end, _text in got] == [
            (0, 400),
            (500, 900),
            # Zero width in, +1 ms out: ELAN cannot hold (1000, 1000).
            (1000, 1001),
        ]
        assert [text.split(" · ")[0] for _s, _e, text in got] == [
            "hello", "multi line", "zero"]

    def test_a_newline_in_a_producer_string_cannot_reach_the_file(self,
                                                                 dataset: dict[str, Path]
                                                                 ) -> None:
        text = " ".join(value for _s, _e, value in annotations(eaf_of(dataset), "words"))
        assert "\n" not in text and "\r" not in text
        assert "multi line" in text

    def test_the_turn_text_names_the_engine_that_produced_it(self,
                                                             dataset: dict[str, Path]
                                                             ) -> None:
        """Two turn tables, two engine names, decided by the tier and not by the filename.

        The label leads with the word ``speaker`` because `SPEAKER_00` (pyannote) and
        ``speaker_0`` (Nemotron) are unrelated clusters over unrelated channels and TalkNet's
        ``track_id`` is a third namespace again (§20.2). A tier label is the one piece of the
        file that travels — pasted into an issue, quoted in a paper — without the tier header
        that says which engine wrote it, so the label carries the namespace marker itself.
        """
        assert annotations(eaf_of(dataset), "turns_pyannote") == [
            (0, 1000, "speaker SPEAKER_00 (pyannote, exclusive) · t-0")]

    def test_a_turn_label_carries_its_turn_id_so_the_fusion_tier_links_to_it(self,
                                                                            dataset: dict[str,
                                                                                          Path]
                                                                            ) -> None:
        """`turn_id` is the key both tables publish, so the labels can share it.

        The fusion row is built *from* a turn and carries the same `turn_id`; printed on both
        sides, it is the one legitimate link between a diarizer's turn and the A/V verdict on
        it — unlike a speaker id, which must never cross engines.
        """
        root = dataset["dir"]
        _write(SPEAKER_FUSION_SCHEMA, root / "speaker" / "fusion_pyannote.parquet",
               [_fusion_row("t-0", "SPEAKER_00", 0.0, 1.0)])
        eaf, _report = build_eaf(root, dataset["video"], log=lambda *a, **k: None)
        turn_id = annotations(eaf, "turns_pyannote")[0][2].rsplit("·", 1)[-1].strip()
        fused = annotations(eaf, "fusion_pyannote")[0][2]
        assert turn_id == "t-0"
        assert "turn t-0" in fused
    def test_face_to_no_face_to_face_collapses_into_three_blocks(self,
                                                                 dataset: dict[str, Path]
                                                                 ) -> None:
        """The behaviour the operator asked for: runs, not frames.

        Expected ends carry one ASD step (0.04 s) because a block's last frame was *sampled*
        at its end timestamp, so the interval covers that frame.
        """
        assert annotations(eaf_of(dataset), "asd_speaking") == [
            (0, 80, "speaking track 0"),
            (80, 160, "no face"),
            (160, 200, "speaking track 1"),
        ]

    def test_a_face_track_is_widened_by_the_step_of_its_own_frames_table(self,
                                                                        dataset: dict[str, Path]
                                                                        ) -> None:
        """The tracks table has no frame step, so the frames table supplies it.

        Without the step the tier would end 40 ms before the matching ``asd_speaking`` block
        and the two tiers would disagree about when a face left the screen.
        """
        assert annotations(eaf_of(dataset), "face_tracks") == [
            (0, 80, "track 0 · 2/2 act · mean 1.234")]

    def test_presence_blocks_are_split_by_a_low_confidence_frame(self,
                                                                dataset: dict[str, Path]
                                                                 ) -> None:
        """A frame below the confidence floor is a hole in coverage, not a bridged gap.

        Timed on the pose table's own 29.97 fps grid, which is not the ASD grid: frame 0 is
        0 ms, frame 1 is 33 ms, frame 2 (0.1 confidence) is 67 ms, frame 3 is 100 ms.
        """
        assert annotations(eaf_of(dataset), "pose_presence") == [
            (0, 67, "body present"),
            (100, 133, "body present"),
        ]

    def test_the_media_descriptor_carries_both_urls_and_they_both_resolve(self,
                                                                        dataset: dict[str, Path]
                                                                        ) -> None:
        """Absolute *and* relative, because ELAN needs both and neither alone is enough.

        The relative form is the one that survives a move, so it is asserted by resolving it
        from the directory the .eaf is written into — the base ELAN uses — rather than by
        comparing strings. Resolving it from the dataset directory instead is the mistake that
        shipped first: the .eaf lives in ``<dataset>/elan``, so a path computed from
        ``<dataset>`` is short one ``../``, resolves to a sibling that has never existed, and
        still round-trips through pympi without a word of complaint.
        """
        eaf = eaf_of(dataset)
        descriptor = eaf.media_descriptors[0]
        video = dataset["video"].resolve()
        root = dataset["dir"].resolve()
        assert descriptor["MEDIA_URL"] == video.as_uri()
        relative = descriptor["RELATIVE_MEDIA_URL"]
        assert relative == "../../../input_videos/clip.mp4"
        # Where ELAN actually resolves it: the .eaf's own directory, from the registry.
        eaf_dir = root / eaf_directory()
        assert (eaf_dir / relative).resolve() == video
        # ... and the wrong base now resolves to something that is not the video, so this test
        # dies if the base ever moves back to the dataset directory.
        assert (root / relative).resolve() != video
        assert descriptor["MIME_TYPE"] == "video/mp4"
        assert int(descriptor["TIME_ORIGIN"]) == 0

    @pytest.mark.parametrize("suffix,mimetype", [(".mp4", "video/mp4"),
                                                 (".mov", "video/quicktime"),
                                                 (".m4v", "video/x-m4v")])
    def test_the_mimetype_follows_the_suffix(self, tmp_path: Path, suffix: str,
                                             mimetype: str) -> None:
        """pympi's own guess table has no video types, so leaving it to the library raises."""
        root = tmp_path / "processed" / "clip"
        video = tmp_path / "input_videos" / f"clip{suffix}"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"stub")
        _write(WORDS_SCHEMA, root / "speech" / "words.parquet", [_word("hi", 0.0, 0.2)])
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        assert eaf.media_descriptors[0]["MIME_TYPE"] == mimetype

    def test_an_unknown_suffix_still_produces_a_well_formed_descriptor(self,
                                                                     tmp_path: Path) -> None:
        """An unknown container costs the declared type, never a lie and never the export.

        The empty string serialises as ``MIME_TYPE=""`` and round-trips; ELAN plays off the
        extension either way. Naming video/mp4 for a .mkv would describe a container nobody
        measured — the corpus's person_demo.avi turned out to be a QuickTime container, which
        is how this rule earned its test.
        """
        root = tmp_path / "processed" / "clip"
        video = tmp_path / "videos" / "clip.mkv"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"stub")
        _write(WORDS_SCHEMA, root / "speech" / "words.parquet", [_word("hi", 0.0, 0.2)])
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        assert eaf.media_descriptors[0]["MIME_TYPE"] == UNKNOWN_MIME_TYPE
        assert eaf.media_descriptors[0]["MIME_TYPE"] != "video/mp4"
        # and the document still writes and parses with the empty type
        out = tmp_path / "clip.eaf"
        eaf.to_file(str(out))
        ET.parse(out)

    def test_the_relative_url_resolves_when_the_eaf_sits_at_its_registered_path(
            self, tmp_path: Path) -> None:
        """End to end, at the path the registry actually writes to.

        The unit test above resolves a string built in memory; this one writes the file where
        the pipeline writes it (``<dataset>/elan/annotations.eaf``) and opens it again, which is
        the only shape that can catch a base-directory mistake: the off-by-one version passed
        every in-memory check because nothing between `build_eaf` and the assertion ever put the
        .eaf on disk where it belongs.
        """
        from pympi.Elan import Eaf

        from multimodal_pipeline.artifacts import ARTIFACT_LAYOUT

        dataset = make_dataset(tmp_path)
        root, video = dataset["dir"], dataset["video"]
        out = root / ARTIFACT_LAYOUT["elan_annotations"]
        out.parent.mkdir(parents=True, exist_ok=True)
        build_eaf(root, video, log=lambda *a, **k: None)[0].to_file(str(out))

        reopened = Eaf(str(out))
        descriptor = reopened.media_descriptors[0]
        from_xml = (out.parent / descriptor["RELATIVE_MEDIA_URL"]).resolve()
        assert from_xml == video.resolve()
        assert from_xml.is_file()

    def test_the_relative_url_follows_the_artifact_when_the_registry_moves_it(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The base is derived from the registry, so base and destination cannot drift.

        Moving the artifact and leaving the base behind is the same off-by-N directory error in a
        different costume: every file already written would link a path that no longer resolves.
        Repointing ``ARTIFACT_LAYOUT`` is how a future refactor would move the .eaf, so the
        relative URL has to follow it. Measured: with the artifact at ``nested/deeper``, the URL
        becomes ``../../../../input_videos/clip.mp4`` and still resolves.
        """
        from pympi.Elan import Eaf

        from multimodal_pipeline import artifacts
        from multimodal_pipeline.artifacts import ARTIFACT_LAYOUT

        dataset = make_dataset(tmp_path)
        root, video = dataset["dir"], dataset["video"]
        monkeypatch.setitem(ARTIFACT_LAYOUT, "elan_annotations", "nested/deeper/annotations.eaf")
        assert eaf_directory() == "nested/deeper"

        out = root / ARTIFACT_LAYOUT["elan_annotations"]
        out.parent.mkdir(parents=True, exist_ok=True)
        build_eaf(root, video, log=lambda *a, **k: None)[0].to_file(str(out))
        rel = Eaf(str(out)).media_descriptors[0]["RELATIVE_MEDIA_URL"]
        assert rel == "../../../../input_videos/clip.mp4"
        assert (out.parent / rel).resolve() == video.resolve()
        monkeypatch.undo()
        assert artifacts.ARTIFACT_LAYOUT["elan_annotations"] == "elan/annotations.eaf"

    def test_one_nan_timestamp_costs_its_row_and_not_the_tier(self,
                                                              tmp_path: Path) -> None:
        """Review finding R4-nan-timestamp-aborts-export, reproduced before it was fixed.

        `interval_ms` and `add_annotation` run *after* the per-tier try/except, so before this
        test existed a single non-finite timestamp raised out of `build_eaf` and lost the
        whole export — every other tier included. A NaN in a timestamp column is producible
        upstream (a division nobody checked), so "impossible" is not an answer; the tier's
        other rows were measured and belong in the file, dropped ones are counted in the log.
        """
        root = tmp_path / "processed" / "clip"
        video = tmp_path / "input_videos" / "clip.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"stub")
        good = _word("hello", 0.0, 0.4)
        bad = _word("nan-word", float("nan"), 1.0)
        _write(WORDS_SCHEMA, root / "speech" / "words.parquet", [good, bad])
        _write(SPEAKER_TURNS_SCHEMA, root / "speech" / "speaker_turns.parquet", [{
            "schema_version": "1.0", "video_id": "clip", "turn_id": "t-0",
            "speaker_id": "SPEAKER_00", "start_time": 0.0, "end_time": 1.0, "duration": 1.0,
            "diarization_type": "exclusive",
        }])
        lines: list[str] = []
        eaf, _report = build_eaf(root, video, log=lambda msg, *a, **k: lines.append(str(msg)))
        # The other tier survived, and the good row of the poisoned tier survived with it.
        assert [(start, end) for start, end, _value in annotations(eaf, "words")] == [(0, 400)]
        assert annotations(eaf, "words")[0][2].startswith("hello")
        assert len(annotations(eaf, "turns_pyannote")) == 1
        assert [line for line in lines if "dropped 1 of 2" in line and "words" in line], lines
        # And the census in the file says one word, not two and not zero. Read by name:
        # pympi puts its own `lastUsedAnnotation` property first, so an index would be a
        # claim about pympi's internals rather than about this file's promise.
        census = dict(eaf.properties)["pipeline-tiers"]
        assert census == "turns_pyannote=1 words=1"

    def test_a_tier_whose_rows_are_all_non_finite_is_empty_not_fatal(self,
                                                                      tmp_path: Path) -> None:
        """The extreme of the same rule: a tier may end with zero annotations.

        An empty tier is a different claim from an absent one (absent means the producer never
        ran, and that is logged as a skip) — so it still gets its tier element and its own
        drop line, and the export keeps going.
        """
        root = tmp_path / "processed" / "clip"
        video = tmp_path / "input_videos" / "clip.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"stub")
        _write(WORDS_SCHEMA, root / "speech" / "words.parquet", [
            _word("a", float("nan"), 1.0), _word("b", 2.0, float("inf"))])
        lines: list[str] = []
        eaf, _report = build_eaf(root, video, log=lambda msg, *a, **k: lines.append(str(msg)))
        assert annotations(eaf, "words") == []
        assert "words" in tiers_of(eaf)
        assert [line for line in lines if "dropped 2 of 2" in line], lines

    def test_a_word_with_no_time_is_dropped_rather_than_placed_at_zero(self,
                                                                       tmp_path: Path
                                                                       ) -> None:
        """Word timing cannot invent t0 either — the guard lives in `build_eaf`, not in one tier.

        The independent verifier's finding, generalised: a null `end_time` used to serialise as
        an annotation over ``[0, 1)`` ms, i.e. the word was reported as spoken at the very start
        of the clip. A null *start* landed at 0 the same way. Both are dropped, the good sibling
        row is kept, and the count is logged separately from the non-finite one, so "the producer
        wrote no time" and "the producer wrote a NaN" stay two states in the run output exactly
        as they are two states in the table.
        """
        root = tmp_path / "processed" / "clip"
        video = tmp_path / "input_videos" / "clip.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"stub")
        _write(WORDS_SCHEMA, root / "speech" / "words.parquet", [
            _word("hello", 0.0, 0.4),
            _word("no-end", 2.0, None),
            _word("no-start", None, 3.0),
        ])
        _write(SPEAKER_TURNS_SCHEMA, root / "speech" / "speaker_turns.parquet", [{
            "schema_version": "1.0", "video_id": "clip", "turn_id": "t-0",
            "speaker_id": "SPEAKER_00", "start_time": 0.0, "end_time": 1.0, "duration": 1.0,
            "diarization_type": "exclusive",
        }])
        lines: list[str] = []
        eaf, _report = build_eaf(root, video, log=lambda msg, *a, **k: lines.append(str(msg)))
        assert [(start, end) for start, end, _t in annotations(eaf, "words")] == [(0, 400)]
        # A sibling tier is untouched, and the census counts only the row really placed.
        assert len(annotations(eaf, "turns_pyannote")) == 1
        assert dict(eaf.properties)["pipeline-tiers"] == "turns_pyannote=1 words=1"
        missing = [line for line in lines if "missing timestamp" in line]
        assert [line for line in missing if "words" in line and "2 of 3" in line], lines
        # The two states are not merged into one line.
        assert not [line for line in lines if "non-finite" in line], lines

    def test_a_missing_and_a_non_finite_timestamp_are_counted_apart(self,
                                                                    tmp_path: Path
                                                                    ) -> None:
        """Both drop a row; only one of them is "the producer never wrote a time".

        Collapsing them into one line would make a table with null times read like a table full
        of NaNs — the same state collapse B1 refused for scores.
        """
        root = tmp_path / "processed" / "clip"
        video = tmp_path / "input_videos" / "clip.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"stub")
        _write(WORDS_SCHEMA, root / "speech" / "words.parquet", [
            _word("good", 0.0, 0.4),
            _word("null-end", 1.0, None),
            _word("nan-start", float("nan"), 4.0),
        ])
        lines: list[str] = []
        eaf, _report = build_eaf(root, video, log=lambda msg, *a, **k: lines.append(str(msg)))
        assert [start for start, _e, text in annotations(eaf, "words")
                if text.startswith("good")] == [0]
        assert [line for line in lines if "words" in line and "1 of 3" in line
                and "missing timestamp" in line], lines
        assert [line for line in lines if "words" in line and "1 of 3" in line
                and "non-finite" in line], lines

    @pytest.mark.parametrize("start,end", [
        (-2.0, -1.0),
        (-0.5, 0.5),
        (-3.0, -1.0),
    ], ids=["both-negative", "straddles-zero", "wide-negative"])
    def test_a_word_with_a_materially_negative_time_is_dropped_and_not_placed_at_zero(
            self, tmp_path: Path, start: float, end: float) -> None:
        """The non-linguistic tiers laundered a negative into a bar at second zero.

        B2 put the refusal in :func:`interval_ms`, but the check only covered *nulls*: a negative
        endpoint fell through to :func:`seconds_to_ms` and its clamp, so ``(-2.0, -1.0)`` reopened
        as a ``[0, 1)`` ms word and ``(-0.5, 0.5)`` as a half-second bar starting before the clip,
        with **no** drop log at all — the tier looked complete. The clamp is right for a converter
        and wrong for a placement: ELAN cannot tell that bar from one measured at t=0. Both shapes
        are now refused on the existing missing-timestamp line.
        """
        root, video = _clip(tmp_path)
        _write(WORDS_SCHEMA, root / "speech" / "words.parquet", [
            _word("good", 0.0, 0.4),
            _word("negative", start, end),
        ])
        lines: list[str] = []
        eaf, _report = build_eaf(root, video, log=lambda msg, *a, **k: lines.append(str(msg)))
        assert [(s, e) for s, e, _t in annotations(eaf, "words")] == [(0, 400)], (start, end)
        assert [line for line in lines if "words" in line and "1 of 2" in line
                and "missing timestamp" in line], (start, end, lines)

    def test_a_word_time_inside_rounding_noise_of_zero_is_kept_and_clamped(self,
                                                                          tmp_path: Path
                                                                          ) -> None:
        """The legitimate half of the rule: `-1e-6` at t=0 is noise, and the row is real.

        A producer that subtracts an offset emits such a value for its first frame; dropping it
        would cost a measured word to buy a sign bit nobody read. It lands at 0 ms, which is where
        the millisecond grid puts it anyway.
        """
        root, video = _clip(tmp_path)
        _write(WORDS_SCHEMA, root / "speech" / "words.parquet", [
            _word("first", -1e-06, 0.4),
        ])
        lines: list[str] = []
        eaf, _report = build_eaf(root, video, log=lambda msg, *a, **k: lines.append(str(msg)))
        assert [(s, e) for s, e, _t in annotations(eaf, "words")] == [(0, 400)]
        assert not [line for line in lines if "missing timestamp" in line], lines

    def test_a_negative_endpoint_is_refused_on_a_second_tier_and_its_sibling_survives(
            self, tmp_path: Path) -> None:
        """The refusal sits in the export's one time path, so no tier needs its own guard.

        Written against the turn tier rather than a copy of the word test: the point is that
        `build_eaf` — not a builder — decides, which is what keeps a future tier from
        re-introducing the clamp by omission. One bad row costs one bar; the tier stays.
        """
        root, video = _clip(tmp_path)
        _write(WORDS_SCHEMA, root / "speech" / "words.parquet", [_word("hello", 0.0, 0.4)])
        _write(SPEAKER_TURNS_SCHEMA, root / "speech" / "speaker_turns.parquet", [
            {"schema_version": "1.0", "video_id": "clip", "turn_id": "t-0",
             "speaker_id": "SPEAKER_00", "start_time": 0.0, "end_time": 1.0, "duration": 1.0,
             "diarization_type": "exclusive"},
            {"schema_version": "1.0", "video_id": "clip", "turn_id": "t-1",
             "speaker_id": "SPEAKER_00", "start_time": -4.0, "end_time": -2.0,
             "duration": 2.0, "diarization_type": "exclusive"},
        ])
        lines: list[str] = []
        eaf, _report = build_eaf(root, video, log=lambda msg, *a, **k: lines.append(str(msg)))
        assert [(s, e) for s, e, _t in annotations(eaf, "turns_pyannote")] == [(0, 1000)]
        assert [line for line in lines if "turns_pyannote" in line and "1 of 2" in line
                and "missing timestamp" in line], lines
        assert dict(eaf.properties)["pipeline-tiers"] == "turns_pyannote=1 words=1"

    def test_the_document_is_well_formed_xml_and_reloads(self, dataset: dict[str, Path]
                                                         ) -> None:
        """ELAN reads bytes, not this object: write, parse, and read it back."""
        eaf = eaf_of(dataset)
        path = dataset["dir"] / "annotations.eaf"
        eaf.to_file(str(path))
        root = ET.parse(path).getroot()
        assert root.tag.endswith("ANNOTATION_DOCUMENT")
        names = sorted(real_tier_names(root))
        assert names == sorted(["words", "turns_pyannote", "asd_speaking", "face_tracks",
                                "pose_presence"])
        assert list(root.iter("MEDIA_DESCRIPTOR"))
        from pympi.Elan import Eaf

        reloaded = Eaf(str(path), suppress_version_warning=True)
        assert sorted(tiers_of(reloaded)) == sorted(names)
        assert tier_counts(reloaded) == tier_counts(eaf)

    def test_one_unreadable_table_costs_its_own_tier_and_not_the_other_four(self,
                                                                           dataset: dict[str, Path]
                                                                           ) -> None:
        """The asymmetry with `enabled`: a corrupt table is a skipped tier, not a failed video.

        A half-good .eaf and a logged line is worth more to a user than no .eaf, and the file
        still parses — so this has to stay a skip rather than a crash.
        """
        dataset["dir"].joinpath("pose/body.parquet").write_bytes(b"not parquet at all")
        lines: list[str] = []
        eaf, _report = build_eaf(dataset["dir"], dataset["video"],
                                 log=lambda *a, **k: lines.append(str(a[0])))
        assert "pose_presence" not in tiers_of(eaf)
        assert any("pose_presence skipped" in line and "unreadable" in line for line in lines)
        assert {"words", "asd_speaking", "face_tracks", "turns_pyannote"} <= set(tiers_of(eaf))

    def test_the_tier_census_in_the_document_matches_the_document(self,
                                                                 dataset: dict[str, Path]
                                                                 ) -> None:
        recorded = dict(eaf_of(dataset).properties)["pipeline-tiers"]
        counts = tier_counts(eaf_of(dataset))
        assert recorded == " ".join(f"{name}={counts[name]}" for name in sorted(counts))


# --------------------------------------------------- identity, state and semantics (B1)


class TestSegmentIdentityLinksTiers:
    """Every text tier carries the producer's own id, so a reader can link across tiers.

    The first twelve tiers used to print prose only: a word, a source segment, an English segment.
    Nothing in the file said which segment a word belonged to or which source line a
    translation answered, so linking them in ELAN meant eyeballing timestamps — and the ids
    that do the linking already exist in the tables (`WORDS_SCHEMA.segment_id` / `word_id`,
    `TRANSLATION_SCHEMA.segment_id`). Printing them is a display change, not a new measurement.
    """

    def _three_tier_clip(self, tmp_path: Path) -> Path:
        root = tmp_path / "processed" / "clip"
        video = tmp_path / "input_videos" / "clip.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"stub")
        _write(WORDS_SCHEMA, root / "speech" / "words.parquet", [
            _word("Hola", 0.0, 0.4, segment_id="seg000001", word_id="seg000001-w00000"),
            _word("mundo", 0.5, 0.9, segment_id="seg000001", word_id="seg000001-w00001"),
            _word("adios", 1.0, 1.4, segment_id="seg000002", word_id="seg000002-w00000"),
        ])
        _write(SEGMENTS_SCHEMA, root / "speech" / "segments.parquet", [
            _segment("seg000001", 0.0, 0.9, "Hola mundo"),
            _segment("seg000002", 1.0, 1.4, "Adios", speaker_id="SPEAKER_01"),
        ])
        _write(TRANSLATION_SCHEMA, root / "translation" / "segments_en.parquet", [
            _translation("seg000001", 0.0, 0.9, "Hello world"),
            _translation("seg000002", 1.0, 1.4, "Goodbye", speaker_id="SPEAKER_01"),
        ])
        return root

    def _build(self, tmp_path: Path) -> Any:
        root = self._three_tier_clip(tmp_path)
        return build_eaf(root, tmp_path / "input_videos" / "clip.mp4",
                         log=lambda *a, **k: None)[0]

    def test_a_word_names_its_speaker_and_its_own_id_and_its_segment(self,
                                                                     tmp_path: Path
                                                                     ) -> None:
        """The word stays first: the tier must remain readable at a glance.

        Ids are what link the tiers, but a tier whose every row begins with
        `seg000001-w00000` is a tier nobody reads. The text leads and the ids trail — which is
        also why the ids are appended to the existing label rather than the row becoming a JSON
        dump of the row.
        """
        got = annotations(self._build(tmp_path), "words")
        assert [text for _s, _e, text in got] == [
            "Hola · SPEAKER_00 · seg000001-w00000 · [seg000001]",
            "mundo · SPEAKER_00 · seg000001-w00001 · [seg000001]",
            "adios · SPEAKER_00 · seg000002-w00000 · [seg000002]",
        ]

    def test_the_segment_tier_leads_with_the_speaker_then_names_the_segment(self,
                                                                           tmp_path: Path
                                                                           ) -> None:
        got = annotations(self._build(tmp_path), "segments_src")
        assert [text for _s, _e, text in got] == [
            "SPEAKER_00: Hola mundo · [seg000001]",
            "SPEAKER_01: Adios · [seg000002]",
        ]

    def test_the_translation_names_the_segment_it_answers_and_who_spoke(self,
                                                                       tmp_path: Path
                                                                       ) -> None:
        """`gloss_en` is keyed by segment_id in the schema; the tier used to drop both it and
        the speaker, so two segments of one translation were indistinguishable in ELAN.
        """
        got = annotations(self._build(tmp_path), "gloss_en")
        assert [text for _s, _e, text in got] == [
            "SPEAKER_00: Hello world · [seg000001]",
            "SPEAKER_01: Goodbye · [seg000002]",
        ]

    def test_a_missing_id_is_printed_as_unknown_rather_than_none_or_blank(self,
                                                                        tmp_path: Path
                                                                        ) -> None:
        """A producer that wrote no id is a missing measurement, not the string ``None``.

        `finalization` requires every translation row to resolve to a transcript segment, but
        words from an older alignment path can lack one, and `"None"` in a tier label reads as
        a rendering bug while `unknown` reads as the state it is.
        """
        root = tmp_path / "processed" / "clip"
        video = tmp_path / "input_videos" / "clip.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"stub")
        _write(WORDS_SCHEMA, root / "speech" / "words.parquet", [
            _word("hello", 0.0, 0.4, segment_id=None, word_id=None)])
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        assert annotations(eaf, "words") == [
            (0, 400, "hello · SPEAKER_00 · unknown · [unknown]")]

    def test_a_null_speaker_is_printed_as_unknown_on_both_segment_tiers(self,
                                                                       tmp_path: Path
                                                                       ) -> None:
        """`speaker_id` is nullable in both schemas, and the tier used to print the word.

        Diarization can be off, or a segment can be left unassigned, and then
        `SEGMENTS_SCHEMA.speaker_id` / `TRANSLATION_SCHEMA.speaker_id` are null. The ids on the
        same label already went through :func:`_id`; the speaker did not, so one row could read
        `hello · SPEAKER_00 · unknown · [unknown]` on one tier and `None: …` on another. A
        leading `None:` reads as a rendering bug, and it is the one part of the label an
        analyst quotes.
        """
        root = tmp_path / "processed" / "clip"
        video = tmp_path / "input_videos" / "clip.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"stub")
        _write(SEGMENTS_SCHEMA, root / "speech" / "segments.parquet",
               [_segment("seg000001", 0.0, 0.9, "Hola mundo", speaker_id=None)])
        _write(TRANSLATION_SCHEMA, root / "translation" / "segments_en.parquet",
               [_translation("seg000001", 0.0, 0.9, "Hello world", speaker_id=None)])
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        assert [text for _s, _e, text in annotations(eaf, "segments_src")] == [
            "unknown: Hola mundo · [seg000001]"]
        assert [text for _s, _e, text in annotations(eaf, "gloss_en")] == [
            "unknown: Hello world · [seg000001]"]

    def test_the_segment_id_survives_the_link_across_the_two_text_tiers(self,
                                                                       tmp_path: Path
                                                                       ) -> None:
        """The property the ids exist for: a word and the segment it sits in agree by name.

        Asserted as a set relationship rather than two independent label checks, so a tier
        that printed ids from some other column would fail here even if its own labels still
        looked well formed.
        """
        eaf = self._build(tmp_path)
        word_segments = {text.rsplit("[", 1)[-1].rstrip("]")
                         for _s, _e, text in annotations(eaf, "words")}
        segment_ids = {text.rsplit("[", 1)[-1].rstrip("]")
                       for _s, _e, text in annotations(eaf, "segments_src")}
        translation_ids = {text.rsplit("[", 1)[-1].rstrip("]")
                           for _s, _e, text in annotations(eaf, "gloss_en")}
        assert word_segments == {"seg000001", "seg000002"}
        assert segment_ids == translation_ids == word_segments


class TestNamespaceInLabels:
    """Which ids are separate namespaces, and which one is a real link."""

    def _fusion_clip(self, tmp_path: Path) -> Path:
        root = tmp_path / "processed" / "clip"
        video = tmp_path / "input_videos" / "clip.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"stub")
        _write(WORDS_SCHEMA, root / "speech" / "words.parquet",
               [_word("hola", 0.0, 0.4)])
        _write(SPEAKER_FUSION_SCHEMA, root / "speaker" / "fusion_pyannote.parquet", [
            _fusion_row("turn000001", "SPEAKER_00", 0.0, 1.0),
            _fusion_row("turn000002", "SPEAKER_01", 1.0, 2.0, agreement="face_never_active",
                        detail="no track active in window", face_track_id=None),
        ])
        return root

    def test_a_fusion_label_says_which_face_track_without_equating_it_to_the_speaker(
            self, tmp_path: Path) -> None:
        """The winning face track is a TalkNet id, and the schema says so twice.

        `SPEAKER_FUSION_SCHEMA.face_track_id` is documented as "An ASD track id, not a speaker
        id", and the turn's `speaker_id` is pyannote's namespace. Printing both in one label
        is only safe when the label says which is which — the same argument §20.2 makes for
        `person_id` versus `track_id`, applied to the tier a human actually reads. The
        verdict's own arithmetic (`agreement_detail`) is kept verbatim after them.
        """
        root = self._fusion_clip(tmp_path)
        eaf, _report = build_eaf(root, tmp_path / "input_videos" / "clip.mp4",
                             log=lambda *a, **k: None)
        assert [text for _s, _e, text in annotations(eaf, "fusion_pyannote")] == [
            "face_matched: turn turn000001 · turn speaker SPEAKER_00 (pyannote) | "
            "face track 0 | track 0 active on 4/5 frames",
            "face_never_active: turn turn000002 · turn speaker SPEAKER_01 (pyannote) | "
            "face track unknown | no track active in window",
        ]

    def test_the_fusion_face_track_id_is_the_face_tracks_tier_track_id(self,
                                                                      tmp_path: Path
                                                                      ) -> None:
        """`face_track_id` is not a fourth namespace — it *is* TalkNet's `track_id`.

        Measured here rather than asserted from prose: `fusion.fuse_turn_table` copies the
        winning frame's `track_id` straight through, so on every corpus dataset the fusion
        tables' `face_track_id` values are a subset of `active_speaker_tracks.track_id`
        (KABC {0} and {0,1} ⊆ [0, 1]; La-1 {0, 4} ⊆ [0, 1, 2, 4]). Printing it as if it were
        another unrelated space would hide the one legitimate link this tier has to the
        `face_tracks` tier, so the link is asserted the way a reader uses it: the id in the
        fusion label names an annotation that really exists on the face-track tier.
        """
        root = tmp_path / "processed" / "clip"
        video = tmp_path / "input_videos" / "clip.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"stub")
        _write(ACTIVE_SPEAKER_TRACKS_SCHEMA, root / "speaker" / "active_speaker_tracks.parquet",
               [{"schema_version": "1.0", "video_id": "clip", "track_id": 4,
                 "first_timestamp": 0.0, "last_timestamp": 0.04, "frame_count": 3,
                 "active_frame_count": 2, "active_ratio": 0.66, "mean_score": 1.5,
                 "max_score": 2.0, "scenes": [1], "mean_bbox_area": 10.0}])
        _write(SPEAKER_FUSION_SCHEMA, root / "speaker" / "fusion_pyannote.parquet",
               [_fusion_row("turn000001", "SPEAKER_00", 0.0, 1.0, face_track_id=4)])
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)

        fused = annotations(eaf, "fusion_pyannote")[0][2]
        face_track = fused.split("face track ", 1)[1].split(" ", 1)[0]
        track_tier = {text.split(" ")[1]
                      for _s, _e, text in annotations(eaf, "face_tracks")}
        assert face_track == "4"
        assert face_track in track_tier, "fusion names a face track the face-track tier lacks"
        # And the document says so, rather than lumping it in with the namespaces that really
        # are separate (§20.2 is about diarizer speaker vs TalkNet face vs YOLO person).
        semantics = dict(eaf.properties)["pipeline-tier-semantics"]
        assert "same" in semantics and FACE_TRACK_NS in semantics
        assert "track_id" in semantics

    def test_the_fusion_label_names_the_engine_from_its_own_row(self, tmp_path: Path) -> None:
        """The engine is read from `engine`, not inferred from the tier name or an id.

        A fusion label is the part of the file that gets quoted on its own, and a tier header
        does not travel with it: quoted out of `fusion_nemotron`, a bare `turn speaker
        speaker_0` is indistinguishable from pyannote's `SPEAKER_00` apart from its shape.
        `SPEAKER_FUSION_SCHEMA.engine` already records which diarizer wrote the row and which
        speaker-id namespace it came from, so the label prints that column — which is why a row
        whose `engine` disagrees with the file it sits in prints the row's own value, and a null
        one prints `unknown` rather than whatever the tier is called.
        """
        root = tmp_path / "processed" / "clip"
        video = tmp_path / "input_videos" / "clip.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"stub")
        _write(SPEAKER_FUSION_SCHEMA, root / "speaker" / "fusion_pyannote.parquet", [
            _fusion_row("turn000001", "speaker_0", 0.0, 1.0, engine="nemotron"),
            _fusion_row("turn000002", "SPEAKER_00", 1.0, 2.0, engine=None),
        ])
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        assert [text for _s, _e, text in annotations(eaf, "fusion_pyannote")] == [
            "face_matched: turn turn000001 · turn speaker speaker_0 (nemotron) | "
            "face track 0 | track 0 active on 4/5 frames",
            "face_matched: turn turn000002 · turn speaker SPEAKER_00 (unknown) | "
            "face track 0 | track 0 active on 4/5 frames",
        ]
        assert ENGINE_NS in dict(eaf.properties)["pipeline-tier-semantics"]

    def test_a_turn_label_says_its_speaker_id_is_a_diarizer_label(self,
                                                                 dataset: dict[str, Path]
                                                                 ) -> None:
        """The namespace marker is the tier's job, and it is asserted on the *other* engine too.

        `turns_nemotron` reads a different file whose ids arrive-ordered (`speaker_0`), so the
        marker has to come from the builder and not from a prefix that happens to look right on
        pyannote's ``SPEAKER_00``.
        """
        root = dataset["dir"]
        _write(SPEAKER_TURNS_NEMOTRON_SCHEMA, root / "speech" / "speaker_turns_nemotron.parquet",
               [{"schema_version": "1.0", "video_id": "clip", "turn_id": "nt-1",
                 "speaker_id": "speaker_0", "start_time": 0.0, "end_time": 0.5,
                 "duration": 0.5, "diarization_type": "overlapping", "overlap_s": 0.2}])
        eaf, _report = build_eaf(root, dataset["video"], log=lambda *a, **k: None)
        assert annotations(eaf, "turns_nemotron") == [
            (0, 500, "speaker speaker_0 (nemotron, overlapping) · nt-1")]


class TestUnknownIsNotZero:
    """A missing number used to print as a measured zero. That is the one bug in this file
    that puts a false measurement in front of an analyst."""

    @pytest.mark.parametrize("places", [0, 2, 3])
    def test_a_missing_number_says_unknown_at_every_precision(self, places: int) -> None:
        assert elan_core._num(None, places) == "unknown"

    @pytest.mark.parametrize("value,places,expected", [
        (0.0, 3, "0.000"),          # an actual zero stays a zero — the other half of the rule
        (0.0, 2, "0.00"),
        (1.2345, 3, "1.234"),
        (-0.0004, 3, "-0.000"),     # rounds to zero, still prints as a number
        (2.506753, 2, "2.51"),
    ])
    def test_a_finite_number_still_prints_rounded(self, value: float, places: int,
                                                  expected: str) -> None:
        assert elan_core._num(value, places) == expected

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_number_says_unknown_rather_than_nan(self, bad: float) -> None:
        """A `nan` or `inf` in a tier label reads as a broken export; the row's number is not a
        measurement, so it is reported as unknown exactly like a null."""
        assert elan_core._num(bad, 3) == "unknown"

    def test_a_track_with_no_mean_score_is_unknown_not_zero(self, tmp_path: Path) -> None:
        """`mean_score` is nullable: a track can be reported without ever being scored."""
        root = tmp_path / "processed" / "clip"
        video = tmp_path / "input_videos" / "clip.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"stub")
        _write(ACTIVE_SPEAKER_TRACKS_SCHEMA, root / "speaker" / "active_speaker_tracks.parquet",
               [{"schema_version": "1.0", "video_id": "clip", "track_id": 4,
                 "first_timestamp": 0.0, "last_timestamp": 0.04, "frame_count": 3,
                 "active_frame_count": 0, "active_ratio": 0.0, "mean_score": None,
                 "max_score": None, "scenes": [1], "mean_bbox_area": 10.0}])
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        assert annotations(eaf, "face_tracks") == [
            (0, 40, "track 4 · 0/3 act · mean unknown")]

    def test_a_person_track_with_no_confidence_is_unknown_not_zero(self,
                                                                  tmp_path: Path
                                                                  ) -> None:
        """The track table's nullable `mean_confidence`, on the tier that now reads it.

        Rewritten with the frames and index tables B2 needs: the assertion is the same one B1
        added (a null confidence is `unknown`, never `0.000`), only the fixture grew.
        """
        root, video = person_clip(tmp_path)
        _persons(root, [_person_frame(0, 2)],
                 [_person_track(2, 0.0, 1.0, 5, mean_confidence=None)])
        _frame_index(root, [_frame_row(0, 0.0)])
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        _start, _end, text = annotations(eaf, "person_tracks")[0]
        assert "conf unknown" in text
        assert "conf 0.000" not in text


class TestAsdStates:
    """The ASD tier's states, including the two the current builder reads wrong."""

    def test_no_face_wins_over_a_stale_active_flag(self) -> None:
        """Already true and kept: a frame with no located face has no track to be active."""
        assert asd_label({"face_status": "no_face", "frame_reason": "no_face",
                          "is_active_speaker": True, "track_id": None}) == "no face"

    @pytest.mark.parametrize("reason", [
        "score_not_finite", "track_has_no_scores", "past_scored_tail",
        "tail_score_not_finite", "unknown",
    ])
    def test_a_face_that_could_not_be_scored_is_not_called_not_speaking(self,
                                                                       reason: str) -> None:
        """`tracked_unscored` means "face located, no measurement". Calling that "not
        speaking" is the collapse the ASD schema grew `face_status` to prevent: it turns
        missing evidence into negative evidence, which is the mistake §17 names.

        The five reasons are `stages.activespeaker.UNSCORED_FRAME_REASONS` — the closed set
        that carries no measurement — and `unknown` is in it on purpose.
        """
        row = {"face_status": "tracked_unscored", "frame_reason": reason,
               "is_active_speaker": False, "track_id": 3}
        assert asd_label(row) == ASD_NOT_EVALUATED

    @pytest.mark.parametrize("reason", [
        "score_not_finite", "track_has_no_scores", "past_scored_tail",
        "tail_score_not_finite", "unknown",
    ])
    def test_an_unscored_row_with_a_stale_active_flag_is_still_not_evaluated(
            self, reason: str) -> None:
        """The precedence that matters.

        `is_active_speaker` is a *derived* verdict: `ActiveSpeakerStage.validate` rejects a
        `tracked_unscored` row that is imputed or marked active, so such a row is a producer
        defect and its flag is stale. Reporting "speaking" from it would let a broken flag
        outlive the measurement it was derived from, so the measurement state wins.
        """
        row = {"face_status": "tracked_unscored", "frame_reason": reason,
               "is_active_speaker": True, "track_id": 3}
        assert asd_label(row) == ASD_NOT_EVALUATED

    def test_a_missing_cause_column_still_honours_tracked_unscored(self) -> None:
        """An old dataset carries `face_status` but no usable `frame_reason`.

        The compat shim in `ActiveSpeakerStage._frame_row` derives a missing reason; a tier
        that only read `frame_reason` would call those rows "not speaking".
        """
        assert asd_label({"face_status": "tracked_unscored", "frame_reason": None,
                          "is_active_speaker": False, "track_id": 3}) == ASD_NOT_EVALUATED

    def test_a_scored_face_that_is_not_active_is_still_not_speaking(self) -> None:
        """The distinction the whole class is about: this one *is* a measurement."""
        assert asd_label({"face_status": "tracked", "frame_reason": "scored",
                          "is_active_speaker": False, "track_id": 1}) == ASD_NOT_SPEAKING

    def test_an_imputed_tail_frame_says_so_rather_than_reading_as_raw(self) -> None:
        """`score_imputed` / `frame_reason='imputed_tail'` means the score was carried from the
        last real one. The verdict is the stage's, so the tier keeps it; but the label says the
        provenance, so nobody reads an extrapolation as a measurement."""
        label = asd_label({"face_status": "tracked", "frame_reason": "imputed_tail",
                           "score_imputed": True, "is_active_speaker": True,
                           "track_id": 2})
        assert label.startswith("speaking track 2")
        assert "imputed" in label

    def test_a_speaking_frame_carries_its_track_id(self) -> None:
        assert asd_label({"face_status": "tracked", "frame_reason": "scored",
                          "is_active_speaker": True, "track_id": 7}) == "speaking track 7"

    @pytest.mark.parametrize("is_active", [True, False])
    def test_an_imputed_score_says_so_whatever_the_activity_state(self,
                                                                 is_active: bool) -> None:
        """The suffix is a statement about the *number*, not about the verdict.

        The first version appended it only to "speaking", which is half the contract: the row
        still says the score was carried forward whether or not it cleared the threshold. Seven
        corpus rows are `imputed_tail` across four datasets and two of them are **not** active
        (La-1 frame 60 at 2.40 s, track 0, carried score −1.4667; `person_demo` frame 96 at
        3.84 s), and both printed a plain "not speaking", hiding that the measurement was an
        extrapolation.
        """
        label = asd_label({"face_status": "tracked", "frame_reason": "imputed_tail",
                           "score_imputed": True, "is_active_speaker": is_active,
                           "track_id": 0})
        assert label.endswith(ASD_IMPUTED_SUFFIX)
        expected = "speaking track 0" if is_active else "not speaking"
        assert label == expected + ASD_IMPUTED_SUFFIX

    def test_a_scored_inactive_frame_still_prints_the_plain_state(self) -> None:
        """The negative half: the suffix is earned by `score_imputed`, never by inactivity.

        If every inactive row carried it, the label would stop distinguishing a measured zero-
        verdict from a carried one, which is the whole reason for printing the provenance.
        """
        assert asd_label({"face_status": "tracked", "frame_reason": "scored",
                          "score_imputed": False, "is_active_speaker": False,
                          "track_id": 0}) == ASD_NOT_SPEAKING

    def test_the_three_states_collapse_into_three_blocks(self, dataset: dict[str, Path]
                                                        ) -> None:
        """The existing behaviour, re-pinned after the state split: a run of equal labels is
        still one block, and the two unscored readings must not merge with "not speaking"."""
        assert annotations(eaf_of(dataset), "asd_speaking") == [
            (0, 80, "speaking track 0"),
            (80, 160, "no face"),
            (160, 200, "speaking track 1"),
        ]

    def test_an_unscored_frame_breaks_a_speaking_block_rather_than_joining_it(self,
                                                                            tmp_path: Path
                                                                            ) -> None:
        """Three labels, three blocks: speaking / not evaluated / speaking.

        If "not evaluated" collapsed into "not speaking" the tier would report a silent middle
        section where the real state is "we have no idea".
        """
        root = tmp_path / "processed" / "clip"
        video = tmp_path / "input_videos" / "clip.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"stub")
        rows = [
            _asd_frame(0, "speaking", 0),
            _asd_frame(1, "speaking", 0),
            _asd_frame(1, "not_speaking", 0, face_status="tracked_unscored",
                       frame_reason="track_has_no_scores", talknet_score=None,
                       talknet_score_raw=None),
            _asd_frame(2, "speaking", 0),
        ]
        for index, row in enumerate(rows):  # unique timestamps after the copy above
            row["timestamp"] = round(index * STEP, 6)
            row["source_timestamp"] = row["timestamp"]
            row["frame_number"] = index
        _write(ACTIVE_SPEAKER_FRAMES_SCHEMA,
               root / "speaker" / "active_speaker_frames.parquet", rows)
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        assert annotations(eaf, "asd_speaking") == [
            (0, 80, "speaking track 0"),
            (80, 120, ASD_NOT_EVALUATED),
            (120, 160, "speaking track 0"),
        ]

    def test_an_inactive_imputed_frame_breaks_its_block_too(self, tmp_path: Path) -> None:
        """The La-1 shape, end to end: frames 59→60→61 are scored-inactive, imputed-inactive,
        scored-active.

        Reproduced from the real table (frame 60 at 2.40 s, `frame_reason='imputed_tail'`,
        `is_active_speaker=False`) rather than invented, because that is the row whose
        provenance the tier dropped: the block it sits in has to end where the measurement does.
        """
        root = tmp_path / "processed" / "clip"
        video = tmp_path / "input_videos" / "clip.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"stub")
        rows = [
            _asd_frame(0, "not_speaking", 0),
            _asd_frame(1, "not_speaking", 0, frame_reason="imputed_tail",
                       score_imputed=True, talknet_score=-1.4667),
            _asd_frame(2, "speaking", 0),
        ]
        _write(ACTIVE_SPEAKER_FRAMES_SCHEMA,
               root / "speaker" / "active_speaker_frames.parquet", rows)
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        assert annotations(eaf, "asd_speaking") == [
            (0, 40, "not speaking"),
            (40, 80, "not speaking" + ASD_IMPUTED_SUFFIX),
            (80, 120, "speaking track 0"),
        ]


class TestTierSemanticsProperty:
    """The file has to explain itself, because a tier label is read without the source."""

    def semantics(self, eaf: Any) -> str:
        return dict(eaf.properties)["pipeline-tier-semantics"]

    def test_the_document_states_how_to_read_the_tiers(self, dataset: dict[str, Path]
                                                      ) -> None:
        text = self.semantics(eaf_of(dataset))
        # Segment-level translation, not a word gloss: the tier is called `gloss_en` and the
        # word "gloss" means something else to every reader who opens this in ELAN.
        assert "gloss_en" in text and "segment-level" in text and "not word" in text
        # Ids are namespaced.
        assert SPEAKER_NS in text and WORD_NS in text and SEGMENT_NS in text
        # The TalkNet score is an unbounded logit-like number, not a probability.
        assert "logit" in text and "not a probability" in text
        # Missing numbers print as `unknown`, never 0.000.
        assert "unknown" in text
        # What is *not* in here, stated in the file rather than only in the README.
        assert "summary" in text and "not every number" in text

    def test_the_semantics_survive_the_round_trip_and_parse(self, dataset: dict[str, Path]
                                                          ) -> None:
        """Written, re-read from disk, and still well-formed XML.

        The value contains `&`, `<`, `>` and quotes on purpose (see TIER_SEMANTICS): pympi
        escapes attribute-free text nodes, and a property that broke the header would cost the
        whole file rather than one tier.
        """
        out = dataset["dir"] / "semantics.eaf"
        eaf_of(dataset).to_file(str(out))
        ET.parse(out)
        from pympi.Elan import Eaf

        reopened = Eaf(str(out), suppress_version_warning=True)
        assert self.semantics(reopened) == TIER_SEMANTICS

    def test_the_asd_clause_scopes_not_speaking_to_the_selected_face(self,
                                                                    dataset: dict[str, Path]
                                                                    ) -> None:
        """`not speaking` is evidence about *one face*, not about the audio.

        The clause used to end "only the last is evidence about silence", which is a claim the
        producer never made: the ASD row's `is_active_speaker` is TalkNet's verdict on the one
        track this frame selected, and the same second can carry an off-screen or out-of-frame
        voice — the fusion table's own `no_face_visible` detail names that case ("off-screen
        narrator or audio bed"). A reader who took the old wording literally would mark a block
        silent while a diarizer turn on the same clip was busy. So the property says whose
        mouth the verdict is about and says plainly that it is not an audio claim.
        """
        text = self.semantics(eaf_of(dataset))
        clause = text[text.index("ASD states:"):text.index("Blocks are half-open")]
        assert "not speaking" in clause
        assert "track" in clause or "face" in clause
        assert "not" in clause and "audio" in clause
        assert "evidence about silence" not in clause

    def test_the_engine_and_face_track_clauses_survive_the_round_trip(self,
                                                                     dataset: dict[str, Path]
                                                                     ) -> None:
        """The two clauses this correction added are checked on disk, not only in the constant.

        The semantics property is one long string written through pympi's XML text node; a
        character in it that the library mangles would otherwise show up as a silently reworded
        document rather than a failing build.
        """
        out = dataset["dir"] / "semantics-clauses.eaf"
        eaf_of(dataset).to_file(str(out))
        from pympi.Elan import Eaf

        text = self.semantics(Eaf(str(out), suppress_version_warning=True))
        assert ENGINE_NS in text
        assert FACE_TRACK_NS in text
        assert "same" in text

    def test_the_semantics_are_about_the_tiers_this_file_writes(
            self, dataset: dict[str, Path]) -> None:
        """No tier name in the property that the tier list does not contain.

        The property is the file's own description; naming a tier that was never written (a
        future B3/B4 tier, or one that was renamed) would make the document describe a file
        that does not exist.
        """
        text = self.semantics(eaf_of(dataset))
        declared = {spec.tier for spec in TIERS}
        named = {name for name in declared if name in text}
        assert named, "the semantics property names none of the tiers"
        # Nothing outside the tier list appears in the shape `<tier> =`.
        for token in re.findall(r"([a-z_]+)\s*=", text):
            assert token in declared, f"semantics describes an unknown tier: {token}"

    def test_the_block_clause_is_scoped_to_the_grid_based_tiers(self,
                                                               dataset: dict[str, Path]
                                                               ) -> None:
        """"the last sampled frame is inside the block" is true of three tiers and false of one.

        The clause used to be global. It is correct for `asd_speaking`, `pose_presence` and
        `voiced_blocks`, whose ends are extended by their own median grid step; it is false for
        `person_tracks`, whose runs deliberately end **at** the last sighting's measured PTS with
        no step added (see :func:`person_track_rows`). A reader who applied the global sentence to
        a person run would believe the file asserted the person was still on screen one grid step
        after the last frame that placed them. So the sentence names the tiers it describes, and
        the person clause says the opposite about its own intervals.
        """
        text = self.semantics(eaf_of(dataset))
        clause = text[text.index("Blocks are half-open"):text.index("Persons:")]
        for tier in ("asd_speaking", "pose_presence", "voiced_blocks"):
            assert tier in clause, f"{tier} is not named by the block clause"
        assert "person_tracks" in clause
        assert "no grid step" in clause
        person = text[text.index("Persons:"):text.index("Coverage:")]
        assert "ends at" in person and "last sighting" in person

    def test_the_person_clause_calls_the_mark_width_a_display_minimum(self,
                                                                     dataset: dict[str, Path]
                                                                     ) -> None:
        """1 ms is what ELAN can store, not what the sighting lasted.

        Kept as its own test because the wording is the claim: "minimum representable interval"
        says the width is a property of the format, and the clause says the interval spans the
        sightings' own endpoints rather than a duration.
        """
        person = self.semantics(eaf_of(dataset))
        person = person[person.index("Persons:"):person.index("Coverage:")]
        assert "minimum representable" in person
        assert "not a measured duration" in person
        assert "1 ms" in person

    def test_the_gap_clause_names_the_column_it_prints(self, dataset: dict[str, Path]
                                                       ) -> None:
        """The person gap is the producer's reported column, and the property says so by name.

        A label fragment that paraphrases a column ("max gap elapsed") invites a later reader to
        recompute it; naming `longest_gap_seconds` makes the source checkable in the table and
        makes an invented value visible as one.
        """
        text = self.semantics(eaf_of(dataset))
        assert "longest_gap_seconds" in text
        assert "as reported" in text

    def test_the_census_property_is_still_the_census(self, dataset: dict[str, Path]) -> None:
        """Adding a property must not disturb the one the stage validates against itself."""
        assert "pipeline-tiers" in dict(eaf_of(dataset).properties)


class TestColumnNamesAgainstSchemas:
    """The B1 columns come from the schema objects, not from this file's memory."""

    def test_the_b1_columns_exist_in_their_schemas(self) -> None:
        expected = {
            "speech_words": (WORDS_SCHEMA, ("start_time", "end_time", "word", "word_id",
                                            "segment_id")),
            "speech_segments": (SEGMENTS_SCHEMA, ("start_time", "end_time", "speaker_id",
                                                  "text", "segment_id")),
            "translation_segments": (TRANSLATION_SCHEMA, ("start_time", "end_time",
                                                          "english_text", "speaker_id",
                                                          "segment_id",
                                                          "translation_model")),
            "speaker_fusion_pyannote": (SPEAKER_FUSION_SCHEMA, ("start_time", "end_time",
                                                                "agreement",
                                                                "agreement_detail",
                                                                "speaker_id",
                                                                "face_track_id",
                                                                "engine")),
        }
        for artifact, (schema, columns) in expected.items():
            known = {field.name for field in schema}
            assert set(columns) <= known, f"{artifact}: {set(columns) - known}"


def real_tier_names(root: ET.Element) -> list[str]:
    """The document's tier names, minus the one pympi always writes.

    ``pympi`` emits a ``default`` TIER in every document (its implicit tier, which nothing here
    annotates). Asserting against the raw element list would expect six names for five tiers and
    would stop being able to see a tier that really went missing.
    """
    return [element.attrib["TIER_ID"] for element in root.iter("TIER")
            if element.attrib.get("TIER_ID") != "default"]


def eaf_of(dataset: dict[str, Path]) -> Any:
    return build_eaf(dataset["dir"], dataset["video"], log=lambda *a, **k: None)[0]


class TestTierRegistration:
    """Every tier reads a registered artifact; no file name is invented here."""

    def test_the_fifteen_tiers_are_the_ones_the_design_named(self) -> None:
        assert [spec.tier for spec in TIERS] == [
            "words", "segments_src", "gloss_en", "turns_pyannote", "turns_nemotron",
            "fusion_pyannote", "fusion_nemotron", "asd_speaking", "face_tracks",
            "person_tracks", "pose_presence",
            "voiced_blocks", "f0_blocks", "intensity_blocks", "formant_blocks"]

    def test_every_tier_input_is_a_registered_artifact(self) -> None:
        unregistered = sorted({spec.artifact for spec in TIERS} - set(ARTIFACT_LAYOUT))
        assert not unregistered, f"tiers read artifacts that do not exist: {unregistered}"

    def test_four_tiers_share_the_frame_table_and_everything_else_is_one_to_one(
            self) -> None:
        """One producer per tier, except the deliberate exception the Praat tiers are.

        The rule this replaces was absolute — "no two tiers read the same table" — and the
        withdrawal of the acoustic summary tier broke it in exactly one place: four tiers now
        block over ``acoustic_frames``, each reading a different column of it and answering a
        different question (voiced or not / how high / how loud / which vowel). Asserting the
        exception by name rather than by count means a *fifth* tier joining that table, or one
        of the four being pointed at a different file, both fail here.

        Secondary inputs stay outside the comparison: `person_tracks` reads its own table *and*
        two more, and those two are read by no other tier.
        """
        artifacts = [spec.artifact for spec in TIERS]
        shared = [name for name in set(artifacts) if artifacts.count(name) > 1]
        assert shared == ["acoustic_frames"], shared
        assert artifacts.count("acoustic_frames") == len(PRAAT_FRAME_TIERS)
        assert [spec.tier for spec in TIERS
                if spec.artifact == "acoustic_frames"] == list(PRAAT_FRAME_TIERS)
        secondary = {name for _tier, names in SECONDARY_INPUTS.items() for name in names}
        assert not secondary & set(artifacts), "a secondary input is also some tier's primary"

    def test_all_inputs_is_de_duplicated_in_tier_order(self) -> None:
        """`Stage.inputs` feeds `status --plan`, where a repeated name reads as a dependency."""
        assert len(ALL_INPUTS) == len(set(ALL_INPUTS))
        # Tier order first: the first entry is the first tier's table.
        assert ALL_INPUTS[0] == TIERS[0].artifact
        # Every tier's table is in there exactly once.
        assert all(artifacts in ALL_INPUTS for artifacts in
                   {spec.artifact for spec in TIERS})

    def test_the_columns_every_builder_reads_exist_in_their_schemas(self) -> None:
        """The failure this closes is silent: a projected read of a renamed column raises
        inside ``build_eaf``'s per-tier guard and the tier just disappears.

        So the columns are checked against the schema objects that write the tables, which is
        the same coupling the real readers depend on.
        """
        expected = {
            "speech_words": (WORDS_SCHEMA, ("start_time", "end_time", "word")),
            "speech_segments": (SEGMENTS_SCHEMA, ("start_time", "end_time", "speaker_id",
                                                  "text")),
            "speaker_turns": (SPEAKER_TURNS_SCHEMA, ("start_time", "end_time", "speaker_id",
                                                     "diarization_type")),
            "active_speaker_frames": (ACTIVE_SPEAKER_FRAMES_SCHEMA,
                                      ("timestamp", "is_active_speaker", "face_status",
                                       "frame_reason", "track_id")),
            "active_speaker_tracks": (ACTIVE_SPEAKER_TRACKS_SCHEMA,
                                      ("track_id", "first_timestamp", "last_timestamp",
                                       "frame_count", "active_frame_count", "mean_score")),
            "person_tracks": (PERSON_TRACKS_SCHEMA,
                              ("person_id", "first_timestamp", "last_timestamp",
                               "frame_count", "mean_confidence")),
            # The two secondary inputs the person tier reads. They are checked here as well as
            # in `test_the_secondary_inputs_are_registered_and_read_from_their_schemas` because
            # the failure mode is the same silent one: a projected read of a renamed column
            # makes the tier fall back to "adjacency unverified" and nothing complains.
            "person_frames": (PERSON_FRAMES_SCHEMA,
                              ("frame_number", "timestamp", "person_id")),
            "frame_index": (FRAME_INDEX_SCHEMA, ("frame_number", "pts_seconds")),
            "pose_body": (BODY_SCHEMA, ("timestamp", "confidence")),
            # Read by four tiers; the union of what they read is checked, because a column only
            # `formant_block_rows` reads would otherwise never be verified against the schema.
            "acoustic_frames": (ACOUSTIC_FRAMES_SCHEMA,
                                ("timestamp", "f0_hz", "intensity_db",
                                 "f1_hz", "f2_hz", "f3_hz")),
        }
        for artifact, (schema, columns) in expected.items():
            known = {field.name for field in schema}
            assert set(columns) <= known, f"{artifact}: {set(columns) - known}"
        # The two ASD schemas are checked straight from `schemas` rather than through
        # TABLE_SCHEMAS: they are deliberately absent from that registry (they are written with
        # the schema object passed explicitly), which is pre-existing design and noted as such
        # in test_persons_config. What is asserted here is that this tier list reads columns the
        # writer actually publishes, not that every schema is registered.


class TestAgainstTheCorpus:
    """The real tables, when this machine has them: the column names stop being ours."""

    CORPUS = PROCESSED / "2017-12-30_0735_US_KABC_Jimmy_Kimmel_Live_1120_696_1124_896_hear"
    VIDEO = (ROOT / "data" / "input_videos"
             / "2017-12-30_0735_US_KABC_Jimmy_Kimmel_Live_1120.696_1124.896_hear.mp4")

    def test_the_corpus_praat_tiers_place_only_measured_frames(self) -> None:
        """The real frame table, asked the question a fixture cannot: does every bar sit on
        frames Praat actually measured?

        Two claims, read off the corpus's own `acoustic/frame_features.parquet`. Every bar
        covers at least one frame whose own labelled columns are all measured — the bar's
        interval must contain a real measurement instant, not a hole. And no bar covers a
        *majority* of unmeasured frames: that window was refused, and a bar there would say
        "this held here" over a stretch where Praat mostly measured nothing.

        Counted rather than assumed, with the counts asserted non-zero: a corpus with no
        measured frame would make both checks vacuous, and that would be a silent test.
        """
        if not self.CORPUS.is_dir():
            pytest.skip(f"corpus dataset not present under {PROCESSED}")
        eaf, _report = build_eaf(self.CORPUS, self.VIDEO, log=lambda *a, **k: None)
        counts = tier_counts(eaf)
        for tier in ("f0_blocks", "intensity_blocks", "formant_blocks"):
            assert tier in counts, (tier, sorted(counts))
            assert counts[tier] > 0, tier

        frames = read_table(self.CORPUS / ARTIFACT_LAYOUT["acoustic_frames"],
                            columns=["timestamp", "f0_hz", "intensity_db",
                                     "f1_hz", "f2_hz", "f3_hz"]).to_pylist()
        frames.sort(key=lambda row: row["timestamp"])
        columns_by_tier = {"f0_blocks": ("f0_hz",), "intensity_blocks": ("intensity_db",),
                           "formant_blocks": ("f1_hz", "f2_hz", "f3_hz")}
        for tier, columns in columns_by_tier.items():
            stamps = [seconds_to_ms(row["timestamp"]) for row in frames]
            measured = [all(row[c] is not None for c in columns) for row in frames]
            assert any(measured), f"{tier}: corpus clip has no measured frame"
            bars = annotations(eaf, tier)
            assert bars, tier
            for start_ms, end_ms, text in bars:
                inside = [k for k, stamp in enumerate(stamps) if start_ms <= stamp < end_ms]
                # A bar must sit on real measurement instants, not span the whole grid.
                assert inside, (tier, start_ms, end_ms)
                hit = sum(1 for k in inside if measured[k])
                assert hit >= 1, (tier, start_ms, end_ms)
                # and it may not be mostly holes: that is the refusal this export promises.
                assert hit / len(inside) >= elan_core.FRAME_MIN_FILL, (
                    tier, start_ms, end_ms, hit, len(inside))
                assert text.startswith(("f0 med ", "int med ", "med ")), text

    def test_the_corpus_praat_tiers_cover_every_frame_their_own_rule_accepts(self) -> None:
        """A pitched frame is inside an f0 block unless its own window was refused for thinness.

        `voiced_blocks` groups runs of pitch presence; `f0_blocks` labels 100 ms windows of the
        same column. They are not the same shape and must not be asserted equal (the first
        version of this test did, and passed only because it compared integers loosely). What
        does hold, and is the fact a reader needs: a pitched frame is either inside a block or
        sitting in a window the export said it could not label — never silently dropped.
        """
        if not self.CORPUS.is_dir():
            pytest.skip(f"corpus dataset not present under {PROCESSED}")
        eaf, _report = build_eaf(self.CORPUS, self.VIDEO, log=lambda *a, **k: None)
        counts = tier_counts(eaf)
        if "voiced_blocks" not in counts or "f0_blocks" not in counts:
            pytest.skip("corpus clip has no acoustic frames")
        frames = read_table(self.CORPUS / ARTIFACT_LAYOUT["acoustic_frames"],
                            columns=["timestamp", "f0_hz"]).to_pylist()
        frames.sort(key=lambda row: row["timestamp"])
        pitched = [row["timestamp"] for row in frames if row["f0_hz"] is not None]
        assert pitched, "corpus clip has no pitched frame"

        bars = annotations(eaf, "f0_blocks")
        inside = [t for t in pitched if any(s <= seconds_to_ms(t) < e for s, e, _t in bars)]
        outside = [t for t in pitched if t not in inside]
        # Coverage is real but not total: the thin windows are the honest remainder.
        assert inside, "no pitched frame is inside any f0 block"
        # Which window a frame belongs to is asked of the module's own clock, not recomputed
        # here: `grid + i*step` and `(grid + i*step) + step` are different floats, and a test
        # with its own arithmetic would fail on a clip for a reason that is the test's.
        edges = elan_core.window_edges(
            frames[0]["timestamp"],
            elan_core.window_count(frames[0]["timestamp"], frames[-1]["timestamp"]))
        for t in outside:
            window = next(i for i in range(len(edges) - 1) if edges[i] <= t < edges[i + 1])
            members = [row for row in frames
                       if edges[window] <= row["timestamp"] < edges[window + 1]]
            pitched_in = sum(1 for row in members if row["f0_hz"] is not None)
            assert pitched_in / len(members) < elan_core.FRAME_MIN_FILL, (
                t, window, pitched_in, len(members))

    def test_the_corpus_export_has_every_tier_and_real_words(self, tmp_path: Path) -> None:
        """The real tables, so the column names stop being this file's invention.

        The XML check writes to ``tmp_path`` and not beside the dataset: ``data/processed/`` is
        the operator's corpus and this test has no business creating files in it, even ones it
        deletes afterwards — a crash between the write and the unlink would leave a stray .eaf
        that the next manifest would list as an artifact.
        """
        if not (self.CORPUS / "manifest.json").is_file():
            pytest.skip(f"corpus dataset not present under {PROCESSED}")
        eaf, _report = build_eaf(self.CORPUS, self.VIDEO, log=lambda *a, **k: None)
        counts = tier_counts(eaf)
        assert len(counts) >= 8, counts
        assert counts.get("words", 0) > 0
        descriptor = eaf.media_descriptors[0]
        path = self.CORPUS / eaf_directory() / descriptor["RELATIVE_MEDIA_URL"]
        assert path.resolve() == self.VIDEO.resolve()
        out = tmp_path / "corpus-check.eaf"
        eaf.to_file(str(out))
        root = ET.parse(out).getroot()
        assert list(root.iter("MEDIA_DESCRIPTOR"))
        assert self.VIDEO.name in ET.tostring(root, encoding="unicode")
        # And the tables this export read are the ones the manifest declares, so a tier that
        # silently vanished from the corpus would be named rather than skipped.
        manifest = json.loads((self.CORPUS / "manifest.json").read_text(encoding="utf-8"))
        assert "speech_words" in manifest["artifacts"]

    @pytest.mark.parametrize("name,expected_runs,expected_marks", [
        # The four clips on this disk with a persons stage. Expected runs and marks are
        # **re-derived here** from source/frame_index.parquet and persons/frames.parquet — they
        # are not copied out of a built .eaf — so the assertion fails when the tier and the
        # tables disagree rather than restating a measurement someone wrote down.
        ("2017-12-30_0735_US_KABC_Jimmy_Kimmel_Live_1120_696_1124_896_hear", 8, 3),
        ("2017-12-30_1930_US_CNN_Global_Warning_Arctic_Melt_1237_273_1241_393_hear", 4, 0),
        ("2019-06-29_2000_ES_La-1_Telediario_1_542-550", 9, 0),
        ("person_demo", 110, 28),
    ])
    def test_the_corpus_person_runs_match_a_grouping_derived_from_its_own_tables(
            self, tmp_path: Path, name: str, expected_runs: int, expected_marks: int) -> None:
        """The real tables, so the source-adjacency rule is measured and not asserted.

        The counts below are not a forecast: La-1's eight trajectories become nine runs because
        id 10 breaks once (source frames 114 → 144, 29 source frames with no row for that id in
        between), and `person_demo`'s 75 ids become 110 runs plus 28 single-frame marks. The .eaf
        is written to ``tmp_path``, never beside the corpus.
        """
        root = PROCESSED / name
        if not (root / "manifest.json").is_file():
            pytest.skip(f"corpus dataset not present under {PROCESSED}")
        if not (root / ARTIFACT_LAYOUT["person_tracks"]).is_file():
            pytest.skip(f"{name} has no person tracks table")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        video = Path(manifest["source"]["path"])
        if not video.is_file():
            pytest.skip(f"source video {video} is not on this disk")
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        # The **logical** rows: two ids sighted in the same frame overlap, and an ELAN independent
        # tier cannot hold both, so the emitted annotations are segments carrying both labels. The
        # grouping this test re-derives is a property of the producers' rows, which is what the
        # document's projection metadata preserves — one entry per sighting run, at its own
        # endpoints. Asserting the emitted list here would measure the projection, not the grouping.
        labels = logical_texts(eaf, "person_tracks")
        runs = [text for text in labels if "sighting run " in text]
        marks = [text for text in labels if "sighting mark " in text]
        # The grouping the tier claims to have used, re-derived here from the source tables.
        frames = read_table(root / ARTIFACT_LAYOUT["person_frames"],
                            columns=["frame_number", "timestamp", "person_id"]).to_pylist()
        index = {row["frame_number"]: row["pts_seconds"] for row in read_table(
            root / ARTIFACT_LAYOUT["frame_index"],
            columns=["frame_number", "pts_seconds"]).to_pylist()}
        grouped: dict[int, dict[int, Any]] = {}
        for row in frames:
            if row["person_id"] is None or row["frame_number"] is None:
                continue
            frame = int(row["frame_number"])
            grouped.setdefault(int(row["person_id"]), {})[frame] = row["timestamp"]
        # Walk each id's de-duplicated, ordered frames exactly as the tier documents it: a run
        # continues only across consecutive source frames the index places at the time the row
        # claims. Re-derived with the same tolerance, deliberately without calling the tier.
        def placed(frame: int, stamp: Any) -> bool:
            return (frame in index and stamp is not None
                    and abs(index[frame] - float(stamp)) <= PTS_TOLERANCE_SECONDS)

        runs_derived = marks_derived = 0
        for seen in grouped.values():
            ordered = sorted(seen)
            run = 0
            for position, frame in enumerate(ordered):
                adjacent = (position > 0 and frame == ordered[position - 1] + 1
                            and placed(frame, seen[frame])
                            and placed(ordered[position - 1], seen[ordered[position - 1]]))
                if not adjacent:
                    if run == 1:
                        marks_derived += 1
                    elif run > 1:
                        runs_derived += 1
                    run = 1
                else:
                    run += 1
            if run == 1:
                marks_derived += 1
            elif run > 1:
                runs_derived += 1
        assert (runs_derived, marks_derived) == (expected_runs, expected_marks)
        assert (len(runs), len(marks)) == (runs_derived, marks_derived)
        # Nothing in the corpus's own tables moved: this is a re-expression, and the raw bytes
        # are still what the persons stage wrote.
        out = tmp_path / f"person-check-{name[:12]}.eaf"
        eaf.to_file(str(out))
        assert out.is_file()


# ------------------------------------------------------------------ person sightings


def _person_frame(index: int | None, person_id: int, *,
                  timestamp: Any = _DEFAULT, confidence: float = 0.8,
                  **extra: Any) -> dict[str, Any]:
    """One row of ``persons/frames.parquet``.

    Only the three columns the sighting tier reads are interesting: which source frame, which
    id, and when. ``timestamp`` defaults to the 25 FPS grid of the synthetic clip, and passing
    ``None`` builds the row the schema allows — a sighting with no time.
    """
    stamp = (round(index * STEP, 6) if timestamp is _DEFAULT else timestamp
             if index is not None or timestamp is not _DEFAULT else None)
    return {"schema_version": "1.0", "video_id": "clip",
            "frame_number": index,
            "timestamp": stamp,
            "person_id": person_id, "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0,
            "confidence": confidence, "track_confidence": None,
            "confidence_reason": "no_track_confidence", "bbox_area": 100.0,
            "persons_in_frame": 1, **extra}


def _frame_row(index: int, pts: float) -> dict[str, Any]:
    return {"schema_version": "1.0", "video_id": "clip", "frame_number": index,
            "pts_seconds": pts}


def _person_track(person_id: int, first: float | None, last: float | None,
                  frame_count: int, *, longest_gap: float = 0.04,
                  mean_confidence: float | None = 0.8) -> dict[str, Any]:
    """The track row the persons stage writes for the frames below.

    The tier reads ``frame_count`` and ``mean_confidence`` from it, so a test that writes
    frames without the matching track row would be testing a dataset the producer cannot make.
    """
    return {"schema_version": "1.0", "video_id": "clip", "person_id": person_id,
            "first_timestamp": first, "last_timestamp": last,
            "duration_seconds": None if first is None or last is None else last - first,
            "frame_count": frame_count, "frame_coverage": 0.5,
            "longest_gap_seconds": longest_gap, "mean_confidence": mean_confidence,
            "max_confidence": mean_confidence, "mean_bbox_area": 100.0,
            "max_bbox_area": 100.0, "appearance_order": person_id}


def _persons(root: Path, frames: list[dict[str, Any]],
             tracks: list[dict[str, Any]]) -> None:
    """Write both persons tables — the tier needs the frames to exist to group anything."""
    _write(PERSON_FRAMES_SCHEMA, root / "persons" / "frames.parquet", frames)
    _write(PERSON_TRACKS_SCHEMA, root / "persons" / "tracks.parquet", tracks)


def _frame_index(root: Path, rows: list[dict[str, Any]]) -> None:
    _write(FRAME_INDEX_SCHEMA, root / "source" / "frame_index.parquet", rows)


def _reported_gap(text: str) -> str:
    match = re.search(r"max gap reported ([0-9.]+) s|max gap reported (unknown)", text)
    assert match, f"no reported gap in {text!r}"
    return match.group(1) or match.group(2)


class TestPersonSightings:
    """The person tier reports sighting runs over **verified source frames**.

    The track table's two endpoints are a span, and a span says nothing about the frames in
    between: on La-1 the id that disappears for a whole second and the one that is present in
    every sampled frame both produce one annotation each. This class is the reason B2 exists.
    """

    def test_consecutive_source_frames_become_one_run(self, tmp_path: Path) -> None:
        root, video = person_clip(tmp_path)
        _persons(root, [_person_frame(i, 1) for i in (0, 1, 2)],
                 [_person_track(1, 0.0, 0.08, 3)])
        _frame_index(root, [_frame_row(i, round(i * STEP, 6)) for i in (0, 1, 2)])
        labels = annotations(eaf_of({"dir": root, "video": video}), "person_tracks")
        assert len(labels) == 1
        start, end, text = labels[0]
        assert "sighting run 3 frames" in text
        # The run ends at the last sighting's own time. No grid step is added — the next frame
        # may never have been sampled — and nothing is subtracted either.
        assert (start, end) == (0, 80)
        assert "max gap reported 0.040 s" in text
        assert "of 3" in text and "track coverage" in text

    def test_a_source_frame_that_exists_but_holds_no_detection_splits_the_run(
            self, tmp_path: Path) -> None:
        """The La-1 case, in three frames.

        Source frames 0, 1, 2 all exist in the clip's frame list and the id has rows in 0 and 2
        only, so the pair is not adjacent and the run breaks. What this fixture does **not**
        establish is *why* frame 1 has no row for the id: `persons/frames.parquet` records
        detections and never which frames were examined, so the tier reports a break and nothing
        about whether that frame was sampled, looked at, or rejected.
        """
        root, video = person_clip(tmp_path)
        _persons(root, [_person_frame(0, 1), _person_frame(2, 1)],
                 [_person_track(1, 0.0, 0.08, 2, longest_gap=0.08)])
        _frame_index(root, [_frame_row(i, round(i * STEP, 6)) for i in (0, 1, 2)])
        labels = annotations(eaf_of({"dir": root, "video": video}), "person_tracks")
        assert len(labels) == 2
        assert [start for start, _e, _t in labels] == [0, 80]
        assert all("sighting mark 1 frame" in text for _s, _e, text in labels)
        # The reported gap survives the split because it is a property of the id, not of the run:
        # the producer's own column says 80 ms, and that is the number worth quoting even though
        # each sighting now stands alone.
        assert all("max gap reported 0.080 s" in text for _s, _e, text in labels)
        assert all("sighting mark 1 frame of 2" in text for _s, _e, text in labels)

    def test_a_gap_in_the_source_index_does_not_join_across_it(self, tmp_path: Path) -> None:
        """Frames 0 and 2 exist, frame 1 was never decoded.

        ``frame_number + 1`` is not adjacency when the index skipped a frame: the two instants
        are 80 ms apart and nothing was measured between them, so the pair is not one sighting.
        """
        root, video = person_clip(tmp_path)
        _persons(root, [_person_frame(0, 1), _person_frame(2, 1)],
                 [_person_track(1, 0.0, 0.08, 2, longest_gap=0.08)])
        _frame_index(root, [_frame_row(0, 0.0), _frame_row(2, 0.08)])
        labels = annotations(eaf_of({"dir": root, "video": video}), "person_tracks")
        assert len(labels) == 2
        assert all("sighting mark" in text for _s, _e, text in labels)

    def test_a_sparse_stride_never_becomes_one_run(self, tmp_path: Path) -> None:
        """Every third source frame sampled, one id seen in all of them.

        The frames table is detections, not a sampling grid: nothing in it says the stride was
        3. Merging the three sightings into one run because the id appears "every measured
        frame" would claim the person was on screen during the two frames never looked at —
        exactly the inference B2 forbids.
        """
        root, video = person_clip(tmp_path)
        _persons(root, [_person_frame(i, 1) for i in (0, 3, 6)],
                 [_person_track(1, 0.0, 0.24, 3, longest_gap=0.12)])
        _frame_index(root, [_frame_row(i, round(i * STEP, 6)) for i in (0, 3, 6)])
        labels = annotations(eaf_of({"dir": root, "video": video}), "person_tracks")
        assert len(labels) == 3
        assert all("sighting mark 1 frame" in text for _s, _e, text in labels)
        assert [start for start, _e, _t in labels] == [0, 120, 240]

    def test_without_the_frames_table_the_tier_is_skipped_rather_than_spanned(
            self, tmp_path: Path) -> None:
        """The pre-B2 shape: one annotation per id, drawn from two timestamps.

        A track span is evidence that something was seen at both ends and nothing in between is
        known, so it must not be re-labelled as sightings. The tier is skipped with the missing
        dependency named, and every other tier still builds.
        """
        root, video = person_clip(tmp_path)
        _persons(root, [_person_frame(0, 1)], [_person_track(1, 0.0, 1.0, 5)])
        _frame_index(root, [_frame_row(0, 0.0)])
        (root / "persons" / "frames.parquet").unlink()
        lines: list[str] = []
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: lines.append(str(a[0])))
        assert "person_tracks" not in tier_counts(eaf)
        assert any("person_tracks skipped" in line and "persons/frames.parquet" in line
                   for line in lines), lines
        assert "words" in tier_counts(eaf), "one missing dependency must not lose the export"

    def test_an_unreadable_frames_table_skips_the_tier_with_the_exception_named(
            self, tmp_path: Path) -> None:
        root, video = person_clip(tmp_path)
        _persons(root, [_person_frame(0, 1)], [_person_track(1, 0.0, 0.0, 1)])
        _frame_index(root, [_frame_row(0, 0.0)])
        (root / "persons" / "frames.parquet").write_bytes(b"not parquet at all")
        lines: list[str] = []
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: lines.append(str(a[0])))
        assert "person_tracks" not in tier_counts(eaf)
        assert any("person_tracks skipped" in line and "unreadable" in line
                   for line in lines), lines
        assert "words" in tier_counts(eaf)

    def test_a_missing_frame_index_cannot_verify_adjacency(self, tmp_path: Path) -> None:
        """Consecutive frame numbers are not verified adjacency without the source index.

        Same frames as the run case; without ``source/frame_index.parquet`` there is nothing to
        check them against, so each sighting stands alone and says so. The alternative — trusting
        the persons table's own integers — is §20.2's mistake with the evidence removed.
        """
        root, video = person_clip(tmp_path)
        _persons(root, [_person_frame(i, 1) for i in (0, 1, 2)],
                 [_person_track(1, 0.0, 0.08, 3)])
        lines: list[str] = []
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: lines.append(str(a[0])))
        labels = annotations(eaf, "person_tracks")
        assert len(labels) == 3
        assert all("sighting mark 1 frame" in text for _s, _e, text in labels)
        assert all(ADJACENCY_NO_INDEX in text for _s, _e, text in labels)
        # The cause is in the marker: without the clip's frame list, coverage is unknown too,
        # and a reader must not mistake "nobody could check" for "checked, and alone".
        assert all("coverage unknown" in text for _s, _e, text in labels)
        assert not any(ADJACENCY_SPLIT in text for _s, _e, text in labels)
        assert not any("person_tracks skipped" in line for line in lines)

    def test_a_corrupt_frame_index_is_the_same_state_as_an_absent_one(self,
                                                                     tmp_path: Path) -> None:
        root, video = person_clip(tmp_path)
        _persons(root, [_person_frame(i, 1) for i in (0, 1, 2)],
                 [_person_track(1, 0.0, 0.08, 3)])
        _frame_index(root, [_frame_row(0, 0.0)])
        (root / "source" / "frame_index.parquet").write_bytes(b"corrupt")
        labels = annotations(eaf_of({"dir": root, "video": video}), "person_tracks")
        assert len(labels) == 3
        assert all(ADJACENCY_NO_INDEX in text for _s, _e, text in labels)

    def test_a_frame_number_absent_from_the_index_cannot_verify_either_side(
            self, tmp_path: Path) -> None:
        """The index exists and is readable, but names no such frame.

        Producers are not repaired here; the tier simply cannot claim adjacency through a frame
        it cannot place, so the run breaks at that point rather than at the number.
        """
        root, video = person_clip(tmp_path)
        _persons(root, [_person_frame(i, 1) for i in (0, 1, 2)],
                 [_person_track(1, 0.0, 0.08, 3)])
        _frame_index(root, [_frame_row(0, 0.0), _frame_row(1, 0.04)])
        labels = annotations(eaf_of({"dir": root, "video": video}), "person_tracks")
        assert ["sighting run" in text for _s, _e, text in labels] == [True, False]
        assert ADJACENCY_UNVERIFIED in labels[1][2]
        assert ADJACENCY_SPLIT in labels[0][2]

    def test_a_sighting_whose_frame_the_index_does_not_name_is_never_verified(
            self, tmp_path: Path) -> None:
        """One lone sighting at a frame the clip's own index does not list.

        There is no neighbour to compare it with, so no boundary can report anything, and the
        only remaining question is whether this sighting itself is placed. It is not: the index
        has no such frame, so `source adjacency verified` would be a claim built on nothing.
        """
        root, video = person_clip(tmp_path)
        _persons(root, [_person_frame(7, 1)], [_person_track(1, 0.28, 0.28, 1)])
        _frame_index(root, [_frame_row(i, round(i * STEP, 6)) for i in range(5)])
        text = annotations(eaf_of({"dir": root, "video": video}), "person_tracks")[0][2]
        assert ADJACENCY_UNVERIFIED in text
        assert ADJACENCY_VERIFIED not in text

    def test_a_frame_time_that_disagrees_with_the_index_is_not_verified(
            self, tmp_path: Path) -> None:
        """Frame numbers line up, the clock does not.

        The corpus's four clips have ``persons.frames.timestamp == frame_index.pts_seconds``
        exactly; a dataset where they differ is a mismatched pair of tables, and the one thing
        to do with a mismatch is stop calling the frames adjacent rather than pick a clock.
        """
        root, video = person_clip(tmp_path)
        _persons(root, [_person_frame(0, 1, timestamp=0.0),
                        _person_frame(1, 1, timestamp=5.0)],
                 [_person_track(1, 0.0, 5.0, 2, longest_gap=5.0)])
        _frame_index(root, [_frame_row(0, 0.0), _frame_row(1, 0.04)])
        labels = annotations(eaf_of({"dir": root, "video": video}), "person_tracks")
        assert len(labels) == 2
        assert all(ADJACENCY_UNVERIFIED in text for _s, _e, text in labels)
        # The sighting keeps the time the persons table measured, not the index's.
        assert [start for start, _e, _t in labels] == [0, 5000]

    def test_the_representational_millisecond_is_visible_on_a_single_sighting(
            self, tmp_path: Path) -> None:
        """A sighting measured at one instant has no duration, and the file must not pretend.

        ELAN cannot store (t, t), so the annotation is 1 ms wide. The label says the frame it
        covers, so a reader who measures the bar gets the format's minimum and not a claim.
        """
        root, video = person_clip(tmp_path)
        _persons(root, [_person_frame(1, 1)], [_person_track(1, 0.04, 0.04, 1)])
        _frame_index(root, [_frame_row(0, 0.0), _frame_row(1, 0.04)])
        start, end, text = annotations(eaf_of({"dir": root, "video": video}),
                                       "person_tracks")[0]
        assert (start, end) == (40, 41)
        assert "covers src frame 1" in text
        assert "sighting mark" in text
    def test_duplicate_rows_for_the_same_person_and_frame_count_once(self,
                                                                    tmp_path: Path) -> None:
        """A sighting is a (person, source frame) pair, however many rows carry it.

        Counting rows would let a table with the same detection written twice report two frames
        of a person, which is the direction that inflates.
        """
        root, video = person_clip(tmp_path)
        # The third row is the same sighting written twice; the fourth is the same (id, frame)
        # pair carrying a *different* timestamp, and the fifth the same pair with a disagreeing
        # `persons_in_frame`. Either way it is one sighting of one frame, not two or three:
        # grouping is on the (id, source frame) pair, and every other column of a duplicated row
        # is a redundant copy of it. The first row's time wins.
        _persons(root, [_person_frame(0, 1), _person_frame(0, 1),
                        _person_frame(0, 1, timestamp=0.041),
                        _person_frame(0, 1, persons_in_frame=7), _person_frame(1, 1)],
                 [_person_track(1, 0.0, 0.04, 2)])
        _frame_index(root, [_frame_row(0, 0.0), _frame_row(1, 0.04)])
        labels = annotations(eaf_of({"dir": root, "video": video}), "person_tracks")
        assert len(labels) == 1
        assert "sighting run 2 frames of 2" in labels[0][2]
        assert labels[0][:2] == (0, 40)

    def test_two_people_in_one_frame_do_not_inflate_either_sighting(self,
                                                                    tmp_path: Path) -> None:
        """``persons_in_frame`` is a property of the frame; a sighting is a property of an id."""
        root, video = person_clip(tmp_path)
        rows = [_person_frame(0, 1, persons_in_frame=2), _person_frame(0, 2, persons_in_frame=2),
                _person_frame(1, 1, persons_in_frame=2)]
        _persons(root, rows, [_person_track(1, 0.0, 0.04, 2), _person_track(2, 0.0, 0.0, 1)])
        _frame_index(root, [_frame_row(0, 0.0), _frame_row(1, 0.04)])
        # Read from the logical rows, not the emitted bars: the two ids overlap in frame 0, so
        # the tier projects them and the emitted list is shorter than the sighting history this
        # test is about. The grouping rule is a property of the rows, projection or not.
        texts = logical_texts(eaf_of({"dir": root, "video": video}), "person_tracks")
        assert [text.split(" · ")[1] for text in texts] == [
            "sighting run 2 frames of 2", "sighting mark 1 frame of 1"]
        assert "track coverage 0.500" in texts[0]

    def test_the_run_carries_source_coverage_and_the_producers_reported_gap(
            self, tmp_path: Path) -> None:
        """The two numbers that make a run honest, both read straight out of the tables.

        `of M` is this id's own total sightings, so a reader can see that the run in front of
        them is not the whole story. The gap is the track table's own
        ``longest_gap_seconds`` column, printed as reported and **never recomputed here** — which
        is why the fixture deliberately reports a number the frames cannot reproduce: if the tier
        derived the gap from sightings again, this assertion would print the derived value and
        pass. The source is the column, and the label names the source.
        """
        root, video = person_clip(tmp_path)
        # id 1 seen at 0,1 and 4,5; source frames 0..5 all exist. The frames imply a widest
        # separation of 0.120 s; the producer's column says 999.000 s. Only one of those two can
        # appear in the label, and it is the column.
        _persons(root, [_person_frame(i, 1) for i in (0, 1, 4, 5)],
                 [_person_track(1, 0.0, 0.20, 4, longest_gap=999.0)])
        _frame_index(root, [_frame_row(i, round(i * STEP, 6)) for i in range(6)])
        labels = annotations(eaf_of({"dir": root, "video": video}), "person_tracks")
        assert len(labels) == 2
        derived_from_frames = 0.120
        for _start, _end, text in labels:
            assert "of 4" in text
            assert _reported_gap(text) == "999.000"
            assert f"{derived_from_frames:.3f}" not in text

    def test_a_reported_gap_of_zero_is_the_columns_zero_not_a_recomputed_one(
            self, tmp_path: Path) -> None:
        """A real 0.000 in the column prints, so the word `unknown` keeps meaning "no value".

        The same fixture with a null column prints `unknown`, so the two states stay distinguishable
        in the file rather than both landing on a number.
        """
        root, video = person_clip(tmp_path)
        _persons(root, [_person_frame(i, 1) for i in (0, 1)],
                 [_person_track(1, 0.0, 0.04, 2, longest_gap=0.0)])
        _frame_index(root, [_frame_row(i, round(i * STEP, 6)) for i in (0, 1)])
        text = annotations(eaf_of({"dir": root, "video": video}), "person_tracks")[0][2]
        assert _reported_gap(text) == "0.000"

        _persons(root, [_person_frame(i, 1) for i in (0, 1)],
                 [_person_track(1, 0.0, 0.04, 2, longest_gap=None)])
        text = annotations(eaf_of({"dir": root, "video": video}), "person_tracks")[0][2]
        assert _reported_gap(text) == "unknown"

    def test_an_unmeasurable_gap_says_unknown_rather_than_zero(self, tmp_path: Path) -> None:
        """One sighting has no pair to separate, so the label refuses the column's placeholder.

        The producer writes 0.0 for an id seen in a single frame (`PersonsStage.person_track_rows`
        has no gap to take a maximum of), and 0.000 in a tier reads as "never lost sight of them".
        The tier prints the column as reported whenever there is a pair, and `unknown` where the
        reported 0.0 is only a placeholder for "no pair exists".
        """
        root, video = person_clip(tmp_path)
        _persons(root, [_person_frame(0, 1)], [_person_track(1, 0.0, 0.0, 1,
                                                             longest_gap=0.0)])
        _frame_index(root, [_frame_row(0, 0.0)])
        text = annotations(eaf_of({"dir": root, "video": video}), "person_tracks")[0][2]
        assert _reported_gap(text) == "unknown"
        # The unit travels with the number, so a missing value never borrows one: `unknown s`
        # would put a quantity on a non-number, which is the same small lie as 0.000 there.
        assert "unknown s" not in text

    def test_a_sighting_with_no_timestamp_is_dropped_rather_than_placed_at_zero(
            self, tmp_path: Path) -> None:
        """Regression: a null sighting time used to export as ``[0, 1)`` ms.

        The pre-fix behaviour put the sighting at second zero of the clip — a person reported on
        screen at the start of a video where the frames table says nothing about when they were
        seen. ELAN has no "time unknown" annotation, so the honest export drops the row, keeps
        its timed siblings where they were measured, and counts the drop separately from a
        non-finite one.
        """
        root, video = person_clip(tmp_path)
        _persons(root, [_person_frame(2, 1, timestamp=None), _person_frame(1, 1)],
                 [_person_track(1, 0.04, 0.04, 2)])
        _frame_index(root, [_frame_row(1, 0.04), _frame_row(2, 0.08)])
        lines: list[str] = []
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: lines.append(str(a[0])))
        labels = annotations(eaf, "person_tracks")
        # The timed sighting is still there, at the time its own table measured.
        assert [(start, end) for start, end, _t in labels] == [(40, 41)]
        assert "covers src frame 1" in labels[0][2]
        # The untimed sighting is not in the file at all, at t=0 or anywhere else.
        assert not [1 for start, _e, text in labels if start == 0], labels
        assert not [1 for _s, _e, text in labels if "covers src frame 2" in text], labels
        assert [line for line in lines if "person_tracks" in line
                and "1 of 2" in line and "missing timestamp" in line], lines
        # The other tiers still build.
        assert "words" in tier_counts(eaf)

    def test_rows_the_schema_allows_but_nobody_wants_are_neither_fatal_nor_inflating(
            self, tmp_path: Path) -> None:
        """A null `person_id` names no one, and a null `frame_number` places nothing.

        Both columns are nullable in ``PERSON_FRAMES_SCHEMA``, and the schema's own comment says
        why a null there is a missing measurement rather than a zero. Neither may crash the tier
        (that would lose all sixteen sibling tiers through the per-tier guard's log line), neither
        may be counted as a sighting of somebody, and the row that has a time but no frame still
        belongs in the file as a mark that cannot be placed.
        """
        root, video = person_clip(tmp_path)
        unnamed = dict(_person_frame(2, 1))
        unnamed["person_id"] = None
        unplaced = _person_frame(None, 1, timestamp=9.0)
        _persons(root, [_person_frame(0, 1), _person_frame(1, 1), unnamed, unplaced],
                 [_person_track(1, 0.0, 9.0, 4)])
        _frame_index(root, [_frame_row(i, round(i * STEP, 6)) for i in (0, 1, 2)])
        labels = annotations(eaf_of({"dir": root, "video": video}), "person_tracks")
        assert len(labels) == 2, labels
        assert "sighting run 2 frames of 3" in labels[0][2], "the id-less row must not be a sighting"
        assert "covers src frame unknown" in labels[1][2]
        assert ADJACENCY_UNVERIFIED in labels[1][2] and ADJACENCY_VERIFIED not in labels[1][2]
        # The mark keeps the time the frames table measured, at ELAN's minimum width.
        assert labels[1][:2] == (9000, 9001)

    def test_the_two_id_tables_agree_on_who_gets_an_annotation(self, tmp_path: Path) -> None:
        """A sighting needs an observation; a track row alone is not one.

        An id present only in `persons/tracks.parquet` produces nothing (there is no frame to
        place), and an id present only in `persons/frames.parquet` still produces its sightings —
        with `unknown` for the two numbers only the track table carries, rather than a zero.
        """
        root, video = person_clip(tmp_path)
        _persons(root, [_person_frame(0, 4)], [_person_track(9, 0.0, 0.0, 1)])
        _frame_index(root, [_frame_row(0, 0.0)])
        labels = annotations(eaf_of({"dir": root, "video": video}), "person_tracks")
        assert len(labels) == 1
        _start, _end, text = labels[0]
        assert "person 4 ·" in text and "person 9" not in text
        assert "conf unknown" in text and "track coverage unknown" in text

    @pytest.mark.parametrize("confidence,expected", [
        (0.9282, "conf 0.928"),
        (None, "conf unknown"),
        (float("nan"), "conf unknown"),
    ])
    def test_confidence_is_the_track_mean_and_a_missing_one_is_not_zero(
            self, tmp_path: Path, confidence: float | None, expected: str) -> None:
        """Per-frame confidence varies as a person turns; the track table's mean is the one
        number worth quoting, and B1's unknown rule keeps applying to it."""
        root, video = person_clip(tmp_path)
        _persons(root, [_person_frame(0, 1)],
                 [_person_track(1, 0.0, 0.0, 1, mean_confidence=confidence)])
        _frame_index(root, [_frame_row(0, 0.0)])
        text = annotations(eaf_of({"dir": root, "video": video}), "person_tracks")[0][2]
        assert expected in text

    def test_person_ids_are_tracker_trajectories_and_the_semantics_say_so(
            self, dataset: dict[str, Path]) -> None:
        """"person 7" in a tier is not seven humans.

        ByteTracker recycles ids and loses them; on this disk `person_demo` reports 75 ids over
        205 sampled frames. The property is what travels with a quoted label.
        """
        text = dict(eaf_of(dataset).properties)["pipeline-tier-semantics"]
        clause = text[text.index("Persons:"):text.index("Coverage:")]
        assert "trajectory" in clause and "not a human" in clause
        # Run count against this id's total, and the elapsed-gap wording.
        assert "N de-duplicated sightings" in clause and "M sightings in total" in clause
        assert "elapsed" in clause and "sampling interval" in clause
        # The one claim that must not be missing: verified grouping is not coverage.
        assert "Sampling coverage is not established" in clause
        assert "not evidence that nobody was there" in clause
        # The 1 ms is the format's minimum, not a duration.
        assert "minimum representable interval" in clause


def person_clip(tmp_path: Path) -> tuple[Path, Path]:
    """The synthetic clip's transcript plus empty persons dirs, ready for the frames tables.

    Returns ``(dataset_dir, video)``. The transcript is there because ``build_eaf`` reports a
    tier census that a reader compares against the rest of the file, and it keeps the other
    tiers alive in the tests that assert one tier was skipped.
    """
    root = tmp_path / "processed" / "clip"
    video_dir = tmp_path / "input_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video = video_dir / "clip.mp4"
    video.write_bytes(b"stub")
    _write(WORDS_SCHEMA, root / "speech" / "words.parquet", [_word("hello", 0.0, 0.4)])
    return root, video


# --------------------------------------------- non-overlapping independent tiers (B2a)

#: The document property that carries the logical intervals a projection split up.
PROJECTION_PROPERTY = "pipeline-overlap-projection"


def _segment_row(segment_id: str, start: float, end: float, text: str) -> dict[str, Any]:
    """One `speech/segments.parquet` row for the overlap fixtures.

    `duration` is left null when either endpoint is, exactly as `_word` does it: a row with no time
    has no duration to report, and a fixture that subtracted through a null would both crash and
    invent a number the producer could not have written.
    """
    duration = (end - start) if (start is not None and end is not None) else None
    return {"schema_version": "1.0", "video_id": "clip", "segment_id": segment_id,
            "start_time": start, "end_time": end, "duration": duration,
            "language": "en", "speaker_id": "SPEAKER_00", "text": text, "confidence": -0.15}


def _clip(tmp_path: Path) -> tuple[Path, Path]:
    """A dataset directory and a video, with nothing written yet."""
    root = tmp_path / "processed" / "clip"
    video_dir = tmp_path / "input_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video = video_dir / "clip.mp4"
    video.write_bytes(b"stub")
    return root, video


def _sweep_boundaries(rows: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Every maximal disjoint segment of a set of intervals, from a boundary sweep.

    Deliberately written a second time, here in the test, on the seconds the fixture was built
    from and with a different algorithm (sort the endpoints, walk them, carry the active set):
    the projection in `elan.py` runs on **milliseconds** and re-derives its boundaries from the
    rounded endpoints, so if the two implementations ever disagree the disagreement is the bug.
    """
    points = sorted({value for pair in rows for value in pair})
    segments: list[tuple[float, float]] = []
    for earlier, later in zip(points, points[1:]):
        if any(start <= earlier and later <= end for start, end in rows):
            segments.append((earlier, later))
    return segments


def projection_of(eaf: Any) -> dict[str, Any]:
    """The projection document's per-tier map, parsed — `{}` when no tier was affected.

    Absence is the state a clean export is written in (see
    `test_only_an_affected_tier_appears_in_the_property`), so reading it as an empty dict here
    keeps the "nothing was re-cut" assertions from depending on whether the key exists. The
    version marker stays out of the way: this returns the `tiers` map, and the version is asserted
    separately against the raw property.
    """
    raw = dict(eaf.properties).get(PROJECTION_PROPERTY)
    return json.loads(raw)["tiers"] if raw else {}


def projection_version(eaf: Any) -> int:
    """The shape marker of the projection property."""
    return json.loads(dict(eaf.properties)[PROJECTION_PROPERTY])["version"]


class TestIndependentTierProjection:
    """Same-tier overlap is illegal in ELAN; the export must project rather than invent.

    The MPI manual states that annotations in one tier may not overlap in time, and pympi never
    enforced it — neither did this file's own round-trip test, which only ever asked whether the
    bytes came back. So the tiers built from simultaneous producers (`person_tracks`: two people
    in one frame; `face_tracks`: two TalkNet tracks alive at once; `asd_speaking`: overlapping
    runs from a re-scored grid) wrote bars ELAN cannot render, and the pipeline's own tables
    already show such rows today.

    Three rules the whole class is about:

    * **Nothing is dropped and nothing is staggered.** Every source row keeps its exact measured
      endpoints in the document's `pipeline-overlap-projection` property; only the *emitted*
      intervals are re-cut, into disjoint half-open segments built from the source endpoints
      themselves, each carrying the text of every row active in it.
    * **A projection is a re-expression, not a new measurement.** A tier with no overlap is
      byte-identical to what this module wrote before: same intervals, same single-string label,
      no mention in the property at all.
    * **Identities are never merged.** Two rows that round to the same millisecond pair stay two
      source intervals with two ids and two texts, in an order that does not depend on the order
      Parquet returned them in.
    """

    # ------------------------------------------------------------- the segments split

    def test_two_sightings_in_one_frame_become_one_segment_holding_both_labels(
            self, tmp_path: Path) -> None:
        """The corpus case, made small: two people, one frame, one legal tier.

        Before this change the tier emitted two annotations over the same pair of time slots,
        which is exactly what an ELAN independent tier may not contain.
        """
        root, video = _clip(tmp_path)
        _write(WORDS_SCHEMA, root / "speech" / "words.parquet", [_word("hello", 0.0, 0.4)])
        _persons(root, [_person_frame(0, 1), _person_frame(0, 2)],
                 [_person_track(1, 0.0, 0.0, 1), _person_track(2, 0.0, 0.0, 1)])
        _frame_index(root, [_frame_row(i, round(i * STEP, 6)) for i in range(3)])
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        emitted = annotations(eaf, "person_tracks")
        assert len(emitted) == 1, emitted
        start, end, value = emitted[0]
        assert (start, end) == (0, 1)
        texts = json.loads(value)
        assert sorted(texts) == ["person 1 · sighting mark 1 frame of 1 · conf 0.800 · track "
                                 "coverage 0.500 · max gap reported unknown · covers src frame 0"
                                 " · source adjacency verified",
                                 "person 2 · sighting mark 1 frame of 1 · conf 0.800 · track "
                                 "coverage 0.500 · max gap reported unknown · covers src frame 0"
                                 " · source adjacency verified"]

    @pytest.mark.parametrize("id_a,id_b", [(0, 1), (1, 0)], ids=["track-0-first", "track-1-first"])
    def test_coincident_face_tracks_keep_both_texts_in_one_order(
            self, tmp_path: Path, id_a: int, id_b: int) -> None:
        """Same interval, two producers: the label is a group, and its order is ours.

        The rows are written in the opposite order for the second parameter, so a label whose
        order came from Parquet would flip between the two runs and this test would say so.
        """
        root, video = _clip(tmp_path)
        _write(WORDS_SCHEMA, root / "speech" / "words.parquet", [_word("hello", 0.0, 0.4)])
        rows = [{"schema_version": "1.0", "video_id": "clip", "track_id": id,
                 "first_timestamp": 0.0, "last_timestamp": 0.04, "frame_count": 2,
                 "active_frame_count": 2, "active_ratio": 1.0, "mean_score": 1.0,
                 "max_score": 1.0, "scenes": [1], "mean_bbox_area": 100.0}
                for id in (id_a, id_b)]
        _write(ACTIVE_SPEAKER_TRACKS_SCHEMA,
               root / "speaker" / "active_speaker_tracks.parquet", rows)
        values = [text for _s, _e, text in annotations(eaf_of({"dir": root, "video": video}),
                                                       "face_tracks")]
        assert len(values) == 1
        # Ordered by text, so the second parameter — the same table with its rows reversed —
        # produces the identical label. That is the point of the parametrization.
        assert json.loads(values[0]) == sorted([f"track {id_a} · 2/2 act · mean 1.000",
                                               f"track {id_b} · 2/2 act · mean 1.000"])

    @pytest.mark.parametrize("rows,expected", [
        # nested: one segment per distinct endpoint pair, outer label alone then both.
        ([(0.0, 2.0), (0.5, 1.0)], [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0)]),
        # partial overlap: the intersection is its own segment.
        ([(0.0, 1.0), (0.5, 1.5)], [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5)]),
        # coincident: one segment, both labels.
        ([(0.0, 1.0), (0.0, 1.0)], [(0.0, 1.0)]),
        # touching: legal, and it stays two annotations — no gap, no merge.
        ([(0.0, 1.0), (1.0, 2.0)], [(0.0, 1.0), (1.0, 2.0)]),
        # an interior endpoint that only one row owns must survive as a boundary.
        ([(0.0, 2.0), (0.5, 2.0)], [(0.0, 0.5), (0.5, 2.0)]),
    ], ids=["nested", "partial", "coincident", "touching", "shared-end"])
    def test_the_segments_are_the_sweep_of_the_source_endpoints(
            self, tmp_path: Path, rows: list[tuple[float, float]],
            expected: list[tuple[float, float]]) -> None:
        """Boundaries come from the producers' own endpoints — never an invented offset.

        `expected` is written out per case rather than computed, and the emitted pairs are also
        compared against an independent sweep, so a projection that rounded a boundary or slid
        one by a grid step dies twice.
        """
        root, video = _clip(tmp_path)
        _write(SEGMENTS_SCHEMA, root / "speech" / "segments.parquet",
               [_segment_row(f"seg-{i}", start, end, f"text {i}")
                for i, (start, end) in enumerate(rows)])
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        got = [(start, end) for start, end, _text in annotations(eaf, "segments_src")]
        assert got == [(int(start * 1000), int(end * 1000)) for start, end in expected]
        assert got == [(int(a * 1000), int(b * 1000)) for a, b in _sweep_boundaries(rows)]
        assert not any(next_start < previous_end
                       for (_s1, previous_end, _t1), (next_start, _s2, _t2) in zip(
                           annotations(eaf, "segments_src"),
                           annotations(eaf, "segments_src")[1:]))

    def test_touching_annotations_are_allowed_and_not_merged(self, tmp_path: Path) -> None:
        """Half-open neighbours share a boundary; that is not an overlap.

        Asserted on its own because the sweep can get this wrong in two opposite ways: merging
        the pair invents one interval where the producer wrote two, and treating a shared
        boundary as a collision opens a hole nobody measured.
        """
        root, video = _clip(tmp_path)
        _write(SEGMENTS_SCHEMA, root / "speech" / "segments.parquet", [
            _segment_row("seg-0", 0.0, 1.0, "first"),
            _segment_row("seg-1", 1.0, 2.0, "second"),
        ])
        got = annotations(eaf_of({"dir": root, "video": video}), "segments_src")
        assert [(s, e) for s, e, _t in got] == [(0, 1000), (1000, 2000)]
        assert [t for _s, _e, t in got] == ["SPEAKER_00: first · [seg-0]",
                                            "SPEAKER_00: second · [seg-1]"]
        assert projection_of(build_eaf(root, video, log=lambda *a, **k: None)[0]) == {}

    # ------------------------------------------------------------------- the label shape

    def test_a_single_active_label_is_still_a_plain_string(self, tmp_path: Path) -> None:
        """Where nothing overlapped, the file must be exactly what it was before.

        JSON everywhere would rewrite every tier in the corpus for a rule that only bites on
        simultaneous rows, and a reviewer would not be able to see which intervals changed.
        """
        root, video = _clip(tmp_path)
        _write(SEGMENTS_SCHEMA, root / "speech" / "segments.parquet", [
            _segment_row("seg-0", 0.0, 1.0, "hola")])
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        assert annotations(eaf, "segments_src") == [(0, 1000, "SPEAKER_00: hola · [seg-0]")]
        assert projection_of(eaf) == {}

    def test_the_group_label_is_machine_readable_and_names_no_invented_delimiter(
            self, tmp_path: Path) -> None:
        """A group is a JSON list, so a label containing the delimiter is not ambiguous.

        The alternative the design considered — joining with a delimiter and listing the members
        in metadata — has to survive a producer that *writes that delimiter*. JSON survives any
        text, so that is what the format is.
        """
        root, video = _clip(tmp_path)
        _write(SEGMENTS_SCHEMA, root / "speech" / "segments.parquet", [
            _segment_row("seg-0", 0.0, 1.0, 'weird · "quoted" · [seg-9]'),
            _segment_row("seg-1", 0.5, 1.5, "normal"),
        ])
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        overlap = [t for _s, _e, t in annotations(eaf, "segments_src") if t.startswith("[")]
        assert len(overlap) == 1
        members = json.loads(overlap[0])
        assert len(members) == 2
        assert any('weird · "quoted" · [seg-9]' in member for member in members)
        assert any("seg-9" in member for member in members)

    def test_a_single_label_that_looks_like_a_group_stays_a_plain_string(
            self, tmp_path: Path) -> None:
        """A producer's text may begin with `[`; that must not be read as a group.

        A lone label is written as the plain text it always was, even inside a projected tier, so a
        transcript that opens with a bracket (`[inaudible] …`) survives unchanged. Exact membership
        never depends on guessing from the first character: the property lists the one text of each
        single-label segment's row.
        """
        root, video = _clip(tmp_path)
        _write(SEGMENTS_SCHEMA, root / "speech" / "segments.parquet", [
            _segment_row("seg-0", 0.0, 1.0, "[inaudible] 3 words"),
            _segment_row("seg-1", 0.5, 1.0, "cover"),
        ])
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        emitted = annotations(eaf, "segments_src")
        assert emitted[0][2] == "SPEAKER_00: [inaudible] 3 words · [seg-0]"
        json.loads(emitted[1][2])  # the overlapping segment is the only list
        rows = {row["text"]: row for row in projection_of(eaf)["segments_src"]["logical"]}
        lone = rows["SPEAKER_00: [inaudible] 3 words · [seg-0]"]
        # Alone on the first segment, sharing the second — and both memberships are recorded.
        assert lone["segments"] == [[0, 500], [500, 1000]]
        assert lone["source"]["segment_id"] == "seg-0"

    def test_unicode_and_whitespace_survive_a_group_label(self, tmp_path: Path) -> None:
        """The projection must not be where a producer's text finally gets mangled."""
        root, video = _clip(tmp_path)
        _write(SEGMENTS_SCHEMA, root / "speech" / "segments.parquet", [
            _segment_row("seg-0", 0.0, 1.0, "  Ñandú — 75%  ·  ¡oye!  "),
            _segment_row("seg-1", 0.5, 1.0, "😀 你好"),
        ])
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        out = root / "unicode.eaf"
        eaf.to_file(str(out))
        from pympi.Elan import Eaf

        reopened = Eaf(str(out), suppress_version_warning=True)
        group = json.loads([t for _s, _e, t in annotations(reopened, "segments_src")
                            if t.startswith("[")][0])
        assert group == ["SPEAKER_00: Ñandú — 75% · ¡oye! · [seg-0]",
                         "SPEAKER_00: 😀 你好 · [seg-1]"]

    # ------------------------------------------------------- identities are never merged

    def test_two_rows_with_the_same_text_keep_two_ids_in_the_group(self,
                                                                  tmp_path: Path) -> None:
        """Deduplicating by text would erase one producer's row.

        Two segments with the same words and different `segment_id`s are two measurements, and
        the ids are the only thing in the label that says so.
        """
        root, video = _clip(tmp_path)
        _write(SEGMENTS_SCHEMA, root / "speech" / "segments.parquet", [
            _segment_row("seg-0", 0.0, 1.0, "same words"),
            _segment_row("seg-1", 0.0, 1.0, "same words"),
        ])
        group = json.loads(annotations(eaf_of({"dir": root, "video": video}),
                                       "segments_src")[0][2])
        assert group == ["SPEAKER_00: same words · [seg-0]",
                         "SPEAKER_00: same words · [seg-1]"]

    @pytest.mark.parametrize("order", ["forward", "reversed"],
                             ids=["forward", "reversed"])
    def test_the_group_order_does_not_depend_on_the_input_order(
            self, tmp_path: Path, order: str) -> None:
        """Determinism: identical tables in, identical bytes out.

        Rows with identical endpoints are the hard case — a producer's file order decides the
        sort's tie, so a group ordered by the sort would flip its label between two runs of the
        same data.
        """
        rows = [_segment_row("seg-1", 0.0, 1.0, "beta"),
                _segment_row("seg-0", 0.0, 1.0, "alpha"),
                _segment_row("seg-2", 0.0, 1.0, "alpha")]
        if order == "reversed":
            rows.reverse()
        root, video = _clip(tmp_path)
        _write(SEGMENTS_SCHEMA, root / "speech" / "segments.parquet", rows)
        group = json.loads(annotations(eaf_of({"dir": root, "video": video}),
                                       "segments_src")[0][2])
        assert group == ["SPEAKER_00: alpha · [seg-0]", "SPEAKER_00: alpha · [seg-2]",
                         "SPEAKER_00: beta · [seg-1]"]

    def test_two_rows_that_round_to_one_interval_stay_two_source_intervals(
            self, tmp_path: Path) -> None:
        """The millisecond grid collides sub-millisecond rows; identity and metadata must not.

        ELAN can only store one interval here, so the projection has one *emitted* segment — but
        it carries both texts and the property records both source intervals, so neither row is
        lost just because the format is coarse.
        """
        root, video = _clip(tmp_path)
        _write(SEGMENTS_SCHEMA, root / "speech" / "segments.parquet", [
            _segment_row("seg-0", 1.2341, 1.2349, "first"),
            _segment_row("seg-1", 1.2344, 1.2348, "second"),
        ])
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        emitted = annotations(eaf, "segments_src")
        assert [s for s, _e, _t in emitted] == [1234]
        assert json.loads(emitted[0][2])[1].startswith("SPEAKER_00: second")
        logical = projection_of(eaf)["segments_src"]["logical"]
        assert [(row["start_ms"], row["end_ms"]) for row in logical] == [(1234, 1235),
                                                                        (1234, 1235)]
        assert sorted(row["row_id"] for row in logical) == ["segments_src:0", "segments_src:1"]

    # ------------------------------------------------------------------- traceability

    def test_the_property_maps_every_source_interval_and_its_segment_membership(
            self, tmp_path: Path) -> None:
        """An interval split across three segments stays one row with three memberships.

        The union of a row's segments has to be the row's own interval — that is what makes the
        property a *reconstruction* rather than a summary: given it, a reader recovers which
        producer row said what, and over what range.
        """
        root, video = _clip(tmp_path)
        _write(SEGMENTS_SCHEMA, root / "speech" / "segments.parquet", [
            _segment_row("seg-0", 0.0, 2.0, "outer"),
            _segment_row("seg-1", 0.5, 1.5, "inner"),
        ])
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        document = projection_of(eaf)["segments_src"]
        logical = {row["row_id"]: row for row in document["logical"]}
        assert sorted(logical) == ["segments_src:0", "segments_src:1"]
        outer = logical["segments_src:0"]
        assert outer["source"] == {"segment_id": "seg-0", "speaker_id": "SPEAKER_00"}
        assert outer["start_seconds"] == 0.0 and outer["end_seconds"] == 2.0
        assert (outer["start_ms"], outer["end_ms"]) == (0, 2000)
        assert outer["text"] == "SPEAKER_00: outer · [seg-0]"
        assert outer["segments"] == [[0, 500], [500, 1500], [1500, 2000]]
        assert logical["segments_src:1"]["segments"] == [[500, 1500]]
        for row in logical.values():
            union_start = min(start for start, _e in row["segments"])
            union_end = max(end for _s, end in row["segments"])
            assert (union_start, union_end) >= (row["start_ms"], row["end_ms"])
            assert union_start == row["start_ms"]

    def test_the_projection_counts_final_and_logical_annotations_apart(
            self, tmp_path: Path) -> None:
        """Two numbers, two states: what ELAN shows and what the producer wrote.

        One number would have to be either "the tables had 5 rows" (false of the file) or "the
        file has 3 annotations" (false of the tables), and the census line is quoted as evidence.
        """
        root, video = _clip(tmp_path)
        _write(SEGMENTS_SCHEMA, root / "speech" / "segments.parquet", [
            _segment_row("seg-0", 0.0, 2.0, "outer"),
            _segment_row("seg-1", 0.5, 1.5, "inner"),
            _segment_row("seg-2", 3.0, 4.0, "alone"),
        ])
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        document = projection_of(eaf)["segments_src"]
        assert document["logical_row_count"] == 3
        assert document["final_annotation_count"] == 4
        assert len(annotations(eaf, "segments_src")) == 4
        census = dict(eaf.properties)["pipeline-tiers"]
        assert "segments_src=4" in census, census

    def test_the_projection_document_carries_a_version_and_survives_the_round_trip(
            self, tmp_path: Path) -> None:
        """A reader must be able to tell this shape from whatever replaces it.

        Written, re-read from disk, still valid JSON: a property nobody can reopen is not
        traceability, and the projection is the only place the producers' own intervals live once
        a tier has been re-cut.
        """
        root, video = _clip(tmp_path)
        _write(SEGMENTS_SCHEMA, root / "speech" / "segments.parquet", [
            _segment_row("seg-0", 0.0, 2.0, "outer"),
            _segment_row("seg-1", 0.5, 1.5, "inner")])
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        assert projection_version(eaf) == 1
        out = root / "projection.eaf"
        eaf.to_file(str(out))
        ET.parse(out)
        from pympi.Elan import Eaf

        reopened = Eaf(str(out), suppress_version_warning=True)
        assert projection_of(reopened) == projection_of(eaf)
        assert projection_version(reopened) == 1
        # Every logical row's segments tile its own interval with no internal gap or overlap.
        for row in projection_of(reopened)["segments_src"]["logical"]:
            segments = row["segments"]
            assert segments[0][0] == row["start_ms"]
            assert segments[-1][1] == row["end_ms"]
            assert all(segments[i][1] == segments[i + 1][0]
                       for i in range(len(segments) - 1)), segments

    def test_only_an_affected_tier_appears_in_the_property(self, dataset: dict[str, Path]
                                                         ) -> None:
        """The synthetic clip has no same-tier overlap, so it gets no property at all.

        Checked by absence rather than by an empty dict per tier: a document that lists seventeen
        unaffected tiers is a document where a reader cannot see which two were rewritten.
        """
        eaf = eaf_of(dataset)
        assert PROJECTION_PROPERTY not in dict(eaf.properties)
        assert projection_of(eaf) == {}

    def test_the_semantics_property_explains_the_projection_as_reexpression(
            self, tmp_path: Path) -> None:
        """A reader of the file alone must not take a segment for a new event.

        The property names the rule it enforces, says the segments are not new measurements, and
        points at the property holding the producers' own intervals.
        """
        root, video = _clip(tmp_path)
        _write(SEGMENTS_SCHEMA, root / "speech" / "segments.parquet", [
            _segment_row("seg-0", 0.0, 1.0, "a"), _segment_row("seg-1", 0.5, 1.5, "b")])
        text = dict(eaf_of({"dir": root, "video": video}).properties)["pipeline-tier-semantics"]
        clause = text[text.index("Independent tiers:"):]
        assert "may not overlap" in clause
        assert "not new" in clause and "measurement" in clause
        assert PROJECTION_PROPERTY in clause
        assert "logical_row_count" in clause and "final_annotation_count" in clause

    def test_a_dropped_row_is_not_counted_in_the_logical_rows(self, tmp_path: Path) -> None:
        """The two drop rules stay independent: a row with no time never reaches the sweep.

        It is not a projection artefact and must not inflate `logical_row_count` — the count has
        to reconcile against the tier's own rows, or the property is unverifiable.
        """
        root, video = _clip(tmp_path)
        _write(WORDS_SCHEMA, root / "speech" / "words.parquet", [
            _word("hello", 0.0, 0.4), _word("no-time", 1.0, None)])
        lines: list[str] = []
        eaf, _report = build_eaf(root, video, log=lambda msg, *a, **k: lines.append(str(msg)))
        assert len(annotations(eaf, "words")) == 1
        assert [line for line in lines if "words" in line and "missing timestamp" in line]
        assert PROJECTION_PROPERTY not in dict(eaf.properties)
        assert dict(eaf.properties)["pipeline-tiers"] == "words=1"

    # -------------------------------------------------- the sweep against a second oracle

    @pytest.mark.parametrize("rows", [
        [(0.0, 1.0), (0.2, 0.4), (0.9, 1.4)],
        [(0.0, 0.3), (0.3, 0.6), (0.6, 0.9)],
        [(1.0, 2.0), (0.0, 3.0), (0.5, 1.5), (2.5, 2.75)],
        [(0.0, 1.0), (0.0, 0.5), (0.5, 1.0)],
        [(2.0, 3.0), (0.0, 0.5)],
    ], ids=["nested-three", "chain-touching", "crossing", "split-at-half", "disjoint"])
    def test_no_emitted_interval_of_a_tier_overlaps_any_other(self, tmp_path: Path,
                                                             rows: list[tuple[float, float]]
                                                             ) -> None:
        """Brute force over the pairs, plus an independently swept boundary set.

        This is the check the format demands and the one pympi cannot do: pairwise disjointness
        of the emitted pairs, with touching allowed, over shapes no hand-written case covers.
        """
        root, video = _clip(tmp_path)
        _write(SEGMENTS_SCHEMA, root / "speech" / "segments.parquet",
               [_segment_row(f"seg-{i}", start, end, f"t{i}")
                for i, (start, end) in enumerate(rows)])
        emitted = annotations(eaf_of({"dir": root, "video": video}), "segments_src")
        pairs = sorted((start, end) for start, end, _t in emitted)
        assert all(pairs[i][1] <= pairs[i + 1][0] for i in range(len(pairs) - 1)), pairs
        assert [s for s, _e in pairs] == sorted({s for s, _e in pairs})
        # Every source endpoint survives as some emitted boundary.
        boundaries = {value for pair in pairs for value in pair}
        for start, end in rows:
            assert int(start * 1000) in boundaries, (start, end, pairs)
            assert int(end * 1000) in boundaries or int(end * 1000) == int(start * 1000) + 1
        # ... and every source row is represented somewhere in the labels.
        joined = " ".join(t for _s, _e, t in emitted)
        for i, (_start, _end) in enumerate(rows):
            assert f"[seg-{i}]" in joined
        assert pairs == [(int(a * 1000), int(b * 1000)) for a, b in _sweep_boundaries(rows)]

    def test_a_projected_tier_leaves_every_other_tier_alone(self, tmp_path: Path) -> None:
        """Per-tier isolation survives the projection.

        `segments_src` overlaps and is re-cut; `words` does not overlap and must come out with the
        intervals, labels and *order* it always had, and must not appear in the property at all.
        A projection that bled across tiers would re-time the transcript of every clip that has two
        speakers, which is the failure an analyst would notice last and trust first.
        """
        root, video = _clip(tmp_path)
        _write(WORDS_SCHEMA, root / "speech" / "words.parquet", [
            _word("hello", 0.0, 0.4), _word("there", 0.4, 0.6)])
        _write(SEGMENTS_SCHEMA, root / "speech" / "segments.parquet", [
            _segment_row("seg-0", 0.0, 2.0, "outer"),
            _segment_row("seg-1", 0.5, 1.5, "inner")])
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        assert projection_of(eaf)["segments_src"]["final_annotation_count"] == 3
        assert list(projection_of(eaf)) == ["segments_src"]
        assert annotations(eaf, "words") == [
            (0, 400, "hello · SPEAKER_00 · w-0.0 · [seg-0]"),
            (400, 600, "there · SPEAKER_00 · w-0.4 · [seg-0]")]

    def test_rows_written_out_of_time_order_project_to_the_same_document(
            self, tmp_path: Path) -> None:
        """The tier's rows come from a sort; the projection must not depend on that sort's ties.

        Two rows with identical endpoints are the case a sort cannot order (both key to the same
        tuple), so the group order is decided by the projection's own tie-break. Written here in
        reverse table order and compared against the forward case's expected label, this dies if
        the group ever inherits Parquet's row order.
        """
        root, video = _clip(tmp_path)
        _write(SEGMENTS_SCHEMA, root / "speech" / "segments.parquet", [
            _segment_row("seg-1", 0.5, 1.0, "inner"),
            _segment_row("seg-0", 0.0, 1.0, "outer")])
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        emitted = annotations(eaf, "segments_src")
        assert [(s, e) for s, e, _t in emitted] == [(0, 500), (500, 1000)]
        assert emitted[0][2] == "SPEAKER_00: outer · [seg-0]"
        assert json.loads(emitted[1][2]) == ["SPEAKER_00: inner · [seg-1]",
                                             "SPEAKER_00: outer · [seg-0]"]
        # ... and the same document the forward-ordered table produces, row for row.
        forward = _clip(tmp_path.parent / "forward")
        _write(SEGMENTS_SCHEMA, forward[0] / "speech" / "segments.parquet", [
            _segment_row("seg-0", 0.0, 1.0, "outer"),
            _segment_row("seg-1", 0.5, 1.0, "inner")])
        other = build_eaf(forward[0], forward[1], log=lambda *a, **k: None)[0]
        assert projection_of(other) == projection_of(eaf)
        assert annotations(other, "segments_src") == emitted

    def test_a_reopened_document_reads_its_projection_through_the_shared_reader(
            self, tmp_path: Path) -> None:
        """Consumers get one reader, and a document without the property reads as "nothing".

        A legacy .eaf — non-overlapping, so it carries no property at all — must be readable by the
        same call that reads a projected one: absence is the compatible state the rule was designed
        to leave alone, not an error and not an empty dict that has to be special-cased.
        """
        from pympi.Elan import Eaf

        from multimodal_pipeline.elan import overlap_projection

        root, video = _clip(tmp_path)
        _write(SEGMENTS_SCHEMA, root / "speech" / "segments.parquet", [
            _segment_row("seg-0", 0.0, 1.0, "first")])
        legacy, _report = build_eaf(root, video, log=lambda *a, **k: None)
        assert overlap_projection(legacy) == {}
        out = root / "legacy.eaf"
        legacy.to_file(str(out))
        assert overlap_projection(Eaf(str(out), suppress_version_warning=True)) == {}

        _write(SEGMENTS_SCHEMA, root / "speech" / "segments.parquet", [
            _segment_row("seg-0", 0.0, 1.0, "first"),
            _segment_row("seg-1", 0.5, 1.5, "second")])
        projected, _report = build_eaf(root, video, log=lambda *a, **k: None)
        projected.to_file(str(out))
        reopened = Eaf(str(out), suppress_version_warning=True)
        document = overlap_projection(reopened)["segments_src"]
        assert document["logical_row_count"] == 2
        assert document["final_annotation_count"] == 3
        assert [row["source"]["segment_id"] for row in document["logical"]] == ["seg-0", "seg-1"]

    def test_the_corpus_projection_maps_back_to_the_producers_rows(self) -> None:
        """The traceability claim, tested on the tables that motivated it.

        For every dataset on this disk, each logical `person_tracks` row has to name a `person_id`
        the frames table reports and an interval equal to that id's own sighting endpoints —
        re-derived here from `persons/frames.parquet`, without calling the tier. That is the check
        that says the property reconstructs the source mapping, rather than merely describing the
        bars it was generated with.
        """
        checked = 0
        for root in sorted(p for p in PROCESSED.iterdir()
                           if (p / "manifest.json").is_file()):
            frames_path = root / ARTIFACT_LAYOUT["person_frames"]
            if not frames_path.is_file():
                continue
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            video = Path(manifest["source"]["path"])
            if not video.is_file():
                continue
            eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
            document = projection_of(eaf).get("person_tracks")
            if document is None:
                continue
            rows = read_table(frames_path,
                              columns=["person_id", "timestamp"]).to_pylist()
            endpoints: dict[int, list[float]] = {}
            for row in rows:
                if row["person_id"] is None or row["timestamp"] is None:
                    continue
                endpoints.setdefault(int(row["person_id"]), []).append(float(row["timestamp"]))
            logical = document["logical"]
            assert document["logical_row_count"] == len(logical)
            assert document["final_annotation_count"] == len(annotations(eaf, "person_tracks"))
            for entry in logical:
                person_id = entry["source"]["person_id"]
                stamps = endpoints.get(person_id)
                assert stamps, f"{root.name}: projected row for unknown id {person_id}"
                # The tier's logical row is one sighting run: its endpoints are measurements taken
                # from this id's own timestamps, never an invented or grid-extended value.
                assert entry["start_seconds"] in stamps, (root.name, entry)
                assert entry["end_seconds"] in stamps, (root.name, entry)
                assert entry["start_ms"] <= entry["end_ms"]
                assert entry["segments"], (root.name, entry)
                checked += 1
        assert checked, "no projected person tier found on this disk; the loop proved nothing"

    def test_the_corpus_export_emits_no_same_tier_overlap(self) -> None:
        """The rule is measured on the real tables, where simultaneous rows already exist.

        Two people in one frame and two TalkNet tracks alive at once are not hypothetical here:
        `persons/frames.parquet` and the ASD tracks table carry them on this disk.
        """
        for name in sorted(p.name for p in PROCESSED.glob("*") if (p / "manifest.json").is_file()):
            root = PROCESSED / name
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            video = Path(manifest["source"]["path"])
            if not video.is_file():
                continue
            eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
            for tier, (annotations_of_tier, _ref, _dict, _type) in eaf.tiers.items():
                if tier == "default":
                    continue
                pairs = sorted((int(eaf.timeslots[start]), int(eaf.timeslots[end]))
                               for start, end, _v, _svg in annotations_of_tier.values())
                assert all(pairs[i][1] <= pairs[i + 1][0]
                           for i in range(len(pairs) - 1)), f"{name}/{tier}: {pairs}"


def eaf_only(dataset_dir: Path, video: Path) -> Any:
    """The document from `build_eaf`, without its report.

    The tuple is what B5 added, so most of this file unpacks it. Where a test cares only about the
    bars, this says so rather than leaving a throwaway `_report` in the signature.
    """
    return build_eaf(dataset_dir, video, log=lambda *a, **k: None)[0]


#: The three normalised tables no tier reads on this corpus, named once here so the tests below
#: assert the *set* rather than three copies of a guess.
UNREAD_TABLES = ("pose_face", "pose_hands", "pose_normalized")


def coverage_map(eaf: Any) -> dict[str, Any]:
    """The document's own per-artifact inventory, parsed, or `{}` when it carries none."""
    return elan_core.coverage_of(eaf)


def corpus_datasets() -> list[Path]:
    """Every dataset directory on this disk, in name order."""
    if not PROCESSED.is_dir():
        return []
    return sorted(p for p in PROCESSED.iterdir() if (p / "manifest.json").is_file())


def corpus_video(root: Path) -> Path:
    """The source video the dataset's own manifest names — never a path this file invents."""
    return Path(json.loads((root / "manifest.json").read_text(encoding="utf-8"))
                ["source"]["path"])


def registry_parquet_states(dataset_dir: Path) -> dict[str, str]:
    """Every `.parquet` artifact's state, recomputed here from the registry and the filesystem.

    Deliberately **not** `coverage_inventory`: it re-derives "which table does a tier read" from
    `TIERS`/`SECONDARY_INPUTS` and re-stats the path from `ARTIFACT_LAYOUT`, so the state map is
    rebuilt rather than echoed. A test that compared the property against the writer's own function
    would pass when the writer was wrong, which is the failure B5 has to be able to see.
    """
    exported = {spec.artifact: spec.tier for spec in TIERS}
    consumers: dict[str, str] = {}
    for tier, artifacts in SECONDARY_INPUTS.items():
        for artifact in artifacts:
            consumers.setdefault(artifact, tier)
    states: dict[str, str] = {}
    for artifact, relative in ARTIFACT_LAYOUT.items():
        if not relative.endswith(".parquet"):
            continue
        if not (dataset_dir / relative).is_file():
            states[artifact] = elan_core.COVERAGE_ABSENT
        elif artifact in exported:
            states[artifact] = elan_core.COVERAGE_EXPORTED
        elif artifact in consumers:
            states[artifact] = elan_core.COVERAGE_SUMMARISED
        else:
            states[artifact] = elan_core.COVERAGE_PRESENT_NOT_EXPORTED
    return states


class TestCoverageInventory:
    """What the export represents, what it read as support, and what it left out — in the file.

    The defect is not a wrong number. An opened `.eaf` proves what it contains and nothing in it
    proves what was never exported, so "this clip has no person data" and "this export never
    represents pose" both arrive at the reader as *a tier that is not there*. One of those is a
    fact about the video, the other is a fact about this file, and only the first is answerable
    from the grid. These tests pin the four states, the one distinction between the two "not in
    the file" states, and the two ways the inventory could quietly stop being true (a hand-edited
    document, and a table added to the registry after the list was last written).
    """

    # --------------------------------------------------------------- the property itself

    def test_the_document_inventories_every_parquet_artifact_with_one_state(
            self, dataset: dict[str, Path]) -> None:
        """22 tables in, 22 entries out, each holding exactly one of the four words.

        The count is taken from the registry rather than typed, because the claim is "every
        normalised table appears", not "22 appear today". The state vocabulary is asserted as a
        closed set so a fifth state cannot be introduced without being named.
        """
        eaf = eaf_of(dataset)
        inventory = coverage_map(eaf)
        parquet = {name for name, relative in ARTIFACT_LAYOUT.items()
                   if relative.endswith(".parquet")}
        assert set(inventory) == parquet
        assert len(inventory) == 23, "the registry's normalised-table count changed shape"
        assert {entry["state"] for entry in inventory.values()} <= set(elan_core.COVERAGE_STATES)

    def test_the_inventory_is_a_json_property_that_survives_the_round_trip(
            self, dataset: dict[str, Path], tmp_path: Path) -> None:
        """The claim has to travel with the file, so it has to survive `to_file`.

        pympi's in-memory property dict is not what an operator opens; the reopened document is.
        Reading the raw XML too, because that is the copy a hand edit reaches.
        """
        from pympi.Elan import Eaf

        eaf = eaf_of(dataset)
        assert elan_core.COVERAGE_PROPERTY in dict(eaf.properties)
        out = dataset["dir"] / "round_trip.eaf"
        eaf.to_file(str(out))
        reopened = Eaf(str(out), suppress_version_warning=True)
        assert coverage_map(reopened) == coverage_map(eaf)
        root = ET.fromstring(out.read_text(encoding="utf-8"))
        raw = next(element.text for element in root.iter("PROPERTY")
                   if element.attrib.get("NAME") == elan_core.COVERAGE_PROPERTY)
        document = json.loads(raw)
        assert document["version"] == elan_core.COVERAGE_VERSION
        assert document["artifacts"] == coverage_map(eaf)

    def test_a_table_with_a_tier_is_exported_and_names_that_tier(
            self, dataset: dict[str, Path]) -> None:
        """The state carries the tier's name, so a reader can go from table to bar."""
        inventory = coverage_map(eaf_of(dataset))
        assert inventory["speech_words"] == {
            "state": elan_core.COVERAGE_EXPORTED, "tiers": ["words"],
            "path": ARTIFACT_LAYOUT["speech_words"]}

    def test_a_table_read_as_support_is_summarised_and_names_the_consuming_tier(
            self, tmp_path: Path) -> None:
        """`persons/frames.parquet` places every sighting bar and no bar is built *from* it.

        That is a third state, not a variant of "exported": the table is read, and it is still not
        an exported analysis of its own. The test builds the persons tables so the tier really is
        written, because an absent-people fixture would report `absent` and prove nothing about the
        distinction between *read as support* and *written as a tier*.
        """
        root, video = person_clip(tmp_path)
        _persons(root, [_person_frame(0, 1), _person_frame(1, 1)], [_person_track(1, 0.0, 0.04, 2)])
        _frame_index(root, [_frame_row(0, 0.0), _frame_row(1, 1 / 29.97)])
        inventory = coverage_map(eaf_of({"dir": root, "video": video}))
        for artifact in SECONDARY_INPUTS["person_tracks"]:
            assert inventory[artifact]["state"] == elan_core.COVERAGE_SUMMARISED, artifact
        assert inventory["person_frames"]["tiers"] == ["person_tracks"]
        assert inventory["frame_index"]["tiers"] == ["person_tracks"]
        # And the distinction holds inside one document: the tier's own table is `exported`.
        assert inventory["person_tracks"]["state"] == elan_core.COVERAGE_EXPORTED
        assert inventory["person_tracks"]["tiers"] == ["person_tracks"]

    # ------------------------------------------- the two "not in the file" states are two

    def test_the_same_artifact_is_present_not_exported_or_absent_dependent_on_the_file(
            self, tmp_path: Path) -> None:
        """The whole point of the property, on one artifact name and one tier set.

        `pose_face` is read by no tier, so writing the table then deleting it moves the state from
        `present, not exported` to `absent` with `TIERS` untouched. Merge the two states — into one
        "not exported", or by deciding states from the tier list alone and never looking at the
        disk — and this test dies, which is the point: the merged version is the one that reads as
        "this clip has no person data" while 16,799 rows sit unread on the shelf.
        """
        root, video = _clip(tmp_path)
        _write(WORDS_SCHEMA, root / "speech" / "words.parquet", [_word("hello", 0.0, 0.4)])
        path = root / ARTIFACT_LAYOUT["pose_face"]

        _write(FACE_SCHEMA, path, [_face_row(0, 0)])
        present = coverage_map(eaf_only(root, video))["pose_face"]
        assert present["state"] == elan_core.COVERAGE_PRESENT_NOT_EXPORTED
        assert "dense per-joint numeric tracks" in present["reason"]

        path.unlink()
        absent = coverage_map(eaf_only(root, video))["pose_face"]
        assert absent["state"] == elan_core.COVERAGE_ABSENT
        assert absent != present
        assert "reason" not in absent, "an artifact nobody wrote has no export decision to explain"

    def test_an_empty_but_present_table_is_still_present_not_exported(
            self, tmp_path: Path) -> None:
        """0 rows is a result; no file is a different result, and the tier set sees neither.

        Every `pipeline_demo` on this disk has a real, empty `pose/face.parquet`. Calling that
        `absent` would tell the operator the stage never ran, when the honest sentence is that it
        ran, found nothing, and this export does not represent those rows anyway.
        """
        root, video = _clip(tmp_path)
        _write(WORDS_SCHEMA, root / "speech" / "words.parquet", [_word("hello", 0.0, 0.4)])
        _write(FACE_SCHEMA, root / ARTIFACT_LAYOUT["pose_face"], [])
        entry = coverage_map(eaf_only(root, video))["pose_face"]
        assert entry["state"] == elan_core.COVERAGE_PRESENT_NOT_EXPORTED

    def test_the_three_unread_tables_are_the_only_ones_with_a_reason_and_each_is_specific(
            self, tmp_path: Path) -> None:
        """Every `present, not exported` names what was deferred; nothing else carries a reason.

        The reason is the half that must not be promotional or vague, and it must not be one
        sentence pasted onto three tables that differ: `pose_normalized` is a change of basis over
        the same BODY_25 keypoints, not a third set of joints, and a reader handed the hands/face
        wording there would go looking for a body-pose tier that already exists.
        """
        dataset = make_dataset(tmp_path)
        _write_unread_pose_tables(dataset["dir"])
        inventory = coverage_map(eaf_of(dataset))
        unread = {name for name, entry in inventory.items()
                  if entry["state"] == elan_core.COVERAGE_PRESENT_NOT_EXPORTED}
        assert unread == set(UNREAD_TABLES), inventory
        for name in unread:
            assert inventory[name]["reason"], name
        assert inventory["pose_hands"]["reason"] == inventory["pose_face"]["reason"]
        assert inventory["pose_normalized"]["reason"] != inventory["pose_face"]["reason"]
        # The checkable half of each sentence: it names the table a pose tier does read.
        for name in unread:
            assert ARTIFACT_LAYOUT["pose_body"].split("/")[-1] in inventory[name]["reason"], name

    def test_a_reason_the_export_cannot_name_says_so_instead_of_staying_silent(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unmapped unread table gets the honest blank, not a neighbour's wording.

        The alternative to admitting it has no specific reason is the tempting one — reuse the
        pose sentence, which sounds right for anything numeric. That is how a documented reason
        becomes a plausible fiction, so the fallback is a separate state-carrying string.
        """
        root, video = _clip(tmp_path)
        _write(WORDS_SCHEMA, root / "speech" / "words.parquet", [_word("hello", 0.0, 0.4)])
        _write(FACE_SCHEMA, root / ARTIFACT_LAYOUT["pose_face"], [_face_row(0, 0)])
        monkeypatch.setattr(elan_core, "TIER_ABSENT_REASONS", {})
        entry = coverage_map(eaf_only(root, video))["pose_face"]
        assert entry["state"] == elan_core.COVERAGE_PRESENT_NOT_EXPORTED
        assert entry["reason"] == elan_core.COVERAGE_REASON_UNKNOWN
        assert "no specific reason" in entry["reason"]

    # -------------------------------------------------------- absence outranks the tier list

    def test_a_tier_named_after_a_table_that_was_never_written_does_not_make_it_exported(
            self, tmp_path: Path) -> None:
        """`gloss_en` exists in `TIERS`; this dataset has no translation table.

        Deciding the states from the tier list alone — which is the cheap implementation, and the
        one that never looks at a file — would report `translation_segments` as `exported` and then
        have to explain a tier with no bars in it. Disk first is what makes the property answer
        "what could this export read?" rather than "what tiers does the code declare?"
        """
        root, video = _clip(tmp_path)
        _write(WORDS_SCHEMA, root / "speech" / "words.parquet", [_word("hello", 0.0, 0.4)])
        inventory = coverage_map(eaf_only(root, video))
        assert inventory["translation_segments"]["state"] == elan_core.COVERAGE_ABSENT
        assert "reason" not in inventory["translation_segments"]
        # The tier itself is absent from the document, and the census says nothing about it.
        assert "gloss_en" not in tiers_of(eaf_only(root, video))

    def test_a_table_that_exists_but_cannot_be_read_is_still_exported(
            self, tmp_path: Path) -> None:
        """A tier that reads a table and fails is not a table this export never represents.

        The file is there; the tier is skipped and the skip is reported by name in the census line
        and in the stage's `skipped_tiers`. Relabelling the artifact `present, not exported` would
        turn a corrupt producer into a design decision about the export — the exact collapse of
        "broken" into "not represented" the whole property exists to prevent.
        """
        root, video = _clip(tmp_path)
        _write(WORDS_SCHEMA, root / "speech" / "words.parquet", [_word("hello", 0.0, 0.4)])
        path = root / ARTIFACT_LAYOUT["speech_segments"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a parquet file")
        inventory = coverage_map(eaf_only(root, video))
        assert inventory["speech_segments"]["state"] == elan_core.COVERAGE_EXPORTED
        assert inventory["speech_segments"]["tiers"] == ["segments_src"]

    # ------------------------------------------------------------ derived, never hand-listed

    def test_a_registry_artifact_nobody_listed_appears_as_absent(
            self, dataset: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
        """A future table shows up without anyone editing a coverage list.

        The registry is patched rather than the property: an assertion that walked a hand-written
        list of 22 names would keep passing after the 23rd table arrived, which is the only way a
        coverage inventory can actually rot.
        """
        import multimodal_pipeline.artifacts as artifacts

        monkeypatch.setitem(artifacts.ARTIFACT_LAYOUT, "eye_gaze", "gaze/eye.parquet")
        inventory = coverage_map(eaf_of(dataset))
        assert "eye_gaze" in inventory
        assert inventory["eye_gaze"]["state"] == elan_core.COVERAGE_ABSENT
        assert inventory["eye_gaze"]["path"] == "gaze/eye.parquet"

    def test_a_new_registry_artifact_on_disk_is_present_not_exported_without_a_reason(
            self, dataset: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
        """Same patch, file written: the state flips, and the reason is the admitted blank.

        Two tests rather than one because the two states a new artifact can land in are the two
        the operator has to be able to tell apart, and a table nobody has written a reason for yet
        must not inherit one.
        """
        import multimodal_pipeline.artifacts as artifacts

        monkeypatch.setitem(artifacts.ARTIFACT_LAYOUT, "eye_gaze", "gaze/eye.parquet")
        target = dataset["dir"] / "gaze" / "eye.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        inventory = coverage_map(eaf_of(dataset))
        assert inventory["eye_gaze"]["state"] == elan_core.COVERAGE_PRESENT_NOT_EXPORTED
        assert inventory["eye_gaze"]["reason"] == elan_core.COVERAGE_REASON_UNKNOWN

    def test_raw_tool_outputs_are_not_inventoried(self, dataset: dict[str, Path]) -> None:
        """The inventory is about the normalised tables the export summarises, not every file.

        `pose/raw/` and `manifest.json` are registry artifacts too, and a document that mixed the
        two would bury the answer under 25 rows a reader cannot act on. The `path` of every entry
        ends in `.parquet`, which is the rule the inventory is built from.
        """
        inventory = coverage_map(eaf_of(dataset))
        assert all(entry["path"].endswith(".parquet") for entry in inventory.values())
        for raw in ("pose_raw", "manifest", "elan_annotations", "acoustic_raw"):
            assert raw not in inventory

    def test_the_inventory_matches_a_state_map_rebuilt_from_the_registry_and_disk(
            self, tmp_path: Path) -> None:
        """Independent recomputation over a directory with real tables in it.

        `registry_parquet_states` re-walks `ARTIFACT_LAYOUT` and stats every path itself. Where the
        two disagree, one of them is wrong about a file that exists.
        """
        root, video = person_clip(tmp_path)
        _persons(root, [_person_frame(0, 1)], [_person_track(1, 0.0, 0.0, 1)])
        _frame_index(root, [_frame_row(0, 0.0)])
        _write(FACE_SCHEMA, root / ARTIFACT_LAYOUT["pose_face"], [_face_row(0, 0)])
        inventory = coverage_map(eaf_only(root, video))
        assert {name: entry["state"] for name, entry in inventory.items()} == \
            registry_parquet_states(root)

    def test_the_property_agrees_with_the_tiers_the_same_document_emits(
            self, tmp_path: Path) -> None:
        """Every `exported`/`summarised` entry names a tier that is really in this file.

        The property and the tier list are written by one pass over `TIERS`, so this can disagree
        only if the inventory is derived from something other than what `build_eaf` iterates — which
        is exactly how a second source of truth drifts. `person_tracks` is in the fixture's table
        set but not in its tier list (its secondary input is absent), so the assertion is one
        direction: bars imply an entry that names them.
        """
        dataset = make_dataset(tmp_path)
        _write_unread_pose_tables(dataset["dir"])
        eaf = eaf_of(dataset)
        declared = set(tiers_of(eaf))
        inventory = coverage_map(eaf)
        claimed = {tier for entry in inventory.values() for tier in entry.get("tiers", [])
                   if entry["state"] in (elan_core.COVERAGE_EXPORTED,
                                         elan_core.COVERAGE_SUMMARISED)}
        assert declared <= claimed, declared - claimed
        census = dict(eaf.properties)["pipeline-tiers"]
        for name, count in (part.split("=") for part in census.split()):
            assert int(count) >= 0 and name in declared

    def test_the_coverage_clause_of_the_semantics_names_all_four_states(
            self, dataset: dict[str, Path]) -> None:
        """A reader who finds the semantics property must not need the README for the vocabulary.

        The clause has to name the property it is describing, or it is a paragraph about a thing
        that is not in the file.
        """
        text = dict(eaf_of(dataset).properties)["pipeline-tier-semantics"]
        clause = text[text.index("Coverage:"):]
        assert elan_core.COVERAGE_PROPERTY in clause
        for state in elan_core.COVERAGE_STATES:
            assert state in clause, state
        assert "never merged" in clause

    # ------------------------------------------------------------------------ corpus truth

    def test_the_corpus_inventories_the_pose_tables_as_present_not_exported(self) -> None:
        """On the real tables, the three pose files are on disk and unread — not "absent".

        This is the operator's original question in executable form. KABC's `pose/face.parquet`
        holds 16,799 rows; a document that could only say "no pose tier" would let a reader
        conclude the clip has no person data, and the clip has plenty.
        """
        checked = 0
        for root in corpus_datasets():
            eaf, _report = build_eaf(root, corpus_video(root), log=lambda *a, **k: None)
            inventory = coverage_map(eaf)
            assert len(inventory) == 23, root.name
            for name in UNREAD_TABLES:
                on_disk = (root / ARTIFACT_LAYOUT[name]).is_file()
                expected = (elan_core.COVERAGE_PRESENT_NOT_EXPORTED if on_disk
                            else elan_core.COVERAGE_ABSENT)
                assert inventory[name]["state"] == expected, (root.name, name)
                checked += 1
        assert checked, "no corpus datasets on this disk; the loop proved nothing"

    def test_the_corpus_inventory_matches_a_state_map_rebuilt_from_the_filesystem(self) -> None:
        """The same recomputation, over the seven real corpora, with the real registry.

        Asserted per corpus, and the set of states actually met is asserted too rather than
        assumed. The `stories` table changed what this disk can show: the two datasets without
        speech skip the stage with a stated reason, so `absent` became a state the real corpora
        actually meet (measured 2026-10-01: person_demo and pipeline_silent), and a guard that
        still forbade it would forbid the truth. The absent set is therefore asserted exactly:
        a third dataset losing a producer is what this catches.
        """
        seen = set()
        absent_by_root = {}
        for root in corpus_datasets():
            eaf, _report = build_eaf(root, corpus_video(root), log=lambda *a, **k: None)
            inventory = coverage_map(eaf)
            recomputed = registry_parquet_states(root)
            assert {name: entry["state"] for name, entry in inventory.items()} == recomputed, \
                root.name
            seen.update(recomputed.values())
            absent_by_root[root.name] = {n for n, state in recomputed.items()
                                         if state == elan_core.COVERAGE_ABSENT}
        assert seen == set(elan_core.COVERAGE_STATES), seen
        assert {name: missing for name, missing in absent_by_root.items() if missing} == {
            "person_demo": {"stories"}, "pipeline_silent": {"stories"},
        }, "which datasets lack a table changed; re-measure the README's counts"


def _frame(timestamp: float, *, f0: Any = 120.0, intensity: Any = 60.0, voiced: bool = True,
           f1: Any = 500.0, f2: Any = 1500.0, f3: Any = 2500.0) -> dict[str, Any]:
    """One ``acoustic/frame_features.parquet`` row on Praat's 10 ms grid.

    Defaults are a fully-measured voiced frame, because the state a frame test has to leave
    deliberately is a null: a fixture that nulled things by default could not tell "the block
    stopped" from "the block was never allowed to start". `overrides`-style keyword arguments
    are how a test asks for the gap cases — f0 null (unvoiced), intensity null, one formant
    null — which are the whole behaviour of the three tiers.
    """
    return {"schema_version": "1.0", "video_id": "clip", "timestamp": timestamp,
            "f0_hz": f0, "intensity_db": intensity, "voiced": voiced,
            "f1_hz": f1, "f2_hz": f2, "f3_hz": f3}


def praat_clip(tmp_path: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    """A dataset holding only the Praat frame table (plus a video to link).

    Only the frame table is written, so every tier that needs another producer's file is
    *skipped* rather than empty: the frame tiers are the only ones expected to appear, and an
    assertion that they did is also the assertion that the others stayed out.
    """
    root, video = _clip(tmp_path)
    _write(ACOUSTIC_FRAMES_SCHEMA, root / ARTIFACT_LAYOUT["acoustic_frames"], rows)
    return root, video


def _frames(count: int, *, start: float = 0.0, step: float = 0.01, **values) -> list[dict]:
    """`count` frames on the 10 ms grid carrying the same measured values."""
    return [_frame(start + i * step, **values) for i in range(count)]


def _window_edge(index: int, grid: float = 0.0) -> int:
    """One window boundary in ms, computed here so the test does not trust the module's."""
    return seconds_to_ms(grid + index * elan_core.FRAME_WINDOW_SECONDS)


class TestPraatFrameTiers:
    """Three tiers of 100 ms window bars over Praat's grid, with the bin in the label.

    These tiers exist because ELAN cannot show a curve through pympi and the operator asked for
    the Praat data in the file. They block by *window*, not by equal label, and the tests below
    are mostly about what that buys:

    * a bar must never sit over a window that was mostly unmeasured — the refusal tests are the
      important ones, not the happy path;
    * a label must name a bin some measured frame inside the bar actually supported — so the
      low-median tests distinguish it from a mean of two frames;
    * a window boundary must be the same instant on all three tiers, which is the property the
      equal-label rule cannot offer.
    """

    def test_one_window_of_equal_measurements_is_one_bar_to_the_window_edge(
            self, tmp_path: Path) -> None:
        """Ten frames inside one 100 ms window: one bar, ending at the boundary."""
        root, video = praat_clip(tmp_path, _frames(10, f0=120.0))
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        bars = annotations(eaf, "f0_blocks")
        assert len(bars) == 1, bars
        assert (bars[0][0], bars[0][1]) == (0, _window_edge(1)), bars
        assert bars[0][2] == f"f0 med {elan_core.pitch_label(120.0)}", bars[0][2]

    def test_a_second_window_with_the_same_label_shares_the_bar(self, tmp_path: Path) -> None:
        """Merging happens on the printed label, so two identical windows become one bar."""
        root, video = praat_clip(tmp_path, _frames(20, f0=120.0))
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        bars = annotations(eaf, "f0_blocks")
        assert [(s, e) for s, e, _t in bars] == [(0, _window_edge(2))], bars

    def test_a_window_whose_label_differs_starts_a_new_bar(self, tmp_path: Path) -> None:
        """The label decides where a bar ends, and the boundary is the window's, not a frame's."""
        root, video = praat_clip(tmp_path, _frames(10, f0=120.0) + _frames(10, start=0.1, f0=400.0))
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        bars = annotations(eaf, "f0_blocks")
        assert [(s, e) for s, e, _t in bars] == [(0, _window_edge(1)), (_window_edge(1),
                                                                         _window_edge(2))], bars
        assert bars[0][2] != bars[1][2], (bars[0][2], bars[1][2])

    def test_the_bar_ends_at_the_window_boundary_and_adds_no_grid_step(
            self, tmp_path: Path) -> None:
        """The frame tiers are the documented exception to the median-step extension.

        `voiced_blocks` widens its last frame by one grid step because a sampled frame stands
        for a span; a frame window *is* the span, so widening it would cover 10 ms of a window
        whose label came from other frames. Asserted against `voiced_blocks` on the same rows
        so the difference is visible rather than assumed.
        """
        root, video = praat_clip(tmp_path, _frames(3, f0=120.0))   # 30 ms of grid
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        voiced = annotations(eaf, "voiced_blocks")
        f0 = annotations(eaf, "f0_blocks")
        assert voiced[0][1] == seconds_to_ms(0.03), voiced          # 20 ms + one 10 ms step
        assert f0[0][1] == _window_edge(1), f0                     # the boundary, not 30 ms
        assert f0[0][1] > voiced[0][1], (voiced, f0)

    @pytest.mark.parametrize("tier,prefix,values,bin_of", [
        ("f0_blocks", "f0 med ", {"f0": 120.0}, lambda v: elan_core.pitch_label(v)),
        ("intensity_blocks", "int med ", {"intensity": 61.0},
         lambda v: elan_core.intensity_label(v)),
        ("formant_blocks", "med ", {"f1": 500.0, "f2": 1500.0, "f3": 2500.0},
         lambda v: elan_core.formant_label(500.0, 1500.0, 2500.0)),
    ])
    def test_every_tier_prints_its_bin_prefixed_as_a_median(
            self, tmp_path: Path, tier: str, prefix: str, values: dict[str, Any],
            bin_of) -> None:
        """The label says which bin, and says it is a median, so it cannot be read as a value."""
        root, video = praat_clip(tmp_path, _frames(4, **values))
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        text = annotations(eaf, tier)[0][2]
        assert text == prefix + bin_of(list(values.values())[0]), text

    def test_the_label_is_the_low_median_a_frame_had_not_the_mean_of_two(
            self, tmp_path: Path) -> None:
        """100 Hz and 300 Hz in one window: the label is G2 (the lower frame), never G3.

        The arithmetic mean is 200 Hz, which lands in a third bin and is a pitch no frame in
        that window had. :func:`lower_median` is what keeps the label a measurement, and this
        is the test that would fail if someone swapped it for :func:`statistics.median`.
        """
        root, video = praat_clip(tmp_path, [_frame(0.0, f0=100.0), _frame(0.01, f0=300.0)])
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        text = annotations(eaf, "f0_blocks")[0][2]
        assert text == f"f0 med {elan_core.pitch_label(100.0)}", text
        assert elan_core.pitch_label(200.0) != elan_core.pitch_label(100.0), (
            "pick f0 values in different bins for this to test anything")

    def test_a_window_that_is_mostly_unmeasured_gets_no_bar_and_is_counted(
            self, tmp_path: Path) -> None:
        """The refusal that matters: 4 pitched frames out of 10 is not enough to claim a pitch.

        Asserted twice over — no bar covers the window, and the run log says how many windows
        were refused. The count is the part a reader of a sparse tier cannot reconstruct from
        the document, so a build that quietly dropped it would fail here.
        """
        frames = _frames(4, f0=120.0) + _frames(6, start=0.04, f0=None)
        root, video = praat_clip(tmp_path, frames)
        logged: list[str] = []
        eaf, _report = build_eaf(root, video, log=lambda msg, *a, **k: logged.append(msg))
        bars = annotations(eaf, "f0_blocks")
        assert bars == [], bars
        thin = [line for line in logged if "unlabelled" in line]
        assert len(thin) == 1, logged
        assert "f0_blocks" in thin[0] and "100 ms" in thin[0], thin[0]

    def test_a_refused_window_breaks_a_bar_rather_than_being_spanned_by_one(
            self, tmp_path: Path) -> None:
        """Good window, thin window, good window: two bars and a visible hole, never one bar.

        A single bar across all three would print the outer windows' label over the middle one
        and be the exact lie the refusal exists to prevent.
        """
        frames = (_frames(10, f0=120.0)
                  + _frames(4, start=0.1, f0=120.0) + _frames(6, start=0.14, f0=None)
                  + _frames(10, start=0.2, f0=120.0))
        root, video = praat_clip(tmp_path, frames)
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        bars = annotations(eaf, "f0_blocks")
        assert [(s, e) for s, e, _t in bars] == [
            (0, _window_edge(1)), (_window_edge(2), _window_edge(3))], bars
        assert all(t.startswith("f0 med ") for _s, _e, t in bars)

    def test_one_null_formant_makes_only_the_formant_windows_thin(self, tmp_path: Path) -> None:
        """A null F2 costs the formant tier the windows it sits in, not the pitch or loudness ones.

        Two nulls out of ten frames is 20% and stays above the refusal line, so this asserts the
        *shape*: the formant tier loses the bar the two nulls fall in, keeps the rest, and the
        other two tiers are untouched because they never read F2.
        """
        frames = _frames(4) + _frames(2, start=0.04, f2=None) + _frames(4, start=0.06)
        root, video = praat_clip(tmp_path, frames)
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        formant = annotations(eaf, "formant_blocks")
        assert len(formant) == 1, formant
        assert len(annotations(eaf, "f0_blocks")) == 1, "f0 blocked on a null F2 it never reads"
        assert len(annotations(eaf, "intensity_blocks")) == 1

    def test_a_majority_of_null_formants_refuses_the_formant_window_only(
            self, tmp_path: Path) -> None:
        """Six null F2 frames in ten: the formant tier has no bar, the other two still do."""
        frames = (_frames(4, f1=500.0, f2=1500.0, f3=2500.0)
                  + _frames(6, start=0.04, f1=500.0, f2=None, f3=2500.0))
        root, video = praat_clip(tmp_path, frames)
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        assert annotations(eaf, "formant_blocks") == []
        assert len(annotations(eaf, "f0_blocks")) == 1
        assert len(annotations(eaf, "intensity_blocks")) == 1

    def test_intensity_blocks_do_not_filter_on_voicing(self, tmp_path: Path) -> None:
        """An unvoiced frame with a measured level is still loudness data."""
        root, video = praat_clip(tmp_path, [
            _frame(0.0, f0=120.0, intensity=60.0), _frame(0.01, f0=None, intensity=60.0)])
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        assert len(annotations(eaf, "intensity_blocks")) == 1
        assert len(annotations(eaf, "f0_blocks")) == 1   # the pitched frame alone is enough

    def test_a_non_finite_value_is_unmeasured_like_any_null(self, tmp_path: Path) -> None:
        """NaN is not a measurement: it cannot fill a window, and it cannot be the median.

        The frame schema allows a float and the worker's docstring says Praat answers None/NaN
        for unmeasured frames, so six NaN frames out of ten must refuse the window rather than
        let one NaN set the label.
        """
        frames = _frames(4, f0=120.0) + _frames(6, start=0.04, f0=float("nan"))
        root, video = praat_clip(tmp_path, frames)
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        assert annotations(eaf, "f0_blocks") == []

    def test_a_frame_table_with_nothing_measured_writes_empty_tiers_not_errors(
            self, tmp_path: Path) -> None:
        """All-null frames: the tiers are present (the table was read) and carry no bars."""
        root, video = praat_clip(tmp_path, [
            _frame(0.0, f0=None, intensity=None, voiced=False, f1=None, f2=None, f3=None)])
        eaf, report = build_eaf(root, video, log=lambda *a, **k: None)
        counts = tier_counts(eaf)
        assert counts.get("f0_blocks", 0) == 0
        assert counts.get("intensity_blocks", 0) == 0
        assert counts.get("formant_blocks", 0) == 0
        assert not any(counts.get(tier) for tier in PRAAT_FRAME_TIERS)

    def test_an_absent_frame_table_skips_all_four_acoustic_tiers(self, tmp_path: Path) -> None:
        """No file, four skips — and each names the missing file, not a fabricated empty tier."""
        root, video = _clip(tmp_path)
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        counts = tier_counts(eaf)
        assert not [tier for tier in PRAAT_FRAME_TIERS if tier in counts], counts

    def test_the_three_frame_tiers_share_their_window_boundaries(self, tmp_path: Path) -> None:
        """The property the window rule exists to give: three tiers, one set of boundaries.

        Equal-label runs would put each tier's edges wherever its own column happened to
        change, and a reader comparing pitch against loudness at a moment would be comparing
        different intervals. Here every bar edge of every frame tier is a window boundary.
        """
        frames = (_frames(10, f0=120.0, intensity=60.0, f1=500.0, f2=1500.0, f3=2500.0)
                  + _frames(10, start=0.1, f0=400.0, intensity=70.0,
                            f1=700.0, f2=1700.0, f3=2700.0))
        root, video = praat_clip(tmp_path, frames)
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        edges: dict[str, set[int]] = {}
        for tier in ("f0_blocks", "intensity_blocks", "formant_blocks"):
            bars = annotations(eaf, tier)
            assert len(bars) == 2, (tier, bars)
            edges[tier] = {value for s, e, _t in bars for value in (s, e)}
        assert edges["f0_blocks"] == edges["intensity_blocks"] == edges["formant_blocks"], edges
        assert edges["f0_blocks"] <= {_window_edge(i) for i in range(3)}, edges


class TestPraatQuantisation:
    """The binning functions and the window rule, asserted against values computed in the test.

    Deliberately not asserted against the module's own helpers — that would be a tautology. The
    expected values below are hand-computed from the definitions: A4 = 440 Hz, 12 semitones per
    octave, floor binning, a 2 dB floor step, formant bands of 300/600/1000 Hz.
    """

    @pytest.mark.parametrize("f0,expected", [
        (440.0, "A4"),        # the anchor itself
        (439.9, "G#4"),       # one cent under the anchor is the bin below
        (466.1638, "A#4"),    # just over the true A#4 boundary 440·2^(1/12) = 466.16376...
        (466.16, "A4"),       # four thousandths of a Hz under it — still A4
        (110.0, "A2"),        # two octaves down
        (1000.0, "B5"),       # MIDI 83.21 floored to 83
    ])
    def test_pitch_bins_are_floor_binned_from_a4(self, f0: float, expected: str) -> None:
        assert elan_core.pitch_label(f0) == expected, f0

    @pytest.mark.parametrize("db,expected", [
        (60.0, "60 to 62 dB"),     # exact bin bottom is inclusive
        (61.9, "60 to 62 dB"),
        (62.0, "62 to 64 dB"),     # the edge belongs to the bin above
        (-3.5, "-4 to -2 dB"),     # floor works below zero
        (-300.0, "-300 to -298 dB"),  # Praat's own floor, and the separator cannot be read as a sign
    ])
    def test_intensity_bins_are_a_two_db_floor_with_a_sign_safe_separator(
            self, db: float, expected: str) -> None:
        """'x to y dB', never 'x-y dB': -300 to -298 printed with a hyphen is unreadable."""
        assert elan_core.intensity_label(db) == expected, db

    def test_formant_label_names_all_three_bands_of_the_measured_triplet(self) -> None:
        # 500//300=1 -> 300; 1500//600=2 -> 1200; 2500//1000=2 -> 2000.
        assert elan_core.formant_label(500.0, 1500.0, 2500.0) == \
            "F1~300Hz F2~1200Hz F3~2000Hz"

    def test_the_bands_are_the_measured_widths(self) -> None:
        """The widths are a measured choice (legibility on the real clips), pinned by name."""
        assert FORMANT_BANDS_HZ == (300, 600, 1000)
        assert INTENSITY_DB_STEP == 2
        assert elan_core.FRAME_WINDOW_SECONDS == 0.1
        assert elan_core.FRAME_MIN_FILL == 0.5

    def test_the_low_median_is_always_one_of_the_values(self) -> None:
        """The definition of the guarantee the label rests on, over odd and even counts."""
        for values in ([100.0], [100.0, 300.0], [300.0, 100.0, 200.0],
                       [5.0, 5.0, 5.0, 5.0], [1.0, 2.0, 3.0, 4.0]):
            picked = elan_core.lower_median(values)
            assert picked in values, (values, picked)
        assert elan_core.lower_median([1.0, 2.0, 3.0, 4.0]) == 2.0

    @pytest.mark.parametrize("measured,expect_bar", [(4, False), (5, True)])
    def test_the_window_rule_refuses_below_half_and_labels_at_exactly_half(
            self, tmp_path: Path, measured: int, expect_bar: bool) -> None:
        """4 of 10 is refused, 5 of 10 is labelled: the boundary of the promise.

        The rule is `fill < FRAME_MIN_FILL` refuses, so exactly half is kept — "half these frames
        were measured" is still a claim about the frames under the bar. Pinned at the boundary
        because a `<=` here would silently discard every half-measured window on a real clip
        (measured: 3-8% of the corpus's pitched frames), and loosening the comparison the other
        way would label windows that are mostly holes.
        """
        frames = (_frames(measured, f0=120.0)
                  + _frames(10 - measured, start=measured * 0.01, f0=None))
        root, video = praat_clip(tmp_path, frames)
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        bars = annotations(eaf, "f0_blocks")
        assert bool(bars) is expect_bar, (measured, bars)

    def test_frame_window_labels_counts_refusals_and_covers_the_rest(self) -> None:
        """The unit-level view of the same rule, without a document in the way."""
        frames = _frames(10, f0=120.0) + _frames(10, start=0.1, f0=None)
        slots, notes = elan_core.frame_window_labels(frames, ("f0_hz",))
        assert notes == {"thin_windows": 1, "windows": 2}, notes
        assert slots[0] is not None and slots[1] is None, slots
        first, last, medians, covered = slots[0]
        assert (first, last) == (0, 1) and covered == 10 and medians == (120.0,), slots[0]

    def test_a_window_with_no_frames_at_all_is_not_counted_as_refused(self) -> None:
        """Two states, not one number: 'Praat measured too little here' is not 'no frame here'.

        The table's own grid ends where the audio ends, so the last window is routinely partly
        empty; counting an empty window as a refusal would put a number on the log line that
        means two different defects, which is the collapse this repository refuses elsewhere.
        """
        frames = _frames(10, f0=120.0) + _frames(10, start=0.2, f0=120.0)  # window 1 empty
        slots, notes = elan_core.frame_window_labels(frames, ("f0_hz",))
        assert notes == {"thin_windows": 0, "windows": 3}, notes
        assert [slot is None for slot in slots] == [False, True, False], slots


class TestWithdrawnTiersStayWithdrawn:
    """The five tiers the operator removed, and the tables that keep their evidence.

    The defect this guards is a quiet one: a table whose tier was withdrawn must not fall back
    to :data:`COVERAGE_REASON_UNKNOWN` — that reads as "nobody thought about this table" when
    the truth is a decision taken after opening the file in ELAN. And the tier names must not
    come back as a rename: `spacy_*` reappearing under a new label would put a borrowed
    timeline back in the document.
    """

    WITHDRAWN_TIERS = ("spacy_source_tokens", "spacy_source_sentences", "spacy_english_tokens",
                       "spacy_english_sentences", "acoustic_segments")
    WITHDRAWN_TABLES = WITHDRAWN_TIERS + ("acoustic_segments",)

    def test_no_tier_is_named_after_a_withdrawn_one(self) -> None:
        names = {spec.tier for spec in TIERS}
        assert not names & set(self.WITHDRAWN_TIERS), names & set(self.WITHDRAWN_TIERS)

    @pytest.mark.parametrize("artifact", sorted(set(WITHDRAWN_TABLES)))
    def test_a_withdrawn_table_names_its_reason_when_it_is_on_disk(
            self, tmp_path: Path, artifact: str) -> None:
        """On disk and unread → a reason that says *which decision*, never the fallback.

        The dataset is written without the artifact's file for the absent case; here the file
        is dropped in by hand so the state is the one that carries a reason.
        """
        root, video = _clip(tmp_path)
        target = root / ARTIFACT_LAYOUT[artifact]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"not really parquet, and the inventory only stats it")
        inventory = elan_core.coverage_inventory(root)
        entry = inventory[artifact]
        assert entry["state"] == elan_core.COVERAGE_PRESENT_NOT_EXPORTED, entry
        reason = entry.get("reason", "")
        assert reason and reason != elan_core.COVERAGE_REASON_UNKNOWN, entry
        # Each reason names the replacement or the withdrawn tier, so it cannot be reused
        # boilerplate: the sentence is falsifiable by the tier list it cites.
        assert "withdrawn" in reason or "not represented" in reason, reason


class TestFrameTableCoverageNamesEveryTier:
    """One table, four tiers: the inventory names all four or a reader cannot find the bars."""

    def test_the_frame_table_lists_all_four_acoustic_tiers(self, tmp_path: Path) -> None:
        root, video = praat_clip(tmp_path, [_frame(0.0, f0=120.0)])
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        entry = coverage_map(eaf)["acoustic_frames"]
        assert entry["state"] == elan_core.COVERAGE_EXPORTED, entry
        assert entry["tiers"] == list(PRAAT_FRAME_TIERS), entry

    def test_a_single_tier_table_still_reports_a_one_element_list(self, tmp_path: Path) -> None:
        """`tiers` is a list for every exported artifact, not only for the four-tier one.

        Built from a dataset that writes the frame table, because `make_dataset` does not: an
        artifact only exists in the inventory as `exported` when a tier read it.
        """
        root, video = praat_clip(tmp_path, [_frame(0.0, f0=120.0)])
        _write(WORDS_SCHEMA, root / "speech" / "words.parquet", [_word("hello", 0.0, 0.4)])
        eaf, _report = build_eaf(root, video, log=lambda *a, **k: None)
        entry = coverage_map(eaf)["speech_words"]
        assert entry["tiers"] == ["words"], entry


class TestDropCountsReachTheRecord:
    """A tier with fewer bars than rows says why in the record, not only in the run log.

    The two drop rules already existed and already logged. What did not exist is a way to answer
    "why does this tier show 20 bars for the 24 rows in the table" after the log has rotated: the
    numbers were in one line of terminal output and nowhere else. Nothing here changes a drop rule
    — same two exceptions, same two counters, same two log lines — it returns them.
    """

    def test_a_built_tier_reports_its_two_drop_counters(self, tmp_path: Path) -> None:
        """The counters come back with the document, and the two states stay two keys."""
        root, video = _clip(tmp_path)
        _write(WORDS_SCHEMA, root / "speech" / "words.parquet", [
            _word("good", 0.0, 0.4),
            _word("no-end", 2.0, None),
            _word("nan", float("nan"), 3.0),
        ])
        eaf, report = build_eaf(root, video, log=lambda *a, **k: None)
        assert elan_core.drop_counts(report) == {"words": {"missing_time": 1, "non_finite": 1}}
        assert report["dropped"]["words"]["rows"] == 3
        assert tier_counts(eaf) == {"words": 1}

    def test_a_clean_tier_contributes_nothing_to_the_record(self, dataset: dict[str, Path]
                                                           ) -> None:
        """`{}` means no row was refused; it does not mean the export could not tell.

        Suppressing zero rows is what keeps the record readable, and it is why the record's
        *absence of a key* has to be a real answer rather than silence: every tier built cleanly
        here, so the map is empty and the reader can trust that emptiness.
        """
        _eaf, report = build_eaf(dataset["dir"], dataset["video"], log=lambda *a, **k: None)
        assert elan_core.drop_counts(report) == {}
        assert set(report["dropped"]) == set(tier_counts(_eaf))
        assert all(counts["missing_time"] == 0 and counts["non_finite"] == 0
                   for counts in report["dropped"].values())

    def test_the_two_drop_states_are_never_summed_into_one_number(self, tmp_path: Path) -> None:
        """The run log keeps them apart and so does the record.

        B1's rule one level up: "the producer wrote no time" and "the producer wrote a NaN" are
        different upstream defects, and one total would send somebody to fix the wrong column.
        """
        root, video = _clip(tmp_path)
        _write(WORDS_SCHEMA, root / "speech" / "words.parquet", [
            _word("a", None, None), _word("b", None, None), _word("c", float("inf"), 1.0)])
        lines: list[str] = []
        _eaf, report = build_eaf(root, video, log=lambda msg, *a, **k: lines.append(str(msg)))
        assert elan_core.drop_counts(report) == {"words": {"missing_time": 2, "non_finite": 1}}
        assert not any("dropped 3" in line for line in lines), lines
        assert [line for line in lines if "2 of 3" in line and "missing timestamp" in line]
        assert [line for line in lines if "1 of 3" in line and "non-finite" in line]

    def test_the_log_lines_are_unchanged_by_the_record(self, tmp_path: Path) -> None:
        """Adding a machine-readable channel must not edit the human-readable one.

        Asserted as the exact substrings the log has always printed, because a refactor that
        folded the two lines together would satisfy every other test in this class.
        """
        root, video = _clip(tmp_path)
        _write(WORDS_SCHEMA, root / "speech" / "words.parquet", [
            _word("a", 1.0, None), _word("b", float("nan"), 1.0)])
        lines: list[str] = []
        build_eaf(root, video, log=lambda msg, *a, **k: lines.append(str(msg)))[0]
        assert any("elan: tier words dropped 1 of 2 annotation(s) with a missing timestamp"
                   in line for line in lines), lines
        assert any("elan: tier words dropped 1 of 2 annotation(s) with a non-finite timestamp"
                   in line for line in lines), lines

    def test_a_dropped_row_is_counted_and_the_projection_is_untouched(self, tmp_path: Path
                                                                     ) -> None:
        """Two drop rules and a re-cut in one tier, each reported by its own mechanism.

        The projection counts rows that were placed and shared an instant; the record counts rows
        that were refused. Feeding both to one number would report a producer defect as a layout
        choice, so this pins the three numbers apart: 3 rows in, 1 refused, 2 placed, 3 bars out.
        """
        from multimodal_pipeline.schemas import SEGMENTS_SCHEMA

        root, video = _clip(tmp_path)
        _write(SEGMENTS_SCHEMA, root / ARTIFACT_LAYOUT["speech_segments"], [
            _segment_row("seg-0", 0.0, 2.0, "first"),
            _segment_row("seg-1", 1.0, 3.0, "second"),
            _segment_row("seg-2", None, 4.0, "no time"),
        ])
        eaf, report = build_eaf(root, video, log=lambda *a, **k: None)
        assert elan_core.drop_counts(report) == {"segments_src": {"missing_time": 1,
                                                                 "non_finite": 0}}
        assert projection_of(eaf)["segments_src"]["logical_row_count"] == 2
        assert projection_of(eaf)["segments_src"]["final_annotation_count"] == 3
        assert tier_counts(eaf)["segments_src"] == 3
        assert report["dropped"]["segments_src"]["rows"] == 3

    def test_the_corpus_carries_no_dropped_rows_and_the_counters_are_still_reported(
            self) -> None:
        """Measured: zero on every dataset on this disk, and the record says so per tier.

        An honest negative. The non-zero path is proven by the tests above against fixtures the
        corpus does not contain; what this test establishes is that the seven real clips pay
        nothing for the guard, and that a clean run still reports a counter for every built tier
        rather than reporting nothing at all.
        """
        checked = 0
        for root in corpus_datasets():
            _eaf, report = build_eaf(root, corpus_video(root), log=lambda *a, **k: None)
            assert elan_core.drop_counts(report) == {}, root.name
            assert report["dropped"], root.name
            checked += 1
        assert checked, "no corpus datasets on this disk; the loop proved nothing"
