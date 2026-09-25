"""The audio/visual fusion: five agreement states, computed by hand.

Two kinds of test live here, and both matter for different reasons.

`TestAgreementStates` drives the pure core (`fusion.fuse_turn_table`) with synthetic turns
and frames built from the **real schema column sets** — every row is projected through
``SPEAKER_TURNS_SCHEMA`` / ``ACTIVE_SPEAKER_FRAMES_SCHEMA``, so a renamed or dropped column
upstream fails here instead of being quietly ignored by a fixture that had already drifted
from the table the pipeline writes. Every expected number is hand-computed from the frame
list in the test itself, because "the ratio is about a half" cannot catch an off-by-one in
the window arithmetic.

`TestStageWiring` drives the stage against real Parquet on disk, which is where the
guarantees that are *not* arithmetic live: two engines write two files, an absent table
skips with a reason rather than emitting an empty table, and the fingerprint notices a
re-diarization or a re-run ASD.

The five states are the reason the stage exists, so each one is tested on its own, and the
pair that is easiest to collapse — `no_face_visible` (measured: nobody on screen) and
`no_frames_measured` (nothing measured at all) — is tested against a shared helper that
differs *only* in whether the ASD table covers the window.
"""

from __future__ import annotations

from typing import Any, Sequence

import pyarrow as pa
import pytest
from multimodal_pipeline.exceptions import ValidationError
from multimodal_pipeline.fusion import (
    AGREEMENT_STATES,
    TURN_TABLES,
    fuse_turn_table,
)
from multimodal_pipeline.schemas import (
    ACTIVE_SPEAKER_FRAMES_SCHEMA,
    SPEAKER_FUSION_SCHEMA,
    SPEAKER_TURNS_NEMOTRON_SCHEMA,
    SPEAKER_TURNS_SCHEMA,
    read_table,
    write_table,
)
from multimodal_pipeline.stages.speaker_fusion import SpeakerFusionStage

FPS = 25
STEP = 1.0 / FPS


# ------------------------------------------------------------------ row builders


def build_row(schema, **values: Any) -> dict[str, Any]:
    """A row holding exactly the schema's columns, built *through* that schema.

    The projection is the point: `row["typo_column"] = 1` used to look fine in a fixture
    and silently stop matching what the stage reads. Unknown keys raise, and the row is
    round-tripped through the real Arrow schema so its types are the types a reader sees.
    """
    known = {field.name for field in schema}
    unknown = set(values) - known
    if unknown:
        raise KeyError(f"{sorted(unknown)} not in {sorted(known)}: the schema moved, "
                       f"update the builder rather than let the fixture drift")
    table = pa.Table.from_pylist([values], schema=schema)
    return table.to_pylist()[0]


def turn_row(*, start: float, end: float, speaker: str = "SPEAKER_00",
             turn_id: str = "turn000001", overlap_s: float | None = None,
             nemotron: bool = False, video_id: str = "conversation_001") -> dict[str, Any]:
    """One diarization turn, in the shape its own engine's table has."""
    schema = SPEAKER_TURNS_NEMOTRON_SCHEMA if nemotron else SPEAKER_TURNS_SCHEMA
    values: dict[str, Any] = {
        "schema_version": "1.0",
        "video_id": video_id,
        "turn_id": turn_id,
        "speaker_id": speaker,
        "start_time": start,
        "end_time": end,
        "duration": round(end - start, 6),
        "diarization_type": "overlapping" if nemotron else "exclusive",
    }
    if nemotron:
        values["overlap_s"] = overlap_s
    return build_row(schema, **values)


def frame_row(index: int, *, track_id: int | None = None, active: bool = False,
              score: float | None = None, video_id: str = "conversation_001",
              **overrides: Any) -> dict[str, Any]:
    """One dense 25 FPS ASD row.

    `track_id=None` is the no_face case, and it is the only state where that is honest —
    the frames stage guarantees a timestamp on every row, including the empty ones.
    """
    values: dict[str, Any] = {
        "schema_version": "1.0",
        "video_id": video_id,
        "frame_number": index,
        "timestamp": round(index * STEP, 6),
        "source_timestamp": round(index * STEP, 6),
        "scene_id": 1,
        "track_id": track_id,
        "face_status": "tracked" if track_id is not None else "no_face",
        "frame_reason": "scored" if track_id is not None else "no_face",
        "x1": 10.0 if track_id is not None else None,
        "y1": 20.0 if track_id is not None else None,
        "x2": 50.0 if track_id is not None else None,
        "y2": 70.0 if track_id is not None else None,
        "talknet_score_raw": score,
        "talknet_score": score,
        "score_imputed": False,
        "is_active_speaker": active,
    }
    values.update(overrides)
    return build_row(ACTIVE_SPEAKER_FRAMES_SCHEMA, **values)


def run_core(turns: Sequence[dict[str, Any]], frames: Sequence[dict[str, Any]],
             *, engine: str = "pyannote", **kwargs: Any) -> list[dict[str, Any]]:
    return fuse_turn_table(video_id="conversation_001", engine=engine, turns=turns,
                           frames=frames, **kwargs)


# ------------------------------------------------------- five agreement states


class TestFaceMatched:
    def test_a_track_clearing_both_thresholds_wins_the_turn(self):
        # Frames 0..7 fall in 0.00-0.28 (inclusive both ends): 8 ASD rows.
        # Track 0 is the face on frames 0..5 (6 rows) and is flagged active on 3 of them.
        # ratio = 3/6 = 0.50 >= min_active_ratio 0.5, and 3 >= min_face_frames 2.
        frames = ([frame_row(i, track_id=0, active=i < 3, score=1.0 + i * 0.1)
                   for i in range(6)]
                  + [frame_row(6), frame_row(7)])
        rows = run_core([turn_row(start=0.0, end=0.28)], frames)
        assert len(rows) == 1
        row = rows[0]
        assert row["agreement"] == "face_matched"
        assert row["face_track_id"] == 0
        assert row["face_active_frames"] == 3
        assert row["face_frames_in_turn"] == 6
        assert row["frames_in_turn"] == 8  # the two no_face rows are counted
        # mean of 1.0..1.5 -> 1.25, over the frames where *this* track was on screen:
        # pooling the window's no_face rows would understate a face that walks off
        # mid-turn, and averaging only the active frames would report the confidence of
        # the frames already selected as active.
        assert row["face_mean_score"] == pytest.approx(1.25, abs=1e-6)
        assert row["face_score_max"] == pytest.approx(1.5, abs=1e-6)
        assert "track 0 active on 3/6 frames in turn" in row["agreement_detail"]

    def test_detail_names_the_thresholds_it_cleared(self):
        # The detail is the column a human reads first; "looks like a match" is not a
        # reason. It has to carry the numbers and the knobs.
        frames = [frame_row(i, track_id=0, active=True, score=2.0) for i in range(4)]
        rows = run_core([turn_row(start=0.0, end=0.12)], frames)
        detail = rows[0]["agreement_detail"]
        assert "4/4 frames in turn" in detail
        assert "ratio 1.00" in detail
        assert "min_active_ratio 0.5" in detail
        assert "min_face_frames 2" in detail
        assert "mean score 2.00" in detail
        # A window with a face on every frame carries no qualification to carry.
        assert "no face located" not in detail

    def test_a_matched_turn_with_a_faceless_stretch_says_so(self):
        """The ratio describes the frames where a face was visible, not the whole turn.

        Measured on the real La 1 clip: its second pyannote turn matches track 4 on 80/80
        of its tracked frames and still contains 44 frames with nobody on screen — a
        voice-over over a different shot. A detail that reported only "80/80, ratio 1.00"
        would read as though the whole turn was a face on camera, which is precisely the
        flattening this table exists to prevent.
        """
        frames = ([frame_row(i, track_id=4, active=True, score=2.0) for i in range(10)]
                  + [frame_row(i) for i in range(10, 15)])
        row = run_core([turn_row(start=0.0, end=0.56)], frames)[0]
        assert row["agreement"] == "face_matched"
        assert row["face_active_frames"] == 10
        assert row["face_frames_in_turn"] == 10
        assert row["frames_in_turn"] == 15
        assert "; no face located on 5/15 measured frames of the window" in \
            row["agreement_detail"]


class TestFacePartial:
    def test_enough_frames_but_the_ratio_is_short(self):
        # Track 0 present on 7 frames, active on 3 -> 3/7 = 0.4286 < 0.5, while
        # 3 >= min_face_frames. Partial on the ratio alone.
        frames = [frame_row(i, track_id=0, active=i < 3, score=0.5) for i in range(7)]
        rows = run_core([turn_row(start=0.0, end=0.24)], frames)
        row = rows[0]
        assert row["agreement"] == "face_partial"
        assert row["face_track_id"] == 0
        assert row["face_active_frames"] == 3
        assert row["face_frames_in_turn"] == 7
        assert row["frames_in_turn"] == 7
        assert "ratio 0.43 is below min_active_ratio 0.5" in row["agreement_detail"]

    def test_a_single_active_frame_is_a_sighting_not_a_speaker(self):
        """min_face_frames is not redundant with the ratio.

        One active frame is ratio 1.00 — the most confident number a ratio can produce —
        and it is still one frame. Drop min_face_frames from the matched condition and the
        same evidence is labelled face_matched.
        """
        frames = [frame_row(0, track_id=0, active=True, score=3.0),
                  frame_row(1, track_id=1, active=False, score=3.0)]
        only_one = run_core([turn_row(start=0.0, end=0.0)], frames)
        assert only_one[0]["frames_in_turn"] == 1
        assert only_one[0]["face_active_frames"] == 1
        assert only_one[0]["agreement"] == "face_partial"
        assert "1 active frame(s) is below min_face_frames 2" in only_one[0]["agreement_detail"]
        # Same numbers, threshold lowered to 1 -> the identical evidence now clears it.
        relaxed = run_core([turn_row(start=0.0, end=0.0)], frames, min_face_frames=1)
        assert relaxed[0]["agreement"] == "face_matched"

    def test_both_shortfalls_are_named_together(self):
        frames = ([frame_row(0, track_id=0, active=True, score=0.2)]
                  + [frame_row(i, track_id=0, active=False, score=0.2) for i in range(1, 9)])
        rows = run_core([turn_row(start=0.0, end=0.32)], frames)
        detail = rows[0]["agreement_detail"]
        assert "below min_face_frames 2" in detail
        assert "below min_active_ratio" in detail


class TestNoFaceVisible:
    def test_a_voice_with_nothing_visible_is_measured_absence(self):
        # Off-screen narrator / audio bed: the ASD table covers the window and reports
        # no face in any of it. This is a *measurement*, not a gap.
        frames = [frame_row(i) for i in range(5)]
        rows = run_core([turn_row(start=0.0, end=0.16)], frames)
        row = rows[0]
        assert row["agreement"] == "no_face_visible"
        assert row["face_track_id"] is None
        assert row["face_active_frames"] == 0
        assert row["face_frames_in_turn"] == 0
        assert row["frames_in_turn"] == 5
        assert row["face_mean_score"] is None
        assert row["face_score_max"] is None
        assert "no face was located in any of the 5 ASD frames" in row["agreement_detail"]


class TestFaceNeverActive:
    def test_faces_visible_but_nobody_ever_flagged(self):
        # Silent mouth or a cutaway face: the opposite conclusion from no_face_visible,
        # and it must not share that state's row.
        frames = [frame_row(i, track_id=0, active=False, score=-1.5) for i in range(4)]
        rows = run_core([turn_row(start=0.0, end=0.12)], frames)
        row = rows[0]
        assert row["agreement"] == "face_never_active"
        assert row["face_track_id"] is None
        assert row["face_active_frames"] == 0
        assert row["face_frames_in_turn"] == 4
        assert row["frames_in_turn"] == 4
        assert "faces visible on 4/4 measured frames across 1 track(s)" in row["agreement_detail"]
        assert "no track was ever flagged an active speaker" in row["agreement_detail"]


class TestNoFramesMeasured:
    def test_a_turn_the_frames_table_never_covers_is_not_a_no_face_turn(self):
        frames = [frame_row(i) for i in range(3)]  # 0.00 .. 0.08
        rows = run_core([turn_row(start=20.0, end=21.0)], frames)
        row = rows[0]
        assert row["agreement"] == "no_frames_measured"
        assert row["frames_in_turn"] == 0
        assert row["face_frames_in_turn"] == 0
        assert row["face_active_frames"] == 0
        assert "absence of measurement" in row["agreement_detail"]

    def test_distinct_from_no_face_visible_by_one_input(self):
        # The two states differ only in whether the table reaches into the window, so the
        # test that protects them from collapsing differs only in that input too.
        turn = turn_row(start=0.2, end=0.36)  # frames 5..9, if the table reaches them
        before_the_window = [frame_row(i) for i in range(5)]        # ends at 0.16
        across_the_window = [frame_row(i) for i in range(5, 10)]    # 0.20 .. 0.36
        unmeasured = run_core([turn], before_the_window)
        measured = run_core([turn], across_the_window)
        assert unmeasured[0]["agreement"] == "no_frames_measured"
        assert unmeasured[0]["frames_in_turn"] == 0
        assert measured[0]["agreement"] == "no_face_visible"
        assert measured[0]["frames_in_turn"] == 5
        assert measured[0]["face_frames_in_turn"] == 0
        # The two must never share a value: one says "nobody was on screen", the other
        # "nobody looked", and a reader cannot recover the difference afterwards.
        assert unmeasured[0]["agreement"] != measured[0]["agreement"]

    def test_window_covers_every_dense_row_in_range_including_no_face_rows(self):
        # Frames at 0.04 steps; window 0.20-0.36 inclusive -> indices 5..9 = 5 rows,
        # of which one carries a face.
        frames = [frame_row(i) for i in (5, 7, 8, 9)] + [frame_row(6, track_id=1,
                                                                   active=False, score=0.1)]
        rows = run_core([turn_row(start=0.2, end=0.36)], frames)
        assert rows[0]["frames_in_turn"] == 5
        assert rows[0]["face_frames_in_turn"] == 1
        assert rows[0]["agreement"] == "face_never_active"


class TestBoundaries:
    #: (active frames, frames the track is present on, expected state), hand-computed
    #: against the defaults min_active_ratio=0.5 / min_face_frames=2. A ratio of exactly
    #: 0.50 counts as matched; a count that clears the ratio but misses min_face_frames
    #: does not.
    CASES = (
        (1, 2, "face_partial"),   # ratio 0.50 clears, 1 < min_face_frames 2
        (2, 2, "face_matched"),   # exactly at both thresholds
        (2, 5, "face_partial"),   # 0.40 < 0.50
        (3, 5, "face_matched"),   # 0.60 and 3 >= 2
        (2, 3, "face_matched"),   # 0.667 and 2 >= 2
        (1, 1, "face_partial"),   # ratio 1.00 on a single frame
    )

    @pytest.mark.parametrize("active,total,expected", CASES)
    def test_threshold_boundaries_are_inclusive(self, active: int, total: int,
                                                expected: str):
        frames = [frame_row(i, track_id=0, active=i < active, score=1.0)
                  for i in range(total)]
        rows = run_core([turn_row(start=0.0, end=(total - 1) * STEP)], frames)
        assert rows[0]["face_active_frames"] == active
        assert rows[0]["face_frames_in_turn"] == total
        assert rows[0]["frames_in_turn"] == total
        assert rows[0]["agreement"] == expected

    def test_ratio_of_exactly_one_is_matched_and_zero_is_not(self):
        matched = run_core([turn_row(start=0.0, end=0.04)],
                           [frame_row(0, track_id=0, active=True, score=1.0),
                            frame_row(1, track_id=0, active=True, score=1.0)])
        assert matched[0]["agreement"] == "face_matched"
        # min_active_ratio can be raised to demand a sustained speaker: the same track
        # active on 9 of its own 10 frames is 0.90, which a 1.00 bar refuses.
        strict = run_core([turn_row(start=0.0, end=9 * STEP)],
                          [frame_row(i, track_id=0, active=i < 9, score=1.0)
                           for i in range(10)],
                          min_active_ratio=1.0)
        assert strict[0]["frames_in_turn"] == 10
        assert strict[0]["face_active_frames"] == 9
        assert strict[0]["agreement"] == "face_partial"
        assert "ratio 0.90 is below min_active_ratio 1" in strict[0]["agreement_detail"]

    def test_turn_spanning_no_frame_at_all_between_two_blocks(self):
        # Frames 0..1 (0.00-0.04) and 50..51 (2.00-2.04); a turn at 1.0-1.2 sits in the gap.
        frames = ([frame_row(i) for i in (0, 1)]
                  + [frame_row(i, track_id=0, active=True, score=1.0) for i in (50, 51)])
        rows = run_core([turn_row(start=1.0, end=1.2)], frames)
        assert rows[0]["agreement"] == "no_frames_measured"


class TestTieBreak:
    def test_higher_mean_score_wins_and_the_tie_is_declared(self):
        # Tracks 0 and 2 both present on 2 frames, both active on both -> a tie on active
        # count. Track 2's mean score is higher, so it wins, and the row must say so.
        frames = ([frame_row(0, track_id=0, active=True, score=0.5),
                   frame_row(1, track_id=0, active=True, score=0.5),
                   frame_row(2, track_id=2, active=True, score=1.5),
                   frame_row(3, track_id=2, active=True, score=1.5)])
        rows = run_core([turn_row(start=0.0, end=0.12)], frames)
        row = rows[0]
        assert row["face_track_id"] == 2
        assert row["face_active_frames"] == 2
        assert row["face_mean_score"] == pytest.approx(1.5, abs=1e-6)
        assert row["agreement"] == "face_matched"
        assert "tie between tracks 0, 2 resolved to 2 by mean score, then track id" in \
            row["agreement_detail"]

    def test_equal_scores_fall_to_the_lower_track_id(self):
        frames = ([frame_row(0, track_id=3, active=True, score=1.0),
                   frame_row(1, track_id=3, active=True, score=1.0),
                   frame_row(2, track_id=1, active=True, score=1.0),
                   frame_row(3, track_id=1, active=True, score=1.0)])
        rows = run_core([turn_row(start=0.0, end=0.12)], frames)
        assert rows[0]["face_track_id"] == 1
        assert "tie between tracks 1, 3" in rows[0]["agreement_detail"]

    def test_more_active_frames_beats_a_better_score(self):
        # Track 0: 3 active of 3. Track 1: 2 active of 2 with a much higher score.
        frames = ([frame_row(i, track_id=0, active=True, score=0.1) for i in range(3)]
                  + [frame_row(i, track_id=1, active=True, score=5.0) for i in range(3, 5)])
        rows = run_core([turn_row(start=0.0, end=0.2)], frames)
        assert rows[0]["face_track_id"] == 0
        assert "tie" not in rows[0]["agreement_detail"]

    def test_a_single_candidate_is_not_reported_as_a_tie(self):
        frames = [frame_row(i, track_id=7, active=True, score=1.0) for i in range(3)]
        rows = run_core([turn_row(start=0.0, end=0.12)], frames)
        assert "tie between" not in rows[0]["agreement_detail"]


class TestQualifyingTrackWinsOverTheBusiestTrack:
    """A track that clears both thresholds beats a track with more active frames.

    This is the case the fusion got wrong on its first pass and no test saw it: ranking by
    absolute active frames and testing the ratio only on that winner answers *who was on
    screen longest* while the column claims to answer *who is talking*. The dense ASD table
    gives every frame to at most one track, so the two tracks here own disjoint frames —
    exactly what a cutaway looks like when the frame table is read honestly.
    """

    # Track 0 is on screen for 40 of the 46 frames and talks on 10 of them (ratio 0.25).
    # Track 1 is on screen for 6 frames and talks on all 6 (ratio 1.00, 6 >= min_face_frames).
    BUSIEST = list(range(0, 40))
    TALKER = list(range(40, 46))

    @pytest.fixture
    def frames(self) -> list[dict[str, Any]]:
        return ([frame_row(i, track_id=0, active=(i % 4 == 0), score=0.1)
                 for i in self.BUSIEST]
                + [frame_row(i, track_id=1, active=True, score=2.5) for i in self.TALKER])

    def test_the_track_that_clears_both_thresholds_wins_the_turn(self, frames):
        row = run_core([turn_row(start=0.0, end=46 * STEP)], frames)[0]
        assert row["agreement"] == "face_matched"
        assert row["face_track_id"] == 1
        # Hand-computed: track 1 is present on 6 frames and active on all 6.
        assert row["face_active_frames"] == 6
        assert row["face_frames_in_turn"] == 46
        assert row["frames_in_turn"] == 46
        assert row["face_mean_score"] == pytest.approx(2.5)

    def test_the_losing_sighting_is_still_counted_in_the_row(self, frames):
        row = run_core([turn_row(start=0.0, end=46 * STEP)], frames)[0]
        # The busy track's 10 active frames are not lost: they show up as the 40 face frames
        # that are not the winner's, and the detail says so.
        assert row["face_frames_in_turn"] - row["face_active_frames"] == 40
        assert "6/6" in row["agreement_detail"]

    def test_more_active_frames_still_wins_when_both_tracks_qualify(self, frames):
        # Qualification is a filter, not the ranking: with both tracks eligible, most active
        # frames still owns the turn, so the fix does not invert the ordering.
        both = [frame_row(i, track_id=0, active=True, score=0.1) for i in self.BUSIEST]
        row = run_core([turn_row(start=0.0, end=46 * STEP)], both + frames[len(self.BUSIEST):])[0]
        assert row["face_track_id"] == 0
        assert row["agreement"] == "face_matched"

    def test_a_qualifying_tie_is_broken_by_score_not_by_frame_ownership(self):
        # Two qualifying tracks, equal active frames: score decides, and the row declares it.
        frames = ([frame_row(i, track_id=0, active=True, score=0.4) for i in (0, 1)]
                  + [frame_row(i, track_id=2, active=True, score=3.0) for i in (2, 3)])
        row = run_core([turn_row(start=0.0, end=4 * STEP)], frames)[0]
        assert row["face_track_id"] == 2
        assert "tie between tracks 0, 2" in row["agreement_detail"]

    def test_with_no_qualifying_track_the_least_bad_one_is_still_named(self, frames):
        # Raise min_face_frames past both tracks' reach: nothing qualifies, so the turn is
        # face_partial and the row names the busiest candidate rather than hiding it.
        row = run_core([turn_row(start=0.0, end=46 * STEP)], frames,
                       min_face_frames=20)[0]
        assert row["agreement"] == "face_partial"
        assert row["face_track_id"] == 0
        assert "below min_face_frames 20" in row["agreement_detail"]


class TestTurnColumnsAreCopiedVerbatim:
    def test_timing_speaker_and_type_come_from_the_turn_unchanged(self):
        turn = turn_row(start=3.034719, end=8.097219, speaker="SPEAKER_00",
                        turn_id="turn000002")
        row = run_core([turn], [])[0]
        assert row["turn_id"] == "turn000002"
        assert row["speaker_id"] == "SPEAKER_00"
        assert row["start_time"] == pytest.approx(3.034719)
        assert row["end_time"] == pytest.approx(8.097219)
        assert row["duration"] == pytest.approx(5.0625)
        assert row["diarization_type"] == "exclusive"
        assert row["engine"] == "pyannote"
        assert row["schema_version"] == "1.0"
        assert row["video_id"] == "conversation_001"

    def test_one_row_per_turn_in_the_order_given(self):
        turns = [turn_row(start=1.0, end=2.0, turn_id="turn000002"),
                 turn_row(start=0.0, end=0.5, turn_id="turn000001")]
        rows = run_core(turns, [])
        assert [row["turn_id"] for row in rows] == ["turn000002", "turn000001"]


class TestEngineParameterisation:
    """T20: Nemotron is a second call of the same core, not a second fusion."""

    def test_the_core_reads_whichever_turn_table_it_is_handed(self):
        assert {spec.engine for spec in TURN_TABLES.values()} == {"pyannote", "nemotron"}
        assert TURN_TABLES["pyannote"].artifact == "speaker_turns"
        assert TURN_TABLES["nemotron"].artifact == "speaker_turns_nemotron"
        # The fused schema is one schema for both, so the only structural difference is
        # which column the overlap number comes from.
        assert TURN_TABLES["pyannote"].overlap_column is None
        assert TURN_TABLES["nemotron"].overlap_column == "overlap_s"

    def test_nemotron_overlap_is_carried_and_pyannote_overlap_is_absent_not_zero(self):
        frames = [frame_row(i, track_id=0, active=True, score=1.0) for i in range(3)]
        nemotron = run_core([turn_row(start=0.0, end=0.08, speaker="speaker_0",
                                      overlap_s=0.85, nemotron=True)],
                            frames, engine="nemotron")
        assert nemotron[0]["overlap_s"] == pytest.approx(0.85)
        assert nemotron[0]["engine"] == "nemotron"
        assert nemotron[0]["diarization_type"] == "overlapping"
        pyannote = run_core([turn_row(start=0.0, end=0.08)], frames)
        # 0.0 would read as "measured: no overlap". pyannote never measured it.
        assert pyannote[0]["overlap_s"] is None

    def test_a_zero_overlap_still_survives_as_zero(self):
        rows = run_core([turn_row(start=0.0, end=0.08, speaker="speaker_0", overlap_s=0.0,
                                  nemotron=True)], [], engine="nemotron")
        assert rows[0]["overlap_s"] == 0.0

    def test_an_unknown_engine_is_refused_not_silently_fused(self):
        with pytest.raises(ValueError, match="unknown speaker-fusion engine"):
            run_core([turn_row(start=0.0, end=0.1)], [], engine="pyannote-v2")

    def test_the_core_refuses_a_threshold_the_config_validator_would_reject(self):
        frames = [frame_row(0, track_id=0, active=True, score=1.0)]
        with pytest.raises(ValueError, match="min_active_ratio"):
            run_core([turn_row(start=0.0, end=0.04)], frames, min_active_ratio=0.0)
        with pytest.raises(ValueError, match="min_face_frames"):
            run_core([turn_row(start=0.0, end=0.04)], frames, min_face_frames=0)


class TestEmptyAndDegenerateInput:
    def test_no_turns_is_no_rows_not_a_crash(self):
        assert run_core([], [frame_row(0)]) == []

    def test_no_frames_at_all_makes_every_turn_unmeasured(self):
        rows = run_core([turn_row(start=0.0, end=8.0)], [])
        assert [row["agreement"] for row in rows] == ["no_frames_measured"]

    def test_a_turn_without_usable_times_measures_nothing(self):
        # Defensive: a hand-edited table with a null start. Fabricating a window would
        # attribute frames to a turn we cannot locate.
        turn = turn_row(start=0.0, end=0.2)
        broken = dict(turn, start_time=None, end_time=None)
        row = run_core([broken], [frame_row(0, track_id=0, active=True, score=1.0)])[0]
        assert row["agreement"] == "no_frames_measured"
        assert row["start_time"] is None

    def test_unscored_active_frames_leave_the_score_columns_null(self):
        # A frame can be flagged active while its score did not survive normalisation
        # (face_status tracked_unscored). Averaging that as 0.0 would report a measured
        # verdict of "not talking" for a frame that was never scored.
        frames = [frame_row(i, track_id=0, active=True, score=None,
                            face_status="tracked_unscored",
                            frame_reason="past_scored_tail") for i in range(3)]
        row = run_core([turn_row(start=0.0, end=0.08)], frames)[0]
        assert row["agreement"] == "face_matched"
        assert row["face_mean_score"] is None
        assert row["face_score_max"] is None


# ------------------------------------------------------------- the stage itself


def seed_frames(context, rows: Sequence[dict[str, Any]]) -> None:
    write_table(context.artifact("active_speaker_frames"),
                pa.Table.from_pylist(list(rows), schema=ACTIVE_SPEAKER_FRAMES_SCHEMA),
                ACTIVE_SPEAKER_FRAMES_SCHEMA)


def seed_turns(context, rows: Sequence[dict[str, Any]], *, nemotron: bool = False) -> None:
    name = "speaker_turns_nemotron" if nemotron else "speaker_turns"
    schema = SPEAKER_TURNS_NEMOTRON_SCHEMA if nemotron else SPEAKER_TURNS_SCHEMA
    write_table(context.artifact(name),
                pa.Table.from_pylist(list(rows), schema=schema), schema)


def log_recorder(context) -> list[str]:
    messages: list[str] = []

    def _log(message: str, level: int = 20) -> None:
        messages.append(str(message))

    context.log = _log
    return messages


#: 0.00-0.08 with a talking face, then silence on screen, then nothing visible.
MATCH_FRAMES = [frame_row(i, track_id=0, active=True, score=2.0) for i in range(3)]


class TestStageGating:
    def test_disabled_by_config(self, context):
        context.config.speaker_fusion.enabled = False
        enabled, reason = SpeakerFusionStage().enabled(context)
        assert enabled is False
        assert reason == "speaker_fusion.enabled = false"

    def test_enabled_by_default_with_pyannote_only(self, context):
        # The default reproduces T13 exactly: fuse the pyannote turns. Nemotron is the
        # operator's extra option (T20), so it must not arrive switched on by an upgrade.
        assert context.config.speaker_fusion.enabled is True
        assert context.config.speaker_fusion.engines == ["pyannote"]

    def test_missing_active_speaker_frames_skips_with_the_stage_to_run(self, context):
        seed_turns(context, [turn_row(start=0.0, end=0.2)])
        enabled, reason = SpeakerFusionStage().enabled(context)
        assert enabled is False
        assert "active speaker frames unavailable" in reason
        assert "activespeaker" in reason

    def test_both_selected_turn_tables_absent_skips(self, context):
        seed_frames(context, MATCH_FRAMES)
        context.config.speaker_fusion.engines = ["pyannote", "nemotron"]
        enabled, reason = SpeakerFusionStage().enabled(context)
        assert enabled is False
        assert "no selected engine produced a turn table" in reason
        # The reason names both artifacts, which is what tells an operator which stage to
        # run rather than which file to go looking for.
        assert "speaker_turns" in reason and "speaker_turns_nemotron" in reason

    def test_one_engine_present_is_enough_to_run(self, context):
        seed_frames(context, MATCH_FRAMES)
        seed_turns(context, [turn_row(start=0.0, end=0.08)])
        context.config.speaker_fusion.engines = ["pyannote", "nemotron"]
        enabled, reason = SpeakerFusionStage().enabled(context)
        assert enabled is True
        assert reason == ""

    def test_nothing_to_fuse_does_not_emit_an_empty_table(self, context):
        """Spec: skip with a reason, not an empty table a reader would misread."""
        stage = SpeakerFusionStage()
        outcome = stage.run(context)
        assert outcome.status == "skipped"
        assert not context.artifact("speaker_fusion_pyannote").exists()


class TestStageExecution:
    def test_one_selected_engine_writes_its_own_table(self, context):
        seed_frames(context, MATCH_FRAMES)
        seed_turns(context, [turn_row(start=0.0, end=0.08)])
        outcome = SpeakerFusionStage().run(context)
        assert outcome.status == "completed"
        table = read_table(context.artifact("speaker_fusion_pyannote"))
        assert table.schema == SPEAKER_FUSION_SCHEMA
        assert table.num_rows == 1
        assert table.column("engine").to_pylist() == ["pyannote"]
        assert not context.artifact("speaker_fusion_nemotron").exists()

    def test_two_engines_write_two_tables_distinguished_by_the_engine_column(self, context):
        """The two speaker-id namespaces must stay in their own rows and files."""
        seed_frames(context, MATCH_FRAMES)
        seed_turns(context, [turn_row(start=0.0, end=0.08, speaker="SPEAKER_00",
                                      turn_id="turn000001")])
        seed_turns(context, [turn_row(start=0.0, end=0.08, speaker="speaker_0",
                                      turn_id="turn000001", overlap_s=1.2,
                                      nemotron=True)], nemotron=True)
        context.config.speaker_fusion.engines = ["pyannote", "nemotron"]
        outcome = SpeakerFusionStage().run(context)
        assert outcome.status == "completed"

        pyannote = read_table(context.artifact("speaker_fusion_pyannote")).to_pylist()
        nemotron = read_table(context.artifact("speaker_fusion_nemotron")).to_pylist()
        # Both namespaces appear — in their own rows, with their own engine tag.
        assert [(row["engine"], row["speaker_id"]) for row in pyannote] == \
            [("pyannote", "SPEAKER_00")]
        assert [(row["engine"], row["speaker_id"]) for row in nemotron] == \
            [("nemotron", "speaker_0")]
        assert nemotron[0]["overlap_s"] == pytest.approx(1.2)
        assert pyannote[0]["overlap_s"] is None
        # turn_id repeats across engines; it is only unique inside one engine's table,
        # which is what makes `engine` part of the row identity rather than decoration.
        assert pyannote[0]["turn_id"] == nemotron[0]["turn_id"]

    def test_selected_engine_without_its_table_is_skipped_and_logged(self, context):
        seed_frames(context, MATCH_FRAMES)
        seed_turns(context, [turn_row(start=0.0, end=0.08)])
        context.config.speaker_fusion.engines = ["pyannote", "nemotron"]
        messages = log_recorder(context)
        outcome = SpeakerFusionStage().run(context)
        assert outcome.status == "completed"
        assert context.artifact("speaker_fusion_pyannote").is_file()
        assert not context.artifact("speaker_fusion_nemotron").exists()
        assert any("skipping engine nemotron" in message for message in messages)
        assert outcome.detail["provenance"]["extra"]["skipped_engines"] == {
            "nemotron": "speaker_turns_nemotron absent — this engine never produced turns, "
                        "so it was not fused"}
        # The skipped engine is not validated, because it was never asked to be written.
        assert SpeakerFusionStage().validate(context)["speaker_fusion_pyannote"]["rows"] == 1

    def test_zero_turns_is_logged_and_not_a_failure(self, context):
        """A video with no speech is legitimate."""
        seed_frames(context, MATCH_FRAMES)
        seed_turns(context, [])
        messages = log_recorder(context)
        outcome = SpeakerFusionStage().run(context)
        assert outcome.status == "completed"
        assert read_table(context.artifact("speaker_fusion_pyannote")).num_rows == 0
        assert any("produced 0 turns" in message for message in messages)
        assert SpeakerFusionStage().validate(context)["speaker_fusion_pyannote"]["rows"] == 0

    def test_a_stale_frames_table_names_the_missing_column_before_any_row_is_read(self, context):
        """An ASD table that predates `is_active_speaker` must not fuse into empty verdicts.

        Such a table parses, has rows, and would produce a fused table of
        `no_face_visible` for a video that had a face on screen the whole time — a lie, not
        a gap. The stage names the column and the stage to rerun instead.
        """
        import pyarrow.parquet as pq

        seed_frames(context, MATCH_FRAMES)
        seed_turns(context, [turn_row(start=0.0, end=0.08)])
        path = context.artifact("active_speaker_frames")
        pq.write_table(pq.read_table(path).drop(["is_active_speaker"]), path)
        with pytest.raises(ValidationError, match=r"missing columns: is_active_speaker"):
            SpeakerFusionStage().run(context)

    def test_run_end_to_end_writes_a_row_the_validator_accepts(self, context):
        seed_frames(context, MATCH_FRAMES + [frame_row(3), frame_row(4)])
        seed_turns(context, [turn_row(start=0.0, end=0.16)])
        outcome = SpeakerFusionStage().run(context)
        summary = outcome.detail["validation"]
        assert summary["speaker_fusion_pyannote"] == {
            "rows": 1,
            "agreement": {"face_matched": 1},
            "engines": ["pyannote"],
        }

    def test_every_agreement_state_is_reachable_from_one_real_run(self, context):
        """The closed vocabulary and the code that fills it cannot drift apart.

        A state no input can produce is a state a consumer will never handle; a state the
        writer invents fails validation. One turn per state, in one table, computed by the
        same code path a real run takes.
        """
        frames = (
            # 0.00-0.08 track 0 active on all three -> face_matched
            [frame_row(i, track_id=0, active=True, score=1.0) for i in range(3)]
            # 1.00-1.16 track 1 active on 2 of 5 -> 0.40, below the ratio -> face_partial
            + [frame_row(i, track_id=1, active=(i - 25) < 2, score=0.4) for i in range(25, 30)]
            # 2.00-2.16 nothing on screen -> no_face_visible
            + [frame_row(i) for i in range(50, 55)]
            # 3.00-3.08 a face, never flagged -> face_never_active
            + [frame_row(i, track_id=2, active=False, score=-0.5) for i in range(75, 78)]
        )
        turns = [
            turn_row(start=0.0, end=0.08, turn_id="turn000001"),
            turn_row(start=1.0, end=1.16, turn_id="turn000002"),
            turn_row(start=2.0, end=2.16, turn_id="turn000003"),
            turn_row(start=3.0, end=3.08, turn_id="turn000004"),
            turn_row(start=20.0, end=21.0, turn_id="turn000005"),   # nothing measured
        ]
        seed_frames(context, frames)
        seed_turns(context, turns)
        summary = SpeakerFusionStage().run(context).detail["validation"]
        produced = set(summary["speaker_fusion_pyannote"]["agreement"])
        assert produced == set(AGREEMENT_STATES), (
            f"reachable states {sorted(produced)} != vocabulary {sorted(AGREEMENT_STATES)}")
        assert summary["speaker_fusion_pyannote"]["rows"] == 5

    def test_the_existing_diarization_tables_are_untouched(self, context):
        """This is a second result beside v1, never a rewrite of it."""
        seed_frames(context, MATCH_FRAMES)
        seed_turns(context, [turn_row(start=0.0, end=0.08)])
        original = context.artifact("speaker_turns").read_bytes()
        SpeakerFusionStage().run(context)
        assert context.artifact("speaker_turns").read_bytes() == original


class TestStageFingerprint:
    def seeded(self, context) -> SpeakerFusionStage:
        seed_frames(context, MATCH_FRAMES)
        seed_turns(context, [turn_row(start=0.0, end=0.08)])
        return SpeakerFusionStage()

    def test_a_rewritten_turn_table_invalidates_the_fusion(self, context):
        """Re-diarizing must not leave the old agreement verdicts cached."""
        stage = self.seeded(context)
        before = stage.config_fingerprint(context)["pyannote_turns_digest"]
        seed_turns(context, [turn_row(start=0.0, end=0.12)])
        context.scratch.clear()  # the digest is memoised per run, as in every other stage
        assert stage.config_fingerprint(context)["pyannote_turns_digest"] != before
        assert before is not None

    def test_a_rewritten_frames_table_invalidates_the_fusion(self, context):
        stage = self.seeded(context)
        before = stage.config_fingerprint(context)["frames_digest"]
        seed_frames(context, MATCH_FRAMES + [frame_row(3, track_id=1, active=True,
                                                       score=0.5)])
        context.scratch.clear()  # same per-run memoisation
        assert stage.config_fingerprint(context)["frames_digest"] != before

    def test_an_absent_input_is_recorded_as_absent(self, context):
        stage = SpeakerFusionStage()
        payload = stage.config_fingerprint(context)
        assert payload["frames_digest"] is None
        assert payload["nemotron_turns_digest"] is None

    def test_thresholds_and_engine_selection_participate(self, context):
        stage = self.seeded(context)
        before = stage.config_fingerprint(context)
        context.config.speaker_fusion.min_active_ratio = 0.9
        assert stage.config_fingerprint(context) != before
        context.config.speaker_fusion.min_active_ratio = 0.5
        context.config.speaker_fusion.min_face_frames = 5
        assert stage.config_fingerprint(context) != before
        context.config.speaker_fusion.min_face_frames = 2
        context.config.speaker_fusion.engines = ["pyannote", "nemotron"]
        assert stage.config_fingerprint(context) != before

    def test_identical_inputs_reproduce_the_fingerprint(self, context):
        stage = self.seeded(context)
        assert stage.config_fingerprint(context) == SpeakerFusionStage().config_fingerprint(context)


class TestStageValidation:
    def seeded_stage(self, context) -> SpeakerFusionStage:
        seed_frames(context, MATCH_FRAMES)
        seed_turns(context, [turn_row(start=0.0, end=0.08)])
        return SpeakerFusionStage()

    def test_a_stale_table_names_the_missing_column_instead_of_raising_keyerror(self, context):
        import pyarrow.parquet as pq

        stage = self.seeded_stage(context)
        stage.run(context)
        path = context.artifact("speaker_fusion_pyannote")
        table = pq.read_table(path).drop(["agreement_detail", "face_active_frames"])
        pq.write_table(table, path)
        with pytest.raises(ValidationError, match=r"missing columns: .*face_active_frames"):
            stage.validate(context)

    def test_a_missing_output_is_named(self, context):
        stage = self.seeded_stage(context)
        stage.run(context)
        context.artifact("speaker_fusion_pyannote").unlink()
        with pytest.raises(ValidationError, match="speaker_fusion_pyannote missing"):
            stage.validate(context)

    def rewrite(self, context, **column_values: Any) -> None:
        """Overwrite one column of the fused table on disk.

        Each mutation below is a table the writer could not have produced; the validator
        has to catch it, because a fused table that validates while lying about its counts
        is worse than no fused table at all.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = context.artifact("speaker_fusion_pyannote")
        table = pq.read_table(path)
        for name, value in column_values.items():
            index = table.schema.get_field_index(name)
            table = table.set_column(index, name,
                                     pa.array([value], type=table.schema.field(name).type))
        pq.write_table(table, path)

    def test_an_agreement_outside_the_vocabulary_is_rejected(self, context):
        stage = self.seeded_stage(context)
        stage.run(context)
        self.rewrite(context, agreement="probably_a_face")
        with pytest.raises(ValidationError, match="unrecognised agreement"):
            stage.validate(context)

    def test_unnested_counts_are_rejected(self, context):
        # face_active_frames > face_frames_in_turn means the window arithmetic is wrong,
        # so nothing in the table can be trusted.
        stage = self.seeded_stage(context)
        stage.run(context)
        self.rewrite(context, face_frames_in_turn=1)
        with pytest.raises(ValidationError, match="counts are not nested"):
            stage.validate(context)

    def test_a_no_face_verdict_backed_by_visible_frames_is_rejected(self, context):
        stage = self.seeded_stage(context)
        stage.run(context)
        self.rewrite(context, agreement="no_face_visible")
        with pytest.raises(ValidationError, match="claims no_face_visible"):
            stage.validate(context)

    def test_no_frames_measured_backed_by_counted_frames_is_rejected(self, context):
        stage = self.seeded_stage(context)
        stage.run(context)
        self.rewrite(context, agreement="no_frames_measured")
        with pytest.raises(ValidationError, match="claims no_frames_measured"):
            stage.validate(context)

    def test_the_other_engines_id_in_this_table_is_rejected(self, context):
        stage = self.seeded_stage(context)
        stage.run(context)
        self.rewrite(context, engine="nemotron")
        with pytest.raises(ValidationError, match="engine 'nemotron' in the pyannote table"):
            stage.validate(context)

    def test_a_duplicate_turn_id_is_rejected(self, context):
        import pyarrow.parquet as pq

        stage = self.seeded_stage(context)
        stage.run(context)
        path = context.artifact("speaker_fusion_pyannote")
        table = pq.read_table(path)
        pq.write_table(pa.concat_tables([table, table]), path)
        with pytest.raises(ValidationError, match="turn_id is not unique"):
            stage.validate(context)

    def test_a_turn_ending_before_it_starts_is_rejected(self, context):
        stage = self.seeded_stage(context)
        stage.run(context)
        self.rewrite(context, end_time=0.0, start_time=5.0)
        with pytest.raises(ValidationError, match="ends before it starts"):
            stage.validate(context)

    def test_a_winner_with_no_active_frames_is_rejected(self, context):
        stage = self.seeded_stage(context)
        stage.run(context)
        self.rewrite(context, face_active_frames=0)
        with pytest.raises(ValidationError, match="while reporting 0 active frames"):
            stage.validate(context)

    def test_an_empty_detail_is_rejected(self, context):
        stage = self.seeded_stage(context)
        stage.run(context)
        self.rewrite(context, agreement_detail="")
        with pytest.raises(ValidationError, match="empty agreement_detail"):
            stage.validate(context)


class TestWiring:
    def test_registered_in_order_after_its_three_inputs(self):
        from multimodal_pipeline.stages.base import STAGE_ORDER

        assert STAGE_ORDER.index("diarization") < STAGE_ORDER.index("speaker_fusion")
        assert STAGE_ORDER.index("diarization_nemotron") < STAGE_ORDER.index("speaker_fusion")
        assert STAGE_ORDER.index("activespeaker") < STAGE_ORDER.index("speaker_fusion")
        assert STAGE_ORDER[-1] == "finalization"

    def test_dependencies_are_exactly_its_three_producers(self):
        from multimodal_pipeline.stages.base import STAGE_DEPENDENCIES

        assert STAGE_DEPENDENCIES["speaker_fusion"] == (
            "diarization", "diarization_nemotron", "activespeaker")
        # finalization must follow it, or the manifest will not describe the fused tables.
        assert "speaker_fusion" in STAGE_DEPENDENCIES["finalization"]

    def test_the_stage_is_constructible_by_the_orchestrator(self):
        from multimodal_pipeline.orchestrator import STAGE_CLASSES, build_stages
        from multimodal_pipeline.stages.base import STAGE_ORDER

        assert STAGE_CLASSES["speaker_fusion"] is SpeakerFusionStage
        assert "speaker_fusion" in [stage.name for stage in build_stages()]
        assert len(STAGE_CLASSES) == len(STAGE_ORDER)

    def test_declares_its_contract(self):
        stage = SpeakerFusionStage()
        assert stage.name == "speaker_fusion"
        assert stage.inputs == ("active_speaker_frames", "speaker_turns",
                                "speaker_turns_nemotron")
        assert set(stage.outputs) == {"speaker_fusion_pyannote", "speaker_fusion_nemotron"}
        assert stage.config_keys == ("speaker_fusion",)

    def test_it_is_pure_python_with_no_environment_of_its_own(self):
        """No subprocess, no uv env: two parquet in, one parquet out.

        Asserted structurally, because every other speaker stage reaches a GPU through
        `WorkerStage` and a reader of this class should be able to see that it does not.
        """
        from multimodal_pipeline.stages.base import Stage, WorkerStage

        assert issubclass(SpeakerFusionStage, Stage)
        assert not issubclass(SpeakerFusionStage, WorkerStage)
        assert not hasattr(SpeakerFusionStage, "raw_artifact")
        assert not hasattr(SpeakerFusionStage, "uv_project")

    def test_the_orchestrator_reuses_a_completed_fusion(self, context):
        """A completed fusion with unchanged inputs is reused, and only then.

        The hash is built the way ``VideoRunner.config_payload_for`` builds it (stage
        fingerprint + stage name + schema version), because that is the value
        ``should_reuse`` compares against; hashing the bare fingerprint here would compare
        a value the orchestrator never writes.
        """
        from multimodal_pipeline.config import stable_hash
        from multimodal_pipeline.stages.base import should_reuse

        seed_frames(context, MATCH_FRAMES)
        seed_turns(context, [turn_row(start=0.0, end=0.08)])
        stage = SpeakerFusionStage()
        stage.run(context)
        # Stage.run only executes; VideoRunner records completion. Marking it here is what
        # lets should_reuse get past "status is pending".
        context.state.mark_completed(stage.name)

        payload = dict(stage.config_fingerprint(context))
        payload.setdefault("stage", stage.name)
        payload.setdefault("schema_version", context.tools.get("schema_version"))
        digest = stable_hash(payload, length=16)

        # What VideoRunner writes at completion, so the first comparison below is the one
        # a resume actually makes.
        context.state.stage(stage.name).config_hash = digest
        context.state.stage(stage.name).dependency_hash = "y"
        assert should_reuse(stage, context, config_hash=digest,
                            dependency_hash="y", force=False)[0] is True
        assert should_reuse(stage, context, config_hash="stale",
                            dependency_hash="y", force=False) == (
            False, "configuration changed")

        context.config.speaker_fusion.min_active_ratio = 0.9
        moved = stable_hash(dict(stage.config_fingerprint(context), stage=stage.name,
                                 schema_version=context.tools["schema_version"]), length=16)
        assert moved != digest
        assert should_reuse(stage, context, config_hash=moved,
                            dependency_hash="y", force=False) == (
            False, "configuration changed")

    def test_reuse_does_not_demand_an_unselected_engine_s_table(self, context):
        seed_frames(context, MATCH_FRAMES)
        seed_turns(context, [turn_row(start=0.0, end=0.08)])
        context.config.speaker_fusion.engines = ["pyannote"]
        stage = SpeakerFusionStage()
        stage.run(context)
        assert not context.artifact("speaker_fusion_nemotron").exists()
        assert stage.outputs_present(context) is True


class TestStaleEngineOutputIsNotLeftBehind:
    """A fused table outlives its engine, and that is a defect, not a leftover.

    `validate` and the reuse test both look only at engines that are fusible *now*, so a
    `fusion_nemotron.parquet` left by an earlier run — before Nemotron was deselected, or
    before its turn table was deleted — survives forever, validates fine, and is indistinguish
    to a reader from a table computed against the current inputs. Joining it to current turns
    produces confident nonsense, which is the failure this stage exists to prevent.
    """

    def _fuse_both(self, context):
        seed_frames(context, MATCH_FRAMES)
        seed_turns(context, [turn_row(start=0.0, end=0.08)])
        seed_turns(context, [turn_row(start=0.0, end=0.08, speaker="speaker_0",
                                      turn_id="turn000001", overlap_s=0.4, nemotron=True)],
                   nemotron=True)
        context.config.speaker_fusion.engines = ["pyannote", "nemotron"]
        assert SpeakerFusionStage().run(context).status == "completed"
        assert context.artifact("speaker_fusion_nemotron").is_file()

    def test_deselecting_an_engine_removes_its_table_on_the_next_run(self, context):
        self._fuse_both(context)
        context.config.speaker_fusion.engines = ["pyannote"]
        messages = log_recorder(context)
        outcome = SpeakerFusionStage().run(context)
        assert outcome.status == "completed"
        assert context.artifact("speaker_fusion_pyannote").is_file()
        assert not context.artifact("speaker_fusion_nemotron").exists()
        assert outcome.detail["provenance"]["extra"]["pruned_outputs"] == {
            "speaker_fusion_nemotron": "engine no longer selected"}
        assert any("removed stale fusion_nemotron.parquet" in message
                   for message in messages)

    def test_a_turn_table_that_disappears_prunes_the_table_it_fed(self, context):
        self._fuse_both(context)
        context.artifact("speaker_turns_nemotron").unlink()
        messages = log_recorder(context)
        outcome = SpeakerFusionStage().run(context)
        assert outcome.status == "completed"
        assert not context.artifact("speaker_fusion_nemotron").exists()
        assert outcome.detail["provenance"]["extra"]["pruned_outputs"] == {
            "speaker_fusion_nemotron": "engine selected but its turn table is absent"}
        assert any("turn table is absent" in message for message in messages)

    def test_a_leftover_table_makes_the_previous_result_unreusable(self, context):
        """Reuse must not be the path that keeps a stale table published.

        Without this, the engine disappears, `outputs_present` still sees the table it wants,
        the stage is skipped as "valid previous result", and the stale file is never reached.
        """
        self._fuse_both(context)
        stage = SpeakerFusionStage()
        context.config.speaker_fusion.engines = ["pyannote"]
        assert stage.outputs_present(context) is False
        context.artifact("speaker_turns_nemotron").unlink()
        context.config.speaker_fusion.engines = ["pyannote", "nemotron"]
        assert stage.outputs_present(context) is False

    def test_pruning_only_reaches_this_stage_s_declared_outputs(self, context):
        """The prune loop must never be able to delete somebody else's artifact.

        The pyannote turn table is an *input* owned by `diarization` and the frames table by
        `activespeaker`; deselecting an engine drops the fused table this stage wrote and
        leaves both inputs exactly where they were, because deleting an upstream table would
        erase the evidence that the engine ever ran.
        """
        self._fuse_both(context)
        turn_table = context.artifact("speaker_turns_nemotron")
        frames = context.artifact("active_speaker_frames")
        turns = context.artifact("speaker_turns")
        context.config.speaker_fusion.engines = ["pyannote"]
        SpeakerFusionStage().run(context)
        assert not context.artifact("speaker_fusion_nemotron").exists()
        assert turn_table.is_file() and frames.is_file() and turns.is_file()
