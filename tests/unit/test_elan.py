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
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pytest
from multimodal_pipeline.artifacts import ARTIFACT_LAYOUT
from multimodal_pipeline.elan import (
    TIERS,
    UNKNOWN_MIME_TYPE,
    NonFiniteTimestamp,
    build_eaf,
    collapse_runs,
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
    PERSON_TRACKS_SCHEMA,
    SEGMENTS_SCHEMA,
    SPEAKER_TURNS_SCHEMA,
    WORDS_SCHEMA,
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

    def test_a_null_timestamp_lands_at_zero(self) -> None:
        assert seconds_to_ms(None) == 0
        assert seconds_to_ms(None, end=True) == 1

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
            (-0.5, 0.5, (0, 500)),
            (None, None, (0, 1)),
        ],
    )
    def test_an_interval_always_satisfies_start_less_than_end(self, start: Any, end: Any,
                                                              expected: tuple[int, int]) -> None:
        got = interval_ms(start, end)
        assert got == expected
        assert got[0] < got[1], "ELAN refuses an annotation whose start is not before its end"

    def test_no_interval_ever_comes_out_negative_or_inverted(self) -> None:
        """The property, over a sweep — the pair rule has to hold for every input.

        Parametrised cases show the interesting ones; this one says the rule is total, which
        is what the format actually requires. Includes values that round to zero width and
        values whose end precedes their start.
        """
        values = [-3.0, -0.0004, 0.0, 0.0004, 0.0006, 0.0009, 0.01, 0.039999, 1.0, 4.169999]
        for start in values:
            for end in values:
                low, high = interval_ms(start, end)
                assert low >= 0, (start, end)
                assert high > low, (start, end, low, high)


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


def _word(word: str, start: float, end: float, **extra: Any) -> dict[str, Any]:
    return {"schema_version": "1.0", "video_id": "clip", "segment_id": "seg-0",
            "word_id": f"w-{start}", "start_time": start, "end_time": end,
            "duration": end - start, "speaker_id": "SPEAKER_00", "word": word,
            "confidence": 0.95, "alignment_status": "aligned", **extra}


def _asd_frame(index: int, label: str, track_id: int | None) -> dict[str, Any]:
    """One ASD frame: `label` is speaking | not_speaking | no_face."""
    stamp = round(index * STEP, 6)
    face_status = "no_face" if label == "no_face" else "tracked"
    return {
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
        "diarization_type": "pyannote",
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


class TestBuildEaf:
    """The twelve tiers, built from real Parquet and read back out of real XML."""

    def test_only_the_tiers_with_input_files_are_present(self, dataset: dict[str, Path]) -> None:
        eaf = build_eaf(dataset["dir"], dataset["video"], log=lambda *a, **k: None)
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
        build_eaf(dataset["dir"], dataset["video"], log=lambda *a, **k: lines.append(str(a[0])))
        # "skipped (" and not bare "skipped": the closing census line also reports the count,
        # and counting it would make this assertion pass at seven tiers or at seventy.
        skipped = [line for line in lines if "skipped (" in line]
        assert len(skipped) == 7, skipped
        assert sum(1 for line in skipped if "gloss_en" in line) == 1
        assert any("translation/segments_en.parquet not produced" in line for line in skipped)
        # The census line agrees with the per-tier lines rather than restating a constant.
        assert any(line.startswith("elan: 5 tier(s)") and "skipped 7" in line for line in lines)

    def test_a_known_word_lands_on_the_expected_millisecond_pair(self,
                                                                dataset: dict[str, Path]
                                                                ) -> None:
        assert annotations(eaf_of(dataset), "words") == [
            (0, 400, "hello"),
            (500, 900, "multi line"),
            # Zero width in, +1 ms out: ELAN cannot hold (1000, 1000).
            (1000, 1001, "zero"),
        ]

    def test_a_newline_in_a_producer_string_cannot_reach_the_file(self,
                                                                 dataset: dict[str, Path]
                                                                 ) -> None:
        text = " ".join(value for _s, _e, value in annotations(eaf_of(dataset), "words"))
        assert "\n" not in text and "\r" not in text
        assert "multi line" in text

    def test_the_turn_text_names_the_engine_that_produced_it(self,
                                                             dataset: dict[str, Path]
                                                             ) -> None:
        """Two turn tables, two engine names, decided by the tier and not by the filename."""
        assert annotations(eaf_of(dataset), "turns_pyannote") == [
            (0, 1000, "SPEAKER_00 (pyannote, pyannote)")]

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
        back to the video rather than by comparing strings — a ``./clip.mp4`` would match a
        literal but be wrong from any dataset directory that is not the video's own.
        """
        eaf = eaf_of(dataset)
        descriptor = eaf.media_descriptors[0]
        video = dataset["video"].resolve()
        root = dataset["dir"].resolve()
        assert descriptor["MEDIA_URL"] == video.as_uri()
        relative = descriptor["RELATIVE_MEDIA_URL"]
        assert relative == "../../input_videos/clip.mp4"
        assert (root / relative).resolve() == video
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
        eaf = build_eaf(root, video, log=lambda *a, **k: None)
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
        eaf = build_eaf(root, video, log=lambda *a, **k: None)
        assert eaf.media_descriptors[0]["MIME_TYPE"] == UNKNOWN_MIME_TYPE
        assert eaf.media_descriptors[0]["MIME_TYPE"] != "video/mp4"
        # and the document still writes and parses with the empty type
        out = tmp_path / "clip.eaf"
        eaf.to_file(str(out))
        ET.parse(out)

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
            "diarization_type": "pyannote",
        }])
        lines: list[str] = []
        eaf = build_eaf(root, video, log=lambda msg, *a, **k: lines.append(str(msg)))
        # The other tier survived, and the good row of the poisoned tier survived with it.
        assert annotations(eaf, "words") == [(0, 400, "hello")]
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
        eaf = build_eaf(root, video, log=lambda msg, *a, **k: lines.append(str(msg)))
        assert annotations(eaf, "words") == []
        assert "words" in tiers_of(eaf)
        assert [line for line in lines if "dropped 2 of 2" in line], lines

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
        eaf = build_eaf(dataset["dir"], dataset["video"], log=lambda *a, **k: lines.append(str(a[0])))
        assert "pose_presence" not in tiers_of(eaf)
        assert any("pose_presence skipped" in line and "unreadable" in line for line in lines)
        assert {"words", "asd_speaking", "face_tracks", "turns_pyannote"} <= set(tiers_of(eaf))

    def test_the_tier_census_in_the_document_matches_the_document(self,
                                                                 dataset: dict[str, Path]
                                                                 ) -> None:
        recorded = dict(eaf_of(dataset).properties)["pipeline-tiers"]
        counts = tier_counts(eaf_of(dataset))
        assert recorded == " ".join(f"{name}={counts[name]}" for name in sorted(counts))


def real_tier_names(root: ET.Element) -> list[str]:
    """The document's tier names, minus the one pympi always writes.

    ``pympi`` emits a ``default`` TIER in every document (its implicit tier, which nothing here
    annotates). Asserting against the raw element list would expect six names for five tiers and
    would stop being able to see a tier that really went missing.
    """
    return [element.attrib["TIER_ID"] for element in root.iter("TIER")
            if element.attrib.get("TIER_ID") != "default"]


def eaf_of(dataset: dict[str, Path]) -> Any:
    return build_eaf(dataset["dir"], dataset["video"], log=lambda *a, **k: None)


class TestTierRegistration:
    """Every tier reads a registered artifact; no file name is invented here."""

    def test_the_twelve_tiers_are_the_ones_the_design_named(self) -> None:
        assert [spec.tier for spec in TIERS] == [
            "words", "segments_src", "gloss_en", "turns_pyannote", "turns_nemotron",
            "fusion_pyannote", "fusion_nemotron", "asd_speaking", "face_tracks",
            "person_tracks", "pose_presence", "voiced_blocks"]

    def test_every_tier_input_is_a_registered_artifact(self) -> None:
        unregistered = sorted({spec.artifact for spec in TIERS} - set(ARTIFACT_LAYOUT))
        assert not unregistered, f"tiers read artifacts that do not exist: {unregistered}"

    def test_no_two_tiers_read_the_same_table(self) -> None:
        """One producer per tier, so a tier's absence names exactly one producer."""
        artifacts = [spec.artifact for spec in TIERS]
        assert len(artifacts) == len(set(artifacts))

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
            "pose_body": (BODY_SCHEMA, ("timestamp", "confidence")),
            "acoustic_frames": (ACOUSTIC_FRAMES_SCHEMA, ("timestamp", "f0_hz")),
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

    def test_the_corpus_export_has_every_tier_and_real_words(self, tmp_path: Path) -> None:
        """The real tables, so the column names stop being this file's invention.

        The XML check writes to ``tmp_path`` and not beside the dataset: ``data/processed/`` is
        the operator's corpus and this test has no business creating files in it, even ones it
        deletes afterwards — a crash between the write and the unlink would leave a stray .eaf
        that the next manifest would list as an artifact.
        """
        if not (self.CORPUS / "manifest.json").is_file():
            pytest.skip(f"corpus dataset not present under {PROCESSED}")
        eaf = build_eaf(self.CORPUS, self.VIDEO, log=lambda *a, **k: None)
        counts = tier_counts(eaf)
        assert len(counts) >= 8, counts
        assert counts.get("words", 0) > 0
        descriptor = eaf.media_descriptors[0]
        path = self.CORPUS / descriptor["RELATIVE_MEDIA_URL"]
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
