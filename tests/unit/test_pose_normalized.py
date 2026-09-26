"""Our computed coordinates against `dfMaker`'s, on every key the two tables share.

This is the test the feature rests on: the stage runs over a real ``pose/body.parquet``
and the output is read back and compared to the committed reference CSVs at 1e-9. No
mocks, and no reimplementation of the formula inside the test -- a second implementation
would only prove both were computed from the same numbers.

The pixel table is rebuilt from the fixture's own ``x``/``y``/``c`` columns rather than
read from ``data/processed/``, which is gitignored: a test that earns a feature cannot be
one that skips wherever nobody has run OpenPose. That rebuilding is itself verified
against the real table where the corpus is present.


One class lives here: `TestAgainstTheReference`, the test that earns the feature. It runs
the *stage* over the ``pose/body.parquet`` of two processed clips and compares every
computed coordinate with ``dfMaker()`` from CRAN ``multimolang`` 0.1.1, whose output is
committed under ``tests/fixtures/pose_normalized/``. Those CSVs were produced by the
reference, not by this code: the join key is ``(frame, people_id - 1, points) ==
(frame_number, detection_index, keypoint_id)``. A sign or transposition error in the change
of basis produces a table that looks entirely plausible — the neck still lands where it
should and only the feet are wrong — so this is the check that cannot be a mock and cannot
be a hand-derived expectation from the same algebra it is testing.

The tests that cover what the corpus cannot are in the two sibling files, not here, because
they were committed as separate review links and each one has to be small enough to review:

* ``test_pose_normalize_math.py`` — `TestMaskingIsPerCoordinate` (no half-zero keypoint
  exists in either fixture video, so the fixtures are silent on the per-coordinate rule)
  and `TestBasisStates` (the degenerate basis).
* ``test_pose_normalized_stage.py`` — `TestAbsenceIsNamed`, the three absence states §20.4
  asks for.

Every expected number in those two files is hand-computed from the coordinates written in
the test.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pytest
from multimodal_pipeline.pose_normalize import (
    BASIS_OK,
    VALUE_BASIS_UNUSABLE,
    VALUE_NORMALIZED,
)
from multimodal_pipeline.schemas import (
    BODY_25_KEYPOINT_NAMES,
    BODY_SCHEMA,
    read_table,
    write_table,
)
from multimodal_pipeline.stages.pose_normalized import PoseNormalizedStage

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "pose_normalized"
PROCESSED = ROOT / "data" / "processed"

#: The reference run, and the processed dataset its input JSON came from. Both are needed:
#: the fixture carries the reference's answer, the Parquet table is what this stage reads.
REFERENCE_RUNS = (
    pytest.param("kabc", id="kabc"),
    pytest.param("cnn", id="cnn"),
)

#: Every fixture coordinate agrees with the pipeline's pixel table at this tolerance, and
#: the computed coordinates agree with the reference at it too. Measured agreement, on this
#: machine, with the tolerance forced to zero so the tests report it: 6.66e-15 over the 154
#: numeric kabc rows and 7.11e-15 over the 290 numeric cnn rows, and 9.55e-15 across all
#: 53588 numeric points of the whole corpus (section 21 of the ODD task). 1e-9 leaves room
#: for a different last-bit path and still refuses anything a reflection or a swapped axis
#: could produce (those are O(1) or larger).
TOLERANCE = 1e-9


def fixture_rows(tag: str) -> list[dict[str, str]]:
    path = FIXTURES / f"dfmaker_0.1.1_{tag}_midhip_neck.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def dataset_for(tag: str) -> Path:
    """The processed dataset whose raw JSON produced this fixture, or skip.

    Only used by the one test that cross-checks the fixture against the table the pipeline
    really wrote. ``data/processed/`` is gitignored, so that check skips on a fresh clone —
    which is why the reference comparison itself does **not** depend on it.
    """
    fixture_id = fixture_rows(tag)[0]["id"]
    # video_id is slugify(stem) of the source file, i.e. the fixture id with '.' -> '_'.
    for candidate in (Path(fixture_id), Path(fixture_id.replace(".", "_"))):
        path = PROCESSED / candidate / "pose" / "body.parquet"
        if path.is_file():
            return path.parent.parent
    pytest.skip(f"processed dataset for fixture {tag!r} not present under data/processed/ "
                f"(looked for {fixture_id} and its slugified form); regenerate with "
                f"`uv run multimodal-pipeline run`")
    raise AssertionError  # unreachable


def body_rows_from_fixture(tag: str) -> list[dict[str, Any]]:
    """The pixel table the reference was run on, rebuilt from the fixture itself.

    This is possible because the CSV carries the reference's *inputs* as well as its
    outputs: a row's ``x``/``y``/``c`` are OpenPose's, and a row whose ``x`` is ``NA`` is a
    keypoint the normalizer never wrote (``score <= 0``). Verified against the real
    ``pose/body.parquet`` — see ``test_the_rebuilt_table_is_the_real_one`` — the rebuilt and
    on-disk tables are identical key-for-key and value-for-value over the 5 fixture frames
    of both clips (154 and 349 keypoints).

    Rebuilding rather than reading ``data/processed/`` is what makes the reference comparison
    run on a fresh clone: that directory is gitignored and regenerable, and a test that
    earns a feature cannot be one that skips wherever nobody has run OpenPose.

    ``timestamp`` is null rather than invented. The fixture carries no timing, the transform
    does not read it, and a value computed as ``frame / 25`` would be a claim about a clip
    whose frame rate the CSV does not state.
    """
    video_id = fixture_video_id(tag)
    rows: list[dict[str, Any]] = []
    for row in fixture_rows(tag):
        if row["x"] == "NA":
            continue  # a keypoint OpenPose did not find, so the body table has no row
        keypoint_id = int(row["points"])
        rows.append({
            "schema_version": "1.0",
            "video_id": video_id,
            "frame_number": int(row["frame"]),
            "timestamp": None,
            # R's people_id is 1-based; our detection_index is 0-based.
            "detection_index": int(row["people_id"]) - 1,
            "keypoint_id": keypoint_id,
            "keypoint_name": BODY_25_KEYPOINT_NAMES[keypoint_id],
            "x": float(row["x"]),
            "y": float(row["y"]),
            "confidence": float(row["c"]),
        })
    # The stage streams one frame at a time and refuses a table that revisits a frame, so
    # the rebuilt input has to be as contiguous as the one openpose writes.
    rows.sort(key=lambda row: (row["frame_number"], row["detection_index"],
                               row["keypoint_id"]))
    return rows


def fixture_video_id(tag: str) -> str:
    """The video_id the fixture's keypoints belong to, in our slugified form."""
    return fixture_rows(tag)[0]["id"].replace(".", "_")


def seeded_context(context, tag: str):
    """Give the shared `context` fixture a body table, under the id the keypoints own.

    Two sources, in this order: the real ``pose/body.parquet`` when this machine has the
    processed corpus, otherwise the same table rebuilt from the fixture (proved identical by
    ``test_the_rebuilt_table_is_the_real_one``). Either way ``source`` is replaced so
    ``ctx.video_id`` is the id the keypoints really belong to — the stage stamps
    ``ctx.video_id`` into its output the way every other normalizer does, and a mismatch here
    would be a fixture artifact rather than a fact about the data.
    """
    from multimodal_pipeline.discovery import VideoSource

    processed = PROCESSED_DIR_FOR(tag)
    target = context.artifact("pose_body")
    target.parent.mkdir(parents=True, exist_ok=True)
    if processed is not None:
        write_table(target, pa.Table.from_pylist(
            [row for row in read_table(processed / "pose" / "body.parquet").to_pylist()
             if row["frame_number"] in fixture_frames(tag)], schema=BODY_SCHEMA), BODY_SCHEMA)
    else:
        write_table(target, pa.Table.from_pylist(body_rows_from_fixture(tag), schema=BODY_SCHEMA),
                    BODY_SCHEMA)
    context.source = VideoSource(path=context.source.path,
                                 relative_path=context.source.relative_path,
                                 video_id=fixture_video_id(tag))
    context.scratch.clear()  # the body digest is memoised per run, as in every other stage
    return context


def fixture_frames(tag: str) -> set[int]:
    return {int(row["frame"]) for row in fixture_rows(tag)}


def PROCESSED_DIR_FOR(tag: str) -> Path | None:
    """Like :func:`dataset_for` but answering None instead of skipping."""
    fixture_id = fixture_rows(tag)[0]["id"]
    for candidate in (Path(fixture_id), Path(fixture_id.replace(".", "_"))):
        path = PROCESSED / candidate / "pose" / "body.parquet"
        if path.is_file():
            return path.parent.parent
    return None


def run_stage(context) -> dict[str, Any]:
    outcome = PoseNormalizedStage().run(context)
    assert outcome.status == "completed", outcome.message
    return outcome.detail["provenance"]["extra"]


def normalized_index(context) -> dict[tuple[int, int, int], dict[str, Any]]:
    rows = read_table(context.artifact("pose_normalized")).to_pylist()
    index = {(row["frame_number"], row["detection_index"], row["keypoint_id"]): row
             for row in rows}
    assert len(index) == len(rows), "two normalized rows share one (frame, person, keypoint)"
    return index


def fixture_index(rows: Sequence[dict[str, str]]) -> dict[tuple[int, int, int], dict[str, str]]:
    """Fixture rows keyed the way our table is keyed.

    ``people_id`` is R's 1-based person index and ``points`` is OpenPose's 0-based keypoint
    index, so only one of the two needs adjusting. Asserted on the shared keys below rather
    than trusted here: if this mapping were wrong the comparison would compare nothing.
    """
    return {(int(row["frame"]), int(row["people_id"]) - 1, int(row["points"])): row
            for row in rows}


# ------------------------------------------------------- the reference comparison


class TestAgainstTheReference:
    """Our numbers against `dfMaker`'s, on every key the two tables share."""

    @pytest.mark.parametrize("tag", REFERENCE_RUNS)
    def test_the_join_key_finds_the_measured_number_of_shared_rows(self, context, tag):
        """The comparison below is only worth anything if the join is the real one.

        Measured, not inherited: kabc shares 154 fixture rows with the pixel table and all
        154 have a reference coordinate; cnn shares 349, of which 290 are numeric and 59 are
        NA because their person-frame has no MidHip. (The cnn fixture holds 210 NA rows in
        total; the other 151 are keypoints our table never had a row for, so they are not
        shared and this test cannot see them.) If the join key drifted — a different
        person-index convention, a filtered keypoint — these numbers change and this fails
        loudly instead of the comparison quietly comparing less.
        """
        seeded_context(context, tag)
        run_stage(context)
        ours = normalized_index(context)
        fixture = fixture_index(fixture_rows(tag))
        shared = set(ours) & set(fixture)
        numeric = {key for key, row in fixture.items() if row["nx"] != "NA"}
        assert numeric <= shared, f"{tag}: the reference has coordinates we have no row for"
        # Everything we share that the reference NA-ised is a row of a person-frame the
        # reference could not frame: the two tables hold the same keypoints, they disagree
        # only about whether a frame of reference existed.
        assert all(fixture[key]["nx"] == "NA" for key in shared - numeric)
        assert len(shared) == {"kabc": 154, "cnn": 349}[tag]
        assert len(numeric) == {"kabc": 154, "cnn": 290}[tag]
        # And the join is not vacuous in the other direction either.
        assert len(ours) >= len(shared)

    @pytest.mark.parametrize("tag", REFERENCE_RUNS)
    def test_the_pixels_under_the_join_are_the_same_pixels(self, context, tag):
        """Guards the comparison: same key must mean same (x, y), not just same index.

        If the pixel table and the fixture disagreed about a coordinate, agreement between
        our output and the reference would prove nothing about this corpus — it would only
        prove both were computed from the same *numbers*, which is what the fixture is for.
        """
        seeded_context(context, tag)
        body = read_table(context.artifact("pose_body")).to_pylist()
        pixels = {(row["frame_number"], row["detection_index"], row["keypoint_id"]):
                  (row["x"], row["y"]) for row in body}
        disagreements = []
        for key, row in fixture_index(fixture_rows(tag)).items():
            if row["x"] == "NA" or key not in pixels:
                continue
            if (float(row["x"]), float(row["y"])) != pixels[key]:
                disagreements.append((key, row["x"], row["y"], pixels[key]))
        assert not disagreements, f"{tag}: fixture and pose_body disagree on pixels: " \
                                  f"{disagreements[:3]}"

    @pytest.mark.parametrize("tag", REFERENCE_RUNS)
    def test_the_rebuilt_table_is_the_real_one(self, tag):
        """The claim `body_rows_from_fixture` rests on, checked against the real table.

        The hermetic path rebuilds ``pose/body.parquet`` from the fixture because
        ``data/processed/`` is gitignored. That rebuild is only a stand-in while it keeps
        exactly the rows the pipeline's own normalizer kept — so where the corpus exists,
        the rebuilt table and the real one are compared key-for-key and value-for-value
        over the five fixture frames of each clip. Skips without the corpus; the rule
        itself is pinned hermetically by the next test.
        """
        real = {
            (row["frame_number"], row["detection_index"], row["keypoint_id"]):
                (row["x"], row["y"], row["confidence"], row["keypoint_name"])
            for row in read_table(dataset_for(tag) / "pose" / "body.parquet").to_pylist()
            if row["frame_number"] in fixture_frames(tag)
        }
        rebuilt = {
            (row["frame_number"], row["detection_index"], row["keypoint_id"]):
                (row["x"], row["y"], row["confidence"], row["keypoint_name"])
            for row in body_rows_from_fixture(tag)
        }
        only_rebuilt = sorted(set(rebuilt) - set(real))
        only_real = sorted(set(real) - set(rebuilt))
        differs = [k for k in set(rebuilt) & set(real) if rebuilt[k] != real[k]]
        assert not (only_rebuilt or only_real or differs), (
            f"{tag}: rebuilt-from-fixture differs from pose/body.parquet — "
            f"only-rebuilt={only_rebuilt[:3]} only-real={only_real[:3]} "
            f"value-differs={[(k, rebuilt[k], real[k]) for k in differs[:3]]}")

    @pytest.mark.parametrize("tag", REFERENCE_RUNS)
    def test_the_rebuilt_table_follows_the_real_normalizer_rule(self, tag):
        """The fresh-clone path, checked against the normalizer that owns the rule.

        ``body_rows_from_fixture`` keeps the fixture rows whose ``x`` is not ``NA``. The
        rule that actually decides ``pose/body.parquet`` lives in
        :func:`openpose_frame_rows`, which drops ``score <= 0`` and ``Background`` — a rule
        this file does not own and must not restate from memory. So an OpenPose JSON
        document is rebuilt from the fixture (a row the reference NA-ed becomes a
        ``0, 0, 0`` triple, which is what dfMaker received), the pipeline's real
        normalizer runs over it, and what it keeps must equal what the rebuild keeps.
        Without this, a change to the normalizer's filter would leave every fresh clone
        feeding the stage a table production would never have produced, and the reference
        comparison would still pass.

        The fixture covers keypoints 0-24 (its ``0`` is Nose, the row OpenPose numbers
        zero); ``Background`` appears in neither table, so this pins the score filter, not
        the Background drop — that one is openpose's own test's job.
        """
        from multimodal_pipeline.normalization import openpose_frame_rows

        rows = fixture_rows(tag)
        by_frame: dict[int, dict[int, dict[int, dict[str, str]]]] = {}
        for row in rows:
            by_frame.setdefault(int(row["frame"]), {}) \
                     .setdefault(int(row["people_id"]), {})[int(row["points"])] = row

        rebuilt = {
            (row["frame_number"], row["detection_index"], row["keypoint_id"]):
                (row["x"], row["y"], row["confidence"], row["keypoint_name"])
            for row in body_rows_from_fixture(tag)
        }
        normalised: dict[tuple[int, int, int], tuple[float, float, float, str]] = {}
        for frame, people in sorted(by_frame.items()):
            ids = sorted(people)
            # detection_index is the array position, so the fixture's people ids have to be
            # contiguous from 1 within a frame for the document below to say what we mean.
            assert ids == list(range(1, len(ids) + 1)), f"{tag} frame {frame}: people ids {ids}"
            document = {"version": "1.5.1", "people": []}
            for person_id in ids:
                keypoints = by_frame[frame][person_id]
                triples = [0.0] * (25 * 3)
                for keypoint_id in range(25):
                    match = keypoints.get(keypoint_id)
                    if match is not None and match["x"] != "NA":
                        base = keypoint_id * 3
                        triples[base] = float(match["x"])
                        triples[base + 1] = float(match["y"])
                        triples[base + 2] = float(match["c"])
                document["people"].append({"pose_keypoints_2d": triples})
            for row in openpose_frame_rows(document, fixture_video_id(tag), frame, 0.0)["body"]:
                normalised[(frame, row["detection_index"], row["keypoint_id"])] = (
                    row["x"], row["y"], row["confidence"], row["keypoint_name"])

        only_rebuilt = sorted(set(rebuilt) - set(normalised))
        only_normalised = sorted(set(normalised) - set(rebuilt))
        differs = [k for k in set(rebuilt) & set(normalised) if rebuilt[k] != normalised[k]]
        assert not (only_rebuilt or only_normalised or differs), (
            f"{tag}: the fixture rebuild no longer matches openpose_frame_rows — "
            f"only-rebuilt={only_rebuilt[:3]} only-normalizer={only_normalised[:3]} "
            f"value-differs={[(k, rebuilt[k], normalised[k]) for k in differs[:3]]} — "
            "the hermetic stand-in has drifted from the rule that writes pose/body.parquet")

    @pytest.mark.parametrize("tag", REFERENCE_RUNS)
    def test_every_computed_coordinate_matches_the_reference(self, context, tag):
        """The test this feature rests on: 1e-9 against dfMaker, on every shared key.

        No mocks, no reimplementation of the formula in the test: the stage runs over the
        real table and its output is read back.
        """
        seeded_context(context, tag)
        run_stage(context)
        ours = normalized_index(context)
        worst = 0.0
        worst_key = None
        compared = 0
        for key, row in fixture_index(fixture_rows(tag)).items():
            if row["nx"] == "NA" or key not in ours:
                continue
            our_row = ours[key]
            assert our_row["value_status"] == VALUE_NORMALIZED, (
                f"{key}: the reference has a coordinate here, our table says "
                f"{our_row['value_status']!r}")
            error = max(abs(our_row["x_norm"] - float(row["nx"])),
                        abs(our_row["y_norm"] - float(row["ny"])))
            compared += 1
            if error > worst:
                worst, worst_key = error, (key, (our_row["x_norm"], our_row["y_norm"]),
                                           (float(row["nx"]), float(row["ny"])))
        assert compared > 0, f"{tag}: nothing was compared"
        assert worst < TOLERANCE, (
            f"{tag}: worst |error| {worst:g} over {compared} point(s), at {worst_key}")

    @pytest.mark.parametrize("tag", REFERENCE_RUNS)
    def test_our_absence_matches_the_reference_absence(self, context, tag):
        """Where dfMaker says NA, we must say "no coordinate" — and nowhere else.

        The two encodings of absence are not identical by construction: dfMaker masks
        ``x == 0 or y == 0`` per coordinate, while the normalizer that produced our input
        dropped every keypoint with ``score <= 0``. On this corpus they coincide, and this
        test is what makes that coincidence visible rather than assumed: a fixture row that
        is NA *and* has a row in our table must be NA in ours too.
        """
        seeded_context(context, tag)
        run_stage(context)
        ours = normalized_index(context)
        body_keys = {(row["frame_number"], row["detection_index"], row["keypoint_id"])
                     for row in read_table(context.artifact("pose_body")).to_pylist()}
        false_present: list[Any] = []
        for key, row in fixture_index(fixture_rows(tag)).items():
            if row["nx"] != "NA" or key not in ours:
                continue
            if ours[key]["x_norm"] is not None or ours[key]["y_norm"] is not None:
                false_present.append((key, ours[key]["x_norm"], ours[key]["y_norm"]))
        assert not false_present, f"{tag}: we produced coordinates where the reference has " \
                                  f"NA: {false_present[:3]}"
        # The converse, restricted to keys both tables actually hold a keypoint for: our
        # table cannot have a row where the pixel table has none.
        for key, row in fixture_index(fixture_rows(tag)).items():
            if row["nx"] == "NA" and key in body_keys:
                assert key in ours, f"{tag}: the body table has this keypoint but the " \
                                    f"normalised table dropped it"

    @pytest.mark.parametrize("tag", REFERENCE_RUNS)
    def test_the_person_frames_the_reference_could_not_frame_are_named(self, context, tag):
        """Every fixture person-frame with NA everywhere is a non-``basis_ok`` frame here.

        The cnn clip has 21 of its 80 fixture person-frames missing ``MidHip``. dfMaker's
        answer is "no coordinates for anybody in that frame"; ours must be the same answer
        *plus the reason*, because §20.4 forbids encoding that as a dropped row.
        """
        seeded_context(context, tag)
        run_stage(context)
        ours = normalized_index(context)
        by_person_frame: dict[tuple[int, int], list[dict[str, str]]] = {}
        for (frame, person, _point), row in fixture_index(fixture_rows(tag)).items():
            by_person_frame.setdefault((frame, person), []).append(row)

        framed = unframed = 0
        for key, rows in by_person_frame.items():
            our_rows = [row for (frame, person, _point), row in ours.items()
                        if (frame, person) == key]
            if not our_rows:
                continue
            states = {row["basis_state"] for row in our_rows}
            assert len(states) == 1, f"{key}: one person-frame reports {states}"
            state = states.pop()
            all_na = all(row["nx"] == "NA" for row in rows)
            if all_na:
                unframed += 1
                assert state != BASIS_OK, (
                    f"{key}: the reference has no coordinate for any joint in this frame "
                    f"while we claim {state!r}")
                assert all(row["x_norm"] is None for row in our_rows)
                assert all(row["value_status"] == VALUE_BASIS_UNUSABLE for row in our_rows)
            else:
                framed += 1
                assert state == BASIS_OK, (
                    f"{key}: the reference computed coordinates here, so our basis must be "
                    f"usable; got {state!r}")
        assert framed > 0, f"{tag}: nothing in this fixture had a usable basis"
        if tag == "cnn":
            # Measured on the fixture: 21 of its 80 person-frames have no MidHip.
            assert unframed > 0, (
                "the cnn fixture stopped exercising the unframeable person-frame, so this "
                "test would quietly stop covering the state §20.4 asks for")

    def test_the_default_basis_is_the_one_the_fixtures_were_made_with(self, context):
        """A cheap, direct check that indices 8 and 1 are MidHip and Neck.

        The fixture command was ``transformation_coords = c(1, 8, 1, 1)``, so the whole
        comparison above silently tests a different frame if these two names are ever
        renumbered — and the numbers would still look like a body.
        """
        assert BODY_25_KEYPOINT_NAMES[8] == "MidHip"
        assert BODY_25_KEYPOINT_NAMES[1] == "Neck"
        seeded_context(context, "kabc")
        run_stage(context)
        rows = read_table(context.artifact("pose_normalized")).to_pylist()
        assert {row["origin_keypoint_name"] for row in rows} == {"MidHip"}
        assert {row["basis_keypoint_name"] for row in rows} == {"Neck"}

    def test_frame_zero_of_the_kabc_clip_reproduces_the_reference_by_hand(self, context):
        """One hand-worked frame, so a passing suite has a number a human can check.

        Taken from ``dfmaker_0.1.1_kabc_midhip_neck.csv``: origin MidHip (438.724, 274.397),
        basis Neck (449.493, 116.910), so vi = (10.769, -157.487) and the perpendicular is
        (-157.487, -10.769). The reference's own answers, quoted verbatim:
        MidHip -> (0, 0), Neck -> (1, 0), Nose -> (1.26513258448808, 0.030695948251933),
        LShoulder -> (0.977995035086614, -0.405284064507879).
        """
        seeded_context(context, "kabc")
        run_stage(context)
        ours = normalized_index(context)
        reference = {
            8: (0.0, 0.0),
            1: (1.0, 0.0),
            0: (1.26513258448808, 0.030695948251933),
            5: (0.977995035086614, -0.405284064507879),
        }
        for keypoint_id, (expected_x, expected_y) in reference.items():
            row = ours[(0, 0, keypoint_id)]
            assert row["x_norm"] == pytest.approx(expected_x, abs=TOLERANCE), keypoint_id
            assert row["y_norm"] == pytest.approx(expected_y, abs=TOLERANCE), keypoint_id

    def test_the_opposite_perpendicular_is_not_the_reference(self, context):
        """The mutation this feature was most likely to ship, killed by a number.

        ``dfMaker`` uses ``(vi.y, -vi.x)`` in the branch that rotates and ``(-vi.y, vi.x)``
        in its scaling branch. Reflecting the second axis negates every ``y'``: the origin
        and the basis point are unaffected (their offsets are zero and vi), so a spot check
        on the neck passes while the feet are wrong by twice their distance from the torso
        axis. Computed here from the same pixels, so it is a claim about the algebra and not
        about the fixtures: at the fixture's own kabc frame 0, the reflected LShoulder comes
        out at +0.405284 where the reference says -0.405284064507879.
        """
        seeded_context(context, "kabc")
        run_stage(context)
        ours = normalized_index(context)
        origin = ours[(0, 0, 8)]
        body = read_table(context.artifact("pose_body")).to_pylist()
        pixels = {(row["keypoint_id"]): (row["x"], row["y"]) for row in body
                  if row["frame_number"] == 0 and row["detection_index"] == 0}
        ox, oy = pixels[8]
        vix, viy = pixels[1][0] - ox, pixels[1][1] - oy
        # The WRONG sign, deliberately: (-vi.y, vi.x).
        wjx, wjy = -viy, vix
        den = vix * wjy - wjx * viy
        lx, ly = pixels[5][0] - ox, pixels[5][1] - oy
        reflected_y = (vix * ly - lx * viy) / den
        assert reflected_y == pytest.approx(0.405284064507879, abs=1e-9)
        assert ours[(0, 0, 5)]["y_norm"] == pytest.approx(-0.405284064507879, abs=1e-9)
        # The basis point is the row a spot check would pick, and the reflection leaves it
        # exactly right — which is precisely why this bug reaches a committed table.
        assert ours[(0, 0, 1)]["y_norm"] == pytest.approx(0.0, abs=1e-12)
        assert origin["value_status"] == VALUE_NORMALIZED
