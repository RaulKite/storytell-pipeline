"""``pose_normalized``: BODY_25 keypoints in a body-centred frame (§20.4, T14).

A pure-python stage — no subprocess, no new uv environment, nothing heavy. It reads one
Parquet table that already exists and writes one beside it, so changing the basis triple
costs a re-normalise (seconds) rather than another OpenPose pass (minutes per video, and
the slowest stage in the pipeline).

Why it is a *new table*: ``pose/body.parquet`` carries pixels, and pixels are the
measured quantity. Every dataset already produced joins on them, so re-expressing them
in place would silently redefine the corpus — the same reasoning that keeps
``speaker_fusion`` out of ``speaker_turns``. The normalised table is a derived
interpretation, and the pipeline's rule is that raw survives so a later decision can be
recomputed without re-running the tool.

Skip semantics are inherited from ``openpose`` alone, as §20.4 requires. The stage reads
``pose/body.parquet`` and nothing else — no audio, no transcript, no diarizer — so a
corpus that ran no OpenPose has nothing to normalise, and the stage says so instead of
emitting an empty table a consumer would read as "nobody was on screen".

Absence is a named state at two levels, which is the ``face_status`` lesson from §17
applied to pose:

* per **person-frame**, ``basis_state`` says whether a frame of reference exists at all
  (``basis_ok``) and, when it does not, whether the cause is a joint that was never
  measured (``basis_missing_joint``), two joints that coincide (``basis_degenerate``), or
  coordinates that are numbers but not pixel positions (``basis_non_finite``);
* per **keypoint**, ``value_status`` says whether this joint got coordinates, lost its
  own coordinate, or had nowhere to be put.

Every (frame, detection_index) that has a keypoint in the body table still has rows here,
and no absence is written as a dropped row. A zero *coordinate* is not an absence marker
either: the origin joint legitimately normalises to (0, 0) and the basis joint to (1, 0),
which the reference itself emits. Absence is carried by the two state columns and by
nulls, never by a number.

The maths lives in ``pose_normalize.py`` so the comparison against the R reference can
drive it without a dataset; this file is the reader, the writer, and the guarantees
around them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

from ..exceptions import StageError, ValidationError
from ..pose_normalize import (
    BASIS_OK,
    BASIS_STATES,
    VALUE_BASIS_UNUSABLE,
    VALUE_STATES,
    normalized_rows,
    person_frame_bases,
)
from ..schemas import (
    BODY_25_KEYPOINT_NAMES,
    ChunkedParquetWriter,
    POSE_NORMALIZED_SCHEMA,
    iter_rows,
    read_table,
    table_columns,
    table_rows,
)
from .base import Stage, StageContext, StageOutcome

#: Rows buffered per Parquet row group. One frame of BODY_25 is ~25 rows per person, so
#: this keeps thousands of frames resident and a small constant of memory regardless of
#: video length — the same reason openpose's writer is chunked.
ROWS_PER_GROUP = 100_000

#: The columns the transform reads. Projecting them keeps the streaming pass cheap on a
#: table with millions of rows, and names the stage's real dependency on the body table.
INPUT_COLUMNS = ("frame_number", "timestamp", "detection_index", "keypoint_id",
                 "keypoint_name", "x", "y")

#: The columns `validate` reads, and the only eight it reads. Projecting them keeps
#: validation cheap on a million-row table: the checks need two ids, two states, two
#: coordinates, the video id and the detail string, and nothing else.
VALIDATE_COLUMNS = ("video_id", "keypoint_id", "keypoint_name", "basis_state",
                    "basis_detail", "x_norm", "y_norm", "value_status")

#: The three columns that record *which* frame the numbers are expressed in. Reading them
#: back is how a table can be caught answering a different question than the config asked.
FRAME_COLUMNS = ("origin_keypoint_name", "basis_keypoint_name", "second_axis")


class PoseNormalizedStage(Stage):
    """Re-express one pose table in a body-centred frame, or say exactly why not."""

    name = "pose_normalized"
    # The only input, on purpose: §20.4 makes this a downstream normalisation of OpenPose
    # output, not a new detector and not a consumer of the audio branch.
    inputs = ("pose_body",)
    outputs = ("pose_normalized",)
    config_keys = ("pose_normalized",)

    # ------------------------------------------------------------------ fingerprint

    def config_fingerprint(self, ctx: StageContext) -> dict[str, Any]:
        """The basis triple plus the bytes of the table it is applied to.

        The triple is here because it changes every number in the output; recording it is
        what makes "which frame is this table in?" answerable from the state file rather
        than from a reader's memory (§20.4 asks for exactly this).

        ``body_digest`` is digested rather than merely named because the stage's output is
        a function of those bytes, and a dependency hash that tracks only upstream
        *configuration* would let a re-run OpenPose leave a table of coordinates computed
        against the keypoints that used to be there.
        """
        cfg = ctx.config.pose_normalized
        return {
            "stage": self.name,
            # dfMaker's own `transformation_coords = c(type, origin, i, j)`, in names.
            "transformation": [cfg.origin_keypoint, cfg.basis_keypoint, cfg.second_axis],
            "body_digest": self._digest(ctx),
        }

    @staticmethod
    def _digest(ctx: StageContext) -> str | None:
        """SHA256 of ``pose/body.parquet``, or None when it is absent.

        Absent is recorded as None rather than dropped from the payload: a table that
        appears later must change the fingerprint, and a key that vanishes from the dict
        would hash the same as some other combination of missing inputs.
        """
        from ..stages.metadata import sha256_of

        path = ctx.artifact("pose_body")
        cache_key = "pose_normalized_digest:pose_body"
        if cache_key in ctx.scratch:
            return ctx.scratch[cache_key]
        digest = sha256_of(path) if path.is_file() else None
        ctx.scratch[cache_key] = digest
        return digest

    # ------------------------------------------------------------------ enablement

    def enabled(self, ctx: StageContext) -> tuple[bool, str]:
        """Skip with a reason when there is nothing to normalise, never emit an empty table.

        The ways the input can be missing get distinct messages because they have distinct
        fixes. All of them inherit openpose's skip semantics — the stage never fails a
        video because the operator does not want pose data — and all are decided from the
        filesystem rather than from the openpose config alone, so a dataset whose pose was
        deleted is described accurately instead of excused by a flag.
        """
        cfg = ctx.config.pose_normalized
        if not cfg.enabled:
            return False, "pose_normalized.enabled = false"
        if ctx.artifact("pose_body").is_file():
            return True, ""
        if not ctx.config.openpose.enabled:
            return False, ("pose/body.parquet unavailable: openpose.enabled = false, so no "
                           "keypoints were extracted and there is nothing to normalise")
        if not ctx.config.openpose.body.enabled:
            return False, ("pose/body.parquet unavailable: openpose.body.enabled = false "
                           "publishes no body keypoints")
        return False, ("pose/body.parquet unavailable: run the openpose stage first — "
                       "there are no pixel keypoints to re-express")

    # ------------------------------------------------------------------- staleness

    def wanted_frame(self, ctx: StageContext) -> dict[str, str]:
        """The frame this configuration asks the table to be written in."""
        cfg = ctx.config.pose_normalized
        return {"origin_keypoint_name": cfg.origin_keypoint,
                "basis_keypoint_name": cfg.basis_keypoint,
                "second_axis": cfg.second_axis}

    def recorded_frame(self, path: Path) -> dict[str, str] | None:
        """The frame the table on disk *says* it was written in, or None if unknowable.

        Costs one column-projected read and never trusts a filename. None means "cannot
        tell" — unreadable, or empty — and every caller treats it as "delete nothing":
        an empty table is the honest result for a video with no people, and a broken one
        is ``validate``'s to report.
        """
        if not path.is_file():
            return None
        try:
            rows = read_table(path, columns=list(FRAME_COLUMNS)).to_pylist()
        except Exception:  # noqa: BLE001 - a broken file is validate()'s problem, not ours
            return None
        if not rows:
            return None
        # Heterogeneous values here would be a corrupted table; validate's vocabulary
        # checks are where that is reported, so the first row's answer is enough to
        # decide "same frame as the config?" and never enough to delete on.
        first = rows[0]
        return {name: str(first[name]) for name in FRAME_COLUMNS}

    def _foreign_frame_outputs(self, ctx: StageContext) -> dict[str, str]:
        """Declared outputs whose recorded frame is not the frame the config asks for.

        This is *positive* evidence in the sense ``speaker_fusion`` uses for a
        deselection: the file states one basis triple and the configuration states
        another. The absence of an input is not such evidence and never appears here.
        """
        wanted = self.wanted_frame(ctx)
        found: dict[str, str] = {}
        for name in self.outputs:
            path = ctx.artifact(name)
            recorded = self.recorded_frame(path)
            if recorded is not None and recorded != wanted:
                listed = ", ".join(f"{key}={recorded[key]}" for key in sorted(recorded))
                asks = ", ".join(f"{key}={wanted[key]}" for key in sorted(wanted))
                found[name] = f"written in a different basis frame ({listed}); this " \
                              f"configuration asks for {asks}"
        return found

    def outputs_present(self, ctx: StageContext) -> bool:
        """Only reusable when the file is there *and* is in the frame this config wants.

        The base implementation checks existence, which would let a table written for one
        basis triple be reused after the triple changed without a state change — a reset
        ``status.json``, or a table copied in from another dataset. Both leave coordinates
        that are plausible, validate structurally, and answer a different question than
        the one asked. Refusing reuse sends the stage back through ``execute``, which
        rewrites the file from the current input.
        """
        for name in self.outputs:
            path = ctx.paths.get(name)
            if path is None or not path.exists():
                return False
        return not self._foreign_frame_outputs(ctx)

    # ------------------------------------------------------------------ execution

    def prepare(self, ctx: StageContext) -> None:
        # Hard input: `enabled()` already refused a run without it, so a raise here means
        # the file vanished between the two checks, which is worth failing loudly for.
        ctx.input("pose_body")

    def run(self, ctx: StageContext) -> StageOutcome:
        """Prune a foreign-frame table even when the run skips.

        Pruning was a property of ``execute``, so a skip — pose deleted, OpenPose switched
        off — left a table computed for a *different* basis triple sitting in the dataset
        indefinitely. Nothing else ever visits that file: the reuse test does not send the
        stage back through ``execute`` for a different reason, and ``validate`` reporting
        the mismatch is a report, not a fix.

        The gate is the config flag, not the wording of the skip reason, for the reason
        ``speaker_fusion`` records: ``pose_normalized.enabled: false`` is the operator
        pausing this stage over a table they still want, and coupling deletion to a reason
        string would let a reworded ``enabled()`` message decide whose files get deleted.
        """
        outcome = super().run(ctx)
        if outcome.status == "skipped" and ctx.config.pose_normalized.enabled:
            # Only a table that positively states a foreign frame is deleted. A missing
            # input proves nothing about the file that is already there, and the set of
            # outputs this run could produce is empty — which is never a licence to prune.
            self._prune_stale_outputs(ctx, self._foreign_frame_outputs(ctx))
        return outcome

    def execute(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.config.pose_normalized
        # Resolve and check everything before an output file is opened: a body table
        # missing a column, or configured with a triple that cannot be built, must fail
        # with the file untouched rather than half-published.
        self._check_input(ctx)
        plan = {
            "origin_id": BODY_25_KEYPOINT_NAMES.index(cfg.origin_keypoint),
            "basis_id": BODY_25_KEYPOINT_NAMES.index(cfg.basis_keypoint),
            "origin_name": cfg.origin_keypoint,
            "basis_name": cfg.basis_keypoint,
            "second_axis": cfg.second_axis,
        }
        path = ctx.artifact("pose_normalized")
        writer = ChunkedParquetWriter(
            path, POSE_NORMALIZED_SCHEMA, rows_per_group=ROWS_PER_GROUP,
            extra_metadata={
                "video_id": ctx.video_id,
                "coordinate_space": "body-centred basis coordinates (not pixels)",
                # Both names, not one index: a reader of the file alone must be able to
                # tell which question its numbers answer.
                "origin_keypoint": cfg.origin_keypoint,
                "basis_keypoint": cfg.basis_keypoint,
                "second_axis": cfg.second_axis,
                "reference": "dfMaker (CRAN multimolang 0.1.1), fast_scaling = FALSE",
            })
        states: dict[str, int] = {}
        for row in self._normalized_rows(ctx, plan):
            states[row["basis_state"]] = states.get(row["basis_state"], 0) + 1
            writer.add(row)
        rows_written = writer.close()

        # No pruning here: this stage has exactly one declared output and a completed run
        # rewrites it, so there is nothing left behind to remove. The case that *does*
        # leave a foreign table is a run that never got here — pose deleted, openpose
        # switched off — and `run` handles that one on the skip path.
        summary = {"rows": rows_written, "basis_states": dict(sorted(states.items())),
                   "transformation": [cfg.origin_keypoint, cfg.basis_keypoint,
                                      cfg.second_axis],
                   "pruned_outputs": {}}
        ctx.scratch["pose_normalized"] = summary
        ctx.log(f"pose_normalized: {rows_written} row(s) -> {path.name} in the "
                f"{cfg.origin_keypoint}->{cfg.basis_keypoint} frame "
                f"({', '.join(f'{state}={count}' for state, count in sorted(states.items()))})")
        return {"tool_version": None, "model_version": None, "extra": summary}

    def _prune_stale_outputs(self, ctx: StageContext, evidence: dict[str, str]) -> dict[str, str]:
        """Delete declared outputs this configuration positively cannot justify.

        Only this stage's declared outputs are candidates, so the loop can never reach a
        neighbouring stage's artifact, and only files that exist are touched.

        ``evidence`` is the caller's proof, per artifact. It is required rather than
        inferred from "this run wrote nothing", which is the trap ``speaker_fusion``
        records: an absent input is the absence of evidence, not evidence of staleness, and
        deleting on it would turn a half-built dataset into lost work. Here the only
        accepted proof is a file that states a basis frame different from the one the
        configuration asks for.
        """
        pruned: dict[str, str] = {}
        for name in self.outputs:
            reason = evidence.get(name)
            if reason is None:
                continue
            path = ctx.artifact(name)
            if not path.is_file():
                continue
            path.unlink()
            pruned[name] = reason
            ctx.log(f"pose_normalized: removed stale {path.name} ({reason}); the table is "
                    f"recomputable from pose/body.parquet in seconds", logging.WARNING)
        return pruned

    # ------------------------------------------------------------------ streaming

    def _check_input(self, ctx: StageContext) -> None:
        """Refuse a body table that predates a column this stage reads.

        Such a table parses, has rows, and would yield a normalised table that is mostly
        nulls and validates structurally. Naming the column is the whole diagnosis, and it
        is the guard the openpose stage applies to its own output, applied to the input
        this stage reads.
        """
        path = ctx.input("pose_body")
        missing = [name for name in INPUT_COLUMNS if name not in set(table_columns(path))]
        if missing:
            raise ValidationError(self.name,
                                  [f"input {path.name} missing columns: {', '.join(missing)} "
                                   f"(rerun the openpose stage)"])

    def _normalized_rows(self, ctx: StageContext, plan: dict[str, Any]
                         ) -> Iterator[dict[str, Any]]:
        """Stream the body table one frame at a time, so memory does not grow with length.

        A basis needs one person's two joints, so a row-at-a-time pass cannot compute it —
        but buffering the whole table is not acceptable either: a 4-hour 50 fps recording
        is tens of millions of keypoint rows. So the unit of work is one frame: its rows
        are read, its person-frames get their bases, and the normalised rows go out before
        the next frame is touched. Resident memory is a function of people-per-frame.

        That requires the input's person-frames to be contiguous, which is how the openpose
        normalizer writes them (it streams ``pose/raw`` in frame order). A table that
        revisits a frame is refused with the reason rather than quietly normalised against
        a half-seen person-frame.
        """
        buffer: list[dict[str, Any]] = []
        current: int | None = None
        last_flushed: int | None = None

        def flush(rows: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
            bases = person_frame_bases(
                rows, origin_id=plan["origin_id"], basis_id=plan["basis_id"],
                origin_name=plan["origin_name"], basis_name=plan["basis_name"])
            return normalized_rows(rows, bases, video_id=ctx.video_id,
                                   origin_name=plan["origin_name"],
                                   basis_name=plan["basis_name"],
                                   second_axis=plan["second_axis"])

        for row in iter_rows(ctx.input("pose_body"), INPUT_COLUMNS):
            frame = int(row["frame_number"])
            if current is not None and frame != current:
                if last_flushed is not None and frame <= last_flushed:
                    raise StageError(
                        f"{self.name}: pose/body.parquet revisits frame {frame} after it was "
                        f"already normalised, so the person-frames are not contiguous. The "
                        f"basis of a person-frame needs both of its joints in the same read, "
                        f"so a non-contiguous table cannot be streamed; rerun the openpose "
                        f"stage to rewrite pose/body.parquet in frame order.")
                yield from flush(buffer)
                last_flushed = current
                buffer = []
            current = frame
            buffer.append(row)
        if buffer:
            yield from flush(buffer)

    # ------------------------------------------------------------------ validation

    def validate(self, ctx: StageContext) -> dict[str, Any]:
        """Check the table's shape, its vocabularies, and that it answers this config.

        Row-level checks say the rows that are here are honest; the comparison against the
        body table says *all* of them are. A normalised table that lost rows — truncated,
        or written from a pose table that has since been re-extracted — would otherwise
        validate clean while a consumer's join silently lost a person.
        """
        path = ctx.artifact("pose_normalized")
        if not path.is_file():
            raise ValidationError(self.name, [f"pose_normalized missing ({path.name})"])
        columns = set(table_columns(path))
        missing = [field.name for field in POSE_NORMALIZED_SCHEMA if field.name not in columns]
        if missing:
            raise ValidationError(self.name,
                                  [f"{path.name} missing columns: {', '.join(missing)}"])

        wanted = self.wanted_frame(ctx)
        recorded = self.recorded_frame(path)
        if recorded is not None and recorded != wanted:
            # The table disagrees with the configuration about what its numbers mean, so
            # every coordinate in it answers a different question than the one asked.
            raise ValidationError(self.name, [
                f"{path.name} is written in the basis frame "
                + ", ".join(f"{key}={recorded[key]}" for key in sorted(recorded))
                + " while the configuration asks for "
                + ", ".join(f"{key}={wanted[key]}" for key in sorted(wanted))
                + " — rerun pose_normalized"])

        rows_read = 0
        problems: list[str] = []
        basis_states: dict[str, int] = {}
        value_states: dict[str, int] = {}
        video_ids: set[str] = set()
        # Streamed rather than read whole: this runs inside the reuse test on every
        # `status`/`run`, and a 4-hour recording's normalised table is tens of millions of
        # rows. Only the counters and a capped problem list are kept.
        for row in iter_rows(path, VALIDATE_COLUMNS):
            rows_read += 1
            video_ids.add(str(row["video_id"]))
            _tally_into(basis_states, row["basis_state"])
            _tally_into(value_states, row["value_status"])
            if len(problems) < 20:
                problem = self._check_row(row)
                if problem is not None:
                    problems.append(f"row {rows_read - 1}: {problem}")
        if problems:
            raise ValidationError(self.name, problems)
        if rows_read and video_ids != {ctx.video_id}:
            # One video per dataset directory: a second id means the file was copied in from
            # elsewhere, and its keypoints describe another body.
            raise ValidationError(self.name,
                                  [f"rows belong to video(s) {sorted(video_ids)}, expected "
                                   f"{ctx.video_id!r}"])

        summary: dict[str, Any] = {"rows": rows_read,
                                   "basis_states": dict(sorted(basis_states.items())),
                                   "value_states": dict(sorted(value_states.items())),
                                   "frame": wanted}
        body = ctx.artifact("pose_body")
        if body.is_file():
            # One normalised row per body row, by construction. A differing count means
            # this table was written against a different pose table than the one on disk.
            expected = table_rows(body)
            if expected != rows_read:
                raise ValidationError(self.name, [
                    f"pose_normalized ({path.name}) has {rows_read} row(s) while its input "
                    f"pose_body has {expected} keypoint row(s): {path.name} is stale — it "
                    f"was not written from these keypoints (pose/body.parquet reads fine, so "
                    f"this is staleness, not corruption); rerun pose_normalized to recompute "
                    f"it from pose_body"])
        else:
            # Not a failure: the stage skips without its input, and validate must describe
            # the dataset rather than fail over an artifact that was never requested.
            # Reported so a reader knows the row count above is unchecked.
            summary["row_count_checked"] = False
        if not rows_read:
            ctx.log(f"{path.name}: 0 rows (the video contains no person, or pose_body was "
                    f"empty)", logging.WARNING)
        return summary

    @staticmethod
    def _check_row(row: dict[str, Any]) -> str | None:
        """The first problem with one row, or None.

        One problem per row is enough: the row index is in the message, and a table whose
        first hundred rows are all wrong needs the first diagnosis, not a hundred copies of
        it. That also lets the caller cap the list and keep streaming memory flat.
        """
        # The two vocabularies are the whole contract: a reader switches on these
        # strings and cannot handle a sixth value.
        basis_state = row["basis_state"]
        value_status = row["value_status"]
        if basis_state not in BASIS_STATES:
            return (f"unrecognised basis_state {basis_state!r} (expected one of: "
                    f"{', '.join(BASIS_STATES)})")
        if value_status not in VALUE_STATES:
            return (f"unrecognised value_status {value_status!r} (expected one of: "
                    f"{', '.join(VALUE_STATES)})")
        if not row["basis_detail"]:
            return "empty basis_detail"
        placed = row["x_norm"] is not None and row["y_norm"] is not None
        if placed and value_status != "normalized":
            return f"carries coordinates but claims {value_status!r}"
        if placed and basis_state != "basis_ok":
            # A coordinate cannot exist in a frame the table says was never built.
            return f"has coordinates while basis_state is {basis_state!r}"
        if not placed and value_status == "normalized":
            return "claims normalized with null coordinates"
        if value_status == VALUE_BASIS_UNUSABLE and basis_state == BASIS_OK:
            # The mirror of the check above, and the reason the two vocabularies are
            # disjoint: "there was no frame" and "the frame was fine, this joint was not
            # measured" cannot both be true of one row.
            return "claims basis_unusable in a person-frame whose basis_state is basis_ok"
        keypoint_id = row["keypoint_id"]
        name = row["keypoint_name"]
        # Background (id 25) is excluded by range: it is a legal BODY_25 name that the body
        # table never carries, so a normalised table containing one was not written from
        # this input.
        if keypoint_id is None or not 0 <= int(keypoint_id) < len(BODY_25_KEYPOINT_NAMES) - 1:
            return (f"keypoint_id {keypoint_id} is outside the joints pose/body.parquet can "
                    f"hold (0..{len(BODY_25_KEYPOINT_NAMES) - 2}; Background is a filler "
                    f"channel and is never written)")
        if BODY_25_KEYPOINT_NAMES[int(keypoint_id)] != name:
            # id and name must agree, or a consumer that selects by one gets the joint the
            # other names.
            return (f"pairs keypoint_id {keypoint_id} with {name!r}, not "
                    f"{BODY_25_KEYPOINT_NAMES[int(keypoint_id)]!r}")
        return None


def _tally_into(counts: dict[str, int], state: Any) -> None:
    """Count one row into a state tally, keeping the validation summary flat.

    Reported rather than asserted: the summary is what lands in the stage record, and a
    reader of `status.json` should see that a video produced 100% missing-basis rows
    without opening the table.
    """
    key = str(state)
    counts[key] = counts.get(key, 0) + 1
