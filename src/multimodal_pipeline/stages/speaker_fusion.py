"""`speaker_fusion`: diarization turns × per-frame active speaker (§20.1, T13/T20).

A pure-python stage — no subprocess, no new uv environment. It reads two Parquet tables
that already exist and writes one per selected engine, so the expensive part (TalkNet,
the diarizer) is never re-run here and a change of fusion thresholds costs a re-normalise.

Why it exists as a *new* table instead of an extra column on `speaker_turns`: the fused
verdict is a different measurement, and the existing tables are the input to
`speaker_assignment` and to every dataset already produced. Writing the agreement into
them would silently relabel the corpus. So `speech/speaker_turns.parquet` keeps meaning
"pyannote turns" and `speaker/fusion_pyannote.parquet` is the second opinion beside it.

Both engines are one implementation: `fusion.TURN_TABLES` says which turn table to read,
and T20 (fuse Nemotron turns) is a second call of that function, not a second fusion.
``speaker_fusion.engines`` picks the calls.

Skip semantics are inherited from both upstreams, as §20.1 requires: if the ASD table is
absent, or no selected engine produced a turn table, there is nothing to fuse and the
stage says so instead of emitting an empty table that a consumer would read as "no face
ever matched". Within one run, one engine's absent table skips that engine only.
"""

from __future__ import annotations

import logging
from typing import Any

import pyarrow as pa

from ..exceptions import ValidationError
from ..fusion import (
    AGREEMENT_STATES,
    FUSION_ENGINES,
    TURN_TABLES,
    TurnTableSpec,
    fuse_turn_table,
)
from ..schemas import (
    ACTIVE_SPEAKER_FRAMES_SCHEMA,
    SPEAKER_FUSION_SCHEMA,
    read_table,
    table_columns,
    write_table,
)
from ..validation import check_intervals
from .base import Stage, StageContext

#: Which fused artifact each engine writes. Kept next to the config key that selects it so
#: an engine cannot quietly start overwriting the other engine's file.
FUSION_OUTPUTS: dict[str, str] = {
    "pyannote": "speaker_fusion_pyannote",
    "nemotron": "speaker_fusion_nemotron",
}


class SpeakerFusionStage(Stage):
    """Agree (or visibly disagree) between a diarizer's turns and TalkNet's frames."""

    name = "speaker_fusion"
    # `active_speaker_frames` is the one input the stage cannot do without; the turn
    # tables are declared because the stage really does consume them, but they are read
    # *softly* (see `_turn_rows`) — `ctx.input()` raises on a missing artifact, and an
    # optional engine that never ran is a supported state, not an error. Same reasoning as
    # SpacySourceStage._language_detection reading whisperx_raw off disk.
    inputs = ("active_speaker_frames", "speaker_turns", "speaker_turns_nemotron")
    # Declared for every engine the schema can hold, written only for selected ones.
    # `outputs_present` below narrows the reuse test to what this config asks for, so an
    # unselected engine's absent file never makes a completed fusion look stale.
    outputs = ("speaker_fusion_pyannote", "speaker_fusion_nemotron")
    config_keys = ("speaker_fusion",)

    # ------------------------------------------------------------------ fingerprint

    def config_fingerprint(self, ctx: StageContext) -> dict[str, Any]:
        """Config + the exact input tables, so a re-run of either model invalidates this.

        The turn tables and the frames table are digested rather than merely named: this
        stage's output is a function of their bytes, and a dependency hash that only tracks
        upstream *configuration* would let a `--force-stage diarization` produce a new turn
        table while the fusion kept the verdicts computed against the old one.
        """
        cfg = ctx.config.speaker_fusion
        return {
            "stage": self.name,
            "engines": list(cfg.engines),
            "min_active_ratio": cfg.min_active_ratio,
            "min_face_frames": cfg.min_face_frames,
            "frames_digest": self._digest(ctx, "active_speaker_frames"),
            **{f"{engine}_turns_digest": self._digest(ctx, spec.artifact)
               for engine, spec in TURN_TABLES.items()},
        }

    @staticmethod
    def _digest(ctx: StageContext, artifact: str) -> str | None:
        """SHA256 of an input table, or None when it is absent.

        Absent is recorded as None rather than skipped from the payload: a table that
        appears later must change the fingerprint, and a key that vanishes from the dict
        would hash the same as some other combination of missing inputs.
        """
        from ..stages.metadata import sha256_of

        path = ctx.artifact(artifact)
        cache_key = f"speaker_fusion_digest:{artifact}"
        if cache_key in ctx.scratch:
            return ctx.scratch[cache_key]
        digest = sha256_of(path) if path.is_file() else None
        ctx.scratch[cache_key] = digest
        return digest

    # ------------------------------------------------------------------ enablement

    def enabled(self, ctx: StageContext) -> tuple[bool, str]:
        """Skip with a reason when there is nothing to fuse, never emit an empty table.

        §20.1 inherits both upstreams' skip semantics: a corpus that ran no diarizer, or
        one where TalkNet never ran, has nothing to compare, and a zero-row fused table
        would read as "every turn failed to match" rather than "nothing was measured".
        """
        cfg = ctx.config.speaker_fusion
        if not cfg.enabled:
            return False, "speaker_fusion.enabled = false"
        if not ctx.artifact("active_speaker_frames").is_file():
            return False, ("active speaker frames unavailable (speaker/active_speaker_frames"
                           ".parquet): run the activespeaker stage — there is no per-frame "
                           "face signal to fuse the turns against")
        missing = [spec for spec in self.selected_specs(ctx)
                   if not ctx.artifact(spec.artifact).is_file()]
        if len(missing) == len(cfg.engines):
            listed = ", ".join(f"{spec.engine} ({spec.artifact})" for spec in missing)
            return False, (f"no selected engine produced a turn table: {listed} — run the "
                           f"diarizer first; there is nothing to fuse against the frames")
        return True, ""

    def selected_specs(self, ctx: StageContext) -> tuple[TurnTableSpec, ...]:
        return tuple(TURN_TABLES[name] for name in ctx.config.speaker_fusion.engines)

    def selected_outputs(self, ctx: StageContext) -> tuple[str, ...]:
        """The fused tables this configuration will actually write.

        An engine whose turn table was never produced is skipped at run time, so its fused
        file does not exist and must not be demanded by the reuse test. Requiring it would
        make the stage re-execute on every run forever, or push it to write a table the
        config never asked for.
        """
        return tuple(FUSION_OUTPUTS[spec.engine] for spec in self.fusible_specs(ctx))

    def fusible_specs(self, ctx: StageContext) -> tuple[TurnTableSpec, ...]:
        """Selected engines that really have a turn table on disk right now."""
        return tuple(spec for spec in self.selected_specs(ctx)
                     if ctx.artifact(spec.artifact).is_file())

    def outputs_present(self, ctx: StageContext) -> bool:
        """Only the tables this run would write have to exist for the stage to be reusable.

        The base implementation requires every declared output, which would force an empty
        `speaker/fusion_nemotron.parquet` to exist whenever Nemotron was not selected.
        Staleness in the other direction is handled here too: a fused table whose engine is
        no longer fusible (its turn table went away, or it was deselected) makes the previous
        result unusable rather than acceptable, so the stage re-runs and `execute` deletes it.
        Calling that reusable would keep publishing last month's verdicts with no code path
        left to notice — `validate` only looks at engines that are fusible right now.
        """
        targets = self.selected_outputs(ctx)
        if not all(ctx.paths.artifact(name).exists() for name in targets):
            return False
        return not self._stale_outputs(ctx)

    # ------------------------------------------------------------------ execution

    def prepare(self, ctx: StageContext) -> None:
        # Hard input only: a missing turn table is a per-engine skip decided in execute(),
        # and it was already checked in aggregate by enabled().
        ctx.input("active_speaker_frames")

    def execute(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.config.speaker_fusion
        frames = self._frame_rows(ctx)
        counts: dict[str, int] = {}
        agreements: dict[str, dict[str, int]] = {}
        skipped: dict[str, str] = {}

        for spec in self.selected_specs(ctx):
            turns = self._turn_rows(ctx, spec)
            if turns is None:
                skipped[spec.engine] = (f"{spec.artifact} absent — this engine never "
                                        f"produced turns, so it was not fused")
                ctx.log(f"speaker_fusion: skipping engine {spec.engine}: "
                        f"{spec.artifact}.parquet not found", logging.WARNING)
                continue
            rows = fuse_turn_table(
                video_id=ctx.video_id,
                engine=spec.engine,
                turns=turns,
                frames=frames,
                min_active_ratio=cfg.min_active_ratio,
                min_face_frames=cfg.min_face_frames,
            )
            output = FUSION_OUTPUTS[spec.engine]
            write_table(ctx.artifact(output),
                        pa.Table.from_pylist(rows, schema=SPEAKER_FUSION_SCHEMA),
                        SPEAKER_FUSION_SCHEMA,
                        extra_metadata={"engine": spec.engine,
                                        "video_id": ctx.video_id,
                                        "turn_table": spec.artifact,
                                        "min_active_ratio": cfg.min_active_ratio,
                                        "min_face_frames": cfg.min_face_frames})
            counts[output] = len(rows)
            agreements[spec.engine] = _tally(rows)
            if not rows:
                # A video with no speech is legitimate, so this is a log line and not a
                # validation failure: an empty fused table for a silent clip is the truth.
                ctx.log(f"speaker_fusion: engine {spec.engine} produced 0 turns, so "
                        f"{output} is empty for this video", logging.WARNING)
            else:
                ctx.log("speaker_fusion: " + ", ".join(
                    f"{state}={n}" for state, n in sorted(agreements[spec.engine].items()))
                    + f" for {len(rows)} {spec.engine} turn(s)")

        # A completed run leaves exactly the tables this config and these inputs produce.
        # An engine that used to fuse and no longer can (its turn table went away, or it was
        # deselected) would otherwise leave last month's `fusion_nemotron.parquet` sitting in
        # the dataset, structurally indistinguishable from a fresh one — and `validate` only
        # inspects engines that are fusible right now, so nothing downstream would ever
        # complain about it. A reader would join a stale table to current turns and get
        # confident nonsense. This runs *after* every write succeeded: pruning first would
        # destroy a good dataset when this run goes on to fail.
        #
        # Pruning is deliberately a property of a *completed* run only. `run` skips before
        # `execute` when the operator set `speaker_fusion.enabled: false`, and when the ASD
        # table is not there yet (a fresh dataset mid-build); deleting fused tables in either
        # state would destroy work the operator still wants. The narrower consequence is
        # recorded in the ODD document: if *no* selected engine has a turn table the stage
        # skips and any earlier fused tables stay on disk until the stage completes again.
        pruned = self._prune_stale_outputs(ctx, kept=set(counts))
        ctx.scratch["speaker_fusion"] = {"tables": counts, "agreement": agreements,
                                         "skipped_engines": skipped, "pruned_outputs": pruned}
        return {"tool_version": None, "model_version": None,
                "extra": {"tables": counts, "agreement": agreements,
                          "skipped_engines": skipped, "pruned_outputs": pruned}}

    def _stale_outputs(self, ctx: StageContext) -> list[str]:
        """Declared fused outputs on disk that this configuration would not write."""
        keep = set(self.selected_outputs(ctx))
        return [name for name in self.outputs
                if name not in keep and ctx.artifact(name).is_file()]

    def _prune_stale_outputs(self, ctx: StageContext, *, kept: set[str]) -> dict[str, str]:
        """Delete fused tables this run did not write, and say why each one went.

        Only declared fusion outputs are candidates, so this can never reach a neighbouring
        stage's artifact, and only files that actually exist are touched.
        """
        chosen = {FUSION_OUTPUTS[spec.engine] for spec in self.selected_specs(ctx)}
        pruned: dict[str, str] = {}
        for name in self.outputs:
            if name in kept:
                continue
            path = ctx.artifact(name)
            if not path.is_file():
                continue
            reason = ("engine selected but its turn table is absent" if name in chosen
                      else "engine no longer selected")
            path.unlink()
            pruned[name] = reason
            ctx.log(f"speaker_fusion: removed stale {path.name} ({reason}; keeping only the "
                    f"tables this run could actually compute)", logging.WARNING)
        return pruned

    def _frame_rows(self, ctx: StageContext) -> list[dict[str, Any]]:
        """The ASD frames rows, guarded on the columns the fusion reads.

        A frames table written before `frame_reason`/`is_active_speaker` existed still parses
        and still has rows, and fusing over it yields a table of `no_face_visible` verdicts
        for a video that had faces the whole time. Naming the missing column is the same
        guard the ASD stage applies to its own output, applied to the input this stage reads.
        """
        path = ctx.input("active_speaker_frames")
        columns = set(table_columns(path))
        needed = [field.name for field in ACTIVE_SPEAKER_FRAMES_SCHEMA
                  if field.name in ("timestamp", "track_id", "is_active_speaker",
                                    "talknet_score")
                  and field.name not in columns]
        if needed:
            raise ValidationError(self.name,
                                  [f"input {path.name} missing columns: {', '.join(needed)} "
                                   f"(rerun the activespeaker stage)"])
        return read_table(path).to_pylist()

    @staticmethod
    def _turn_rows(ctx: StageContext, spec: TurnTableSpec) -> list[dict[str, Any]] | None:
        """Read one engine's turn table, or None when that engine never ran."""
        path = ctx.artifact(spec.artifact)
        if not path.is_file():
            return None
        return read_table(path).to_pylist()

    # ------------------------------------------------------------------ validation

    def validate(self, ctx: StageContext) -> dict[str, Any]:
        """Check every fused table this configuration is expected to hold.

        Engines without a turn table are skipped here exactly as `execute` skips them, so
        ``validate`` describes the same dataset the stage writes instead of failing over an
        artifact that was never requested.
        """
        summary: dict[str, Any] = {}
        for spec in self.fusible_specs(ctx):
            output = FUSION_OUTPUTS[spec.engine]
            path = ctx.artifact(output)
            if not path.is_file():
                raise ValidationError(self.name, [f"{output} missing ({path.name})"])
            columns = set(table_columns(path))
            missing = [field.name for field in SPEAKER_FUSION_SCHEMA
                       if field.name not in columns]
            if missing:
                # A table written before this schema grew must say which column it lacks,
                # which is also which stage to rerun. Reading rows first turned this into a
                # bare KeyError, recorded as a crash instead of a one-line diagnosis (the
                # lesson ActiveSpeakerStage.validate already carries).
                raise ValidationError(self.name,
                                      [f"{path.name} missing columns: {', '.join(missing)}"])
            rows = read_table(path).to_pylist()
            self._check_rows(ctx, rows, engine=spec.engine)
            summary[output] = {
                "rows": len(rows),
                "agreement": _tally(rows),
                "engines": sorted({str(row["engine"]) for row in rows}),
            }
            if not rows:
                ctx.log(f"{path.name}: 0 rows for engine {spec.engine} (a video with no "
                        f"speech, or an engine that emitted no turns)", logging.WARNING)
        return summary

    def _check_rows(self, ctx: StageContext, rows: list[dict[str, Any]], *, engine: str) -> None:
        problems: list[str] = []
        # The closed vocabulary is the whole contract: a reader switches on these five
        # strings and cannot handle a sixth.
        allowed_engines = set(FUSION_ENGINES)
        for index, row in enumerate(rows):
            row_engine = row["engine"]
            if row_engine not in allowed_engines:
                problems.append(f"row {index} names engine {row_engine!r}, not one of "
                                f"{', '.join(sorted(allowed_engines))}")
            elif row_engine != engine:
                # Two engines in one file would put two speaker-id namespaces in one
                # speaker_id column — precisely the join this layout exists to prevent.
                problems.append(f"row {index} is engine {row_engine!r} in the {engine} table")
            agreement = row["agreement"]
            if agreement not in AGREEMENT_STATES:
                problems.append(f"row {index} has an unrecognised agreement "
                                f"{agreement!r} (expected one of: {', '.join(AGREEMENT_STATES)})")
            active = _int(row["face_active_frames"])
            face_frames = _int(row["face_frames_in_turn"])
            frames = _int(row["frames_in_turn"])
            # The three counts are a nested set, and the nesting is what makes the
            # no-face/no-measurement distinction readable. A violated ordering means the
            # window arithmetic is wrong, so no verdict in this table can be trusted.
            if not active <= face_frames <= frames:
                problems.append(
                    f"row {index} counts are not nested: face_active_frames={active} "
                    f"face_frames_in_turn={face_frames} frames_in_turn={frames}")
            if row["face_track_id"] is not None and active == 0:
                # A winning track is chosen from tracks with at least one active frame.
                problems.append(f"row {index} names track {row['face_track_id']} as the "
                                f"winner while reporting 0 active frames")
            if agreement == "no_frames_measured" and frames != 0:
                problems.append(f"row {index} claims no_frames_measured but counts "
                                f"{frames} frames in the turn")
            if agreement == "no_face_visible" and not (frames > 0 and face_frames == 0):
                problems.append(f"row {index} claims no_face_visible with "
                                f"face_frames_in_turn={face_frames} of {frames}")
            if not row["agreement_detail"]:
                problems.append(f"row {index} has an empty agreement_detail")
        if len({row["turn_id"] for row in rows}) != len(rows):
            problems.append("turn_id is not unique inside one engine's fused table")
        check_intervals([row["start_time"] for row in rows],
                        [row["end_time"] for row in rows],
                        stage=self.name, label=f"{engine} fused turn",
                        max_time=self._duration(ctx))
        if problems:
            raise ValidationError(self.name, problems[:20])

    @staticmethod
    def _duration(ctx: StageContext) -> float | None:
        from ..artifacts import read_json

        path = ctx.artifact("metadata")
        if not path.is_file():
            return None
        try:
            return read_json(path).get("duration_seconds")
        except Exception:  # noqa: BLE001 - a missing duration only narrows the check
            return None


def _tally(rows: list[dict[str, Any]]) -> dict[str, int]:
    tally: dict[str, int] = {}
    for row in rows:
        state = str(row["agreement"])
        tally[state] = tally.get(state, 0) + 1
    return dict(sorted(tally.items()))


def _int(value: Any) -> int:
    return int(value) if value is not None else 0
