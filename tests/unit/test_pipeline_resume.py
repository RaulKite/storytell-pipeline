"""Resume, reuse and invalidation: the contract that makes an interrupted run safe.

Stages are replaced by instrumented fakes so each test controls exactly which
stage fails, changes or is interrupted, and can count how many times work ran.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from multimodal_pipeline.artifacts import ArtifactRegistry, VideoPaths
from multimodal_pipeline.orchestrator import VideoRunner
from multimodal_pipeline.state import STATUS_COMPLETED, STATUS_FAILED, STATUS_SKIPPED
from multimodal_pipeline.stages.base import STAGE_ORDER, Stage, StageContext, StageError


class FakeStage(Stage):
    """Writes its own output file and records how often it actually executed."""

    #: shared across instances so a test can count executions per stage name
    executions: dict[str, int] = {}
    #: stage names whose execute() should raise
    fail_on: set[str] = set()
    #: optional extra config keys folded into the fingerprint
    extra_config: dict[str, Any] = {}

    def __init__(self, name: str, depends_on: tuple[str, ...] = ()) -> None:
        self.name = name
        self.inputs = tuple(ARTIFACT_OF[dep] for dep in depends_on)
        self.outputs = (ARTIFACT_OF[name],)

    def config_fingerprint(self, ctx: StageContext) -> dict[str, Any]:
        return {"stage": self.name, **type(self).extra_config.get(self.name, {})}

    def prepare(self, ctx: StageContext) -> None:
        for artifact_name in self.inputs:
            ctx.input(artifact_name)

    def execute(self, ctx: StageContext) -> dict[str, Any]:
        if self.name in type(self).fail_on:
            raise StageError(f"{self.name} exploded")
        type(self).executions[self.name] = type(self).executions.get(self.name, 0) + 1
        path = ctx.artifact(ARTIFACT_OF[self.name])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"stage": self.name, "run": type(self).executions[self.name]}),
                        encoding="utf-8")
        return {"tool_version": "fake 1.0"}

    def validate(self, ctx: StageContext) -> dict[str, Any]:
        path = ctx.artifact(ARTIFACT_OF[self.name])
        if not path.is_file():
            from multimodal_pipeline.exceptions import ValidationError

            raise ValidationError(self.name, [f"missing {path.name}"])
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            from multimodal_pipeline.exceptions import ValidationError

            raise ValidationError(self.name, [f"unreadable {path.name}: {exc}"]) from exc
        return {"ok": True}


#: The fake DAG mirrors the real dependency shape closely enough to exercise
#: propagation: a branch (pose) that does not depend on the speech chain.
FAKE_DAG = [
    ("metadata", ()),
    ("audio", ("metadata",)),
    ("whisperx", ("audio",)),
    ("pose", ("metadata",)),
    ("assign", ("whisperx",)),
    ("report", ("assign", "pose")),
]

#: Fake stages reuse real layout slots so ctx.artifact()/outputs_present work
#: against the production artifact registry rather than a parallel one.
ARTIFACT_OF = {
    "metadata": "metadata",
    "audio": "audio",
    "whisperx": "speech_segments",
    "pose": "pose_body",
    "assign": "speech_words",
    "report": "acoustic_frames",
}


@pytest.fixture(autouse=True)
def fake_dag(monkeypatch, tmp_path):
    FakeStage.executions = {}
    FakeStage.fail_on = set()
    FakeStage.extra_config = {}

    stages = [FakeStage(name, deps) for name, deps in FAKE_DAG]
    names = [s.name for s in stages]
    dependencies = dict(FAKE_DAG)

    monkeypatch.setattr("multimodal_pipeline.stages.base.STAGE_ORDER", tuple(names))
    monkeypatch.setattr("multimodal_pipeline.stages.base.STAGE_DEPENDENCIES", dependencies)
    monkeypatch.setattr("multimodal_pipeline.orchestrator.STAGE_ORDER", tuple(names))
    monkeypatch.setattr("multimodal_pipeline.orchestrator.build_stages", lambda: list(stages))
    monkeypatch.setattr("multimodal_pipeline.stages.base.STAGE_ORDER", tuple(names))
    yield stages


def runner_for(config, monkeypatch, tmp_path: Path) -> VideoRunner:
    from multimodal_pipeline.discovery import VideoSource

    source = VideoSource(path=config.input.directory / "clip.mp4", relative_path=Path("clip.mp4"),
                         video_id="clip")
    source.path.write_bytes(b"video-bytes")
    # No fake stage writes manifest.json, so the runner's closing summary refresh
    # finds nothing to rewrite and stays out of these tests.
    return VideoRunner(config, source, tools={"schema_version": "1.0"})


def statuses(runner: VideoRunner) -> dict[str, str]:
    state = type(runner.state).load(runner.paths, runner.source.video_id)
    return {name: state.status_of(name) for name, _ in FAKE_DAG}


class TestFirstRun:
    def test_every_stage_runs_once(self, config, monkeypatch) -> None:
        runner = runner_for(config, monkeypatch, Path(""))
        result = runner.run()
        assert result.status == "completed"
        assert FakeStage.executions == {name: 1 for name, _ in FAKE_DAG}

    def test_outputs_land_on_disk(self, config, monkeypatch) -> None:
        runner = runner_for(config, monkeypatch, Path(""))
        runner.run()
        for name, _ in FAKE_DAG:
            assert runner.paths.artifact(ARTIFACT_OF[name]).is_file()

    def test_state_records_hashes_and_versions(self, config, monkeypatch) -> None:
        runner = runner_for(config, monkeypatch, Path(""))
        runner.run()
        record = runner.state.stage("whisperx")
        assert record.status == STATUS_COMPLETED
        assert record.config_hash and record.dependency_hash
        assert record.tool_version == "fake 1.0"
        assert record.started_at and record.completed_at


class TestResume:
    def test_completed_run_resumes_with_zero_work(self, config, monkeypatch) -> None:
        runner = runner_for(config, monkeypatch, Path(""))
        runner.run()
        before = dict(FakeStage.executions)
        resumed = VideoRunner(runner.config, runner.source, tools={"schema_version": "1.0"})
        resumed.run()
        assert FakeStage.executions == before
        # Reuse is counted on the process that resumed, from state on disk.
        reused = {name: record.reuse_count for name, record in resumed.state.stages.items()}
        assert all(reused[name] >= 1 for name, _ in FAKE_DAG), reused

    def test_interruption_resumes_from_the_failed_stage(self, config, monkeypatch) -> None:
        first = runner_for(config, monkeypatch, Path(""))
        FakeStage.fail_on = {"assign"}
        first.run()
        mid = statuses(first)
        assert mid["assign"] == STATUS_FAILED
        assert mid["whisperx"] == STATUS_COMPLETED
        assert mid["pose"] == STATUS_COMPLETED  # independent branch still finished
        assert mid["report"] == STATUS_SKIPPED
        executed_after_failure = dict(FakeStage.executions)

        FakeStage.fail_on = set()
        second = VideoRunner(first.config, first.source, tools={"schema_version": "1.0"})
        second.run()
        # Nothing upstream of the failure was redone.
        assert FakeStage.executions["metadata"] == executed_after_failure["metadata"]
        assert FakeStage.executions["whisperx"] == executed_after_failure["whisperx"]
        # A stage that raises never reaches its counter, so the failed attempt
        # counts as zero executions and the retry is the first recorded one.
        assert FakeStage.executions["assign"] == executed_after_failure.get("assign", 0) + 1
        assert statuses(second)["report"] == STATUS_COMPLETED

    def test_resume_reports_reused_not_done(self, config, monkeypatch) -> None:
        runner = runner_for(config, monkeypatch, Path(""))
        runner.run()
        seen: list[tuple[str, str, str]] = []
        runner.progress = lambda name, status, note: seen.append((name, status, note))
        FakeStage.executions.clear()
        runner.run()
        assert all(note == "reused" for _name, _status, note in seen)
        assert FakeStage.executions == {}


class TestFailurePropagation:
    def test_only_dependants_are_blocked(self, config, monkeypatch) -> None:
        FakeStage.fail_on = {"whisperx"}
        runner = runner_for(config, monkeypatch, Path(""))
        runner.run()
        result = statuses(runner)
        assert result["whisperx"] == STATUS_FAILED
        assert result["assign"] == STATUS_SKIPPED
        assert result["report"] == STATUS_SKIPPED
        # pose depends on metadata only, so a speech failure must not stop it.
        assert result["pose"] == STATUS_COMPLETED
        assert FakeStage.executions.get("pose") == 1

    def test_a_branch_failure_does_not_stop_another(self, config, monkeypatch) -> None:
        FakeStage.fail_on = {"pose"}
        runner = runner_for(config, monkeypatch, Path(""))
        runner.run()
        result = statuses(runner)
        assert result["assign"] == STATUS_COMPLETED
        assert result["report"] == STATUS_SKIPPED

    def test_failure_is_recorded_with_context(self, config, monkeypatch) -> None:
        FakeStage.fail_on = {"audio"}
        runner = runner_for(config, monkeypatch, Path(""))
        outcome = runner.run()
        record = runner.state.stage("audio")
        assert record.error["type"] == "StageError"
        assert "exploded" in record.error["message"]
        assert "audio" in outcome.errors

    def test_video_status_is_partial_when_a_stage_fails(self, config, monkeypatch) -> None:
        FakeStage.fail_on = {"assign"}
        runner = runner_for(config, monkeypatch, Path(""))
        assert runner.run().status in {"partial", "failed"}


class TestInvalidation:
    def test_changed_stage_config_reruns_only_that_stage(self, config, monkeypatch) -> None:
        runner = runner_for(config, monkeypatch, Path(""))
        runner.run()
        baseline = dict(FakeStage.executions)
        FakeStage.extra_config = {"whisperx": {"model": "other"}}
        VideoRunner(runner.config, runner.source, tools={"schema_version": "1.0"}).run()
        assert FakeStage.executions["whisperx"] == baseline["whisperx"] + 1
        # Its dependants followed...
        assert FakeStage.executions["assign"] == baseline["assign"] + 1
        assert FakeStage.executions["report"] == baseline["report"] + 1
        # ...and its independent ancestors did not.
        assert FakeStage.executions["metadata"] == baseline["metadata"]
        assert FakeStage.executions["pose"] == baseline["pose"]

    def test_plan_explains_the_reason(self, config, monkeypatch) -> None:
        runner = runner_for(config, monkeypatch, Path(""))
        runner.run()
        plan = {item.name: item for item in runner.plan()}
        assert not plan["metadata"].will_run
        assert plan["metadata"].reason == "valid previous result"
        FakeStage.extra_config = {"audio": {"bitrate": 99}}
        plan = {item.name: item for item in runner.plan()}
        assert plan["audio"].will_run
        assert plan["audio"].reason == "configuration changed"
        assert plan["whisperx"].will_run
        assert plan["whisperx"].reason == "upstream dependency changed"

    def test_deleted_output_is_regenerated(self, config, monkeypatch) -> None:
        runner = runner_for(config, monkeypatch, Path(""))
        runner.run()
        runner.paths.artifact("speech_segments").unlink()
        plan = {item.name: item for item in runner.plan()}
        assert plan["whisperx"].reason == "output artifacts missing"
        runner.run()
        assert runner.paths.artifact("speech_segments").is_file()

    def test_unreadable_output_is_not_reused(self, config, monkeypatch) -> None:
        """Existence is not validation: a corrupt artifact must trigger a rerun."""
        runner = runner_for(config, monkeypatch, Path(""))
        runner.run()
        baseline = dict(FakeStage.executions)
        runner.paths.artifact("speech_segments").write_text("{ not json", encoding="utf-8")
        plan = {item.name: item for item in runner.plan()}
        assert plan["whisperx"].will_run
        assert plan["whisperx"].reason.startswith("validation failed")
        runner.run()
        assert FakeStage.executions["whisperx"] == baseline["whisperx"] + 1

    def test_force_flag_reruns_a_completed_stage(self, config, monkeypatch) -> None:
        runner = runner_for(config, monkeypatch, Path(""))
        runner.run()
        baseline = dict(FakeStage.executions)
        plan = {item.name: item for item in runner.plan(force_stages=["whisperx"])}
        assert plan["whisperx"].will_run
        assert plan["whisperx"].reason == "forced recomputation"
        runner.run(force_stages=["whisperx"])
        assert FakeStage.executions["whisperx"] == baseline["whisperx"] + 1

    def test_forcing_a_stage_makes_dependants_stale(self, config, monkeypatch) -> None:
        """The subtle one: identical config + forced rerun must still cascade."""
        runner = runner_for(config, monkeypatch, Path(""))
        runner.run()
        baseline = dict(FakeStage.executions)
        runner.run(force_stages=["whisperx"])
        after_force = dict(FakeStage.executions)
        assert after_force["whisperx"] == baseline["whisperx"] + 1
        # Dependants follow in the same pass: the stage loop re-reads each upstream
        # record as it goes, so nothing stale survives to a later run.
        assert after_force["assign"] == baseline["assign"] + 1
        assert after_force["report"] == baseline["report"] + 1
        # The independent branch and the shared ancestor were left alone.
        assert after_force["metadata"] == baseline["metadata"]
        assert after_force["pose"] == baseline["pose"]
        # And the pipeline settles: a following pass does no work at all.
        VideoRunner(runner.config, runner.source, tools={"schema_version": "1.0"}).run()
        assert dict(FakeStage.executions) == after_force

    def test_schema_version_change_invalidates_everything(self, config, monkeypatch) -> None:
        runner = runner_for(config, monkeypatch, Path(""))
        runner.run()
        baseline = dict(FakeStage.executions)
        VideoRunner(runner.config, runner.source, tools={"schema_version": "2.0"}).run()
        assert all(FakeStage.executions[name] > baseline[name] for name, _ in FAKE_DAG)


class TestStageSelection:
    def test_to_stage_stops_early(self, config, monkeypatch) -> None:
        runner = runner_for(config, monkeypatch, Path(""))
        runner.run(to_stage="whisperx")
        assert FakeStage.executions.get("assign") is None
        assert FakeStage.executions["whisperx"] == 1
        assert statuses(runner)["report"] == STATUS_SKIPPED

    def test_only_stage_still_runs_its_dependencies(self, config, monkeypatch) -> None:
        runner = runner_for(config, monkeypatch, Path(""))
        runner.run(only_stage=["assign"])
        # assign needs whisperx needs audio needs metadata.
        assert set(FakeStage.executions) == {"metadata", "audio", "whisperx", "assign"}

    def test_only_stage_pulls_the_whole_dependency_chain(self, config, monkeypatch) -> None:
        runner = runner_for(config, monkeypatch, Path(""))
        plan = {item.name: item for item in runner.plan(only_stage=["report"])}
        assert all(plan[name].will_run for name, _ in FAKE_DAG)

    def test_from_stage_skips_earlier_completed_work(self, config, monkeypatch) -> None:
        runner = runner_for(config, monkeypatch, Path(""))
        runner.run(to_stage="whisperx")
        FakeStage.executions.clear()
        runner.run(from_stage="assign")
        assert FakeStage.executions.get("metadata") is None
        assert FakeStage.executions["assign"] == 1


class TestInterruption:
    def test_a_process_kill_leaves_a_rerunnable_state(self, config, monkeypatch) -> None:
        """Simulate a crash: a stage left in 'running' must be picked up again."""
        runner = runner_for(config, monkeypatch, Path(""))
        runner.run(to_stage="whisperx")
        runner.state.mark_running("assign")  # as if killed mid-stage
        runner.state.save()

        second = VideoRunner(runner.config, runner.source, tools={"schema_version": "1.0"})
        second.run()
        assert statuses(second)["assign"] == STATUS_COMPLETED
        assert second.state.stage("metadata").status == STATUS_COMPLETED

    def test_running_stage_is_never_reused(self, config, monkeypatch) -> None:
        runner = runner_for(config, monkeypatch, Path(""))
        runner.run()
        runner.state.stage("whisperx").status = "running"
        runner.state.save()
        plan = {item.name: item for item in runner.plan()}
        assert plan["whisperx"].will_run
        assert plan["whisperx"].reason == "status is running"

    def test_status_file_survives_a_failing_stage(self, config, monkeypatch) -> None:
        FakeStage.fail_on = {"report"}
        runner = runner_for(config, monkeypatch, Path(""))
        runner.run()
        payload = json.loads(runner.paths.status.read_text())
        assert payload["stages"]["report"]["status"] == STATUS_FAILED
        assert payload["stages"]["metadata"]["status"] == STATUS_COMPLETED



def _write_rows(path: Path, rows: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"segment_id": [f"s{i}" for i in range(rows)]}), path)


def read_parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def _source_for(config):
    from multimodal_pipeline.discovery import VideoSource

    source = VideoSource(path=config.input.directory / "clip.mp4", relative_path=Path("clip.mp4"),
                         video_id="clip")
    source.path.write_bytes(b"video-bytes")
    return source


class ParquetStage(Stage):
    """A stage whose output is a real Parquet table, so integrity is measurable.

    ``FakeStage`` writes JSON into artifact slots, which is enough for reuse
    tests; artifact integrity needs the real format because the fingerprint is
    the row count.
    """

    executions: dict[str, int] = {}

    def __init__(self, name: str, artifact: str, *, depends_on: tuple[str, ...] = (),
                 rows: int = 5) -> None:
        self.name = name
        self.artifact = artifact
        self.rows = rows
        self.inputs = tuple(dep_artifact[dep] for dep in depends_on)
        self.outputs = (artifact,)

    def config_fingerprint(self, ctx: StageContext) -> dict[str, Any]:
        return {"stage": self.name, "rows": self.rows}

    def execute(self, ctx: StageContext) -> dict[str, Any]:
        type(self).executions[self.name] = type(self).executions.get(self.name, 0) + 1
        _write_rows(ctx.artifact(self.artifact), self.rows)
        return {}

    def validate(self, ctx: StageContext) -> dict[str, Any]:
        path = ctx.artifact(self.artifact)
        if not path.is_file():
            from multimodal_pipeline.exceptions import ValidationError

            raise ValidationError(self.name, [f"missing {path.name}"])
        return {"rows": read_parquet_rows(path)}


#: Both stages write the same slot, the way speaker_assignment rewrites the
#: transcript tables that whisperx produced.
TRANSCRIPT = "speech_segments"
dep_artifact = {"asr": TRANSCRIPT}


@pytest.fixture
def parquet_dag(monkeypatch, config):
    ParquetStage.executions = {}
    stages = [ParquetStage("asr", TRANSCRIPT, rows=5),
              ParquetStage("post", TRANSCRIPT, depends_on=("asr",), rows=2)]
    monkeypatch.setattr("multimodal_pipeline.orchestrator.build_stages", lambda: list(stages))
    monkeypatch.setattr("multimodal_pipeline.stages.base.STAGE_ORDER", ("asr", "post"))
    monkeypatch.setattr("multimodal_pipeline.orchestrator.STAGE_ORDER", ("asr", "post"))
    monkeypatch.setattr("multimodal_pipeline.stages.base.STAGE_DEPENDENCIES",
                        {"asr": (), "post": ("asr",)})
    return VideoRunner(config, _source_for(config), tools={"schema_version": "1.0"})


class TestArtifactIntegrity:
    def test_completion_records_a_row_count_fingerprint(self, parquet_dag) -> None:
        parquet_dag.run()
        record = type(parquet_dag.state).load(parquet_dag.paths, "clip").stage("post")
        assert record.status == STATUS_COMPLETED
        assert record.output_row_counts == {TRANSCRIPT: 2}

    def test_a_truncated_output_forces_a_recompute(self, parquet_dag) -> None:
        """An artifact that exists and parses but lost rows must not be trusted."""
        parquet_dag.run()
        baseline = dict(ParquetStage.executions)
        _write_rows(parquet_dag.paths.artifact(TRANSCRIPT), 1)
        plan = {item.name: item for item in parquet_dag.plan()}
        assert plan["post"].will_run
        assert "outputs changed" in plan["post"].reason
        assert "1 rows, 2 when validated" in plan["post"].reason
        parquet_dag.run()
        assert ParquetStage.executions["post"] == baseline["post"] + 1

    def test_an_untouched_dataset_is_never_recomputed(self, parquet_dag) -> None:
        parquet_dag.run()
        baseline = dict(ParquetStage.executions)
        parquet_dag.run()
        assert ParquetStage.executions == baseline

    def test_a_later_stages_rewrite_of_the_same_file_is_not_corruption(
        self, parquet_dag
    ) -> None:
        """``post`` owns the slot now; judging it against ``asr``'s fingerprint
        would flag every healthy dataset."""
        parquet_dag.run()
        _write_rows(parquet_dag.paths.artifact(TRANSCRIPT), 1)
        plan = {item.name: item for item in parquet_dag.plan()}
        assert plan["asr"].reason == "valid previous result"
        assert plan["post"].will_run
