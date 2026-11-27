"""The uv worker contract: how the orchestrator talks to every heavy tool.

A real throwaway uv project is used rather than a patched ``subprocess``, because
the contract under test is precisely "run this in another environment and trust
its result file": argument passing, environment forwarding, the ``status: ok``
proof of work and stale-result handling.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from multimodal_pipeline.exceptions import ValidationError, WorkerError
from multimodal_pipeline.stages.base import WorkerStage
from multimodal_pipeline.uv_worker import (
    request_hash,
    run_worker,
    worker_argv,
    worker_result_path,
    write_worker_request,
)

uv = pytest.fixture(scope="session")(lambda: shutil.which("uv"))

WORKER_SOURCE = '''
#!/usr/bin/env python3
"""A worker that behaves however the test told it to."""
import argparse, json, os, sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--raw-output", type=Path)
parser.add_argument("--result-json", type=Path)
parser.add_argument("--mode", default="ok")
parser.add_argument("--echo-env", default=None)
parser.add_argument("--extra", default=None)
args = parser.parse_args()

behaviour = args.mode
if args.echo_env:
    os.environ.setdefault("ECHOED", os.environ.get(args.echo_env, "<missing>"))

if behaviour == "ok":
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text(json.dumps({"native": True, "extra": args.extra}))
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps({
        "status": "ok", "tool_version": "fake 1.0", "model_version": "fake-model",
        "echoed": os.environ.get("ECHOED"), "argv_len": len(sys.argv),
    }))
    sys.exit(0)
if behaviour == "fails":
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps({"status": "error", "error": "worker said no"}))
    print("worker said no", file=sys.stderr)
    sys.exit(1)
if behaviour == "silent":
    sys.exit(0)          # claims success, writes nothing
if behaviour == "lies":
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps({"status": "ok"}))
    sys.exit(0)          # claims success, produces no raw artifact
if behaviour == "no_raw":
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps({"status": "error", "error": "missing input"}))
    sys.exit(2)
sys.exit(9)
'''


@pytest.fixture(scope="module")
def uv_project(tmp_path_factory) -> Path:
    """A minimal uv project with one script — the shape every real env has."""
    root = tmp_path_factory.mktemp("uvproj")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "mp-fake-env"\nversion = "0.1.0"\n'
        'requires-python = ">=3.10"\ndependencies = []\n',
        encoding="utf-8",
    )
    scripts = root / "workers"
    scripts.mkdir()
    (scripts / "fake_worker.py").write_text(WORKER_SOURCE, encoding="utf-8")
    if shutil.which("uv") is None:
        pytest.skip("uv not installed")
    return root


@pytest.fixture
def workspace(tmp_path: Path) -> dict:
    return {"log": tmp_path / "stage.log", "result": tmp_path / "raw" / "worker_result.json",
            "raw": tmp_path / "raw" / "native.json"}


def invoke(uv_project: Path, workspace: dict, mode: str = "ok", *, args=None, **kwargs):
    argv = ["--raw-output", str(workspace["raw"]), "--result-json", str(workspace["result"]),
            "--mode", mode, *(args or [])]
    return run_worker(uv_project=uv_project, worker_script=uv_project / "workers" / "fake_worker.py",
                      args=argv, log_path=workspace["log"], result_path=workspace["result"], **kwargs)


class TestArgv:
    def test_project_and_script_come_first(self) -> None:
        argv = worker_argv(None, Path("env"), Path("w.py"), ["--x", "1"])
        assert argv[:4] == ["uv", "run", "--project", "env"]
        assert argv[4:6] == ["python", "w.py"]
        assert argv[6:] == ["--x", "1"]

    def test_python_version_is_forwarded(self) -> None:
        argv = worker_argv("3.12", Path("env"), Path("w.py"), [])
        assert argv[4:6] == ["--python", "3.12"]

    def test_uv_executable_is_configurable(self) -> None:
        assert worker_argv(None, Path("e"), Path("w"), [], uv_executable="/opt/uv")[0] == "/opt/uv"


class TestSuccessPath:
    def test_result_is_read_back(self, uv_project, workspace) -> None:
        result = invoke(uv_project, workspace)
        assert result.ok
        assert result.tool_version == "fake 1.0"
        assert result.model_version == "fake-model"
        assert result.exit_code == 0

    def test_raw_artifact_is_produced(self, uv_project, workspace) -> None:
        invoke(uv_project, workspace)
        assert json.loads(workspace["raw"].read_text())["native"] is True

    def test_arguments_reach_the_worker(self, uv_project, workspace) -> None:
        result = invoke(uv_project, workspace, args=["--extra", "hello"])
        assert json.loads(workspace["raw"].read_text())["extra"] == "hello"
        assert result.payload["argv_len"] > 1

    def test_environment_is_forwarded(self, uv_project, workspace) -> None:
        result = invoke(uv_project, workspace, env={"MY_SECRET": "forwarded"},
                        args=["--echo-env", "MY_SECRET"])
        assert result.payload["echoed"] == "forwarded"

    def test_log_file_records_the_command(self, uv_project, workspace) -> None:
        invoke(uv_project, workspace)
        assert "fake_worker.py" in workspace["log"].read_text()

    def test_duration_is_measured(self, uv_project, workspace) -> None:
        assert invoke(uv_project, workspace).duration_seconds > 0


class TestFailurePath:
    def test_a_reported_error_becomes_a_worker_error(self, uv_project, workspace) -> None:
        with pytest.raises(WorkerError, match="worker said no"):
            invoke(uv_project, workspace, "fails")

    def test_the_error_payload_is_carried(self, uv_project, workspace) -> None:
        with pytest.raises(WorkerError) as excinfo:
            invoke(uv_project, workspace, "fails")
        assert excinfo.value.details["worker_result"]["status"] == "error"

    def test_a_silent_exit_is_not_trusted(self, uv_project, workspace) -> None:
        """Exit 0 with no result file proves nothing; the stage must fail."""
        with pytest.raises(WorkerError, match="no result JSON"):
            invoke(uv_project, workspace, "silent")

    def test_a_nonzero_exit_with_a_result_keeps_its_message(self, uv_project, workspace) -> None:
        with pytest.raises(WorkerError) as excinfo:
            invoke(uv_project, workspace, "no_raw")
        assert "missing input" in str(excinfo.value)

    def test_a_missing_result_is_never_inherited_from_the_previous_run(
        self, uv_project, workspace
    ) -> None:
        """The old result file must be deleted before the worker starts."""
        invoke(uv_project, workspace, "ok")
        assert workspace["result"].is_file()
        with pytest.raises(WorkerError):
            invoke(uv_project, workspace, "silent")
        assert not workspace["result"].exists()

    def test_missing_worker_script_fails_clearly(self, uv_project, workspace) -> None:
        with pytest.raises(WorkerError, match="worker script not found"):
            run_worker(uv_project=uv_project, worker_script=uv_project / "absent.py", args=[],
                       log_path=workspace["log"], result_path=workspace["result"])

    def test_missing_uv_project_names_the_remedy(self, tmp_path, workspace) -> None:
        with pytest.raises(WorkerError, match="uv sync"):
            run_worker(uv_project=tmp_path / "nope", worker_script=Path("w.py"), args=[],
                       log_path=workspace["log"], result_path=workspace["result"])

    def test_nonexistent_uv_executable_is_reported(self, uv_project, workspace, tmp_path) -> None:
        with pytest.raises(Exception, match="uv"):
            run_worker(uv_project=uv_project, worker_script=uv_project / "workers" / "fake_worker.py",
                       args=[], log_path=workspace["log"], result_path=workspace["result"],
                       uv_executable=str(tmp_path / "no-uv-here"))


class TestRequestRecords:
    def test_request_is_written_atomically(self, tmp_path: Path) -> None:
        path = write_worker_request(tmp_path / "req.json", {"model": "m", "api_key": "sk-1"})
        assert json.loads(path.read_text())["model"] == "m"

    def test_hash_is_stable_and_order_insensitive(self) -> None:
        assert request_hash({"a": 1, "b": 2}) == request_hash({"b": 2, "a": 1})

    def test_hash_changes_with_content(self) -> None:
        assert request_hash({"model": "a"}) != request_hash({"model": "b"})

    def test_hash_length_is_bounded(self) -> None:
        assert len(request_hash({"x": "y" * 5000})) == 16

    def test_result_path_helper(self, tmp_path: Path) -> None:
        assert worker_result_path(tmp_path) == tmp_path / "worker_result.json"
        assert worker_result_path(tmp_path, "x.json") == tmp_path / "x.json"


class _RawStage(WorkerStage):
    """A concrete worker stage: WorkerStage is abstract until validate() exists."""

    name = "asr"
    raw_artifact = "whisperx_raw"

    def validate(self, ctx):  # noqa: D401 - unused by validate_raw
        return {}


class TestRawArtifactContract:
    """The raw artifact is the one thing a rerun of normalization depends on."""

    stage = _RawStage()

    @pytest.fixture
    def ctx(self, tmp_path: Path):
        from multimodal_pipeline.artifacts import VideoPaths

        resolved = VideoPaths(tmp_path / "dataset")
        resolved.ensure_dirs()

        class _Ctx:
            def artifact(self, name: str) -> Path:
                return resolved.artifact(name)

        return _Ctx()

    def write_raw(self, ctx, text: str) -> None:
        path = ctx.artifact("whisperx_raw")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def test_a_dict_payload_is_returned(self, ctx) -> None:
        self.write_raw(ctx, '{"segments": []}')
        assert self.stage.validate_raw(ctx) == {"segments": []}

    def test_a_missing_raw_file_is_named(self, ctx) -> None:
        with pytest.raises(ValidationError, match="raw artifact missing: whisperx.json"):
            self.stage.validate_raw(ctx)

    def test_malformed_json_is_reported_as_unreadable(self, ctx) -> None:
        self.write_raw(ctx, "{ not json")
        with pytest.raises(ValidationError, match="raw artifact unreadable"):
            self.stage.validate_raw(ctx)

    def test_a_json_array_names_the_wrong_type(self, ctx) -> None:
        """"missing keys" about an array implies a rename; the real problem is the type."""
        self.write_raw(ctx, "[1, 2, 3]")
        with pytest.raises(ValidationError, match="is a JSON list, expected an object"):
            self.stage.validate_raw(ctx)

    def test_a_truncated_object_is_still_caught(self, ctx) -> None:
        self.write_raw(ctx, '{"segments": [{"start": 0}')
        with pytest.raises(ValidationError, match="unreadable"):
            self.stage.validate_raw(ctx)
