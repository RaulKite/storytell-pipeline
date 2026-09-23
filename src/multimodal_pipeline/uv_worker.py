"""Uniform invocation of the heavy ML workers inside their own uv projects.

The orchestrator never imports torch/pyannote/spacy: it calls
``uv run --project <env> python workers/<worker>.py`` and reads back a
machine-readable result JSON. That keeps mutually-incompatible CUDA/torch pins
out of the orchestrator environment and makes every stage resumable from the
worker's own status file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import atomic_write_json
from .config import mask_command, stable_hash
from .exceptions import WorkerError
from .subprocess_utils import CommandError, require_executable, run_command


@dataclass
class WorkerResult:
    argv: list[str]
    argv_masked: list[str]
    payload: dict[str, Any]
    exit_code: int
    duration_seconds: float
    tool_version: str | None
    model_version: str | None

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and self.payload.get("status") == "ok"


def worker_argv(config_python: str | None, uv_project: Path, worker_script: Path,
                args: Sequence[str], *, uv_executable: str = "uv") -> list[str]:
    """``uv run --project <env> <worker.py> <args...>`` with explicit interpreter."""
    argv: list[str] = [uv_executable, "run", "--project", str(uv_project)]
    if config_python:
        argv += ["--python", config_python]
    argv += ["python", str(worker_script), *args]
    return argv


def run_worker(
    *,
    uv_project: Path,
    worker_script: Path,
    args: Sequence[str],
    log_path: Path,
    result_path: Path,
    python_version: str | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    cwd: Path | None = None,
    line_callback=None,
    uv_executable: str = "uv",
) -> WorkerResult:
    """Run one worker, requiring its ``status=ok`` result JSON as proof of work."""
    project = Path(uv_project)
    # Environment first: "your ML environment is not set up" is the actionable
    # answer, and it is the one a fresh clone actually needs.
    if not project.is_dir():
        raise WorkerError(
            f"uv project not found: {project}. Create it and run `uv sync` there first."
        )
    if not worker_script.is_file():
        raise WorkerError(f"worker script not found: {worker_script}")
    require_executable(uv_executable, hint="install uv (https://docs.astral.sh/uv/)")
    argv = worker_argv(python_version, project, worker_script, list(args), uv_executable=uv_executable)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        result_path.unlink()  # a stale result must never be mistaken for success
    try:
        completed = run_command(
            argv,
            cwd=cwd,
            env=env,
            timeout=timeout,
            log_path=log_path,
            line_callback=line_callback,
        )
    except CommandError as exc:
        payload = _read_result(result_path)
        details = {"worker_result": payload} if payload else {}
        details["stderr_tail"] = (exc.result.stderr_tail if exc.result else str(exc))[:4000]
        # The worker's own diagnosis beats a bare exit code: a recorded error that
        # only says "exited with 1" sends someone digging through stderr.
        reported = (payload or {}).get("error")
        prefix = (f"worker reported status={(payload or {}).get('status')}: {reported}"
                  if reported else f"worker exited with {getattr(exc.result, 'returncode', '?')}")
        raise WorkerError(f"{prefix}: {exc}", details=details) from exc
    payload = _read_result(result_path)
    if payload is None:
        raise WorkerError(
            f"worker produced no result JSON at {result_path}",
            details={"stderr_tail": completed.stderr_tail[-4000:]},
        )
    if payload.get("status") != "ok":
        raise WorkerError(
            f"worker reported status={payload.get('status')}: {payload.get('error') or 'no error message'}",
            details={"worker_result": payload},
        )
    return WorkerResult(
        argv=argv,
        argv_masked=mask_command(argv),
        payload=payload,
        exit_code=completed.returncode,
        duration_seconds=completed.duration_seconds,
        tool_version=payload.get("tool_version"),
        model_version=payload.get("model_version"),
    )


def _read_result(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def worker_result_path(stage_dir: Path, name: str = "worker_result.json") -> Path:
    return stage_dir / name


def write_worker_request(path: Path, payload: dict[str, Any]) -> Path:
    """Record what was asked of a worker (hashed into the stage fingerprint)."""
    return atomic_write_json(path, payload)


def request_hash(payload: dict[str, Any]) -> str:
    return stable_hash(payload, length=16)
