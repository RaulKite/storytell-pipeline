"""Robust subprocess execution with capture, timeouts and masked provenance.

Every external tool (ffmpeg, ffprobe, uv workers, OpenPose) goes through
:func:`run_command`, so the orchestrator always knows argv, cwd, environment
overrides, timings, exit code and bounded output tails — with secrets masked
before anything reaches a log or ``status.json``.

stdout and stderr are merged into one ordered stream: interleaving matters when
diagnosing a tool that prints progress on stdout and errors on stderr, and the
combined text is what lands in the per-stage log file.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .config import mask_command

_SECRET_KEY_RE = re.compile(r"(token|key|secret|password)", re.I)
_MAX_TAIL_CHARS = 200_000  # bounded in-memory tails, never unbounded subprocess output


class CommandError(RuntimeError):
    """Raised when an external command fails or times out."""

    def __init__(self, message: str, *, result: "CommandResult | None" = None) -> None:
        super().__init__(message)
        self.result = result


@dataclass
class CommandResult:
    argv: list[str]
    argv_masked: list[str]
    cwd: str | None
    returncode: int
    output: str
    started_at: float
    duration_seconds: float
    timed_out: bool = False
    output_lines: int = 0
    env_overrides: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    # Backwards-friendly aliases: the captured stream is merged output.
    @property
    def stdout(self) -> str:
        return self.output

    @property
    def stderr(self) -> str:
        return self.output

    def tail(self, limit: int = 40) -> str:
        return "\n".join(self.output.splitlines()[-limit:])

    @property
    def stderr_tail(self) -> str:
        return self.tail()

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.argv_masked,
            "cwd": self.cwd,
            "exit_code": self.returncode,
            "duration_seconds": round(self.duration_seconds, 3),
            "timed_out": self.timed_out,
            "output_lines": self.output_lines,
            "output_tail": self.tail(20),
            "env_overrides": {
                key: ("***masked***" if _SECRET_KEY_RE.search(key) else val)
                for key, val in self.env_overrides.items()
            },
        }


def which(executable: str) -> str | None:
    return shutil.which(executable)


def require_executable(executable: str | Path, *, hint: str = "") -> Path:
    """Resolve an executable path or raise a helpful error before wasting a run."""
    candidate = Path(executable)
    if candidate.is_absolute():
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
        raise FileNotFoundError(f"executable not found or not runnable: {candidate}. {hint}".strip())
    found = shutil.which(str(candidate))
    if found:
        return Path(found)
    raise FileNotFoundError(f"executable '{candidate}' not found on PATH. {hint}".strip())


def run_command(
    argv: Sequence[str | Path],
    *,
    cwd: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    log_path: Path | None = None,
    line_callback: Callable[[str], None] | None = None,
    check: bool = True,
) -> CommandResult:
    """Run ``argv`` (argument array, never a shell string) and capture everything.

    ``timeout`` is enforced by a watchdog that kills the process even when it
    hangs without producing output.
    """
    argv_str = [str(a) for a in argv]
    if not argv_str:
        raise ValueError("run_command requires a non-empty argv")
    overrides: dict[str, str] = {k: str(v) for k, v in (env or {}).items()}
    merged_env = dict(os.environ)
    merged_env.update(overrides)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    masked = mask_command(argv_str)
    started = time.time()

    try:
        process = subprocess.Popen(
            argv_str,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors="replace",
        )
    except FileNotFoundError as exc:
        _log(log_path, f"$ {' '.join(masked)}\nERROR: {exc}\n")
        raise CommandError(f"executable not runnable: {argv_str[0]} ({exc})") from exc

    _log(log_path, f"\n$ {' '.join(masked)}\n(cwd={cwd or os.getcwd()})\n")

    killed = threading.Event()

    def _kill() -> None:
        killed.set()
        try:
            process.kill()
        except OSError:  # pragma: no cover - already gone
            pass

    watchdog = threading.Timer(timeout, _kill) if timeout else None
    if watchdog:
        watchdog.start()

    chunks: list[str] = []
    line_count = 0
    try:
        assert process.stdout is not None
        for line in process.stdout:
            line_count += 1
            chunks.append(line)
            if sum(map(len, chunks)) > _MAX_TAIL_CHARS * 2:
                joined = "".join(chunks)[-(_MAX_TAIL_CHARS * 2):]
                chunks = [joined]
            if log_path:
                _log(log_path, line)
            if line_callback:
                line_callback(line.rstrip())
        returncode = process.wait()
    finally:
        if watchdog:
            watchdog.cancel()

    duration = time.time() - started
    timed_out = killed.is_set()
    output = "".join(chunks)[-(_MAX_TAIL_CHARS * 2):]
    _log(log_path, f"[exit={returncode} timed_out={timed_out} duration={duration:.1f}s]\n")

    result = CommandResult(
        argv=argv_str,
        argv_masked=masked,
        cwd=str(cwd) if cwd else str(Path.cwd()),
        returncode=returncode,
        output=output,
        started_at=started,
        duration_seconds=duration,
        timed_out=timed_out,
        output_lines=line_count,
        env_overrides=overrides,
    )
    if timed_out:
        raise CommandError(
            f"command timed out after {timeout}s: {' '.join(masked)}\n{result.tail()}", result=result
        )
    if check and returncode != 0:
        raise CommandError(
            f"command failed with exit code {returncode}: {' '.join(masked)}\n{result.tail()}",
            result=result,
        )
    return result


def _log(log_path: Path | None, text: str) -> None:
    if not log_path:
        return
    try:
        with log_path.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(text)
    except OSError:  # logging must never break processing
        pass


def probe_version(argv: Iterable[str | Path], *, timeout: float = 60.0) -> str | None:
    """Best-effort ``--version`` capture for provenance; never raises."""
    try:
        result = run_command(list(argv), timeout=timeout, check=False)
    except (CommandError, OSError):
        return None
    text = (result.output or "").strip()
    return text.splitlines()[0][:400] if text else None
