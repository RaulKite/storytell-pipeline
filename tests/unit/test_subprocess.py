"""Subprocess execution: timeouts, bounded capture, masking, executable checks."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from multimodal_pipeline.subprocess_utils import (
    CommandError,
    CommandResult,
    probe_version,
    require_executable,
    run_command,
    which,
)

PY = sys.executable


def python(code: str) -> list[str]:
    return [PY, "-c", code]


class TestBasics:
    def test_success_returns_output(self) -> None:
        result = run_command(python("print('hello')"))
        assert result.returncode == 0
        assert result.ok
        assert "hello" in result.output

    def test_stderr_is_captured_in_the_same_stream(self) -> None:
        result = run_command(python("import sys; print('boom', file=sys.stderr)"))
        assert "boom" in result.output
        assert result.stderr == result.output

    def test_interleaving_is_preserved(self) -> None:
        """Merged capture must show events in the order the tool produced them.

        The child flushes both streams explicitly: a piped stdout is block-buffered
        by default, and blaming that ordering on the capture would be wrong.
        """
        code = ("import sys; "
                "print('out1',flush=True); print('err1',file=sys.stderr,flush=True); "
                "print('out2',flush=True)")
        output = run_command(python(code)).output
        assert output.index("out1") < output.index("err1") < output.index("out2")

    def test_nonzero_exit_raises_by_default(self) -> None:
        with pytest.raises(CommandError) as excinfo:
            run_command(python("raise SystemExit(3)"))
        assert excinfo.value.result.returncode == 3

    def test_check_false_reports_the_failure_instead(self) -> None:
        result = run_command(python("raise SystemExit(4)"), check=False)
        assert result.returncode == 4 and not result.ok

    def test_missing_executable_raises_command_error(self) -> None:
        with pytest.raises(CommandError, match="not runnable"):
            run_command(["definitely-not-an-executable-xyz"])

    def test_empty_argv_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            run_command([])

    def test_cwd_is_applied(self, tmp_path: Path) -> None:
        result = run_command(python("import os; print(os.getcwd())"), cwd=tmp_path)
        assert Path(result.output.strip()).resolve() == tmp_path.resolve()

    def test_arguments_are_never_run_through_a_shell(self, tmp_path: Path) -> None:
        # If argv were joined into a shell string, the redirection would create a
        # file. Passing argv as a list means the child only ever sees one argument.
        victim = tmp_path / "injected.txt"
        run_command([PY, "-c", "import sys; print(len(sys.argv[1:]))", f"x > {victim}"])
        assert not victim.exists()

    def test_stdin_is_closed(self) -> None:
        # A child reading stdin must see EOF instead of hanging the pipeline.
        result = run_command(python("import sys; print('eof' if sys.stdin.read()=='' else 'data')"))
        assert "eof" in result.output


class TestTimeouts:
    def test_a_hanging_command_is_killed(self) -> None:
        code = "import time; print('started', flush=True); time.sleep(30); print('never')"
        with pytest.raises(CommandError) as excinfo:
            run_command(python(code), timeout=2.0)
        result = excinfo.value.result
        assert result.timed_out
        assert "started" in result.output  # output before the hang survives

    def test_a_command_that_finishes_in_time_is_untouched(self) -> None:
        result = run_command(python("print('quick')"), timeout=30.0)
        assert result.returncode == 0 and not result.timed_out

    def test_timeout_output_records_the_kill(self) -> None:
        with pytest.raises(CommandError) as excinfo:
            run_command(python("import time; time.sleep(30)"), timeout=1.0)
        assert "timed out" in str(excinfo.value).lower() or excinfo.value.result.timed_out


class TestBoundedCapture:
    def test_flooded_output_does_not_grow_without_limit(self) -> None:
        # ~8 MB of output through a 200k-char cap.
        code = "import sys; sys.stdout.write('x' * (8*1024*1024))"
        result = run_command(python(code))
        assert len(result.output) <= 500_000
        assert result.output.endswith("x")

    def test_line_count_counts_every_line(self) -> None:
        code = "print('\\n'.join(str(i) for i in range(500)))"
        result = run_command(python(code))
        assert result.output_lines == 500

    def test_tail_returns_the_last_lines(self) -> None:
        code = "print('\\n'.join(str(i) for i in range(100)))"
        result = run_command(python(code))
        lines = result.tail(5).splitlines()
        assert lines == ["95", "96", "97", "98", "99"]

    def test_invalid_utf8_is_replaced_not_fatal(self) -> None:
        code = "import sys; sys.stdout.buffer.write(b'ok \\xff\\xfe end')"
        result = run_command(python(code))
        assert "ok" in result.output


class TestLoggingAndMasking:
    def test_output_lands_in_the_log_file(self, tmp_path: Path) -> None:
        log = tmp_path / "stage.log"
        run_command(python("print('to the log')"), log_path=log)
        text = log.read_text()
        assert "to the log" in text
        assert text.lstrip().startswith("$")  # the command line is recorded

    def test_credentials_are_masked_in_the_log(self, tmp_path: Path) -> None:
        log = tmp_path / "stage.log"
        run_command([PY, "-c", "print('x')", "--token", "super-secret-value"], log_path=log)
        text = log.read_text()
        assert "super-secret-value" not in text
        assert "***masked***" in text

    def test_result_keeps_both_the_real_and_masked_command(self) -> None:
        result = run_command([PY, "-c", "print(1)", "--api-key", "hunter2"])
        assert "hunter2" in " ".join(result.argv)
        assert "hunter2" not in " ".join(result.argv_masked)

    def test_secret_environment_values_are_masked_in_the_dict(self) -> None:
        result = run_command(python("print(1)"), env={"HF_TOKEN": "real-token"})
        payload = result.to_dict()
        assert payload["env_overrides"]["HF_TOKEN"] == "***masked***"

    def test_ordinary_environment_overrides_are_visible(self) -> None:
        result = run_command(python("print(1)"), env={"CUDA_VISIBLE_DEVICES": "0"})
        assert result.to_dict()["env_overrides"]["CUDA_VISIBLE_DEVICES"] == "0"

    def test_environment_overrides_reach_the_child(self) -> None:
        result = run_command(python("import os; print(os.environ['MP_TEST_VAR'])"),
                             env={"MP_TEST_VAR": "visible"})
        assert "visible" in result.output

    def test_overrides_do_not_replace_the_whole_environment(self) -> None:
        result = run_command(python("import os; print(bool(os.environ.get('PATH')))"),
                             env={"MP_ONLY": "1"})
        assert "True" in result.output


class TestCommandResult:
    def test_to_dict_shape(self) -> None:
        result = run_command(python("print('one\\ntwo')"))
        payload = result.to_dict()
        assert payload["exit_code"] == 0
        assert payload["timed_out"] is False
        assert payload["output_tail"].endswith("two")
        assert payload["command"]

    def test_duration_is_recorded(self) -> None:
        result = run_command(python("import time; time.sleep(0.2)"))
        assert result.duration_seconds >= 0.15


class TestRequireExecutable:
    def test_finds_a_bare_name_on_path(self) -> None:
        assert require_executable("python3")

    def test_absolute_path_is_accepted(self) -> None:
        assert require_executable(Path(sys.executable)) == Path(sys.executable)

    def test_missing_bare_name_raises_with_hint(self) -> None:
        with pytest.raises(FileNotFoundError, match="PATH"):
            require_executable("nope-not-here", hint="install it")

    def test_missing_absolute_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found or not runnable"):
            require_executable(tmp_path / "missing")

    def test_directory_is_not_an_executable(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            require_executable(tmp_path)

    def test_non_executable_file_is_rejected(self, tmp_path: Path) -> None:
        script = tmp_path / "not-executable"
        script.write_text("#!/bin/sh\necho hi\n")
        script.chmod(0o644)
        with pytest.raises(FileNotFoundError):
            require_executable(script)


class TestProbeVersion:
    def test_reads_a_version_string(self) -> None:
        assert probe_version([PY, "--version"])

    def test_failure_is_reported_as_unknown(self) -> None:
        assert probe_version(["definitely-not-an-executable-xyz", "--version"]) is None

    def test_nonzero_exit_is_unknown(self) -> None:
        assert probe_version(python("raise SystemExit(1)")) is None
