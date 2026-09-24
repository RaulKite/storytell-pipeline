"""The README's checkable claims, checked against the tree.

Documentation rots silently: a sentence that was true when a stage was added stays
true forever in the file while the thing it describes moves. Two commit messages in
this repository's history already claimed README content that had not been written
(see ``odd/tasks/multimodal-video-pipeline.md`` §17), so the fix is not "be careful"
— it is a test that fails when the prose and the tree disagree.

Only claims with a mechanical answer belong here. "The frames table is dense" is a
sentence about behaviour that the stage tests own; "the quick start invokes the CLI
in a way a fresh clone can actually run" is a fact about this file and
``pyproject.toml``, and nothing else checks it.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text(encoding="utf-8")
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def fenced_blocks(language: str | None = None) -> list[str]:
    pattern = re.compile(r"^```(\w*)\n(.*?)^```", re.DOTALL | re.MULTILINE)
    return [
        body
        for lang, body in pattern.findall(README)
        if language is None or lang == language
    ]


class TestLicense:
    def test_license_file_exists_and_is_mit(self) -> None:
        license_path = ROOT / "LICENSE"
        assert license_path.is_file(), "README's License section points at a missing file"
        text = license_path.read_text(encoding="utf-8")
        assert text.startswith("MIT License")
        # A license with no copyright line grants nothing to nobody.
        assert re.search(r"^Copyright \(c\) \d{4} .+", text, re.MULTILINE)

    def test_pyproject_declares_the_same_license(self) -> None:
        declared = PYPROJECT["project"].get("license", {})
        assert declared.get("text", "").strip() == "MIT"

    def test_readme_does_not_claim_an_undeclared_license(self) -> None:
        # The sentence this test was written for: pyproject declared MIT for the whole
        # project's life while the README still said nothing had been chosen.
        assert "Not yet declared" not in README


class TestCliInvocation:
    """A bare console-script name is not on PATH in a fresh clone."""

    console_script = PYPROJECT["project"]["scripts"]["multimodal-pipeline"]

    def test_console_script_name_matches_the_readme(self) -> None:
        assert self.console_script == "multimodal_pipeline.cli:app_main"

    @pytest.mark.parametrize("index", range(len(fenced_blocks("bash"))))
    def test_bash_blocks_invoke_the_cli_through_uv_run(self, index: int) -> None:
        block = fenced_blocks("bash")[index]
        offenders = [
            line.strip()
            for line in block.splitlines()
            # A line that calls the CLI without `uv run` in front of it, ignoring
            # shell-variable assignments like `HF_TOKEN=... uv run ...`.
            if re.search(r"(?<![\w./-])multimodal-pipeline\s+\w", line)
            and "uv run" not in line
            and not line.lstrip().startswith("#")
        ]
        assert not offenders, (
            "the quick start must be copy-pasteable in a fresh clone, where the "
            f"console script is not yet on PATH: {offenders}"
        )


class TestEnvironmentCount:
    """The count in the prose is measured, not remembered."""

    def environments(self) -> list[str]:
        return sorted(
            path.name for path in (ROOT / "environments").iterdir() if path.is_dir()
        )

    def test_readme_syncs_every_environment_directory(self) -> None:
        synced = set(re.findall(r"environments/(\w+)\s+&& uv sync", README))
        missing = set(self.environments()) - synced
        assert not missing, f"environments present but never synced by the quick start: {missing}"

    def test_readme_counts_the_environments_it_ships(self) -> None:
        words = {3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
        count = len(self.environments())
        expected = words[count]
        assert f"[Why {expected} environments]" in README
        assert f"{expected} environments" in README.lower()
        for other in words.values():
            if other != expected:
                assert f"Why {other} environments" not in README

    def test_no_stale_heavy_tool_count_remains(self) -> None:
        # "The four heavy tools" survived the fifth environment being added, because the
        # sentence lives far away from the heading that counts them.
        count = len(self.environments())
        words = {3: "three", 4: "four", 5: "five", 6: "six"}
        for number, word in words.items():
            if number != count:
                assert f"{word} heavy" not in README.lower()


class TestDocumentedCommandsExist:
    """Every command in the Commands table is a real typer command."""

    def test_table_commands_are_reachable_from_the_cli(self) -> None:
        import typer

        from multimodal_pipeline.cli import app

        assert isinstance(app, typer.Typer)
        names = {
            command.name or command.callback.__name__
            for command in app.registered_commands
        }
        assert names, "the CLI exposes no registered commands to check against"
        table = re.search(r"## Commands\n(.*?)\n\n", README, re.DOTALL)
        assert table, "the Commands section moved; update this test's anchor"
        documented = set(re.findall(r"^\| `([a-z-]+)`", table.group(1), re.MULTILINE))
        assert documented, "no commands parsed from the table"
        assert documented <= names, f"README documents commands that do not exist: {sorted(documented - names)}"


class TestDatasetLayoutMatchesTheRegistry:
    """The tree diagram lists artifacts the registry actually declares."""

    def test_every_parquet_in_the_diagram_is_registered(self) -> None:
        from multimodal_pipeline.artifacts import ARTIFACT_LAYOUT

        listed = set(re.findall(r"([a-z_]+)\.parquet", README))
        known = {
            Path(relative).name.removesuffix(".parquet")
            for relative in ARTIFACT_LAYOUT.values()
            if relative.endswith(".parquet")
        }
        unknown = listed - known
        assert not unknown, f"README names Parquet artifacts that are not registered: {sorted(unknown)}"


def test_pytest_is_available_for_the_documented_test_command() -> None:
    """The README tells you to run `uv run --with pytest pytest tests/unit`."""
    assert (ROOT / "tests" / "unit").is_dir()
    assert (ROOT / "tests" / "e2e").is_dir()
    assert sys.version_info >= (3, 10)
