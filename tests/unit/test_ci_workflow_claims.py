"""The CI workflow's own claims, checked against the tree.

`.github/workflows/unit.yml` opens by stating what it covers. That prose is the only
reason an operator trusts a green check, and it had already rotted once: it described
"the 718-test unit suite" and a "drop coverage by 15 tests" penalty long after the suite
had grown past a thousand, and called the ffmpeg-dependent tests "two tests" when there
are seventeen. Nothing failed, because nothing read the file.

So this checks the structural claims, which have mechanical answers: that the workflow is
the unit-only workflow its name promises, that it keeps the flags that make the run
reproducible, and that it installs the tool its coverage statement depends on.

The counts are deliberately *not* restated here and are now banned from the workflow,
which is what `test_workflow_states_no_test_counts` enforces. Asserting "1065 passed" from
inside the unit suite is impossible without running the suite again (~33 s, doubling the
job) or building a fake `/usr/bin` of 1500 symlinks to measure the ffmpeg-gated delta, and
neither is a fair price for a comment. What a comment can say durably is *which* tests need
what; what it cannot say for free is how many there are.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML ships with the orchestrator environment")

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "unit.yml"
SOURCE = WORKFLOW.read_text(encoding="utf-8")


def steps() -> list[dict]:
    """Every step of the single job, in order."""
    document = yaml.safe_load(SOURCE)
    jobs = document["jobs"]
    assert len(jobs) == 1, f"expected one job, found {sorted(jobs)}"
    return next(iter(jobs.values()))["steps"]


def run_script(step: dict) -> str:
    return str(step.get("run", ""))


def all_run_scripts() -> str:
    return "\n".join(run_script(step) for step in steps())


class TestWorkflowExistsAndIsLegible:
    def test_the_file_parses_as_yaml(self) -> None:
        # An unparseable workflow does not fail loudly: GitHub shows a red triangle on a
        # commit and every claim in the file becomes a claim about nothing.
        assert isinstance(yaml.safe_load(SOURCE), dict)

    def test_it_runs_on_push_to_master_and_on_pull_requests(self) -> None:
        # PyYAML reads an unquoted `on:` as the boolean True (YAML 1.1), which is why the
        # key is looked up both ways rather than assumed.
        document = yaml.safe_load(SOURCE)
        triggers = document.get("on", document.get(True))
        assert triggers["push"]["branches"] == ["master"]
        assert "pull_request" in triggers

    def test_it_cannot_stale_out_and_pile_up(self) -> None:
        document = yaml.safe_load(SOURCE)
        assert document["concurrency"]["cancel-in-progress"] is True

    def test_it_cannot_write_to_the_repository(self) -> None:
        # Read-only contents is the whole story for a test workflow: it has no reason to
        # push, comment, or create a check that could be mistaken for a review receipt.
        document = yaml.safe_load(SOURCE)
        assert document["permissions"] == {"contents": "read"}


class TestItCoversWhatItsNameSays:
    def test_it_is_the_unit_suite_only_not_the_whole_suite(self) -> None:
        """The workflow is named `unit`, so `tests/e2e` must not quietly join it.

        e2e shells out to real `uv` project creation and to the OpenPose binary, so a
        hosted runner cannot be expected to pass it — and a workflow that starts passing
        e2e by accident would report a check it has never measured.
        """
        scripts = all_run_scripts()
        assert "pytest tests/unit" in scripts
        assert "tests/e2e" not in scripts

    def test_it_keeps_the_flag_that_makes_the_run_reproducible(self) -> None:
        # `-p no:randomly` is load-bearing, not cosmetic: some tests share module-level
        # state and a randomised order fails them for unrelated reasons. The CI command and
        # the one in AGENTS.md are kept byte-identical for the same reason.
        for step in steps():
            if "pytest tests/unit" in run_script(step):
                assert "-p no:randomly" in run_script(step)

    def test_it_installs_ffmpeg_because_its_coverage_depends_on_it(self) -> None:
        # Without this step the suite still goes green while ~17 tests skip, which is the
        # specific way a CI job loses coverage without ever reporting a failure.
        assert re.search(r"apt-get install.*ffmpeg", all_run_scripts()) is not None

    def test_it_syncs_the_interpreter_the_project_declares(self) -> None:
        expected = "3.12"
        assert f"uv sync --python {expected}" in all_run_scripts()

    def test_it_keeps_a_timeout_instead_of_running_forever(self) -> None:
        document = yaml.safe_load(SOURCE)
        job = next(iter(document["jobs"].values()))
        assert job["timeout-minutes"] <= 30


class TestWorkflowStatesNoTestCounts:
    """No number of tests, and no number of passes or skips, anywhere in the workflow.

    Every such figure in this file was true when written and false within days, and a
    stale count in a coverage statement is worse than no count: it is the number an
    operator quotes when deciding what the check means. The README's counts are the
    exception because `test_readme_claims.py` re-derives them from pytest's own collection
    on every run; the workflow has no equivalent that is cheap enough to be worth it.
    """

    def test_workflow_states_no_test_counts(self) -> None:
        offenders = re.findall(
            r"\b\d[\d,]*\s*(?:tests?|passed|skipped|failed|collected)\b", SOURCE)
        assert not offenders, (
            f"the workflow states test counts nobody verifies: {offenders}. Describe which "
            "tests need which tool instead of how many there are, or add a ratchet that "
            "re-derives the number the way test_readme_claims.py does.")

    def test_the_workflow_still_says_what_it_does_not_cover(self) -> None:
        """Removing the numbers must not remove the honesty.

        The point of the header was never the count; it was naming the GPU/OpenPose/token
        paths that stay unverified, so a green check is not read as a reviewed release.
        """
        header = SOURCE.split("on:", 1)[0].lower()
        for gap in ("gpu", "openpose", "token", "translation", "tests/e2e"):
            assert gap in header, f"header no longer names {gap!r} as unverified"
