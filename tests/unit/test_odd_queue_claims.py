"""The prose capability queue cannot describe finished work as pending.

`odd/tasks/multimodal-video-pipeline.md` §26 is a table of the operator's requested
capabilities with a prose `state` cell. It has lied twice, the same way each time: the row
said a capability was blocked or unstarted, the blocker was cleared, the stage was built,
ran on the corpus and gained a test suite, and the row was never re-read. The second
instance is the one that is easy to wave away — `20.3 stories | blocked, not started |
needs the operator's endpoint and a spend decision`, kept four commits after the stage
existed — and nobody noticed it in a test dying. A human read the document.

So this guard exists, and it asserts **one direction only**:

    a row that claims work has not started must not name a stage that exists

The obvious stronger rule — "a row saying `built` must name a stage, and its artifacts must
be on disk" — was tried against the table as written and rejected, because it is wrong four
times before it is right:

* `20.1` says `built` and names `diarization_v2`, which is not a stage: the shipped stage is
  `speaker_fusion`. The row is accurate, the name is an aspiration that got renamed.
* `20.5` says `built` and names `pose_skeletons`, which deliberately has no stage of its own
  — it is served inside `openpose` (`write_images`, `image_max_side`, `pose_images_raw`) so
  the rendered view and the measured keypoints cannot disagree about which frames had a
  usable detection.
* `20.6` says `built` and names documentation, not a stage at all.
* A disk check would then fail on any machine that has not run the corpus.

A guard that needs four exceptions is not a guard, it is noise with a failure message, and
noise is how guards rot into things people tick. The negative direction needs none: "not
started" and "the stage exists" is a contradiction in every case, including the ones where
the row's name is an alias, a sub-feature, or prose. That is the whole defect, priced at one
assertion.

"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from multimodal_pipeline.stages.base import STAGE_ORDER

ROOT = Path(__file__).resolve().parents[2]
MASTER_ODD = ROOT / "odd" / "tasks" / "multimodal-video-pipeline.md"

# Words that claim work has not happened. Deliberately narrow: it names a state, it does not
# try to read the row's prose for tone. "deferred" is absent on purpose — a deferred item can
# legitimately be built elsewhere (20.5 is exactly that), and an unstarted-but-blocked item
# says "blocked".
NOT_STARTED = ("not started", "unstarted", "blocked", "pending", "not begun")


def primary_state(cell: str) -> str:
    """The state the row *declares*, stripped of the history it may be carrying.

    This distinction is not pedantry, it is the first thing this guard got wrong. The honest
    way to fix a stale queue row is to keep the correction visible — §26/20.3 now reads
    "**built** (corrected 2026-10-02 — this row said \"blocked, not started\" for a stage
    that had shipped)". Matching the whole cell calls that row a lie, which means the guard
    would punish the exact behaviour that stopped the original defect: deleting history so a
    checker stays quiet. So the state is the first bolded run, or the cell up to its first
    parenthetical/em-dash aside, and only that is matched against NOT_STARTED.
    """
    bold = re.match(r"\s*\*\*([^*]+)\*\*", cell)
    if bold:
        return bold.group(1).strip().lower()
    return re.split(r"\s*[\u2014(]", cell, maxsplit=1)[0].strip().lower()


def capability_rows(text: str) -> list[tuple[str, str, str]]:
    """The §26 queue as (entry, name, state) triples.

    Parsed by the `20.N` entry number rather than by counting a table position, so an
    inserted column or a re-wrapped cell does not silently shift a cell into the wrong
    slot — which is how a table-asserting test starts asserting the wrong thing while
    still passing.
    """
    rows = []
    for line in text.splitlines():
        # The name is optional and may be prose: 20.6 is "documentation that installs",
        # which is a capability with no stage to name. Dropping that row would quietly
        # shrink what the guard covers, so the name slot accepts either form.
        match = re.match(r"^\|\s*(20\.\d+)\s+(?:`([^`]+)`|([^|]*?))\s*\|([^|]*)\|", line)
        if match:
            rows.append((match.group(1), match.group(2) or "", match.group(4).strip()))
    return rows


@pytest.fixture(scope="module")
def queue_text() -> str:
    return MASTER_ODD.read_text(encoding="utf-8")


class TestQueueDoesNotCallShippedWorkUnstarted:
    def test_the_queue_table_is_found_at_all(self, queue_text: str) -> None:
        """A guard that parses nothing is green, which is worse than a guard that fails.

        The expected entry set is derived from the document (every line that starts a `20.N`
        table row) rather than hardcoded. That is not tidiness: the first version of this
        parser silently dropped 20.6 because its name is prose ("documentation that
        installs") and not a backticked stage, and a hardcoded `>= 6` would have stayed
        green at six parsed rows out of seven. Deriving the set catches any row the parser
        fails to read — which is the failure mode that makes a prose-parsing guard decorative
        — while never needing an edit when the operator legitimately adds a 20.7.
        """
        declared = set(re.findall(r"^\|\s*(20\.\d+)\s", queue_text, re.M))
        parsed = {entry for entry, _name, _state in capability_rows(queue_text)}
        assert declared, "no §26 queue rows found at all"
        assert parsed == declared, (
            f"§26 rows the guard did not parse: {sorted(declared - parsed)} — a row this "
            "test cannot read is a row this guard cannot police"
        )

    @pytest.mark.parametrize("entry,name,state", capability_rows(
        MASTER_ODD.read_text(encoding="utf-8")))
    def test_a_row_claiming_no_work_did_not_name_an_existing_stage(
        self, entry: str, name: str, state: str
    ) -> None:
        if not any(phrase in primary_state(state) for phrase in NOT_STARTED):
            pytest.skip(f"{entry} does not claim the work is unstarted")
        assert name not in STAGE_ORDER, (
            f"§26 says {entry} `{name}` is {state!r}, but {name} is a stage in STAGE_ORDER. "
            "This is the exact defect the queue has had twice: the row still describes the "
            "state before the work landed. Update the row (and keep the correction visible "
            "rather than deleting it — §26/§41 show why the history is the useful part)."
        )

    def test_this_guard_would_have_caught_the_real_defect(self, tmp_path: Path) -> None:
        """A guard nobody has seen fail is not evidence, so here is its own red case.

        The row is the literal text §26 carried for four commits after `stories` shipped,
        recovered from `git show b4af917~1`, not a paraphrase of it.
        """
        lying = (
            "| entry | state | evidence |\n"
            "|---|---|---|\n"
            "| 20.3 `stories` | **blocked, not started** | needs the operator's endpoint "
            "and a spend decision |\n"
        )
        rows = capability_rows(lying)
        assert rows == [("20.3", "stories", "**blocked, not started**")]
        entry, name, state = rows[0]
        assert any(phrase in state.lower() for phrase in NOT_STARTED)
        assert name in STAGE_ORDER, "`stories` left STAGE_ORDER; this case is now vacuous"

    def test_the_shipped_table_passes_the_same_check(self, queue_text: str) -> None:
        """The pair of the test above: same predicate, real table, nothing may trip."""
        for entry, name, state in capability_rows(queue_text):
            if any(phrase in primary_state(state) for phrase in NOT_STARTED):
                assert name not in STAGE_ORDER, f"{entry} trips the guard it exists for"

    def test_a_corrected_row_keeping_its_history_does_not_trip(self) -> None:
        """The row that records its own correction must pass, not be punished.

        This is the literal corrected §26/20.3 cell. The whole-cell reading of the guard
        flagged it, which would have taught the next operator to delete the correction so
        the checker stops complaining — the opposite of what the queue needs. The state the
        row *declares* is `built`; the stale wording inside the parenthetical is the audit
        trail, and the guard has to be able to tell those two apart.
        """
        corrected = ('**built** (corrected 2026-10-02 \u2014 this row said "blocked, not '
                     'started" for a stage that had shipped)')
        assert primary_state(corrected) == "built"
        assert not any(phrase in primary_state(corrected) for phrase in NOT_STARTED)

    def test_a_row_that_regresses_to_pending_still_trips(self) -> None:
        """Reading only the primary state must not become a way to hide a stale row."""
        for state in ("**blocked, not started**", "**pending**", "**unstarted**",
                      "not begun", "**blocked** (needs the operator's endpoint)"):
            assert any(phrase in primary_state(state) for phrase in NOT_STARTED), state

    def test_alias_rows_are_not_mistaken_for_stages(self, queue_text: str) -> None:
        """`diarization_v2` and `pose_skeletons` are real rows naming non-stages.

        Recorded so the next person does not "fix" the guard into the affirmative
        direction and then have to add exceptions: 20.1 shipped as `speaker_fusion` and
        20.5 ships inside `openpose` on purpose.
        """
        names = {name for _e, name, _s in capability_rows(queue_text)}
        assert "diarization_v2" in names and "pose_skeletons" in names
        assert not set(names) & set(STAGE_ORDER) - {"persons", "stories", "pose_normalized"}
