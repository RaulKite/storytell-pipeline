"""The compact ODD tracker preserves open work without obsolete task numbering."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from multimodal_pipeline.stages.base import STAGE_ORDER

ROOT = Path(__file__).resolve().parents[2]
MASTER_ODD = ROOT / "odd/tasks/multimodal-video-pipeline.md"
NOT_STARTED = ("not started", "unstarted", "blocked", "pending", "not begun")


def capability_rows(text: str) -> list[tuple[str, str]]:
    section = text.split("## Delivered capabilities\n", 1)[1].split("\n## ", 1)[0]
    return re.findall(r"^\| `([a-z0-9_]+)` \| ([^|]+) \|", section, re.MULTILINE)


def validate_capabilities(text: str) -> None:
    rows = capability_rows(text)
    assert rows, "no delivered capabilities found"
    assert len({name for name, _ in rows}) == len(rows), "duplicate capability"
    for name, state in rows:
        assert name in STAGE_ORDER, f"unknown stage: {name}"
        assert not any(phrase in state.lower() for phrase in NOT_STARTED), (name, state)


def test_only_one_live_tracker_remains() -> None:
    assert sorted(p.name for p in (ROOT / "odd/tasks").glob("*.md")) == [MASTER_ODD.name]


def test_current_capabilities_are_real_and_not_reported_as_unstarted() -> None:
    validate_capabilities(MASTER_ODD.read_text())


@pytest.mark.parametrize("state", ["pending", "blocked, not started", "unstarted", "not begun"])
def test_a_stale_capability_state_is_rejected(state: str) -> None:
    text = f"## Delivered capabilities\n| `stories` | {state} | stale |\n"
    with pytest.raises(AssertionError):
        validate_capabilities(text)


def test_a_stage_alias_is_rejected() -> None:
    with pytest.raises(AssertionError, match="unknown stage"):
        validate_capabilities("## Delivered capabilities\n| `diarization_v2` | built | alias |\n")


def test_tracker_is_short_and_keeps_the_human_only_check_open() -> None:
    text = MASTER_ODD.read_text()
    assert len(text.splitlines()) <= 120
    assert re.search(r"^- \[ \].*ELAN.*(?:laptop|desktop)", text, re.MULTILINE)
    assert "headless" in text
    assert "Git" in text
