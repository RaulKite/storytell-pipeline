"""Structure tests for the ``status`` human render.

The defect these tests exist for was silent: ``rich`` never reports that a table does
not fit, it steals width until the row identity disappears. At 80 columns (what rich
uses when stdout is not a tty, which is what every CI run and every pipe sees) the old
table needed 196 columns on this corpus's real 72-character ids, so ``video_id``
collapsed to a single ``…`` and each header to ``m…``. No row said which video it was,
and the exit code was 0. It fitted the e2e fixture only because six stages are switched
off there: 78 of 80 columns, two columns of margin, which is why the failure was read as
"a new stage broke a test" rather than as a render that had never worked.

So these tests assert *structure* — which column a mark sits in, and whether an id
survives as one string — rather than the presence of a word. ``assert "alpha" in
stderr`` is exactly the assertion that passed while the render was broken elsewhere
and failed for an unrelated reason here; the id is not optional, it is the row.

Everything is driven through the real typer command with a temp config, because the
folding happened at print time: a test of a pure string builder would have missed the
mechanism that caused the bug.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import pytest
import yaml
from typer.testing import CliRunner

from multimodal_pipeline.artifacts import VideoPaths
from multimodal_pipeline.cli import _STATUS_MARKS, app
from multimodal_pipeline.orchestrator import enabled_stage_names
from multimodal_pipeline.state import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
    STATUS_SKIPPED,
    VideoState,
)
from multimodal_pipeline.stages.base import STAGE_ORDER

# The real ids in this corpus look like this (72 characters). slugify allows 80, so a
# test id of this length is not hypothetical: it is what the operator's files produce.
LONG_VIDEO_ID = "2017-12-30_1930_US_CNN_Global_Warning_Arctic_Melt_1237_273_1241_393_hear"

#: Stages switched off for the second case of the header test: a subset of what the
#: pipeline can run, so the header is exercised at two different widths. Counted from
#: ``STAGE_ORDER`` below rather than named, because the count moves when a stage lands.
SOME_STAGES_DISABLED = {
    "whisperx": False,
    "diarization": False,
    "translation": False,
    "acoustic": False,
    "openpose": False,
    "speaker_fusion": False,
}


def tokens_with_columns(line: str, after: int = 0) -> list[tuple[str, int]]:
    """Every non-blank token at or past column ``after``, with its start column.

    Positions are read off the header itself instead of being hardcoded, so a render
    that changes the field width still has to keep the marks under their index — the
    only property that matters to a human reading the output.
    """
    return [(match.group(0), match.start())
            for match in re.finditer(r"\S+", line) if match.start() >= after]


def status_header(lines: Sequence[str], stage_count: int) -> tuple[str, list[tuple[str, int]]]:
    """The one line whose tokens are exactly ``1..stage_count`` followed by ``ov``.

    Exact token equality is the point: rich dropped whole columns silently, so a test
    that only looked for "1" would not notice a header that stopped at 8.
    """
    expected = [str(index) for index in range(1, stage_count + 1)] + ["ov"]
    for line in lines:
        tokens = tokens_with_columns(line)
        if [token for token, _ in tokens] == expected:
            return line, tokens
    raise AssertionError(
        f"no header line with columns {expected}; rendered:\n" + "\n".join(lines)
    )


def marks_for(lines: Sequence[str], video_id: str) -> list[tuple[str, int]]:
    """The marks (and only the marks) belonging to ``video_id``, with start columns.

    Two shapes are legal: a short id shares its line with the marks, a long id gets a
    line to itself and the marks go directly beneath it. What is not legal is the id
    being folded, truncated or elided, which is what the previous render did. Tokens
    inside the id's own span are dropped, so a shared line yields marks, not ``alpha``.
    """
    for index, line in enumerate(lines):
        if line.strip() == video_id:
            if index + 1 >= len(lines):
                raise AssertionError(f"{video_id}: id line is the last line, no marks beneath it")
            return tokens_with_columns(lines[index + 1])
        if line.startswith(video_id):
            return tokens_with_columns(line, after=len(video_id))
    raise AssertionError(f"no line starts with the whole id {video_id!r}:\n" + "\n".join(lines))


def build_project(tmp_path: Path, video_ids: Sequence[str], *, disabled: dict[str, bool] | None = None):
    """A real on-disk project: videos on disk, real ``status.json`` files, real config.

    Nothing is mocked. ``status`` reads ``status.json`` through ``VideoState``, so the
    fixture writes those files with the same class the pipeline writes them with; a
    dict hand-assembled here would test the test.
    """
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    payload: dict[str, object] = {
        "project_root": str(tmp_path),
        "input": {"directory": str(input_dir)},
        "output": {"directory": str(output_dir)},
        "logging": {"level": "INFO", "console": False},
    }
    payload.update({name: {"enabled": enabled} for name, enabled in (disabled or {}).items()})
    config_path = tmp_path / "config" / "config.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    for video_id in video_ids:
        (input_dir / f"{video_id}.mp4").write_bytes(b"")
    return config_path, input_dir, output_dir


def write_state(output_dir: Path, input_dir: Path, video_id: str, statuses: dict[str, str]) -> None:
    """Persist one video's ``status.json`` with the requested per-stage states."""
    paths = VideoPaths(output_dir / video_id)
    paths.dataset_dir.mkdir(parents=True, exist_ok=True)
    state = VideoState.load(paths, video_id, str(input_dir / f"{video_id}.mp4"))
    state.bind_stages(STAGE_ORDER)
    for name, status in statuses.items():
        state.stage(name).status = status
    state.save()


def invoke_status(config_path: Path, *args: str):
    return CliRunner().invoke(app, ["status", "-c", str(config_path), *args])


class TestLongVideoIdSurvives:
    """The row identity must not be the first thing a narrow terminal loses.

    Written first, and run against the old table render: it failed there because ``rich``
    folded a 72-character id into a single ``…`` cell and exited 0.
    """

    def test_the_full_id_appears_as_one_unbroken_string(self, tmp_path: Path) -> None:
        config_path, input_dir, output_dir = build_project(tmp_path, [LONG_VIDEO_ID])
        write_state(output_dir, input_dir, LONG_VIDEO_ID,
                    {name: STATUS_COMPLETED for name in STAGE_ORDER})

        result = invoke_status(config_path)
        assert result.exit_code == 0, result.stderr
        lines = result.stderr.splitlines()

        assert LONG_VIDEO_ID in result.stderr, (
            f"the id is not present as one contiguous string:\n{result.stderr}"
        )
        assert "…" not in result.stderr, (
            f"an ellipsis means a cell was elided, so a row lost its identity:\n{result.stderr}"
        )
        # A folded id leaves fragments: the tail or the head alone on a line. If any
        # 12-character window of the id appears anywhere, the whole id must be there.
        windows = {LONG_VIDEO_ID[i:i + 12] for i in range(len(LONG_VIDEO_ID) - 11)}
        for line in lines:
            fragments = [window for window in windows if window in line]
            if fragments and LONG_VIDEO_ID not in line:
                raise AssertionError(
                    f"line holds part of the id, not the whole id: {line!r}\n"
                    f"fragments present: {sorted(fragments)[:3]}"
                )


#: The five states the state machine knows, cycled across the stages. The point is that
#: neighbouring stages differ, so a mark sitting in the wrong column is visible: a
#: constant row (all ``c``) would survive an off-by-one shift, and a shift is exactly the
#: failure a positional render can have. Cycling over ``STAGE_ORDER`` keeps this honest
#: when a stage is added -- the guard in the test below still demands it be non-constant.
_STATE_CYCLE = [STATUS_COMPLETED, STATUS_SKIPPED, STATUS_FAILED, STATUS_RUNNING,
                STATUS_PENDING]
DISTINCT_STATUSES = {name: _STATE_CYCLE[i % len(_STATE_CYCLE)]
                     for i, name in enumerate(STAGE_ORDER)}


class TestColumnAlignment:
    """The mark under index *i* is the state of stage *i*, and nothing else.

    Status is positional in this render, so an off-by-one column is a wrong answer about
    a specific stage — the kind of wrong answer that makes an operator re-run a good
    stage or trust a broken one.
    """

    def test_each_mark_sits_under_its_own_stage_index(self, tmp_path: Path) -> None:
        from multimodal_pipeline.config import load_config

        config_path, input_dir, output_dir = build_project(tmp_path, ["alpha"])
        write_state(output_dir, input_dir, "alpha", DISTINCT_STATUSES)
        names = enabled_stage_names(load_config(config_path))
        assert len(names) == len(STAGE_ORDER) == len(DISTINCT_STATUSES), (
            "the stage list and the status fixture no longer describe the same pipeline")

        result = invoke_status(config_path)
        assert result.exit_code == 0, result.stderr
        lines = result.stderr.splitlines()
        _, header = status_header(lines, len(names))
        marks = marks_for(lines, "alpha")

        assert len(marks) == len(names) + 1, (
            f"expected {len(names)} stage marks plus ov, got {marks}\n{result.stderr}"
        )
        # The pattern must actually be non-constant, or this test cannot see a shift.
        stage_marks = [_STATUS_MARKS[status] for status in DISTINCT_STATUSES.values()]
        assert len(set(stage_marks)) > 1, "the fixture state pattern is constant"

        for position, (token, column) in enumerate(marks[:len(names)]):
            expected = _STATUS_MARKS[DISTINCT_STATUSES[names[position]]]
            start_column = header[position][1]
            assert token == expected, (
                f"column {position + 1} ({names[position]}): header says '{expected}', "
                f"render printed '{token}'"
            )
            assert column == start_column, (
                f"column {position + 1} ({names[position]}): mark at column {column}, "
                f"its index is at column {start_column}"
            )

    def test_single_and_double_digit_indices_share_the_fixed_field(self, tmp_path: Path) -> None:
        """Width 3 is what keeps ``1`` and ``10`` in step; a variable field does not."""
        config_path, input_dir, output_dir = build_project(tmp_path, ["alpha"])
        write_state(output_dir, input_dir, "alpha",
                    {name: STATUS_COMPLETED for name in STAGE_ORDER})

        result = invoke_status(config_path)
        assert result.exit_code == 0, result.stderr
        lines = result.stderr.splitlines()
        header, columns = status_header(lines, len(STAGE_ORDER))
        starts = [column for _, column in columns]
        steps = {later - earlier for earlier, later in zip(starts, starts[1:])}
        assert steps == {3}, f"stage fields are not a fixed width of 3: {header!r} {steps}"
        assert " 10 " in header and " 1 " in header, header


class TestHeaderCompleteness:
    """rich dropped whole columns without saying so; the header count is now asserted."""

    @pytest.mark.parametrize("disabled", [{}, SOME_STAGES_DISABLED],
                            ids=["all-stages", "subset-disabled"])
    def test_the_header_numbers_every_enabled_stage(self, tmp_path: Path, disabled: dict) -> None:
        from multimodal_pipeline.config import load_config

        config_path, input_dir, output_dir = build_project(tmp_path, ["alpha", "beta"],
                                                          disabled=disabled)
        write_state(output_dir, input_dir, "alpha", {})
        write_state(output_dir, input_dir, "beta", {})
        names = enabled_stage_names(load_config(config_path))
        # Derived, never hardcoded: this is the assertion that would otherwise go stale
        # the week a stage is added, and a stale expected count proves nothing.
        assert len(names) == len(STAGE_ORDER) - len(disabled)

        result = invoke_status(config_path)
        assert result.exit_code == 0, result.stderr
        # Both videos are named, which is the assertion the broken render could not pass.
        assert "alpha" in result.stderr and "beta" in result.stderr, result.stderr

        _, columns = status_header(result.stderr.splitlines(), len(names))
        assert [token for token, _ in columns] == (
            [str(index) for index in range(1, len(names) + 1)] + ["ov"]
        )
        assert result.stdout.strip() == "", "human status belongs on stderr"


class TestOverallColumn:
    """``ov`` is the last column and it is aligned with the others."""

    def test_ov_holds_the_videos_overall_state_in_its_own_column(self, tmp_path: Path) -> None:
        from multimodal_pipeline.config import load_config

        config_path, input_dir, output_dir = build_project(tmp_path, ["alpha"])
        # One completed stage and nothing else settled: the overall state is partial,
        # a letter no single stage column shows here, so it cannot be confused with one.
        write_state(output_dir, input_dir, "alpha", {"metadata": STATUS_COMPLETED})
        names = enabled_stage_names(load_config(config_path))

        result = invoke_status(config_path)
        assert result.exit_code == 0, result.stderr
        lines = result.stderr.splitlines()
        _, header = status_header(lines, len(names))
        marks = marks_for(lines, "alpha")

        assert header[-1][0] == "ov" and header[-1][1] > header[-2][1], header
        overall_token, overall_column = marks[-1]
        assert overall_token == _STATUS_MARKS["partial"]
        assert overall_column == header[-1][1], (
            f"ov mark at column {overall_column}, its header is at {header[-1][1]}"
        )
        assert "ov=overall" in result.stderr, "the legend must explain ov as it explains the letters"
