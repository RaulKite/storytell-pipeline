"""Check current documentation contracts against code, not local run history."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from multimodal_pipeline.artifacts import ARTIFACT_LAYOUT
from multimodal_pipeline.stages.base import STAGE_DEPENDENCIES, STAGE_ORDER

ROOT = Path(__file__).resolve().parents[2]
GUIDE_PATHS = (
    "docs/getting-started.md", "docs/cli.md", "docs/datasets.md", "docs/modalities.md",
    "docs/elan.md", "docs/architecture.md", "docs/configuration.md", "docs/development.md",
)
DOCUMENTATION = "\n\n".join(
    (ROOT / path).read_text(encoding="utf-8") for path in ("README.md", *GUIDE_PATHS)
)
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def fenced_blocks(language: str) -> list[str]:
    return re.findall(rf"^```{language}\n(.*?)^```", DOCUMENTATION, re.DOTALL | re.MULTILINE)


def options(command: str) -> set[str]:
    import typer.main
    from multimodal_pipeline.cli import app

    target = typer.main.get_command(app).commands[command]
    return {opt for param in target.params for opt in param.opts}


def documented_dependencies(text: str) -> dict[str, tuple[str, ...]]:
    """Read the explicitly headed stage/dependency table, excluding other tables."""
    section = text.split("## Stage dependencies\n", 1)[1].split("\n## ", 1)[0]
    rows = re.findall(r"^\| `([a-z_]+)` \| ([^|]+) \|", section, re.MULTILINE)
    dependencies: dict[str, tuple[str, ...]] = {}
    for name, cell in rows:
        dependencies[name] = (tuple(dependencies) if cell.strip() == "All producer stages above"
                              else tuple(re.findall(r"`([a-z_]+)`", cell)))
    return dependencies


def documented_tables(text: str) -> dict[str, str]:
    rows = re.findall(r"^\| `([a-z_]+)` \| `([^`]+\.parquet)` \|", text, re.MULTILINE)
    return dict(rows)


def test_license_matches_the_project() -> None:
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("MIT License")
    assert re.search(r"^Copyright \(c\) \d{4} .+", text, re.MULTILINE)
    assert PYPROJECT["project"]["license"]["text"] == "MIT"
    assert "Not yet declared" not in DOCUMENTATION


def test_console_script_is_the_documented_entry_point() -> None:
    assert PYPROJECT["project"]["scripts"]["multimodal-pipeline"] == "multimodal_pipeline.cli:app_main"


@pytest.mark.parametrize("block", fenced_blocks("bash"))
def test_shell_examples_invoke_the_cli_through_uv_run(block: str) -> None:
    offenders = [line for line in block.splitlines()
                 if re.search(r"(?<![\w./-])multimodal-pipeline\s+\w", line)
                 and "uv run" not in line and not line.lstrip().startswith("#")]
    assert not offenders, offenders


def test_install_instructions_cover_every_isolated_environment() -> None:
    environments = {p.name for p in (ROOT / "environments").iterdir() if p.is_dir()}
    synced = set(re.findall(r"environments/(\w+)\s+&& uv sync", DOCUMENTATION))
    assert synced == environments
    from multimodal_pipeline.config import PipelineConfig

    config = PipelineConfig.model_validate({"input": {"directory": "/tmp/in"},
                                           "output": {"directory": "/tmp/out"}})
    for dotted in re.findall(r"`([a-z_]+\.[a-z_]+)`", DOCUMENTATION):
        section, field = dotted.split(".")
        if field in {"py", "json", "yaml", "md"}:
            continue  # filenames are not settings
        if hasattr(config, section):
            sub = type(getattr(config, section))
            if hasattr(sub, "model_fields"):
                assert field in sub.model_fields, dotted
    for prerequisite in ("HF_TOKEN", "flite", "openpose.root", "activespeaker.talknet_root",
                         "install_spacy_models.sh", ".env.example", "config/config.example.yaml"):
        assert prerequisite in DOCUMENTATION


def test_cli_table_lists_exactly_the_shipped_commands() -> None:
    from multimodal_pipeline.cli import app

    real = {command.name or command.callback.__name__ for command in app.registered_commands}
    text = (ROOT / "docs/cli.md").read_text(encoding="utf-8")
    section = text.split("## Commands\n", 1)[1].split("\n## ", 1)[0]
    documented = set(re.findall(r"^\| `([a-z-]+)(?: <file>)?`", section, re.MULTILINE))
    assert documented == real


def test_run_flags_are_real_and_process_video_has_no_video_flag() -> None:
    for flag in ("--only-stage", "--from-stage", "--to-stage", "--force-stage"):
        assert flag in DOCUMENTATION
        assert flag in options("run")
        assert flag in options("process-video")
    assert "--video" in options("run")
    assert "--video" not in options("process-video")
    assert "--plan" in options("status")
    for command in ("status", "validate"):
        assert "--json" in options(command)
    assert "--json" not in options("inspect-environment")
    assert "no flag needed" in DOCUMENTATION


def test_architecture_dependencies_match_the_dag_in_execution_order() -> None:
    text = (ROOT / "docs/architecture.md").read_text(encoding="utf-8")
    documented = documented_dependencies(text)
    assert tuple(documented) == STAGE_ORDER
    assert documented == STAGE_DEPENDENCIES


def test_dependency_guard_catches_a_wrong_visual_dependency() -> None:
    text = "## Stage dependencies\n| `openpose` | `whisperx` | wrong |\n"
    assert documented_dependencies(text)["openpose"] != STAGE_DEPENDENCIES["openpose"]


def test_producer_shorthand_expands_only_preceding_documented_stages() -> None:
    text = ("## Stage dependencies\n| `metadata` | — | source |\n"
            "| `audio` | `metadata` | extract |\n"
            "| `finalization` | All producer stages above | summarize |\n"
            "| `elan` | `finalization` | export |\n")
    assert documented_dependencies(text)["finalization"] == ("metadata", "audio")


def test_dataset_reference_lists_every_registered_parquet_table() -> None:
    text = (ROOT / "docs/datasets.md").read_text(encoding="utf-8")
    expected = {name: path for name, path in ARTIFACT_LAYOUT.items() if path.endswith(".parquet")}
    assert documented_tables(text) == expected


def test_table_guard_catches_a_wrong_path() -> None:
    text = "| `speech_words` | `speech/segments.parquet` | words |"
    assert documented_tables(text)["speech_words"] != ARTIFACT_LAYOUT["speech_words"]


def test_no_unregistered_parquet_filename_is_documented() -> None:
    known = {Path(path).stem for path in ARTIFACT_LAYOUT.values() if path.endswith(".parquet")}
    # Python's pyarrow.parquet module is not a dataset filename.
    prose = re.sub(r"^\s*(?:import|from) .*$", "", DOCUMENTATION, flags=re.MULTILINE)
    listed = set(re.findall(r"([a-z_]+)\.parquet", prose))
    assert listed <= known


def test_working_grid_and_join_rule_remain_explicit() -> None:
    assert re.search(r"join on `source_timestamp`,? never (?:on )?`frame_number`", DOCUMENTATION)
    assert not re.search(r"join on `frame_number`,? never (?:on )?`source_timestamp`", DOCUMENTATION)
    mentions = [line for line in DOCUMENTATION.splitlines() if "25 FPS" in line]
    assert mentions
    claims = [line for line in mentions if "row" in line or "dense" in line]
    assert claims
    assert all("grid" in line or "source" in line for line in claims)
    stage = (ROOT / "src/multimodal_pipeline/stages/activespeaker.py").read_text()
    assert '"frame_number": row.get("frame_25fps")' in stage
    assert "OUTPUT_FPS = 25" in (ROOT / "workers/activespeaker_worker.py").read_text()


def test_test_commands_use_collection_instead_of_remembered_counts() -> None:
    text = (ROOT / "docs/development.md").read_text()
    assert "--collect-only" in text
    assert not re.search(r"\b\d+ (?:unit |CLI-driven )?tests\b", text)
    for suite in ("tests/unit", "tests/e2e"):
        assert f"pytest {suite} -q -p no:randomly" in text


def test_coverage_omissions_and_unknowns_are_documented() -> None:
    text = (ROOT / "docs/elan.md").read_text()
    for term in ("pipeline-coverage", "pipeline-overlap-projection", "not evaluated",
                 "unknown", "imputed", "stories", "linguistic", "100 ms", "low median"):
        assert term in text
    assert "not a word gloss" in text


def test_user_guides_do_not_embed_local_run_receipts() -> None:
    assert "Output, verbatim, against the local dataset" not in DOCUMENTATION
    assert "five of the seven datasets" not in DOCUMENTATION
    assert "Measured corpus census" not in DOCUMENTATION
