"""``multimodal-pipeline`` command-line interface.

Every command is a thin wrapper over the orchestrator so a shell script and a
human get identical behaviour. Stage controls (``--only-stage``,
``--force-stage``, ``--from-stage``/``--to-stage``) are understood by the
planner, not by individual stages.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .artifacts import VideoPaths
from .config import PipelineConfig, load_config
from .discovery import VideoSource, discover_single, discover_videos
from .log import configure, get_logger
from .orchestrator import VideoRunner, build_stages, enabled_stage_names
from .stages.base import artifact_owners, integrity_problems
from .provenance import system_report, tools_report
from .report import BatchReport, collect_report, write_batch_report
from .state import STATUS_COMPLETED, STATUS_PENDING, STATUS_SKIPPED, VideoState

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Sequential multimodal video-processing pipeline.")
console = Console(stderr=True)
# Machine-readable output goes to stdout so `--json | jq` and CI capture work;
# the human tables stay on stderr, where diagnostics belong.
out_console = Console(stderr=False)
log = get_logger("cli")

def _stage_list(value: Optional[str]) -> list[str] | None:
    if not value:
        return None
    names = [item.strip() for item in value.split(",") if item.strip()]
    from .stages.base import STAGE_ORDER

    unknown = [name for name in names if name not in STAGE_ORDER]
    if unknown:
        raise typer.BadParameter(f"unknown stage(s): {', '.join(unknown)}. Known: {', '.join(STAGE_ORDER)}")
    return names


def _check_stage(name: Optional[str], flag: str) -> None:
    from .stages.base import STAGE_ORDER

    if name and name not in STAGE_ORDER:
        raise typer.BadParameter(f"{flag}: unknown stage '{name}'. Known: {', '.join(STAGE_ORDER)}")


def load_config_or_exit(path: Optional[Path]) -> PipelineConfig:
    """Load a config, turning every expected failure into one readable line.

    A typo in a YAML key is the single most common way to break a run, and the
    default pydantic traceback hides the one sentence that matters (which key,
    in which file) behind 60 lines of internals. Anything that is not a config
    error is re-raised untouched: a real bug must still show its stack.
    """
    import yaml as _yaml
    from pydantic import ValidationError as _ValidationError

    from .exceptions import ConfigError as _ConfigError

    try:
        return load_config(path)
    except FileNotFoundError as exc:
        _fail(str(exc))
    except _ConfigError as exc:
        # A malformed .env line otherwise surfaces as a bare traceback, and the
        # operator cannot tell which of their credential lines the pipeline rejected.
        _fail(str(exc))
    except _yaml.YAMLError as exc:
        _fail(f"config is not valid YAML ({path}): {_one_line(str(exc).replace(chr(10), ' '))}")
    except _ValidationError as exc:
        _fail(f"invalid configuration ({path}):\n{_config_problems(exc)}")
    raise AssertionError("unreachable")  # pragma: no cover


def _config_problems(exc: BaseException) -> str:
    lines = []
    for error in exc.errors():  # type: ignore[attr-defined]
        location = ".".join(str(part) for part in error.get("loc", ())) or "config"
        detail = _one_line(error.get("msg", "invalid"))
        if error.get("type") == "extra_forbidden":
            detail = f"unknown setting (known: {_known_settings(location)})"
        lines.append(f"  {location}: {detail}")
    return "\n".join(lines[:12])


def _known_settings(location: str) -> str:
    """Name the valid keys of the section that got a typo."""
    from .config import PipelineConfig

    node: Any = PipelineConfig
    for part in location.split("."):
        children = getattr(node, "model_fields", {})
        if part not in children:
            return ", ".join(sorted(children)) or "none"
        annotation = children[part].annotation
        node = annotation if isinstance(annotation, type) else object
    return ", ".join(sorted(getattr(node, "model_fields", {}))) or "none"


def _one_line(text: str) -> str:
    return " ".join(str(text).split())[:300]


def _fail(message: str) -> None:
    console.print(f"[red]error:[/] {message}")
    raise typer.Exit(code=2)


def _sources(config: PipelineConfig, video: Optional[Path]) -> list[VideoSource]:
    if video is not None:
        return [discover_single(video, config)]
    return discover_videos(config)


def _tools(config: PipelineConfig) -> dict[str, Any]:
    """Collected once per batch and shared by every video (system probing is slow)."""
    return {
        "schema_version": "1.0",
        "pipeline_version": __version__,
        "run_started_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "tools": tools_report(config),
    }


def _progress_printer(show: bool = True):
    """Return ``runner_factory``-friendly progress callback printing the stage table."""
    if not show:
        return lambda *args: None

    state: dict[str, Any] = {"video_header_printed": False}

    def callback(stage: str, status: str, note: str) -> None:
        if not state["video_header_printed"]:
            state["video_header_printed"] = True
        icon = {"completed": "+", "running": ">", "failed": "x", "skipped": "-", "pending": "."}.get(status, " ")
        console.print(f"    [dim]{icon}[/dim] {stage:<20} {status:<10} [dim]{note}[/dim]")

    return callback


def _print_banner(config: PipelineConfig, sources: list[VideoSource]) -> None:
    console.print(f"[bold]Configuration:[/bold] {config.config_path}")
    console.print(f"[bold]Input:[/bold]    {config.input.directory}")
    console.print(f"[bold]Output:[/bold]   {config.output.directory}")
    console.print(f"[bold]Videos:[/bold]   {len(sources)}   [bold]Execution:[/bold] {config.execution.mode}"
                  f"   [bold]GPU:[/bold] {config.execution.gpu}")
    console.print()


def _print_video_header(index: int, total: int, source: VideoSource) -> None:
    console.print(f"[bold cyan][{index}/{total}][/] [bold]{source.filename}[/]")


def _print_video_footer(runner: VideoRunner, result) -> None:
    console.print(f"    [bold]{result.status}[/] in {_human_duration(result.total_seconds)}")
    console.print(f"    Dataset: {result.dataset_dir}")
    console.print()


def _human_duration(seconds: float) -> str:
    seconds = int(seconds)
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:02d}m {secs:02d}s"
    return f"{secs}s"


# --------------------------------------------------------------------- commands


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="YAML configuration file."),
    video: Optional[Path] = typer.Option(None, "--video", help="Process a single video file instead of the input directory."),
    only_stage: Optional[str] = typer.Option(None, "--only-stage", help="Run only this stage plus its prerequisites (comma separated)."),
    force_stage: Optional[str] = typer.Option(None, "--force-stage", help="Recompute these stages even if valid."),
    from_stage: Optional[str] = typer.Option(None, "--from-stage", help="Start the stage range here."),
    to_stage: str = typer.Option("finalization", "--to-stage", help="End the stage range here."),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress the per-stage console table."),
) -> None:
    """Process every discovered video, sequentially, one stage at a time."""
    configure("INFO", console=not quiet)
    pipeline_config = load_config_or_exit(config)
    sources = _sources(pipeline_config, video)
    if not sources:
        console.print(f"[yellow]No videos found in {config.input.directory}[/]")
        raise typer.Exit(code=1)
    _print_banner(pipeline_config, sources)
    report = _execute_batch(pipeline_config, sources, only_stage=_stage_list(only_stage),
                            force_stages=_stage_list(force_stage) or [], from_stage=from_stage,
                            to_stage=to_stage, quiet=quiet)
    _finish(report, pipeline_config)


@app.command()
def resume(
    config: Path = typer.Option(..., "--config", "-c"),
    video: Optional[Path] = typer.Option(None, "--video"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Continue an interrupted run, reusing every stage whose result is still valid."""
    run(config=config, video=video, only_stage=None, force_stage=None, from_stage=None,
        to_stage=None, quiet=quiet)


@app.command("retry-failed")
def retry_failed(
    config: Path = typer.Option(..., "--config", "-c"),
    video: Optional[Path] = typer.Option(None, "--video"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Force-recompute exactly the stages that failed, then their dependants."""
    configure("INFO", console=not quiet)
    pipeline_config = load_config_or_exit(config)
    sources = _sources(pipeline_config, video)
    forced: list[str] = []
    targets: list[VideoSource] = []
    for source in sources:
        state = VideoState.load(VideoPaths(pipeline_config.output.directory / source.video_id),
                               source.video_id, str(source.path))
        failed = [name for name in state.stage_order if state.stage(name).status == "failed"]
        if not failed:
            continue
        from .stages.base import dependants_of

        targets.append(source)
        for name in failed:
            forced.extend(dependants_of(name))
    forced = sorted(set(forced))
    if not targets:
        console.print("[green]Nothing to retry: no failed stages found.[/]")
        raise typer.Exit(code=0)
    console.print(f"[bold]Retrying[/] {len(forced)} stage(s) across {len(targets)} video(s): "
                  f"{', '.join(forced)}")
    report = _execute_batch(pipeline_config, targets, only_stage=forced, force_stages=forced,
                            from_stage=None, to_stage=None, quiet=quiet)
    _finish(report, pipeline_config)


@app.command("process-video")
def process_video(
    video: Path = typer.Argument(..., help="Video file to process."),
    config: Path = typer.Option(..., "--config", "-c"),
    only_stage: Optional[str] = typer.Option(None, "--only-stage"),
    force_stage: Optional[str] = typer.Option(None, "--force-stage"),
    from_stage: Optional[str] = typer.Option(None, "--from-stage"),
    to_stage: str = typer.Option("finalization", "--to-stage"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Process one video file (the smoke-test entry point)."""
    run(config=config, video=video, only_stage=only_stage, force_stage=force_stage,
        from_stage=from_stage, to_stage=to_stage, quiet=quiet)


@app.command()
def status(
    config: Path = typer.Option(..., "--config", "-c"),
    video: Optional[str] = typer.Option(None, "--video-id", help="Limit to one video id."),
    plan: bool = typer.Option(False, "--plan", help="Explain why each stage would or would not rerun."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Show per-video, per-stage processing state.

    Statuses are rendered as single letters with a legend: eleven stage names in
    full cannot fit a terminal, and a table nobody can read tells nobody anything.
    Use ``--plan`` for the wordy version and ``--json`` for scripting.
    """
    configure("WARNING", console=False)
    pipeline_config = load_config_or_exit(config)
    sources = discover_videos(pipeline_config)
    if video:
        sources = [source for source in sources if source.video_id == video]
        if not sources:
            console.print(f"[red]No video with id {video}[/]")
            raise typer.Exit(code=1)
    names = enabled_stage_names(pipeline_config)
    rows = []
    for source in sources:
        paths = VideoPaths(pipeline_config.output.directory / source.video_id)
        state = VideoState.load(paths, source.video_id, str(source.path))
        rows.append((source.video_id, [state.status_of(name) for name in names], state.overall_status))

    if as_json:
        out_console.print_json(json.dumps({
            "output_directory": str(pipeline_config.output.directory),
            "stages": names,
            "videos": [{"video_id": vid, "stages": dict(zip(names, cells)), "overall": overall}
                       for vid, cells, overall in rows],
        }))
        return

    table = Table(title=f"Pipeline status — {pipeline_config.output.directory}", expand=False)
    table.add_column("video_id", overflow="fold")
    for name in names:
        table.add_column(_stage_initials(name), justify="center", width=max(2, len(_stage_initials(name))))
    table.add_column("overall", justify="center")
    for vid, cells, overall in rows:
        table.add_row(vid, *[_STATUS_MARKS.get(cell, cell[:3]) for cell in cells],
                      _STATUS_MARKS.get(overall, overall[:3]))
    console.print(table)
    console.print("  " + "  ".join(f"{code}={word}" for word, code in sorted(_STATUS_MARKS.items())))
    console.print("  stages: " + " ".join(f"{i+1}={name}" for i, name in enumerate(names)))
    if plan:
        for source in sources:
            runner = VideoRunner(pipeline_config, source, tools=_tools(pipeline_config))
            console.print(f"\n[bold]{source.video_id}[/]")
            for item in runner.plan(only_stage=None, from_stage=None, to_stage=None):
                marker = "run " if item.will_run else "keep"
                console.print(f"  [{marker}] {item.name:<20} {item.reason}")


#: One letter per status keeps an eleven-stage table inside a terminal width.
_STATUS_MARKS = {
    "completed": "c",
    "skipped": "s",
    "failed": "F",
    "running": "r",
    "pending": ".",
    "partial": "P",
}


_SHORT_STAGE_NAMES = {
    "metadata": "meta",
    "audio": "audio",
    "whisperx": "asr",
    "diarization": "diar",
    "speaker_assignment": "spk",
    "translation": "trans",
    "spacy_source": "nlp_src",
    "spacy_english": "nlp_en",
    "acoustic": "acou",
    "openpose": "pose",
    "finalization": "final",
}


def _stage_initials(name: str) -> str:
    """Short, readable column headers for an eleven-stage table."""
    return _SHORT_STAGE_NAMES.get(name, name[:6])



@app.command()
def validate(
    config: Path = typer.Option(..., "--config", "-c"),
    video: Optional[Path] = typer.Option(None, "--video"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable report on stdout."),
) -> None:
    """Re-run semantic validation over artifacts already on disk (no processing).

    A stage that was legitimately skipped (disabled by configuration, or blocked
    by a missing credential) is reported as skipped, not as a failure: validating
    the outputs of a stage that never ran would make every partial dataset look
    broken and hide the problems that are real.
    """
    configure("WARNING", console=False)
    pipeline_config = load_config_or_exit(config)
    sources = _sources(pipeline_config, video)
    failures = 0
    results: list[dict[str, Any]] = []
    for source in sources:
        runner = VideoRunner(pipeline_config, source, tools=_tools(pipeline_config))
        context = runner.stage_context()
        from .log import StageLogger as _StageLogger

        logger = _StageLogger(runner.paths.dataset_dir / "logs", "validate")
        context.log = logger
        problems: dict[str, str] = {}
        skipped: list[str] = []
        # Artifacts a later stage rewrote in place must not be judged against the
        # earlier stage's fingerprint.
        owners = artifact_owners(context.state)
        for stage in build_stages():
            status = context.state.stage(stage.name).status
            if status == STATUS_SKIPPED:
                reason = (context.state.stage(stage.name).validation_result or {}).get("reason", "")
                skipped.append(f"{stage.name} ({reason})" if reason else stage.name)
                continue
            if status == STATUS_PENDING:
                skipped.append(f"{stage.name} (never run)")
                continue
            if not stage.outputs_present(context):
                problems[stage.name] = "outputs missing"
                continue
            changed = integrity_problems(context, stage, owners=owners)
            if changed:
                problems[stage.name] = "; ".join(changed[:3])
                continue
            try:
                stage.validate(context)
            except Exception as exc:  # noqa: BLE001 - aggregated into the report
                problems[stage.name] = str(exc)[:300]
        logger.close()
        results.append({"video_id": source.video_id, "ok": not problems,
                        "problems": problems, "skipped": skipped})
        if problems:
            failures += 1
            console.print(f"[red]FAIL[/] {source.video_id}")
            for name, message in problems.items():
                console.print(f"    {name}: {message}")
        else:
            console.print(f"[green]OK[/]   {source.video_id}")
        for item in skipped:
            console.print(f"    [dim]skipped:[/] {item}")
    if as_json:
        out_console.print_json(json.dumps({
            "output_directory": str(pipeline_config.output.directory),
            "ok": not failures,
            "results": results,
        }))
    raise typer.Exit(code=1 if failures else 0)


@app.command("inspect-environment")
def inspect_environment(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Report discovered tools, models and GPU/CUDA versions (as JSON)."""
    configure("WARNING", console=False)
    payload: dict[str, Any] = {"system": system_report()}
    if config is not None:
        pipeline_config = load_config_or_exit(config)
        payload["tools"] = tools_report(pipeline_config)
        payload["environment_warnings"] = _environment_warnings(pipeline_config)
    else:
        payload["note"] = "pass --config for the full tool/OpenPose inventory"
    out_console.print_json(json.dumps(payload, ensure_ascii=False))
    return payload


def _environment_warnings(config: PipelineConfig) -> list[str]:
    """Things that will make a stage skip or degrade, reported *before* a long run starts.

    Ordered by how much the operator has to act on them: credentials and endpoints they
    must supply, then installs they must provide, then the silent-quality-loss cases
    (a stage that will run with no language model, which produces empty linguistics that
    look like results).
    """
    import os

    warnings: list[str] = []
    if config.diarization.enabled and not os.environ.get(config.diarization.hf_token_env):
        warnings.append(f"{config.diarization.hf_token_env} is not set: diarization will be skipped")
    if config.translation.enabled and not config.translation.endpoint_configured:
        warnings.append("translation endpoint is not configured: translation will be skipped")
    if not Path(config.input.directory).is_dir():
        warnings.append(f"input directory does not exist: {config.input.directory}")
    from .provenance import openpose_report

    discovery = openpose_report(config)
    if config.openpose.enabled and not discovery.get("executable"):
        warnings.append(f"OpenPose binary not found under {config.openpose.root}")
    warnings.extend(_uv_project_warnings(config))
    warnings.extend(_activespeaker_warnings(config))
    warnings.extend(_spacy_warnings(config))
    return warnings


def _uv_project_warnings(config: PipelineConfig) -> list[str]:
    """Warn only about uv project *directories* that are missing.

    A project that exists but has never been synced is not warned about: `uv run
    --project` resolves and syncs it on first use, verified on this machine against a
    throwaway project. Only the disabled-stage filter is a judgement call — a stage the
    operator switched off does not need its environment, and warning about it buries the
    warnings that matter.
    """
    warnings: list[str] = []
    for name, cfg in config.stage_configs.items():
        project = getattr(cfg, "uv_project", None)
        if project is None:
            continue
        if not getattr(cfg, "enabled", True):
            continue
        if not config.resolve(project).is_dir():
            warnings.append(f"{name}: uv project missing at {config.resolve(project)}")
    return warnings


def _activespeaker_warnings(config: PipelineConfig) -> list[str]:
    """Say up front what ``ActiveSpeakerStage.enabled`` will decide at runtime.

    The reasons are the stage's own strings, reused rather than reworded: two copies of
    this gate would drift, and the whole point is that the pre-flight warning and the
    skip reason agree.
    """
    cfg = config.activespeaker
    if not cfg.enabled:
        return []
    if cfg.talknet_root is None:
        return [
            "activespeaker.talknet_root is not set: active speaker detection will be "
            "skipped (point it at a TalkNet-ASD checkout to enable it)"
        ]
    root = Path(cfg.talknet_root)
    if not root.is_dir():
        return [f"activespeaker.talknet_root does not exist: {root}"]
    if not (root / "run_talknet.py").is_file():
        return [
            f"activespeaker.talknet_root has no run_talknet.py: {root} "
            "(is it a TalkNet-ASD checkout?) — active speaker detection will be skipped"
        ]
    return []


def _spacy_warnings(config: PipelineConfig) -> list[str]:
    """Warn when the **English** spaCy model is missing, and only when English is coming.

    The English variant is the one case where the model is known before a run: its input
    language is always ``en``, and it runs exactly when the translation stage produces
    segments. So "english_model is not installed + translation is configured" predicts a
    real, avoidable outcome — ``linguistic/english/*`` would carry empty lemmas/POS/dep —
    and naming it here is worth a line of output.

    Source-language models are deliberately **not** inventoried. The configured mapping is
    a default dictionary covering eight languages; warning for every name that is not
    installed told this operator five things about German, French, Italian, Dutch and
    Portuguese while their corpus held English and Spanish. The language is only known
    after transcription, so that warning belongs at the moment the language is known: the
    spaCy worker already records ``selected_model`` and ``model_selection_status`` in its
    raw output and the stage logs the model per video, which is where a ``blank`` fallback
    can be read against the language it was chosen for.

    A configured model that has an installed same-family substitute is not warned about:
    that is a full pipeline for the language, and the substitution is already recorded in
    provenance as ``substituted_family``.
    """
    from .stages.spacy_source import installed_model_inventory, select_model

    cfg = config.spacy
    if not cfg.enabled or not cfg.process_english:
        return []
    # No translation output means no English text, so the English model would never run.
    if not (config.translation.enabled and config.translation.endpoint_configured):
        return []
    installed = set(installed_model_inventory(config.resolve(cfg.uv_project)))
    decision = select_model("en", {"en": cfg.english_model}, installed, fallback=cfg.fallback_model)
    if decision["status"] not in {"fallback_missing_model", "fallback_no_model"}:
        return []
    return [
        f"spaCy model {cfg.english_model} is not installed in "
        f"{config.resolve(cfg.uv_project)}: English linguistics fall back to "
        f"'{decision['model']}' — tokenization and sentences only, empty lemmas/POS/"
        f"dependencies. Run scripts/install_spacy_models.sh {cfg.english_model} to install it."
    ]


# ------------------------------------------------------------------ internals


def _execute_batch(config: PipelineConfig, sources: list[VideoSource], *, only_stage=None,
                   force_stages=None, from_stage=None, to_stage=None,
                   quiet: bool = False) -> BatchReport:
    tools = _tools(config)
    tools["schema_version"] = "1.0"
    tools["run_started_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    started = time.time()
    results = []
    total = len(sources)
    for index, source in enumerate(sources, start=1):
        _print_video_header(index, total, source)
        runner = VideoRunner(config, source, tools=tools, progress=_progress_printer(not quiet))
        try:
            result = runner.run(only_stage=only_stage, force_stages=force_stages or [],
                                from_stage=from_stage, to_stage=to_stage)
        except Exception as exc:  # noqa: BLE001 - one bad video must not end the batch
            log.exception("video %s crashed outside stage execution", source.video_id)
            from .orchestrator import VideoResult

            result = VideoResult(video_id=source.video_id, source_path=str(source.path),
                                 dataset_dir=str(config.output.directory / source.video_id),
                                 status="failed", errors={"pipeline": {"type": type(exc).__name__,
                                                                       "message": str(exc)[:1000]}})
        results.append(result)
        _print_video_footer(runner, result)
        if result.status == "failed" and config.execution.stop_on_video_error:
            console.print("[red]execution.stop_on_video_error is set: stopping the batch[/]")
            break
    report = collect_report(results, discovered=len(sources), elapsed_seconds=time.time() - started,
                            output_directory=config.output.directory)
    write_batch_report(report)
    return report


def _finish(report: BatchReport, config: PipelineConfig) -> None:
    console.print("[bold]Processing complete[/]")
    console.print(f"  Videos discovered: {report.videos_discovered}")
    console.print(f"  Completed:         {report.completed}")
    console.print(f"  Partial:           {report.partial}")
    console.print(f"  Failed:            {report.failed}")
    console.print(f"  Total time:        {_human_duration(report.elapsed_seconds)}")
    for entry in report.entries:
        if entry["status"] != STATUS_COMPLETED:
            console.print(f"\n  [yellow]{entry['video_id']}[/]")
            for stage, message in (entry.get("errors") or {}).items():
                detail = message.get("message") if isinstance(message, dict) else message
                console.print(f"    {stage}: {detail}")
    console.print(f"\nBatch report:\n  {report.path}")
    if report.failed or report.partial:
        raise typer.Exit(code=2)


def app_main() -> None:  # pragma: no cover - console-script shim
    try:
        app()
    except typer.Exit:
        raise
    except KeyboardInterrupt:  # pragma: no cover - interactive
        console.print("[red]Interrupted[/]")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    app_main()
