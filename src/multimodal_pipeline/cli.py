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
from .provenance import system_report, tools_report
from .report import BatchReport, collect_report, write_batch_report
from .state import STATUS_COMPLETED, VideoState

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Sequential multimodal video-processing pipeline.")
console = Console(stderr=True)
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
    pipeline_config = load_config(config, overrides=None)
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
        to_stage="finalization", quiet=quiet)


@app.command("retry-failed")
def retry_failed(
    config: Path = typer.Option(..., "--config", "-c"),
    video: Optional[Path] = typer.Option(None, "--video"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Force-recompute exactly the stages that failed, then their dependants."""
    configure("INFO", console=not quiet)
    pipeline_config = load_config(config)
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
                            from_stage=None, to_stage="finalization", quiet=quiet)
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
) -> None:
    """Show per-video, per-stage processing state."""
    configure("WARNING", console=False)
    pipeline_config = load_config(config)
    sources = discover_videos(pipeline_config)
    if video:
        sources = [source for source in sources if source.video_id == video]
        if not sources:
            console.print(f"[red]No video with id {video}[/]")
            raise typer.Exit(code=1)
    table = Table(title=f"Pipeline status — {pipeline_config.output.directory}")
    table.add_column("video_id")
    for name in enabled_stage_names(pipeline_config):
        table.add_column(name.replace("_", "\n"), justify="center", overflow="fold")
    table.add_column("overall", justify="center")
    for source in sources:
        paths = VideoPaths(pipeline_config.output.directory / source.video_id)
        state = VideoState.load(paths, source.video_id, str(source.path))
        cells = [state.status_of(name) for name in enabled_stage_names(pipeline_config)]
        table.add_row(source.video_id, *cells, state.overall_status)
    console.print(table)
    if plan:
        for source in sources:
            runner = VideoRunner(pipeline_config, source, tools=_tools(pipeline_config))
            console.print(f"\n[bold]{source.video_id}[/]")
            for item in runner.plan(only_stage=None, from_stage=None, to_stage="finalization"):
                marker = "run " if item.will_run else "keep"
                console.print(f"  [{marker}] {item.name:<20} {item.reason}")


@app.command()
def validate(
    config: Path = typer.Option(..., "--config", "-c"),
    video: Optional[Path] = typer.Option(None, "--video"),
) -> None:
    """Re-run semantic validation over artifacts already on disk (no processing)."""
    configure("WARNING", console=False)
    pipeline_config = load_config(config)
    sources = _sources(pipeline_config, video)
    failures = 0
    for source in sources:
        runner = VideoRunner(pipeline_config, source, tools=_tools(pipeline_config))
        context = runner.stage_context()
        from .log import StageLogger as _StageLogger

        logger = _StageLogger(runner.paths.dataset_dir / "logs", "validate")
        context.log = logger
        problems: dict[str, str] = {}
        for stage in build_stages():
            if not stage.outputs_present(context):
                problems[stage.name] = "outputs missing"
                continue
            try:
                stage.validate(context)
            except Exception as exc:  # noqa: BLE001 - aggregated into the report
                problems[stage.name] = str(exc)[:300]
        logger.close()
        if problems:
            failures += 1
            console.print(f"[red]FAIL[/] {source.video_id}")
            for name, message in problems.items():
                console.print(f"    {name}: {message}")
        else:
            console.print(f"[green]OK[/]   {source.video_id}")
    raise typer.Exit(code=1 if failures else 0)


@app.command("inspect-environment")
def inspect_environment(config: Optional[Path] = typer.Option(None, "--config", "-c")) -> None:
    """Report discovered tools, models and GPU/CUDA versions (as JSON)."""
    configure("WARNING", console=False)
    payload: dict[str, Any] = {"system": system_report()}
    if config is not None:
        pipeline_config = load_config(config)
        payload["tools"] = tools_report(pipeline_config)
        payload["environment_warnings"] = _environment_warnings(pipeline_config)
    else:
        payload["note"] = "pass --config for the full tool/OpenPose inventory"
    console.print_json(json.dumps(payload, ensure_ascii=False))


def _environment_warnings(config: PipelineConfig) -> list[str]:
    """Things that will make a stage skip, reported *before* a long run starts."""
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
    for name, cfg in config.stage_configs.items():
        project = getattr(cfg, "uv_project", None)
        if project is not None and not config.resolve(project).is_dir():
            warnings.append(f"{name}: uv project missing at {config.resolve(project)}")
    return warnings


# ------------------------------------------------------------------ internals


def _execute_batch(config: PipelineConfig, sources: list[VideoSource], *, only_stage=None,
                   force_stages=None, from_stage=None, to_stage="finalization",
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
