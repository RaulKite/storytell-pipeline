"""Stage construction, DAG execution and resume decisions for one video.

Execution is strictly sequential: one video at a time, one stage at a time,
because every GPU-heavy stage (WhisperX, Pyannote, spaCy transformers,
OpenPose) would otherwise contend for the single RTX 4090.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .artifacts import ArtifactRegistry, VideoPaths
from .config import PipelineConfig, stable_hash
from .discovery import VideoSource
from .log import get_logger
from .state import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_SKIPPED,
    VideoState,
    utc_now,
)
from .stages.acoustic import AcousticStage
from .stages.audio import AudioStage
from .stages.base import (
    STAGE_DEPENDENCIES,
    STAGE_ORDER,
    Stage,
    StageContext,
    StageError,
    StageOutcome,
    dependency_chain,
    fingerprint_config,
    should_reuse,
    stage_selection,
)
from .stages.diarization import DiarizationStage
from .stages.finalization import FinalizationStage
from .stages.metadata import MetadataStage
from .stages.openpose import OpenPoseStage
from .stages.speaker_assignment import SpeakerAssignmentStage
from .stages.spacy_english import SpacyEnglishStage
from .stages.spacy_source import SpacySourceStage
from .stages.translation import TranslationStage
from .stages.whisperx import WhisperXStage

log = get_logger("orchestrator")

STAGE_CLASSES: dict[str, type[Stage]] = {
    cls.name: cls
    for cls in (
        MetadataStage,
        AudioStage,
        WhisperXStage,
        DiarizationStage,
        SpeakerAssignmentStage,
        TranslationStage,
        SpacySourceStage,
        SpacyEnglishStage,
        AcousticStage,
        OpenPoseStage,
        FinalizationStage,
    )
}


@dataclass
class StagePlan:
    name: str
    will_run: bool
    reason: str
    config_hash: str
    dependency_hash: str
    forced: bool = False


@dataclass
class VideoResult:
    video_id: str
    source_path: str
    dataset_dir: str
    status: str  # completed | partial | failed
    stage_outcomes: dict[str, str] = field(default_factory=dict)
    durations: dict[str, float] = field(default_factory=dict)
    errors: dict[str, dict[str, Any]] = field(default_factory=dict)
    total_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "source_path": self.source_path,
            "dataset_dir": self.dataset_dir,
            "status": self.status,
            "stages": dict(self.stage_outcomes),
            "duration_seconds": round(self.total_seconds, 2),
            "stage_durations": self.durations,
            "errors": self.errors,
        }


def build_stages() -> list[Stage]:
    return [STAGE_CLASSES[name]() for name in STAGE_ORDER]


class VideoRunner:
    """Runs (or resumes) the whole stage sequence for a single video."""

    def __init__(
        self,
        config: PipelineConfig,
        source: VideoSource,
        *,
        dataset_root: Path | None = None,
        tools: dict[str, Any] | None = None,
        progress: Callable[[str, str, str], None] | None = None,
    ) -> None:
        self.config = config
        self.source = source
        self.paths = VideoPaths(dataset_root or (config.output.directory / source.video_id))
        self.state = VideoState.load(self.paths, source.video_id, str(source.path))
        self.registry = ArtifactRegistry(self.paths)
        self.tools = tools or {}
        self.progress = progress or (lambda *_: None)
        self.stage_logger_factory: Callable[[str], Any] | None = None
        self.started_at: str = utc_now()

    # ------------------------------------------------------------------ planning

    def plan(self, *, only_stage: Sequence[str] | None = None, from_stage: str | None = None,
             to_stage: str = "finalization", force_stages: Sequence[str] = ()) -> list[StagePlan]:
        """Decide, per stage, whether it runs — and say why (used by ``status``)."""
        context = self.stage_context()
        stages = build_stages()
        context.state.bind_stages(STAGE_ORDER)
        selected = set(stage_selection(STAGE_ORDER, only_stage=only_stage, from_stage=from_stage, to_stage=to_stage))
        planned: list[StagePlan] = []
        for stage in stages:
            config_hash = self.config_hash_for(stage, context)
            dependency_hash = self.dependency_hash_for(stage, context)
            forced = stage.name in set(force_stages)
            if stage.name not in selected:
                planned.append(StagePlan(stage.name, False, "outside requested stage range",
                                         config_hash, dependency_hash, forced))
                continue
            hash_reason = self._hash_reason(context, stage, config_hash, dependency_hash)
            if forced:
                reason = "forced recomputation"
            elif hash_reason:
                reason = hash_reason
            else:
                enabled, why = stage.enabled(context)
                reason = "" if enabled else f"disabled: {why}"
            planned.append(StagePlan(stage.name, not reason, reason or "valid previous result",
                                     config_hash, dependency_hash, forced))
        return planned

    def _hash_reason(self, context: StageContext, stage: Stage, config_hash: str,
                     dependency_hash: str) -> str:
        record = context.state.stage(stage.name)
        if record.status != STATUS_COMPLETED:
            return f"status is {record.status}"
        if record.config_hash != config_hash:
            return "configuration changed"
        if record.dependency_hash != dependency_hash:
            return "upstream dependency changed"
        if not stage.outputs_present(context):
            return "output artifacts missing"
        return ""

    def config_hash_for(self, stage: Stage, context: StageContext) -> str:
        return stable_hash(self.config_payload_for(stage, context), length=16)

    def config_payload_for(self, stage: Stage, context: StageContext) -> dict[str, Any]:
        """Stage fingerprint = stage config + resolved tool identity + source identity."""
        payload: dict[str, Any] = dict(stage.config_fingerprint(context) or {})
        payload.setdefault("stage", stage.name)
        payload.setdefault("schema_version", self.tools.get("schema_version"))
        return payload

    def dependency_hash_for(self, stage: Stage, context: StageContext) -> str:
        payload = {dep: context.state.stage(dep).config_hash for dep in dependency_chain(stage.name)}
        payload["schema_version"] = self.tools.get("schema_version")
        return stable_hash(payload, length=16)

    # ----------------------------------------------------------------- execution

    def run(self, *, only_stage: Sequence[str] | None = None, from_stage: str | None = None,
            to_stage: str = "finalization", force_stages: Sequence[str] = ()) -> VideoResult:
        self.paths.ensure_dirs()
        self.registry.refresh()
        logger = self.stage_logger_factory or _default_logger_factory(self.paths)
        context = StageContext(
            config=self.config,
            source=self.source,
            paths=self.paths,
            state=self.state,
            registry=self.registry,
            log=_NullLogger(),
            tools=self.tools,
        )
        self.state.bind_stages(STAGE_ORDER)
        selected = set(stage_selection(STAGE_ORDER, only_stage=only_stage, from_stage=from_stage, to_stage=to_stage))
        forced = set(force_stages)

        result = VideoResult(
            video_id=self.source.video_id,
            source_path=str(self.source.path),
            dataset_dir=str(self.paths.dataset_dir),
            status=STATUS_COMPLETED,
        )
        started = time.time()
        for stage in build_stages():
            stage_log = logger(stage.name)
            context.log = _StageLogAdapter(stage_log)
            outcome = self._run_stage(stage, context, stage_log, selected, forced)
            self.registry.refresh()
            record = self.state.stage(stage.name)
            result.stage_outcomes[stage.name] = record.status
            result.durations[stage.name] = record.duration_seconds or 0.0
            if record.error:
                result.errors[stage.name] = record.error
            stage_log.close()
            if record.status == STATUS_FAILED and stage.name in selected:
                # A failed stage poisons its dependants; stop this video's path,
                # report the video as partial, and let the batch continue.
                self._mark_dependants_skipped(stage, context, selected, "upstream stage failed", result)
                break
        result.total_seconds = round(time.time() - started, 2)
        result.status = self.state.overall_status
        return result

    def _run_stage(self, stage: Stage, context: StageContext, stage_log: Any,
                   selected: set[str], forced: set[str]) -> StageOutcome:
        name = stage.name
        if name not in selected:
            if self.state.stage(name).status != STATUS_COMPLETED:
                self.state.mark_skipped(name, "outside requested stage range")
            self.progress(name, STATUS_SKIPPED, "outside requested stage range")
            return StageOutcome.skipped("outside requested stage range")

        config_hash = self.config_hash_for(stage, context)
        dependency_hash = self.dependency_hash_for(stage, context)
        enabled, why_disabled = stage.enabled(context)
        if not enabled:
            self.state.mark_skipped(name, why_disabled or "disabled by configuration")
            self.progress(name, STATUS_SKIPPED, why_disabled or "disabled")
            return StageOutcome.skipped(why_disabled or "disabled")

        force = name in forced
        if force:
            self.state.reset_stage(name)
        reuse, reason = should_reuse(stage, context, config_hash=config_hash,
                                     dependency_hash=dependency_hash, force=force)
        if reuse:
            self.state.mark_reused(name)
            self.progress(name, STATUS_COMPLETED, "reused")
            stage_log(f"reuse: {reason}")
            return StageOutcome.reused(reason)

        stage_log(f"start ({reason})")
        self.progress(name, STATUS_RUNNING, reason)
        self.state.mark_running(name)
        try:
            outcome = stage.run(context)
        except Exception as exc:  # noqa: BLE001 - every failure becomes recorded state
            self.state.mark_failed(
                name,
                {
                    "type": type(exc).__name__,
                    "message": str(exc)[:2000],
                    "details": getattr(exc, "details", {}) or {},
                    "stderr_tail": getattr(getattr(exc, "result", None), "stderr_tail", "")[:4000],
                    "log_file": str(self.paths.log(name)),
                    "traceback": traceback.format_exc()[-4000:],
                },
                exit_code=getattr(getattr(exc, "result", None), "returncode", None),
                command=getattr(getattr(exc, "result", None), "argv_masked", None),
            )
            self.progress(name, STATUS_FAILED, str(exc)[:80])
            stage_log(f"failed: {exc}")
            return StageOutcome(executed=True, status="failed", message=str(exc))

        record = self.state.stage(name)
        record.config_hash = config_hash
        record.dependency_hash = dependency_hash
        record.input_artifacts = list(stage.inputs)
        record.output_artifacts = list(stage.outputs)
        extras = outcome.detail.get("provenance") or {}
        record.tool_version = extras.get("tool_version")
        record.model_version = extras.get("model_version")
        if extras.get("command"):
            record.command = extras["command"]
        if extras.get("executable"):
            record.executable = extras["executable"]
        if extras.get("exit_code") is not None:
            record.exit_code = extras["exit_code"]
        record.validation_result = outcome.detail.get("validation")
        if record.status != STATUS_COMPLETED:
            # ``execute`` may have run outside mark_running (rare); settle it now.
            self.state.mark_completed(
                name,
                config_hash=config_hash,
                dependency_hash=dependency_hash,
                input_artifacts=stage.inputs,
                output_artifacts=stage.outputs,
                tool_version=extras.get("tool_version"),
                model_version=extras.get("model_version"),
                validation_result=outcome.detail.get("validation"),
                command=extras.get("command"),
                executable=extras.get("executable"),
                exit_code=extras.get("exit_code"),
            )
        else:
            self.state.save()
        self.progress(name, STATUS_COMPLETED, "done")
        stage_log("completed")
        return outcome

    def _mark_dependants_skipped(self, stage: Stage, context: StageContext, selected: set[str],
                                 reason: str, result: VideoResult) -> None:
        blocked = [name for name in dependency_chain("finalization") if name in selected]
        for name in blocked:
            record = self.state.stage(name)
            if record.status in {"pending", "running"}:
                self.state.mark_skipped(name, reason)
                result.stage_outcomes[name] = STATUS_SKIPPED
                self.progress(name, STATUS_SKIPPED, reason)

    # ------------------------------------------------------------------- context

    def stage_context(self, logger: Any | None = None) -> StageContext:
        self.paths.ensure_dirs()
        self.registry.refresh()
        self.state.bind_stages(STAGE_ORDER)
        return StageContext(
            config=self.config,
            source=self.source,
            paths=self.paths,
            state=self.state,
            registry=self.registry,
            log=logger or _NullLogger(),
            tools=self.tools,
        )


class _NullLogger:
    def __call__(self, message: str, level: int = 20) -> None:
        log.debug(message)


class _StageLogAdapter:
    """Lets stages write to both their file log and the console progress line."""

    def __init__(self, stage_log: Any) -> None:
        self._stage_log = stage_log

    def __call__(self, message: str, level: int = 20) -> None:
        self._stage_log(str(message))
        if level >= 30:
            log.warning(message)


def _default_logger_factory(paths: VideoPaths):
    from .log import make_stage_logger_factory

    return make_stage_logger_factory(paths.dataset_dir / "logs")


def stage_names() -> tuple[str, ...]:
    return STAGE_ORDER


def enabled_stage_names(config: PipelineConfig) -> list[str]:
    """Stages that configuration allows to run (for ``status`` reporting)."""
    mapping = {
        "whisperx": config.whisperx.enabled,
        "diarization": config.diarization.enabled,
        "translation": config.translation.enabled,
        "spacy_source": config.spacy.enabled and config.spacy.process_source,
        "spacy_english": config.spacy.enabled and config.spacy.process_english,
        "acoustic": config.acoustic.enabled,
        "openpose": config.openpose.enabled,
    }
    return [name for name in STAGE_ORDER if mapping.get(name, True)]
