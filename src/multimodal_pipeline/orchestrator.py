"""Stage construction, DAG execution and resume decisions for one video.

Execution is strictly sequential: one video at a time, one stage at a time,
because every GPU-heavy stage (WhisperX, Pyannote, spaCy transformers,
OpenPose) would otherwise contend for the single RTX 4090.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
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
    STAGE_ORDER,
    Stage,
    StageContext,
    StageOutcome,
    dependency_chain,
    should_reuse,
    stage_selection,
)
from .stages.diarization import DiarizationStage
from .stages.diarization_nemotron import DiarizationNemotronStage
from .stages.finalization import FinalizationStage
from .stages.metadata import MetadataStage
from .stages.activespeaker import ActiveSpeakerStage
from .stages.openpose import OpenPoseStage
from .stages.pose_normalized import PoseNormalizedStage
from .stages.speaker_assignment import SpeakerAssignmentStage
from .stages.speaker_fusion import SpeakerFusionStage
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
        DiarizationNemotronStage,
        SpeakerAssignmentStage,
        TranslationStage,
        SpacySourceStage,
        SpacyEnglishStage,
        AcousticStage,
        OpenPoseStage,
        PoseNormalizedStage,
        ActiveSpeakerStage,
        SpeakerFusionStage,
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
        #: Stages that failed during this run; poisons only their dependants.
        self._failed: set[str] = set()

    # ------------------------------------------------------------------ planning

    def plan(self, *, only_stage: Sequence[str] | None = None, from_stage: str | None = None,
             to_stage: str | None = None, force_stages: Sequence[str] = ()) -> list[StagePlan]:
        """Decide, per stage, whether it runs — and say why (used by ``status``).

        ``_hash_reason`` names why a previous result is *not* reusable, which is
        precisely why the stage must run; only an explicit disable or an outside-
        range selection prevents it. Keeping that straight matters because ``status``
        is how an operator decides whether to intervene.
        """
        context = self.stage_context()
        stages = build_stages()
        context.state.bind_stages(STAGE_ORDER)
        selected = set(stage_selection(STAGE_ORDER, only_stage=only_stage, from_stage=from_stage, to_stage=to_stage))
        planned: list[StagePlan] = []
        blocked: set[str] = set()
        # Stages decided so far in this same pass. A dependant's dependency hash
        # must be evaluated against them, otherwise ``status`` promises "valid
        # previous result" for a stage the very next ``run`` recomputes.
        pending: dict[str, tuple[str | None, int | None]] = {}
        counter = context.state.sequence
        for stage in stages:
            config_hash = self.config_hash_for(stage, context)
            dependency_hash = self.dependency_hash_for(stage, context, pending)
            forced = stage.name in set(force_stages)
            if stage.name not in selected:
                planned.append(StagePlan(stage.name, False, "outside requested stage range",
                                         config_hash, dependency_hash, forced))
                continue
            upstream_failed = sorted(blocked & set(dependency_chain(stage.name)))
            if upstream_failed:
                blocked.add(stage.name)
                planned.append(StagePlan(stage.name, False,
                                         f"blocked by failed upstream: {', '.join(upstream_failed)}",
                                         config_hash, dependency_hash, forced))
                continue
            enabled, why_disabled = stage.enabled(context)
            if not enabled:
                planned.append(StagePlan(stage.name, False, f"disabled: {why_disabled}",
                                         config_hash, dependency_hash, forced))
                continue
            if forced:
                reason = "forced recomputation"
            else:
                reason = self._hash_reason(context, stage, config_hash, dependency_hash)
            if context.state.stage(stage.name).status == STATUS_FAILED:
                reason = reason or "previous run failed"
            if reason:
                counter += 1
                pending[stage.name] = (config_hash, counter)
                if context.state.stage(stage.name).status == STATUS_FAILED:
                    blocked.add(stage.name)
                planned.append(StagePlan(stage.name, True, reason, config_hash, dependency_hash, forced))
            else:
                planned.append(StagePlan(stage.name, False, "valid previous result",
                                         config_hash, dependency_hash, forced))
        return planned

    def _hash_reason(self, context: StageContext, stage: Stage, config_hash: str,
                     dependency_hash: str) -> str:
        """Why this stage cannot reuse its previous result, or "" if it can.

        Delegates to :func:`should_reuse` so ``status`` reports exactly what a
        following ``run`` will do. A plan that ignored semantic validation would
        tell an operator "valid previous result" about a dataset the next run
        recomputes anyway.
        """
        reusable, reason = should_reuse(stage, context, config_hash=config_hash,
                                        dependency_hash=dependency_hash, force=False)
        return "" if reusable else reason

    def config_hash_for(self, stage: Stage, context: StageContext) -> str:
        return stable_hash(self.config_payload_for(stage, context), length=16)

    def config_payload_for(self, stage: Stage, context: StageContext) -> dict[str, Any]:
        """Stage fingerprint = stage config + resolved tool identity + source identity."""
        payload: dict[str, Any] = dict(stage.config_fingerprint(context) or {})
        payload.setdefault("stage", stage.name)
        payload.setdefault("schema_version", self.tools.get("schema_version"))
        return payload

    def dependency_hash_for(self, stage: Stage, context: StageContext,
                            pending: dict[str, tuple[str | None, int | None]] | None = None) -> str:
        """Fingerprint of everything this stage consumes upstream.

        For each transitive dependency this combines the configuration that
        produced it *and that stage's execution sequence number*. Configuration
        alone misses the resume case that matters most: a stage rerun with
        identical settings (``--force-stage``, or a crash mid-write) whose outputs
        really did change, so its dependants must follow. A sequence counter is
        used rather than artifact mtimes because this filesystem rounds mtimes to
        roughly 16 ms: a rerun finishing inside one tick would otherwise look
        unchanged and leave stale Parquet on disk.

        ``pending`` carries the (config, sequence) a planning pass has already
        decided to assign to upstream stages, so a plan reflects the cascade.
        """
        payload: dict[str, Any] = {"schema_version": self.tools.get("schema_version")}
        for name in dependency_chain(stage.name):
            record = context.state.stage(name)
            override = (pending or {}).get(name)
            payload[name] = list(override) if override else [record.config_hash, record.run_sequence]
        return stable_hash(payload, length=16)

    # ----------------------------------------------------------------- execution

    def run(self, *, only_stage: Sequence[str] | None = None, from_stage: str | None = None,
            to_stage: str | None = None, force_stages: Sequence[str] = ()) -> VideoResult:
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
            if record.status == STATUS_FAILED:
                # Record the poison and keep going: independent branches of the DAG
                # (OpenPose from metadata, audio from the source) still run.
                self._failed.add(stage.name)
        result.total_seconds = round(time.time() - started, 2)
        result.status = self.state.overall_status
        # The loop's log handlers are closed; summaries are written to the video log.
        context.log = _NullLogger()
        self._refresh_summary(context)
        return result

    def _refresh_summary(self, context: StageContext) -> None:
        """Rewrite manifest/provenance so they describe the video as it finished.

        ``finalization`` runs inside the stage loop, so the manifest it writes can
        only ever record itself as ``running`` and the overall status as whatever
        preceded it. Without this closing rewrite every dataset on disk misreports
        its own completion forever.
        """
        if not self.paths.manifest.is_file():
            return  # the dataset was never finalized; nothing to correct
        from .stages.finalization import write_dataset_summary

        try:
            write_dataset_summary(context)
        except Exception as exc:  # noqa: BLE001 - a stale summary must not mask results
            log.warning("%s: could not refresh manifest: %s", self.source.video_id, exc)

    def _run_stage(self, stage: Stage, context: StageContext, stage_log: Any,
                   selected: set[str], forced: set[str]) -> StageOutcome:
        name = stage.name
        if name not in selected:
            if self.state.stage(name).status != STATUS_COMPLETED:
                self.state.mark_skipped(name, "outside requested stage range")
            self.progress(name, STATUS_SKIPPED, "outside requested stage range")
            return StageOutcome.skipped("outside requested stage range")

        # A failure poisons only the stages that actually consume its output.
        # OpenPose depends on metadata alone, so a transcription failure must not
        # stop pose extraction on an otherwise healthy video.
        blocked_by = sorted(self._failed & set(dependency_chain(name)))
        if blocked_by:
            reason = f"upstream stage failed: {', '.join(blocked_by)}"
            self.state.mark_skipped(name, reason)
            self.progress(name, STATUS_SKIPPED, reason)
            stage_log(f"blocked: {reason}")
            return StageOutcome.skipped(reason)

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
        # Allocated per real execution, so a dependant's dependency hash moves even
        # when the rerun used identical configuration.
        run_sequence = self.state.next_sequence()
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
        record.run_sequence = run_sequence
        record.output_row_counts = stage.output_row_counts(context)
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
                run_sequence=run_sequence,
                output_row_counts=record.output_row_counts,
            )
        else:
            self.state.save()
        self.progress(name, STATUS_COMPLETED, "done")
        stage_log("completed")
        return outcome

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
    """Stages that configuration allows to run (for ``status`` reporting).

    Every stage with its own enable check has to appear here or the status table lists it
    as enabled when `Stage.enabled` will report a skip reason — the mapping's default is
    True, so a new optional stage is invisible to this view unless it is added.
    """
    mapping = {
        "whisperx": config.whisperx.enabled,
        "diarization": config.diarization.enabled,
        "translation": config.translation.enabled,
        "spacy_source": config.spacy.enabled and config.spacy.process_source,
        "spacy_english": config.spacy.enabled and config.spacy.process_english,
        "acoustic": config.acoustic.enabled,
        "openpose": config.openpose.enabled,
        # Its own enable check, so it has to be listed or the status table calls it
        # enabled while Stage.enabled reports a skip reason (the mapping defaults to True).
        "pose_normalized": config.pose_normalized.enabled,
        "speaker_fusion": config.speaker_fusion.enabled,
    }
    return [name for name in STAGE_ORDER if mapping.get(name, True)]
