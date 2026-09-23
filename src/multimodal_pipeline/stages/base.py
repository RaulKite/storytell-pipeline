"""Stage interface, dependency DAG and the reuse/invalidation decision.

A stage is a small, uniform contract — ``prepare / run / validate / outputs``
plus a ``config_fingerprint`` — so the orchestrator can decide resume-vs-rerun
for every stage without knowing anything about WhisperX or OpenPose.

Reuse requires *all* of:

1. recorded status ``completed``;
2. recorded own ``config_hash`` equals the current fingerprint;
3. recorded ``dependency_hash`` equals the current upstream fingerprint;
4. declared output artifacts still exist;
5. stage-specific semantic ``validate()`` passes.

Any single failure re-runs just that stage (and, transitively, its
dependants), which is what makes "changed the Whisper model" refresh the
transcript-dependent stages only.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..artifacts import ArtifactRegistry, VideoPaths
from ..config import PipelineConfig, stable_hash
from ..discovery import VideoSource
from ..exceptions import StageError, ValidationError, ValidationIssue
from ..state import STATUS_COMPLETED, VideoState

# Execution order for one video. Also the DAG's topological order.
STAGE_ORDER: tuple[str, ...] = (
    "metadata",
    "audio",
    "whisperx",
    "diarization",
    "speaker_assignment",
    "translation",
    "spacy_source",
    "spacy_english",
    "acoustic",
    "openpose",
    "finalization",
)

STAGE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "metadata": (),
    "audio": ("metadata",),
    "whisperx": ("audio",),
    "diarization": ("audio",),
    "speaker_assignment": ("whisperx", "diarization"),
    "translation": ("speaker_assignment",),
    "spacy_source": ("speaker_assignment",),
    "spacy_english": ("translation",),
    "acoustic": ("audio", "speaker_assignment"),
    "openpose": ("metadata",),
    "finalization": tuple(name for name in STAGE_ORDER if name != "finalization"),
}


@dataclass
class StageContext:
    """Everything a stage is allowed to touch."""

    config: PipelineConfig
    source: VideoSource
    paths: VideoPaths
    state: VideoState
    registry: ArtifactRegistry
    log: Callable[..., None]
    tools: dict[str, Any] = field(default_factory=dict)
    # Cross-stage in-memory hints (e.g. detected language) that are also
    # recoverable from disk, so resume works without them.
    scratch: dict[str, Any] = field(default_factory=dict)

    def artifact(self, name: str) -> Path:
        return self.paths.artifact(name)

    def input(self, name: str) -> Path:
        path = self.paths.artifact(name)
        if not path.exists():
            raise StageError(f"required input artifact missing: {name} ({path})")
        return path

    @property
    def video_id(self) -> str:
        return self.source.video_id


@dataclass
class StageOutcome:
    executed: bool
    status: str  # completed | reused | skipped | failed
    message: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ran(cls, **kw: Any) -> "StageOutcome":
        return cls(executed=True, status="completed", **kw)

    @classmethod
    def reused(cls, message: str = "valid previous result reused") -> "StageOutcome":
        return cls(executed=False, status="reused", message=message)

    @classmethod
    def skipped(cls, reason: str) -> "StageOutcome":
        return cls(executed=False, status="skipped", message=reason)


class Stage(abc.ABC):
    """Uniform processing unit."""

    name: str = ""
    #: Logical artifact names this stage consumes / produces.
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    #: Configuration keys whose change invalidates this stage.
    config_keys: tuple[str, ...] = ()

    @property
    def dependencies(self) -> tuple[str, ...]:
        return STAGE_DEPENDENCIES.get(self.name, ())

    # ------------------------------------------------------------------ hooks

    @abc.abstractmethod
    def config_fingerprint(self, ctx: StageContext) -> dict[str, Any]:
        """Values whose change forces this stage to rerun."""

    def enabled(self, ctx: StageContext) -> tuple[bool, str]:
        return True, ""

    def prepare(self, ctx: StageContext) -> None:  # noqa: D401 - optional hook
        """Optional pre-run step (mkdir, input checks, command construction)."""

    @abc.abstractmethod
    def execute(self, ctx: StageContext) -> dict[str, Any]:
        """Do the work; return provenance extras (tool_version/model_version/...)."""

    @abc.abstractmethod
    def validate(self, ctx: StageContext) -> dict[str, Any]:
        """Semantic validation. Raise :class:`ValidationError` when broken."""

    # ------------------------------------------------------------- execution

    def run(self, ctx: StageContext) -> StageOutcome:
        enabled, reason = self.enabled(ctx)
        if not enabled:
            return StageOutcome.skipped(reason or f"{self.name} disabled")
        ctx.paths.ensure_dirs()
        self.prepare(ctx)
        extras = self.execute(ctx) or {}
        validation = self.validate(ctx)
        return StageOutcome.ran(detail={"provenance": extras, "validation": validation})

    def outputs_present(self, ctx: StageContext) -> bool:
        for name in self.outputs:
            path = ctx.paths.get(name)
            if path is None or not path.exists():
                return False
        return True


def dependency_chain(stage: str) -> list[str]:
    """Stages that must exist for ``stage``, in canonical order."""
    seen: set[str] = set()
    stack = list(STAGE_DEPENDENCIES.get(stage, ()))
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(STAGE_DEPENDENCIES.get(current, ()))
    return [name for name in STAGE_ORDER if name in seen]


def dependants_of(stage: str) -> list[str]:
    """Stages that transitively consume ``stage`` output."""
    result: list[str] = []
    frontier = [name for name in STAGE_ORDER if stage in STAGE_DEPENDENCIES.get(name, ())]
    while frontier:
        current = frontier.pop()
        if current in result:
            continue
        result.append(current)
        frontier.extend(name for name in STAGE_ORDER if current in STAGE_DEPENDENCIES.get(name, ()))
    return [name for name in STAGE_ORDER if name in result]


def stage_selection(
    order: Sequence[str],
    *,
    only_stage: Iterable[str] | None = None,
    from_stage: str | None = None,
    to_stage: str = "finalization",
) -> list[str]:
    """Resolve CLI stage controls into an explicit ordered allow-list.

    ``--only-stage`` wins over ``--from-stage/--to-stage``; a range always
    includes its endpoints and stays inside the canonical order.
    """
    only = [s for s in order if s in set(only_stage or ())]
    if only:
        # A stage is useless without its prerequisites, so keep dependencies.
        keep = set(only)
        for name in only:
            keep.update(dependency_chain(name))
        return [s for s in order if s in keep]
    if from_stage:
        if from_stage not in order:
            raise StageError(f"unknown --from-stage: {from_stage}")
        start = order.index(from_stage)
        end = order.index(to_stage) if to_stage in order else len(order) - 1
        if end < start:
            raise StageError(f"--to-stage {to_stage} precedes --from-stage {from_stage}")
        return list(order[start : end + 1])
    return list(order)


def should_reuse(
    stage: Stage,
    ctx: StageContext,
    *,
    config_hash: str,
    dependency_hash: str,
    force: bool,
) -> tuple[bool, str]:
    """Apply the five-part reuse test to decide reuse vs execution."""
    if force:
        return False, "forced recomputation"
    record = ctx.state.stage(stage.name)
    if record.status != STATUS_COMPLETED:
        return False, f"status is {record.status}"
    if record.config_hash != config_hash:
        return False, "configuration changed"
    if record.dependency_hash != dependency_hash:
        return False, "upstream dependency changed"
    if not stage.outputs_present(ctx):
        return False, "output artifacts missing"
    try:
        stage.validate(ctx)
    except ValidationError as exc:
        return False, f"validation failed: {exc}"
    except StageError as exc:  # unreadable outputs are simply not reusable
        return False, f"validation error: {exc}"
    return True, "valid previous result"


def fingerprint_config(ctx: StageContext, keys: Sequence[str]) -> dict[str, Any]:
    """Pull named sub-configs out of the pipeline config for hashing."""
    payload: dict[str, Any] = {}
    for key in keys:
        section = getattr(ctx.config, key, None)
        if section is None:
            continue
        model = getattr(section, "model_dump", None)
        payload[key] = model(mode="json") if callable(model) else section
    return payload


def dependency_hash_for(stage: Stage, ctx: StageContext) -> str:
    """Hash of every upstream stage's config hash: one change ripples down."""
    payload = {
        dep: ctx.state.stage(dep).config_hash for dep in dependency_chain(stage.name)
    }
    payload["schema_version"] = ctx.tools.get("schema_version")
    return stable_hash(payload, length=16)


# --------------------------------------------------------------------------
# Worker-backed stages
# --------------------------------------------------------------------------

#: Keys written into a preserved raw artifact so a later run can decide whether
#: the expensive native result is still valid for the current configuration.
RAW_REQUEST_KEY = "_pipeline_request"
RAW_PROVENANCE_KEY = "_pipeline_provenance"


def utc_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class WorkerStage(Stage):
    """Stage whose heavy work runs in an isolated uv worker environment.

    Contract: :meth:`request` describes the work (and is hashed), the worker
    writes a native artifact which is preserved verbatim plus two
    ``_pipeline_*`` bookkeeping keys, and :meth:`normalize` turns that raw file
    into canonical Parquet. Normalisation is therefore always rerunnable from
    disk after a schema change, with no model re-run.
    """

    #: Logical name of the preserved raw artifact.
    raw_artifact: str = ""
    #: Environment variable names holding credentials that must be forwarded.
    required_env: tuple[str, ...] = ()

    # ------------------------------------------------------------- subclass API

    def request(self, ctx: StageContext) -> dict[str, Any]:
        """Everything that can change the native output (also the fingerprint)."""
        raise NotImplementedError

    def worker_argv(self, ctx: StageContext, raw_path: Path, request_digest: str) -> list[str]:
        raise NotImplementedError

    def worker_environment(self, ctx: StageContext) -> dict[str, str]:
        return {}

    def normalize(self, ctx: StageContext) -> dict[str, Any]:
        raise NotImplementedError

    # ------------------------------------------------------------------ plumbing

    def config_fingerprint(self, ctx: StageContext) -> dict[str, Any]:
        return self.request(ctx)

    def request_digest(self, ctx: StageContext) -> str:
        from ..config import stable_hash

        return stable_hash(self.request(ctx), length=16)

    def uv_project(self, ctx: StageContext) -> Path:
        raise NotImplementedError

    def worker_script(self, ctx: StageContext) -> Path:
        raise NotImplementedError

    def python_version(self, ctx: StageContext) -> str | None:
        return None

    def worker_timeout(self) -> float | None:
        return None

    # ---------------------------------------------------------------- execution

    def prepare(self, ctx: StageContext) -> None:
        raw_path = ctx.artifact(self.raw_artifact)
        raw_path.parent.mkdir(parents=True, exist_ok=True)

    def execute(self, ctx: StageContext) -> dict[str, Any]:
        from ..config import stable_hash

        request = self.request(ctx)
        digest = stable_hash(request, length=16)
        raw_path = ctx.artifact(self.raw_artifact)
        raw_path.parent.mkdir(parents=True, exist_ok=True)

        reused = raw_request_matches(raw_path, digest)
        if reused:
            ctx.log(f"reusing existing raw output for request {digest} (no model run)")
        else:
            self.run_model(ctx, request, raw_path, digest)

        summary = self.normalize(ctx) or {}
        provenance = raw_provenance(raw_path) or {}
        return {
            "tool_version": provenance.get("tool_version"),
            "model_version": provenance.get("model_version"),
            "command": provenance.get("command"),
            "executable": provenance.get("executable"),
            "exit_code": provenance.get("exit_code"),
            "extra": {"request_hash": digest, "raw_reused": reused, **summary},
        }

    def run_model(self, ctx: StageContext, request: dict[str, Any], raw_path: Path,
                  digest: str) -> None:
        from ..uv_worker import run_worker, worker_result_path

        cfg_env = self._env_block(ctx)
        argv = self.worker_argv(ctx, raw_path, digest)
        result_path = worker_result_path(raw_path.parent, f"{self.name}_worker_result.json")
        ctx.log(f"running {self.name} worker")
        try:
            worker = run_worker(
                uv_project=self.uv_project(ctx),
                worker_script=self.worker_script(ctx),
                args=argv,
                log_path=ctx.paths.log(self.name),
                result_path=result_path,
                python_version=self.python_version(ctx),
                env=self.worker_environment(ctx),
                timeout=self.worker_timeout(),
                cwd=ctx.config.project_root,
            )
        except Exception as exc:  # noqa: BLE001 - translated into a recorded failure
            details = dict(getattr(exc, "details", {}) or {})
            details["request"] = _mask_request(request)
            raise StageError(f"{self.name} worker failed: {exc}", details=details) from exc
        if not raw_path.is_file():
            raise StageError(f"{self.name} worker reported success but raw artifact is missing: {raw_path}")
        stamp_raw(raw_path, request=request, digest=digest, worker=worker)

    def _env_block(self, ctx: StageContext) -> dict[str, str]:
        return {}

    # ---------------------------------------------------------------- validation

    def validate_raw(self, ctx: StageContext) -> dict[str, Any]:
        from ..artifacts import read_json

        raw_path = ctx.artifact(self.raw_artifact)
        if not raw_path.is_file():
            raise ValidationError(self.name, [f"raw artifact missing: {raw_path.name}"])
        try:
            return read_json(raw_path)
        except (OSError, ValueError) as exc:
            raise ValidationError(self.name, [f"raw artifact unreadable ({raw_path.name}): {exc}"]) from exc


def raw_sidecar_path(raw_path: Path) -> Path:
    """``whisperx.json`` → ``whisperx.json.provenance.json``.

    Provenance lives in a sidecar so the native artifact stays byte-identical to
    what the tool produced: auditable against the tool's own output, and safe to
    hand to anyone expecting the tool's schema.
    """
    return raw_path.with_name(raw_path.name + ".provenance.json")


def raw_request_matches(raw_path: Path, digest: str) -> bool:
    """Is this preserved native output the product of exactly this request?"""
    if not raw_path.is_file():
        return False
    sidecar = _read_sidecar(raw_path)
    if sidecar is None:
        return False
    if sidecar.get("request_hash") != digest:
        return False
    # Guard against a deleted-and-replaced raw file with a stale sidecar.
    recorded, current = sidecar.get("raw_sha256"), _sha256(raw_path)
    return recorded is None or current is None or recorded == current


def raw_provenance(raw_path: Path) -> dict[str, Any] | None:
    sidecar = _read_sidecar(raw_path)
    return sidecar.get("provenance") if isinstance(sidecar, dict) else None


def _read_sidecar(raw_path: Path) -> dict[str, Any] | None:
    from ..artifacts import read_json

    path = raw_sidecar_path(raw_path)
    if not path.is_file():
        return None
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _sha256(path: Path) -> str | None:
    try:
        from ..stages.metadata import sha256_of

        return sha256_of(path)
    except (OSError, ImportError):  # pragma: no cover - unreadable artifact
        return None


def stamp_raw(raw_path: Path, *, request: dict[str, Any], digest: str, worker: Any) -> None:
    """Write the sidecar describing which request produced a raw artifact."""
    from ..artifacts import atomic_write_json

    atomic_write_json(raw_sidecar_path(raw_path), {
        "schema_version": "1.0",
        "raw_artifact": raw_path.name,
        "raw_sha256": _sha256(raw_path),
        "request_hash": digest,
        "request": _mask_request(request),
        "recorded_at": utc_timestamp(),
        "provenance": {
            "tool_version": worker.tool_version if worker else None,
            "model_version": worker.model_version if worker else None,
            "command": worker.argv_masked if worker else None,
            "executable": "uv",
            "exit_code": worker.exit_code if worker else None,
            "duration_seconds": round(worker.duration_seconds, 3) if worker else None,
            "uv_project": str(worker.argv[3]) if worker and len(worker.argv) > 3 else None,
        },
    })


def _mask_request(request: dict[str, Any]) -> dict[str, Any]:
    from ..config import mask_secrets

    return mask_secrets(request)
