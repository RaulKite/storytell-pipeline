"""Typed runtime configuration with YAML loading and secret masking.

The orchestrator validates the whole configuration at startup: every stage
reads its own typed sub-model, so a typo in ``config.local.yaml`` fails before
any GPU work starts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Values whose keys match this pattern are masked in logs/provenance/manifests.
SECRET_KEY_RE = re.compile(r"(api[_-]?key|token|secret|password|authorization|access[_-]?key)", re.I)

# ...unless the key names a *holder* rather than a value: ``hf_token_env``
# stores the name of an environment variable, which is not a secret and is
# needed to diagnose why a stage skipped.
SECRET_HOLDER_RE = re.compile(r"(_env|_var|_variable|_env_var)$", re.I)

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

PLACEHOLDER_VALUES = {
    "",
    "your_api_key",
    "your-api-key",
    "changeme",
    "none",
    "null",
    "litellm_host:port",
}


def interpolate_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` inside YAML values."""
    if isinstance(value, str):

        def _sub(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            return os.environ.get(name, default if default is not None else match.group(0))

        return _ENV_RE.sub(_sub, value)
    if isinstance(value, dict):
        return {k: interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate_env(v) for v in value]
    return value


class _Model(BaseModel):
    """Base model: unknown keys are an error, so config drift fails loudly.

    ``validate_default`` keeps declared defaults inside the same invariants as
    user-supplied values (normalised extensions, validated signal ranges), so a
    stage fingerprint never depends on whether a key was written in the YAML.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True, validate_default=True)


class InputConfig(_Model):
    directory: Path
    recursive: bool = False
    extensions: list[str] = Field(default_factory=lambda: [".mp4", ".mov", ".mkv", ".avi", ".webm"])

    @field_validator("extensions")
    @classmethod
    def _normalise_extensions(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        for ext in value:
            ext = ext.strip().lower()
            if not ext:
                continue
            out.append(ext if ext.startswith(".") else f".{ext}")
        if not out:
            raise ValueError("input.extensions must contain at least one entry")
        return sorted(set(out))


class OutputConfig(_Model):
    directory: Path
    temp_directory: Path | None = None


class ExecutionConfig(_Model):
    mode: str = "sequential"
    gpu: int = 0
    # Fail the whole batch on the first video-level crash (still isolates stage failures).
    stop_on_video_error: bool = False
    keep_temp_files: bool = False

    @field_validator("mode")
    @classmethod
    def _mode_sequential(cls, value: str) -> str:
        if value != "sequential":
            raise ValueError("execution.mode currently supports only 'sequential'")
        return value


class FFmpegConfig(_Model):
    executable: str = "ffmpeg"
    ffprobe: str = "ffprobe"


class WhisperXConfig(_Model):
    enabled: bool = True
    uv_project: Path = Path("environments/whisperx")
    worker: Path = Path("workers/whisperx_worker.py")
    model: str = "large-v3"
    language: str = "auto"
    device: str = "cuda"
    device_index: int = 0
    compute_type: str = "float16"
    batch_size: str | int = "auto"
    beam_size: int = 5
    # WhisperX alignment models are language specific; unsupported languages
    # fall back to unaligned words with an explicit alignment_status.
    align_model: str | None = None
    # VAD is always applied by WhisperX 3.8.x; this chooses which VAD model and
    # how long the merged speech chunks may be (its ``chunk_size``).
    vad_method: str = "pyannote"
    vad_merge_chunk_seconds: int = 30
    threads: int = 4
    download_root: Path | None = None
    # Raw dict forwarded to ``load_model(asr_options=...)``.
    asr_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("vad_method")
    @classmethod
    def _vad(cls, value: str) -> str:
        if value not in {"pyannote", "silero"}:
            raise ValueError("whisperx.vad_method must be 'pyannote' or 'silero'")
        return value
    python_version: str = "3.12"
    extra_args: list[str] = Field(default_factory=list)


class DiarizationConfig(_Model):
    enabled: bool = True
    uv_project: Path = Path("environments/diarization")
    worker: Path = Path("workers/diarization_worker.py")
    provider: str = "pyannote"
    pipeline: str = "pyannote/speaker-diarization-community-1"
    hf_token_env: str = "HF_TOKEN"
    device: str = "cuda"
    device_index: int = 0
    use_exclusive_diarization_for_alignment: bool = True
    min_speakers: int | None = None
    max_speakers: int | None = None
    num_speakers: int | None = None
    python_version: str = "3.12"
    extra_args: list[str] = Field(default_factory=list)

    @field_validator("provider")
    @classmethod
    def _provider(cls, value: str) -> str:
        if value != "pyannote":
            raise ValueError("diarization.provider currently supports only 'pyannote'")
        return value


class TranslationConfig(_Model):
    enabled: bool = True
    provider: str = "openai-compatible"
    base_url: str = "http://LITELLM_HOST:PORT/v1"
    api_key: str = ""
    model: str = ""
    temperature: float = 0.0
    timeout_seconds: float = 120.0
    max_retries: int = 3
    backoff_base_seconds: float = 1.0
    batch_size: int = 10
    # Number of neighbouring segments sent alongside each batch as context.
    context_segments: int = 2
    prompt_version: str = "v1"
    max_output_tokens: int | None = None
    cache: bool = True
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider")
    @classmethod
    def _provider(cls, value: str) -> str:
        if value not in {"openai-compatible", "mock"}:
            raise ValueError("translation.provider must be 'openai-compatible' or 'mock'")
        return value

    @property
    def endpoint_configured(self) -> bool:
        if self.provider == "mock":
            return True
        host = self.base_url.lower().replace("http://", "").replace("https://", "")
        return (
            "litellm_host" not in host
            and "your_api_key" not in self.api_key.lower()
            and bool(self.model.strip())
            and self.api_key.strip().lower() not in PLACEHOLDER_VALUES
        )


class SpacyConfig(_Model):
    enabled: bool = True
    uv_project: Path = Path("environments/spacy")
    worker: Path = Path("workers/spacy_worker.py")
    process_source: bool = True
    process_english: bool = True
    english_model: str = "en_core_web_trf"
    source_models: dict[str, str] = Field(
        default_factory=lambda: {
            "en": "en_core_web_trf",
            "es": "es_dep_news_trf",
            "de": "de_dep_news_trf",
            "fr": "fr_dep_news_trf",
            "it": "it_core_news_lg",
            "pt": "pt_core_news_lg",
            "nl": "nl_core_news_lg",
        }
    )
    # Used when no language-specific model is configured/installed. ``blank``
    # keeps tokenisation/sentence splitting working for any language.
    fallback_model: str = "blank"
    max_length: int = 1_000_000
    python_version: str = "3.12"

    @field_validator("source_models")
    @classmethod
    def _langs(cls, value: dict[str, str]) -> dict[str, str]:
        return {k.strip().lower(): v for k, v in value.items()}


class AcousticConfig(_Model):
    enabled: bool = True
    uv_project: Path = Path("environments/acoustic")
    worker: Path = Path("workers/acoustic_worker.py")
    backend: str = "parselmouth"
    time_step: float = 0.01
    pitch_floor: float = 75.0
    pitch_ceiling: float = 500.0
    number_of_formants: int = 5
    formant_ceiling: float = 5500.0
    silence_threshold_db: float | None = None
    minimum_pause_duration: float = 0.2
    # Frames analysed per chunk keeps memory bounded for long recordings.
    chunk_seconds: float = 120.0
    python_version: str = "3.12"

    @field_validator("backend")
    @classmethod
    def _backend(cls, value: str) -> str:
        if value != "parselmouth":
            raise ValueError("acoustic.backend currently supports only 'parselmouth'")
        return value

    @model_validator(mode="after")
    def _sane(self) -> "AcousticConfig":
        if not 0 < self.time_step < 1:
            raise ValueError("acoustic.time_step must be between 0 and 1 seconds")
        if self.pitch_floor >= self.pitch_ceiling:
            raise ValueError("acoustic.pitch_floor must be below acoustic.pitch_ceiling")
        if not 2 <= self.number_of_formants <= 5:
            raise ValueError("acoustic.number_of_formants must be between 2 and 5")
        return self


class OpenPoseBodyConfig(_Model):
    enabled: bool = True
    model: str = "BODY_25"


class OpenPoseConfig(_Model):
    enabled: bool = True
    root: Path = Path("/opt/openpose")
    executable: str = "auto"
    model_folder: str = "auto"
    gpu: int = 0
    body: OpenPoseBodyConfig = Field(default_factory=OpenPoseBodyConfig)
    hands: dict[str, bool] = Field(default_factory=lambda: {"enabled": True})
    face: dict[str, bool] = Field(default_factory=lambda: {"enabled": True})
    # OpenPose keeps every intermediate frame in RAM when multithreading is on.
    disable_multi_thread: bool = True
    timeout_seconds: float | None = None
    extra_args: list[str] = Field(default_factory=list)

    @property
    def hands_enabled(self) -> bool:
        return bool(self.hands.get("enabled", True))

    @property
    def face_enabled(self) -> bool:
        return bool(self.face.get("enabled", True))


class LoggingConfig(_Model):
    level: str = "INFO"
    console: bool = True
    # Seconds between terminal progress refreshes for long-running stages.
    progress_interval_seconds: float = 10.0


class PipelineConfig(_Model):
    input: InputConfig
    output: OutputConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    ffmpeg: FFmpegConfig = Field(default_factory=FFmpegConfig)
    whisperx: WhisperXConfig = Field(default_factory=WhisperXConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    spacy: SpacyConfig = Field(default_factory=SpacyConfig)
    acoustic: AcousticConfig = Field(default_factory=AcousticConfig)
    openpose: OpenPoseConfig = Field(default_factory=OpenPoseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # ``--project-root`` is resolved at load time; relative uv projects/workers
    # are interpreted against it rather than the process cwd.
    project_root: Path = Field(default_factory=Path.cwd)
    config_path: Path | None = None

    @property
    def stage_configs(self) -> dict[str, _Model]:
        return {
            "whisperx": self.whisperx,
            "diarization": self.diarization,
            "translation": self.translation,
            "spacy": self.spacy,
            "acoustic": self.acoustic,
            "openpose": self.openpose,
        }

    def resolve(self, path: Path | str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else (self.project_root / candidate).resolve()

    def masked_dict(self) -> dict[str, Any]:
        return mask_secrets(self.model_dump(mode="json"))

    def behaviour_dict(self) -> dict[str, Any]:
        """Everything that changes outputs, minus placement/identity fields."""
        payload = self.model_dump(mode="json", exclude={"config_path", "project_root"})
        return mask_secrets(payload)


def configuration_hash(config: "PipelineConfig") -> str:
    """Global fingerprint of behaviour-affecting configuration.

    ``config_path`` and ``project_root`` are excluded: moving the project or
    renaming the config file must not invalidate a completed run.
    """
    return stable_hash(config.behaviour_dict(), length=16)


def mask_secrets(value: Any) -> Any:
    """Recursively replace secret-looking values with ``***masked***``."""
    if isinstance(value, Mapping):
        return {
            key: ("***masked***" if _is_secret_key(key) and val not in (None, "", 0) else mask_secrets(val))
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [mask_secrets(v) for v in value]
    return value


def _is_secret_key(key: object) -> bool:
    name = str(key)
    return bool(SECRET_KEY_RE.search(name)) and not SECRET_HOLDER_RE.search(name)


def mask_command(args: Iterable[str]) -> list[str]:
    """Mask token-looking values inside a command line for provenance."""
    out: list[str] = []
    previous: str | None = None
    for arg in args:
        if previous is not None and re.search(r"(token|key|secret|password)$", previous.lstrip("-"), re.I):
            out.append("***masked***")  # a value passed after a credential-shaped flag
        else:
            out.append(arg)
        previous = arg
    return out


def stable_hash(payload: Any, length: int = 16) -> str:
    """Deterministic hash of JSON-serialisable data (config/stage fingerprints)."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:length]


def load_config(path: Path | str, *, project_root: Path | None = None,
                overrides: Mapping[str, Any] | None = None) -> PipelineConfig:
    """Load, environment-interpolate, override and validate a YAML config."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {config_path}")
    data = interpolate_env(raw)
    if overrides:
        data = _deep_merge(data, dict(overrides))
    root = project_root or config_path.parent.parent
    data.setdefault("project_root", str(root))
    config = PipelineConfig.model_validate(data)
    config.config_path = config_path
    return config


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
