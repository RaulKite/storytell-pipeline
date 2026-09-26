"""Typed runtime configuration with YAML loading and secret masking.

The orchestrator validates the whole configuration at startup: every stage
reads its own typed sub-model, so a typo in ``config.local.yaml`` fails before
any GPU work starts.

Credentials live in an untracked ``.env`` at the project root and reach the YAML
only through ``${VAR}`` interpolation, so no secret is ever written into a config
file, a log, or a provenance record.
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

from .exceptions import ConfigError
from .pose_normalize import SECOND_AXIS_PERPENDICULAR

# Values whose keys match this pattern are masked in logs/provenance/manifests.
SECRET_KEY_RE = re.compile(r"(api[_-]?key|token|secret|password|authorization|access[_-]?key)", re.I)

# ...unless the key names a *holder* rather than a value: ``hf_token_env``
# stores the name of an environment variable, which is not a secret and is
# needed to diagnose why a stage skipped.
SECRET_HOLDER_RE = re.compile(r"(_env|_var|_variable|_env_var)$", re.I)

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

# A .env assignment: optional ``export``, optional quotes, optional trailing comment.
_DOTENV_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")

PLACEHOLDER_VALUES = {
    "",
    "your_api_key",
    "your-api-key",
    "changeme",
    "none",
    "null",
    "litellm_host:port",
}


def load_dotenv(path: Path | str | None = None, *, override: bool = False) -> list[str]:
    """Load ``KEY=value`` lines from an untracked ``.env`` into the environment.

    Returns the keys actually set. Credentials live in one gitignored file rather
    than inside the YAML because a config file gets copied between machines, pasted
    into issues and committed by accident; a pipeline whose secrets are embedded in it
    leaks every time someone shares their config. ``${VAR}`` interpolation then keeps
    the YAML free of them.

    Precedence is deliberate: a variable already in the environment wins unless
    ``override=True``, so ``HF_TOKEN=... multimodal-pipeline run`` and CI secrets both
    beat a stale ``.env`` left over from a different account.

    A missing file is not an error — ``.env`` is optional by design, and a stage that
    needs a credential reports it as skipped with the variable to set.

    """
    dotenv = Path(path) if path is not None else Path.cwd() / ".env"
    if not dotenv.is_file():
        return []
    loaded: list[str] = []
    for lineno, raw in enumerate(dotenv.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _DOTENV_LINE_RE.match(line)
        if match is None:
            # Named by file and line: a half-edited .env otherwise shows up as a stage
            # mysteriously skipping for a missing credential.
            raise ConfigError(f"{dotenv.name}:{lineno}: expected KEY=value, got {line[:60]!r}")
        key, value = match.group(1), _clean_dotenv_value(match.group(2))
        if override or key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def _clean_dotenv_value(raw: str) -> str:
    """Reduce a ``.env`` right-hand side to its value.

    Quoted and unquoted forms are resolved in one direction only. Stripping a
    trailing comment first and then the quotes would misread ``A="x #y"`` (the comment
    marker is inside the quotes and part of the value); stripping quotes first would
    misread ``A=x #y`` (the comment is not part of the value). Quoting wins, exactly
    as shells and python-dotenv treat it.
    """
    if raw[:1] in ('"', "'"):
        end = raw.find(raw[0], 1)
        if end != -1:
            return raw[1:end]  # everything after the closing quote is a comment
        return raw[1:]  # unterminated quote: take what is there rather than lose it
    if " #" in raw:  # unquoted trailing comment
        return raw.split(" #", 1)[0].rstrip()
    return raw


def interpolate_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` inside YAML values.

    Each string is rewritten in a single pass. Rescanning a substituted value would
    interpolate the *result*, so a credential whose own text contains ``${...}`` could
    be silently mangled — and a truncated API key is worse than a loud failure.
    """
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


class DiarizationNemotronConfig(_Model):
    """NVIDIA Nemotron 3 Diarization, as a second engine next to pyannote.

    A parallel opinion, not a replacement: it reads ``audio/audio.wav`` and writes its own
    raw JSON and its own table. ``speaker_assignment`` never reads either, so the existing
    dataset meaning is untouched and the operator can compare the two tables and then
    decide which engine to adopt.

    The model is **not gated**, so unlike pyannote community-1 there is no model-condition
    acceptance step; ``hf_token_env`` is a rate-limit convenience, and the stage runs
    without it. It is still read from the environment and never from YAML, because the
    token would otherwise land in ``provenance/config.json``.
    """

    #: Off by default. This is an extra engine whose environment is not part of the
    #: default set a fresh clone syncs, and whose checkpoint is a large download from the
    #: Hugging Face hub; a fresh clone must not start that download unasked. A corpus still
    #: completes with this stage off, which is the point of a second opinion.
    enabled: bool = False
    uv_project: Path = Path("environments/diarization_nemotron")
    worker: Path = Path("workers/nemotron_diarization_worker.py")
    python_version: str = "3.12"
    model: str = "nvidia/Nemotron-3-Diarization"
    hf_token_env: str = "HF_TOKEN"
    device: str = "cuda"
    device_index: int = 0
    #: The model emits a fixed number of speaker channels. Raising it above the
    #: checkpoint's own channel count cannot invent speakers, so it is capped instead.
    max_speakers: int = 8
    #: Frame probability above which a frame counts as that speaker's speech. Default is
    #: the documented default of ``extract_speaker_dict``. Lower it to admit weaker speech
    #: and expect more, shorter segments.
    threshold: float = 0.5
    timeout_seconds: float | None = None
    #: When true, a worker that asks for cuda and cannot see a GPU degrades to cpu and
    #: records why. When false it fails. The pyannote worker only ever degrades, which is
    #: right there; here the second engine is optional, so a silent CPU run is worth being
    #: able to forbid: an hour-long batch that quietly crawled on CPU is a worse outcome
    #: than a fast failure.
    fallback_to_cpu: bool = True
    extra_args: list[str] = Field(default_factory=list)

    @field_validator("max_speakers")
    @classmethod
    def _max_speakers(cls, value: int) -> int:
        # The checkpoint's resolved streaming_config carries num_speakers = 8 and the
        # logits are (batch, frames, 8): a 9th channel does not exist, so a request for
        # more would silently do nothing. Refuse it at parse time instead.
        if not 1 <= value <= 8:
            raise ValueError(
                "diarization_nemotron.max_speakers must be between 1 and 8: the model "
                "emits exactly 8 speaker channels and cannot produce more"
            )
        return value

    @field_validator("threshold")
    @classmethod
    def _threshold(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("diarization_nemotron.threshold must be a probability in (0, 1)")
        return value

    @field_validator("device")
    @classmethod
    def _device(cls, value: str) -> str:
        if value not in {"cuda", "cpu"}:
            raise ValueError("diarization_nemotron.device must be 'cuda' or 'cpu'")
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
    # en_core_web_lg is the installed default: a router model with word vectors
    # and no torch dependency, so it never fights the CUDA-pinned ML environments.
    english_model: str = "en_core_web_lg"
    source_models: dict[str, str] = Field(
        default_factory=lambda: {
            "en": "en_core_web_lg",
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
    # WhisperX grades its own auto-detection (``language_detection`` in
    # ``speech/raw/whisperx.json``) as ``configured`` / ``ok`` / ``low``. This key
    # decides what the source-language linguistics stage does with ``low``.
    #
    # Default true preserves today's behaviour: in this corpus every clip is shorter
    # than WhisperX's 30 s detection window, so *every* auto-detection is graded
    # ``low`` — and the model each one produced was checked and is correct (the La 1
    # clip's Spanish lemmas come from a real Spanish pipeline, the English clips from
    # ``en_core_web_lg``). Refusing those would strip the full pipeline from the only
    # Spanish clip in the corpus on the strength of a grade that is always pessimistic
    # here; the choice is kept and the low grade is logged as a warning instead.
    #
    # Set false to refuse to build a full linguistic layer on a sub-window guess:
    # a ``low`` detection then demotes the source variant to the honest-empty path
    # (no language is handed to the resolver, so it reports ``fallback_no_model`` and
    # a blank pipeline yields tokens + sentences with no lemmas, POS or dependencies).
    trust_low_language_detection: bool = True
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
    #: Render one skeleton image per processed frame into ``pose/raw_images``.
    #: Opt-in on purpose: both keys below are mixed into the ``openpose`` stage
    #: fingerprint, so turning this on invalidates every pose dataset already on
    #: disk and re-runs OpenPose -- the slowest stage in the pipeline -- on all of
    #: them. Enabling it has to be a deliberate decision, never an upgrade side
    #: effect, which is why the default reproduces the pre-feature command exactly.
    write_images: bool = False
    #: Longest rendered side in pixels. OpenPose's own ``--output_resolution``
    #: default is ``-1x-1``, i.e. full input resolution, which costs hundreds of MB
    #: to GBs per video; ``None`` keeps that default and the stage logs the warning
    #: once per run instead of silently choosing a downscale for you.
    image_max_side: int | None = None
    timeout_seconds: float | None = None
    extra_args: list[str] = Field(default_factory=list)

    @field_validator("image_max_side")
    @classmethod
    def _image_max_side(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("openpose.image_max_side must be a positive pixel count or null")
        return value

    @property
    def hands_enabled(self) -> bool:
        return bool(self.hands.get("enabled", True))

    @property
    def face_enabled(self) -> bool:
        return bool(self.face.get("enabled", True))


class ActiveSpeakerConfig(_Model):
    """TalkNet-ASD active-speaker detection over the original video.

    ``talknet_root`` points at a TalkNet-ASD checkout rather than an installed
    package: the upstream project is research code that resolves its checkpoints
    and its ``model/`` imports relative to the current working directory, so it has
    to be invoked in place. ``weights_dir`` is an escape hatch for a read-only
    checkout -- without it the runner downloads its two checkpoints into the repo.
    """

    enabled: bool = True
    uv_project: Path = Path("environments/activespeaker")
    worker: Path = Path("workers/activespeaker_worker.py")
    python_version: str = "3.12"
    talknet_root: Path | None = None
    weights_dir: Path | None = None
    device: str = "auto"
    device_index: int = 0
    #: Smoothed score at or above which a frame counts as an active speaker.
    #: TalkNet scores are unbounded logits; the upstream convention is 0.
    speaker_threshold: float = 0.0
    #: Centered smoothing window in 25 FPS frames, clipped at scene boundaries.
    score_window: int = 5
    #: A challenger must beat the incumbent by this much...
    switch_margin: float = 0.5
    #: ...for this many consecutive frames before it takes over.
    switch_frames: int = 3
    timeout_seconds: float | None = None
    extra_args: list[str] = Field(default_factory=list)

    @field_validator("device")
    @classmethod
    def _device(cls, value: str) -> str:
        if value not in ("auto", "cuda", "cpu"):
            raise ValueError("activespeaker.device must be 'auto', 'cuda' or 'cpu'")
        return value

    @field_validator("score_window")
    @classmethod
    def _window(cls, value: int) -> int:
        if value <= 0 or value % 2 == 0:
            raise ValueError("activespeaker.score_window must be a positive odd integer")
        return value

    @field_validator("switch_frames")
    @classmethod
    def _switch_frames(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("activespeaker.switch_frames must be positive")
        return value

    @field_validator("switch_margin")
    @classmethod
    def _margin(cls, value: float) -> float:
        if value < 0:
            raise ValueError("activespeaker.switch_margin must not be negative")
        return value


class SpeakerFusionConfig(_Model):
    """Fuse diarization turns with the per-frame active-speaker table (§20.1 / T13).

    A *second* diarization result, kept beside the existing ones. Nothing here is read by
    ``speaker_assignment``, so enabling it relabels no existing artifact — which is the
    whole reason it is a new stage rather than a change to ``diarization``.

    ``engines`` is the T20 requirement made explicit: the fusion core takes *which* turn
    table to read as a parameter, so ``["pyannote"]`` and ``["pyannote", "nemotron"]`` are
    two calls of the same code. Each selected engine is one output table.
    """

    enabled: bool = True
    #: Turn tables to fuse, from ``fusion.TURN_TABLES``. Selecting ``nemotron`` costs
    #: nothing but a table when the diarizer ran; a selected engine whose turn table is
    #: absent is skipped with a logged reason rather than failing the stage.
    engines: list[str] = Field(default_factory=lambda: ["pyannote"])
    #: Share of a winning track's in-window frames that must be flagged active for a
    #: ``face_matched`` verdict. Lower it to accept a face that talks intermittently.
    min_active_ratio: float = 0.5
    #: Frames a track must be flagged active on before it counts as evidence at all.
    #: One frame is a sighting, not a match; raise it to require a sustained speaker.
    min_face_frames: int = 2

    @field_validator("engines")
    @classmethod
    def _engines(cls, value: list[str]) -> list[str]:
        from .fusion import FUSION_ENGINES, TURN_TABLES

        if not value:
            raise ValueError(
                "speaker_fusion.engines must name at least one turn table "
                f"(allowed: {', '.join(FUSION_ENGINES)})"
            )
        unknown = [name for name in value if name not in TURN_TABLES]
        if unknown:
            raise ValueError(
                f"speaker_fusion.engines has unknown entries {sorted(unknown)} "
                f"(allowed: {', '.join(FUSION_ENGINES)})"
            )
        duplicates = sorted({name for name in value if value.count(name) > 1})
        if duplicates:
            # A repeated engine writes the same table twice and doubles the row counts in
            # it, which validates fine and is simply wrong data.
            raise ValueError(
                f"speaker_fusion.engines lists {', '.join(duplicates)} more than once"
            )
        return list(value)

    @field_validator("min_active_ratio")
    @classmethod
    def _min_active_ratio(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError(
                "speaker_fusion.min_active_ratio must be a share in (0, 1]: it is the "
                "fraction of a turn's in-range frames that must be active"
            )
        return value

    @field_validator("min_face_frames")
    @classmethod
    def _min_face_frames(cls, value: int) -> int:
        if value < 1:
            raise ValueError(
                "speaker_fusion.min_face_frames must be >= 1: a track needs at least one "
                "active frame before it can be evidence at all"
            )
        return value


class PoseNormalizedConfig(_Model):
    """Re-express BODY_25 keypoints in a body-centred frame (§20.4 / T14).

    A *second* pose table, written beside ``pose/body.parquet``. Nothing here is read by
    the openpose stage, so enabling it rewrites no pixel table — which is the whole
    reason it is a new stage rather than a column.

    The basis triple is configuration, not a constant, because changing it changes
    every number in the table: it is mixed into the stage fingerprint, so a different
    frame invalidates the file instead of quietly redefining it. ``second_axis`` names
    the joint that supplies the second axis; the only value the reference implementation
    was measured against is ``perpendicular`` (dfMaker's ``i == j`` branch), and that is
    what ``MidHip -> Neck`` uses.
    """

    enabled: bool = True
    #: Keypoint that becomes the origin, by BODY_25 name.
    origin_keypoint: str = "MidHip"
    #: Keypoint the first basis vector points at.
    basis_keypoint: str = "Neck"
    #: What supplies the second axis. Only ``perpendicular`` is implemented: it is
    #: dfMaker's ``i == j`` branch, and the only branch the reference comparison was
    #: run against. A third joint name is refused rather than quietly ignored — the
    #: transform would keep using the perpendicular and label every row with a frame it
    #: did not build, which is the one outcome a derived table cannot survive.
    second_axis: str = SECOND_AXIS_PERPENDICULAR

    @field_validator("origin_keypoint", "basis_keypoint", "second_axis")
    @classmethod
    def _keypoint_names(cls, value: str) -> str:
        from .schemas import BODY_25_KEYPOINT_NAMES

        # Names, not indices: `transformation_coords = c(1, 8, 1, 1)` is legible in R
        # and illegible here, and a fingerprint that reads "8" cannot be reviewed.
        if value == "Background":
            # In BODY_25's name list but never in pose/body.parquet: the normalizer
            # drops it as a filler channel, so a frame defined by it could only ever
            # report basis_missing_joint for every person-frame. Refusing it here beats
            # writing a table of nulls and calling the stage healthy.
            raise ValueError(
                "pose_normalized cannot use Background as a keypoint: OpenPose's "
                "Background channel is a filler and is not written to "
                "pose/body.parquet, so it is never measurable"
            )
        if value not in BODY_25_KEYPOINT_NAMES and value != SECOND_AXIS_PERPENDICULAR:
            raise ValueError(
                f"pose_normalized keypoint {value!r} is not a BODY_25 keypoint name "
                f"(and is not {SECOND_AXIS_PERPENDICULAR!r}): valid names are "
                f"{', '.join(BODY_25_KEYPOINT_NAMES[:-1])} — the last BODY_25 entry, "
                "Background, is a filler channel and is never written to "
                "pose/body.parquet, so it cannot define a frame either"
            )
        return value

    @model_validator(mode="after")
    def _basis_triple_is_buildable(self) -> "PoseNormalizedConfig":
        if self.origin_keypoint == self.basis_keypoint:
            # Not a preference: with origin == basis the vector between them is the zero
            # vector, so the determinant is 0 and every coordinate would be 0/0. The
            # stage would write a table of nulls and call it a normalisation.
            raise ValueError(
                "pose_normalized.origin_keypoint and basis_keypoint must name different "
                f"keypoints: both are {self.origin_keypoint!r}, so the basis vector has "
                "zero length and the transform's determinant is 0 — nothing could be "
                "divided by it"
            )
        if self.second_axis != SECOND_AXIS_PERPENDICULAR:
            # Reachable: the field validator lets a BODY_25 name through here.
            raise ValueError(
                f"pose_normalized.second_axis={self.second_axis!r} is not implemented: the "
                f"only second axis validated against the reference is "
                f"{SECOND_AXIS_PERPENDICULAR!r} (dfMaker's i == j branch). Accepting a "
                "joint name and then computing the perpendicular anyway would label every "
                "row with a frame that was never built"
            )
        return self


PERSON_TRACKER_TYPES: tuple[str, ...] = ("botsort", "bytetrack", "ocsort", "deepocsort",
                                         "tracktrack", "fasttrack")


#: COCO class id for ``person``. Mirrors ``PERSON_CLASS`` in workers/persons_worker.py, which
#: cannot import this module: the worker runs inside the persons uv project, which does not
#: install the pipeline package (the same reason it re-implements its frame-index reading).
PERSON_COCO_CLASS_ID = 0


class PersonsConfig(_Model):
    """YOLO person detection + tracking over the original video (§20.2 / T15).

    A *third* visual signal, next to OpenPose's bodies and TalkNet's faces. It answers the
    two questions neither of them answers: how many distinct people appear in a video, and
    when each one is on screen. Its ids live in their own namespace and are written to
    their own directory — see :data:`multimodal_pipeline.schemas.PERSON_FRAMES_SCHEMA`.

    Off by default, like the other optional GPU stages. A fresh clone is not asked to sync
    a fifth torch environment it may not want; with this off the pipeline completes and the
    stage reports why it did not run.
    """

    enabled: bool = False
    uv_project: Path = Path("environments/persons")
    worker: Path = Path("workers/persons_worker.py")
    python_version: str = "3.12"
    #: Checkpoint *name*, resolved under ``weights_dir`` when that is set and otherwise
    #: left to ultralytics to fetch on first use. The name is in the stage fingerprint, so
    #: switching from yolo11n to yolo11s invalidates every person table instead of letting
    #: a count computed by one model be read as the other's.
    model: str = "yolo11n.pt"
    #: Where that checkpoint is read from, named for TalkNet's setting of the same name.
    #: Two facts make this setting necessary rather than cosmetic:
    #:   * ultralytics keeps its own ``weights_dir`` in a user-level settings file and it
    #:     is a *relative* path (``weights``), so it resolves against whoever happened to
    #:     start the process and is not governed by this config file at all;
    #:   * the default resolution writes a downloaded ``yolo11n.pt`` next to whatever the
    #:     cwd is, so a read-only checkout fails in the middle of the first video.
    #: Point it at a directory holding the checkpoint to get a hermetic run.
    weights_dir: Path | None = None
    device: str = "auto"
    device_index: int = 0
    #: Refuse any COCO class other than 0 (person), and refuse an empty ``classes``.
    #: Default ``true``. The reason is what a wrong value costs a *reader*, not what it costs a
    #: run: every id in ``persons/frames.parquet`` is called ``person_id`` and the summary table
    #: says how many people appear, so with ``classes: [2]`` that column counts cars and
    #: nothing in the Parquet says so. The raw document and the fingerprint do carry the
    #: classes, so the run is traceable -- but a notebook that joined on the column months
    #: later sees only a person column that quietly means something else. A measurement
    #: artifact should not be able to look like a different measurement than it is.
    #: Set it to ``false`` to track other classes on purpose; the column keeps its name, so the
    #: honest reading of such a dataset is "detected objects".
    person_classes_only: bool = True
    #: ByteTracker. NOT the ultralytics default — in 8.4.163 the library default is
    #: ``tracktrack.yaml`` (see ``ultralytics/cfg/default.yaml``), and the two give different
    #: answers on the same clips, which is why this is a deliberate choice rather than an
    #: inherited one; the measured comparison is in config.example.yaml.
    #:
    #: Not ``botsort`` either: BoT-Sort is the tracker that owns the camera-motion-compensation
    #: state this stage has to work around, and its re-identification branch can pull an
    #: appearance-embedding checkpoint that ultralytics downloads on *first use* rather than
    #: with the model weights, so the default would depend on a download nobody asked for.
    #: ``bytetrack`` needs only ``lap``, which is pinned in the environment.
    #:
    #: The tracker decides how ids are carried across frames, so it is in the fingerprint:
    #: changing it changes every count in the table.
    tracker: str = "bytetrack"
    #: Person-class detections at or below this confidence are dropped. Lower it to keep
    #: distant or partially occluded people, and expect more fragmented ids in return.
    conf: float = 0.25
    #: Class filter. 0 is COCO's ``person`` and nothing else is wanted; leaving it null
    #: would put cars and chairs into a "persons per video" table.
    classes: list[int] = Field(default_factory=lambda: [0])
    #: Image side fed to the network, in pixels. Ultralytics' own default. Raising it costs
    #: speed roughly quadratically and is the knob to turn when small people are missed.
    imgsz: int = 640
    #: Kill a wedged worker after N seconds. No default: a 4-hour recording at 50 fps is
    #: 720k frames and a stage with a guessed timeout would fail a legitimate long run.
    timeout_seconds: float | None = None
    extra_args: list[str] = Field(default_factory=list)

    @field_validator("device")
    @classmethod
    def _device(cls, value: str) -> str:
        if value not in ("auto", "cuda", "cpu"):
            raise ValueError("persons.device must be 'auto', 'cuda' or 'cpu'")
        return value

    @field_validator("tracker")
    @classmethod
    def _tracker(cls, value: str) -> str:
        # Ultralytics accepts a YAML *path* here as well as a tracker name. That form is
        # refused rather than passed through: the stage fingerprint would record a path, so
        # editing the YAML in place would leave every existing person table looking
        # reusable while the tracker parameters behind it had changed.
        if value not in PERSON_TRACKER_TYPES:
            raise ValueError(
                f"persons.tracker={value!r} is not one of the built-in trackers "
                f"({', '.join(PERSON_TRACKER_TYPES)}). A path to a custom tracker YAML is "
                "not accepted: the fingerprint would record the filename rather than the "
                "parameters inside it, and editing that file would silently invalidate "
                "nothing."
            )
        return value

    @field_validator("conf")
    @classmethod
    def _conf(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError(
                "persons.conf must be a confidence in (0, 1): it is the floor a person "
                "detection has to clear. 0 would keep every box the model can produce and "
                "1 would keep none, and neither is a measurement of who is on screen."
            )
        return value

    @field_validator("classes")
    @classmethod
    def _classes(cls, value: list[int]) -> list[int]:
        for index in value:
            # COCO's ids are 0..79; ultralytics raises ValueError past the last one, and
            # a stage that failed on the first frame of every video is a worse outcome
            # than a config that refuses to load.
            if not 0 <= index <= 79:
                raise ValueError(
                    f"persons.classes contains {index}, which is not a COCO class id "
                    "(0..79). This stage's contract is person detections; keep it [0]."
                )
        return list(value)

    @model_validator(mode="after")
    def _person_classes(self) -> "PersonsConfig":
        """The guard that keeps ``person_id`` meaning a person. See ``person_classes_only``.

        An empty list is refused here as well: the worker then omits ``classes`` entirely and
        ultralytics returns all 80 COCO classes, so "do not filter" is the most indirect way to
        fill a person column with a car.
        """
        if not self.person_classes_only:
            return self
        stray = [index for index in self.classes if index != PERSON_COCO_CLASS_ID]
        if stray:
            raise ValueError(
                f"persons.classes contains {stray}, which is not the person class "
                f"({PERSON_COCO_CLASS_ID}). Every id this stage writes is called person_id, so "
                "a table built from other classes would not say what it measures. Set "
                "persons.person_classes_only to false to opt into that deliberately."
            )
        if not self.classes:
            raise ValueError(
                "persons.classes is empty, which tells ultralytics to detect all 80 COCO "
                "classes; persons.person_classes_only=true refuses that because the column it "
                "writes is called person_id. Set persons.classes to "
                f"[{PERSON_COCO_CLASS_ID}], or person_classes_only to false."
            )
        return self

    @field_validator("imgsz")
    @classmethod
    def _imgsz(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("persons.imgsz must be a positive pixel count")
        return value


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
    diarization_nemotron: DiarizationNemotronConfig = Field(default_factory=DiarizationNemotronConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    spacy: SpacyConfig = Field(default_factory=SpacyConfig)
    acoustic: AcousticConfig = Field(default_factory=AcousticConfig)
    openpose: OpenPoseConfig = Field(default_factory=OpenPoseConfig)
    activespeaker: ActiveSpeakerConfig = Field(default_factory=ActiveSpeakerConfig)
    speaker_fusion: SpeakerFusionConfig = Field(default_factory=SpeakerFusionConfig)
    pose_normalized: PoseNormalizedConfig = Field(default_factory=PoseNormalizedConfig)
    persons: PersonsConfig = Field(default_factory=PersonsConfig)
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
            "diarization_nemotron": self.diarization_nemotron,
            "translation": self.translation,
            "spacy": self.spacy,
            "acoustic": self.acoustic,
            "openpose": self.openpose,
            "activespeaker": self.activespeaker,
            "speaker_fusion": self.speaker_fusion,
            "pose_normalized": self.pose_normalized,
            # Carries `uv_project`, so `provenance/tools.json` inventories it and
            # `inspect-environment` warns a fresh clone that the environment is absent.
            "persons": self.persons,
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
    root = project_root or config_path.parent.parent
    # Credentials are read here, before interpolation, so a run needs neither a shell
    # wrapper nor an explicit ``source .env``. The project root is used rather than the
    # process cwd so the same file applies however the CLI is invoked.
    #
    # MULTIMODAL_PIPELINE_NO_DOTENV=1 skips exactly this implicit read. Without it the
    # suite would be non-hermetic on a machine that has real credentials: assertions
    # like "diarization skips because the token is missing" would pass on CI and fail at
    # that desk. An explicit ``load_dotenv(path)`` call is never disabled -- tests of the
    # loader itself must keep working.
    if os.environ.get("MULTIMODAL_PIPELINE_NO_DOTENV") != "1":
        load_dotenv(root / ".env")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {config_path}")
    data = interpolate_env(raw)
    if overrides:
        data = _deep_merge(data, dict(overrides))
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
