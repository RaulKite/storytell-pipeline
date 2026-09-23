"""English translation through an OpenAI-compatible (LiteLLM) endpoint.

Design choices that matter for correctness:

* transcript **segments** are the stable translation unit, so an English row can
  always be traced back to exactly one source ``segment_id``;
* nearby segments travel along as *context* (better pronoun and register
  handling) but the model must return **only** the requested ids;
* the response is required to be strict JSON keyed by ``segment_id`` and is
  validated before use — a batch that omits, duplicates or invents an id is
  retried as a batch, never partially accepted;
* completed batches are cached by request hash, so an interrupted video does not
  pay twice;
* the API key lives in configuration only and is masked in logs and provenance.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import httpx
import pyarrow as pa

from ..artifacts import atomic_write_json, read_json
from ..config import stable_hash
from ..exceptions import StageError, ValidationError
from ..schemas import TRANSLATION_SCHEMA, read_table, write_table
from .base import Stage, StageContext


class TranslationTransportError(RuntimeError):
    """Transient endpoint problem worth retrying."""


PROMPTS: dict[str, str] = {
    "v1": (
        "You are a professional translator. Translate the requested transcript "
        "segments into natural, accurate English.\n"
        "\n"
        "Rules:\n"
        "1. Translate ONLY the segments listed under REQUESTED.\n"
        "2. Segments under CONTEXT are context only - do not translate them.\n"
        "3. Preserve meaning, speaker perspective, register and hedging. Do not "
        "summarise, expand or add commentary.\n"
        "4. Keep numbers, names and technical terms faithful; transliterate only "
        "when the source is not in Latin script.\n"
        "5. Return ONLY a JSON object mapping each requested segment_id to its "
        "English translation string. Every requested id must appear exactly once.\n"
        "\n"
        "SOURCE LANGUAGE: {language}\n"
        "\n"
        "CONTEXT:\n{context}\n"
        "\n"
        "REQUESTED:\n{requested}\n"
    )
}


@dataclass
class TranslationRequest:
    """One endpoint call: what to translate plus what to show as context."""

    requested: list[dict[str, Any]]
    context: list[dict[str, Any]]
    language: str
    prompt_version: str
    model: str
    temperature: float
    max_output_tokens: int | None

    @property
    def requested_ids(self) -> list[str]:
        return [str(row["segment_id"]) for row in self.requested]

    def prompt(self) -> str:
        template = PROMPTS.get(self.prompt_version)
        if not template:
            raise StageError(f"unknown translation prompt version: {self.prompt_version}")
        return template.format(
            language=self.language or "unknown",
            context=_render(self.context, mark=False),
            requested=_render(self.requested, mark=True),
        )

    def key(self) -> str:
        return stable_hash({
            "requested": self.requested_ids,
            "context": [str(row["segment_id"]) for row in self.context],
            "texts": [str(row.get("text") or "") for row in self.requested],
            "context_texts": [str(row.get("text") or "") for row in self.context],
            "language": self.language,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }, length=24)


def _render(rows: Sequence[dict[str, Any]], *, mark: bool) -> str:
    if not rows:
        return "(none)"
    lines = []
    for row in rows:
        prefix = "REQUESTED" if mark else "context"
        speaker = row.get("speaker_id") or "UNKNOWN"
        lines.append(f"[{prefix}] segment_id={row['segment_id']} speaker={speaker}: {row.get('text') or ''}")
    return "\n".join(lines)


class TranslationClient:
    """Minimal OpenAI-compatible chat client with structured JSON output."""

    def __init__(self, *, base_url: str, api_key: str, model: str, temperature: float = 0.0,
                 timeout_seconds: float = 120.0, max_retries: int = 3,
                 backoff_base_seconds: float = 1.0, extra_body: dict[str, Any] | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_base = backoff_base_seconds
        self.extra_body = extra_body or {}
        self._sleep = sleep
        self.request_count = 0
        self.timings_ms: list[float] = []
        self.usage: dict[str, int] = {}

    # ------------------------------------------------------------------ request

    def translate(self, request: TranslationRequest) -> dict[str, str]:
        prompt = request.prompt()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            started = time.time()
            try:
                payload = self._post(prompt, request)
                self.timings_ms.append(round((time.time() - started) * 1000, 1))
                self._record_usage(payload)
                raw_text = extract_message_content(payload)
                parsed = parse_structured_translations(raw_text)
                validated = validate_translations(parsed, request.requested_ids)
                return validated
            except TranslationTransportError as exc:
                last_error = exc
            except (ValidationError, ValueError) as exc:
                # A malformed answer is the model's fault, not the network's, but
                # resampling usually fixes it; treat it as retryable up to a point.
                last_error = exc
            if attempt < self.max_retries:
                delay = self.backoff_base * (2**attempt) + random.uniform(0, self.backoff_base)
                self._sleep(delay)
        raise StageError(
            f"translation batch failed after {self.max_retries + 1} attempts: {last_error}",
            details={"requested_ids": request.requested_ids, "attempts": self.max_retries + 1},
        )

    def _post(self, prompt: str, request: TranslationRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": request.model or self.model,
            "temperature": request.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if request.max_output_tokens:
            body["max_tokens"] = request.max_output_tokens
        body.update(self.extra_body)
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        self.request_count += 1
        try:
            response = httpx.post(f"{self.base_url}/chat/completions", json=body, headers=headers,
                                  timeout=self.timeout_seconds)
        except httpx.TimeoutException as exc:
            raise TranslationTransportError(f"request timed out after {self.timeout_seconds}s") from exc
        except httpx.HTTPError as exc:
            raise TranslationTransportError(f"transport error: {exc}") from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise TranslationTransportError(f"HTTP {response.status_code}: {response.text[:300]}")
        if response.status_code >= 400:
            raise StageError(f"translation endpoint rejected the request "
                             f"(HTTP {response.status_code}): {response.text[:400]}")
        try:
            return response.json()
        except ValueError as exc:
            raise TranslationTransportError(f"non-JSON response body: {response.text[:200]}") from exc

    def _record_usage(self, payload: dict[str, Any]) -> None:
        usage = payload.get("usage") or {}
        for key, value in usage.items():
            if isinstance(value, int):
                self.usage[key] = self.usage.get(key, 0) + value


def extract_message_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise TranslationTransportError("response contained no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):  # some gateways return content parts
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not isinstance(content, str) or not content.strip():
        raise TranslationTransportError("response message content was empty")
    return content


def parse_structured_translations(text: str) -> dict[str, str]:
    """Parse the required ``{segment_id: english}`` object, tolerating code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"translation response was not JSON: {text[:200]!r}")
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"translation response was not parseable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"translation response must be a JSON object, got {type(payload).__name__}")
    out: dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(value, dict):  # tolerate {"segment_id": {"text": ...}} shapes
            value = value.get("text") or value.get("translation") or value.get("english_text")
        if value is None:
            raise ValueError(f"translation for {key!r} was null")
        out[str(key)] = str(value).strip()
    return out


def validate_translations(translations: dict[str, str], requested_ids: Sequence[str]) -> dict[str, str]:
    """Every requested id exactly once, nothing invented, nothing empty."""
    requested = [str(item) for item in requested_ids]
    missing = [item for item in requested if item not in translations]
    extra = [item for item in translations if item not in set(requested)]
    empty = [key for key, value in translations.items() if not value.strip()]
    problems = []
    if missing:
        problems.append(f"missing translations for {missing[:5]}")
    if extra:
        problems.append(f"unexpected segment ids {extra[:5]}")
    if empty:
        problems.append(f"empty translations for {empty[:5]}")
    if problems:
        raise ValidationError("translation", problems)
    # Return in requested order so raw artifacts stay deterministic.
    return {key: translations[key] for key in requested}


# ------------------------------------------------------------------ stage

class TranslationStage(Stage):
    name = "translation"
    inputs = ("speech_segments",)
    outputs = ("translation_segments", "translation_raw")
    config_keys = ("translation",)

    def config_fingerprint(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.config.translation
        return {
            "stage": self.name,
            "provider": cfg.provider,
            "model": cfg.model,
            "base_url": cfg.base_url,
            "temperature": cfg.temperature,
            "batch_size": cfg.batch_size,
            "context_segments": cfg.context_segments,
            "prompt_version": cfg.prompt_version,
            "max_output_tokens": cfg.max_output_tokens,
            "max_retries": cfg.max_retries,
            "extra_body": cfg.extra_body,
            # Any transcript change (including speaker labels) changes the work.
            "segments_digest": self._segments_digest(ctx),
        }

    @staticmethod
    def _segments_digest(ctx: StageContext) -> str | None:
        from ..stages.metadata import sha256_of

        try:
            path = ctx.input("speech_segments")
        except StageError:
            return None
        return sha256_of(path)

    def enabled(self, ctx: StageContext) -> tuple[bool, str]:
        cfg = ctx.config.translation
        if not cfg.enabled:
            return False, "translation.enabled = false"
        if not cfg.endpoint_configured:
            return False, ("translation endpoint is not configured (set translation.base_url, "
                           "translation.api_key and translation.model in the config file)")
        return True, ""

    def prepare(self, ctx: StageContext) -> None:
        ctx.input("speech_segments")
        ctx.artifact("translation_raw").mkdir(parents=True, exist_ok=True)

    def build_requests(self, ctx: StageContext) -> list[TranslationRequest]:
        cfg = ctx.config.translation
        segments = read_table(ctx.input("speech_segments")).to_pylist()
        language = self._language(ctx, segments)
        return build_requests(
            segments,
            language=language,
            batch_size=cfg.batch_size,
            context_segments=cfg.context_segments,
            prompt_version=cfg.prompt_version,
            model=cfg.model,
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
        )

    @staticmethod
    def _language(ctx: StageContext, segments: Sequence[dict[str, Any]]) -> str:
        for row in segments:
            if row.get("language"):
                return str(row["language"])
        whisperx = ctx.scratch.get("whisperx") or {}
        return str(whisperx.get("language") or "")

    def execute(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.config.translation
        requests = self.build_requests(ctx)
        client = self._client(ctx)
        cache_dir = ctx.artifact("translation_raw") / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        translations: dict[str, str] = {}
        reused = 0
        for index, request in enumerate(requests, start=1):
            cached = self._load_cached(cache_dir / f"{request.key()}.json", request)
            if cached is not None:
                translations.update(cached)
                reused += 1
                ctx.log(f"batch {index}/{len(requests)} reused from cache")
                continue
            ctx.log(f"translating batch {index}/{len(requests)} ({len(request.requested_ids)} segments)")
            result = client.translate(request)
            atomic_write_json(cache_dir / f"{request.key()}.json", {
                "schema_version": "1.0",
                "request_key": request.key(),
                "requested_ids": request.requested_ids,
                "context_ids": [str(row["segment_id"]) for row in request.context],
                "model": request.model,
                "temperature": request.temperature,
                "prompt_version": request.prompt_version,
                "translations": result,
            })
            translations.update(result)

        segments = read_table(ctx.input("speech_segments")).to_pylist()
        rows = self._translation_rows(ctx.video_id, segments, translations, cfg.model, cfg.prompt_version)
        write_table(ctx.artifact("translation_segments"),
                    pa.Table.from_pylist(rows, schema=TRANSLATION_SCHEMA), TRANSLATION_SCHEMA,
                    extra_metadata={"translation_model": cfg.model, "prompt_version": cfg.prompt_version,
                                    "video_id": ctx.video_id})
        atomic_write_json(ctx.artifact("translation_raw") / "translation_summary.json", {
            "schema_version": "1.0",
            "video_id": ctx.video_id,
            "model": cfg.model,
            "base_url": cfg.base_url,
            "temperature": cfg.temperature,
            "prompt_version": cfg.prompt_version,
            "batches": len(requests),
            "batches_reused": reused,
            "requests_made": getattr(client, "request_count", 0),
            "request_timings_ms": getattr(client, "timings_ms", [])[-200:],
            "token_usage": getattr(client, "usage", {}),
            "segments": len(rows),
        })
        ctx.scratch["translation"] = {"segments": len(rows), "model": cfg.model}
        ctx.log(f"translated {len(rows)} segments ({reused}/{len(requests)} batches reused)")
        return {"tool_version": "openai-compatible", "model_version": cfg.model,
                "extra": {"segments": len(rows), "batches": len(requests), "batches_reused": reused}}

    def _client(self, ctx: StageContext) -> Any:
        cfg = ctx.config.translation
        if cfg.provider == "mock":
            return MockTranslationClient(model=cfg.model)
        return TranslationClient(base_url=cfg.base_url, api_key=cfg.api_key, model=cfg.model,
                                 temperature=cfg.temperature, timeout_seconds=cfg.timeout_seconds,
                                 max_retries=cfg.max_retries,
                                 backoff_base_seconds=cfg.backoff_base_seconds,
                                 extra_body=cfg.extra_body)

    @staticmethod
    def _load_cached(path: Path, request: TranslationRequest) -> dict[str, str] | None:
        if not path.is_file():
            return None
        try:
            payload = read_json(path)
        except (OSError, ValueError):
            return None
        cached = payload.get("translations") or {}
        if list(cached.keys()) != request.requested_ids:
            return None
        try:
            return validate_translations(cached, request.requested_ids)
        except ValidationError:
            return None

    @staticmethod
    def _translation_rows(video_id: str, segments: Sequence[dict[str, Any]],
                          translations: dict[str, str], model: str,
                          prompt_version: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for segment in segments:
            segment_id = str(segment["segment_id"])
            rows.append({
                "schema_version": "1.0",
                "video_id": video_id,
                "segment_id": segment_id,
                "speaker_id": segment.get("speaker_id"),
                "start_time": segment.get("start_time"),
                "end_time": segment.get("end_time"),
                "source_language": segment.get("language"),
                "source_text": segment.get("text"),
                "english_text": translations.get(segment_id),
                "translation_model": model,
                "translation_prompt_version": prompt_version,
            })
        return rows

    # ---------------------------------------------------------------- validation

    def validate(self, ctx: StageContext) -> dict[str, Any]:
        path = ctx.artifact("translation_segments")
        if not path.is_file():
            raise ValidationError(self.name, ["translation/segments_en.parquet missing"])
        rows = read_table(path).to_pylist()
        segments = read_table(ctx.artifact("speech_segments")).to_pylist()
        expected = [str(row["segment_id"]) for row in segments]
        produced = [str(row["segment_id"]) for row in rows]
        if sorted(produced) != sorted(expected):
            missing = sorted(set(expected) - set(produced))
            extra = sorted(set(produced) - set(expected))
            raise ValidationError(self.name, [f"segment id mismatch (missing={missing[:5]}, extra={extra[:5]})"])
        duplicates = {item for item in produced if produced.count(item) > 1}
        if duplicates:
            raise ValidationError(self.name, [f"duplicated segment ids: {sorted(duplicates)[:5]}"])
        empty = [row["segment_id"] for row in rows if not (row.get("english_text") or "").strip()]
        if empty:
            raise ValidationError(self.name, [f"segments without English text: {empty[:5]}"])
        source_mismatch = [
            row["segment_id"] for row, original in zip(rows, segments)
            if row.get("source_text") != original.get("text")
        ]
        if source_mismatch:
            raise ValidationError(self.name, [f"source text drifted for: {source_mismatch[:5]}"])
        return {"segments": len(rows), "all_segments_translated": True}


def build_requests(segments: Sequence[dict[str, Any]], *, language: str, batch_size: int,
                   context_segments: int, prompt_version: str, model: str, temperature: float,
                   max_output_tokens: int | None) -> list[TranslationRequest]:
    """Chunk segments into batches, each carrying neighbouring context.

    Context comes from the batch neighbourhood, so a long transcript is
    translated in a few passes with no batch ever losing its immediate context.
    """
    if batch_size <= 0:
        raise StageError("translation.batch_size must be positive")
    requests: list[TranslationRequest] = []
    for start in range(0, len(segments), batch_size):
        stop = min(len(segments), start + batch_size)
        batch = list(segments[start:stop])
        context_start = max(0, start - context_segments)
        context_end = min(len(segments), stop + context_segments)
        # Index slicing keeps neighbours even when two segments share identical text.
        context = list(segments[context_start:start]) + list(segments[stop:context_end])
        requests.append(TranslationRequest(
            requested=batch,
            context=context,
            language=language,
            prompt_version=prompt_version,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ))
    return requests


class MockTranslationClient:
    """Deterministic offline stand-in used by tests and dry runs."""

    def __init__(self, model: str = "mock") -> None:
        self.model = model
        self.request_count = 0
        self.timings_ms: list[float] = []
        self.usage: dict[str, int] = {}

    def translate(self, request: TranslationRequest) -> dict[str, str]:
        self.request_count += 1
        self.timings_ms.append(0.0)
        return {str(row["segment_id"]): f"[en] {row.get('text') or ''}".strip()
                for row in request.requested}
