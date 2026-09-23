"""spaCy linguistic analysis over the original-language transcript.

Two problems have to be solved together:

1. **model resolution** — WhisperX detected a language; we need the best
   *installed* spaCy pipeline for it, with an honest fallback when none is
   available (tokenisation and sentence splitting still work from a blank
   pipeline with the right language vocabulary);
2. **timestamp alignment** — spaCy tokenises the *segment text*, whose
   whitespace differs from the transcript's own word list. Character spans are
   therefore mapped back onto WhisperX word spans by deterministic matching, and
   every token carries an explicit ``timestamp_alignment_status`` so a consumer
   never trusts a timestamp that was approximated.

The worker writes one raw JSON document (native spaCy fields + alignment
verdicts); this stage normalises that document into the two Parquet tables, so
schema fixes never require re-running spaCy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa

from ..artifacts import read_json
from ..exceptions import StageError, ValidationError
from ..schemas import SENTENCES_SCHEMA, TOKENS_SCHEMA, read_table, write_table
from .base import StageContext, WorkerStage

#: What a blank/lemma-only fallback pipeline can still provide.
FALLBACK_CAPABILITIES = ("tokenization", "sentencizer")


def select_model(language: str | None, configured: dict[str, str], available: set[str],
                 *, fallback: str = "blank") -> dict[str, Any]:
    """Choose the pipeline for a detected language and record *why*.

    Preference order: the configured model for the language (if installed) →
    another installed model of the same language family → a discovered model
    whose name starts with the language code → the configured fallback. The
    reason travels with the result into provenance instead of only the log.
    """
    language_key = (language or "").strip().lower() or "und"
    requested = configured.get(language_key)
    if requested and requested in available:
        return {"model": requested, "requested_model": requested, "status": "configured",
                "language": language_key, "capabilities": "full"}
    if requested:
        family = requested.split("_")[0]
        candidates = sorted(name for name in available if name.startswith(f"{family}_"))
        if candidates:
            return {"model": candidates[0], "requested_model": requested, "status": "substituted_family",
                    "language": language_key, "capabilities": "full"}
        return {"model": fallback, "requested_model": requested, "status": "fallback_missing_model",
                "language": language_key, "capabilities": ",".join(FALLBACK_CAPABILITIES)}
    candidates = sorted(name for name in available if name.startswith(f"{language_key}_"))
    if candidates:
        return {"model": candidates[0], "requested_model": None, "status": "discovered",
                "language": language_key, "capabilities": "full"}
    return {"model": fallback, "requested_model": None, "status": "fallback_no_model",
            "language": language_key, "capabilities": ",".join(FALLBACK_CAPABILITIES)}


class SpacySourceStage(WorkerStage):
    name = "spacy_source"
    raw_artifact = "spacy_source_raw"
    inputs = ("speech_segments", "speech_words")
    outputs = ("spacy_source_raw", "spacy_source_tokens", "spacy_source_sentences")
    config_keys = ("spacy",)
    variant = "source"

    # ------------------------------------------------------------------ request

    def request(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.config.spacy
        return {
            "stage": self.name,
            "variant": self.variant,
            "source_models": cfg.source_models,
            "english_model": cfg.english_model,
            "fallback_model": cfg.fallback_model,
            "max_length": cfg.max_length,
            "uv_project": str(ctx.config.resolve(cfg.uv_project)),
            "worker": str(ctx.config.resolve(cfg.worker)),
            "segments_digest": self._digest(ctx, "speech_segments"),
            "words_digest": self._digest(ctx, "speech_words"),
        }

    @staticmethod
    def _digest(ctx: StageContext, artifact: str) -> str | None:
        from ..stages.metadata import sha256_of

        try:
            return sha256_of(ctx.input(artifact))
        except StageError:
            return None

    # ------------------------------------------------------------------ worker

    def enabled(self, ctx: StageContext) -> tuple[bool, str]:
        cfg = ctx.config.spacy
        if not cfg.enabled:
            return False, "spacy.enabled = false"
        return True, ""

    def uv_project(self, ctx: StageContext) -> Path:
        return ctx.config.resolve(ctx.config.spacy.uv_project)

    def worker_script(self, ctx: StageContext) -> Path:
        return ctx.config.resolve(ctx.config.spacy.worker)

    def python_version(self, ctx: StageContext) -> str | None:
        return ctx.config.spacy.python_version

    def worker_argv(self, ctx: StageContext, raw_path: Path, request_digest: str) -> list[str]:
        cfg = ctx.config.spacy
        return [
            "--variant", self.variant,
            "--video-id", ctx.video_id,
            "--segments", str(ctx.input("speech_segments")),
            "--words", str(ctx.input("speech_words")),
            "--raw-output", str(raw_path),
            "--source-models", json.dumps(cfg.source_models),
            "--english-model", cfg.english_model,
            "--fallback-model", cfg.fallback_model,
            "--max-length", str(cfg.max_length),
            "--request-hash", request_digest,
        ]

    # ------------------------------------------------------------ normalisation

    def normalize(self, ctx: StageContext) -> dict[str, Any]:
        payload = self.validate_raw(ctx)
        tokens = [{**row, "variant": self.variant, "video_id": ctx.video_id, "schema_version": "1.0"}
                  for row in payload.get("tokens") or []]
        sentences = [{**row, "variant": self.variant, "video_id": ctx.video_id, "schema_version": "1.0"}
                     for row in payload.get("sentences") or []]
        tokens_path, sentences_path = self.parquet_outputs(ctx)
        write_table(tokens_path, pa.Table.from_pylist(tokens, schema=TOKENS_SCHEMA), TOKENS_SCHEMA,
                    extra_metadata={"variant": self.variant,
                                    "spacy_model": str(payload.get("selected_model")),
                                    "video_id": ctx.video_id})
        write_table(sentences_path, pa.Table.from_pylist(sentences, schema=SENTENCES_SCHEMA), SENTENCES_SCHEMA,
                    extra_metadata={"variant": self.variant,
                                    "spacy_model": str(payload.get("selected_model")),
                                    "video_id": ctx.video_id})
        aligned = sum(1 for row in tokens if row.get("timestamp_alignment_status") == "aligned")
        summary = {
            "tokens": len(tokens),
            "sentences": len(sentences),
            "aligned_tokens": aligned,
            "selected_spacy_model": payload.get("selected_model"),
            "spacy_model_version": payload.get("model_version"),
            "model_selection_status": payload.get("model_selection_status"),
            "available_capabilities": payload.get("capabilities"),
            "detected_language": payload.get("language"),
        }
        ctx.scratch[self.name] = summary
        rate = f"{aligned / len(tokens):.0%}" if tokens else "n/a"
        ctx.log(f"{self.variant} linguistics: {len(tokens)} tokens / {len(sentences)} sentences "
                f"(model={payload.get('selected_model')}, timestamps aligned={rate})")
        return summary

    def parquet_outputs(self, ctx: StageContext) -> tuple[Path, Path]:
        return ctx.artifact("spacy_source_tokens"), ctx.artifact("spacy_source_sentences")

    # ---------------------------------------------------------------- validation

    def validate(self, ctx: StageContext) -> dict[str, Any]:
        tokens_path, sentences_path = self.parquet_outputs(ctx)
        for path, schema in ((tokens_path, TOKENS_SCHEMA), (sentences_path, SENTENCES_SCHEMA)):
            if not path.is_file():
                raise ValidationError(self.name, [f"missing table: {path.name}"])
            from ..schemas import table_columns

            columns = set(table_columns(path))
            missing = [field.name for field in schema if field.name not in columns]
            if missing:
                raise ValidationError(self.name, [f"{path.name} missing columns: {', '.join(missing)}"])
        tokens = read_table(tokens_path).to_pylist()
        sentences = read_table(sentences_path).to_pylist()
        segment_ids = {row["segment_id"] for row in read_table(ctx.artifact("speech_segments")).to_pylist()}
        unknown = {row["segment_id"] for row in tokens} - segment_ids
        if unknown:
            raise ValidationError(self.name, [f"tokens reference unknown segments: {sorted(unknown)[:5]}"])
        sentence_ids = {row["sentence_id"] for row in sentences}
        orphans = {row["sentence_id"] for row in tokens} - sentence_ids
        if orphans:
            raise ValidationError(self.name, [f"tokens reference unknown sentences: {sorted(orphans)[:5]}"])
        token_ids = [row["token_id"] for row in tokens]
        if len(set(token_ids)) != len(token_ids):
            raise ValidationError(self.name, ["token_id values are not unique"])
        if tokens and not all(row.get("timestamp_alignment_status") for row in tokens):
            raise ValidationError(self.name, ["some tokens lack timestamp_alignment_status"])
        return {"tokens": len(tokens), "sentences": len(sentences)}
