"""spaCy analysis of the English translation.

Same feature schema as the source-language pass, and the same stage code — the
differences are only the input table (translation instead of transcript) and the
output paths. Timing is inherited from the source segment: the English text has
no audio of its own, so its temporal identity *is* the segment it came from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..exceptions import StageError
from .base import StageContext
from .spacy_source import SpacySourceStage


class SpacyEnglishStage(SpacySourceStage):
    name = "spacy_english"
    raw_artifact = "spacy_english_raw"
    inputs = ("translation_segments",)
    outputs = ("spacy_english_raw", "spacy_english_tokens", "spacy_english_sentences")
    config_keys = ("spacy",)
    variant = "english"

    def request(self, ctx: StageContext) -> dict[str, Any]:
        base = super().request(ctx)
        base["variant"] = "english"
        base["stage"] = self.name
        # The English pass depends on the translation table, not the transcript.
        base["segments_digest"] = self._digest(ctx, "translation_segments")
        base["words_digest"] = None
        base["english_model"] = ctx.config.spacy.english_model
        # Source-language detection says nothing about this pass: it forces `en`
        # (status `english_default`). Leaving the grade in the request would make this
        # stage's cache depend on a WhisperX re-grade that cannot change its output,
        # and would imply a dependency the worker never honours.
        base.pop("language_detection", None)
        base.pop("trust_low_language_detection", None)
        return base

    def enabled(self, ctx: StageContext) -> tuple[bool, str]:
        cfg = ctx.config.spacy
        if not cfg.enabled:
            return False, "spacy.enabled = false"
        if not cfg.process_english:
            return False, "spacy.process_english = false"
        if not ctx.artifact("translation_segments").is_file():
            return False, "no English translation available (translation stage did not run)"
        return True, ""

    def worker_argv(self, ctx: StageContext, raw_path: Path, request_digest: str) -> list[str]:
        cfg = ctx.config.spacy
        return [
            "--variant", "english",
            "--video-id", ctx.video_id,
            # Segments carry speaker/timing; translations carry the English text.
            "--segments", str(ctx.input("speech_segments")),
            "--translations", str(ctx.input("translation_segments")),
            "--words", str(ctx.artifact("speech_words")),
            "--raw-output", str(raw_path),
            "--source-models", json.dumps(cfg.source_models),
            "--english-model", cfg.english_model,
            "--fallback-model", cfg.fallback_model,
            # No --language-detection here on purpose: this variant forces `en`, so the
            # source-language grade is not an input to it.
            "--max-length", str(cfg.max_length),
            "--request-hash", request_digest,
        ]

    def parquet_outputs(self, ctx: StageContext) -> tuple[Path, Path]:
        return ctx.artifact("spacy_english_tokens"), ctx.artifact("spacy_english_sentences")
