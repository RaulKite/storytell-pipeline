#!/usr/bin/env python3
"""spaCy worker — runs inside ``environments/spacy`` only.

One code path for both variants:

* ``--variant source``  — original-language transcript, with WhisperX word
  timestamps mapped onto spaCy tokens;
* ``--variant english`` — the English translation. It has no audio of its own,
  so it deliberately receives **no** word list: its temporal identity is its
  source segment, and inventing word timestamps from another language's word
  boundaries would be worse than reporting ``no_timing``.

Writes one raw JSON document with native spaCy fields plus explicit
timestamp-alignment verdicts. The orchestrator normalises that document into
Parquet, so a schema fix never needs spaCy to run again.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
import unicodedata
from typing import Any

FALLBACK_CAPABILITIES = ("tokenization", "sentencizer")

#: Explicit alignment verdicts; a consumer must never have to guess.
ALIGNED = "aligned"
APPROXIMATE = "approximate"
UNMATCHED = "unmatched"
NO_TIMING = "no_timing"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="spaCy linguistic extraction worker")
    parser.add_argument("--variant", required=True, choices=("source", "english"))
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--segments", required=True, help="speech/segments.parquet (timing + speakers)")
    parser.add_argument("--words", help="speech/words.parquet (source variant only)")
    parser.add_argument("--translations", help="translation/segments_en.parquet (english variant)")
    parser.add_argument("--raw-output", required=True)
    parser.add_argument("--source-models", default="{}")
    parser.add_argument("--english-model", default="en_core_web_trf")
    parser.add_argument("--fallback-model", default="blank")
    parser.add_argument("--max-length", type=int, default=1_000_000)
    parser.add_argument("--request-hash", default=None)
    parser.add_argument("--result-json", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    from pathlib import Path

    raw_output = Path(args.raw_output)
    result_path = Path(args.result_json) if args.result_json else (
        raw_output.parent / f"spacy_{args.variant}_worker_result.json")
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"stage": f"spacy_{args.variant}", "status": "running",
                                  "request_hash": args.request_hash}
    try:
        import pyarrow.parquet as pq
        import spacy

        segments = pq.read_table(args.segments).to_pylist()
        language = next((row["language"] for row in segments if row.get("language")), None)
        configured = json.loads(args.source_models or "{}")

        if args.variant == "english":
            if not args.translations:
                raise ValueError("--translations is required for variant=english")
            texts = english_texts(pq.read_table(args.translations).to_pylist())
            language = "en"
            word_source: list[dict] = []
            selection = {"model": args.english_model, "requested_model": args.english_model,
                         "status": "english_default", "language": "en", "capabilities": "full"}
        else:
            texts = None
            selection = select_model(language, configured, installed_models(),
                                     fallback=args.fallback_model)
            word_source = pq.read_table(args.words).to_pylist() if args.words else []

        nlp = load_pipeline(spacy, selection, args.english_model, args.max_length, args.fallback_model)
        document = build_document(
            variant=args.variant,
            video_id=args.video_id,
            segments=segments,
            segment_words=word_source,
            nlp=nlp,
            selection=selection,
            model_version_value=model_version(selection["model"]),
            texts=texts,
            language=language,
        )
        raw_output.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
        payload.update({
            "status": "ok",
            "tool_version": spacy.__version__,
            "model_version": document["model_version"],
            "selected_model": document["selected_model"],
            "model_selection_status": selection["status"],
            "tokens": len(document["tokens"]),
            "sentences": len(document["sentences"]),
            "duration_seconds": round(time.time() - started, 3),
        })
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 0
    except Exception as exc:  # noqa: BLE001 - the orchestrator reads this file, not stderr
        payload.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"[:2000],
                        "traceback": traceback.format_exc()[-6000:],
                        "duration_seconds": round(time.time() - started, 3)})
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"spacy worker failed: {exc}", file=sys.stderr)
        return 1


def english_texts(rows: list[dict]) -> dict[str, str]:
    return {str(row["segment_id"]): str(row.get("english_text") or "") for row in rows}


def installed_models() -> set[str]:
    """Names of spaCy models actually importable in this environment."""
    try:
        from spacy.cli.info import info  # noqa: F401  (ensures the CLI machinery is importable)
    except Exception:  # noqa: BLE001 - irrelevant for listing
        pass
    try:
        from spacy.util import get_installed_models

        return set(get_installed_models())
    except Exception:  # noqa: BLE001 - older/newer spaCy layout
        return set()


def select_model(language, configured: dict[str, str], available: set[str],
                 *, fallback: str = "blank") -> dict[str, Any]:
    """Mirror of the orchestrator's resolver (kept in sync deliberately).

    Order: configured model for the language if installed → another installed
    model of the same language family → any installed model named for the
    language → the fallback. ``status`` says which branch fired, and it is
    carried into provenance.
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


def load_pipeline(spacy_module, selection: dict[str, Any], english_model: str,
                  max_length: int, fallback: str):
    """Load the chosen pipeline, degrading to a language-aware blank pipeline.

    A missing transformer model must not cost the whole linguistic layer: a
    blank pipeline with the right language vocabulary still yields tokenisation
    and sentence boundaries, and ``capabilities`` records exactly that.
    """
    name = selection.get("model") or fallback
    language_key = selection.get("language") or "xx"
    if selection.get("status") == "english_default":
        # Prefer the configured transformer model, then any smaller English one.
        for candidate in (name, english_model, "en_core_web_sm"):
            if candidate and candidate != "blank":
                try:
                    return _load(spacy_module, candidate, max_length)
                except Exception:  # noqa: BLE001 - try the next candidate
                    continue
        return _load_blank(spacy_module, "en", max_length)
    try:
        return _load(spacy_module, name, max_length)
    except Exception:  # noqa: BLE001 - model not installed for this language
        return _load_blank(spacy_module, language_key, max_length)


def _load(spacy_module, name: str, max_length: int):
    nlp = spacy_module.load(name)
    nlp.max_length = max_length
    return nlp


def _load_blank(spacy_module, language: str, max_length: int):
    try:
        nlp = spacy_module.blank(language)
    except Exception:  # noqa: BLE001 - spaCy has no vocab for that code
        nlp = spacy_module.blank("xx")
    if "sentencizer" not in nlp.pipe_names:
        try:
            nlp.add_pipe("sentencizer", first=True)
        except Exception:  # noqa: BLE001 - already present
            pass
    nlp.max_length = max_length
    return nlp


def model_version(model_name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(model_name)
    except PackageNotFoundError:
        # Blank pipelines are not packages: no version is the honest answer.
        return None


def build_document(*, variant: str, video_id, segments: list[dict], segment_words: list[dict],
                   nlp, selection: dict[str, Any], model_version_value, texts,
                   language) -> dict[str, Any]:
    """Run spaCy per segment; emit native features plus timing verdicts.

    Token/sentence ids are derived from the segment id so they stay stable and
    self-describing (``seg000001-s002-t0007``), and ``head_token_id`` keeps the
    dependency edge resolvable inside the table.
    """
    words_by_segment: dict[str, list[dict]] = {}
    for word in segment_words:
        words_by_segment.setdefault(str(word["segment_id"]), []).append(word)

    tokens_out: list[dict[str, Any]] = []
    sentences_out: list[dict[str, Any]] = []
    capabilities = capabilities_of(nlp)

    for segment in segments:
        segment_id = str(segment["segment_id"])
        text = (texts or {}).get(segment_id, "") if variant == "english" else (segment.get("text") or "")
        if not (text or "").strip():
            continue
        doc = analyse(nlp, text)
        words = words_by_segment.get(segment_id, [])
        for sentence_index, sent in enumerate(doc.sents):
            sentence_id = f"{segment_id}-s{sentence_index + 1:03d}"
            sentences_out.append({
                "segment_id": segment_id,
                "sentence_id": sentence_id,
                "sentence_index": sentence_index,
                "speaker_id": segment.get("speaker_id"),
                "text": sent.text,
                "token_count": sent.end - sent.start,
                "char_start": sent.start_char,
                "char_end": sent.end_char,
                "segment_start_time": segment.get("start_time"),
                "segment_end_time": segment.get("end_time"),
            })
            for token in sent:
                offset = token.i - sent.start
                head_offset = token.head.i - sent.start
                in_sentence = sent.start <= token.head.i < sent.end
                tokens_out.append({
                    "segment_id": segment_id,
                    "sentence_id": sentence_id,
                    "token_id": f"{sentence_id}-t{offset + 1:04d}",
                    "token_index": token.i,
                    "speaker_id": segment.get("speaker_id"),
                    "text": token.text,
                    "lower": token.lower_,
                    "lemma": token.lemma_,
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "morph": str(token.morph),
                    "dep": token.dep_,
                    "head_token_id": f"{sentence_id}-t{head_offset + 1:04d}" if in_sentence else None,
                    "head_text": token.head.text,
                    "head_pos": token.head.pos_,
                    "ent_type": token.ent_type_ or None,
                    "is_alpha": bool(token.is_alpha),
                    "is_stop": bool(token.is_stop),
                    "is_digit": bool(token.is_digit),
                    "like_num": bool(token.like_num),
                    "shape": shape_of(token.text),
                    "char_start": token.idx,
                    "char_end": token.idx + len(token.text),
                    "segment_start_time": segment.get("start_time"),
                    "segment_end_time": segment.get("end_time"),
                    **token_timing(token, words),
                })
    return {
        "schema_version": "1.0",
        "video_id": video_id,
        "variant": variant,
        "language": language,
        "selected_model": selection.get("model"),
        "requested_model": selection.get("requested_model"),
        "model_selection_status": selection.get("status"),
        "model_version": model_version_value,
        "capabilities": ",".join(capabilities),
        "tokens": tokens_out,
        "sentences": sentences_out,
    }


def analyse(nlp, text: str):
    """Analyse a segment, windowing oversized text instead of failing the stage."""
    try:
        return nlp(text)
    except ValueError:
        pass
    from spacy.tokens import Doc

    windows = [window for window in re.split(r"(?<=[.!?。！？])\s+", text) if window.strip()]
    if not windows:
        windows = [text]
    docs = []
    limit = max(1, nlp.max_length - 1)
    for window in windows:
        # A single window can still exceed the limit: split it by characters.
        for start in range(0, len(window), limit):
            piece = window[start : start + limit]
            if piece.strip():
                docs.append(nlp(piece))
    if not docs:
        return nlp("")
    words = [token.text for doc in docs for token in doc]
    if not words:
        return docs[0]
    return Doc(docs[0].vocab, words=words, whitespace=[" "] * len(words))


def capabilities_of(nlp) -> tuple[str, ...]:
    """Which linguistic layers the loaded pipeline actually provides."""
    components = set(getattr(nlp, "pipe_names", ()) or ())
    capabilities = ["tokenization"]
    capabilities.append("dependency_sentences" if "parser" in components else "sentencizer")
    if "tagger" in components or "attribute_ruler" in components:
        capabilities.append("pos")
    if "parser" in components:
        capabilities.append("dependency")
    if "ner" in components:
        capabilities.append("entities")
    if "lemmatizer" in components or "attribute_ruler" in components:
        capabilities.append("lemma")
    return tuple(dict.fromkeys(capabilities))


def shape_of(text: str, max_length: int = 12) -> str:
    """spaCy-style word shape (``xxxx``, ``Xxxx``, ``dddd``), truncated."""
    out = []
    for char in text[:max_length]:
        if char.isupper():
            out.append("X")
        elif char.islower():
            out.append("x")
        elif char.isdigit():
            out.append("d")
        else:
            out.append(char)
    return "".join(out) + "…" if len(text) > max_length else "".join(out)


def normalise_token(text: str) -> str:
    return unicodedata.normalize("NFKC", (text or "")).strip().lower()


def token_timing(token, segment_words: list[dict]) -> dict[str, Any]:
    """Map a spaCy token onto WhisperX word timing by deterministic matching.

    spaCy and WhisperX split text differently (clitics, punctuation, merged
    tokens), so positional equality is tried first and reported as ``aligned``;
    a bounded ±2-word search is allowed and reported as ``approximate`` with a
    reduced confidence. Anything else is ``unmatched`` with null timestamps
    rather than a silent guess.
    """
    if not segment_words:
        return {"token_start_time": None, "token_end_time": None,
                "timestamp_alignment_status": NO_TIMING, "timestamp_alignment_confidence": 0.0}
    match = find_word_for_token(token.text, segment_words, token.i)
    if match is None:
        return {"token_start_time": None, "token_end_time": None,
                "timestamp_alignment_status": UNMATCHED, "timestamp_alignment_confidence": 0.0}
    word = match["word"]
    start, end = word.get("start_time"), word.get("end_time")
    if start is None or end is None:
        return {"token_start_time": None, "token_end_time": None,
                "timestamp_alignment_status": word.get("alignment_status") or UNMATCHED,
                "timestamp_alignment_confidence": 0.0}
    return {"token_start_time": round(float(start), 6), "token_end_time": round(float(end), 6),
            "timestamp_alignment_status": match["status"],
            "timestamp_alignment_confidence": match["confidence"]}


def find_word_for_token(token_text: str, segment_words: list[dict],
                        token_index: int) -> dict[str, Any] | None:
    target = normalise_token(token_text)
    if not target:
        return None
    if token_index < len(segment_words):
        candidate = segment_words[token_index]
        if normalise_token(candidate.get("word") or "") == target:
            return {"word": candidate, "status": ALIGNED, "confidence": 1.0}
    for offset in range(-2, 3):
        index = token_index + offset
        if 0 <= index < len(segment_words):
            candidate = segment_words[index]
            if normalise_token(candidate.get("word") or "") == target:
                return {"word": candidate, "status": APPROXIMATE,
                        "confidence": round(max(0.2, 1.0 - abs(offset) * 0.3), 3)}
    # Punctuation-only token: inherit the neighbouring word's span, flagged.
    if not normalise_token(token_text).strip(".,;:!?¡¿\"'()[]{}—–-…"):
        return {"word": segment_words[min(token_index, len(segment_words) - 1)],
                "status": APPROXIMATE, "confidence": 0.2}
    return None


if __name__ == "__main__":
    sys.exit(main())
