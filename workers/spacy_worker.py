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
import logging
import re
import sys
import time
import traceback
import unicodedata
from typing import Any, Sequence

FALLBACK_CAPABILITIES = ("tokenization", "sentencizer")

#: Worker diagnostics go through logging so they reach the per-stage log file that
#: ``run_command`` captures (stdout+stderr merged) instead of vanishing with the process.
#: With no configured handler, ``logging.lastResort`` still writes WARNING to stderr.
LOGGER = logging.getLogger("spacy_worker")

#: Explicit alignment verdicts; a consumer must never have to guess.
ALIGNED = "aligned"
APPROXIMATE = "approximate"
UNMATCHED = "unmatched"
NO_TIMING = "no_timing"

#: Stripped when comparing a spaCy token against a WhisperX word. WhisperX emits
#: "world," where spaCy emits "world" and ","; without this the two never match.
#: A character set for str.strip(): spaces and ASCII punctuation plus the
#: quotation, dash and inversion marks this corpus's languages actually use.
PUNCTUATION = (
    " \t\n\r\f\v"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    "\u00a1\u00bf\u2013\u2014\u2018\u2019\u201c\u201d\u2026\u00ab\u00bb"
)


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
    parser.add_argument("--language-detection", default=None,
                        help="WhisperX language_detection grade as compact JSON, or 'none'")
    parser.add_argument("--trust-low-language-detection", action="store_true",
                        help="keep the configured model when the detection was graded low")
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
            # Nothing was detected here: this variant forces English, so there is no
            # guess to grade and no policy to apply. Recorded as such rather than as
            # "absent", which would read like a missing WhisperX grade.
            trusted = True
            reliability_record: dict[str, Any] = {"status": "not_applicable"}
        else:
            texts = None
            grade, grade_available = parse_language_detection(
                getattr(args, "language_detection", None))
            trust_low = bool(getattr(args, "trust_low_language_detection", False))
            # The orchestrator's sentinel for "whisperx.json exists but is not readable
            # JSON". It arrives as a grade so the fingerprint records it, but it is not a
            # verdict: it says the check could not run, not that the detection is bad.
            unreadable = bool(grade_available and grade.get("status") == "unreadable")
            low_grade = bool(grade_available and grade.get("status") == "low")
            trusted = not (low_grade and not trust_low)
            reliability_record = grade if grade_available else {"status": "absent"}
            # The policy is applied *here*, not inside select_model: that resolver's
            # contract is "which installed model serves this language", and quietly
            # teaching it about detection quality would make a model-availability
            # decision depend on an ASR confidence number.
            selection = select_model(language if trusted else None, configured,
                                     installed_models(), fallback=args.fallback_model)
            if not trusted:
                LOGGER.warning(
                    "language %r was auto-detected with low reliability (probability %s; %s). "
                    "spacy.trust_low_language_detection = false, so the detection was not used "
                    "to choose a model: the source variant was demoted to model=%r "
                    "(status=%r, capabilities=%s), which still yields tokens and sentences but "
                    "no lemmas, POS tags or dependencies. Set "
                    "spacy.trust_low_language_detection = true to accept a low-confidence "
                    "detection again.",
                    language, _probability_text(grade), _reasons_text(grade),
                    selection.get("model"), selection.get("status"),
                    selection.get("capabilities"))
            elif unreadable:
                # A grade dict that reports the document could not be read: the default
                # behaviour is kept (the language is trusted), but the reason is not "no
                # grade shipped with this dataset" — someone's upstream artifact is
                # corrupt, and saying "absent" would send them looking for a missing file.
                LOGGER.warning(
                    "the raw WhisperX document for this dataset exists but could not be read, "
                    "so the reliability of the detected language %r could not be checked; the "
                    "detection was trusted by default and the model choice was left as-is "
                    "(model=%r, status=%r). Re-run the whisperx stage to rewrite "
                    "speech/raw/whisperx.json.",
                    language, selection.get("model"), selection.get("status"))
            elif not grade_available:
                # No grade at all: the flag was not passed, the argv value was the
                # literal "none", or it did not parse. The default behaviour is kept,
                # but silently skipping the check would make the policy look stronger
                # than it is on exactly the datasets that predate the grade.
                LOGGER.warning(
                    "no WhisperX language_detection grade was available, so the reliability of "
                    "the detected language %r could not be checked; the model choice was left "
                    "as-is (model=%r, status=%r). Re-run the whisperx stage to record a grade.",
                    language, selection.get("model"), selection.get("status"))
            elif grade_available and low_grade:
                LOGGER.warning(
                    "language %r was auto-detected with low reliability (probability %s; %s); "
                    "the full pipeline for it was still selected (model=%r, status=%r) because "
                    "spacy.trust_low_language_detection = true — set "
                    "spacy.trust_low_language_detection = false to require a trustworthy "
                    "detection before building the linguistic layer.",
                    language, _probability_text(grade), _reasons_text(grade),
                    selection.get("model"), selection.get("status"))
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
            language_reliability=reliability_record,
            language_reliability_trusted=trusted,
        )
        raw_output.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
        payload.update({
            "status": "ok",
            "tool_version": spacy.__version__,
            "model_version": document["model_version"],
            "selected_model": document["selected_model"],
            "model_selection_status": selection["status"],
            "language_reliability": document.get("language_reliability"),
            "language_reliability_trusted": document.get("language_reliability_trusted"),
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


#: Literal the orchestrator passes when the dataset carries no grade at all.
NO_GRADE = "none"


def parse_language_detection(raw) -> tuple[dict[str, Any] | None, bool]:
    """WhisperX's reliability grade as ``(grade, available)``.

    ``(None, False)`` covers every way a grade can fail to arrive — the flag was never
    passed (an older orchestrator), the literal ``none`` (no raw document in this
    dataset), unparseable JSON, or JSON that is not an object. None of them may cost a
    transcript: the caller keeps today's model choice and says the check could not run.

    The two-tuple shape is the point: a caller that only looked at ``grade`` could not
    tell "no grade exists" from "the grade exists and is trustworthy", and those get
    different warnings. A grade whose ``status`` is ``unreadable`` is the third case —
    available, present in the fingerprint, and not a verdict about the detection. It is
    reported by the orchestrator when the raw document exists but does not parse, and is
    handled by the caller as "available but not low".
    """
    if raw is None:
        return None, False
    text = str(raw).strip()
    if not text or text.lower() == NO_GRADE:
        return None, False
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return None, False
    if not isinstance(parsed, dict):
        return None, False
    return parsed, True


def _probability_text(grade: dict[str, Any] | None) -> str:
    probability = (grade or {}).get("probability")
    if isinstance(probability, (int, float)):
        return f"{float(probability):.2f}"
    return "unknown"


def _reasons_text(grade: dict[str, Any] | None) -> str:
    reasons = (grade or {}).get("reasons")
    if isinstance(reasons, list) and reasons:
        return "; ".join(str(reason) for reason in reasons)
    return "no reasons recorded"


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
                   language, language_reliability: dict[str, Any] | None = None,
                   language_reliability_trusted: bool = True) -> dict[str, Any]:
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
        sentences = list(doc.sents)
        words = words_by_segment.get(segment_id, [])
        # Align over the whole segment at once: monotonic matching needs the
        # sequence of tokens, and a sentence boundary must not reset the cursor.
        timings = align_token_texts([token.text for token in doc], words) if variant == "source" \
            else [{"token_start_time": None, "token_end_time": None,
                   "timestamp_alignment_status": NO_TIMING, "timestamp_alignment_confidence": 0.0}
                  for _ in doc]

        for sentence_index, sent in enumerate(sentences):
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
                    **timings[token.i],
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
        # How much the language above was trusted, next to the model it produced: a
        # reader of this document must be able to see that a full pipeline rests on a
        # sub-window guess, without going back to the WhisperX raw file.
        "language_reliability": language_reliability or {"status": "absent"},
        "language_reliability_trusted": bool(language_reliability_trusted),
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


def core_text(text: str) -> str:
    """Comparison key: NFKC, casefolded, with surrounding punctuation stripped.

    WhisperX returns ``world,`` where spaCy returns ``world``. Comparing on the
    word core is what makes those two the same word. Only the *edges* are
    stripped: an apostrophe inside ``don't`` is part of the word.
    """
    normalised = unicodedata.normalize("NFKC", text or "").strip().casefold()
    return normalised.strip(PUNCTUATION)


def has_core(text: str) -> bool:
    """True when a token carries lexical content (not just punctuation/space)."""
    return bool(core_text(text))


def raw_key(text: str) -> str:
    """Comparison key that keeps punctuation: for matching an ASR punctuation word."""
    return unicodedata.normalize("NFKC", text or "").strip().casefold()


def _is_fragment(core: str, word_core: str) -> bool:
    """True when a token is a proper piece of a word the cursor is sitting on.

    spaCy splits ``l'homme`` into ``l'`` + ``homme`` while the ASR kept one token.
    Requiring containment is what separates that real case from a word that is
    simply not in the recording, which must stay ``unmatched`` instead of being
    handed the timestamp of whatever word came next.
    """
    return bool(core) and len(word_core) > len(core) and core in word_core


def align_token_texts(token_texts: Sequence[str], segment_words: list[dict]) -> list[dict[str, Any]]:
    """Align spaCy tokens to WhisperX word timings in a single monotonic pass.

    The two tokenisers disagree constantly: WhisperX attaches punctuation to the
    preceding word (``world,``) while spaCy emits ``world`` and ``,`` separately.
    Matching token *i* to word *i* therefore drifts by one after the first comma
    and mislabels almost every timestamp — which is worse than admitting there is
    no timestamp. Word order is identical in both sequences, so this walks them
    together with a single forward cursor:

    * a token whose core equals the word the cursor is on is ``aligned`` (1.0);
    * an exact core match a few words ahead is ``approximate``: the timestamp is
      right but the 1:1 pairing is not provable when a word repeats;
    * a token that is a proper fragment of the current word (``l'`` + ``homme``)
      borrows that word's span *without consuming it*, so the next token can
      still match the same word;
    * punctuation borrows the span of the word it accompanies (0.2), unless the
      ASR timed that mark as its own word, in which case that timing is used;
    * anything else gets null timestamps and ``unmatched``. Only an exact match
      against the cursor is ever reported as ``aligned``.

    The pass never reorders and never invents a timestamp for a word it could not
    name. When there are no word timings at all every token is ``no_timing``.
    """
    if not segment_words:
        return [{"token_start_time": None, "token_end_time": None,
                 "timestamp_alignment_status": NO_TIMING, "timestamp_alignment_confidence": 0.0}
                for _ in token_texts]

    timings: list[dict[str, Any]] = []
    cursor = 0
    last_matched: dict[str, Any] | None = None
    total = len(segment_words)

    for text in token_texts:
        core = core_text(text)

        if not core:
            # Punctuation/whitespace: the ASR sometimes times it as its own word.
            punct = _match_raw(segment_words, cursor, text, window=1)
            if punct is not None:
                index, distance = punct
                word = segment_words[index]
                last_matched = word
                cursor = index + 1
                timings.append(_timing_from_word(
                    word, ALIGNED if distance == 0 else APPROXIMATE,
                    1.0 if distance == 0 else 0.5, fallback=UNMATCHED))
                continue
            borrowed = last_matched
            if borrowed is None and cursor < total:
                borrowed = segment_words[cursor]  # leading mark: belongs to what follows
            timings.append(_timing_from_word(borrowed, APPROXIMATE, 0.2, fallback=UNMATCHED))
            continue

        exact = _match_word(segment_words, cursor, core)
        if exact is not None:
            index, distance = exact
            word = segment_words[index]
            last_matched = word
            cursor = index + 1
            timings.append(_timing_from_word(
                word, ALIGNED if distance == 0 else APPROXIMATE,
                1.0 if distance == 0 else round(max(0.3, 1.0 - distance * 0.25), 3),
                fallback=UNMATCHED))
            continue

        current_core = core_text(segment_words[cursor].get("word") or "") if cursor < total else ""
        if _is_fragment(core, current_core):
            # Do not advance: the sibling fragment must still match this word.
            timings.append(_timing_from_word(segment_words[cursor], APPROXIMATE, 0.3, fallback=UNMATCHED))
            continue

        timings.append({"token_start_time": None, "token_end_time": None,
                        "timestamp_alignment_status": UNMATCHED, "timestamp_alignment_confidence": 0.0})
    return timings


def _match_word(segment_words: list[dict], cursor: int, core: str,
                window: int = 3) -> tuple[int, int] | None:
    """Nearest core match at or after ``cursor`` within ``window`` words."""
    for distance in range(0, window + 1):
        index = cursor + distance
        if index >= len(segment_words):
            break
        if core_text(segment_words[index].get("word") or "") == core:
            return index, distance
    return None


def _match_raw(segment_words: list[dict], cursor: int, text: str,
               window: int = 1) -> tuple[int, int] | None:
    """Exact (punctuation-included) match for a mark the ASR timed on its own."""
    target = raw_key(text)
    if not target:
        return None
    for distance in range(0, window + 1):
        index = cursor + distance
        if index >= len(segment_words):
            break
        if raw_key(segment_words[index].get("word") or "") == target:
            return index, distance
    return None


def _timing_from_word(word: dict | None, status: str, confidence: float,
                      *, fallback: str) -> dict[str, Any]:
    if word is None:
        return {"token_start_time": None, "token_end_time": None,
                "timestamp_alignment_status": fallback, "timestamp_alignment_confidence": 0.0}
    start, end = word.get("start_time"), word.get("end_time")
    if start is None or end is None:
        # The word exists but its own timing is unknown (unaligned or truncated).
        return {"token_start_time": None, "token_end_time": None,
                "timestamp_alignment_status": word.get("alignment_status") or fallback,
                "timestamp_alignment_confidence": 0.0}
    return {"token_start_time": round(float(start), 6), "token_end_time": round(float(end), 6),
            "timestamp_alignment_status": status, "timestamp_alignment_confidence": round(confidence, 3)}


if __name__ == "__main__":
    sys.exit(main())
