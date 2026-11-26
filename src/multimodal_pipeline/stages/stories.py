"""`stories`: storytelling events, the stories told inside them, and uncertain spans.

Prompt **v2** implements the domain expert's definition: a *storytelling event* is the
conversational activity of launching and telling (pre-telling, telling, coda), and a *story*
is the represented "what" inside it. They are marked separately because they are different
claims — the pre-telling belongs to the event and not necessarily to the story, a coda can
close the event while the story ended earlier, and one event can carry several stories. v1
asked for "an anecdote with a beginning and an end" in a single span with ``parent_id``
nesting; that model is gone from the code (it stays in Git history) and asking for
``stories.prompt_version: v1`` now fails at config-parse time.

Four properties carry the design:

* **Emptiness is a result, not a failure.** Empty arrays are the expected and correct
  answer when the transcript supports nothing, and the count is recorded per window. A
  model that is not allowed to answer "nothing" stretches a greeting into a story.
* **No time is invented anywhere.** The model cites a ``segment_id`` and copies a verbatim
  ``transcript_quote``; this module locates that quote inside the cited segment's text and
  maps its **edge words** to word times through ``speech/words.parquet``. A quote that is
  not there is a validation breach, and so is a quote that occurs more than once in the
  cited segment: neither says where the boundary is, and both are retried. The quote has to
  cover whole words (a one-letter quote cannot match inside ``Ahora``), the alignment
  accepts the split and merged tokens a different tokenizer produces (``can`` + ``'t`` for
  ``can't``), and an edge word with no exact alignment falls back to the segment's own
  boundary, labelled ``segment``, rather than borrowing a neighbour's time.
* **Speaker labels are attributed, not guessed.** Labels come from automatic diarization —
  Nemotron by default, whose turns *overlap* — so a word is labelled only when one speaker
  dominates it, and ``unknown`` otherwise (``story_speakers``).
* **Windowing is honest about what it clips.** The default is the whole transcript in one
  request. When a window size is configured, an event whose boundary (or whose marked
  interruption) falls outside the window is dropped *and counted* as one gap, with the
  number of stories it took with it counted apart, never quietly kept — and never rejected
  because a neighbouring window's stretch was visible as context.

Nothing here imports a heavy tool: the stage is httpx, pyarrow and stdlib, reached over
HTTP by the same OpenAI-compatible endpoint translation uses.
"""

from __future__ import annotations

import json
import random
import re
import string
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import httpx
import pyarrow as pa

from ..artifacts import atomic_write_json, atomic_write_text, read_json
from ..config import stable_hash
from ..exceptions import StageError, ValidationError
from ..schemas import (
    STORIES_SCHEMA,
    STORYTELLING_EVENTS_SCHEMA,
    UNCERTAIN_SPANS_SCHEMA,
    read_table,
    table_columns,
    write_table,
)
from ..story_speakers import (
    DEFAULT_MAX_COMPETING,
    DEFAULT_MIN_COVERAGE,
    UNKNOWN_SPEAKER,
    attribute_interval,
    attribute_words,
    emitted_labels,
    render_runs,
    speaker_runs,
)
from .base import Stage, StageContext
from .translation import extract_message_content

#: Bumped whenever a table's columns change meaning. 2.0 was the event/story split (v1's
#: `parent_id`, `title` and numeric `confidence` gone); 2.1 is `story_evidence` becoming the
#: expert's list of explanations, with the segments it used to hold moved to
#: `evidence_segment_ids`. Both versions exist only in local dry runs: no corpus table and
#: no pushed commit carries 2.0, so nothing is rewritten for it.
SCHEMA_VERSION = "2.1"

#: The only supported prompt version. See `PROMPTS`.
PROMPT_VERSION = "v2"

#: Decimal places a computed boundary is compared at. Times now come from word alignment,
#: so this is only the tolerance for float noise (and for a real 1 ms difference between a
#: story's start and its event's), never a rounding of what the model said — it says no
#: times at all.
BOUNDARY_TOLERANCE = 1e-3

#: Closed vocabularies. A value outside them is a breach rather than a coercion: the model
#: picking a fourth boundary type means it answered a different question, and quietly
#: mapping it to "uncertain" would hide that.
START_BOUNDARY_TYPES = ("pre_telling", "story_launch", "uncertain")
END_BOUNDARY_TYPES = ("coda", "listener_acknowledgment", "activity_shift", "uncertain")
CONFIDENCE_VALUES = ("high", "medium", "low")

#: Boundary resolution precisions: word alignment, or the fallback to the cited segment.
PRECISION_WORD = "word"
PRECISION_SEGMENT = "segment"

#: Characters a quote or a token may carry at its edges without those edges being part
#: of the word: ASCII punctuation, string.whitespace, and the marks Spanish and quoted
#: speech actually use (``¿¡«»“”‘’``) plus the dashes and ellipsis a transcript carries.
_EDGE_CHARACTERS = string.whitespace + string.punctuation + "¿¡«»“”‘’—–…"
_TOKEN_RE = re.compile(r"\S+")

#: How many tokens (or words) one split or merged correspondence may span. Two covers
#: ``can`` + ``'t``; four is slack for a tokenizer that broke a numeral into pieces.
_MAX_SPLIT_OR_MERGE = 4

PROMPTS: dict[str, str] = {
    # Armine's definition. Her annotation rules are kept as she wrote them — the sentence
    # that decides a case ("an empty answer is the expected and correct answer", "never
    # invent") is not paraphrased — and three sections were adapted because they describe
    # *this* pipeline rather than her WhisperX JSON input: the INPUT section describes our
    # transcript lines, the OUTPUT contract names `segment_id` where she had
    # `time_seconds` (the pipeline computes times; a model that types seconds is typing the
    # one thing nothing can check), and `story_evidence` keeps her three-part explanation
    # while `evidence_segment_ids` is added beside it for the segments themselves.
    "v2": (
        "You are an analyst marking storytelling events and the stories told inside them "
        "in a conversation transcript.\n"
        "\n"
        "Your goal is to mark the boundaries of each storytelling event and, separately, "
        "the boundaries of each story told within it. A story is the \"what\" being "
        "represented; a storytelling event is the conversational activity of launching and "
        "telling it.\n"
        "\n"
        "SPEAKER ATTRIBUTION LIMITATION\n"
        "Speaker labels come from automatic diarization. They are arbitrary ids, they may "
        "be wrong, and a word the diarizer could not decide is labelled \"unknown\". Treat "
        "a label as weak evidence about who is talking, never as a person's name, and say "
        "so when a boundary depends on it. A speaker you name must be one of the labels "
        "shown in square brackets in the transcript below — in the segments you are "
        "answering about or in their context — or \"unknown\".\n"
        "\n"
        "WORKING CRITERIA\n"
        "A stretch of the transcript is a possible story when it satisfies one or both of "
        "these criteria:\n"
        "(1) it represents one or more events - including events that are causally linked "
        "to one another, and parts of a single event - not necessarily in chronological "
        "order, and not necessarily involving the speakers who are talking now; and/or\n"
        "(2) it foregrounds experience: embodied or intentional experience, or an "
        "evaluation of an experience (for example, recounting one's own reaction to a "
        "book, a place or a conversation).\n"
        "A report, an opinion, an explanation or a description is not automatically a "
        "story. Some stretches are genuinely ambiguous; when one is, explain the evidence "
        "for and against marking it rather than choosing silently.\n"
        "\n"
        "STORY EVIDENCE\n"
        "For each story, say in `story_evidence` why it is a story: a short list of "
        "explanations in your own words naming which events are represented, how they are "
        "causally or experientially connected, and where embodiment, intentionality or "
        "evaluation shows up. Give the segments that carry that evidence in "
        "`evidence_segment_ids`; they must lie inside the story's own start and end.\n"
        "\n"
        "BOUNDARY GUIDANCE\n"
        "Start of a storytelling event: a possible pre-telling - the gist or bottom line "
        "of the story, a question or a reference that initiates a story sequence (such as "
        "asking whether the story has been told before), or an explicit preface such as "
        "\"Let me tell you something\". Include the pre-telling in the event span; it does "
        "not automatically belong to the story. The pre-telling may be absent, in which "
        "case the event and the story may begin together.\n"
        "Start of a story: the point where the speaker enters the story world - references "
        "to time, place or person that shift from the here-and-now into the represented "
        "experience. These are clues, not formulas.\n"
        "End of a story: the last utterance belonging to the represented events or "
        "experience. Listener assessments stay outside it unless they add to the "
        "represented experience, and a story may end before the event that contains it.\n"
        "End of a storytelling event: a coda by the storyteller (\"That's it\", \"What a "
        "night\", \"That was embarrassing\"), a listener assessment that treats the story "
        "as complete, or a clear shift to another activity. A coda may belong to the event "
        "but not to the story. Listener responses that merely acknowledge that the story "
        "is over belong OUTSIDE the event. If it is unclear whether a stretch is a "
        "boundary, mark its boundary type as \"uncertain\" and say what the alternatives "
        "are.\n"
        "\n"
        "MULTIPLE TELLERS, STORIES AND INTERRUPTIONS\n"
        "An event may have several tellers; track them only when the attribution is "
        "reliable, and say \"unknown\" rather than guessing. One event may contain more "
        "than one story, including a second story. A story may be interrupted by unrelated "
        "talk and resumed: if a later segment clearly continues the earlier story, treat "
        "it as the same story and the same event, and mark the interruption in "
        "`interruptions_or_resumptions` with the segments it spans and the words that show "
        "it. Do not join stretches by topic similarity only. Do not force every stretch "
        "into a story or an event: unrelated conversation is the normal case.\n"
        "\n"
        "ANNOTATION RULES\n"
        "- Copy exact wording into transcript_quote from the `text:` line of the cited "
        "segment. Never invent words and never silently correct them: quotes are checked "
        "against the transcript, and a quote that is not there invalidates the whole "
        "answer.\n"
        "- Quote enough surrounding words that the quote occurs only ONCE in that segment. "
        "A quote that occurs twice does not say where the boundary is and invalidates the "
        "answer; extend a short, repeating quote with the words around it.\n"
        "- Do not output times. The pipeline computes them from the segment ids and quotes "
        "you cite, so there is no timestamp precision for you to invent.\n"
        "- Cite the segment_id of every boundary. A quote must be copied from that ONE "
        "segment's text, and a boundary must lie inside the segments you are answering "
        "about.\n"
        "- Keep the story and event boundaries separate even when they coincide.\n"
        "- When something is unclear, give the best-supported estimate and explain it.\n"
        "- Use \"unknown\" when there is no reliable speaker label.\n"
        "- Use empty arrays when nothing is supported. An empty answer is the expected and "
        "correct answer when the transcript supports no storytelling event and no story. "
        "Never stretch a non-story into a story to please the request.\n"
        "- Return ONE JSON object only, with no prose and no code fences.\n"
        "\n"
        "INPUT\n"
        "One entry per transcript segment, in three lines:\n"
        "- segment_id=<id> start=<seconds> end=<seconds>\n"
        "  text: <the segment's transcribed words, exactly>\n"
        "  speakers: [speaker label] words ... [speaker label] words ...\n"
        "\n"
        "The `text:` line is the segment's transcribed words, and it is the only place a "
        "quote may be copied from. The `speakers:` line is the same speech split into "
        "diarized runs: a label in square brackets followed by the words attributed to it, "
        "merged when consecutive, with \"unknown\" where no single speaker dominated. Read "
        "the speaker line only to decide who is talking, never to quote from it.\n"
        "Do not infer pauses, overlap, intonation, gaze or gesture from this text: it is "
        "transcribed words with diarized speaker runs and segment times, and nothing else.\n"
        "\n"
        "OUTPUT\n"
        "Return one JSON object with exactly these three keys:\n"
        "{\"storytelling_events\": [{\"event_id\": \"SE1\", "
        "\"start\": {\"segment_id\": \"seg000001\", \"transcript_quote\": \"exact words\", "
        "\"speaker\": \"speaker_0\", \"boundary_type\": \"pre_telling\", "
        "\"reason\": \"why this boundary\"}, "
        "\"end\": {\"segment_id\": \"seg000004\", \"transcript_quote\": \"exact words\", "
        "\"speaker\": \"speaker_0\", \"boundary_type\": \"coda\", "
        "\"reason\": \"why this boundary\"}, "
        "\"story_ids\": [\"S1\"], \"tellers\": [\"speaker_0\"], "
        "\"interruptions_or_resumptions\": [{\"start_segment_id\": \"seg000002\", "
        "\"end_segment_id\": \"seg000003\", \"transcript_quote\": \"exact words\", "
        "\"description\": \"what happens instead\"}], "
        "\"confidence\": \"high\", \"notes\": \"\"}], "
        "\"stories\": [{\"story_id\": \"S1\", \"event_id\": \"SE1\", "
        "\"start\": {\"segment_id\": \"seg000001\", \"transcript_quote\": \"exact words\", "
        "\"speaker\": \"speaker_0\", \"reason\": \"why this boundary\"}, "
        "\"end\": {\"segment_id\": \"seg000003\", \"transcript_quote\": \"exact words\", "
        "\"speaker\": \"speaker_0\", \"reason\": \"why this boundary\"}, "
        "\"tellers\": [\"speaker_0\"], \"summary\": \"the represented what\", "
        "\"story_evidence\": [\"an explanation of the evidence\", "
        "\"another explanation\"], "
        "\"evidence_segment_ids\": [\"seg000001\", \"seg000002\"], "
        "\"confidence\": \"high\", \"notes\": \"\"}], "
        "\"uncertain_spans\": [{\"start_segment_id\": \"seg000002\", "
        "\"end_segment_id\": \"seg000003\", \"transcript_quote\": \"exact words\", "
        "\"issue\": \"what could not be decided\"}]}\n"
        "Allowed boundary_type for a start: pre_telling, story_launch, uncertain.\n"
        "Allowed boundary_type for an end: coda, listener_acknowledgment, activity_shift, "
        "uncertain.\n"
        "confidence is one of: high, medium, low. story_evidence and evidence_segment_ids "
        "are required for every story and must not be empty. Use empty arrays when nothing "
        "is supported; omit nothing.\n"
        "\n"
        "VIDEO: {video_id}\n"
        "\n"
        "CONTEXT (context only - never cite these ids):\n{context}\n"
        "\n"
        "TRANSCRIPT (the segments you are answering about; cite only these ids):\n"
        "{requested}\n"
    )
}


class StoriesTransportError(RuntimeError):
    """Transient endpoint problem worth retrying."""


# --------------------------------------------------------------- quotes and word times


def _normalise_with_map(text: str) -> tuple[str, list[int]]:
    """Casefolded, whitespace-collapsed text plus the source index of each character.

    Only two things are normalised, and the brief allows exactly these: case, and runs of
    whitespace collapsing to one space. Every other character is kept, so a quote that
    paraphrases (drops a comma, changes a word) does not match — which is the point: the
    answer must be checkable against the transcript, not approximately related to it.
    """
    out: list[str] = []
    index: list[int] = []
    for position, char in enumerate(text):
        if char.isspace():
            if out and out[-1] == " ":
                continue
            out.append(" ")
            index.append(position)
            continue
        for piece in char.casefold():
            out.append(piece)
            index.append(position)
    return "".join(out), index


def _normalise_quote(quote: str) -> str:
    """A quote as it is compared: case and whitespace normalised, edges stripped.

    Stripping leading and trailing punctuation is the third permitted normalisation: a
    model that quotes ``let me tell you something`` without the sentence's final comma has
    quoted the segment, and refusing that would reject a correct answer. Nothing inside the
    quote is stripped.
    """
    normalised, _index = _normalise_with_map(quote)
    return normalised.strip(_EDGE_CHARACTERS)


def _token_cores(text: str) -> list[tuple[int, int]]:
    """Each word token of ``text`` as its punctuation-stripped **core** range.

    A token's core is what remains after leading and trailing punctuation is removed, and
    it includes the non-ASCII marks Spanish and quoted speech bring (``¿¡«»“”``). The core
    is what a quote has to cover: quoting ``not that one`` out of ``'not that one!'`` is
    quoting that token, while quoting ``can`` out of ``can't`` is not — the apostrophe is
    inside the token, so ``can`` does not end at the token's end. A token made only of
    punctuation has no core and cannot be a quote edge.
    """
    cores: list[tuple[int, int]] = []
    for match in _TOKEN_RE.finditer(text):
        start, end = match.start(), match.end()
        while start < end and text[start] in _EDGE_CHARACTERS:
            start += 1
        while end > start and text[end - 1] in _EDGE_CHARACTERS:
            end -= 1
        if end > start:
            cores.append((start, end))
    return cores


def quote_occurrences(quote: str, text: str) -> list[tuple[int, int]]:
    """Every **word-aligned** occurrence of ``quote`` in ``text``, in order.

    A hit counts only when it starts at a token core's start and ends at a token core's
    end. That is the whole difference from a plain substring search, and it is a real
    defect it removes: on a Spanish transcript the one-letter quote ``a`` used to match
    inside ``Ahora`` and gave a citable boundary the model never meant, and a quote like
    ``that`` matched its first appearance in a segment that said it three times.

    The comparison itself is unchanged and stays exact after the three permitted
    normalisations (case, whitespace, surrounding punctuation): no fuzzy matching, no
    ``difflib``, no longest-common-substring.
    """
    needle = _normalise_quote(quote)
    if not needle:
        return []
    haystack, index = _normalise_with_map(text)
    cores = _token_cores(text)
    starts = {start for start, _end in cores}
    ends = {end for _start, end in cores}
    occurrences: list[tuple[int, int]] = []
    position = haystack.find(needle)
    while position >= 0:
        start = index[position]
        end = index[position + len(needle) - 1] + 1
        if start in starts and end in ends:
            occurrences.append((start, end))
        position = haystack.find(needle, position + 1)
    return occurrences


def locate_quote(quote: str, text: str) -> tuple[int, int] | None:
    """The character range of a quote that occurs exactly once in ``text``, else None.

    ``None`` covers both "not there" and "ambiguous", because neither is a location. The
    two are different answers to the model (the validator asks it for more surrounding
    words in the second case), so a caller that has to say which one it is uses
    :func:`quote_occurrences`.
    """
    occurrences = quote_occurrences(quote, text)
    return occurrences[0] if len(occurrences) == 1 else None


def _squash(token: str) -> str:
    """A token reduced to what comparison uses: its letters and digits, case-folded.

    Punctuation is dropped *including* punctuation inside the token, because that is how
    the two sides of a split or merged correspondence meet: the alignment's ``can`` + ``'t``
    and the transcript's ``can't`` both squash to ``cant``, and ``¿Qué`` meets ``Qué``.
    Digits are kept, so ``12`` and ``twelve`` stay different words — a boundary that would
    have to cross that gap falls back to the segment rather than borrowing a neighbour.
    """
    return "".join(char for char in token.casefold() if char.isalnum())


def _alignment_moves(token_keys: Sequence[str],
                     word_keys: Sequence[str]) -> list[tuple[int, int, int, int]]:
    """Ordered correspondence between text tokens and alignment words.

    Each move is ``(token_index, token_count, word_index, word_count)``: a run of text
    tokens covered by a run of alignment words whose squashed keys concatenate to the same
    string. Three shapes are possible, and the third is why this is not a zip:

    * 1:1 — the ordinary case, one spoken word per transcript token;
    * several words to one token — ``can`` + ``'t`` is the transcript's ``can't``;
    * one word to several tokens — the reverse asymmetry, a tokenizer that split what the
      aligner kept together.

    Tokens and words that do not correspond are skipped and contribute no timing, which is
    how a word the text does not contain (``twelve`` for ``12``) ends up with none. The
    best alignment is the one that covers the most words; ties keep the plainest shape,
    because a 1:1 match is always the intended reading of two equal keys.
    """
    tokens, words = list(token_keys), list(word_keys)
    memo: dict[tuple[int, int], tuple[int, tuple[tuple[int, int, int, int], ...]]] = {}

    def solve(index: int, position: int) -> tuple[int, tuple[tuple[int, int, int, int], ...]]:
        if index >= len(tokens) or position >= len(words):
            return 0, ()
        key = (index, position)
        if key in memo:
            return memo[key]
        options: list[tuple[int, tuple[tuple[int, int, int, int], ...]]] = []
        if tokens[index] and tokens[index] == words[position]:
            score, rest = solve(index + 1, position + 1)
            options.append((1 + score, ((index, 1, position, 1),) + rest))
        for count in range(2, _MAX_SPLIT_OR_MERGE + 1):
            if position + count > len(words):
                break
            joined = "".join(words[position:position + count])
            if joined and joined == tokens[index]:
                score, rest = solve(index + 1, position + count)
                options.append((count + score, ((index, 1, position, count),) + rest))
        for count in range(2, _MAX_SPLIT_OR_MERGE + 1):
            if index + count > len(tokens):
                break
            joined = "".join(tokens[index:index + count])
            if joined and joined == words[position]:
                score, rest = solve(index + count, position + 1)
                options.append((count + score, ((index, count, position, 1),) + rest))
        options.append(solve(index + 1, position))
        options.append(solve(index, position + 1))
        memo[key] = max(options, key=lambda option: option[0])
        return memo[key]

    return list(solve(0, 0)[1])


def align_token_times(text: str,
                      words: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One entry per text token: its core range and the times of the words covering it.

    ``speech/words.parquet`` carries ``character_start``/``character_end`` and on the corpus
    this was checked against (``data/processed/*/speech/words.parquet``, eight datasets)
    **every one of them is NULL**, so token placement does not depend on them: it is the
    alignment of :func:`_alignment_moves` over the segment's own tokens.

    A token covered by exactly one word (or by the words one token was split into) takes
    that word's own start and end. A token that is part of a *merged* alignment word — one
    word spanning several tokens — only takes the edge its position really is: the first
    such token may claim the word's start, the last its end, and the tokens between them
    claim neither, because the word's boundary is not their boundary.
    """
    cores = _token_cores(text)
    entries: list[dict[str, Any]] = [
        {"char_start": start, "char_end": end, "start_time": None, "end_time": None,
         "word_indexes": []}
        for start, end in cores
    ]
    token_keys = [_squash(text[start:end]) for start, end in cores]
    word_list = list(words)
    word_keys = [_squash(str(word.get("word") or "")) for word in word_list]
    for token_index, token_count, word_index, word_count in _alignment_moves(token_keys,
                                                                             word_keys):
        group = list(range(word_index, word_index + word_count))
        for offset in range(token_count):
            entries[token_index + offset]["word_indexes"] = group
        if token_count > 1:
            # One alignment word over several tokens: only its outer edges belong to a
            # token, so only those tokens may claim a time.
            word = word_list[word_index]
            entries[token_index]["start_time"] = word.get("start_time")
            entries[token_index + token_count - 1]["end_time"] = word.get("end_time")
            continue
        # One token, one word — or the several words it was split into. Both ends belong
        # to that token, and both come from the word(s) that really cover it.
        words_here = [word_list[index] for index in group]
        entries[token_index]["start_time"] = words_here[0].get("start_time")
        entries[token_index]["end_time"] = words_here[-1].get("end_time")
    return entries


def _boundary_time(segment: Mapping[str, Any], span: tuple[int, int],
                   role: str) -> tuple[float, str]:
    """The boundary instant and how precisely it is known.

    A start boundary is the **first word of the quote**'s start, an end boundary the **last
    word of the quote**'s end. Both are read from the token that really sits at that edge
    of the quote, so the time can only come from the edge word: when that token has no
    exact alignment (or no timing), the cited segment's own boundary is the honest answer
    and ``time_precision`` says ``segment``. Borrowing a neighbouring word's time would
    claim word precision for a boundary nothing places.
    """
    tokens = [token for token in segment.get("tokens") or []
              if int(token["char_start"]) >= span[0] and int(token["char_end"]) <= span[1]]
    if tokens:
        edge = tokens[0] if role == "start" else tokens[-1]
        value = edge.get("start_time") if role == "start" else edge.get("end_time")
        if value is not None:
            return float(value), PRECISION_WORD
    fallback = segment.get("start_time") if role == "start" else segment.get("end_time")
    return float(fallback), PRECISION_SEGMENT


# ------------------------------------------------------------------- speaker preparation


def prepare_segments(segments: Sequence[Mapping[str, Any]],
                     words: Sequence[Mapping[str, Any]] = (),
                     turns: Sequence[Mapping[str, Any]] = (), *,
                     speaker_source: str = "none",
                     speaker_min_coverage: float = DEFAULT_MIN_COVERAGE,
                     speaker_max_competing: float = DEFAULT_MAX_COMPETING) -> list[dict[str, Any]]:
    """Segment rows with their rendered speaker runs and their word positions attached.

    Three label sources, and the difference between them is real:

    * ``nemotron`` attributes every word against the overlapping Nemotron turns, and a
      segment with no word rows falls back to the same rule over the segment's own
      interval (the only timing evidence left).
    * ``pyannote`` uses the ``speaker_id`` already on each word (or on the segment when it
      has no words) and only replaces a *null* id with ``unknown`` — the id is already an
      assignment, so re-deciding it here would be inventing a second diarizer.
    * ``none`` labels everything ``unknown``.

    Nothing falls back from one engine to the other: ``speaker_source`` is what the operator
    asked for, and the stage records whether it was available.
    """
    ordered = sorted(segments, key=lambda row: (float(row["start_time"]),
                                                float(row["end_time"]),
                                                str(row["segment_id"])))
    by_segment: dict[str, list[dict[str, Any]]] = {}
    for word in words:
        by_segment.setdefault(str(word.get("segment_id") or ""), []).append(dict(word))

    prepared: list[dict[str, Any]] = []
    for row in ordered:
        segment_id = str(row["segment_id"])
        text = str(row.get("text") or "")
        segment_words = by_segment.get(segment_id, [])
        tokens = align_token_times(text, segment_words)
        if speaker_source == "pyannote":
            labels = [str(word.get("speaker_id") or "").strip() or UNKNOWN_SPEAKER
                      for word in segment_words]
        elif speaker_source == "nemotron":
            labels = attribute_words(segment_words, turns,
                                     min_coverage=speaker_min_coverage,
                                     max_competing=speaker_max_competing)
        else:
            labels = [UNKNOWN_SPEAKER] * len(segment_words)

        if segment_words:
            spans = [(token["char_start"], token["char_end"])
                     if token["word_indexes"] else None for token in tokens]
            token_labels = []
            for token in tokens:
                labels_here = {labels[index] for index in token["word_indexes"]}
                # Words merged into one token that disagree about the speaker make the
                # token disputed, rather than awarding it to whichever came first.
                token_labels.append(labels_here.pop() if len(labels_here) == 1
                                    else UNKNOWN_SPEAKER)
            runs = speaker_runs(text, spans, token_labels)
        else:
            fallback = UNKNOWN_SPEAKER
            if speaker_source == "pyannote":
                fallback = str(row.get("speaker_id") or "").strip() or UNKNOWN_SPEAKER
            elif speaker_source == "nemotron":
                fallback = attribute_interval(turns, row.get("start_time"), row.get("end_time"),
                                              min_coverage=speaker_min_coverage,
                                              max_competing=speaker_max_competing)
            runs = [(fallback, text.strip())]

        prepared.append({
            "segment_id": segment_id,
            "start_time": float(row["start_time"]),
            "end_time": float(row["end_time"]),
            "text": text,
            "speaker_runs": render_runs(runs),
            # The labels the prompt really shows, not the labels the word rows carry: a
            # word whose position could not be resolved is folded into a neighbouring run
            # and never appears under its own label, so naming it would describe words the
            # model was not shown.
            "speaker_labels": emitted_labels(runs),
            "tokens": tokens,
        })
    return prepared


# ------------------------------------------------------------------------ request


@dataclass
class StoriesRequest:
    """One endpoint call: one window of the transcript plus what to show as context."""

    video_id: str
    window_index: int
    requested: list[dict[str, Any]]
    context: list[dict[str, Any]]
    #: Every prepared segment of the transcript by id. Boundary times are computed from
    #: these, and an id that is not here was invented.
    segments: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: The ordered ids of the whole transcript, for "the quote is somewhere between these
    #: two segments" checks on interruptions and uncertain spans.
    segment_order: list[str] = field(default_factory=list)
    speaker_source: str = "none"
    speaker_min_coverage: float = DEFAULT_MIN_COVERAGE
    speaker_max_competing: float = DEFAULT_MAX_COMPETING
    prompt_version: str = PROMPT_VERSION
    model: str = ""
    temperature: float = 0.0
    max_output_tokens: int | None = None
    #: Which endpoint this window is addressed to. Part of the key because the answer is
    #: that endpoint's opinion: moving base_url while the model name stays a LiteLLM alias
    #: ("chat") changes the answer and nothing else in the request.
    endpoint: str = ""
    #: The provider's extra body parameters (top-p, reasoning effort, a vendor's switches).
    #: Part of the key through the request body, because a parameter that changes the
    #: answer but not one character of the prompt would otherwise reuse the old answer.
    extra_body: dict[str, Any] = field(default_factory=dict)

    @property
    def requested_ids(self) -> list[str]:
        return [str(row["segment_id"]) for row in self.requested]

    @property
    def context_ids(self) -> list[str]:
        return [str(row["segment_id"]) for row in self.context]

    @property
    def visible_ids(self) -> frozenset[str]:
        """Every segment id this prompt shows: the window plus its context neighbours.

        A reference outside it was not visible when the answer was written, so it is a
        breach rather than a window drop.
        """
        return frozenset(self.requested_ids) | frozenset(self.context_ids)

    @property
    def all_segment_ids(self) -> frozenset[str]:
        return frozenset(self.segments)

    @property
    def id_prefix(self) -> str:
        """Namespace the model's ``SE1``/``S1`` belong to: two windows both start at SE1."""
        return f"w{self.window_index}"

    @property
    def speaker_labels(self) -> frozenset[str]:
        """Labels the rendered prompt shows, plus ``unknown``.

        Read from the rows' rendered labels, which are the labels the speaker runs really
        emitted (``story_speakers.emitted_labels``) — never from the word rows, where a
        word whose position could not be resolved still carries a label the prompt never
        shows. Context labels count: the rendered context is visible, so it is allowed
        evidence for a teller or an interruption even though it is not citable.
        """
        labels = {UNKNOWN_SPEAKER}
        for row in list(self.requested) + list(self.context):
            labels.update(str(label) for label in row.get("speaker_labels") or [])
        return frozenset(labels)

    def prompt(self) -> str:
        template = PROMPTS.get(self.prompt_version)
        if not template:
            raise StageError(f"unknown stories prompt version: {self.prompt_version}")
        # Substituted rather than str.format'ed: the answer contract in the prompt is a
        # literal JSON object, and .format would need every one of its braces doubled —
        # turning the sentence an operator most needs to read into noise.
        return (template
                .replace("{video_id}", self.video_id)
                .replace("{context}", _render(self.context))
                .replace("{requested}", _render(self.requested)))

    def body(self, *, model_fallback: str = "",
             extra_body_fallback: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """The HTTP body this request really posts.

        Built once and used twice on purpose: the client sends exactly this, and the cache
        key hashes exactly this. A hand-picked list of fields in the key is how the cache
        ends up reusing an answer to a question that was not asked — ``extra_body`` and the
        context segments' timestamps were both missing from it.
        """
        body: dict[str, Any] = {
            "model": self.model or model_fallback,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": self.prompt()}],
        }
        if self.max_output_tokens:
            body["max_tokens"] = self.max_output_tokens
        body.update(dict(extra_body_fallback or {}))
        body.update(dict(self.extra_body))
        return body

    def key(self) -> str:
        """Request digest: the rendered prompt and the request body that carries it.

        The prompt is hashed as text, so *everything* the model can read is bound — every
        segment id, its times, its plain text and its speaker runs, the context and the
        instruction. The body is hashed as the client will post it, so every sampling
        parameter (and ``extra_body``) is bound too. Only then do the two identifiers that
        are not part of the question, the video and the window, name the file. ``endpoint``
        stays because the answer is that endpoint's opinion: the same body sent to another
        ``base_url`` is another question.
        """
        return stable_hash({
            "video_id": self.video_id,
            "window_index": self.window_index,
            "prompt_version": self.prompt_version,
            "endpoint": self.endpoint,
            "prompt": self.prompt(),
            "request_body": self.body(),
        }, length=24)


def _render(rows: Sequence[dict[str, Any]]) -> str:
    """Transcript lines the model can cite: id and boundaries, then text and speakers.

    Text and speaker runs are separate lines because they are separate claims. The quote
    the answer must copy is the segment's text, so it is shown as text, exactly as
    ``speech/segments.parquet`` has it (a quote is checked against that, not against a
    rendering); the runs say who the diarizer thought was talking, which is evidence for a
    boundary and never a source of quoted words. When a word could not be placed, its
    characters still appear in the runs inside a neighbouring run, and the text line is
    what the quote is checked against either way.
    """
    if not rows:
        return "(none)"
    lines: list[str] = []
    for row in rows:
        text = str(row.get("text") or "").strip()
        runs = str(row.get("speaker_runs") or "").strip() or f"[{UNKNOWN_SPEAKER}]"
        lines.append(f"- segment_id={row['segment_id']} "
                     f"start={float(row['start_time']):.3f} "
                     f"end={float(row['end_time']):.3f}")
        lines.append(f"  text: {text}")
        lines.append(f"  speakers: {runs}")
    return "\n".join(lines)


def build_requests(segments: Sequence[Mapping[str, Any]], *, video_id: str,
                   max_segments_per_request: int | None, context_segments: int,
                   prompt_version: str = PROMPT_VERSION, model: str = "",
                   temperature: float = 0.0, max_output_tokens: int | None = None,
                   speaker_source: str = "none",
                   speaker_min_coverage: float = DEFAULT_MIN_COVERAGE,
                   speaker_max_competing: float = DEFAULT_MAX_COMPETING,
                   endpoint: str = "",
                   extra_body: Mapping[str, Any] | None = None) -> list[StoriesRequest]:
    """One request for the whole transcript (default), or start-ordered windows.

    ``max_segments_per_request=None`` is the whole transcript: a storytelling event and its
    coda can sit at opposite ends of a short conversation, and splitting one manufactures
    the boundary problem the model is being asked to solve. A number restores v1's
    windows, which partition the transcript (they never overlap — overlapping *requests*
    would produce the same event twice under two ids) with ``context_segments`` neighbours
    travelling along as CONTEXT that is explicitly not citable.
    """
    if max_segments_per_request is not None and max_segments_per_request < 1:
        raise StageError("stories.max_segments_per_request must be null or positive")
    ordered = sorted(segments, key=lambda row: (float(row["start_time"]),
                                                float(row["end_time"]),
                                                str(row["segment_id"])))
    prepared = [_prepare_row(row) for row in ordered]
    index = {str(row["segment_id"]): row for row in prepared}
    order = [str(row["segment_id"]) for row in prepared]
    size = len(prepared) if max_segments_per_request is None else max_segments_per_request

    def make(index_in_transcript: int, requested: list[dict[str, Any]],
             context: list[dict[str, Any]]) -> StoriesRequest:
        return StoriesRequest(
            video_id=video_id,
            window_index=index_in_transcript,
            requested=requested,
            context=context,
            segments=index,
            segment_order=order,
            speaker_source=speaker_source,
            speaker_min_coverage=speaker_min_coverage,
            speaker_max_competing=speaker_max_competing,
            prompt_version=prompt_version,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            endpoint=endpoint,
            extra_body=dict(extra_body or {}),
        )

    if not prepared:
        return []
    if max_segments_per_request is None:
        return [make(0, prepared, [])]
    requests: list[StoriesRequest] = []
    for window, start in enumerate(range(0, len(prepared), size)):
        stop = min(len(prepared), start + size)
        context_start = max(0, start - context_segments)
        context_end = min(len(prepared), stop + context_segments)
        requests.append(make(window, prepared[start:stop],
                             prepared[context_start:start] + prepared[stop:context_end]))
    return requests


def _prepare_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """A segment row in request shape, tolerant of a caller that prepared nothing.

    Tests and any future caller may hand raw segment rows; rendering them with no
    attribution is better than a KeyError, and it is what ``speaker_source: none`` does
    anyway. A caller that hands ``tokens`` (already-aligned text tokens) gets word
    precision; one that hands nothing gets segment precision, which is the honest answer
    for a segment with no alignment behind it.
    """
    text = str(row.get("text") or "")
    runs = row.get("speaker_runs")
    labels = row.get("speaker_labels")
    tokens = row.get("tokens")
    if tokens is None:
        tokens = align_token_times(text, [])
    return {
        "segment_id": str(row["segment_id"]),
        "start_time": float(row["start_time"]),
        "end_time": float(row["end_time"]),
        "text": text,
        "speaker_runs": str(runs) if runs is not None
        else (f"[{UNKNOWN_SPEAKER}] {text.strip()}" if text.strip()
              else f"[{UNKNOWN_SPEAKER}]"),
        "speaker_labels": list(labels) if labels is not None else [UNKNOWN_SPEAKER],
        "tokens": [dict(token) for token in tokens],
    }


# ------------------------------------------------------------------ parse and validate


def parse_stories_response(text: str) -> dict[str, Any]:
    """Parse the required JSON object, tolerating code fences and a chatty wrapper."""
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
            raise ValueError(f"stories response was not JSON: {text[:200]!r}")
        try:
            payload = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"stories response was not parseable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"stories response must be a JSON object, got {type(payload).__name__}")
    return payload


@dataclass
class ValidatedAnswer:
    """One answer, split into table rows and the drops that keep a clip a known gap."""

    events: list[dict[str, Any]]
    stories: list[dict[str, Any]]
    uncertain_spans: list[dict[str, Any]]
    dropped: list[dict[str, Any]]

    @property
    def empty(self) -> bool:
        return not self.events and not self.stories and not self.uncertain_spans


def validate_response(payload: dict[str, Any], request: StoriesRequest, *,
                      stage: str = "stories") -> ValidatedAnswer:
    """Reject or drop. Returns rows for the three tables plus the dropped items.

    Two different refusals, because they mean different things:

    * a **breach** — an invented ``segment_id``, a quote that is not verbatim in the cited
      segment, a quote that occurs more than once there, an enum value outside its
      vocabulary, a duplicate id, a story whose event is not in the answer, inconsistent
      ``event.story_ids``, a story outside its event's span, an end before its start, a
      missing or blank ``story_evidence`` explanation, missing ``evidence_segment_ids`` or
      evidence outside the story's own span, a reference to a segment the prompt did not
      show — raises :class:`ValidationError`. The answer describes a transcript that does
      not exist, so nothing in it can be trusted and the whole batch is retried;
    * a **drop** is one event **group** — the event, its stories and its interruptions —
      anchored outside this window, plus any uncertain span that lies entirely outside. It
      is not this window's answer, another window may well make it, and it is returned so
      the caller can count the event once and its stories apart.

    Accepted rows carry window-namespaced ids (``w0-SE1``, ``w0-S1``): the model answers
    ``SE1`` in every window, and an id that is only unique per request cannot be a table
    key.
    """
    problems: list[str] = []
    events_raw = payload.get("storytelling_events")
    stories_raw = payload.get("stories")
    uncertain_raw = payload.get("uncertain_spans", [])
    if not isinstance(events_raw, list):
        raise ValidationError(stage, [f"storytelling_events must be a list, got "
                                      f"{type(events_raw).__name__}"])
    if not isinstance(stories_raw, list):
        raise ValidationError(stage, [f"stories must be a list, got "
                                      f"{type(stories_raw).__name__}"])
    if uncertain_raw is None:
        uncertain_raw = []
    if not isinstance(uncertain_raw, list):
        raise ValidationError(stage, [f"uncertain_spans must be a list, got "
                                      f"{type(uncertain_raw).__name__}"])

    events: list[dict[str, Any]] = []
    seen_events: set[str] = set()
    for position, raw in enumerate(events_raw):
        if not isinstance(raw, dict):
            problems.append(f"storytelling_events[{position}] is a "
                            f"{type(raw).__name__}, not an object")
            continue
        event_id = str(raw.get("event_id") or "").strip()
        label = event_id or f"event {position}"
        if not event_id:
            problems.append(f"event {position} has no event_id")
        elif event_id in seen_events:
            # Two events called SE1 make story_ids and the table key meaningless.
            problems.append(f"duplicate event_id {event_id!r} in one answer")
            continue
        else:
            seen_events.add(event_id)

        start = _resolve_boundary(raw.get("start"), request, label, "start", problems)
        end = _resolve_boundary(raw.get("end"), request, label, "end", problems)
        _check_boundary_type(raw.get("start"), START_BOUNDARY_TYPES, label, "start", problems)
        _check_boundary_type(raw.get("end"), END_BOUNDARY_TYPES, label, "end", problems)
        if start and end:
            if float(end["time"]) < float(start["time"]):
                problems.append(f"{label}: ends at {end['time']} before it starts at "
                                f"{start['time']}")
        start_type = _boundary_type(raw.get("start"))
        end_type = _boundary_type(raw.get("end"))
        story_ids = _string_list(raw.get("story_ids"), label, "story_ids", problems)
        tellers = _speaker_list(raw.get("tellers"), request, label, "tellers", problems)
        confidence = _check_confidence(raw.get("confidence"), label, problems)
        interruptions = _resolve_interruptions(raw.get("interruptions_or_resumptions"),
                                               request, label, problems)
        events.append({
            "model_id": event_id,
            "start": start,
            "end": end,
            # Carried on the entry so the row and the validated vocabulary cannot
            # disagree; a value outside the vocabulary has already raised by then.
            "start_boundary_type": start_type,
            "end_boundary_type": end_type,
            "story_ids": story_ids,
            "tellers": tellers,
            "interruptions": interruptions,
            "confidence": confidence,
            "notes": str(raw.get("notes") or "").strip(),
        })

    stories: list[dict[str, Any]] = []
    seen_stories: set[str] = set()
    for position, raw in enumerate(stories_raw):
        if not isinstance(raw, dict):
            problems.append(f"stories[{position}] is a {type(raw).__name__}, not an object")
            continue
        story_id = str(raw.get("story_id") or "").strip()
        label = story_id or f"story {position}"
        if not story_id:
            problems.append(f"story {position} has no story_id")
        elif story_id in seen_stories:
            problems.append(f"duplicate story_id {story_id!r} in one answer")
            continue
        else:
            seen_stories.add(story_id)

        event_id = str(raw.get("event_id") or "").strip()
        if not event_id:
            problems.append(f"{label}: event_id is missing")
        start = _resolve_boundary(raw.get("start"), request, label, "start", problems)
        end = _resolve_boundary(raw.get("end"), request, label, "end", problems)
        if start and end and float(end["time"]) < float(start["time"]):
            problems.append(f"{label}: ends at {end['time']} before it starts at "
                            f"{start['time']}")
        tellers = _speaker_list(raw.get("tellers"), request, label, "tellers", problems)
        confidence = _check_confidence(raw.get("confidence"), label, problems)
        evidence = _evidence_list(raw.get("story_evidence"), label, problems)
        evidence_ids = _string_list(raw.get("evidence_segment_ids"), label,
                                    "evidence_segment_ids", problems)
        if not evidence_ids:
            problems.append(f"{label}: evidence_segment_ids is required and must not be "
                            f"empty: name the segments of this story that carry its "
                            f"evidence")
        invented = [item for item in evidence_ids if item not in request.segments]
        if invented:
            problems.append(f"{label}: evidence_segment_ids names segments that are not in "
                            f"this transcript: {invented[:5]}")
        if start and end and evidence_ids:
            # The evidence has to be evidence *of this story*: an id outside the story's
            # own span describes a stretch the model did not mark as this story.
            story_range = _segment_range(start["segment_id"], end["segment_id"], request,
                                         label, problems)
            if story_range is not None:
                outside = [item for item in evidence_ids if item not in story_range]
                if outside:
                    problems.append(
                        f"{label}: evidence_segment_ids names segments outside the story's "
                        f"own {start['segment_id']}..{end['segment_id']} range: "
                        f"{outside[:5]}")
        stories.append({
            "model_id": story_id,
            "event_id": event_id,
            "start": start,
            "end": end,
            "tellers": tellers,
            "summary": str(raw.get("summary") or "").strip(),
            "story_evidence": evidence,
            "evidence_segment_ids": evidence_ids,
            "confidence": confidence,
            "notes": str(raw.get("notes") or "").strip(),
        })

    # Cross-references: an event and its stories must agree, in both directions. One-sided
    # checking was a v1 defect (a dropped child could cite a parent nobody answered), so
    # both halves are checked here and both are breaches.
    by_event: dict[str, set[str]] = {}
    for story in stories:
        if story["event_id"]:
            by_event.setdefault(story["event_id"], set()).add(story["model_id"])
    event_by_id = {event["model_id"]: event for event in events if event["model_id"]}
    for story in stories:
        if story["event_id"] and story["event_id"] not in event_by_id:
            problems.append(f"{story['model_id']}: event_id {story['event_id']!r} is not a "
                            f"storytelling_event of this answer")
        event = event_by_id.get(story["event_id"])
        if event is None or story["start"] is None or story["end"] is None:
            continue
        if event["start"] is None or event["end"] is None:
            continue
        if float(story["start"]["time"]) < float(event["start"]["time"]) - BOUNDARY_TOLERANCE:
            problems.append(f"{story['model_id']}: starts before its event "
                            f"{story['event_id']} starts")
        if float(story["end"]["time"]) > float(event["end"]["time"]) + BOUNDARY_TOLERANCE:
            problems.append(f"{story['model_id']}: ends after its event "
                            f"{story['event_id']} ends")
    for event in events:
        if not event["model_id"]:
            continue
        listed = set(event["story_ids"])
        answered = by_event.get(event["model_id"], set())
        if listed != answered:
            problems.append(
                f"{event['model_id']}: story_ids {sorted(listed)} do not match the stories "
                f"that name it as their event {sorted(answered)}")

    uncertain: list[dict[str, Any]] = []
    for position, raw in enumerate(uncertain_raw):
        if not isinstance(raw, dict):
            problems.append(f"uncertain_spans[{position}] is a {type(raw).__name__}, not "
                            f"an object")
            continue
        label = f"uncertain span {position}"
        start_id = str(raw.get("start_segment_id") or "").strip()
        end_id = str(raw.get("end_segment_id") or "").strip()
        if not start_id or not end_id:
            problems.append(f"{label}: start_segment_id and end_segment_id are required")
            continue
        range_ids = _segment_range(start_id, end_id, request, label, problems)
        if range_ids is None:
            continue
        quote = raw.get("transcript_quote")
        if not isinstance(quote, str) or not quote.strip():
            problems.append(f"{label}: transcript_quote is missing")
        elif not _quote_in_range(quote, range_ids, request):
            problems.append(f"{label}: transcript_quote is not verbatim in {start_id}"
                            f"..{end_id}: {quote[:120]!r}")
        uncertain.append({
            "start_segment_id": start_id,
            "end_segment_id": end_id,
            "start_time": float(request.segments[start_id]["start_time"]),
            "end_time": float(request.segments[end_id]["end_time"]),
            "transcript_quote": str(quote or ""),
            "issue": str(raw.get("issue") or "").strip(),
        })

    if problems:
        raise ValidationError(stage, problems)

    # Drops, after every structural check: a fabricated reference inside an event that
    # would have been dropped is still a breach (v1's finding 4), so the reject/drop line is
    # drawn on *structure* first and *window membership* second.
    #
    # The unit is the event **group** — the event, the stories told in it, and the
    # interruptions it marks — because those three are one claim. An event whose boundary
    # or interruption sits in a context segment belongs to the neighbouring window, and
    # dropping it whole is what keeps this window's tables free of a half-event; counting
    # it once (with its stories counted separately) is what keeps the gap visible instead
    # of silently shrinking the answer.
    window_ids = set(request.requested_ids)
    dropped: list[dict[str, Any]] = []
    kept_events: list[dict[str, Any]] = []
    dropped_event_ids: set[str] = set()
    stories_by_event: dict[str, list[str]] = {}
    for story in stories:
        if story["event_id"]:
            stories_by_event.setdefault(story["event_id"], []).append(story["model_id"])
    for event in events:
        outside = [boundary["segment_id"]
                   for boundary in (event["start"], event["end"])
                   if boundary["segment_id"] not in window_ids]
        outside += [segment_id
                    for interruption in event["interruptions"]
                    for segment_id in (interruption["start_segment_id"],
                                       interruption["end_segment_id"])
                    if segment_id not in window_ids]
        if outside:
            dropped.append({"kind": "event",
                            "model_id": event["model_id"],
                            "child_story_ids": stories_by_event.get(event["model_id"], []),
                            "reason": f"outside this window: {sorted(set(outside))[:5]}"})
            dropped_event_ids.add(event["model_id"])
        else:
            kept_events.append(event)

    # A window that keeps an event cannot keep a story outside it: the story-inside-event
    # check already makes that a breach, so this is the defensive half of the same rule.
    kept_stories: list[dict[str, Any]] = []
    for story in stories:
        if story["event_id"] in dropped_event_ids:
            continue
        outside = [boundary["segment_id"]
                   for boundary in (story["start"], story["end"])
                   if boundary["segment_id"] not in window_ids]
        if outside:
            dropped.append({"kind": "story", "model_id": story["model_id"],
                            "reason": f"boundary outside this window: {outside[:5]}"})
            continue
        kept_stories.append(story)

    kept_uncertain: list[dict[str, Any]] = []
    for span in uncertain:
        if (span["start_segment_id"] not in window_ids
                and span["end_segment_id"] not in window_ids):
            dropped.append({"kind": "uncertain_span",
                            "model_id": f"{span['start_segment_id']}.."
                                        f"{span['end_segment_id']}",
                            "reason": "entirely outside this window"})
            continue
        kept_uncertain.append(span)

    return ValidatedAnswer(
        events=[_event_row(event, request) for event in kept_events],
        stories=[_story_row(story, request) for story in kept_stories],
        uncertain_spans=[_uncertain_row(span, request) for span in kept_uncertain],
        dropped=dropped,
    )


def _resolve_boundary(raw: Any, request: StoriesRequest, label: str, role: str,
                      problems: list[str]) -> dict[str, Any] | None:
    """Validate one boundary object and compute its time from the transcript itself."""
    if not isinstance(raw, dict):
        problems.append(f"{label}: {role} boundary is not an object")
        return None
    segment_id = str(raw.get("segment_id") or "").strip()
    quote = raw.get("transcript_quote")
    speaker = str(raw.get("speaker") or "").strip()
    reason = str(raw.get("reason") or "").strip()
    segment = request.segments.get(segment_id)
    if not segment_id:
        problems.append(f"{label}: {role}.segment_id is missing")
        return None
    if segment is None:
        problems.append(f"{label}: {role}.segment_id {segment_id!r} is not a segment of "
                        f"this transcript")
        return None
    if not isinstance(quote, str) or not quote.strip():
        problems.append(f"{label}: {role}.transcript_quote is missing")
        return None
    if not speaker:
        problems.append(f"{label}: {role}.speaker is missing")
    elif speaker not in request.speaker_labels:
        problems.append(f"{label}: {role}.speaker {speaker!r} is not a label shown in this "
                        f"request (shown: {sorted(request.speaker_labels)})")
    occurrences = quote_occurrences(quote, str(segment.get("text") or ""))
    if not occurrences:
        problems.append(f"{label}: {role}.transcript_quote is not verbatim in {segment_id}: "
                        f"{quote[:120]!r}")
        return None
    if len(occurrences) > 1:
        # Ambiguity is its own breach, with its own instruction: the quote is really in
        # the cited segment, just not once, so the model can fix it by quoting more of the
        # words around it. Calling this "not verbatim" would teach the retry nothing.
        problems.append(
            f"{label}: {role}.transcript_quote occurs {len(occurrences)} times in "
            f"{segment_id}, so it does not say where the boundary is; quote more "
            f"surrounding words so the quote is unique in that segment: {quote[:120]!r}")
        return None
    span = occurrences[0]
    when, precision = _boundary_time(segment, span, role)
    return {"segment_id": segment_id, "quote": quote, "speaker": speaker or UNKNOWN_SPEAKER,
            "reason": reason, "time": when, "precision": precision}


def _boundary_type(raw: Any) -> str:
    return str((raw or {}).get("boundary_type") or "").strip() if isinstance(raw, dict) else ""


def _check_boundary_type(raw: Any, vocabulary: Sequence[str], label: str, role: str,
                         problems: list[str]) -> None:
    if not isinstance(raw, dict):
        return
    value = _boundary_type(raw)
    if value not in vocabulary:
        problems.append(f"{label}: {role}.boundary_type {value!r} is not one of "
                        f"{list(vocabulary)}")


def _string_list(raw: Any, label: str, field_name: str, problems: list[str]) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        problems.append(f"{label}: {field_name} must be a list, got {type(raw).__name__}")
        return []
    items: list[str] = []
    for item in raw:
        if not isinstance(item, (str, int)):
            problems.append(f"{label}: {field_name} contains a non-string {item!r}")
            continue
        items.append(str(item).strip())
    return items


def _evidence_list(raw: Any, label: str, problems: list[str]) -> list[str]:
    """``story_evidence``: her short explanations, required and each one non-blank.

    It is a list of *reasons*, not of ids — which events are represented, how they connect,
    where embodiment, intentionality or evaluation shows up. An empty list says nothing
    about why the stretch is a story, so it is a breach rather than an accepted blank, and
    so is a list of blanks. The segments carrying the evidence live in their own field
    (``evidence_segment_ids``), where the range check can see them.
    """
    if raw is None:
        problems.append(f"{label}: story_evidence is required: explain which events are "
                        f"represented, how they connect, and where embodiment, "
                        f"intentionality or evaluation shows up")
        return []
    if not isinstance(raw, list):
        problems.append(f"{label}: story_evidence must be a list of explanations, got "
                        f"{type(raw).__name__}")
        return []
    if not raw:
        problems.append(f"{label}: story_evidence must not be empty")
        return []
    items: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            problems.append(f"{label}: story_evidence contains a non-string {item!r}")
            continue
        explanation = item.strip()
        if not explanation:
            problems.append(f"{label}: story_evidence contains a blank explanation")
            continue
        items.append(explanation)
    return items


def _speaker_list(raw: Any, request: StoriesRequest, label: str, field_name: str,
                  problems: list[str]) -> list[str]:
    tellers = _string_list(raw, label, field_name, problems)
    for teller in tellers:
        if teller not in request.speaker_labels:
            problems.append(f"{label}: {field_name} names {teller!r}, which is not a label "
                            f"shown in this request")
    return tellers


def _check_confidence(raw: Any, label: str, problems: list[str]) -> str:
    value = str(raw or "").strip()
    if value not in CONFIDENCE_VALUES:
        problems.append(f"{label}: confidence {value!r} is not one of "
                        f"{list(CONFIDENCE_VALUES)}")
    return value


def _segment_range(start_id: str, end_id: str, request: StoriesRequest, label: str,
                   problems: list[str]) -> list[str] | None:
    """Ids from ``start_id`` to ``end_id`` inclusive, in transcript order."""
    for role, segment_id in (("start", start_id), ("end", end_id)):
        if segment_id not in request.segments:
            problems.append(f"{label}: {role}_segment_id {segment_id!r} is not a segment of "
                            f"this transcript")
            return None
    order = request.segment_order
    try:
        start = order.index(start_id)
        end = order.index(end_id)
    except ValueError:  # pragma: no cover - every id is in `order` by construction
        problems.append(f"{label}: {start_id}..{end_id} cannot be ordered")
        return None
    if end < start:
        problems.append(f"{label}: ends at {end_id} before it starts at {start_id}")
        return None
    return order[start:end + 1]


def _quote_in_range(quote: str, range_ids: Sequence[str],
                    request: StoriesRequest) -> bool:
    """Is the quote in one of these segments, word-aligned? Ambiguity does not matter here.

    An interruption or an uncertain span cites a stretch of segments, and its times come
    from the first and last of them: which occurrence of the quote showed the stretch is
    not a claim this row makes, so a repeated quote is still evidence that the stretch
    exists.
    """
    return any(quote_occurrences(quote, str(request.segments[segment_id].get("text") or ""))
               for segment_id in range_ids)


def _resolve_interruptions(raw: Any, request: StoriesRequest, label: str,
                           problems: list[str]) -> list[dict[str, Any]]:
    """Interruptions/resumptions: segment range plus the quote that shows it happened.

    The times are the start of the first cited segment and the end of the last: an
    interruption is a stretch of other activity, and its edges are the segments the model
    named. Nothing about the quote is used for timing, so a quote that happens to sit in
    the middle cannot shorten the interruption.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        problems.append(f"{label}: interruptions_or_resumptions must be a list, got "
                        f"{type(raw).__name__}")
        return []
    # Context segments are shown, so an interruption that happened in one is a reference
    # the model could see. Whether it lies inside this window is a *drop* question, decided
    # after the answer is structurally valid (see the drop pass) — not a reason to reject a
    # whole answer that is about a neighbouring window's event.
    visible_ids = set(request.visible_ids)
    interruptions: list[dict[str, Any]] = []
    for position, item in enumerate(raw):
        where = f"{label}: interruption {position}"
        if not isinstance(item, dict):
            problems.append(f"{where} is a {type(item).__name__}, not an object")
            continue
        start_id = str(item.get("start_segment_id") or "").strip()
        end_id = str(item.get("end_segment_id") or "").strip()
        if not start_id or not end_id:
            problems.append(f"{where}: start_segment_id and end_segment_id are required")
            continue
        invented = [segment_id for segment_id in (start_id, end_id)
                    if segment_id not in request.segments]
        if invented:
            problems.append(f"{where}: names segments that are not in this transcript: "
                            f"{invented}")
            continue
        unseen = [segment_id for segment_id in (start_id, end_id)
                  if segment_id not in visible_ids]
        if unseen:
            problems.append(f"{where}: names segments that were not shown in this request: "
                            f"{unseen}")
            continue
        range_ids = _segment_range(start_id, end_id, request, where, problems)
        if range_ids is None:
            continue
        quote = item.get("transcript_quote")
        if not isinstance(quote, str) or not quote.strip():
            problems.append(f"{where}: transcript_quote is missing")
            continue
        if not _quote_in_range(quote, range_ids, request):
            problems.append(f"{where}: transcript_quote is not verbatim in {start_id}..{end_id}")
            continue
        interruptions.append({
            "start_segment_id": start_id,
            "end_segment_id": end_id,
            "start_time": float(request.segments[start_id]["start_time"]),
            "end_time": float(request.segments[end_id]["end_time"]),
            "transcript_quote": quote,
            "description": str(item.get("description") or "").strip(),
        })
    return interruptions


# ---------------------------------------------------------------------- table rows


def _named(prefix: str, model_id: str) -> str:
    return f"{prefix}-{model_id}" if model_id else ""


def _event_row(event: Mapping[str, Any], request: StoriesRequest) -> dict[str, Any]:
    start, end = event["start"], event["end"]
    return {
        "schema_version": SCHEMA_VERSION,
        "video_id": request.video_id,
        "event_id": _named(request.id_prefix, str(event["model_id"])),
        "start_time": float(start["time"]),
        "end_time": float(end["time"]),
        "start_segment_id": start["segment_id"],
        "end_segment_id": end["segment_id"],
        "start_quote": start["quote"],
        "end_quote": end["quote"],
        "start_speaker": start["speaker"],
        "end_speaker": end["speaker"],
        "start_boundary_type": str(event["start_boundary_type"]),
        "end_boundary_type": str(event["end_boundary_type"]),
        "start_time_precision": start["precision"],
        "end_time_precision": end["precision"],
        "start_reason": start["reason"],
        "end_reason": end["reason"],
        "story_ids": json.dumps([_named(request.id_prefix, story_id)
                                 for story_id in event["story_ids"]], ensure_ascii=False),
        "tellers": json.dumps(list(event["tellers"]), ensure_ascii=False),
        "interruptions": json.dumps([dict(item) for item in event["interruptions"]],
                                    ensure_ascii=False),
        "confidence": event["confidence"],
        "notes": event["notes"],
        "window_index": request.window_index,
        "model": request.model,
        "prompt_version": request.prompt_version,
        "request_key": request.key(),
    }


def _story_row(story: Mapping[str, Any], request: StoriesRequest) -> dict[str, Any]:
    start, end = story["start"], story["end"]
    return {
        "schema_version": SCHEMA_VERSION,
        "video_id": request.video_id,
        "story_id": _named(request.id_prefix, str(story["model_id"])),
        "event_id": _named(request.id_prefix, str(story["event_id"])),
        "start_time": float(start["time"]),
        "end_time": float(end["time"]),
        "start_segment_id": start["segment_id"],
        "end_segment_id": end["segment_id"],
        "start_quote": start["quote"],
        "end_quote": end["quote"],
        "start_speaker": start["speaker"],
        "end_speaker": end["speaker"],
        "start_reason": start["reason"],
        "end_reason": end["reason"],
        "start_time_precision": start["precision"],
        "end_time_precision": end["precision"],
        "tellers": json.dumps(list(story["tellers"]), ensure_ascii=False),
        "summary": story["summary"],
        "story_evidence": json.dumps(list(story["story_evidence"]), ensure_ascii=False),
        "evidence_segment_ids": json.dumps(list(story["evidence_segment_ids"]),
                                           ensure_ascii=False),
        "confidence": story["confidence"],
        "notes": story["notes"],
        "window_index": request.window_index,
        "model": request.model,
        "prompt_version": request.prompt_version,
        "request_key": request.key(),
    }


def _uncertain_row(span: Mapping[str, Any], request: StoriesRequest) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "video_id": request.video_id,
        "start_segment_id": span["start_segment_id"],
        "end_segment_id": span["end_segment_id"],
        "start_time": float(span["start_time"]),
        "end_time": float(span["end_time"]),
        "transcript_quote": span["transcript_quote"],
        "issue": span["issue"],
        "window_index": request.window_index,
        "model": request.model,
        "prompt_version": request.prompt_version,
        "request_key": request.key(),
    }


@dataclass
class StoriesVerdict:
    """One window's answer: the raw text, the parsed payload, and the rows it produced."""

    content: str
    payload: dict[str, Any]
    answer: ValidatedAnswer

    @property
    def events(self) -> list[dict[str, Any]]:
        return self.answer.events

    @property
    def stories(self) -> list[dict[str, Any]]:
        return self.answer.stories

    @property
    def uncertain_spans(self) -> list[dict[str, Any]]:
        return self.answer.uncertain_spans

    @property
    def dropped(self) -> list[dict[str, Any]]:
        return self.answer.dropped


# ------------------------------------------------------------------ clients


class StoriesClient:
    """OpenAI-compatible chat client for story detection, with retry and usage.

    The httpx/retry/backoff/usage shape is the one ``TranslationClient`` established,
    kept deliberately close to it: same endpoint, same failure modes (429, 5xx, a timeout,
    a non-JSON body, a chatty answer), and an operator who reads one reads the other.
    """

    #: Where rejected answers are kept, one file per attempt. Set by the stage to
    #: ``stories/raw/rejected``; left None, nothing is written.
    rejected_dir: Path | None = None

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

    def detect(self, request: StoriesRequest) -> StoriesVerdict:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            started = time.time()
            raw_text: str | None = None
            try:
                payload = self._post(request)
                self.timings_ms.append(round((time.time() - started) * 1000, 1))
                self._record_usage(payload)
                raw_text = extract_message_content(payload)
                parsed = parse_stories_response(raw_text)
                answer = validate_response(parsed, request)
                return StoriesVerdict(content=raw_text, payload=parsed, answer=answer)
            except StoriesTransportError as exc:
                last_error = exc
            except (ValidationError, ValueError) as exc:
                # A malformed answer is the model's mistake, not the network's, but
                # resampling usually fixes it — up to a point. Every rejected answer is
                # kept: the accepted response alone would make the previous attempt
                # unauditable, and "it failed twice then passed" is exactly the history a
                # reader needs to judge a batch.
                last_error = exc
                self._record_rejection(request, attempt, raw_text, exc)
            if attempt < self.max_retries:
                delay = self.backoff_base * (2**attempt) + random.uniform(0, self.backoff_base)
                self._sleep(delay)
        raise StageError(
            f"stories window {request.window_index} failed after {self.max_retries + 1} "
            f"attempts: {last_error}",
            details={"window_index": request.window_index,
                     "requested_ids": request.requested_ids,
                     "attempts": self.max_retries + 1},
        )

    def _record_rejection(self, request: StoriesRequest, attempt: int, raw_text: str | None,
                          error: Exception) -> None:
        """Keep the text of a rejected answer, with the reason it was rejected.

        ``raw/rejected/window_<i>_<key>_attempt<n>.txt``, so a batch's failed attempts are
        as auditable as its accepted one. Only the response body is stored — never a
        header — and a transport failure that produced no text is not stored here: it is
        reported as a failure, not as a response.
        """
        if self.rejected_dir is None or raw_text is None:
            return
        path = (self.rejected_dir
                / f"window_{request.window_index}_{request.key()}_attempt{attempt + 1}.txt")
        reason = f"{type(error).__name__}: {error}"
        try:
            atomic_write_text(
                path,
                f"# rejected answer (attempt {attempt + 1} of {self.max_retries + 1}) for "
                f"window {request.window_index} of {request.video_id}\n"
                f"# request_key: {request.key()}\n"
                f"# reason: {reason}\n"
                f"\n{raw_text}",
            )
        except OSError:  # pragma: no cover - a full disk must not hide the real failure
            pass

    def _post(self, request: StoriesRequest) -> dict[str, Any]:
        # The body the cache key hashed and the body this posts are built by the same
        # method: the client's own model/extra_body are only fallbacks for a request built
        # without them, so the two cannot drift.
        body = request.body(model_fallback=self.model,
                            extra_body_fallback=self.extra_body)
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        self.request_count += 1
        try:
            response = httpx.post(f"{self.base_url}/chat/completions", json=body,
                                  headers=headers, timeout=self.timeout_seconds)
        except httpx.TimeoutException as exc:
            raise StoriesTransportError(f"request timed out after {self.timeout_seconds}s") from exc
        except httpx.HTTPError as exc:
            # str(httpx error) carries the URL and the failure, never the authorization
            # header, which is why the key is not reachable from this message.
            raise StoriesTransportError(f"transport error: {exc}") from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise StoriesTransportError(f"HTTP {response.status_code}: {response.text[:300]}")
        if response.status_code >= 400:
            raise StageError(f"stories endpoint rejected the request "
                             f"(HTTP {response.status_code}): {response.text[:400]}")
        try:
            return response.json()
        except ValueError as exc:
            raise StoriesTransportError(f"non-JSON response body: {response.text[:200]}") from exc

    def _record_usage(self, payload: dict[str, Any]) -> None:
        usage = payload.get("usage") or {}
        for key, value in usage.items():
            if isinstance(value, int):
                self.usage[key] = self.usage.get(key, 0) + value


class MockStoriesClient:
    """Deterministic offline stand-in, so no test or dry run needs the endpoint.

    It answers a *valid v2 object*: one storytelling event spanning the request's first and
    last segment, one story inside it, every quote taken from real segment text, and every
    speaker one of the labels the request actually showed. A single segment cannot hold an
    event and a story, so it answers with the three empty arrays — the answer the contract
    says is correct.
    """

    EVENT_NOTE = ("mock provider: one event spanning the requested segments, with its story "
                  "inside it")

    def __init__(self, model: str = "mock") -> None:
        self.model = model
        self.request_count = 0
        self.timings_ms: list[float] = []
        self.usage: dict[str, int] = {}

    def detect(self, request: StoriesRequest) -> StoriesVerdict:
        self.request_count += 1
        self.timings_ms.append(0.0)
        payload = self._payload(request)
        content = json.dumps(payload, ensure_ascii=False)
        answer = validate_response(payload, request)
        return StoriesVerdict(content=content, payload=payload, answer=answer)

    @classmethod
    def _payload(cls, request: StoriesRequest) -> dict[str, Any]:
        # Only segments with real text can be quoted: a blank segment's rendering is the
        # synthetic ``[unknown]`` marker, which is not in the transcript and must never be
        # answered as if it were a quote.
        rows = [row for row in request.requested if str(row.get("text") or "").strip()]
        quotes = [cls._quote(rows[0], first=True), cls._quote(rows[-1], first=False),
                  cls._quote(rows[0], first=False), cls._quote(rows[-1], first=False)] \
            if len(rows) >= 2 else []
        if len(rows) < 2 or not all(quotes):
            return {"storytelling_events": [], "stories": [], "uncertain_spans": []}
        first, last = rows[0], rows[-1]
        event_start, event_end, story_start, story_end = quotes
        evidence_segments = [str(row["segment_id"]) for row in rows]
        teller = cls._speaker(first)
        payload = {
            "storytelling_events": [{
                "event_id": "SE1",
                "start": {"segment_id": str(first["segment_id"]),
                          "transcript_quote": event_start, "speaker": cls._speaker(first),
                          "boundary_type": "pre_telling",
                          "reason": "mock provider: the request opens with this line."},
                "end": {"segment_id": str(last["segment_id"]),
                        "transcript_quote": event_end, "speaker": cls._speaker(last),
                        "boundary_type": "coda",
                        "reason": "mock provider: the request closes on this line."},
                "story_ids": ["S1"],
                "tellers": [teller],
                "interruptions_or_resumptions": [],
                "confidence": "medium",
                "notes": cls.EVENT_NOTE,
            }],
            "stories": [{
                "story_id": "S1",
                "event_id": "SE1",
                "start": {"segment_id": str(first["segment_id"]),
                          "transcript_quote": story_start, "speaker": cls._speaker(first),
                          "reason": "mock provider: the telling starts here."},
                "end": {"segment_id": str(last["segment_id"]),
                        "transcript_quote": story_end, "speaker": cls._speaker(last),
                        "reason": "mock provider: the telling ends here."},
                "tellers": [teller],
                "summary": "Mock story: the requested segments read as one telling.",
                "story_evidence": [
                    "mock provider: the requested segments represent one event",
                    "mock provider: the event is told from experience, in order",
                ],
                "evidence_segment_ids": evidence_segments,
                "confidence": "medium",
                "notes": "",
            }],
            "uncertain_spans": [],
        }
        return payload

    @staticmethod
    def _words(text: str) -> list[str]:
        return [word for word in re.split(r"\s+", text.strip()) if word]

    @classmethod
    def _quote(cls, row: Mapping[str, Any], *, first: bool) -> str:
        """A quote from this segment's real text that occurs exactly once in it.

        An empty string means "no quotable text": the caller answers emptily rather than
        quoting the synthetic ``[unknown]`` marker or a blank, either of which would be a
        fabrication dressed as evidence.
        """
        text = str(row.get("text") or "").strip()
        words = cls._words(text)
        if not words:
            return ""
        candidate = " ".join(words[:4] if first else words[-4:])
        if len(quote_occurrences(candidate, text)) == 1:
            return candidate
        # A short window repeated inside the segment ("no no no no no") says nothing about
        # where the boundary is; the whole segment is the longest quote available.
        return text if len(quote_occurrences(text, text)) == 1 else ""

    @staticmethod
    def _speaker(row: Mapping[str, Any]) -> str:
        labels = [str(label) for label in row.get("speaker_labels") or []]
        labels = [label for label in labels if label != UNKNOWN_SPEAKER]
        return labels[0] if labels else UNKNOWN_SPEAKER


def _drop_counts(dropped: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Counts a window's drops, with an event counted once and its stories apart.

    ``{"events": 1, "stories": 2, "uncertain_spans": 0}`` reads as "one storytelling event
    that straddled this window was left to the neighbouring one, and it carried two
    stories". Summing the dropped items instead would report three and say nothing about
    what was lost.
    """
    counts = {"events": 0, "stories": 0, "uncertain_spans": 0}
    for item in dropped:
        kind = str(item.get("kind") or "")
        if kind == "event":
            counts["events"] += 1
            counts["stories"] += len(item.get("child_story_ids") or [])
        elif kind == "story":
            counts["stories"] += 1
        elif kind == "uncertain_span":
            counts["uncertain_spans"] += 1
    return counts


# ------------------------------------------------------------------ stage


class StoriesStage(Stage):
    """Ask the translation endpoint where the storytelling events and stories are."""

    name = "stories"
    # Hard input: the transcript with speakers applied. The words and the diarizer turns
    # are read *softly* (see `_read_soft`) because they refine the prompt rather than being
    # the thing answered about: a corpus with no diarizer still has stories, with every
    # label "unknown".
    inputs = ("speech_segments",)
    outputs = ("storytelling_events", "stories", "uncertain_spans", "stories_raw")
    config_keys = ("stories",)

    # ------------------------------------------------------------------ fingerprint

    def config_fingerprint(self, ctx: StageContext) -> dict[str, Any]:
        """Config + the digests of everything this answer is computed from.

        ``segments_digest`` is what makes a re-transcription invalidate the answers, and
        ``words_digest``/``speaker_turns_digest`` do the same for the two files the
        boundary times and the speaker labels come from — a re-alignment or a re-diarization
        changes the prompt and the computed times without changing one word of
        ``segments.parquet``. ``_python_code_sha256`` covers this module *and*
        ``story_speakers``: both are pure python in this process, so a fix to the
        attribution rule or the validator would otherwise leave every cached window looking
        reusable.
        """
        from .. import story_speakers
        from . import stories as stories_stage
        from .base import python_source_digest

        cfg = ctx.config.stories
        return {
            "stage": self.name,
            "provider": cfg.provider,
            "model": cfg.model,
            "base_url": cfg.base_url,
            "temperature": cfg.temperature,
            "prompt_version": cfg.prompt_version,
            "max_output_tokens": cfg.max_output_tokens,
            "max_retries": cfg.max_retries,
            "max_segments_per_request": cfg.max_segments_per_request,
            "context_segments": cfg.context_segments,
            "speaker_source": cfg.speaker_source,
            "speaker_min_coverage": cfg.speaker_min_coverage,
            "speaker_max_competing": cfg.speaker_max_competing,
            "extra_body": cfg.extra_body,
            "segments_digest": self._digest(ctx, "speech_segments"),
            "words_digest": self._digest(ctx, "speech_words"),
            "speaker_turns_digest": self._digest(ctx, "speaker_turns_nemotron"),
            "_python_code_sha256": python_source_digest(stories_stage, story_speakers),
        }

    @staticmethod
    def _digest(ctx: StageContext, artifact: str) -> str | None:
        from ..stages.metadata import sha256_of

        path = ctx.artifact(artifact)
        if not path.is_file():
            return None
        return sha256_of(path)

    # ------------------------------------------------------------------ enablement

    def enabled(self, ctx: StageContext) -> tuple[bool, str]:
        """An unconfigured endpoint is a skip with the keys to set, never a crash."""
        cfg = ctx.config.stories
        if not cfg.enabled:
            return False, "stories.enabled = false"
        if not cfg.endpoint_configured:
            return False, ("stories endpoint is not configured (set stories.base_url, "
                           "stories.api_key and stories.model in the config file)")
        segments = self._segment_rows(ctx)
        if not segments:
            return False, ("no speech segments to read stories from "
                           "(speech/segments.parquet is empty or missing): run whisperx "
                           "and speaker_assignment first")
        return True, ""

    def prepare(self, ctx: StageContext) -> None:
        ctx.input("speech_segments")
        ctx.artifact("stories_raw").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ inputs

    @staticmethod
    def _segment_rows(ctx: StageContext) -> list[dict[str, Any]]:
        path = ctx.artifact("speech_segments")
        if not path.is_file():
            return []
        return read_table(path).to_pylist()

    @classmethod
    def _word_rows(cls, ctx: StageContext) -> list[dict[str, Any]]:
        return cls._read_soft(ctx, "speech_words",
                              required={"segment_id", "word", "start_time", "end_time"})

    @staticmethod
    def _read_soft(ctx: StageContext, artifact: str,
                   required: set[str] | None = None) -> list[dict[str, Any]]:
        """A table read off disk, if it exists and carries the columns needed.

        Read softly rather than declared as a hard input, the way ``SpacySourceStage``
        reads ``whisperx_raw``: ``ctx.input()`` raises on a missing artifact, and "this
        corpus ran no diarizer" is a supported state for a stage whose subject is the
        words. A table that exists but lacks a needed column is treated the same way — its
        rows could not be used, and crashing on it would turn an old corpus into a failure.
        """
        from ..schemas import table_columns

        path = ctx.artifact(artifact)
        if not path.is_file():
            return []
        if required:
            columns = set(table_columns(path))
            if not required <= columns:
                return []
        return read_table(path).to_pylist()

    def _speaker_inputs(self, ctx: StageContext) -> tuple[list[dict[str, Any]], str]:
        """The turns to attribute against, and the source actually used.

        A missing selected source is supported and recorded, never silently replaced: a
        Nemotron-configured corpus with no Nemotron table labels everything "unknown"
        instead of quietly switching to pyannote, because that swap would relabel a
        transcript under a name the operator did not ask for and would be invisible in the
        table.
        """
        cfg = ctx.config.stories
        if cfg.speaker_source == "none":
            return [], "none"
        if cfg.speaker_source == "pyannote":
            # The labels are already on the words/segments; no turn table is needed. But a
            # corpus whose word rows and segments both carry no `speaker_id` has no
            # pyannote labels to use, and that is the same state as a missing source: say
            # so, label everything unknown, and record `used: none` rather than presenting
            # an all-unknown transcript as a successful diarization.
            labeled_words = any(str(word.get("speaker_id") or "").strip()
                                for word in self._word_rows(ctx))
            labeled_segments = any(str(row.get("speaker_id") or "").strip()
                                   for row in self._segment_rows(ctx))
            if not labeled_words and not labeled_segments:
                ctx.log("stories: speaker_source=pyannote found no speaker_id on any word "
                        "or segment; every speaker label will be 'unknown'",
                        30)
                return [], "none"
            return [], "pyannote"
        turns = self._read_soft(ctx, "speaker_turns_nemotron",
                                required={"speaker_id", "start_time", "end_time"})
        if not turns:
            ctx.log(f"stories: speaker_source=nemotron has no usable "
                    f"speech/speaker_turns_nemotron.parquet; every speaker label will be "
                    f"'unknown' (no fallback to pyannote)", 30)
            return [], "none"
        return turns, "nemotron"

    # ------------------------------------------------------------------ execution

    def build_requests(self, ctx: StageContext) -> list[StoriesRequest]:
        cfg = ctx.config.stories
        turns, used = self._speaker_inputs(ctx)
        prepared = prepare_segments(self._segment_rows(ctx), self._word_rows(ctx), turns,
                                    speaker_source=used,
                                    speaker_min_coverage=cfg.speaker_min_coverage,
                                    speaker_max_competing=cfg.speaker_max_competing)
        ctx.scratch["stories_speaker_source_used"] = used
        return build_requests(
            prepared,
            video_id=ctx.video_id,
            max_segments_per_request=cfg.max_segments_per_request,
            context_segments=cfg.context_segments,
            prompt_version=cfg.prompt_version,
            model=cfg.model,
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
            speaker_source=used,
            speaker_min_coverage=cfg.speaker_min_coverage,
            speaker_max_competing=cfg.speaker_max_competing,
            endpoint=cfg.base_url,
            extra_body=cfg.extra_body,
        )

    def execute(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.config.stories
        requests = self.build_requests(ctx)
        used_source = str(ctx.scratch.get("stories_speaker_source_used") or "none")
        client = self._client(ctx)
        raw_dir = ctx.artifact("stories_raw")
        cache_dir = raw_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        rejected_dir = raw_dir / "rejected"
        rejected_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(client, "rejected_dir"):
            # Every rejected answer is kept beside the accepted one, so a batch's failed
            # attempts are auditable rather than inferred from a retry count.
            client.rejected_dir = rejected_dir

        events: list[dict[str, Any]] = []
        stories: list[dict[str, Any]] = []
        uncertain: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []
        drop_counts = {"events": 0, "stories": 0, "uncertain_spans": 0}
        reused = 0
        empty_windows = 0
        for request in requests:
            index = request.window_index
            verdict = self._load_cached(cache_dir / f"{request.key()}.json", request) \
                if cfg.cache else None
            if verdict is not None:
                reused += 1
                ctx.log(f"window {index + 1}/{len(requests)} reused from cache")
                # A reused window still owes the dataset its raw file: the cache may have
                # survived a hand-deleted stories/raw/window_*.json.
                self._ensure_raw(raw_dir, request, verdict)
            else:
                ctx.log(f"detecting storytelling events in window {index + 1}/"
                        f"{len(requests)} ({len(request.requested_ids)} segments)")
                verdict = client.detect(request)
                # Raw first, byte-identical: a normalised table with no original next to it
                # cannot be re-validated later (repo rule), and this file is also what a
                # human reads to see what the model actually said. Written unconditionally
                # rather than only when absent: at temperature > 0 the endpoint can answer
                # the same request differently, so keeping a previous response here would
                # file the wrong evidence beside these rows.
                atomic_write_text(raw_dir / f"window_{index}_{request.key()}.json",
                                  verdict.content)
                atomic_write_json(cache_dir / f"{request.key()}.json", {
                    "schema_version": SCHEMA_VERSION,
                    "request_key": request.key(),
                    "video_id": request.video_id,
                    "window_index": index,
                    "requested_ids": request.requested_ids,
                    "context_ids": [str(row["segment_id"]) for row in request.context],
                    "model": request.model,
                    "temperature": request.temperature,
                    "prompt_version": request.prompt_version,
                    "speaker_source": request.speaker_source,
                    # The model's own payload, unprefixed: reuse re-validates it against
                    # the current window rather than trusting what is on disk.
                    "payload": verdict.payload,
                    # And its exact response text, so a reuse can restore a deleted raw
                    # file byte-identically instead of writing a re-serialisation that
                    # only looks like what the endpoint sent.
                    "content": verdict.content,
                })
            events.extend(verdict.events)
            stories.extend(verdict.stories)
            uncertain.extend(verdict.uncertain_spans)
            if verdict.answer.empty:
                empty_windows += 1
                ctx.log(f"window {index + 1}/{len(requests)}: no storytelling event, "
                        f"story or uncertain span")
            for item in verdict.dropped:
                children = item.get("child_story_ids") or []
                ctx.log(f"window {index + 1}: dropped {item['kind']} "
                        f"{item['model_id']}"
                        + (f" with {len(children)} story/stories" if children else "")
                        + f" — {item['reason']}", 30)
            dropped.extend({"window_index": index, **item} for item in verdict.dropped)
            for kind, count in _drop_counts(verdict.dropped).items():
                drop_counts[kind] += count

        metadata = {"video_id": ctx.video_id, "model": cfg.model,
                    "prompt_version": cfg.prompt_version, "windows": len(requests),
                    "speaker_source": used_source}
        write_table(ctx.artifact("storytelling_events"),
                    pa.Table.from_pylist(events, schema=STORYTELLING_EVENTS_SCHEMA),
                    STORYTELLING_EVENTS_SCHEMA, extra_metadata=metadata)
        write_table(ctx.artifact("stories"),
                    pa.Table.from_pylist(stories, schema=STORIES_SCHEMA), STORIES_SCHEMA,
                    extra_metadata=metadata)
        write_table(ctx.artifact("uncertain_spans"),
                    pa.Table.from_pylist(uncertain, schema=UNCERTAIN_SPANS_SCHEMA),
                    UNCERTAIN_SPANS_SCHEMA, extra_metadata=metadata)
        atomic_write_json(raw_dir / "stories_summary.json", {
            "schema_version": SCHEMA_VERSION,
            "video_id": ctx.video_id,
            "provider": cfg.provider,
            "model": cfg.model,
            "base_url": cfg.base_url,
            "temperature": cfg.temperature,
            "prompt_version": cfg.prompt_version,
            "max_segments_per_request": cfg.max_segments_per_request,
            "context_segments": cfg.context_segments,
            # Both halves of the speaker decision: what was asked for and what the files on
            # disk could actually support. A reader can tell a corpus that never ran
            # Nemotron from one that did and produced no turns.
            "speaker_source_requested": cfg.speaker_source,
            "speaker_source_used": used_source,
            "speaker_min_coverage": cfg.speaker_min_coverage,
            "speaker_max_competing": cfg.speaker_max_competing,
            "windows": len(requests),
            "empty_windows": empty_windows,
            "batches_reused": reused,
            "requests_made": getattr(client, "request_count", 0),
            "request_timings_ms": getattr(client, "timings_ms", [])[-200:],
            "token_usage": getattr(client, "usage", {}),
            "storytelling_events": len(events),
            "stories": len(stories),
            "uncertain_spans": len(uncertain),
            # An event is counted once, with the stories it took with it counted apart:
            # one cross-window event is one gap in the answer, however many stories it
            # held, and a reader who needs the story count has it.
            "dropped_outside_window": drop_counts["events"],
            "dropped_stories_outside_window": drop_counts["stories"],
            "dropped_uncertain_spans_outside_window": drop_counts["uncertain_spans"],
            "dropped_detail": dropped[:50],
        })
        ctx.scratch["stories"] = {"storytelling_events": len(events),
                                  "stories": len(stories),
                                  "uncertain_spans": len(uncertain),
                                  "windows": len(requests),
                                  "empty_windows": empty_windows,
                                  "dropped_outside_window": drop_counts["events"],
                                  "speaker_source_used": used_source,
                                  "model": cfg.model}
        ctx.log(f"stories: {len(events)} storytelling event(s), {len(stories)} story/stories, "
                f"{len(uncertain)} uncertain span(s) across {len(requests)} window(s); "
                f"{empty_windows} empty, {drop_counts['events']} event(s) dropped outside "
                f"a window (with {drop_counts['stories']} story/stories), "
                f"{reused}/{len(requests)} reused; speaker_source "
                f"{cfg.speaker_source} -> {used_source}")
        return {"tool_version": cfg.provider, "model_version": cfg.model,
                "extra": {"events_total": len(events), "stories_total": len(stories),
                          "uncertain_spans_total": len(uncertain),
                          "windows": len(requests),
                          "empty_window_count": empty_windows,
                          "dropped_outside_window": drop_counts["events"],
                          "dropped_stories_outside_window": drop_counts["stories"],
                          "dropped_uncertain_spans_outside_window":
                              drop_counts["uncertain_spans"],
                          "batches_reused": reused,
                          "speaker_source_requested": cfg.speaker_source,
                          "speaker_source_used": used_source}}

    def _client(self, ctx: StageContext) -> Any:
        cfg = ctx.config.stories
        if cfg.provider == "mock":
            return MockStoriesClient(model=cfg.model or "mock")
        return StoriesClient(base_url=cfg.base_url, api_key=cfg.api_key, model=cfg.model,
                             temperature=cfg.temperature, timeout_seconds=cfg.timeout_seconds,
                             max_retries=cfg.max_retries,
                             backoff_base_seconds=cfg.backoff_base_seconds,
                             extra_body=cfg.extra_body)

    @staticmethod
    def _ensure_raw(raw_dir: Path, request: StoriesRequest, verdict: StoriesVerdict) -> None:
        """Guarantee the raw window file *is* the answer the tables will carry.

        Three states, and the middle one is a defect this used to hide: absent (write it),
        present and parsing to the same payload (leave the bytes alone — they are the
        original), present and parsing to something else (rewrite from the cached content,
        because a raw file that disagrees with its rows files the wrong evidence beside
        them). A verdict with no content cannot be written at all and is not turned into a
        re-serialisation that only looks like what the endpoint sent.
        """
        path = raw_dir / f"window_{request.window_index}_{request.key()}.json"
        if not verdict.content:
            return
        if path.is_file():
            try:
                if parse_stories_response(path.read_text(encoding="utf-8")) == verdict.payload:
                    return
            except (OSError, ValueError):
                pass
        atomic_write_text(path, verdict.content)

    @staticmethod
    def _load_cached(path: Path, request: StoriesRequest) -> StoriesVerdict | None:
        """A cached window, re-validated against *this* window's segments.

        The re-validation is the point. A cache file whose segment ids or quotes no longer
        fit the transcript is a file about a clip that no longer exists, and handing its
        rows straight to the tables would publish them with no check at all. An entry
        without its exact response text is a **miss**, not a reusable answer: the original
        bytes are the evidence this dataset keeps, and a re-serialisation of the payload
        would be filed as if the endpoint had sent it.
        """
        if not path.is_file():
            return None
        try:
            payload = read_json(path)
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        if list(payload.get("requested_ids") or []) != request.requested_ids:
            return None
        stored = payload.get("payload")
        if not isinstance(stored, dict):
            return None
        content = payload.get("content")
        if not isinstance(content, str) or not content.strip():
            # No original bytes: refetch rather than invent them.
            return None
        try:
            if parse_stories_response(content) != stored:
                return None
        except (ValueError, ValidationError):
            return None
        try:
            answer = validate_response(stored, request)
        except (ValidationError, ValueError):
            return None
        return StoriesVerdict(content=content, payload=stored, answer=answer)

    # ------------------------------------------------------------------ validation

    def validate(self, ctx: StageContext) -> dict[str, Any]:
        """The three tables parse, agree with each other, and sit on real transcript rows.

        Recomputed from disk, not from memory: the row counts, the event/story link in both
        directions, every story inside its event's span, and every boundary's time inside
        the segment it cites. An empty table is valid — zero rows is a real answer — so the
        checks are about consistency, not about having found something.
        """
        problems: list[str] = []
        tables: dict[str, list[dict[str, Any]]] = {}
        schemas = {"storytelling_events": STORYTELLING_EVENTS_SCHEMA,
                   "stories": STORIES_SCHEMA,
                   "uncertain_spans": UNCERTAIN_SPANS_SCHEMA}
        for name, schema in schemas.items():
            path = ctx.artifact(name)
            if not path.is_file():
                problems.append(f"stories/{name}.parquet missing")
                continue
            # Columns first: a zero-row table reads back as an empty list whatever it was
            # written with, so "no rows" and "the wrong table" are indistinguishable unless
            # the schema is checked on its own. An empty table with the wrong columns is a
            # defect, not an empty answer.
            actual = list(table_columns(path))
            expected = [field.name for field in schema]
            if actual != expected:
                problems.append(f"stories/{name}.parquet columns do not match its schema: "
                                f"missing {[c for c in expected if c not in actual][:5]}, "
                                f"unexpected {[c for c in actual if c not in expected][:5]}, "
                                f"or in another order")
                continue
            tables[name] = read_table(path).to_pylist()
        if problems:
            raise ValidationError(self.name, problems)

        summary_path = ctx.artifact("stories_raw") / "stories_summary.json"
        if not summary_path.is_file():
            raise ValidationError(self.name,
                                  ["stories/raw/stories_summary.json missing: the tables have "
                                   "no record of how many windows were asked about"])
        summary = read_json(summary_path)

        events = tables["storytelling_events"]
        stories = tables["stories"]
        uncertain = tables["uncertain_spans"]

        event_ids = [str(row["event_id"]) for row in events]
        duplicates = sorted({item for item in event_ids if event_ids.count(item) > 1})
        if duplicates:
            problems.append(f"duplicate event_id values: {duplicates[:5]}")
        story_ids = [str(row["story_id"]) for row in stories]
        duplicates = sorted({item for item in story_ids if story_ids.count(item) > 1})
        if duplicates:
            problems.append(f"duplicate story_id values: {duplicates[:5]}")

        known_events = set(event_ids)
        by_event: dict[str, set[str]] = {}
        for row in stories:
            event_id = str(row["event_id"])
            if event_id not in known_events:
                problems.append(f"story {row['story_id']} names event {event_id}, which is "
                                f"not a storytelling event of this video")
            by_event.setdefault(event_id, set()).add(str(row["story_id"]))
        for row in events:
            listed = set(json.loads(row["story_ids"] or "[]"))
            answered = by_event.get(str(row["event_id"]), set())
            if listed != answered:
                problems.append(f"event {row['event_id']} lists stories {sorted(listed)} but "
                                f"{sorted(answered)} name it as their event")
            if float(row["end_time"]) < float(row["start_time"]):
                problems.append(f"event {row['event_id']} ends before it starts")

        event_by_id = {str(row["event_id"]): row for row in events}
        for row in stories:
            event = event_by_id.get(str(row["event_id"]))
            if event is None:
                continue
            if (float(row["start_time"])
                    < float(event["start_time"]) - BOUNDARY_TOLERANCE):
                problems.append(f"story {row['story_id']} starts before its event "
                                f"{row['event_id']} starts")
            if (float(row["end_time"])
                    > float(event["end_time"]) + BOUNDARY_TOLERANCE):
                problems.append(f"story {row['story_id']} ends after its event "
                                f"{row['event_id']} ends")
            if float(row["end_time"]) < float(row["start_time"]):
                problems.append(f"story {row['story_id']} ends before it starts")

        spans = self._segment_spans(ctx)
        if spans is None:
            problems.append("speech/segments.parquet is unreadable or missing, so no "
                            "boundary in these tables can be checked")
        else:
            for row in events:
                problems.extend(self._boundary_problems(spans, row["event_id"], row))
            for row in stories:
                problems.extend(self._boundary_problems(spans, row["story_id"], row))
            for row in uncertain:
                label = f"uncertain span {row['start_segment_id']}..{row['end_segment_id']}"
                problems.extend(self._inside(spans=spans, label=label, row=row,
                                             start="start_time", end="end_time",
                                             require=("start_segment_id", "end_segment_id")))

        for key, rows in (("storytelling_events", events), ("stories", stories),
                          ("uncertain_spans", uncertain)):
            if int(summary.get(key) or 0) != len(rows):
                problems.append(f"stories_summary.json says {summary.get(key)} {key}, the "
                                f"table has {len(rows)}")
        # Every row names the response it came from, so that response has to be there: a
        # table whose provenance points at a missing file cannot be audited, and the raw
        # window is the only place the model's own words survive.
        raw_dir = ctx.artifact("stories_raw")
        for name, rows in (("storytelling_events", events), ("stories", stories),
                           ("uncertain_spans", uncertain)):
            for row in rows:
                referenced = (raw_dir / f"window_{row['window_index']}_"
                                        f"{row['request_key']}.json")
                if not referenced.is_file():
                    problems.append(f"{name} row {row.get('event_id') or row.get('story_id') or row.get('start_segment_id')}"
                                    f" names request {row['request_key']} but "
                                    f"stories/raw/{referenced.name} is missing")
                    break
        windows = int(summary.get("windows") or 0)
        named = {int(row["window_index"]) for row in events + stories + uncertain}
        if windows and windows < len(named):
            problems.append("the tables name more windows than the raw summary recorded")
        if problems:
            raise ValidationError(self.name, problems[:20])
        return {"storytelling_events": len(events), "stories": len(stories),
                "uncertain_spans": len(uncertain), "windows": windows,
                "empty_windows": max(windows - len(named), 0)}

    def _boundary_problems(self, spans: dict[str, tuple[float, float]],
                           label: str, row: Mapping[str, Any]) -> list[str]:
        return self._inside(spans=spans, label=label, row=row, start="start_time",
                            end="end_time",
                            require=("start_segment_id", "end_segment_id"))

    @staticmethod
    def _inside(*, spans: dict[str, tuple[float, float]], label: str,
                row: Mapping[str, Any], start: str, end: str,
                require: tuple[str, ...]) -> list[str]:
        """Both times must sit inside the segment the row says they belong to.

        This is the checkable half of "the model did not invent a time": the time came from
        a word of that segment, so it cannot fall outside the segment's own span. A row
        whose ``*_segment_id`` names a segment this transcript no longer has is reported
        too — it is the same defect a re-transcription causes.
        """
        problems: list[str] = []
        for role, time_column in (("start", start), ("end", end)):
            segment_id = str(row.get(require[0] if role == "start" else require[1]) or "")
            bounds = spans.get(segment_id)
            if bounds is None:
                problems.append(f"{label}: {role} segment {segment_id!r} is not a segment "
                                f"of this transcript")
                continue
            low, high = bounds
            value = float(row[time_column])
            if value < low - BOUNDARY_TOLERANCE or value > high + BOUNDARY_TOLERANCE:
                problems.append(f"{label}: {time_column} {value} lies outside its {role} "
                                f"segment {segment_id} ({low}..{high})")
        return problems

    @staticmethod
    def _segment_spans(ctx: StageContext) -> dict[str, tuple[float, float]] | None:
        """Per segment id, the span a boundary's computed time must lie inside."""
        path = ctx.artifact("speech_segments")
        if not path.is_file():
            return None
        try:
            rows = read_table(path, columns=["segment_id", "start_time",
                                             "end_time"]).to_pylist()
        except Exception:  # noqa: BLE001 - reported as "cannot check", not as a crash
            return None
        return {str(row["segment_id"]): (float(row["start_time"]), float(row["end_time"]))
                for row in rows}
