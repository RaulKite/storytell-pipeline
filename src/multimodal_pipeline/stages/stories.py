"""`stories`: storytelling events, the stories told inside them, and uncertain spans.

Prompt **v2** implements the domain expert's definition: a *storytelling event* is the
conversational activity of launching and telling (pre-telling, telling, coda), and a *story*
is the represented "what" inside it. They are marked separately because they are different
claims — the pre-telling belongs to the event and not necessarily to the story, a coda can
close the event while the story ended earlier, and one event can carry several stories. v1
asked for "an anecdote with a beginning and an end" in a single span with ``parent_id``
nesting; that model is gone from the code (it stays in Git history) and asking for
``stories.prompt_version: v1`` now fails at config-parse time. v2 is **her prompt**, not a
description of it: her goal, speaker-attribution limitation, working criteria, boundary
guidance, multiple-tellers rule and annotation rules are quoted word for word from
``docs/stories-annotation-guidelines.md``, which holds her original text, and
``tests/unit/test_stories.py`` fails when the two drift apart. Two sections are adapted
because they describe *her* input and output — her WhisperX JSON becomes our INPUT section,
and her numeric boundary times become a ``segment_id`` + verbatim ``transcript_quote`` the
stage resolves itself — and every pipeline-specific rule lives in its own clearly separated
section rather than inside one of her sentences.

Four properties carry the design:

* **Emptiness is a result, not a failure.** Empty arrays are the expected and correct
  answer when the transcript supports nothing, and the count is recorded per window. A
  model that is not allowed to answer "nothing" stretches a greeting into a story.
* **No time is invented anywhere.** The model cites a ``segment_id`` and copies a verbatim
  ``transcript_quote``; this module locates that quote inside the cited segment's text and
  maps its **edge words** to word times through ``speech/words.parquet``. A quote that is
  not there is a validation breach. A quote that occurs more than once is not a location
  either, so the answer must say which occurrence it means (``occurrence``, 1 = first):
  requiring the quote to be unique instead made a correct boundary inexpressible wherever
  the segment repeats itself, and no retry could converge on one. The quote has to cover
  whole words (a one-letter quote cannot match inside ``Ahora``), the alignment accepts the
  split and merged tokens a different tokenizer produces (``can`` + ``'t`` for ``can't``),
  and an edge word with no exact alignment falls back to the segment's own boundary,
  labelled ``segment``, rather than borrowing a neighbour's time.
* **Speaker labels are attributed, not guessed.** Labels come from automatic diarization —
  Nemotron by default, whose turns *overlap* — so a word is labelled only when one speaker
  dominates it, and ``unknown`` otherwise (``story_speakers``).
* **Windowing is honest about what it clips.** The default is the whole transcript in one
  request. When a window size is configured, an event whose boundary (or whose marked
  interruption) falls outside the window is dropped *and counted* as one gap, with the
  number of stories it took with it counted apart, never quietly kept — and never rejected
  because a neighbouring window's stretch was visible as context. A segment the prompt
  never showed at all is a different thing: a quote out of it cannot have been copied from
  what the model read, so that is a breach and the answer is retried.

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
from .translation import TranslationTransportError, extract_message_content

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

#: A segment id as the answer types it; an "explanation" made only of these explains
#: nothing about why the stretch is a story.
_SEGMENT_ID_RE = re.compile(r"(?i)^seg\d+$")

#: How many (token, word) cells one alignment may fill. The grid is bounded so a segment
#: whose transcript and alignment have nothing in common cannot spend unbounded time on a
#: correspondence that does not exist; above it, equal keys are placed positionally.
_ALIGNMENT_CELL_LIMIT = 400_000

PROMPTS: dict[str, str] = {
    # Prompt v2 is the domain expert's annotation prompt, kept word for word: goal,
    # speaker-attribution limitation, working criteria, boundary guidance, multiple
    # tellers/stories/interruptions and annotation rules are hers as she wrote them, and
    # `docs/stories-annotation-guidelines.md` holds her original text so an edit to either
    # side is caught instead of drifting. Two sections are adapted because they describe
    # *her* input and *her* output, not this pipeline's: the INPUT section describes our
    # transcript lines (not a WhisperX JSON object), and the OUTPUT contract names
    # `segment_id` where she had a numeric time (the pipeline computes times; a model that
    # types seconds is typing the one thing nothing can check), with `story_evidence`
    # keeping her three-part explanation and `evidence_segment_ids` added beside it for the
    # segments themselves. Exactly one sentence of hers is rewritten, her annotation rule
    # on timestamps, into the rule that the model outputs no times at all; her intent (do
    # not invent precision) is the reason for it. Every pipeline-specific rule — cite a
    # segment id, quote from `text:` and only once in that segment, name only a shown
    # speaker label, labels are automatic and may be wrong, an empty answer is correct —
    # lives in its own "Pipeline rules" section rather than inside her sentences.
    "v2": (
        "You are annotating a conversation transcript. The guidelines come first, quoted as the domain expert wrote them; the sections marked as the pipeline's say how this input and this answer differ from hers." "\n"
        "\n"
        "Your goal is to mark the boundaries of each storytelling event and, separately, the boundaries of each story told within it. A story is the “what” being represented; a storytelling event is the conversational activity of launching and telling it." "\n"
        "\n"
        "## Important limitation: speaker attribution" "\n"
        "\n"
        "This transcript may not include speaker labels. Do not infer speaker identities from the text alone." "\n"
        "\n"
        "- If the transcript includes speaker labels, use them." "\n"
        "- If it has no speaker labels, use \"unknown\" for the speaker unless the transcript itself clearly identifies who is speaking." "\n"
        "- Do not invent labels such as “Speaker 1” or “Speaker 2” based on assumptions about turn-taking." "\n"
        "- You may describe a voice as a distinct speaker only if the input contains reliable speaker or diarization information." "\n"
        "- If attribution is unclear, mark the speaker as \"unknown\" and explain the uncertainty." "\n"
        "\n"
        "## Working criteria" "\n"
        "\n"
        "Treat a stretch as a possible story when it does one or both of the following:" "\n"
        "\n"
        "1. Represents one or more events, including events linked causally or parts of a single event. The events need not be told in chronological order, and they need not involve the people currently speaking." "\n"
        "2. Foregrounds experience, such as a person’s embodied or intentional experience, or an evaluation of an experience. For example, someone may recount their reaction to a book without narrating a sequence of external events." "\n"
        "\n"
        "A report, opinion, explanation, or description is not automatically a story. For ambiguous cases, explain the evidence for treating the stretch as a story." "\n"
        "\n"
        "## Boundary guidance" "\n"
        "\n"
        "### Start of the storytelling event" "\n"
        "Look for possible pre-telling before the story itself, including:" "\n"
        "- A gist or “bottom line” that previews the story." "\n"
        "- A question or reference that initiates a story sequence, such as asking whether the story has been told before." "\n"
        "- An explicit preface, such as “Let me tell you something.”" "\n"
        "\n"
        "If present, include the pre-telling in the storytelling-event span. Do not automatically treat it as part of the story itself. Pre-telling may be absent or difficult to identify; do not assume it must occur." "\n"
        "\n"
        "### Start of the story" "\n"
        "Mark where the speaker enters the story world, rather than where they merely announce or preview the telling. Look for references to a time, place, person, or combination that shift from the conversational here-and-now into the represented experience. These are clues, not mandatory formulas." "\n"
        "\n"
        "### End of the story" "\n"
        "Mark the last utterance that belongs to the represented events or experience. Keep listener assessments or comments outside the story span unless they add to the represented experience." "\n"
        "\n"
        "### End of the storytelling event" "\n"
        "Look for evidence that the telling has closed, such as:" "\n"
        "- A coda by the storyteller (“That’s it,” “What a night,” “That was embarrassing”)." "\n"
        "- A listener’s assessment or comment that treats the story as complete." "\n"
        "- A clear shift back to another conversational activity." "\n"
        "\n"
        "A coda may belong to the storytelling event while falling outside the story itself. Listener responses that acknowledge the story is over belong outside the storytelling event. If the transcript does not make the ending clear, mark it as uncertain." "\n"
        "\n"
        "## Multiple tellers, stories, and interruptions" "\n"
        "\n"
        "- A storytelling event can involve more than one teller. Track contributions by each speaker only when the input provides speaker labels or reliable attribution." "\n"
        "- One storytelling event may contain more than one story, including a second story. Mark stories separately when the transcript supports that distinction." "\n"
        "- A story may be interrupted by unrelated talk and then resumed. If a later segment clearly continues or retrieves the earlier story, treat it as part of the same story and storytelling event, and mark the interruption. Do not join stretches based only on topic similarity." "\n"
        "- If a new story begins within an ongoing storytelling event, mark it as a second story within that event when appropriate." "\n"
        "- Do not force every conversational stretch into a story or storytelling event." "\n"
        "\n"
        "## Pipeline rules: what this pipeline adds" "\n"
        "\n"
        "These rules are the pipeline's own, not the expert's: they say what an answer must contain" "\n"
        "for the stage to check it against the transcript. They do not change the guidelines above." "\n"
        "\n"
        "- Cite a `segment_id` at every boundary. A boundary must lie inside the segments you are" "\n"
        "  answering about; the `CONTEXT` segments are shown so you can see where a boundary falls," "\n"
        "  never as ids you may cite." "\n"
        "- Copy exact wording into transcript_quote from the `text:` line of the cited segment." "\n"
        "  Never invent words and never silently correct them: quotes are checked against the" "\n"
        "  transcript, and a quote that is not there rejects the whole answer." "\n"
        "- If your quote occurs more than once in the segment's text, add \"occurrence\": n" "\n"
        "  (1 = first) to that boundary object, so it says which occurrence you mean. Without it a" "\n"
        "  repeated quote does not say where the boundary is, and the answer is rejected. When the" "\n"
        "  quote occurs once, `occurrence` is optional and must be 1 if you give it." "\n"
        "- A speaker you name must be one of the labels shown in square brackets in the" "\n"
        "  `speakers:` lines — in the segments you are answering about or in their context — or" "\n"
        "  \"unknown\"." "\n"
        "- Those labels come from automatic diarization, not from listening: they are arbitrary ids," "\n"
        "  they may be wrong, and a word the diarizer could not decide is labelled \"unknown\". Treat a" "\n"
        "  label as weak evidence about who is talking, and say so when a boundary depends on it." "\n"
        "- If the transcript itself clearly says who is speaking, say that in `notes`, but keep" "\n"
        "  `speaker` a label shown in square brackets or \"unknown\": those are the only values the" "\n"
        "  pipeline can check against what it showed you." "\n"
        "- `story_evidence` is the expert's list of explanations: short statements of what is" "\n"
        "  represented, how it connects and where experience shows, never a list of segment ids." "\n"
        "- `story_evidence` is the expert's list of explanations. Add `evidence_segment_ids` for the" "\n"
        "  segments that carry that evidence: every story needs both lists non-empty, and every id" "\n"
        "  named there must lie inside the story's own start and end." "\n"
        "- Use empty arrays when nothing is supported. An empty answer is the expected and correct" "\n"
        "  answer when the transcript supports no storytelling event and no story. Never stretch a" "\n"
        "  non-story into a story to please the request." "\n"
        "\n"
        "## INPUT: the transcript (adapted from her input-format section)" "\n"
        "\n"
        "The input is one entry per transcript segment, in three lines:" "\n"
        "\n"
        "- segment_id=<id> start=<seconds> end=<seconds>" "\n"
        "  text: <the segment's transcribed words, exactly>" "\n"
        "  speakers: [speaker label] words ... [speaker label] words ..." "\n"
        "\n"
        "The `text:` line is the segment's transcribed words, and it is the only place a quote may" "\n"
        "be copied from. The `speakers:` line is the same speech split into diarized runs: a label" "\n"
        "in square brackets followed by the words attributed to it, merged when consecutive, with" "\n"
        "\"unknown\" where no single speaker dominated. Read the speaker line only to decide who is" "\n"
        "talking, never to quote from it." "\n"
        "\n"
        "- Quote exact wording from the segment text. Do not invent words or silently correct" "\n"
        "  transcription errors." "\n"
        "- Segment boundaries may be coarse and may combine multiple conversational turns, so a" "\n"
        "  boundary is reported as the segment it falls in plus the words that mark it, not as an" "\n"
        "  estimate of a moment inside the segment." "\n"
        "- Do not infer pauses, overlap, intonation, gaze, gesture, or other nonverbal cues from" "\n"
        "  text alone." "\n"
        "- Speaker labels may be missing from every run. If they are, state that limitation and do" "\n"
        "  not invent them." "\n"
        "\n"
        "## Output (adapted from the expert's \"Output\" section)" "\n"
        "\n"
        "Return valid JSON only:" "\n"
        "\n"
        "{\"storytelling_events\": [{\"event_id\": \"SE1\", \"start\": {\"segment_id\": \"seg000001\", \"transcript_quote\": \"Exact opening words from the transcript\", \"occurrence\": 1, \"speaker\": \"speaker_0\", \"boundary_type\": \"pre_telling\", \"reason\": \"Brief evidence for this boundary\"}, \"end\": {\"segment_id\": \"seg000004\", \"transcript_quote\": \"Exact closing words from the transcript\", \"speaker\": \"speaker_0\", \"boundary_type\": \"coda\", \"reason\": \"Brief evidence for this boundary\"}, \"story_ids\": [\"S1\"], \"tellers\": [\"speaker_0\"], \"interruptions_or_resumptions\": [{\"start_segment_id\": \"seg000002\", \"end_segment_id\": \"seg000003\", \"transcript_quote\": \"Exact words or span, if present\", \"description\": \"Briefly explain the interruption or later resumption\"}], \"confidence\": \"high\", \"notes\": \"Mention ambiguity or limitations, including unclear speaker attribution\"}], \"stories\": [{\"story_id\": \"S1\", \"event_id\": \"SE1\", \"start\": {\"segment_id\": \"seg000001\", \"transcript_quote\": \"Exact opening words from the transcript\", \"speaker\": \"speaker_0\", \"reason\": \"Brief evidence that the story world or experience begins here\"}, \"end\": {\"segment_id\": \"seg000003\", \"transcript_quote\": \"Exact closing words from the transcript\", \"speaker\": \"speaker_0\", \"reason\": \"Brief evidence that the represented events or experience end here\"}, \"tellers\": [\"speaker_0\"], \"summary\": \"One or two sentences describing the represented events or experience\", \"story_evidence\": [\"Represented event(s)\", \"Causal or experiential connection\", \"Embodiment, intentionality, or evaluation, if relevant\"], \"evidence_segment_ids\": [\"seg000001\", \"seg000002\"], \"confidence\": \"high\", \"notes\": \"Mention ambiguity, alternative plausible boundaries, or unclear speaker attribution\"}], \"uncertain_spans\": [{\"start_segment_id\": \"seg000002\", \"end_segment_id\": \"seg000003\", \"transcript_quote\": \"Exact words from the transcript\", \"issue\": \"What is uncertain and why\"}]}" "\n"
        "\n"
        "The one change from her object: a boundary cites a `segment_id` and a verbatim" "\n"
        "`transcript_quote` instead of her numeric time field, because the pipeline computes the" "\n"
        "times itself, and every story carries `evidence_segment_ids` beside `story_evidence`." "\n"
        "`occurrence` is optional: omit it when the quote occurs once in the cited segment's text," "\n"
        "and give it when it occurs more than once, selecting which one (1 = first)." "\n"
        "\n"
        "Allowed boundary_type for a start: pre_telling, story_launch, uncertain." "\n"
        "Allowed boundary_type for an end: coda, listener_acknowledgment, activity_shift, uncertain." "\n"
        "confidence is one of: high, medium, low. story_evidence and evidence_segment_ids are" "\n"
        "required for every story and must not be empty. Use empty arrays when nothing is" "\n"
        "supported; omit nothing." "\n"
        "\n"
        "## Annotation rules" "\n"
        "\n"
        "- Return one JSON object and no surrounding commentary or Markdown." "\n"
        "- Use exact transcript wording in quotes. Do not invent words or silently correct the transcript." "\n"
        "- Do not output times: the pipeline computes them from the segment ids and quotes you cite, so there is no timestamp precision for you to invent." "\n"
        "- Keep story boundaries and storytelling-event boundaries separate, even when they happen to coincide." "\n"
        "- If a boundary is unclear, provide the best-supported estimate and explain the uncertainty." "\n"
        "- Use \"unknown\" for speaker attribution when no reliable label is available." "\n"
        "- Return empty arrays when no storytelling events, stories, or uncertain spans are supported by the transcript." "\n"
        "\n"
        "VIDEO: {video_id}" "\n"
        "\n"
        "CONTEXT (context only - never cite these ids):" "\n"
        "{context}" "\n"
        "\n"
        "TRANSCRIPT (the segments you are answering about; cite only these ids):" "\n"
        "{requested}" "\n"
    )
}


class StoriesTransportError(RuntimeError):
    """Transient endpoint problem worth retrying."""


#: What a credential is replaced with in retained text and log messages.
REDACTED = "[redacted]"

#: Header shapes a gateway can echo back when it rejects a request. Both are redacted
#: because a provider that quotes the request in an error message can quote the
#: credential along with it.
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[^\s\"',;]+")
_AUTHORIZATION_RE = re.compile(r"(?i)\bauthorization\b\s*[:=]\s*[^\s\"',;]+")


# --------------------------------------------------------------- quotes and word times


def _normalise_with_map(text: str) -> tuple[str, list[int]]:
    """Folded, whitespace-collapsed text plus the source index of each character.

    Only two things are normalised, and the brief allows exactly these: case, and runs of
    whitespace collapsing to one space. Every other character is kept, so a quote that
    paraphrases (drops a comma, changes a word) does not match — which is the point: the
    answer must be checkable against the transcript, not approximately related to it.

    One character folds to **one** character always. ``str.casefold`` expands some
    characters — ``ß`` becomes ``ss``, ``ﬀ`` becomes ``ff`` — and expanding one source
    position into two output positions makes the index map lie about where a match started:
    ``sa`` then matches inside ``ßa``, because the two ``s`` share one source index. A
    character whose folded form is longer than itself is therefore kept as it is, which is
    the exact-case fallback: ``Straße`` matches ``straße`` and no longer pretends to match
    ``STRASSE``. Nothing in a Spanish or English transcript needs the expansion.
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
        folded = char.casefold()
        out.append(folded if len(folded) == 1 else char)
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


def _alignment_moves(token_keys: Sequence[str], word_keys: Sequence[str], *,
                     token_offset: int = 0,
                     word_offset: int = 0) -> list[tuple[int, int, int, int]]:
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

    It is a **bottom-up** dynamic program, not a recursion: the recursive version recursed
    once per unmatched token and word, so a segment of a few thousand tokens raised
    ``RecursionError`` instead of aligning. The common head and tail — where both sides are
    equal word for word — are matched first, which is what keeps the grid small on the
    normal case; a segment whose remaining grid would be enormous is placed positionally
    (:data:`_ALIGNMENT_CELL_LIMIT`) rather than spending unbounded time on a correspondence
    nothing can find.
    """
    tokens, words = list(token_keys), list(word_keys)
    head = 0
    while (head < len(tokens) and head < len(words)
           and tokens[head] and tokens[head] == words[head]):
        head += 1
    tail = 0
    while (tail < len(tokens) - head and tail < len(words) - head
           and tokens[-1 - tail] and tokens[-1 - tail] == words[-1 - tail]):
        tail += 1
    moves = [(token_offset + index, 1, word_offset + index, 1) for index in range(head)]
    moves.extend(_core_alignment_moves(tokens[head:len(tokens) - tail],
                                       words[head:len(words) - tail],
                                       token_offset + head, word_offset + head))
    stop_tokens, stop_words = len(tokens) - tail, len(words) - tail
    moves.extend((token_offset + stop_tokens + index, 1, word_offset + stop_words + index, 1)
                 for index in range(tail))
    return moves


def _core_alignment_moves(tokens: Sequence[str], words: Sequence[str], token_offset: int,
                          word_offset: int) -> list[tuple[int, int, int, int]]:
    """The dynamic program over the part of a segment that is not already matched."""
    count_tokens, count_words = len(tokens), len(words)
    if not count_tokens or not count_words:
        return []
    if count_tokens * count_words > _ALIGNMENT_CELL_LIMIT:
        # No correspondence worth finding inside this budget: keep the words that sit at the
        # same position and leave the rest without a time, which is what "no exact
        # alignment" already means.
        return [(token_offset + index, 1, word_offset + index, 1)
                for index in range(min(count_tokens, count_words))
                if tokens[index] and tokens[index] == words[index]]
    skip_token, skip_word, one_to_one = 0, 1, 2
    words_to_token = 3
    tokens_to_word = 3 + _MAX_SPLIT_OR_MERGE
    decisions = [bytearray(count_words) for _ in range(count_tokens)]
    # Rows of the score table, newest first: row i needs rows i+1 .. i+4 only.
    later: list[list[int]] = [[0] * (count_words + 1)]
    for index in range(count_tokens - 1, -1, -1):
        token = tokens[index]
        row = [0] * (count_words + 1)
        for position in range(count_words - 1, -1, -1):
            word = words[position]
            best, code = 0, skip_token
            if token and token == word:
                best, code = 1 + later[0][position + 1], one_to_one
            if token:
                joined = word
                for span in range(2, _MAX_SPLIT_OR_MERGE + 1):
                    if position + span > count_words:
                        break
                    joined += words[position + span - 1]
                    if joined == token:
                        score = span + later[0][position + span]
                        if score > best:
                            best, code = score, words_to_token + span
            if word:
                joined = token
                for span in range(2, _MAX_SPLIT_OR_MERGE + 1):
                    if index + span > count_tokens:
                        break
                    joined += tokens[index + span - 1]
                    if joined == word:
                        score = span + later[span - 1][position + 1]
                        if score > best:
                            best, code = score, tokens_to_word + span
            if later[0][position] > best:
                best, code = later[0][position], skip_token
            if row[position + 1] > best:
                best, code = row[position + 1], skip_word
            row[position] = best
            decisions[index][position] = code
        later.insert(0, row)
        if len(later) > _MAX_SPLIT_OR_MERGE + 1:
            later.pop()
    moves: list[tuple[int, int, int, int]] = []
    index = position = 0
    while index < count_tokens and position < count_words:
        code = decisions[index][position]
        if code == skip_token:
            index += 1
        elif code == skip_word:
            position += 1
        elif code == one_to_one:
            moves.append((token_offset + index, 1, word_offset + position, 1))
            index += 1
            position += 1
        elif code < tokens_to_word:
            span = code - words_to_token
            moves.append((token_offset + index, 1, word_offset + position, span))
            index += 1
            position += span
        else:
            span = code - tokens_to_word
            moves.append((token_offset + index, span, word_offset + position, 1))
            index += span
            position += 1
    return moves


def _offset_groups(text: str, words: Sequence[Mapping[str, Any]], keys: Sequence[str],
                   cores_by_start: Mapping[int, int],
                   cores_by_end: Mapping[int, int]) -> list[tuple[int, list[int]]]:
    """Word→token anchors read from the alignment's own character offsets.

    ``character_start``/``character_end`` are NULL in every corpus measured so far, which
    is why placement did not depend on them. A present-but-wrong offset must not move a
    boundary (all-zeros offsets are a real defect, not a hypothesis), so an offset is
    honoured only when the slice it names is *exactly this word* after the permitted
    normalisation **and** exactly one token. When that holds it says something the token
    aligner cannot: two identical words are placed by their own offsets instead of by
    order.

    The result is a chain of ``(token_index, [word positions])`` groups in increasing
    order; several words may anchor to one token (a contraction the aligner split), and a
    group that would run backwards is dropped rather than used to place anything.
    """
    groups: list[tuple[int, list[int]]] = []
    for position, (word, key) in enumerate(zip(words, keys)):
        start, end = word.get("character_start"), word.get("character_end")
        if isinstance(start, bool) or isinstance(end, bool):
            continue
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if not 0 <= start < end <= len(text) or _squash(text[start:end]) != key:
            continue
        token_index = cores_by_start.get(start)
        if token_index is None or cores_by_end.get(end) != token_index:
            continue
        if groups and groups[-1][0] == token_index:
            if position > groups[-1][1][-1]:
                groups[-1][1].append(position)
            continue
        if groups and token_index <= groups[-1][0]:
            continue
        groups.append((token_index, [position]))
    return groups


def align_token_times(text: str,
                      words: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One entry per text token: its core range and the times of the words covering it.

    Placement is the alignment of :func:`_alignment_moves` over the segment's own tokens,
    with two refinements:

    * a word whose ``character_start``/``character_end`` really slice it out of the text
      anchors its own token (:func:`_offset_groups`), and only the stretches between
      anchors are aligned;
    * an alignment row that contains no word at all (a tokenizer that emitted a separate
      ``¿``) takes part in nothing, because it can only ever be *near* a token and its
      time is not that token's boundary.

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
    kept = [index for index, key in enumerate(word_keys) if key]
    kept_keys = [word_keys[index] for index in kept]
    groups = _offset_groups(text, [word_list[index] for index in kept], kept_keys,
                            {start: index for index, (start, _end) in enumerate(cores)},
                            {end: index for index, (_start, end) in enumerate(cores)})
    # (token index, token count, word positions) in "kept" space; the end sentinel aligns
    # whatever is left after the last anchor.
    plan: list[tuple[int, int, list[int]]] = []
    token_from = word_from = 0
    for token_index, word_positions in groups + [(len(cores), [len(kept)])]:
        first_word = word_positions[0]
        for move in _alignment_moves(token_keys[token_from:token_index],
                                     kept_keys[word_from:first_word],
                                     token_offset=token_from, word_offset=word_from):
            move_token, move_tokens, move_word, move_words = move
            plan.append((move_token, move_tokens,
                         list(range(move_word, move_word + move_words))))
        if token_index < len(cores):
            plan.append((token_index, 1, list(range(first_word, word_positions[-1] + 1))))
        token_from = token_index + 1
        word_from = word_positions[-1] + 1
    for token_index, token_count, positions in plan:
        originals = [kept[position] for position in positions]
        for offset in range(token_count):
            entries[token_index + offset]["word_indexes"] = list(originals)
        # One token, one word — or the several words it was split into. Both ends come from
        # the word(s) that really cover it. One word over several tokens only fills its own
        # outer edges: the word's boundary is not the boundary of the tokens between them.
        entries[token_index]["start_time"] = word_list[originals[0]].get("start_time")
        entries[token_index + token_count - 1]["end_time"] = word_list[
            originals[-1]].get("end_time")
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

    def followup_body(self, previous: str | None, problems: Sequence[str], *,
                      model_fallback: str = "",
                      extra_body_fallback: Mapping[str, Any] | None = None
                      ) -> dict[str, Any]:
        """A retry of the same question: the rejected answer, then what was wrong with it.

        The prompt is untouched and the repaired request is not part of :meth:`key`, so the
        answer is filed as the answer to the request that was asked, not to a conversation
        that happened to converge. Resampling the identical prompt and hoping for a
        different answer is not a plan; telling the model which checks it failed is.
        """
        body = self.body(model_fallback=model_fallback,
                         extra_body_fallback=extra_body_fallback)
        messages = list(body.get("messages") or [])
        if previous:
            messages.append({"role": "assistant", "content": previous})
        messages.append({"role": "user", "content": feedback_message(problems)})
        body["messages"] = messages
        return body


def feedback_message(problems: Sequence[str]) -> str:
    """The repair instruction a rejected attempt earns on the next one.

    It names the checks that failed, in the validator's own words, and asks for the whole
    object again: a model that returns only the repaired fragment leaves the rest of the
    answer missing, which is a different defect wearing the same shape.
    """
    lines = [f"- {problem}" for problem in problems if str(problem).strip()]
    if not lines:
        lines = ["- the previous answer could not be read as the required JSON object"]
    return ("Your previous answer was rejected by the pipeline's validation. Fix every "
            "problem listed below and return the corrected answer as one complete JSON "
            "object with the same contract as before: include every storytelling event, "
            "story and uncertain span you still claim, not only the parts you changed. "
            "Return only the JSON object.\n\nProblems:\n" + "\n".join(lines))


def _problems_from(error: Exception) -> list[str]:
    """The validator's problems behind a rejection, as lines a model can act on."""
    issues = getattr(error, "details", {}).get("issues")
    if issues:
        return [str(issue) for issue in issues]
    return [f"{type(error).__name__}: {error}"]


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
      segment, a repeated quote with no (or an impossible) ``occurrence``, an enum value
      outside its vocabulary, a duplicate id, a story whose event is not in the answer,
      inconsistent ``event.story_ids``, a story outside its event's span, an end before its
      start, a missing or blank ``story_evidence`` explanation (or one that is only segment
      ids), missing ``evidence_segment_ids`` or evidence outside the story's own span, a
      reference to a real segment the prompt did not show — raises
      :class:`ValidationError`. The answer describes a transcript that does not exist, so
      nothing in it can be trusted and the whole batch is retried;
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
        unseen = [item for item in evidence_ids
                  if item in request.segments and item not in request.visible_ids]
        if unseen:
            # A real segment nobody was shown is not evidence the model saw: the prompt
            # never carried its text, so the id came from somewhere other than this
            # request. That is a breach, not a window drop.
            problems.append(f"{label}: evidence_segment_ids names segments that were not "
                            f"shown in this request: {unseen[:5]}")
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
        unseen = [segment_id for segment_id in (start_id, end_id)
                  if segment_id not in request.visible_ids]
        if unseen:
            problems.append(f"{label}: names segments that were not shown in this request: "
                            f"{unseen}")
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
        # A span that touches a context segment is the neighbouring window's answer: kept
        # here it would publish an interval reaching into another window's territory, so it
        # is dropped and counted like every other cross-window claim.
        if (span["start_segment_id"] not in window_ids
                or span["end_segment_id"] not in window_ids):
            dropped.append({"kind": "uncertain_span",
                            "model_id": f"{span['start_segment_id']}.."
                                        f"{span['end_segment_id']}",
                            "reason": "touches a context segment outside this window"})
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
    if segment_id not in request.visible_ids:
        # Real segment, never shown: the prompt did not carry its text, so a quote out of it
        # cannot have been copied from what the model read. A breach, not a window drop —
        # a drop would quietly record the model citing text it never had.
        problems.append(f"{label}: {role}.segment_id {segment_id} is a segment of this "
                        f"transcript that was not shown in this request, so no quote from "
                        f"it can support a boundary")
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
    span = _pick_occurrence(raw.get("occurrence"), occurrences, quote, segment_id,
                            label, role, problems)
    if span is None:
        return None
    when, precision = _boundary_time(segment, span, role)
    return {"segment_id": segment_id, "quote": quote, "speaker": speaker or UNKNOWN_SPEAKER,
            "reason": reason, "time": when, "precision": precision}


def _pick_occurrence(raw: Any, occurrences: Sequence[tuple[int, int]], quote: str,
                     segment_id: str, label: str, role: str,
                     problems: list[str]) -> tuple[int, int] | None:
    """Which word-aligned occurrence of the quote the boundary means.

    A repeated quote is ambiguous only until the answer says which occurrence it means, and
    ``occurrence`` is that answer (1 = first). Asking the model to quote *more* words was
    the earlier rule and it could not be met: a segment whose text is ``no no`` has no
    longer quote that is unique, so a correct boundary could not be expressed at all and the
    retry could not converge. Selecting an occurrence always can be.

    ``None`` means the answer did not say, or named an occurrence that does not exist — a
    breach either way, reported with the count so the model can fix it.
    """
    count = len(occurrences)
    if raw is None:
        if count == 1:
            return occurrences[0]
        problems.append(
            f"{label}: {role}.transcript_quote occurs {count} times in {segment_id}, so it "
            f"does not say where the boundary is; add \"occurrence\": n (1 = first) to "
            f"choose which one: {quote[:120]!r}")
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        problems.append(f"{label}: {role}.occurrence {raw!r} is not a whole number")
        return None
    if count == 1:
        if raw != 1:
            problems.append(f"{label}: {role}.occurrence is {raw}, but the quote occurs "
                            f"once in {segment_id}, so the only occurrence is 1")
            return None
        return occurrences[0]
    if raw < 1 or raw > count:
        problems.append(f"{label}: {role}.occurrence {raw} is out of range: the quote "
                        f"occurs {count} times in {segment_id} (1..{count})")
        return None
    return occurrences[raw - 1]


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
    (``evidence_segment_ids``), where the range check can see them, so an "explanation"
    that is only segment ids is a breach too: it says *where* to look and never *why*, and
    it would satisfy the required-and-non-empty rule while explaining nothing.
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
        if _only_segment_ids(explanation):
            problems.append(
                f"{label}: story_evidence entry {explanation[:60]!r} is only segment ids; "
                f"explain what the stretch represents and where its experience shows, and "
                f"leave the ids to evidence_segment_ids")
            continue
        items.append(explanation)
    return items


def _only_segment_ids(explanation: str) -> bool:
    """Is this "explanation" nothing but segment ids (``seg000001``, ``["seg000002"]``)?"""
    pieces = [piece for piece in re.split(r"[^0-9A-Za-z_]+", explanation) if piece]
    return bool(pieces) and all(_SEGMENT_ID_RE.match(piece) for piece in pieces)


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
        # The body of the next attempt. A rejected *answer* earns a repair: the same prompt,
        # the rejected answer as an assistant message and the validator's problems as the
        # follow-up question. A rejected *transport* keeps the body it already has.
        body = request.body(model_fallback=self.model,
                            extra_body_fallback=self.extra_body)
        for attempt in range(self.max_retries + 1):
            started = time.time()
            raw_text: str | None = None
            feedback: list[str] | None = None
            try:
                payload = self._post(body)
                self.timings_ms.append(round((time.time() - started) * 1000, 1))
                self._record_usage(payload)
                try:
                    raw_text = extract_message_content(payload)
                except TranslationTransportError as exc:
                    # No message content is the model's empty answer, not a broken network:
                    # it enters the validation path (kept, retried with feedback) instead of
                    # escaping as a transport failure after one attempt.
                    raise ValueError(f"response carried no message content ({exc})") from exc
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
                feedback = _problems_from(exc)
            if attempt < self.max_retries:
                if feedback is not None:
                    body = request.followup_body(raw_text, feedback,
                                                 model_fallback=self.model,
                                                 extra_body_fallback=self.extra_body)
                delay = self.backoff_base * (2**attempt) + random.uniform(0, self.backoff_base)
                self._sleep(delay)
        raise StageError(
            f"stories window {request.window_index} failed after {self.max_retries + 1} "
            f"attempts: {self.redact(str(last_error))}",
            details={"window_index": request.window_index,
                     "requested_ids": request.requested_ids,
                     "attempts": self.max_retries + 1},
        )

    def _record_rejection(self, request: StoriesRequest, attempt: int, raw_text: str | None,
                          error: Exception) -> None:
        """Keep the text of a rejected answer, with the reason it was rejected.

        ``raw/rejected/window_<i>_<key>_attempt<n>_<utc>.txt``, so a batch's failed attempts
        are as auditable as its accepted one, and a second run of the same window appends to
        the history instead of overwriting the first: the attempt number alone repeats
        across runs, and an auditor who lost the earlier text cannot tell a retry that
        failed from one that never happened. Only the response body is stored — never a
        header — and the credential is redacted out of both the body and the reason,
        because a gateway that echoes the request echoes what it was sent. A transport
        failure that produced no text is not stored here: it is reported as a failure, not
        as a response.
        """
        if self.rejected_dir is None or raw_text is None:
            return
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        reason = self.redact(f"{type(error).__name__}: {error}")
        body = self.redact(raw_text)
        try:
            path = self._unique_path(
                self.rejected_dir /
                f"window_{request.window_index}_{request.key()}_attempt{attempt + 1}_{stamp}.txt")
            atomic_write_text(
                path,
                f"# rejected answer (attempt {attempt + 1} of {self.max_retries + 1}) for "
                f"window {request.window_index} of {request.video_id}\n"
                f"# request_key: {request.key()}\n"
                f"# reason: {reason}\n"
                f"\n{body}",
            )
        except OSError:  # pragma: no cover - a full disk must not hide the real failure
            pass

    @staticmethod
    def _unique_path(path: Path) -> Path:
        """A path that does not exist yet: ``name``, then ``name_2``, ``name_3``, …

        Reserved by creating the file, so two attempts in the same second cannot pick the
        same name and one of them silently disappear.
        """
        candidate = path
        counter = 1
        while True:
            try:
                with open(candidate, "x", encoding="utf-8"):
                    pass
                return candidate
            except FileExistsError:
                counter += 1
                candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        # The caller builds the body (the first attempt from the request, a retry from the
        # request plus the rejected answer and the problems): the cache key hashes the
        # request's own body, so a repair can never be mistaken for another question.
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

    def redact(self, text: str) -> str:
        """The same text with the configured credential taken out of it.

        Applied to every retained attempt and to the failure message, because both are
        written to disk and to the stage log: a provider that echoes the request (or the
        header it received) must not turn a rejected answer into a leaked API key.
        """
        cleaned = str(text)
        if self.api_key:
            cleaned = cleaned.replace(self.api_key, REDACTED)
        cleaned = _BEARER_RE.sub(f"Bearer {REDACTED}", cleaned)
        cleaned = _AUTHORIZATION_RE.sub(f"authorization: {REDACTED}", cleaned)
        return cleaned

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
        raw_windows: list[dict[str, Any]] = []
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
            # Every window this run answered, by the file that holds its answer — empty
            # windows included. The summary is what ``validate()`` walks, so a window left
            # out of it is a window whose raw evidence is never read back, which is how a
            # corrupt raw file used to pass validation.
            raw_windows.append({
                "window_index": index,
                "request_key": request.key(),
                "file": f"window_{index}_{request.key()}.json",
                "requested_ids": request.requested_ids,
                "context_ids": [str(row["segment_id"]) for row in request.context],
                "empty": verdict.answer.empty,
            })
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
            # The raw window files this answer consists of, each with the request key its
            # rows carry. Empty windows are listed too: "nothing was found" is an answer
            # that also has a response behind it.
            "raw_windows": raw_windows,
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

        Three states: absent (write it), present with exactly the cached bytes (leave them
        alone — they are the original), present with anything else (rewrite from the cached
        content, because a raw file that disagrees with its rows files the wrong evidence
        beside them). The comparison is on **bytes**, not on the parsed payload: a
        re-serialisation and a hand-edited response both parse to the accepted object and
        neither is what the endpoint sent, so parsed equality left the wrong file in place.
        A verdict with no content cannot be written at all and is not turned into a
        re-serialisation that only looks like what the endpoint sent.
        """
        path = raw_dir / f"window_{request.window_index}_{request.key()}.json"
        if not verdict.content:
            return
        expected = verdict.content.encode("utf-8")
        if path.is_file():
            try:
                if path.read_bytes() == expected:
                    return
            except OSError:
                pass
        # Bytes, not the parsed payload: two answers can parse to the same object and one of
        # them can still be text the endpoint never sent. The cache holds the exact bytes, so
        # the raw file next to the rows has to be exactly those.
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
        # Every row names the response it came from, so that response has to be there *and*
        # have to say what the row says: a table whose provenance points at a missing or
        # edited file cannot be audited, and the raw window is the only place the model's
        # own words survive.
        problems.extend(self._raw_evidence_problems(ctx, summary, tables))
        windows = int(summary.get("windows") or 0)
        named = {int(row["window_index"]) for row in events + stories + uncertain}
        if windows and windows < len(named):
            problems.append("the tables name more windows than the raw summary recorded")
        if problems:
            raise ValidationError(self.name, problems[:20])
        return {"storytelling_events": len(events), "stories": len(stories),
                "uncertain_spans": len(uncertain), "windows": windows,
                "empty_windows": max(windows - len(named), 0)}

    def _raw_evidence_problems(self, ctx: StageContext, summary: Mapping[str, Any],
                               tables: Mapping[str, list[dict[str, Any]]]) -> list[str]:
        """Every window the summary lists, re-read and re-validated from disk.

        The tables are a conclusion; ``stories/raw`` holds the evidence for it. Checking
        only that a file exists accepts a truncated or hand-edited response, because nothing
        ever reads it: a corrupt raw file next to a plausible row passed ``validate()``. So
        each listed window is parsed and re-run through the same validator the run used,
        against a request rebuilt from the current transcript, and the ids it yields must be
        the rows the tables attribute to that window.
        """
        problems: list[str] = []
        raw_dir = ctx.artifact("stories_raw")
        listed = summary.get("raw_windows")
        if not isinstance(listed, list) or not listed:
            problems.append("stories_summary.json does not list the raw window files "
                            "(`raw_windows`), so no response can be checked against the "
                            "tables")
            return problems
        try:
            rebuilt = {request.key(): request for request in self.build_requests(ctx)}
        except (StageError, OSError, ValueError) as exc:
            problems.append(f"the windows cannot be rebuilt to check the raw evidence: {exc}")
            return problems
        events = tables["storytelling_events"]
        stories = tables["stories"]
        uncertain = tables["uncertain_spans"]
        seen: set[str] = set()
        for entry in listed:
            if not isinstance(entry, dict):
                problems.append(f"stories_summary.json raw_windows contains {entry!r}, not an "
                                f"object")
                continue
            key = str(entry.get("request_key") or "")
            name = str(entry.get("file") or "")
            index = int(entry.get("window_index") or 0)
            if not key or not name:
                problems.append("stories_summary.json lists a raw window without its file "
                                "or request_key")
                continue
            if key in seen:
                problems.append(f"stories_summary.json lists window {index} twice ({key})")
                continue
            seen.add(key)
            path = raw_dir / name
            if not path.is_file():
                problems.append(f"stories/raw/{name} is missing, so the window it answers "
                                f"cannot be audited")
                continue
            request = rebuilt.get(key)
            if request is None:
                problems.append(f"stories/raw/{name} names request {key}, which the current "
                                f"transcript and configuration no longer produce")
                continue
            try:
                payload = parse_stories_response(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                problems.append(f"stories/raw/{name} is not a parseable stories answer: {exc}")
                continue
            try:
                answer = validate_response(payload, request)
            except ValidationError as exc:
                issues = "; ".join(str(item) for item in exc.details.get("issues") or [])
                problems.append(f"stories/raw/{name} no longer passes the stories validation "
                                f"against the current transcript: {issues[:300]}")
                continue
            for table_name, rows, field in (("storytelling_events", events, "event_id"),
                                            ("stories", stories, "story_id")):
                on_disk = {str(row[field]) for row in rows
                           if int(row["window_index"]) == index}
                in_raw = {str(row[field]) for row in answer.events if field == "event_id"}
                if field == "story_id":
                    in_raw = {str(row[field]) for row in answer.stories}
                if on_disk != in_raw:
                    problems.append(
                        f"stories/{table_name}.parquet and stories/raw/{name} disagree about "
                        f"window {index}: the table has {sorted(on_disk)}, the response "
                        f"yields {sorted(in_raw)}")
            spans_on_disk = sum(1 for row in uncertain
                                if int(row["window_index"]) == index)
            if spans_on_disk != len(answer.uncertain_spans):
                problems.append(f"stories/uncertain_spans.parquet has {spans_on_disk} row(s) "
                                f"for window {index}, stories/raw/{name} yields "
                                f"{len(answer.uncertain_spans)}")
        for table_name, rows, field in (("storytelling_events", events, "event_id"),
                                        ("stories", stories, "story_id"),
                                        ("uncertain_spans", uncertain,
                                         "start_segment_id")):
            for row in rows:
                if str(row["request_key"]) in seen:
                    continue
                problems.append(f"{table_name} row {row.get(field)} names request "
                                f"{row['request_key']}, which stories_summary.json does not "
                                f"list as a window")
                break
        missing = [key for key in rebuilt if key not in seen]
        if missing:
            problems.append(f"stories_summary.json does not list {len(missing)} window(s) the "
                            f"current configuration produces, so their responses were never "
                            f"checked")
        return problems

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
