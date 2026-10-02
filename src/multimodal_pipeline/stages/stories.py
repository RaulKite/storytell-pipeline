"""`stories`: narrative windows in the transcript, found by the pipeline's own LLM.

ODD §20.3 becoming code, shaped by a probe of the real endpoint (the measurements are in
``odd/tasks/stories-detection.md``). Three properties carry the design:

* **Emptiness is a result, not a failure.** The prompt states that an empty list is the
  expected and correct answer for fragments, greetings and announcements, and the probe
  measured that rule holding: two of three real clips answered ``"stories": []`` with a
  stated ``no_story_reason``, unprompted. An empty answer writes a zero-row table and is
  *counted* per window, because §20.3 also demands that something count how often the
  answer is empty.
* **The response is validated or the batch is rejected.** The probe's stories cited only
  real ``segment_id``s and real boundary times, which makes rejection a real path rather
  than a hypothetical one: an invented id or a fabricated timestamp is the model's error,
  the batch is retried as a batch, and the stage fails loudly after ``max_retries``. A
  partially accepted answer would put a story in the corpus that no transcript supports.
* **Windowing is honest about what it clips.** Long transcripts are split into
  start-ordered windows; a story whose evidence is not entirely inside one window is
  dropped *and counted* (``dropped_outside_window``), never quietly kept.

Nothing here imports a heavy tool: the stage is httpx, pyarrow and stdlib, reached over
HTTP by the same OpenAI-compatible endpoint translation already uses.
"""

from __future__ import annotations

import json
import hashlib
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import httpx
import pyarrow as pa

from ..artifacts import atomic_write_json, atomic_write_text, read_json
from ..config import stable_hash
from ..exceptions import StageError, ValidationError
from ..schemas import STORIES_SCHEMA, read_table, write_table
from .base import Stage, StageContext
from .translation import extract_message_content

#: Decimal places a transcript boundary is compared at. The probe's endpoint echoes the
#: times it was shown, which are rendered with three; a model that rounds 12.3456789 to
#: 12.346 has quoted a real boundary, and refusing it would reject the correct answer.
BOUNDARY_PRECISION = 3

PROMPTS: dict[str, str] = {
    # Wording taken from the probe that measured this endpoint's behaviour
    # (/tmp/probe_stories.py, PROMPT_VERSION "probe-v1"). Rule 4 is the load-bearing
    # sentence and is reproduced deliberately: without permission to answer "no story",
    # the model stretches a greeting into one to satisfy the request.
    "v1": (
        "You are an analyst who marks NARRATIVE windows in a short video transcript.\n"
        "\n"
        "A narrative window (\"story\") is a stretch of speech that tells something with "
        "a beginning and an end: an anecdote, a recalled event, a reported episode with a "
        "point. News framing, greetings, announcements, and a single bare statement of "
        "fact are NOT stories.\n"
        "\n"
        "Rules:\n"
        "1. A window must start and end at the boundaries of the transcript segments "
        "listed below under REQUESTED: use the given start/end seconds exactly. Never "
        "invent times.\n"
        "2. Every story must cite the segment_ids that carry it (\"evidence\"). Only ids "
        "listed under REQUESTED.\n"
        "3. Stories may nest: a story inside another reports parent_id of the outer one.\n"
        "4. MOST IMPORTANT: if the transcript contains no narrative window, return "
        "\"stories\": [] and say why in \"no_story_reason\". An empty list is the expected "
        "and correct answer for fragments, greetings, and announcements. Never stretch a "
        "non-story into a story to please the request.\n"
        "5. Return ONLY this JSON object, no prose, no fences:\n"
        "{\"stories\": [{\"story_id\": \"s1\", \"parent_id\": null, \"start_time\": 0.0, "
        "\"end_time\": 0.0, \"title\": \"up to 8 words\", "
        "\"why_it_is_a_story\": \"one sentence\", "
        "\"evidence_segment_ids\": [\"seg000001\"], \"confidence\": 0.0}], "
        "\"no_story_reason\": null}\n"
        "\n"
        "VIDEO: {video_id}\n"
        "\n"
        "SPEAKER TURNS (context only):\n{turns}\n"
        "\n"
        "CONTEXT (context only - never cite these ids):\n{context}\n"
        "\n"
        "REQUESTED (the transcript you are answering about):\n{requested}\n"
    )
}


class StoriesTransportError(RuntimeError):
    """Transient endpoint problem worth retrying."""


@dataclass
class StoriesRequest:
    """One endpoint call: one window of the transcript plus what to show as context."""

    video_id: str
    window_index: int
    requested: list[dict[str, Any]]
    context: list[dict[str, Any]]
    turns: list[dict[str, Any]]
    #: Every id and boundary in the whole transcript. A story citing one of these but
    #: outside this window is a story that belongs to another window (dropped and
    #: counted); an id outside this set was invented (the batch is rejected).
    all_segment_ids: frozenset[str] = field(default_factory=frozenset)
    all_starts: frozenset[float] = field(default_factory=frozenset)
    all_ends: frozenset[float] = field(default_factory=frozenset)
    prompt_version: str = "v1"
    model: str = ""
    temperature: float = 0.0
    max_output_tokens: int | None = None
    #: Which endpoint this window is addressed to. Part of the key because the answer
    #: is that endpoint's opinion: moving base_url while the model name stays a LiteLLM
    #: alias ("chat") is exactly the change that alters the answer and changes nothing
    #: else in the request (independent verification of 9f39903, finding 1).
    endpoint: str = ""

    @property
    def requested_ids(self) -> list[str]:
        return [str(row["segment_id"]) for row in self.requested]

    @property
    def id_prefix(self) -> str:
        """Namespace the model's ``s1``/``s2`` belong to: two windows both start at s1."""
        return f"w{self.window_index}"

    def prompt(self) -> str:
        template = PROMPTS.get(self.prompt_version)
        if not template:
            raise StageError(f"unknown stories prompt version: {self.prompt_version}")
        # Substituted rather than str.format'ed: the answer contract in the prompt is a
        # literal JSON object, and .format would need every one of its braces doubled —
        # which turns the sentence an operator most needs to read into noise. The probe
        # rendered it the same way for the same reason.
        return (template
                .replace("{video_id}", self.video_id)
                .replace("{turns}", _render_turns(self.turns))
                .replace("{context}", _render(self.context))
                .replace("{requested}", _render(self.requested)))

    def key(self) -> str:
        """Request digest: prompt + endpoint + model + temperature + exactly what was shown.

        The docstring is a contract, not decoration, and an independent verifier proved
        it was broken: with only `prompt_version` inside, editing the *text* of the v1
        prompt left the key identical, so a forced rerun re-read the old prompt's
        answers. The rendered template's own hash binds the promise now, and `endpoint`
        binds where the question goes — together with the stage fingerprint (which
        already hashes both), no reachable path reuses an answer across a change that
        would have changed it.
        """
        template = PROMPTS.get(self.prompt_version, "")
        return stable_hash({
            "video_id": self.video_id,
            "window_index": self.window_index,
            "requested": self.requested_ids,
            "context": [str(row["segment_id"]) for row in self.context],
            "texts": [str(row.get("text") or "") for row in self.requested],
            "context_texts": [str(row.get("text") or "") for row in self.context],
            # Speakers and turns change the answer without changing one word of text, so
            # they belong in the key rather than only in the rendered prompt.
            "speakers": [str(row.get("speaker_id") or "") for row in self.requested],
            "times": [[float(row["start_time"]), float(row["end_time"])]
                      for row in self.requested],
            "turns": [[str(turn.get("turn_id")), str(turn.get("speaker_id")),
                       float(turn["start_time"]), float(turn["end_time"])]
                      for turn in self.turns],
            "prompt_version": self.prompt_version,
            "prompt_text_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
            "endpoint": self.endpoint,
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }, length=24)


def _render(rows: Sequence[dict[str, Any]]) -> str:
    """Transcript lines the model can cite: id, speaker, real boundaries, text."""
    if not rows:
        return "(none)"
    return "\n".join(
        f"- segment_id={row['segment_id']} speaker={row.get('speaker_id') or 'UNKNOWN'} "
        f"start={float(row['start_time']):.{BOUNDARY_PRECISION}f} "
        f"end={float(row['end_time']):.{BOUNDARY_PRECISION}f}: {row.get('text') or ''}"
        for row in rows)


def _render_turns(turns: Sequence[dict[str, Any]]) -> str:
    if not turns:
        return "(none)"
    return "\n".join(
        f"- turn {turn['turn_id']} speaker={turn.get('speaker_id') or 'UNKNOWN'} "
        f"{float(turn['start_time']):.{BOUNDARY_PRECISION}f}-"
        f"{float(turn['end_time']):.{BOUNDARY_PRECISION}f}"
        for turn in turns)


def build_requests(segments: Sequence[dict[str, Any]], *, video_id: str,
                   max_segments_per_request: int, context_segments: int,
                   prompt_version: str, model: str, temperature: float,
                   max_output_tokens: int | None,
                   turns: Sequence[dict[str, Any]] | None = None,
                   endpoint: str = "") -> list[StoriesRequest]:
    """Split a transcript into start-ordered windows, each with neighbouring context.

    Windows partition the transcript (they do not overlap): overlapping *requests* would
    produce the same story twice under two ids. ``context_segments`` neighbours travel
    along as CONTEXT and are explicitly not citable, which is how the model sees what
    comes in and out of the window without being allowed to answer about it.
    """
    if max_segments_per_request < 1:
        raise StageError("stories.max_segments_per_request must be positive")
    ordered = sorted(segments, key=lambda row: (float(row["start_time"]),
                                                float(row["end_time"]),
                                                str(row["segment_id"])))
    all_ids = frozenset(str(row["segment_id"]) for row in ordered)
    all_starts = frozenset(round(float(row["start_time"]), BOUNDARY_PRECISION) for row in ordered)
    all_ends = frozenset(round(float(row["end_time"]), BOUNDARY_PRECISION) for row in ordered)
    requests: list[StoriesRequest] = []
    for index, start in enumerate(range(0, len(ordered), max_segments_per_request)):
        stop = min(len(ordered), start + max_segments_per_request)
        context_start = max(0, start - context_segments)
        context_end = min(len(ordered), stop + context_segments)
        requests.append(StoriesRequest(
            video_id=video_id,
            window_index=index,
            requested=list(ordered[start:stop]),
            context=list(ordered[context_start:start]) + list(ordered[stop:context_end]),
            turns=list(turns or []),
            all_segment_ids=all_ids,
            all_starts=all_starts,
            all_ends=all_ends,
            prompt_version=prompt_version,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            endpoint=endpoint,
        ))
    return requests


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


def validate_stories(payload: dict[str, Any], request: StoriesRequest, *,
                     stage: str = "stories") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reject or drop. Returns ``(accepted stories, dropped stories)``.

    Two different refusals, because they mean different things:

    * a **breach** (invented id, fabricated boundary, confidence out of range, a parent
      that is not a story of this answer, a cycle) raises :class:`ValidationError`. The
      answer describes a transcript that does not exist, so nothing in it can be trusted
      and the whole batch is retried;
    * a **drop** is a story anchored outside this window — its evidence or a boundary
      belongs to another window of the same transcript. That story is not this window's
      answer and another window may well make it; it is returned so the caller can count
      it, which is what keeps a clipped long transcript a known gap.

    Accepted rows carry window-namespaced ids (``w0-s1``): the model is allowed to answer
    with ``s1`` in every window, and a story id that is only unique per request cannot be
    a table key.
    """
    problems: list[str] = []
    dropped: list[dict[str, Any]] = []
    raw_stories = payload.get("stories")
    if not isinstance(raw_stories, list):
        raise ValidationError(stage, [f"stories must be a list, got "
                                     f"{type(raw_stories).__name__}"])
    if not raw_stories:
        reason = payload.get("no_story_reason")
        # An empty list *with* a reason is the answer §20.3 asked for. Without one it is
        # indistinguishable from a model that said nothing at all, so it is a breach.
        if not isinstance(reason, str) or not reason.strip():
            problems.append("empty stories list without a non-empty no_story_reason")

    window_ids = set(request.requested_ids)
    window_starts = {round(float(row["start_time"]), BOUNDARY_PRECISION) for row in request.requested}
    window_ends = {round(float(row["end_time"]), BOUNDARY_PRECISION) for row in request.requested}

    entries: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for position, story in enumerate(raw_stories):
        if not isinstance(story, dict):
            problems.append(f"story {position} is a {type(story).__name__}, not an object")
            continue
        model_id = str(story.get("story_id") or "").strip()
        label = model_id or f"story {position}"
        if not model_id:
            problems.append(f"story {position} has no story_id")
        elif model_id in seen:
            # Two rows called s1 make both parent_id resolution and the table's story_id
            # column meaningless, so the batch is the model's to redo.
            problems.append(f"duplicate story_id {model_id!r} in one answer")
        else:
            seen.add(model_id)
            entries.append((model_id, story))

        reasons: list[str] = []
        evidence = story.get("evidence_segment_ids")
        if not isinstance(evidence, list) or not evidence:
            problems.append(f"{label}: evidence_segment_ids missing or empty")
        else:
            ids = [str(item) for item in evidence]
            invented = [item for item in ids if item not in request.all_segment_ids]
            if invented:
                problems.append(f"{label}: invented evidence ids {invented[:5]}")
            elif any(item not in window_ids for item in ids):
                outside = [item for item in ids if item not in window_ids]
                reasons.append(f"evidence outside this window: {outside[:5]}")

        start, end = _number(story.get("start_time")), _number(story.get("end_time"))
        if start is None or end is None:
            problems.append(f"{label}: missing or non-numeric start_time/end_time")
        elif start >= end:
            problems.append(f"{label}: window ends at or before it starts "
                            f"({start}..{end})")
        else:
            rounded_start = round(start, BOUNDARY_PRECISION)
            rounded_end = round(end, BOUNDARY_PRECISION)
            if rounded_start not in window_starts or rounded_end not in window_ends:
                real = (rounded_start in request.all_starts and rounded_end in request.all_ends)
                if real:
                    reasons.append(f"boundary {start}..{end} belongs to another window")
                else:
                    problems.append(f"{label}: boundary {start}..{end} is not a real "
                                    f"segment boundary of this transcript")

        confidence = _number(story.get("confidence"))
        if confidence is None:
            problems.append(f"{label}: confidence is missing or not a number")
        elif not 0.0 <= confidence <= 1.0:
            problems.append(f"{label}: confidence {confidence} is outside [0, 1]")

        if reasons:
            dropped.append({"model_story_id": label, "reason": "; ".join(reasons),
                            "story": story})

    if problems:
        raise ValidationError(stage, problems)

    # A story whose parent was dropped cannot be placed inside this window either, and
    # that is a drop rather than a breach. The distinction matters operationally: at
    # temperature 0 the model answers the same way when asked again, so refusing the batch
    # for it would retry a fixed answer until the stage failed and the video lost every
    # story in it. Dropping keeps the child's absence counted, which is the same treatment
    # its parent got. Applied to fixpoint so a chain of nested stories unwinds.
    dropped_ids = {item["model_story_id"] for item in dropped}
    changed = True
    while changed:
        changed = False
        for story_id, story in entries:
            if story_id in dropped_ids:
                continue
            parent = _parent_id(story)
            if parent is not None and parent in dropped_ids:
                dropped.append({"model_story_id": story_id,
                                "reason": f"parent {parent} is not anchored inside this "
                                          f"window",
                                "story": story})
                dropped_ids.add(story_id)
                changed = True

    accepted_entries = [(story_id, story) for story_id, story in entries
                        if story_id not in dropped_ids]
    accepted_ids = {story_id for story_id, _story in accepted_entries}
    # A fabricated parent name is a breach wherever the child ended up, accepted or
    # dropped: the row or the drop record would both cite a story no part of this answer
    # claims. Only a parent that IS in the answer can legitimately be missing from this
    # window (propagated drop above). Independent verification of 9f39903, finding 4:
    # this check used to run for accepted entries only, so a dropped child slipped past
    # it pointing at a name nobody answered.
    entry_ids = {story_id for story_id, _story in entries}
    for story_id, story in entries:
        parent = _parent_id(story)
        if parent is not None and parent not in entry_ids:
            raise ValidationError(stage, [f"{story_id}: parent_id {parent!r} is not a story "
                                          f"of this answer"])
    for story_id, story in accepted_entries:
        parent = _parent_id(story)
        if parent is None:
            continue
        if parent not in accepted_ids:
            # A parent from another window cannot be joined here either: story ids are
            # per-request, and inventing a cross-window link would be a claim this
            # answer cannot support. A name that appears nowhere in the answer is the
            # same breach — the row would point at a story nobody claimed.
            raise ValidationError(stage, [f"{story_id}: parent_id {parent!r} is not a story "
                                          f"of this answer"])
        ancestors: set[str] = set()
        cursor: str | None = parent
        while cursor is not None:
            if cursor in ancestors or cursor == story_id:
                raise ValidationError(stage, [f"{story_id}: parent_id chain contains a "
                                              f"cycle at {cursor!r}"])
            ancestors.add(cursor)
            cursor = _parent_of(accepted_entries, cursor)

    rows = [_row(story, request, story_id) for story_id, _story in accepted_entries]
    return rows, dropped


def _parent_id(story: dict[str, Any]) -> str | None:
    parent = story.get("parent_id")
    if parent is None or (isinstance(parent, str) and not parent.strip()):
        return None
    return str(parent).strip()


def _parent_of(entries: Sequence[tuple[str, dict[str, Any]]], story_id: str) -> str | None:
    for current_id, story in entries:
        if current_id == story_id:
            parent = story.get("parent_id")
            if parent is None or (isinstance(parent, str) and not parent.strip()):
                return None
            return str(parent).strip()
    return None


def _number(value: Any) -> float | None:
    """A JSON number as a float, or None for anything else.

    ``True`` is excluded on purpose: ``isinstance(True, int)`` is true in python, and a
    model that answered ``true`` for a confidence is giving a non-number, not a 1.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _row(story: dict[str, Any], request: StoriesRequest, model_id: str) -> dict[str, Any]:
    """A validated story as one table row, with window-namespaced ids.

    ``model_id`` is the entry key validation built — whitespace-stripped — not the raw
    ``story_id`` field. The independent verifier (finding 2) showed why: the validator
    matched and de-duplicated stripped ids while this function used the padded original,
    so a story sent as ``" s1 "`` published ``w0- s1 `` while its child cited ``w0-s1``:
    an orphan the validator had guaranteed could not exist. Normalisation that lives only
    in the checker is not a guarantee; it has to reach the row.
    """
    parent = story.get("parent_id")
    parent_id = None if parent is None or not str(parent).strip() else \
        f"{request.id_prefix}-{str(parent).strip()}"
    evidence = [str(item) for item in story.get("evidence_segment_ids") or []]
    return {
        "schema_version": "1.0",
        "video_id": request.video_id,
        "story_id": f"{request.id_prefix}-{model_id}",
        "parent_id": parent_id,
        "start_time": float(story["start_time"]),
        "end_time": float(story["end_time"]),
        "title": str(story.get("title") or "").strip(),
        "why_it_is_a_story": str(story.get("why_it_is_a_story") or "").strip(),
        # JSON array as a string: the flat-table rule every other artifact follows.
        "evidence_segment_ids": json.dumps(evidence, ensure_ascii=False),
        "confidence": float(story["confidence"]),
        "window_index": request.window_index,
        "model": request.model,
        "prompt_version": request.prompt_version,
        "request_key": request.key(),
    }


@dataclass
class StoriesVerdict:
    """One window's answer: the raw text, the parsed payload, and the split it produced."""

    content: str
    payload: dict[str, Any]
    stories: list[dict[str, Any]]
    dropped: list[dict[str, Any]]


# ------------------------------------------------------------------ clients


class StoriesClient:
    """OpenAI-compatible chat client for story detection, with retry and usage.

    The httpx/retry/backoff/usage shape is the one ``TranslationClient`` established,
    kept deliberately close to it: same endpoint, same failure modes (429, 5xx, a timeout,
    a non-JSON body, a chatty answer), and an operator who reads one reads the other.
    """

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
        prompt = request.prompt()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            started = time.time()
            try:
                payload = self._post(prompt, request)
                self.timings_ms.append(round((time.time() - started) * 1000, 1))
                self._record_usage(payload)
                raw_text = extract_message_content(payload)
                parsed = parse_stories_response(raw_text)
                stories, dropped = validate_stories(parsed, request)
                return StoriesVerdict(content=raw_text, payload=parsed, stories=stories,
                                      dropped=dropped)
            except StoriesTransportError as exc:
                last_error = exc
            except (ValidationError, ValueError) as exc:
                # A malformed answer is the model's mistake, not the network's, but
                # resampling usually fixes it — up to a point.
                last_error = exc
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

    def _post(self, prompt: str, request: StoriesRequest) -> dict[str, Any]:
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

    It answers the way the probe's endpoint answered on the two clips with no story: one
    window with a real boundary and full evidence when there is something to span, an
    empty list *with a stated reason* when a single segment cannot hold an arc.
    """

    EMPTY_REASON = ("mock provider: a single segment cannot hold a beginning and an end, "
                    "so no narrative window is claimed")

    def __init__(self, model: str = "mock") -> None:
        self.model = model
        self.request_count = 0
        self.timings_ms: list[float] = []
        self.usage: dict[str, int] = {}

    def detect(self, request: StoriesRequest) -> StoriesVerdict:
        self.request_count += 1
        self.timings_ms.append(0.0)
        rows = request.requested
        if len(rows) >= 2:
            payload: dict[str, Any] = {"stories": [{
                "story_id": "s1",
                "parent_id": None,
                "start_time": float(rows[0]["start_time"]),
                "end_time": float(rows[-1]["end_time"]),
                "title": "Mock narrative window",
                "why_it_is_a_story": "mock provider: these segments run beginning to end.",
                "evidence_segment_ids": [str(row["segment_id"]) for row in rows],
                "confidence": 0.5,
            }], "no_story_reason": None}
        else:
            payload = {"stories": [], "no_story_reason": self.EMPTY_REASON}
        content = json.dumps(payload, ensure_ascii=False)
        stories, dropped = validate_stories(payload, request)
        return StoriesVerdict(content=content, payload=payload, stories=stories,
                              dropped=dropped)


# ------------------------------------------------------------------ stage


class StoriesStage(Stage):
    """Ask the translation endpoint where the stories in this transcript are."""

    name = "stories"
    # Hard input: the transcript with speakers applied. The speaker turns are read
    # *softly* (see `_turn_rows`) because they are context for the prompt, not the thing
    # being answered about, and a corpus that ran no diarizer still has stories.
    inputs = ("speech_segments",)
    outputs = ("stories", "stories_raw")
    config_keys = ("stories",)

    # ------------------------------------------------------------------ fingerprint

    def config_fingerprint(self, ctx: StageContext) -> dict[str, Any]:
        """Config + the digest of the transcript this reads.

        The ``segments_digest`` is what makes a re-transcription invalidate the answers:
        config alone cannot see that the words changed. ``_python_code_sha256`` closes the
        hole from the other side — the parser, validator and prompt live in this process
        with no worker file to digest, so a fix to the rejection rules would otherwise
        leave every cached window looking reusable (the defect class
        ``python_source_digest`` documents).
        """
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
            "extra_body": cfg.extra_body,
            "segments_digest": self._segments_digest(ctx),
            "_python_code_sha256": python_source_digest(stories_stage),
        }

    @staticmethod
    def _segments_digest(ctx: StageContext) -> str | None:
        from ..stages.metadata import sha256_of

        try:
            path = ctx.input("speech_segments")
        except StageError:
            return None
        return sha256_of(path)

    # ------------------------------------------------------------------ enablement

    def enabled(self, ctx: StageContext) -> tuple[bool, str]:
        """An unconfigured endpoint is a skip with the keys to set, never a crash.

        The same shape as translation's: this stage shares an endpoint with it, and a
        corpus configured for transcription only should report "not configured" rather
        than fail every video on an empty URL.
        """
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

    # ------------------------------------------------------------------ execution

    def build_requests(self, ctx: StageContext) -> list[StoriesRequest]:
        cfg = ctx.config.stories
        return build_requests(
            self._segment_rows(ctx),
            video_id=ctx.video_id,
            max_segments_per_request=cfg.max_segments_per_request,
            context_segments=cfg.context_segments,
            prompt_version=cfg.prompt_version,
            model=cfg.model,
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
            turns=self._turn_rows(ctx),
            endpoint=cfg.base_url,
        )

    @staticmethod
    def _segment_rows(ctx: StageContext) -> list[dict[str, Any]]:
        path = ctx.artifact("speech_segments")
        if not path.is_file():
            return []
        return read_table(path).to_pylist()

    @staticmethod
    def _turn_rows(ctx: StageContext) -> list[dict[str, Any]]:
        """Speaker turns for the prompt, if a diarizer produced any.

        Read off disk rather than declared as a hard input, exactly as
        ``SpacySourceStage`` reads ``whisperx_raw``: ``ctx.input()`` raises on a missing
        artifact, and "this corpus ran no diarizer" is a supported state for a stage whose
        subject is the words.
        """
        from ..schemas import table_columns

        path = ctx.artifact("speaker_turns")
        if not path.is_file():
            return []
        columns = set(table_columns(path))
        if not {"turn_id", "speaker_id", "start_time", "end_time"} <= columns:
            return []
        return read_table(path).to_pylist()

    def execute(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.config.stories
        requests = self.build_requests(ctx)
        client = self._client(ctx)
        raw_dir = ctx.artifact("stories_raw")
        cache_dir = raw_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []
        reused = 0
        empty_windows = 0
        for request in requests:
            index = request.window_index
            verdict = self._load_cached(cache_dir / f"{request.key()}.json", request) if cfg.cache else None
            if verdict is not None:
                reused += 1
                ctx.log(f"window {index + 1}/{len(requests)} reused from cache")
                # A reused window still owes the dataset its raw file: the cache may have
                # survived a hand-deleted stories/raw/window_*.json.
                self._ensure_raw(raw_dir, request, verdict)
            else:
                ctx.log(f"detecting stories in window {index + 1}/{len(requests)} "
                        f"({len(request.requested_ids)} segments)")
                verdict = client.detect(request)
                # Raw first, byte-identical: a normalised table with no original next to
                # it cannot be re-validated later (repo rule), and this file is also what
                # a human reads to see what the model actually said. Written
                # unconditionally rather than only when absent: this is a fresh answer,
                # and at temperature > 0 the endpoint can answer the same request
                # differently, so keeping a previous response here would file the wrong
                # evidence beside these rows. atomic_write_text writes through a temp file
                # + os.replace, so an interrupted run cannot leave half a JSON body that a
                # later reader takes for the answer.
                atomic_write_text(raw_dir / f"window_{index}_{request.key()}.json",
                                  verdict.content)
                atomic_write_json(cache_dir / f"{request.key()}.json", {
                    "schema_version": "1.0",
                    "request_key": request.key(),
                    "video_id": request.video_id,
                    "window_index": index,
                    "requested_ids": request.requested_ids,
                    "context_ids": [str(row["segment_id"]) for row in request.context],
                    "model": request.model,
                    "temperature": request.temperature,
                    "prompt_version": request.prompt_version,
                    # The model's own payload, unprefixed: reuse re-validates it against
                    # the current window rather than trusting what is on disk.
                    "payload": verdict.payload,
                    # And its exact response text, so a reuse can restore a deleted raw
                    # file byte-identically instead of writing a re-serialisation that
                    # only looks like what the endpoint sent.
                    "content": verdict.content,
                })
            rows.extend(verdict.stories)
            if not verdict.stories:
                empty_windows += 1
                ctx.log(f"window {index + 1}/{len(requests)}: no narrative window "
                        f"(reason: {str(verdict.payload.get('no_story_reason') or 'unstated')[:160]})")
            for item in verdict.dropped:
                ctx.log(f"window {index + 1}: dropped story {item['model_story_id']} — "
                        f"{item['reason']} (evidence not entirely inside this window)", 30)
            dropped.extend({"window_index": index, **item} for item in verdict.dropped)

        write_table(ctx.artifact("stories"),
                    pa.Table.from_pylist(rows, schema=STORIES_SCHEMA), STORIES_SCHEMA,
                    extra_metadata={"video_id": ctx.video_id, "model": cfg.model,
                                    "prompt_version": cfg.prompt_version,
                                    "windows": len(requests)})
        atomic_write_json(raw_dir / "stories_summary.json", {
            "schema_version": "1.0",
            "video_id": ctx.video_id,
            "provider": cfg.provider,
            "model": cfg.model,
            "base_url": cfg.base_url,
            "temperature": cfg.temperature,
            "prompt_version": cfg.prompt_version,
            "max_segments_per_request": cfg.max_segments_per_request,
            "context_segments": cfg.context_segments,
            "windows": len(requests),
            "empty_windows": empty_windows,
            "batches_reused": reused,
            "requests_made": getattr(client, "request_count", 0),
            "request_timings_ms": getattr(client, "timings_ms", [])[-200:],
            "token_usage": getattr(client, "usage", {}),
            "stories": len(rows),
            "dropped_outside_window": len(dropped),
            "dropped_detail": dropped[:50],
        })
        ctx.scratch["stories"] = {"stories": len(rows), "windows": len(requests),
                                  "empty_windows": empty_windows,
                                  "dropped_outside_window": len(dropped), "model": cfg.model}
        ctx.log(f"stories: {len(rows)} story window(s) across {len(requests)} window(s), "
                f"{empty_windows} empty, {len(dropped)} dropped outside a window, "
                f"{reused}/{len(requests)} reused")
        return {"tool_version": cfg.provider, "model_version": cfg.model,
                "extra": {"stories_total": len(rows), "windows": len(requests),
                          "empty_window_count": empty_windows,
                          "dropped_outside_window": len(dropped),
                          "batches_reused": reused}}

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
        """Guarantee the raw window file exists for every row the table will carry.

        Normalised Parquet without its raw beside it cannot be re-validated later, which
        the repository treats as a hard rule rather than a preference. The normal path
        writes the file before the cache entry; this covers the reused path, where a raw
        file deleted by hand would otherwise leave a table whose evidence is gone and
        whose cache says nothing about it. It writes the stored response text, so the
        restored file is the endpoint's bytes, not this process's re-serialisation.
        """
        path = raw_dir / f"window_{request.window_index}_{request.key()}.json"
        if path.exists() or not verdict.content:
            return
        atomic_write_text(path, verdict.content)

    @staticmethod
    def _load_cached(path: Path, request: StoriesRequest) -> StoriesVerdict | None:
        """A cached window, re-validated against *this* window's segments.

        The re-validation is the point. A cache file whose evidence ids or boundaries no
        longer fit the transcript is a file about a clip that no longer exists, and
        handing its rows straight to the table would publish them with no check at all.
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
        if not isinstance(content, str):
            # A cache entry written before the raw text was stored. Reuse is still
            # correct — the payload re-validates — but there are no original bytes to
            # hand back, so `_ensure_raw` has nothing to restore from.
            content = ""
        try:
            stories, dropped = validate_stories(stored, request)
        except (ValidationError, ValueError):
            return None
        if content:
            # An entry whose two halves disagree describes nothing honestly: the payload
            # would pass while `_ensure_raw` restores a response text citing evidence the
            # table does not carry (independent verification of 9f39903, finding 3). The
            # content must re-parse to the payload it claims to be the original bytes of,
            # or the entry is refetched instead of half-trusted.
            try:
                if parse_stories_response(content) != stored:
                    return None
            except (ValueError, ValidationError):
                return None
        # The raw text on disk is the payload re-serialised, not the model's own bytes:
        # the byte-identical copy lives in window_<i>_<key>.json and is never rewritten.
        return StoriesVerdict(content=content or json.dumps(stored, ensure_ascii=False),
                              payload=stored, stories=stories, dropped=dropped)

    # ------------------------------------------------------------------ validation

    def validate(self, ctx: StageContext) -> dict[str, Any]:
        """The table parses, its rows are self-consistent, and its boundaries are still real.

        Returns the three numbers ``status.json`` shows, read from the artifacts rather
        than from memory: the row count of the table, the window count of the raw summary
        (an empty window has no rows to count), and the difference between them.
        """
        path = ctx.artifact("stories")
        if not path.is_file():
            raise ValidationError(self.name, ["stories/stories.parquet missing"])
        rows = read_table(path).to_pylist()

        summary_path = ctx.artifact("stories_raw") / "stories_summary.json"
        if not summary_path.is_file():
            raise ValidationError(self.name,
                                  ["stories/raw/stories_summary.json missing: the table has "
                                   "no record of how many windows were asked about"])
        summary = read_json(summary_path)
        windows = int(summary.get("windows") or 0)

        problems: list[str] = []
        ids = [str(row["story_id"]) for row in rows]
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        if duplicates:
            problems.append(f"duplicate story_id values: {duplicates[:5]}")
        known = set(ids)
        for row in rows:
            parent = row.get("parent_id")
            if parent is not None and str(parent) not in known:
                problems.append(f"story {row['story_id']} names parent {parent}, which is "
                                f"not a story of this video")
        boundaries = self._boundaries(ctx)
        if boundaries is None:
            problems.append("speech/segments.parquet is unreadable or missing, so no "
                            "story boundary in this table can be checked")
        else:
            starts, ends = boundaries
            for row in rows:
                start = round(float(row["start_time"]), BOUNDARY_PRECISION)
                end = round(float(row["end_time"]), BOUNDARY_PRECISION)
                if start not in starts or end not in ends:
                    problems.append(f"story {row['story_id']} window {row['start_time']}.."
                                    f"{row['end_time']} is no longer a real segment "
                                    f"boundary of the transcript")
        if windows and windows < len({int(row["window_index"]) for row in rows}):
            problems.append("the table names more windows than the raw summary recorded")
        if problems:
            raise ValidationError(self.name, problems[:20])
        return {"stories": len(rows), "windows": windows,
                "empty_windows": max(windows - len({int(row["window_index"]) for row in rows}), 0)}

    @staticmethod
    def _boundaries(ctx: StageContext) -> tuple[set[float], set[float]] | None:
        """Transcript boundaries the table's times are checked against, or None."""
        path = ctx.artifact("speech_segments")
        if not path.is_file():
            return None
        try:
            rows = read_table(path, columns=["start_time", "end_time"]).to_pylist()
        except Exception:  # noqa: BLE001 - reported as "cannot check", not as a crash
            return None
        return ({round(float(row["start_time"]), BOUNDARY_PRECISION) for row in rows},
                {round(float(row["end_time"]), BOUNDARY_PRECISION) for row in rows})
