"""`stories`: storytelling events, stories inside them, and uncertain spans (prompt v2).

Four things are tested, and each has a different failure it exists to catch.

**Boundary times are computed, never typed.** The model cites a `segment_id` and copies a
verbatim quote; this module maps the quote to word times. `TestBoundaryTimes` covers the
word-precision path, the segment-precision fallback, and the normalisations that are
allowed (case, whitespace, surrounding punctuation) — a quote that paraphrases is a breach,
because an approximate anchor is not evidence.

**The rejection path** (`TestValidation`). An invented segment id, a quote that is not in
the cited segment, an enum value outside its vocabulary, a duplicate id, a story whose
event is not in the answer, `story_ids` that disagree with the stories, a story outside its
event's span, an end before its start: each is a breach, the whole answer is refused, and
the batch is retried. Empty arrays are a valid answer.

**Speaker labels are attributed** (`TestSpeakerSources`): Nemotron's turns overlap, so a
word no single speaker dominates is `unknown`; a missing Nemotron table is a supported state
that labels everything `unknown` and records requested→used, and never falls back to
the max-overlap `speaker_id` on segments/words (Nemotron is the only diarizer; the
`pyannote` source was removed on 2026-11-26).

**The stage contract** (`TestStageExecution` and below): whole-transcript default,
windowed drop-and-count when configured, the digest cache and its re-validation, raw
preserved byte-identically, three tables written (including zero-row ones), and the
fingerprint noticing a change in the words or the diarization.

The endpoint is never contacted: the mock provider and a fake client do the work, and the
retry/transport code is exercised against a real local HTTP server as in
`test_translation.py`.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from multimodal_pipeline.artifacts import ARTIFACT_LAYOUT, STAGE_LOG_NAMES
from multimodal_pipeline.exceptions import StageError, ValidationError
from multimodal_pipeline.schemas import (
    SEGMENTS_SCHEMA,
    SPEAKER_TURNS_NEMOTRON_SCHEMA,
    STORIES_SCHEMA,
    STORYTELLING_EVENTS_SCHEMA,
    TABLE_SCHEMAS,
    UNCERTAIN_SPANS_SCHEMA,
    WORDS_SCHEMA,
    read_table,
    write_table,
)
from multimodal_pipeline.stages import stories as stories_module
from multimodal_pipeline.stages.base import STAGE_DEPENDENCIES, STAGE_ORDER
from multimodal_pipeline.stages.stories import (
    PROMPTS,
    MockStoriesClient,
    StoriesClient,
    StoriesStage,
    StoriesTransportError,
    align_token_times,
    _drop_counts,
    build_requests,
    feedback_message,
    locate_quote,
    parse_stories_response,
    prepare_segments,
    quote_occurrences,
    validate_response,
)


# ------------------------------------------------------------------ builders


def segment(index: int, *, text: str | None = None, speaker: str = "SPEAKER_00",
            start: float | None = None, end: float | None = None) -> dict[str, Any]:
    """One transcript row in the shape `speech/segments.parquet` really has."""
    start = float(index) * 3.0 if start is None else start
    end = start + 2.5 if end is None else end
    return {
        "schema_version": "1.0",
        "video_id": "conversation_001",
        "segment_id": f"seg{index:06d}",
        "start_time": start,
        "end_time": end,
        "duration": round(end - start, 6),
        "language": "en",
        "speaker_id": speaker,
        "text": text or f"line number {index}",
        "confidence": 0.9,
        "speaker_overlap_seconds": 0.0,
        "speaker_overlap_ratio": 0.0,
        "speaker_assignment_method": "pyannote",
    }


SEGMENTS = [
    segment(1, text="I was fiddling with my wedding ring"),
    segment(2, text="it rolled down the side of the sofa", speaker="SPEAKER_01"),
    segment(3, text="that was embarrassing"),
]

#: Nemotron turns that cover SEGMENTS one speaker at a time (no overlap).
NEMOTRON_TURNS = [
    {"schema_version": "1.0", "video_id": "conversation_001", "turn_id": f"turn{i:06d}",
     "speaker_id": spk, "start_time": a, "end_time": b, "duration": b - a,
     "diarization_type": "inclusive", "overlap_s": 0.0}
    for i, (spk, a, b) in enumerate([("speaker_0", 3.0, 5.5), ("speaker_1", 6.0, 8.5),
                                     ("speaker_0", 9.0, 11.5)], start=1)]


def word_rows(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Word rows covering each segment's text evenly across its own interval.

    Timings are spread over the whole segment so a whole-segment quote resolves to the
    segment's own boundaries: that makes a word-precision result checkable against numbers
    the test wrote.
    """
    rows: list[dict[str, Any]] = []
    for row in segments:
        words = str(row["text"]).split()
        step = (float(row["end_time"]) - float(row["start_time"])) / max(len(words), 1)
        for position, word in enumerate(words):
            rows.append({
                "schema_version": "1.0",
                "video_id": row["video_id"],
                "segment_id": row["segment_id"],
                "word_id": f"{row['segment_id']}-w{position:05d}",
                "start_time": float(row["start_time"]) + position * step,
                "end_time": float(row["start_time"]) + (position + 1) * step,
                "duration": step,
                "speaker_id": row["speaker_id"],
                "word": word,
                "confidence": 0.9,
                "alignment_status": "aligned",
                "character_start": None,
                "character_end": None,
            })
    return rows


WORDS = word_rows(SEGMENTS)


def prepared(segments: list[dict[str, Any]] = SEGMENTS, words: Any = None,
             turns: list[dict[str, Any]] | None = None,
             speaker_source: str = "none", **kwargs: Any) -> list[dict[str, Any]]:
    return prepare_segments(segments, WORDS if words is None else words, turns or [],
                            speaker_source=speaker_source, **kwargs)


def request_for(segments: list[dict[str, Any]] = SEGMENTS, *, index: int = 0,
                rows: list[dict[str, Any]] | None = None,
                words: Any = None,
                turns: list[dict[str, Any]] | None = None,
                speaker_source: str = "none", **cfg: Any):
    """The request a stage would send for these segments."""
    rows = rows if rows is not None else prepared(segments, words=words, turns=turns,
                                                 speaker_source=speaker_source)
    options: dict[str, Any] = {"max_segments_per_request": None, "context_segments": 5,
                              "prompt_version": "v2", "model": "chat", "temperature": 0.0,
                              "max_output_tokens": None, "speaker_source": speaker_source}
    options.update(cfg)
    return build_requests(rows, video_id="conversation_001", **options)[index]


def label(request, segment_id: str) -> str:
    """The one non-unknown label shown for a segment, or "unknown".

    ``unknown`` is also the answer for a segment outside this window: a windowed test
    deliberately cites one to produce a drop, and "unknown" is always an allowed speaker.
    """
    rows = [row for row in request.requested if row["segment_id"] == segment_id]
    if not rows:
        return "unknown"
    labels = [item for item in rows[0]["speaker_labels"] if item != "unknown"]
    return labels[0] if labels else "unknown"


def event_payload(request, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event_id": "SE1",
        "start": {"segment_id": "seg000001",
                  "transcript_quote": SEGMENTS[0]["text"],
                  "speaker": label(request, "seg000001"),
                  "boundary_type": "pre_telling",
                  "reason": "the gist comes first"},
        "end": {"segment_id": "seg000003", "transcript_quote": SEGMENTS[2]["text"],
                "speaker": label(request, "seg000003"),
                "boundary_type": "coda",
                "reason": "a coda closes it"},
        "story_ids": ["S1"],
        "tellers": [label(request, "seg000001")],
        "interruptions_or_resumptions": [],
        "confidence": "high",
        "notes": "",
    }
    payload.update(overrides)
    return payload


def story_payload(request, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "story_id": "S1",
        "event_id": "SE1",
        "start": {"segment_id": "seg000001", "transcript_quote": "with my wedding ring",
                  "speaker": label(request, "seg000001"),
                  "reason": "the speaker enters the story world"},
        "end": {"segment_id": "seg000002", "transcript_quote": SEGMENTS[1]["text"],
                "speaker": label(request, "seg000002"),
                "reason": "the last represented event"},
        "tellers": [label(request, "seg000001"), label(request, "seg000002")],
        "summary": "a wedding ring rolled down the side of the sofa",
        "story_evidence": ["a wedding ring rolling away is one represented event",
                           "the speaker's embarrassment evaluates that experience"],
        "evidence_segment_ids": ["seg000001", "seg000002"],
        "confidence": "high",
        "notes": "",
    }
    payload.update(overrides)
    return payload


def answer(*, events: list[dict[str, Any]] | None = None,
           stories: list[dict[str, Any]] | None = None,
           uncertain: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"storytelling_events": events or [], "stories": stories or [],
            "uncertain_spans": uncertain or []}


def honest_answer(request) -> dict[str, Any]:
    return answer(events=[event_payload(request)], stories=[story_payload(request)])


def unknown_speaker_answer() -> dict[str, Any]:
    """A valid answer whose speakers are all "unknown", so any label set accepts it."""
    request = request_for(speaker_source="none")
    return answer(events=[event_payload(request)], stories=[story_payload(request)])


def envelope(payload: dict[str, Any]) -> str:
    return json.dumps(payload)


class FakeClient:
    """A client-shaped object handing back scripted content, counting the calls."""

    def __init__(self, responses: list[Any], *, model: str = "chat") -> None:
        self.responses = list(responses)
        self.model = model
        self.request_count = 0
        self.timings_ms: list[float] = []
        self.usage: dict[str, int] = {"prompt_tokens": 10}
        self.prompts: list[str] = []

    def detect(self, request):
        from multimodal_pipeline.stages.stories import StoriesVerdict

        self.request_count += 1
        self.timings_ms.append(0.0)
        self.prompts.append(request.prompt())
        outcome = self.responses.pop(0) if self.responses else self.responses
        if isinstance(outcome, Exception):
            raise outcome
        payload = parse_stories_response(outcome)
        return StoriesVerdict(content=outcome, payload=payload,
                              answer=validate_response(payload, request))


def seed_segments(context, rows: list[dict[str, Any]]) -> None:
    write_table(context.artifact("speech_segments"),
                pa.Table.from_pylist(rows, schema=SEGMENTS_SCHEMA), SEGMENTS_SCHEMA)


def seed_words(context, rows: list[dict[str, Any]]) -> None:
    write_table(context.artifact("speech_words"),
                pa.Table.from_pylist(rows, schema=WORDS_SCHEMA), WORDS_SCHEMA)


def seed_nemotron(context, rows: list[dict[str, Any]]) -> None:
    write_table(context.artifact("speaker_turns_nemotron"),
                pa.Table.from_pylist(rows, schema=SPEAKER_TURNS_NEMOTRON_SCHEMA),
                SPEAKER_TURNS_NEMOTRON_SCHEMA)


def turn(speaker: str, start: float, end: float, index: int = 1) -> dict[str, Any]:
    return {"schema_version": "1.0", "video_id": "conversation_001",
            "turn_id": f"turn{index:06d}", "speaker_id": speaker,
            "start_time": start, "end_time": end, "duration": end - start,
            "diarization_type": "inclusive", "overlap_s": 0.0}


def use_stage_config(context, **stories_cfg: Any) -> None:
    """A stories section that runs offline by default."""
    values: dict[str, Any] = {"provider": "mock", "speaker_source": "none"}
    values.update(stories_cfg)
    for key, value in values.items():
        setattr(context.config.stories, key, value)


def run_stage(context, client=None) -> dict[str, Any]:
    stage = StoriesStage()
    if client is not None:
        stage._client = lambda _ctx: client  # type: ignore[method-assign]
    outcome = stage.run(context)
    assert outcome.status == "completed", outcome.message
    return outcome.detail["provenance"]["extra"]


def rows_of(context, artifact: str) -> list[dict[str, Any]]:
    return read_table(context.artifact(artifact)).to_pylist()


def summary(context) -> dict[str, Any]:
    return json.loads((context.artifact("stories_raw") / "stories_summary.json")
                      .read_text(encoding="utf-8"))


# ------------------------------------------------------------------ local endpoint


class EndpointState:
    def __init__(self, responses: list[tuple[int, Any]]) -> None:
        self.responses = list(responses)
        self.bodies: list[dict] = []
        self.headers: list[dict] = []

    def handler(self) -> type[BaseHTTPRequestHandler]:
        state = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("content-length", 0))
                state.bodies.append(json.loads(self.rfile.read(length) or b"{}"))
                headers = dict(self.headers)
                headers["_path"] = self.path
                state.headers.append(headers)
                status, payload = (state.responses.pop(0) if state.responses
                                   else (200, ok_body("{}")))
                raw = (json.dumps(payload) if not isinstance(payload, str)
                       else payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args) -> None:  # silence the test server
                pass

        return Handler


def ok_body(content: str, **usage: int) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}],
            **({"usage": usage} if usage else {})}


@pytest.fixture
def endpoint() -> Any:
    servers: list[HTTPServer] = []

    def start(responses: list[tuple[int, Any]]) -> tuple[HTTPServer, EndpointState]:
        state = EndpointState(responses)
        server = HTTPServer(("127.0.0.1", 0), state.handler())
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server, state

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def stories_client(server: HTTPServer, **kwargs: Any) -> StoriesClient:
    options: dict[str, Any] = {"base_url": f"http://127.0.0.1:{server.server_port}/v1",
                               "api_key": "sk-test-123", "model": "chat",
                               "max_retries": 1, "backoff_base_seconds": 0.0,
                               "sleep": lambda _seconds: None}
    options.update(kwargs)
    return StoriesClient(**options)


def log_recorder(context) -> list[str]:
    messages: list[str] = []

    def _log(message: str, level: int = 20) -> None:
        messages.append(str(message))

    context.log = _log
    return messages


# ------------------------------------------------------------------ prompt

#: The expert's own annotation prompt, committed verbatim. The prompt below quotes it, so
#: the document is the source the quotes are checked against; `PROMPTS["v2"]` embedded a
#: paraphrase once, and a paraphrase of a domain expert's criteria is a different criterion.
GUIDELINES = (Path(__file__).resolve().parents[2] / "docs"
              / "stories-annotation-guidelines.md")

#: Her sections, split on the ``##`` heading that introduces each one. Three of her last
#: sections are not quoted: her input format (a WhisperX JSON object) and her output
#: contract (times in seconds) describe *her* pipeline, and her prompt's trailing
#: ``## Transcript`` placeholder is where our rendered transcript goes instead.
ADAPTED_SECTIONS: tuple[str, ...] = ("Using WhisperX JSON", "Output", "Transcript")

#: The single sentence of hers the prompt rewrites, and what it becomes. Her rule points at
#: timestamps the model would have to read; this pipeline makes the model report no time at
#: all, which is the same intent (never invent precision) pushed to where nothing is typed.
HER_TIMESTAMP_RULE = ("- Use the transcript\u2019s timestamps. Do not invent timestamp "
                      "precision that the segments do not support.")
PIPELINE_TIMESTAMP_RULE = ("- Do not output times: the pipeline computes them from the "
                           "segment ids and quotes you cite, so there is no timestamp "
                           "precision for you to invent.")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def guidelines_sections() -> dict[str, str]:
    """Her document as ``{section name: heading + body}``, plus ``""`` for the goal.

    Read from the committed file rather than restated in the test so that an edit to either
    side is a failure: the check is that the *same* words are in both places, not that two
    hand-maintained copies agree.
    """
    text = GUIDELINES.read_text(encoding="utf-8")
    head, *blocks = text.split("\n## ")
    sections: dict[str, str] = {"": head.split("\n\n")[-1]}
    for block in blocks:
        name = block.partition("\n")[0]
        sections[name.strip()] = "## " + block.rstrip("\n")
    return sections


def quoted_sections() -> list[str]:
    """Every section the prompt must quote: hers, minus the adapted ones, plus the goal."""
    return [name for name in guidelines_sections() if name not in ADAPTED_SECTIONS]


def test_the_guidelines_document_is_the_experts_own_text() -> None:
    """The committed source still starts with her goal sentence, after a short header."""
    text = GUIDELINES.read_text(encoding="utf-8")
    assert text.startswith("# Stories annotation guidelines\n")
    assert "Your goal is to mark the boundaries of each storytelling event" in text
    assert "2026-11-26" in text


class TestPromptQuotesTheExpertVerbatim:
    """Every section of her prompt except the two adapted ones is in the prompt word for word.

    Whitespace is normalised and nothing else: no rewording, no abbreviation, no reordered
    bullets. A future edit that "tidies" her prose while looking at the prompt — or the
    reverse — fails here instead of shipping a criterion the expert never wrote.
    """

    def test_the_adapted_sections_are_exactly_the_ones_declared(self) -> None:
        sections = guidelines_sections()
        assert set(ADAPTED_SECTIONS) <= set(sections)
        assert "" in sections
        # Her goal paragraph is a section like the rest, and is quoted like the rest.
        assert "" in quoted_sections()

    @pytest.mark.parametrize("name", quoted_sections(), ids=lambda name: name or "goal")
    def test_the_section_is_quoted_word_for_word(self, name: str) -> None:
        expected = _norm(guidelines_sections()[name])
        if name == "Annotation rules":
            # One bullet of hers is rewritten, and only that one.
            assert _norm(HER_TIMESTAMP_RULE) in expected
            expected = expected.replace(_norm(HER_TIMESTAMP_RULE),
                                        _norm(PIPELINE_TIMESTAMP_RULE))
        assert expected in _norm(PROMPTS["v2"])

    def test_only_one_sentence_of_hers_is_rewritten(self) -> None:
        """Her timestamps rule is gone, its replacement is there, the other bullets stay."""
        shipped = _norm(PROMPTS["v2"])
        assert _norm(HER_TIMESTAMP_RULE) not in shipped
        assert _norm(PIPELINE_TIMESTAMP_RULE) in shipped
        rules = guidelines_sections()["Annotation rules"]
        bullets = [line for line in rules.splitlines() if line.startswith("- ")]
        assert HER_TIMESTAMP_RULE in bullets
        for bullet in bullets:
            if bullet == HER_TIMESTAMP_RULE:
                continue
            assert _norm(bullet) in shipped, bullet

    def test_her_story_evidence_example_survives_in_the_contract(self) -> None:
        prompt = PROMPTS["v2"]
        for item in ("Represented event(s)", "Causal or experiential connection",
                     "Embodiment, intentionality, or evaluation, if relevant"):
            assert item in prompt, item

    def test_her_input_limitations_survive_the_adapted_input_section(self) -> None:
        """Her WhisperX section is replaced, but not every bullet in it was about WhisperX."""
        prompt = _norm(PROMPTS["v2"])
        for bullet in ("Do not invent words or silently correct transcription errors",
                       "may combine multiple conversational turns",
                       "Do not infer pauses, overlap, intonation, gaze, gesture, or other "
                       "nonverbal cues from text alone",
                       "do not invent them"):
            assert bullet in prompt, bullet


class TestPrompt:
    def test_v2_is_the_only_prompt_version(self) -> None:
        assert list(PROMPTS) == ["v2"]

    def test_the_goal_sentence_is_the_domain_experts(self) -> None:
        prompt = _norm(PROMPTS["v2"])
        assert ("Your goal is to mark the boundaries of each storytelling event and, "
                "separately, the boundaries of each story told within it.") in prompt
        # Her words, including her typographic quotes: the straight-quote version was a
        # paraphrase, and `TestPromptQuotesTheExpertVerbatim` is what keeps it honest.
        assert ("A story is the \u201cwhat\u201d being represented; a storytelling event is "
                "the conversational activity of launching and telling it.") in prompt

    def test_the_working_criteria_and_boundary_guidance_are_present(self) -> None:
        lowered = _norm(PROMPTS["v2"]).lower()
        for phrase in ("foregrounds experience", "not automatically a story",
                       "pre-telling", "coda", "listener_acknowledgment", "activity_shift",
                       "interruptions", "do not invent words", "uncertain"):
            assert phrase in lowered, phrase

    def test_the_emptiness_rule_stays_prominent(self) -> None:
        prompt = _norm(PROMPTS["v2"])
        assert "An empty answer is the expected and correct answer" in prompt
        assert "empty arrays" in prompt

    def test_the_input_format_is_ours_not_whisperx_json(self) -> None:
        prompt = PROMPTS["v2"]
        assert "WhisperX" not in prompt
        assert "segment_id=" in prompt
        assert "[speaker label]" in prompt
        assert "Do not output times" in prompt
        assert "transcript_quote" in prompt
        assert ("Do not infer pauses, overlap, intonation, gaze, gesture, or other "
                "nonverbal cues from text alone" in _norm(prompt))

    def test_the_output_contract_keeps_the_experts_field_names(self) -> None:
        prompt = PROMPTS["v2"]
        for field in ("storytelling_events", "story_ids", "interruptions_or_resumptions",
                      "tellers", "story_evidence", "evidence_segment_ids",
                      "uncertain_spans", "summary", "issue"):
            assert field in prompt, field
        assert "time_seconds" not in prompt

    def test_the_transcript_lines_show_text_and_speaker_runs_separately(self) -> None:
        prompt = request_for(turns=NEMOTRON_TURNS, speaker_source="nemotron").prompt()
        assert "VIDEO: conversation_001" in prompt
        for row in SEGMENTS:
            assert row["segment_id"] in prompt
            assert f"{float(row['start_time']):.3f}" in prompt
        assert ("- segment_id=seg000001 start=3.000 end=5.500\n"
                "  text: I was fiddling with my wedding ring\n"
                "  speakers: [speaker_0] I was fiddling with my wedding ring") in prompt
        assert ("  text: it rolled down the side of the sofa\n"
                "  speakers: [speaker_1] it rolled down the side of the sofa") in prompt

    def test_the_prompt_says_quotes_come_from_the_text_line_and_how_to_pick_one(self) -> None:
        prompt = PROMPTS["v2"]
        assert "Copy exact wording into transcript_quote from the `text:` line" in prompt
        assert "add \"occurrence\": n" in prompt
        assert "(1 = first)" in prompt
        # The uniqueness rule is gone: a segment whose text repeats itself ("no no") has no
        # longer quote that is unique, so that rule made a correct boundary unexpressible.
        assert "occurs only ONCE" not in prompt
        # And the renderer really shows the line the rule talks about.
        assert "\n  text: " in request_for().prompt()

    def test_an_unknown_prompt_version_fails_loudly(self) -> None:
        request = request_for(prompt_version="v1")
        with pytest.raises(StageError, match="prompt version"):
            request.prompt()

    def test_the_key_moves_with_text_model_prompt_and_temperature(self) -> None:
        base = request_for().key()
        assert request_for(model="other").key() != base
        assert request_for(temperature=0.7).key() != base
        changed = [dict(SEGMENTS[0], text="a different line")] + SEGMENTS[1:]
        assert request_for(changed).key() != base

    def test_the_key_follows_the_rendered_labels(self) -> None:
        """A relabelled transcript is a different question, not the same one re-asked."""
        base_rows = prepared(turns=NEMOTRON_TURNS, speaker_source="nemotron")
        base = request_for(rows=base_rows, speaker_source="nemotron").key()
        assert request_for(speaker_source="none").key() != base
        # A threshold that leaves every rendered run unchanged leaves the question
        # unchanged: the model is shown the same transcript, so the answer is reusable.
        # Binding the *threshold* rather than the rendering would throw a paid answer away
        # for a setting that changed nothing.
        same_runs = prepared(turns=NEMOTRON_TURNS, speaker_source="nemotron",
                             speaker_min_coverage=0.9)
        assert request_for(rows=same_runs, speaker_source="nemotron").key() == base
        turns = [turn("speaker_0", 3.0, 11.5, index=1), turn("speaker_1", 3.0, 11.5, index=2)]
        strict = prepared(turns=turns, speaker_source="nemotron", speaker_max_competing=0.1)
        lenient = prepared(turns=turns, speaker_source="nemotron", speaker_max_competing=99.0)
        assert [row["speaker_runs"] for row in strict] != [
            row["speaker_runs"] for row in lenient]
        assert (request_for(rows=strict).key() != request_for(rows=lenient).key())

    def test_the_key_moves_with_extra_body_and_the_posted_body(self) -> None:
        """`extra_body` changes the answer without changing one character of the prompt."""
        base = request_for().key()
        assert request_for(extra_body={"top_p": 0.5}).key() != base
        assert request_for(max_output_tokens=64).key() != base
        rendered = request_for(extra_body={"top_p": 0.5})
        assert rendered.body()["top_p"] == 0.5
        assert rendered.body()["model"] == "chat"
        assert rendered.body()["messages"][0]["content"] == rendered.prompt()

    def test_the_key_moves_with_a_context_segment_time(self) -> None:
        """Context timestamps are part of the prompt, and were once missing from the key."""
        rows = prepared()
        base = request_for(rows=rows, max_segments_per_request=2, context_segments=1,
                           index=0).key()
        moved = [dict(row) for row in rows]
        moved[2]["start_time"] = 8.75  # a context segment of the first window
        assert request_for(rows=moved, max_segments_per_request=2, context_segments=1,
                           index=0).key() != base

    def test_the_key_moves_when_only_the_rendered_runs_move(self) -> None:
        base = request_for().key()
        relabelled = prepared()
        relabelled[0]["speaker_runs"] = "[someone_else] I was fiddling with my wedding ring"
        relabelled[0]["speaker_labels"] = ["someone_else"]
        assert request_for(rows=relabelled).key() != base

    def test_the_key_moves_when_the_prompt_text_moves(self, monkeypatch) -> None:
        before = request_for().key()
        monkeypatch.setitem(PROMPTS, "v2", PROMPTS["v2"] + "\nAsk politely.")
        assert request_for().key() != before


# ------------------------------------------------------------------ windowing


class TestWindowing:
    def test_the_default_is_the_whole_transcript_in_one_request(self) -> None:
        segments = [segment(i) for i in range(1, 41)]
        rows = prepared(segments, words=word_rows(segments))
        requests = build_requests(rows, video_id="v", max_segments_per_request=None,
                                  context_segments=5)
        assert len(requests) == 1
        assert len(requests[0].requested_ids) == 40
        assert requests[0].context == []

    def test_a_configured_window_still_partitions_in_start_order(self) -> None:
        segments = [segment(i) for i in range(1, 8)]
        rows = prepared(segments, words=word_rows(segments))
        requests = build_requests(rows, video_id="v", max_segments_per_request=3,
                                  context_segments=1)
        assert [r.requested_ids for r in requests] == [
            ["seg000001", "seg000002", "seg000003"],
            ["seg000004", "seg000005", "seg000006"],
            ["seg000007"],
        ]
        assert [r.window_index for r in requests] == [0, 1, 2]
        assert [r.id_prefix for r in requests] == ["w0", "w1", "w2"]

    def test_context_never_overlaps_the_window(self) -> None:
        segments = [segment(i) for i in range(1, 8)]
        rows = prepared(segments, words=word_rows(segments))
        requests = build_requests(rows, video_id="v", max_segments_per_request=3,
                                  context_segments=2)
        for request in requests:
            assert not set(request.requested_ids) & {r["segment_id"] for r in request.context}
        assert [len(r.context) for r in requests] == [2, 3, 2]

    def test_ids_outside_the_window_are_still_known_transcript_ids(self) -> None:
        segments = [segment(i) for i in range(1, 8)]
        rows = prepared(segments, words=word_rows(segments))
        request = build_requests(rows, video_id="v", max_segments_per_request=3,
                                 context_segments=1)[0]
        assert request.all_segment_ids == {f"seg{i:06d}" for i in range(1, 8)}
        assert "seg000007" not in request.requested_ids

    def test_an_empty_transcript_makes_no_request(self) -> None:
        assert build_requests([], video_id="v", max_segments_per_request=None,
                              context_segments=1) == []

    def test_a_non_positive_window_size_is_refused(self) -> None:
        with pytest.raises(StageError, match="max_segments_per_request"):
            build_requests(prepared(), video_id="v", max_segments_per_request=0,
                           context_segments=1)


# ------------------------------------------------------------------ parsing


class TestParsing:
    def test_a_plain_object_parses(self) -> None:
        assert parse_stories_response('{"storytelling_events": []}') == {
            "storytelling_events": []}

    def test_code_fences_and_prose_are_tolerated(self) -> None:
        assert parse_stories_response('```json\n{"stories": []}\n```') == {"stories": []}
        assert parse_stories_response('Here you go:\n{"stories": []}\nEnjoy.') == {
            "stories": []}

    @pytest.mark.parametrize("text", ["", "not json at all", "[1, 2]", '"word"', "null"])
    def test_a_response_that_is_not_an_object_is_refused(self, text: str) -> None:
        with pytest.raises(ValueError):
            parse_stories_response(text)


# ------------------------------------------------------------------ boundary times


class TestBoundaryTimes:
    def test_a_quote_is_resolved_to_its_first_and_last_word(self) -> None:
        request = request_for()
        validated = validate_response(honest_answer(request), request)
        event = validated.events[0]
        # The start quote is the whole first segment's text (segment 1 is 3.0-5.5).
        assert event["start_time"] == pytest.approx(3.0)
        assert event["start_time_precision"] == "word"
        story = validated.stories[0]
        # "with my wedding ring" begins at word index 3 of 7 → 3.0 + 3 * (2.5 / 7).
        assert story["start_time"] == pytest.approx(3.0 + 3 * (2.5 / 7))
        assert story["start_time_precision"] == "word"
        # Its end quote is the whole second segment, so it ends on that segment's end.
        assert story["end_time"] == pytest.approx(8.5)
        assert story["end_time_precision"] == "word"

    def test_no_words_falls_back_to_the_cited_segment_and_says_so(self) -> None:
        rows = prepared(words=[])
        request = request_for(rows=rows)
        validated = validate_response(honest_answer(request), request)
        event = validated.events[0]
        assert event["start_time"] == pytest.approx(SEGMENTS[0]["start_time"])
        assert event["end_time"] == pytest.approx(SEGMENTS[2]["end_time"])
        assert event["start_time_precision"] == "segment"
        assert event["end_time_precision"] == "segment"

    def test_untimed_words_fall_back_to_segment_precision(self) -> None:
        words = [dict(row, start_time=None, end_time=None) for row in WORDS]
        request = request_for(rows=prepared(words=words))
        event = validate_response(honest_answer(request), request).events[0]
        assert event["start_time_precision"] == "segment"
        assert event["start_time"] == pytest.approx(SEGMENTS[0]["start_time"])

    def test_case_whitespace_and_surrounding_punctuation_are_normalised(self) -> None:
        text = "Well, I said: 'not that one!' Then she left."
        assert locate_quote("not that one", text) is not None
        assert locate_quote("  WELL,   i said  ", text) == (0, 12)
        assert locate_quote("'not that one!'", text) is not None

    def test_a_paraphrased_quote_does_not_match(self) -> None:
        # Internal punctuation and wording are not normalised: no fuzzy matching.
        assert locate_quote("not that 1", "not that one") is None
        assert locate_quote("well I said", "Well, I said") is None

    def test_a_quote_inside_another_word_is_not_a_match(self) -> None:
        """Telediario seg000002: the quote "a" used to match inside "Ahora"."""
        text = "Ahora vamos a retroceder."
        assert quote_occurrences("a", text) == [(12, 13)]
        assert locate_quote("a", text) == (12, 13)
        # A prefix of a word is not a word either: "Ah" does not end where a token ends.
        assert quote_occurrences("Ah", text) == []

    def test_a_word_boundary_quote_reads_the_real_word(self) -> None:
        """Juman seg000007 repeats "that"; the quote has to be word-aligned to be found."""
        text = "that was the thing that rolled away"
        assert quote_occurrences("that", text) == [(0, 4), (19, 23)]
        assert locate_quote("that", text) is None  # ambiguous, so not a location
        assert locate_quote("that was the thing", text) == (0, 18)

    def test_the_word_alignment_does_not_depend_on_the_column_offsets(self) -> None:
        """Every character offset in the corpus is NULL; a wrong one must not be trusted.

        The placement is the token alignment of the segment's own text, so offsets that
        point at the wrong characters (all of them point at character 0 here) change
        nothing.
        """
        words = [dict(row, character_start=0, character_end=3) for row in WORDS]
        request = request_for(rows=prepared(words=words))
        validated = validate_response(honest_answer(request), request)
        assert validated.stories[0]["start_time"] == pytest.approx(3.0 + 3 * (2.5 / 7))

    def test_a_contraction_split_into_two_alignment_words_is_one_token(self) -> None:
        """``can`` + ``'t`` is the transcript's ``can't``: the token's own edges are used."""
        segments = [segment(1, text="I can't do it"), segment(2, text="she left")]
        words = [
            {"segment_id": "seg000001", "word": "I", "start_time": 3.0, "end_time": 3.2},
            {"segment_id": "seg000001", "word": "can", "start_time": 3.2, "end_time": 3.4},
            {"segment_id": "seg000001", "word": "'t", "start_time": 3.4, "end_time": 3.6},
            {"segment_id": "seg000001", "word": "do", "start_time": 3.6, "end_time": 3.8},
            {"segment_id": "seg000001", "word": "it", "start_time": 3.8, "end_time": 4.0},
            {"segment_id": "seg000002", "word": "she", "start_time": 6.0, "end_time": 6.2},
            {"segment_id": "seg000002", "word": "left", "start_time": 6.2, "end_time": 6.4},
        ]
        request = request_for(segments, words=words)
        payload = answer(events=[{
            "event_id": "SE1",
            "start": {"segment_id": "seg000001", "transcript_quote": "can't do",
                      "speaker": "unknown", "boundary_type": "pre_telling", "reason": "x"},
            "end": {"segment_id": "seg000002", "transcript_quote": "she left",
                    "speaker": "unknown", "boundary_type": "coda", "reason": "x"},
            "story_ids": ["S1"], "tellers": ["unknown"],
            "interruptions_or_resumptions": [], "confidence": "high", "notes": ""}],
            stories=[{
                "story_id": "S1", "event_id": "SE1",
                "start": {"segment_id": "seg000001", "transcript_quote": "can't",
                          "speaker": "unknown", "reason": "x"},
                "end": {"segment_id": "seg000001", "transcript_quote": "can't do it",
                        "speaker": "unknown", "reason": "x"},
                "tellers": ["unknown"], "summary": "s",
                "story_evidence": ["one represented event"],
                "evidence_segment_ids": ["seg000001"], "confidence": "high", "notes": ""}])
        validated = validate_response(payload, request)
        story = validated.stories[0]
        # "can't" starts at the "can" that was split out of it and ends at the "'t".
        assert story["start_time"] == pytest.approx(3.2)
        assert story["start_time_precision"] == "word"
        assert story["end_time"] == pytest.approx(4.0)
        assert story["end_time_precision"] == "word"

    def test_an_unaligned_edge_word_falls_back_to_the_segment_not_a_neighbour(self) -> None:
        """WhisperX aligned ``twelve``; the transcript says ``12`` — so no time is borrowed.

        The end of ``is 12`` must not become the end of ``is`` (word precision it has not
        earned) nor the end of ``twelve`` (a word the quote does not contain). It is the
        cited segment's own end, labelled ``segment``.
        """
        segments = [segment(1, text="the value is 12"), segment(2, text="she left")]
        words = [
            {"segment_id": "seg000001", "word": "the", "start_time": 3.0, "end_time": 3.2},
            {"segment_id": "seg000001", "word": "value", "start_time": 3.2, "end_time": 3.4},
            {"segment_id": "seg000001", "word": "is", "start_time": 3.4, "end_time": 3.6},
            {"segment_id": "seg000001", "word": "twelve", "start_time": 3.6, "end_time": 3.8},
            {"segment_id": "seg000002", "word": "she", "start_time": 6.0, "end_time": 6.2},
            {"segment_id": "seg000002", "word": "left", "start_time": 6.2, "end_time": 6.4},
        ]
        request = request_for(segments, words=words)
        payload = answer(events=[{
            "event_id": "SE1",
            "start": {"segment_id": "seg000001", "transcript_quote": "the value is 12",
                      "speaker": "unknown", "boundary_type": "pre_telling", "reason": "x"},
            "end": {"segment_id": "seg000002", "transcript_quote": "she left",
                    "speaker": "unknown", "boundary_type": "coda", "reason": "x"},
            "story_ids": ["S1"], "tellers": ["unknown"],
            "interruptions_or_resumptions": [], "confidence": "high", "notes": ""}],
            stories=[{
                "story_id": "S1", "event_id": "SE1",
                "start": {"segment_id": "seg000001", "transcript_quote": "the value is 12",
                          "speaker": "unknown", "reason": "x"},
                "end": {"segment_id": "seg000001", "transcript_quote": "is 12",
                        "speaker": "unknown", "reason": "x"},
                "tellers": ["unknown"], "summary": "s",
                "story_evidence": ["one represented event"],
                "evidence_segment_ids": ["seg000001"], "confidence": "high", "notes": ""}])
        validated = validate_response(payload, request)
        event = validated.events[0]
        story = validated.stories[0]
        assert event["start_time"] == pytest.approx(3.0)
        assert event["start_time_precision"] == "word"
        assert story["end_time"] == pytest.approx(5.5)  # the segment's own end
        assert story["end_time_precision"] == "segment"
        assert story["end_time"] != pytest.approx(3.6)  # not the "is" that precedes it
        assert story["end_time"] != pytest.approx(3.8)  # not the "twelve" it never said

    def test_inverted_marks_do_not_hide_the_edge_word(self) -> None:
        """``¿Qué`` is the token ``Qué``: the mark is at the edge, not part of the word."""
        segments = [segment(1, text="¿Qué pasa aquí?"), segment(2, text="nada")]
        words = [
            {"segment_id": "seg000001", "word": "Qué", "start_time": 3.0, "end_time": 3.2},
            {"segment_id": "seg000001", "word": "pasa", "start_time": 3.2, "end_time": 3.4},
            {"segment_id": "seg000001", "word": "aquí", "start_time": 3.4, "end_time": 3.6},
            {"segment_id": "seg000002", "word": "nada", "start_time": 6.0, "end_time": 6.2},
        ]
        rows = prepare_segments(segments, words, [], speaker_source="none")
        assert [token["start_time"] for token in rows[0]["tokens"]] == [
            pytest.approx(3.0), pytest.approx(3.2), pytest.approx(3.4)]
        request = build_requests(rows, video_id="v", max_segments_per_request=None,
                                 context_segments=0)
        assert quote_occurrences("¿Qué pasa!", "¿Qué pasa!") == [(1, 9)]

    def test_a_word_the_text_does_not_contain_gets_no_timing(self) -> None:
        words = [dict(row, word="GONE") if row["segment_id"] == "seg000001" else row
                 for row in WORDS]
        request = request_for(rows=prepared(words=words))
        # No word from the first segment can be attributed, so the start boundary is the
        # segment's own boundary and says so.
        event = validate_response(honest_answer(request), request).events[0]
        assert event["start_time_precision"] == "segment"

    def test_a_punctuation_only_alignment_row_never_times_an_edge(self) -> None:
        """A separate ``?``/ ``¿`` row is not a word: its time is not a word's boundary.

        The aligner used to *prefer* merging a token with the word plus the punctuation
        after it — two rows covered beats one — so the quoted word's end became the
        question mark's end.
        """
        segments = [segment(1, text="pasa"), segment(2, text="nada")]
        words = [
            {"segment_id": "seg000001", "word": "¿", "start_time": 2.8,
             "end_time": 3.0},
            {"segment_id": "seg000001", "word": "pasa", "start_time": 3.0,
             "end_time": 3.2},
            {"segment_id": "seg000001", "word": "?", "start_time": 3.2,
             "end_time": 3.4},
            {"segment_id": "seg000002", "word": "nada", "start_time": 6.0,
             "end_time": 6.2},
        ]
        rows = prepare_segments(segments, words, [], speaker_source="none")
        token = rows[0]["tokens"][0]
        assert token["start_time"] == pytest.approx(3.0)  # not the ¿ before it
        assert token["end_time"] == pytest.approx(3.2)  # not the ? after it

    def test_character_offsets_place_the_occurrence_the_aligner_cannot(self) -> None:
        """Offsets say *which* repeated word a row is; the order-only aligner does not.

        The text says ``no`` three times and the alignment carries two rows: by order the
        second row belongs to the second ``no``, and the third gets no time at all. The
        offset says the row is the third one, and it is honoured.
        """
        segments = [segment(1, text="no no no")]
        words = [
            {"segment_id": "seg000001", "word": "no", "start_time": 3.0,
             "end_time": 3.2},
            {"segment_id": "seg000001", "word": "no", "start_time": 3.2,
             "end_time": 3.4, "character_start": 6, "character_end": 8},
        ]
        by_order = prepare_segments(
            segments, [dict(row, character_start=None, character_end=None) for row in words],
            [], speaker_source="none")
        assert by_order[0]["tokens"][2]["start_time"] is None
        placed = prepare_segments(segments, words, [], speaker_source="none")
        tokens = placed[0]["tokens"]
        assert tokens[1]["start_time"] is None
        assert tokens[2]["start_time"] == pytest.approx(3.2)
        assert tokens[2]["end_time"] == pytest.approx(3.4)

    def test_the_aligner_handles_more_than_two_thousand_tokens(self) -> None:
        """It was a recursion that descended once per token: ~1000 tokens was a crash."""
        count = 2100
        words_in_text = [f"w{index}" for index in range(count)]
        text = " ".join(words_in_text)
        rows = [{"segment_id": "seg000001", "word": word,
                 "start_time": 3.0 + index, "end_time": 3.5 + index}
                for index, word in enumerate(words_in_text)]
        # A contraction the sub-word tokenizer split, deep inside: the dynamic program runs
        # for real, on a grid of a few cells rather than the whole transcript.
        rows[999] = {"segment_id": "seg000001", "word": "w99", "start_time": 1002.0,
                     "end_time": 1002.25}
        rows.insert(1000, {"segment_id": "seg000001", "word": "9", "start_time": 1002.25,
                           "end_time": 1002.5})
        entries = align_token_times(text, rows)
        assert len(entries) == count
        assert entries[0]["start_time"] == pytest.approx(3.0)
        assert entries[998]["end_time"] == pytest.approx(1001.5)
        # w99 + 9 is the token w999: the merged run gives the token its own two edges.
        assert entries[999]["start_time"] == pytest.approx(1002.0)
        assert entries[999]["end_time"] == pytest.approx(1002.5)
        assert entries[count - 1]["end_time"] == pytest.approx(3.5 + count - 1)

    def test_an_offset_may_cover_the_token_with_its_punctuation(self) -> None:
        """``go,`` at [3, 6) and ``go`` at [3, 5) name the same token.

        An aligner that ran on the original text reports the word *with* its edge
        punctuation; one that stripped it reports the core. Honouring only the core dropped
        the second form, and the occurrence it named then fell back to word order.
        """
        words = [
            {"segment_id": "seg000001", "word": "go", "start_time": 3.2, "end_time": 3.4,
             "character_start": 3, "character_end": 6},
            {"segment_id": "seg000001", "word": "go", "start_time": 3.0, "end_time": 3.2},
        ]
        entries = align_token_times("go go,", words)
        assert entries[1]["start_time"] == pytest.approx(3.2)
        assert entries[1]["end_time"] == pytest.approx(3.4)
        # The anchor is what moved it: by order the 3.2 row belongs to the first "go".
        assert entries[0]["start_time"] is None

    def test_an_offset_may_start_at_the_leading_punctuation(self) -> None:
        """``¿go?`` at [3, 7) names the second token although its core starts at 4."""
        words = [
            {"segment_id": "seg000001", "word": "go", "start_time": 3.2, "end_time": 3.4,
             "character_start": 3, "character_end": 7},
            {"segment_id": "seg000001", "word": "go", "start_time": 3.0, "end_time": 3.2},
        ]
        entries = align_token_times("go ¿go?", words)
        assert entries[1]["start_time"] == pytest.approx(3.2)
        assert entries[0]["start_time"] is None

    def test_beyond_the_alignment_budget_nothing_is_placed_positionally(
            self, monkeypatch) -> None:
        """Above the cell budget the interior is unaligned, not guessed at by position.

        The positional fallback claimed token *i* was word *i* without having compared the
        neighbourhoods that decide it. Leaving the interior unresolved is the same answer
        as "no exact alignment", so an affected boundary says ``segment`` instead of
        borrowing a time it has not earned.
        """
        rows = [segment(1, text="x a b y"), segment(2, text="nada")]
        words = [
            {"segment_id": "seg000001", "word": "p", "start_time": 3.0, "end_time": 3.2},
            {"segment_id": "seg000001", "word": "a", "start_time": 3.2, "end_time": 3.4},
            {"segment_id": "seg000001", "word": "q", "start_time": 3.4, "end_time": 3.6},
            {"segment_id": "seg000001", "word": "r", "start_time": 3.6, "end_time": 3.8},
            {"segment_id": "seg000002", "word": "nada", "start_time": 6.0,
             "end_time": 6.2},
        ]
        # Inside the budget the one truly equal key is found: "a" is the second token.
        placed = align_token_times("x a b y", words[:4])
        assert placed[1]["start_time"] == pytest.approx(3.2)
        monkeypatch.setattr(stories_module, "_ALIGNMENT_CELL_LIMIT", 0)
        assert all(entry["start_time"] is None
                   for entry in align_token_times("x a b y", words[:4]))
        request = request_for(rows=prepared(rows, words=words))
        payload = answer(events=[{
            "event_id": "SE1",
            "start": {"segment_id": "seg000001", "transcript_quote": "x a b y",
                      "speaker": "unknown", "boundary_type": "pre_telling", "reason": "x"},
            "end": {"segment_id": "seg000002", "transcript_quote": "nada",
                    "speaker": "unknown", "boundary_type": "coda", "reason": "x"},
            "story_ids": ["S1"], "tellers": ["unknown"],
            "interruptions_or_resumptions": [], "confidence": "high", "notes": ""}],
            stories=[{
                "story_id": "S1", "event_id": "SE1",
                "start": {"segment_id": "seg000001", "transcript_quote": "a b",
                          "speaker": "unknown", "reason": "x"},
                "end": {"segment_id": "seg000001", "transcript_quote": "a b y",
                        "speaker": "unknown", "reason": "x"},
                "tellers": ["unknown"], "summary": "s",
                "story_evidence": ["one represented event"],
                "evidence_segment_ids": ["seg000001"], "confidence": "high", "notes": ""}])
        story = validate_response(payload, request).stories[0]
        assert story["start_time"] == pytest.approx(3.0)  # the segment's own start
        assert story["start_time_precision"] == "segment"

    def test_a_case_expanding_character_does_not_create_a_quote(self) -> None:
        """``sidefold('ß') == 'ss'`` used to make ``sa`` match inside ``ßa``.

        One character must map to one character, or the index that says where a match
        started is a lie. Folding that would change the length is dropped back to the
        character as written — the exact-case fallback.
        """
        assert quote_occurrences("sa", "ßa") == []
        assert locate_quote("sa", "ßa") is None
        assert locate_quote("ßa", "ßa") == (0, 2)
        assert locate_quote("BETA", "el beta") == (3, 7)  # case folding still works
        assert locate_quote("AÑO", "el año pasado") == (3, 6)
        assert locate_quote("STRASSE", "die Straße") is None  # no expansion, no match
        assert locate_quote("straße", "die Straße") == (4, 10)


# ------------------------------------------------------------------ rejection


class TestValidation:
    def test_an_honest_answer_validates_and_namespaces_ids(self) -> None:
        validated = validate_response(honest_answer(request_for()), request_for())
        assert [row["event_id"] for row in validated.events] == ["w0-SE1"]
        assert [row["story_id"] for row in validated.stories] == ["w0-S1"]
        assert json.loads(validated.events[0]["story_ids"]) == ["w0-S1"]
        assert validated.stories[0]["event_id"] == "w0-SE1"
        assert not validated.empty

    def test_empty_arrays_are_a_valid_answer(self) -> None:
        validated = validate_response(answer(), request_for())
        assert validated.empty
        assert validated.events == [] and validated.stories == []

    def test_an_absent_uncertain_spans_key_is_an_empty_list(self) -> None:
        payload = honest_answer(request_for())
        payload.pop("uncertain_spans")
        assert validate_response(payload, request_for()).uncertain_spans == []

    @pytest.mark.parametrize("key", ["storytelling_events", "stories", "uncertain_spans"])
    def test_a_section_that_is_not_a_list_is_refused(self, key: str) -> None:
        payload = honest_answer(request_for())
        payload[key] = {"SE1": {}}
        with pytest.raises(ValidationError, match="must be a list"):
            validate_response(payload, request_for())

    def test_an_invented_segment_id_is_refused(self) -> None:
        request = request_for()
        payload = answer(events=[event_payload(
            request, start={"segment_id": "seg999999", "transcript_quote": "x",
                            "speaker": "unknown", "boundary_type": "pre_telling",
                            "reason": "invented"})])
        with pytest.raises(ValidationError, match="seg999999"):
            validate_response(payload, request)

    def test_a_quote_that_is_not_in_the_cited_segment_is_refused(self) -> None:
        request = request_for()
        payload = answer(events=[event_payload(
            request, start={"segment_id": "seg000001",
                            "transcript_quote": "words nobody said",
                            "speaker": label(request, "seg000001"),
                            "boundary_type": "pre_telling", "reason": "fabricated"})])
        with pytest.raises(ValidationError, match="not verbatim"):
            validate_response(payload, request)

    def test_a_quote_taken_from_a_different_segment_is_refused(self) -> None:
        # The quote exists in the transcript, but not in the segment the boundary cites.
        request = request_for()
        payload = answer(events=[event_payload(
            request, start={"segment_id": "seg000001",
                            "transcript_quote": SEGMENTS[1]["text"],
                            "speaker": label(request, "seg000001"),
                            "boundary_type": "pre_telling", "reason": "wrong segment"})])
        with pytest.raises(ValidationError, match="not verbatim"):
            validate_response(payload, request)

    @pytest.mark.parametrize("boundary_type", ["start", "narrative_launch", ""])
    def test_an_unknown_start_boundary_type_is_refused(self, boundary_type: str) -> None:
        request = request_for()
        event = event_payload(request)
        event["start"]["boundary_type"] = boundary_type
        with pytest.raises(ValidationError, match="boundary_type"):
            validate_response(answer(events=[event]), request)

    def test_an_end_boundary_type_from_the_start_vocabulary_is_refused(self) -> None:
        request = request_for()
        event = event_payload(request)
        event["end"]["boundary_type"] = "pre_telling"
        with pytest.raises(ValidationError, match="boundary_type"):
            validate_response(answer(events=[event]), request)

    @pytest.mark.parametrize("confidence", ["certain", "0.9", ""])
    def test_confidence_outside_the_vocabulary_is_refused(self, confidence: str) -> None:
        request = request_for()
        with pytest.raises(ValidationError, match="confidence"):
            validate_response(answer(events=[event_payload(request, confidence=confidence)]),
                              request)

    def test_a_duplicated_event_id_is_refused(self) -> None:
        request = request_for()
        payload = answer(events=[event_payload(request), event_payload(request)])
        with pytest.raises(ValidationError, match="duplicate event_id"):
            validate_response(payload, request)

    def test_a_duplicated_story_id_is_refused(self) -> None:
        request = request_for()
        payload = answer(events=[event_payload(request)],
                         stories=[story_payload(request), story_payload(request)])
        with pytest.raises(ValidationError, match="duplicate story_id"):
            validate_response(payload, request)

    def test_a_story_whose_event_is_not_an_event_is_refused(self) -> None:
        request = request_for()
        payload = answer(events=[event_payload(request)],
                         stories=[story_payload(request, event_id="SE7")])
        with pytest.raises(ValidationError, match="SE7"):
            validate_response(payload, request)

    def test_story_ids_that_omit_an_answered_story_are_refused(self) -> None:
        request = request_for()
        payload = answer(events=[event_payload(request, story_ids=[])],
                         stories=[story_payload(request)])
        with pytest.raises(ValidationError, match="story_ids"):
            validate_response(payload, request)

    def test_story_ids_naming_a_story_that_does_not_exist_are_refused(self) -> None:
        request = request_for()
        payload = answer(events=[event_payload(request, story_ids=["S1", "S9"])],
                         stories=[story_payload(request)])
        with pytest.raises(ValidationError, match="story_ids"):
            validate_response(payload, request)

    def test_a_story_starting_before_its_event_is_refused(self) -> None:
        # A story cannot begin inside a telling that has not started: 0.0 is the event's
        # own start, so a start quote from the *end* of the first segment is outside it.
        request = request_for()
        payload = answer(events=[event_payload(
            request, start={"segment_id": "seg000001",
                            "transcript_quote": "ring",
                            "speaker": label(request, "seg000001"),
                            "boundary_type": "story_launch", "reason": "late start"})],
            stories=[story_payload(request)])
        with pytest.raises(ValidationError, match="before its event"):
            validate_response(payload, request)

    def test_a_story_ending_after_its_event_is_refused(self) -> None:
        request = request_for()
        payload = answer(events=[event_payload(
            request, end={"segment_id": "seg000002",
                          "transcript_quote": "it rolled",
                          "speaker": label(request, "seg000002"),
                          "boundary_type": "coda", "reason": "too early"})],
            stories=[story_payload(request)])
        with pytest.raises(ValidationError, match="after its event"):
            validate_response(payload, request)

    def test_an_end_before_its_start_is_refused(self) -> None:
        request = request_for()
        with pytest.raises(ValidationError, match="before it starts"):
            validate_response(answer(events=[event_payload(
                request, start={"segment_id": "seg000003",
                                "transcript_quote": SEGMENTS[2]["text"],
                                "speaker": label(request, "seg000003"),
                                "boundary_type": "pre_telling", "reason": "reversed"},
                end={"segment_id": "seg000001",
                     "transcript_quote": SEGMENTS[0]["text"],
                     "speaker": label(request, "seg000001"),
                     "boundary_type": "coda", "reason": "reversed"})]), request)

    def test_a_speaker_not_shown_in_the_request_is_refused(self) -> None:
        request = request_for()
        event = event_payload(request)
        event["start"]["speaker"] = "speaker_9"
        with pytest.raises(ValidationError, match="speaker_9"):
            validate_response(answer(events=[event]), request)

    def test_a_missing_speaker_is_refused(self) -> None:
        request = request_for()
        event = event_payload(request)
        event["start"]["speaker"] = ""
        with pytest.raises(ValidationError, match="speaker"):
            validate_response(answer(events=[event]), request)

    def test_an_unknown_teller_is_refused(self) -> None:
        request = request_for()
        with pytest.raises(ValidationError, match="tellers"):
            validate_response(answer(events=[event_payload(request, tellers=["narrator"])],
                                     stories=[story_payload(request)]), request)

    @pytest.mark.parametrize("evidence", [None, [], ["   "], "one explanation", [1],
                                           ["", "an explanation"]])
    def test_story_evidence_is_required_and_every_entry_is_an_explanation(
            self, evidence: Any) -> None:
        request = request_for()
        payload = answer(events=[event_payload(request)],
                         stories=[story_payload(request, story_evidence=evidence)])
        with pytest.raises(ValidationError, match="story_evidence"):
            validate_response(payload, request)

    def test_story_evidence_explanations_are_stored_verbatim(self) -> None:
        request = request_for()
        validated = validate_response(honest_answer(request), request)
        evidence = json.loads(validated.stories[0]["story_evidence"])
        assert evidence == ["a wedding ring rolling away is one represented event",
                            "the speaker's embarrassment evaluates that experience"]

    @pytest.mark.parametrize("evidence", [["seg000001"],
                                          ["seg000001", "seg000002"],
                                          ['["seg000001", "seg000002"]'],
                                          ["the ring rolled away", "seg000002"]])
    def test_a_story_evidence_entry_of_only_segment_ids_is_refused(
            self, evidence: list[str]) -> None:
        """An "explanation" that is only ids says where to look, never why.

        It would pass the required-and-non-empty rule while explaining nothing, and the ids
        already have their own field.
        """
        request = request_for()
        payload = answer(events=[event_payload(request)],
                        stories=[story_payload(request, story_evidence=evidence)])
        with pytest.raises(ValidationError, match="is only segment ids"):
            validate_response(payload, request)

    def test_an_explanation_that_mentions_an_id_among_words_is_fine(self) -> None:
        request = request_for()
        payload = answer(events=[event_payload(request)],
                        stories=[story_payload(
                            request,
                            story_evidence=["seg000001 shows the ring rolling away, an "
                                            "event told from experience"])])
        assert validate_response(payload, request).stories

    def test_a_story_without_evidence_segment_ids_is_refused(self) -> None:
        request = request_for()
        story = story_payload(request)
        story.pop("evidence_segment_ids")
        with pytest.raises(ValidationError, match="evidence_segment_ids is required"):
            validate_response(answer(events=[event_payload(request)], stories=[story]),
                              request)

    def test_an_evidence_id_outside_the_story_range_is_refused(self) -> None:
        """Evidence for a story that spans seg000001..seg000002 cannot be seg000003."""
        request = request_for()
        payload = answer(events=[event_payload(request)],
                         stories=[story_payload(request,
                                                evidence_segment_ids=["seg000003"])])
        with pytest.raises(ValidationError, match="outside the story's own"):
            validate_response(payload, request)

    def test_an_invented_evidence_segment_id_is_refused(self) -> None:
        request = request_for()
        payload = answer(events=[event_payload(request)],
                         stories=[story_payload(request,
                                                evidence_segment_ids=["seg000001",
                                                                      "nope"])])
        with pytest.raises(ValidationError, match="not in this transcript"):
            validate_response(payload, request)

    def test_a_repeated_quote_needs_an_occurrence_and_then_says_where_it_is(self) -> None:
        """A quote that occurs twice says nothing about where the boundary is — until the
        answer selects one. Quoting more words cannot always fix it (``no no``), so the
        selection is what makes the retry converge."""
        segments = [segment(1, text="no no"), segment(2, text="then she left")]
        words = [
            {"segment_id": "seg000001", "word": "no", "start_time": 3.0,
             "end_time": 4.0},
            {"segment_id": "seg000001", "word": "no", "start_time": 4.0,
             "end_time": 5.0},
            {"segment_id": "seg000002", "word": "then", "start_time": 6.0,
             "end_time": 6.5},
            {"segment_id": "seg000002", "word": "she", "start_time": 6.5,
             "end_time": 7.0},
            {"segment_id": "seg000002", "word": "left", "start_time": 7.0,
             "end_time": 8.0},
        ]
        request = request_for(segments, words=words)
        payload = answer(events=[{
            "event_id": "SE1",
            "start": {"segment_id": "seg000001", "transcript_quote": "no",
                      "speaker": "unknown", "boundary_type": "pre_telling",
                      "reason": "which one?"},
            "end": {"segment_id": "seg000002", "transcript_quote": "then she left",
                    "speaker": "unknown", "boundary_type": "coda", "reason": "x"},
            "story_ids": [], "tellers": ["unknown"],
            "interruptions_or_resumptions": [], "confidence": "high", "notes": ""}])
        with pytest.raises(ValidationError, match="occurs 2 times") as excinfo:
            validate_response(payload, request)
        assert "occurrence" in str(excinfo.value)
        # The retry the message asks for is the answer that passes: the second "no".
        payload["storytelling_events"][0]["start"]["occurrence"] = 2
        event = validate_response(payload, request).events[0]
        assert event["start_time"] == pytest.approx(4.0)
        assert event["start_time_precision"] == "word"
        # And occurrence 1 is the first one, with its own time.
        payload["storytelling_events"][0]["start"]["occurrence"] = 1
        assert (validate_response(payload, request).events[0]["start_time"]
                == pytest.approx(3.0))
        # A model that answers the same problem by quoting more words still passes.
        payload["storytelling_events"][0]["start"] = {
            "segment_id": "seg000001", "transcript_quote": "no no",
            "speaker": "unknown", "boundary_type": "pre_telling", "reason": "x"}
        assert (validate_response(payload, request).events[0]["start_time"]
                == pytest.approx(3.0))

    @pytest.mark.parametrize("occurrence,message", [
        (3, "out of range"),
        (0, "out of range"),
        (True, "not a whole number"),
        (1.5, "not a whole number"),
    ])
    def test_an_impossible_occurrence_is_a_breach(self, occurrence: Any,
                                                  message: str) -> None:
        segments = [segment(1, text="no no"), segment(2, text="then she left")]
        request = request_for(segments, words=word_rows(segments))
        payload = answer(events=[{
            "event_id": "SE1",
            "start": {"segment_id": "seg000001", "transcript_quote": "no",
                      "occurrence": occurrence, "speaker": "unknown",
                      "boundary_type": "pre_telling", "reason": "x"},
            "end": {"segment_id": "seg000002", "transcript_quote": "then she left",
                    "speaker": "unknown", "boundary_type": "coda", "reason": "x"},
            "story_ids": [], "tellers": ["unknown"],
            "interruptions_or_resumptions": [], "confidence": "high", "notes": ""}])
        with pytest.raises(ValidationError, match=message):
            validate_response(payload, request)

    def test_a_single_occurrence_accepts_only_occurrence_one(self) -> None:
        request = request_for()
        payload = honest_answer(request)
        payload["storytelling_events"][0]["start"]["occurrence"] = 1
        assert validate_response(payload, request).events
        payload["storytelling_events"][0]["start"]["occurrence"] = 2
        with pytest.raises(ValidationError, match="the only occurrence is 1"):
            validate_response(payload, request)

    def test_an_interruption_outside_the_event_window_ids_is_refused(self) -> None:
        request = request_for()
        event = event_payload(request, interruptions_or_resumptions=[{
            "start_segment_id": "seg999999", "end_segment_id": "seg000002",
            "transcript_quote": "x", "description": "invented"}])
        with pytest.raises(ValidationError, match="seg999999"):
            validate_response(answer(events=[event]), request)

    def test_an_interruption_whose_quote_is_not_in_its_range_is_refused(self) -> None:
        request = request_for()
        event = event_payload(request, interruptions_or_resumptions=[{
            "start_segment_id": "seg000002", "end_segment_id": "seg000003",
            "transcript_quote": "I was fiddling with my wedding ring",
            "description": "quote from the wrong stretch"}])
        with pytest.raises(ValidationError, match="not verbatim"):
            validate_response(answer(events=[event]), request)

    def test_an_honest_interruption_gets_computed_times(self) -> None:
        request = request_for()
        event = event_payload(request, interruptions_or_resumptions=[{
            "start_segment_id": "seg000002", "end_segment_id": "seg000002",
            "transcript_quote": SEGMENTS[1]["text"],
            "description": "aside before the resumption"}])
        validated = validate_response(answer(events=[event], stories=[story_payload(request)]),
                                      request)
        interruptions = json.loads(validated.events[0]["interruptions"])
        assert interruptions[0]["start_time"] == pytest.approx(SEGMENTS[1]["start_time"])
        assert interruptions[0]["end_time"] == pytest.approx(SEGMENTS[1]["end_time"])

    def test_an_uncertain_span_gets_segment_boundaries(self) -> None:
        request = request_for()
        validated = validate_response(answer(uncertain=[{
            "start_segment_id": "seg000002", "end_segment_id": "seg000003",
            "transcript_quote": SEGMENTS[2]["text"],
            "issue": "could be a coda or a shift"}]), request)
        span = validated.uncertain_spans[0]
        assert span["start_time"] == pytest.approx(SEGMENTS[1]["start_time"])
        assert span["end_time"] == pytest.approx(SEGMENTS[2]["end_time"])
        assert span["issue"] == "could be a coda or a shift"

    def test_an_uncertain_span_with_an_invented_id_is_refused(self) -> None:
        request = request_for()
        payload = answer(uncertain=[{
            "start_segment_id": "seg000002", "end_segment_id": "seg999999",
            "transcript_quote": "x", "issue": "invented"}])
        with pytest.raises(ValidationError, match="seg999999"):
            validate_response(payload, request)

    def test_an_uncertain_span_ending_before_it_starts_is_refused(self) -> None:
        request = request_for()
        payload = answer(uncertain=[{
            "start_segment_id": "seg000003", "end_segment_id": "seg000001",
            "transcript_quote": SEGMENTS[0]["text"], "issue": "reversed"}])
        with pytest.raises(ValidationError, match="before it starts"):
            validate_response(payload, request)

    def test_several_problems_are_all_reported(self) -> None:
        request = request_for()
        payload = answer(events=[event_payload(request, confidence="sure"),
                                 event_payload(request, event_id="SE2",
                                               start={"segment_id": "seg999999",
                                                      "transcript_quote": "x",
                                                      "speaker": "unknown",
                                                      "boundary_type": "pre_telling",
                                                      "reason": "x"})])
        with pytest.raises(ValidationError) as excinfo:
            validate_response(payload, request)
        assert len(excinfo.value.issues) >= 3


class TestWindowedDrops:
    """At event level: an event anchored outside the window is a counted gap."""

    def windowed(self, **cfg: Any):
        rows = prepared()
        options = {"max_segments_per_request": 2, "context_segments": 1}
        options.update(cfg)
        return build_requests(rows, video_id="conversation_001", **options)

    def test_an_event_anchored_outside_the_window_is_dropped_with_its_story(self) -> None:
        request = self.windowed()[0]  # seg000001..seg000002
        payload = answer(events=[event_payload(request)], stories=[story_payload(request)])
        validated = validate_response(payload, request)
        assert validated.events == [] and validated.stories == []
        # One dropped event, carrying one story: the group is one gap in the answer.
        assert [item["kind"] for item in validated.dropped] == ["event"]
        assert validated.dropped[0]["child_story_ids"] == ["S1"]
        assert "seg000003" in validated.dropped[0]["reason"]

    def test_an_event_inside_the_window_is_kept(self) -> None:
        request = self.windowed()[0]
        payload = answer(events=[event_payload(
            request, end={"segment_id": "seg000002", "transcript_quote": "sofa",
                          "speaker": label(request, "seg000002"),
                          "boundary_type": "coda", "reason": "ends here"})],
            stories=[story_payload(request)])
        validated = validate_response(payload, request)
        assert len(validated.events) == 1 and len(validated.stories) == 1
        assert validated.dropped == []

    def test_a_cross_window_event_with_a_context_interruption_is_dropped_not_rejected(
            self) -> None:
        """A neighbouring window's event is a counted gap, not a reason to retry forever."""
        request = self.windowed()[0]  # seg000001..seg000002, context seg000003
        payload = answer(events=[event_payload(request, interruptions_or_resumptions=[{
            "start_segment_id": "seg000003", "end_segment_id": "seg000003",
            "transcript_quote": SEGMENTS[2]["text"],
            "description": "a coda that belongs to the next window"}])],
            stories=[story_payload(request)])
        validated = validate_response(payload, request)
        assert validated.events == [] and validated.stories == []
        assert [item["kind"] for item in validated.dropped] == ["event"]
        assert validated.dropped[0]["child_story_ids"] == ["S1"]
        assert "seg000003" in validated.dropped[0]["reason"]

    def test_the_dropped_event_is_counted_once_and_its_stories_apart(self) -> None:
        request = self.windowed()[0]
        payload = answer(events=[event_payload(request, story_ids=["S1", "S2"])],
                         stories=[story_payload(request),
                                  story_payload(request, story_id="S2",
                                                summary="a second telling")])
        validated = validate_response(payload, request)
        assert _drop_counts(validated.dropped) == {"events": 1, "stories": 2,
                                                  "uncertain_spans": 0}

    def test_an_interruption_from_a_segment_nobody_was_shown_is_still_a_breach(self) -> None:
        rows = prepared()
        request = build_requests(rows, video_id="v", max_segments_per_request=2,
                                 context_segments=0)[0]
        payload = answer(events=[event_payload(request, interruptions_or_resumptions=[{
            "start_segment_id": "seg000003", "end_segment_id": "seg000003",
            "transcript_quote": SEGMENTS[2]["text"], "description": "invented visibility"}])])
        with pytest.raises(ValidationError, match="not shown in this request"):
            validate_response(payload, request)

    def test_an_uncertain_span_entirely_outside_the_window_is_dropped(self) -> None:
        request = self.windowed()[0]
        payload = answer(uncertain=[{
            "start_segment_id": "seg000003", "end_segment_id": "seg000003",
            "transcript_quote": SEGMENTS[2]["text"], "issue": "outside"}])
        validated = validate_response(payload, request)
        assert validated.uncertain_spans == []
        assert [item["kind"] for item in validated.dropped] == ["uncertain_span"]

    def test_an_uncertain_span_touching_context_is_dropped_and_counted(self) -> None:
        """A span reaching into a context segment is the neighbouring window's answer.

        Keeping it would publish an interval that runs into another window's territory, so
        it is dropped and counted — the same question a cross-window event asks.
        """
        request = self.windowed()[0]  # seg000001..seg000002, context seg000003
        payload = answer(uncertain=[{
            "start_segment_id": "seg000001", "end_segment_id": "seg000003",
            "transcript_quote": SEGMENTS[2]["text"], "issue": "reaches into context"}])
        validated = validate_response(payload, request)
        assert validated.uncertain_spans == []
        assert [item["kind"] for item in validated.dropped] == ["uncertain_span"]
        assert _drop_counts(validated.dropped)["uncertain_spans"] == 1

    def test_a_boundary_in_an_unshown_segment_is_a_breach_not_a_drop(self) -> None:
        """A real segment the prompt never showed cannot support a quoted boundary.

        There is no context here, so ``seg000003`` was never rendered: the model cannot
        have copied a quote out of it, and calling that a window drop would publish an
        answer written about text it never read.
        """
        rows = prepared()
        request = build_requests(rows, video_id="v", max_segments_per_request=2,
                                 context_segments=0)[0]
        payload = answer(events=[event_payload(request)], stories=[story_payload(request)])
        with pytest.raises(ValidationError, match="not shown in this request"):
            validate_response(payload, request)

    def test_an_uncertain_span_in_an_unshown_segment_is_a_breach(self) -> None:
        rows = prepared()
        request = build_requests(rows, video_id="v", max_segments_per_request=2,
                                 context_segments=0)[0]
        payload = answer(uncertain=[{
            "start_segment_id": "seg000003", "end_segment_id": "seg000003",
            "transcript_quote": SEGMENTS[2]["text"], "issue": "never shown"}])
        with pytest.raises(ValidationError, match="not shown in this request"):
            validate_response(payload, request)

    def test_an_evidence_id_from_an_unshown_segment_is_a_breach(self) -> None:
        """The evidence rule is the boundary rule: only what the prompt showed counts.

        A window normally partitions the transcript contiguously, so every id between two
        shown ids is shown too; this request shows ``seg000001`` and ``seg000003`` and
        leaves ``seg000002`` in the transcript but out of the prompt, which is the state the
        check exists for (a caller that assembles its own window).
        """
        request = request_for()
        partial = replace(
            request,
            requested=[row for row in request.requested
                       if row["segment_id"] in ("seg000001", "seg000003")],
            context=[])
        assert "seg000002" not in partial.visible_ids
        assert "seg000002" in partial.segments
        story = story_payload(partial, end={"segment_id": "seg000003",
                                           "transcript_quote": SEGMENTS[2]["text"],
                                           "speaker": label(partial, "seg000003"),
                                           "reason": "the last represented event"},
                              evidence_segment_ids=["seg000001", "seg000002"])
        payload = answer(events=[event_payload(partial)], stories=[story])
        with pytest.raises(ValidationError, match="were not shown in this request"):
            validate_response(payload, partial)


# ------------------------------------------------------------------ speakers


class TestSpeakerSources:
    def test_nemotron_labels_words_from_non_overlapping_turns(self) -> None:
        turns = [turn("speaker_0", 3.0, 4.3, index=1),
                 turn("speaker_1", 4.3, 8.5, index=2),
                 turn("speaker_0", 9.0, 11.5, index=3)]
        rows = prepared(turns=turns, speaker_source="nemotron")
        runs = {row["segment_id"]: row["speaker_runs"] for row in rows}
        assert "[speaker_0]" in runs["seg000001"] and "[speaker_1]" in runs["seg000001"]
        assert "[speaker_0]" in runs["seg000003"]

    def test_nemotron_overlap_makes_the_disputed_word_unknown(self) -> None:
        # Both turns cover the whole first segment, so every word is disputed.
        turns = [turn("speaker_0", 3.0, 11.5, index=1),
                 turn("speaker_1", 3.0, 11.5, index=2)]
        rows = prepared(turns=turns, speaker_source="nemotron")
        assert rows[0]["speaker_runs"].startswith("[unknown] I was fiddling")
        assert rows[0]["speaker_labels"] == ["unknown"]

    def test_nemotron_covers_a_segment_with_no_words_over_its_interval(self) -> None:
        turns = [turn("speaker_0", 3.0, 5.5, index=1),
                 turn("speaker_1", 6.0, 8.5, index=2),
                 turn("speaker_0", 9.0, 11.5, index=3)]
        rows = prepared(words=[], turns=turns, speaker_source="nemotron")
        assert [row["speaker_runs"].split("]")[0] + "]" for row in rows] == [
            "[speaker_0]", "[speaker_1]", "[speaker_0]"]

    def test_allowed_labels_are_the_ones_the_runs_really_emitted(self) -> None:
        """A word folded into a neighbour's run never shows its own label."""
        segments = [segment(1, text="hello there"), segment(2, text="yes")]
        words = [
            {"segment_id": "seg000001", "word": "hello", "start_time": 3.0,
             "end_time": 3.5, "speaker_id": "speaker_0"},
            # Not a word of the text, so it is placed nowhere: it stays inside
            # speaker_0's run and speaker_1 never appears in the prompt.
            {"segment_id": "seg000001", "word": "GONE", "start_time": 3.5, "end_time": 4.0,
             "speaker_id": "speaker_1"},
            {"segment_id": "seg000001", "word": "there", "start_time": 4.0,
             "end_time": 4.5, "speaker_id": "speaker_0"},
            {"segment_id": "seg000002", "word": "yes", "start_time": 6.0, "end_time": 6.5,
             "speaker_id": "speaker_0"},
        ]
        turns = [turn("speaker_0", 3.0, 3.5, index=1), turn("speaker_1", 3.5, 4.0, index=2),
                 turn("speaker_0", 4.0, 4.5, index=3), turn("speaker_0", 6.0, 6.5, index=4)]
        rows = prepared(segments, words=words, turns=turns, speaker_source="nemotron")
        assert rows[0]["speaker_runs"] == "[speaker_0] hello there"
        assert rows[0]["speaker_labels"] == ["speaker_0"]
        request = build_requests(rows, video_id="v", max_segments_per_request=None,
                                 context_segments=0)[0]
        assert request.speaker_labels == frozenset({"speaker_0", "unknown"})
        payload = answer(events=[{
            "event_id": "SE1",
            "start": {"segment_id": "seg000001", "transcript_quote": "hello there",
                      "speaker": "speaker_1", "boundary_type": "pre_telling",
                      "reason": "a label the prompt never showed"},
            "end": {"segment_id": "seg000002", "transcript_quote": "yes",
                    "speaker": "speaker_0", "boundary_type": "coda", "reason": "x"},
            "story_ids": [], "tellers": ["speaker_0"],
            "interruptions_or_resumptions": [], "confidence": "high", "notes": ""}])
        with pytest.raises(ValidationError, match="speaker_1"):
            validate_response(payload, request)

    def test_a_context_label_is_visible_and_therefore_allowed(self) -> None:
        rows = prepared(turns=[turn("speaker_0", 3.0, 5.5, index=1),
                               turn("speaker_1", 6.0, 8.5, index=2),
                               turn("speaker_2", 9.0, 11.5, index=3)],
                        speaker_source="nemotron")
        request = build_requests(rows, video_id="v", max_segments_per_request=2,
                                 context_segments=1)[0]
        assert request.requested_ids == ["seg000001", "seg000002"]
        assert "speaker_2" in request.speaker_labels  # shown in the context only
        payload = answer(events=[event_payload(
            request, end={"segment_id": "seg000002", "transcript_quote": "sofa",
                          "speaker": "speaker_1", "boundary_type": "coda",
                          "reason": "ends here"},
            tellers=["speaker_2"])],
            stories=[story_payload(request, tellers=["speaker_2"])])
        validated = validate_response(payload, request)
        assert json.loads(validated.events[0]["tellers"]) == ["speaker_2"]

    def test_pyannote_is_no_longer_a_speaker_source(self) -> None:
        """Nemotron is the only diarizer (operator decision, 2026-11-26)."""
        import pydantic

        from multimodal_pipeline.config import PipelineConfig

        with pytest.raises(pydantic.ValidationError, match="speaker_source"):
            PipelineConfig.model_validate({
                "input": {"directory": "/tmp/in"}, "output": {"directory": "/tmp/out"},
                "stories": {"speaker_source": "pyannote"}})

    def test_none_labels_everything_unknown(self) -> None:
        rows = prepared(speaker_source="none")
        assert all(row["speaker_labels"] == ["unknown"] for row in rows)

    def test_a_missing_nemotron_table_is_all_unknown_and_recorded(
            self, context) -> None:
        use_stage_config(context, speaker_source="nemotron")
        seed_segments(context, SEGMENTS)
        seed_words(context, WORDS)
        messages = log_recorder(context)
        run_stage(context)
        assert all(row["start_speaker"] == "unknown"
                   for row in rows_of(context, "storytelling_events"))
        recorded = summary(context)
        assert recorded["speaker_source_requested"] == "nemotron"
        assert recorded["speaker_source_used"] == "none"
        assert any("nemotron" in message.lower() for message in messages)

    def test_never_falls_back_to_the_segment_speaker_ids(self, context) -> None:
        """Words/segments carry speaker ids; a missing Nemotron table must still not use them."""
        use_stage_config(context, speaker_source="nemotron")
        seed_segments(context, SEGMENTS)
        seed_words(context, WORDS)  # words carry SPEAKER_00 / SPEAKER_01
        run_stage(context)
        assert summary(context)["speaker_source_used"] == "none"
        assert {row["start_speaker"] for row in rows_of(context, "storytelling_events")} == {
            "unknown"}

    def test_a_nemotron_corpus_records_the_source_it_used(self, context) -> None:
        use_stage_config(context, speaker_source="nemotron")
        seed_segments(context, SEGMENTS)
        seed_words(context, WORDS)
        seed_nemotron(context, [turn("speaker_0", 3.0, 5.5, index=1),
                                turn("speaker_1", 6.0, 8.5, index=2),
                                turn("speaker_0", 9.0, 11.5, index=3)])
        run_stage(context)
        recorded = summary(context)
        assert recorded["speaker_source_requested"] == "nemotron"
        assert recorded["speaker_source_used"] == "nemotron"


# ------------------------------------------------------------------ config


class TestConfig:
    def test_the_defaults_are_the_v2_settings(self, context) -> None:
        cfg = context.config.stories
        assert cfg.enabled is True
        assert cfg.provider == "openai-compatible"
        assert cfg.prompt_version == "v2"
        assert cfg.max_segments_per_request is None
        assert cfg.speaker_source == "nemotron"
        assert cfg.speaker_min_coverage == 0.6
        assert cfg.speaker_max_competing == 0.25
        assert cfg.repair_max_answer_chars == 20000

    def test_the_v1_prompt_version_is_refused_with_a_reason(self) -> None:
        import pydantic

        from multimodal_pipeline.config import PipelineConfig

        with pytest.raises(pydantic.ValidationError, match="v2"):
            PipelineConfig.model_validate({
                "input": {"directory": "/tmp/in"}, "output": {"directory": "/tmp/out"},
                "stories": {"prompt_version": "v1"}})

    def test_an_unknown_speaker_source_is_refused(self) -> None:
        import pydantic

        from multimodal_pipeline.config import PipelineConfig

        with pytest.raises(pydantic.ValidationError, match="speaker_source"):
            PipelineConfig.model_validate({
                "input": {"directory": "/tmp/in"}, "output": {"directory": "/tmp/out"},
                "stories": {"speaker_source": "whisper"}})

    def test_an_invalid_coverage_fraction_is_refused(self) -> None:
        import pydantic

        from multimodal_pipeline.config import PipelineConfig

        with pytest.raises(pydantic.ValidationError, match="speaker_min_coverage"):
            PipelineConfig.model_validate({
                "input": {"directory": "/tmp/in"}, "output": {"directory": "/tmp/out"},
                "stories": {"speaker_min_coverage": 0.0}})

    def test_a_zero_window_size_is_refused_at_parse_time(self) -> None:
        import pydantic

        from multimodal_pipeline.config import PipelineConfig

        with pytest.raises(pydantic.ValidationError, match="max_segments_per_request"):
            PipelineConfig.model_validate({
                "input": {"directory": "/tmp/in"}, "output": {"directory": "/tmp/out"},
                "stories": {"max_segments_per_request": 0}})

    def test_a_zero_repair_answer_budget_is_refused(self) -> None:
        import pydantic

        from multimodal_pipeline.config import PipelineConfig

        with pytest.raises(pydantic.ValidationError, match="repair_max_answer_chars"):
            PipelineConfig.model_validate({
                "input": {"directory": "/tmp/in"}, "output": {"directory": "/tmp/out"},
                "stories": {"repair_max_answer_chars": 0}})

    def test_the_client_gets_the_configured_repair_budget(self, context) -> None:
        use_stage_config(context, provider="openai-compatible", base_url="https://x.invalid/v1",
                         api_key="sk-test", model="chat", repair_max_answer_chars=1234)
        assert StoriesStage()._client(context).repair_max_answer_chars == 1234

    def test_an_unconfigured_endpoint_degrades_to_a_skip_not_a_crash(self, context) -> None:
        seed_segments(context, SEGMENTS)
        enabled, reason = StoriesStage().enabled(context)
        assert enabled is False
        assert "stories.base_url" in reason and "stories.api_key" in reason

    def test_the_mock_provider_is_always_configured(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        assert StoriesStage().enabled(context) == (True, "")

    def test_disabled_by_config(self, context) -> None:
        use_stage_config(context, enabled=False)
        seed_segments(context, SEGMENTS)
        enabled, reason = StoriesStage().enabled(context)
        assert enabled is False
        assert reason == "stories.enabled = false"

    def test_a_transcript_with_no_speech_skips_with_a_reason(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, [])
        enabled, reason = StoriesStage().enabled(context)
        assert enabled is False and "no speech segments" in reason

    def test_the_mock_provider_is_the_one_handed_to_the_stage(self, context) -> None:
        use_stage_config(context, model="mock-model")
        assert isinstance(StoriesStage()._client(context), MockStoriesClient)


# ------------------------------------------------------------------ execution


class TestStageExecution:
    def test_the_mock_client_writes_three_tables_with_namespaced_ids(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        seed_words(context, WORDS)
        extra = run_stage(context)
        events = rows_of(context, "storytelling_events")
        stories = rows_of(context, "stories")
        assert len(events) == 1 and len(stories) == 1
        assert (events[0]["event_id"], stories[0]["story_id"]) == ("w0-SE1", "w0-S1")
        assert stories[0]["event_id"] == "w0-SE1"
        assert json.loads(events[0]["story_ids"]) == ["w0-S1"]
        assert events[0]["start_time"] == pytest.approx(3.0)
        assert events[0]["end_time"] == pytest.approx(11.5)
        assert events[0]["start_time_precision"] == "word"
        assert events[0]["schema_version"] == "2.1"
        assert rows_of(context, "uncertain_spans") == []
        assert extra["events_total"] == 1 and extra["stories_total"] == 1
        assert extra["uncertain_spans_total"] == 0
        assert extra["windows"] == 1 and extra["empty_window_count"] == 0

    def test_an_empty_answer_writes_zero_row_tables(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS[:1])  # one segment: the mock answers emptily
        extra = run_stage(context)
        for artifact in ("storytelling_events", "stories", "uncertain_spans"):
            assert rows_of(context, artifact) == []
        assert extra["empty_window_count"] == 1
        assert extra["events_total"] == 0 and extra["stories_total"] == 0

    def test_the_whole_transcript_is_one_request_by_default(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        seed_words(context, WORDS)
        client = FakeClient([envelope(unknown_speaker_answer())])
        run_stage(context, client)
        assert client.request_count == 1
        assert len(client.prompts) == 1
        # Every segment of the transcript is citable in that one request.
        for row in SEGMENTS:
            assert row["segment_id"] in client.prompts[0]

    def test_a_windowed_run_drops_and_counts(self, context) -> None:
        use_stage_config(context, max_segments_per_request=2, context_segments=1)
        seed_segments(context, SEGMENTS)
        seed_words(context, WORDS)
        client = FakeClient([
            envelope(answer(events=[event_payload(request_for(max_segments_per_request=2,
                                                            context_segments=1,
                                                            index=0))],
                            stories=[story_payload(request_for(
                                max_segments_per_request=2, context_segments=1, index=0))])),
            envelope(answer()),
        ])
        messages = log_recorder(context)
        extra = run_stage(context, client)
        assert extra["windows"] == 2
        assert extra["dropped_outside_window"] == 1  # the event, counted once
        assert extra["dropped_stories_outside_window"] == 1  # and its story, apart
        assert rows_of(context, "storytelling_events") == []
        assert any("dropped" in message for message in messages)
        recorded = summary(context)
        assert recorded["dropped_outside_window"] == 1
        assert recorded["dropped_stories_outside_window"] == 1
        assert recorded["dropped_detail"][0]["window_index"] == 0
        assert recorded["dropped_detail"][0]["child_story_ids"] == ["S1"]

    def test_the_raw_response_is_preserved_byte_identical(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        content = envelope(honest_answer(request_for()))
        run_stage(context, FakeClient([content]))
        raw = sorted(context.artifact("stories_raw").glob("window_*.json"))
        assert raw[0].read_text(encoding="utf-8") == content

    def test_the_summary_records_counts_and_speaker_source(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        run_stage(context)
        recorded = summary(context)
        assert recorded["video_id"] == "conversation_001"
        assert recorded["windows"] == 1
        assert recorded["storytelling_events"] == 1
        assert recorded["stories"] == 1
        assert recorded["uncertain_spans"] == 0
        assert recorded["speaker_source_requested"] == "none"
        assert recorded["speaker_source_used"] == "none"
        assert recorded["max_segments_per_request"] is None

    def test_the_cache_avoids_a_second_request(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        client = FakeClient([envelope(honest_answer(request_for()))] * 2)
        stage = StoriesStage()
        stage._client = lambda _ctx: client  # type: ignore[method-assign]
        assert stage.run(context).status == "completed"
        assert client.request_count == 1
        assert stage.run(context).status == "completed"
        assert client.request_count == 1, "the cached window was requested again"
        assert len(rows_of(context, "storytelling_events")) == 1

    def test_a_reused_window_is_reported_in_the_summary(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        client = FakeClient([envelope(honest_answer(request_for()))] * 2)
        stage = StoriesStage()
        stage._client = lambda _ctx: client  # type: ignore[method-assign]
        stage.run(context)
        extra = stage.run(context).detail["provenance"]["extra"]
        assert extra["batches_reused"] == 1

    def test_a_changed_transcript_invalidates_the_cache(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        client = FakeClient([envelope(honest_answer(request_for())), envelope(answer())])
        stage = StoriesStage()
        stage._client = lambda _ctx: client  # type: ignore[method-assign]
        stage.run(context)
        assert len(rows_of(context, "storytelling_events")) == 1
        seed_segments(context, [dict(row, text="totally different words") for row in SEGMENTS])
        stage.run(context)
        assert client.request_count == 2
        assert rows_of(context, "storytelling_events") == []

    def test_a_changed_speaker_source_invalidates_the_cache(self, context) -> None:
        """The same text with a different attribution rule is a different question."""
        use_stage_config(context, speaker_source="none")
        seed_segments(context, SEGMENTS)
        seed_words(context, WORDS)
        seed_nemotron(context, [turn("speaker_0", 3.0, 5.5, index=1),
                                turn("speaker_1", 6.0, 8.5, index=2),
                                turn("speaker_0", 9.0, 11.5, index=3)])
        client = FakeClient([envelope(unknown_speaker_answer())] * 2)
        stage = StoriesStage()
        stage._client = lambda _ctx: client  # type: ignore[method-assign]
        stage.run(context)
        context.config.stories.speaker_source = "nemotron"
        stage.run(context)
        assert client.request_count == 2

    def test_the_cache_is_revalidated_before_reuse(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        run_stage(context, FakeClient([envelope(honest_answer(request_for()))]))
        cache = sorted((context.artifact("stories_raw") / "cache").glob("*.json"))
        assert len(cache) == 1
        entry = json.loads(cache[0].read_text(encoding="utf-8"))
        entry["payload"]["stories"][0]["story_evidence"] = ["seg999999"]
        cache[0].write_text(json.dumps(entry), encoding="utf-8")
        client = FakeClient([envelope(honest_answer(request_for()))])
        extra = run_stage(context, client)
        assert client.request_count == 1, "a stale cache entry was accepted"
        assert extra["batches_reused"] == 0

    def test_a_cache_whose_content_contradicts_its_payload_is_refetched(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        run_stage(context, FakeClient([envelope(honest_answer(request_for()))]))
        cache_files = list((context.artifact("stories_raw") / "cache").glob("*.json"))
        entry = json.loads(cache_files[0].read_text(encoding="utf-8"))
        tampered = json.loads(entry["content"])
        tampered["stories"][0]["story_evidence"] = ["seg999999"]
        entry["content"] = json.dumps(tampered)
        cache_files[0].write_text(json.dumps(entry), encoding="utf-8")
        for raw in context.artifact("stories_raw").glob("window_*.json"):
            raw.unlink()
        client = FakeClient([envelope(honest_answer(request_for()))])
        extra = run_stage(context, client)
        assert client.request_count == 1
        assert extra["batches_reused"] == 0
        restored = list(context.artifact("stories_raw").glob("window_*.json"))
        assert restored and "seg999999" not in restored[0].read_text(encoding="utf-8")

    def test_a_cache_entry_without_its_content_is_refetched(self, context) -> None:
        """No original bytes, no reuse: a re-serialisation is not what the endpoint sent."""
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        content = envelope(honest_answer(request_for()))
        run_stage(context, FakeClient([content]))
        cache_files = list((context.artifact("stories_raw") / "cache").glob("*.json"))
        entry = json.loads(cache_files[0].read_text(encoding="utf-8"))
        entry.pop("content")
        cache_files[0].write_text(json.dumps(entry), encoding="utf-8")
        for raw in context.artifact("stories_raw").glob("window_*.json"):
            raw.unlink()
        client = FakeClient([content])
        extra = run_stage(context, client)
        assert client.request_count == 1, "a cache entry without content was reused"
        assert extra["batches_reused"] == 0
        restored = sorted(context.artifact("stories_raw").glob("window_*.json"))
        assert [path.read_text(encoding="utf-8") for path in restored] == [content]

    def test_a_raw_file_that_disagrees_with_the_cache_is_rewritten(self, context) -> None:
        """The raw file beside a row must be the answer that row came from."""
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        content = envelope(honest_answer(request_for()))
        run_stage(context, FakeClient([content]))
        raw = sorted(context.artifact("stories_raw").glob("window_*.json"))
        raw[0].write_text(envelope(answer()), encoding="utf-8")  # a different answer
        client = FakeClient([content])
        extra = run_stage(context, client)
        assert client.request_count == 0 and extra["batches_reused"] == 1
        assert raw[0].read_text(encoding="utf-8") == content
        assert len(rows_of(context, "storytelling_events")) == 1

    def test_a_reused_window_restores_the_cached_bytes_not_a_respacing(self, context) -> None:
        """Same payload, different bytes: the raw file must be what the endpoint sent.

        Parsed equality used to decide this, so a re-indented (or otherwise edited) file
        that still parsed to the accepted payload was left in place — a file the endpoint
        never produced, filed as its original response.
        """
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        content = envelope(honest_answer(request_for()))
        respaced = json.dumps(honest_answer(request_for()), indent=2)
        assert respaced != content
        run_stage(context, FakeClient([content]))
        raw = sorted(context.artifact("stories_raw").glob("window_*.json"))
        raw[0].write_text(respaced, encoding="utf-8")
        run_stage(context, FakeClient([content]))
        assert raw[0].read_text(encoding="utf-8") == content

    def test_rejected_answers_are_kept_beside_the_accepted_one(self, endpoint, tmp_path) -> None:
        request = request_for()
        rejected = "I cannot help with that."
        server, _state = endpoint([(200, ok_body(rejected)),
                                   (200, ok_body(envelope(answer())))])
        client = stories_client(server, max_retries=2)
        client.rejected_dir = tmp_path / "rejected"
        client.rejected_dir.mkdir()
        verdict = client.detect(request)
        assert verdict.answer.empty
        attempts = sorted(client.rejected_dir.glob("window_0_*_attempt1_*.txt"))
        assert len(attempts) == 1, list(client.rejected_dir.iterdir())
        stored = attempts[0].read_text(encoding="utf-8")
        assert rejected in stored
        assert "rejected answer (attempt 1 of 3)" in stored
        assert "reason: ValueError" in stored

    def test_the_stage_keeps_the_rejected_attempts_it_produces(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        stage = StoriesStage()
        directory: list[Any] = []

        class RejectingClient(FakeClient):
            rejected_dir = None

            def detect(self, request):
                if self.rejected_dir is not None:
                    directory.append(self.rejected_dir)
                    self.rejected_dir.mkdir(parents=True, exist_ok=True)
                return super().detect(request)

        run_stage(context, RejectingClient([envelope(honest_answer(request_for()))]))
        assert directory and directory[0] == context.artifact("stories_raw") / "rejected"
        assert directory[0].is_dir()

    def test_a_reused_window_restores_a_deleted_raw_file(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        content = envelope(honest_answer(request_for()))
        client = FakeClient([content, content])
        stage = StoriesStage()
        stage._client = lambda _ctx: client  # type: ignore[method-assign]
        stage.run(context)
        raw = sorted(context.artifact("stories_raw").glob("window_*.json"))
        assert len(raw) == 1
        raw[0].unlink()
        stage.run(context)
        restored = sorted(context.artifact("stories_raw").glob("window_*.json"))
        assert len(restored) == 1
        assert restored[0].read_text(encoding="utf-8") == content
        assert client.request_count == 1

    def test_cache_can_be_switched_off(self, context) -> None:
        use_stage_config(context, cache=False)
        seed_segments(context, SEGMENTS)
        client = FakeClient([envelope(honest_answer(request_for()))] * 2)
        stage = StoriesStage()
        stage._client = lambda _ctx: client  # type: ignore[method-assign]
        stage.run(context)
        stage.run(context)
        assert client.request_count == 2

    def test_a_client_that_gives_up_fails_the_stage(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        client = FakeClient([StageError("stories window 0 failed after 4 attempts: bad JSON",
                                        details={"window_index": 0})])
        stage = StoriesStage()
        stage._client = lambda _ctx: client  # type: ignore[method-assign]
        with pytest.raises(StageError, match="failed after"):
            stage.run(context)
        assert client.request_count == 1
        assert not context.artifact("stories").exists()
        assert not context.artifact("storytelling_events").exists()

    def test_the_api_key_never_reaches_the_raw_artifacts(self, context) -> None:
        use_stage_config(context, provider="openai-compatible",
                         base_url="https://x.invalid/v1", api_key="sk-secret-123456",
                         model="chat")
        seed_segments(context, SEGMENTS)
        run_stage(context, FakeClient([envelope(honest_answer(request_for()))]))
        blob = "".join(path.read_text(encoding="utf-8")
                       for path in context.artifact("stories_raw").rglob("*")
                       if path.is_file())
        assert "sk-secret-123456" not in blob


# ------------------------------------------------------------------ fingerprint


class TestFingerprint:
    def test_the_fingerprint_carries_the_config_and_the_input_digests(self, context) -> None:
        from multimodal_pipeline.stages.metadata import sha256_of

        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        seed_words(context, WORDS)
        payload = StoriesStage().config_fingerprint(context)
        assert payload["stage"] == "stories"
        assert payload["max_segments_per_request"] is None
        assert payload["prompt_version"] == "v2"
        assert payload["speaker_source"] == "none"
        assert payload["segments_digest"] == sha256_of(context.artifact("speech_segments"))
        assert payload["words_digest"] == sha256_of(context.artifact("speech_words"))
        assert payload["speaker_turns_digest"] is None

    def test_a_changed_transcript_changes_the_fingerprint(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        stage = StoriesStage()
        before = stage.config_fingerprint(context)
        seed_segments(context, [dict(row, text="a different transcription") for row in SEGMENTS])
        assert stage.config_fingerprint(context) != before

    def test_a_changed_word_table_changes_the_fingerprint(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        seed_words(context, WORDS)
        stage = StoriesStage()
        before = stage.config_fingerprint(context)
        seed_words(context, [dict(row, start_time=(row["start_time"] or 0) + 0.01)
                             for row in WORDS])
        assert stage.config_fingerprint(context)["words_digest"] != before["words_digest"]

    def test_a_changed_diarization_changes_the_fingerprint(self, context) -> None:
        use_stage_config(context, speaker_source="nemotron")
        seed_segments(context, SEGMENTS)
        seed_nemotron(context, [turn("speaker_0", 3.0, 11.5)])
        stage = StoriesStage()
        before = stage.config_fingerprint(context)
        seed_nemotron(context, [turn("speaker_1", 3.0, 11.5)])
        assert (stage.config_fingerprint(context)["speaker_turns_digest"]
                != before["speaker_turns_digest"])

    def test_the_fingerprint_names_the_python_that_computed_it(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        fingerprint = StoriesStage().config_fingerprint(context)
        digest = fingerprint["_python_code_sha256"]
        assert isinstance(digest, str) and len(digest) == 64

    def test_a_speaker_threshold_change_changes_the_fingerprint(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        stage = StoriesStage()
        before = stage.config_fingerprint(context)
        context.config.stories.speaker_min_coverage = 0.9
        assert stage.config_fingerprint(context) != before


# ------------------------------------------------------------------ validate


class TestValidate:
    def test_the_counts_it_reports_are_the_counts_on_disk(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        run_stage(context, FakeClient([envelope(answer(uncertain=[{
            "start_segment_id": "seg000002", "end_segment_id": "seg000003",
            "transcript_quote": SEGMENTS[2]["text"], "issue": "unclear"}]) )]))
        result = StoriesStage().validate(context)
        assert result == {"storytelling_events": 0, "stories": 0, "uncertain_spans": 1,
                          "windows": 1, "empty_windows": 0}

    def test_a_missing_table_is_a_validation_failure(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        with pytest.raises(ValidationError, match="missing"):
            StoriesStage().validate(context)

    def test_an_event_whose_stories_disagree_is_refused(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        run_stage(context, FakeClient([envelope(honest_answer(request_for()))]))
        rows = rows_of(context, "storytelling_events")
        rows[0]["story_ids"] = "[]"
        write_table(context.artifact("storytelling_events"),
                    pa.Table.from_pylist(rows, schema=STORYTELLING_EVENTS_SCHEMA),
                    STORYTELLING_EVENTS_SCHEMA)
        with pytest.raises(ValidationError, match="lists stories"):
            StoriesStage().validate(context)

    def test_a_story_naming_an_unknown_event_is_refused(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        run_stage(context, FakeClient([envelope(honest_answer(request_for()))]))
        rows = rows_of(context, "stories")
        rows[0]["event_id"] = "w0-SE9"
        write_table(context.artifact("stories"),
                    pa.Table.from_pylist(rows, schema=STORIES_SCHEMA), STORIES_SCHEMA)
        with pytest.raises(ValidationError, match="not a storytelling event"):
            StoriesStage().validate(context)

    def test_a_boundary_outside_its_segment_is_refused(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        run_stage(context, FakeClient([envelope(honest_answer(request_for()))]))
        rows = rows_of(context, "storytelling_events")
        rows[0]["start_time"] = 99.0
        write_table(context.artifact("storytelling_events"),
                    pa.Table.from_pylist(rows, schema=STORYTELLING_EVENTS_SCHEMA),
                    STORYTELLING_EVENTS_SCHEMA)
        with pytest.raises(ValidationError, match="outside its start segment"):
            StoriesStage().validate(context)

    def test_an_empty_table_with_the_wrong_columns_is_refused(self, context) -> None:
        """Zero rows read back as [] whatever the file was written with: check the schema."""
        import pyarrow.parquet as pq

        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        run_stage(context, FakeClient([envelope(honest_answer(request_for()))]))
        path = context.artifact("stories")
        pq.write_table(pa.table({"schema_version": pa.array([], type=pa.string()),
                                 "story_id": pa.array([], type=pa.string()),
                                 "parent_id": pa.array([], type=pa.string())}), path)
        with pytest.raises(ValidationError, match="columns do not match its schema"):
            StoriesStage().validate(context)

    def test_a_request_key_without_its_raw_window_file_is_refused(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        run_stage(context, FakeClient([envelope(honest_answer(request_for()))]))
        for raw in context.artifact("stories_raw").glob("window_*.json"):
            raw.unlink()
        with pytest.raises(ValidationError, match="is missing"):
            StoriesStage().validate(context)

    def test_a_corrupt_raw_file_is_caught_even_with_the_rows_intact(self, context) -> None:
        """The raw window is the evidence behind the rows, so validate() reads it back.

        Truncating it used to pass: nothing parsed the file, so "it exists" was the whole
        check and the tables' only evidence was gone.
        """
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        run_stage(context, FakeClient([envelope(honest_answer(request_for()))]))
        raw = sorted(context.artifact("stories_raw").glob("window_*.json"))
        raw[0].write_text(raw[0].read_text(encoding="utf-8")[:40], encoding="utf-8")
        with pytest.raises(ValidationError, match="not a parseable stories answer"):
            StoriesStage().validate(context)

    def test_a_raw_file_whose_answer_disagrees_with_the_table_is_caught(self, context) -> None:
        """The response still parses and still validates — it just is a different answer."""
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        run_stage(context, FakeClient([envelope(honest_answer(request_for()))]))
        raw = sorted(context.artifact("stories_raw").glob("window_*.json"))
        raw[0].write_text(envelope(answer()), encoding="utf-8")
        with pytest.raises(ValidationError, match="disagree about window"):
            StoriesStage().validate(context)

    def test_a_raw_response_whose_time_moved_under_the_same_id_is_caught(
            self, context) -> None:
        """The edited response still validates; it just resolves to another time.

        Ids and counts are unchanged — one event, one story, the same ``w0-S1`` — because
        only the ``occurrence`` moved from the first "the ring" to the second. An id-set
        comparison passed this, and the table then carried a time its evidence no longer
        supports.
        """
        rows = [segment(1, text="the ring the ring"),
                segment(2, text=SEGMENTS[1]["text"], speaker="SPEAKER_01"),
                segment(3, text=SEGMENTS[2]["text"])]
        words = word_rows(rows)
        request = request_for(rows=prepared(rows, words=words))
        story = story_payload(request)
        story["start"] = {"segment_id": "seg000001", "transcript_quote": "the ring",
                         "occurrence": 1, "speaker": label(request, "seg000001"),
                         "reason": "the speaker enters the story world"}
        event = event_payload(request, start={
            "segment_id": "seg000001", "transcript_quote": "the ring the ring",
            "speaker": label(request, "seg000001"), "boundary_type": "pre_telling",
            "reason": "the gist comes first"})
        use_stage_config(context)
        seed_segments(context, rows)
        seed_words(context, words)
        run_stage(context, FakeClient([envelope(answer(events=[event], stories=[story]))]))
        assert StoriesStage().validate(context)["stories"] == 1
        raw = min(context.artifact("stories_raw").glob("window_*.json"))
        edited = json.loads(raw.read_text(encoding="utf-8"))
        assert edited["stories"][0]["start"]["occurrence"] == 1
        edited["stories"][0]["start"]["occurrence"] = 2
        raw.write_text(json.dumps(edited), encoding="utf-8")
        with pytest.raises(ValidationError, match="disagree about window") as excinfo:
            StoriesStage().validate(context)
        assert "start_time" in str(excinfo.value), str(excinfo.value)

    def test_values_swapped_between_rows_are_caught(self) -> None:
        """Each column keeps its multiset when two rows swap a value; the rows do not."""
        expected = [{"story_id": "w0-S1", "start_time": 3.0},
                    {"story_id": "w0-S2", "start_time": 4.25}]
        swapped = [{"story_id": "w0-S1", "start_time": 4.25},
                   {"story_id": "w0-S2", "start_time": 3.0}]
        problems = stories_module._row_disagreements("stories", "window_0.json", 0,
                                                     expected, swapped)
        assert problems and "different rows" in problems[0]
        # Reordering whole rows is not a disagreement.
        assert stories_module._row_disagreements("stories", "window_0.json", 0, expected,
                                                 list(reversed(expected))) == []

    def test_two_windows_filed_under_each_others_request_keys_are_caught(self, context) -> None:
        """The rows are scoped by the request_key they carry, not by their window index.

        Swapping the two keys leaves every window index, segment id and row count exactly
        where it was, so a comparison that grouped the table by ``window_index`` found
        nothing wrong: the rows it compared to window 0's response were still window 0's
        rows.
        """
        rows = [segment(1), segment(2), segment(3), segment(4, text="and then we laughed")]
        use_stage_config(context, max_segments_per_request=2, context_segments=0)
        seed_segments(context, rows)
        run_stage(context)
        assert StoriesStage().validate(context)["windows"] == 2
        events = rows_of(context, "storytelling_events")
        assert len(events) == 2, events
        keys = [str(row["request_key"]) for row in events]
        assert keys[0] != keys[1]
        events[0]["request_key"], events[1]["request_key"] = keys[1], keys[0]
        write_table(context.artifact("storytelling_events"),
                    pa.Table.from_pylist(events, schema=STORYTELLING_EVENTS_SCHEMA),
                    STORYTELLING_EVENTS_SCHEMA)
        with pytest.raises(ValidationError, match="disagree about window") as excinfo:
            StoriesStage().validate(context)
        assert "event_id" in str(excinfo.value), str(excinfo.value)

    def test_a_summary_that_does_not_list_its_raw_windows_is_refused(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        run_stage(context)
        path = context.artifact("stories_raw") / "stories_summary.json"
        recorded = json.loads(path.read_text(encoding="utf-8"))
        recorded.pop("raw_windows")
        path.write_text(json.dumps(recorded), encoding="utf-8")
        with pytest.raises(ValidationError, match="does not list the raw window files"):
            StoriesStage().validate(context)

    def test_every_window_is_listed_even_an_empty_one(self, context) -> None:
        use_stage_config(context, max_segments_per_request=2, context_segments=1)
        seed_segments(context, SEGMENTS)
        run_stage(context, FakeClient([envelope(answer()), envelope(answer())]))
        recorded = summary(context)
        assert [entry["window_index"] for entry in recorded["raw_windows"]] == [0, 1]
        assert all(entry["empty"] is None or isinstance(entry["empty"], bool)
                   for entry in recorded["raw_windows"])
        assert {entry["file"] for entry in recorded["raw_windows"]} == {
            path.name for path in context.artifact("stories_raw").glob("window_*.json")}
        assert StoriesStage().validate(context)["windows"] == 2

    def test_a_table_row_whose_request_is_not_a_listed_window_is_refused(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        run_stage(context, FakeClient([envelope(honest_answer(request_for()))]))
        rows = rows_of(context, "storytelling_events")
        rows[0]["request_key"] = "0" * 24
        write_table(context.artifact("storytelling_events"),
                    pa.Table.from_pylist(rows, schema=STORYTELLING_EVENTS_SCHEMA),
                    STORYTELLING_EVENTS_SCHEMA)
        with pytest.raises(ValidationError, match="does not list as a window"):
            StoriesStage().validate(context)

    def test_a_summary_count_that_disagrees_with_the_table_is_refused(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        run_stage(context)
        path = context.artifact("stories_raw") / "stories_summary.json"
        recorded = json.loads(path.read_text(encoding="utf-8"))
        recorded["storytelling_events"] = 7
        path.write_text(json.dumps(recorded), encoding="utf-8")
        with pytest.raises(ValidationError, match="stories_summary.json says"):
            StoriesStage().validate(context)


# ------------------------------------------------------------------ schema and wiring


class TestSchemaAndWiring:
    def test_the_three_schemas_are_flat_and_versioned(self) -> None:
        for schema in (STORYTELLING_EVENTS_SCHEMA, STORIES_SCHEMA, UNCERTAIN_SPANS_SCHEMA):
            names = [field.name for field in schema]
            assert names[0] == "schema_version"
            assert not any("list" in str(field.type) for field in schema)

    def test_the_story_schema_carries_the_event_and_separate_boundaries(self) -> None:
        names = [field.name for field in STORIES_SCHEMA]
        assert "event_id" in names
        assert "parent_id" not in names and "why_it_is_a_story" not in names
        assert {"start_segment_id", "end_segment_id", "start_quote", "end_quote",
                "start_time_precision", "end_time_precision", "story_evidence",
                "evidence_segment_ids", "summary", "tellers", "confidence",
                "notes"} <= set(names)

    def test_the_event_schema_keeps_the_experts_field_names(self) -> None:
        names = [field.name for field in STORYTELLING_EVENTS_SCHEMA]
        assert {"story_ids", "tellers", "interruptions", "confidence", "notes",
                "start_boundary_type", "end_boundary_type"} <= set(names)

    def test_the_tables_are_registered_under_their_artifact_names(self) -> None:
        assert TABLE_SCHEMAS["storytelling_events"] is STORYTELLING_EVENTS_SCHEMA
        assert TABLE_SCHEMAS["stories"] is STORIES_SCHEMA
        assert TABLE_SCHEMAS["uncertain_spans"] is UNCERTAIN_SPANS_SCHEMA

    def test_the_artifact_paths_live_in_the_stories_directory(self) -> None:
        assert ARTIFACT_LAYOUT["storytelling_events"] == "stories/storytelling_events.parquet"
        assert ARTIFACT_LAYOUT["stories"] == "stories/stories.parquet"
        assert ARTIFACT_LAYOUT["uncertain_spans"] == "stories/uncertain_spans.parquet"
        assert ARTIFACT_LAYOUT["stories_raw"] == "stories/raw"

    def test_ensure_dirs_creates_the_stories_directory(self, tmp_path) -> None:
        from multimodal_pipeline.artifacts import VideoPaths

        VideoPaths(tmp_path / "ds").ensure_dirs()
        assert (tmp_path / "ds" / "stories" / "raw").is_dir()

    def test_the_stage_declares_its_io(self) -> None:
        stage = StoriesStage()
        assert stage.name == "stories"
        assert stage.inputs == ("speech_segments",)
        assert stage.outputs == ("storytelling_events", "stories", "uncertain_spans",
                                 "stories_raw")
        assert stage.config_keys == ("stories",)

    def test_it_is_numbered_after_speaker_fusion_and_before_finalization(self) -> None:
        assert STAGE_ORDER.index("speaker_fusion") < STAGE_ORDER.index("stories")
        assert STAGE_ORDER.index("stories") < STAGE_ORDER.index("finalization")

    def test_it_depends_on_the_speaker_branch_only(self) -> None:
        assert STAGE_DEPENDENCIES["stories"] == ("speaker_assignment",)

    def test_it_owns_a_stage_log(self) -> None:
        assert "stories" in STAGE_LOG_NAMES

    def test_it_is_registered_with_the_orchestrator(self) -> None:
        from multimodal_pipeline.orchestrator import STAGE_CLASSES

        assert STAGE_CLASSES["stories"] is StoriesStage

    def test_the_coverage_inventory_still_explains_the_tables(self) -> None:
        from multimodal_pipeline import elan

        reason = elan.TIER_ABSENT_REASONS["stories"]
        assert "narrative" in reason
        assert reason != elan.COVERAGE_REASON_UNKNOWN
        assert "no tier" in reason
        # Falsifiable half: the wording must name the tables it is about.
        assert "storytelling_events" in reason


# ------------------------------------------------------------------ clients


class TestClients:
    def test_the_mock_client_is_deterministic_and_valid(self) -> None:
        request = request_for()
        first = MockStoriesClient(model="m").detect(request)
        second = MockStoriesClient(model="m").detect(request)
        assert first.content == second.content
        assert [row["event_id"] for row in first.events] == ["w0-SE1"]
        assert [row["story_id"] for row in first.stories] == ["w0-S1"]

    def test_the_mock_client_never_quotes_the_unknown_marker_or_a_blank(self) -> None:
        """The ``[unknown]`` marker is a rendering artifact, not transcript text."""
        segments = [segment(1, text=" "), segment(2, text="   "), segment(3, text="real one")]
        request = request_for(segments, words=[], speaker_source="none")
        verdict = MockStoriesClient().detect(request)
        # One quotable segment cannot hold an event and a story, so the honest answer is
        # the empty one rather than a quote of the marker.
        assert verdict.answer.empty
        assert "[unknown]" not in verdict.content

    def test_the_mock_client_quotes_only_real_text(self) -> None:
        segments = [segment(1, text=" "), segment(2, text="a real line here"),
                    segment(3, text="another real line")]
        request = request_for(segments, words=[], speaker_source="none")
        verdict = MockStoriesClient().detect(request)
        assert not verdict.answer.empty
        assert "[unknown]" not in verdict.content
        assert "a real line here" in verdict.content

    def test_the_mock_client_is_empty_below_two_segments(self) -> None:
        request = request_for(SEGMENTS[:1], words=WORDS)
        verdict = MockStoriesClient().detect(request)
        assert verdict.answer.empty
        assert json.loads(verdict.content) == {"storytelling_events": [], "stories": [],
                                               "uncertain_spans": []}

    def test_the_real_client_is_built_from_the_config(self, context) -> None:
        use_stage_config(context, provider="openai-compatible",
                         base_url="https://x.invalid/v1", api_key="sk-test", model="chat",
                         max_retries=1, backoff_base_seconds=0.25, timeout_seconds=7.0)
        client = StoriesStage()._client(context)
        assert isinstance(client, StoriesClient)
        assert client.model == "chat"
        assert client.max_retries == 1
        assert client.backoff_base == 0.25
        assert client.timeout_seconds == 7.0

    def test_a_good_answer_comes_back_from_a_real_server(self, endpoint) -> None:
        request = request_for()
        server, state = endpoint([(200, ok_body(envelope(honest_answer(request)),
                                               prompt_tokens=610,
                                               completion_tokens=72))])
        client = stories_client(server)
        verdict = client.detect(request_for())
        assert [row["event_id"] for row in verdict.events] == ["w0-SE1"]
        assert client.usage == {"prompt_tokens": 610, "completion_tokens": 72}
        body = state.bodies[0]
        assert body["model"] == "chat" and body["temperature"] == 0.0
        assert state.headers[0]["authorization"] == "Bearer sk-test-123"

    def test_a_5xx_is_retried_and_a_4xx_is_not(self, endpoint) -> None:
        slept: list[float] = []
        server, _state = endpoint([(503, "upstream down"), (400, "bad request")])
        client = stories_client(server, max_retries=2, backoff_base_seconds=0.5,
                                sleep=slept.append)
        with pytest.raises(StageError, match="HTTP 400"):
            client.detect(request_for())
        assert client.request_count == 2
        assert len(slept) == 1 and slept[0] >= 0.5

    def test_a_malformed_answer_is_resampled_by_the_client(self, endpoint) -> None:
        server, _state = endpoint([(200, ok_body("I cannot help with that.")),
                                   (200, ok_body(envelope(answer())))])
        client = stories_client(server, max_retries=2)
        verdict = client.detect(request_for())
        assert client.request_count == 2
        assert verdict.answer.empty

    def test_a_rejected_answer_is_retried_with_the_problems_as_feedback(self, endpoint) -> None:
        """Resampling the identical prompt is not a retry plan: say what was wrong.

        The second body carries the original question, the rejected answer as an
        ``assistant`` message and the validator's problems in a following ``user`` message.
        """
        request = request_for()
        broken = answer(events=[event_payload(request)], stories=[story_payload(request)])
        broken["stories"][0]["evidence_segment_ids"] = ["seg999999"]
        server, state = endpoint([(200, ok_body(envelope(broken))),
                                  (200, ok_body(envelope(answer())))])
        client = stories_client(server, max_retries=2)
        client.detect(request)
        assert client.request_count == 2, "the repaired answer was accepted on attempt 2"
        first, second = state.bodies
        assert first["messages"] == [{"role": "user", "content": request.prompt()}]
        assert [message["role"] for message in second["messages"]] == [
            "user", "assistant", "user"]
        assert second["messages"][0]["content"] == request.prompt()
        assert second["messages"][1]["content"] == envelope(broken)
        feedback = second["messages"][2]["content"]
        assert "seg999999" in feedback
        assert "return the corrected answer" in feedback
        # Everything except the conversation is the same request.
        assert {key: value for key, value in first.items() if key != "messages"} == {
            key: value for key, value in second.items() if key != "messages"}

    def test_a_long_rejected_answer_is_omitted_from_the_repair(self, endpoint) -> None:
        """A repair is a second request, so the rejected answer has a size limit.

        Resending a very long answer is what pushed the follow-up past the endpoint's
        context limit; the problems alone still name every failed check.
        """
        request = request_for()
        long_answer = "I cannot help with that. " * 30
        assert len(long_answer) > 200
        server, state = endpoint([(200, ok_body(long_answer)),
                                  (200, ok_body(envelope(answer())))])
        client = stories_client(server, max_retries=1, repair_max_answer_chars=200)
        verdict = client.detect(request)
        assert verdict.answer.empty
        repair = state.bodies[1]
        assert [message["role"] for message in repair["messages"]] == ["user", "user"]
        assert repair["messages"][0]["content"] == request.prompt()
        assert "return the corrected answer" in repair["messages"][1]["content"]

    def test_a_short_rejected_answer_is_still_resent_in_the_repair(self, endpoint) -> None:
        request = request_for()
        broken = answer(events=[event_payload(request)], stories=[story_payload(request)])
        broken["stories"][0]["evidence_segment_ids"] = ["seg999999"]
        server, state = endpoint([(200, ok_body(envelope(broken))),
                                  (200, ok_body(envelope(answer())))])
        client = stories_client(server, max_retries=1, repair_max_answer_chars=20000)
        client.detect(request)
        assert [message["role"] for message in state.bodies[1]["messages"]] == [
            "user", "assistant", "user"]

    def test_the_repair_feedback_is_length_bounded(self) -> None:
        """The problem list is part of a request too: it is bounded, and says so."""
        limit = stories_module._REPAIR_MAX_PROBLEMS
        message = feedback_message([f"problem {index}" for index in range(200)])
        assert f"- problem {limit - 1}\n" in message
        assert f"- problem {limit}\n" not in message
        assert f"{200 - limit} further problem(s) omitted" in message
        huge = feedback_message(["x" * 10_000])
        assert len(huge) < stories_module._REPAIR_MAX_PROBLEM_CHARS + 1_000
        assert "return the corrected answer" in huge

    def test_the_truncation_notice_fits_inside_the_cap(self) -> None:
        message = feedback_message(["x" * 10_000])
        listed = message.split("Problems:\n", 1)[1]
        assert len(listed) <= stories_module._REPAIR_MAX_PROBLEM_CHARS
        assert listed.endswith("(further problem text omitted for length)")

    @pytest.mark.parametrize("status", [400, 413])
    def test_a_repair_the_endpoint_refuses_for_length_falls_back_to_the_plain_body(
            self, endpoint, status: int) -> None:
        """A context-limit error on the follow-up must not end the sequence.

        The endpoint judged the repair too large, not the question. Resending the same
        repair would be refused again, so the next attempt is the plain original body —
        and the refusal is counted instead of disappearing into the retry history.
        """
        request = request_for()
        broken = answer(events=[event_payload(request)], stories=[story_payload(request)])
        broken["stories"][0]["evidence_segment_ids"] = ["seg999999"]
        server, state = endpoint([(200, ok_body(envelope(broken))),
                                  (status, "request too large"),
                                  (200, ok_body(envelope(answer())))])
        client = stories_client(server, max_retries=2)
        verdict = client.detect(request)
        assert client.repair_fallbacks == 1
        assert client.request_count == 3
        assert [message["role"] for message in state.bodies[1]["messages"]] == [
            "user", "assistant", "user"]
        assert [message["role"] for message in state.bodies[2]["messages"]] == ["user"]
        assert state.bodies[2]["messages"][0]["content"] == request.prompt()
        assert verdict.answer.empty

    def test_the_stage_records_a_repair_the_endpoint_refused(self, context, endpoint) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        request = request_for()
        broken = answer(events=[event_payload(request)], stories=[story_payload(request)])
        broken["stories"][0]["evidence_segment_ids"] = ["seg999999"]
        server, _state = endpoint([(200, ok_body(envelope(broken))),
                                   (413, "request too large"),
                                   (200, ok_body(envelope(honest_answer(request))))])
        stage = StoriesStage()
        stage._client = lambda _ctx: stories_client(server, max_retries=2)
        assert stage.run(context).status == "completed"
        assert summary(context)["repair_fallbacks"] == 1

    def test_a_fixable_answer_converges_on_the_retry(self, endpoint) -> None:
        """The first attempt invents a segment id; the repaired one is accepted."""
        request = request_for()
        broken = answer(events=[event_payload(request)], stories=[story_payload(request)])
        broken["stories"][0]["evidence_segment_ids"] = ["seg999999"]
        fixed = answer(events=[event_payload(request)], stories=[story_payload(request)])
        server, _state = endpoint([(200, ok_body(envelope(broken))),
                                   (200, ok_body(envelope(fixed)))])
        client = stories_client(server, max_retries=1)
        verdict = client.detect(request)
        assert client.request_count == 2
        assert [row["event_id"] for row in verdict.events] == ["w0-SE1"]
        assert json.loads(verdict.content) == fixed

    def test_a_transport_error_retries_the_unchanged_body(self, endpoint) -> None:
        """A broken connection is not the model's mistake, so it earns no feedback."""
        request = request_for()
        server, state = endpoint([(503, "upstream down"),
                                  (200, ok_body(envelope(answer())))])
        client = stories_client(server, max_retries=1)
        client.detect(request)
        assert len(state.bodies) == 2
        assert state.bodies[0] == state.bodies[1]
        assert state.bodies[1]["messages"] == [{"role": "user", "content": request.prompt()}]

    def test_an_empty_message_is_a_model_error_retried_with_feedback(self, endpoint,
                                                                    tmp_path) -> None:
        """No content is an empty answer, not a dead endpoint.

        It used to escape as a transport failure on the first attempt: the answer was never
        read, never kept and never repaired.
        """
        request = request_for()
        server, state = endpoint([(200, {"choices": [{"message": {"content": "   "}}]}),
                                  (200, ok_body(envelope(answer())))])
        client = stories_client(server, max_retries=1)
        client.rejected_dir = tmp_path / "rejected"
        client.rejected_dir.mkdir()
        verdict = client.detect(request)
        assert verdict.answer.empty
        assert client.request_count == 2
        assert len(state.bodies[1]["messages"]) == 2  # no assistant message: there was none
        assert "no message content" in state.bodies[1]["messages"][-1]["content"]

    def test_a_rejection_file_is_unique_and_redacted(self, endpoint, tmp_path) -> None:
        request = request_for()
        leaked = ("I cannot help. My key is sk-test-123 and I saw "
                  "authorization: Bearer sk-test-123 too.")
        server, _state = endpoint([(200, ok_body(leaked)), (200, ok_body(leaked)),
                                   (200, ok_body(envelope(answer())))])
        client = stories_client(server, max_retries=2)
        client.rejected_dir = tmp_path / "rejected"
        client.rejected_dir.mkdir()
        client.detect(request)
        files = sorted(client.rejected_dir.iterdir())
        assert len(files) == 2, files
        assert files[0].name != files[1].name
        for path in files:
            stored = path.read_text(encoding="utf-8")
            assert "sk-test-123" not in stored
            assert "[redacted]" in stored
            assert "attempt" in stored

    def test_a_401_excerpt_is_redacted_before_it_is_raised(self, endpoint) -> None:
        """The response excerpt goes to the log and the report, so it is redacted first.

        Both credentials are removed: ours, echoed back, and a *different* one quoted in a
        JSON field the gateway serialised — a provider that echoes its own configuration
        leaks that one, and it is not ours to publish.
        """
        echoed = ('{"error": "invalid_api_key", "authorization": "Bearer sk-test-123", '
                  '"api_key": "sk-other-999", "token": "t-other-123"}')
        server, _state = endpoint([(401, echoed)])
        client = stories_client(server, max_retries=0)
        with pytest.raises(StageError, match="HTTP 401") as excinfo:
            client.detect(request_for())
        message = str(excinfo.value) + str(excinfo.value.details)
        assert "sk-test-123" not in message
        assert "sk-other-999" not in message
        assert "t-other-123" not in message
        assert "[redacted]" in message

    def test_a_credential_cut_by_the_excerpt_or_escaped_is_still_redacted(
            self, endpoint) -> None:
        """Redaction runs on the whole body before the excerpt is cut.

        Cutting first split a JSON credential at the 400-character mark, and the half that
        survived no longer matched the pattern; a value with an escaped quote stopped the
        pattern early and left its tail visible.
        """
        padding = "x" * 380
        echoed = ('{"error": "' + padding + '", "api_key": "sk-cut-ABCDEFGHIJKLMNOP", '
                  '"token": "esc\\"aped-TAILSECRET"}')
        server, _state = endpoint([(401, echoed)])
        client = stories_client(server, max_retries=0)
        with pytest.raises(StageError, match="HTTP 401") as excinfo:
            client.detect(request_for())
        message = str(excinfo.value) + str(excinfo.value.details)
        assert "sk-cut-" not in message
        assert "ABCDEFGH" not in message
        client2 = stories_client(server, max_retries=0)
        assert "TAILSECRET" not in client2.redact('{"token": "esc\\"aped-TAILSECRET"}')
        assert "unterminated" not in client2.redact('{"api_key": "unterminated-value')

    def test_the_api_key_never_appears_in_a_failure_report(self) -> None:
        client = StoriesClient(base_url="http://127.0.0.1:1/v1", api_key="sk-secret-123456",
                               model="chat", max_retries=0, sleep=lambda _s: None)
        with pytest.raises(StageError) as excinfo:
            client.detect(request_for())
        assert "sk-secret-123456" not in str(excinfo.value)
        assert "sk-secret-123456" not in str(excinfo.value.details)

    def test_an_exhausted_client_becomes_a_stage_error(self) -> None:
        client = StoriesClient(base_url="http://127.0.0.1:1/v1", api_key="", model="chat",
                               max_retries=1, backoff_base_seconds=0.0,
                               sleep=lambda _seconds: None)
        with pytest.raises(StageError, match="failed after 2 attempts") as excinfo:
            client.detect(request_for())
        assert client.request_count == 2
        assert excinfo.value.details["window_index"] == 0

    def test_a_transport_failure_propagates_out_of_execute(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        client = FakeClient([StoriesTransportError("HTTP 503: upstream unavailable")])
        stage = StoriesStage()
        stage._client = lambda _ctx: client  # type: ignore[method-assign]
        with pytest.raises(StoriesTransportError, match="503"):
            stage.run(context)
        assert not context.artifact("stories").exists()
        assert not context.artifact("storytelling_events").exists()
