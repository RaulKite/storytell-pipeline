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
pyannote.

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
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
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
from multimodal_pipeline.stages.base import STAGE_DEPENDENCIES, STAGE_ORDER
from multimodal_pipeline.stages.stories import (
    PROMPTS,
    MockStoriesClient,
    StoriesClient,
    StoriesStage,
    StoriesTransportError,
    build_requests,
    locate_quote,
    parse_stories_response,
    prepare_segments,
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
        "story_evidence": ["seg000001", "seg000002"],
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


class TestPrompt:
    def test_v2_is_the_only_prompt_version(self) -> None:
        assert list(PROMPTS) == ["v2"]

    def test_the_goal_sentence_is_the_domain_experts(self) -> None:
        prompt = PROMPTS["v2"]
        assert ("Your goal is to mark the boundaries of each storytelling event and, "
                "separately, the boundaries of each story told within it.") in prompt
        assert ('A story is the "what" being represented; a storytelling event is the '
                'conversational activity of launching and telling it.') in prompt

    def test_the_working_criteria_and_boundary_guidance_are_present(self) -> None:
        prompt = PROMPTS["v2"]
        lowered = prompt.lower()
        for phrase in ("foregrounds experience", "not automatically a story",
                       "pre-telling", "coda", "listener_acknowledgment", "activity_shift",
                       "interruptions", "never invent words", "uncertain"):
            assert phrase in lowered, phrase

    def test_the_emptiness_rule_stays_prominent(self) -> None:
        assert ("An empty answer is the expected and correct answer" in PROMPTS["v2"])
        assert "empty arrays" in PROMPTS["v2"]

    def test_the_input_format_is_ours_not_whisperx_json(self) -> None:
        prompt = PROMPTS["v2"]
        assert "WhisperX" not in prompt
        assert "segment_id=" in prompt
        assert "[speaker label]" in prompt
        assert "Do not output times" in prompt
        assert "transcript_quote" in prompt
        assert ("Do not infer pauses, overlap, intonation, gaze or gesture"
                in prompt)

    def test_the_output_contract_keeps_the_experts_field_names(self) -> None:
        prompt = PROMPTS["v2"]
        for field in ("storytelling_events", "story_ids", "interruptions_or_resumptions",
                      "tellers", "story_evidence", "uncertain_spans", "summary", "issue"):
            assert field in prompt, field
        assert "time_seconds" not in prompt

    def test_the_transcript_lines_show_ids_boundaries_and_speaker_runs(self) -> None:
        prompt = request_for(speaker_source="pyannote").prompt()
        assert "VIDEO: conversation_001" in prompt
        for row in SEGMENTS:
            assert row["segment_id"] in prompt
            assert f"{float(row['start_time']):.3f}" in prompt
        assert "[SPEAKER_00] I was fiddling with my wedding ring" in prompt
        assert "[SPEAKER_01] it rolled down the side of the sofa" in prompt

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

    def test_the_key_moves_with_the_speaker_source_and_its_thresholds(self) -> None:
        """A relabelled transcript is a different question, not the same one re-asked."""
        base = request_for(speaker_source="pyannote").key()
        assert request_for(speaker_source="none").key() != base
        assert request_for(speaker_source="pyannote",
                           speaker_min_coverage=0.9).key() != base
        assert request_for(speaker_source="pyannote",
                           speaker_max_competing=0.01).key() != base

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

    def test_character_offsets_are_used_when_they_index_into_the_text(self) -> None:
        words = word_rows(SEGMENTS)
        text = SEGMENTS[0]["text"]
        offset = 0
        for row in words:
            if row["segment_id"] == "seg000001":
                row["character_start"] = text.index(row["word"], offset)
                row["character_end"] = row["character_start"] + len(row["word"])
                offset = row["character_end"]
        request = request_for(rows=prepared(words=words))
        validated = validate_response(honest_answer(request), request)
        assert validated.stories[0]["start_time"] == pytest.approx(3.0 + 3 * (2.5 / 7))

    def test_offsets_that_do_not_index_into_the_text_are_ignored(self) -> None:
        """The real corpus has NULL offsets; a *wrong* one must not be trusted either.

        Every column offset here points at character 0, so the slice never equals the word
        it claims to be. The ordered scan has to place the words anyway.
        """
        words = [dict(row, character_start=0, character_end=3) for row in WORDS]
        request = request_for(rows=prepared(words=words))
        validated = validate_response(honest_answer(request), request)
        assert validated.stories[0]["start_time"] == pytest.approx(3.0 + 3 * (2.5 / 7))

    def test_a_word_the_text_does_not_contain_gets_no_timing(self) -> None:
        words = [dict(row, word="GONE") if row["segment_id"] == "seg000001" else row
                 for row in WORDS]
        request = request_for(rows=prepared(words=words))
        # No word from the first segment can be attributed, so the start boundary is the
        # segment's own boundary and says so.
        event = validate_response(honest_answer(request), request).events[0]
        assert event["start_time_precision"] == "segment"


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

    def test_an_invented_story_evidence_id_is_refused(self) -> None:
        request = request_for()
        payload = answer(events=[event_payload(request)],
                         stories=[story_payload(request,
                                                story_evidence=["seg000001", "nope"])])
        with pytest.raises(ValidationError, match="story_evidence"):
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
        assert {item["kind"] for item in validated.dropped} == {"event", "story"}
        assert any("seg000003" in item["reason"] for item in validated.dropped)

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

    def test_an_uncertain_span_entirely_outside_the_window_is_dropped(self) -> None:
        request = self.windowed()[0]
        payload = answer(uncertain=[{
            "start_segment_id": "seg000003", "end_segment_id": "seg000003",
            "transcript_quote": SEGMENTS[2]["text"], "issue": "outside"}])
        validated = validate_response(payload, request)
        assert validated.uncertain_spans == []
        assert [item["kind"] for item in validated.dropped] == ["uncertain_span"]


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

    def test_pyannote_uses_the_word_id_and_unknown_only_for_a_null_one(self) -> None:
        words = [dict(row, speaker_id=None) if row["segment_id"] == "seg000002" else row
                 for row in WORDS]
        rows = prepared(words=words, speaker_source="pyannote")
        assert rows[0]["speaker_labels"] == ["SPEAKER_00"]
        assert rows[1]["speaker_labels"] == ["unknown"]

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

    def test_never_falls_back_to_pyannote(self, context) -> None:
        """Pyannote words exist; a missing Nemotron table must still not use them."""
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
        assert events[0]["schema_version"] == "2.0"
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
        assert extra["dropped_outside_window"] == 2  # the event and its story
        assert rows_of(context, "storytelling_events") == []
        assert any("dropped" in message for message in messages)
        recorded = summary(context)
        assert recorded["dropped_outside_window"] == 2
        assert recorded["dropped_detail"][0]["window_index"] == 0

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
        use_stage_config(context, speaker_source="pyannote")
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
                "summary", "tellers", "confidence", "notes"} <= set(names)

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
