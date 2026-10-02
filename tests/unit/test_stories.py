"""`stories`: narrative-window detection with the pipeline's own LLM (T16, ODD §20.3).

Three things are tested, and each has a different failure it exists to catch.

**The rejection path** (`TestValidation`). The probe measured that this endpoint answers
with real ``segment_id``s and real boundary times on a clip that has a story, and with an
empty list plus a stated reason on two that do not (`odd/tasks/stories-detection.md`). So a
parser that accepted an invented id or a fabricated timestamp would not be "lenient", it
would be writing fiction into the corpus: the whole point of the table is that a human can
check a story against the transcript. Every rejection test below names the one breach it
refuses, and each guard was broken once to confirm the test dies.

**Emptiness as a result** (`TestEmptyIsAResult`). §20.3 requires the output be allowed to
be empty *and* something to count how often it is. An empty list with no reason is refused
— that is the difference between "the model looked and found nothing" and "the model said
nothing at all".

**The stage contract** (`TestStageExecution` and below): windowing, the digest cache, raw
preserved byte-identical, the fingerprint noticing a transcript change. The endpoint is
never contacted: the mock provider and a fake client do the work, so these tests run
offline and the retry/transport code is exercised through the fake's exceptions instead.
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
    STORIES_SCHEMA,
    TABLE_SCHEMAS,
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
    parse_stories_response,
    validate_stories,
)


# ------------------------------------------------------------------ builders


def segment(index: int, *, text: str | None = None, speaker: str = "SPEAKER_00",
            start: float | None = None, end: float | None = None) -> dict[str, Any]:
    """One transcript row in the shape `speech/segments.parquet` really has.

    Times step by 3 s so every boundary is distinct at 3-decimal rounding, and the ids
    match the ``seg%06d`` the assignment stage writes.
    """
    start = float(index) * 3.0 if start is None else start
    end = start + 2.5 if end is None else end
    return {
        "schema_version": "1.0",
        "video_id": "conversation_001",
        "segment_id": f"seg{index:06d}",
        "start_time": start,
        "end_time": end,
        "duration": round(end - start, 6),
        "language": "es",
        "speaker_id": speaker,
        "text": text or f"line number {index}",
        "confidence": 0.9,
        "speaker_overlap_seconds": 0.0,
        "speaker_overlap_ratio": 0.0,
        "speaker_assignment_method": "pyannote",
    }


SEGMENTS = [segment(1, speaker="SPEAKER_00"), segment(2, speaker="SPEAKER_01"),
            segment(3, speaker="SPEAKER_00")]


def request_for(segments: list[dict[str, Any]], index: int = 0, **cfg: Any):
    """The window request a stage would send for these segments."""
    options: dict[str, Any] = {"max_segments_per_request": 60, "context_segments": 5,
                              "prompt_version": "v1", "model": "chat", "temperature": 0.0,
                              "max_output_tokens": None}
    options.update(cfg)
    requests = build_requests(segments, video_id="conversation_001", **options)
    return requests[index]


def story_payload(**overrides: Any) -> dict[str, Any]:
    """A story citing the first two segments of the default fixture."""
    payload: dict[str, Any] = {
        "story_id": "s1",
        "parent_id": None,
        "start_time": SEGMENTS[0]["start_time"],
        "end_time": SEGMENTS[1]["end_time"],
        "title": "A recalled evening",
        "why_it_is_a_story": "It opens on a memory and closes on its point.",
        "evidence_segment_ids": ["seg000001", "seg000002"],
        "confidence": 0.9,
    }
    payload.update(overrides)
    return payload


def envelope(*stories: dict[str, Any], reason: str | None = None) -> str:
    """The JSON text the endpoint returns for this contract."""
    return json.dumps({"stories": list(stories), "no_story_reason": reason})


class FakeClient:
    """A client-shaped object handing back scripted content, counting the calls.

    Each entry is either a string (returned as the message content) or an Exception to
    raise. ``request_count`` is what the cache test reads.
    """

    def __init__(self, responses: list[Any], *, model: str = "chat") -> None:
        self.responses = list(responses)
        self.model = model
        self.request_count = 0
        self.timings_ms: list[float] = []
        self.usage: dict[str, int] = {"prompt_tokens": 10}
        self.prompts: list[str] = []

    def detect(self, request):
        self.request_count += 1
        self.timings_ms.append(0.0)
        self.prompts.append(request.prompt())
        outcome = self.responses.pop(0) if self.responses else self.responses
        if isinstance(outcome, Exception):
            raise outcome
        from multimodal_pipeline.stages.stories import StoriesVerdict

        payload = parse_stories_response(outcome)
        stories, dropped = validate_stories(payload, request)
        return StoriesVerdict(content=outcome, payload=payload, stories=stories,
                              dropped=dropped)


def seed_segments(context, rows: list[dict[str, Any]]) -> None:
    write_table(context.artifact("speech_segments"),
                pa.Table.from_pylist(rows, schema=SEGMENTS_SCHEMA), SEGMENTS_SCHEMA)


def use_stage_config(context, **stories_cfg: Any) -> None:
    """A stories section that runs offline by default."""
    values: dict[str, Any] = {"provider": "mock"}
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


def stories_rows(context) -> list[dict[str, Any]]:
    return read_table(context.artifact("stories")).to_pylist()


# ------------------------------------------------------------------ local endpoint
#
# The client is exercised against a real HTTP server rather than a patched httpx, the way
# tests/unit/test_translation.py does it: status codes, headers, retry order and body
# shape are then actually tested instead of assumed.


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
                status, payload = state.responses.pop(0) if state.responses else (200, ok_body("{}"))
                raw = json.dumps(payload).encode() if not isinstance(payload, str) else payload.encode()
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
    def test_the_prompt_offers_version_v1(self) -> None:
        assert "v1" in PROMPTS

    def test_the_window_and_its_context_are_both_rendered(self) -> None:
        request = request_for(SEGMENTS, max_segments_per_request=2, context_segments=1)
        prompt = request.prompt()
        assert "REQUESTED" in prompt and "CONTEXT" in prompt
        # The window's own ids are the ones it may cite; the neighbour is context.
        assert "seg000001" in prompt and "seg000002" in prompt
        assert "seg000003" in prompt
        assert "conversation_001" in prompt

    def test_the_prompt_shows_ids_speakers_and_times(self) -> None:
        # §20.3: boundaries are referential, so the model has to see timestamps and
        # speakers, not a wall of plain text.
        prompt = request_for(SEGMENTS).prompt()
        for row in SEGMENTS:
            assert row["segment_id"] in prompt
            assert f"{row['start_time']:.3f}" in prompt
            assert f"{row['end_time']:.3f}" in prompt
            assert row["speaker_id"] in prompt
            assert row["text"] in prompt

    def test_the_emptiness_rule_is_stated(self) -> None:
        # The probe's central measurement rests on this wording: empty must be allowed
        # to be the *correct* answer, or the model stretches a fragment into a story.
        prompt = PROMPTS["v1"]
        assert "no_story_reason" in prompt
        assert "empty list is the expected and correct answer" in prompt

    def test_an_unknown_prompt_version_fails_loudly(self) -> None:
        request = request_for(SEGMENTS, prompt_version="nope")
        with pytest.raises(StageError, match="prompt version"):
            request.prompt()

    def test_the_key_moves_with_text_model_prompt_and_temperature(self) -> None:
        base = request_for(SEGMENTS).key()
        assert request_for(SEGMENTS, model="other").key() != base
        assert request_for(SEGMENTS, temperature=0.7).key() != base
        assert request_for(SEGMENTS, prompt_version="v1").key() == base
        changed = [dict(SEGMENTS[0], text="a different line")] + SEGMENTS[1:]
        assert request_for(changed).key() != base
        # A different context window is a different request, as in translation. Observable
        # only once there is more than one window: a single window covering the whole
        # transcript has no context either way, so its key correctly does not move.
        many = [segment(i) for i in range(1, 8)]

        def middle_key(**kwargs: Any) -> str:
            return build_requests(many, video_id="v", max_segments_per_request=3,
                                 prompt_version="v1", model="m", temperature=0.0,
                                 max_output_tokens=None, **kwargs)[1].key()

        assert middle_key(context_segments=1) != middle_key(context_segments=0)


# ------------------------------------------------------------------ windowing


class TestWindowing:
    def test_windows_partition_the_transcript_in_start_order(self) -> None:
        segments = [segment(i) for i in range(1, 8)]
        requests = build_requests(segments, video_id="v", max_segments_per_request=3,
                                  context_segments=1, prompt_version="v1", model="m",
                                  temperature=0.0, max_output_tokens=None)
        assert [r.requested_ids for r in requests] == [
            ["seg000001", "seg000002", "seg000003"],
            ["seg000004", "seg000005", "seg000006"],
            ["seg000007"],
        ]
        assert [r.window_index for r in requests] == [0, 1, 2]

    def test_context_never_overlaps_the_window(self) -> None:
        segments = [segment(i) for i in range(1, 8)]
        requests = build_requests(segments, video_id="v", max_segments_per_request=3,
                                  context_segments=2, prompt_version="v1", model="m",
                                  temperature=0.0, max_output_tokens=None)
        for request in requests:
            assert not set(request.requested_ids) & {r["segment_id"] for r in request.context}
        assert [len(r.context) for r in requests] == [2, 3, 2]

    def test_ids_outside_the_window_are_still_known_transcript_ids(self) -> None:
        # The validator needs both sets: an id from another window is a *drop*, an id
        # that exists nowhere is a fabrication and a rejection.
        segments = [segment(i) for i in range(1, 8)]
        request = build_requests(segments, video_id="v", max_segments_per_request=3,
                                 context_segments=1, prompt_version="v1", model="m",
                                 temperature=0.0, max_output_tokens=None)[0]
        assert request.all_segment_ids == {f"seg{i:06d}" for i in range(1, 8)}
        assert "seg000007" not in request.requested_ids

    def test_an_empty_transcript_makes_no_window(self) -> None:
        assert build_requests([], video_id="v", max_segments_per_request=3,
                             context_segments=1, prompt_version="v1", model="m",
                             temperature=0.0, max_output_tokens=None) == []

    def test_a_non_positive_window_size_is_refused(self) -> None:
        with pytest.raises(StageError, match="max_segments_per_request"):
            build_requests(SEGMENTS, video_id="v", max_segments_per_request=0,
                           context_segments=1, prompt_version="v1", model="m",
                           temperature=0.0, max_output_tokens=None)

    def test_story_ids_are_prefixed_per_window(self) -> None:
        segments = [segment(i) for i in range(1, 8)]
        requests = build_requests(segments, video_id="v", max_segments_per_request=3,
                                  context_segments=0, prompt_version="v1", model="m",
                                  temperature=0.0, max_output_tokens=None)
        assert [r.id_prefix for r in requests] == ["w0", "w1", "w2"]


# ------------------------------------------------------------------ parsing


class TestParsing:
    def test_a_plain_object_parses(self) -> None:
        assert parse_stories_response('{"stories": [], "no_story_reason": "x"}') == {
            "stories": [], "no_story_reason": "x"}

    def test_code_fences_and_prose_are_tolerated(self) -> None:
        assert parse_stories_response('```json\n{"stories": []}\n```') == {"stories": []}
        assert parse_stories_response('Here you go:\n{"stories": []}\nEnjoy.') == {
            "stories": []}

    @pytest.mark.parametrize("text", ["", "not json at all", "[1, 2]", '"word"', "null"])
    def test_a_response_that_is_not_an_object_is_refused(self, text: str) -> None:
        with pytest.raises(ValueError):
            parse_stories_response(text)


# ------------------------------------------------------------------ rejection


class TestValidation:
    def test_a_honest_story_validates(self) -> None:
        request = request_for(SEGMENTS)
        stories, dropped = validate_stories({"stories": [story_payload()]}, request)
        assert len(stories) == 1 and dropped == []
        assert stories[0]["story_id"] == "w0-s1"

    def test_empty_with_a_reason_is_accepted(self) -> None:
        stories, dropped = validate_stories(
            {"stories": [], "no_story_reason": "A single announcement, no narrative arc."},
            request_for(SEGMENTS))
        assert stories == [] and dropped == []

    def test_empty_without_a_reason_is_refused(self) -> None:
        # "Nothing found" must be distinguishable from "nothing said".
        with pytest.raises(ValidationError, match="no_story_reason"):
            validate_stories({"stories": [], "no_story_reason": None}, request_for(SEGMENTS))
        with pytest.raises(ValidationError, match="no_story_reason"):
            validate_stories({"stories": []}, request_for(SEGMENTS))

    def test_stories_must_be_a_list(self) -> None:
        with pytest.raises(ValidationError, match="list"):
            validate_stories({"stories": {"s1": story_payload()}}, request_for(SEGMENTS))

    def test_an_invented_evidence_id_is_refused(self) -> None:
        payload = {"stories": [story_payload(evidence_segment_ids=["seg000001", "seg999999"])]}
        with pytest.raises(ValidationError, match="seg999999"):
            validate_stories(payload, request_for(SEGMENTS))

    def test_an_empty_evidence_list_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="evidence"):
            validate_stories({"stories": [story_payload(evidence_segment_ids=[])]},
                             request_for(SEGMENTS))

    def test_a_fabricated_boundary_is_refused(self) -> None:
        # The probe's rule 1: never invent times. 4.444 is no segment's boundary here.
        with pytest.raises(ValidationError, match="boundary"):
            validate_stories({"stories": [story_payload(start_time=4.444)]},
                             request_for(SEGMENTS))
        with pytest.raises(ValidationError, match="boundary"):
            validate_stories({"stories": [story_payload(end_time=99.999)]},
                             request_for(SEGMENTS))

    def test_a_real_boundary_rounded_to_three_decimals_is_accepted(self) -> None:
        precise = [segment(1, start=1.0004, end=2.0006), segment(2, start=3.0, end=4.0)]
        request = request_for(precise)
        payload = {"stories": [dict(story_payload(), start_time=1.0005, end_time=2.0005,
                                    evidence_segment_ids=["seg000001"])]}
        stories, _dropped = validate_stories(payload, request)
        assert len(stories) == 1

    def test_a_window_that_ends_before_it_starts_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="ends"):
            validate_stories({"stories": [story_payload(start_time=3.0, end_time=3.0)]},
                             request_for(SEGMENTS))
        with pytest.raises(ValidationError, match="ends"):
            validate_stories({"stories": [story_payload(start_time=6.0, end_time=2.5)]},
                             request_for(SEGMENTS))

    @pytest.mark.parametrize("confidence", [-0.1, 1.5])
    def test_confidence_outside_the_unit_interval_is_refused(self, confidence: float) -> None:
        with pytest.raises(ValidationError, match="confidence"):
            validate_stories({"stories": [story_payload(confidence=confidence)]},
                             request_for(SEGMENTS))

    def test_confidence_must_be_a_number(self) -> None:
        with pytest.raises(ValidationError, match="confidence"):
            validate_stories({"stories": [story_payload(confidence="high")]},
                             request_for(SEGMENTS))

    def test_a_duplicated_story_id_is_refused(self) -> None:
        # Parent resolution and the table's story_id column both become meaningless.
        payload = {"stories": [story_payload(), story_payload(title="second")]}
        with pytest.raises(ValidationError, match="duplicate"):
            validate_stories(payload, request_for(SEGMENTS))

    def test_a_parent_must_be_a_story_of_this_batch(self) -> None:
        # The parent of a nested story has to be a story the same answer named; a story
        # from another window lives in another request and cannot be joined here.
        with pytest.raises(ValidationError, match="parent"):
            validate_stories({"stories": [story_payload(parent_id="s7")]},
                             request_for(SEGMENTS))

    def test_a_parent_cycle_is_refused(self) -> None:
        a = story_payload(story_id="s1", parent_id="s2", evidence_segment_ids=["seg000001"])
        b = story_payload(story_id="s2", parent_id="s1", evidence_segment_ids=["seg000002"])
        with pytest.raises(ValidationError, match="cycle"):
            validate_stories({"stories": [a, b]}, request_for(SEGMENTS))

    def test_a_self_parenting_story_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="cycle"):
            validate_stories({"stories": [story_payload(parent_id="s1")]},
                             request_for(SEGMENTS))

    def test_a_real_parent_is_accepted(self) -> None:
        outer = story_payload(story_id="s1", evidence_segment_ids=["seg000001", "seg000002"])
        inner = story_payload(story_id="s2", parent_id="s1", evidence_segment_ids=["seg000002"])
        stories, dropped = validate_stories({"stories": [outer, inner]}, request_for(SEGMENTS))
        assert [s["story_id"] for s in stories] == ["w0-s1", "w0-s2"]
        assert stories[1]["parent_id"] == "w0-s1"

    def test_a_child_of_a_dropped_parent_is_dropped_too_not_a_rejection(self) -> None:
        # The parent's evidence lies in the next window, so the child cannot be placed
        # inside this one. That is a gap to count, not a malformed answer: at temperature 0
        # the model repeats itself, so rejecting the batch over it would retry a fixed
        # answer until the stage failed and the whole video lost its stories.
        request = request_for(SEGMENTS, max_segments_per_request=2, context_segments=1)
        outer = story_payload(story_id="s1", evidence_segment_ids=["seg000003"])
        inner = story_payload(story_id="s2", parent_id="s1",
                              evidence_segment_ids=["seg000001"])
        stories, dropped = validate_stories({"stories": [outer, inner]}, request)
        assert stories == []
        assert {item["model_story_id"] for item in dropped} == {"s1", "s2"}

    def test_a_story_anchored_outside_the_window_is_dropped_not_rejected(self) -> None:
        # The evidence belongs to the next window: that story is *not this window's
        # answer*, which is a known gap to count, not a malformed answer to retry.
        request = request_for(SEGMENTS, max_segments_per_request=2, context_segments=1)
        payload = {"stories": [story_payload(evidence_segment_ids=["seg000003"])]}
        stories, dropped = validate_stories(payload, request)
        assert stories == []
        assert len(dropped) == 1
        assert "seg000003" in dropped[0]["reason"]

    def test_a_boundary_outside_the_window_is_dropped_and_counted(self) -> None:
        request = request_for(SEGMENTS, max_segments_per_request=2, context_segments=1)
        payload = {"stories": [story_payload(evidence_segment_ids=["seg000001"],
                                             end_time=SEGMENTS[2]["end_time"])]}
        stories, dropped = validate_stories(payload, request)
        assert stories == [] and len(dropped) == 1

    def test_several_problems_are_all_reported(self) -> None:
        payload = {"stories": [story_payload(story_id="s1", confidence=5.0),
                              story_payload(story_id="s2", start_time=4.444)]}
        with pytest.raises(ValidationError) as excinfo:
            validate_stories(payload, request_for(SEGMENTS))
        assert len(excinfo.value.issues) >= 2


# ------------------------------------------------------------------ config


class TestConfig:
    def test_the_defaults_are_the_probe_settings(self, context) -> None:
        cfg = context.config.stories
        assert cfg.enabled is True
        assert cfg.provider == "openai-compatible"
        assert cfg.temperature == 0.0
        assert cfg.timeout_seconds == 120.0
        assert cfg.max_retries == 3
        assert cfg.backoff_base_seconds == 1.0
        assert cfg.prompt_version == "v1"
        assert cfg.max_output_tokens is None
        assert cfg.cache is True
        assert cfg.max_segments_per_request == 60
        assert cfg.context_segments == 5
        assert cfg.extra_body == {}

    def test_an_unconfigured_endpoint_degrades_to_a_skip_not_a_crash(self, context) -> None:
        stage = StoriesStage()
        seed_segments(context, SEGMENTS)
        enabled, reason = stage.enabled(context)
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

    @pytest.mark.parametrize("provider", ["openai-compatible", "mock"])
    def test_the_declared_providers_are_accepted(self, provider: str) -> None:
        from multimodal_pipeline.config import PipelineConfig

        config = PipelineConfig.model_validate({
            "input": {"directory": "/tmp/in"}, "output": {"directory": "/tmp/out"},
            "stories": {"provider": provider}})
        assert config.stories.provider == provider

    def test_an_unknown_provider_is_refused(self) -> None:
        import pydantic

        from multimodal_pipeline.config import PipelineConfig

        with pytest.raises(pydantic.ValidationError, match="stories.provider"):
            PipelineConfig.model_validate({
                "input": {"directory": "/tmp/in"}, "output": {"directory": "/tmp/out"},
                "stories": {"provider": "anthropic"}})

    def test_a_zero_window_size_is_refused_at_parse_time(self) -> None:
        import pydantic

        from multimodal_pipeline.config import PipelineConfig

        with pytest.raises(pydantic.ValidationError, match="max_segments_per_request"):
            PipelineConfig.model_validate({
                "input": {"directory": "/tmp/in"}, "output": {"directory": "/tmp/out"},
                "stories": {"max_segments_per_request": 0}})


# ------------------------------------------------------------------ execution


class TestStageExecution:
    def test_the_mock_client_through_execute_writes_one_prefixed_story(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        extra = run_stage(context)
        rows = stories_rows(context)
        assert len(rows) == 1
        row = rows[0]
        # The mock answers with "s1"; the table namespaced it to its window.
        assert row["story_id"] == "w0-s1"
        assert row["parent_id"] is None
        assert row["start_time"] == pytest.approx(SEGMENTS[0]["start_time"])
        assert row["end_time"] == pytest.approx(SEGMENTS[2]["end_time"])
        assert json.loads(row["evidence_segment_ids"]) == [s["segment_id"] for s in SEGMENTS]
        assert row["window_index"] == 0
        assert row["prompt_version"] == context.config.stories.prompt_version
        assert row["video_id"] == "conversation_001"
        assert row["request_key"]
        assert extra["stories_total"] == 1
        assert extra["windows"] == 1
        assert extra["empty_window_count"] == 0
        assert extra["dropped_outside_window"] == 0
        assert extra["batches_reused"] == 0

    def test_a_single_segment_window_gets_the_mock_empty_answer(self, context) -> None:
        use_stage_config(context, max_segments_per_request=1)
        seed_segments(context, SEGMENTS[:1])
        extra = run_stage(context)
        assert stories_rows(context) == []
        assert extra["stories_total"] == 0
        assert extra["empty_window_count"] == 1

    def test_an_empty_answer_writes_no_rows_and_is_counted(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        client = FakeClient([envelope(reason="Two greetings and nothing else.")])
        extra = run_stage(context, client)
        assert stories_rows(context) == []
        assert extra["stories_total"] == 0
        assert extra["empty_window_count"] == 1
        assert extra["windows"] == 1

    def test_the_empty_reason_is_kept_in_the_raw_sidecar(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        reason = "A station announcement: no beginning and no end."
        run_stage(context, FakeClient([envelope(reason=reason)]))
        raw = sorted(context.artifact("stories_raw").glob("window_*.json"))
        assert len(raw) == 1
        assert json.loads(raw[0].read_text())["no_story_reason"] == reason

    def test_the_raw_response_is_preserved_byte_identical(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        content = envelope(story_payload())
        run_stage(context, FakeClient([content]))
        raw = sorted(context.artifact("stories_raw").glob("window_*.json"))
        assert raw[0].read_text() == content

    def test_several_windows_are_numbered_in_order(self, context) -> None:
        use_stage_config(context, max_segments_per_request=2, context_segments=0)
        segments = [segment(i) for i in range(1, 6)]
        seed_segments(context, segments)
        client = FakeClient([
            envelope(story_payload(evidence_segment_ids=["seg000001", "seg000002"],
                                   end_time=segments[1]["end_time"])),
            envelope(reason="Nothing narrative in the middle."),
            envelope(reason="Nothing narrative in the tail."),
        ])
        extra = run_stage(context, client)
        rows = stories_rows(context)
        assert [(row["story_id"], row["window_index"]) for row in rows] == [("w0-s1", 0)]
        assert extra["windows"] == 3  # 5 segments in windows of 2
        assert extra["empty_window_count"] == 2

    def test_a_dropped_story_is_counted_and_logged(self, context) -> None:
        use_stage_config(context, max_segments_per_request=2, context_segments=1)
        seed_segments(context, SEGMENTS)
        client = FakeClient([
            envelope(story_payload(evidence_segment_ids=["seg000003"])),
            envelope(reason="Nothing left."),
        ])
        messages = log_recorder(context)
        extra = run_stage(context, client)
        assert extra["dropped_outside_window"] == 1
        assert stories_rows(context) == []
        assert any("dropped" in message for message in messages)

    def test_the_drop_is_recorded_in_the_raw_summary(self, context) -> None:
        use_stage_config(context, max_segments_per_request=2, context_segments=1)
        seed_segments(context, SEGMENTS)
        run_stage(context, FakeClient([
            envelope(story_payload(evidence_segment_ids=["seg000003"])),
            envelope(reason="Nothing left."),
        ]))
        summary = json.loads((context.artifact("stories_raw") / "stories_summary.json")
                             .read_text())
        assert summary["dropped_outside_window"] == 1
        assert summary["dropped_detail"][0]["window_index"] == 0
        assert "seg000003" in summary["dropped_detail"][0]["reason"]

    def test_the_cache_avoids_a_second_request(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        client = FakeClient([envelope(story_payload()), envelope(story_payload())])
        stage = StoriesStage()
        stage._client = lambda _ctx: client  # type: ignore[method-assign]
        assert stage.run(context).status == "completed"
        assert client.request_count == 1
        assert stage.run(context).status == "completed"
        assert client.request_count == 1, "the cached window was requested again"
        assert len(stories_rows(context)) == 1

    def test_a_reused_batch_is_reported_in_the_summary(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        client = FakeClient([envelope(story_payload()), envelope(story_payload())])
        stage = StoriesStage()
        stage._client = lambda _ctx: client  # type: ignore[method-assign]
        stage.run(context)
        extra = stage.run(context).detail["provenance"]["extra"]
        assert extra["batches_reused"] == 1

    def test_a_changed_transcript_invalidates_the_cache(self, context) -> None:
        # The cache is keyed on window content, so a re-transcribed clip must not reuse
        # the answer given to the old one.
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        client = FakeClient([envelope(story_payload()), envelope(reason="Rewritten.")])
        stage = StoriesStage()
        stage._client = lambda _ctx: client  # type: ignore[method-assign]
        stage.run(context)
        assert len(stories_rows(context)) == 1
        seed_segments(context, [dict(row, text="totally different words") for row in SEGMENTS])
        stage.run(context)
        assert client.request_count == 2
        assert stories_rows(context) == []

    def test_the_cache_is_revalidated_before_reuse(self, context) -> None:
        """A cache entry whose rows no longer fit the window is not reused as-is."""
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        run_stage(context, FakeClient([envelope(story_payload())]))
        cache = sorted((context.artifact("stories_raw") / "cache").glob("*.json"))
        assert len(cache) == 1
        payload = json.loads(cache[0].read_text())
        payload["payload"]["stories"][0]["evidence_segment_ids"] = ["seg999999"]
        cache[0].write_text(json.dumps(payload))
        client = FakeClient([envelope(story_payload())])
        extra = run_stage(context, client)
        assert client.request_count == 1, "a stale cache entry was accepted"
        assert extra["batches_reused"] == 0

    def test_a_reused_window_restores_a_deleted_raw_file(self, context) -> None:
        """Reuse must not leave a table whose evidence has gone missing.

        The cache can outlive a hand-deleted `stories/raw/window_*.json`. Republishing the
        rows with no raw beside them would break the rule that a normalised table is always
        re-validatable against what the tool produced, so the cached response text — the
        endpoint's own bytes, stored alongside the payload — is written back.
        """
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        content = envelope(story_payload())
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
        assert restored[0].read_text() == content
        assert client.request_count == 1

    def test_a_fresh_request_overwrites_the_raw_window_file(self, context) -> None:
        # At temperature > 0 the endpoint can answer the same request differently, so a
        # re-request must not leave the previous response filed beside the new rows.
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        run_stage(context, FakeClient([envelope(reason="first answer.")]))
        cache = sorted((context.artifact("stories_raw") / "cache").glob("*.json"))
        cache[0].unlink()  # as if the cache were cleared but the raw file left behind
        run_stage(context, FakeClient([envelope(reason="second answer, same request.")]))
        raw = sorted(context.artifact("stories_raw").glob("window_*.json"))
        assert len(raw) == 1
        assert json.loads(raw[0].read_text())["no_story_reason"] == "second answer, same request."

    def test_cache_can_be_switched_off(self, context) -> None:
        use_stage_config(context, cache=False)
        seed_segments(context, SEGMENTS)
        client = FakeClient([envelope(story_payload()), envelope(story_payload())])
        stage = StoriesStage()
        stage._client = lambda _ctx: client  # type: ignore[method-assign]
        stage.run(context)
        stage.run(context)
        assert client.request_count == 2

    def test_a_client_that_gives_up_fails_the_stage(self, context) -> None:
        """Retries live in the client; a client that exhausts them fails the stage loudly.

        A silently skipped window would look like "this window has no stories" — the one
        outcome indistinguishable from an honest empty answer. The exception reaches the
        orchestrator, which records the stage as failed and poisons only its dependants.
        """
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

    def test_a_summary_is_written_beside_the_raw_windows(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        run_stage(context, FakeClient([envelope(story_payload())]))
        summary = json.loads((context.artifact("stories_raw") / "stories_summary.json")
                             .read_text())
        assert summary["video_id"] == "conversation_001"
        assert summary["windows"] == 1
        assert summary["requests_made"] == 1
        assert summary["token_usage"] == {"prompt_tokens": 10}

    def test_the_api_key_never_reaches_the_raw_artifacts(self, context) -> None:
        use_stage_config(context, provider="openai-compatible", base_url="https://x.invalid/v1",
                         api_key="sk-secret-123456", model="chat")
        seed_segments(context, SEGMENTS)
        run_stage(context, FakeClient([envelope(story_payload())]))
        blob = "".join(path.read_text() for path in context.artifact("stories_raw").rglob("*")
                       if path.is_file())
        assert "sk-secret-123456" not in blob


# ------------------------------------------------------------------ fingerprint


class TestFingerprint:
    def test_the_fingerprint_carries_the_window_size_and_the_transcript_digest(
            self, context) -> None:
        from multimodal_pipeline.stages.metadata import sha256_of

        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        stage = StoriesStage()
        payload = stage.config_fingerprint(context)
        assert payload["stage"] == "stories"
        assert payload["max_segments_per_request"] == 60
        assert payload["context_segments"] == 5
        assert payload["prompt_version"] == "v1"
        assert payload["segments_digest"] == sha256_of(context.artifact("speech_segments"))

    def test_a_changed_transcript_changes_the_fingerprint(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        stage = StoriesStage()
        before = stage.config_fingerprint(context)
        seed_segments(context, [dict(row, text="a different transcription") for row in SEGMENTS])
        after = stage.config_fingerprint(context)
        assert before != after
        assert before["segments_digest"] != after["segments_digest"]

    def test_the_fingerprint_names_the_python_that_computed_it(self, context) -> None:
        # Same hole `9b5f056` left in pose_normalized: pure-python rows, no worker file,
        # so a parser fix invalidated nothing.
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        digest = StoriesStage().config_fingerprint(context)["_python_code_sha256"]
        assert isinstance(digest, str) and len(digest) == 64

    def test_a_config_change_changes_the_fingerprint(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        stage = StoriesStage()
        before = stage.config_fingerprint(context)
        context.config.stories.max_segments_per_request = 30
        assert stage.config_fingerprint(context) != before


# ------------------------------------------------------------------ validate


class TestValidate:
    def test_the_counts_it_reports_are_the_counts_on_disk(self, context) -> None:
        use_stage_config(context, max_segments_per_request=2, context_segments=0)
        segments = [segment(i) for i in range(1, 6)]
        seed_segments(context, segments)
        client = FakeClient([
            envelope(story_payload(evidence_segment_ids=["seg000001", "seg000002"],
                                   end_time=segments[1]["end_time"])),
            envelope(reason="Middle."),
            envelope(story_payload(story_id="s1", start_time=segments[4]["start_time"],
                                   end_time=segments[4]["end_time"],
                                   evidence_segment_ids=["seg000005"])),
        ])
        run_stage(context, client)
        result = StoriesStage().validate(context)
        assert result == {"stories": 2, "windows": 3, "empty_windows": 1}

    def test_a_missing_table_is_a_validation_failure(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        with pytest.raises(ValidationError, match="missing"):
            StoriesStage().validate(context)

    def test_a_boundary_that_is_no_longer_real_is_refused(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        run_stage(context, FakeClient([envelope(story_payload())]))
        shifted = [dict(row, start_time=row["start_time"] + 7.0,
                        end_time=row["end_time"] + 7.0) for row in SEGMENTS]
        seed_segments(context, shifted)
        with pytest.raises(ValidationError, match="boundary"):
            StoriesStage().validate(context)

    def test_a_parent_that_is_not_a_story_of_the_video_is_refused(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        run_stage(context, FakeClient([envelope(story_payload())]))
        rows = stories_rows(context)
        rows[0]["parent_id"] = "w9-s9"
        write_table(context.artifact("stories"),
                    pa.Table.from_pylist(rows, schema=STORIES_SCHEMA), STORIES_SCHEMA)
        with pytest.raises(ValidationError, match="parent"):
            StoriesStage().validate(context)

    def test_a_duplicated_story_id_is_refused(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        run_stage(context, FakeClient([envelope(story_payload())]))
        rows = stories_rows(context) * 2
        write_table(context.artifact("stories"),
                    pa.Table.from_pylist(rows, schema=STORIES_SCHEMA), STORIES_SCHEMA)
        with pytest.raises(ValidationError, match="duplicate"):
            StoriesStage().validate(context)


# ------------------------------------------------------------------ schema and layout


class TestSchemaAndWiring:
    def test_the_schema_is_flat_and_starts_with_its_version(self) -> None:
        names = [field.name for field in STORIES_SCHEMA]
        assert names[0] == "schema_version"
        assert names == ["schema_version", "video_id", "story_id", "parent_id",
                         "start_time", "end_time", "title", "why_it_is_a_story",
                         "evidence_segment_ids", "confidence", "window_index", "model",
                         "prompt_version", "request_key"]
        assert not any("list" in str(field.type) for field in STORIES_SCHEMA)

    def test_evidence_ids_round_trip_through_the_json_column(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        run_stage(context, FakeClient([envelope(story_payload())]))
        row = stories_rows(context)[0]
        assert json.loads(row["evidence_segment_ids"]) == ["seg000001", "seg000002"]

    def test_the_table_is_registered_under_its_artifact_name(self) -> None:
        assert TABLE_SCHEMAS["stories"] is STORIES_SCHEMA

    def test_the_artifact_paths_live_in_their_own_directory(self) -> None:
        assert ARTIFACT_LAYOUT["stories"] == "stories/stories.parquet"
        assert ARTIFACT_LAYOUT["stories_raw"] == "stories/raw"

    def test_ensure_dirs_creates_the_stories_directories(self, tmp_path) -> None:
        from multimodal_pipeline.artifacts import VideoPaths

        VideoPaths(tmp_path / "ds").ensure_dirs()
        assert (tmp_path / "ds" / "stories" / "raw").is_dir()

    def test_the_stage_declares_its_io(self) -> None:
        stage = StoriesStage()
        assert stage.name == "stories"
        assert stage.inputs == ("speech_segments",)
        assert stage.outputs == ("stories", "stories_raw")
        assert stage.config_keys == ("stories",)

    def test_it_is_numbered_after_speaker_fusion_and_before_finalization(self) -> None:
        assert STAGE_ORDER.index("speaker_fusion") < STAGE_ORDER.index("stories")
        assert STAGE_ORDER.index("stories") < STAGE_ORDER.index("finalization")

    def test_it_depends_on_the_speaker_branch_only(self) -> None:
        # It reads speech_segments + turns, neither of which translation produces.
        assert STAGE_DEPENDENCIES["stories"] == ("speaker_assignment",)

    def test_it_owns_a_stage_log(self) -> None:
        assert "stories" in STAGE_LOG_NAMES

    def test_it_is_registered_with_the_orchestrator(self) -> None:
        from multimodal_pipeline.orchestrator import STAGE_CLASSES

        assert STAGE_CLASSES["stories"] is StoriesStage

    def test_the_coverage_inventory_explains_the_new_table(self) -> None:
        from multimodal_pipeline import elan

        reason = elan.TIER_ABSENT_REASONS["stories"]
        assert "narrative" in reason
        assert reason != elan.COVERAGE_REASON_UNKNOWN
        # Falsifiable half: it names the tier-level fact a reader can check.
        assert "no tier" in reason

    def test_the_mock_client_is_deterministic(self) -> None:
        request = request_for(SEGMENTS)
        first = MockStoriesClient(model="m").detect(request)
        second = MockStoriesClient(model="m").detect(request)
        assert first.content == second.content
        assert [s["story_id"] for s in first.stories] == ["w0-s1"]

    def test_the_real_client_is_built_from_the_config(self, context) -> None:
        use_stage_config(context, provider="openai-compatible", base_url="https://x.invalid/v1",
                         api_key="sk-test", model="chat", max_retries=1,
                         backoff_base_seconds=0.25, timeout_seconds=7.0)
        client = StoriesStage()._client(context)
        assert isinstance(client, StoriesClient)
        assert client.model == "chat"
        assert client.max_retries == 1
        assert client.backoff_base == 0.25
        assert client.timeout_seconds == 7.0

    def test_a_good_answer_comes_back_from_a_real_server(self, endpoint) -> None:
        server, state = endpoint([(200, ok_body(envelope(story_payload()),
                                               prompt_tokens=610, completion_tokens=72))])
        client = stories_client(server)
        verdict = client.detect(request_for(SEGMENTS))
        assert [row["story_id"] for row in verdict.stories] == ["w0-s1"]
        assert client.usage == {"prompt_tokens": 610, "completion_tokens": 72}
        body = state.bodies[0]
        assert body["model"] == "chat" and body["temperature"] == 0.0
        assert state.headers[0]["authorization"] == "Bearer sk-test-123"

    def test_a_5xx_is_retried_and_a_4xx_is_not(self, endpoint) -> None:
        slept: list[float] = []
        server, _state = endpoint([(503, "upstream down"), (400, "bad request")])
        client = stories_client(server, max_retries=2, backoff_base_seconds=0.5, sleep=slept.append)
        with pytest.raises(StageError, match="HTTP 400"):
            client.detect(request_for(SEGMENTS))
        # One retry of the 503 (with exponential backoff), then the 400 ends the loop:
        # a rejected request is not resampled.
        assert client.request_count == 2
        assert len(slept) == 1 and slept[0] >= 0.5

    def test_a_malformed_answer_is_resampled_by_the_client(self, endpoint) -> None:
        server, _state = endpoint([(200, ok_body("I cannot help with that.")),
                                   (200, ok_body(envelope(reason="No arc here.")))])
        client = stories_client(server, max_retries=2)
        verdict = client.detect(request_for(SEGMENTS))
        assert client.request_count == 2
        assert verdict.stories == []
        assert verdict.payload["no_story_reason"] == "No arc here."

    def test_the_api_key_never_appears_in_a_failure_report(self) -> None:
        # Nothing runs on port 1, so this is the transport-error path: httpx's own message
        # carries the URL and the reason, and the authorization header lives somewhere else
        # in the request object. status.json and the stage log print exactly this text, so
        # the assertion is about what reaches disk.
        client = StoriesClient(base_url="http://127.0.0.1:1/v1", api_key="sk-secret-123456",
                              model="chat", max_retries=0, sleep=lambda _s: None)
        with pytest.raises(StageError) as excinfo:
            client.detect(request_for(SEGMENTS))
        assert "sk-secret-123456" not in str(excinfo.value)
        assert "sk-secret-123456" not in str(excinfo.value.details)

    def test_a_4xx_fails_the_stage_without_retrying(self, endpoint) -> None:
        server, state = endpoint([(401, "unauthorized")])
        client = stories_client(server, max_retries=3)
        with pytest.raises(StageError, match="HTTP 401"):
            client.detect(request_for(SEGMENTS))
        assert client.request_count == 1
        assert len(state.bodies) == 1


class TestFailurePropagation:
    """The stage does not swallow an endpoint failure; the orchestrator records it.

    A window that could not be answered must never be recorded as a window with no
    stories — that is the one state indistinguishable from an honest empty answer, and it
    would silently understate the corpus. ``VideoRunner`` is the place that turns an
    exception into a ``failed`` record (see tests/unit/test_pipeline_resume.py for that
    machinery), so the stage-level contract is simply: it propagates.
    """

    def test_a_transport_failure_propagates_out_of_execute(self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        client = FakeClient([StoriesTransportError("HTTP 503: upstream unavailable")])
        stage = StoriesStage()
        stage._client = lambda _ctx: client  # type: ignore[method-assign]
        with pytest.raises(StoriesTransportError, match="503"):
            stage.run(context)
        # Nothing was published, so a later run cannot read a half-written table as a
        # completed answer.
        assert not context.artifact("stories").exists()

    def test_an_exhausted_client_becomes_a_stage_error(self) -> None:
        # Port 1 is not listening, so every attempt is a transport error and the retry
        # budget runs out. This is the message that reaches status.json and the log.
        client = StoriesClient(base_url="http://127.0.0.1:1/v1", api_key="", model="chat",
                              max_retries=1, backoff_base_seconds=0.0,
                              sleep=lambda _seconds: None)
        with pytest.raises(StageError, match="failed after 2 attempts") as excinfo:
            client.detect(request_for(SEGMENTS))
        assert client.request_count == 2
        assert excinfo.value.details["window_index"] == 0

    def test_the_api_key_never_reaches_a_failure_message(self) -> None:
        client = StoriesClient(base_url="http://127.0.0.1:1/v1", api_key="sk-secret-123456",
                              model="chat", max_retries=0, sleep=lambda _s: None)
        with pytest.raises(StageError) as excinfo:
            client.detect(request_for(SEGMENTS))
        assert "sk-secret-123456" not in str(excinfo.value)
        assert "sk-secret-123456" not in str(excinfo.value.details)


class TestTheCacheKeyBindsWhatActuallyChangesTheAnswer:
    """The independent verifier falsified the key's promise (finding 1, HIGH).

    `key()`'s docstring claims "an endpoint, model or prompt change produces a different
    key". Measured on a temporary corpus: changing the endpoint, or editing the prompt
    source, left the request key unchanged, so a forced rerun reported
    `batches_reused: 1` and the client was never called again — the table kept the old
    endpoint's answers. The stage fingerprint did move in both cases (it hashes
    base_url and the module source), which is why only a forced rerun, or a moved
    cache directory, reaches the stale cache — the narrowness is real and the broken
    promise is equally real. These tests bind the key to both.
    """

    def test_the_request_key_changes_when_the_endpoint_changes(self) -> None:
        first = request_for(SEGMENTS, endpoint="https://one.example/v1")
        second = request_for(SEGMENTS, endpoint="https://two.example/v1")
        assert first.key() != second.key()

    def test_the_request_key_changes_when_the_prompt_text_changes(
            self, monkeypatch) -> None:
        before = request_for(SEGMENTS).key()
        monkeypatch.setitem(PROMPTS, "v1", PROMPTS["v1"] + "\nAsk politely.")
        after = request_for(SEGMENTS).key()
        assert before != after, (
            "the key ignored the prompt it will actually render, so a prompt edit "
            "reuses the old prompt's answers when the cache is reachable")

    def test_a_forced_rerun_after_an_endpoint_change_calls_the_client_again(
            self, context) -> None:
        # The verifier's exact reproduction, one level up: same context, second
        # execution, base_url moved. Before the fix: batches_reused 1, calls 1 total.
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        context.config.stories.base_url = "https://one.example/v1"
        client = FakeClient([envelope(story_payload()), envelope(story_payload())])
        first = run_stage(context, client)
        assert first["batches_reused"] == 0
        context.config.stories.base_url = "https://two.example/v1"
        stage = StoriesStage()
        stage._client = lambda _ctx: client  # type: ignore[method-assign]
        outcome = stage.run(context)
        assert outcome.status == "completed", outcome.message
        second = outcome.detail["provenance"]["extra"]
        assert second["batches_reused"] == 0, (
            "the endpoint moved but the old window was reused: the table now claims "
            "the new endpoint produced answers the old one gave")
        assert client.request_count == 2


class TestStoryIdsAreNormalizedBeforeAnythingCitesThem:
    """Finding 2 (MEDIUM): the validator stripped ids, serialization did not.

    `story_id: " s1 "` validated (stripped for the duplicate check) and then the row
    writer used the padded original, so a child citing `"s1"` was published pointing at
    `w0-s1` while its parent carried `w0- s1 ` — an orphan the validator had sworn did
    not exist. The normalization therefore moves to where the entry is built, before
    parent resolution and before rows.
    """

    def test_a_padded_story_id_is_normalized_and_children_still_link(self) -> None:
        payload = {"stories": [story_payload(story_id=" s1 "),
                               story_payload(story_id="s2", parent_id="s1")]}
        stories, dropped = validate_stories(payload, request_for(SEGMENTS))
        assert dropped == []
        by_id = {row["story_id"]: row["parent_id"] for row in stories}
        assert sorted(by_id) == ["w0-s1", "w0-s2"]
        assert by_id["w0-s2"] == "w0-s1", (
            f"child lost its parent across whitespace: {by_id}")

    def test_an_id_that_only_differs_in_whitespace_is_a_duplicate(self) -> None:
        payload = {"stories": [story_payload(story_id="s1"), story_payload(story_id=" s1 ")]}
        with pytest.raises(ValidationError, match="duplicate"):
            validate_stories(payload, request_for(SEGMENTS))


class TestACacheEntryMustDescribeItselfConsistently:
    """Finding 3 (MEDIUM): a cache file whose content contradicts its payload.

    Reuse re-validated the payload — so no invented id could reach the table (the
    verifier confirmed this) — but `_ensure_raw` happily restored the *content* string,
    so a tampered cache could publish a raw window file citing evidence the table does
    not carry. An entry that does not agree with itself is not evidence of anything: it
    is refetched, not half-trusted.
    """

    def test_a_cache_whose_content_cites_an_invented_id_is_not_reused(
            self, context) -> None:
        use_stage_config(context)
        seed_segments(context, SEGMENTS)
        run_stage(context, FakeClient([envelope(story_payload())]))
        cache_files = list((context.artifact("stories_raw") / "cache").glob("*.json"))
        assert len(cache_files) == 1
        entry = json.loads(cache_files[0].read_text())
        tampered = json.loads(entry["content"])
        tampered["stories"][0]["evidence_segment_ids"] = ["seg999999"]
        entry["content"] = json.dumps(tampered)
        cache_files[0].write_text(json.dumps(entry))
        # Delete the raw file so a reuse would have to restore it from the poisoned entry.
        for raw in context.artifact("stories_raw").glob("window_*.json"):
            raw.unlink()
        client = FakeClient([envelope(story_payload())])
        extra = run_stage(context, client)
        assert client.request_count == 1, (
            "the poisoned entry was reused instead of refetched")
        assert extra["batches_reused"] == 0
        restored = list(context.artifact("stories_raw").glob("window_*.json"))
        assert restored and "seg999999" not in restored[0].read_text()


class TestAFabricatedParentIsABreachEvenForADroppedStory:
    """Finding 4 (LOW): the reject/drop line leaked a fabricated reference.

    A story dropped for anchoring outside the window skipped the parent check entirely,
    so an answer citing parent `"not-in-answer"` was counted as a gap when it describes
    a story nobody answered. The accepted-entry rule (breach) now applies to dropped
    entries too; a parent that IS in the answer but was itself dropped remains the
    counted propagation the deviation-1 tests guard.
    """

    def test_a_dropped_story_with_a_parent_nobody_answered_is_a_breach(self) -> None:
        request = request_for(SEGMENTS, max_segments_per_request=2, context_segments=1)
        payload = {"stories": [story_payload(evidence_segment_ids=["seg000003"],
                                             parent_id="not-in-answer")]}
        with pytest.raises(ValidationError, match="not-in-answer"):
            validate_stories(payload, request)
