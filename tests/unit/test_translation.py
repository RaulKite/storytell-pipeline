"""Translation: prompting, batching, JSON parsing, validation, retries, caching.

The endpoint is exercised through a real local HTTP server rather than a patched
client, so status codes, headers, timeouts and retry timing are actually tested.
"""

from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable

import pytest

from multimodal_pipeline.exceptions import StageError, ValidationError
from multimodal_pipeline.stages.translation import (
    MockTranslationClient,
    TranslationClient,
    TranslationTransportError,
    build_requests,
    extract_message_content,
    parse_structured_translations,
    validate_translations,
)


def segment(index: int, text: str | None = None, speaker: str | None = None) -> dict[str, Any]:
    return {
        "segment_id": f"seg{index:06d}",
        "text": text or f"segment number {index}",
        "speaker_id": speaker,
        "start_time": float(index),
        "end_time": float(index) + 1.0,
    }


SEGMENTS = [segment(i, speaker=f"SPEAKER_{i % 2:02d}") for i in range(1, 13)]


class TestBatching:
    def test_batches_cover_every_segment_once(self) -> None:
        requests = build_requests(SEGMENTS, language="es", batch_size=5, context_segments=2,
                                  prompt_version="v1", model="m", temperature=0.0, max_output_tokens=None)
        requested = [i for request in requests for i in request.requested_ids]
        assert requested == [row["segment_id"] for row in SEGMENTS]

    def test_batch_sizes(self) -> None:
        requests = build_requests(SEGMENTS, language="es", batch_size=5, context_segments=1,
                                  prompt_version="v1", model="m", temperature=0.0, max_output_tokens=None)
        assert [len(r.requested) for r in requests] == [5, 5, 2]

    def test_context_excludes_the_requested_rows(self) -> None:
        requests = build_requests(SEGMENTS, language="es", batch_size=5, context_segments=2,
                                  prompt_version="v1", model="m", temperature=0.0, max_output_tokens=None)
        for request in requests:
            assert not set(request.requested_ids) & {row["segment_id"] for row in request.context}

    def test_context_is_the_neighbourhood_not_the_whole_transcript(self) -> None:
        requests = build_requests(SEGMENTS, language="es", batch_size=5, context_segments=2,
                                  prompt_version="v1", model="m", temperature=0.0, max_output_tokens=None)
        # 12 segments in batches of 5 -> [0:5] [5:10] [10:12]; each batch sees at
        # most 2 neighbours on each side, never the whole transcript.
        assert [len(r.context) for r in requests] == [2, 4, 2]

    def test_first_batch_has_no_leading_context(self) -> None:
        requests = build_requests(SEGMENTS, language="es", batch_size=4, context_segments=3,
                                  prompt_version="v1", model="m", temperature=0.0, max_output_tokens=None)
        assert requests[0].context and requests[0].context[0]["segment_id"] == "seg000005"

    def test_context_neighbours_survive_identical_texts(self) -> None:
        twin = [segment(1, "same"), segment(2, "same"), segment(3, "same"), segment(4, "same")]
        requests = build_requests(twin, language="es", batch_size=2, context_segments=1,
                                  prompt_version="v1", model="m", temperature=0.0, max_output_tokens=None)
        assert [row["segment_id"] for row in requests[0].context] == ["seg000003"]
        assert [row["segment_id"] for row in requests[1].context] == ["seg000002"]

    def test_empty_transcript(self) -> None:
        assert build_requests([], language="es", batch_size=5, context_segments=1,
                              prompt_version="v1", model="m", temperature=0.0, max_output_tokens=None) == []

    def test_non_positive_batch_size_is_rejected(self) -> None:
        with pytest.raises(StageError):
            build_requests(SEGMENTS, language="es", batch_size=0, context_segments=1,
                           prompt_version="v1", model="m", temperature=0.0, max_output_tokens=None)

    def test_prompt_marks_requested_and_context(self) -> None:
        requests = build_requests(SEGMENTS, language="es", batch_size=2, context_segments=1,
                                  prompt_version="v1", model="m", temperature=0.0, max_output_tokens=None)
        prompt = requests[0].prompt()
        assert "REQUESTED" in prompt and "CONTEXT" in prompt
        assert "SOURCE LANGUAGE: es" in prompt
        assert "segment_id=seg000001" in prompt
        assert "speaker=SPEAKER_01" in prompt

    def test_unknown_prompt_version_fails(self) -> None:
        requests = build_requests(SEGMENTS, language="es", batch_size=2, context_segments=0,
                                  prompt_version="nope", model="m", temperature=0.0, max_output_tokens=None)
        with pytest.raises(StageError, match="prompt version"):
            requests[0].prompt()

    def test_key_changes_with_text_model_and_prompt(self) -> None:
        def key(**changes):
            options = {"language": "es", "batch_size": 3, "context_segments": 1,
                       "prompt_version": "v1", "model": "m", "temperature": 0.0, "max_output_tokens": None}
            options.update(changes)
            return build_requests(SEGMENTS, **options)[0].key()

        base = key()
        assert key(model="other") != base
        assert key(temperature=0.7) != base
        assert key(language="de") != base
        assert key(prompt_version="v1") == base
        # Adding neighbours changes the context the model sees, hence the key.
        assert key(context_segments=4) != base


class TestParsing:
    def test_plain_object(self) -> None:
        assert parse_structured_translations('{"seg000001": "hello"}') == {"seg000001": "hello"}

    def test_code_fenced_json(self) -> None:
        text = '```json\n{"seg000001": "hello"}\n```'
        assert parse_structured_translations(text) == {"seg000001": "hello"}

    def test_bare_fence(self) -> None:
        assert parse_structured_translations('```\n{"a": "b"}\n```') == {"a": "b"}

    def test_prose_wrapped_json(self) -> None:
        text = 'Sure! Here is the translation:\n{"seg000001": "hello"}\nHope that helps.'
        assert parse_structured_translations(text) == {"seg000001": "hello"}

    def test_nested_text_shapes_are_unwrapped(self) -> None:
        assert parse_structured_translations('{"a": {"text": "x"}}') == {"a": "x"}
        assert parse_structured_translations('{"a": {"translation": "y"}}') == {"a": "y"}

    def test_values_are_stringified_and_trimmed(self) -> None:
        assert parse_structured_translations('{"a": 42, "b": "  padded  "}') == {"a": "42", "b": "padded"}

    @pytest.mark.parametrize("text", ["", "not json at all", "[1, 2, 3]", '"a string"', "null"])
    def test_non_object_responses_are_rejected(self, text: str) -> None:
        with pytest.raises(ValueError):
            parse_structured_translations(text)

    def test_null_translation_is_rejected_not_silently_kept(self) -> None:
        with pytest.raises(ValueError, match="null"):
            parse_structured_translations('{"a": null}')


class TestValidation:
    def test_complete_set_is_returned_in_requested_order(self) -> None:
        result = validate_translations({"b": "two", "a": "one"}, ["a", "b"])
        assert list(result) == ["a", "b"]

    def test_missing_id_fails(self) -> None:
        with pytest.raises(ValidationError, match="missing"):
            validate_translations({"a": "one"}, ["a", "b"])

    def test_invented_id_fails(self) -> None:
        with pytest.raises(ValidationError, match="unexpected"):
            validate_translations({"a": "one", "ghost": "boo"}, ["a"])

    def test_empty_translation_fails(self) -> None:
        with pytest.raises(ValidationError, match="empty"):
            validate_translations({"a": "   "}, ["a"])


class TestContentExtraction:
    def test_standard_shape(self) -> None:
        payload = {"choices": [{"message": {"content": "hello"}}]}
        assert extract_message_content(payload) == "hello"

    def test_content_parts(self) -> None:
        payload = {"choices": [{"message": {"content": [{"text": "he"}, {"text": "llo"}]}}]}
        assert extract_message_content(payload) == "hello"

    @pytest.mark.parametrize("payload", [
        {}, {"choices": []}, {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": ""}}]},
        {"choices": [{"message": {"content": "   "}}]},
    ])
    def test_usable_shapes_raise_transport_error(self, payload: dict) -> None:
        with pytest.raises(TranslationTransportError):
            extract_message_content(payload)


# ------------------------------------------------------------------ live client


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
                status, payload = state.responses.pop(0) if state.responses else (200, ok_body({}))
                raw = json.dumps(payload).encode() if not isinstance(payload, str) else payload.encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args) -> None:  # silence the test server
                pass

        return Handler


def ok_body(translations: dict[str, str], **usage: int) -> dict[str, Any]:
    return {"choices": [{"message": {"content": json.dumps(translations)}}],
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


def client(server: HTTPServer, **kwargs: Any) -> TranslationClient:
    options = {"base_url": f"http://127.0.0.1:{server.server_port}/v1", "api_key": "sk-test-123",
               "model": "gpt-test", "max_retries": 2, "backoff_base_seconds": 0.0,
               "sleep": lambda _seconds: None}
    options.update(kwargs)
    return TranslationClient(**options)


def request_for(count: int = 2) -> Any:
    return build_requests(SEGMENTS[:count], language="es", batch_size=count, context_segments=0,
                          prompt_version="v1", model="gpt-test", temperature=0.0,
                          max_output_tokens=None)[0]


class TestClientAgainstARealServer:
    def test_successful_call_sends_the_right_request(self, endpoint) -> None:
        request = request_for(2)
        translations = {i: f"en {i}" for i in request.requested_ids}
        server, state = endpoint([(200, ok_body(translations, prompt_tokens=10, completion_tokens=5))])
        result = client(server).translate(request)
        assert result == translations
        body = state.bodies[0]
        assert body["model"] == "gpt-test"
        assert body["temperature"] == 0.0
        assert body["messages"][0]["role"] == "user"
        assert "SOURCE LANGUAGE: es" in body["messages"][0]["content"]
        assert state.headers[0]["authorization"] == "Bearer sk-test-123"

    def test_url_targets_the_chat_completions_path(self, endpoint) -> None:
        """A base_url ending in /v1 must become /v1/chat/completions, not a guess."""
        request = request_for(1)
        server, state = endpoint([(200, ok_body({request.requested_ids[0]: "x"}))])
        client(server).translate(request)
        assert state.headers[0]["_path"] == "/v1/chat/completions"

    def test_trailing_slash_in_base_url_does_not_double_the_path(self, endpoint) -> None:
        request = request_for(1)
        server, state = endpoint([(200, ok_body({request.requested_ids[0]: "x"}))])
        client(server, base_url=f"http://127.0.0.1:{server.server_port}/v1/").translate(request)
        assert state.headers[0]["_path"] == "/v1/chat/completions"

    def test_usage_is_accumulated(self, endpoint) -> None:
        request = request_for(1)
        good = (200, ok_body({request.requested_ids[0]: "x"}, total_tokens=7))
        server, _ = endpoint([good, good])
        instance = client(server)
        instance.translate(request)
        instance.translate(request)
        assert instance.usage["total_tokens"] == 14

    def test_max_output_tokens_is_forwarded(self, endpoint) -> None:
        request = build_requests(SEGMENTS[:1], language="es", batch_size=1, context_segments=0,
                                 prompt_version="v1", model="gpt-test", temperature=0.0,
                                 max_output_tokens=256)[0]
        server, state = endpoint([(200, ok_body({request.requested_ids[0]: "x"}))])
        client(server).translate(request)
        assert state.bodies[0]["max_tokens"] == 256

    def test_extra_body_is_merged(self, endpoint) -> None:
        request = request_for(1)
        server, state = endpoint([(200, ok_body({request.requested_ids[0]: "x"}))])
        client(server, extra_body={"response_format": {"type": "json_object"}}).translate(request)
        assert state.bodies[0]["response_format"] == {"type": "json_object"}

    def test_rate_limit_then_success(self, endpoint) -> None:
        request = request_for(1)
        good = (200, ok_body({request.requested_ids[0]: "x"}))
        server, _ = endpoint([(429, {"error": "slow down"}), good])
        instance = client(server)
        assert instance.translate(request) == {request.requested_ids[0]: "x"}
        assert instance.request_count == 2

    def test_server_error_then_success(self, endpoint) -> None:
        request = request_for(1)
        good = (200, ok_body({request.requested_ids[0]: "x"}))
        server, _ = endpoint([(500, "boom"), (503, "nope"), good])
        assert client(server).translate(request) == {request.requested_ids[0]: "x"}

    def test_retries_are_bounded(self, endpoint) -> None:
        request = request_for(1)
        server, _ = endpoint([(500, "a"), (500, "b"), (500, "c"), (500, "d")])
        instance = client(server, max_retries=2)
        with pytest.raises(StageError, match="3 attempts"):
            instance.translate(request)
        assert instance.request_count == 3

    def test_client_rejection_is_not_retried(self, endpoint) -> None:
        """A 4xx means our request is wrong; hammering the endpoint won't help."""
        request = request_for(1)
        server, _ = endpoint([(400, {"error": "bad request"}), (200, ok_body({"x": "y"}))])
        instance = client(server)
        with pytest.raises(StageError, match="rejected"):
            instance.translate(request)
        assert instance.request_count == 1

    def test_malformed_json_body_is_retried(self, endpoint) -> None:
        request = request_for(1)
        good = (200, ok_body({request.requested_ids[0]: "x"}))
        server, _ = endpoint([(200, "<html>not json</html>"), good])
        assert client(server).translate(request) == {request.requested_ids[0]: "x"}

    def test_incomplete_answer_is_retried(self, endpoint) -> None:
        """The model may answer only part of a batch; that must not be accepted."""
        request = request_for(2)
        partial = (200, ok_body({request.requested_ids[0]: "only one"}))
        complete = {i: f"en {i}" for i in request.requested_ids}
        server, state = endpoint([partial, (200, ok_body(complete))])
        assert client(server).translate(request) == complete
        assert state.bodies and len(state.bodies) == 2

    def test_backoff_sequence_grows_exponentially(self, endpoint) -> None:
        request = request_for(1)
        delays: list[float] = []
        server, _ = endpoint([(500, "a"), (500, "b"), (500, "c"),
                              (200, ok_body({request.requested_ids[0]: "x"}))])
        client(server, max_retries=3, backoff_base_seconds=1.0, sleep=delays.append).translate(request)
        assert len(delays) == 3
        # base * 2**attempt plus jitter in [0, base): 1.x, 2.x, 4.x
        assert [math.floor(d) for d in delays] == [1, 2, 4]
        assert delays == sorted(delays)

    def test_timings_are_recorded(self, endpoint) -> None:
        request = request_for(1)
        server, _ = endpoint([(200, ok_body({request.requested_ids[0]: "x"}))])
        instance = client(server)
        instance.translate(request)
        assert len(instance.timings_ms) == 1 and instance.timings_ms[0] >= 0.0

    def test_api_key_may_be_absent(self, endpoint) -> None:
        request = request_for(1)
        server, state = endpoint([(200, ok_body({request.requested_ids[0]: "x"}))])
        client(server, api_key="").translate(request)
        assert "authorization" not in {k.lower() for k in state.headers[0]}


class TestMockClient:
    def test_deterministic_and_complete(self) -> None:
        request = request_for(3)
        mock = MockTranslationClient()
        first = mock.translate(request)
        assert set(first) == set(request.requested_ids)
        assert first[request.requested_ids[0]].startswith("[en]")
        assert mock.translate(request) == first

    def test_mock_output_passes_the_same_validation_as_the_real_client(self) -> None:
        request = request_for(3)
        mock = MockTranslationClient()
        validate_translations(mock.translate(request), request.requested_ids)

    def test_every_requested_id_gets_a_key_even_for_empty_source_text(self) -> None:
        request = build_requests([segment(1, ""), segment(2, "y")], language="es", batch_size=2,
                                 context_segments=0, prompt_version="v1", model="m", temperature=0.0,
                                 max_output_tokens=None)[0]
        assert set(MockTranslationClient().translate(request)) == {"seg000001", "seg000002"}
