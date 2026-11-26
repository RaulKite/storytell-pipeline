"""Attribution of words to overlapping diarization turns (`story_speakers`).

The rule under test is the one the operator chose for Nemotron, whose turns overlap: a word
belongs to a speaker only when that speaker covers most of the word **and** nobody else
covers a meaningful share, otherwise it is ``unknown``. Every test below names the one case
it refuses, because the failure this guards against — a confident-looking label built on a
simultaneous stretch — is invisible in the output table afterwards.

The renderer is tested with the property that makes quotes checkable: every piece it emits
is a substring of the segment text, so a quote the model copies is still findable in the
transcript.
"""

from __future__ import annotations

import re

import pytest

from multimodal_pipeline.story_speakers import (
    UNKNOWN_SPEAKER,
    attribute_interval,
    attribute_segments,
    attribute_words,
    coverage,
    render_speaker_runs,
)


def turn(speaker: str, start: float, end: float) -> dict[str, object]:
    return {"speaker_id": speaker, "start_time": start, "end_time": end}


def word(start: float | None, end: float | None) -> dict[str, object]:
    return {"word": "w", "start_time": start, "end_time": end}


class TestIntervalAttribution:
    def test_one_speaker_covering_the_whole_word_is_attributed(self) -> None:
        turns = [turn("speaker_0", 0.0, 2.0)]
        assert attribute_interval(turns, 0.5, 0.8) == "speaker_0"

    def test_an_overlapping_second_speaker_refuses_the_label(self) -> None:
        """Nemotron's real shape: two turns hold the floor over the same instant.

        speaker_0 covers the whole 0.5-0.8 word, and speaker_1 covers 0.55-2.0 — 0.25 s of
        its 0.3 s duration, i.e. ≥ the 0.25 competing share. Neither gets the word.
        """
        turns = [turn("speaker_0", 0.0, 2.0), turn("speaker_1", 0.55, 2.0)]
        assert attribute_interval(turns, 0.5, 0.8) == UNKNOWN_SPEAKER

    def test_a_small_competing_share_is_tolerated(self) -> None:
        # speaker_1 covers 0.02 s of a 0.3 s word: real turn-boundary jitter, not a claim.
        turns = [turn("speaker_0", 0.0, 2.0), turn("speaker_1", 0.78, 2.0)]
        assert attribute_interval(turns, 0.5, 0.8) == "speaker_0"

    def test_low_coverage_is_unknown(self) -> None:
        # The word is 1 s long and the only turn covers 0.4 s of it: no majority.
        turns = [turn("speaker_0", 0.0, 0.4)]
        assert attribute_interval(turns, 0.0, 1.0) == UNKNOWN_SPEAKER

    def test_no_turns_is_unknown(self) -> None:
        assert attribute_interval([], 0.0, 1.0) == UNKNOWN_SPEAKER

    def test_missing_timing_is_unknown(self) -> None:
        assert attribute_interval([turn("speaker_0", 0.0, 2.0)], None, 1.0) == UNKNOWN_SPEAKER
        assert attribute_interval([turn("speaker_0", 0.0, 2.0)], 0.0, None) == UNKNOWN_SPEAKER
        assert attribute_interval([turn("speaker_0", 0.0, 2.0)], 1.0, 1.0) == UNKNOWN_SPEAKER
        assert attribute_interval([turn("speaker_0", 0.0, 2.0)], 2.0, 1.0) == UNKNOWN_SPEAKER

    def test_a_boolean_is_not_a_time(self) -> None:
        # `isinstance(True, int)` is true in Python, and a JSON true here is not 1 second.
        assert attribute_interval([turn("speaker_0", 0.0, 2.0)], True, 1.0) == UNKNOWN_SPEAKER

    def test_turns_with_missing_edges_are_ignored(self) -> None:
        turns = [{"speaker_id": "speaker_0"},
                 turn("speaker_1", 0.0, 1.0)]
        assert attribute_interval(turns, 0.0, 1.0) == "speaker_1"

    def test_a_turn_without_a_speaker_id_is_ignored(self) -> None:
        turns = [{"speaker_id": "", "start_time": 0.0, "end_time": 1.0},
                 turn("speaker_1", 0.0, 1.0)]
        assert attribute_interval(turns, 0.0, 1.0) == "speaker_1"

    def test_the_thresholds_are_relative_to_the_interval_not_seconds(self) -> None:
        # Same shape at two durations: the verdict must not depend on how long the word is.
        turns = [turn("speaker_0", 0.0, 10.0), turn("speaker_1", 0.5, 10.0)]
        assert attribute_interval(turns, 0.0, 0.4) == "speaker_0"
        assert attribute_interval(turns, 0.0, 4.0) == UNKNOWN_SPEAKER

    def test_the_thresholds_are_configurable(self) -> None:
        turns = [turn("speaker_0", 0.0, 10.0), turn("speaker_1", 0.9, 10.0)]
        assert attribute_interval(turns, 0.0, 1.0) == "speaker_0"
        assert attribute_interval(turns, 0.0, 1.0,
                                  max_competing=0.05) == UNKNOWN_SPEAKER

    def test_coverage_sums_per_speaker(self) -> None:
        turns = [turn("speaker_0", 0.0, 0.5), turn("speaker_0", 0.5, 1.0)]
        assert coverage(turns, 0.0, 1.0) == {"speaker_0": pytest.approx(1.0)}


class TestWordAttribution:
    def test_each_timed_word_is_attributed_on_its_own(self) -> None:
        turns = [turn("speaker_0", 0.0, 1.0), turn("speaker_1", 1.0, 2.0)]
        labels = attribute_words([word(0.1, 0.4), word(1.1, 1.4)], turns)
        assert labels == ["speaker_0", "speaker_1"]

    def test_an_untimed_word_is_unknown_not_inherited(self) -> None:
        turns = [turn("speaker_0", 0.0, 5.0)]
        labels = attribute_words([word(None, None), word(0.1, 0.4)], turns)
        assert labels == [UNKNOWN_SPEAKER, "speaker_0"]
        labels = attribute_words([word(0.1, None)], turns)
        assert labels == [UNKNOWN_SPEAKER]

    def test_a_word_split_between_two_speakers_is_unknown(self) -> None:
        turns = [turn("speaker_0", 0.0, 0.35), turn("speaker_1", 0.35, 1.0)]
        assert attribute_words([word(0.2, 0.5)], turns) == [UNKNOWN_SPEAKER]


class TestSegmentAttribution:
    def test_segments_use_the_same_rule_over_their_own_interval(self) -> None:
        segments = [{"start_time": 0.0, "end_time": 1.0},
                    {"start_time": 1.0, "end_time": 2.0}]
        turns = [turn("speaker_0", 0.0, 1.0), turn("speaker_1", 1.0, 2.0)]
        assert attribute_segments(segments, turns) == ["speaker_0", "speaker_1"]

    def test_a_segment_with_no_turns_is_unknown(self) -> None:
        assert attribute_segments([{"start_time": 0.0, "end_time": 1.0}], []) == [
            UNKNOWN_SPEAKER]


class TestSpeakerRuns:
    TEXT = "I was fiddling with it and then it rolled away"

    def spans(self, *pairs: tuple[int, int] | None) -> list[tuple[int, int] | None]:
        return list(pairs)

    def test_consecutive_same_speaker_words_merge_into_one_run(self) -> None:
        text = "one two three four"
        spans = [(0, 3), (4, 7), (8, 13), (14, 18)]
        rendered = render_speaker_runs(text, spans, ["a", "a", "b", "b"])
        assert rendered == "[a] one two [b] three four"

    def test_every_rendered_piece_is_a_substring_of_the_text(self) -> None:
        # The property that makes a copied quote checkable: only whitespace at a run
        # boundary is collapsed, no character is invented or reordered.
        text = "Well, I said: 'not that one!' Then she left."
        spans = [(0, 5), (6, 7), (8, 13), (14, 27), (28, 32), (33, 36), (37, 41)]
        labels = ["a", "a", "a", "b", "b", "b", "b"]
        rendered = render_speaker_runs(text, spans, labels)
        pieces = re.split(r"\[[^\]]+\] ", rendered)[1:]
        assert pieces
        for piece in pieces:
            assert piece in text

    def test_an_unplaced_word_stays_inside_the_surrounding_run(self) -> None:
        text = "one two three four"
        # "three" could not be located, so its label cannot start a run: its text stays
        # in the run that spans it, and its own label is not shown at all.
        rendered = render_speaker_runs(text, [(0, 3), (4, 7), None, (14, 18)],
                                       ["a", "a", "b", "c"])
        assert rendered == "[a] one two [c] three four"

    def test_no_words_renders_one_unknown_run(self) -> None:
        assert render_speaker_runs("hello there", [], []) == "[unknown] hello there"
        assert render_speaker_runs("", [], []) == "[unknown]"

    def test_leading_and_trailing_text_stay_with_the_nearest_run(self) -> None:
        text = "... one two."
        rendered = render_speaker_runs(text, [(4, 7), (8, 12)], ["a", "a"])
        assert rendered == "[a] ... one two."
