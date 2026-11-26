# Stories annotation guidelines

The domain expert's original annotation prompt for the `stories` stage (Armine, received
2026-11-26), reproduced byte-identically below and the source of truth for prompt `v2`:
every section is quoted word for word except the two adapted here — "Using WhisperX JSON"
→ the stage's INPUT section, and "Output" → the stage's contract, citing `segment_id` +
`transcript_quote` instead of `time_seconds` (the pipeline computes times) plus
`evidence_segment_ids`. `tests/unit/test_stories.py` fails if prompt and document drift.

Your goal is to mark the boundaries of each storytelling event and, separately, the boundaries of each story told within it. A story is the “what” being represented; a storytelling event is the conversational activity of launching and telling it.

## Important limitation: speaker attribution

This transcript may not include speaker labels. Do not infer speaker identities from the text alone.

- If the transcript includes speaker labels, use them.
- If it has no speaker labels, use "unknown" for the speaker unless the transcript itself clearly identifies who is speaking.
- Do not invent labels such as “Speaker 1” or “Speaker 2” based on assumptions about turn-taking.
- You may describe a voice as a distinct speaker only if the input contains reliable speaker or diarization information.
- If attribution is unclear, mark the speaker as "unknown" and explain the uncertainty.

## Working criteria

Treat a stretch as a possible story when it does one or both of the following:

1. Represents one or more events, including events linked causally or parts of a single event. The events need not be told in chronological order, and they need not involve the people currently speaking.
2. Foregrounds experience, such as a person’s embodied or intentional experience, or an evaluation of an experience. For example, someone may recount their reaction to a book without narrating a sequence of external events.

A report, opinion, explanation, or description is not automatically a story. For ambiguous cases, explain the evidence for treating the stretch as a story.

## Boundary guidance

### Start of the storytelling event
Look for possible pre-telling before the story itself, including:
- A gist or “bottom line” that previews the story.
- A question or reference that initiates a story sequence, such as asking whether the story has been told before.
- An explicit preface, such as “Let me tell you something.”

If present, include the pre-telling in the storytelling-event span. Do not automatically treat it as part of the story itself. Pre-telling may be absent or difficult to identify; do not assume it must occur.

### Start of the story
Mark where the speaker enters the story world, rather than where they merely announce or preview the telling. Look for references to a time, place, person, or combination that shift from the conversational here-and-now into the represented experience. These are clues, not mandatory formulas.

### End of the story
Mark the last utterance that belongs to the represented events or experience. Keep listener assessments or comments outside the story span unless they add to the represented experience.

### End of the storytelling event
Look for evidence that the telling has closed, such as:
- A coda by the storyteller (“That’s it,” “What a night,” “That was embarrassing”).
- A listener’s assessment or comment that treats the story as complete.
- A clear shift back to another conversational activity.

A coda may belong to the storytelling event while falling outside the story itself. Listener responses that acknowledge the story is over belong outside the storytelling event. If the transcript does not make the ending clear, mark it as uncertain.

## Multiple tellers, stories, and interruptions

- A storytelling event can involve more than one teller. Track contributions by each speaker only when the input provides speaker labels or reliable attribution.
- One storytelling event may contain more than one story, including a second story. Mark stories separately when the transcript supports that distinction.
- A story may be interrupted by unrelated talk and then resumed. If a later segment clearly continues or retrieves the earlier story, treat it as part of the same story and storytelling event, and mark the interruption. Do not join stretches based only on topic similarity.
- If a new story begins within an ongoing storytelling event, mark it as a second story within that event when appropriate.
- Do not force every conversational stretch into a story or storytelling event.

## Using WhisperX JSON

The input may be a WhisperX JSON object containing `text` and `segments`. Use the timestamped `segments` as the primary source for boundary locations.

- Use the segment `start` and `end` values to report approximate boundary times in seconds.
- Quote exact wording from the segment text. Do not invent words or silently correct transcription errors.
- Segment boundaries may be coarse and may combine multiple conversational turns. When a boundary falls inside a segment, estimate its time only if the transcript supports a reasonable estimate; otherwise use the segment boundary and mark the timing as uncertain.
- Ignore technical fields such as `tokens`, `temperature`, `avg_logprob`, and `compression_ratio`.
- Do not infer pauses, overlap, intonation, gaze, gesture, or other nonverbal cues from text alone.
- If timestamps or speaker labels are missing, state that limitation and do not invent them.

## Output

Return valid JSON only:

{
  "storytelling_events": [
    {
      "event_id": "SE1",
      "start": {
        "time_seconds": 0.0,
        "transcript_quote": "Exact opening words from the transcript",
        "speaker": "unknown",
        "boundary_type": "pre_telling | story_launch | uncertain",
        "reason": "Brief evidence for this boundary"
      },
      "end": {
        "time_seconds": 0.0,
        "transcript_quote": "Exact closing words from the transcript",
        "speaker": "unknown",
        "boundary_type": "coda | listener_acknowledgment | activity_shift | uncertain",
        "reason": "Brief evidence for this boundary"
      },
      "story_ids": ["S1"],
      "tellers": ["unknown"],
      "interruptions_or_resumptions": [
        {
          "time_range_seconds": [0.0, 0.0],
          "transcript_quote": "Exact words or span, if present",
          "description": "Briefly explain the interruption or later resumption"
        }
      ],
      "confidence": "high | medium | low",
      "notes": "Mention ambiguity or limitations, including unclear speaker attribution"
    }
  ],
  "stories": [
    {
      "story_id": "S1",
      "event_id": "SE1",
      "start": {
        "time_seconds": 0.0,
        "transcript_quote": "Exact opening words from the transcript",
        "speaker": "unknown",
        "reason": "Brief evidence that the story world or experience begins here"
      },
      "end": {
        "time_seconds": 0.0,
        "transcript_quote": "Exact closing words from the transcript",
        "speaker": "unknown",
        "reason": "Brief evidence that the represented events or experience end here"
      },
      "tellers": ["unknown"],
      "summary": "One or two sentences describing the represented events or experience",
      "story_evidence": [
        "Represented event(s)",
        "Causal or experiential connection",
        "Embodiment, intentionality, or evaluation, if relevant"
      ],
      "confidence": "high | medium | low",
      "notes": "Mention ambiguity, alternative plausible boundaries, or unclear speaker attribution"
    }
  ],
  "uncertain_spans": [
    {
      "time_range_seconds": [0.0, 0.0],
      "transcript_quote": "Exact words from the transcript",
      "issue": "What is uncertain and why"
    }
  ]
}

## Annotation rules

- Return one JSON object and no surrounding commentary or Markdown.
- Use exact transcript wording in quotes. Do not invent words or silently correct the transcript.
- Use the transcript’s timestamps. Do not invent timestamp precision that the segments do not support.
- Keep story boundaries and storytelling-event boundaries separate, even when they happen to coincide.
- If a boundary is unclear, provide the best-supported estimate and explain the uncertainty.
- Use "unknown" for speaker attribution when no reliable label is available.
- Return empty arrays when no storytelling events, stories, or uncertain spans are supported by the transcript.

## Transcript

[PASTE THE WHISPERX JSON HERE]
