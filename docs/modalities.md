# Modality semantics

Interpret each model's answer in its own scope. Diarization, face activity, person
tracks and narrative candidates are different kinds of evidence, not interchangeable
identities or ground truth.

Back to the overview: [Storytel Pipeline](../README.md).

## Contents

- [Speech and language](#speech-and-language)
- [Acoustics](#acoustics)
- [Active-speaker evidence](#active-speaker-evidence)
- [Pose coordinates](#pose-coordinates)
- [Person tracks](#person-tracks)
- [Storytelling events and stories](#storytelling-events-and-stories)

## Speech and language

WhisperX produces segments, words and alignment states. `speaker_assignment` applies
Nemotron speaker labels to the normalized transcript; original WhisperX bytes remain
in `speech/raw/`. Word confidence and segment confidence are not interchangeable:
segment scores can be negative because they are log-probability-derived.

Nemotron is the only diarizer (`diarization_nemotron`). It writes
`speech/speaker_turns_nemotron.parquet`, whose `speaker_id` values (`speaker_0`) are its
own arrival-ordered namespace. Its channels overlap, so a turn records
`diarization_type: overlapping` and the seconds it overlaps another speaker in `overlap_s`.
It uses a whole-clip offline route with at most eight channels; its threshold controls
which frame probabilities become intervals.

Assignment is by maximum overlap, aggregated per speaker: a transcript interval that two
speakers talk over goes to whoever covers most of it, with a deterministic tie break, and
the runner-up is recorded in `runner_up_speaker_id`. There is no exclusive timeline to
prefer. A channel is not proof of a distinct real person: two channels can be one voice
bleeding, and an off-screen speaker can own a channel no face matches.

`speaker_assignment` is the only stage that decides what `speaker_id` means, so it never
leaves another engine's labels behind. A turn table already in the dataset is used even
when `diarization_nemotron` was not re-run (its environment missing, or switched off);
without one, `segments` and `words` are rewritten with `speaker_id: null` and
`speaker_assignment_method: no_diarization` — a recorded fact, not a gap. Transcripts
produced under the removed pyannote engine therefore lose their `SPEAKER_00` labels on
the next run instead of publishing a speaker no table in the dataset contains. It labels
whichever of the two transcript tables the dataset has, so a words-only dataset is not
left publishing a foreign label. The preserved raw WhisperX document is never rewritten.

That makes the turn table an input worth checking, and a damaged one **fails** the stage
rather than clearing labels: a table that is unreadable, lacks `start_time`/`end_time`/
`speaker_id`, holds a reversed or non-numeric interval, runs past the video's audio, names
another `video_id`, or was diarized from a different `audio_sha256` (Nemotron's sidecar
records it) is a defect, and labelling from it would silently mislabel or null everything.
Only genuine absence clears.

English translation is **segment-level translation**, not transliteration or
interlinear glossing. Source and English spaCy annotations are separate. Their
alignment/model-capability fields distinguish token timings, approximate timings
and segment context; model choice does not establish annotation correctness.
See [language policy](configuration.md#language-policy).

## Acoustics

Parselmouth/Praat writes frame pitch, intensity, voicing and formants, plus
segment aggregates and pauses. The frame grid follows `acoustic.time_step`, not
video FPS. Unvoiced pitch is missing, not 0 Hz. Pitch/formant ranges, chunk size
and pause thresholds affect results and belong to recorded configuration.

Read aggregate columns with their producer's denominator. In particular, reported
`duration` is media duration, not automatically a segment span; pause statistics
must not be reinterpreted as a different denominator. Dense numerical analysis
belongs in Parquet; ELAN bins are summaries, not substitutes for measurements.

## Active-speaker evidence

TalkNet connects audio with **one selected visible face per working-grid frame**.
A face's inactivity does not establish global silence or exclude an off-screen voice.
Scores are unbounded logit-like values, not probabilities.

| Field | Meaning |
|---|---|
| `face_status = no_face` | No face located |
| `face_status = tracked` | A located face with a usable score |
| `face_status = tracked_unscored` | Face located but no usable measurement |
| `score_imputed = true` | Last measured score carried over at most two tail frames |
| `frame_reason` | Why a measurement, carry or missing score exists |

The closed reason set is `scored`, `imputed_tail`, `no_face`, `score_not_finite`,
`track_has_no_scores`, `past_scored_tail`, `tail_score_not_finite`, and `unknown`.
`unknown` supports older raw artifacts whose cause cannot be reconstructed.
An imputed score is not a new measurement, even if smoothing changes its displayed
value. Unscored rows retain null scores rather than becoming zero-confidence faces.

The dense table follows a 25 FPS **working grid**, not source frames. Use the
[dataset clock rules](datasets.md#consumer-rules) before joining it to pose.

![Synthetic schema demo: measured, carried and missing face activity](assets/active_speaker_strip.png)

`activespeaker.write_video` optionally retains the native 25 FPS AVI with audio:
all tracked-face boxes and native smoothed scores (green at score ≥ 0, red otherwise).
These overlays are not the normalized table's scene-clipped smoothing, configurable
threshold and selected-face hysteresis. An unscored overlay is not evidence of silence.

## Pose coordinates

OpenPose publishes body, hand and face keypoints in source pixels. Confidence is
per keypoint; missing or partially detected landmarks are normal. Rendered skeleton
images are optional views and can be downscaled without rescaling table coordinates.

`pose_normalized` writes a **separate** body-centred table. The default origin is
`MidHip`, the first axis points to `Neck`, and the second is its clockwise
perpendicular in image coordinates. The origin becomes `(0, 0)` and basis joint
`(1, 0)`. This is a change of basis, not replacement of measured pixels.

`basis_state` distinguishes missing joints, a degenerate basis and non-finite math;
`value_status` describes each keypoint. Missing coordinates remain null, not zeros.

![Synthetic schema demo: BODY_25 keypoints with missing detections](assets/pose_skeleton_strip.png)
Reference fixtures from R's `multimolang::dfMaker` are committed under
[`tests/fixtures/pose_normalized/`](../tests/fixtures/pose_normalized).

## Person tracks

YOLO/ByteTrack IDs are trajectories, **not a count of unique humans**. Cuts,
occlusion and tracker association can split one person into many IDs or merge
tracks. Read frame count, confidence, coverage and gaps before quoting a count.
`longest_gap_seconds` is elapsed time between sightings, including sampling
spacing; it is not measured duration of absence.

The default is `yolo11n.pt` with ByteTrack. Raw output records model path/hash,
tracker and settings. An explicitly configured missing weights location is refused
rather than silently replaced with a download. With no explicit location, model
resolution can download weights. Changing the tracker can change counts more than
changing checkpoint size; neither default claims optimal accuracy.

The detection table contains sightings, not a complete sampled/empty-frame grid.
An absent detection cannot prove that frame was inspected. YOLO person IDs, OpenPose
per-frame person indices and TalkNet face tracks cannot be equated by matching digits.

`persons.write_video` optionally keeps those boxes and `id=` labels as a constant-rate
`persons/tracks.mp4` preview, drawn inside the tracking pass. It is a rendering of the
tables, not another measurement: each `id=` is a ByteTracker track id, and on a
variable-frame-rate source a constant-rate clock drifts against the picture, so
`source/frame_index.parquet` stays the timing authority.

## Storytelling events and stories

The `stories` stage asks an LLM to mark **storytelling events** (the conversational
activity of launching and telling: pre-telling, telling, coda) and, **separately**,
the **stories** told inside them (the represented "what"), plus spans it could not
place. The three tables are `storytelling_events`, `stories` and `uncertain_spans`;
a story names its event, and one event may carry several stories.

The prompt is the domain expert's own annotation prompt, quoted word for word in
[stories annotation guidelines](stories-annotation-guidelines.md) — her goal,
speaker-attribution limitation, working criteria, boundary guidance, multiple
tellers rule and annotation rules are hers as she wrote them. Two of her sections are
adapted, because they describe her input and her output rather than this stage's: her
WhisperX JSON section becomes the input format below, and her `time_seconds` report
becomes a `segment_id` + `transcript_quote` citation the pipeline resolves itself.
Prompts that paraphrase a domain expert's criteria are the failure this avoids.

The model cites a `segment_id` and copies a `transcript_quote` from that one segment;
it never reports times. The pipeline finds the quote (whole words only, case folded one
character to one character) in `speech/segments.parquet` and times its **first and last
word** through `speech/words.parquet`. `*_time_precision` is `word`, or `segment` when the
edge word has no exact alignment; a neighbouring word's time is never borrowed, and
character offsets are used only when they slice exactly that word. A quote that occurs
more than once in its segment needs `occurrence` (1 = first), because a segment such as
"no no" has no longer unique quote. The whole answer is rejected for an invented or
unshown segment id, a non-verbatim or unresolved quote, an enum value outside the closed
vocabularies, a duplicate id, a story outside its event, inconsistent `story_ids`, a
missing `story_evidence` explanation (or one made only of segment ids), or
`evidence_segment_ids` that are empty or outside the story's span. The retry is a repair:
the same prompt, the rejected answer (omitted above `stories.repair_max_answer_chars`)
and a bounded list of problems; a repair the endpoint refuses for size (HTTP 400/413) is
resent as the plain question. Transport failures are retried unchanged. `story_evidence` holds the model's explanations of why the stretch is
a story; `evidence_segment_ids` names the segments they rest on.

Each segment is shown as `segment_id`/`start`/`end`, its own `text:` (the only place
quotes come from) and its `speakers:` runs, e.g. `[speaker_0] words … [unknown] words`.
`stories.speaker_source` is `nemotron` (default; the only diarizer stories read) or `none`. Nemotron's
turns overlap, so a word gets a label only when one speaker's merged turns cover at least
`stories.speaker_min_coverage` of it and no other covers `stories.speaker_max_competing`;
otherwise it is `unknown`. Without Nemotron turns everything is `unknown`, recorded as
`speaker_source_requested`/`speaker_source_used` in `stories_summary.json`; the
max-overlap `speaker_id` on segments and words is never read instead. An answer may only
name labels the prompt showed (context included). Labels are arbitrary ids, not
identities. Where the transcript itself names a speaker, the model says so in `notes`.

Empty arrays are a legitimate, expected answer and are counted per window; a video
with no transcript segments skips the stage. The default is the whole transcript in
one request; setting `stories.max_segments_per_request` restores windowing. An event
whose boundary — or whose marked interruption — straddles a window is dropped and
counted **once**, with the number of stories it took with it counted separately
(`dropped_outside_window`, `dropped_stories_outside_window`), and an uncertain span
that touches a context segment is dropped and counted the same way, so absence is not a
recall result; references to the *visible* context segments are evidence, not a reason
to reject the answer. Raw responses make claims auditable: the
raw window holds the model's message text exactly as returned (not the HTTP envelope),
every rejected attempt with message text is kept under `stories/raw/rejected/` with a
unique name and credentials redacted (an empty message leaves no file), and the cache
(keyed by the rendered prompt and posted request body) restores those exact bytes on
reuse. `stories_summary.json` lists every window's raw file, empty windows included;
`validate()` re-parses and re-validates each one against the current transcript and
compares the recomputed rows field by field with the table rows of that request. Low temperature alone
does not make service responses deterministic.

There is no validated ground truth or accuracy guarantee here. **Storytelling events
and stories are not exported as an ELAN tier**. Transcript evidence supports auditing
a claim, not treating the LLM's summary or confidence as a measurement of narrative
truth.
