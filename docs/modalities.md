# Modalities in depth

How the stages that measure the same moment from different directions actually decide: two diarization engines, audio/visual agreement, the normalised pose frame, person tracks, and narrative windows.

Back to the overview: [Storytel Pipeline](../README.md).

## Contents

- [Two diarization engines, on purpose](#two-diarization-engines-on-purpose)
- [Audio × visual agreement: `speaker_fusion`](#audio--visual-agreement-speaker_fusion)
- [Normalised pose: `pose_normalized`](#normalised-pose-pose_normalized)
- [Person tracks: `persons`](#person-tracks-persons)
- [Narrative windows: `stories`](#narrative-windows-stories)

---

## Two diarization engines, on purpose

`speaker_turns.parquet` is pyannote. `speaker_turns_nemotron.parquet` is
`nvidia/Nemotron-3-Diarization`. They are **two independent measurements of the same
audio**, both produced when `diarization_nemotron.enabled` is true, so the choice of engine
can be made from evidence instead of from a blog post. Neither one replaces the other, and
`speaker_assignment` — the stage that labels transcript segments — reads **only** pyannote,
so enabling the second engine changes no existing label in any dataset.

Do not join the two tables on `speaker_id`. Pyannote emits `SPEAKER_00`, Nemotron emits
`speaker_0` ordered by arrival, and the numbers are unrelated: identical digits name
different people. Compare them by time overlap, not by label.

The shapes differ because the engines disagree about what a diarization is. Pyannote's
table here is the **exclusive** timeline — exactly one speaker per instant — because that is
what makes labelling a transcript segment well-defined. Nemotron's is **overlapping**: two
channels can be active at once, which is the feature the model exists for and which an
exclusive table cannot represent at all. So `speaker_turns_nemotron.parquet` carries
`overlap_s`: the seconds of that segment spent overlapping a *different* speaker's segment
(summed over all of them). `diarization_type` is `overlapping` in every row, so a reader
cannot mistake it for the other table.

What that disagreement looks like on real clips from this corpus, measured 2026-09-25:

| clip | pyannote (exclusive) | Nemotron |
|---|---|---|
| KABC, 4.20 s | 1 turn, 1 speaker | 3 segments, 2 speakers, 2 overlapping pairs |
| La1, 8.01 s | 2 turns, **1 speaker** | 4 segments, **2 speakers**, 2 overlapping pairs |

Two things worth knowing before reading that as a verdict. Nemotron found a second voice
where pyannote heard one — which is *also* the failure mode of an overlapping-speech model:
it can split one talkative speaker or promote background speech to a channel. And the
speed difference is far smaller than the raw inference time suggests. Nemotron's *inference*
is ~0.15 s per clip, but the worker is one process per video, so it pays a model load every
time: the two runs above recorded `load_seconds` of 1.55 and 1.22 against
`inference_seconds` of 0.148 and 0.153. Measured stage cost over this whole corpus, pyannote
took 37 s across 7 videos (5.3 s per video). So neither engine is the cheap one, and
"Nemotron is faster" is not a reason to prefer it. Two clips is not a comparison either.
Run the stage over the corpus you care about and read the tables; `status --plan` will tell
you it is the only stage that reruns.

Enabling it costs a separate uv environment and a model download, and the environment pin
is an unreleased `transformers` commit — see
[Why seven environments](architecture.md#why-seven-environments). If the environment is absent the stage
*skips* with the reason naming the fix, and the corpus still completes, because a second
opinion is not a prerequisite.

---

---

## Audio × visual agreement: `speaker_fusion`

`speaker_fusion` is the third thing in this space, and it is not a tie-breaker. It fuses a
diarizer's turn table with `activespeaker`'s per-frame table and writes
`speaker/fusion_pyannote.parquet` (plus `fusion_nemotron.parquet` when you select that
engine) **beside** the existing tables. `speaker_assignment` still reads pyannote, so
nothing already produced changes label. That is deliberate: a fused verdict quietly written
into `speaker_turns.parquet` would relabel every dataset in the corpus.

It runs no model and needs no uv environment of its own — two Parquet files in, one out, in
process, like `speaker_assignment`. Re-tuning the thresholds costs seconds, not a TalkNet
re-run.

**The disagreement is the product.** A diarizer answers *when does a voice speak*; TalkNet
answers *which visible face is talking* at 25 FPS. They agree often and diverge exactly
where it matters — an off-screen narrator, a cutaway, two faces with one voice, a mouth that
moves while silent. So each turn keeps its audio timing and speaker and gains an explicit
`agreement` state instead of one flattened label:

| `agreement` | What was measured |
|---|---|
| `face_matched` | a track cleared both `min_face_frames` and `min_active_ratio` |
| `face_partial` | a best track exists, below one of the two — the detail names which |
| `no_face_visible` | the ASD table covers the window and located **no** face in it: voice with nothing visible (off-screen narrator, audio bed) |
| `face_never_active` | faces were visible and measured, and no track was ever flagged active: a silent mouth or a cutaway face |
| `no_frames_measured` | the ASD table covers **no time** in the window: nothing was measured, which is not evidence about who spoke |

The last two are the reason the table has three count columns instead of a ratio.
`frames_in_turn` counts every dense ASD row in the window **including** the `no_face` rows,
so `frames_in_turn = 0` (not measured) and `frames_in_turn > 0` with
`face_frames_in_turn = 0` (measured: nobody there) cannot collide — the same lesson
`face_status` and `frame_reason` already learned. The validated invariant is
`face_active_frames ≤ face_frames_in_turn ≤ frames_in_turn`.

`agreement_detail` is the column to read first; it carries the numbers and the knob that
decided them. This is a real row from the La 1 clip, measured on a copy of
`data/processed/` under `/tmp`:

```text
track 4 active on 80/80 frames in turn (ratio 1.00 >= min_active_ratio 0.5, 80 >= min_face_frames 2), mean score 2.35; no face located on 44/124 measured frames of the window
```

The trailing clause is not decoration. That turn really does contain 44 frames with nobody on
screen — the ratio describes the frames where a face was visible, and a detail that reported
only "80/80, ratio 1.00" would read as though the whole five seconds were a face on camera.

Two engines, **one implementation**: the core takes *which* turn table to read as a
parameter, so Nemotron is a second call of the same code, not a second fusion.
`speaker_fusion.engines` selects the calls, and each engine's table is written to its own
file, because `SPEAKER_00` and `speaker_0` remain unrelated namespaces — do not join the two
fused tables on `speaker_id` any more than you join the two turn tables. `face_track_id` is *not*
a fourth namespace: it is `active_speaker_frames.track_id` copied through from the winning frame,
so it joins the ASD tables and the ELAN `face_tracks` tier — it is only never a **speaker** id.
`overlap_s` is carried from the
turn table and stays `null` on pyannote rows, because a `0.0` there would read as "measured:
no overlap".

```yaml
speaker_fusion:
  enabled: true
  engines: [pyannote]     # or [pyannote, nemotron]; each writes its own file
  min_active_ratio: 0.5   # share of the track's in-turn frames that must be active
  min_face_frames: 2      # one frame is a sighting, not a speaker
```

A selected engine whose turn table was never produced is **skipped with a logged reason**
while the other engine still fuses; if *no* selected engine has a table, or `activespeaker`
never ran, the whole stage skips rather than emitting an empty table that a consumer would
read as "every turn failed to match". A video with no speech is the exception that proves
the rule: zero turns is legitimate, so it is logged, not failed.

The reverse case is handled too, because it is the dangerous one. A completed run deletes any
fused table its configuration can no longer compute — deselect Nemotron, or delete its turn
table, and `fusion_nemotron.parquet` goes, with a warning naming the reason. Left in place it
would look exactly like a current table, and nothing downstream would notice: both the reuse
test and validation look only at engines that are fusible right now.

---

---

## Normalised pose: `pose_normalized`

`pose/body.parquet` carries **pixels**, so a presenter who steps back looks like they shrank.
`pose_normalized` is a second pose table that re-expresses the same keypoints in a
body-centred frame, which is what makes poses comparable across people, camera framing and
shot scale. It is a **new file next to the pixel table** — `pose/normalized.parquet` — because
pixels are the measured quantity and every dataset already produced joins on them.

It is a change of basis, not a rescaling: pick one joint as the origin and a second to define
the first axis, and every keypoint is re-expressed in that frame. This is the
linear-transformation branch of `dfMaker()` from **multimolang** (CRAN, the MULTIFLOW
project), reimplemented in Python and **validated against the reference** rather than assumed:
the fixtures under `tests/fixtures/pose_normalized/` are `dfMaker`'s own output, produced by
the reference R implementation, regenerated byte-for-byte by
`scripts/make_pose_normalized_fixtures.R`. The committed fixtures hold the first 5 frames of
two clips and the unit tests assert agreement on all 503 shared keypoints at 1e-9; the
agreement actually observed on this machine, over **every** raw frame of all four videos that
have pose (53588 numeric points), is 9.5e-15 — so the tolerance is headroom for a different
last-bit path, not load-bearing slack.

Which triple defines the frame is a real decision, not a detail, so it is configuration and it
is written into the stage fingerprint — change it and the table is invalidated instead of
quietly redefined. The default is **`MidHip → Neck`**, chosen over `dfMaker`'s own default
(`Neck → LShoulder`) by measuring the divisor over the four processed clips in `data/processed/`
— 695 frames, 3029 person-frames, of which a `MidHip → Neck` basis is usable in 2661 and a
`Neck → LShoulder` one in 2901. The basis length *is* the divisor, so it decides
how much an OpenPose jitter is amplified. Measured with the code in this repository:

| basis | median \|basis\| | median max(\|x'\|,\|y'\|) | p99 max(\|x'\|,\|y'\|) |
|---|---|---|---|
| `Neck → LShoulder` (dfMaker's default) | **17.7 px** | 8.79 | **1102.08** |
| `MidHip → Neck` (the default here) | 72.4 px | 1.58 | **2.37** |

A 17.7 px shoulder segment turns a half-pixel OpenPose jitter into a ~0.03 swing, and a wrist
four segments away lands over a thousand units out: those "normalised" coordinates are *less*
stable than the pixels they came from. `MidHip → Neck` is the longest two-point torso segment
BODY_25 offers, and both endpoints are in the top availability band (measured over those 695
frames: `Neck` 100.0%, `MidHip` 94.2%).

```yaml
pose_normalized:
  enabled: true
  origin_keypoint: MidHip   # becomes (0, 0)
  basis_keypoint: Neck      # becomes (1, 0)
  second_axis: perpendicular  # the branch the reference was validated against
```

**Missing keypoints are the normal case, and they are named, never zeroed.** A basis needs two
joints, and this corpus contains people who are partially off-frame; the frame either exists or
it does not, and "it does not" gets a reason in `basis_state` — `basis_missing_joint` (a
defining joint was never measured), `basis_degenerate` (both were, and they coincide, so the
determinant is zero), or `basis_non_finite` (both hold a number, but the numbers overflow the
basis, so every coordinate from them would be infinite or NaN). Per keypoint, `value_status`
says whether the joint got coordinates, lost its own coordinate, or had nowhere to be put. Every person-frame that has a keypoint still has
rows, because a dropped row is a person who stopped existing. A zero is never used for absence:
on these axes a zero means *exactly at the hip*, which is a measurement.

The stage depends on `openpose` **only** — no audio, no transcript, no diarizer — so a
transcription failure never costs the normalised poses, and it inherits openpose's skip
semantics: no `pose/body.parquet`, no table, with the reason naming which switch to turn on.
It runs in-process over Parquet, like `speaker_fusion`: one table in, one out, no
subprocess, no uv environment, seconds per video (37 857 keypoints of `person_demo` in 0.2 s
measured here).

---

---

## Person tracks: `persons`

Every other stage asks something about a *frame* or about *speech*. `persons` answers the
count question no stage answers: **how many distinct people appear in this video, and when is
each one on screen**. It runs Ultralytics YOLO detection with a ByteTracker over the source
video and writes three files: `persons/raw/yolo_track.json` (the raw per-frame tool output,
preserved byte-identical like every other stage's raw layer), `persons/frames.parquet` (one
row per person per frame, with bbox and confidence) and `persons/tracks.parquet` (one summary
row per person: first/last timestamp, frame count, coverage, longest gap, confidence and bbox
statistics, and an `appearance_order`).

Measured on this machine (RTX 4090, `yolo11n.pt`, `conf=0.25`, `imgsz=640`, `bytetrack`, one
worker process per video), with the worker's own reported numbers:

| clip | frames | frames with a person | distinct ids | max in one frame | wall |
|---|---|---|---|---|---|
| KABC | 126 | 126 | 3 | 3 | 1.4 s |
| CNN | 124 | 124 | 4 | 4 | 1.4 s |
| La-1 | 240 | 238 | 8 | 3 | 1.8 s |
| `person_demo` | 205 | 205 | **75** | 14 | 1.8 s |
| `pipeline_demo` | 249 | 0 | 0 | 0 | 1.6 s |

Read `person_demo`'s 75 as a measurement of a hard clip, not as a fact about how many people
are in it: it is 205 frames of a person walking in and out of frame, and the summary table
records 13 ids that last two frames or fewer precisely so a reader can see the fragmentation
instead of inheriting a tidy number. `pipeline_demo`'s zeros are the honest empty case — the
synthetic TTS/`testsrc` clip contains no person, exactly like its `pose_body` 0 rows — and
they are distinguishable from "the stage never ran" because a run that could not start
**skips** and names the reason instead of writing an empty table.

**The tracker changes the answer more than the model does**, which is why the tracker is
configuration and the numbers above name it. Same weights, same frames, only the tracker and
threshold moving, measured here:

| clip | bytetrack @ 0.25 (default) | bytetrack @ 0.10 | tracktrack @ 0.10 |
|---|---|---|---|
| KABC | 3 | 3 | 2 |
| CNN | 4 | 4 | 4 |
| La-1 | 8 | 8 | 6 |
| `person_demo` | 75 | 71 | **8** |

`tracktrack` merges aggressively (its `new_track_thresh` is 0.7 against bytetrack's 0.25) and
never saw more than 7 people on a frame where bytetrack saw 14. The default is `bytetrack`
because a fragmented track leaves evidence to inspect and a merged one leaves nothing. Note
also that **omitting `conf` is not a neutral choice**: `ultralytics`'s `track()` sets
`conf = 0.1` when the caller omits it, so a stage that reported 0.25 in its documentation and
forwarded nothing would be quietly reporting the counts a 0.1 threshold produced. The worker
forwards `conf` and `imgsz` explicitly and records both in the raw document's `parameters`.

A `person_id` is **not** a `track_id`. TalkNet's `track_id` counts faces, `person_id` counts
bodies, and on KABC TalkNet reports 2 face tracks where YOLO reports 3 person ids — both
correct, measuring different things. Join them by time overlap, never by label
([invariants](datasets.md#the-invariants-a-consumer-may-rely-on)). And `persons/frames.parquet`'s
`frame_number` is the worker's own 0-based count over the frames it read, **not** a source
frame and not the ASD table's 25 FPS grid index: `timestamp`, taken from
`source/frame_index.parquet`, is the only column safe to join on.

One process per video is load-bearing, not stylistic. Ultralytics keeps camera-motion
compensation state (`prevFrame`) on the tracker, and if that object is reused for a second
source the size assertion fails on **every** frame thereafter, the tracker silently falls
back to identity warps, and the only trace is one `WARNING` line per frame. Ids fragment,
which looks exactly like a model-quality problem. The parent measured 489 such warnings from
one batch that reused a single `YOLO(...)` across five clips, and 0 from the same five clips
in one process each. The worker therefore builds the model inside its tracking function,
passes `persist=False` unconditionally, and counts GMC failures into
`gmc_failure_count` so the failure is a number in the artifact rather than a line in a log.

---

## Narrative windows: `stories`

Every other stage measures something. This is the only stage whose job is a **judgement**: which
stretch of this transcript actually tells a story. The pipeline's own LLM endpoint is asked, and
its answer is treated as a claim to check, not a result to publish. It reads
`speech/segments.parquet` and, when a diarizer ran, the speaker turns — never frames, poses or
faces — and one of two stages that spend tokens per video, the other being `translation`.

One story exists on this corpus. On KABC:

```python
row = read("stories").iloc[0].to_dict()
# {'story_id': 'w0-s1', 'parent_id': None, 'start_time': 0.071, 'end_time': 4.051,
#  'title': 'Recalling hearing a voice at a Laker game',
#  'why_it_is_a_story': "The speaker recalls a specific past event of hearing someone's
#                        voice at a basketball game.",
#  'evidence_segment_ids': '["seg000001", "seg000002"]', 'confidence': 0.9,
#  'window_index': 0, 'model': 'chat', 'prompt_version': 'v1',
#  'request_key': '6af24e5d243048792e2584bb'}
```

That row cost one request and 680 tokens (549 prompt + 131 completion), and its two evidence ids
are two of the segment ids it was shown. Across the corpus: CNN, La-1, `pipeline_demo` and
`pipeline_demo_ntsc` each ran and returned an **empty table with a stated `no_story_reason`**,
and `person_demo` and `pipeline_silent` skipped for having no speech segments at all. Four of
the five datasets that ran produced nothing.

**There is no ground truth, so there is no precision or recall number to quote.** One positive
case in seven datasets is a working mechanism with one measured example, not a validated
detector — and that example is the same 4-second clip that carries most of this repository's
other measured evidence. Read the stage as cheap to rerun and easy to audit, not as a solved
problem.

**The empty answer is the designed answer, and it is defended in the prompt.** Rule 4 of
`PROMPTS["v1"]` is "MOST IMPORTANT: if the transcript contains no narrative window, return
`"stories": []` and say why in `no_story_reason`". That sentence is there because of a
measurement, not taste: in the probe that preceded the schema, an endpoint with no permission
to answer "no story" stretched a greeting into a story to satisfy the request. So `stories: 0`
on four clips is the stage working. The same probe is why `no_story_reason` became a column —
the endpoint volunteered it unprompted, and a stage that cannot record "nothing here, because…"
cannot distinguish an empty transcript from a failed judgement.

**Nothing reaches the table that was not checked against the transcript it was shown.** A story
citing a `segment_id` outside the window it was offered, quoting times that are not that
segment's boundaries, or omitting its evidence is dropped before `write_table` and counted as
`dropped_outside_window` in the raw summary, so a clipped answer is a recorded gap rather than a
story that never existed. `request_key` names the call that produced a row and `model` /
`prompt_version` name what answered, so every claim traces to retained bytes in `stories/raw/`.

**Ids are namespaced by window, which is not decoration.** The endpoint returns `s1`, and two
windows can each return one; the table stores `"w0-" + "s1"`. `parent_id` points inside that
namespace, because stories nest — and a parent the endpoint never actually answered with is
recorded as a breach, not silently re-parented.

**What the export deliberately does not do is the most consequential fact here.** No tier reads
this table, and the `.eaf` says so in its own coverage inventory with the reason spelled out:
*"narrative-level claims are not represented by this export: no tier reads this table, and no
existing tier represents a stretch of speech as a story — the transcript tiers (`segments_src`,
`words`) block over individual segments' own rows and never group them, so a story's span and
its `why_it_is_a_story` have no bar to reach."* On the two datasets that skipped the stage, the
same inventory reports the table `absent` instead of implying it exists. A story is a claim
about a *stretch* of speech, and every tier in ELAN is a claim about a segment: a bar drawn over
`w0-s1` would look like a measurement.

Two settings shape the answers more than the model choice. `max_segments_per_request` (default
60) is the window: a story whose evidence straddles the boundary is dropped **and counted**.
`temperature: 0.0` is deliberate — windows are compared across runs, so the same transcript
should give the same answer. `cache: true` reuses a completed window across runs, and the key
covers the endpoint **and the rendered prompt text**, so editing the prompt invalidates the
cache instead of silently serving stale answers; an earlier key hashed neither, which independent
verification found and `8f41689` fixed — the observable symptom was a rerun after a prompt edit
reporting `batches_reused: 1` with zero requests made.

`provider: mock` needs no endpoint at all and is what the suite uses: it returns one window
spanning the requested segments when shown two or more, and an empty answer with a reason when
shown one, because a single segment cannot hold a beginning and an end. It goes through the same
validator as the real provider, so the rejection path is exercised either way.

`stories` is **on by default**, and `inspect-environment` announces an unconfigured endpoint
before a run starts, immediately after translation's. That warning exists because silence about a stage that costs money is a defect, not a detail.
