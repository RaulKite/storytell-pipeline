# Architecture

The stage graph, what each stage decides and why, how reuse decides to rerun, and why the heavy tools live in seven isolated uv projects.

Back to the overview: [Storytel Pipeline](../README.md).

## Contents

- [Stage graph](#stage-graph)
- [What each stage decides](#what-each-stage-decides)
- [Why the frame tables are dense](#why-the-frame-tables-are-dense)
- [What `face_status` and `frame_reason` distinguish](#what-face_status-and-frame_reason-distinguish)
- [What `score_imputed` costs you](#what-score_imputed-costs-you)
- [What `spacy_model: blank` costs you](#what-spacy_model-blank-costs-you)
- [What `language_detection.status: low` warns about](#what-language_detectionstatus-low-warns-about)
- [Resume, reuse and invalidation](#resume-reuse-and-invalidation)
- [Why seven environments](#why-seven-environments)

---

## Stage graph

```
metadata
  ├─ audio
  │    ├─ whisperx ────┐
  │    ├─ diarization ─┴─ speaker_assignment
  │    │                   ├─ translation ── spacy_english
  │    │                   ├─ spacy_source
  │    │                   └─ stories
  │    ├─ diarization_nemotron
  │    └─ acoustic ◄── (also reads speaker_assignment)
  ├─ openpose ── pose_normalized
  ├─ persons
  └─ activespeaker ◄── (also reads audio)

  diarization + diarization_nemotron + activespeaker ──► speaker_fusion

  all 16 producers above ──► finalization ──► elan
```

`openpose` depends only on `metadata`, so a transcription failure never stops pose
extraction — verified: with whisperx failing, `openpose` completed while the speech
chain reported `upstream stage failed: whisperx`. Propagation is **per branch**: a
stage is blocked only by a failed stage in its own dependency chain, and a *skipped*
prerequisite (disabled, or a missing credential) is not a failure — `speaker_assignment`
degrades with its own reason instead of poisoning the run, so a dataset without a
Hugging Face token still gets transcript, linguistics, acoustics and pose.

OpenPose can also emit the rendered skeleton frames it draws, and it is **off by
default**: `openpose.write_images: true` adds `--write_images pose/raw_images` plus the
per-module switches this build actually exposes (`--render_pose -1` to inherit, and
`--face_render` / `--hand_render`, each following `face.enabled` / `hands.enabled`), so
body, hand and face skeletons land in one pass over the frames OpenPose already
computed. Set it deliberately: both it and `image_max_side` are hashed into the
`openpose` fingerprint, so enabling them re-runs the slowest stage on every dataset you
already produced. And bring disk space — OpenPose's own `--output_resolution` default is
`-1x-1`, full input resolution, which is hundreds of MB to several GB per video; give
`image_max_side` a pixel budget (640 is enough to read a pose) or the stage logs the
warning once and renders at source size anyway. What comes back is a *view*, not data:
`pose/*.parquet` stays the measured keypoints, the JSON in `pose/raw/` stays
byte-identical, and a run that was asked to render and wrote zero images fails loudly
instead of completing an empty dataset — that last check only means anything because a
rendering run empties `pose/raw_images` first, so the count it inspects belongs to that run
and not to whatever rendered there last. Turning rendering off deletes nothing. Capping the
render does not rescale the data:
measured on a 1280×720 clip with `image_max_side: 640`, the images came back 640×360
while `--keypoint_scale` kept its default and the JSON coordinates still reached x≈1223
— so the tables stay in source pixels and the images are a downscaled view of them
(exit 0, 205 frames, 205 images, 31 MB).

A second request is refused *before* the binary is invoked, because no post-run check can
catch it: `write_images: true` with `body.enabled`, `face.enabled` and `hands.enabled` all
false asks for images that nothing will draw. On this build `--write_images` still writes one
image per processed frame with every renderer off **and** `--output_resolution` does not bound
those files. Measured on a 249-frame clip: that request wrote all 249 images at the full source
640×480 while asking for 320×240, and they were the source frames themselves — mean absolute
difference 0.69 grey levels against the frame ffmpeg extracts. Maximum cost, zero skeletons,
exit 0 with a full image directory, so the zero-image guard above cannot see it. Enabling any
one module (face alone is enough — verified) still runs normally.

`activespeaker` answers a question the audio-only stages cannot: **which visible face
is producing the audio**. Pyannote says when someone speaks and OpenPose says where
bodies are; only TalkNet connects the two. Its frames table is deliberately **dense**
— exactly one row per 25 FPS frame of its working timeline, including frames where no
face was found — and every row also carries the nearest original-video timestamp,
because TalkNet thinks in constant-rate 25 FPS and the rest of the dataset does not.

That working timeline is the one thing about this table a reader gets wrong, so it is
stated twice and in the invariant list: the worker re-encodes the clip at `fps=25`, so
`frame_number` counts **its own grid**, `timestamp` is that grid's second, and
`source_timestamp` is the real one. On a 2997/100 clip only 1 row in 105 has
`timestamp == source_timestamp`; join on `source_timestamp`, never on `frame_number`. See
[Why the frame tables are dense](#why-the-frame-tables-are-dense).

Each frame carries `face_status`: `no_face` (nothing located), `tracked` (a face with a
score), or `tracked_unscored` (S3FD located a face, TalkNet had no measurement for it).
The third state matters: collapsing it into `no_face` would report "we could not score
this person" as "nobody was here", and a reader would draw the opposite conclusion from
the same null. A malformed bounding box is still dropped rather than invented — a
meaningless box is not a location.

`frame_reason` answers the next question, *why* the row does or does not carry a score,
because four different causes otherwise produce byte-identical rows. It is set by the
worker at the branch that knows (neither a frame's position inside its track nor the
track's score count survives into the table, so nothing downstream could recover it):

| `frame_reason` | Meaning |
|---|---|
| `scored` | a face was located and TalkNet produced a finite score for it |
| `imputed_tail` | the score is the last real score carried over the bounded tail — `score_imputed` is `true` and the value was not measured |
| `no_face` | no face was located in this frame at all (the only reason that pairs with `face_status = no_face`) |
| `score_not_finite` | a score existed at this position and was NaN or infinite |
| `track_has_no_scores` | the track produced zero scores, so there was nothing to carry |
| `past_scored_tail` | the frame sits beyond the last score plus the two-frame carry window |
| `tail_score_not_finite` | inside the carry window, but the value available to carry was not finite |
| `unknown` | only for raw artifacts written before this field existed, where the cause cannot be recovered — never a guess at one of the four above |

| Stage | Runs | Needs |
|---|---|---|
| `metadata` | ffprobe + SHA256 | ffmpeg |
| `audio` | ffmpeg → 16 kHz mono | ffmpeg |
| `whisperx` | uv env worker | GPU (or CPU), model download on first use |
| `diarization` | uv env worker | `HF_TOKEN` + pyannote community-1 EULA |
| `diarization_nemotron` | uv env worker (Nemotron 3) | optional second engine; its uv env + model download |
| `speaker_assignment` | in-process interval math | diarization output |
| `translation` | OpenAI-compatible HTTP | `translation.base_url` / `api_key` / `model` |
| `spacy_source` | uv env worker | spaCy model for the detected language |
| `spacy_english` | uv env worker | translation output + `en` model |
| `acoustic` | uv env worker (Parselmouth) | audio |
| `openpose` | `/opt/openpose` binary | OpenPose install + models, GPU |
| `pose_normalized` | in-process Parquet arithmetic | `pose/body.parquet` (nothing else) |
| `activespeaker` | uv env worker (TalkNet-ASD) | TalkNet checkout + `environments/activespeaker` |
| `persons` | uv env worker (Ultralytics YOLO) | `environments/persons` + weights (auto-download unless `weights_dir`) |
| `speaker_fusion` | in-process Parquet arithmetic | diarization turns **and** `activespeaker` output |
| `stories` | OpenAI-compatible HTTP | transcript segments; speaker turns when available |
| `finalization` | in-process | everything above |
| `elan` | in-process XML (pympi) | every producer's tables + the source video path — runs **after** `finalization` |

What happens to a stage whose prerequisites are absent depends on **which kind of
prerequisite is missing**, and the distinction is deliberate — measured, not styled:

1. **A choice you can opt out of** (disabled section, missing credential, unconfigured
   endpoint): the stage is **skipped with a reason** — `missing credential HF_TOKEN
   (export it to enable diarization)`. `status --plan`, the batch report and
   `validate` echo those reasons. The rest of the dataset is still produced and still
   valid.
2. **An install you explicitly asked for**: the stage **fails** with the fix in its
   message. With `openpose.enabled: true` and a wrong `openpose.root`, the run ends
   `openpose: OpenPose binary not found under … Set openpose.executable explicitly`
   (verified on this machine against a nonexistent root: `status.json` records
   `failed`, the video is `partial`, exit code 2). Same for a missing uv environment:
   `whisperx worker failed: uv project not found: … Create it and run \`uv sync\` there
   first.` Skipping instead would turn a broken install into a silently incomplete
   dataset — an operator who enabled a stage asked for its data or for an error, not a
   shrug.
3. **A stage blocked by a failed upstream** is *skipped* with
   `blocked by failed upstream: whisperx` (per-branch, as described above) — that skip
   is bookkeeping for the run, not a claim that the stage was configured wrong.

---

## What each stage decides

The table above says what runs. This is the reasoning a reader needs in order to *trust* a
result — the five decisions that change how a number should be read. Where a decision is
enforced by a `validate()` it is named; where it is only what the writer does today, this
says so, because the difference is exactly what a consumer needs.

### Why the frame tables are dense

Three tables are per-frame, and they are dense on **three different clocks**:

| Table | Grid | What is actually enforced |
|---|---|---|
| `source/frame_index.parquet` | every frame the container holds, on real PTS seconds | `metadata.validate()` fails if the table is missing or has 0 rows. The rows come from ffprobe *packet* timings; if none can be read the stage falls back to a uniform `1/fps` grid and logs a warning, so a table is always written but is not always measured |
| `acoustic/frame_features.parquet` | a fixed `acoustic.time_step` (default 0.01 s = 10 ms, Praat's own pitch step) | columns, non-negative and in-media timestamps, and monotonic order. Nothing asserts a fixed step — the spacing is what the worker's `frame_step_seconds` header says it is |
| `speaker/active_speaker_frames.parquet` | a **25 FPS grid TalkNet's worker creates** by re-encoding the clip at `fps=25` | `activespeaker.validate()` raises `frame_number is not a dense 0..N-1 sequence: …`, and a gap, a duplicate and an out-of-order row each get their own message |

Dense means the row exists even when nothing was found, so "this frame was not measured"
and "this frame was measured and nobody was on screen" are different rows rather than the
same missing one. What that looks like on disk: across the seven datasets in
`data/processed/`, `frame_index` has exactly `source.frame_count` rows in all seven, its
`frame_number` runs `0..N-1`, and `pts_seconds` is strictly increasing. For `pipeline_demo`
(a true 25 fps clip) the acoustic grid is uniform: 1001 rows, first timestamp 0.024, all
1000 gaps exactly 0.01, last 10.024.

Because three clocks are involved, one question comes up at every join and it belongs in
the invariant list rather than in a footnote:

> **`speaker/active_speaker_frames.parquet` is dense on the 25 FPS grid, not on the
> source frames.** `frame_number` is the index of that grid and `timestamp` is the grid
> second (`index / 25`); `source_timestamp` is the real source time, the PTS of the
> nearest actual frame. They coincide only when the clip is genuinely 25 fps.

Measured on the KABC clip (`2997/100` fps, 126 source frames): the ASD table has **105**
rows numbered 0..104, `timestamp` runs 0.0 / 0.04 / … / 4.16 while `source_timestamp` runs
0.0 / 0.033367 / … / 4.170838. Only 1 of the 105 rows has `timestamp == source_timestamp`.
`pose/body.parquet` in the same dataset covers all **126** source frames and its
`timestamp` reaches 4.170838. La 1 is the same shape: 240 source frames → 200 ASD rows.
`pipeline_demo`, at a true `25/1`, is the case where the two agree: 249 source frames →
249 rows, all equal. So the join rule is `timestamp`/`source_timestamp` or
`frame_index.pts_seconds`, **never `frame_number`**, unless you have checked
`source.frame_rate_rational` is `25/1`. On the KABC clip a `frame_number` join happens to
agree on 3 rows out of 105 and is wrong on the other 102 — its last ASD row
(`frame_number` 104, `source_timestamp` 4.170838) belongs to source frame **125**, while
`pose` row with `frame_number` 104 is a frame at 3.470137 s.

### What `face_status` and `frame_reason` distinguish

They answer two different questions about the same row, and the stage refuses to write a
table where they contradict each other:

- `face_status` — *was a face located at all?* `no_face` / `tracked` / `tracked_unscored`.
  Collapsing `tracked_unscored` into `no_face` would report "we could not score this
  person" as "nobody was here", which is the opposite conclusion from the same null.
- `frame_reason` — *why does this row carry, or not carry, a TalkNet score?* The eight
  values in the table above; four different causes produce an unscored row and only the
  worker's own branch knows which one ran.

`validate()` enforces the pair: a row is invalid unless
`face_status == "no_face"` exactly when `frame_reason == "no_face"` (the one reason that
means both), a score may only be present when the reason is `scored` or `imputed_tail`,
and any `frame_reason` outside the closed set of eight is a validation failure — a reader
switches on those strings and cannot handle a ninth.

### What `score_imputed` costs you

A carried score is **not a measurement**. TalkNet's MFCC windowing leaves its score array
one or two samples short of its frame list, so the worker carries the last real score over
that tail rather than inventing a third score. The ceiling is two frames per track, and
`score_imputed: true` marks exactly those rows.

What that costs a reader: for those frames the table tells you what the previous frame
measured, not what this one did. On the KABC clip the single imputed row carries
`talknet_score_raw` 0.9 — the same carried value as the measured row before it — while its
smoothed `talknet_score` is 0.9667 against that row's 0.95, because the smoothing window has
moved. So the imputed row is not a duplicate of its predecessor and must not be deduplicated
away; it is a re-expression of a measurement that belongs to an earlier frame. A "score
above threshold" count over-states the evidence by up to two frames per track. On KABC
exactly 1 of 105 rows is imputed (and 104 are `scored`); `pipeline_demo` has none.

### What `spacy_model: blank` costs you

`blank` means the source-language variant ran a spaCy pipeline with **no trained model**:
`capabilities` reads `tokenization,sentencizer` in
`linguistic/source/raw/spacy_source.json`, and what you lose is lemmas, POS tags,
dependency parses and named entities. The tokens and sentences are still there, still
segmented, still timed — so the table looks populated and only the annotation columns are
gone. That is why the fallback is loud in three places: `spacy_model` is written into the
Parquet file's schema metadata, `selected_model` /
`model_selection_status` / `capabilities` into the raw JSON beside it, and the chosen model
into `logs/spacy_source.log` per video.

Measured in this corpus: `pipeline_demo` records `en_core_web_lg` with status `configured`;
`person_demo` and `pipeline_silent` record `blank` with status `fallback_no_model` and
capabilities `tokenization,sentencizer` — those two clips have no detectable language, so
nothing reached the resolver. (Their token tables have 0 rows for a different and more
boring reason: WhisperX produced 0 words, so there was no text to annotate. An empty table
and a degraded model are two different absences and the two fields above tell them apart.)
The English variant is resolved separately and independently — `person_demo`'s
`linguistic/english/raw/spacy_english.json` records `en_core_web_lg` with status
`english_default`.

A *wrong* model is worse than a blank one, because it looks like a result: on Spanish text
`ca_core_news_lg` labelled `Muy buena entrada` as three `PROPN` tokens with plausible
dependencies attached. Resolution is configured language → same family → any installed
model named for the language → `blank`, and the status column says which branch fired
(`configured`, `substituted_family`, `discovered`, `fallback_missing_model`,
`fallback_no_model`). Installing or removing a model invalidates the linguistics stages,
because which models exist changes the output as much as the configuration does.

### What `language_detection.status: low` warns about

`speech/raw/whisperx.json` carries `language_detection` — `{status, probability, reasons}`
— and it is a grade of **the language guess, not of the transcript**. `status` is
`configured` when you pinned `whisperx.language`, and otherwise `ok` or `low`, where `low`
means the audio is shorter than WhisperX's 30-second detection window or the probability is
missing or below 0.5. The reasons array says which.

Two consequences:

1. It is normal here. Every auto-detected clip in this corpus is `low`, because every clip
   is under 30 s. `pipeline_demo` records `status: low`, `probability: 0.95703125`, reasons
   `["audio is 10.0s, below the 30s detection window"]`.
2. It is what tells you a `language` value is not a measurement. `person_demo` and
   `pipeline_silent` come back as `nn` with probabilities 0.238037109375 and 0.215 and a
   second reason naming the low probability — a reader who looks only at `language` takes a
   silent video's `nn` as a finding.

What it *drives* is `spacy.trust_low_language_detection` (see
[How far to trust the detected language](configuration.md#how-far-to-trust-the-detected-language)), and the
decision lands in provenance: `language_reliability` (the grade, or
`{"status": "absent"}`) and `language_reliability_trusted` in
`linguistic/source/raw/spacy_source.json`. It changes nothing else — not the transcript,
not the acoustics, not pose.

---

## Resume, reuse and invalidation

`status.json` is the only source of truth for resume. A stage reuses its previous
result only when **all five** hold:

1. its status is `completed`;
2. its own `config_hash` is unchanged (stage config + resolved tool identity + source identity);
3. its `dependency_hash` is unchanged — every transitive upstream stage's config **and** execution sequence;
4. all its output artifacts exist;
5. they pass its own `validate()`, and each Parquet output still has the row count
   it had when the stage completed.

`status --plan` runs the same decision function as `run`, so what it reports is
what will happen:

```
   metadata             valid previous result
   whisperx             configuration changed
   diarization          disabled: missing credential HF_TOKEN (export it to enable diarization)
   speaker_assignment   disabled: diarization produced no speaker turns
   acoustic             outputs changed: frame_features.parquet has 900 rows, 1001 when validated
```

Two details that took real debugging to get right:

- **Sequence numbers, not mtimes.** This filesystem rounds mtimes to ~16 ms, so a
  stage that reran quickly looked unchanged and left stale Parquet behind. Each
  video carries a monotonic `run_sequence`; a dependant goes stale when an upstream
  stage's sequence moves, even with identical configuration (`--force-stage`, or a
  crash mid-write).
- **A stage must not invalidate itself.** `finalization` writes the manifest, so
  folding artifact sizes into its fingerprint made every later run rewrite the
  whole summary, forever. Fingerprints cover only what a stage does not write.

`--force-stage` recomputes the named stages and everything downstream of them, and
nothing else. A completed run's rerun costs ~0 s.

---

## Why seven environments

whisperx pins `torch~=2.8.0`, pyannote.audio pulls its own transformers/torchcodec
combination, TalkNet needs an *older* torch than both, ultralytics pulls the newest torch
build it can find, spaCy wants neither, and the orchestrator should import none of them.
Installing everything together produces an unsatisfiable resolution or — worse — a
"working" resolution where one tool silently gets another's CUDA build.

So each heavy tool is its own uv project, invoked as
`uv run --project environments/<x> python workers/<x>.py`. The orchestrator imports
no `torch`, `pyannote` or `spacy`; a worker's environment can be rebuilt without
touching the pipeline. Verified pins on this machine:

| Env | Pins |
|---|---|
| `whisperx` | whisperx 3.8.6, torch 2.8.0 **+cu126**, torchaudio 2.8.0, torchvision 0.23.0 |
| `diarization` | pyannote.audio 4.0.7, torch 2.8.0, **torchcodec 0.7.0** |
| `spacy` | spaCy 3.8.16, pyarrow ≥17 |
| `acoustic` | praat-parselmouth 0.4.7 (Praat 6.1.38), numpy ≥1.26,<3 |
| `activespeaker` | **torch 2.5.1 +cu124**, torchvision 0.20.1, facenet-pytorch 2.5.3, scenedetect 0.6.5, numpy 2.0.2 |
| `diarization_nemotron` | **torch 2.8.0 +cu128**, transformers from git `5880561a`, librosa 1.0.0, accelerate 1.15.0 |
| `persons` | ultralytics 8.4.163, **torch 2.8.0 +cu126**, torchvision 0.23.0, lap 0.5.13 — no `opencv-python` pin, deliberately |

`diarization_nemotron` is the one pin that is **not a release**. NVIDIA ships
`nvidia/Nemotron-3-Diarization` two ways, and only one of them runs here: NeMo 3.0.0
cannot load the checkpoint at all (`self_attention_model='rope' is not supported`), and no
*released* `transformers` contains the architecture yet. So the environment pins an exact
commit of `transformers` from git — reproducible, but unreleased by construction. When a
release contains `nemotron3_diarization`, swap the git source for a version pin and expect
the Nemotron artifacts to be invalidated and recomputed. Full probe table:
`environments/diarization_nemotron/pyproject.toml`.

Three pins exist because of specific failures, not taste: the driver here is 555.42.06
(CUDA 12.5) and cu126 wheels are what was verified on it; latest `torchcodec` ships a
CUDA-13 build that dies with `libnvrtc.so.13`; and TalkNet cannot take a modern torch
because `talkNet.py` and its S3FD detector call `torch.load()` without `weights_only=`,
whose default flipped to `True` in torch 2.6 and rejects the project's 2021
checkpoints. That last one is why `activespeaker` is on cu124 while `whisperx` and
`diarization` are on cu126, and why `diarization_nemotron` was resolved separately on
cu128: cu128 is the build that was measured working for this model on this driver, and the
other three environments were never re-resolved to match it.

TalkNet is also the one stage whose model lives *outside* this repository: point
`activespeaker.talknet_root` at a checkout, and the two checkpoints either download
themselves into it or come from `activespeaker.weights_dir` if you keep the checkout
read-only. Unset, the stage skips with a reason naming the setting.

The `mock` translation provider (`translation.provider: mock`) lets you exercise the
whole graph, English linguistics included, with no network and no credentials.
