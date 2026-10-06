# Datasets and analysis

The output directory, file by file: what each artifact holds, a worked example with real numbers, how to read the tables from Python, and the invariants a consumer may rely on.

Back to the overview: [Storytel Pipeline](../README.md).

Commands use repository-relative paths: run them from the repository root.

## Contents

- [What a dataset looks like](#what-a-dataset-looks-like)
- [Two layers on purpose](#two-layers-on-purpose)
- [What a stage emits, as a picture](#what-a-stage-emits-as-a-picture)
- [One dataset, file by file](#one-dataset-file-by-file)
- [How to consume it](#how-to-consume-it)
- [The invariants a consumer may rely on](#the-invariants-a-consumer-may-rely-on)

---

## What a dataset looks like

```
data/processed/<video_id>/
├── manifest.json                 ← programmatic entry point: every artifact + status
├── status.json                   ← per-stage state machine (resume reads this)
├── source/
│   ├── metadata.json             ← ffprobe: streams, rational frame rate, SHA256, tags
│   └── frame_index.parquet       ← frame_number → true PTS in seconds
├── audio/
│   ├── audio.wav                 ← 16 kHz mono PCM s16le (WhisperX/Pyannote/Parselmouth)
│   └── audio_info.json
├── speech/
│   ├── segments.parquet          ← segment_id, start/end, language, speaker_id, text
│   ├── words.parquet             ← word-level times, confidence, alignment_status
│   ├── speaker_turns.parquet     ← pyannote diarization (when available)
│   ├── speaker_turns_nemotron.parquet  ← second engine, if enabled (see modality/ELAN guides)
│   └── raw/{whisperx,diarization,exclusive_diarization,nemotron_diarization}.json
│       + diarization.rttm
├── translation/
│   ├── segments_en.parquet
│   └── raw/                      ← every raw response + per-batch cache
├── linguistic/
│   ├── source/{tokens,sentences}.parquet   + raw/spacy_source.json
│   └── english/{tokens,sentences}.parquet  + raw/spacy_english.json
├── acoustic/
│   ├── frame_features.parquet    ← timestamp, f0_hz, intensity_db, voiced, f1..f3_hz
│   ├── segment_features.parquet  ← per-segment aggregates + pause statistics
│   └── raw/acoustic_features.jsonl
├── pose/
│   ├── body.parquet              ← BODY_25 keypoints, one row per person/keypoint (pixels)
│   ├── normalized.parquet        ← the same keypoints in a body-centred frame (see modality/ELAN guides)
│   ├── hands.parquet             ← 21 points × left/right
│   ├── face.parquet              ← 70 points
│   ├── raw/<video>_NNNNNNNNNNNN_keypoints.json   ← OpenPose's own output, untouched
│   └── raw_images/<video>_NNNNNNNNNNNNN_rendered.jpg   ← only with write_images: true
├── speaker/
│   ├── active_speaker_frames.parquet   ← dense on a 25 FPS grid TalkNet invents (see modality/ELAN guides)
│   ├── active_speaker_tracks.parquet   ← one row per TalkNet face track
│   ├── fusion_pyannote.parquet         ← pyannote turns × per-frame active speaker (see modality/ELAN guides)
│   ├── fusion_nemotron.parquet         ← same fusion, Nemotron turns, if that engine is selected
│   └── raw/{active_speaker.json,tracks.pckl,scores.pckl,scenes.csv}
├── persons/
│   ├── frames.parquet             ← per-frame person detections (when enabled)
│   ├── tracks.parquet             ← per-id track summaries
│   └── raw/yolo_track.json
├── stories/
│   ├── stories.parquet            ← transcript-bound narrative candidates (when produced)
│   └── raw/                       ← retained responses and window caches
├── elan/
│   └── annotations.eaf            ← up to 15 summary tiers; 12 table inputs + 2 support inputs
├── logs/                         ← pipeline.log + one log per stage
└── provenance/
    ├── config.json               ← resolved config, secrets masked, config hash
    ├── tools.json                ← tool + model versions, GPU/CUDA, OpenPose inventory
    └── processing.json           ← per-stage commands, hashes, durations, errors
```

For producer semantics, see [Modalities](modalities.md); for tier rules, see
[ELAN export](elan.md). The comments above are an inventory, not a full schema reference.

Start from `manifest.json`. It lists every artifact as a path relative to the
dataset directory, plus `artifacts_not_generated` for what was legitimately
skipped — so an absent file is always *declared*, never silently missing.
`temporal_model` states the unit and which columns are intervals versus instants.

### Two layers on purpose

Raw tool output and normalized Parquet are both kept, and normalization is
rerunnable on its own (`--only-stage <stage>` with raw present). Re-deriving a
Parquet table costs seconds; re-running WhisperX large-v3 or OpenPose costs minutes.
Raw artifacts stay **byte-identical** to what the tool produced — provenance is
written to a sidecar (`*.provenance.json`) rather than stamped into the raw file.

### What a stage emits, as a picture

These three figures illustrate the Parquet schemas with **synthetic** data, never
processed broadcast frames. The active-speaker strip separates measured scores,
carried scores and missing faces; the other strips show speaker turns and BODY_25
keypoints. They demonstrate the data shapes, not model accuracy.

![Synthetic schema demo: active-speaker frame states](assets/active_speaker_strip.png)
![Synthetic schema demo: speaker turns against word timings](assets/speaker_turn_strip.png)
![Synthetic schema demo: BODY_25 pose skeletons](assets/pose_skeleton_strip.png)

The [asset notes](assets/README.md) describe the generator. The command below renders
all four figures, including a stage graph not displayed here. Matplotlib is optional;
the figure-rendering tests require it too. Reproducibility depends on the current
stage definitions as well as the seed:

```bash
uv run --with matplotlib python scripts/make_dataset_figures.py --synthetic --seed 7 --out docs/assets
```

### One dataset, file by file

The example is `pipeline_demo`, because `scripts/make_fixtures.sh` rebuilds it from
nothing — ffmpeg's `flite` TTS over the `testsrc` test pattern, no copyrighted media — so
every number below is reproducible on your machine. Start from the two JSON files rather
than from the directory; that is what they are for:

```python
import json
from pathlib import Path

dataset = Path("data/processed/pipeline_demo")
manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
status = json.loads((dataset / "status.json").read_text(encoding="utf-8"))

print(manifest["video_id"], manifest["source"]["duration_seconds"], "s at",
      manifest["source"]["frame_rate_rational"], "fps,",
      manifest["source"]["frame_count"], "source frames")
print("temporal_model:", manifest["temporal_model"])
print("artifacts:", len(manifest["artifacts"]),
      "| declared not generated:", manifest["artifacts_not_generated"])
for name in status["stage_order"]:
    stage = status["stages"][name]
    print(f"  {name:17} {stage['status']:9} rows={stage['output_row_counts']}")
```

Output, verbatim, against the local dataset under `data/processed/` on this machine:

```text
pipeline_demo 9.985 s at 25/1 fps, 249 source frames
temporal_model: {'unit': 'seconds_from_video_start', 'interval_columns': ['start_time', 'end_time'], 'instant_columns': ['timestamp'], 'frame_columns': ['frame_number']}
artifacts: 45 | declared not generated: {'pose_images_raw': 'not_generated'}
  metadata          completed rows={'frame_index': 249}
  audio             completed rows={}
  whisperx          completed rows={'speech_segments': 1, 'speech_words': 23}
  diarization       completed rows={'speaker_turns': 2}
  diarization_nemotron completed rows={'speaker_turns_nemotron': 1}
  speaker_assignment completed rows={'speech_segments': 1, 'speech_words': 23}
  translation       completed rows={'translation_segments': 1}
  spacy_source      completed rows={'spacy_source_tokens': 26, 'spacy_source_sentences': 1}
  spacy_english     completed rows={'spacy_english_tokens': 28, 'spacy_english_sentences': 2}
  acoustic          completed rows={'acoustic_frames': 1001, 'acoustic_segments': 1}
  openpose          completed rows={'pose_body': 0, 'pose_hands': 0, 'pose_face': 0}
  pose_normalized   completed rows={'pose_normalized': 0}
  activespeaker     completed rows={'active_speaker_frames': 249, 'active_speaker_tracks': 0}
  persons           completed rows={'person_frames': 0, 'person_tracks': 0}
  speaker_fusion    completed rows={'speaker_fusion_pyannote': 2, 'speaker_fusion_nemotron': 1}
  stories           completed rows={'stories': 0}
  finalization      completed rows={}
  elan              completed rows={}
```

Two things to notice before opening a single Parquet file. `artifacts_not_generated` holds
exactly one entry on this dataset — the skeleton renderings under `pose_images_raw`, which
exist only when `openpose.write_images: true` and are off by default — and it is the
*declaration* channel: an absent file is always named there, never silently missing. The
`pose_*` and `person_*` zeros are a **result**, not a failure: the clip is a synthetic test
pattern with a synthesised voice over it, so there is no person for OpenPose or YOLO to find,
and both stages still wrote their schema-complete tables and their raw documents.
`manifest.json` carries the same 45 keys in `artifacts` (key → dataset-relative path) and
their sizes in `artifact_details`.

Why 18 stages and 45 artifacts when the registry declares 46 manifest keys? Because one key
is opt-in: `pose_images_raw`, the skeleton renderings, exists only with
`openpose.write_images: true`, which is false by default, and it is the single entry datasets
with speech declare absent. The counts are **not uniform across this disk** and the prose
says so honestly: five of the seven datasets list **45** artifacts with that one absence,
while `person_demo` and `pipeline_silent` list **43** and declare three — `pose_images_raw`
plus `stories` and `stories_raw`, because the `stories` stage skips a video with no speech
segments, with the reason in its status record. A full batch run after `stories` was added
rewrote the manifests, and an ELAN refresh after that moved the coverage inventory to the
23-table registry (12 entries exported to a tier + 2 summarised, i.e. the 14 tables an
export actually reads; the withdrawn `spacy_*` and `acoustic_segments` tables are present-and-
named-unexported, as is `stories`).

Where a reviewer should look to check that number: nowhere in git. `data/processed/` is
gitignored, so no commit in this repository contains a manifest, and a diff that moves this
count cannot carry the evidence for it. The evidence is the guard
`tests/unit/test_readme_claims.py`, which parses the two sentences above and re-reads every
manifest under `data/processed/` when one exists — so a re-run that changes the count breaks
the prose, and editing the prose breaks it against disk — and falls back to checking the
sentence's own arithmetic against the registry constant when the corpus is absent, which is
the half that runs on a fresh clone and in CI.
That uniformity is the point of reading the manifest instead of the
prose: a month ago these same datasets listed 36, one of them listed 38, and the README
sentence that claimed a stable per-dataset count was false against the bytes until a test
started comparing the sentence to them.

Now the files, with what those numbers mean:

| File | Rows | What it decides |
|---|---|---|
| `source/metadata.json` | — | ffprobe verbatim: 640×480, `frame_rate_rational` `"25/1"`, `frame_count` 249, `duration_seconds` 9.985, plus the source SHA256. Every other table is checked against this duration. |
| `source/frame_index.parquet` | 249 | `(frame_number, pts_seconds)` for each of those 249 frames — the authority that turns a second into a frame. First row `(0, 0.0)`, last `(248, 9.92)`. |
| `audio/audio.wav` | — | 16 kHz mono s16le; `audio/audio_info.json` records the exact ffmpeg argv and the resulting 160 768 frames / 10.048 s. |
| `speech/words.parquet` | 23 | Word-level times. Row 0: `word_id` `seg000001-w00000`, `start_time` 0.233, `end_time` 0.554, `duration` 0.321, `speaker_id` `SPEAKER_00`, `word` `Hello`, `confidence` 0.709, `alignment_status` `aligned`. 16 columns; `speaker_id` plus the three `speaker_overlap_seconds` / `speaker_overlap_ratio` / `speaker_assignment_method` columns are written by `speaker_assignment`, which re-reads this table with `segments.parquet` and `speaker_turns.parquet` and rewrites both — not by WhisperX. |
| `speech/segments.parquet` | 1 | The whole utterance is one segment, `seg000001`, 0.233→9.689 s, `language` `en`, `confidence` −0.1412 (log-probability-derived — see [Troubleshooting](configuration.md#troubleshooting)). |
| `speech/speaker_turns.parquet` | 2 | pyannote's **exclusive** timeline: `turn000001` 0.199719→1.026594 and `turn000002` 1.127844→9.869094, both `SPEAKER_00`, `diarization_type` `exclusive`. |
| `speech/speaker_turns_nemotron.parquet` | 1 | The second engine's answer on the same audio. Different id namespace — never join it to the row above on `speaker_id`. |
| `translation/segments_en.parquet` | 1 | `segment_id` `seg000001` again (every translation row must resolve to a transcript segment — `finalization` checks it), `source_language` `en`, `translation_model` `chat`, `translation_prompt_version` `v1`. |
| `linguistic/source/tokens.parquet` | 26 | spaCy over the source text. Token 0: `text` `Hello`, `lemma` `hello`, `pos` `INTJ`, `dep` `intj`, `head_token_id` `seg000001-s001-t0002`, and `token_start_time` 0.233 / `token_end_time` 0.554 inherited from the word alignment, with `timestamp_alignment_status` `aligned`. |
| `linguistic/source/sentences.parquet` | 1 | One sentence, `seg000001-s001`, `token_count` 26 — so the token and sentence tables agree by construction. |
| `linguistic/english/{tokens,sentences}.parquet` | 28 / 2 | The same schema over the English translation, variant `english`. 28 tokens for 26 source tokens: translation is not a 1:1 map, which is why the two variants are separate files. |
| `acoustic/frame_features.parquet` | 1001 | 10 ms Praat frames. 634 rows are `voiced` and **exactly those 634 carry a non-null `f0_hz`**; the other 367 are null. `timestamp` runs 0.024→10.024 in steps of exactly 0.01 — note the tail: the acoustic grid is laid over `audio.wav`, which is 10.048 s, so it can run slightly past the 9.985 s video. That is inside the duration + 1 s `finalization` allows, not a defect. |
| `acoustic/segment_features.parquet` | 1 | Per-segment F0/intensity/formant aggregates and pause statistics, keyed by `segment_id`. |
| `pose/{body,hands,face}.parquet` | 0 / 0 / 0 | The honest empty case: schema present, about 2 KB each, no person in frame. `pose/raw/` still holds all 249 `_keypoints.json` files, so the emptiness is auditable rather than asserted. |
| `pose/normalized.parquet` | 0 | The body-centred transform of the rows above, in its own coordinates (origin `MidHip`, axis toward `Neck`) — 15 columns, so the empty case has a schema too. It is **not** a rescaling of the `pose/body.parquet` rows above: same `frame_number`, different coordinate space, so the two are never interchangeable. |
| `persons/{frames,tracks}.parquet` | 0 / 0 | YOLO + ByteTrack over the same frames. Empty here for the same reason OpenPose was empty, but the tables still carry their Parquet metadata (`model`, `weights_sha256`, `tracker`, `device`, `coco_classes`) and `persons/raw/yolo_track.json` records the run that produced them, so an empty count is attributable to a specific model rather than asserted. `person_id` is its own namespace — never join it to TalkNet's `track_id`. |
| `speaker/fusion_{pyannote,nemotron}.parquet` | 2 / 1 | One row per diarization turn per engine: the turn's times plus what the face evidence said about it, as a verdict in `agreement` and the arithmetic behind it in `agreement_detail`. On this clip both engines answer `no_face_visible` — "a voice with nothing visible" — which is the correct reading of a test pattern, not a failure to decide. |
| `speaker/active_speaker_frames.parquet` | 249 | Dense on the 25 FPS grid, and on this clip the grid *is* the source grid. Every row is `face_status='no_face'`, `frame_reason='no_face'`, `score_imputed=False`, `is_active_speaker=False`, `track_id=None`. |
| `speaker/active_speaker_tracks.parquet` | 0 | No track, because no face was ever located. |
| `stories/stories.parquet` | 0 | The endpoint was asked once (`requests_made: 1`, 541 tokens) and answered *"no story"*, recording `windows: 1, empty_windows: 1`. The empty table is the verdict, not a missing stage. `stories/raw/` holds two `window_0_*.json` responses and two cache entries for that one request because the dataset kept the bytes produced under the stage's **first** cache key as well; the pre-fix response is the original evidence behind an earlier published row, so deleting it would erase raw output. |
| `elan/annotations.eaf` | 15 tiers / 276 annotations *(on disk)* | The ELAN export of the tiers above: fixed flat tiers, the video linked by both an absolute and a relative URL, and the per-frame Praat signals cut into 100 ms windows. It is XML, so the count is annotations and tiers rather than rows, and it is a **summary** — label text and window boundaries are derived, so this is not a row-for-row copy of the tables. **Regenerated on 2026-10-05** by `--only-stage elan` after four `spacy_*` tiers and the `acoustic_segments` tier were withdrawn and the three frame tiers were re-cut as windows; this clip went 17 tiers / 86 annotations → 15 tiers / 276, and the annotations went *up* because `intensity_blocks` alone now carries 85 of them. Measured on this clip: `words` 23, `voiced_blocks` 27, `f0_blocks` 43, `intensity_blocks` 85, `formant_blocks` 89, `face_tracks` / `person_tracks` / `pose_presence` 0 — the export repeats the tables' own emptiness, it does not invent a subject the clip does not have. |
| `provenance/{config,tools,processing}.json` | — | Resolved config with secrets masked, the machine inventory, and every stage's exact command, hashes and duration. |

One row read straight out of `speech/words.parquet`, which is the row every other modality
is anchored to:

```python
row = read("speech_words").iloc[0].to_dict()
# {'word_id': 'seg000001-w00000', 'start_time': 0.233, 'end_time': 0.554,
#  'speaker_id': 'SPEAKER_00', 'word': 'Hello', 'confidence': 0.709,
#  'alignment_status': 'aligned', ...}
```

That single row is worth reading carefully, because it is the intersection of three
stages: WhisperX produced the times, the confidence and `alignment_status`; pyannote
produced the speaker that `speaker_assignment` copied into `speaker_id`; and the
token table's `token_start_time` is this `start_time`. When those three disagree, the
dataset is broken and `finalization` is what says so.

#### The same walk on a real clip

`pipeline_demo` exercises the easy path: a true 25 fps clip, one speaker, nobody on
screen. Here is a real broadcast clip from this corpus —
`2017-12-30_0735_US_KABC_Jimmy_Kimmel_Live_1120_696_1124_896_hear`, 4.204204 s at
`2997/100` fps, 126 source frames — where none of that holds:

| Artifact | Rows | What is different |
|---|---|---|
| `speech/words.parquet` | 20 | Two segments, 1 pyannote turn, 2 translation rows, 24 source tokens. |
| `acoustic/frame_features.parquet` | 417 | Same 10 ms grid, four seconds of audio. |
| `pose/body.parquet` | 3 775 | All **126** source frames present, `timestamp` up to 4.170838 s. |
| `pose/face.parquet` | 16 799 | One row per landmark; the OpenPose face model has 70, and 171 of the 252 (frame, face) pairs in this clip carry all 70 while the rest carry 59–62. A partially-detected face is the normal case, not a corruption. |
| `speaker/active_speaker_frames.parquet` | **105** | Not 126: the 25 FPS grid, frames 0..104 contiguous, `timestamp` 0.0/0.04/…/4.16 against `source_timestamp` 0.0/0.033367/…/4.170838. All 105 rows are `face_status='tracked'`; 104 are `frame_reason='scored'` and 1 is `imputed_tail`; `score_imputed` is true exactly once; `is_active_speaker` is true on 102 rows; `talknet_score` is present on all 105 (min −2.24, max 3.86). |
| `speaker/active_speaker_tracks.parquet` | 2 | Two face tracks, one handing over to the other at 3.12 s. |

The tracks table is the cheapest way to read a clip before touching the frames:

| `track_id` | first → last (s) | frames | active | ratio | mean | max | mean bbox area |
|---|---|---|---|---|---|---|---|
| 0 | 0.00 → 3.08 | 78 | 75 | 0.9615 | 2.5054 | 3.86 | 2 971.59 |
| 1 | 3.12 → 4.16 | 27 | 27 | 1.0 | 2.5554 | 3.28 | 2 169.37 |

Two tracks, one after the other, with a 0.04 s gap that is exactly one grid frame: track 0
ends at 3.08 and track 1 starts at 3.12. Both are flagged active for nearly all of their
frames, so a naive "who is speaking?" query answers "both". That is not a bug in the
table — TalkNet was never asked to pick one voice for the whole clip — and it is why
`speaker_fusion` exists: it joins these rows to the diarization turns and writes a verdict
per turn. Read `active_speaker_tracks.parquet` as "here are the visible faces and how
speaking each one looked", not as "here are the speakers".

And do not read the handover as a camera cut. This clip is **one** scene
(`speaker/raw/scenes.csv` reports a single scene spanning all 105 frames, and every row's
`scene_id` is 1); track 1 is S3FD starting a new *track*, which happens when a face is lost
and re-found just as much as when the shot changes. The scene boundary and the track
boundary are different measurements and only one of them is in the tracks table.

### How to consume it

Every snippet below was run on this machine against the datasets in `data/processed/`, and
each one resolves paths through `manifest.json` rather than reassembling them — the
manifest is the file that knows what was actually produced.

`pandas` is **not** a project dependency (the orchestrator depends on `pydantic`, `PyYAML`,
`typer`, `rich`, `pyarrow`, `openai` and `httpx`), so run these as
`uv run --with pandas --with pyarrow python your_script.py`. `pyarrow` is a dependency and
works with a plain `uv run`.

#### Load a table

```python
import json
from pathlib import Path

import pandas as pd
from pyarrow import parquet as pq

DATASET = Path("data/processed/pipeline_demo")
artifacts = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))["artifacts"]


def read(key: str, columns: list[str] | None = None) -> pd.DataFrame:
    """One manifest key -> a DataFrame. The manifest, not a string template, knows the path."""
    return pq.read_table(DATASET / artifacts[key], columns=columns).to_pandas()


words = read("speech_words", ["word_id", "start_time", "end_time", "speaker_id",
                              "word", "confidence", "alignment_status"])
print(len(words), words.iloc[0].to_dict())
```

```text
23 {'word_id': 'seg000001-w00000', 'start_time': 0.233, 'end_time': 0.554, 'speaker_id': 'SPEAKER_00', 'word': 'Hello', 'confidence': 0.709, 'alignment_status': 'aligned'}
```

#### Stay in Arrow when you do not need a DataFrame

```python
import json
from pathlib import Path

from pyarrow import parquet as pq
import pyarrow.compute as pc

DATASET = Path("data/processed/pipeline_demo")
artifacts = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))["artifacts"]

# Row counts without reading a single cell — this is what the pipeline's own validate does.
print("rows:", pq.read_metadata(DATASET / artifacts["acoustic_frames"]).num_rows)

# Project only what you need; Parquet is columnar, so this costs a fraction of the file.
frames = pq.read_table(DATASET / artifacts["acoustic_frames"],
                       columns=["timestamp", "f0_hz", "voiced"])
voiced = frames.filter(pc.field("voiced"))
unvoiced = frames.filter(pc.invert(pc.field("voiced")))
print("voiced:", voiced.num_rows,
      "| f0 non-null among them:", voiced.num_rows - voiced["f0_hz"].null_count,
      "| f0 nulls where unvoiced:", unvoiced["f0_hz"].null_count)
```

```text
rows: 1001
voiced: 634 | f0 non-null among them: 634 | f0 nulls where unvoiced: 367
```

That pair of numbers is the invariant, checked on the real file: pitch exists exactly when
`voiced` is true, and never as a zero elsewhere.

#### Join pose to the active-speaker frames

This is the one join that goes silently wrong, because both tables have a `frame_number`
and a `timestamp` and only one of each pair means the same thing. The ASD table's pair
belongs to its own 25 FPS grid; `source_timestamp` is the column that means what `pose`'s
`timestamp` means. Drop the grid columns before merging so pandas cannot keep two
`timestamp`s and pick the wrong one:

```python
import json
from pathlib import Path

import pandas as pd
from pyarrow import parquet as pq

DATASET = Path("data/processed/2017-12-30_0735_US_KABC_Jimmy_Kimmel_Live_1120_696_1124_896_hear")
artifacts = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))["artifacts"]


def read(key: str, columns: list[str]) -> pd.DataFrame:
    return pq.read_table(DATASET / artifacts[key], columns=columns).to_pandas()


asf = read("active_speaker_frames", ["frame_number", "timestamp", "source_timestamp",
                                     "track_id", "talknet_score", "is_active_speaker"])
pose = read("pose_body", ["frame_number", "timestamp", "keypoint_name", "x", "y", "confidence"])
neck = pose[pose.keypoint_name == "Neck"]
print("ASD grid:", len(asf), "rows numbered", asf.frame_number.min(), "..", asf.frame_number.max())
print("source frames in pose:", pose.frame_number.nunique(),
      "numbered", pose.frame_number.min(), "..", pose.frame_number.max())

asf = asf.drop(columns=["frame_number", "timestamp"])
joined = pd.merge_asof(neck.sort_values("timestamp"),
                       asf.sort_values("source_timestamp"),
                       left_on="timestamp", right_on="source_timestamp",
                       direction="nearest", tolerance=1e-4)
print("matched:", joined.track_id.notna().sum(), "of", len(neck))
print(joined[["frame_number", "timestamp", "source_timestamp", "track_id",
              "talknet_score"]].iloc[[0, 251]].to_string())
```

```text
ASD grid: 105 rows numbered 0 .. 104
source frames in pose: 126 numbered 0 .. 125
matched: 210 of 252
     frame_number  timestamp  source_timestamp  track_id  talknet_score
0               0   0.000000          0.000000       0.0         2.4000
251           125   4.170838          4.170838       1.0         0.9667
```

`210 of 252` for two reasons, both worth knowing. 252 because two people are on screen and
`Neck` has one row per person per frame; 210 because the 25 FPS grid samples a 29.97 fps
clock and cannot land on every source frame — 105 of the 126 source frames have an ASD row
within the tolerance, and the other 21 are 0.0333 s from the nearest one. A join that must
not drop a frame needs `how="left"` and a tolerance you chose, not one you inherited. The
last row is the one to look at: source frame **125**, the final frame of the clip, matched
to an ASD row whose `source_timestamp` is 4.170838 — the ASD table has no row numbered 125
at all.

If you would rather not depend on the nearest-match merge, map both tables onto
`source/frame_index.parquet` first: every pose `timestamp` equals its own frame's
`pts_seconds` exactly (checked for all rows of all four pose-bearing datasets in
`data/processed/`), so `frame_index` is the shared spine and `merge_asof` is only a
convenience.

#### Decide what is trustworthy before using it

`manifest.json` and `status.json` together answer that, and the snippet that prints them is
at the top of [One dataset, file by file](#one-dataset-file-by-file). The fields worth
gating on:

- `manifest["artifacts"]` vs `manifest["artifacts_not_generated"]` — a path is either
  listed or declared not generated. `finalization`'s own `validate()` fails if any listed
  artifact is missing, so the first set is safe to open without an existence check.
- `status["stages"][name]["status"]` — one of `pending`, `running`, `completed`, `failed`,
  `skipped` (`overall_status` at the top of the file additionally has `partial`, which is a
  verdict about the video, not a stage state). A skipped stage puts its reason in
  `validation_result` as `{"skipped": true, "reason": "…"}` — that object is where the exact
  skip strings quoted elsewhere in these guides live.
- `status["stages"][name]["output_row_counts"]` — the row count each table had when the
  stage completed. Reuse re-validates against it, so a mismatch you find later means the
  file changed after the run.
- `status["stages"][name]["config_hash"]` / `"dependency_hash"` — the fingerprint pair that
  decides reuse. `provenance/processing.json` carries the same plus the exact command,
  `tool_version` and `model_version`.

### The invariants a consumer may rely on

Stated in tiers, because "checked by a test" and "true of the seven datasets on this disk"
are different promises and only the first one survives a hand-edited file. Tier 1 is
enforced by a `validate()` that the `validate` command re-runs on artifacts already on
disk. Tier 2 is measured on the seven datasets under `data/processed/` and is **not**
enforced anywhere — treat it as what the writers do today, not as a guarantee.

**Tier 1 — validated.**

1. **One timeline, declared.** `manifest["temporal_model"]` states the unit
   (`seconds_from_video_start`), which columns are intervals (`start_time`, `end_time`),
   which are instants (`timestamp`), and which are frames (`frame_number`). Every timestamp
   in every timed table is ≥ 0 and within the media duration + 1 s — `finalization`'s
   cross-modal check over the nine tables in its `TIMED_TABLES` list.
2. **Identifiers resolve across modalities.** Every `translation/segments_en.parquet`
   `segment_id` exists in `speech/segments.parquet`, and every `speaker_id` the transcript
   cites exists in `speech/speaker_turns.parquet`. Both checks run independently, so a
   missing translation table does not silently disable the speaker check.
3. **The ASD frame table is dense, on its own grid.** `validate()` raises
   `frame_number is not a dense 0..N-1 sequence: …` for a gap, a duplicate or an
   out-of-order row — where "frame" is the worker's 25 FPS index, not a source frame.
   `frame_index.parquet` is validated as non-empty, and it is written from the container's
   own packet timings.
4. **The disclosure columns cannot contradict each other.** `score_imputed` is true exactly
   on `frame_reason = 'imputed_tail'` rows — `validate()` rejects any other combination,
   including an imputation hiding on a `no_face` row — a score may only sit on a `scored`
   or `imputed_tail` row, `face_status = 'no_face'` iff `frame_reason = 'no_face'`, and a
   ninth `frame_reason` value is a failure rather than a surprise.
5. **Every declared artifact exists.** Every path in `manifest["artifacts"]` is checked for
   existence and for staying inside the dataset directory, so that set is safe to open
   without an existence test; anything else is *named* in `artifacts_not_generated`.
6. **A pose table never fakes a coordinate.** `openpose.validate()` fails on a null `x`/`y`
   or a non-positive `confidence`, so a missing joint is a missing *row* in `pose/*` and the
   raw JSON beside it says whether the frame was ever processed.
7. **Every Parquet table carries the columns its schema declares.** Checked per stage; a
   stale table written before a column existed names the missing columns instead of raising
   a `KeyError`, which is what tells you which stage to rerun.

**Tier 2 — measured here, not enforced by the pipeline.**

- **Raw survives beside the normalized tables.** Provenance is written to a sidecar
  (`*.provenance.json`) rather than stamped into the tool's own output, so normalization can
  be redone with `--only-stage <stage>` and the original is still there to re-read. That is
  the convention [Two layers on purpose](#two-layers-on-purpose) describes and the stage
  tests for the sidecar hold; nothing re-hashes a raw file to prove it is untouched, so
  after a hand-edit it is your checksum, not the pipeline's.
- **Nulls mean absence and never zero.** `f0_hz` is null exactly when `voiced` is false
  (634 and 367 of 1001 in `pipeline_demo`); `talknet_score` is null exactly when
  `frame_reason` says there is no measurement; `overlap_s` stays null on pyannote rows
  because a `0.0` there would read as "measured: no overlap". Assert it on your own data.
- **Timestamp columns are non-negative, inside the duration and monotonically ordered.**
  True for all 44 non-empty (table, column) pairs of the 63 the seven datasets have — but
  only `acoustic/*` validates ordering, and `finalization` only checks the bounds.
- **`frame_index` has exactly `source.frame_count` rows** with `frame_number` `0..N-1` and
  strictly increasing `pts_seconds`: all seven datasets.
- **Every pose `timestamp` equals its frame's `frame_index.pts_seconds` exactly** — all rows
  of all four pose-bearing datasets, which is what makes `frame_index` a usable join spine.
- **Empty is a result, not a failure.** `pose/body.parquet` with 0 rows means the clip
  contained no person and `pose/raw/` still holds one JSON per processed frame (249 for
  `pipeline_demo`) as proof; `active_speaker_tracks.parquet` with 0 rows means no face was
  ever located. Both are `completed` stages.
- **Id namespaces do not cross.** pyannote's `SPEAKER_00`, Nemotron's
  arrival-ordered `speaker_0`, TalkNet's `track_id` and YOLO's `person_id` are four
  unrelated id spaces in four different files. `track_id` counts **faces** and `person_id`
  counts **bodies**: on the KABC clip TalkNet reports 2 face tracks and YOLO 3 person ids,
  and neither number is wrong. Compare them by time overlap, never by label and never by
  joining `track_id = person_id` — enforced by *separate files and separate schemas*, so
  nothing can join them by accident, but no runtime check will catch you trying.
