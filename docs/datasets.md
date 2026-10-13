# Datasets and analysis

Start from the manifest to discover outputs, then read the typed Parquet tables.
This guide maps the files and explains the clocks, identifiers and missing values
that matter when combining modalities.

Back to the overview: [Storytel Pipeline](../README.md).

Commands use repository-relative paths: run them from the repository root.

## Contents

- [Directory layout](#directory-layout)
- [Table reference](#table-reference)
- [Speaker timing example](#speaker-timing-example)
- [Read a dataset](#read-a-dataset)
- [Consumer rules](#consumer-rules)

## Directory layout

```text
<output.directory>/<video_id>/
├── manifest.json       artifact paths, details, declared absences and temporal model
├── status.json         stage outcomes, fingerprints and recorded validation results
├── source/             metadata.json + frame_index.parquet
├── audio/              audio.wav + audio_info.json
├── speech/             transcript, alignment, diarization + raw tool output
├── translation/        English segments + raw responses/cache
├── linguistic/         source/ and english/ tokens, sentences + raw results
├── acoustic/           frame and segment features + raw JSONL
├── pose/               body, hands, face, normalized + raw/ and optional raw_images/
├── persons/            detections, tracks + raw/yolo_track.json
├── speaker/            active-speaker evidence, fusion + raw/
├── stories/            candidates + raw responses/cache
├── elan/               annotations.eaf
├── logs/               pipeline.log + <stage>.log
└── provenance/         config.json, tools.json, processing.json
```

This is the possible layout, not a promise that every file exists in every run.
`manifest.artifacts` maps logical artifact names to **dataset-relative paths**;
`artifacts_not_generated` declares absent outputs. Artifact totals vary with
configuration, available prerequisites and input. Do not hardcode a corpus count.

Set `activespeaker.write_video: true` to retain TalkNet's native annotated video
with audio as `speaker/raw/video_out.avi` (`activespeaker_video_raw` in the manifest).
It is opt-in and requires ASD recomputation. Turning it off preserves an existing
video; manifest presence reports files on disk, not the current toggle.

Raw outputs remain beside normalized tables. Provenance is stored in sidecars and
the provenance directory rather than injected into original raw bytes. The config
snapshot masks secrets, but logs/tool provenance can still contain machine paths.

## Table reference

These are the registered Parquet outputs. For every field/type, consult
[`schemas.py`](../src/multimodal_pipeline/schemas.py); for all raw and bookkeeping
paths, consult [`ARTIFACT_LAYOUT`](../src/multimodal_pipeline/artifacts.py).
Presence is determined by the manifest, not this inventory.

| Artifact key | Dataset-relative path | Meaning |
|---|---|---|
| `frame_index` | `source/frame_index.parquet` | Source frame number → PTS seconds |
| `speech_segments` | `speech/segments.parquet` | Transcript segments, language and assigned speaker |
| `speech_words` | `speech/words.parquet` | Words, alignment state and confidence |
| `speaker_turns` | `speech/speaker_turns.parquet` | pyannote intervals |
| `speaker_turns_nemotron` | `speech/speaker_turns_nemotron.parquet` | Independent Nemotron intervals |
| `translation_segments` | `translation/segments_en.parquet` | English text keyed by transcript segment |
| `spacy_source_tokens` | `linguistic/source/tokens.parquet` | Source token annotations and timing state |
| `spacy_source_sentences` | `linguistic/source/sentences.parquet` | Source sentence annotations |
| `spacy_english_tokens` | `linguistic/english/tokens.parquet` | English translation token annotations |
| `spacy_english_sentences` | `linguistic/english/sentences.parquet` | English translation sentence annotations |
| `acoustic_frames` | `acoustic/frame_features.parquet` | Pitch, intensity, voicing and formants |
| `acoustic_segments` | `acoustic/segment_features.parquet` | Segment aggregates and pause statistics |
| `pose_body` | `pose/body.parquet` | BODY_25 keypoints in source pixels |
| `pose_hands` | `pose/hands.parquet` | Left/right hand keypoints |
| `pose_face` | `pose/face.parquet` | Face landmarks |
| `pose_normalized` | `pose/normalized.parquet` | Body-centred keypoints and basis/value status |
| `active_speaker_frames` | `speaker/active_speaker_frames.parquet` | Dense selected-face activity on TalkNet's 25 FPS grid |
| `active_speaker_tracks` | `speaker/active_speaker_tracks.parquet` | TalkNet face-track summaries |
| `person_frames` | `persons/frames.parquet` | YOLO detections and body-track IDs |
| `person_tracks` | `persons/tracks.parquet` | Body-track summaries and sighting gaps |
| `speaker_fusion_pyannote` | `speaker/fusion_pyannote.parquet` | pyannote turn × face evidence agreement |
| `speaker_fusion_nemotron` | `speaker/fusion_nemotron.parquet` | Nemotron turn × face evidence agreement |
| `stories` | `stories/stories.parquet` | LLM candidates with transcript evidence and request identity |

## Speaker timing example

Speaker-turn intervals and word-timing markers are separate records. This synthetic
example illustrates `speaker_turns` alongside `speech_words`, not measured accuracy
or data from a recording. See [Consumer rules](#consumer-rules) before joining them.

![Synthetic schema demo: speaker turns and word timings](assets/speaker_turn_strip.png)

## Read a dataset

Run the following with `uv run python` in the root environment. Replace the dataset
path with one produced by your configuration; there is no required local demo dataset.

```python
import json
from pathlib import Path
import pyarrow.parquet as pq

dataset = Path("data/processed/pipeline_demo")
manifest = json.loads((dataset / "manifest.json").read_text())
status = json.loads((dataset / "status.json").read_text())

print("source:", manifest["source"])
print("not generated:", manifest["artifacts_not_generated"])
for name, stage in status["stages"].items():
    print(name, stage["status"])

def read(name):
    relative = manifest["artifacts"].get(name)
    if relative is None:
        raise KeyError(f"Artifact not produced: {name}")
    return pq.read_table(dataset / relative)

words = read("speech_words")
print(words.schema)               # field types and metadata
print(words.slice(0, 5).to_pylist())
```

Use Arrow directly for large tables. For pandas analysis, add pandas to your analysis
environment and call `read("speech_words").to_pandas()`. Group within a single video
before joining; IDs from two datasets need not be globally unique.

## Consumer rules

| Rule | Safe interpretation |
|---|---|
| Time unit | Seconds from video start; intervals use `start_time`/`end_time`, instants use `timestamp` |
| Source clock | `frame_index.pts_seconds` maps source frames; metadata logs a fallback if packet timing cannot be read |
| Acoustic clock | Praat's configured time step, not video-frame numbers; audio duration can differ slightly from container duration |
| TalkNet clock | One row per 25 FPS working-grid frame, **not** per source frame; `timestamp` is grid time and `source_timestamp` maps to the nearest source PTS |
| Track namespaces | Each diarizer's `speaker_id`, YOLO `person_id` and TalkNet `track_id` are independent |
| Explicit face link | Fusion `face_track_id` refers to TalkNet `track_id` in the same dataset, not to a diarizer speaker |
| Empty vs absent | Zero rows with a schema can mean a completed analysis found nothing; absence needs a stage/manifest reason |
| Missing vs zero | Null is not a measured zero; inspect alignment, basis, face and model-capability status fields |
| Imputed vs measured | `score_imputed` means carried evidence, not a fresh TalkNet measurement |

For pose/ASD comparisons, join on `source_timestamp`, never on `frame_number`:
the TalkNet index counts re-encoded frames. Map pose's source `timestamp` to that
column, preserving the possibility that several working-grid rows map to one source
frame. Do not assume a one-to-one frame join, even when both columns are integers.
For interval/instant joins, state your boundary rule explicitly (usually
`start_time <= timestamp < end_time`) and inspect overlapping diarizer turns.

Translation joins transcript segments by `segment_id`, not word position. English
tokens are not one-to-one source tokens, and their times can be approximate or segment
context. Speaker assignment rewrites transcript tables with pyannote labels; it does
not rename the original WhisperX raw output.

Read [Modalities](modalities.md) before interpreting confidence, person counts or
story candidates. Read [ELAN export](elan.md) for what an annotation summary leaves out.
