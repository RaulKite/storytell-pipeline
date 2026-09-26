# multimodal-pipeline

Sequential, resumable multimodal video-processing pipeline.

Give it a directory of video recordings; it produces **one self-contained dataset
directory per video** containing media metadata, extracted audio, WhisperX
transcription with word-level alignment, Pyannote Community diarization, a
speaker-assigned transcript, an English translation through a LiteLLM
OpenAI-compatible endpoint, spaCy linguistic features (source language and
English), Praat/Parselmouth acoustic features and OpenPose BODY_25 + hands + face
keypoints — all normalised to Parquet on a single video timeline, with raw tool
artifacts, logs, status, provenance and a manifest.

Every modality lands on the same clock (`seconds_from_video_start`), so a query
like *"show me everything between 12.4 s and 15.1 s in video X"* is a filter, not
a join across five timestamp conventions.

---

## Quick start

```bash
uv sync --python 3.12                       # orchestrator (light dependencies only)
cp config/config.example.yaml config/config.local.yaml
$EDITOR config/config.local.yaml            # input/output dirs, model, endpoints
cp .env.example .env && $EDITOR .env        # credentials (optional: diarization, translation)
uv run multimodal-pipeline inspect-environment -c config/config.local.yaml
uv run multimodal-pipeline run -c config/config.local.yaml
```

Every command in this file is written as `uv run multimodal-pipeline …`. That is the
form that works with no activation step: `multimodal-pipeline` is a console script of
the root project, so the bare name is only on `PATH` inside an activated
`.venv/bin` (`source .venv/bin/activate`, or `uv shell`) — verified on a fresh clone,
where the bare command is not found. Use whichever form you prefer; they are the same
entry point.

`.env` at the project root is read automatically before the config is interpolated,
so a run needs no shell wrapper and no `source .env`. Variables you export yourself
always win over the file, which keeps CI secrets and one-off overrides working.

`inspect-environment` before a long run is worth the two seconds: it prints JSON on
stdout with the ffmpeg/OpenPose/GPU inventory it actually found. Its
`environment_warnings` name what will cost you data: missing `HF_TOKEN`, an
unconfigured translation endpoint, a missing OpenPose binary or input directory, an
absent uv project directory, a `talknet_root` that will make `activespeaker` skip, and
a missing English spaCy model (which would produce `linguistic/english/*` tables with
empty lemmas — the one language degradation knowable before a run, because English
linguistics always run on English text). Stages with a required install are not listed
because they do not skip — see
[Resume, reuse and invalidation](#resume-reuse-and-invalidation).

The seven heavy tool environments live in their own uv projects and must be synced
separately. This is deliberate — see [Why seven environments](#why-seven-environments).

```bash
(cd environments/whisperx   && uv sync --python 3.12)
(cd environments/diarization && uv sync --python 3.12)
(cd environments/spacy      && uv sync --python 3.12)
(cd environments/acoustic   && uv sync --python 3.12)
(cd environments/activespeaker && uv sync --python 3.12)   # optional: TalkNet needs its own torch
(cd environments/diarization_nemotron && uv sync --python 3.12)  # optional: second diarizer, see below
(cd environments/persons && uv sync --python 3.12)  # optional: person tracking, see below
scripts/install_spacy_models.sh             # language models you actually need
```

Then generate the test fixtures and run a smoke test end to end:

```bash
scripts/make_fixtures.sh                    # ~1 s, uses ffmpeg's flite TTS
uv run multimodal-pipeline process-video -c config/config.local.yaml data/input_videos/pipeline_demo.mp4
```

`make_fixtures.sh` needs the ffmpeg **`flite` filter**, which comes from libflite and is
missing from some packaged ffmpeg builds; when it is absent the script stops and says how
to check (`ffmpeg -filters | grep flite`) rather than dying inside a lavfi parse error. It
also verifies every fixture it writes with ffprobe (exists, ≥ 1 s, has both a video and an
audio stream), because `-shortest` makes the output as long as the shorter stream and an
ffmpeg that exits 0 on junk used to still print "generated …". The clip with a real person
is copied from your OpenPose install: the root is `--openpose-root`, else `openpose.root`
from the config given with `--config`, else from `config/config.local.yaml`, else the
schema default — the same setting the pipeline already uses, and it prints which one it
used. `pipeline_demo.mp4` is 9.985 s rather than the 14 s requested because that is how
long the synthesised speech is; `-shortest` then trims the video to match.

---

## Install from nothing

The quick start is the short form. This is the whole chain in the order that makes each
failure readable, with the check that proves each step and what skipping it costs. The
ordering is not ceremony: the pipeline is built so that a missing prerequisite announces
itself at the step where it can still be fixed — either as an `inspect-environment`
warning before the batch, or as a skip reason naming the setting to change.

**1. Tools, in this order.**

| Step | Check it with | What its absence costs |
|---|---|---|
| `uv` | `uv --version` | Everything. Every command here runs through `uv run`, and every heavy stage is `uv run --project environments/<x> python workers/<x>_worker.py`. `provenance/tools.json` records the version under `system.uv`. |
| Python 3.10–3.13 for the orchestrator | `pyproject.toml` `requires-python = ">=3.10,<3.14"` | The root install fails. Every `uv sync` in the quick start names `--python 3.12`, which is what this corpus was produced with — and it is the only version inside **all seven** environment ranges, which are narrower than the root's: `activespeaker` and `persons` are `>=3.10,<3.13` and `diarization_nemotron` is `>=3.12,<3.13`, so 3.13 is available to the orchestrator and to four of the seven, never to those three. |
| `ffmpeg` and `ffprobe` on `PATH` (or `ffmpeg.executable` / `ffmpeg.ffprobe` in the config) | `ffmpeg -version` | `metadata` and `audio` — and therefore every stage — cannot run. The verified build here is ffmpeg 7.x; `tests/e2e/test_cli_smoke.py::test_the_ffmpeg_major_this_pipeline_was_verified_against` skips rather than fails on another major. |
| ffmpeg's `flite` filter | `ffmpeg -filters \| grep flite` | Only `scripts/make_fixtures.sh`, which synthesises the demo clip with it. A pipeline run is unaffected; the fixture script stops and names the check. |
| OpenPose under `openpose.root` | `inspect-environment` → `tools.openpose.executable` | With `openpose.enabled: true` and no binary, `inspect-environment` prints `OpenPose binary not found under /opt/openpose` and the stage **fails** the run. With `openpose.enabled: false` it skips with `openpose.enabled = false`. |
| `HF_TOKEN` for pyannote community-1 (EULA accepted first) | `inspect-environment` → `HF_TOKEN is not set: diarization will be skipped` | `diarization` skips, so `speaker_turns.parquet` is absent and `speaker_assignment` degrades with its own reason. Transcript, linguistics, acoustics and pose still land. |
| An OpenAI-compatible endpoint for translation | `inspect-environment` → `translation endpoint is not configured: translation will be skipped` | `translation` skips, so `linguistic/english/*` has no English text to read. `translation.provider: mock` exercises the whole graph offline with no credential. |
| A TalkNet-ASD checkout for `activespeaker` (`activespeaker.talknet_root`, containing `run_talknet.py`) | `inspect-environment` → `activespeaker.talknet_root is not set: active speaker detection will be skipped (point it at a TalkNet-ASD checkout to enable it)` | `speaker/*` is absent and `speaker_fusion` skips with it. The example config ships `activespeaker.enabled: false` precisely because this checkout is external to the repository. |

`openpose.root` defaults to `/opt/openpose` and `openpose.executable: auto` searches
`<root>/build/examples/openpose/openpose.bin`, `<root>/bin/openpose.bin` and
`<root>/openpose.bin`, with models under `<root>/models` or
`<root>/share/openpose/models`. That list is not folklore either: it is what
`provenance.openpose_report()` probes, and `tools.json` in a finished dataset records what
it found for that video.

**2. Install the projects.** Root project first, then the seven heavy environments, then
the language models (the quick start has the exact commands). A project directory that
exists but has never been synced is *not* a problem — `uv run --project` resolves and
syncs it on first use, verified on this machine against a throwaway project — so
`inspect-environment` warns only about a genuinely absent directory
(`whisperx: uv project missing at …`). `scripts/install_spacy_models.sh` runs *after*
`(cd environments/spacy && uv sync --python 3.12)`, because it installs into that
environment's own venv and says so when the venv is missing.

That laziness has one cost worth knowing before a long batch: the resolution happens
*inside* the stage that first uses the project, so a dependency that cannot resolve fails
as a stage failure mid-run rather than as an install error up front — and the first stage
of each kind is the slow one, since it pays the download. `uv sync` each environment once
before a batch you want to finish unattended; `inspect-environment` will not tell you that
you skipped it, because skipping it is legitimate.

**3. Secrets, then config, in that order.** `cp .env.example .env` and fill it in; the
committed template is deliberately unusable (`HF_TOKEN=` empty,
`LITELLM_API_KEY=sk-example-not-a-real-key`). Copying it is not enough for translation:
`config/config.example.yaml` leaves `translation.base_url: http://LITELLM_HOST:PORT/v1`,
and the config treats that host as a placeholder, so the endpoint stays "not configured"
until you point it somewhere real. Then `cp config/config.example.yaml
config/config.local.yaml` and set `input.directory` and `output.directory` — until the
input directory exists, `inspect-environment` prints
`input directory does not exist: /data/videos`.

**4. Verify the machine before believing the install.**

```bash
uv run multimodal-pipeline inspect-environment -c config/config.local.yaml
```

The payload is `system`, `tools` (GPU/CUDA, ffmpeg and ffprobe versions, the OpenPose
binary and model inventory, and whether each uv project directory exists) and
`environment_warnings`. What a fresh clone with a copied `.env.example` and the example
config reports on a machine that has ffmpeg, OpenPose and all seven environments is three
lines — and all three are correct:

```json
[
  "HF_TOKEN is not set: diarization will be skipped",
  "translation endpoint is not configured: translation will be skipped",
  "input directory does not exist: /data/videos"
]
```

What it does *not* warn about is worth as much: an environment that exists but has never
been synced (because `uv` syncs it on first use), a stage you explicitly disabled, or a
source-language spaCy model you may never need — the language is not known until after
transcription, so that decision is recorded per video instead of guessed up front.

---

## Commands

| Command | What it does |
|---|---|
| `run` | Process every discovered video, sequentially, one stage at a time |
| `resume` | Continue an interrupted run, reusing every stage whose result is still valid |
| `retry-failed` | Force-recompute exactly the stages that failed, then their dependants |
| `process-video <file>` | Process one file by path (the smoke-test entry point) |
| `status` | Per-video, per-stage state. `--plan` explains every reuse decision, `--json` for scripts |
| `validate` | Re-run semantic validation over artifacts already on disk, without processing |
| `inspect-environment` | Discovered tools, models and GPU/CUDA versions — JSON on stdout, always, no flag needed |

Stage controls on `run`; `process-video` takes the same four stage controls and takes
the file as its positional argument instead of a flag:

```
--only-stage   acoustic              this stage plus whatever it needs
--from-stage   whisperx              start here
--to-stage     acoustic              stop here            (default: finalization)
--force-stage  whisperx,acoustic     recompute even if valid
--video        clip.mp4               one file (run only); a bare name resolves
                                     against input.directory
```

`status` and `validate` take `--json` for machine output, and `inspect-environment`
*is* JSON unconditionally; on all three, the human-readable tables go to **stderr**,
so `| jq` and CI capture work without scraping. (`inspect-environment --json` is not a
thing — it exits 2 with typer's "No such option", measured.)

`status` numbers its stage columns by position instead of naming them, because fifteen
stage names (or even fifteen abbreviations) do not fit the 80 columns `rich` assumes
when its output is not a tty. The legend under the table gives both mappings — the letter
and the stage behind each index — and each video's id is printed whole on its own line
when it is too long to share one:

```
        1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 ov
alpha   c  c  c  c  c  c  c  c  c  c  c  c  c  c  c  c
2017-12-30_1930_US_CNN_Global_Warning_Arctic_Melt_1237_273_1241_393_hear
        c  c  c  c  c  c  c  c  c  c  c  c  c  c  c  c
  c=completed  F=failed  P=partial  .=pending  r=running  s=skipped
  stages: 1=metadata 2=audio … 15=finalization  ov=overall
```

That shape is not cosmetic. The previous `rich` table measured 143 columns against those
80, and `rich` resolves that by stealing width in silence: `video_id` collapsed to a
single `…` and every header to `m…`, exit code 0, so no row said which video it was.
Shortening the headers could not fix it — a real id in this corpus is 72 characters, so
the table never fits. `tests/unit/test_cli_status_layout.py` asserts the structure (which
index holds which stage's mark, and that an id survives as one string) rather than the
presence of a word, because "the id is in the output" is exactly the assertion that
cannot see this failure.

Exit codes are part of the API:

| Code | Meaning |
|---|---|
| `0` | every video completed |
| `1` | `validate` found a broken artifact, or `status` was given an unknown video id |
| `2` | `run`/`resume` had at least one failed or partial video; or a config/usage error |

`run` uses `2` rather than `1` so a CI job can tell "a video needs attention" apart
from "the pipeline was invoked wrongly".

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
│   ├── speaker_turns_nemotron.parquet  ← second engine, if enabled (see below)
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
│   ├── normalized.parquet        ← the same keypoints in a body-centred frame (see below)
│   ├── hands.parquet             ← 21 points × left/right
│   ├── face.parquet              ← 70 points
│   ├── raw/<video>_NNNNNNNNNNNN_keypoints.json   ← OpenPose's own output, untouched
│   └── raw_images/<video>_NNNNNNNNNNNNN_rendered.jpg   ← only with write_images: true
├── speaker/
│   ├── active_speaker_frames.parquet   ← dense on a 25 FPS grid TalkNet invents (see below)
│   ├── active_speaker_tracks.parquet   ← one row per TalkNet face track
│   ├── fusion_pyannote.parquet         ← pyannote turns × per-frame active speaker (see below)
│   ├── fusion_nemotron.parquet         ← same fusion, Nemotron turns, if that engine is selected
│   └── raw/{active_speaker.json,tracks.pckl,scores.pckl,scenes.csv}
├── logs/                         ← pipeline.log + one log per stage
└── provenance/
    ├── config.json               ← resolved config, secrets masked, config hash
    ├── tools.json                ← tool + model versions, GPU/CUDA, OpenPose inventory
    └── processing.json           ← per-stage commands, hashes, durations, errors
```

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

Four figures under [`docs/assets/`](docs/assets/README.md), drawn from the Parquet tables
themselves by `scripts/make_dataset_figures.py` — and built from a **synthetic** dataset,
never from processed video: the pipeline's real input is copyrighted broadcast material,
so no derived frame is committed here. The stage graph comes from `STAGE_ORDER` and
`STAGE_DEPENDENCIES` (dashed = the stage has its own enable check and may skip). The
active-speaker strip puts the dense frame table on one axis — ticks coloured by
`frame_reason`, the score trace, the `is_active_speaker` band — which is where "nothing
was measured" and "the last score was carried" finally look different from each other.

![Pipeline stage graph](docs/assets/stage_graph.png)
![Active-speaker frames strip](docs/assets/active_speaker_strip.png)
![Speaker turns against word timings](docs/assets/speaker_turn_strip.png)
![BODY_25 skeletons from pose/body.parquet](docs/assets/pose_skeleton_strip.png)

Regenerate all four (matplotlib is not a project dependency, so `--with` supplies it; the
output is byte-identical for a given seed, and a test asserts these committed bytes):

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

Output, verbatim, against the dataset committed under `data/processed/` on this machine:

```text
pipeline_demo 9.985 s at 25/1 fps, 249 source frames
temporal_model: {'unit': 'seconds_from_video_start', 'interval_columns': ['start_time', 'end_time'], 'instant_columns': ['timestamp'], 'frame_columns': ['frame_number']}
artifacts: 36 | declared not generated: {}
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
  activespeaker     completed rows={'active_speaker_frames': 249, 'active_speaker_tracks': 0}
  finalization      completed rows={}
```

Two things to notice before opening a single Parquet file. `artifacts_not_generated` is
empty because all 13 stages in `status.json` completed — an absent file is always
*declared* there, never silently missing — and the `pose_*` zeros are a **result**, not a
failure: the clip is a synthetic test pattern with a synthesised voice over it, so there is
no person for OpenPose to find. `manifest.json` carries the same 36 keys in `artifacts`
(key → dataset-relative path) and their sizes in `artifact_details`.

Why 13 stages and 36 artifacts when the stage table above lists more? Because a manifest
describes the run that produced it. This dataset was last written before `pose_normalized`
and `speaker_fusion` were added to the queue, so its `stage_order` has 13 entries and its
manifest lists 36 of the 40 artifacts the registry can now declare; the other four
(`pose/normalized.parquet`, `pose/raw_images`, `speaker/fusion_pyannote.parquet`,
`speaker/fusion_nemotron.parquet`) are absent from every dataset under `data/processed/`
because no run here has produced them since. Another reason to read `status.json` rather
than assume the queue: it is the file that records what actually ran.

Now the files, with what those numbers mean:

| File | Rows | What it decides |
|---|---|---|
| `source/metadata.json` | — | ffprobe verbatim: 640×480, `frame_rate_rational` `"25/1"`, `frame_count` 249, `duration_seconds` 9.985, plus the source SHA256. Every other table is checked against this duration. |
| `source/frame_index.parquet` | 249 | `(frame_number, pts_seconds)` for each of those 249 frames — the authority that turns a second into a frame. First row `(0, 0.0)`, last `(248, 9.92)`. |
| `audio/audio.wav` | — | 16 kHz mono s16le; `audio/audio_info.json` records the exact ffmpeg argv and the resulting 160 768 frames / 10.048 s. |
| `speech/words.parquet` | 23 | Word-level times. Row 0: `word_id` `seg000001-w00000`, `start_time` 0.233, `end_time` 0.554, `duration` 0.321, `speaker_id` `SPEAKER_00`, `word` `Hello`, `confidence` 0.709, `alignment_status` `aligned`. 16 columns; `speaker_id` plus the three `speaker_overlap_seconds` / `speaker_overlap_ratio` / `speaker_assignment_method` columns are written by `speaker_assignment`, which re-reads this table with `segments.parquet` and `speaker_turns.parquet` and rewrites both — not by WhisperX. |
| `speech/segments.parquet` | 1 | The whole utterance is one segment, `seg000001`, 0.233→9.689 s, `language` `en`, `confidence` −0.1412 (log-probability-derived — see Troubleshooting). |
| `speech/speaker_turns.parquet` | 2 | pyannote's **exclusive** timeline: `turn000001` 0.199719→1.026594 and `turn000002` 1.127844→9.869094, both `SPEAKER_00`, `diarization_type` `exclusive`. |
| `speech/speaker_turns_nemotron.parquet` | 1 | The second engine's answer on the same audio. Different id namespace — never join it to the row above on `speaker_id`. |
| `translation/segments_en.parquet` | 1 | `segment_id` `seg000001` again (every translation row must resolve to a transcript segment — `finalization` checks it), `source_language` `en`, `translation_model` `chat`, `translation_prompt_version` `v1`. |
| `linguistic/source/tokens.parquet` | 26 | spaCy over the source text. Token 0: `text` `Hello`, `lemma` `hello`, `pos` `INTJ`, `dep` `intj`, `head_token_id` `seg000001-s001-t0002`, and `token_start_time` 0.233 / `token_end_time` 0.554 inherited from the word alignment, with `timestamp_alignment_status` `aligned`. |
| `linguistic/source/sentences.parquet` | 1 | One sentence, `seg000001-s001`, `token_count` 26 — so the token and sentence tables agree by construction. |
| `linguistic/english/{tokens,sentences}.parquet` | 28 / 2 | The same schema over the English translation, variant `english`. 28 tokens for 26 source tokens: translation is not a 1:1 map, which is why the two variants are separate files. |
| `acoustic/frame_features.parquet` | 1001 | 10 ms Praat frames. 634 rows are `voiced` and **exactly those 634 carry a non-null `f0_hz`**; the other 367 are null. `timestamp` runs 0.024→10.024 in steps of exactly 0.01 — note the tail: the acoustic grid is laid over `audio.wav`, which is 10.048 s, so it can run slightly past the 9.985 s video. That is inside the duration + 1 s `finalization` allows, not a defect. |
| `acoustic/segment_features.parquet` | 1 | Per-segment F0/intensity/formant aggregates and pause statistics, keyed by `segment_id`. |
| `pose/{body,hands,face}.parquet` | 0 / 0 / 0 | The honest empty case: schema present, about 2 KB each, no person in frame. `pose/raw/` still holds all 249 `_keypoints.json` files, so the emptiness is auditable rather than asserted. |
| `speaker/active_speaker_frames.parquet` | 249 | Dense on the 25 FPS grid, and on this clip the grid *is* the source grid. Every row is `face_status='no_face'`, `frame_reason='no_face'`, `score_imputed=False`, `is_active_speaker=False`, `track_id=None`. |
| `speaker/active_speaker_tracks.parquet` | 0 | No track, because no face was ever located. |
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
  skip strings quoted elsewhere in this file live.
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
- **Speaker-id namespaces do not cross.** pyannote's `SPEAKER_00`, Nemotron's
  arrival-ordered `speaker_0` and TalkNet's `track_id` are three unrelated id spaces in
  three different files. Compare them by time overlap, never by label — enforced by
  *separate files and separate schemas*, so nothing can join them by accident, but no
  runtime check will catch you trying.

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
[Why seven environments](#why-seven-environments). If the environment is absent the stage
*skips* with the reason naming the fix, and the corpus still completes, because a second
opinion is not a prerequisite.

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
fused tables on `speaker_id` any more than you join the two turn tables. `face_track_id` is a
third id space again (a TalkNet track), never a speaker id. `overlap_s` is carried from the
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
defining joint was never measured) or `basis_degenerate` (both were, and they coincide, so the
determinant is zero). Per keypoint, `value_status` says whether the joint got coordinates, lost
its own coordinate, or had nowhere to be put. Every person-frame that has a keypoint still has
rows, because a dropped row is a person who stopped existing. A zero is never used for absence:
on these axes a zero means *exactly at the hip*, which is a measurement.

The stage depends on `openpose` **only** — no audio, no transcript, no diarizer — so a
transcription failure never costs the normalised poses, and it inherits openpose's skip
semantics: no `pose/body.parquet`, no table, with the reason naming which switch to turn on.
It runs in-process over Parquet, like `speaker_fusion`: one table in, one out, no
subprocess, no uv environment, seconds per video (37 857 keypoints of `person_demo` in 0.2 s
measured here).

---

## Stage graph

```
metadata
  ├─ audio
  │    ├─ whisperx ────┐
  │    ├─ diarization ─┴─ speaker_assignment
  │    │                                  ├─ translation ── spacy_english
  │    │                                  ├─ spacy_source
  │    └─ acoustic ◄──────────────(also)──┘
  ├─ openpose                     (metadata only — no audio, no transcript)
  └─ activespeaker                (metadata + audio — independent of the transcript)

  diarization + diarization_nemotron + activespeaker  ──►  speaker_fusion
                                  (turns × frames — it reads tables, it runs no model)

  metadata, audio, whisperx, diarization, speaker_assignment, translation,
  spacy_source, spacy_english, acoustic, openpose, activespeaker, speaker_fusion
                                                                        ──►  finalization
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
| `speaker_fusion` | in-process Parquet arithmetic | diarization turns **and** `activespeaker` output |
| `finalization` | in-process | everything above |

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
[How far to trust the detected language](#how-far-to-trust-the-detected-language)), and the
decision lands in provenance: `language_reliability` (the grade, or
`{"status": "absent"}`) and `language_reliability_trusted` in
`linguistic/source/raw/spacy_source.json`. It changes nothing else — not the transcript,
not the acoustics, not pose.

---

## Configuration

`config/config.example.yaml` is the documented template (a test loads it, so it
cannot drift from the schema). Values support `${VAR}` and `${VAR:-default}`
environment interpolation, so credentials stay out of the file. Unknown keys are
rejected with the valid names listed:

```
error: invalid configuration (config/config.local.yaml):
  logging.capture_subprocess_output: unknown setting (known: console, level,
progress_interval_seconds)
```

The interesting sections (a sketch of the schema, not a copy of the shipped file:
`config/config.example.yaml` leaves `translation.base_url` literal as
`http://LITELLM_HOST:PORT/v1` and pins `model: claude-sonnet-4-5`, so the `${…}` form shown
for them is what you add, not what is there):

```yaml
input:    {directory: /data/input_videos, recursive: false}
output:   {directory: /data/processed}
execution: {mode: sequential, gpu: 0, stop_on_video_error: false}

whisperx:
  model: large-v3          # any Whisper arch
  language: auto           # or an ISO code to force it
  device: cuda             # cpu works, slowly
  compute_type: float16

diarization:
  pipeline: pyannote/speaker-diarization-community-1
  hf_token_env: HF_TOKEN   # the *variable name*; its value is never written
  num_speakers: null       # or min_speakers / max_speakers to constrain it

translation:               # any OpenAI-compatible endpoint (LiteLLM in particular)
  base_url: ${LITELLM_BASE_URL:-http://LITELLM_HOST:PORT/v1}
  api_key: ${LITELLM_API_KEY:-}
  model: ${LITELLM_MODEL:-}
  batch_size: 10           # segments per request
  context_segments: 2      # neighbours sent as context, never re-translated

spacy:
  english_model: en_core_web_lg
  source_models: {en: en_core_web_lg, es: es_dep_news_trf, ...}
  fallback_model: blank    # any language still gets tokens + sentences
  trust_low_language_detection: true   # false = no full pipeline on a shaky detection

acoustic:
  time_step: 0.01          # 10 ms, Praat's default pitch granularity
  pitch_floor: 75.0
  pitch_ceiling: 500.0
  chunk_seconds: 120.0     # bounded memory on long recordings

openpose:
  root: /opt/openpose      # binary and models are discovered under here
  body: {enabled: true, model: BODY_25}
  hands: {enabled: true}
  face:  {enabled: true}
```

Install only the spaCy models whose language you actually expect. `spacy_model` in the
table's Parquet schema metadata (and `selected_model` in the raw JSON beside it) records
which one ran, and `blank` is a degradation you can see. A *wrong*
model is worse, because it looks like a result: on Spanish text, `ca_core_news_lg`
labelled `Muy buena entrada` as three `PROPN` tokens with the first as `ROOT` and
plausible dependencies attached, where the blank pipeline's empty lemmas were at least
honest about knowing nothing. Resolution goes configured language → same family → any
installed model → blank, so an installed model *will* be picked up for a language it
does not serve. Either install the model that matches the language or leave that family
out and accept `blank`.

Installing or removing a model invalidates the linguistics stages instead of leaving
them serving a cached `blank` result: which models exist changes the output as much as
the configuration does.

#### How far to trust the detected language

The language the resolver is handed is WhisperX's *detection*, and WhisperX grades its
own work: `speech/raw/whisperx.json` carries `language_detection` with status
`configured` (you pinned `whisperx.language`), `ok`, or `low` — `low` when the audio is
shorter than WhisperX's 30 s detection window or the probability is missing or below
0.5. `spacy.trust_low_language_detection` decides what `spacy_source` does with `low`.

**`true` (default)** keeps the model the detection selected and logs a warning naming the
language, the probability, the model that was still chosen, and the key that reverses it:

```text
language 'es' was auto-detected with low reliability (probability 0.88; audio is 8.0s,
below the 30s detection window); the full pipeline for it was still selected
(model='es_core_news_lg', status='configured') because
spacy.trust_low_language_detection = true — set spacy.trust_low_language_detection =
false to require a trustworthy detection before building the linguistic layer.
```

Both texts are copied from what the worker logs, not paraphrased.

The default is `true` because the grade is pessimistic on short clips: on this
pipeline's corpus every auto-detected clip is `low` — seven out of seven, because every
clip is under 30 s — and the resulting choice is measurably correct where it was checked
(Spanish lemmas from a real Spanish pipeline on the Spanish clip, `en_core_web_lg` on the
English ones). Demoting `low` unconditionally would take the linguistic layer off the
only Spanish clip in the corpus on the strength of a grade that never says `ok` here.

**`false`** refuses to build a full linguistic layer on a sub-window guess. No language
reaches the resolver, so it reports what it can do without one and the raw document says
so (`model_selection_status: fallback_no_model`, model `blank`, capabilities
tokenisation + sentencizer — no lemmas, POS tags or dependencies). The warning names the
demotion and the key that reverses it:

```text
language 'es' was auto-detected with low reliability (probability 0.88; audio is 8.0s,
below the 30s detection window). spacy.trust_low_language_detection = false, so the
detection was not used to choose a model: the source variant was demoted to model='blank'
(status='fallback_no_model', capabilities=tokenization,sentencizer), which still yields
tokens and sentences but no lemmas, POS tags or dependencies. Set
spacy.trust_low_language_detection = true to accept a low-confidence detection again.
```

Both branches record the decision in provenance — `language_reliability` (the grade
itself, or `{"status": "absent"}`) and `language_reliability_trusted` — in
`linguistic/source/raw/spacy_source.json` and in the worker result. `spacy_english` is
excluded: it forces `en`, so it has no detection to trust and never sees the grade.

If no grade is available — a dataset produced before WhisperX graded anything, or a raw
file that could not be read — the model choice is left exactly as it was and the worker
warns that reliability could not be checked. A missing check is never reported as a
passed one.

### Secrets

Credentials live in an untracked `.env` at the project root — never in the YAML.
A config file gets copied between machines and pasted into issues; a pipeline whose
secrets are embedded in it leaks every time someone shares their config. The YAML
references them as `${VAR}` / `${VAR:-default}` and `load_config` reads `.env` first:

```bash
HF_TOKEN=hf_...                      # pyannote community-1 (accept its EULA first)
LITELLM_BASE_URL=https://host/v1     # any OpenAI-compatible endpoint
LITELLM_API_KEY=sk-...
LITELLM_MODEL=chat
```

`.env.example` is the committed template with fake values. Precedence is
environment-over-file on purpose: a stale `.env` from another account can never
shadow an explicit `HF_TOKEN=... uv run multimodal-pipeline run` or a CI secret. A
malformed line names itself (`.env:2: expected KEY=value`) and exits 2 rather than
raising a traceback. `MULTIMODAL_PIPELINE_NO_DOTENV=1` skips the implicit read —
that is what keeps the test suite independent of the credentials on your machine.

Anything matching `token|key|secret|password` is then replaced with `***masked***` in
logs, `status.json`, manifests, provenance and the batch report. Keys that *name*
where a secret lives (`hf_token_env: HF_TOKEN`) survive masking on purpose — the
variable name is documentation, its value is not. An e2e test greps every file a
run writes for two planted credentials and fails if either appears.

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

---

## Testing

```bash
uv run --with pytest pytest tests/unit -q     # 1300 tests, ~35 s
uv run --with pytest pytest tests/e2e -q      # 42 tests, ~110 s (needs ffmpeg + uv)
```

Unit tests avoid mocks wherever a mock would hide the bug: media tests call real
ffmpeg/ffprobe (including the NTSC `30000/1001` case), translation tests drive a real
threaded `HTTPServer` so status codes, timeouts and retry timing are actually
exercised, the uv-worker tests build a throwaway uv project so the trust boundary is
real, and OpenPose normalization runs against a real frame JSON captured from the
binary.

`tests/e2e/test_cli_smoke.py` invokes the `multimodal-pipeline` command as a
subprocess against a real on-disk project, because exit codes and stream routing are
part of the API: it asserts a corrupted file in the input directory fails *that*
video without stopping the batch, that killing a run mid-stage never leaves the
in-flight stage `completed`, that a second run rewrites nothing, and that no
credential reaches any written file.

Mutation-checked behaviour (restoring the defect fails a test): resume invalidation,
per-branch failure propagation, artifact integrity, the cross-modal checks, secret
masking, and the manifest's promise that every listed artifact exists.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `diarization ... missing credential HF_TOKEN` | Put `HF_TOKEN=...` in `.env` and accept the model's EULA on Hugging Face. Everything except speakers and English linguistics still runs without it. |
| `translation endpoint is not configured` | Set `translation.base_url`/`api_key`/`model` (or `provider: mock` to test the graph). |
| `uv project not found: .../environments/whisperx` | Run `uv sync --python 3.12` in that environment directory. |
| `libnvrtc.so.13` on import | `torchcodec` drifted past 0.7.0. Re-pin and re-sync the diarization env. |
| `OpenPose binary not found under /opt/openpose` | Check `openpose.root`; `inspect-environment` prints the resolved path. |
| Everything reruns after one config edit | Expected: `--only-stage X --force-stage X` reruns X and its dependants only. Check `status --plan` for the named reason. |
| `pose/*.parquet` have 0 rows | The video contains no person. That is a valid outcome; the raw JSON in `pose/raw/` confirms it. |
| `activespeaker.talknet_root is not set` | Clone TalkNet-ASD and set `activespeaker.talknet_root`. Without it the stage skips and the rest of the dataset is unaffected. |
| `activespeaker.talknet_root has no run_talknet.py` | That path is not a TalkNet-ASD checkout — the stage checks for the entrypoint rather than letting a confusing `torch.load` error surface minutes later. |
| TalkNet dies with an unpickling or `weights_only` error | The environment drifted past torch 2.5. Re-sync `environments/activespeaker`; the pin is load-bearing (see [Why seven environments](#why-seven-environments)). |
| `speaker/active_speaker_frames.parquet` has rows with `track_id = null` | No face was detected in those frames — off-screen, back-turned, or too small. S3FD tracks near-frontal faces only; a person walking away legitimately loses the track. Absence is recorded as a row, not dropped. `frame_reason = no_face` says the same thing from the score side. |
| `score_imputed = true` on some frames | TalkNet scores fewer frames than it tracks (an unexplained `-1` in its MFCC windowing), so the last score was carried forward rather than measured. At most two frames per track are affected; treat those as unmeasured, not as low confidence. `frame_reason = imputed_tail` names the same rows. |
| A frame has `face_status = "tracked_unscored"` | S3FD located a face there but TalkNet produced no usable score for it. `frame_reason` names which of the four causes: `score_not_finite` (a score existed and was NaN/inf), `track_has_no_scores` (the track never produced one), `past_scored_tail` (beyond the two frames that may be imputed), or `tail_score_not_finite` (inside the window, the value to carry was broken). The scores stay `null` and the frame is never active — this is "we could not measure", not "nobody was on screen" and not a confidence of zero. |
| A frame has `frame_reason = "unknown"` | That table was normalised from a raw artifact written before the field existed, so the cause is not recoverable. Rerun `activespeaker` to get a diagnosed table. |
| `acoustic/segment_features.parquet` is empty | No voiced audio (silent track, or music with no speech-like f0). |
| `avg_segments_confidence` is negative | WhisperX reports log-probability-derived segment confidence; word confidences are the 0–1 ones. |

---

## Repo layout

```
src/multimodal_pipeline/   orchestrator: config, discovery, DAG, state, CLI, normalization
workers/                   heavy ML entry points, run inside the isolated envs
                         (whisperx, diarization, nemotron diarization, spacy,
                         acoustic, activespeaker)
environments/              one uv project per dependency-heavy tool
config/                    example template (committed) + local config (ignored)
tests/unit/                1300 tests
tests/e2e/                 42 CLI-driven tests
scripts/                   fixture + spaCy model installers, dataset figure renderer
docs/assets/               committed figures (synthetic-schema demos, regenerable)
odd/tasks/                 Gentle-AI ODD feature document (decisions, evidence)
data/input_videos/         synthetic fixtures (committed, ~330 KB)
```

`data/processed/` and `data/input_videos/person_demo.avi` are deliberately not
version-controlled: the datasets are fully regenerable, and the person clip is
OpenPose's own example media, copied on demand by `scripts/make_fixtures.sh`.

## License

MIT. The `LICENSE` file is the text; `pyproject.toml` declares the same SPDX
identifier, so the machine-readable and human-readable declarations agree.
