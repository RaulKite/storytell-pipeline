# Getting started

Install only the tools you intend to run, configure a local input/output directory,
and process your first video. Root installation and model installation are separate.

Back to the overview: [Storytel Pipeline](../README.md).

Commands use repository-relative paths: run them from the repository root.

## Contents

- [What runs by default](#what-runs-by-default)
- [Prerequisites](#prerequisites)
- [Install](#install)
- [Configure](#configure)
- [Check and run](#check-and-run)
- [Optional synthetic demo](#optional-synthetic-demo)

## What runs by default

Every stage below is implemented at this commit; none is experimental. What differs is
what a fresh clone gets for free.

| Stage | Shipped example | Also needs | Main outputs | Read |
|---|---|---|---|---|
| `metadata` | always | ffprobe | `source/metadata.json`, `source/frame_index.parquet` | [Datasets](datasets.md#table-reference) |
| `audio` | always | ffmpeg | `audio/audio.wav` | [Datasets](datasets.md#directory-layout) |
| `whisperx` | on | its environment | `speech/segments.parquet`, `speech/words.parquet` | [Modalities](modalities.md#speech-and-language) |
| `diarization_nemotron` | on | its environment, model download | `speech/speaker_turns_nemotron.parquet` | [Modalities](modalities.md#speech-and-language) |
| `speaker_assignment` | conditional | Nemotron turns and a transcript | rewrites the transcript tables with `speaker_id` | [Modalities](modalities.md#speech-and-language) |
| `translation` | on | endpoint, key, model | `translation/segments_en.parquet` | [Modalities](modalities.md#speech-and-language) |
| `spacy_source` | on | its environment; a trained model is optional | `linguistic/source/tokens.parquet`, `linguistic/source/sentences.parquet` | [Configuration](configuration.md#language-policy) |
| `spacy_english` | on, if `spacy.process_english` | an English translation, its environment | `linguistic/english/tokens.parquet`, `linguistic/english/sentences.parquet` | [Configuration](configuration.md#language-policy) |
| `acoustic` | on | its environment | `acoustic/frame_features.parquet`, `acoustic/segment_features.parquet` | [Modalities](modalities.md#acoustics) |
| `openpose` | on | OpenPose binary and models under `openpose.root` | `pose/body.parquet`, `pose/hands.parquet`, `pose/face.parquet` | [Modalities](modalities.md#pose-coordinates) |
| `pose_normalized` | on | OpenPose body output | `pose/normalized.parquet` | [Modalities](modalities.md#pose-coordinates) |
| `activespeaker` | **off** | TalkNet-ASD checkout at `activespeaker.talknet_root`, its environment | `speaker/active_speaker_frames.parquet`, `speaker/active_speaker_tracks.parquet` | [Modalities](modalities.md#active-speaker-evidence) |
| `persons` | **off** | its environment; a checkpoint under `persons.weights_dir`, or the worker downloads one | `persons/frames.parquet`, `persons/tracks.parquet` | [Modalities](modalities.md#person-tracks) |
| `stories` | on | endpoint, key, model | `stories/stories.parquet` | [Modalities](modalities.md#story-candidates) |
| `finalization` | always | — | `manifest.json`, `provenance/` | [Datasets](datasets.md#read-a-dataset) |
| `elan` | on | a transcript (words or segments) | `elan/annotations.eaf` | [ELAN export](elan.md) |

- `always` — no enable flag of its own: nothing in configuration turns it off, though a
  stage range, a reused record or a failed upstream can still keep it from executing.
- `on` / `off` — the `enabled` value in
  [`config/config.example.yaml`](../config/config.example.yaml).
- `conditional` — no flag of its own; it runs when the input it reads exists.
  `speaker_assignment` wants Nemotron's turns specifically, so a corpus with
  `diarization_nemotron.enabled: false` leaves it skipped.
- `on` is not a promise: `translation` and `stories` ship with a placeholder endpoint, so
  they skip until you configure them, and `diarization_nemotron` skips when its
  environment is missing. A missing `HF_TOKEN` slows its model download and skips nothing.
- A trained spaCy model is an optimisation, not a prerequisite: with no model for the
  detected language the worker falls back to a blank pipeline and records it, so tokens and
  sentence boundaries arrive with empty lemmas, POS and dependencies.
- An explicit `persons.weights_dir` must hold the checkpoint or the stage skips; unset,
  Ultralytics resolves it and may download it on first use.

Three outputs are opt-in inside an enabled stage: `openpose.write_images`,
`openpose.write_video` and `activespeaker.write_video`. All default to off, none is needed
for a table above, and each one both reruns its stage and costs disk.

The two `off` stages need a download or an external checkout, so a fresh clone does not pay
for them. **Deferred, not shipped as stages:** a renderer for normalized selected-face labels
and a non-English synthetic fixture with e2e coverage. Those are decisions and evidence gaps
rather than half-written code; the [ODD tracker](../odd/tasks/multimodal-video-pipeline.md)
holds them.

## Prerequisites

| Requirement | Used by |
|---|---|
| `uv` and Python 3.12 | Root project and all isolated environment ranges |
| `ffmpeg` and `ffprobe` on PATH | Metadata and extracted audio |
| Compatible GPU/driver, or CPU settings where supported | Enabled ML stages; CPU can be much slower |
| OpenPose binary and models | `openpose`; separate installation under `openpose.root` |
| `HF_TOKEN` (optional, rate-limit convenience) | Nemotron diarization |
| OpenAI-compatible endpoint | Translation and stories; may send text externally and cost money |
| TalkNet-ASD checkout containing `run_talknet.py` | Optional active-speaker stage; set `activespeaker.talknet_root` |

The root project supports Python 3.10–3.13; **3.12** also fits all six worker
environments. Syncing an environment can download packages, and first model use can
download weights. Complete installation before starting an unattended batch.

## Install

Step 1, the root project and the system binaries:

```bash
uv --version
ffmpeg -version
ffprobe -version
uv sync --python 3.12
```

Step 2, one `uv sync` per environment whose stage you enable. Root sync does not install
these, and the numbers are what an enabled stage pulls in:

```mermaid
flowchart LR
    root["uv sync --python 3.12<br/>CLI, Parquet, HTTP stages, ELAN writer,<br/>metadata, audio, speaker_assignment,<br/>pose_normalized, finalization"]
    root --> e1["environments/whisperx → whisperx"]
    root --> e3["environments/diarization_nemotron → diarization_nemotron"]
    root --> e4["environments/spacy → spacy_source, spacy_english"]
    root --> e5["environments/acoustic → acoustic"]
    root --> e6["environments/activespeaker → activespeaker"]
    root --> e7["environments/persons → persons"]
    root --> bin["system binaries: ffmpeg, ffprobe,<br/>OpenPose under openpose.root"]
    root --> cred["credentials in .env: HF_TOKEN,<br/>LITELLM_* for translation and stories"]
```

All six commands, so you can pick the lines you need:

```bash
(cd environments/whisperx && uv sync --python 3.12)
(cd environments/spacy && uv sync --python 3.12)
(cd environments/acoustic && uv sync --python 3.12)
(cd environments/activespeaker && uv sync --python 3.12)
(cd environments/diarization_nemotron && uv sync --python 3.12)
(cd environments/persons && uv sync --python 3.12)
scripts/install_spacy_models.sh
```

The spaCy installer needs the spaCy environment first and installs `en_core_web_lg` when
given no argument; see its options before selecting others. A missing language model
degrades linguistics to tokenisation and sentences rather than failing the video.
For the isolation rationale, see [Why six environments](architecture.md#why-six-environments).

OpenPose is not installed by `uv`. Its default root is `/opt/openpose`; configure
the root or explicit executable/model paths to match your installation. TalkNet is
also external to this repository. Neither checkout is created by the root sync.

GPU compatibility is per environment, not a global rule: each one pins its own CUDA
wheel train, explained by the comments in its `pyproject.toml` (some record the driver and
card measured). Read those rather than comparing numbers — Nemotron's cu128 wheels ran on a
driver reporting CUDA 12.5. A wheel train your driver cannot serve may install cleanly and
then report no CUDA device: check with `inspect-environment` and a short run.

A missing environment directory and an existing one that was never synced are
different failures. `uv` syncs an existing project on first use, so a checkout that was
never synced usually costs time, though that sync can itself fail (network, no wheels for
your platform) and fail the stage. A *missing* directory is refused before `uv` is called:
a skip with a reason for `diarization_nemotron` and `persons`, which check first, and a
stage failure naming the path for every other worker stage.
`inspect-environment` warns about missing project directories before a batch starts.

## Configure

For a fresh setup:

```bash
cp config/config.example.yaml config/config.local.yaml
cp .env.example .env
$EDITOR config/config.local.yaml
$EDITOR .env
```

**Keep existing files rather than overwriting them.** Set `input.directory` and
`output.directory` to real local paths. Put secrets in `.env`; it is read
automatically, with explicitly exported environment variables taking precedence.
[`.env.example`](../.env.example) lists the names — `HF_TOKEN` for the Nemotron
checkpoint, which is not gated and therefore optional, and `LITELLM_BASE_URL`,
`LITELLM_API_KEY`, `LITELLM_MODEL` for the endpoint. Defining a variable does not replace
a literal YAML value; reference it as `${VAR}` or edit it.

The example enables WhisperX, Nemotron diarization, spaCy, acoustics, OpenPose, pose
normalization, translation, stories and ELAN; TalkNet and YOLO are optional and disabled.
Disable any tool you have not installed. Nemotron skips when
`environments/diarization_nemotron` is absent, while a missing enabled OpenPose binary is
an error, not an optional skip. A config that still carries a `diarization:` or
`speaker_fusion:` section is refused with an instruction naming the replacement.

Translation and stories are enabled in the example but their endpoint host is a
placeholder: set each section's `base_url`, `api_key` and `model` deliberately, or set
`enabled: false`. Their `provider: mock` exercises the request/validation path offline
with test data, not a real translation or story analysis. See
[Configuration](configuration.md) for interpolation and policy settings.

## Check and run

```bash
uv run multimodal-pipeline inspect-environment -c config/config.local.yaml
uv run multimodal-pipeline process-video -c config/config.local.yaml data/input_videos/clip.mp4
uv run multimodal-pipeline status -c config/config.local.yaml --video-id clip --plan
uv run multimodal-pipeline validate -c config/config.local.yaml --video data/input_videos/clip.mp4
```

Put your clip in `input.directory`. `--video` and `--video-id` scope the commands that
follow, which otherwise rediscover the configured directory and can report a different
clip; `status` cannot see a file outside it at all. `inspect-environment` prints JSON and
warnings; it is a prerequisite inventory, **not** an inference or credential test.
Typical warnings include `translation endpoint is not configured: translation will be
skipped`, `stories endpoint is not configured: stories will be skipped`, and
`diarization_nemotron: uv project missing at …`. A missing `HF_TOKEN` is not a warning:
the checkpoint is not gated, so the download merely runs anonymous.
An environment directory existing does not prove all of its packages/models work.

`process-video` writes to `<output.directory>/<video_id>/`: `manifest.json` has artifact
paths, `status.json` the per-stage outcomes, `logs/<stage>.log` the failures. A zero-row
table can be a valid result; a skipped stage is different.

For a whole input directory, use `run`; see the [CLI reference](cli.md), which also
walks through one video end to end. To understand the resulting files, go to
[Datasets and analysis](datasets.md).

## Optional synthetic demo

```bash
ffmpeg -filters | grep flite
scripts/make_fixtures.sh
uv run multimodal-pipeline process-video -c config/config.local.yaml data/input_videos/pipeline_demo.mp4
```

The fixture generator requires ffmpeg's **`flite`** TTS filter. It generates the
`pipeline_demo`, NTSC and silent fixtures; speech duration and `-shortest` determine
the final demo length. It can also copy OpenPose's example person clip if available.
The synthetic test-pattern clip has speech but no person to detect: empty pose or
person tables do not indicate that installation failed.
