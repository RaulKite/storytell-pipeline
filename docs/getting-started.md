# Getting started

Install only the tools you intend to run, configure a local input/output directory,
and process your first video. Root installation and model installation are separate.

Back to the overview: [Storytel Pipeline](../README.md).

Commands use repository-relative paths: run them from the repository root.

## Contents

- [Prerequisites](#prerequisites)
- [Install](#install)
- [Configure](#configure)
- [Check and run](#check-and-run)
- [Optional synthetic demo](#optional-synthetic-demo)

## Prerequisites

| Requirement | Used by |
|---|---|
| `uv` and Python 3.12 | Root project and all isolated environment ranges |
| `ffmpeg` and `ffprobe` on PATH | Metadata and extracted audio |
| Compatible GPU/driver, or CPU settings where supported | Enabled ML stages; CPU can be much slower |
| OpenPose binary and models | `openpose`; separate installation under `openpose.root` |
| `HF_TOKEN`, with pyannote community-1 terms accepted | pyannote diarization |
| OpenAI-compatible endpoint | Translation and stories; may send text externally and cost money |
| TalkNet-ASD checkout containing `run_talknet.py` | Optional active-speaker stage; set `activespeaker.talknet_root` |

The root project supports Python 3.10–3.13; **3.12** also fits all seven worker
environments. Syncing an environment can download packages, and first model use can
download weights. Complete installation before starting an unattended batch.

## Install

```bash
uv --version
ffmpeg -version
ffprobe -version
uv sync --python 3.12
```

Sync the environments needed by your enabled stages (all available commands shown):

```bash
(cd environments/whisperx && uv sync --python 3.12)
(cd environments/diarization && uv sync --python 3.12)
(cd environments/spacy && uv sync --python 3.12)
(cd environments/acoustic && uv sync --python 3.12)
(cd environments/activespeaker && uv sync --python 3.12)
(cd environments/diarization_nemotron && uv sync --python 3.12)
(cd environments/persons && uv sync --python 3.12)
scripts/install_spacy_models.sh
```

The spaCy installer needs the spaCy environment first. Install language models you
expect to use; see the script's options before selecting them.
For the isolation rationale, see [Why seven environments](architecture.md#why-seven-environments).

OpenPose is not installed by `uv`. Its default root is `/opt/openpose`; configure
the root or explicit executable/model paths to match your installation. TalkNet is
also external to this repository. Neither checkout is created by the root sync.

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

The example enables WhisperX, pyannote, spaCy, acoustics and OpenPose. Nemotron,
TalkNet and YOLO are optional and disabled in the example. Disable any tool you
have not installed. Missing `HF_TOKEN` skips pyannote; a missing enabled OpenPose
binary is an error, not an optional skip.

Translation and stories are enabled in the example but their endpoint host is a
placeholder. Set each section's `base_url`, `api_key` and `model` deliberately, or
set `enabled: false`. Their `provider: mock` exercises the request/validation path
offline; its text is test data, not a real translation or story analysis.
See [Configuration](configuration.md) for interpolation and policy settings.

## Check and run

```bash
uv run multimodal-pipeline inspect-environment -c config/config.local.yaml
uv run multimodal-pipeline process-video -c config/config.local.yaml path/to/clip.mp4
uv run multimodal-pipeline status -c config/config.local.yaml --plan
uv run multimodal-pipeline validate -c config/config.local.yaml --json
```

Replace `path/to/clip.mp4` with your file. `inspect-environment` prints JSON and
warnings; it is a prerequisite inventory, **not** an inference or credential test.
Typical warnings include `HF_TOKEN is not set: diarization will be skipped`,
`translation endpoint is not configured: translation will be skipped`, and
`stories endpoint is not configured: stories will be skipped`.
An environment directory existing does not prove all of its packages/models work.

`process-video` writes to `<output.directory>/<video_id>/`. Open `manifest.json`
for artifact paths, `status.json` for per-stage outcomes, and `logs/<stage>.log`
for failures. A zero-row table can be a valid result; a skipped stage is different.

For a whole input directory, use `run`; see [CLI reference](cli.md). To understand
the resulting files, go to [Datasets and analysis](datasets.md).

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
