# Getting started

Install the pipeline and get one video through it, from a machine that has nothing yet: what to install, in which order, the check that proves each step, and what skipping it costs.

Back to the overview: [Storytel Pipeline](../README.md).

Commands use repository-relative paths: run them from the repository root.

## Contents

- [Quick start](#quick-start)
- [Install from nothing](#install-from-nothing)

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

Every command in this guide is written as `uv run multimodal-pipeline …`. That is the
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
[Resume, reuse and invalidation](architecture.md#resume-reuse-and-invalidation).

The seven heavy tool environments live in their own uv projects and must be synced
separately. This is deliberate — see [Why seven environments](architecture.md#why-seven-environments).

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
| The same endpoint for story detection | `inspect-environment` → `stories endpoint is not configured: stories will be skipped` | `stories` skips with the reason in its status record. It is on by default and one of two stages that spend tokens per video (`translation` is the other); `stories.provider: mock` answers without any endpoint. |
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
config reports on a machine that has ffmpeg, OpenPose and all seven environments is four
lines — and all four are correct:

```json
[
  "HF_TOKEN is not set: diarization will be skipped",
  "translation endpoint is not configured: translation will be skipped",
  "stories endpoint is not configured: stories will be skipped",
  "input directory does not exist: /data/videos"
]
```

The two endpoint warnings sit adjacent on purpose: `translation` and `stories` share one
gateway and both default to on, so a fresh install is always missing both or neither, and
`stories` and `translation` are the two stages that cost tokens when they run.

What it does *not* warn about is worth as much: an environment that exists but has never
been synced (because `uv` syncs it on first use), a stage you explicitly disabled, or a
source-language spaCy model you may never need — the language is not known until after
transcription, so that decision is recorded per video instead of guessed up front.
