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
multimodal-pipeline inspect-environment -c config/config.local.yaml
multimodal-pipeline run -c config/config.local.yaml
```

`inspect-environment` before a long run is worth the two seconds: it reports the
ffmpeg/OpenPose/GPU inventory it actually found and lists, in
`environment_warnings`, every stage that is about to be skipped and why.

The four heavy tools live in their own uv projects and must be synced
separately. This is deliberate — see [Why four environments](#why-four-environments).

```bash
(cd environments/whisperx   && uv sync --python 3.12)
(cd environments/diarization && uv sync --python 3.12)
(cd environments/spacy      && uv sync --python 3.12)
(cd environments/acoustic   && uv sync --python 3.12)
scripts/install_spacy_models.sh             # language models you actually need
```

Then generate the test fixtures and run a smoke test end to end:

```bash
scripts/make_fixtures.sh                    # ~1 s, uses ffmpeg's flite TTS
multimodal-pipeline process-video -c config/config.local.yaml data/input_videos/pipeline_demo.mp4
```

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
| `inspect-environment` | Discovered tools, models and GPU/CUDA versions as JSON |

Stage controls on `run` and `process-video`:

```
--only-stage   acoustic              this stage plus whatever it needs
--from-stage   whisperx              start here
--to-stage     acoustic              stop here            (default: finalization)
--force-stage  whisperx,acoustic     recompute even if valid
--video        clip.mp4               one file; a bare name resolves against input.directory
```

`--json` on `status` and `validate`, and `inspect-environment`, write JSON to
**stdout**; the human-readable tables go to **stderr**, so `| jq` and CI capture
work without scraping.

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
│   ├── speaker_turns.parquet     ← diarization output (when available)
│   └── raw/{whisperx,diarization,exclusive_diarization}.json + diarization.rttm
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
│   ├── body.parquet              ← BODY_25 keypoints, one row per person/keypoint
│   ├── hands.parquet             ← 21 points × left/right
│   ├── face.parquet              ← 70 points
│   └── raw/<video>_NNNNNNNNNNNN_keypoints.json   ← OpenPose's own output, untouched
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
  └─ openpose                     (metadata only — no audio, no transcript)

  metadata, audio, whisperx, diarization, speaker_assignment, translation,
  spacy_source, spacy_english, acoustic, openpose  ──►  finalization
```

`openpose` depends only on `metadata`, so a transcription failure never stops pose
extraction — verified: with whisperx failing, `openpose` completed while the speech
chain reported `upstream stage failed: whisperx`. Propagation is **per branch**: a
stage is blocked only by a failed stage in its own dependency chain, and a *skipped*
prerequisite (disabled, or a missing credential) is not a failure — `speaker_assignment`
degrades with its own reason instead of poisoning the run, so a dataset without a
Hugging Face token still gets transcript, linguistics, acoustics and pose.

| Stage | Runs | Needs |
|---|---|---|
| `metadata` | ffprobe + SHA256 | ffmpeg |
| `audio` | ffmpeg → 16 kHz mono | ffmpeg |
| `whisperx` | uv env worker | GPU (or CPU), model download on first use |
| `diarization` | uv env worker | `HF_TOKEN` + pyannote community-1 EULA |
| `speaker_assignment` | in-process interval math | diarization output |
| `translation` | OpenAI-compatible HTTP | `translation.base_url` / `api_key` / `model` |
| `spacy_source` | uv env worker | spaCy model for the detected language |
| `spacy_english` | uv env worker | translation output + `en` model |
| `acoustic` | uv env worker (Parselmouth) | audio |
| `openpose` | `/opt/openpose` binary | OpenPose install + models, GPU |
| `finalization` | in-process | everything above |

A stage whose prerequisites are missing is **skipped with a reason**, not failed:
`missing credential HF_TOKEN (export it to enable diarization)`. `status --plan`
and `validate` both echo those reasons, so you learn what to fix without reading
a log file. The rest of the dataset is still produced and still valid.

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

The interesting sections:

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

### Secrets

Anything matching `token|key|secret|password` is replaced with `***masked***` in
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

## Why four environments

whisperx pins `torch~=2.8.0`, pyannote.audio pulls its own transformers/torchcodec
combination, spaCy wants neither, and the orchestrator should import none of them.
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

Two pins exist because of specific failures, not taste: the driver here is 555.42.06
(CUDA 12.5) and cu126 wheels are what was verified on it; latest `torchcodec` ships a
CUDA-13 build that dies with `libnvrtc.so.13`.

The `mock` translation provider (`translation.provider: mock`) lets you exercise the
whole graph, English linguistics included, with no network and no credentials.

---

## Testing

```bash
uv run --with pytest pytest tests/unit -q     # 561 tests, ~25 s
uv run --with pytest pytest tests/e2e -q      # 28 tests, ~110 s (needs ffmpeg + uv)
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
| `diarization ... missing credential HF_TOKEN` | `export HF_TOKEN=...` and accept the model's EULA on Hugging Face. Everything except speakers and English linguistics still runs without it. |
| `translation endpoint is not configured` | Set `translation.base_url`/`api_key`/`model` (or `provider: mock` to test the graph). |
| `uv project not found: .../environments/whisperx` | Run `uv sync --python 3.12` in that environment directory. |
| `libnvrtc.so.13` on import | `torchcodec` drifted past 0.7.0. Re-pin and re-sync the diarization env. |
| `OpenPose binary not found under /opt/openpose` | Check `openpose.root`; `inspect-environment` prints the resolved path. |
| Everything reruns after one config edit | Expected: `--only-stage X --force-stage X` reruns X and its dependants only. Check `status --plan` for the named reason. |
| `pose/*.parquet` have 0 rows | The video contains no person. That is a valid outcome; the raw JSON in `pose/raw/` confirms it. |
| `acoustic/segment_features.parquet` is empty | No voiced audio (silent track, or music with no speech-like f0). |
| `avg_segments_confidence` is negative | WhisperX reports log-probability-derived segment confidence; word confidences are the 0–1 ones. |

---

## Repo layout

```
src/multimodal_pipeline/   orchestrator: config, discovery, DAG, state, CLI, normalization
workers/                   heavy ML entry points, run inside the isolated envs
environments/              one uv project per dependency-heavy tool
config/                    example template (committed) + local config (ignored)
tests/unit/                561 tests
tests/e2e/                 28 CLI-driven tests
scripts/                   fixture + spaCy model installers
odd/tasks/                 Gentle-AI ODD feature document (decisions, evidence)
data/input_videos/         synthetic fixtures (committed, ~330 KB)
```

`data/processed/` and `data/input_videos/person_demo.avi` are deliberately not
version-controlled: the datasets are fully regenerable, and the person clip is
OpenPose's own example media, copied on demand by `scripts/make_fixtures.sh`.

## License

Not yet declared.
