# ODD Feature: multimodal-video-pipeline

- Workflow: Gentle-AI ODD (Organic Driven Development)
- Feature branch: `master` (fresh repo, no upstream) → feature work committed as reviewable work units
- Status: **IN PROGRESS — Task Group 2 (Foundation)**
- Engram mirror: `odd/multimodal-video-pipeline/tasks` (project scope)

---

## 1. Objective

Build an automatic, sequential, resumable multimodal video-processing pipeline.
Given a configurable input directory of video recordings, produce one dataset
directory per video containing media metadata, extracted audio, WhisperX
transcription with word-level alignment, Pyannote Community diarization,
speaker-assigned transcript, English translation via LiteLLM, spaCy linguistic
features (source + English), Parselmouth acoustic features, OpenPose
BODY_25/hands/face keypoints, all normalized to Parquet on a single video
timeline, plus raw tool artifacts, logs, status, provenance and a manifest.

## 2. Problem / Context

Repository `/data/home/raulagent/miscosas/repos/storytel-pipeline` was empty
(only `.gitignore` + `.atl/` Pi state). Git repo initialized at this path
during exploration; the enclosing `/data/home/raulagent` git dir is unrelated
home-directory state and is not the project root.

The machine is a single-GPU workstation. Everything heavy (WhisperX, Pyannote,
spaCy transformers, OpenPose) contends for one RTX 4090, which justifies the
fixed sequential execution decision.

## 3. Scope

In scope: full orchestrator package, four isolated uv environments, four Python
workers, OpenPose + ffmpeg subprocess stages, LiteLLM HTTP translation stage,
Parquet normalization, semantic validation per stage, atomic resumable state,
config-hash invalidation, CLI (`run/resume/retry-failed/status/process-video/validate/inspect-environment`),
unit tests, per-video manifest, batch report, README.

Out of scope (explicit non-goals): parallel/concurrent processing, GPU
multi-device fan-out, web UI, database storage, speaker *identification*
(names), retraining models, subtitle rendering.

## 4. Architectural decisions

| # | Decision | Rationale |
|---|----------|-----------|
| A1 | Orchestrator is a thin Python package; heavy deps live in 4 uv projects invoked via `uv run --project <env> python workers/<x>.py` | Isolates mutually-incompatible CUDA/torch/transformers pins from the orchestrator's light deps |
| A2 | Sequential execution: one video at a time, one stage at a time | Single RTX 4090; predictable VRAM |
| A3 | Two artifact layers: `raw/` native tool output + normalized Parquet | Normalization is rerunnable without re-running expensive models |
| A4 | Single canonical timeline: seconds from source start; `start_time`/`end_time` for intervals, `timestamp` (+`frame_number`) for instants | Cross-modal joins for `video X, A→B` |
| A5 | Stage interface `prepare/run/validate/outputs/status` + explicit dependency DAG | Uniform resume/invalidation logic |
| A6 | Stage-level `config_hash` + upstream artifact hashes gate reuse | Selective recomputation on config/model change |
| A7 | Speaker assignment is its own stage, preferring pyannote *exclusive* diarization | Community-1 exposes `exclusive_speaker_diarization`; better reconciliation with ASR timestamps |
| A8 | Parquet via PyArrow with explicit schemas + `schema_version`; chunked writers | Bounded memory for long recordings |
| A9 | Secrets masked everywhere (logs, provenance, status) | API key / HF token hygiene |
| A10 | Fixtures generated with ffmpeg `flite` TTS (present on machine) into short real videos | Enables real end-to-end smoke tests without copyrighted media |
| A11 | Translation uses OpenAI-compatible chat endpoint with strict JSON structured output keyed by `segment_id`, batched, cached, backoff-retried | Segment-stable, verifiable translations |

## 5. Implementation constraints

- Python 3.10.12 system; uv-managed CPython 3.12.14 already installed locally
  (`~/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu`).
- whisperx 3.8.6 requires `>=3.10,<3.14`, `torch~=2.8.0` → pin Python 3.12 in
  ML environments.
- CUDA 12.5 driver (555.42.06) with RTX 4090; torch 2.8 cu128 wheels run under
  12.5 driver via minor-version compatibility → **must verify empirically**.
- OpenPose binary is linked against CUDA 11.8 (`/usr/local/cuda-11.8`) +
  `libcudnn.so.9` — prebuilt, do not rebuild.
- No `nvcc` on PATH (irrelevant: no source builds expected).
- System ffmpeg 7.1.1 with `flite` filter available (fixture TTS).

## 6. Machine evidence (gathered 2026-09-23)

```
repository root   /data/home/raulagent/miscosas/repos/storytel-pipeline (git init, branch master, 0 commits)
OS                Ubuntu 22.04.5 LTS, Linux 5.15.0-168-generic, x86_64, host asterion.inf.um.es
CPU               AMD Ryzen 9 7900X3D 12c/24t
RAM               62 GiB (57 GiB available), swap 8 GiB
Disk              /data 11T (7.6T free); / 1.8T (1.7T free)
GPU               1x NVIDIA GeForce RTX 4090 24564MiB (idle)
Driver/CUDA       555.42.06 / CUDA Version 12.5 (reported by nvidia-smi)
CUDA toolkit      /usr/local/cuda -> cuda-11.8 (11.8.0); nvcc NOT on PATH
Python            3.10.12 system; uv 0.12.17; uv CPython 3.12.14 available
ffmpeg/ffprobe    7.1.1 at /usr/bin
Gentle-AI         3.6.1 (doctor: 7 ok / 1 fail: optional `gga` missing)
OpenPose root     /opt/openpose (source tree + build, root-owned, readable)
OpenPose binary   /opt/openpose/build/examples/openpose/openpose.bin (113896 B, 2024-07-10)
OpenPose models   /opt/openpose/models/{pose,hand,face,cameraParameters}
  pose/body_25    pose_deploy.prototxt + pose_iter_584000.caffemodel (104 MB)
  hand            pose_deploy.prototxt + pose_iter_102000/120000.caffemodel
  face            pose_deploy.prototxt + pose_iter_116000.caffemodel (116 MB)
OpenPose links    libcudart.so.11.0, libcublas.so.11, libcudnn.so.9 (resolved)
Sample video      /opt/openpose/examples/media/video.avi (probe in progress)
PyPI reachability 200; huggingface.co API 200
Engram            reachable via MCP (doctor ok)
```

OpenPose CLI facts verified from `openpose.bin --help` (gflags):
`--video`, `--model_folder`, `--model_pose` (e.g. `BODY_25`), `--hand`, `--face`,
`--write_json <dir>`, `--num_gpu`, `--num_gpu_start`, `--display 0`, `--render 0`,
`--disable_multi_thread`. NOTE: `--disable_display` **does not exist** (first probe
failed with `unknown command line flag 'disable_display'`); use `--display 0 --render 0`.

Upstream versions verified from PyPI/official docs (2026-09-23):
```
whisperx                  3.8.6   py>=3.10,<3.14; torch~=2.8.0, ctranslate2>=4.5, faster-whisper>=1.2
pyannote.audio            4.0.7   py>=3.10; torch>=2.8; pyannote-core>=6.0.1
spacy                     3.8.16  official models compatible with 3.8 (85 models listed)
praat-parselmouth         0.4.7
litellm                   1.102.1 (orchestrator uses openai-compatible client directly)
pyarrow                   25.0.1 ; pandas 3.0.6 ; pydantic 2.13.5 ; typer 0.27.2
```
pyannote model `pyannote/speaker-diarization-community-1` — HF revision
`3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`, CC-BY-4.0, 16 kHz mono input,
returns both `speaker_diarization` and `exclusive_speaker_diarization`, GPU via
`pipeline.to(torch.device("cuda"))`, HF access token + EULA acceptance required.

spaCy 3.8-compatible official models confirmed present for: ca, da, de, el, en,
es, fi, fr, hr, it, ja, ko (+ more) — trf variants for en/es/de/fr/ja/..., `lg`
for it/pt/nl etc. Resolver stays config-driven.

## 7. Open questions / external prerequisites

| Q | Blocking? | Handling |
|---|-----------|----------|
| `HF_TOKEN` absent on machine (no `~/.cache/huggingface/token`, no env var) | Blocks **live** pyannote diarization run | Stage implemented + unit-tested; live run gated on token. Asked user once. |
| LiteLLM `base_url`/`api_key`/`model` unknown | Blocks **live** translation run | Stage implemented with mock-server tests; live run gated on endpoint. Asked user once. |
| WhisperX alignment model for arbitrary detected languages | Non-blocking | WhisperX auto-resolves via torchaudio/HF; unsupported language degrades to unaligned words with explicit `alignment_status` |

## 8. Actionable task checklist

Progress legend: `[ ]` pending · `[~]` in progress · `[x]` done (evidence recorded)

### TG1 Environment exploration — DONE
- [x] Resolve project root (git init at repo path, parent home repo ignored)
- [x] Inspect repository (empty) / gentle-ai doctor / Engram reachable
- [x] Python, uv, GPU/CUDA, ffmpeg, OpenPose binary+models, versions recorded
- [x] Upstream dependency compatibility verified against PyPI + official docs

### TG2 Foundation
- [ ] `pyproject.toml` + `src/multimodal_pipeline/` package + console script
- [ ] Typed config (pydantic) + YAML load + env interpolation + secret masking
- [ ] Discovery (ext filter, recursive flag, deterministic sort, stable IDs + hash collision suffix)
- [ ] Stage base interface, DAG, topological order, only/force/from/to selection
- [ ] Atomic `status.json` state machine (pending/running/completed/failed/skipped)
- [ ] Artifact registry + manifest writer
- [ ] Subprocess utils (argv, capture, timeout, log streaming, masked provenance)
- [ ] Logging (console + per-stage log files) + provenance framework (config/tools/processing)
- [ ] Config hashing + invalidation rules
- [ ] CLI: run/resume/retry-failed/status/process-video/validate/inspect-environment
- [ ] Unit tests for all of the above

### TG3 Media
- [ ] ffprobe metadata (rational FPS 24000/1001 etc., streams, creation tags) + SHA256
- [ ] ffmpeg audio extraction (16 kHz mono PCM s16le for WhisperX/Pyannote/Parselmouth, timeline-exact)
- [ ] Frame PTS mapping via ffprobe for precise timestamps
- [ ] Validation + unit tests

### TG4 WhisperX
- [ ] `environments/whisperx` uv project (py3.12, whisperx 3.8.6, cu128 torch) + CUDA smoke
- [ ] `workers/whisperx_worker.py` — detect language, transcribe, segment, align, JSON dump
- [ ] Raw preservation + Parquet normalization (segments, words) + validation
- [ ] Real short-video test

### TG5 Pyannote
- [ ] `environments/diarization` uv project (py3.12, pyannote.audio 4.0.7, cu128 torch)
- [ ] `workers/diarization_worker.py` — community-1, HF_TOKEN, GPU, exclusive diarization, RTTM/JSON raw
- [ ] Normalized `speaker_turns.parquet` + validation

### TG6 Speaker assignment
- [ ] Interval-overlap assignment (segments + words), overlap seconds/ratio, method field
- [ ] Unit tests: inside-one-turn, straddling two turns, multi-turn segment, gaps, overlapped speech, exact boundaries

### TG7 Translation
- [ ] OpenAI-compatible client, contextual batching, structured JSON keyed by segment_id
- [ ] exactly-one-translation validation, retries/backoff, batch cache, raw responses
- [ ] Mock-server tests (real endpoint gated on credentials)

### TG8/TG9 spaCy
- [ ] `environments/spacy` uv project + model install script (multilingual set)
- [ ] Model resolver + multilingual fallback + capability recording
- [ ] source tokens/sentences with WhisperX-word timestamp alignment + alignment status/confidence
- [ ] english tokens/sentences linked to source segments

### TG10 Acoustic
- [ ] `environments/acoustic` uv project (parselmouth) + worker
- [ ] frame_features.parquet (f0, intensity, voiced, F1-F3), segment_features.parquet (aggregates, pauses)
- [ ] NaN semantics + validation + tests

### TG11 OpenPose
- [ ] Installation discovery under /opt/openpose (binary/models/version)
- [ ] Command construction (`BODY_25` + hands + face, GPU configurable, `--display 0 --render 0`)
- [ ] Streaming/chunked raw JSON → body/hands/face Parquet with frame timestamps
- [ ] Validation + tests on synthetic raw JSON

### TG12 Finalization
- [ ] Per-video final validation, manifest from registry, batch_report.json + terminal summary

### TG13 End-to-end verification
- [ ] Synthetic real-content fixture video (flite TTS + person footage) through every stage
- [ ] Inspect dir layout, schemas, row counts, temporal ranges, manifest, status, logs, provenance

### TG14 Resume / failure tests
- [ ] Interrupt + resume, failed stage retry, forced stage, config/model change invalidation, raw-reuse normalization

### TG15 Small sequential batch
- [ ] Several videos, one dataset each, correct state + batch report, bounded memory

### TG16 Final check / ODD close
- [ ] Acceptance criteria 1-26 verified, README, docs schemas, clean reproducible repo, Engram close

## 9. Acceptance criteria

See project prompt §41 (26 criteria). Tracked in §10 evidence table as they are demonstrated.

## 10. Evidence

| Criterion | Evidence | Status |
|---|---|---|
| (populated as tasks close) | | |

## 11. Accepted changes during implementation

- (none yet)

## 12. Next step

TG2 Foundation: scaffold package, config, discovery, DAG, state, subprocess,
manifest/provenance, CLI, with unit tests, committed as work units.
