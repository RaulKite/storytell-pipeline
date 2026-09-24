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
  (`~/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu`). All four uv environments and the
  orchestrator now run on 3.12.
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
repository root   /data/home/raulagent/miscosas/repos/storytel-pipeline (git init, branch master)
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

### TG2 Foundation — DONE
- [x] `pyproject.toml` + `src/multimodal_pipeline/` package + console script
- [x] Typed config (pydantic, `extra="forbid"`) + YAML load + env interpolation + secret masking
- [x] Discovery (ext filter, recursive flag, deterministic sort, stable IDs + hash collision suffix)
- [x] Stage base interface, DAG, topological order, only/force/from/to selection
- [x] Atomic `status.json` state machine (pending/running/completed/failed/skipped)
- [x] Artifact registry + manifest writer
- [x] Subprocess utils (argv, capture, timeout watchdog, bounded tails, masked provenance)
- [x] Logging (console + per-stage log files) + provenance framework (config/tools/processing)
- [x] Config hashing + invalidation rules (config hash + monotonic per-video run sequence)
- [x] CLI: run/resume/retry-failed/status/process-video/validate/inspect-environment
- [x] Unit tests (config 25, discovery 28, state 19, artifacts/manifest/report 48,
      subprocess 37, uv_worker 22)

### TG3 Media — DONE
- [x] ffprobe metadata (rational FPS, streams, creation tags) + SHA256
- [x] ffmpeg audio extraction (16 kHz mono PCM s16le, timeline-exact)
- [x] Frame PTS mapping via ffprobe (`source/frame_index.parquet`, 249 rows on the demo)
- [x] Validation + unit tests against real ffmpeg (NTSC 30000/1001 case included)

### TG4 WhisperX — DONE
- [x] `environments/whisperx` (whisperx 3.8.6, torch 2.8.0+**cu126**) + CUDA verified
- [x] `workers/whisperx_worker.py` — language detect, transcribe, segment, align, JSON dump
- [x] Raw preserved byte-identical + Parquet normalization (segments, words) + validation
- [x] Real GPU test: large-v3 transcribed the demo, 23/23 words aligned, `en` auto-detected

### TG5 Pyannote — DONE (env verified; live inference needs a token)
- [x] `environments/diarization` (pyannote.audio 4.0.7, torch 2.8.0, torchcodec 0.7.0 pinned)
- [x] `workers/diarization_worker.py` — community-1, HF_TOKEN, GPU, exclusive diarization, RTTM+JSON
- [x] Normalized `speaker_turns.parquet` + validation
- [x] Without `HF_TOKEN` the stage reports `skipped: missing credential HF_TOKEN` and every
      downstream stage degrades cleanly instead of failing (verified on the 4-video batch)
- [x] Live diarization run — DONE with a user-supplied `HF_TOKEN`: GPU inference on all six videos, 2 speakers in the Kimmel clip, 0 in the silent fixture. Three real defects were found and fixed by this run (see section 16).

### TG6 Speaker assignment — DONE
- [x] Interval-overlap assignment (segments + words), overlap seconds/ratio, method field
- [x] 38 unit tests: inside-one-turn, straddling two turns, multi-turn segment, gaps,
      overlapped speech, exact boundaries

### TG7 Translation — DONE (client verified; live endpoint needs credentials)
- [x] OpenAI-compatible client, contextual batching, structured JSON keyed by `segment_id`
- [x] exactly-one-translation validation, retries/backoff, batch cache, raw responses
- [x] 52 tests against a real threaded `HTTPServer` (status codes, timeouts, retry timing)
- [x] Live LiteLLM run — DONE against the user's vLLM endpoint (two usable chat models probed; `chat` selected for latency, `modelo-gordo` also works and its reasoning preamble parses). Its `tts` endpoint returns HTTP 500, so it could not be used to synthesize non-English fixtures.

### TG8/TG9 spaCy — DONE
- [x] `environments/spacy` + `scripts/install_spacy_models.sh` (multilingual set)
- [x] Model resolver + multilingual fallback (`blank` + sentencizer) + capability recording
- [x] source tokens/sentences with WhisperX-word timestamp alignment + status/confidence
- [x] english tokens/sentences linked to source segments (27 worker tests)

### TG10 Acoustic — DONE
- [x] `environments/acoustic` (praat-parselmouth 0.4.7 / Praat 6.1.38) + worker
- [x] frame_features.parquet (f0, intensity, voiced, F1-F3) + segment_features.parquet
- [x] NaN semantics + validation + 62 tests; real run measured 1001 frames, f0 median 147.6 Hz

### TG11 OpenPose — DONE
- [x] Installation discovery under /opt/openpose (binary/models/version)
- [x] Command construction (`BODY_25` + hands + face, `--display 0 --render 0`)
- [x] Streaming raw JSON → body/hands/face Parquet with frame timestamps
- [x] Golden-frame test + real 205-frame run (37 857 body / 56 054 hands / 37 768 face rows)

### TG12 Finalization — DONE
- [x] Cross-modal validation (timeline bounds, unresolvable segment/speaker ids)
- [x] Manifest generated from the registry, batch_report.json + terminal summary
- [x] 21 tests, mutation-verified

### TG13 End-to-end verification — DONE
- [x] 28 CLI-driven e2e tests (`tests/e2e/test_cli_smoke.py`) running the real command
- [x] Synthetic flite fixtures + real-person OpenPose clip through every stage
- [x] Layout, schemas, row counts, temporal ranges, manifest, status, logs, provenance inspected

### TG14 Resume / failure tests — DONE
- [x] 24 resume/DAG tests + e2e kill-mid-run/resume test
- [x] failed stage retry, forced stage, config-change invalidation, artifact truncation detected

### TG15 Small sequential batch — DONE
- [x] 4 videos → 4 datasets, 0 failed, 2m01s, correct status + batch report
- [x] Second run reuses everything (0 s); third run confirms it

### TG16 Final check / ODD close — DONE
- [x] README written; acceptance criteria mapped to evidence in §10
- [x] Repo reproducible: 74 tracked files, generated datasets and third-party media excluded
- [x] Engram session summary saved

## 9. Acceptance criteria

Mapped to observable evidence in §10. Two criteria (live diarization, live
translation) were gated on credentials this machine did not have when this section
was written; they are now verified live — see criteria 10 and 12. What remains
unverifiable here is a **non-English source**: flite ships English-only voices, there
is no espeak, and the endpoint's tts returns 500, so language detection, the `es`
spaCy model and genuine es->en translation are covered only by unit tests against a
real HTTP server. Legacy text retained for context: their code
paths are tested against a real HTTP server and a real uv worker contract, and
the stages degrade to `skipped` with an actionable reason rather than failing.

## 10. Evidence

Test suite: **561 unit + 28 e2e tests, all passing**
(`timeout 300 uv run --with pytest pytest tests/unit -q` → 561 passed in ~25 s;
`pytest tests/e2e` → 28 passed in ~108 s).

Real batch (`uv run multimodal-pipeline run -c config/config.local.yaml`, clean
output, 4 videos, sequential):

```
Videos discovered: 4   Completed: 4   Partial: 0   Failed: 0   Total time: 02m 01s
person_demo          completed 59.1s   pose+acoustic+whisperx+spacy_source completed
pipeline_demo        completed 21.8s
pipeline_demo_ntsc   completed 23.0s
pipeline_silent      completed 15.9s
```

| # | Criterion | Evidence | Status |
|---|---|---|---|
| 1 | One output dataset dir per video | 4 dirs under `data/processed/`, one per fixture | ✅ |
| 2 | Configurable input/output dirs | `input.directory` / `output.directory` in YAML; e2e fixture uses tmp dirs | ✅ |
| 3 | Sequential execution, one stage at a time | `execution.mode` accepts only `sequential` (validated); stage loop is serial | ✅ |
| 4 | ffprobe metadata incl. rational frame rate | `frame_rate_rational "30000/1001"` on the NTSC fixture; `test_media.py` uses real ffprobe | ✅ |
| 5 | Source SHA256 + size | `source.SHA256` in metadata and manifest | ✅ |
| 6 | 16 kHz mono PCM audio | `audio/audio.wav`; `test_media.py` asserts rate/channels/sample width | ✅ |
| 7 | Frame index with true PTS | `source/frame_index.parquet`, 249 rows for the demo | ✅ |
| 8 | WhisperX transcription + word alignment | GPU run: 1 segment, 23 words, 100 % `alignment_status=aligned`, `language=en` | ✅ |
| 9 | Language auto-detection | `language="auto"` resolved to `en`, surfaced in manifest `source.detected_language` | ✅ |
| 10 | Pyannote community-1 diarization | **Live GPU run** with a real token: community-1 found 2 speakers in the KABC/Kimmel clip (0.03-4.25 SPEAKER_00, 3.05-3.14 SPEAKER_01), 1 in the CNN clip, and correctly 0 turns in pipeline_silent. Turns land in speech/speaker_turns.parquet. | ✅ |
| 11 | Speaker-assigned transcript + overlap metrics | 38 unit tests on real interval arithmetic; `speaker_overlap_ratio`, `speaker_assignment_method` columns present | ✅ (code) |
| 12 | English translation via LiteLLM | **Live run** against a real OpenAI-compatible endpoint (vLLM, model chat): 1 request, 281 tokens, 450 ms; 'Now that you say that, I can remember hearing your voice at the Laker game.' became 'Now that you mention it, I remember hearing your voice at the Lakers game.' translation/segments_en.parquet populated and spacy_english ran on it. | ✅ |
| 13 | spaCy linguistics, source language | 26 tokens × 32 columns, POS/lemma/dep/NER + per-token timestamps | ✅ |
| 14 | spaCy linguistics, English | same schema under `linguistic/english/`, runs when translation exists | ✅ (code) |
| 15 | Acoustics via Parselmouth | 1001 frames (`f0_hz`, `intensity_db`, `voiced`, `f1..f3_hz`) + 1 segment row | ✅ |
| 16 | OpenPose BODY_25 + hands + face | real run on `person_demo.avi`: 37 857 / 56 054 / 37 768 rows over 205 frames | ✅ |
| 17 | Single normalized timeline | `temporal_model` declared in every manifest; finalization rejects timestamps < 0 or > duration | ✅ |
| 18 | Raw tool output preserved | `speech/raw/whisperx.json`, `pose/raw/*.json`, `acoustic/raw/*.jsonl` stay byte-identical (sidecar provenance) | ✅ |
| 19 | Normalization independently rerunnable | `--only-stage` + raw-preserving workers; raw reuse covered in resume tests | ✅ |
| 20 | Parquet with explicit schemas + version | every table carries `schema_version` and `video_id` | ✅ |
| 21 | Per-video manifest describing the dataset | `manifest.json`, generated from the registry; `validate` proves each promise exists | ✅ |
| 22 | `status.json` + per-stage logs | 13 log files + atomic state per dataset | ✅ |
| 23 | Provenance (config/tools/processing) with secrets masked | `provenance/config.json` shows `api_key: ***masked***`, keeps `hf_token_env: HF_TOKEN`; e2e greps every written file | ✅ |
| 24 | Resume after interruption | e2e kills a real run mid-stage, asserts the in-flight stage is not `completed`, then `resume` finishes it | ✅ |
| 25 | Idempotent rerun | run 2 and run 3 reuse all 28 stage results (0 s); artifact row-count fingerprint catches hand-truncated Parquet | ✅ |
| 26 | Batch report + honest per-video status | `batch_report.json` with counts and per-stage outcomes; a video whose first stage failed is `failed`, not `partial` | ✅ |

## 11. Accepted changes during implementation

Deviation from the original spec, each with the reason:

1. **cu126 torch wheels, not cu128.** The driver is 555.42.06 (CUDA 12.5); cu126
   wheels verified working on the RTX 4090.
2. **`torchcodec==0.7.0` pinned** in the diarization env — the latest release pulls a
   CUDA-13 build that fails with `libnvrtc.so.13`.
3. **WhisperX worker written against the installed 3.8.6 API**, not the README
   (`transcribe()` has no `vad=`/`beam_size=`; beam size lives in `asr_options`).
4. **Sidecar provenance files** (`*.provenance.json`) instead of stamping raw artifacts,
   so raw tool output stays byte-identical as required.
5. **`en_core_web_lg` as the default spaCy model** rather than `_trf`: a router model
   with vectors and no torch dependency, so it never fights the CUDA-pinned envs.
   `scripts/install_spacy_models.sh` still installs `_trf` on request.
6. **Artifact layout uses `linguistic/`** (not `linguistics/`) and
   `speech/{segments,words}.parquet` (not `transcript_*`).
7. **`speaker_assignment` degrades instead of cascading** when diarization is absent, so
   a missing HF token still yields transcript + linguistics + acoustics + pose.

## 12. Real defects found and fixed while verifying

Each of these was surfaced by a test or a real run, not by reading the code, and each
has a regression test verified by mutation (restoring the defect fails the test):

| Defect | Impact | Fix |
|---|---|---|
| `finalization` fingerprint included the sizes of files finalization itself wrote | every run called it stale and rewrote the whole summary, forever | fingerprint only what it does not write; dependency hash still covers content changes |
| Registry treated an empty directory as an artifact | manifest promised `translation_raw` with 0 files (observed on `pipeline_demo`) | an empty directory is not a dataset |
| No artifact integrity check | truncating `frame_index.parquet` passed `validate` | Parquet row-count fingerprint recorded at completion, checked by reuse and validate |
| Cross-check ordering | speaker/segment consistency never validated without a translation table | three independent cross-checks |
| `overall_status` counted skips as usable | a video whose first stage failed was `partial` | `failed` means nothing usable was produced |
| Worker errors reported only an exit code | the worker's own diagnosis was buried in stderr | the worker's reported status/error is the headline |
| Invalidation used mtimes | this filesystem rounds mtimes to ~16 ms, so a fast rerun looked unchanged | monotonic per-video `run_sequence` |
| Machine output printed to stderr | `--json \| jq` got nothing | JSON on stdout, human tables on stderr |
| Unknown YAML key raised a pydantic traceback | the actionable message was invisible | names the offending key and the valid ones |
| `schemas._coerce` called `table.append_columns` (plural) | a table missing a declared column raised AttributeError instead of writing nulls (latent: no stage hits it today) | use pyarrow's singular `append_column`; 100 % coverage on `schemas.py` |
| `config.example.yaml` did not load | shipped template was unusable (`no:` parsed as boolean; misplaced key) | fixed, and a test now loads it |

## 13. Native review outcome (RDD)

RDD is enabled, so the final work-unit commit was frozen for native review:

```
lineage      review-0d54936f4c1fd6cd
candidate    f342aa0..HEAD (7 paths, 518 changed lines, tier high)
lenses       review-risk, review-resilience, review-readability, review-reliability
outcome      did NOT close
```

Three of the four reviewers were captured and submitted; the fourth capture failed,
and bound STATUS returned `action: stop`, `state: escalated`,
`reason_code: native_stop_required` with `escalation.cause = unknown_causality` on
finding `R3-001`. Per the lifecycle contract a `stop` ends the transition and never
approves delivery, so **this candidate has no review approval**, and the finding's
causality is unknown rather than established as pre-existing.

What that means concretely: the commit stands on its own verification (589 passing
tests, mutation-checked regressions, real batch), not on a review receipt. Resolving
`R3-001` requires maintainer inspection of the lineage
(`gentle-ai review inspect`/`recover`), which is a deliberate human decision, not
something to route around by restarting the transaction. The two earlier attempts also
burned a consent binding that expired unused.

## 14. Remaining work (user decisions)

1. ~~Live diarization~~ — DONE, see criterion 10.
2. ~~Live translation + `spacy_english`~~ — DONE, see criterion 12.
2b. **A non-English source video** — the only remaining credential-free gap. Needs a real
   clip with non-English speech to exercise language detection, the `es` spaCy model
   and genuine es->en translation. Cannot be synthesized on this machine.
2c. **`.env` support** — implemented: `load_dotenv`, environment-over-file precedence,
   `ConfigError` naming the offending line, the
   `MULTIMODAL_PIPELINE_NO_DOTENV` escape hatch, and a committed `.env.example`.
3. `push` / pull-request remain the user's call; 12 work-unit commits exist on `master`.

## 15. RESUME POINT (session interrupted by context limit)

Read this first when continuing. Persisted also in Engram (project `raulagent`,
title starting "RESUME POINT").

**Repository state**: branch `master`, HEAD `8885891`, 16 commits, clean tree except
`.gitignore` modified. 561 unit + 28 e2e tests green. Not pushed yet, remote not added.

**In flight — `.env` credential support.** Decision: credentials go in an untracked
`.env` at the project root and reach the pipeline only through the existing `${VAR}`
interpolation, so nothing secret lives in `config/config.local.yaml` (a config file
gets copied, pasted into issues and committed far more easily than a dedicated secret
file). Done so far:

- `.gitignore` extended with `.env`, `.env.*`, `!.env.example` — **verified** with
  `git check-ignore` (`.env.example` correctly NOT ignored).
- Verified no credential ever reached the working tree or any commit
  (`git log --all -S` for each secret returns nothing).
- Blocked: the agent's write tool refuses `.env`/`.env.example` as a sensitive path
  under the Gentle AI safety policy. Awaiting the user's explicit plan. Proposed: the
  user creates `.env` themselves; the agent commits only `.env.example` (fake values).

**Still to implement** in `src/multimodal_pipeline/config.py`: `load_dotenv()` reading
the project-root `.env`, with real environment taking precedence over the file (so
`HF_TOKEN=... multimodal-pipeline run`, and CI secrets, still win), quoted values,
`#` comments, malformed line -> `ConfigError`, missing file not an error. Keep
interpolation single-pass: `_ENV_RE`'s default group is `[^}]*`, so no nested `${}`.

**Live endpoints, already probed** (values deliberately not recorded here):
reachable; models available are `chat`, `modelo-gordo`, `embeddings`, `whisper`, `tts`.
`chat` answered in 0.17 s but wraps JSON in ```` ```json ```` fences; `modelo-gordo`
took 0.69 s and emits a reasoning preamble — `parse_structured_translations` already
strips fences, so either works. `tts` returns HTTP 500. ffmpeg's flite ships only
English voices (`awb kal kal16 rms slt`) and there is no espeak, so Spanish speech
**cannot** be synthesized on this machine: a real Spanish clip is the only way to
exercise language detection, the `es` spaCy model and es→en translation.

**Next, in order**: 1) resolve the `.env` guard; 2) implement `load_dotenv` + tests;
3) live run of diarization + translation + spacy_english over the fixtures and
re-verify; 4) `git remote add origin git@github.com:RaulKite/storytell-pipeline.git`
and push (deploy key verified working against the empty repo), after re-checking the
pushed tree contains no secret; 5) real videos from the user.

**Requested real videos** (30–90 s, small): one with **two or more Spanish speakers**
(the only way to test diarization + `es` linguistics + real es→en translation), one
with a visible body and hands (OpenPose on real movement, not colour bars), one quiet
or ambient (empty-transcription path on real audio). Drop them in `data/input_videos/`
— that directory's contents are gitignored except the three committed fixtures, so
they stay local and out of the push.

## 16. Live credentials: what the first real run actually caught

Credentials arrived after the review escalation, so diarization and translation ran
for the first time ever. All three defects below were invisible until then, and each
was found by running rather than by reading:

| # | Symptom | Root cause | Why tests missed it |
|---|---|---|---|
| 1 | every completed diarization rejected as `raw result was produced by a different configuration` | `validate()` compared a `request_hash` read out of the worker's JSON, which never contains one → `None != hash` always | the stage always skipped without a token, so `validate` never ran |
| 2 | `person_demo`/`pipeline_silent` failed validation with `diarization: outputs missing` | the RTTM was written only when truthy; a no-speech diarization has a legitimately empty RTTM | same — plus the empty-transcript path was never reached |
| 3 | `speaker_assignment: expected bytes, NoneType found`, then `segments table is empty after assignment` | `write_table` passed `diarization_type=None` into Parquet metadata (string-only, rejected inside Cython); `validate` demanded a non-empty transcript and averaged over it | unit tests never combined a speaker timeline with zero segments |

Plus a dead `from pyannote.core import Segmentation` (removed in pyannote 4.x) that
made the worker fail at import on the first real invocation.

Each fix carries a mutation-verified test: the diarization validation test was run
against the original buggy code and fails exactly as production did, and reverting
only the `if rttm:` guard to its old truthy form fails the RTTM test.

**Final state**: clean six-video batch (four fixtures + two real broadcast clips) in
03m37s, 6 completed / 0 partial / 0 failed, `validate` globally `ok: true`, rerun in
6s reusing every stage. 610 tests pass (561→610 unit+e2e growth this session).

## 17. Active speaker detection (TalkNet-ASD) — 2026-09-24

Added as a new `activespeaker` stage plus a fifth uv environment, at the user's
request, reusing their own per-frame active-speaker script's semantics.

### What landed (six work-unit commits, each independently green)

| Commit | Unit |
| --- | --- |
| `8d54c54` | `WorkerStage.worker_timeout` received no context — config timeouts were dead code |
| `afb666a` | spaCy fingerprint carries the installed-model inventory; install script fixed |
| `653204a` | `environments/activespeaker` uv project (torch 2.5.1+cu124 pin) |
| `fdc5963` | `workers/activespeaker_worker.py` (pure logic + TalkNet child process) |
| `78e25a7` | config section, artifact layout, two Parquet schemas |
| `516c12b` | the stage, DAG wiring, 39 tests |
| `abadea0` | README + example config |

### Live verification

- TalkNet smoke run on the KABC clip: 105 frames at 25 fps, 1 scene, 2 tracks, exit 0,
  inference 1.48 s. Scores of length 104 against tracks of 105 — the operator script's
  tail imputation is load-bearing, not superstition.
- Selection sanity on real pickles: track 0 (mean +1.13) won 78/105 frames over track 1
  (mean −0.94); every emitted frame lands within 0.02 s of a real source frame.
- Full batch: 7 videos (4 fixtures + 2 real clips + La 1), Completed 7, Failed 0, 6m49s;
  `speaker/active_speaker_frames.parquet` and `active_speaker_tracks.parquet` written for
  all of them. Rerun idempotent in 11 s. `validate --json`: global `ok: true`, 7/7 videos,
  0 problems, 0 skipped.
- Test suite: 659 passing (629 unit + 30 e2e). Four activespeaker tests are mutation-
  locked (density check, request-hash coverage, active-frame-without-face,
  score_imputed preservation).

### Defects this feature caught

1. `worker_timeout()` had no `ctx` parameter — every `*.timeout_seconds` setting in the
   config was unreachable.
2. `validate()` crashed with a bare `TypeError` on a row with `track_id` set but null
   bbox; must be a `ValidationError` naming the frame.
3. `track_summary_rows()` subtracted possibly-null bbox coordinates.
4. spaCy fingerprint omitted the installed-model inventory: installing `es_core_news_lg`
   did not invalidate a cached `blank` result (found while enabling Catalan/Spanish).
5. `install_spacy_models.sh` built wheel URLs from the spaCy version (models ship 3.8.0,
   spaCy is 3.8.16 → 404), and its `spacy download` fallback reported success without
   linking into the uv venv.

### Known limitation, documented in README

A language model that does not match the text language is worse than `blank`:
`ca_core_news_lg` on Spanish output labelled "Muy buena entrada" as three PROPNs, first
token ROOT. WhisperX auto-detected the La 1 clip as Catalan (`ca`), which pulls in the
Catalan model. Resolution prefers family matches, but a discovered model of any family
still beats an honest `blank`. Kept visible via `spacy_model` in the tokens table.

### Review disposition

The first candidate (all 15 paths as one workspace target) was rejected by the provider
with `lens_context_budget_exceeded` — no authority created, candidate evidence never
truncated. Response: split into the six work units above, each of which the provider can
size independently. Native review of the individual units was not started this session;
no review receipt exists and none is claimed. Push follows ordinary repository policy
and the user's standing authorization, not a review outcome.

## 18. Native review receipt for the activespeaker slice (2026-09-24)

Splitting the 15-path candidate worked. The provider sized the stage slice
(`README.md`, `config/config.example.yaml`, this document, `orchestrator.py`,
`stages/activespeaker.py`, `stages/base.py`, `tests/unit/test_activespeaker.py` — 7 files,
925 changed lines, medium tier, one `review-reliability` lens) and it closed **approved**.

- Lineage `review-66668d42e1a11acc`, store revision `sha256:a16e970f…c9b1a`.
- Acknowledgement completed: `burn_evidence: gentle-ai.review-acknowledged/v1`,
  `authority: burned`, `delivery: ordinary-repository-policy`.
- One advisory finding, explicitly non-blocking and reopening nothing:
  **R3-001** (WARNING, `stages/activespeaker.py:264-267`) — the dense-sequence check
  rejects a frame table that is dense-but-reordered rather than logging the anomaly.
  Verified: the stage has no logging call at all, so this is a real observation about the
  whole stage, not just those lines. Filed as follow-up work, not fixed under this receipt.

### Follow-ups carried out of this review

1. `activespeaker` emits no warnings anywhere; a malformed-but-parseable worker result is
   silently strict-rejected. Add the stage's first warning log where a distinct diagnosis
   is possible (R3-001).
2. Frames past the two that may be imputed fall into the same `track_id = null` rows as
   "no face detected"; `score_imputed` cannot distinguish them because no row is emitted
   for them at all. Documented in README; a real fix needs a distinct reason column.
3. WhisperX auto-detected the La 1 clip as Catalan, so source linguistics use
   `ca_core_news_lg` and produce confidently wrong POS on Spanish text. Needs either a
   genuinely Spanish clip or a policy decision about trusting auto-detection.

## 19. Autonomous piloting (2026-09-24)

The operator asked the agent to pilot this repository without asking: choose the best
option, always.

- **`gentle-ai review mode disable --scope clone`** — RDD is now `off (decided by
  clone_local)` for this clone, which removes the per-candidate review consent prompt.
  Verified with `review mode status`. This is the operator's sanctioned user-owned
  switch; it is scoped to this clone and reversible with `... mode enable`.
- **`.pi/modes.config.json`** — `defaultMode: "yolo"`, versioned, with `extraInstructions`
  that append (verified in `pi-agent-modes/src/config.ts`: `extra` is concatenated to
  `baseInstructions`, never replaces it) rather than overwrite the built-in mode text.
- **`AGENTS.md`** — the decision policy that replaces asking: prefer the reversible
  option, announce rather than ask, and three reservations kept with the operator
  (force-push/history rewrite/remote deletion/merge, anything outside this repository,
  hand-made operator data). Plus the commands and constraints that cost a failed run
  each: `-p no:randomly`, `validate` has no `-o`, writers need `## Allowed edit
  surfaces`, worktree launches fail in this clone.

Verified rather than assumed:
- `hasTrustRequiringProjectResources(...)` is **false** for this repo, so
  `projectTrusted` resolves true and `.pi/modes.config.json` loads with **no trust
  prompt**. `modes.config.json` is deliberately not in pi's
  `TRUST_REQUIRING_PROJECT_CONFIG_RESOURCES` list; `.pi/settings.json` or `.pi/SYSTEM.md`
  **would** add a prompt, so they are gitignored and not created.
- No installed extension registers a `project_trust` handler, so nothing else can
  re-introduce a prompt.
- `AGENTS.md` loads regardless of trust (pi docs, `docs/security.md`).

Note for a fresh clone: RDD is on by default elsewhere, so this document's claim that
review consent is disabled describes *this* clone. An agent must read
`gentle-ai review mode status` rather than assume it.

## 20. Requested next capabilities (operator, 2026-09-24) — not started

Ordered by how much they depend on what already exists. Each is a separate feature
entry with its own task list when it starts; none of them is scoped yet.

### 20.1 `diarization_v2` — fuse pyannote with active speaker detection

Output a *second* diarization result alongside v1 rather than replacing it: v1 is
audio-only and is the input to `speaker_assignment`, acoustics and the existing
datasets. A replacement would silently relabel every existing artefact.

The fusion is the interesting part and it is not a join. pyannote answers *when does a
voice speak*; `activespeaker` answers *which visible face is talking at 25 FPS*. They
agree often and disagree in exactly the cases that matter: an off-screen narrator (voice
with no face), a cutaway shot (voice continues over a different face), two people
visible while only one talks, and a face that keeps moving its mouth while silent.
Those disagreements are signal, not noise, so the table should keep the audio turn, the
winning face per turn and an explicit agreement state per turn rather than flattening to
one label.

Depends on: `diarization` + `activespeaker` (so it lands after both, and inherits both
their skip semantics — if either is absent there is nothing to fuse and it must skip with
a reason, not emit an empty table).

### 20.2 `persons` — YOLO detection + tracking, persons per video

Ultralytics YOLO for person detection with a tracker, to answer things no current stage
answers: how many distinct people appear in a video, when each is on screen, and whether
a scene changed. Cross-checking the tracker's own re-identification against `scenedetect`
cuts is worth doing deliberately: `activespeaker` already trusts scenedetect for scene
ids, so a disagreement between the two is a defect report on one of them.

Constraints that will shape this more than the model choice:
- **It needs its own uv environment.** Ultralytics pulls a torch build of its own and
  will fight the `whisperx`/`diarization`/`activespeaker` pins if installed beside them.
  Five environments already exist for exactly this reason.
- Choose a **weights policy up front**: ultralytics downloads checkpoints on first use,
  like TalkNet does. Reuse the `weights_dir` staging idea or accept the download.
- Its output is a *person* track, not a *face* track. Do not pretend the two are the
  same id space; TalkNet's `track_id` and a YOLO tracker id are unrelated and a consumer
  that joins on them gets nonsense.

### 20.3 `stories` — narrative windows mined with the LLM

The hardest of the three and the one to prototype before building. Input: transcript +
word timings + `speaker_assignment` + (once it exists) diarization_v2/persons. Output:
candidate narrative windows with a start and an end, plus why the model thinks the story
begins and ends there.

Why it is not a single prompt over the transcript:
- Stories **nest** (an anecdote inside a report) and **interrupt** each other, so the
  output is a forest, not a list of intervals. A schema that cannot express nesting will
  force the model to flatten the very structure being asked for.
- Boundaries are usually *prosodic and referential* ("volviendo a lo de antes…", a
  speaker change, a topic return), which means the prompt needs the transcript *with*
  timestamps and speaker labels, not plain text.
- The endpoint already emits fenced JSON (seen in translation), so the parse path exists;
  what does not exist is a **rejection path**. A model asked hard enough will always
  find a story. The output must be allowed to be empty, and something must count how
  often it is.
- Cost and determinism: one request per candidate window vs one request per video is a
  real design fork. Temperature 0 alone will not make it reproducible; the request digest
  must bind the prompt version, the model name and the transcript hash.

Prototype first, small: hand-pick three clips, write the prompt, read the output, and
only then decide the schema. Building the stage before seeing what the endpoint actually
returns is how you end up with a schema that cannot hold the answers.

### 20.4 `pose_normalized` — body keypoints normalised à la dfMaker (multimolang)

Requested by the operator. Depends on **`openpose` output only**, so it is a downstream
normalisation stage, not a new detector: it consumes `pose/*.parquet` and the raw
OpenPose JSON already in `pose/raw/`.

What it is, verified rather than assumed: `dfMaker()` is the first tool of **multimolang**
(the MULTIFLOW project, daedalusLAB), published on CRAN. It structures OpenPose keypoints
and applies a **user-defined linear transformation** to the raw coordinates: pick one
keypoint as the origin and two more to define the new basis vectors, which re-expresses
every joint in a body-centred frame instead of pixels. That is what makes poses
comparable across people, camera framing and shot scale — the current `pose/*.parquet`
carries pixel coordinates, so a presenter who steps back looks like they shrank.

Facts that will shape the implementation more than the maths:

- **It is R, not Python.** CRAN `multimolang` depends on R ≥ 4.1.0 and imports `arrow`.
  Two honest routes, and the choice must be made deliberately:
  1. run R as a **sidecar** (a `environments/pose_normalized/` holding an R script,
     invoked like the workers are invoked today). Keeps semantics identical to the tool
     the operator referenced, keeps results comparable with published multimolang work,
     and inherits a GPL-3 dependency plus a second runtime on every machine.
  2. reimplement the transformation **in Python** inside the pipeline. The vignette
     defines the algebra (origin + two basis keypoints), so the reimplementation is
     small and stays inside the existing Parquet/provenance/validation discipline.
     Cost: it is *compatible with* dfMaker, not dfMaker, and must be verified against
     the reference rather than asserted.
  Recommended: route 2, validated against route 1 on the fixtures, because the pipeline's
  guarantees (atomic writes, row-count integrity, request-digest caching, provenance
  sidecars that leave raw bytes untouched) would otherwise be duplicated in R.
- **The reference frame is a real decision, not a parameter to fill in later.** Which
  keypoint is the origin (sternum? pelvis? neck?) and which two define the basis decides
  whether the output answers "how does the body move" or "how do the arms move relative
  to the torso". BODY_25 gives us both candidates. Record the chosen triple in the
  request digest, because changing it changes every number in the table.
- **Missing keypoints are the normal case, not the error case.** OpenPose reports
  confidence per joint and this corpus contains people who are partially off-frame; a
  basis defined by a joint that is absent in a frame cannot produce a frame of
  coordinates for that frame. The table needs an explicit state for "no valid basis in
  this frame" — the same lesson as `face_status` in 20.2/§17: absence must not be
  encoded as a zero or as a silently dropped row.
- Keep it as a **new output next to the raw pixel tables**. Pixels are the measured
  quantity; normalised coordinates are a derived interpretation, and the pipeline's rule
  is that raw survives so a later decision can be recomputed without re-running OpenPose.
- Depends on `openpose` only, so it must inherit openpose's skip semantics and stay
  independent of the audio/transcript branch.

### 20.5 `pose_skeletons` — rendered OpenPose frames (body + hands + face)

Requested by the operator: ask OpenPose to emit the rendered skeleton images when it
runs. Verified against the installed binary (`/opt/openpose/build/examples/openpose/openpose.bin`)
rather than from documentation:

- The stage currently passes `--render_pose 0 --display 0` and **no `--write_images`**,
  so no rendered frame exists anywhere today. The raw `_keypoints.json` files are the
  only pose output.
- The flags exist and are compatible with headless operation: `--write_images <dir>`
  (format via `--write_images_format`), and rendering is per-module — `--render_pose`,
  `--face_render` and `--hand_render` are separate switches, each accepting `-1` to
  inherit `render_pose`. So body+hands+face skeletons in one pass is one extra flag
  group, not a second run: OpenPose renders the same pass it already computes.
- `--display 0` stays exactly as it is. The help text is explicit that rendering is
  independent of visual display, and this build has no display to disable anyway.

What is *not* free, and why this is a task rather than a one-line edit:

- **Disk, and it is the whole cost.** One PNG per frame per video, on top of the raw
  JSON that already dominates `pose/raw/`. At 25 fps a 4-minute clip is 6000 renders;
  `--output_resolution` (default `-1x-1`, i.e. input resolution) decides whether that is
  hundreds of MB or a few GB per video. This needs an explicit default and probably a
  downscale, not the binary's default.
- **It changes the `openpose` stage fingerprint**, which invalidates every existing
  pose dataset and re-runs OpenPose (the slowest stage) on all of them. Better as an
  opt-in config flag, defaulting off, so enabling it is a deliberate decision.
- **Do not disturb `--keypoint_scale`.** It is not currently passed, which means the JSON
  coordinates keep the meaning the published `pose/*.parquet` tables assert. Touching it
  to make images and JSON "match" would silently rescale every already-published
  number; images and coordinates are allowed to differ in scale, and if they do, that
  belongs in the artifact metadata.
- Raw renders belong under `pose/` beside the raw JSON with provenance, and the Parquet
  tables must keep describing measured keypoints — a rendered PNG is a view, not data.
- Depends on `openpose` and inherits its skip semantics. Pairs naturally with 20.4
  (normalised keypoints à la dfMaker): the skeleton images are the human-facing view of
  the same signal, so the two should agree on which frames had a usable detection.
