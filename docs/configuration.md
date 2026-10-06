# Configuration

The YAML schema, how `${VAR}` interpolation keeps secrets out of config files, how far to trust a detected language, and the troubleshooting table for the messages a run actually prints.

Back to the overview: [Storytel Pipeline](../README.md).

Commands use repository-relative paths: run them from the repository root.

## Contents

- [Configuration](#configuration)
- [Secrets](#secrets)
- [Troubleshooting](#troubleshooting)

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

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `diarization ... missing credential HF_TOKEN` | Put `HF_TOKEN=...` in `.env` and accept the model's EULA on Hugging Face. Everything except speakers and English linguistics still runs without it. |
| `translation endpoint is not configured` | Set `translation.base_url`/`api_key`/`model` (or `provider: mock` to test the graph). |
| `stories endpoint is not configured` | Set `stories.base_url`/`api_key`/`model` — or `stories.enabled: false` to stop paying for a stage whose output you do not read. |
| `uv project not found: .../environments/whisperx` | Run `uv sync --python 3.12` in that environment directory. |
| `libnvrtc.so.13` on import | `torchcodec` drifted past 0.7.0. Re-pin and re-sync the diarization env. |
| `OpenPose binary not found under /opt/openpose` | Check `openpose.root`; `inspect-environment` prints the resolved path. |
| Everything reruns after one config edit | Expected: `--only-stage X --force-stage X` reruns X and its dependants only. Check `status --plan` for the named reason. |
| `pose/*.parquet` have 0 rows | The video contains no person. That is a valid outcome; the raw JSON in `pose/raw/` confirms it. |
| `activespeaker.talknet_root is not set` | Clone TalkNet-ASD and set `activespeaker.talknet_root`. Without it the stage skips and the rest of the dataset is unaffected. |
| `activespeaker.talknet_root has no run_talknet.py` | That path is not a TalkNet-ASD checkout — the stage checks for the entrypoint rather than letting a confusing `torch.load` error surface minutes later. |
| TalkNet dies with an unpickling or `weights_only` error | The environment drifted past torch 2.5. Re-sync `environments/activespeaker`; the pin is load-bearing (see [Why seven environments](architecture.md#why-seven-environments)). |
| `speaker/active_speaker_frames.parquet` has rows with `track_id = null` | No face was detected in those frames — off-screen, back-turned, or too small. S3FD tracks near-frontal faces only; a person walking away legitimately loses the track. Absence is recorded as a row, not dropped. `frame_reason = no_face` says the same thing from the score side. |
| `score_imputed = true` on some frames | TalkNet scores fewer frames than it tracks (an unexplained `-1` in its MFCC windowing), so the last score was carried forward rather than measured. At most two frames per track are affected; treat those as unmeasured, not as low confidence. `frame_reason = imputed_tail` names the same rows. |
| A frame has `face_status = "tracked_unscored"` | S3FD located a face there but TalkNet produced no usable score for it. `frame_reason` names which of the four causes: `score_not_finite` (a score existed and was NaN/inf), `track_has_no_scores` (the track never produced one), `past_scored_tail` (beyond the two frames that may be imputed), or `tail_score_not_finite` (inside the window, the value to carry was broken). The scores stay `null` and the frame is never active — this is "we could not measure", not "nobody was on screen" and not a confidence of zero. |
| A frame has `frame_reason = "unknown"` | That table was normalised from a raw artifact written before the field existed, so the cause is not recoverable. Rerun `activespeaker` to get a diagnosed table. |
| `acoustic/segment_features.parquet` is empty | No voiced audio (silent track, or music with no speech-like f0). |
| `avg_segments_confidence` is negative | WhisperX reports log-probability-derived segment confidence; word confidences are the 0–1 ones. |
