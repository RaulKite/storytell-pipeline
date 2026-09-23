# multimodal-pipeline

Sequential, resumable multimodal video-processing pipeline.

Give it a directory of video recordings; it produces **one self-contained
dataset directory per video** containing media metadata, extracted audio,
WhisperX transcription with word-level alignment, Pyannote Community
diarization, a speaker-assigned transcript, an English translation through a
LiteLLM OpenAI-compatible endpoint, spaCy linguistic features (source language
and English), Praat/Parselmouth acoustic features and OpenPose BODY_25 + hands
+ face keypoints — all normalised to Parquet on a single video timeline, with
raw tool artifacts, logs, status, provenance and a manifest.

> Full documentation (installation, uv environments, schemas, resume semantics,
> troubleshooting) is being filled in as the pipeline lands — see
> `odd/tasks/multimodal-video-pipeline.md` for authoritative progress.

## Quick start

```bash
uv sync --python 3.12
cp config/config.example.yaml config/config.local.yaml
$EDITOR config/config.local.yaml
multimodal-pipeline run --config config/config.local.yaml
```

## Layout

```
src/multimodal_pipeline/   orchestrator (light dependencies only)
workers/                   heavy ML code, run inside isolated uv environments
environments/              one uv project per dependency-heavy tool
config/                    YAML runtime configuration
tests/                     unit + integration tests
odd/tasks/                 Gentle-AI ODD feature document
```
