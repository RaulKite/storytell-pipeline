# Configuration

Use a local YAML file for paths and stage settings, and `.env` for credentials.
This guide identifies the main knobs and the failure messages worth investigating.

Back to the overview: [Storytel Pipeline](../README.md).

Commands use repository-relative paths: run them from the repository root.

## Contents

- [Configuration sources](#configuration-sources)
- [Stage settings](#stage-settings)
- [Language policy](#language-policy)
- [Secrets and external requests](#secrets-and-external-requests)
- [Troubleshooting](#troubleshooting)

## Configuration sources

[`config/config.example.yaml`](../config/config.example.yaml) is the annotated
starting point; [`config.py`](../src/multimodal_pipeline/config.py) defines accepted
fields and defaults. Unknown settings are rejected, not ignored.

The local copy `config/config.local.yaml` and root `.env` are gitignored.
Interpolation accepts `${VAR}` and `${VAR:-default}`. `.env` is loaded before
interpolation; exported variables win. Set `MULTIMODAL_PIPELINE_NO_DOTENV=1` to
opt out of the automatic read.

```yaml
input: {directory: data/input_videos, recursive: false}
output: {directory: data/processed}
translation:
  base_url: ${LITELLM_BASE_URL:-http://LITELLM_HOST:PORT/v1}
  api_key: ${LITELLM_API_KEY:-}
  model: ${LITELLM_MODEL:-}
```

This fragment shows interpolation, not a complete config. The shipped template
uses a literal placeholder host/model; defining `LITELLM_BASE_URL` alone does not
replace a literal YAML value. Reference the variable in YAML or edit the value.
The same applies independently to the `stories` section.

## Stage settings

| Section | Important settings / cost |
|---|---|
| `input`, `output` | Directory, recursive discovery and accepted extensions; dataset root |
| `execution` | Sequential mode, GPU index, stop-on-video-error policy |
| `whisperx` | Model, language, device, compute type, batching and VAD |
| `diarization` | pyannote pipeline, speaker constraints, exclusive alignment preference |
| `diarization_nemotron` | Optional second engine, threshold and up to eight speaker channels |
| `translation`, `stories` | Independent endpoint/model, prompt version, retries, context and cache |
| `spacy` | Source/English models, fallback and language-reliability policy |
| `acoustic` | Frame step, pitch range, formant ceiling, pause policy and chunk size |
| `openpose` | Root/executable/models, enabled keypoint modules, optional rendered images |
| `pose_normalized` | Origin and basis keypoints; changing them changes coordinates |
| `activespeaker` | TalkNet checkout/weights, device, threshold, smoothing and switch hysteresis |
| `persons` | Model/weights, tracker, confidence, classes and input image size |
| `speaker_fusion` | Selected engines, minimum face frames and active ratio |
| `elan` | Enable/disable derived annotation export |
| `logging` | Level, console output and progress refresh interval |

OpenPose renderings are opt-in: `openpose.write_images: true` adds files under
`pose/raw_images/`. Set `openpose.image_max_side` to bound their size. Changing
render settings invalidates OpenPose; images can use substantial disk. Coordinates
remain source pixels, not resized-image pixels. At least one drawing module must
be enabled. Turning rendering off does not delete old renderings.

YOLO's default class filter is people (`[0]`). Deliberately changing object classes
requires disabling `persons.person_classes_only`; interpret such output as object
tracks, even though the ID column retains its name.

## Language policy

WhisperX records `language_detection` in its raw JSON. `configured` means you
pinned the language; `low` can mean a clip shorter than the 30 s detection window,
a missing probability, or probability below 0.5. This grades the language guess,
not transcript quality.

- `spacy.trust_low_language_detection: true` (default) accepts the guess and warns.
- `false` demotes low-reliability source-language analysis to `blank`.
- The English variant forces English and does not use this detection grade.

Model resolution tries the configured language model, a matching model family,
other installed models for that language, then the configured fallback. `blank`
provides tokenization/sentence boundaries without trained lemmas, POS, dependencies
or entities. Inspect `selected_model`, `model_selection_status`, `capabilities` and
`language_reliability_trusted` in the raw result; a populated table alone proves
none of those capabilities. Installing/removing models invalidates linguistic reuse.

## Secrets and external requests

Use [`.env.example`](../.env.example) for the variable names. pyannote reads the
variable named by `diarization.hf_token_env` (normally `HF_TOKEN`). API keys belong
in `.env` and YAML interpolation, not committed literals.

Credential-like fields are masked as `***masked***` in pipeline bookkeeping.
That is not a guarantee that every artifact is safe to publish: raw responses,
transcript content, source media, hostnames and paths can still be sensitive.

Translation and stories are separate HTTP stages. Enable them only when you accept
sending text/context to the configured endpoint and its token cost. `provider: mock`
is for offline tests. `temperature: 0.0` does not guarantee deterministic service
responses; retained responses and cache keys are the reproducibility evidence.

## Troubleshooting

| Symptom | Check / remedy |
|---|---|
| Missing `HF_TOKEN` | Add the credential and accept pyannote model terms; diarization otherwise skips |
| Endpoint not configured | Replace the placeholder in that stage's YAML; supply key/model, use mock or disable |
| `uv project not found` | Restore/correct the configured project directory, then sync it; sync cannot create a missing project definition |
| `libnvrtc.so.13` | Check the diarization torchcodec pin and re-sync; see environment comments |
| OpenPose binary not found | Fix `openpose.root` or executable/model paths; inspect the environment report |
| TalkNet root missing / no `run_talknet.py` | Point at a valid TalkNet-ASD checkout |
| TalkNet `weights_only` / unpickling error | Preserve the torch 2.5.1 compatibility pin and re-sync its environment |
| Stage reruns unexpectedly | Read `status --plan`; compare config/tool/source hashes, upstream sequence and validation |
| Empty pose/person table | Inspect raw output and stage status; no detection can be a valid result, not proof nobody was present |
| Empty acoustic segment table | Inspect transcript segments and raw acoustic output, not only voicing |
| Negative segment confidence | WhisperX segment scores are log-probability-derived; not word probabilities |
| Unscored/imputed active-speaker rows | Read `face_status`, `frame_reason` and `score_imputed`; see [Modalities](modalities.md#active-speaker-evidence) |

Use `logs/<stage>.log` for the command/tool error, `status.json` for the recorded
outcome, and `validate` for current artifact checks. Re-enabling or changing tools
can trigger expensive reruns; check the plan before running the batch.
