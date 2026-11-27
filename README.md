# Storytel Pipeline

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/brand/banner-dark.svg">
    <img alt="" src="docs/assets/brand/banner-light.svg" width="100%"></picture></p>

<p align="center">
  <a href="docs/getting-started.md#prerequisites"><img alt="Python 3.10–3.13" src="https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg"></a> <a href="https://docs.astral.sh/uv/"><img alt="Runs on uv" src="https://img.shields.io/badge/runs%20on-uv-261230.svg"></a><br>
  <a href="#quickstart">Get started</a> · <a href="#what-it-does">Outputs</a> · <a href="docs/architecture.md">Architecture</a> · <a href="#citing">Cite</a></p>

**Video recordings → traceable, time-aligned multimodal datasets.**

Built for researchers studying face-to-face conversation and storytelling through words, voice
and bodily movement. Each video yields one dataset directory: normalized Parquet tables, retained
tool output and stage caches, a log and provenance, times in seconds from video start, and stated
[join rules](docs/datasets.md#consumer-rules). No accuracy claim: each stage records what it produced, skipped and why.

## What it does

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/brand/modalities-dark.svg">
    <img alt="Synthetic illustration with invented values: six labelled lanes — speech, language, acoustics, vision, speakers and stories — share one axis in seconds from video start. A marker at 33 seconds reads every lane at once, inside an example story candidate span, and the lanes feed one dataset directory of Parquet tables, retained tool output, provenance and an optional ELAN export." src="docs/assets/brand/modalities-light.svg" width="100%"></picture></p>

| Layer | Tools and results | Output |
|---|---|---|
| Speech | WhisperX transcription/alignment; Nemotron speaker diarization | `speech/words.parquet`, `speech/speaker_turns_nemotron.parquet` |
| Language | English segment translations; spaCy source/English tokens and sentences | `translation/segments_en.parquet`, `linguistic/source/tokens.parquet`, `linguistic/english/tokens.parquet` |
| Acoustics | Praat/Parselmouth pitch, intensity, formants and segment statistics | `acoustic/frame_features.parquet` |
| Visual | OpenPose keypoints and body-centred coordinates; YOLO person tracks | `pose/body.parquet`, `persons/tracks.parquet` |
| Speakers | TalkNet visible-face activity — a separate, visual signal | `speaker/active_speaker_frames.parquet` |
| Stories | LLM candidates for storytelling events and the stories told inside them, following the expert's [annotation guidelines](docs/modalities.md#storytelling-events-and-stories); quote-anchored, Nemotron speaker labels or `unknown`, not validated detections | `stories/storytelling_events.parquet`, `stories/stories.parquet` |
| Inspection | ELAN `.eaf` files written with pympi-ling (requires a transcript), HTML preview, corpus-specific bundle | `elan/annotations.eaf` |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/brand/flow-dark.svg">
  <img alt="Five steps, run sequentially for one video: probe with ffprobe and ffmpeg; run each tool in its own uv environment or the OpenPose binary; normalize to Parquet and retain tool output; validate and record status, manifest and provenance so valid results are reused; inspect through an optional ELAN file, an HTML preview helper and explanation guides." src="docs/assets/brand/flow-light.svg" width="100%"></picture>

Heavy tools live in `uv` environments the root sync does not install ([what runs by default](docs/getting-started.md#what-runs-by-default)); the full stage graph is in [Architecture](docs/architecture.md#stage-dependencies).

## Quickstart

Root sync installs the CLI, Parquet and the stages that need no model. `config.example.yaml`
enables **every** stage — CUDA WhisperX, Nemotron diarization, spaCy, acoustics, OpenPose,
pose normalization, TalkNet, YOLO and ELAN — and installs none of their worker environments,
so sync each one you want the way [Install](docs/getting-started.md#install) shows
(`cd environments/<name> && uv sync --python 3.12`). A stage whose environment, binary,
checkout or checkpoint is absent **skips with a reason** instead of failing the video, so the
first run reports what a full one would add. OpenPose, TalkNet, `HF_TOKEN` and an LLM endpoint
are further [prerequisites](docs/getting-started.md#prerequisites).

```bash
uv sync --python 3.12
cp config/config.example.yaml config/config.local.yaml && cp .env.example .env
$EDITOR config/config.local.yaml   # input and output paths, stages, endpoints
$EDITOR .env                       # credentials; never commit this file
uv run multimodal-pipeline inspect-environment -c config/config.local.yaml
# put one clip in input.directory, then smoke-test the plumbing: ffmpeg/ffprobe only
uv run multimodal-pipeline run -c config/config.local.yaml --video clip.mp4 --to-stage audio
uv run multimodal-pipeline run -c config/config.local.yaml   # the full configured stage list
```

Keep existing configuration and credentials instead of overwriting them; translation and stories send transcript text out, so they can cost money.

<details><summary>What one video becomes: one <a href="docs/datasets.md#directory-layout">dataset directory</a> under <code>output.directory</code></summary>

```text
<video_id>/
├── manifest.json  status.json   what exists, what is absent, what each stage decided
├── source/ audio/ speech/ translation/ linguistic/ acoustic/ pose/ persons/ speaker/ stories/
└── elan/ logs/ provenance/      ELAN summary, pipeline logs, config/tools/processing records
```
</details>

## Documentation

| Question | Guide |
|---|---|
| How do I install and process a video? | [Getting started](docs/getting-started.md) |
| How do I run, resume or select stages? | [CLI reference](docs/cli.md) |
| What settings, credentials and failure messages matter? | [Configuration](docs/configuration.md) |
| What files do I get, and how do I join their tables? | [Datasets and analysis](docs/datasets.md) |
| What do the measurements mean, and what are their limits? | [Modalities](docs/modalities.md) |
| What is exported to ELAN, and how do I share it? | [ELAN export](docs/elan.md) |
| How is the pipeline structured, and why? | [Architecture](docs/architecture.md) |
| Where is the code, and how do I test a change? | [Development](docs/development.md) |

## Status, scope and limitations

- Live work and deferrals: the [ODD tracker](odd/tasks/multimodal-video-pipeline.md).
- `.github/workflows/unit.yml` covers the unit suite on Linux with ffmpeg only: no GPU models, OpenPose, live endpoints or accuracy, and no hosted run is claimed here.
- Stories are LLM candidates, track IDs are not unique-human counts, and one video runs at a time (no server or queue).

## Citing

If you use this pipeline, please cite the STORYTEL article it accompanies. [CITATION.cff](CITATION.cff)
declares it as the preferred citation, so GitHub's **Cite this repository** button exports it as APA or BibTeX.

> Bonifazi, A., Al-qaoud, J., Alcaraz Carrión, D., Crosilla, G., Debreslioska, S., Del Rosario Illan Castillo, M., Eide, Ø., Frings, M., Garibyan, A., Gipper, S., Grice, M., Iabdounane, Y., Pagán Cánovas, C., Patel, M., Rau, F., Sánchez Sánchez, R., Somrani Achouri, M., Turner, M., Uhrig, P., Vitali, F., & Yu, M. (2026).
> Enabling AI to find multimodal signals of storytelling cross-culturally: A multi-step approach.
> In *Companion Publication of the 28th International Conference on Multimodal Interaction (ICMI Companion ’26)* (pp. 370–374). ACM.
> https://doi.org/10.1145/3776591.3836646

## Acknowledgements

Developed for [STORYTEL](https://storytel.uni-koeln.de/), which studies spontaneous storytelling in face-to-face conversation through words, voice and bodily movement.

<details><summary>Tools this pipeline drives</summary>

WhisperX, NVIDIA Nemotron, [spaCy](https://spacy.io/),
[Praat](https://www.fon.hum.uva.nl/praat/) via Parselmouth, [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose),
[Ultralytics YOLO](https://docs.ultralytics.com/), TalkNet-ASD and [ELAN](https://archive.mpi.nl/tla/elan) are driven
rather than reimplemented; their licences and terms stay with their authors. This is attribution, not endorsement.
</details>

---

Code written and maintained by **Raúl Sánchez** ([raul@um.es](mailto:raul@um.es)).
