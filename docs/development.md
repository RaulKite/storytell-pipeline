# Development

How to run the suite and what it refuses to mock, plus where things live in the repository.

Back to the overview: [Storytel Pipeline](../README.md).

Commands use repository-relative paths: run them from the repository root.

## Contents

- [Testing](#testing)
- [Repo layout](#repo-layout)

---

## Testing

```bash
uv run --with pytest pytest tests/unit -q -p no:randomly     # 2036 tests, ~90 s
uv run --with pytest pytest tests/e2e -q -p no:randomly      # 42 tests, ~285 s (needs ffmpeg + uv)
```

`-p no:randomly` keeps the documented order stable; some tests share module-level
state. The counts above are collected cases, including skipped optional checks.

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
masking, the manifest's promise that every listed artifact exists, and the three stages
that compute rows in-process noticing an edit to the python that computes them.

---

## Repo layout

```
src/multimodal_pipeline/   orchestrator: config, discovery, DAG, state, CLI, normalization
workers/                   heavy ML entry points, run inside the isolated envs
                         (whisperx, diarization, nemotron diarization, spacy,
                         acoustic, activespeaker, persons)
environments/              one uv project per dependency-heavy tool
config/                    example template (committed) + local config (ignored)
tests/unit/                2036 tests
tests/e2e/                 42 CLI-driven tests
scripts/                   fixture + spaCy model installers, dataset figure renderer,
                           ELAN .eaf HTML viewer (no ELAN needed to eyeball a tier layout),
                           and the shareable review bundle builder
docs/                      installation, CLI, datasets, modalities, ELAN and contributor guides
docs/assets/               committed figures (synthetic-schema demos, regenerable)
odd/tasks/                 Gentle-AI ODD feature document (decisions, evidence)
data/input_videos/         synthetic fixtures (committed, ~330 KB)
```

`data/processed/` and `data/input_videos/person_demo.avi` are deliberately not
version-controlled: the datasets are fully regenerable, and the person clip is
OpenPose's own example media, copied on demand by `scripts/make_fixtures.sh`.
